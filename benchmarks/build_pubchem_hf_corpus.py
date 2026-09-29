#!/usr/bin/env python3
"""Build a second speed corpus from a pinned 200k PubChem HF sample."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import tempfile
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

PUBCHEM_REPO = "jablonkagroup/pubchem-smiles-molecular-formula"
PUBCHEM_REVISION = "cd46fb8a3110cc12d6a669ad0d8e260b3acf6d0b"
SAMPLE_SEED = 20260929
SELECTION_SEED = 20260930
SAMPLE_SIZE = 200_000
CORPUS_SIZE = 5_000
FIELDS = (
    "source_index",
    "smiles",
    "canonical_smiles",
    "heavy_atoms",
    "ring_count",
    "hetero_atoms",
    "formal_charge",
    "has_stereo",
)


def _stable_priority(seed: int, canonical_smiles: str) -> bytes:
    value = f"{seed}\0{canonical_smiles}".encode()
    return hashlib.blake2b(value, digest_size=16).digest()


def _excluded_canonical_smiles(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["canonical_smiles"] for row in csv.DictReader(handle) if row.get("canonical_smiles")}


def _molecule_row(source_index: int, smiles: str) -> dict[str, Any] | None:
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    return {
        "source_index": source_index,
        "smiles": smiles,
        "canonical_smiles": canonical,
        "heavy_atoms": mol.GetNumHeavyAtoms(),
        "ring_count": mol.GetRingInfo().NumRings(),
        "hetero_atoms": sum(atom.GetAtomicNum() not in (1, 6) for atom in mol.GetAtoms()),
        "formal_charge": sum(atom.GetFormalCharge() for atom in mol.GetAtoms()),
        "has_stereo": int("@" in smiles or "/" in smiles or "\\" in smiles),
    }


def sample_candidates(dataset: Any, *, size: int, seed: int, excluded: set[str]) -> list[dict[str, Any]]:
    """Select exactly ``size`` unique valid molecules from a pinned dataset."""

    rng = random.Random(seed)
    used_indices: set[int] = set()
    seen = set(excluded)
    rows: list[dict[str, Any]] = []
    while len(rows) < size:
        batch_size = min(max((size - len(rows)) * 2, 10_000), len(dataset) - len(used_indices))
        if batch_size <= 0:
            break
        indices: list[int] = []
        while len(indices) < batch_size:
            index = rng.randrange(len(dataset))
            if index not in used_indices:
                used_indices.add(index)
                indices.append(index)
        smiles_values = dataset.select(indices)["smiles"]
        for index, value in zip(indices, smiles_values, strict=True):
            smiles = str(value or "").strip()
            row = _molecule_row(index, smiles) if smiles else None
            if row is None or row["canonical_smiles"] in seen:
                continue
            seen.add(row["canonical_smiles"])
            rows.append(row)
            if len(rows) == size:
                break
        print(f"sampled {len(rows)}/{size} unique valid molecules", flush=True)
    if len(rows) != size:
        raise RuntimeError(f"only sampled {len(rows)} unique valid molecules")
    return rows


def _verify_chunk(payload: tuple[list[dict[str, Any]], list[str]]) -> list[dict[str, Any]]:
    rows, names = payload
    from py2opsin import py2opsin

    from openclatura.resonance_compare import equivalent_smiles

    tmp_path = Path(tempfile.gettempdir()) / f"openclatura_pubchem_speed_{os.getpid()}_{time.time_ns()}.txt"
    try:
        decoded = list(py2opsin(names, tmp_fpath=str(tmp_path)))
    finally:
        tmp_path.unlink(missing_ok=True)
    if len(decoded) != len(names):
        raise RuntimeError(f"OPSIN returned {len(decoded)} rows for {len(names)} names")
    return [
        {**row, "openclatura_name": name}
        for row, name, opsin_smiles in zip(rows, names, decoded, strict=True)
        if name and opsin_smiles and equivalent_smiles(str(row["smiles"]), opsin_smiles)
    ]


def verify_candidates(rows: list[dict[str, Any]], *, workers: int, chunk_size: int) -> list[dict[str, Any]]:
    from openclatura import name_many

    results = name_many(
        [str(row["smiles"]) for row in rows],
        processes=workers,
        chunksize=500,
    )
    names = [result.name or "" for result in results]
    payloads = [
        (rows[start : start + chunk_size], names[start : start + chunk_size])
        for start in range(0, len(rows), chunk_size)
    ]
    verified: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for completed, matched in enumerate(pool.map(_verify_chunk, payloads), start=1):
            verified.extend(matched)
            print(
                f"verified {min(completed * chunk_size, len(rows))}/{len(rows)}; accepted {len(verified)}",
                flush=True,
            )
    return verified


def _write_csv(path: Path, rows: list[dict[str, Any]], *, include_name: bool) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [*FIELDS, "openclatura_name"] if include_name else list(FIELDS)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _distribution(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    fields = ("heavy_atoms", "ring_count", "hetero_atoms", "formal_charge", "has_stereo")
    return {field: dict(sorted(Counter(str(row[field]) for row in rows).items())) for field in fields}


def build(args: argparse.Namespace) -> dict[str, Any]:
    from datasets import load_dataset
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    dataset = load_dataset(PUBCHEM_REPO, split="train", revision=PUBCHEM_REVISION)
    excluded = _excluded_canonical_smiles(args.exclude_corpus)
    sampled = sample_candidates(dataset, size=args.sample_size, seed=args.sample_seed, excluded=excluded)
    verified = verify_candidates(sampled, workers=args.workers, chunk_size=args.chunk_size)
    if len(verified) < args.corpus_size:
        raise RuntimeError(f"only {len(verified)} molecules passed verification")

    corpus = sorted(
        verified,
        key=lambda row: _stable_priority(args.selection_seed, str(row["canonical_smiles"])),
    )[: args.corpus_size]
    verified_sha256 = _write_csv(args.verified_output, verified, include_name=True)
    corpus_sha256 = _write_csv(args.corpus_output, corpus, include_name=False)
    manifest = {
        "dataset": PUBCHEM_REPO,
        "revision": PUBCHEM_REVISION,
        "split": "train",
        "sample_seed": args.sample_seed,
        "selection_seed": args.selection_seed,
        "sampled_rows": len(sampled),
        "verified_rows": len(verified),
        "rejected_rows": len(sampled) - len(verified),
        "verification": "equivalent_smiles(smiles, OPSIN(openclatura_name))",
        "excluded_corpus": str(args.exclude_corpus),
        "excluded_rows": len(excluded),
        "verified_pool": str(args.verified_output),
        "verified_pool_sha256": verified_sha256,
        "benchmark_rows": len(corpus),
        "benchmark_corpus": str(args.corpus_output),
        "benchmark_corpus_sha256": corpus_sha256,
        "distribution": _distribution(corpus),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", type=int, default=SAMPLE_SIZE)
    parser.add_argument("--corpus-size", type=int, default=CORPUS_SIZE)
    parser.add_argument("--sample-seed", type=int, default=SAMPLE_SEED)
    parser.add_argument("--selection-seed", type=int, default=SELECTION_SEED)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument(
        "--exclude-corpus",
        type=Path,
        default=Path("benchmarks/data/opsin_verified_5000.csv"),
    )
    parser.add_argument(
        "--verified-output",
        type=Path,
        default=Path("benchmarks/data/pubchem_hf_seed20260929_verified.csv"),
    )
    parser.add_argument(
        "--corpus-output",
        type=Path,
        default=Path("benchmarks/data/pubchem_hf_seed20260929_opsin_verified_5000.csv"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("benchmarks/data/pubchem_hf_seed20260929_opsin_verified_5000.manifest.json"),
    )
    args = parser.parse_args()
    if min(args.sample_size, args.corpus_size, args.workers, args.chunk_size) < 1:
        parser.error("sizes, workers, and chunk size must be positive")
    if args.corpus_size > args.sample_size:
        parser.error("--corpus-size cannot exceed --sample-size")
    return args


def main() -> int:
    args = parse_args()
    print(json.dumps(build(args), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
