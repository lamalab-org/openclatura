#!/usr/bin/env python3
"""Build a deterministic, OPSIN-verified performance corpus.

The source CSV is expected to contain an ``input`` SMILES column. Candidates
are deduplicated by canonical isomeric SMILES and ordered by a seeded stable
hash, giving a reproducible sample independent of source-file ordering. Only
rows whose OpenClatura name round-trips to the input structure through OPSIN
are written to the output corpus.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
import time
from collections import Counter
from pathlib import Path

from rdkit import Chem, RDLogger

from openclatura import name_many
from openclatura.resonance_compare import equivalent_smiles


def _stable_priority(seed: int, canonical_smiles: str) -> bytes:
    value = f"{seed}\0{canonical_smiles}".encode()
    return hashlib.blake2b(value, digest_size=16).digest()


def _load_candidates(path: Path, seed: int) -> tuple[list[dict[str, object]], int]:
    candidates: dict[str, dict[str, object]] = {}
    rows_read = 0
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "input" not in reader.fieldnames:
            raise ValueError(f"{path} must contain an 'input' column")
        for source_index, row in enumerate(reader):
            rows_read += 1
            smiles = (row.get("input") or "").strip()
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
            candidates.setdefault(
                canonical,
                {
                    "source_index": source_index,
                    "smiles": smiles,
                    "canonical_smiles": canonical,
                    "heavy_atoms": mol.GetNumHeavyAtoms(),
                    "ring_count": mol.GetRingInfo().NumRings(),
                    "hetero_atoms": sum(atom.GetAtomicNum() not in (1, 6) for atom in mol.GetAtoms()),
                    "formal_charge": sum(atom.GetFormalCharge() for atom in mol.GetAtoms()),
                    "has_stereo": int("@" in smiles or "/" in smiles or "\\" in smiles),
                },
            )
    ordered = sorted(
        candidates.values(),
        key=lambda row: _stable_priority(seed, str(row["canonical_smiles"])),
    )
    return ordered, rows_read


def _opsin_roundtrip(names: list[str]) -> list[str]:
    from py2opsin import py2opsin

    tmp_path = Path(tempfile.gettempdir()) / f"openclatura_corpus_{os.getpid()}_{time.time_ns()}.txt"
    try:
        decoded = list(py2opsin(names, tmp_fpath=str(tmp_path)))
    finally:
        tmp_path.unlink(missing_ok=True)
    if len(decoded) != len(names):
        raise RuntimeError(f"OPSIN returned {len(decoded)} rows for {len(names)} names")
    return [value or "" for value in decoded]


def _distribution(rows: list[dict[str, object]]) -> dict[str, dict[str, int]]:
    fields = ("heavy_atoms", "ring_count", "hetero_atoms", "formal_charge", "has_stereo")
    return {field: dict(sorted(Counter(str(row[field]) for row in rows).items())) for field in fields}


def build_corpus(source: Path, output: Path, *, size: int, seed: int, batch_size: int) -> dict[str, object]:
    RDLogger.DisableLog("rdApp.*")
    candidates, rows_read = _load_candidates(source, seed)
    accepted: list[dict[str, object]] = []
    attempted = 0

    for start in range(0, len(candidates), batch_size):
        batch = candidates[start : start + batch_size]
        results = name_many([str(row["smiles"]) for row in batch], processes=1)
        named = [(row, result.name) for row, result in zip(batch, results, strict=True) if result.name]
        decoded = _opsin_roundtrip([name for _, name in named])
        for (row, _name), opsin_smiles in zip(named, decoded, strict=True):
            if opsin_smiles and equivalent_smiles(str(row["smiles"]), opsin_smiles):
                accepted.append(row)
                if len(accepted) == size:
                    break
        attempted += len(batch)
        print(f"verified {min(start + len(batch), len(candidates))} candidates; accepted {len(accepted)}/{size}")
        if len(accepted) == size:
            break

    if len(accepted) != size:
        raise RuntimeError(f"only {len(accepted)} of {len(candidates)} unique candidates passed OPSIN verification")

    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source_index",
        "smiles",
        "canonical_smiles",
        "heavy_atoms",
        "ring_count",
        "hetero_atoms",
        "formal_charge",
        "has_stereo",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(accepted)

    corpus_sha256 = hashlib.sha256(output.read_bytes()).hexdigest()
    manifest: dict[str, object] = {
        "source": source.name,
        "selection": "canonical-deduplicate then seeded BLAKE2b ordering",
        "verification": "equivalent_smiles(input, OPSIN(openclatura_name))",
        "seed": seed,
        "requested_rows": size,
        "source_rows": rows_read,
        "unique_valid_candidates": len(candidates),
        "candidates_attempted": attempted,
        "verified_rows": len(accepted),
        "corpus_sha256": corpus_sha256,
        "distribution": _distribution(accepted),
    }
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--batch-size", type=int, default=1000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_corpus(args.source, args.output, size=args.size, seed=args.seed, batch_size=args.batch_size)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
