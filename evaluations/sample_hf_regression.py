#!/usr/bin/env python3
"""Create five additional deterministic 100k PubChem and ZINC22 shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path

SEEDS = (101, 211, 307, 401, 503)
ROWS_PER_SHARD = 100_000
POOL_MULTIPLIER = 2


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    repo_id: str
    revision: str
    smiles_column: str


PUBCHEM = DatasetSpec(
    key="pubchem",
    repo_id="jablonkagroup/pubchem-smiles-molecular-formula",
    revision="cd46fb8a3110cc12d6a669ad0d8e260b3acf6d0b",
    smiles_column="smiles",
)
ZINC22 = DatasetSpec(
    key="zinc22",
    repo_id="chandar-lab/ZINC_22",
    revision="f13c8b4132037e38102499e07741dc9a71431ea5",
    smiles_column="SMILES",
)


def _existing_smiles(output_root: Path, key: str) -> set[str]:
    seen = set()
    for path in sorted((output_root / key).glob("*_input.jsonl")):
        # Generated HF shards must not exclude themselves on a repeat run.
        if "_hf_" in path.name:
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                smiles = str(json.loads(line).get("smiles") or "").strip()
                if smiles:
                    seen.add(smiles)
    return seen


def _write_shard(output_root: Path, spec: DatasetSpec, seed: int, rows: list[dict], sources: list[str]) -> None:
    output_dir = output_root / spec.key
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{spec.key}_hf_seed{seed}_{len(rows)}"
    payload = "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows)
    (output_dir / f"{stem}_input.jsonl").write_text(payload, encoding="utf-8")
    manifest = {
        "dataset": spec.key,
        "repo_id": spec.repo_id,
        "revision": spec.revision,
        "seed": seed,
        "sampling": "deterministic seeded sampling without replacement; paper regression rows excluded",
        "rows": len(rows),
        "sources": sources,
        "sha256": hashlib.sha256(payload.encode()).hexdigest(),
    }
    (output_dir / f"{stem}_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _sample_pubchem(output_root: Path, size: int) -> None:
    from datasets import load_dataset

    dataset = load_dataset(PUBCHEM.repo_id, split="train", revision=PUBCHEM.revision)
    excluded = _existing_smiles(output_root, PUBCHEM.key)
    selected_indices: set[int] = set()
    for seed in SEEDS:
        rng = random.Random(seed)
        rows = []
        while len(rows) < size:
            needed = size - len(rows)
            candidates = []
            while len(candidates) < max(needed * 2, 10_000):
                index = rng.randrange(len(dataset))
                if index not in selected_indices:
                    selected_indices.add(index)
                    candidates.append(index)
            for index, smiles in zip(candidates, dataset.select(candidates)[PUBCHEM.smiles_column], strict=True):
                smiles = str(smiles or "").strip()
                if not smiles or smiles in excluded:
                    continue
                excluded.add(smiles)
                rows.append(
                    {
                        "set_seed": seed,
                        "row_in_set": len(rows),
                        "dataset_index": index,
                        "smiles": smiles,
                    }
                )
                if len(rows) == size:
                    break
        _write_shard(output_root, PUBCHEM, seed, rows, ["train"])


def _zinc_parquets() -> list[str]:
    from huggingface_hub import HfApi

    return sorted(
        path
        for path in HfApi(token=os.environ.get("HF_TOKEN")).list_repo_files(
            ZINC22.repo_id,
            repo_type="dataset",
            revision=ZINC22.revision,
        )
        if path.startswith("data/train-") and path.endswith(".parquet")
    )


def _load_zinc_parquet(path: str):
    from datasets import load_dataset
    from huggingface_hub import hf_hub_url

    url = hf_hub_url(
        repo_id=ZINC22.repo_id,
        filename=path,
        repo_type="dataset",
        revision=ZINC22.revision,
    )
    kwargs = {"data_files": url, "split": "train"}
    if token := os.environ.get("HF_TOKEN"):
        kwargs["token"] = token
    return load_dataset("parquet", **kwargs)


def _sample_zinc(output_root: Path, size: int) -> None:
    excluded = _existing_smiles(output_root, ZINC22.key)
    parquets = _zinc_parquets()
    for seed in SEEDS:
        rng = random.Random(seed)
        ordered = parquets[:]
        rng.shuffle(ordered)
        pool = []
        candidate_seen = set(excluded)
        sources = []
        for source in ordered:
            dataset = _load_zinc_parquet(source)
            take = min(len(dataset), size * POOL_MULTIPLIER - len(pool))
            indices = rng.sample(range(len(dataset)), take)
            for index, smiles in zip(indices, dataset.select(indices)[ZINC22.smiles_column], strict=True):
                smiles = str(smiles or "").strip()
                if smiles and smiles not in candidate_seen:
                    pool.append({"smiles": smiles, "source_parquet": source, "source_row": index})
                    candidate_seen.add(smiles)
            sources.append(source)
            if len(pool) >= size * POOL_MULTIPLIER:
                break
        if len(pool) < size:
            raise RuntimeError(f"only collected {len(pool)} unique ZINC22 candidates for seed {seed}")
        rows = rng.sample(pool, size)
        for row in rows:
            excluded.add(row["smiles"])
        _write_shard(output_root, ZINC22, seed, rows, sources)


def sample_all(output_root: Path, size: int = ROWS_PER_SHARD) -> None:
    _sample_pubchem(output_root, size)
    _sample_zinc(output_root, size)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("evaluations/data"))
    parser.add_argument("--size", type=int, default=ROWS_PER_SHARD)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.size < 1:
        raise SystemExit("--size must be positive")
    sample_all(args.output_root, args.size)
