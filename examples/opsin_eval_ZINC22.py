import csv
import multiprocessing as mp
import os
import random
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_url

import numpy as np
import py2opsin
from datasets import load_dataset
from tqdm import tqdm
from bluenamer.utils import standardize_mol

from bluenamer.namer import name_smiles

# --- Configuration ---
ZINC22_REPO_ID = "chandar-lab/ZINC_22"
ZINC22_REVISION = None  # set to a commit hash for fully frozen reproducibility
SMILES_COLUMN = "SMILES"

# Number of parquet files to consider as one random "batch"
PARQUET_BATCH_SIZE = 4

# Build a larger candidate pool, then sample N_PER_SEED from it.
# 2 means: collect ~200K candidates, then sample 100K.
SAMPLE_POOL_MULTIPLIER = 2

HF_TOKEN = os.environ.get("HF_TOKEN")
N_PER_SEED = 100_000
SEEDS = [42, 17, 87, 5, 63]
OUT_DIR = Path("eval_failures")

NAME_CHUNKSIZE = 10
OPSIN_BATCH_SIZE = 1000


NO_NAME = "no name generated"
OPSIN_MISMATCH = "opsin smiles mismatch"
OPSIN_UNRECOGNIZED = "opsin - name is not recognized"


def list_zinc22_parquet_files():
    """
    Lists parquet shards in the HF dataset repo without downloading the dataset.
    """
    api = HfApi(token=HF_TOKEN)

    files = api.list_repo_files(
        repo_id=ZINC22_REPO_ID,
        repo_type="dataset",
        revision=ZINC22_REVISION,
    )

    parquet_files = sorted(f for f in files if f.endswith(".parquet"))

    if not parquet_files:
        raise RuntimeError(f"No parquet files found in {ZINC22_REPO_ID}")

    print(f"Found {len(parquet_files):,} parquet files in {ZINC22_REPO_ID}")
    return parquet_files


def load_parquet_shard(parquet_path):
    """
    Loads one remote parquet shard from the HF dataset repo.

    This downloads/caches only this shard, not the full ZINC22 dataset.
    """
    url = hf_hub_url(
        repo_id=ZINC22_REPO_ID,
        filename=parquet_path,
        repo_type="dataset",
        revision=ZINC22_REVISION,
    )

    kwargs = {
        "data_files": url,
        "split": "train",
    }

    if HF_TOKEN:
        kwargs["token"] = HF_TOKEN

    try:
        ds = load_dataset("parquet", **kwargs)
    except TypeError:
        # Compatibility with older `datasets` versions.
        if HF_TOKEN:
            kwargs.pop("token", None)
            kwargs["use_auth_token"] = HF_TOKEN
        ds = load_dataset("parquet", **kwargs)

    if SMILES_COLUMN not in ds.column_names:
        raise RuntimeError(
            f"Column {SMILES_COLUMN!r} not found in {parquet_path}. "
            f"Available columns: {ds.column_names}"
        )

    return ds


def sample_zinc22_from_random_parquet_batches(parquet_files, seed):
    """
    Deterministically:
      1. shuffle parquet shards using `seed`
      2. read random parquet batches
      3. sample rows from those shards into a candidate pool
      4. sample exactly N_PER_SEED molecules from the pool

    Returns:
      sampled_rows: list of dicts with smiles + source metadata
      sample_stats: summary metadata
    """
    rng = random.Random(seed)

    shuffled_parquets = list(parquet_files)
    rng.shuffle(shuffled_parquets)

    target_pool_size = max(
        N_PER_SEED,
        int(N_PER_SEED * SAMPLE_POOL_MULTIPLIER),
    )

    pool = []
    used_parquets = []
    parquet_batches_read = 0

    progress = tqdm(
        total=target_pool_size,
        desc=f"Sampling ZINC22 parquet batches seed {seed}",
    )

    for batch_start in range(0, len(shuffled_parquets), PARQUET_BATCH_SIZE):
        batch = shuffled_parquets[
            batch_start : batch_start + PARQUET_BATCH_SIZE
        ]
        parquet_batches_read += 1

        print(
            f"Seed {seed}: reading parquet batch {parquet_batches_read} "
            f"with {len(batch)} shard(s)"
        )

        for parquet_path in batch:
            if len(pool) >= target_pool_size:
                break

            ds = load_parquet_shard(parquet_path)
            n_rows = len(ds)

            remaining = target_pool_size - len(pool)
            n_take = min(n_rows, remaining)

            if n_take <= 0:
                continue

            row_indices = rng.sample(range(n_rows), n_take)
            sampled_smiles = ds.select(row_indices)[SMILES_COLUMN]

            before = len(pool)

            for row_i, smi in zip(row_indices, sampled_smiles):
                if smi and str(smi).strip():
                    pool.append(
                        {
                            "smiles": str(smi),
                            "source_parquet": parquet_path,
                            "source_row": int(row_i),
                        }
                    )

            used_parquets.append(parquet_path)

            progress.update(min(len(pool), target_pool_size) - before)

        if len(pool) >= target_pool_size:
            break

    progress.close()

    if len(pool) < N_PER_SEED:
        raise RuntimeError(
            f"Only collected {len(pool):,} valid molecules, "
            f"but need {N_PER_SEED:,}. Try increasing PARQUET_BATCH_SIZE "
            f"or lowering SAMPLE_POOL_MULTIPLIER."
        )

    sampled_rows = rng.sample(pool, N_PER_SEED)

    sample_stats = {
        "candidate_pool_size": len(pool),
        "parquet_batches_read": parquet_batches_read,
        "parquet_files_read": len(used_parquets),
    }

    return sampled_rows, sample_stats


def canon(smi):
    if not smi:
        return None
    try:
        return standardize_mol(smi)
    except Exception:
        return None


def try_name_smiles(smi):
    try:
        name = name_smiles(smi)
        if not name or not str(name).strip():
            return None
        return str(name)
    except Exception:
        return None


def parallel_map(fn, items, desc, chunksize=10):
    max_workers = max(1, os.cpu_count() - 1)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        return list(
            tqdm(
                executor.map(fn, items, chunksize=chunksize),
                total=len(items),
                desc=desc,
            )
        )


def opsin_one(name):
    if not name:
        return None

    try:
        result = py2opsin.py2opsin([name])
        if isinstance(result, list):
            return result[0] if result else None
        return result
    except Exception:
        return None


def opsin_batch_with_fallback(names):
    """
    Converts generated names to SMILES.

    Empty names are left as None.
    If a batch fails, falls back to one-by-one OPSIN calls so failures
    can still be attributed per molecule.
    """
    raw_smiles = [None] * len(names)

    valid_positions = []
    valid_names = []

    for i, name in enumerate(names):
        if name and str(name).strip():
            valid_positions.append(i)
            valid_names.append(name)

    for start in tqdm(
        range(0, len(valid_names), OPSIN_BATCH_SIZE),
        total=(len(valid_names) + OPSIN_BATCH_SIZE - 1) // OPSIN_BATCH_SIZE,
        desc="Converting names with OPSIN",
    ):
        pos_chunk = valid_positions[start : start + OPSIN_BATCH_SIZE]
        name_chunk = valid_names[start : start + OPSIN_BATCH_SIZE]

        try:
            converted = py2opsin.py2opsin(name_chunk)

            if not isinstance(converted, list):
                converted = [converted]

            if len(converted) != len(name_chunk):
                raise ValueError(
                    f"OPSIN returned {len(converted)} results for "
                    f"{len(name_chunk)} names"
                )

        except Exception:
            converted = [opsin_one(name) for name in name_chunk]

        for pos, smi in zip(pos_chunk, converted):
            raw_smiles[pos] = smi if smi and str(smi).strip() else None

    return raw_smiles


def classify_failure(generated_name, opsin_raw_smiles, opsin_canon_smiles, original_canon_smiles):
    if generated_name is None or not str(generated_name).strip():
        return NO_NAME

    if opsin_raw_smiles is None or not str(opsin_raw_smiles).strip():
        return OPSIN_UNRECOGNIZED

    if opsin_canon_smiles is None:
        return OPSIN_MISMATCH

    if original_canon_smiles is None:
        return OPSIN_MISMATCH

    if opsin_canon_smiles != original_canon_smiles:
        return OPSIN_MISMATCH

    return None


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_seed(parquet_files, seed):
    print(f"\n=== Seed {seed} | N={N_PER_SEED:,} ===")

    sampled_rows, sample_stats = sample_zinc22_from_random_parquet_batches(
        parquet_files,
        seed,
    )

    dataset = [row["smiles"] for row in sampled_rows]

    print(
        f"Seed {seed}: sampled {len(dataset):,} molecules from "
        f"{sample_stats['parquet_files_read']:,} parquet file(s), "
        f"{sample_stats['parquet_batches_read']:,} parquet batch(es), "
        f"candidate pool size={sample_stats['candidate_pool_size']:,}"
    )

    print("Converting SMILES to IUPAC names...")
    predicted_names = parallel_map(
        try_name_smiles,
        dataset,
        desc=f"Naming seed {seed}",
        chunksize=NAME_CHUNKSIZE,
    )

    print("Converting generated names back to SMILES with OPSIN...")
    opsin_raw_smiles = opsin_batch_with_fallback(predicted_names)

    print("Canonicalizing original and OPSIN SMILES...")
    original_canon = parallel_map(
        canon,
        dataset,
        desc=f"Canonicalizing original seed {seed}",
        chunksize=100,
    )

    opsin_canon = parallel_map(
        canon,
        opsin_raw_smiles,
        desc=f"Canonicalizing OPSIN seed {seed}",
        chunksize=100,
    )

    failures = []
    matches = []

    for local_i, (
        source_row,
        original_smiles,
        original_canon_smiles,
        generated_name,
        raw_opsin_smiles,
        canon_opsin_smiles,
    ) in enumerate(
        zip(
            sampled_rows,
            dataset,
            original_canon,
            predicted_names,
            opsin_raw_smiles,
            opsin_canon,
        )
    ):
        failure_reason = classify_failure(
            generated_name,
            raw_opsin_smiles,
            canon_opsin_smiles,
            original_canon_smiles,
        )

        is_match = failure_reason is None
        matches.append(is_match)

        if not is_match:
            failures.append(
                {
                    "seed": seed,
                    "local_position": local_i,
                    "dataset_index": f"{source_row['source_parquet']}:{source_row['source_row']}",
                    "source_parquet": source_row["source_parquet"],
                    "source_row": source_row["source_row"],
                    "failure_reason": failure_reason,
                    "original_smiles": original_smiles,
                    "original_canon_smiles": original_canon_smiles,
                    "generated_name": generated_name,
                    "opsin_raw_smiles": raw_opsin_smiles,
                    "opsin_canon_smiles": canon_opsin_smiles,
                }
            )

    matches = np.array(matches, dtype=bool)
    accuracy = float(np.mean(matches))
    counts = Counter(row["failure_reason"] for row in failures)

    summary = {
        "seed": seed,
        "n": N_PER_SEED,
        "matches": int(matches.sum()),
        "failures": len(failures),
        "accuracy": accuracy,
        "candidate_pool_size": sample_stats["candidate_pool_size"],
        "parquet_batches_read": sample_stats["parquet_batches_read"],
        "parquet_files_read": sample_stats["parquet_files_read"],
        NO_NAME: counts[NO_NAME],
        OPSIN_MISMATCH: counts[OPSIN_MISMATCH],
        OPSIN_UNRECOGNIZED: counts[OPSIN_UNRECOGNIZED],
    }

    print(f"Seed {seed} accuracy: {accuracy:.2%}")
    print(f"Failures: {len(failures):,}")
    print(dict(counts))

    failure_fieldnames = [
        "seed",
        "local_position",
        "dataset_index",
        "source_parquet",
        "source_row",
        "failure_reason",
        "original_smiles",
        "original_canon_smiles",
        "generated_name",
        "opsin_raw_smiles",
        "opsin_canon_smiles",
    ]

    write_csv(
        OUT_DIR / f"failures_seed_{seed}.csv",
        failures,
        failure_fieldnames,
    )

    return summary, failures


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Listing ZINC22 parquet shards...")
    parquet_files = list_zinc22_parquet_files()

    all_summaries = []
    all_failures = []

    for seed in SEEDS:
        summary, failures = evaluate_seed(parquet_files, seed)
        all_summaries.append(summary)
        all_failures.extend(failures)

    summary_fieldnames = [
        "seed",
        "n",
        "matches",
        "failures",
        "accuracy",
        "candidate_pool_size",
        "parquet_batches_read",
        "parquet_files_read",
        NO_NAME,
        OPSIN_MISMATCH,
        OPSIN_UNRECOGNIZED,
    ]

    failure_fieldnames = [
        "seed",
        "local_position",
        "dataset_index",
        "source_parquet",
        "source_row",
        "failure_reason",
        "original_smiles",
        "original_canon_smiles",
        "generated_name",
        "opsin_raw_smiles",
        "opsin_canon_smiles",
    ]

    write_csv(OUT_DIR / "summary.csv", all_summaries, summary_fieldnames)
    write_csv(OUT_DIR / "all_failures.csv", all_failures, failure_fieldnames)

    total_n = sum(row["n"] for row in all_summaries)
    total_matches = sum(row["matches"] for row in all_summaries)
    overall_accuracy = total_matches / total_n

    print("\n=== Overall ===")
    print(f"Total molecules: {total_n:,}")
    print(f"Total matches: {total_matches:,}")
    print(f"Overall accuracy: {overall_accuracy:.2%}")
    print(f"Total failures: {len(all_failures):,}")
    print(f"Wrote results to: {OUT_DIR.resolve()}")

if __name__ == "__main__":
    mp.freeze_support()
    main()
