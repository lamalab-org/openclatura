import csv
import multiprocessing as mp
import os
import random
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import py2opsin
from datasets import load_dataset
from tqdm import tqdm
from bluenamer.utils import standardize_mol

from bluenamer.namer import name_smiles

# --- Configuration ---
N_PER_SEED = 100_000
SEEDS = [42, 17, 87, 5, 63]
OUT_DIR = Path("eval_failures/pubchem")

NAME_CHUNKSIZE = 10
OPSIN_BATCH_SIZE = 1000


NO_NAME = "no name generated"
OPSIN_MISMATCH = "opsin smiles mismatch"
OPSIN_UNRECOGNIZED = "opsin - name is not recognized"


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
                raise ValueError(f"OPSIN returned {len(converted)} results for {len(name_chunk)} names")

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


def evaluate_seed(ds, seed):
    print(f"\n=== Seed {seed} | N={N_PER_SEED:,} ===")

    rng = random.Random(seed)

    indices = rng.sample(range(len(ds)), N_PER_SEED)
    dataset = list(ds.select(indices)["smiles"])

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
        dataset_index,
        original_smiles,
        original_canon_smiles,
        generated_name,
        raw_opsin_smiles,
        canon_opsin_smiles,
    ) in enumerate(
        zip(
            indices,
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
                    "dataset_index": dataset_index,
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

    print("Loading PubChem dataset once...")
    ds = load_dataset(
        "jablonkagroup/pubchem-smiles-molecular-formula",
        split="train",
    )

    all_summaries = []
    all_failures = []

    for seed in SEEDS:
        summary, failures = evaluate_seed(ds, seed)
        all_summaries.append(summary)
        all_failures.extend(failures)

    summary_fieldnames = [
        "seed",
        "n",
        "matches",
        "failures",
        "accuracy",
        NO_NAME,
        OPSIN_MISMATCH,
        OPSIN_UNRECOGNIZED,
    ]

    failure_fieldnames = [
        "seed",
        "local_position",
        "dataset_index",
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
