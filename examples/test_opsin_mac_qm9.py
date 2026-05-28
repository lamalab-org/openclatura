import os
import random
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import py2opsin
from datasets import load_dataset
from structure_to_iupac.namer import name_smiles
from rdkit.Chem import CanonSmiles
from tqdm import tqdm
import pandas as pd


# --- Configuration ---
N_TEST = 5_000
SEED = 42
FAILURES_CSV = "qm9_failures.csv"


def canon(smi):
    if not smi:
        return None
    try:
        return CanonSmiles(smi)
    except Exception:
        return None


def try_name_smiles_with_error(smi):
    try:
        name = name_smiles(smi)
        return name, None
    except Exception as e:
        return None, repr(e)


def try_opsin_with_error(name):
    if not name:
        return None, "empty_or_missing_iupac_name"

    try:
        result = py2opsin.py2opsin([name])

        if not result or result[0] is None:
            return None, "opsin_returned_none"

        return result[0], None

    except Exception as e:
        return None, repr(e)


def main():
    random.seed(SEED)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    print("Loading QM9 dataset...")

    ds = load_dataset(
        "yairschiff/qm9",
        split="train",
    )

    all_smiles = ds["smiles"]
    dataset = list(all_smiles)
   
    dataset = random.sample(dataset, min(N_TEST, len(dataset)))

    print(f"Processing {len(dataset)} molecules...")

    max_workers = max(1, os.cpu_count() - 1)

    # 1. SMILES -> IUPAC
    print("Converting SMILES to IUPAC names...")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        name_results = list(
            tqdm(
                executor.map(try_name_smiles_with_error, dataset, chunksize=10),
                total=len(dataset),
            )
        )

    predicted_names = [x[0] for x in name_results]
    naming_errors = [x[1] for x in name_results]

    # 2. IUPAC -> SMILES with per-molecule OPSIN errors
    print("Converting IUPAC names to SMILES with failure collection...")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        opsin_results = list(
            tqdm(
                executor.map(try_opsin_with_error, predicted_names, chunksize=10),
                total=len(predicted_names),
            )
        )

    raw_smiles_strings = [x[0] for x in opsin_results]
    opsin_errors = [x[1] for x in opsin_results]

    # 3. Canonicalize
    print("Canonicalizing and calculating accuracy...")

    smiles_strings = [canon(smi) for smi in raw_smiles_strings]
    original_canon = [canon(smi) for smi in dataset]

    matches = np.array(
        [
            predicted == original and predicted is not None
            for predicted, original in zip(smiles_strings, original_canon)
        ]
    )

    accuracy = np.mean(matches)

    print(f"Accuracy: {accuracy:.2%}")

    # 4. Collect failures
    failure_rows = []

    for i, (
        original_smiles,
        original_c,
        iupac_name,
        naming_error,
        opsin_smiles,
        opsin_error,
        predicted_c,
        match,
    ) in enumerate(
        zip(
            dataset,
            original_canon,
            predicted_names,
            naming_errors,
            raw_smiles_strings,
            opsin_errors,
            smiles_strings,
            matches,
        )
    ):
        if naming_error is not None or opsin_error is not None or not match:
            failure_rows.append(
                {
                    "index": i,
                    "original_smiles": original_smiles,
                    "original_canonical_smiles": original_c,
                    "predicted_iupac_name": iupac_name,
                    "naming_error": naming_error,
                    "opsin_smiles": opsin_smiles,
                    "opsin_error": opsin_error,
                    "predicted_canonical_smiles": predicted_c,
                    "matched": bool(match),
                }
            )

    failures = pd.DataFrame(failure_rows)
    failures.to_csv(FAILURES_CSV, index=False)

    print(f"Failures written to: {FAILURES_CSV}")
    print(f"Total failures: {len(failures)}")


if __name__ == "__main__":
    mp.freeze_support()
    main()