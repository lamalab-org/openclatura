import multiprocessing as mp
import os
import random
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import py2opsin
from datasets import load_dataset
from tqdm import tqdm
from utils import standardize_mol

from bluenamer.namer import name_smiles

# --- Configuration ---
N_TEST = 1000_000
SEED = 42


def canon(smi):
    if not smi:
        return None
    try:
        return standardize_mol(smi)
    except Exception:
        return None


def try_name_smiles(smi):
    try:
        return name_smiles(smi)
    except Exception:
        return None


def main():
    random.seed(SEED)

    # Optional, but avoids some noisy multiprocessing warnings from tokenizers/datasets
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # 1. Load dataset and select random N using indices
    print(f"Testing on {N_TEST} random molecules from the Pubchem dataset")
    ds = load_dataset(
        "jablonkagroup/pubchem-smiles-molecular-formula",
        split="train",
    )

    indices = random.sample(range(len(ds)), N_TEST)
    dataset = ds.select(indices)["smiles"]

    # Convert to plain list before multiprocessing
    dataset = list(dataset)

    # 2. Parallelize SMILES -> IUPAC
    print("Converting SMILES to IUPAC names...")

    max_workers = max(1, os.cpu_count() - 1)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        predicted_names = list(
            tqdm(
                executor.map(try_name_smiles, dataset, chunksize=10),
                total=len(dataset),
            )
        )

    # 3. Batch Process IUPAC -> SMILES
    print("Converting IUPAC names to SMILES batch...")

    valid_names = [n if n is not None else "" for n in predicted_names]

    try:
        raw_smiles_strings = py2opsin.py2opsin(valid_names)
    except Exception as e:
        print(f"Batch conversion failed: {e}")
        raw_smiles_strings = [None] * len(valid_names)

    # 4. Canonicalize results
    print("Canonicalizing and calculating accuracy...")

    smiles_strings = [canon(smi) for smi in raw_smiles_strings]
    original_canon = [canon(smi) for smi in dataset]

    # 5. Calculate Accuracy
    matches = np.array(
        [predicted == original and predicted is not None for predicted, original in zip(smiles_strings, original_canon)]
    )

    accuracy = np.mean(matches)

    print(f"Accuracy: {accuracy:.2%}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
