import random
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import py2opsin
from datasets import load_dataset
from rdkit.Chem import CanonSmiles
from tqdm import tqdm

from openclatura.namer import name_smiles


def canon(smi):
    if not smi:
        return None
    try:
        return CanonSmiles(smi)
    except Exception:
        return None


def try_name_smiles(smi):
    try:
        return name_smiles(smi)
    except Exception:
        return None


# --- Configuration ---
N_TEST = 50_000

# 1. Load dataset and select random N using indices
print(f"Sampling {N_TEST} random molecules using .select()...")
ds = load_dataset("jablonkagroup/pubchem-smiles-molecular-formula", split="train")

# Generate N random distinct integers and select
indices = random.sample(range(len(ds)), N_TEST)
dataset = ds.select(indices)["smiles"]

# 2. Parallelize SMILES -> IUPAC (CPU Bound)
print("Converting SMILES to IUPAC names...")
with ProcessPoolExecutor() as executor:
    predicted_names = list(tqdm(executor.map(try_name_smiles, dataset), total=len(dataset)))

# 3. Batch Process IUPAC -> SMILES
print("Converting IUPAC names to SMILES (Batch)...")
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
matches = np.array([s == o and s is not None for s, o in zip(smiles_strings, original_canon)])
accuracy = np.mean(matches)

print(f"Accuracy: {accuracy:.2%}")
