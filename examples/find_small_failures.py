import numpy as np
import py2opsin
from datasets import load_dataset
from bluenamer.namer import name_smiles
from rdkit.Chem import CanonSmiles
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

def canon(smi):
    if not smi: return None
    try:
        return CanonSmiles(smi)        
    except:
        return None

def try_name_smiles(smi):
    try:
        return name_smiles(smi)
    except Exception:
        return None

def run_pipeline():
    # 1. Load and Filter Data
    print("Loading and filtering dataset...")
    raw_data = load_dataset('jablonkagroup/pubchem-smiles-molecular-formula')["train"]["smiles"][2_300_000:2_600_000]
    
    # Filter: Only molecules < 30 characters
    filtered_dataset = [s for s in raw_data if len(s) < 30]
    print(f"Filtered {len(raw_data)} down to {len(filtered_dataset)} molecules.")

    # 2. Parallelize SMILES -> IUPAC
    print("Converting SMILES to IUPAC names...")
    with ProcessPoolExecutor() as executor:
        predicted_names = list(tqdm(executor.map(try_name_smiles, filtered_dataset), total=len(filtered_dataset)))

    # 3. Batch Process IUPAC -> SMILES
    print("Converting IUPAC names back to SMILES...")
    # Replace None with empty strings for py2opsin batch processing
    query_names = [n if n is not None else "" for n in predicted_names]
    
    try:
        reconstructed_smiles = py2opsin.py2opsin(query_names)
    except Exception as e:
        print(f"Batch conversion failed: {e}")
        reconstructed_smiles = [None] * len(query_names)

    # 4. Identify Failure Cases
    print("Analyzing results and identifying failures...")
    failures = []
    success_count = 0

    for i in range(len(filtered_dataset)):
        orig_smi = filtered_dataset[i]
        iupac_name = predicted_names[i]
        recon_smi = reconstructed_smiles[i]
        
        # Canonicalize for fair comparison
        orig_canon = canon(orig_smi)
        recon_canon = canon(recon_smi)

        failure_reason = None

        if not iupac_name:
            failure_reason = "SMILES_TO_NAME_FAILED"
        elif not recon_smi:
            failure_reason = "NAME_TO_SMILES_FAILED"
        elif orig_canon != recon_canon:
            failure_reason = "SMILES_MISMATCH"
        
        if failure_reason:
            failures.append({
                "original": orig_smi,
                "iupac": iupac_name,
                "reconstructed": recon_smi,
                "reason": failure_reason
            })
        else:
            success_count += 1

    # 5. Reporting
    accuracy = success_count / len(filtered_dataset)
    print(f"\n--- Results ---")
    print(f"Total Processed: {len(filtered_dataset)}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Total Failures: {len(failures)}")
    
    # Categorize failures
    reasons = [f['reason'] for f in failures]
    unique_reasons = set(reasons)
    for r in unique_reasons:
        print(f" - {r}: {reasons.count(r)}")

    # Show top 5 examples of failures
    if failures:
        print("\n--- Example Failures ---")
        for f in failures[:60]:
            print(f"Reason: {f['reason']}")
            print(f"  Orig: {f['original']}")
            print(f"  Name: {f['iupac']}")
            print(f"  Back: {f['reconstructed']}\n")

if __name__ == "__main__":
    run_pipeline()