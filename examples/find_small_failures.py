import csv
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import py2opsin
from datasets import load_dataset
from tqdm import tqdm
from utils import standardize_mol

from bluenamer.namer import name_smiles
from bluenamer.resonance_compare import equivalent_smiles


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


def failure_bucket(failure):
    name = (failure["iupac"] or "").lower()
    original = failure["original"].lower()
    if not name:
        return "blank-name"
    if "hydrazine" in name or "hydrazone" in name or "hydrazinyl" in name:
        return "nitrogen-hydrazine/hydrazone"
    if "diazo" in name or "azido" in name or "iminio" in name or ("imino" in name and "[n-]" in original):
        return "nitrogen-chain/diazo"
    if "sulfonimidoyl" in name or "iminosulfanyl" in name or "sulfoamino" in name or "sulfon" in name:
        return "sulfur-nitrogen-oxo"
    if "lambda" in name or "phosph" in name or "selen" in name or "chloranyl" in name:
        return "hypervalent-p/s/se/halogen"
    if "tricyclo" in name or "tetracyclo" in name or "spiro" in name or "bicyclo" in name:
        return "polycycle"
    if "olate" in name or "oxido" in name or "ium" in name or "thiolate" in name:
        return "formal-charge"
    if "carbonothioyl" in name or "thioxo" in name or "thiol" in name:
        return "thio-carbonyl"
    if any(marker in name for marker in ("(1e)", "(1z)", "(2e)", "(2z)", "(1r)", "(1s)", "(2r)", "(2s)")):
        return "stereo"
    return "other"


def run_pipeline():
    # 1. Load and Filter Data
    print("Loading and filtering dataset...")
    raw_data = load_dataset("jablonkagroup/pubchem-smiles-molecular-formula")["train"]["smiles"][2_300_000:2_600_000]

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
        elif orig_canon != recon_canon and not equivalent_smiles(orig_smi, recon_smi):
            failure_reason = "SMILES_MISMATCH"

        if failure_reason:
            failures.append(
                {"original": orig_smi, "iupac": iupac_name, "reconstructed": recon_smi, "reason": failure_reason}
            )
        else:
            success_count += 1

    # 5. Reporting
    accuracy = success_count / len(filtered_dataset)
    print("\n--- Results ---")
    print(f"Total Processed: {len(filtered_dataset)}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Total Failures: {len(failures)}")

    # Categorize failures
    reasons = [f["reason"] for f in failures]
    unique_reasons = set(reasons)
    for r in unique_reasons:
        print(f" - {r}: {reasons.count(r)}")

    # Show top 5 examples of failures
    if failures:
        bucket_counts = Counter(failure_bucket(f) for f in failures)
        print("\n--- Failure Buckets ---")
        for bucket, count in bucket_counts.most_common():
            print(f" - {bucket}: {count}")

        substring_counts = Counter()
        substrings = [
            "hydrazine",
            "hydrazone",
            "hydrazinyl",
            "diazo",
            "azido",
            "imino",
            "iminio",
            "sulfonimidoyl",
            "iminosulfanyl",
            "sulfoamino",
            "lambda",
            "phosph",
            "selen",
            "chloranyl",
            "tricyclo",
            "tetracyclo",
            "spiro",
            "bicyclo",
            "olate",
            "oxido",
            "thiolate",
            "carbonothioyl",
            "thioxo",
            "methanehydrazine",
            "dihydrazine",
        ]
        for f in failures:
            lower_name = (f["iupac"] or "").lower()
            for substring in substrings:
                if substring in lower_name:
                    substring_counts[substring] += 1
        print("\n--- Failure Name Substrings ---")
        for substring, count in substring_counts.most_common():
            print(f" - {substring}: {count}")

        output_path = "small_failures.csv"
        with open(output_path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["reason", "bucket", "original", "iupac", "reconstructed"])
            writer.writeheader()
            for f in failures:
                writer.writerow({**f, "bucket": failure_bucket(f)})
        print(f"\nWrote failure details to {output_path}")

        print("\n--- Example Failures ---")
        for f in failures[:60]:
            print(f"Reason: {f['reason']}")
            print(f"  Orig: {f['original']}")
            print(f"  Name: {f['iupac']}")
            print(f"  Back: {f['reconstructed']}\n")


if __name__ == "__main__":
    run_pipeline()
