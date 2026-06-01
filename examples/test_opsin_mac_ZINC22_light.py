import os
import random
import tempfile
import contextlib
import warnings
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import py2opsin
from datasets import load_dataset
from huggingface_hub import list_repo_files
from bluenamer.namer import name_smiles
from rdkit.Chem import CanonSmiles
from tqdm import tqdm
from utils import standardize_mol


# --- Configuration ---
N_TEST = 100_000
SEED = 42
FAILURES_CSV = "ZINC22_light_failures.csv"
DATASET_REPO = "chandar-lab/ZINC_22"


@contextlib.contextmanager
def capture_fds():
    old_stdout_fd = os.dup(1)
    old_stderr_fd = os.dup(2)

    stdout_tmp = tempfile.TemporaryFile(mode="w+b")
    stderr_tmp = tempfile.TemporaryFile(mode="w+b")

    try:
        os.dup2(stdout_tmp.fileno(), 1)
        os.dup2(stderr_tmp.fileno(), 2)
        yield stdout_tmp, stderr_tmp
    finally:
        os.dup2(old_stdout_fd, 1)
        os.dup2(old_stderr_fd, 2)

        os.close(old_stdout_fd)
        os.close(old_stderr_fd)

        stdout_tmp.close()
        stderr_tmp.close()


def read_tmp_file(tmp):
    tmp.flush()
    tmp.seek(0)
    return tmp.read().decode("utf-8", errors="replace").strip()


def try_opsin_single(name):
    if not name:
        return None, "empty_or_missing_iupac_name", "", "", ""

    stdout_text = ""
    stderr_text = ""
    warning_text = ""

    try:
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")

            with capture_fds() as (stdout_tmp, stderr_tmp):
                result = py2opsin.py2opsin([name])

                stdout_text = read_tmp_file(stdout_tmp)
                stderr_text = read_tmp_file(stderr_tmp)

            warning_text = "\n".join(str(w.message) for w in caught_warnings).strip()

        console_output = "\n".join(
            x for x in [stdout_text, stderr_text, warning_text] if x
        )

        if not result:
            return None, console_output or "opsin_returned_empty_result", stdout_text, stderr_text, warning_text

        if result[0] is None:
            return None, console_output or "opsin_returned_none", stdout_text, stderr_text, warning_text

        return result[0], console_output, stdout_text, stderr_text, warning_text

    except Exception as e:
        console_output = "\n".join(
            x for x in [stdout_text, stderr_text, warning_text] if x
        )

        err = repr(e)
        if console_output:
            err += f"\nOPSIN_CONSOLE_OR_WARNINGS:\n{console_output}"

        return None, err, stdout_text, stderr_text, warning_text


def canon(smi):
    if not smi:
        return None
    try:
        return standardize_mol(smi)
    except Exception:
        return None


def try_name_smiles(smi):
    try:
        return name_smiles(smi), None
    except Exception as e:
        return None, repr(e)


def pick_random_parquet(repo, seed):
    files = list_repo_files(repo_id=repo, repo_type="dataset")
    parquet_files = sorted(f for f in files if f.endswith(".parquet"))
    if not parquet_files:
        raise RuntimeError("No parquet files found in dataset repo")
    return random.Random(seed).choice(parquet_files)


def main():
    random.seed(SEED)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTHONIOENCODING"] = "utf-8"
    os.environ["JAVA_TOOL_OPTIONS"] = "-Dfile.encoding=UTF-8"

    parquet_file = pick_random_parquet(DATASET_REPO, SEED)
    data_file = f"hf://datasets/{DATASET_REPO}/{parquet_file}"

    print(f"Sampling {N_TEST} random molecules from: {parquet_file}")

    ds = load_dataset(
        "parquet",
        data_files=data_file,
        split="train",
    )

    if "SMILES" in ds.column_names:
        smiles_col = "SMILES"
    elif "smiles" in ds.column_names:
        smiles_col = "smiles"
    else:
        smiles_col = ds.column_names[0]

    n = min(N_TEST, len(ds))
    indices = random.sample(range(len(ds)), n)
    dataset = list(ds.select(indices)[smiles_col])

    print("Converting SMILES to IUPAC names...")

    max_workers = max(1, os.cpu_count() - 1)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        name_results = list(
            tqdm(
                executor.map(try_name_smiles, dataset, chunksize=10),
                total=len(dataset),
            )
        )

    predicted_names = [x[0] for x in name_results]
    naming_errors = [x[1] for x in name_results]

    print("Converting IUPAC names to SMILES batch...")

    valid_names = [n if n is not None else "" for n in predicted_names]

    batch_error = None
    try:
        raw_smiles_strings = py2opsin.py2opsin(valid_names)
    except Exception as e:
        batch_error = repr(e)
        print(f"Batch conversion failed: {batch_error}")
        raw_smiles_strings = [None] * len(valid_names)

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

    print("Collecting failure states...")

    failure_rows = []

    failed_indices = [
        i for i, match in enumerate(matches)
        if not match or naming_errors[i] is not None
    ]

    for i in tqdm(failed_indices):
        iupac_name = predicted_names[i]

        single_opsin_smiles = None
        single_opsin_error = None
        opsin_stdout = None
        opsin_stderr = None
        opsin_warnings = None

        if naming_errors[i] is None:
            (
                single_opsin_smiles,
                single_opsin_error,
                opsin_stdout,
                opsin_stderr,
                opsin_warnings,
            ) = try_opsin_single(iupac_name)

        failure_rows.append(
            {
                "index": i,
                "original_smiles": dataset[i],
                "original_canonical_smiles": original_canon[i],
                "predicted_iupac_name": iupac_name,
                "naming_error": naming_errors[i],
                "batch_opsin_smiles": raw_smiles_strings[i],
                "batch_predicted_canonical_smiles": smiles_strings[i],
                "single_opsin_smiles": single_opsin_smiles,
                "single_opsin_error": single_opsin_error,
                "opsin_stdout": opsin_stdout,
                "opsin_stderr": opsin_stderr,
                "opsin_warnings": opsin_warnings,
                "matched": bool(matches[i]),
                "batch_error": batch_error,
                "source_parquet": parquet_file,
            }
        )

    failures = pd.DataFrame(failure_rows)
    failures.to_csv(FAILURES_CSV, index=False)

    print(f"Failures written to: {FAILURES_CSV}")
    print(f"Total failures: {len(failures)}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
