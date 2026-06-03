import contextlib
import multiprocessing as mp
import os
import tempfile
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import py2opsin
from datasets import load_dataset
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from tqdm import tqdm

from bluenamer.namer import name_smiles

normalizer = rdMolStandardize.Normalizer()
reionizer = rdMolStandardize.Reionizer()
uncharger = rdMolStandardize.Uncharger()
fragment_chooser = rdMolStandardize.LargestFragmentChooser()
tautomer_enum = rdMolStandardize.TautomerEnumerator()


# --- Configuration ---

FAILURES_CSV = "qm9_failures.csv"


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

        console_output = "\n".join(x for x in [stdout_text, stderr_text, warning_text] if x)

        if not result:
            return None, console_output or "opsin_returned_empty_result", stdout_text, stderr_text, warning_text

        if result[0] is None:
            return None, console_output or "opsin_returned_none", stdout_text, stderr_text, warning_text

        return result[0], console_output, stdout_text, stderr_text, warning_text

    except Exception as e:
        console_output = "\n".join(x for x in [stdout_text, stderr_text, warning_text] if x)

        err = repr(e)
        if console_output:
            err += f"\nOPSIN_CONSOLE_OR_WARNINGS:\n{console_output}"

        return None, err, stdout_text, stderr_text, warning_text


def standardize_and_canonicalize_tautomer(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None

    mol = rdMolStandardize.Cleanup(mol)
    mol = normalizer.normalize(mol)
    mol = reionizer.reionize(mol)
    mol = uncharger.uncharge(mol)
    mol = tautomer_enum.Canonicalize(mol)

    return Chem.MolToSmiles(mol, canonical=True)


def normalize_tautomer_smiles(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    taut = tautomer_enum.Canonicalize(mol)
    return Chem.MolToSmiles(taut, canonical=True)


te = rdMolStandardize.TautomerEnumerator()


def canon(smi):
    if not smi:
        return None
    try:
        return standardize_and_canonicalize_tautomer(smi)
    except Exception:
        return None


def try_name_smiles(smi):
    try:
        return name_smiles(smi), None
    except Exception as e:
        return None, repr(e)


def main():

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    print("Loading all QM9 molecules")

    ds = load_dataset(
        "yairschiff/qm9",
        split="train",
    )

    # all_smiles = ds["smiles"]
    dataset = list(ds["smiles"])

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
        [predicted == original and predicted is not None for predicted, original in zip(smiles_strings, original_canon)]
    )

    accuracy = np.mean(matches)
    print(f"Accuracy: {accuracy:.2%}")

    print("Collecting failure states...")

    failure_rows = []

    failed_indices = [i for i, match in enumerate(matches) if not match or naming_errors[i] is not None]

    for i in tqdm(failed_indices):
        original_smiles = dataset[i]
        iupac_name = predicted_names[i]

        single_opsin_smiles = None
        single_opsin_error = None

        if naming_errors[i] is None:
            single_opsin_smiles, single_opsin_error, opsin_stdout, opsin_stderr, opsin_warnings = try_opsin_single(
                iupac_name
            )

        failure_rows.append(
            {
                "index": i,
                "original_smiles": original_smiles,
                "original_canonical_smiles": original_canon[i],
                "predicted_iupac_name": iupac_name,
                "naming_error": naming_errors[i],
                "batch_opsin_smiles": raw_smiles_strings[i],
                "batch_predicted_canonical_smiles": smiles_strings[i],
                "single_opsin_smiles": single_opsin_smiles,
                "single_opsin_error": single_opsin_error,
                "matched": bool(matches[i]),
                "batch_error": batch_error,
                "opsin_stdout": opsin_stdout,
                "opsin_stderr": opsin_stderr,
                "opsin_warnings": opsin_warnings,
            }
        )

    failures = pd.DataFrame(failure_rows)
    failures.to_csv(FAILURES_CSV, index=False)

    print(f"Failures written to: {FAILURES_CSV}")
    print(f"Total failures: {len(failures)}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
