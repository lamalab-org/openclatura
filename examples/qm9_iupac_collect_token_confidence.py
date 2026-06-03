"""Collect final-name token binding confidence for QM9.

This is the token-audit companion to ``qm9_iupac_collect_warnings.py``.
It intentionally does not call OPSIN.  Instead, it names QM9 molecules with
trace collection enabled and writes one row per emitted final-name token so
fallback/ambiguous/unbound token bindings can be inspected and patched.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import random
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

from bluenamer import DEFAULT_NAMING_ENGINE, NamingRequest

# --- Configuration ---
N_TEST = 5_000
SEED = 42

TOKEN_CONFIDENCE_CSV = "qm9_token_confidence.csv"
MOLECULE_SUMMARY_CSV = "qm9_token_confidence_summary.csv"
FALLBACK_TOKENS_CSV = "qm9_token_confidence_fallbacks.csv"

# Match the OPSIN script default: process the whole dataset unless explicitly
# changed for quicker local iterations.
USE_RANDOM_SAMPLE = False


def _jsonish(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set, frozenset)):
        return ",".join(str(item) for item in value)
    return str(value)


def _trace_steps_with_tokens(result) -> list[tuple[int, str, dict]]:
    steps = []
    for step_idx, step in enumerate(result.decisions or []):
        data = getattr(step, "data", None)
        decision = getattr(step, "decision", "")
        if isinstance(data, dict) and data.get("name_token_spans"):
            steps.append((step_idx, decision, data))
    return steps


def _token_rows_for_result(index: int, smiles: str, result) -> list[dict]:
    rows = []
    for step_idx, decision, data in _trace_steps_with_tokens(result):
        component_name = data.get("name", "")
        for token_idx, token in enumerate(data.get("name_token_spans", ())):
            rows.append(
                {
                    "index": index,
                    "original_smiles": smiles,
                    "predicted_iupac_name": result.name,
                    "naming_error": result.error,
                    "trace_step_index": step_idx,
                    "trace_decision": decision,
                    "component_name": component_name,
                    "token_index": token_idx,
                    "token_text": token.get("text", ""),
                    "token_start": token.get("start", ""),
                    "token_end": token.get("end", ""),
                    "confidence": token.get("confidence", ""),
                    "ownership": token.get("ownership", ""),
                    "source": token.get("source", ""),
                    "token_kind": token.get("token_kind", ""),
                    "grammar_role": token.get("grammar_role", ""),
                    "binding_key": token.get("binding_key", ""),
                    "binding_indices": _jsonish(token.get("binding_indices", ())),
                    "atoms": _jsonish(token.get("atoms", ())),
                    "bonds": _jsonish(token.get("bonds", ())),
                    "charge_atoms": _jsonish(token.get("charge_atoms", ())),
                    "locants": _jsonish(token.get("locants", ())),
                }
            )
    return rows


def _summary_for_result(index: int, smiles: str, result, token_rows: list[dict]) -> dict:
    confidence_counts = Counter(row["confidence"] or "missing" for row in token_rows)
    ownership_counts = Counter(row["ownership"] or "missing" for row in token_rows)
    source_counts = Counter(row["source"] or "missing" for row in token_rows)
    token_kind_counts = Counter(row["token_kind"] or "missing" for row in token_rows)

    fallback_rows = [
        row
        for row in token_rows
        if row["confidence"] == "fallback"
        or row["ownership"] in {"ambiguous", "unbound"}
        or row["source"] in {"broad_fallback", "unresolved"}
    ]
    unbound_rows = [row for row in token_rows if row["ownership"] == "unbound" or row["source"] == "unresolved"]

    return {
        "index": index,
        "original_smiles": smiles,
        "predicted_iupac_name": result.name,
        "naming_error": result.error,
        "token_count": len(token_rows),
        "exact_token_count": confidence_counts.get("exact", 0),
        "derived_token_count": confidence_counts.get("derived", 0),
        "fallback_token_count": confidence_counts.get("fallback", 0),
        "ambiguous_token_count": ownership_counts.get("ambiguous", 0),
        "unbound_token_count": len(unbound_rows),
        "broad_fallback_token_count": source_counts.get("broad_fallback", 0),
        "unresolved_token_count": source_counts.get("unresolved", 0),
        "confidence_counts": ";".join(f"{key}:{value}" for key, value in sorted(confidence_counts.items())),
        "ownership_counts": ";".join(f"{key}:{value}" for key, value in sorted(ownership_counts.items())),
        "source_counts": ";".join(f"{key}:{value}" for key, value in sorted(source_counts.items())),
        "token_kind_counts": ";".join(f"{key}:{value}" for key, value in sorted(token_kind_counts.items())),
        "fallback_tokens": "|".join(row["token_text"] for row in fallback_rows),
        "unbound_tokens": "|".join(row["token_text"] for row in unbound_rows),
    }


def try_name_with_token_confidence(item: tuple[int, str]) -> tuple[dict, list[dict]]:
    index, smiles = item
    result = DEFAULT_NAMING_ENGINE.run(NamingRequest(smiles=smiles, include_trace=True, verify_opsin=False))
    token_rows = _token_rows_for_result(index, smiles, result)
    return _summary_for_result(index, smiles, result, token_rows), token_rows


def main() -> None:
    random.seed(SEED)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    print("Loading QM9 dataset...")
    ds = load_dataset("yairschiff/qm9", split="train")
    all_smiles = list(ds["smiles"])

    if USE_RANDOM_SAMPLE:
        print(f"Sampling {N_TEST} random molecules...")
        indices = random.sample(range(len(all_smiles)), min(N_TEST, len(all_smiles)))
    else:
        print(f"Using all {len(all_smiles)} molecules...")
        indices = list(range(len(all_smiles)))

    dataset = [(idx, all_smiles[idx]) for idx in indices]
    max_workers = max(1, (os.cpu_count() or 2) - 1)

    print("Naming molecules and collecting token confidence metadata...")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(
            tqdm(
                executor.map(try_name_with_token_confidence, dataset, chunksize=10),
                total=len(dataset),
            )
        )

    summary_rows = []
    token_rows = []
    fallback_rows = []

    for summary, tokens in results:
        summary_rows.append(summary)
        token_rows.extend(tokens)
        if summary["fallback_token_count"] or summary["ambiguous_token_count"] or summary["unbound_token_count"]:
            fallback_rows.extend(tokens)

    summary_df = pd.DataFrame(summary_rows)
    token_df = pd.DataFrame(token_rows)
    fallback_df = pd.DataFrame(
        [
            row
            for row in fallback_rows
            if row["confidence"] == "fallback"
            or row["ownership"] in {"ambiguous", "unbound"}
            or row["source"] in {"broad_fallback", "unresolved"}
        ]
    )

    token_df.to_csv(TOKEN_CONFIDENCE_CSV, index=False)
    summary_df.to_csv(MOLECULE_SUMMARY_CSV, index=False)
    fallback_df.to_csv(FALLBACK_TOKENS_CSV, index=False)

    print(f"Token confidence rows written to: {TOKEN_CONFIDENCE_CSV}")
    print(f"Molecule summary written to: {MOLECULE_SUMMARY_CSV}")
    print(f"Fallback token rows written to: {FALLBACK_TOKENS_CSV}")

    print(f"Total molecules: {len(summary_df)}")
    print(f"Total token rows: {len(token_df)}")
    print(
        f"Molecules with fallback tokens: {int((summary_df['fallback_token_count'] > 0).sum()) if not summary_df.empty else 0}"
    )
    print(
        f"Molecules with ambiguous tokens: {int((summary_df['ambiguous_token_count'] > 0).sum()) if not summary_df.empty else 0}"
    )
    print(
        f"Molecules with unbound tokens: {int((summary_df['unbound_token_count'] > 0).sum()) if not summary_df.empty else 0}"
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()
