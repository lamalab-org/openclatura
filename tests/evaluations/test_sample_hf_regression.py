"""Tests for deterministic Hugging Face evaluation sampling."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_sampler():
    path = Path(__file__).parents[2] / "evaluations" / "sample_hf_regression.py"
    spec = importlib.util.spec_from_file_location("sample_hf_regression", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SAMPLER = _load_sampler()


def test_existing_smiles_ignores_generated_hf_shards(tmp_path):
    output_dir = tmp_path / "pubchem"
    output_dir.mkdir()
    (output_dir / "pubchem_seed5_100000_input.jsonl").write_text(
        json.dumps({"smiles": "paper-smiles"}) + "\n",
        encoding="utf-8",
    )
    (output_dir / "pubchem_hf_seed101_100000_input.jsonl").write_text(
        json.dumps({"smiles": "generated-smiles"}) + "\n",
        encoding="utf-8",
    )

    assert SAMPLER._existing_smiles(tmp_path, "pubchem") == {"paper-smiles"}
