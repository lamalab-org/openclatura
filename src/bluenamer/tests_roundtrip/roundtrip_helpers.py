"""Helpers for SMILES round-trip testing."""

from __future__ import annotations

import ast
import shutil
import subprocess
from collections.abc import Iterable
from pathlib import Path

import pytest
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

from bluenamer.namer import name_smiles

try:
    import py2opsin
except Exception:  # pragma: no cover - optional dependency
    py2opsin = None


def _java_available() -> bool:
    if shutil.which("java") is None:
        return False
    try:
        subprocess.run(
            ["java", "-version"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, OSError):
        return False
    return True


_JAVA_AVAILABLE = _java_available()


normalizer = rdMolStandardize.Normalizer()
reionizer = rdMolStandardize.Reionizer()
uncharger = rdMolStandardize.Uncharger()
tautomer_enum = rdMolStandardize.TautomerEnumerator()


def standardize_and_canonicalize_tautomer(smi: str) -> str | None:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None

    mol = rdMolStandardize.Cleanup(mol)
    mol = normalizer.normalize(mol)
    mol = reionizer.reionize(mol)
    mol = uncharger.uncharge(mol)
    mol = tautomer_enum.Canonicalize(mol)

    return Chem.MolToSmiles(mol, canonical=True)


def repo_root() -> Path:
    """Return the repository root by walking up to the directory containing pyproject.toml."""

    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("Unable to locate repository root (no pyproject.toml found).")


ORIGINAL_TESTS_DIR = repo_root() / "tests"


def _string_literals_from_file(path: Path) -> Iterable[str]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield node.value


def _is_candidate_smiles(text: str) -> bool:
    if not text or any(ch.isspace() for ch in text):
        return False
    if ":" in text or ";" in text or "$" in text:
        return False
    if text.startswith("http"):
        return False
    return True


def is_valid_smiles(text: str) -> bool:
    if not _is_candidate_smiles(text):
        return False
    mol = Chem.MolFromSmiles(text)
    return mol is not None


def extract_smiles_from_test_file(path: Path) -> list[str]:
    smiles_set: set[str] = set()
    for literal in _string_literals_from_file(path):
        if not is_valid_smiles(literal):
            continue
        canon = standardize_and_canonicalize_tautomer(literal)
        if canon:
            smiles_set.add(canon)
    return sorted(smiles_set)


def roundtrip_smiles(smiles: str) -> None:
    if py2opsin is None:
        pytest.skip("py2opsin is not available")
    if not _JAVA_AVAILABLE:
        pytest.skip("Java runtime not found (OPSIN requires Java)")

    original = standardize_and_canonicalize_tautomer(smiles)
    assert original is not None

    name = name_smiles(original)
    assert name

    result = py2opsin.py2opsin([name])
    assert result and result[0]

    back = standardize_and_canonicalize_tautomer(result[0])
    assert back == original


__all__ = [
    "ORIGINAL_TESTS_DIR",
    "extract_smiles_from_test_file",
    "roundtrip_smiles",
]
