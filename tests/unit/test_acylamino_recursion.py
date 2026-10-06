"""A RecursionError in the acylamino prefix ends naming instead of starting a fallback."""

import subprocess
import sys

import pytest

from openclatura import namer
from openclatura.graph_io import read_smiles

# N-acetylglycine: from the glycine parent, the nitrogen's prefix is acetamido.
ACETYLGLYCINE = "CC(=O)NCC(=O)O"
NITROGEN = 3
PARENT = {4, 5, 6, 7}


def test_acylamino_prefix_names_the_acid():
    mol = read_smiles(ACETYLGLYCINE)
    assert namer._acylamino_amido_prefix(mol, NITROGEN, PARENT, 4) == "acetamido"


def test_acylamino_prefix_declines_when_the_acid_cannot_be_named(monkeypatch):
    def fail(*args, **kwargs):
        raise ValueError("unsupported acid")

    monkeypatch.setattr(namer, "name_component", fail)
    mol = read_smiles(ACETYLGLYCINE)
    assert namer._acylamino_amido_prefix(mol, NITROGEN, PARENT, 4) is None


def test_acylamino_prefix_lets_recursion_error_through(monkeypatch):
    def fail(*args, **kwargs):
        raise RecursionError("maximum recursion depth exceeded")

    monkeypatch.setattr(namer, "name_component", fail)
    mol = read_smiles(ACETYLGLYCINE)
    with pytest.raises(RecursionError):
        namer._acylamino_amido_prefix(mol, NITROGEN, PARENT, 4)


def test_peptide_past_the_recursion_limit_fails_fast():
    """Each residue nests one acylamino prefix. Swallowing the RecursionError
    made every level retry another path: (Ala)15 under a limit of 90 took
    about 7 s to fail, against 0.06 s to name it at the default limit
    (lamalab-org/openclatura#136)."""

    program = """
import sys
import time

import openclatura

openclatura.name("CC(=O)O")
smiles = "N[C@@H](C)C(=O)" * 15 + "O"
sys.setrecursionlimit(90)
started = time.perf_counter()
result = openclatura.name(smiles)
elapsed = time.perf_counter() - started
sys.setrecursionlimit(1000)
print(result.ok, "RecursionError" in (result.error or ""), round(elapsed, 3))
"""
    completed = subprocess.run(
        [sys.executable, "-c", program],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert completed.returncode == 0, completed.stderr
    ok, recursion_error, elapsed = completed.stdout.split()
    assert (ok, recursion_error) == ("False", "True")
    assert float(elapsed) < 2.0
