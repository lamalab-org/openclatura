from __future__ import annotations

import pytest
from rdkit import Chem

from openclatura.utils import standardize_mol


@pytest.mark.parametrize(
    "smiles",
    [
        "N=C1NC(=O)CO1",
        "N=C1OC(=O)CNC1=O",
        "[NH2+]=C1O[CH-]C(=O)C=C1",
        "NC1=[NH+]C(=N)C=C([O-])N1",
        "O=C1O[C-]2CC[NH2+]C2=C1",
    ],
)
def test_standardize_mol_is_parseable_and_idempotent(smiles: str):
    standardized = standardize_mol(smiles)

    assert standardized is not None
    assert Chem.MolFromSmiles(standardized) is not None
    assert standardize_mol(standardized) == standardized


@pytest.mark.parametrize(
    ("tautomer_a", "tautomer_b"),
    [
        ("N=c1[nH]c(O)co1", "Nc1nc(O)co1"),
        ("N=c1oc(O)c[nH]c1=O", "Nc1oc(=O)cnc1O"),
    ],
)
def test_standardize_mol_converges_equivalent_heterocyclic_tautomers(tautomer_a: str, tautomer_b: str):
    assert standardize_mol(tautomer_a) == standardize_mol(tautomer_b)
