"""Round-trip tests for test_polycyclic_descriptors.py."""

import random

import pytest
from rdkit import Chem

from openclatura import name_mol, opsin_available
from openclatura.opsin_verify import verify_with_opsin
from roundtrip.roundtrip_helpers import roundtrip_smiles

SMILES = [
    "c1ccccc1",
    "O=S1CC2CCCCC2C1",
    "O=S1CCCC1",
    "C1=CC2CCCCC2C=C1",
    "C1=CC2CCC1C2",
]


@pytest.mark.parametrize("smiles", SMILES)
def test_roundtrip(smiles):
    roundtrip_smiles(smiles)


def test_has_smiles():
    if not SMILES:
        pytest.skip("No SMILES literals found.")


@pytest.mark.skipif(not opsin_available(), reason="OPSIN is unavailable")
@pytest.mark.parametrize("ordering", ("original", "reversed", "shuffled"))
@pytest.mark.parametrize(
    "smiles",
    (
        "CN1CC11C2C3CC2C13",
        "C1C2C1C13COC21CO3",
        "C1NC23COC12C=CC3",
        "C1NC23COC12COC3",
    ),
)
def test_canonical_polycycle_traversal_preserves_numbered_graph(smiles, ordering):
    mol = Chem.MolFromSmiles(smiles)
    order = list(range(mol.GetNumAtoms()))
    if ordering == "reversed":
        order.reverse()
    elif ordering == "shuffled":
        random.Random(19).shuffle(order)
    reference = name_mol(mol)
    result = name_mol(Chem.RenumberAtoms(mol, order))

    assert reference.error is None
    assert result.error is None
    assert result.name == reference.name
    # Check the actual graph, not a neutralized or tautomer-normalized surrogate.
    check = verify_with_opsin(result.name, smiles, standardize_smiles=False)
    assert check.status == "matched", check
    assert check.canonical_roundtrip == Chem.MolToSmiles(mol, isomericSmiles=True)
