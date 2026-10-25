"""Functional-class azine names keep both C=N descriptors where OPSIN finds them."""

import pytest
from rdkit import Chem

from openclatura import name_mol, opsin_available, verify_with_opsin

# ``<aldehyde> <ylidene>hydrazone`` gave the aldehyde side's descriptor the
# locant 1, which benzaldehyde and ...carbaldehyde parents do not have, and
# dropped the ylidene side's own descriptor (lamalab-org/openclatura#133).
CASES = {
    r"NC(=N/N=C\c1ccccc1)SC": "(Z)-benzaldehyde (amino)(methylsulfanyl)methylidenehydrazone",
    r"CC1(C(=N/N=C/N2CCOCC2)C(C1=O)(C)C)C": (
        "(E)-morpholine-4-carbaldehyde 2,2,4,4-tetramethyl-3-oxocyclobutylidenehydrazone"
    ),
    r"c1ccccc1/C=N/N=C(/C)CC": "(E)-benzaldehyde [(Z)-butan-2-ylidene]hydrazone",
    r"c1ccccc1/C=N/N=C(\C)CC": "(E)-benzaldehyde [(E)-butan-2-ylidene]hydrazone",
    r"c1ccccc1/C=N/N=C\c1ccccc1": "(Z)-benzaldehyde [(E)-benzylidene]hydrazone",
    r"C(=N/N=C/c1ccccc1)c1ccccc1": "(E)-benzaldehyde benzylidenehydrazone",
    r"c1ccncc1/C=N/N=C(/C)CC": "(E)-pyridine-3-carbaldehyde [(Z)-butan-2-ylidene]hydrazone",
    r"c1ccccc1/C=N/N=C1CCCCC1": "(E)-benzaldehyde cyclohexylidenehydrazone",
    r"Clc1ccc(/C=N/N=C\c2ccccc2)cc1": "(Z)-benzaldehyde (1E)-(4-chlorophenyl)methylidenehydrazone",
    r"C/C=N/N=C(/C)CC": "(2Z)-butan-2-one [(E)-ethan-1-ylidene]hydrazone",
    r"C/C=N/N=C\C": "(1Z)-ethanal [(E)-ethan-1-ylidene]hydrazone",
}

# A parent that numbers its C=N carbon keeps the locant; symmetric azines and
# plain hydrazones are named as before.
GUARDS = {
    r"N/N=C(/C)c1ccccc1": "(1Z)-1-phenylethan-1-one hydrazone",
    r"CN(C)/N=C/c1ccccc1": "(E)-N,N-dimethylbenzaldehyde hydrazone",
    r"CC/C(C)=N/N=C1CCCCC1": "(2E)-butan-2-one cyclohexylidenehydrazone",
    r"CCCC/C=N/N=C(C)C": "(1E)-pentanal propan-2-ylidenehydrazone",
    r"CC1CCCC/C1=N/N=C/CCCCCCCCC": "(1E)-decanal (1Z)-2-methylcyclohexylidenehydrazone",
    r"c1ccccc1/C=N/N=C/c1ccccc1": "1,2-bis[(E)-benzylidene]hydrazine",
    r"CC/C(C)=N/N=C(\C)CC": "1,2-bis[(E)-butan-2-ylidene]hydrazine",
    r"NC(=NN=Cc1ccccc1)SC": "benzaldehyde (amino)(methylsulfanyl)methylidenehydrazone",
}


def _name(smiles, reverse):
    mol = Chem.MolFromSmiles(smiles)
    if reverse:
        mol = Chem.RenumberAtoms(mol, list(reversed(range(mol.GetNumAtoms()))))
    result = name_mol(mol)
    assert result.error is None
    return result.name


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("smiles", sorted(CASES))
def test_azine_hydrazone_stereo(smiles, reverse):
    assert _name(smiles, reverse) == CASES[smiles]


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("smiles", sorted(GUARDS))
def test_azine_hydrazone_stereo_guards(smiles, reverse):
    assert _name(smiles, reverse) == GUARDS[smiles]


@pytest.mark.opsin
@pytest.mark.parametrize("smiles", sorted(CASES | GUARDS))
def test_azine_hydrazone_names_round_trip(smiles):
    if not opsin_available():
        pytest.skip("OPSIN and Java are required")
    check = verify_with_opsin((CASES | GUARDS)[smiles], smiles)
    assert check.status == "matched", check.to_dict()
