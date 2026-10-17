"""An -ium ring atom with no ring double bond cites its own indicated hydrogen."""

import pytest

import openclatura
from openclatura import name_smiles, opsin_available, verify_with_opsin

# OPSIN reads a ring -ium as one more bond to that atom and gives it a ring
# double bond unless the name cites hydrogen there. A protonated ring oxygen or
# sulfur kept the neutral parent's 4H instead, which no longer has a double bond
# to give up, and a saturated ring nitrogen lost the hydrogen its neutral parent
# cites as 4(1H) (lamalab-org/openclatura#134).
CASES = {
    "OC(=O)C1=CC(=O)c2c([OH+]1)cccc2": "4-oxo-1H-1-benzopyran-1-ium-2-carboxylic acid",
    "O=C1C=C[OH+]c2ccccc12": "1H-1-benzopyran-1-ium-4-one",
    "O=C1C=C[SH+]C=C1": "1H-thiopyran-1-ium-4-one",
    "O=C1C=C[OH+]C=C1": "1H-pyran-1-ium-4-one",
    "O=C1C=C[OH+]C=C1O": "3-hydroxy-1H-pyran-1-ium-4-one",
    "O=C1C=C[SH+]c2ccccc12": "1H-1-benzothiopyran-1-ium-4-one",
    "OC(=O)C1=CC(=O)C=C[OH+]1": "4-oxo-1H-pyran-1-ium-2-carboxylic acid",
    "O=C1C=C[NH2+]C=C1": "1H-pyridin-1-ium-4-one",
    "O=C1C=C[NH2+]C=C1O": "3-hydroxy-1H-pyridin-1-ium-4-one",
    "O=C1C=C[NH+](C)C=C1": "1-methyl-1H-pyridin-1-ium-4-one",
    "C[N+]1(C)C=CC(=O)C=C1": "1,1-dimethyl-1H-pyridin-1-ium-4-one",
    "OC(=O)C1=CC(=O)C=C[NH2+]1": "4-oxo-1H-pyridin-1-ium-2-carboxylic acid",
    "O=C1N=C[NH2+]C=C1": "1H-pyrimidin-1-ium-4-one",
}

# An -ium atom that keeps a ring double bond states its hydrogen itself, and the
# neutral parents keep their own citation.
GUARDS = {
    "c1cc[nH+]cc1": "pyridin-1-ium",
    "C[n+]1ccccc1": "1-methylpyridin-1-ium",
    "C[N+]1=CC=C(C=C1)O": "1-methylpyridin-1-ium-4-ol",
    "c1c[nH+]c[nH]1": "1H-imidazol-3-ium",
    "N=C1C=CNC=[NH+]1": "1H-pyrimidin-3-ium-4-imine",
    "O=C1C=C[NH+]=CC1": "pyridin-1(3H)-ium-4-one",
    "O=c1cc[nH]c[nH+]1": "pyrimidin-3(1H)-ium-4-one",
    "c1cc[o+]cc1": "1-oxacyclohexa-1,3,5-trien-1-ium",
    "c1ccc2[o+]cccc2c1": "benzo[b]pyran-1-ium",
    "C[O+]1C=CC(=O)C=C1": "1-methyl-4H-pyran-1-ium-4-one",
    "[O-][O+]1C=CC(=S)c2ccccc21": "1-oxido-4-thioxo-4H-1-benzopyran-1-ium",
    "O=C1C[OH+]C=C1": "furan-1(2H)-ium-3-one",
    "C1=C[NH2+]C=CC1": "1,4-dihydropyridin-1-ium",
    "O=C1CC[NH2+]CC1": "piperidinium-4-one",
    "O=C1C=COC=C1": "4H-pyran-4-one",
    "O=c1ccoc2ccccc12": "4H-1-benzopyran-4-one",
    "O=C1C=CNC=C1": "pyridin-4(1H)-one",
    "O=C1C=CN(C)C=C1": "1-methylpyridin-4(1H)-one",
}


@pytest.mark.parametrize("smiles", sorted(CASES))
def test_saturated_ium_atom_cites_its_hydrogen(smiles):
    assert name_smiles(smiles) == CASES[smiles]


@pytest.mark.parametrize("smiles", sorted(CASES))
def test_ium_hydrogen_passes_the_final_audit(smiles):
    """The traced path audits the parent's binding term against the final name."""

    result = openclatura.name(smiles, include_trace=True)
    assert result.error is None
    assert result.name == CASES[smiles]


@pytest.mark.parametrize("smiles", sorted(GUARDS))
def test_ium_hydrogen_guards(smiles):
    assert name_smiles(smiles) == GUARDS[smiles]


@pytest.mark.opsin
@pytest.mark.parametrize("smiles", sorted(CASES | GUARDS))
def test_ium_hydrogen_names_round_trip(smiles):
    if not opsin_available():
        pytest.skip("OPSIN and Java are required")
    check = verify_with_opsin((CASES | GUARDS)[smiles], smiles)
    assert check.status == "matched", check.to_dict()
