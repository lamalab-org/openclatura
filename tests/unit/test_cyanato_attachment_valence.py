"""(Iso)(thio)cyanato prefixes are used only when the linking atom has one other neighbour."""

import pytest

from openclatura import name_smiles, opsin_available, verify_with_opsin

# The prefixes -SC#N, -OC#N, -N=C=S and -N=C=O express one bond from the linking
# atom. A four-coordinate S, an oxonium O or an iminium N has more, and those
# neighbours were left out of the name (lamalab-org/openclatura#131).
CASES = {
    "CC(=O)NS1(CCCC1)C#N": "N-(1-cyanothiolan-1-yl)acetamide",
    "N#CS1(C)CCCC1": "1-methylthiolane-1-carbonitrile",
    "CS(C)(C)C#N": "(trimethyl-lambda^4-sulfanyl)methanenitrile",
    "C[S+](C)C#N": "(dimethylsulfaniumyl)methanenitrile",
    "CC[S@+](C)C#N": "((S)-ethyl(methyl)sulfaniumyl)methanenitrile",
    "N#CS1=CC=CC=C1": "1lambda^4-thiacyclohexa-1,3,5-triene-1-carbonitrile",
    "C[O+](C)C#N": "(dimethyloxidaniumyl)methanenitrile",
    "C[N+](C)=C=S": "N,N-dimethyl-1-thioxomethaniminium",
    "C[N+](C)=C=O": "N,N-dimethyl-1-oxomethaniminium",
}

# A divalent linking atom keeps the prefix.
GUARDS = {
    "CSC#N": "thiocyanatomethane",
    "N#CSc1ccccc1": "thiocyanatobenzene",
    "N#CSCCSC#N": "1,2-dithiocyanatoethane",
    "COC#N": "cyanatomethane",
    "N#COc1ccccc1": "cyanatobenzene",
    "CN=C=S": "isothiocyanatomethane",
    "S=C=Nc1ccccc1": "isothiocyanatobenzene",
    "O=C=NCCCCCCN=C=O": "1,6-diisocyanatohexane",
}


@pytest.mark.parametrize("smiles", sorted(CASES))
def test_crowded_linking_atom_is_not_a_cyanato_prefix(smiles):
    assert name_smiles(smiles) == CASES[smiles]


@pytest.mark.parametrize("smiles", sorted(GUARDS))
def test_divalent_linking_atom_keeps_the_prefix(smiles):
    assert name_smiles(smiles) == GUARDS[smiles]


@pytest.mark.opsin
@pytest.mark.parametrize("smiles", sorted(CASES | GUARDS))
def test_cyanato_names_round_trip(smiles):
    if not opsin_available():
        pytest.skip("OPSIN and Java are required")
    check = verify_with_opsin((CASES | GUARDS)[smiles], smiles)
    assert check.status == "matched", check.to_dict()
