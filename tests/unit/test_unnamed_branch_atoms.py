"""A branch that cannot be named fails the substituent it hangs from instead of dropping out of the name."""

import pytest

import openclatura
from openclatura import name_smiles, opsin_available, verify_with_opsin
from openclatura.graph_io import read_smiles
from openclatura.namer import name_subgraph

# Germyl, stannyl and plumbyl groups had no prefix, and on a chain substituent
# the chain was named without them: 2-ethynylpyridine for the trimethylstannyl
# alkyne (lamalab-org/openclatura#129).
CASES = {
    "C[Sn](C)(C)C#Cc1ccccn1": "2-(2-(trimethylstannyl)ethynyl)pyridine",
    "C[Ge](C)(C)C=Cc1cccs1": "2-(2-(trimethylgermyl)ethenyl)thiophene",
    "C[Sn](C)(C)CCc1ccccn1": "2-(2-(trimethylstannyl)ethyl)pyridine",
    "C[Sn](C)(C)Cc1ccccn1": "2-((trimethylstannyl)methyl)pyridine",
    "C[Sn](C)(C)CC(=O)OC": "methyl 2-(trimethylstannyl)acetate",
    "C[Sn](C)(C)c1ccccn1": "2-(trimethylstannyl)pyridine",
    "C[Ge](C)(C)c1ccccn1": "2-(trimethylgermyl)pyridine",
    "C[Pb](C)(C)CCc1ccccn1": "2-(2-(trimethylplumbyl)ethyl)pyridine",
    "[SnH3]CCc1ccccn1": "2-(2-stannylethyl)pyridine",
    "CCCC[Sn](CCCC)(CCCC)c1ccccn1": "2-(tributylstannyl)pyridine",
    "C[Sn](C)(C)Oc1ccccn1": "2-((trimethylstannyl)oxy)pyridine",
    "C[Sn](C)(C)CC[Si](C)(C)c1ccccn1": "2-(dimethyl(2-(trimethylstannyl)ethyl)silyl)pyridine",
    # PubChem CID 12000583: the =CH-Sn(CH3)3 end of the chain was dropped.
    "C/C=C(\\C)/[C@@H]([C@@H](C)[C@H](C/C=C(\\C)/C=C(\\C)/C/C(=C/[Sn](C)(C)C)/C)O[Si](C1=CC=CC=C1)"
    "(C2=CC=CC=C2)C(C)(C)C)O[Si](CCC(C(C(F)(F)F)(F)F)(F)F)(C(C)C)C(C)C": (
        "((tert-butyl)(((2E,4R,5S,6S,8E,10E,13E)-4-(((3,3,4,4,5,5,5-heptafluoropentyl)bis(propan-2-yl)silyl)oxy)"
        "-3,5,9,11,13-pentamethyl-14-(trimethylstannyl)tetradeca-2,8,10,13-tetraen-6-yl)oxy)(phenyl)silyl)benzene"
    ),
}

# Silicon is named, and a stannane or bismuthane parent is unaffected.
GUARDS = {
    "C[Si](C)(C)C=Cc1cccs1": "2-(2-(trimethylsilyl)ethenyl)thiophene",
    "C[Si](C)(C)C#Cc1ccccn1": "2-(2-(trimethylsilyl)ethynyl)pyridine",
    "C[Si](C)(C)CCOc1ccccn1": "2-((2-(trimethylsilyl)ethyl)oxy)pyridine",
    "C[Sn](C)(C)C#Cc1ccccc1": "trimethyl(2-phenylethynyl)stannane",
    "C[Bi](C)CCc1ccccc1": "dimethyl(2-phenylethyl)bismuthane",
    "CCOCCc1ccccn1": "2-(2-ethoxyethyl)pyridine",
}

# Bismuth and aluminium have no substituent prefix. These were named without
# the metal (2-ethylpyridine, 2-ethoxypyridine, ...).
UNNAMEABLE = [
    "C[Bi](C)CCc1ccccn1",
    "C[Bi](C)C#Cc1ccccn1",
    "C[Bi](C)CCOc1ccccn1",
    "C[Bi](C)CC[Si](C)(C)c1ccccn1",
    "C[Al](C)C=Cc1cccs1",
]


@pytest.mark.parametrize("smiles", sorted(CASES))
def test_group_14_metal_prefixes(smiles):
    assert name_smiles(smiles) == CASES[smiles]


@pytest.mark.parametrize("smiles", sorted(GUARDS))
def test_branch_naming_guards(smiles):
    assert name_smiles(smiles) == GUARDS[smiles]


@pytest.mark.parametrize("smiles", UNNAMEABLE)
def test_unnamed_nested_branch_fails_the_name(smiles):
    result = openclatura.name(smiles)
    assert not result.ok
    assert "UnnamedAtomError" in result.error


@pytest.mark.parametrize("smiles", ["[2H]OCCc1ccccn1", "[2H]C([2H])([2H])Oc1ccccn1", "C[Si](C)(C)[2H]"])
def test_explicit_hydrogen_branch_does_not_fail_the_name(smiles):
    # An explicit hydrogen has no prefix; the component audit skips it as well.
    assert openclatura.name(smiles).ok


def test_chain_branch_with_unnamed_branch_has_no_name():
    # The ethyl branch on the pyridine carries a dimethylbismuthanyl group,
    # which has no prefix.
    mol = read_smiles("C[Bi](C)CCc1ccccn1")
    assert name_subgraph(mol, 4, set(range(5, 11)), upstream_atom=5) == ""


def test_heteroatom_branch_with_unnamed_ligand_has_no_name():
    # Not "hydroxy": the oxygen's ligand is there, it just has no name.
    mol = read_smiles("C[Bi](C)Oc1ccccn1")
    assert name_subgraph(mol, 3, set(range(4, 10)), upstream_atom=4) == ""


@pytest.mark.opsin
@pytest.mark.parametrize("smiles", sorted(CASES | GUARDS))
def test_branch_names_round_trip(smiles):
    if not opsin_available():
        pytest.skip("OPSIN and Java are required")
    check = verify_with_opsin((CASES | GUARDS)[smiles], smiles)
    assert check.status == "matched", check.to_dict()
