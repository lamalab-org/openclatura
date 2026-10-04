"""Regression and feature tests for role-aware chalcogen functional groups."""

import pytest

from openclatura.graph_io import read_smiles
from openclatura.perception import _builtin_perceive_groups
from openclatura.chalcogen_roles import ChalcogenLigandRole, classify_chalcogen_ligand


@pytest.mark.parametrize("smiles", ["CC(=O)OC(C)=O", "O=C(C)OC(C)=O"])
def test_anhydride_perception_is_invariant_to_atom_zero(smiles: str):
    groups = _builtin_perceive_groups(read_smiles(smiles))

    assert [group.key for group in groups].count("anhydride") == 1


@pytest.mark.parametrize(
    ("smiles", "expected_role"),
    [
        ("C=O", ChalcogenLigandRole.DOUBLE_BONDED),
        ("CO", ChalcogenLigandRole.HYDROGEN_BEARING),
        ("C[O-]", ChalcogenLigandRole.ANIONIC),
        ("COC", ChalcogenLigandRole.ORGANIC_LINK),
        ("COO", ChalcogenLigandRole.CHALCOGEN_LINK),
        ("C[SeH]", ChalcogenLigandRole.HYDROGEN_BEARING),
    ],
)
def test_shared_chalcogen_ligand_classifier(smiles: str, expected_role: ChalcogenLigandRole):
    mol = read_smiles(smiles)
    ligand = classify_chalcogen_ligand(mol, 0, 1)

    assert ligand is not None
    assert ligand.role is expected_role


def test_neutral_chalcogen_radical_is_not_hydrogen_bearing():
    mol = read_smiles("C[Se]")

    assert mol.atoms[1].radical_electrons == 1
    assert classify_chalcogen_ligand(mol, 0, 1) is None


@pytest.mark.parametrize(
    ("smiles", "expected_key", "expected_name"),
    [
        ("C[SeH]", "selenol", "methaneselenol"),
        ("C[TeH]", "tellurol", "methanetellurol"),
        ("C[Se-]", "selenolate", "methaneselenolate"),
        ("C[Te-]", "tellurolate", "methanetellurolate"),
        ("CC(=S)C", "thioketone", "propane-2-thione"),
        ("CC(=[Se])C", "selenoketone", "propane-2-selone"),
        ("CC(=[Te])C", "telluroketone", "propane-2-tellone"),
        ("CC(=[Se])", "selenoaldehyde", "ethaneselenal"),
        ("CC(=[Te])", "telluroaldehyde", "ethanetellanal"),
    ],
)
def test_simple_chalcogen_analogues_are_named_from_graph_roles(
    smiles: str, expected_key: str, expected_name: str
):
    import openclatura as oc

    mol = read_smiles(smiles)
    groups = _builtin_perceive_groups(mol)
    matching = [group for group in groups if group.key == expected_key]

    assert len(matching) == 1
    assert matching[0].descriptor is not None
    assert oc.name(smiles).name == expected_name


@pytest.mark.parametrize(
    ("smiles", "expected_key"),
    [("COC", "ether"), ("CSC", "thioether"), ("C[Se]C", "selenoether"), ("C[Te]C", "telluroether")],
)
def test_ether_analogues_share_one_structural_detector(smiles: str, expected_key: str):
    groups = _builtin_perceive_groups(read_smiles(smiles))

    assert [group.key for group in groups].count(expected_key) == 2
    assert all(group.descriptor is not None for group in groups if group.key == expected_key)


@pytest.mark.parametrize(
    ("smiles", "expected"),
    [
        ("CC(=S)O", "ethanethioic O-acid"),
        ("CC(=O)[SH]", "ethanethioic S-acid"),
        ("CC(=S)[SH]", "ethanedithioic acid"),
        ("CC(=[Se])O", "ethaneselenoic O-acid"),
        ("CC(=O)[SeH]", "ethaneselenoic Se-acid"),
        ("CC(=[Se])[SeH]", "ethanediselenoic acid"),
        ("CC(=[Te])O", "ethanetelluroic O-acid"),
        ("CC(=O)[TeH]", "ethanetelluroic Te-acid"),
        ("CC(=[Te])[TeH]", "ethaneditelluroic acid"),
        ("CC(=S)[SeH]", "ethaneselenothioic Se-acid"),
        ("CC(=[Se])[SH]", "ethaneselenothioic S-acid"),
        ("CC(=[Te])[SeH]", "ethaneselenotelluroic Se-acid"),
    ],
)
def test_mixed_chalcogen_acids_preserve_both_sites(smiles: str, expected: str):
    import openclatura as oc

    assert oc.name(smiles).name == expected


@pytest.mark.parametrize(
    ("smiles", "expected"),
    [
        ("CC(=O)SC", "S-methyl ethanethioate"),
        ("CC(=S)OC", "O-methyl ethanethioate"),
        ("CC(=S)SC", "S-methyl ethanedithioate"),
        ("CC(=[Se])OC", "O-methyl ethaneselenoate"),
        ("CC(=O)[Se]C", "Se-methyl ethaneselenoate"),
        ("CC(=[Te])N", "ethanetelluroamide"),
        ("CC(=[Se])NN", "ethaneselenohydrazide"),
        ("NC(=[Se])N", "selenourea"),
        ("NC(=[Te])N", "tellurourea"),
        ("CC(=[Se])Cl", "ethaneselenoyl chloride"),
    ],
)
def test_acyl_derivatives_reuse_the_role_aware_center(smiles: str, expected: str):
    import openclatura as oc

    assert oc.name(smiles).name == expected
