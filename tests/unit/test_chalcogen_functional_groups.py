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
