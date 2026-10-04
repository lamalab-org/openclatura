"""Regression and feature tests for role-aware chalcogen functional groups."""

import pytest

from openclatura.graph_io import read_smiles
from openclatura.perception import _builtin_perceive_groups


@pytest.mark.parametrize("smiles", ["CC(=O)OC(C)=O", "O=C(C)OC(C)=O"])
def test_anhydride_perception_is_invariant_to_atom_zero(smiles: str):
    groups = _builtin_perceive_groups(read_smiles(smiles))

    assert [group.key for group in groups].count("anhydride") == 1
