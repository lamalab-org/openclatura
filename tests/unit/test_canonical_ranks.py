import pytest

from openclatura.canonical_ranks import canonical_ranks
from openclatura.graph_io import read_smiles


def test_subgraph_ranks_ignore_atoms_outside_the_selected_parent():
    parent = read_smiles("c1ccccc1")
    substituted = read_smiles("c1ccccc1Cl")
    parent_atoms = frozenset(range(6))

    assert canonical_ranks(parent) == canonical_ranks(substituted, parent_atoms)


def test_subgraph_ranks_reject_unknown_atoms():
    mol = read_smiles("CC")

    with pytest.raises(KeyError, match="Unknown atom ids"):
        canonical_ranks(mol, {0, 99})


def test_subgraph_ranking_does_not_replace_the_whole_graph_cache():
    mol = read_smiles("CCCl")

    whole = canonical_ranks(mol)
    subset = canonical_ranks(mol, {0, 1})

    assert set(whole) == {0, 1, 2}
    assert set(subset) == {0, 1}
    assert canonical_ranks(mol) is whole
