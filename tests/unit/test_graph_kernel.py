import pytest

from openclatura.graph_kernel import (
    GraphFace,
    adjacency_from_edges,
    connected_components,
    connected_subsets,
    cycle_rank,
    gf2_basis_insert,
)


def test_graph_face_derives_validated_edges_from_its_cycle():
    face = GraphFace(2, (4, 7, 9))

    assert face.atoms == frozenset({4, 7, 9})
    assert face.edges == frozenset({(4, 7), (7, 9), (4, 9)})


def test_adjacency_and_cycle_rank_reject_unknown_edge_endpoints():
    with pytest.raises(ValueError, match="unknown node"):
        adjacency_from_edges({1, 2}, {(1, 3)})
    with pytest.raises(ValueError, match="unknown node"):
        cycle_rank({1, 2}, {(1, 2), (2, 3)})


def test_connected_components_induces_the_requested_subgraph():
    assert connected_components({1, 2}, {(1, 2), (2, 3)}) == [{1, 2}]


def test_connected_subsets_are_unique_and_size_filtered():
    adjacency = {0: {1}, 1: {0, 2}, 2: {1}}

    assert connected_subsets(adjacency, adjacency, (2,)) == (frozenset({0, 1}), frozenset({1, 2}))


def test_gf2_basis_insert_rejects_dependent_vectors():
    basis = gf2_basis_insert((), 0b011)

    assert basis == (0b011,)
    assert gf2_basis_insert(basis, 0b011) is None
    assert gf2_basis_insert(basis, 0b101) == (0b101, 0b011)
