import pytest

from openclatura.graph_kernel import (
    GraphFace,
    adjacency_from_edges,
    biconnected_edge_components,
    connected_components,
    connected_subsets,
    cycle_rank,
    gf2_basis_insert,
)


def test_biconnected_edge_components_separate_spiro_and_linked_rings():
    edges = {
        (0, 1),
        (1, 2),
        (0, 2),
        (2, 3),
        (3, 4),
        (2, 4),
        (4, 5),
        (5, 6),
        (6, 7),
        (5, 7),
    }

    blocks = biconnected_edge_components(range(8), edges)

    assert {frozenset(block) for block in blocks} == {
        frozenset({(0, 1), (1, 2), (0, 2)}),
        frozenset({(2, 3), (3, 4), (2, 4)}),
        frozenset({(4, 5)}),
        frozenset({(5, 6), (6, 7), (5, 7)}),
    }


def test_biconnected_edge_components_keep_fused_rings_in_one_block():
    edges = {(0, 1), (1, 2), (2, 3), (0, 3), (2, 4), (4, 5), (3, 5)}

    assert biconnected_edge_components(range(6), edges) == (frozenset(edges),)


def test_biconnected_edge_components_handle_large_fused_graph_without_recursion():
    ring_count = 1_500
    edges = {
        tuple(sorted(edge))
        for ring in range(ring_count)
        for edge in ((ring, ring + 1), (ring + 1, ring + 2), (ring, ring + 2))
    }

    blocks = biconnected_edge_components(range(ring_count + 2), edges)

    assert blocks == (frozenset(edges),)


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
