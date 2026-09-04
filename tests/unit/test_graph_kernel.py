import pytest

from openclatura.graph_kernel import GraphFace, adjacency_from_edges, connected_components, cycle_rank


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
