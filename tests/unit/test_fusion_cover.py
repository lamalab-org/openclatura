from openclatura.fusion.cover import (
    audit_component_cover,
    block_cut_sets,
    build_component_cover_graph,
    build_cover_graph,
    build_cover_proof,
    component_scope,
    cycle_order_hint,
)


def test_component_scopes_build_typed_overlap_interface_and_exact_cover_audit():
    left = component_scope("left", range(6), [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5)])
    right = component_scope("right", (4, 5, 6, 7, 8, 9), [(4, 5), (4, 6), (6, 7), (7, 8), (8, 9), (5, 9)])
    target_edges = left.edges | right.edges

    audit = audit_component_cover((left, right), target_atom_ids=range(10), target_edges=target_edges)

    assert audit.ok
    assert audit.proof.kind == "tree"
    assert len(audit.graph.interfaces) == 1
    interface = audit.graph.interfaces[0]
    assert interface.shared_atom_ids == frozenset({4, 5})
    assert interface.shared_edges == frozenset({(4, 5)})


def test_one_atom_spiro_overlap_does_not_become_fusion_interface():
    left = component_scope("left", (0, 1, 2), [(0, 1), (1, 2), (0, 2)])
    right = component_scope("right", (2, 3, 4), [(2, 3), (3, 4), (2, 4)])

    graph = build_component_cover_graph((left, right))
    audit = audit_component_cover(
        (left, right),
        target_atom_ids=range(5),
        target_edges=left.edges | right.edges,
    )

    assert graph.interfaces == ()
    assert not audit.ok
    assert audit.proof.kind == "disconnected"


def test_cover_proof_classifies_tree_cycle_cactus_and_complex_graphs():
    tree = build_cover_proof(build_cover_graph(("a", "b", "c"), (("a", "b"), ("b", "c"))))
    cycle = build_cover_proof(build_cover_graph(("a", "b", "c"), (("a", "b"), ("b", "c"), ("c", "a"))))
    cactus = build_cover_proof(
        build_cover_graph(
            ("a", "b", "c", "d", "e"),
            (("a", "b"), ("b", "c"), ("c", "a"), ("c", "d"), ("d", "e"), ("e", "c")),
        )
    )
    complex_proof = build_cover_proof(
        build_cover_graph(
            ("a", "b", "c", "d"),
            (("a", "b"), ("a", "c"), ("a", "d"), ("b", "c"), ("b", "d"), ("c", "d")),
        )
    )

    assert (tree.kind, tree.cycle_rank, tree.articulation_nodes) == ("tree", 0, ("b",))
    assert (cycle.kind, cycle.cycle_rank) == ("cycle", 1)
    assert cycle.cycle_proofs[0].ordered_nodes == ("a", "b", "c")
    assert (cactus.kind, cactus.cycle_rank, cactus.articulation_nodes) == ("cactus", 2, ("c",))
    assert complex_proof.kind == "complex"


def test_cycle_order_hints_and_general_block_cut_sets_are_deterministic():
    cycle_proof = build_cover_proof(build_cover_graph((0, 1, 2), ((0, 1), (1, 2), (2, 0))))
    complete = build_cover_graph(
        (0, 1, 2, 3),
        ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)),
    )
    complete_proof = build_cover_proof(complete)

    assert cycle_order_hint(cycle_proof) == {
        (0, 2): 0,
        (0, 1): 1,
        (1, 0): 0,
        (1, 2): 1,
        (2, 1): 0,
        (2, 0): 1,
    }
    general = complete_proof.blocks[0]
    cuts = block_cut_sets(complete, general, max_sets=3)
    assert len(cuts) == 3
    assert all(len(cut) == 3 for cut in cuts)


def test_component_cover_audit_rejects_missing_edges_and_triple_coverage():
    first = component_scope("a", (0, 1, 2), [(0, 1), (1, 2), (0, 2)])
    second = component_scope("b", (0, 1, 3), [(0, 1), (1, 3), (0, 3)])
    third = component_scope("c", (0, 1, 4), [(0, 1), (1, 4), (0, 4)])
    target_edges = first.edges | second.edges | third.edges | {(4, 5)}

    audit = audit_component_cover(
        (first, second, third),
        target_atom_ids=range(6),
        target_edges=target_edges,
    )

    assert not audit.ok
    assert "component edges do not exactly reconstruct the target" in audit.errors
    assert "a molecular edge belongs to more than two components" in audit.errors
