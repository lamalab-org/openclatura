"""Graph-built descriptor tests for systematic HW fusion components."""

from openclatura.fusion.descriptor import build_fusion_name_ast, render_fusion_name
from openclatura.fusion.faces import GraphCycle
from openclatura.fusion.registry import fusion_component_registry
from openclatura.molecule import Molecule


def _triazolopyridine_graph() -> tuple[Molecule, tuple[GraphCycle, ...]]:
    mol = Molecule()
    symbols = {0: "N", 1: "N", 2: "C", 3: "N", 4: "C", 5: "C", 6: "C", 7: "C", 8: "C"}
    for atom_id, symbol in symbols.items():
        mol.add_atom(symbol, idx=atom_id, is_aromatic=True)
    edges = (
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 0),
        (3, 5),
        (5, 6),
        (6, 7),
        (7, 8),
        (8, 4),
    )
    double_edges = {frozenset(edge) for edge in ((1, 2), (4, 0), (5, 6), (7, 8))}
    for bond_id, edge in enumerate(edges, start=500):
        mol.add_bond(*edge, idx=bond_id, order=2 if frozenset(edge) in double_edges else 1)
    return mol, (
        GraphCycle.from_atoms((0, 1, 2, 3, 4)),
        GraphCycle.from_atoms((3, 5, 6, 7, 8, 4)),
    )


def test_graph_derived_hw_component_builds_a_fusion_descriptor():
    """An unlisted HW ring participates without an exact whole-parent template."""

    mol, faces = _triazolopyridine_graph()
    registry = fusion_component_registry()

    ast = build_fusion_name_ast(mol, registry.match_faces(mol, faces), registry)

    assert render_fusion_name(ast, registry) == "[1,2,4]triazolo[4,3-a]pyridine"
    assert {registry.spec_for_match(match).parent_name for match in ast.component_occurrences} == {
        "[1,2,4]triazole",
        "pyridine",
    }
