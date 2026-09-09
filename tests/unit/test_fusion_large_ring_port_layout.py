"""Two-port macrocycles retain OPSIN's angular fusion numbering geometry."""

import random
import xml.etree.ElementTree as ET
from dataclasses import asdict

import pytest
from rdkit import Chem

from openclatura import name_mol, opsin_available, verify_with_opsin
from openclatura.fusion.descriptor import iter_fusion_name_asts, render_fusion_name
from openclatura.fusion.faces import select_bounded_face_model, typed_face_model
from openclatura.fusion.layout import (
    LayoutSearchBudgetExceeded,
    _audit_layout,
    intrinsic_fused_layouts,
    preferred_intrinsic_layouts,
)
from openclatura.fusion.model import FusionConfirmed, FusionMode
from openclatura.fusion.numbering import _numbering_from_layout, completed_system_numbering_selection
from openclatura.fusion.planner import plan_fusion_parent
from openclatura.fusion.registry import fusion_component_registry
from openclatura.graph_io import read_rdkit_mol, read_smiles
from openclatura.molecule import Molecule

ORIGINAL = "C=C(C)[C@H]1CC[C@@]23O[C@@H]2[C@@H](C/C(C)=C2/C(=O)C=C(C)[C@@]2(O)CC1)OC3=O"
BASE_ATOMS = frozenset({3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 16, 17, 19, 21, 22})
BASE_NAME = "cyclopenta[1',2':1,2]cyclododeca[6,7-b]oxirene"
EXPECTED = {
    7: "1",
    6: "1a",
    5: "2",
    4: "3",
    3: "4",
    22: "5",
    21: "6",
    19: "6a",
    17: "7",
    16: "8",
    14: "9",
    13: "9a",
    11: "10",
    10: "11",
    9: "12",
    8: "12a",
}
OPSIN_PATHS = {
    (7, 6, 5, 4, 3, 22, 21, 19, 17, 16, 14, 13, 11, 10, 9, 8),
    (14, 16, 17, 19, 21, 22, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13),
}


def _selection(mol, atoms):
    bounded = select_bounded_face_model(mol, atoms)
    assert bounded is not None
    model = typed_face_model(mol, bounded)
    layouts = preferred_intrinsic_layouts(model)
    assert layouts
    orders = {face.id: face.atom_cycle for face in model.faces}
    for layout in layouts:
        positions = {atom: (x, y) for atom, x, y in layout.atom_positions}
        assert set(positions) == set(atoms)
        assert _audit_layout(model, orders, positions)
    selection = completed_system_numbering_selection(mol, bounded, face_model=model, layouts=layouts)
    assert selection.accepted
    return bounded, model, layouts, selection


def _angular_case(size, distance=None, *, extra_face=False):
    distance = size // 2 - 1 if distance is None else distance
    cycles = [
        tuple(range(size)),
        (0, 1, size),
        (distance, distance + 1, size + 1, size + 2, size + 3),
    ]
    if extra_face:
        cycles.append((size + 1, size + 2, size + 4, size + 5, size + 6, size + 7))
    mol = Molecule()
    for atom in sorted(set().union(*map(set, cycles))):
        mol.add_atom("O" if atom == size else "C", idx=atom)
    edges = {tuple(sorted((left, right))) for cycle in cycles for left, right in zip(cycle, cycle[1:] + cycle[:1])}
    for edge_id, edge in enumerate(sorted(edges)):
        mol.add_bond(*edge, idx=edge_id)
    return mol


def test_macro88584_expanded_geometry_reproduces_both_opsin_perimeters():
    mol = read_smiles(ORIGINAL)
    bounded, model, layouts, selection = _selection(mol, BASE_ATOMS)
    assert sorted(face.size for face in model.faces) == [3, 5, 12]
    assert {layout.orientation_score for layout in layouts} == {(0, -2, -6, 2, -8)}
    candidates = [
        _numbering_from_layout(mol, bounded, model, layout, index, set(bounded.fusion_atoms))
        for index, layout in enumerate(layouts)
    ]
    assert all(candidate is not None for candidate in candidates)
    assert {candidate.perimeter for candidate in candidates} == OPSIN_PATHS
    assert [candidate.string_map for candidate in selection.accepted] == [EXPECTED]


def test_macro88584_base_passes_the_full_existing_planner_audit():
    result = plan_fusion_parent(read_smiles(ORIGINAL), BASE_ATOMS, mode=FusionMode.GENERAL)
    assert isinstance(result, FusionConfirmed)
    assert result.plan.rendered_base_name == BASE_NAME
    assert not result.plan.audit.errors
    assert result.plan.numbering.string_input_locant_maps() == (EXPECTED,)


@pytest.mark.parametrize("seed", [None, 0, 17, 53, 91, 137])
def test_large_ring_ports_are_invariant_to_atom_and_bond_relabelling(seed):
    graph = Chem.MolFromSmiles(ORIGINAL)
    order = list(range(graph.GetNumAtoms()))
    if seed is None:
        order.reverse()
    else:
        random.Random(seed).shuffle(order)
    inverse = {old: new for new, old in enumerate(order)}
    original = read_rdkit_mol(Chem.RenumberAtoms(graph, order))
    # Also change which fusion bond is selected as the proxy entrance.
    mol = Molecule()
    for atom in original.atoms.values():
        mol.add_atom(**asdict(atom))
    for bond in original.bonds.values():
        mol.add_bond(**{**asdict(bond), "idx": 1000 - bond.idx})
    bounded, _, _, selection = _selection(mol, {inverse[atom] for atom in BASE_ATOMS})
    assert bounded.audit.ok
    assert [{old: candidate.string_map[inverse[old]] for old in BASE_ATOMS} for candidate in selection.accepted] == [
        EXPECTED
    ]


@pytest.mark.parametrize("size", [10, 12, 14, 16, 18, 20])
@pytest.mark.parametrize("extra_face", [False, True])
def test_reduction_is_generic_over_even_sizes_and_short_ring_tree_extensions(size, extra_face):
    mol = _angular_case(size, extra_face=extra_face)
    _, model, layouts, _ = _selection(mol, mol.atoms)
    assert len(model.faces) == (4 if extra_face else 3)
    assert any(shape.startswith(f"two-port-{size}:") for layout in layouts for _, shape in layout.face_shapes)


@pytest.mark.parametrize("size,distance", [(11, 4), (12, 1), (12, 2), (12, 4), (12, 6)])
def test_unproved_large_ring_direction_signatures_still_abstain(size, distance):
    mol = _angular_case(size, distance)
    bounded = select_bounded_face_model(mol, mol.atoms)
    assert bounded is not None
    assert intrinsic_fused_layouts(typed_face_model(mol, bounded)) == ()


@pytest.mark.parametrize("limits", [{"search_budget": 1}, {"max_layouts": 1}])
def test_proxy_layout_search_retains_existing_hard_limits(limits):
    mol = read_smiles(ORIGINAL)
    bounded = select_bounded_face_model(mol, BASE_ATOMS)
    with pytest.raises(LayoutSearchBudgetExceeded):
        intrinsic_fused_layouts(typed_face_model(mol, bounded), **limits)


def _assert_opsin_locant_graph(name, mol, bounded, locants, tmp_path):
    if not opsin_available():
        pytest.skip("OPSIN and Java are required")
    from py2opsin import py2opsin

    cml = py2opsin(name, output_format="CML", tmp_fpath=str(tmp_path / "opsin-input.txt"))
    root = ET.fromstring(cml)
    ns = {"c": "http://www.xml-cml.org/schema"}
    atoms = root.findall(".//c:atom", ns)
    labels = {
        atom.attrib["id"]: next(
            (
                label.attrib["value"]
                for label in atom.findall("c:label", ns)
                if label.attrib["value"] in locants.values()
            ),
            None,
        )
        for atom in atoms
    }
    assert {labels[atom.attrib["id"]]: atom.attrib["elementType"] for atom in atoms if labels[atom.attrib["id"]]} == {
        locant: mol.atoms[atom].symbol for atom, locant in locants.items()
    }
    actual_edges = {
        frozenset(labels[atom] for atom in bond.attrib["atomRefs2"].split())
        for bond in root.findall(".//c:bond", ns)
        if all(labels[atom] for atom in bond.attrib["atomRefs2"].split())
    }
    assert actual_edges == {frozenset(locants[atom] for atom in edge) for edge in bounded.edge_ids}


@pytest.mark.opsin
def test_macro88584_base_citation_has_exact_opsin_locant_labelled_skeleton(tmp_path):
    mol = read_smiles(ORIGINAL)
    bounded, _, _, selection = _selection(mol, BASE_ATOMS)
    _assert_opsin_locant_graph(BASE_NAME, mol, bounded, selection.accepted[0].string_map, tmp_path)


@pytest.mark.opsin
def test_macro88584_production_name_roundtrips_exactly_with_stereochemistry():
    if not opsin_available():
        pytest.skip("OPSIN and Java are required")
    result = name_mol(Chem.MolFromSmiles(ORIGINAL), fusion_mode=FusionMode.GENERAL)
    assert result.error is None
    assert BASE_NAME in result.name
    check = verify_with_opsin(result.name, ORIGINAL, standardize_smiles=False)
    assert check.status == "matched", check.to_dict()
    assert check.canonical_original == check.canonical_roundtrip


@pytest.mark.opsin
@pytest.mark.parametrize("size", [10, 12, 14, 16, 18, 20])
@pytest.mark.parametrize("extra_face", [False, True])
def test_generic_two_port_base_citations_match_opsin_numbering(size, extra_face, tmp_path):
    mol = _angular_case(size, extra_face=extra_face)
    bounded, _, _, selection = _selection(mol, mol.atoms)
    registry = fusion_component_registry()
    ast = next(iter_fusion_name_asts(mol, registry.match_faces(mol, bounded), registry))
    name = render_fusion_name(ast, registry)
    _assert_opsin_locant_graph(name, mol, bounded, selection.accepted[0].string_map, tmp_path)
