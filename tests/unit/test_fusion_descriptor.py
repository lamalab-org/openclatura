"""Graph-built tests for bounded systematic fusion descriptors."""

from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from dataclasses import replace

import pytest

from openclatura.fusion import (
    FusionDescriptorError,
    build_fusion_name_ast,
    component_sides,
    render_fusion_name,
)
from openclatura.fusion.cover import FusionInterface
from openclatura.fusion.descriptor import _multiparent_root_sets, classify_ordered_fusion_interface
from openclatura.fusion.faces import GraphCycle
from openclatura.fusion.model import FusionComponentMatch, FusionConfirmed, FusionDescriptor, FusionJoinKind, FusionMode
from openclatura.fusion.planner import plan_fusion_parent
from openclatura.fusion.registry import FusionComponentRegistry, fusion_component_registry
from openclatura.fusion.rules import component_interface_orbit
from openclatura.hantzsch_widman import mancude_bond_orders
from openclatura.molecule import Molecule
from openclatura.naming_data import load_json_table
from openclatura.opsin_verify import opsin_available, verify_with_opsin

OccurrenceLocant = tuple[int, str]
FusionPair = tuple[OccurrenceLocant, OccurrenceLocant]


class _DisjointSet:
    def __init__(self, values: Iterable[OccurrenceLocant]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: OccurrenceLocant) -> OccurrenceLocant:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: OccurrenceLocant, right: OccurrenceLocant) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def _component_graph(
    component_keys: tuple[str, ...],
    fused_atoms: tuple[FusionPair, ...],
    *,
    atom_id_order: tuple[int, ...] | None = None,
) -> tuple[Molecule, tuple[GraphCycle, ...]]:
    """Construct a molecular graph by identifying local component atoms."""

    registry = fusion_component_registry()
    registered = tuple(registry.get(key) for key in component_keys)
    assert all(component is not None for component in registered)
    specs = tuple(component.spec for component in registered if component is not None)
    local_atoms = tuple((occurrence, locant) for occurrence, spec in enumerate(specs) for locant in spec.locants)
    sets = _DisjointSet(local_atoms)
    for left, right in fused_atoms:
        sets.union(left, right)

    roots = tuple(dict.fromkeys(sets.find(atom) for atom in local_atoms))
    ids = atom_id_order or tuple(range(len(roots)))
    if len(ids) != len(roots) or len(set(ids)) != len(ids):
        raise ValueError("atom_id_order must be a permutation with one id per fused graph atom")
    atom_by_root = dict(zip(roots, ids, strict=True))
    atom_by_local = {local: atom_by_root[sets.find(local)] for local in local_atoms}

    mol = Molecule()
    symbols: dict[int, str] = {}
    for occurrence, spec in enumerate(specs):
        for atom in spec.atoms:
            atom_id = atom_by_local[(occurrence, atom.locant)]
            previous = symbols.setdefault(atom_id, atom.symbol)
            if previous != atom.symbol:
                raise ValueError("fused component atoms disagree on their element")
    for atom_id, symbol in sorted(symbols.items()):
        mol.add_atom(
            symbol,
            idx=atom_id,
            is_aromatic=symbol not in {"O", "S", "Se", "Te"},
        )

    edge_orders: dict[tuple[int, int], int] = {}
    for occurrence, spec in enumerate(specs):
        generated_orders = (
            mancude_bond_orders([atom.symbol for atom in spec.atoms])
            if spec.template.family == "generated_hw_monocycle"
            else [1] * len(spec.bonds)
        )
        for bond, order in zip(spec.bonds, generated_orders, strict=True):
            endpoints = tuple(atom_by_local[(occurrence, locant)] for locant in bond.locants)
            edge = tuple(sorted(endpoints))
            edge_orders[edge] = max(edge_orders.get(edge, 1), order)
    for edge, order in sorted(edge_orders.items()):
        mol.add_bond(*edge, order=order, idx=100 + len(mol.bonds))

    faces = tuple(
        GraphCycle.from_atoms(atom_by_local[(occurrence, locant)] for locant in ring)
        for occurrence, spec in enumerate(specs)
        for ring in spec.rings
    )
    return mol, faces


def _build(
    component_keys: tuple[str, ...],
    fused_atoms: tuple[FusionPair, ...],
    *,
    atom_id_order: tuple[int, ...] | None = None,
    registry: FusionComponentRegistry | None = None,
    experimental: bool = False,
    atomic_components_only: bool = False,
):
    registry = registry or fusion_component_registry()
    mol, faces = _component_graph(component_keys, fused_atoms, atom_id_order=atom_id_order)
    matches = registry.match_faces(mol, faces)
    if atomic_components_only:
        allowed = set(component_keys)
        matches = tuple(match for match in matches if match.spec_key in allowed and len(match.covered_face_ids) == 1)
    kwargs = (
        {"cover_kinds": ("tree", "multiparent"), "join_kinds": ("ortho", "ortho_peri", "higher_order")}
        if experimental
        else {}
    )
    ast = build_fusion_name_ast(mol, matches, registry, **kwargs)
    return mol, ast, render_fusion_name(ast, registry)


def test_component_side_letters_follow_the_directed_peripheral_walk():
    furan = fusion_component_registry().by_key["furan"].spec

    sides = component_sides(furan)

    assert tuple((side.letter, side.start_locant, side.end_locant) for side in sides) == (
        ("a", "1", "2"),
        ("b", "2", "3"),
        ("c", "3", "4"),
        ("d", "4", "5"),
        ("e", "5", "1"),
    )


@pytest.mark.parametrize(
    ("component_keys", "fused_atoms", "expected", "parent_key"),
    [
        (
            ("furan", "thiophene"),
            (((1, "2"), (0, "2")), ((1, "3"), (0, "3"))),
            "thieno[2,3-b]furan",
            "furan",
        ),
        (
            ("thiazole", "imidazole"),
            (((1, "2"), (0, "2")), ((1, "1"), (0, "3"))),
            "imidazo[2,1-b][1,3]thiazole",
            "thiazole",
        ),
    ],
)
def test_two_component_ortho_fusion_is_derived_from_exact_shared_edges(
    component_keys: tuple[str, ...],
    fused_atoms: tuple[FusionPair, ...],
    expected: str,
    parent_key: str,
):
    mol, ast, rendered = _build(component_keys, fused_atoms)

    assert rendered == expected
    assert ast.plan_kind == "two_component"
    assert ast.component_occurrences[ast.parent_occurrences[0]].spec_key == parent_key
    assert len(ast.joins) == 1
    join = ast.joins[0]
    assert len(join.shared_input_atoms) == 2
    assert join.shared_input_bonds == frozenset(
        bond.idx for bond in mol.bonds.values() if frozenset((bond.u, bond.v)) == join.shared_input_atoms
    )


def test_shared_classifier_builds_an_ordered_ortho_peri_interface_from_graph_paths():
    registry = fusion_component_registry()
    attached_spec = registry.by_key["benzene"].spec
    host_spec = registry.by_key["pyridine"].spec
    host_atoms = (0, 1, 2, 3, 4, 5)
    attached_atoms = (1, 2, 3, 6, 7, 8)
    mol = Molecule()
    for atom in range(9):
        mol.add_atom("N" if atom == 0 else "C", idx=atom, is_aromatic=True)
    edges: set[tuple[int, int]] = set()
    for cycle in (host_atoms, attached_atoms):
        for left, right in zip(cycle, cycle[1:] + cycle[:1]):
            edge = tuple(sorted((left, right)))
            if edge not in edges:
                edges.add(edge)
                mol.add_bond(*edge, idx=100 + len(edges))

    attached = FusionComponentMatch(
        occurrence_id=0,
        spec_key="benzene",
        covered_face_ids=frozenset({0}),
        local_to_input_atom=tuple(zip(attached_spec.locants, attached_atoms, strict=True)),
        local_to_skeleton_atom=tuple(zip(attached_spec.locants, attached_atoms, strict=True)),
        topology_key=(),
    )
    host = FusionComponentMatch(
        occurrence_id=1,
        spec_key="pyridine",
        covered_face_ids=frozenset({1}),
        local_to_input_atom=tuple(zip(host_spec.locants, host_atoms, strict=True)),
        local_to_skeleton_atom=tuple(zip(host_spec.locants, host_atoms, strict=True)),
        topology_key=(),
    )
    shared_edges = frozenset({(1, 2), (2, 3)})
    interface = FusionInterface(
        left=0,
        right=1,
        shared_atom_ids=frozenset({1, 2, 3}),
        shared_edges=shared_edges,
    )

    classified = classify_ordered_fusion_interface(
        attached,
        host,
        attached_spec,
        host_spec,
        interface,
        mol,
    )

    assert classified is not None
    evidence, side_rank = classified
    assert evidence.kind is FusionJoinKind.ORTHO_PERI
    assert side_rank == 1
    assert tuple(locant.text for locant in evidence.attached_path) == ("1", "2", "3")
    assert tuple(locant.text for locant in evidence.host_path) == ("2", "3", "4")
    assert tuple(side.letter for side in evidence.host_sides) == ("b", "c")
    assert evidence.ordered_input_atoms == (1, 2, 3)
    assert evidence.shared_input_edges == shared_edges
    descriptor = FusionDescriptor.from_interface(evidence)
    assert descriptor.render() == "[1,2,3-bc]"


def test_shared_classifier_rejects_disconnected_host_sides():
    registry = fusion_component_registry()
    spec = registry.by_key["benzene"].spec
    mol, _faces = _component_graph(
        ("benzene", "benzene"),
        (
            ((0, "1"), (1, "1")),
            ((0, "2"), (1, "2")),
            ((0, "4"), (1, "4")),
            ((0, "5"), (1, "5")),
        ),
    )
    matches = tuple(
        FusionComponentMatch(
            occurrence_id=occurrence,
            spec_key="benzene",
            covered_face_ids=frozenset({occurrence}),
            local_to_input_atom=match.local_to_input_atom,
            local_to_skeleton_atom=match.local_to_skeleton_atom,
            topology_key=match.topology_key,
        )
        for occurrence, match in enumerate(
            next(
                candidate
                for candidate in fusion_component_registry().match_faces(mol, _faces)
                if candidate.spec_key == "benzene" and candidate.covered_face_ids == frozenset({face})
            )
            for face in range(2)
        )
    )
    left_atoms = frozenset(atom for _, atom in matches[0].local_to_input_atom)
    right_atoms = frozenset(atom for _, atom in matches[1].local_to_input_atom)
    shared_atoms = left_atoms & right_atoms
    shared_edges = frozenset(
        tuple(sorted((bond.u, bond.v))) for bond in mol.bonds.values() if {bond.u, bond.v} <= shared_atoms
    )

    assert (
        classify_ordered_fusion_interface(
            matches[0],
            matches[1],
            spec,
            spec,
            FusionInterface(0, 1, shared_atoms, shared_edges),
            mol,
        )
        is None
    )


def test_polycomponent_tree_orders_attached_components_by_seniority():
    mol, ast, rendered = _build(
        ("furan", "thiophene", "pyridine"),
        (
            ((0, "3"), (2, "2")),
            ((0, "2"), (2, "3")),
            ((1, "2"), (2, "5")),
            ((1, "3"), (2, "6")),
        ),
    )

    assert rendered == "furo[3,2-b]thieno[2,3-e]pyridine"
    assert ast.plan_kind == "polycomponent_tree"
    assert ast.component_occurrences[ast.parent_occurrences[0]].spec_key == "pyridine"
    assert tuple(ast.component_occurrences[child.occurrence_id].spec_key for child in ast.citation_tree.children) == (
        "furan",
        "thiophene",
    )
    assert tuple(descriptor.render() for descriptor in ast.descriptors) == (
        "[3,2-b]",
        "[2,3-e]",
    )


def test_long_component_tree_uses_a_central_parent_location_and_is_atom_order_invariant():
    components = ("furan",) * 16
    joins = tuple(
        pair
        for occurrence in range(15)
        for pair in (
            ((occurrence, "4"), (occurrence + 1, "2")),
            ((occurrence, "5"), (occurrence + 1, "3")),
        )
    )

    first_mol, first_ast, first_name = _build(components, joins)
    arbitrary_ids = tuple(500 + 13 * index for index in reversed(range(len(first_mol.atoms))))
    _second_mol, second_ast, second_name = _build(components, joins, atom_id_order=arbitrary_ids)

    assert len(first_ast.component_occurrences) == len(second_ast.component_occurrences) == 16
    assert first_ast.plan_kind == second_ast.plan_kind == "polycomponent_tree"
    assert first_name == second_name

    def maximum_depth(ast) -> int:
        pending = [(ast.citation_tree, 0)]
        result = 0
        while pending:
            node, depth = pending.pop()
            result = max(result, depth)
            pending.extend((child, depth + 1) for child in node.children)
        return result

    assert maximum_depth(first_ast) == maximum_depth(second_ast) == 8


def test_long_five_membered_ring_chain_completes_the_bounded_layout_proof():
    components = ("furan",) * 16
    joins = tuple(
        pair
        for occurrence in range(15)
        for pair in (
            ((occurrence, "4"), (occurrence + 1, "2")),
            ((occurrence, "5"), (occurrence + 1, "3")),
        )
    )
    mol, _faces = _component_graph(components, joins)

    result = plan_fusion_parent(mol, mol.atoms, mode=FusionMode.GENERAL)

    assert isinstance(result, FusionConfirmed)
    assert len(result.plan.ast.component_occurrences) == 16
    assert result.plan.audit.confirmed


def test_second_order_pairwise_component_keeps_ordinary_fusion_descriptor():
    components = ("furan", "thiophene", "pyridine")
    interfaces = (
        ((0, "2"), (1, "4")),
        ((0, "3"), (1, "5")),
        ((1, "2"), (2, "2")),
        ((1, "3"), (2, "3")),
    )

    first_mol, first_ast, first_name = _build(components, interfaces, experimental=True)
    arbitrary_ids = tuple(700 + 23 * index for index in reversed(range(len(first_mol.atoms))))
    _second_mol, second_ast, second_name = _build(
        components,
        interfaces,
        atom_id_order=arbitrary_ids,
        experimental=True,
    )

    assert first_name == second_name
    higher = tuple(join for join in first_ast.joins if join.order == 2)
    assert len(higher) == 1
    assert higher[0].kind is FusionJoinKind.ORTHO
    assert higher[0].host_sides
    assert not higher[0].host_locants
    assert all(descriptor.kind is not FusionJoinKind.HIGHER_ORDER for descriptor in first_ast.descriptors)
    assert all(descriptor.kind is not FusionJoinKind.HIGHER_ORDER for descriptor in second_ast.descriptors)


def test_identical_leaf_components_form_one_primed_multiplicative_group():
    _mol, ast, rendered = _build(
        ("furan", "furan", "pyridine"),
        (
            ((0, "3"), (2, "2")),
            ((0, "2"), (2, "3")),
            ((1, "2"), (2, "5")),
            ((1, "3"), (2, "6")),
        ),
    )

    assert rendered == "difuro[3,2-b:2',3'-e]pyridine"
    assert ast.plan_kind == "multiplicative_tree"
    assert len(ast.multiplicative_groups) == 1
    assert ast.multiplicative_groups[0].multiplier == "di"
    assert tuple(descriptor.render() for descriptor in ast.descriptors) == (
        "[3,2-b]",
        "[2',3'-e]",
    )


def test_multiplicative_interfaces_use_typed_component_automorphism_orbits():
    registry = fusion_component_registry()
    furan = registry.get("furan").spec
    pyridine = registry.get("pyridine").spec

    assert component_interface_orbit(furan, ("3", "2")) == component_interface_orbit(
        furan, ("2", "3")
    )
    assert component_interface_orbit(pyridine, ("2", "3")) == component_interface_orbit(
        pyridine, ("5", "6")
    )
    assert component_interface_orbit(pyridine, ("2", "3")) != component_interface_orbit(
        pyridine, ("3", "4")
    )


def test_retained_polycyclic_components_form_an_exact_face_cover():
    _mol, ast, rendered = _build(
        ("naphthalene", "azulene"),
        (
            ((0, "2"), (1, "5")),
            ((0, "3"), (1, "6")),
        ),
    )

    assert rendered == "naphtho[2,3-f]azulene"
    assert {match.spec_key for match in ast.component_occurrences} == {"naphthalene", "azulene"}
    assert {match.covered_face_ids for match in ast.component_occurrences} == {
        frozenset((0, 1)),
        frozenset((2, 3)),
    }


def test_polycyclic_component_cover_is_atom_renumbering_invariant():
    components = ("naphthalene", "azulene")
    joins = (
        ((0, "2"), (1, "5")),
        ((0, "3"), (1, "6")),
    )
    first_mol, _first_ast, first = _build(components, joins)
    arbitrary_ids = tuple(200 + 11 * index for index in reversed(range(len(first_mol.atoms))))
    _second_mol, _second_ast, second = _build(components, joins, atom_id_order=arbitrary_ids)

    assert first == second == "naphtho[2,3-f]azulene"


def test_rendering_is_context_free_and_atom_renumbering_invariant():
    components = ("furan", "thiophene", "pyridine")
    joins = (
        ((0, "3"), (2, "2")),
        ((0, "2"), (2, "3")),
        ((1, "2"), (2, "5")),
        ((1, "3"), (2, "6")),
    )
    first_mol, first_ast, first = _build(components, joins)
    atom_count = len(first_mol.atoms)
    arbitrary_ids = tuple(100 + 7 * index for index in reversed(range(atom_count)))
    _second_mol, second_ast, second = _build(components, joins, atom_id_order=arbitrary_ids)

    assert first == second == "furo[3,2-b]thieno[2,3-e]pyridine"
    assert render_fusion_name(first_ast, fusion_component_registry()) == first
    assert render_fusion_name(second_ast, fusion_component_registry()) == second


def test_component_policy_iteration_order_does_not_change_the_name():
    data = deepcopy(load_json_table("fusion_components.json"))
    data["components"].reverse()
    reordered_registry = FusionComponentRegistry.from_data(data)
    components = ("furan", "thiophene", "pyridine")
    joins = (
        ((0, "3"), (2, "2")),
        ((0, "2"), (2, "3")),
        ((1, "2"), (2, "5")),
        ((1, "3"), (2, "6")),
    )

    _, _, ordinary = _build(components, joins)
    _, _, reordered = _build(components, joins, registry=reordered_registry)

    assert reordered == ordinary == "furo[3,2-b]thieno[2,3-e]pyridine"


def test_spiro_overlap_is_outside_the_bounded_fusion_tier():
    registry: FusionComponentRegistry = fusion_component_registry()
    mol, faces = _component_graph(
        ("furan", "thiophene"),
        (((0, "2"), (1, "2")),),
    )
    matches = registry.match_faces(mol, faces)

    with pytest.raises(FusionDescriptorError, match="no exact tree-cover"):
        build_fusion_name_ast(mol, matches, registry, cover_kinds=("tree",))


def test_cyclic_multiparent_component_cover_abstains_from_tree_tier():
    registry = fusion_component_registry()
    mol, faces = _component_graph(
        ("benzene", "benzene", "benzene"),
        (
            ((0, "1"), (1, "1")),
            ((0, "2"), (1, "2")),
            ((1, "4"), (2, "1")),
            ((1, "5"), (2, "2")),
            ((2, "4"), (0, "4")),
            ((2, "5"), (0, "5")),
        ),
    )
    matches = tuple(
        match
        for match in registry.match_faces(mol, faces)
        if match.spec_key == "benzene" and len(match.covered_face_ids) == 1
    )

    with pytest.raises(FusionDescriptorError, match="no exact tree-cover"):
        build_fusion_name_ast(mol, matches, registry, cover_kinds=("tree",))


def test_cyclic_cover_retains_cycle_closing_interface_as_numeric_higher_order_join():
    components = ("benzene", "benzene", "benzene")
    interfaces = (
        ((0, "1"), (1, "1")),
        ((0, "2"), (1, "2")),
        ((1, "4"), (2, "1")),
        ((1, "5"), (2, "2")),
        ((2, "4"), (0, "4")),
        ((2, "5"), (0, "5")),
    )

    first_mol, first_ast, first_name = _build(
        components,
        interfaces,
        experimental=True,
        atomic_components_only=True,
    )
    arbitrary_ids = tuple(900 + 17 * index for index in reversed(range(len(first_mol.atoms))))
    _second_mol, second_ast, second_name = _build(
        components,
        interfaces,
        atom_id_order=arbitrary_ids,
        experimental=True,
        atomic_components_only=True,
    )

    assert first_ast.plan_kind == second_ast.plan_kind == "cyclic_component_cover"
    assert first_name == second_name
    assert first_ast.citation_plan is not None
    assert len(first_ast.joins) == 3
    assert first_ast.citation_plan.cycle_closing_join_indices == (2,)
    closing = first_ast.joins[2]
    assert closing.kind is FusionJoinKind.HIGHER_ORDER
    assert closing.host_locants
    assert not closing.host_sides
    assert first_ast.descriptors[2].render().count(":") == 1


def test_multiparent_cover_records_interparent_component_and_is_atom_order_invariant():
    components = ("furan", "benzene", "furan")
    interfaces = (
        ((0, "2"), (1, "1")),
        ((0, "3"), (1, "2")),
        ((2, "2"), (1, "4")),
        ((2, "3"), (1, "5")),
    )

    first_mol, first_ast, first_name = _build(
        components,
        interfaces,
        experimental=True,
        atomic_components_only=True,
    )
    arbitrary_ids = tuple(1200 + 19 * index for index in reversed(range(len(first_mol.atoms))))
    _second_mol, second_ast, second_name = _build(
        components,
        interfaces,
        atom_id_order=arbitrary_ids,
        experimental=True,
        atomic_components_only=True,
    )

    assert first_ast.plan_kind == second_ast.plan_kind == "multiparent"
    assert first_name == second_name
    assert first_ast.citation_tree is None
    assert first_ast.citation_plan is not None
    assert len(first_ast.parent_occurrences) == 2
    assert len(first_ast.citation_plan.interparent_occurrences) == 1
    assert first_ast.citation_plan.interparent_join_indices == (1,)
    assert first_name.startswith("benzo[")
    assert first_name.endswith("difuran")


def test_multiparent_roots_require_identical_graph_template_variants():
    furan = fusion_component_registry().by_key["furan"].spec
    variant = replace(furan, template=replace(furan.template, name="furan-variant"))
    adjacency = {0: (1,), 1: (0, 2), 2: (1,)}

    assert _multiparent_root_sets((0, 2), adjacency, {0: furan, 1: furan, 2: furan})
    assert not _multiparent_root_sets(
        (0, 2),
        adjacency,
        {0: furan, 1: furan, 2: variant},
    )


@pytest.mark.skipif(not opsin_available(), reason="py2opsin/Java is unavailable")
def test_production_multiparent_grammar_round_trips_through_opsin():
    _mol, _ast, rendered = _build(
        ("furan", "benzene", "furan"),
        (
            ((0, "2"), (1, "1")),
            ((0, "3"), (1, "2")),
            ((2, "2"), (1, "4")),
            ((2, "3"), (1, "5")),
        ),
        atomic_components_only=True,
    )

    assert rendered == "benzo[1,2-b:4,5-b']difuran"
    assert verify_with_opsin(rendered, "O1C=2C(C=C1)=CC=1OC=CC1C2").ok


def test_locanted_hw_multiparents_use_data_selected_complex_multiplier():
    oxadiazole = "generated-hw:O.N.C.C.N"
    mol, ast, rendered = _build(
        (oxadiazole, "benzene", oxadiazole),
        (
            ((1, "1"), (0, "3")),
            ((1, "2"), (0, "4")),
            ((1, "3"), (2, "3")),
            ((1, "4"), (2, "4")),
        ),
        atomic_components_only=True,
    )

    assert ast.plan_kind == "multiparent"
    assert rendered == "benzo[1,2-c:3,4-c']bis([1,2,5]oxadiazole)"
    planned = plan_fusion_parent(mol, mol.atoms, mode=FusionMode.GENERAL)
    assert isinstance(planned, FusionConfirmed)
    assert planned.plan.rendered_base_name == rendered
