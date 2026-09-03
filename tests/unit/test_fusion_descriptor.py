"""Graph-built tests for bounded systematic fusion descriptors."""

from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy

import pytest

from openclatura.fusion import (
    FusionDescriptorError,
    build_fusion_name_ast,
    component_sides,
    render_fusion_name,
)
from openclatura.fusion.faces import GraphCycle
from openclatura.fusion.registry import FusionComponentRegistry, fusion_component_registry
from openclatura.molecule import Molecule
from openclatura.naming_data import load_json_table

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
    specs = tuple(registry.by_key[key].spec for key in component_keys)
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

    seen_edges: set[tuple[int, int]] = set()
    for occurrence, spec in enumerate(specs):
        for bond in spec.bonds:
            endpoints = tuple(atom_by_local[(occurrence, locant)] for locant in bond.locants)
            edge = tuple(sorted(endpoints))
            if edge in seen_edges:
                continue
            seen_edges.add(edge)
            mol.add_bond(*edge, idx=100 + len(seen_edges))

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
):
    registry = registry or fusion_component_registry()
    mol, faces = _component_graph(component_keys, fused_atoms, atom_id_order=atom_id_order)
    matches = registry.match_faces(mol, faces)
    ast = build_fusion_name_ast(mol, matches, registry)
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


def test_polycomponent_tree_orders_attached_components_by_seniority():
    _mol, ast, rendered = _build(
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
        build_fusion_name_ast(mol, matches, registry)
