"""Independent graph reconstruction and audit for systematic fusion plans.

The auditor consumes only typed fusion objects and molecular graphs.  It does
not inspect a rendered name and deliberately does not call the fusion planner
or component matcher, so it can detect corruption in either of those stages.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from ..molecule import Molecule
from ..polycycle_topology import normalize_edge
from .cover import audit_component_cover, component_scope
from .model import (
    AuditStatus,
    FusionAuditResult,
    FusionComponentMatch,
    FusionComponentSpec,
    FusionDescriptor,
    FusionGraph,
    FusionJoin,
    FusionMode,
    FusionNameAst,
    FusionNumberingProof,
    ParentBondModel,
)
from .rules import component_spec_seniority_key, pin_ring_size_gate

_Node = tuple[int, str]
_Edge = tuple[int, int]


@dataclass(frozen=True, slots=True)
class _ReconstructedGraph:
    input_atom_by_node: Mapping[_Node, int]
    skeleton_atom_by_node: Mapping[_Node, int]
    input_edges_by_occurrence: Mapping[int, frozenset[_Edge]]
    input_atoms: Mapping[int, tuple[str, int]]
    input_edges: frozenset[_Edge]
    skeleton_atoms: Mapping[int, tuple[str, int]]
    skeleton_edges: Mapping[_Edge, str]


class _DisjointSet:
    def __init__(self, values: Iterable[_Node]) -> None:
        self._parent = {value: value for value in values}

    def find(self, value: _Node) -> _Node:
        parent = self._parent[value]
        if parent != value:
            self._parent[value] = self.find(parent)
        return self._parent[value]

    def union(self, left: _Node, right: _Node) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self._parent[right_root] = left_root


def audit_fusion_plan(
    mol: Molecule,
    parent_atom_ids: Iterable[int],
    *,
    ast: FusionNameAst,
    abstract_parent_graph: FusionGraph,
    numbering: FusionNumberingProof,
    bond_model: ParentBondModel,
    mode: FusionMode = FusionMode.GENERAL,
    registry: object | None = None,
) -> FusionAuditResult:
    """Independently reconstruct and audit a completed fusion candidate.

    ``ABSTAIN`` is reserved for a nomenclatural applicability gate, currently
    the PIN ring-size and component-policy gates.  A self-inconsistent proof or
    graph is a ``MISMATCH``.  Missing registry data and unexpected audit
    failures are reported as ``ERROR`` rather than allowing an unaudited name
    to enter rendering.
    """

    parent_atoms = frozenset(parent_atom_ids)
    unknown = parent_atoms - mol.atoms.keys()
    if unknown:
        return _error(f"selected parent references unknown atom ids: {sorted(unknown)}")
    if not parent_atoms:
        return _error("selected parent graph is empty")

    if mode is FusionMode.AUDITED_PIN:
        ring_sizes = tuple(face.size for face in numbering.selected_face_model.faces)
        if not pin_ring_size_gate(ring_sizes):
            return FusionAuditResult(
                AuditStatus.ABSTAIN,
                checks=("pin_ring_size_gate",),
                errors=("fewer than two rings of size at least five",),
            )

    try:
        specs = {match.occurrence_id: _component_spec(registry, match) for match in ast.component_occurrences}
    except (KeyError, TypeError, ValueError) as exc:
        return _error(str(exc))

    if mode is FusionMode.AUDITED_PIN:
        non_pin = sorted(occurrence for occurrence, spec in specs.items() if not spec.pin_component)
        if non_pin:
            return FusionAuditResult(
                AuditStatus.ABSTAIN,
                checks=("pin_ring_size_gate", "pin_component_policy"),
                errors=(f"component occurrences are not PIN-eligible: {non_pin}",),
            )

    policy_errors = _component_role_errors(ast, specs)
    if policy_errors:
        return FusionAuditResult(AuditStatus.ABSTAIN, checks=("component_role_policy",), errors=policy_errors)

    errors: list[str] = []
    checks: list[str] = []
    try:
        reconstruction = _reconstruct(ast, specs, mol, errors)
        _audit_nomenclature_selection(ast, specs, errors)
        checks.append("nomenclature_selection")
        _audit_component_and_face_coverage(
            mol,
            parent_atoms,
            ast,
            specs,
            numbering,
            reconstruction,
            errors,
        )
        checks.append("component_coverage")

        _audit_descriptors(mol, ast, specs, reconstruction, errors)
        checks.append("descriptor_interfaces")

        _audit_reconstructed_graph(mol, parent_atoms, abstract_parent_graph, reconstruction, errors)
        checks.extend(("abstract_graph_reconstruction", "input_graph_identity"))

        _audit_numbering(mol, parent_atoms, abstract_parent_graph, numbering, errors)
        checks.append("completed_numbering")

        _audit_bond_model(mol, abstract_parent_graph, numbering, bond_model, errors)
        checks.append("parent_bond_model")
    except (KeyError, TypeError, ValueError) as exc:
        return _error(f"fusion audit could not evaluate candidate: {exc}", checks=checks)

    if errors:
        return FusionAuditResult(AuditStatus.MISMATCH, checks=tuple(checks), errors=tuple(dict.fromkeys(errors)))
    return FusionAuditResult(AuditStatus.CONFIRMED, checks=tuple(checks))


def _audit_nomenclature_selection(
    ast: FusionNameAst,
    specs: Mapping[int, FusionComponentSpec],
    errors: list[str],
) -> None:
    """Verify parent identity and intrinsic component seniority independently."""

    if len(ast.parent_occurrences) != 1:
        errors.append("bounded fusion tier requires exactly one parent occurrence")
        return
    parent = ast.parent_occurrences[0]
    if ast.citation_tree.occurrence_id != parent:
        errors.append("fusion citation root does not match the declared parent occurrence")
    matches = {match.occurrence_id: match for match in ast.component_occurrences}
    if parent not in matches:
        errors.append("declared fusion parent is absent from component occurrences")
        return
    eligible = [match for occurrence, match in matches.items() if specs[occurrence].usable_as_parent]
    if not eligible:
        errors.append("fusion plan has no component eligible as a parent")
        return
    preferred = min(component_spec_seniority_key(specs[match.occurrence_id]) for match in eligible)
    if component_spec_seniority_key(specs[parent]) != preferred:
        errors.append("declared fusion parent is not the intrinsically senior eligible component")


def _component_spec(registry: object | None, match: FusionComponentMatch) -> FusionComponentSpec:
    if registry is None:
        from .registry import fusion_component_registry

        registry = fusion_component_registry()

    resolver = getattr(registry, "spec_for_match", None)
    if resolver is not None:
        return resolver(match)
    value: Any = None
    by_key = getattr(registry, "by_key", None)
    if by_key is not None:
        value = by_key.get(match.spec_key)
    elif isinstance(registry, Mapping):
        value = registry.get(match.spec_key)
    else:
        getter = getattr(registry, "get", None)
        if getter is not None:
            value = getter(match.spec_key)
    value = getattr(value, "spec", value)
    if not isinstance(value, FusionComponentSpec):
        raise KeyError(f"unknown fusion component spec: {match.spec_key}")
    return value


def _component_role_errors(
    ast: FusionNameAst,
    specs: Mapping[int, FusionComponentSpec],
) -> tuple[str, ...]:
    parents = set(ast.parent_occurrences)
    errors = []
    for occurrence, spec in specs.items():
        if occurrence in parents and not spec.usable_as_parent:
            errors.append(f"component occurrence {occurrence} is not allowed as a fusion parent")
        if occurrence not in parents and not spec.usable_as_attached:
            errors.append(f"component occurrence {occurrence} is not allowed as an attached component")
    return tuple(errors)


def _reconstruct(
    ast: FusionNameAst,
    specs: Mapping[int, FusionComponentSpec],
    mol: Molecule,
    errors: list[str],
) -> _ReconstructedGraph:
    matches = {match.occurrence_id: match for match in ast.component_occurrences}
    nodes: list[_Node] = []
    labels: dict[_Node, tuple[str, int]] = {}
    input_by_node: dict[_Node, int] = {}
    skeleton_by_node: dict[_Node, int] = {}
    local_edges: list[tuple[_Node, _Node, str]] = []
    input_edges_by_occurrence: dict[int, frozenset[_Edge]] = {}

    for occurrence, match in matches.items():
        spec = specs[occurrence]
        input_map = match.input_atom_by_locant
        skeleton_map = match.skeleton_atom_by_locant
        if set(input_map) != set(spec.locants) or set(skeleton_map) != set(spec.locants):
            errors.append(f"component occurrence {occurrence} does not map every spec locant exactly once")
            continue
        atom_by_locant = {atom.locant: atom for atom in spec.atoms}
        for locant in spec.locants:
            node = (occurrence, locant)
            nodes.append(node)
            atom = atom_by_locant[locant]
            labels[node] = (atom.symbol, atom.charge)
            input_by_node[node] = input_map[locant]
            skeleton_by_node[node] = skeleton_map[locant]
        occurrence_edges: set[_Edge] = set()
        for bond in spec.bonds:
            left, right = bond.locants
            local_edges.append(((occurrence, left), (occurrence, right), bond.bond_class))
            occurrence_edges.add(normalize_edge(input_map[left], input_map[right]))
        input_edges_by_occurrence[occurrence] = frozenset(occurrence_edges)

    disjoint = _DisjointSet(nodes)
    for join in ast.joins:
        if join.attached_occurrence not in matches or join.host_occurrence not in matches:
            errors.append("fusion join references an unknown component occurrence")
            continue
        attached_nodes = tuple((join.attached_occurrence, locant.text) for locant in join.attached_locants)
        host_paths = _host_interface_paths(join, specs[join.host_occurrence], errors)
        if not host_paths:
            continue
        if len(attached_nodes) != join.order + 1:
            errors.append(
                f"join {join.attached_occurrence}->{join.host_occurrence} has an attached interface "
                f"of {len(attached_nodes) - 1} bonds but declares order {join.order}"
            )
            continue
        attached_input = tuple(input_by_node.get(node) for node in attached_nodes)
        matching_paths = [
            path for path in host_paths if tuple(input_by_node.get(node) for node in path) == attached_input
        ]
        if not matching_paths:
            errors.append(
                f"join {join.attached_occurrence}->{join.host_occurrence} cites interfaces that do not map "
                "to the same ordered input atoms"
            )
            continue
        host_nodes = matching_paths[0]
        for attached, host in zip(attached_nodes, host_nodes, strict=True):
            disjoint.union(attached, host)

    classes: dict[_Node, list[_Node]] = defaultdict(list)
    for node in nodes:
        classes[disjoint.find(node)].append(node)

    reconstructed_input_atoms: dict[int, tuple[str, int]] = {}
    reconstructed_skeleton_atoms: dict[int, tuple[str, int]] = {}
    root_input: dict[_Node, int] = {}
    root_skeleton: dict[_Node, int] = {}
    for root, members in classes.items():
        member_labels = {labels[node] for node in members}
        input_ids = {input_by_node[node] for node in members}
        skeleton_ids = {skeleton_by_node[node] for node in members}
        if len(member_labels) != 1:
            errors.append(f"merged component atoms have incompatible element or charge definitions: {members}")
            continue
        if len(input_ids) != 1:
            errors.append(f"merged component atoms map to different input atoms: {members}")
            continue
        if len(skeleton_ids) != 1:
            errors.append(f"merged component atoms map to different abstract atoms: {members}")
            continue
        label = next(iter(member_labels))
        input_id = next(iter(input_ids))
        skeleton_id = next(iter(skeleton_ids))
        if input_id in reconstructed_input_atoms and reconstructed_input_atoms[input_id] != label:
            errors.append(f"input atom {input_id} receives incompatible component definitions")
        if skeleton_id in reconstructed_skeleton_atoms and reconstructed_skeleton_atoms[skeleton_id] != label:
            errors.append(f"abstract atom {skeleton_id} receives incompatible component definitions")
        reconstructed_input_atoms[input_id] = label
        reconstructed_skeleton_atoms[skeleton_id] = label
        root_input[root] = input_id
        root_skeleton[root] = skeleton_id

    input_edges: set[_Edge] = set()
    skeleton_edges: dict[_Edge, str] = {}
    for left, right, bond_class in local_edges:
        left_root = disjoint.find(left)
        right_root = disjoint.find(right)
        if left_root == right_root:
            errors.append(f"fusion collapse creates a self bond from {left} to {right}")
            continue
        if left_root not in root_input or right_root not in root_input:
            continue
        input_edge = normalize_edge(root_input[left_root], root_input[right_root])
        skeleton_edge = normalize_edge(root_skeleton[left_root], root_skeleton[right_root])
        input_edges.add(input_edge)
        previous = skeleton_edges.get(skeleton_edge)
        if previous is not None and previous != bond_class:
            errors.append(f"shared abstract edge {skeleton_edge} has incompatible bond classes")
        skeleton_edges[skeleton_edge] = bond_class

    return _ReconstructedGraph(
        input_atom_by_node=input_by_node,
        skeleton_atom_by_node=skeleton_by_node,
        input_edges_by_occurrence=input_edges_by_occurrence,
        input_atoms=reconstructed_input_atoms,
        input_edges=frozenset(input_edges),
        skeleton_atoms=reconstructed_skeleton_atoms,
        skeleton_edges=skeleton_edges,
    )


def _host_interface_paths(
    join: FusionJoin,
    host: FusionComponentSpec,
    errors: list[str],
) -> tuple[tuple[_Node, ...], ...]:
    occurrence = join.host_occurrence
    if join.host_locants:
        path = tuple((occurrence, locant.text) for locant in join.host_locants)
        if len(path) != join.order + 1:
            errors.append(
                f"join {join.attached_occurrence}->{occurrence} has a numeric host interface "
                f"of {len(path) - 1} bonds but declares order {join.order}"
            )
            return ()
        return (path,)

    side_indices: list[int] = []
    for side in join.host_sides:
        for letter in side.letter:
            index = ord(letter) - ord("a")
            if not 0 <= index < len(host.peripheral_order):
                errors.append(f"host side {letter!r} is not defined by component {host.key}")
                return ()
            side_indices.append(index)
    if len(side_indices) != join.order:
        errors.append(
            f"join {join.attached_occurrence}->{occurrence} cites {len(side_indices)} host sides "
            f"but declares order {join.order}"
        )
        return ()
    edges = [
        (
            host.peripheral_order[index],
            host.peripheral_order[(index + 1) % len(host.peripheral_order)],
        )
        for index in side_indices
    ]
    path = _ordered_side_path(edges)
    if not path:
        errors.append(f"host sides do not form one ordered interface on component {host.key}")
        return ()
    return (tuple((occurrence, locant) for locant in path),)


def _ordered_side_path(edges: list[tuple[str, str]]) -> tuple[str, ...]:
    if not edges:
        return ()
    path = [edges[0][0], edges[0][1]]
    for left, right in edges[1:]:
        if path[-1] != left:
            return ()
        path.append(right)
    return tuple(path)


def _audit_component_and_face_coverage(
    mol: Molecule,
    parent_atoms: frozenset[int],
    ast: FusionNameAst,
    specs: Mapping[int, FusionComponentSpec],
    numbering: FusionNumberingProof,
    reconstruction: _ReconstructedGraph,
    errors: list[str],
) -> None:
    matches = {match.occurrence_id: match for match in ast.component_occurrences}
    scopes = []
    face_owners: Counter[int] = Counter()
    faces = {face.id: face for face in numbering.selected_face_model.faces}
    for occurrence, match in matches.items():
        input_map = match.input_atom_by_locant
        spec = specs[occurrence]
        atom_ids = set(input_map.values())
        edges = {
            normalize_edge(input_map[bond.locants[0]], input_map[bond.locants[1]])
            for bond in spec.bonds
            if set(bond.locants) <= input_map.keys()
        }
        scopes.append(component_scope(occurrence, atom_ids, edges))
        face_owners.update(match.covered_face_ids)
        unknown_faces = match.covered_face_ids - faces.keys()
        if unknown_faces:
            errors.append(f"component occurrence {occurrence} covers unknown faces: {sorted(unknown_faces)}")
            continue
        mapped_rings = {
            frozenset(input_map[locant] for locant in ring) for ring in spec.rings if set(ring) <= input_map.keys()
        }
        selected_rings = {frozenset(faces[face_id].atom_cycle) for face_id in match.covered_face_ids}
        if mapped_rings != selected_rings:
            errors.append(f"component occurrence {occurrence} does not reconstruct its declared faces")

    expected_face_ids = set(faces)
    if set(face_owners) != expected_face_ids or any(count != 1 for count in face_owners.values()):
        errors.append("component occurrences do not cover every selected face exactly once")

    target_edges = _molecule_edges(mol, parent_atoms)
    cover = audit_component_cover(scopes, target_atom_ids=parent_atoms, target_edges=target_edges)
    errors.extend(cover.errors)
    if reconstruction.input_edges != target_edges:
        errors.append("component reconstruction does not exactly cover selected parent edges")


def _audit_descriptors(
    mol: Molecule,
    ast: FusionNameAst,
    specs: Mapping[int, FusionComponentSpec],
    reconstruction: _ReconstructedGraph,
    errors: list[str],
) -> None:
    matches = {match.occurrence_id: match for match in ast.component_occurrences}
    joins_by_pair: dict[frozenset[int], FusionJoin] = {}
    for index, join in enumerate(ast.joins):
        pair = frozenset((join.attached_occurrence, join.host_occurrence))
        if pair in joins_by_pair:
            errors.append(f"component pair {sorted(pair)} has more than one fusion join")
        joins_by_pair[pair] = join

        attached = matches[join.attached_occurrence]
        host = matches[join.host_occurrence]
        attached_map = attached.input_atom_by_locant
        host_map = host.input_atom_by_locant
        attached_path = tuple(attached_map.get(locant.text) for locant in join.attached_locants)
        host_paths = _host_interface_paths(join, specs[join.host_occurrence], errors)
        host_input_paths = tuple(tuple(host_map.get(node[1]) for node in path) for path in host_paths)
        if attached_path not in host_input_paths:
            continue
        derived_edges = frozenset(
            normalize_edge(left, right) for left, right in zip(attached_path, attached_path[1:])
        )
        actual_edges = (
            reconstruction.input_edges_by_occurrence[join.attached_occurrence]
            & reconstruction.input_edges_by_occurrence[join.host_occurrence]
        )
        derived_atoms = frozenset(attached_path)
        if derived_edges != actual_edges:
            errors.append(
                f"join {join.attached_occurrence}->{join.host_occurrence} does not cite every and only shared edge"
            )
        if derived_atoms != frozenset(atom for edge in actual_edges for atom in edge):
            errors.append(f"join {join.attached_occurrence}->{join.host_occurrence} cites wrong shared atoms")
        expected_bond_ids = frozenset(bond.idx for edge in actual_edges if (bond := mol.get_bond(*edge)) is not None)
        if join.shared_input_atoms != derived_atoms:
            errors.append(f"join {join.attached_occurrence}->{join.host_occurrence} stores wrong shared atoms")
        if join.shared_input_bonds != expected_bond_ids:
            errors.append(f"join {join.attached_occurrence}->{join.host_occurrence} stores wrong shared bonds")

        if ast.descriptors:
            descriptor = ast.descriptors[index]
            if not _descriptor_matches_join(descriptor, join):
                errors.append(f"descriptor {index} does not preserve its fusion join")

    occurrences = sorted(matches)
    for position, left in enumerate(occurrences):
        for right in occurrences[position + 1 :]:
            overlap = reconstruction.input_edges_by_occurrence[left] & reconstruction.input_edges_by_occurrence[right]
            if bool(overlap) != (frozenset((left, right)) in joins_by_pair):
                errors.append(
                    f"shared interface between component occurrences {left} and {right} is not described exactly once"
                )

    for group in ast.multiplicative_groups:
        keys = {matches[occurrence].spec_key for occurrence in group.occurrence_ids}
        if len(keys) != 1:
            errors.append(f"multiplicative group {group.occurrence_ids} combines nonidentical components")


def _descriptor_matches_join(descriptor: FusionDescriptor, join: FusionJoin) -> bool:
    return (
        descriptor.attached_locants == join.attached_locants
        and descriptor.parent_sides == join.host_sides
        and descriptor.parent_locants == join.host_locants
        and descriptor.kind is join.kind
    )


def _audit_reconstructed_graph(
    mol: Molecule,
    parent_atoms: frozenset[int],
    abstract: FusionGraph,
    reconstructed: _ReconstructedGraph,
    errors: list[str],
) -> None:
    actual_labels = {atom: (mol.atoms[atom].symbol, mol.atoms[atom].charge) for atom in parent_atoms}
    if reconstructed.input_atoms != actual_labels:
        errors.append("reconstructed component atoms differ from selected input elements or formal charges")
    actual_edges = _molecule_edges(mol, parent_atoms)
    if reconstructed.input_edges != actual_edges:
        errors.append("reconstructed component connectivity differs from the selected input parent")

    abstract_labels = {atom.id: (atom.symbol, atom.formal_charge) for atom in abstract.atoms}
    abstract_edges = {normalize_edge(*bond.atoms): bond.bond_class for bond in abstract.bonds}
    if reconstructed.skeleton_atoms != abstract_labels:
        errors.append("reconstructed component atoms differ from the declared abstract parent graph")
    if reconstructed.skeleton_edges != abstract_edges:
        errors.append("reconstructed component bonds differ from the declared abstract parent graph")


def _audit_numbering(
    mol: Molecule,
    parent_atoms: frozenset[int],
    abstract: FusionGraph,
    numbering: FusionNumberingProof,
    errors: list[str],
) -> None:
    abstract_map = dict(numbering.abstract_atom_to_locant)
    abstract_labels = {atom.id: (atom.symbol, atom.formal_charge) for atom in abstract.atoms}
    abstract_edges = frozenset(normalize_edge(*bond.atoms) for bond in abstract.bonds)
    if set(abstract_map) != set(abstract_labels) or len(set(abstract_map.values())) != len(abstract_map):
        errors.append("completed abstract numbering is not a bijection over the parent graph")

    _audit_face_model(mol, parent_atoms, numbering, errors)
    face_membership = Counter(atom for face in numbering.selected_face_model.faces for atom in face.atom_cycle)
    fusion_atoms = {atom for atom, count in face_membership.items() if count > 1}

    for input_map_items in numbering.input_locant_maps:
        input_map = dict(input_map_items)
        if set(input_map) != parent_atoms or len(set(input_map.values())) != len(input_map):
            errors.append("completed input numbering is not a bijection over the selected parent")
            continue
        abstract_by_locant = {locant: atom for atom, locant in abstract_map.items()}
        input_to_abstract = {atom: abstract_by_locant.get(locant) for atom, locant in input_map.items()}
        if any(atom is None for atom in input_to_abstract.values()):
            errors.append("input numbering uses locants absent from the abstract numbering")
            continue
        for input_atom, abstract_atom in input_to_abstract.items():
            assert abstract_atom is not None
            if (mol.atoms[input_atom].symbol, mol.atoms[input_atom].charge) != abstract_labels[abstract_atom]:
                errors.append("input locant map does not preserve element and formal-charge labels")
                break
        mapped_edges = frozenset(
            normalize_edge(input_to_abstract[left], input_to_abstract[right])
            for left, right in _molecule_edges(mol, parent_atoms)
        )
        if mapped_edges != abstract_edges:
            errors.append("input locant map is not graph preserving")

        for atom, locant in input_map.items():
            if locant.interior_distance is not None:
                continue
            is_fusion_carbon = atom in fusion_atoms and mol.atoms[atom].symbol == "C"
            if is_fusion_carbon != bool(locant.fusion_suffix):
                errors.append("fusion-carbon suffix locants do not match the selected face model")
                break
            if atom in fusion_atoms and mol.atoms[atom].symbol != "C" and locant.fusion_suffix:
                errors.append("fusion heteroatoms must receive integer completed-system locants")
                break


def _audit_face_model(
    mol: Molecule,
    parent_atoms: frozenset[int],
    numbering: FusionNumberingProof,
    errors: list[str],
) -> None:
    model = numbering.selected_face_model
    owners: dict[int, list[int]] = defaultdict(list)
    reconstructed_edges: set[int] = set()
    for face in model.faces:
        if not set(face.atom_cycle) <= parent_atoms:
            errors.append(f"numbering face {face.id} contains atoms outside the selected parent")
            continue
        expected_ids = []
        for left, right in zip(face.atom_cycle, face.atom_cycle[1:] + face.atom_cycle[:1]):
            bond = mol.get_bond(left, right)
            if bond is None:
                errors.append(f"numbering face {face.id} contains a nonexistent input edge")
                continue
            expected_ids.append(bond.idx)
            owners[bond.idx].append(face.id)
            reconstructed_edges.add(bond.idx)
        if tuple(expected_ids) != face.edge_cycle:
            errors.append(f"numbering face {face.id} edge cycle does not follow its atom cycle")

    expected_parent_edges = {
        bond.idx for bond in mol.bonds.values() if bond.u in parent_atoms and bond.v in parent_atoms
    }
    if reconstructed_edges != expected_parent_edges:
        errors.append("numbering face model does not reconstruct every selected parent edge")
    expected_owners = tuple(sorted((edge, tuple(sorted(face_ids))) for edge, face_ids in owners.items()))
    if model.edge_to_faces != expected_owners:
        errors.append("numbering face edge ownership is inconsistent")
    expected_fusion = frozenset(edge for edge, face_ids in owners.items() if len(face_ids) == 2)
    expected_perimeter = frozenset(edge for edge, face_ids in owners.items() if len(face_ids) == 1)
    if model.fusion_edges != expected_fusion or model.perimeter_edges != expected_perimeter:
        errors.append("numbering face perimeter or fusion edge classification is inconsistent")
    expected_adjacency = tuple(
        sorted((min(face_ids), max(face_ids), edge) for edge, face_ids in owners.items() if len(face_ids) == 2)
    )
    if model.face_adjacency != expected_adjacency:
        errors.append("numbering face adjacency is inconsistent with shared edges")
    boundary_edges = []
    for left, right in zip(model.outer_boundary, model.outer_boundary[1:] + model.outer_boundary[:1]):
        bond = mol.get_bond(left, right)
        if bond is None:
            errors.append("numbering outer boundary is not a molecular cycle")
            return
        boundary_edges.append(bond.idx)
    if frozenset(boundary_edges) != expected_perimeter:
        errors.append("numbering outer boundary does not equal the face-model perimeter")


def _audit_bond_model(
    mol: Molecule,
    abstract: FusionGraph,
    numbering: FusionNumberingProof,
    model: ParentBondModel,
    errors: list[str],
) -> None:
    abstract_edges = frozenset(normalize_edge(*bond.atoms) for bond in abstract.bonds)
    known_edges = frozenset(normalize_edge(*edge) for edge in model.required_single_bonds | model.pi_eligible_edges)
    if known_edges != abstract_edges:
        errors.append("parent bond model does not cover every and only abstract parent edge")
        return

    maximum = 0
    for assignment in model.allowed_kekule_assignments:
        orders = {normalize_edge(*edge): order for edge, order in assignment.orders}
        if set(orders) != abstract_edges:
            errors.append("parent bond assignment is incomplete")
            continue
        if any(order not in {1, 2} for order in orders.values()):
            errors.append("fusion parent bond assignments may contain only single and double bonds")
        double_edges = [edge for edge, order in orders.items() if order == 2]
        maximum = max(maximum, len(double_edges))
        if len({atom for edge in double_edges for atom in edge}) != 2 * len(double_edges):
            errors.append("parent bond assignment contains cumulative double bonds")
        if any(orders[edge] != 1 for edge in model.required_single_bonds):
            errors.append("parent bond assignment violates a required-single edge")
    if maximum != model.maximum_non_cumulative_double_bonds:
        errors.append("parent bond model double-bond maximum is inconsistent with its assignments")
    if not model.allowed_kekule_assignments:
        errors.append("parent bond model has no allowed bonding assignment")
        return

    abstract_by_locant = {locant: atom for atom, locant in numbering.abstract_atom_to_locant}
    observed_matches = False
    for input_map_items in numbering.input_locant_maps:
        input_to_abstract = {
            input_atom: abstract_by_locant[locant]
            for input_atom, locant in input_map_items
            if locant in abstract_by_locant
        }
        observed: dict[_Edge, int | None] = {}
        for bond in mol.bonds.values():
            if bond.u not in input_to_abstract or bond.v not in input_to_abstract:
                continue
            edge = normalize_edge(input_to_abstract[bond.u], input_to_abstract[bond.v])
            aromatic = mol.atoms[bond.u].is_aromatic and mol.atoms[bond.v].is_aromatic
            observed[edge] = None if aromatic and edge in model.pi_eligible_edges else bond.order
        for assignment in model.allowed_kekule_assignments:
            allowed = {normalize_edge(*edge): order for edge, order in assignment.orders}
            if set(observed) != abstract_edges:
                continue
            abstract_to_input = {abstract: input_atom for input_atom, abstract in input_to_abstract.items()}
            compatible = True
            for edge, order in observed.items():
                if order is None or allowed[edge] == order:
                    continue
                if (
                    order != 1
                    or allowed[edge] != 2
                    or any(
                        atom not in abstract_to_input or mol.atoms[abstract_to_input[atom]].total_h_count <= 0
                        for atom in edge
                    )
                ):
                    compatible = False
                    break
            if compatible:
                observed_matches = True
                break
        if observed_matches:
            break
    if not observed_matches:
        errors.append("selected input bond orders are not allowed by the parent bond model")


def _molecule_edges(mol: Molecule, atoms: frozenset[int]) -> frozenset[_Edge]:
    return frozenset(
        normalize_edge(bond.u, bond.v) for bond in mol.bonds.values() if bond.u in atoms and bond.v in atoms
    )


def _error(message: str, *, checks: Iterable[str] = ()) -> FusionAuditResult:
    return FusionAuditResult(AuditStatus.ERROR, checks=tuple(checks), errors=(message,))
