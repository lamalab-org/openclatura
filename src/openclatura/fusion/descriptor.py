"""Bounded construction and pure rendering of systematic fusion citations.

The builder in this module starts from exact, graph-backed component matches.
It supports a bounded systematic-fusion tier: independently named components
form an exact cover of the bounded faces and a connected, acyclic
component-overlap graph. A component may itself cover multiple faces.
No line notation, complete-name lookup, or drawing coordinates participate in
descriptor construction.
"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import product
from typing import Protocol

from ..molecule import Molecule
from .cover import ComponentScope, FusionInterface, audit_component_cover, component_scope
from .model import (
    ComponentLocant,
    FusionCitationNode,
    FusionComponentMatch,
    FusionComponentSpec,
    FusionDescriptor,
    FusionJoin,
    FusionJoinKind,
    FusionMultiplicityGroup,
    FusionNameAst,
    FusionSide,
)
from .rules import component_seniority_key

MAX_COMPONENT_OCCURRENCES = 8
MAX_COMPONENT_SELECTIONS = 256
MAX_COMPONENT_SELECTION_STATES = 4096
MAX_LOCANT_MAP_COMBINATIONS = 4096


class FusionDescriptorError(ValueError):
    """Raised when the bounded descriptor tier cannot prove a valid AST."""


class _Registry(Protocol):
    @property
    def by_key(self) -> Mapping[str, object]: ...


@dataclass(frozen=True, slots=True)
class ComponentSide:
    """One directed, lettered side of a component's peripheral walk."""

    letter: str
    start_locant: str
    end_locant: str

    @property
    def bond_key(self) -> frozenset[str]:
        return frozenset((self.start_locant, self.end_locant))


@dataclass(frozen=True, slots=True)
class _OccurrenceOption:
    face_ids: frozenset[int]
    spec_key: str
    atom_ids: frozenset[int]
    mappings: tuple[FusionComponentMatch, ...]


@dataclass(frozen=True, slots=True)
class _Candidate:
    ast: FusionNameAst
    score: tuple
    rendered: str


def component_sides(spec: FusionComponentSpec) -> tuple[ComponentSide, ...]:
    """Return the component's directed sides in ``a``, ``b``, ... order."""

    order = spec.peripheral_order
    if len(order) < 3:
        raise FusionDescriptorError(f"component {spec.key!r} has no usable peripheral walk")
    return tuple(
        ComponentSide(
            letter=_alphabetic_index(index),
            start_locant=left,
            end_locant=right,
        )
        for index, (left, right) in enumerate(zip(order, order[1:] + order[:1]))
    )


def build_fusion_name_ast(
    mol: Molecule,
    component_matches: Sequence[FusionComponentMatch],
    registry: _Registry | Mapping[str, FusionComponentSpec],
) -> FusionNameAst:
    """Build the preferred bounded fusion citation from exact component maps.

    Selected components must cover every face exactly once, reconstruct their
    graph union exactly, and have a tree-shaped overlap graph. Alternative
    component identities, parent locations, and local numbering maps are
    enumerated under explicit bounds and compared without using molecular atom
    identifiers as nomenclature tie-breakers.
    """

    options = _occurrence_options(component_matches, registry)
    face_ids = tuple(sorted(set().union(*(option.face_ids for option in options))))
    if len(face_ids) < 2:
        raise FusionDescriptorError("a fusion citation requires at least two component faces")
    if len(face_ids) > MAX_COMPONENT_OCCURRENCES:
        raise FusionDescriptorError(
            f"component count {len(face_ids)} exceeds bounded limit {MAX_COMPONENT_OCCURRENCES}"
        )

    selections = _exact_component_covers(options, frozenset(face_ids))

    candidates: list[_Candidate] = []
    for selected_options in selections:
        candidates.extend(_candidates_for_component_selection(mol, selected_options, registry))
    if not candidates:
        raise FusionDescriptorError("no exact tree-cover fusion citation was found")
    return min(candidates, key=lambda candidate: (candidate.score, candidate.rendered)).ast


def render_fusion_name(
    ast: FusionNameAst,
    registry: _Registry | Mapping[str, FusionComponentSpec],
) -> str:
    """Render a fusion name using only the AST and component registry."""

    matches = {match.occurrence_id: match for match in ast.component_occurrences}
    descriptors = {
        join.attached_occurrence: descriptor for join, descriptor in zip(ast.joins, ast.descriptors, strict=True)
    }
    groups_by_member: dict[int, FusionMultiplicityGroup] = {}
    for group in ast.multiplicative_groups:
        for occurrence in group.occurrence_ids:
            groups_by_member[occurrence] = group

    rendered_groups: set[tuple[int, ...]] = set()

    def attachment(node: FusionCitationNode) -> str:
        descendants = render_children(node)
        spec = _spec(registry, matches[node.occurrence_id].spec_key)
        return f"{descendants}{spec.attached_prefix}{descriptors[node.occurrence_id].render()}"

    def render_children(node: FusionCitationNode) -> str:
        pieces: list[str] = []
        for child in node.children:
            group = groups_by_member.get(child.occurrence_id)
            if group is None:
                pieces.append(attachment(child))
                continue
            if group.occurrence_ids in rendered_groups:
                continue
            rendered_groups.add(group.occurrence_ids)
            members = tuple(_citation_node(ast.citation_tree, occurrence) for occurrence in group.occurrence_ids)
            if any(member.children for member in members):
                raise FusionDescriptorError("multiplicative rendering is limited to leaf components")
            prefixes = {_spec(registry, matches[member.occurrence_id].spec_key).attached_prefix for member in members}
            if len(prefixes) != 1:
                raise FusionDescriptorError("a multiplicative group must use one attached prefix")
            descriptor = _render_combined_descriptors(tuple(descriptors[member.occurrence_id] for member in members))
            pieces.append(f"{group.multiplier}{min(prefixes)}{descriptor}")
        return "".join(pieces)

    root = ast.citation_tree
    root_match = matches[root.occurrence_id]
    root_spec = _spec(registry, root_match.spec_key)
    return f"{render_children(root)}{root_spec.parent_name}"


def _occurrence_options(
    matches: Sequence[FusionComponentMatch],
    registry: _Registry | Mapping[str, FusionComponentSpec],
) -> tuple[_OccurrenceOption, ...]:
    grouped: dict[tuple[frozenset[int], str, frozenset[int]], list[FusionComponentMatch]] = defaultdict(list)
    for match in matches:
        if not match.covered_face_ids:
            continue
        atom_ids = frozenset(atom for _, atom in match.local_to_input_atom)
        grouped[(match.covered_face_ids, match.spec_key, atom_ids)].append(match)
    if not grouped:
        raise FusionDescriptorError("no exact component matches were supplied")

    options: list[_OccurrenceOption] = []
    for (face_ids, spec_key, atom_ids), mappings in grouped.items():
        _spec(registry, spec_key)
        unique = {mapping.local_to_input_atom: mapping for mapping in mappings}
        ordered = tuple(unique[key] for key in sorted(unique, key=_local_map_key))
        options.append(_OccurrenceOption(face_ids, spec_key, atom_ids, ordered))
    return tuple(sorted(options, key=_occurrence_option_key))


def _exact_component_covers(
    options: Sequence[_OccurrenceOption],
    target_face_ids: frozenset[int],
) -> tuple[tuple[_OccurrenceOption, ...], ...]:
    """Enumerate bounded, deterministic, disjoint covers of all target faces."""

    by_face: dict[int, tuple[_OccurrenceOption, ...]] = {
        face_id: tuple(option for option in options if face_id in option.face_ids)
        for face_id in sorted(target_face_ids)
    }
    if any(not candidates for candidates in by_face.values()):
        raise FusionDescriptorError("component matches do not cover every fusion face")

    covers: list[tuple[_OccurrenceOption, ...]] = []
    visited_states = 0

    def visit(covered: frozenset[int], selected: tuple[_OccurrenceOption, ...]) -> None:
        nonlocal visited_states
        visited_states += 1
        if visited_states > MAX_COMPONENT_SELECTION_STATES:
            raise FusionDescriptorError(
                f"component cover search exceeds bounded limit {MAX_COMPONENT_SELECTION_STATES}"
            )
        if covered == target_face_ids:
            covers.append(selected)
            if len(covers) > MAX_COMPONENT_SELECTIONS:
                raise FusionDescriptorError(
                    f"component selection count exceeds bounded limit {MAX_COMPONENT_SELECTIONS}"
                )
            return
        if len(selected) >= MAX_COMPONENT_OCCURRENCES:
            return
        uncovered = target_face_ids - covered
        pivot = min(uncovered, key=lambda face_id: (len(by_face[face_id]), face_id))
        for option in by_face[pivot]:
            if option.face_ids & covered or not option.face_ids <= target_face_ids:
                continue
            visit(covered | option.face_ids, selected + (option,))

    visit(frozenset(), ())
    if not covers:
        raise FusionDescriptorError("no bounded exact component cover was found")
    return tuple(covers)


def _candidates_for_component_selection(
    mol: Molecule,
    options: Sequence[_OccurrenceOption],
    registry: _Registry | Mapping[str, FusionComponentSpec],
) -> list[_Candidate]:
    ordered_options = tuple(sorted(options, key=_occurrence_option_key))
    canonical_matches = tuple(
        replace(option.mappings[0], occurrence_id=index) for index, option in enumerate(ordered_options)
    )
    specs = {match.occurrence_id: _spec(registry, match.spec_key) for match in canonical_matches}
    specs_by_key = {spec.key: spec for spec in specs.values()}
    scopes = tuple(_scope_for_match(mol, match, specs[match.occurrence_id]) for match in canonical_matches)
    target_atoms = frozenset(atom for scope in scopes for atom in scope.atom_ids)
    target_edges = frozenset(edge for scope in scopes for edge in scope.edges)
    audit = audit_component_cover(scopes, target_atom_ids=target_atoms, target_edges=target_edges)
    if not audit.ok or audit.proof.kind != "tree":
        return []
    if any(
        len(interface.shared_edges) != 1 or len(interface.shared_atom_ids) != 2 for interface in audit.graph.interfaces
    ):
        return []

    eligible_roots = [match for match in canonical_matches if specs[match.occurrence_id].usable_as_parent]
    if not eligible_roots:
        return []
    senior_key = min(component_seniority_key(match, specs_by_key) for match in eligible_roots)
    roots = [match for match in eligible_roots if component_seniority_key(match, specs_by_key) == senior_key]

    mapping_sets: list[tuple[FusionComponentMatch, ...]] = []
    for occurrence_id, option in enumerate(ordered_options):
        mapping_sets.append(tuple(replace(candidate, occurrence_id=occurrence_id) for candidate in option.mappings))
    map_count = _product_size(tuple(len(values) for values in mapping_sets))
    if map_count > MAX_LOCANT_MAP_COMBINATIONS:
        raise FusionDescriptorError(f"locant-map count {map_count} exceeds bounded limit {MAX_LOCANT_MAP_COMBINATIONS}")

    candidates: list[_Candidate] = []
    for root in roots:
        parent_by_child, order_by_occurrence = _orient_tree(audit.graph.adjacency, root.occurrence_id)
        if any(not specs[occurrence].usable_as_attached for occurrence in parent_by_child):
            continue
        interface_by_pair = {
            frozenset((interface.left, interface.right)): interface for interface in audit.graph.interfaces
        }
        for selected_maps in product(*mapping_sets):
            candidate = _build_candidate(
                selected_maps,
                specs,
                root.occurrence_id,
                parent_by_child,
                order_by_occurrence,
                interface_by_pair,
                registry,
                mol,
            )
            if candidate is not None:
                candidates.append(candidate)
    return candidates


def _build_candidate(
    matches: Sequence[FusionComponentMatch],
    specs: Mapping[int, FusionComponentSpec],
    root: int,
    parent_by_child: Mapping[int, int],
    order_by_occurrence: Mapping[int, int],
    interface_by_pair: Mapping[frozenset[int], FusionInterface[int]],
    registry: _Registry | Mapping[str, FusionComponentSpec],
    mol: Molecule,
) -> _Candidate | None:
    match_by_id = {match.occurrence_id: match for match in matches}
    raw_joins: dict[int, FusionJoin] = {}
    side_rank: dict[int, int] = {}
    for child, host in parent_by_child.items():
        interface = interface_by_pair[frozenset((child, host))]
        join_data = _ordinary_join(
            match_by_id[child],
            match_by_id[host],
            specs[child],
            specs[host],
            interface,
            order_by_occurrence[child],
            mol,
        )
        if join_data is None:
            return None
        raw_joins[child], side_rank[child] = join_data

    child_order = _ordered_children(parent_by_child, specs, raw_joins, side_rank)
    prime_depths, groups = _multiplicative_groups(child_order, specs, raw_joins)
    joins_by_child = {child: _with_prime_depths(join, prime_depths) for child, join in raw_joins.items()}
    citation_tree = _build_citation_tree(root, child_order)
    citation_children = _preorder_children(citation_tree)
    joins = tuple(joins_by_child[child] for child in citation_children)
    descriptors = tuple(
        FusionDescriptor(
            attached_locants=join.attached_locants,
            parent_sides=join.host_sides,
            parent_locants=join.host_locants,
            kind=join.kind,
        )
        for join in joins
    )
    ast = FusionNameAst(
        plan_kind=_plan_kind(len(matches), groups),
        parent_occurrences=(root,),
        component_occurrences=tuple(sorted(matches, key=lambda match: match.occurrence_id)),
        joins=joins,
        citation_tree=citation_tree,
        multiplicative_groups=groups,
        descriptors=descriptors,
    )
    rendered = render_fusion_name(ast, registry)
    specs_by_key = {spec.key: spec for spec in specs.values()}
    score = (
        component_seniority_key(match_by_id[root], specs_by_key).as_tuple(),
        len(matches),
        max(order_by_occurrence.values(), default=0),
        tuple(_join_preference_key(joins_by_child[child], side_rank[child]) for child in citation_children),
        tuple(component_seniority_key(match_by_id[child], specs_by_key).as_tuple() for child in citation_children),
        rendered,
    )
    return _Candidate(ast, score, rendered)


def _ordinary_join(
    attached: FusionComponentMatch,
    host: FusionComponentMatch,
    attached_spec: FusionComponentSpec,
    host_spec: FusionComponentSpec,
    interface: FusionInterface[int],
    order: int,
    mol: Molecule,
) -> tuple[FusionJoin, int] | None:
    if len(interface.shared_edges) != 1:
        return None
    shared_edge = next(iter(interface.shared_edges))
    host_map = host.input_atom_by_locant
    attached_inverse = {atom: locant for locant, atom in attached.local_to_input_atom}
    for side_index, side in enumerate(component_sides(host_spec)):
        directed_atoms = (host_map[side.start_locant], host_map[side.end_locant])
        if frozenset(directed_atoms) != frozenset(shared_edge):
            continue
        try:
            attached_text = tuple(attached_inverse[atom] for atom in directed_atoms)
        except KeyError:
            return None
        if frozenset(attached_text) not in {frozenset(bond.locants) for bond in attached_spec.bonds}:
            return None
        molecular_bond = mol.get_bond(*shared_edge)
        if molecular_bond is None:
            return None
        join = FusionJoin(
            attached_occurrence=attached.occurrence_id,
            host_occurrence=host.occurrence_id,
            order=order,
            kind=FusionJoinKind.ORTHO,
            attached_locants=tuple(ComponentLocant(attached.occurrence_id, locant) for locant in attached_text),
            host_sides=(FusionSide(host.occurrence_id, side.letter),),
            shared_input_atoms=interface.shared_atom_ids,
            shared_input_bonds=frozenset((molecular_bond.idx,)),
        )
        return join, side_index
    return None


def _scope_for_match(
    mol: Molecule,
    match: FusionComponentMatch,
    spec: FusionComponentSpec,
) -> ComponentScope[int]:
    atom_by_locant = match.input_atom_by_locant
    edges: list[tuple[int, int]] = []
    for bond in spec.bonds:
        edge = _edge(atom_by_locant[bond.locants[0]], atom_by_locant[bond.locants[1]])
        if mol.get_bond(*edge) is None:
            raise FusionDescriptorError(
                f"component {spec.key!r} maps local bond {bond.locants!r} to a missing molecular edge"
            )
        edges.append(edge)
    return component_scope(match.occurrence_id, atom_by_locant.values(), edges)


def _orient_tree(
    adjacency: Mapping[int, tuple[int, ...]],
    root: int,
) -> tuple[dict[int, int], dict[int, int]]:
    parent_by_child: dict[int, int] = {}
    order = {root: 0}
    pending = deque((root,))
    while pending:
        host = pending.popleft()
        for child in adjacency[host]:
            if child == parent_by_child.get(host) or child in order:
                continue
            parent_by_child[child] = host
            order[child] = order[host] + 1
            pending.append(child)
    return parent_by_child, order


def _ordered_children(
    parent_by_child: Mapping[int, int],
    specs: Mapping[int, FusionComponentSpec],
    joins: Mapping[int, FusionJoin],
    side_rank: Mapping[int, int],
) -> dict[int, tuple[int, ...]]:
    children: dict[int, list[int]] = defaultdict(list)
    synthetic_matches = {
        occurrence: FusionComponentMatch(
            occurrence_id=occurrence,
            spec_key=spec.key,
            covered_face_ids=frozenset((occurrence,)),
            local_to_input_atom=tuple((locant, index) for index, locant in enumerate(spec.locants)),
            local_to_skeleton_atom=tuple((locant, index) for index, locant in enumerate(spec.locants)),
            topology_key=(),
        )
        for occurrence, spec in specs.items()
    }
    specs_by_key = {spec.key: spec for spec in specs.values()}
    for child, parent in parent_by_child.items():
        children[parent].append(child)
    return {
        parent: tuple(
            sorted(
                values,
                key=lambda child: (
                    component_seniority_key(synthetic_matches[child], specs_by_key).as_tuple(),
                    side_rank[child],
                    _attached_locant_key(joins[child]),
                    specs[child].key,
                ),
            )
        )
        for parent, values in children.items()
    }


def _multiplicative_groups(
    child_order: Mapping[int, tuple[int, ...]],
    specs: Mapping[int, FusionComponentSpec],
    joins: Mapping[int, FusionJoin],
) -> tuple[dict[int, int], tuple[FusionMultiplicityGroup, ...]]:
    prime_depths: dict[int, int] = defaultdict(int)
    groups: list[FusionMultiplicityGroup] = []
    for _parent, children in sorted(child_order.items()):
        by_pattern: dict[tuple, list[int]] = defaultdict(list)
        for child in children:
            if child not in child_order:
                join = joins[child]
                pattern = (
                    specs[child].key,
                    join.kind,
                    join.order,
                    len(join.attached_locants),
                    len(join.host_sides),
                    len(join.host_locants),
                )
                by_pattern[pattern].append(child)
        for _pattern, members in sorted(by_pattern.items()):
            if len(members) < 2:
                continue
            ordered = tuple(sorted(members, key=lambda child: _descriptor_order_key(joins[child])))
            multiplier = _simple_multiplier(len(ordered))
            if multiplier is None:
                continue
            for depth, occurrence in enumerate(ordered):
                prime_depths[occurrence] = depth
            groups.append(FusionMultiplicityGroup(ordered, multiplier))
    groups.sort(key=lambda group: group.occurrence_ids)
    return dict(prime_depths), tuple(groups)


def _with_prime_depths(join: FusionJoin, depths: Mapping[int, int]) -> FusionJoin:
    attached_depth = depths.get(join.attached_occurrence, 0)
    host_depth = depths.get(join.host_occurrence, 0)
    return replace(
        join,
        attached_locants=tuple(replace(locant, prime_depth=attached_depth) for locant in join.attached_locants),
        host_sides=tuple(replace(side, prime_depth=host_depth) for side in join.host_sides),
        host_locants=tuple(replace(locant, prime_depth=host_depth) for locant in join.host_locants),
    )


def _build_citation_tree(root: int, children: Mapping[int, tuple[int, ...]]) -> FusionCitationNode:
    return FusionCitationNode(
        root,
        tuple(_build_citation_tree(child, children) for child in children.get(root, ())),
    )


def _preorder_children(root: FusionCitationNode) -> tuple[int, ...]:
    result: list[int] = []
    for child in root.children:
        result.extend(_preorder_children(child))
        result.append(child.occurrence_id)
    return tuple(result)


def _citation_node(root: FusionCitationNode, occurrence: int) -> FusionCitationNode:
    if root.occurrence_id == occurrence:
        return root
    for child in root.children:
        try:
            return _citation_node(child, occurrence)
        except KeyError:
            continue
    raise KeyError(occurrence)


def _render_combined_descriptors(descriptors: tuple[FusionDescriptor, ...]) -> str:
    interiors = tuple(descriptor.render()[1:-1] for descriptor in descriptors)
    return f"[{':'.join(interiors)}]"


def _descriptor_order_key(join: FusionJoin) -> tuple:
    return (
        tuple(_side_sort_key(side.letter) for side in join.host_sides),
        _attached_locant_key(join),
    )


def _join_preference_key(join: FusionJoin, side_rank: int) -> tuple:
    return (join.order, side_rank, _attached_locant_key(join))


def _attached_locant_key(join: FusionJoin) -> tuple:
    return tuple(_locant_sort_key(locant.text) for locant in join.attached_locants)


def _local_map_key(mapping: tuple[tuple[str, int], ...]) -> tuple:
    return tuple((_locant_sort_key(locant), atom) for locant, atom in mapping)


def _occurrence_option_key(option: _OccurrenceOption) -> tuple:
    return (
        min(option.face_ids),
        -len(option.face_ids),
        tuple(sorted(option.face_ids)),
        option.spec_key,
        tuple(sorted(option.atom_ids)),
    )


def _locant_sort_key(locant: str) -> tuple[int, str]:
    position = 0
    while position < len(locant) and locant[position].isdigit():
        position += 1
    number = int(locant[:position]) if position else 1_000_000
    return number, locant[position:]


def _side_sort_key(letter: str) -> tuple[int, ...]:
    return tuple(ord(char) - ord("a") for char in letter)


def _alphabetic_index(index: int) -> str:
    if index < 0:
        raise ValueError("alphabetic index must be non-negative")
    value = index + 1
    letters = ""
    while value:
        value, remainder = divmod(value - 1, 26)
        letters = chr(ord("a") + remainder) + letters
    return letters


def _simple_multiplier(count: int) -> str | None:
    return {2: "di", 3: "tri", 4: "tetra", 5: "penta", 6: "hexa"}.get(count)


def _plan_kind(component_count: int, groups: tuple[FusionMultiplicityGroup, ...]) -> str:
    if groups:
        return "multiplicative_tree"
    return "two_component" if component_count == 2 else "polycomponent_tree"


def _spec(
    registry: _Registry | Mapping[str, FusionComponentSpec],
    key: str,
) -> FusionComponentSpec:
    if isinstance(registry, Mapping):
        value = registry.get(key)
    else:
        value = registry.by_key.get(key)
    if value is None:
        raise FusionDescriptorError(f"unknown fusion component {key!r}")
    spec = getattr(value, "spec", value)
    if not isinstance(spec, FusionComponentSpec):
        raise FusionDescriptorError(f"registry value for {key!r} is not a fusion component spec")
    return spec


def _edge(left: int, right: int) -> tuple[int, int]:
    return (left, right) if left < right else (right, left)


def _product_size(sizes: tuple[int, ...]) -> int:
    result = 1
    for size in sizes:
        result *= size
    return result
