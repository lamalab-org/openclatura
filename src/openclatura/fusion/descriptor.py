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
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import combinations
from typing import Protocol

from ..assembly_parts import NameTokenBinding
from ..locants import retained_locant_sort_key
from ..molecule import Molecule
from ..polycycle_topology import normalize_edge
from ..rules import multipliers
from .config import fusion_nomenclature_config
from .cover import ComponentScope, FusionInterface, audit_component_cover, component_scope
from .model import (
    ComponentLocant,
    FusionCitationNode,
    FusionCitationPlan,
    FusionComponentMatch,
    FusionComponentSpec,
    FusionDescriptor,
    FusionJoin,
    FusionJoinKind,
    FusionMultiplicityGroup,
    FusionNameAst,
    FusionSide,
    OrderedFusionInterface,
    ParentLocationKey,
)
from .rules import (
    component_spec_seniority_key,
    multiplicative_attachment_key,
    multiplicative_member_order_key,
)

_LIMITS = fusion_nomenclature_config().search
_SUPPORT = fusion_nomenclature_config().rules.support
MAX_COMPONENT_OCCURRENCES = _LIMITS.maximum_component_occurrences
MAX_FACES = _LIMITS.maximum_faces
MAX_COMPONENT_SELECTIONS = _LIMITS.maximum_component_selections
MAX_COMPONENT_SELECTION_STATES = _LIMITS.component_selection_states
MAX_LOCANT_MAP_COMBINATIONS = _LIMITS.locant_map_combinations


class FusionDescriptorError(ValueError):
    """Raised when the bounded descriptor tier cannot prove a valid AST."""


class _LocantMapBudgetExceeded(RuntimeError):
    """Internal signal that one candidate cover exceeded its deterministic budget."""


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


@dataclass(frozen=True, slots=True)
class _CitationTopology:
    roots: tuple[int, ...]
    parent_by_child: Mapping[int, int]
    order_by_occurrence: Mapping[int, int]
    interparent_pairs: tuple[frozenset[int], ...]
    cycle_closing_pairs: tuple[frozenset[int], ...]
    interparent_occurrences: tuple[int, ...]
    location_key: ParentLocationKey


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
    *,
    cover_kinds: Iterable[str] | None = None,
    join_kinds: Iterable[str] | None = None,
    multiparent_parents: bool | None = None,
) -> FusionNameAst:
    """Build the preferred bounded fusion citation from exact component maps.

    Selected components must cover every face exactly once, reconstruct their
    graph union exactly, and have a tree-shaped overlap graph. Alternative
    component identities, parent locations, and local numbering maps are
    enumerated under explicit bounds and compared without using molecular atom
    identifiers as nomenclature tie-breakers.
    """

    supported_covers = frozenset(_SUPPORT.cover_kinds if cover_kinds is None else cover_kinds)
    supported_joins = frozenset(_SUPPORT.join_kinds if join_kinds is None else join_kinds)
    allow_multiparent_parents = (
        _SUPPORT.multiparent_parents
        if multiparent_parents is None
        else multiparent_parents
    )
    if not supported_covers <= {"tree", "multiparent"}:
        raise ValueError("unknown fusion cover kind")
    if not supported_joins <= {"ortho", "ortho_peri", "higher_order"}:
        raise ValueError("unknown fusion join kind")

    options = _occurrence_options(component_matches, registry)
    face_ids = tuple(sorted(set().union(*(option.face_ids for option in options))))
    if len(face_ids) < 2:
        raise FusionDescriptorError("a fusion citation requires at least two component faces")
    if len(face_ids) > MAX_FACES:
        raise FusionDescriptorError(
            f"face count {len(face_ids)} exceeds bounded limit {MAX_FACES}"
        )

    selections = _exact_component_covers(options, frozenset(face_ids))

    candidates: list[_Candidate] = []
    budget_exhausted = 0
    for selected_options in selections:
        try:
            candidates.extend(
                _candidates_for_component_selection(
                    mol,
                    selected_options,
                    registry,
                    supported_covers=supported_covers,
                    supported_joins=supported_joins,
                    allow_multiparent_parents=allow_multiparent_parents,
                )
            )
        except _LocantMapBudgetExceeded:
            budget_exhausted += 1
    if not candidates:
        if budget_exhausted:
            raise FusionDescriptorError(
                f"all viable component covers exceeded the locant-map budget of {MAX_LOCANT_MAP_COMBINATIONS} states"
            )
        tier = "tree-cover" if supported_covers == {"tree"} else "supported-cover"
        raise FusionDescriptorError(f"no exact {tier} fusion citation was found")
    return min(candidates, key=lambda candidate: (candidate.score, candidate.rendered)).ast


def render_fusion_name(
    ast: FusionNameAst,
    registry: _Registry | Mapping[str, FusionComponentSpec],
) -> str:
    """Render a fusion name using only the AST and component registry."""

    return "".join(part.text for part in render_fusion_name_parts(ast, registry))


def render_fusion_name_parts(
    ast: FusionNameAst,
    registry: _Registry | Mapping[str, FusionComponentSpec],
    *,
    mol: Molecule | None = None,
) -> tuple[NameTokenBinding, ...]:
    """Render ordered text parts while preserving component/interface scope."""

    matches = {match.occurrence_id: match for match in ast.component_occurrences}
    descriptors_by_attached: dict[int, list[FusionDescriptor]] = defaultdict(list)
    for join, descriptor in zip(ast.joins, ast.descriptors, strict=True):
        descriptors_by_attached[join.attached_occurrence].append(descriptor)
    groups_by_member: dict[int, FusionMultiplicityGroup] = {}
    for group in ast.multiplicative_groups:
        for occurrence in group.occurrence_ids:
            groups_by_member[occurrence] = group

    rendered_groups: set[tuple[int, ...]] = set()

    def rendered_component_scope(occurrence_ids: tuple[int, ...]) -> tuple[frozenset[int], frozenset[int]]:
        atom_ids: set[int] = set()
        bond_ids: set[int] = set()
        for occurrence_id in occurrence_ids:
            match = matches[occurrence_id]
            spec = _spec_for_match(registry, match)
            local = match.input_atom_by_locant
            atom_ids.update(local.values())
            if mol is not None:
                for bond in spec.bonds:
                    molecular_bond = mol.get_bond(local[bond.locants[0]], local[bond.locants[1]])
                    if molecular_bond is not None:
                        bond_ids.add(molecular_bond.idx)
        return frozenset(atom_ids), frozenset(bond_ids)

    def component_part(text: str, role: str, occurrence_ids: tuple[int, ...]) -> NameTokenBinding:
        atom_ids, bond_ids = rendered_component_scope(occurrence_ids)
        occurrence_key = ",".join(map(str, occurrence_ids))
        return NameTokenBinding(
            text=text,
            token_kind="parent",
            source="fusion_renderer",
            grammar_role=f"fusion_{role}",
            binding_key=f"fusion:{role}:occurrences={occurrence_key}",
            atom_ids=set(atom_ids),
            bond_ids=set(bond_ids),
            match_priority=100,
        )

    def descriptor_part(text: str, occurrence_ids: tuple[int, ...]) -> NameTokenBinding:
        interfaces = tuple(join for join in ast.joins if join.attached_occurrence in occurrence_ids)
        occurrence_key = ",".join(map(str, occurrence_ids))
        return NameTokenBinding(
            text=text,
            token_kind="locant",
            source="fusion_renderer",
            grammar_role="fusion_descriptor",
            binding_key=f"fusion:descriptor:interfaces={occurrence_key}",
            atom_ids=frozenset().union(*(join.shared_input_atoms for join in interfaces)),
            bond_ids=frozenset().union(*(join.shared_input_bonds for join in interfaces)),
            match_priority=100,
        )

    def attachment(node: FusionCitationNode) -> list[NameTokenBinding]:
        descendants = render_children(node)
        spec = _spec_for_match(registry, matches[node.occurrence_id])
        descriptors = tuple(descriptors_by_attached[node.occurrence_id])
        descriptor_text = _combine_rendered_descriptors(
            tuple(
                _render_descriptor(
                    descriptor,
                    omit_attached_locants=(
                        len(descriptors) == 1
                        and _omit_attached_locants(registry, spec.key)
                    ),
                )
                for descriptor in descriptors
            )
        )
        return [
            *descendants,
            component_part(spec.attached_prefix, "attached_component", (node.occurrence_id,)),
            descriptor_part(
                descriptor_text,
                (node.occurrence_id,),
            ),
        ]

    def render_children(node: FusionCitationNode) -> list[NameTokenBinding]:
        pieces: list[NameTokenBinding] = []
        for child in node.children:
            group = groups_by_member.get(child.occurrence_id)
            if group is None:
                pieces.extend(attachment(child))
                continue
            if group.occurrence_ids in rendered_groups:
                continue
            rendered_groups.add(group.occurrence_ids)
            members = tuple(_citation_node(ast.citation_tree, occurrence) for occurrence in group.occurrence_ids)
            if any(member.children for member in members):
                raise FusionDescriptorError("multiplicative rendering is limited to leaf components")
            prefixes = {_spec_for_match(registry, matches[member.occurrence_id]).attached_prefix for member in members}
            if len(prefixes) != 1:
                raise FusionDescriptorError("a multiplicative group must use one attached prefix")
            member_descriptors = tuple(
                descriptors_by_attached[member.occurrence_id][0] for member in members
            )
            descriptor = _combine_rendered_descriptors(
                tuple(
                    _render_descriptor(
                        item,
                        omit_attached_locants=_omit_attached_locants(
                            registry,
                            _spec_for_match(registry, matches[member.occurrence_id]).key,
                        ),
                    )
                    for item, member in zip(
                        member_descriptors,
                        members,
                        strict=True,
                    )
                )
            )
            pieces.extend(
                (
                    NameTokenBinding(
                        text=group.multiplier,
                        token_kind="grammar",
                        source="fusion_renderer",
                        grammar_role="fusion_multiplier",
                        binding_key=(
                            "fusion:multiplier:occurrences="
                            + ",".join(map(str, group.occurrence_ids))
                        ),
                        match_priority=100,
                    ),
                    component_part(min(prefixes), "attached_component", group.occurrence_ids),
                    descriptor_part(descriptor, group.occurrence_ids),
                )
            )
        return pieces

    plan = ast.citation_plan
    if plan is None:
        if ast.citation_tree is None:
            raise FusionDescriptorError("fusion AST has no citation plan")
        plan = FusionCitationPlan.from_tree(ast.citation_tree, ast.joins)

    if (
        len(plan.roots) > 1
        or plan.interparent_join_indices
        or plan.cycle_closing_join_indices
    ):
        pieces: list[NameTokenBinding] = []
        for occurrence in plan.render_order:
            spec = _spec_for_match(registry, matches[occurrence])
            descriptors = tuple(descriptors_by_attached[occurrence])
            if not descriptors:
                raise FusionDescriptorError(
                    f"nonparent component occurrence {occurrence} has no fusion descriptor"
                )
            pieces.extend(
                (
                    component_part(spec.attached_prefix, "attached_component", (occurrence,)),
                    descriptor_part(
                        _combine_rendered_descriptors(
                            tuple(
                                _render_descriptor(descriptor, omit_attached_locants=False)
                                for descriptor in descriptors
                            )
                        ),
                        (occurrence,),
                    ),
                )
            )
        root_specs = tuple(
            _spec_for_match(registry, matches[root.occurrence_id]) for root in plan.roots
        )
        parent_names = {spec.parent_name for spec in root_specs}
        if len(parent_names) != 1:
            raise FusionDescriptorError("multiparent rendering requires identical parent components")
        parent_name = next(iter(parent_names))
        if len(root_specs) == 1:
            pieces.append(component_part(parent_name, "parent_component", plan.parent_occurrences))
        else:
            try:
                multiplier = multipliers.basic(len(root_specs))
            except KeyError as exc:
                raise FusionDescriptorError("unsupported multiparent multiplicity") from exc
            pieces.append(
                component_part(
                    f"{multiplier}{parent_name}",
                    "multiparent_components",
                    plan.parent_occurrences,
                )
            )
        return tuple(pieces)

    root = plan.roots[0]
    root_spec = _spec_for_match(registry, matches[root.occurrence_id])
    return tuple(
        [
            *render_children(root),
            component_part(root_spec.parent_name, "parent_component", (root.occurrence_id,)),
        ]
    )


def _occurrence_options(
    matches: Sequence[FusionComponentMatch],
    registry: _Registry | Mapping[str, FusionComponentSpec],
) -> tuple[_OccurrenceOption, ...]:
    grouped: dict[tuple[frozenset[int], str, str, frozenset[int]], list[FusionComponentMatch]] = defaultdict(list)
    for match in matches:
        if not match.covered_face_ids:
            continue
        atom_ids = frozenset(atom for _, atom in match.local_to_input_atom)
        grouped[(match.covered_face_ids, match.spec_key, match.template_name, atom_ids)].append(match)
    if not grouped:
        raise FusionDescriptorError("no exact component matches were supplied")

    options: list[_OccurrenceOption] = []
    for (face_ids, spec_key, _template_name, atom_ids), mappings in grouped.items():
        _spec_for_match(registry, mappings[0])
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
    *,
    supported_covers: frozenset[str],
    supported_joins: frozenset[str],
    allow_multiparent_parents: bool,
) -> list[_Candidate]:
    if len(options) > MAX_COMPONENT_OCCURRENCES:
        return []
    ordered_options = tuple(sorted(options, key=_occurrence_option_key))
    canonical_matches = tuple(
        replace(option.mappings[0], occurrence_id=index) for index, option in enumerate(ordered_options)
    )
    specs = {match.occurrence_id: _spec_for_match(registry, match) for match in canonical_matches}
    scopes = tuple(_scope_for_match(mol, match, specs[match.occurrence_id]) for match in canonical_matches)
    target_atoms = frozenset(atom for scope in scopes for atom in scope.atom_ids)
    target_edges = frozenset(edge for scope in scopes for edge in scope.edges)
    audit = audit_component_cover(scopes, target_atom_ids=target_atoms, target_edges=target_edges)
    if not audit.ok or audit.proof.kind not in supported_covers:
        return []
    if any(_interface_support_key(interface) not in supported_joins for interface in audit.graph.interfaces):
        return []

    eligible_roots = [match for match in canonical_matches if specs[match.occurrence_id].usable_as_parent]
    if not eligible_roots:
        return []
    senior_key = min(component_spec_seniority_key(specs[match.occurrence_id]) for match in eligible_roots)
    roots = [
        match
        for match in eligible_roots
        if component_spec_seniority_key(specs[match.occurrence_id]) == senior_key
    ]
    root_sets = [(root.occurrence_id,) for root in roots]
    if allow_multiparent_parents:
        root_sets.extend(_multiparent_root_sets(tuple(root.occurrence_id for root in roots), audit.graph.adjacency, specs))
    topologies = tuple(
        _citation_topology(audit.graph.adjacency, root_set, specs)
        for root_set in root_sets
    )
    if audit.proof.kind == "multiparent" and "higher_order" not in supported_joins:
        return []
    preferred_location = min(_pre_mapping_location_key(topology.location_key) for topology in topologies)
    topologies = tuple(
        topology
        for topology in topologies
        if _pre_mapping_location_key(topology.location_key) == preferred_location
    )

    mapping_sets: list[tuple[FusionComponentMatch, ...]] = []
    for occurrence_id, option in enumerate(ordered_options):
        mapping_sets.append(tuple(replace(candidate, occurrence_id=occurrence_id) for candidate in option.mappings))
    candidates: list[_Candidate] = []
    for topology in topologies:
        if any(not specs[occurrence].usable_as_attached for occurrence in topology.parent_by_child):
            continue
        interface_by_pair = {
            frozenset((interface.left, interface.right)): interface for interface in audit.graph.interfaces
        }
        if len(topology.roots) == 1 and not (
            topology.interparent_pairs or topology.cycle_closing_pairs
        ):
            candidate = _best_tree_mapping_candidate(
                mapping_sets,
                specs,
                topology,
                interface_by_pair,
                registry,
                mol,
                supported_joins,
                audit.proof.kind,
            )
            if candidate is not None:
                candidates.append(candidate)
            continue
        for selected_maps in _compatible_mapping_assignments(
            mapping_sets,
            specs,
            topology.roots[0],
            topology.parent_by_child,
            topology.order_by_occurrence,
            interface_by_pair,
            mol,
        ):
            candidate = _build_candidate(
                selected_maps,
                specs,
                topology,
                interface_by_pair,
                registry,
                mol,
                supported_joins,
                audit.proof.kind,
            )
            if candidate is not None:
                candidates.append(candidate)
    return candidates


def _best_tree_mapping_candidate(
    mapping_sets: Sequence[tuple[FusionComponentMatch, ...]],
    specs: Mapping[int, FusionComponentSpec],
    topology: _CitationTopology,
    interface_by_pair: Mapping[frozenset[int], FusionInterface[int]],
    registry: _Registry | Mapping[str, FusionComponentSpec],
    mol: Molecule,
    supported_joins: frozenset[str],
    cover_kind: str,
) -> _Candidate | None:
    """Return the exact preferred mapping assignment for a citation tree.

    Local component automorphisms are selected while walking the citation in
    the same postorder used by the final preference key.  Once an emitted
    join-key prefix is lexicographically worse than the best complete prefix,
    no descendant choice can improve it, so that branch can be discarded.
    This avoids materializing the exponential Cartesian product for long
    unbranched systems without changing the ordering rules.
    """

    projected = _deduplicate_interface_equivalent_maps(
        mapping_sets,
        interface_by_pair.values(),
    )
    root = topology.roots[0]
    children: dict[int, tuple[int, ...]] = {
        parent: tuple(child for child, host in topology.parent_by_child.items() if host == parent)
        for parent in range(len(projected))
    }
    selected: dict[int, FusionComponentMatch] = {}
    selected_joins: dict[int, FusionJoin] = {}
    side_ranks: dict[int, int] = {}
    best: _Candidate | None = None
    visited_states = 0

    def prefix_is_worse(prefix: tuple[tuple, ...]) -> bool:
        if best is None:
            return False
        return prefix > best.score[4][: len(prefix)]

    def complete(prefix: tuple[tuple, ...]) -> None:
        nonlocal best
        candidate = _build_candidate(
            tuple(selected[occurrence] for occurrence in range(len(projected))),
            specs,
            topology,
            interface_by_pair,
            registry,
            mol,
            supported_joins,
            cover_kind,
        )
        if candidate is not None and (best is None or candidate.score < best.score):
            best = candidate

    def visit_ordered_children(
        ordered: tuple[int, ...],
        index: int,
        prefix: tuple[tuple, ...],
        continuation,
    ) -> None:
        if index == len(ordered):
            continuation(prefix)
            return
        child = ordered[index]

        def after_child(child_prefix: tuple[tuple, ...]) -> None:
            next_prefix = child_prefix + (
                _join_preference_key(selected_joins[child], side_ranks[child]),
            )
            if not prefix_is_worse(next_prefix):
                visit_ordered_children(ordered, index + 1, next_prefix, continuation)

        visit_node(child, prefix, after_child)

    def visit_node(node: int, prefix: tuple[tuple, ...], continuation) -> None:
        child_ids = children.get(node, ())

        def select_direct_child(position: int) -> None:
            nonlocal visited_states
            if position == len(child_ids):
                ordered = tuple(
                    sorted(
                        child_ids,
                        key=lambda child: (
                            component_spec_seniority_key(specs[child]).as_tuple(),
                            side_ranks[child],
                            _attached_locant_key(selected_joins[child]),
                            specs[child].key,
                        ),
                    )
                )
                visit_ordered_children(ordered, 0, prefix, continuation)
                return

            child = child_ids[position]
            interface = interface_by_pair[frozenset((child, node))]
            for mapping in projected[child]:
                visited_states += 1
                if visited_states > MAX_LOCANT_MAP_COMBINATIONS:
                    raise _LocantMapBudgetExceeded(
                        f"tree locant-map search exceeds bounded limit {MAX_LOCANT_MAP_COMBINATIONS} states"
                    )
                classified = _classified_join(
                    mapping,
                    selected[node],
                    specs[child],
                    specs[node],
                    interface,
                    topology.order_by_occurrence[child],
                    mol,
                )
                if classified is None:
                    continue
                join, side_rank = classified
                selected[child] = mapping
                selected_joins[child] = join
                side_ranks[child] = side_rank
                select_direct_child(position + 1)
                del side_ranks[child]
                del selected_joins[child]
                del selected[child]

        select_direct_child(0)

    for root_mapping in projected[root]:
        selected[root] = root_mapping
        visit_node(root, (), complete)
        del selected[root]
    return best


def _multiparent_root_sets(
    roots: tuple[int, ...],
    adjacency: Mapping[int, tuple[int, ...]],
    specs: Mapping[int, FusionComponentSpec],
) -> tuple[tuple[int, ...], ...]:
    """Return the complete set of identical, nonadjacent senior parents.

    Multiparent nomenclature cites every occurrence of the intrinsically
    senior parent component. Choosing arbitrary subsets changes the citation
    and creates an exponential search, so the check is deliberately linear.
    """

    if len(roots) < 2 or len({specs[root].key for root in roots}) != 1:
        return ()
    selected = tuple(sorted(roots, key=lambda root: _cover_node_key(root, adjacency, specs)))
    if any(right in adjacency[left] for left, right in combinations(selected, 2)):
        return ()
    topology = _citation_topology(adjacency, selected, specs)
    return (selected,) if topology.interparent_occurrences else ()


def _citation_topology(
    adjacency: Mapping[int, tuple[int, ...]],
    roots: tuple[int, ...],
    specs: Mapping[int, FusionComponentSpec],
) -> _CitationTopology:
    """Build one deterministic spanning forest and retain all closing edges."""

    root_set = set(roots)
    parent_by_child: dict[int, int] = {}
    order = {root: 0 for root in roots}
    owner = {root: root for root in roots}
    pending = deque(sorted(roots, key=lambda node: _cover_node_key(node, adjacency, specs)))
    while pending:
        host = pending.popleft()
        for child in sorted(
            adjacency[host],
            key=lambda node: _cover_node_key(node, adjacency, specs),
        ):
            if child in order:
                continue
            parent_by_child[child] = host
            order[child] = order[host] + 1
            owner[child] = owner[host]
            pending.append(child)

    primary_pairs = {
        frozenset((child, host)) for child, host in parent_by_child.items()
    }
    all_pairs = {
        frozenset((left, right))
        for left, neighbors in adjacency.items()
        for right in neighbors
        if left != right
    }
    closing = tuple(
        sorted(
            all_pairs - primary_pairs,
            key=lambda pair: tuple(
                sorted(_cover_node_key(node, adjacency, specs) for node in pair)
            ),
        )
    )
    interparent_pairs = tuple(
        pair
        for pair in closing
        if len({owner.get(node) for node in pair}) > 1
    )
    cycle_closing_pairs = tuple(pair for pair in closing if pair not in interparent_pairs)
    interparents = tuple(
        sorted(
            {
                node
                for pair in interparent_pairs
                for node in pair
                if node not in root_set
            },
            key=lambda node: _cover_node_key(node, adjacency, specs),
        )
    )
    location = _parent_location_key(
        parent_by_child,
        order,
        roots,
        interparents,
        specs,
    )
    return _CitationTopology(
        roots=roots,
        parent_by_child=parent_by_child,
        order_by_occurrence=order,
        interparent_pairs=interparent_pairs,
        cycle_closing_pairs=cycle_closing_pairs,
        interparent_occurrences=interparents,
        location_key=location,
    )


def _cover_node_key(
    occurrence: int,
    adjacency: Mapping[int, tuple[int, ...]],
    specs: Mapping[int, FusionComponentSpec],
) -> tuple:
    return (
        component_spec_seniority_key(specs[occurrence]).as_tuple(),
        -len(adjacency[occurrence]),
        specs[occurrence].key,
        occurrence,
    )


def _tree_parent_location_key(
    adjacency: Mapping[int, tuple[int, ...]],
    root: int,
    specs: Mapping[int, FusionComponentSpec],
) -> ParentLocationKey:
    """Return the topology-only parent-location criteria for a tree cover."""

    parent_by_child, order_by_occurrence = _orient_tree(adjacency, root)
    return _parent_location_key(
        parent_by_child,
        order_by_occurrence,
        (root,),
        (),
        specs,
    )


def _parent_location_key(
    parent_by_child: Mapping[int, int],
    order_by_occurrence: Mapping[int, int],
    roots: tuple[int, ...],
    interparents: tuple[int, ...],
    specs: Mapping[int, FusionComponentSpec],
) -> ParentLocationKey:
    """Return the complete deterministic P-25 parent-location key."""

    maximum_order = max(order_by_occurrence.values(), default=0)
    counts = tuple(
        -sum(order == level for order in order_by_occurrence.values())
        for level in range(1, maximum_order + 1)
    )
    parent_keys = {specs[root].key for root in roots}
    incomplete = int(
        len(roots) == 1
        and any(
            occurrence not in roots and specs[occurrence].key in parent_keys
            for occurrence in specs
        )
    )
    interparent_seniority = tuple(
        sorted(component_spec_seniority_key(specs[occurrence]).as_tuple() for occurrence in interparents)
    )
    attached = tuple(
        sorted(
            component_spec_seniority_key(specs[occurrence]).as_tuple()
            for occurrence in specs
            if occurrence not in roots and occurrence not in interparents
        )
    )
    return ParentLocationKey(
        incomplete_system=incomplete,
        maximum_attachment_order=maximum_order,
        attachment_count_by_order=counts,
        # Exact multiplicative equivalence depends on the selected local
        # interface orbits and is filled after locant-map selection.
        multiplicative_grouping_score=(),
        interparent_seniority=interparent_seniority,
        attached_component_preference=attached,
    )


def _pre_mapping_location_key(location: ParentLocationKey) -> tuple:
    """Criteria that are fully known before local component maps are chosen."""

    return (
        location.incomplete_system,
        location.maximum_attachment_order,
        location.attachment_count_by_order,
    )


def _compatible_mapping_assignments(
    mapping_sets: Sequence[tuple[FusionComponentMatch, ...]],
    specs: Mapping[int, FusionComponentSpec],
    root: int,
    parent_by_child: Mapping[int, int],
    order_by_occurrence: Mapping[int, int],
    interface_by_pair: Mapping[frozenset[int], FusionInterface[int]],
    mol: Molecule,
) -> tuple[tuple[FusionComponentMatch, ...], ...]:
    """Assign local maps incrementally along the component-cover tree.

    A component automorphism is relevant only when it maps the shared atoms to
    a valid directed fusion side. Checking that constraint as each child is
    attached avoids constructing the Cartesian product of unrelated maps.
    """

    projected_mapping_sets = _deduplicate_interface_equivalent_maps(
        mapping_sets,
        interface_by_pair.values(),
    )
    occurrence_order = tuple(
        sorted(range(len(mapping_sets)), key=lambda occurrence: (order_by_occurrence[occurrence], occurrence))
    )
    if not occurrence_order or occurrence_order[0] != root:
        raise FusionDescriptorError("component cover tree has no numbered root")

    assignments: list[tuple[FusionComponentMatch, ...]] = []
    selected: dict[int, FusionComponentMatch] = {}
    visited_states = 0

    def visit(position: int) -> None:
        nonlocal visited_states
        visited_states += 1
        if visited_states > MAX_LOCANT_MAP_COMBINATIONS:
            raise _LocantMapBudgetExceeded(
                f"compatible locant-map search exceeds bounded limit {MAX_LOCANT_MAP_COMBINATIONS}"
            )
        if position == len(occurrence_order):
            assignments.append(tuple(selected[occurrence] for occurrence in range(len(projected_mapping_sets))))
            return

        occurrence = occurrence_order[position]
        host = parent_by_child.get(occurrence)
        for candidate in projected_mapping_sets[occurrence]:
            if host is not None:
                interface = interface_by_pair[frozenset((occurrence, host))]
                if (
                    _classified_join(
                        candidate,
                        selected[host],
                        specs[occurrence],
                        specs[host],
                        interface,
                        order_by_occurrence[occurrence],
                        mol,
                    )
                    is None
                ):
                    continue
            selected[occurrence] = candidate
            visit(position + 1)
            del selected[occurrence]

    visit(0)
    return tuple(assignments)


def _deduplicate_interface_equivalent_maps(
    mapping_sets: Sequence[tuple[FusionComponentMatch, ...]],
    interfaces: Iterable[FusionInterface[int]],
) -> tuple[tuple[FusionComponentMatch, ...], ...]:
    """Keep one full map for each distinct assignment at fusion interfaces."""

    interface_atoms: dict[int, set[int]] = defaultdict(set)
    for interface in interfaces:
        interface_atoms[interface.left].update(interface.shared_atom_ids)
        interface_atoms[interface.right].update(interface.shared_atom_ids)

    result = []
    for occurrence, mappings in enumerate(mapping_sets):
        relevant_atoms = interface_atoms[occurrence]
        by_projection: dict[tuple[tuple[int, str], ...], FusionComponentMatch] = {}
        for mapping in mappings:
            locant_by_atom = {atom: locant for locant, atom in mapping.local_to_input_atom}
            projection = tuple((atom, locant_by_atom[atom]) for atom in sorted(relevant_atoms))
            by_projection.setdefault(projection, mapping)
        result.append(tuple(by_projection[key] for key in sorted(by_projection)))
    return tuple(result)


def _build_candidate(
    matches: Sequence[FusionComponentMatch],
    specs: Mapping[int, FusionComponentSpec],
    topology: _CitationTopology,
    interface_by_pair: Mapping[frozenset[int], FusionInterface[int]],
    registry: _Registry | Mapping[str, FusionComponentSpec],
    mol: Molecule,
    supported_joins: frozenset[str],
    cover_kind: str,
) -> _Candidate | None:
    match_by_id = {match.occurrence_id: match for match in matches}
    primary_joins: dict[int, FusionJoin] = {}
    interparent_joins: list[FusionJoin] = []
    cycle_closing_joins: list[FusionJoin] = []
    side_rank: dict[int, int] = {}
    for child, host in topology.parent_by_child.items():
        interface = interface_by_pair[frozenset((child, host))]
        join_data = _classified_join(
            match_by_id[child],
            match_by_id[host],
            specs[child],
            specs[host],
            interface,
            topology.order_by_occurrence[child],
            mol,
        )
        if join_data is None:
            return None
        join, side_rank[child] = join_data
        if (
            cover_kind == "multiparent"
            and topology.order_by_occurrence[child] > 1
            and "higher_order" in supported_joins
        ):
            join = replace(
                join,
                interface=replace(
                    join.interface,
                    kind=FusionJoinKind.HIGHER_ORDER,
                    host_sides=(),
                    host_locants=join.interface.host_path,
                ),
            )
        primary_joins[child] = join

    for pair in (*topology.interparent_pairs, *topology.cycle_closing_pairs):
        attached, host = _closing_join_direction(
            pair,
            topology.roots,
            topology.order_by_occurrence,
            specs,
        )
        classified = classify_ordered_fusion_interface(
            match_by_id[attached],
            match_by_id[host],
            specs[attached],
            specs[host],
            interface_by_pair[pair],
            mol,
        )
        if classified is None:
            return None
        evidence, _rank = classified
        join = FusionJoin(
            order=max(topology.order_by_occurrence[attached], 1) + 1,
            interface=(
                evidence
                if pair in topology.interparent_pairs
                else replace(
                    evidence,
                    kind=FusionJoinKind.HIGHER_ORDER,
                    host_sides=(),
                    host_locants=evidence.host_path,
                )
            ),
        )
        target = interparent_joins if pair in topology.interparent_pairs else cycle_closing_joins
        target.append(join)

    child_order = _ordered_children(topology.parent_by_child, specs, primary_joins, side_rank)
    prime_depths, groups = _multiplicative_groups(child_order, specs, primary_joins)
    if len(topology.roots) > 1:
        groups = ()
        prime_depths.update({root: depth for depth, root in enumerate(topology.roots)})
    joins_by_child = {
        child: _with_prime_depths(join, prime_depths)
        for child, join in primary_joins.items()
    }
    roots = tuple(_build_citation_tree(root, child_order) for root in topology.roots)
    citation_children = tuple(
        occurrence for root in roots for occurrence in _preorder_children(root)
    )
    primary = tuple(joins_by_child[child] for child in citation_children)
    interparent = tuple(_with_prime_depths(join, prime_depths) for join in interparent_joins)
    cycle_closing = tuple(
        _with_prime_depths(join, prime_depths) for join in cycle_closing_joins
    )
    joins = (*primary, *interparent, *cycle_closing)
    descriptors = tuple(
        FusionDescriptor.from_interface(join.interface)
        for join in joins
    )
    citation_plan = FusionCitationPlan(
        roots=roots,
        primary_join_indices=tuple(range(len(primary))),
        interparent_join_indices=tuple(
            range(len(primary), len(primary) + len(interparent))
        ),
        cycle_closing_join_indices=tuple(
            range(len(primary) + len(interparent), len(joins))
        ),
        interparent_occurrences=topology.interparent_occurrences,
        render_order=tuple(
            occurrence
            for root in roots
            for occurrence in _preorder_children(root)
        ),
    )
    ast = FusionNameAst(
        plan_kind=_plan_kind(len(matches), groups, topology),
        parent_occurrences=topology.roots,
        component_occurrences=tuple(sorted(matches, key=lambda match: match.occurrence_id)),
        joins=joins,
        citation_tree=roots[0] if len(roots) == 1 else None,
        multiplicative_groups=groups,
        descriptors=descriptors,
        citation_plan=citation_plan,
    )
    rendered = render_fusion_name(ast, registry)
    exact_location_key = replace(
        topology.location_key,
        multiplicative_grouping_score=tuple(
            -len(group.occurrence_ids)
            for group in sorted(groups, key=lambda group: (-len(group.occurrence_ids), group.occurrence_ids))
        ),
    )
    score = (
        component_spec_seniority_key(specs[topology.roots[0]]).as_tuple(),
        exact_location_key,
        len(matches),
        max(topology.order_by_occurrence.values(), default=0),
        tuple(_join_preference_key(joins_by_child[child], side_rank[child]) for child in citation_children),
        tuple(component_spec_seniority_key(specs[child]).as_tuple() for child in citation_children),
        rendered,
    )
    return _Candidate(ast, score, rendered)


def _closing_join_direction(
    pair: frozenset[int],
    roots: tuple[int, ...],
    orders: Mapping[int, int],
    specs: Mapping[int, FusionComponentSpec],
) -> tuple[int, int]:
    """Orient a non-tree interface without consulting molecular atom ids."""

    root_set = set(roots)
    left, right = sorted(
        pair,
        key=lambda occurrence: (
            component_spec_seniority_key(specs[occurrence]).as_tuple(),
            specs[occurrence].key,
            occurrence,
        ),
    )
    if (left in root_set) != (right in root_set):
        return (right, left) if left in root_set else (left, right)
    ranked = sorted(
        (left, right),
        key=lambda occurrence: (
            orders[occurrence],
            component_spec_seniority_key(specs[occurrence]).as_tuple(),
            specs[occurrence].key,
            occurrence,
        ),
    )
    return ranked[-1], ranked[0]


def classify_ordered_fusion_interface(
    attached: FusionComponentMatch,
    host: FusionComponentMatch,
    attached_spec: FusionComponentSpec,
    host_spec: FusionComponentSpec,
    interface: FusionInterface[int],
    mol: Molecule,
) -> tuple[OrderedFusionInterface, int] | None:
    """Project one exact graph overlap onto ordered component interfaces.

    Host side order determines direction.  The attached path is then read by
    mapping those same input atoms back into the attached component.  Ordinary
    ortho and contiguous multi-side ortho-peri joins therefore use one proof
    path and one orientation rule.
    """

    if not interface.shared_edges:
        return None
    host_map = host.input_atom_by_locant
    attached_inverse = {atom: locant for locant, atom in attached.local_to_input_atom}
    sides = component_sides(host_spec)
    side_edges = tuple(
        normalize_edge(host_map[side.start_locant], host_map[side.end_locant]) for side in sides
    )
    selected_indices = frozenset(
        index for index, edge in enumerate(side_edges) if edge in interface.shared_edges
    )
    if len(selected_indices) != len(interface.shared_edges) or len(selected_indices) == len(sides):
        return None
    starts = tuple(index for index in selected_indices if (index - 1) % len(sides) not in selected_indices)
    if len(starts) != 1:
        return None
    start = starts[0]
    ordered_indices = tuple((start + offset) % len(sides) for offset in range(len(selected_indices)))
    if frozenset(ordered_indices) != selected_indices:
        return None

    selected_sides = tuple(sides[index] for index in ordered_indices)
    host_text = (selected_sides[0].start_locant,) + tuple(side.end_locant for side in selected_sides)
    ordered_atoms = tuple(host_map[locant] for locant in host_text)
    ordered_edges = tuple(
        normalize_edge(left, right) for left, right in zip(ordered_atoms, ordered_atoms[1:])
    )
    if frozenset(ordered_edges) != interface.shared_edges:
        return None
    if frozenset(ordered_atoms) != interface.shared_atom_ids:
        return None
    try:
        attached_text = tuple(attached_inverse[atom] for atom in ordered_atoms)
    except KeyError:
        return None
    attached_bonds = {frozenset(bond.locants) for bond in attached_spec.bonds}
    if any(
        frozenset(pair) not in attached_bonds
        for pair in zip(attached_text, attached_text[1:])
    ):
        return None
    molecular_bonds = tuple(mol.get_bond(*edge) for edge in ordered_edges)
    if any(bond is None for bond in molecular_bonds):
        return None

    kind = FusionJoinKind.ORTHO if len(ordered_edges) == 1 else FusionJoinKind.ORTHO_PERI
    attached_path = tuple(ComponentLocant(attached.occurrence_id, text) for text in attached_text)
    evidence = OrderedFusionInterface(
        kind=kind,
        attached_occurrence=attached.occurrence_id,
        host_occurrence=host.occurrence_id,
        attached_path=attached_path,
        host_path=tuple(ComponentLocant(host.occurrence_id, text) for text in host_text),
        cited_attached_locants=attached_path,
        host_sides=tuple(FusionSide(host.occurrence_id, side.letter) for side in selected_sides),
        ordered_input_atoms=ordered_atoms,
        ordered_input_edges=ordered_edges,
        ordered_input_bonds=tuple(bond.idx for bond in molecular_bonds if bond is not None),
    )
    return evidence, start


def _classified_join(
    attached: FusionComponentMatch,
    host: FusionComponentMatch,
    attached_spec: FusionComponentSpec,
    host_spec: FusionComponentSpec,
    interface: FusionInterface[int],
    order: int,
    mol: Molecule,
) -> tuple[FusionJoin, int] | None:
    classified = classify_ordered_fusion_interface(
        attached,
        host,
        attached_spec,
        host_spec,
        interface,
        mol,
    )
    if classified is None:
        return None
    evidence, side_rank = classified
    return FusionJoin(order=order, interface=evidence), side_rank


def _interface_support_key(interface: FusionInterface[int]) -> str:
    return "ortho" if len(interface.shared_edges) == 1 else "ortho_peri"


def _scope_for_match(
    mol: Molecule,
    match: FusionComponentMatch,
    spec: FusionComponentSpec,
) -> ComponentScope[int]:
    atom_by_locant = match.input_atom_by_locant
    edges: list[tuple[int, int]] = []
    for bond in spec.bonds:
        edge = normalize_edge(atom_by_locant[bond.locants[0]], atom_by_locant[bond.locants[1]])
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
    for child, parent in parent_by_child.items():
        children[parent].append(child)
    return {
        parent: tuple(
            sorted(
                values,
                key=lambda child: (
                    component_spec_seniority_key(specs[child]).as_tuple(),
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
                pattern = multiplicative_attachment_key(specs[child], join)
                by_pattern[pattern].append(child)
        for _pattern, members in sorted(by_pattern.items()):
            if len(members) < 2:
                continue
            ordered = tuple(sorted(members, key=lambda child: multiplicative_member_order_key(joins[child])))
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
        interface=replace(
            join.interface,
            attached_path=tuple(
                replace(locant, prime_depth=attached_depth)
                for locant in join.interface.attached_path
            ),
            cited_attached_locants=tuple(
                replace(locant, prime_depth=attached_depth)
                for locant in join.interface.cited_attached_locants
            ),
            host_path=tuple(
                replace(locant, prime_depth=host_depth) for locant in join.interface.host_path
            ),
            host_sides=tuple(
                replace(side, prime_depth=host_depth) for side in join.interface.host_sides
            ),
            host_locants=tuple(
                replace(locant, prime_depth=host_depth)
                for locant in join.interface.host_locants
            ),
        ),
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


def _combine_rendered_descriptors(descriptors: tuple[str, ...]) -> str:
    interiors = tuple(descriptor[1:-1] for descriptor in descriptors)
    return f"[{':'.join(interiors)}]"


def _render_descriptor(descriptor: FusionDescriptor, *, omit_attached_locants: bool) -> str:
    if not omit_attached_locants:
        return descriptor.render()
    if descriptor.kind is not FusionJoinKind.ORTHO or len(descriptor.parent_sides) != 1:
        return descriptor.render()
    return f"[{descriptor.parent_sides[0]}]"


def _omit_attached_locants(
    registry: _Registry | Mapping[str, FusionComponentSpec],
    key: str,
) -> bool:
    if isinstance(registry, Mapping):
        return False
    component = registry.get(key)
    return component.omit_attached_locants


def _join_preference_key(join: FusionJoin, side_rank: int) -> tuple:
    return (join.order, side_rank, _attached_locant_key(join))


def _attached_locant_key(join: FusionJoin) -> tuple:
    return tuple(retained_locant_sort_key(locant.text) for locant in join.attached_locants)


def _local_map_key(mapping: tuple[tuple[str, int], ...]) -> tuple:
    return tuple((retained_locant_sort_key(locant), atom) for locant, atom in mapping)


def _occurrence_option_key(option: _OccurrenceOption) -> tuple:
    return (
        min(option.face_ids),
        -len(option.face_ids),
        tuple(sorted(option.face_ids)),
        option.spec_key,
        tuple(sorted(option.atom_ids)),
    )


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
    if count < 2:
        return None
    try:
        return multipliers.basic(count)
    except ValueError:
        return None


def _plan_kind(
    component_count: int,
    groups: tuple[FusionMultiplicityGroup, ...],
    topology: _CitationTopology,
) -> str:
    if len(topology.roots) > 1:
        return "multiparent"
    if topology.cycle_closing_pairs:
        return "cyclic_component_cover"
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
        value = registry.get(key)
    if value is None:
        raise FusionDescriptorError(f"unknown fusion component {key!r}")
    spec = getattr(value, "spec", value)
    if not isinstance(spec, FusionComponentSpec):
        raise FusionDescriptorError(f"registry value for {key!r} is not a fusion component spec")
    return spec


def _spec_for_match(
    registry: _Registry | Mapping[str, FusionComponentSpec],
    match: FusionComponentMatch,
) -> FusionComponentSpec:
    resolver = getattr(registry, "spec_for_match", None)
    if resolver is not None:
        return resolver(match)
    return _spec(registry, match.spec_key)
