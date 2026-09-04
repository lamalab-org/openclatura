"""Typed composition of fused parents with bridge and spiro nomenclature.

Ordinary fusion planning deliberately rejects bridged and spiro-only graphs.
This module keeps that boundary intact: it first proves an independently
named fused parent, then records the additional graph operation around it.
The wrapper never infers a junction locant from rendered name text.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from itertools import combinations

from ..assembly_parts import NameTokenBinding
from ..locants import retained_locant_sort_key
from ..molecule import Molecule, bond_ids_within, edges_within_atoms
from ..polycycle_topology import connected_components, ring_system_topology
from ..retained_fused_templates import match_retained_fused_templates
from ..retained_name_policy import retained_parent_output_name
from ..rules import stems
from .config import fusion_nomenclature_config
from .model import FusionConfirmed, FusionMode, FusionParentPlan

_WRAPPER_SEARCH_STATES = fusion_nomenclature_config().search.component_selection_states


class WrapperParentKind(StrEnum):
    """Source of the independently proven component parent."""

    RETAINED = "retained"
    SYSTEMATIC_FUSION = "systematic_fusion"


class NondetachableBridgeKind(StrEnum):
    """Supported neutral, closed-shell nondetachable bridge classes."""

    CARBO = "carbo"
    EPOXY = "epoxy"
    EPITHIO = "epithio"
    EPIMINO = "epimino"


@dataclass(frozen=True, slots=True)
class WrapperParentPlan:
    """A locant-complete parent that may receive a graph wrapper."""

    kind: WrapperParentKind
    name: str
    atom_ids: frozenset[int]
    locant_maps: tuple[tuple[tuple[int, str], ...], ...]
    fusion_plan: FusionParentPlan | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.atom_ids or not self.locant_maps:
            raise ValueError("wrapper parent requires a name, atoms, and locant maps")
        for entries in self.locant_maps:
            mapping = dict(entries)
            if set(mapping) != set(self.atom_ids) or len(set(mapping.values())) != len(mapping):
                raise ValueError("wrapper parent locant maps must be complete and bijective")
        if self.kind is WrapperParentKind.SYSTEMATIC_FUSION and self.fusion_plan is None:
            raise ValueError("systematic-fusion wrapper parent requires its fusion proof")


@dataclass(frozen=True, slots=True)
class NondetachableBridgeOperation:
    """One path outside a fused parent whose ends attach to that parent."""

    kind: NondetachableBridgeKind
    prefix: str
    atom_ids: tuple[int, ...]
    endpoint_atom_ids: tuple[int, int]
    endpoint_locants: tuple[str, str]
    bond_ids: frozenset[int]

    @property
    def rendered(self) -> str:
        return f"{','.join(self.endpoint_locants)}-{self.prefix}"


@dataclass(frozen=True, slots=True)
class BridgedFusionWrapperPlan:
    """A fused parent plus independently graph-proven bridge operations."""

    parent: WrapperParentPlan
    bridges: tuple[NondetachableBridgeOperation, ...]
    rendered_name: str
    rendered_parts: tuple[NameTokenBinding, ...]
    audit_ok: bool
    audit_checks: tuple[str, ...] = ()
    search_states: int = 0

    def __post_init__(self) -> None:
        if not self.bridges:
            raise ValueError("bridged fusion wrapper requires at least one bridge")
        if not self.audit_ok:
            raise ValueError("bridged fusion wrapper must pass its graph partition audit")
        if not self.audit_checks:
            raise ValueError("bridged fusion wrapper must record its successful audits")
        if self.search_states < 1:
            raise ValueError("bridged fusion wrapper must record bounded search effort")
        if "".join(part.text for part in self.rendered_parts) != self.rendered_name:
            raise ValueError("bridged fusion wrapper parts must reproduce the rendered name")


@dataclass(frozen=True, slots=True)
class FusionSpiroSidePlan:
    """An audited fused component and its graph-derived spiro locant."""

    parent: WrapperParentPlan
    junction_atom_id: int
    junction_locant: str

    def __post_init__(self) -> None:
        if self.parent.kind is not WrapperParentKind.SYSTEMATIC_FUSION:
            raise ValueError("FusionSpiroSidePlan requires a systematic fusion parent")
        if self.junction_atom_id not in self.parent.atom_ids:
            raise ValueError("spiro junction lies outside the fused side parent")
        if self.junction_locant != dict(self.parent.locant_maps[0]).get(self.junction_atom_id):
            raise ValueError("spiro junction locant must come from the fusion proof map")

    def to_spiro_assembly(self, *, parent_locant: str = ""):
        """Adapt to the established spiro renderer without parsing parent text."""

        from ..spiro_assembly import SpiroAssembly

        return SpiroAssembly(
            parent_locant=parent_locant,
            side_locant=self.junction_locant,
            side_parent_name=self.parent.name,
        )


def plan_fusion_spiro_side(
    mol: Molecule,
    side_atom_ids: set[int] | frozenset[int],
    junction_atom_id: int,
    *,
    mode: FusionMode | str,
) -> FusionSpiroSidePlan | None:
    """Plan a fused spiro side only from a confirmed fusion-parent proof."""

    atoms = frozenset(side_atom_ids)
    if junction_atom_id not in atoms or any(mol.atoms[atom].charge for atom in atoms):
        return None
    parent = _systematic_fusion_parent(mol, atoms, FusionMode(mode))
    if parent is None:
        return None
    junction_locant = dict(parent.locant_maps[0]).get(junction_atom_id)
    if junction_locant is None:
        return None
    return FusionSpiroSidePlan(parent, junction_atom_id, junction_locant)


def plan_bridged_fusion_wrapper(
    mol: Molecule,
    atom_ids: set[int] | frozenset[int],
    *,
    mode: FusionMode | str,
    maximum_bridge_atoms: int = 4,
) -> BridgedFusionWrapperPlan | None:
    """Find a retained or systematic fused parent beneath simple bridges.

    Candidate removal is restricted to neutral degree-two atoms and bounded by
    ``maximum_bridge_atoms``.  A candidate is accepted only when every removed
    component is a path with exactly two parent attachments and the remaining
    graph has a locant-complete retained or audited systematic-fusion parent.
    This search runs only for already bridged polycycles.
    """

    atoms = frozenset(atom_ids)
    policy = FusionMode(mode)
    if (
        policy in {FusionMode.DISABLED, FusionMode.LEGACY}
        or maximum_bridge_atoms < 1
        or any(mol.atoms[atom].charge for atom in atoms)
    ):
        return None
    topology = ring_system_topology(mol, atoms)
    if topology.cycle_rank < 3 or not topology.bridgeheads:
        return None

    # RDKit may mark a bridge heteroatom aromatic as part of the input ring
    # model.  Degree and the audited parent remainder, rather than that input
    # flag, determine whether an atom can be a bridge-path interior.
    removable = tuple(
        atom
        for atom in sorted(atoms)
        if len([neighbor for neighbor in mol.get_neighbors(atom) if neighbor in atoms]) == 2
        and (mol.atoms[atom].symbol != "C" or not mol.atoms[atom].is_aromatic)
    )
    visited_states = 0
    for parent_kind in (WrapperParentKind.RETAINED, WrapperParentKind.SYSTEMATIC_FUSION):
        candidates: list[tuple[tuple, BridgedFusionWrapperPlan]] = []
        for count in range(1, min(maximum_bridge_atoms, len(removable)) + 1):
            for removed_tuple in combinations(removable, count):
                visited_states += 1
                if visited_states > _WRAPPER_SEARCH_STATES:
                    return None
                removed = frozenset(removed_tuple)
                parent_atoms = atoms - removed
                parent = (
                    _retained_wrapper_parent(mol, parent_atoms)
                    if parent_kind is WrapperParentKind.RETAINED
                    else _systematic_fusion_parent(mol, parent_atoms, policy)
                )
                if parent is None:
                    continue
                bridge_paths = _bridge_path_components(mol, atoms, parent_atoms, removed)
                if bridge_paths is None:
                    continue
                for map_index, entries in enumerate(parent.locant_maps):
                    locants = dict(entries)
                    operations = _bridge_operations(mol, bridge_paths, parent_atoms, locants)
                    if operations is None:
                        continue
                    operations = tuple(
                        sorted(operations, key=lambda operation: (operation.prefix, operation.endpoint_locants))
                    )
                    audit_checks = _audit_bridge_plan(
                        mol,
                        atoms,
                        parent_atoms,
                        dict(entries),
                        operations,
                    )
                    if audit_checks is None:
                        continue
                    selected_parent = replace(parent, locant_maps=(entries,))
                    rendered_parts = _render_bridge_parts(mol, selected_parent, operations)
                    rendered = "".join(part.text for part in rendered_parts)
                    plan = BridgedFusionWrapperPlan(
                        selected_parent,
                        operations,
                        rendered,
                        rendered_parts,
                        audit_ok=True,
                        audit_checks=audit_checks,
                        search_states=visited_states,
                    )
                    rank = (
                        -len(parent_atoms),
                        map_index,
                        tuple(operation.prefix for operation in operations),
                        tuple(operation.endpoint_locants for operation in operations),
                        rendered,
                    )
                    candidates.append((rank, plan))
            if candidates:
                break
        if candidates:
            return min(candidates, key=lambda item: item[0])[1]
    return None


def _retained_wrapper_parent(mol: Molecule, atoms: frozenset[int]) -> WrapperParentPlan | None:
    matches = match_retained_fused_templates(mol, set(atoms))
    if not matches:
        matches = match_retained_fused_templates(
            mol,
            set(atoms),
            allow_nonaromatic=True,
            allow_relocated_indicated_h=True,
        )
    if not matches:
        return None
    first = matches[0]
    template_name = first.template.name
    same_parent = [match for match in matches if match.template.name == template_name]
    maps = tuple(
        tuple(sorted(((atom, str(locant)) for atom, locant in match.atom_to_locant.items()))) for match in same_parent
    )
    return WrapperParentPlan(
        kind=WrapperParentKind.RETAINED,
        name=min(
            (
                retained_parent_output_name(template_name, "composite_parent"),
                template_name,
                *first.template.aliases,
            ),
            key=lambda candidate: (len(candidate), candidate),
        ),
        atom_ids=atoms,
        locant_maps=tuple(dict.fromkeys(maps)),
    )


def _systematic_fusion_parent(
    mol: Molecule,
    atoms: frozenset[int],
    mode: FusionMode,
) -> WrapperParentPlan | None:
    if mode in {FusionMode.DISABLED, FusionMode.LEGACY}:
        return None
    from .planner import plan_fusion_parent

    result = plan_fusion_parent(mol, atoms, mode=mode)
    if not isinstance(result, FusionConfirmed):
        return None
    maps = tuple(
        tuple(sorted((atom, str(locant)) for atom, locant in entries))
        for entries in result.plan.numbering.input_locant_maps
    )
    return WrapperParentPlan(
        kind=WrapperParentKind.SYSTEMATIC_FUSION,
        name=result.plan.rendered_base_name,
        atom_ids=atoms,
        locant_maps=maps,
        fusion_plan=result.plan,
    )


def _bridge_path_components(
    mol: Molecule,
    all_atoms: frozenset[int],
    parent_atoms: frozenset[int],
    removed: frozenset[int],
) -> tuple[tuple[int, ...], ...] | None:
    removed_edges = frozenset(
        edge for edge in edges_within_atoms(mol, set(all_atoms)) if edge[0] in removed and edge[1] in removed
    )
    components = connected_components(set(removed), removed_edges)
    paths = []
    for component in components:
        internal_degree = {
            atom: sum(neighbor in component for neighbor in mol.get_neighbors(atom)) for atom in component
        }
        if any(degree > 2 for degree in internal_degree.values()):
            return None
        attachments = {
            neighbor for atom in component for neighbor in mol.get_neighbors(atom) if neighbor in parent_atoms
        }
        if len(attachments) != 2:
            return None
        starts = sorted(atom for atom in component if internal_degree[atom] <= 1)
        if not starts:
            return None
        path = _walk_path(mol, component, starts[0])
        if set(path) != set(component):
            return None
        paths.append(path)
    return tuple(sorted(paths))


def _walk_path(mol: Molecule, atoms: set[int], start: int) -> tuple[int, ...]:
    path = [start]
    previous = None
    while len(path) < len(atoms):
        following = sorted(
            neighbor for neighbor in mol.get_neighbors(path[-1]) if neighbor in atoms and neighbor != previous
        )
        if not following:
            break
        previous, current = path[-1], following[0]
        path.append(current)
    return tuple(path)


def _bridge_operations(
    mol: Molecule,
    paths: tuple[tuple[int, ...], ...],
    parent_atoms: frozenset[int],
    locants: dict[int, str],
) -> tuple[NondetachableBridgeOperation, ...] | None:
    operations = []
    for path in paths:
        endpoints = tuple(
            sorted(
                {neighbor for atom in path for neighbor in mol.get_neighbors(atom) if neighbor in parent_atoms},
                key=lambda atom: retained_locant_sort_key(locants[atom]),
            )
        )
        if len(endpoints) != 2:
            return None
        # A path whose parent endpoints already share an edge is an annelated
        # ring component, not a nondetachable bridge over the parent.
        if mol.get_bond(*endpoints) is not None:
            return None
        bridge_class = _bridge_class(mol, path)
        if bridge_class is None:
            return None
        kind, prefix = bridge_class
        endpoint_locants = tuple(locants[atom] for atom in endpoints)
        operation_atoms = set(path) | set(endpoints)
        operations.append(
            NondetachableBridgeOperation(
                kind=kind,
                prefix=prefix,
                atom_ids=path,
                endpoint_atom_ids=endpoints,
                endpoint_locants=endpoint_locants,
                bond_ids=frozenset(bond_ids_within(mol, operation_atoms) - bond_ids_within(mol, set(endpoints))),
            )
        )
    return tuple(operations)


def _bridge_class(mol: Molecule, path: tuple[int, ...]) -> tuple[NondetachableBridgeKind, str] | None:
    symbols = tuple(mol.atoms[atom].symbol for atom in path)
    if all(symbol == "C" for symbol in symbols):
        return NondetachableBridgeKind.CARBO, f"{stems.get(len(symbols)).stem}ano"
    if symbols == ("O",):
        return NondetachableBridgeKind.EPOXY, "epoxy"
    if symbols == ("S",):
        return NondetachableBridgeKind.EPITHIO, "epithio"
    if symbols == ("N",):
        return NondetachableBridgeKind.EPIMINO, "epimino"
    return None


def _render_bridge_parts(
    mol: Molecule,
    parent: WrapperParentPlan,
    operations: tuple[NondetachableBridgeOperation, ...],
) -> tuple[NameTokenBinding, ...]:
    parts: list[NameTokenBinding] = []
    for index, operation in enumerate(operations):
        endpoint_atoms = set(operation.endpoint_atom_ids)
        parts.append(
            NameTokenBinding(
                text=f"{','.join(operation.endpoint_locants)}-",
                token_kind="locant",
                source="fusion_wrapper_renderer",
                grammar_role="nondetachable_bridge_locants",
                binding_key=f"fusion:bridge:{index}:locants",
                atom_ids=endpoint_atoms,
                locants=operation.endpoint_locants,
            )
        )
        parts.append(
            NameTokenBinding(
                text=operation.prefix + ("-" if index + 1 < len(operations) else ""),
                token_kind="prefix",
                source="fusion_wrapper_renderer",
                grammar_role="nondetachable_bridge",
                binding_key=f"fusion:bridge:{index}:path",
                atom_ids=set(operation.atom_ids),
                bond_ids=set(operation.bond_ids),
            )
        )
    parts.append(
        NameTokenBinding(
            text=parent.name,
            token_kind="parent",
            source="fusion_wrapper_renderer",
            grammar_role="wrapped_fusion_parent",
            binding_key="fusion:bridge:parent",
            atom_ids=set(parent.atom_ids),
            bond_ids=bond_ids_within(mol, set(parent.atom_ids)),
        )
    )
    return tuple(parts)


def _audit_bridge_plan(
    mol: Molecule,
    all_atoms: frozenset[int],
    parent_atoms: frozenset[int],
    parent_locants: dict[int, str],
    operations: tuple[NondetachableBridgeOperation, ...],
) -> tuple[str, ...] | None:
    if set(parent_locants) != set(parent_atoms) or len(set(parent_locants.values())) != len(parent_atoms):
        return None
    bridge_atoms = {atom for operation in operations for atom in operation.atom_ids}
    if (
        sum(len(operation.atom_ids) for operation in operations) != len(bridge_atoms)
        or parent_atoms & bridge_atoms
        or parent_atoms | bridge_atoms != all_atoms
    ):
        return None
    expected_edges = set(edges_within_atoms(mol, set(parent_atoms)))
    for operation in operations:
        path = operation.atom_ids
        if not path or any(mol.get_bond(left, right) is None for left, right in zip(path, path[1:])):
            return None
        attachments = {neighbor for atom in path for neighbor in mol.get_neighbors(atom) if neighbor in parent_atoms}
        if attachments != set(operation.endpoint_atom_ids):
            return None
        if operation.endpoint_locants != tuple(parent_locants[atom] for atom in operation.endpoint_atom_ids):
            return None
        scope = set(operation.atom_ids) | set(operation.endpoint_atom_ids)
        operation_edges = edges_within_atoms(mol, scope) - edges_within_atoms(mol, set(operation.endpoint_atom_ids))
        expected_edges.update(operation_edges)
        if operation.bond_ids != bond_ids_within(mol, scope) - bond_ids_within(mol, set(operation.endpoint_atom_ids)):
            return None
    if expected_edges != set(edges_within_atoms(mol, set(all_atoms))):
        return None
    return (
        "complete_bijective_parent_locants",
        "disjoint_parent_and_bridge_atoms",
        "simple_bridge_paths",
        "exact_bridge_endpoints_and_locants",
        "exact_bridge_bond_ownership",
        "complete_wrapper_graph_reconstruction",
    )


__all__ = [
    "BridgedFusionWrapperPlan",
    "FusionSpiroSidePlan",
    "NondetachableBridgeKind",
    "NondetachableBridgeOperation",
    "WrapperParentKind",
    "WrapperParentPlan",
    "plan_bridged_fusion_wrapper",
    "plan_fusion_spiro_side",
]
