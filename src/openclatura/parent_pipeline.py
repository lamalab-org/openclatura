"""Shared parent planning steps for component and subgraph naming."""

from .assembly_parts import AssemblyParts, NameAtomBinding, ParentChargeItem, RetainedParentMetadata
from .fusion.context import current_fusion_mode
from .fusion.model import FusionMode, PinDecision, PinStatus
from .heteroatom_subgraphs import upstream_bond_order
from .locant_sources import LocantMapSource
from .locants import canonical_locant_pair
from .molecule import DecisionTrace, Molecule, TracePhase, bond_ids_within
from .name_bindings import ensure_name_atom_binding_tokens
from .namer_config import RETAINED_RING_ELEMENTS
from .naming_context import NamingIntent, ParentAssemblyPlan
from .numbering import choose_parent_numbering
from .parent_selection import ParentSelection
from .ring_parent import ParentHydrideKind, ParentHydridePlan, RingParent
from .ring_renderer import is_von_baeyer_descriptor
from .rules import retained
from .small_ring_stereo import scoped_small_ring_stereo_features
from .subgraph_tools import subgraph_locant_getter
from .trace_helpers import trace_decision


def plan_fusion_parent(mol: Molecule, parent_atoms: frozenset[int], *, mode: FusionMode):
    """Load the systematic fusion planner only for an enabled request."""

    from .fusion.planner import plan_fusion_parent as _plan_fusion_parent

    return _plan_fusion_parent(mol, parent_atoms, mode=mode)


def resolve_systematic_fusion_parent(
    mol: Molecule,
    selection: ParentSelection,
    *,
    retained_name: str | None,
    decision_trace: DecisionTrace | None = None,
) -> RingParent | None:
    """Resolve an audited fusion parent through the established ring handoff."""

    mode = current_fusion_mode()
    if mode not in {FusionMode.AUDITED_PIN, FusionMode.GENERAL}:
        trace_decision(
            decision_trace,
            TracePhase.PARENT_SELECTION,
            "systematic fusion disabled",
            "The request keeps the legacy ring-nomenclature path.",
            atoms=selection.atom_set,
            data={"fusion_mode": mode.value, "reason": "request_policy"},
        )
        return None
    if retained_name is not None:
        trace_decision(
            decision_trace,
            TracePhase.PARENT_SELECTION,
            "skipped systematic fusion",
            "A retained or independently systematic parent has precedence over fusion nomenclature.",
            atoms=selection.atom_set,
            data={"fusion_mode": mode.value, "reason": "retained_parent_precedence"},
        )
        return None

    result = plan_fusion_parent(mol, selection.atom_set, mode=mode)
    from .fusion.model import FusionConfirmed

    if not isinstance(result, FusionConfirmed):
        trace_decision(
            decision_trace,
            TracePhase.PARENT_SELECTION,
            "systematic fusion fallback",
            "The bounded fusion planner did not produce an audit-confirmed parent, so legacy ring nomenclature remains active.",
            atoms=selection.atom_set,
            data={
                "fusion_mode": mode.value,
                "result": type(result).__name__,
                "reason": result.reason,
                "details": list(getattr(result, "details", ()) or getattr(result, "candidate_summary", ())),
            },
        )
        return None

    plan = result.plan
    if mode is FusionMode.AUDITED_PIN:
        pin_decision = PinDecision(
            status=PinStatus.CONFIRMED,
            checks=(
                "no_preferred_retained_complete_parent",
                "no_preferred_independently_systematic_complete_parent",
                "fusion_rules_satisfied",
                *plan.audit.checks,
            ),
        )
    else:
        pin_decision = PinDecision(
            status=PinStatus.VALID_GENERAL_NAME,
            checks=("fusion_rules_satisfied", *plan.audit.checks),
        )
    from .fusion.planner import PLANNER_TIER
    from .fusion.trace import fusion_proof_counts, trace_confirmed_fusion_plan

    trace_confirmed_fusion_plan(decision_trace, mol, plan, selection.atom_set)
    trace_decision(
        decision_trace,
        TracePhase.PARENT_SELECTION,
        "selected audited systematic fusion parent",
        "The graph-backed fusion plan passed component, descriptor, numbering, bond-model, and reconstruction audits.",
        atoms=selection.atom_set,
        data={
            "fusion_mode": mode.value,
            "parent_nomenclature": "systematic_fusion",
            "base_name": plan.rendered_base_name,
            "pin_status": str(pin_decision.status),
            "pin_checks": list(pin_decision.checks),
            "fusion_support_tier": PLANNER_TIER,
            "proof_source": "fusion_reconstruction",
            "components": [
                {
                    "occurrence_id": match.occurrence_id,
                    "spec_key": match.spec_key,
                    "faces": sorted(match.covered_face_ids),
                }
                for match in plan.ast.component_occurrences
            ],
            "joins": [
                {
                    "attached": join.attached_occurrence,
                    "host": join.host_occurrence,
                    "attached_locants": [str(locant) for locant in join.attached_locants],
                    "host_sides": [str(side) for side in join.host_sides],
                }
                for join in plan.ast.joins
            ],
            "locant_map_count": len(plan.numbering.input_locant_maps),
            "atom_to_locant": {atom: str(locant) for atom, locant in plan.numbering.input_locant_maps[0]},
            "orientation_score": plan.numbering.orientation_score,
            "rule_trace": [
                {
                    "rule": item.rule,
                    "criterion": item.criterion,
                    "outcome": item.outcome,
                    "reason": item.reason,
                }
                for item in plan.rule_trace
            ],
            "audit_checks": list(plan.audit.checks),
            "proof_counts": fusion_proof_counts(plan),
        },
    )
    return RingParent.from_fusion_plan(plan, pin_decision=pin_decision)


def resolve_bridged_fusion_parent(
    mol: Molecule,
    selection: ParentSelection,
    *,
    decision_trace: DecisionTrace | None = None,
) -> RingParent | None:
    """Resolve a proven nondetachable bridge around a fused parent."""

    mode = current_fusion_mode()
    if mode not in {FusionMode.AUDITED_PIN, FusionMode.GENERAL}:
        return None
    from .fusion.wrappers import plan_bridged_fusion_wrapper

    plan = plan_bridged_fusion_wrapper(mol, selection.atom_set, mode=mode)
    if plan is None:
        return None
    trace_decision(
        decision_trace,
        TracePhase.PARENT_SELECTION,
        "selected audited bridged fusion parent",
        "A bounded bridge decomposition exposed a locant-complete retained or systematic fused parent.",
        atoms=selection.atom_set,
        bonds=bond_ids_within(mol, selection.atom_set),
        data={
            "base_name": plan.rendered_name,
            "parent_kind": plan.parent.kind.value,
            "parent_atoms": sorted(plan.parent.atom_ids),
            "bridges": [
                {
                    "kind": bridge.kind.value,
                    "atoms": list(bridge.atom_ids),
                    "endpoint_atoms": list(bridge.endpoint_atom_ids),
                    "endpoint_locants": list(bridge.endpoint_locants),
                }
                for bridge in plan.bridges
            ],
            "proof_source": "fusion_wrapper_reconstruction",
            "audit_checks": list(plan.audit_checks),
            "search_states": plan.search_states,
        },
    )
    status = PinStatus.CONFIRMED if mode is FusionMode.AUDITED_PIN else PinStatus.VALID_GENERAL_NAME
    checks = (
        "no_preferred_retained_complete_parent",
        "no_preferred_independently_systematic_complete_parent",
        "fusion_rules_satisfied",
        *plan.audit_checks,
    )
    return RingParent.from_fusion_wrapper(plan).with_pin_decision(PinDecision(status=status, checks=checks))


def resolve_third_component_fusion_parent(
    mol: Molecule,
    selection: ParentSelection,
    *,
    decision_trace: DecisionTrace | None = None,
) -> RingParent | None:
    """Apply P-25.5.1 after an ordinary cyclic fusion citation is prohibited."""

    mode = current_fusion_mode()
    if mode not in {FusionMode.AUDITED_PIN, FusionMode.GENERAL}:
        return None
    from .fusion.third_component import (
        THIRD_COMPONENT_TIER,
        plan_third_component_fusion_parent,
    )

    plan = plan_third_component_fusion_parent(mol, selection.atom_set, mode=mode)
    if plan is None:
        return None
    status = PinStatus.CONFIRMED if mode is FusionMode.AUDITED_PIN else PinStatus.VALID_GENERAL_NAME
    checks = (
        "no_preferred_retained_complete_parent",
        "no_preferred_independently_systematic_complete_parent",
        "fusion_rules_satisfied",
        *plan.audit_checks,
    )
    parent = plan.parent.with_pin_decision(PinDecision(status=status, checks=checks))
    trace_decision(
        decision_trace,
        TracePhase.PARENT_SELECTION,
        "selected skeletal-replacement fusion parent",
        "A cyclic component cover cannot be emitted as an ordinary pairwise fusion name; the corresponding audited carbon parent is used under P-25.5.1.",
        atoms=selection.atom_set,
        bonds=bond_ids_within(mol, selection.atom_set),
        data={
            "fusion_mode": mode.value,
            "parent_nomenclature": "skeletal_replacement_fusion",
            "base_name": parent.base_name,
            "pin_status": parent.pin_status,
            "fusion_support_tier": THIRD_COMPONENT_TIER,
            "proof_source": parent.proof_source,
            "cover_topology": plan.cover_topology,
            "ring_sizes": list(plan.ring_sizes),
            "replacement_atoms": list(plan.replacement_atom_ids),
            "prohibited_cycle_closing_joins": list(plan.prohibited_citation.citation_plan.cycle_closing_join_indices),
            "audit_checks": list(plan.audit_checks),
        },
    )
    return parent


def resolve_retained_parent(
    mol: Molecule, path: list[int], is_ring: bool, is_bicycle: bool, is_polycycle: bool
) -> tuple[str | None, list[dict[int, str]] | None]:
    """Return a retained parent name and locant maps when valid for a path."""

    temp_retained = retained.get_retained_ring(mol, path) if is_ring else None
    if not temp_retained:
        return None, None
    retained_name, locant_maps = temp_retained
    if any(mol.atoms[idx].symbol not in RETAINED_RING_ELEMENTS for idx in path):
        return None, None
    if locant_maps is None and (is_bicycle or is_polycycle):
        if all(mol.atoms[idx].symbol == "C" and mol.atoms[idx].charge == 0 for idx in path):
            return retained_name, None
        return None, None
    return retained_name, locant_maps


def resolve_parent_hydride_plan(
    mol: Molecule,
    selection: ParentSelection,
    *,
    retained_name: str | None,
    locant_maps: list[dict[int, str]] | None,
    retained_parent_metadata: RetainedParentMetadata | None = None,
    decision_trace: DecisionTrace | None = None,
    retained_proof_source: str = "retained_template",
) -> ParentHydridePlan | None:
    """Resolve one ring parent-hydride handoff without changing selection rules.

    Retained-parent eligibility remains with the existing retained resolvers,
    and fusion eligibility remains with the audited fusion planner.  This
    function only consolidates their outputs before numbering and assembly.
    """

    parent = selection.ring_parent
    if retained_name is not None:
        if parent is None:
            parent = RingParent.from_paths(
                kind="legacy_ring",
                atoms=selection.atom_set,
                descriptor=selection.polycycle_descriptor,
                paths=selection.paths,
            )
        return parent.with_retained_identity(
            name=retained_name,
            locant_maps=locant_maps,
            metadata=retained_parent_metadata,
            proof_source=retained_proof_source,
        )

    if selection.is_bicycle or selection.is_polycycle:
        fusion_parent = resolve_systematic_fusion_parent(
            mol,
            selection,
            retained_name=None,
            decision_trace=decision_trace,
        )
        if fusion_parent is not None:
            return fusion_parent
        third_component_parent = resolve_third_component_fusion_parent(
            mol,
            selection,
            decision_trace=decision_trace,
        )
        if third_component_parent is not None:
            return third_component_parent
        bridged_parent = resolve_bridged_fusion_parent(
            mol,
            selection,
            decision_trace=decision_trace,
        )
        if bridged_parent is not None:
            return bridged_parent
    return parent


def build_parent_assembly_plan(
    mol: Molecule,
    selection: ParentSelection,
    intent: NamingIntent,
    substituent_mapping: dict[int, list],
    locant_maps=None,
    retained_name: str | None = None,
    retained_parent_metadata: RetainedParentMetadata | None = None,
    *,
    parent_hydride: ParentHydridePlan | None = None,
) -> ParentAssemblyPlan:
    """Number a selected parent and create base assembly parts."""

    parent_hydride = parent_hydride or selection.ring_parent
    if (
        parent_hydride is not None
        and retained_name is not None
        and parent_hydride.hydride_kind is not ParentHydrideKind.RETAINED
    ):
        retained_parent_metadata = retained_parent_metadata or retained.parent_metadata(retained_name)
        parent_hydride = parent_hydride.with_retained_identity(
            name=retained_name,
            locant_maps=locant_maps,
            metadata=retained_parent_metadata,
        )
    if parent_hydride is not None:
        retained_name = parent_hydride.retained_name or retained_name
        retained_parent_metadata = parent_hydride.metadata or retained_parent_metadata
        if parent_hydride.numbering_uses_proof_locant_maps:
            proof_maps = parent_hydride.proof_locant_maps
        else:
            proof_maps = ()
        if proof_maps:
            locant_maps = list(proof_maps)
    if parent_hydride is not None and (
        parent_hydride.hydride_kind
        in {
            ParentHydrideKind.SYSTEMATIC_FUSION,
            ParentHydrideKind.SKELETAL_REPLACEMENT_FUSION,
            ParentHydrideKind.BRIDGED_FUSION,
        }
        or is_von_baeyer_descriptor(parent_hydride.descriptor)
    ):
        locant_map_source = LocantMapSource.PROOF
    else:
        locant_map_source = LocantMapSource.SUPPLIED if locant_maps else LocantMapSource.GENERATED
    if (
        locant_maps is None
        and parent_hydride is not None
        and parent_hydride.numbering_candidates
        and is_von_baeyer_descriptor(parent_hydride.descriptor)
    ):
        audited_maps = [numbering.locant_map for numbering in parent_hydride.numbering_candidates if numbering.audit_ok]
        locant_maps = audited_maps or None
        if locant_maps:
            locant_map_source = LocantMapSource.PROOF
    numbered_path, locant_map = choose_parent_numbering(
        mol,
        selection.paths,
        intent.principal_atoms,
        substituent_mapping,
        locant_maps,
        selection.is_ring,
        selection.is_bicycle,
        selection.is_spiro,
        selection.is_polycycle,
        retained_name,
        fixed_start=intent.fixed_start,
    )
    get_loc = subgraph_locant_getter(numbered_path, locant_map)
    parts = build_parent_parts(
        mol,
        numbered_path,
        get_loc,
        retained_name,
        selection,
        intent,
        retained_parent_metadata,
        parent_hydride=parent_hydride,
        locant_map_source=locant_map_source,
    )
    return ParentAssemblyPlan(
        numbered_path=numbered_path,
        locant_map=locant_map,
        locant_map_source=locant_map_source,
        get_loc=get_loc,
        parts=parts,
        parent_hydride=parent_hydride,
    )


def build_parent_parts(
    mol: Molecule,
    numbered_path: list[int],
    get_loc,
    retained_name: str | None,
    selection: ParentSelection,
    intent: NamingIntent,
    retained_parent_metadata: RetainedParentMetadata | None = None,
    *,
    ring_parent: RingParent | None = None,
    parent_hydride: ParentHydridePlan | None = None,
    locant_map_source: LocantMapSource = LocantMapSource.GENERATED,
) -> AssemblyParts:
    """Create shared parent assembly parts for a naming intent."""

    parent_hydride = parent_hydride or ring_parent
    if parent_hydride is not None:
        retained_name = parent_hydride.retained_name or retained_name
        retained_parent_metadata = parent_hydride.metadata or retained_parent_metadata
    if retained_parent_metadata is None and retained_name is not None:
        retained_parent_metadata = retained.parent_metadata(retained_name)
    assembly_overrides = {}
    if intent.is_substituent:
        if intent.root_atom is None:
            raise ValueError("Subgraph naming intent requires a root atom.")
        upstream_order = upstream_bond_order(mol, intent.root_atom, intent.upstream_atom)
        assembly_overrides.update(
            {
                "is_substituent": True,
                "is_double_attach": upstream_order == 2,
                "is_triple_attach": upstream_order == 3,
                "attachment_locant": get_loc(intent.root_atom),
            }
        )

    parts = AssemblyParts(
        parent_length=len(numbered_path),
        is_ring=selection.is_ring,
        is_bicycle=selection.is_bicycle,
        is_spiro=selection.is_spiro,
        is_polycycle=selection.is_polycycle,
        bicycle_xyz=selection.xyz if selection.is_bicycle else (0, 0, 0),
        spiro_xy=(selection.xyz[0], selection.xyz[1]) if selection.is_spiro else (0, 0),
        polycycle_descriptor=selection.polycycle_descriptor,
        retained_name=retained_name,
        retained_parent_metadata=retained_parent_metadata,
        parent_hydride=parent_hydride,
        locant_map_source=locant_map_source,
        omit_redundant_locants=intent.omit_redundant_locants,
        parent_atom_ids=set(numbered_path),
        parent_bond_ids=bond_ids_within(mol, set(numbered_path)),
        **assembly_overrides,
    )
    for atom_idx in numbered_path:
        locant = str(get_loc(atom_idx))
        parts.parent_atom_ids_by_locant[locant] = atom_idx
        parts.parent_atom_symbols_by_locant[locant] = mol.atoms[atom_idx].symbol
        parts.parent_atom_charges_by_locant[locant] = mol.atoms[atom_idx].charge
        parts.parent_atom_isotopes_by_locant[locant] = mol.atoms[atom_idx].isotope
        if mol.atoms[atom_idx].stereo:
            parts.stereo_features.append((get_loc(atom_idx), mol.atoms[atom_idx].stereo))
        if mol.atoms[atom_idx].charge:
            parts.parent_charges.append(
                ParentChargeItem(
                    locant=locant,
                    symbol=mol.atoms[atom_idx].symbol,
                    charge=mol.atoms[atom_idx].charge,
                    atom_id=atom_idx,
                )
            )
    _add_relative_ring_stereo(mol, parts, numbered_path, get_loc)
    parent_set = set(numbered_path)
    for atom_idx in numbered_path:
        locant = str(get_loc(atom_idx))
        for neighbor_idx in mol.get_neighbors(atom_idx):
            if neighbor_idx in parent_set and atom_idx < neighbor_idx:
                neighbor_locant = str(get_loc(neighbor_idx))
                bond = mol.get_bond(atom_idx, neighbor_idx)
                if bond is not None:
                    locant_pair = canonical_locant_pair(locant, neighbor_locant)
                    parts.parent_bond_orders_by_locants[locant_pair] = bond.order
                    parts.parent_bond_ids_by_locants[locant_pair] = bond.idx
    return parts


def _add_accurate_ring_descriptors(mol: Molecule, parts: AssemblyParts, raw_atoms: list[int], get_loc) -> None:
    """Emit per-atom descriptors for ring centres the legacy perception leaves
    unassigned but the accurate labeller does name.

    Every centre must carry a label, and each a numeric parent locant, or nothing
    is emitted: a partial set would describe some of the ring's configuration and
    silently drop the rest."""

    features: list[tuple[str, str]] = []
    for atom_idx in raw_atoms:
        descriptor = mol.accurate_cip.get(atom_idx)
        locant = str(get_loc(atom_idx))
        if descriptor is None or not locant.isdigit():
            return
        features.append((locant, descriptor))
    features.sort(key=lambda item: int(item[0]))
    parts.stereo_features.extend(features)
    parts.name_atom_bindings.append(
        ensure_name_atom_binding_tokens(
            NameAtomBinding(
                stage="assembly",
                role="small_ring_stereo",
                term=",".join(f"{locant}{descriptor}" for locant, descriptor in features),
                atom_ids=set(raw_atoms),
                locants=tuple(locant for locant, _ in features),
            )
        )
    )


def _add_relative_ring_stereo(mol: Molecule, parts: AssemblyParts, numbered_path: list[int], get_loc) -> None:
    """Render unassigned tetrahedral ring stereo as cis/trans for simple rings."""

    if not parts.is_ring or parts.is_bicycle or parts.is_spiro or parts.is_polycycle:
        return
    scoped_features = scoped_small_ring_stereo_features(mol, parts, numbered_path, get_loc)
    if scoped_features:
        parts.stereo_features.extend(scoped_features)
        parts.name_atom_bindings.append(
            ensure_name_atom_binding_tokens(
                NameAtomBinding(
                    stage="assembly",
                    role="small_ring_stereo",
                    term=",".join(f"{locant}{descriptor}" for locant, descriptor in scoped_features),
                    atom_ids={atom_idx for atom_idx in numbered_path if mol.atoms[atom_idx].raw_stereo},
                    locants=tuple(locant for locant, _ in scoped_features),
                )
            )
        )
        return
    raw_atoms = [
        atom_idx for atom_idx in numbered_path if mol.atoms[atom_idx].raw_stereo and not mol.atoms[atom_idx].stereo
    ]
    if len(raw_atoms) != 2:
        return

    first, second = raw_atoms
    first_tag = mol.atoms[first].raw_stereo
    second_tag = mol.atoms[second].raw_stereo
    if first_tag not in {"CW", "CCW"} or second_tag not in {"CW", "CCW"}:
        return
    parent_set = set(numbered_path)
    if any(
        sum(1 for neighbor_idx in mol.get_neighbors(atom_idx) if neighbor_idx not in parent_set) != 1
        for atom_idx in raw_atoms
    ):
        # ``cis``/``trans`` can only relate a pair that each bear exactly one
        # substituent — there is no single face to speak of otherwise — so a ring
        # position carrying two different groups gets its descriptor spelled out
        # instead, from the accurate labeller that does assign these centres.
        _add_accurate_ring_descriptors(mol, parts, raw_atoms, get_loc)
        return

    term = "cis" if first_tag == second_tag else "trans"
    locants = tuple(str(get_loc(atom_idx)) for atom_idx in raw_atoms)
    parts.relative_stereo_prefixes.append(term)
    parts.name_atom_bindings.append(
        ensure_name_atom_binding_tokens(
            NameAtomBinding(
                stage="assembly",
                role="relative_stereo",
                term=term,
                atom_ids=set(raw_atoms),
                locants=locants,
            )
        )
    )
