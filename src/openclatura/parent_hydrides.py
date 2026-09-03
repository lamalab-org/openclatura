"""Resolve audited parent-hydride plans without changing legacy precedence."""

from __future__ import annotations

from .fusion.context import current_fusion_mode
from .fusion.model import FusionConfirmed, ParentHydridePlan
from .fusion.planner import PLANNER_TIER, plan_fusion_parent
from .fusion.rules import fusion_mode_allows_planning
from .molecule import DecisionTrace, Molecule, TracePhase
from .parent_selection import ParentSelection
from .trace_helpers import trace_decision


def resolve_fusion_parent_hydride(
    mol: Molecule,
    selection: ParentSelection,
    *,
    retained_name: str | None,
    decision_trace: DecisionTrace | None = None,
) -> ParentHydridePlan | None:
    """Try systematic fusion after retained-parent resolution, then fall back."""

    mode = current_fusion_mode()
    if not fusion_mode_allows_planning(mode):
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
    if isinstance(result, FusionConfirmed):
        plan = result.plan
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
                "pin_status": str(plan.pin_status),
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
            },
        )
        return ParentHydridePlan.from_fusion(plan)
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
