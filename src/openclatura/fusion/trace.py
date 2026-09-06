"""Decision-trace projection for an already proven fusion parent plan."""

from __future__ import annotations

from collections.abc import Iterable

from ..molecule import DecisionTrace, Molecule, TracePhase, bond_ids_within
from ..trace_helpers import trace_decision
from .model import FusionParentPlan


def fusion_proof_counts(plan: FusionParentPlan) -> dict[str, int]:
    """Summarize bounded proof work already retained by ``plan``.

    These counts are projections of immutable proof objects, not mutable
    instrumentation.  Producing a decision trace therefore does not rerun any
    graph search.
    """

    return {
        "bounded_faces": len(plan.numbering.selected_face_model.faces),
        "component_occurrences": len(plan.ast.component_occurrences),
        "fusion_joins": len(plan.ast.joins),
        "preferred_numberings": len(plan.numbering.input_locant_maps),
        "rejected_numberings": len(plan.numbering.rejected_numberings),
        "mancude_assignments": len(plan.bond_model.allowed_kekule_assignments),
        "audit_checks": len(plan.audit.checks),
    }


def trace_confirmed_fusion_plan(
    trace: DecisionTrace | None,
    mol: Molecule,
    plan: FusionParentPlan,
    parent_atoms: Iterable[int],
) -> None:
    """Expose each existing proof stage without recomputing the plan."""

    if trace is None:
        return
    atoms = frozenset(parent_atoms)
    face_model = plan.numbering.selected_face_model
    trace_decision(
        trace,
        TracePhase.PARENT_SELECTION,
        "selected fusion face model",
        "A bounded face set covers every cyclic parent edge and reconstructs one connected perimeter.",
        atoms=atoms,
        bonds=set(face_model.perimeter_edges) | set(face_model.fusion_edges),
        data={
            "faces": [
                {"id": face.id, "atoms": list(face.atom_cycle), "bonds": list(face.edge_cycle), "size": face.size}
                for face in face_model.faces
            ],
            "perimeter_edges": sorted(face_model.perimeter_edges),
            "fusion_edges": sorted(face_model.fusion_edges),
        },
    )
    for match in plan.ast.component_occurrences:
        component_atoms = set(match.input_atom_by_locant.values())
        trace_decision(
            trace,
            TracePhase.PARENT_SELECTION,
            "matched fusion component",
            "The shared retained-template matcher proved a complete locanted component occurrence.",
            atoms=component_atoms,
            bonds=bond_ids_within(mol, component_atoms),
            data={
                "occurrence_id": match.occurrence_id,
                "spec_key": match.spec_key,
                "template_name": match.template_name,
                "covered_faces": sorted(match.covered_face_ids),
                "component_locants": dict(match.local_to_input_atom),
            },
        )
    trace_decision(
        trace,
        TracePhase.PARENT_SELECTION,
        "selected fusion parent location",
        "The citation parent occurrence or occurrences won the component-seniority and parent-location criteria within the audited component cover.",
        atoms=atoms,
        data={
            "parent_occurrences": list(plan.ast.parent_occurrences),
            "plan_kind": plan.ast.plan_kind,
            "citation_plan": {
                "primary_join_indices": list(plan.ast.citation_plan.primary_join_indices),
                "interparent_join_indices": list(plan.ast.citation_plan.interparent_join_indices),
                "cycle_closing_join_indices": list(plan.ast.citation_plan.cycle_closing_join_indices),
                "interparent_occurrences": list(plan.ast.citation_plan.interparent_occurrences),
                "render_order": list(plan.ast.citation_plan.render_order),
            }
            if plan.ast.citation_plan is not None
            else None,
            "multiplicative_groups": [
                {
                    "occurrences": list(group.occurrence_ids),
                    "multiplier": group.multiplier,
                }
                for group in plan.ast.multiplicative_groups
            ],
            "criteria": [
                {
                    "rule": decision.rule,
                    "criterion": decision.criterion,
                    "outcome": decision.outcome,
                    "reason": decision.reason,
                }
                for decision in plan.rule_trace
            ],
        },
    )
    for join, descriptor in zip(plan.ast.joins, plan.ast.descriptors, strict=True):
        trace_decision(
            trace,
            TracePhase.ASSEMBLY,
            "constructed fusion descriptor",
            "Component-local locants and parent-side letters were taken from the graph-bound fusion interface.",
            atoms=join.shared_input_atoms,
            bonds=join.shared_input_bonds,
            data={
                "attached_occurrence": join.attached_occurrence,
                "host_occurrence": join.host_occurrence,
                "order": join.order,
                "kind": join.kind.value,
                "descriptor": descriptor.render(),
            },
        )
    if plan.lambda_descriptors:
        trace_decision(
            trace,
            TracePhase.ASSEMBLY,
            "cited fusion bonding-number descriptors",
            "Neutral nonstandard-valence parent atoms are cited from the completed-system locant map.",
            atoms={descriptor.atom_id for descriptor in plan.lambda_descriptors},
            data={
                "descriptors": [descriptor.text for descriptor in plan.lambda_descriptors],
                "locants": [str(descriptor.locant) for descriptor in plan.lambda_descriptors],
            },
        )
    if plan.indicated_hydrogens:
        atom_by_locant = {locant: atom for atom, locant in plan.numbering.input_locant_maps[0]}
        trace_decision(
            trace,
            TracePhase.ASSEMBLY,
            "cited fusion indicated hydrogen",
            "Indicated-hydrogen positions were projected through the completed-system numbering proof.",
            atoms={atom_by_locant[locant] for locant in plan.indicated_hydrogens},
            data={"locants": [str(locant) for locant in plan.indicated_hydrogens]},
        )
    layout = plan.numbering.selected_layout
    trace_decision(
        trace,
        TracePhase.NUMBERING,
        "selected preferred fusion orientation",
        "Intrinsic ring layouts were ranked by the configured P-25 orientation criteria.",
        atoms=atoms,
        data={
            "orientation_score": list(layout.orientation_score),
            "face_positions": [list(item) for item in layout.face_positions],
            "face_shapes": [list(item) for item in layout.face_shapes],
            "audit_evidence": list(layout.audit_evidence),
        },
    )
    trace_decision(
        trace,
        TracePhase.NUMBERING,
        "selected completed fusion numbering",
        "The final parent locants come from the preferred oriented perimeter and ordered heteroatom tie-breaks.",
        atoms=atoms,
        data={
            "atom_to_locant": {atom: str(locant) for atom, locant in plan.numbering.input_locant_maps[0]},
            "candidate_count": len(plan.numbering.input_locant_maps),
            "rejected_candidates": [item.reason for item in plan.numbering.rejected_numberings],
            "proof_counts": fusion_proof_counts(plan),
        },
    )
    trace_decision(
        trace,
        TracePhase.ASSEMBLY,
        "audited systematic fusion parent",
        "Independent descriptor, numbering, bond-model, and graph reconstruction checks all passed.",
        atoms=atoms,
        data={
            "status": plan.audit.status.value,
            "checks": list(plan.audit.checks),
            "proof_counts": fusion_proof_counts(plan),
        },
    )
