"""Public facade for audited systematic fusion nomenclature.

The fusion engine is intentionally imported lazily.  Ordinary naming runs in
legacy fusion mode by default and must not pay the cost of loading the planner,
layout search, or reconstruction audit.
"""

from importlib import import_module
from typing import Any

_EXPORTS: dict[str, tuple[str, str]] = {
    "audit_fusion_plan": (".audit", "audit_fusion_plan"),
    "ComponentSide": (".descriptor", "ComponentSide"),
    "FusionDescriptorError": (".descriptor", "FusionDescriptorError"),
    "build_fusion_name_ast": (".descriptor", "build_fusion_name_ast"),
    "component_sides": (".descriptor", "component_sides"),
    "render_fusion_name": (".descriptor", "render_fusion_name"),
    "render_fusion_name_parts": (".descriptor", "render_fusion_name_parts"),
    "AuditStatus": (".model", "AuditStatus"),
    "BondAssignment": (".model", "BondAssignment"),
    "ComponentAtom": (".model", "ComponentAtom"),
    "ComponentBond": (".model", "ComponentBond"),
    "ComponentLocant": (".model", "ComponentLocant"),
    "Face": (".model", "Face"),
    "FaceModel": (".model", "FaceModel"),
    "FusedLayout": (".model", "FusedLayout"),
    "FusionAuditFailed": (".model", "FusionAuditFailed"),
    "FusionAuditResult": (".model", "FusionAuditResult"),
    "FusionCitationNode": (".model", "FusionCitationNode"),
    "FusionComponentMatch": (".model", "FusionComponentMatch"),
    "FusionComponentSpec": (".model", "FusionComponentSpec"),
    "FusionConfirmed": (".model", "FusionConfirmed"),
    "FusionDescriptor": (".model", "FusionDescriptor"),
    "FusionGraph": (".model", "FusionGraph"),
    "FusionGraphAtom": (".model", "FusionGraphAtom"),
    "FusionGraphBond": (".model", "FusionGraphBond"),
    "FusionJoin": (".model", "FusionJoin"),
    "FusionJoinKind": (".model", "FusionJoinKind"),
    "FusionMode": (".model", "FusionMode"),
    "FusionMultiplicityGroup": (".model", "FusionMultiplicityGroup"),
    "FusionNameAst": (".model", "FusionNameAst"),
    "FusionNotApplicable": (".model", "FusionNotApplicable"),
    "FusionNumberingProof": (".model", "FusionNumberingProof"),
    "FusionParentPlan": (".model", "FusionParentPlan"),
    "FusionPlanningResult": (".model", "FusionPlanningResult"),
    "FusionRenderedPart": (".model", "FusionRenderedPart"),
    "FusionRuleDecision": (".model", "FusionRuleDecision"),
    "FusionSide": (".model", "FusionSide"),
    "FusionUnsupported": (".model", "FusionUnsupported"),
    "ParentBondModel": (".model", "ParentBondModel"),
    "PinStatus": (".model", "PinStatus"),
    "RejectedNumbering": (".model", "RejectedNumbering"),
    "SystemLocant": (".model", "SystemLocant"),
    "TypedLocantMap": (".model", "TypedLocantMap"),
    "EARLIEST_SPECIAL_HETEROATOM_PRECEDENCE": (
        ".rules",
        "EARLIEST_SPECIAL_HETEROATOM_PRECEDENCE",
    ),
    "GENERAL_HETEROATOM_COUNT_PRECEDENCE": (
        ".rules",
        "GENERAL_HETEROATOM_COUNT_PRECEDENCE",
    ),
    "ComponentSeniorityKey": (".rules", "ComponentSeniorityKey"),
    "component_seniority_key": (".rules", "component_seniority_key"),
    "component_spec_seniority_key": (".rules", "component_spec_seniority_key"),
    "explain_component_comparison": (".rules", "explain_component_comparison"),
    "fusion_mode_allows_planning": (".rules", "fusion_mode_allows_planning"),
    "pin_ring_size_gate": (".rules", "pin_ring_size_gate"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Load a public fusion symbol only when a caller requests it."""

    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
