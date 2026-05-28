"""Derive high-level nomenclature operations from naming traces."""

from .molecule import NomenclatureOperation, OperationClass, TracePhase, TraceStep


def infer_operations(decisions: list[TraceStep], trace_segments: list[dict]) -> list[NomenclatureOperation]:
    """Return operation records for the supported structure-to-name pipeline.

    This is intentionally derived from structural decision data, not from final
    name strings. It provides an operation ledger for explainability while the
    detailed graph decisions remain in ``TraceStep`` and trace segments.
    """

    operations: list[NomenclatureOperation] = []
    for step in decisions:
        if step.phase == TracePhase.ASSEMBLY and step.decision == "assembled component name":
            principal_key = step.data.get("principal_key")
            substituent_count = int(step.data.get("substituent_count") or 0)
            unsaturation_count = int(step.data.get("unsaturation_count") or 0)
            if principal_key or substituent_count:
                operations.append(
                    NomenclatureOperation(
                        OperationClass.SUBSTITUTIVE,
                        "principal_group_and_substituent_assembly",
                    )
                )
            if unsaturation_count:
                operations.append(NomenclatureOperation(OperationClass.SUBTRACTIVE, "unsaturation"))
        elif step.phase == TracePhase.PARENT_SELECTION and step.decision == "selected parent skeleton":
            if step.data.get("is_polycycle") and step.data.get("polycycle_descriptor"):
                operations.append(NomenclatureOperation(OperationClass.FUSION, "polycyclic_parent"))

    if any(str(segment.get("key", "")).startswith("replacement:") for segment in trace_segments):
        operations.append(NomenclatureOperation(OperationClass.REPLACEMENT, "replacement_prefix"))
    if any(str(segment.get("key", "")).startswith("unsaturation:") for segment in trace_segments):
        operations.append(NomenclatureOperation(OperationClass.SUBTRACTIVE, "unsaturation"))

    return _dedupe_operations(operations)


def _dedupe_operations(operations: list[NomenclatureOperation]) -> list[NomenclatureOperation]:
    """Keep operation records stable and unique."""

    deduped: list[NomenclatureOperation] = []
    seen: set[tuple[OperationClass, str, tuple[str, ...]]] = set()
    for operation in operations:
        key = (operation.operation_class, operation.detail, operation.locants)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(operation)
    return deduped
