from openclatura.molecule import OperationClass, TracePhase, TraceStep
from openclatura.operations import infer_operations


def test_systematic_fusion_extends_legacy_polycycle_operation_metadata():
    decisions = [
        TraceStep(
            phase=TracePhase.PARENT_SELECTION,
            decision="selected parent skeleton",
            reason="legacy polycycle fixture",
            data={"is_polycycle": True, "polycycle_descriptor": "tricyclo[2.2.1.0]"},
        ),
        TraceStep(
            phase=TracePhase.PARENT_SELECTION,
            decision="selected audited systematic fusion parent",
            reason="systematic fusion fixture",
        ),
    ]

    operations = infer_operations(decisions, [])

    assert [(operation.operation_class, operation.detail) for operation in operations] == [
        (OperationClass.FUSION, "polycyclic_parent"),
        (OperationClass.FUSION, "systematic_fusion_parent"),
    ]
