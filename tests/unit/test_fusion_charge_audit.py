"""Independent charge-operation audit rejects duplicate or invalid evidence."""

from copy import copy, deepcopy

import pytest

from openclatura.fusion.audit import _audit_charge_operations
from openclatura.fusion.model import FusionConfirmed
from openclatura.fusion.planner import plan_fusion_parent
from openclatura.graph_io import read_smiles


@pytest.fixture(params=("[N-]1[NH+]=CC=C2C=CN=C12", "O=C1[CH-]NC2=C1C[NH2+]C2"))
def charged_plan(request):
    mol = read_smiles(request.param)
    atoms = frozenset(atom for atom in mol.atoms if len(mol.get_neighbors(atom)) > 1)
    result = plan_fusion_parent(mol, atoms, mode="audited_pin")
    assert isinstance(result, FusionConfirmed), result
    return mol, atoms, result.plan


def _errors(mol, atoms, plan, operations):
    errors = []
    _audit_charge_operations(mol, atoms, plan.abstract_parent_graph, plan.numbering, operations, errors)
    return errors


def test_charge_audit_rejects_duplicate_operations(charged_plan):
    mol, atoms, plan = charged_plan
    assert not _errors(mol, atoms, plan, plan.charge_operations)
    assert "fusion charge operations duplicate a charged atom" in _errors(
        mol, atoms, plan, plan.charge_operations + plan.charge_operations[:1]
    )


def test_charge_audit_independently_checks_proton_removal(charged_plan):
    mol, atoms, plan = charged_plan
    operation = next(op for op in plan.charge_operations if op.observed_charge == -1)
    invalid = deepcopy(mol)
    invalid.update_atom(operation.atom_id, total_h_count=mol.atoms[operation.atom_id].total_h_count + 1)
    assert "fusion deprotonation does not preserve proton-removal valence" in _errors(
        invalid, atoms, plan, plan.charge_operations
    )


def test_charge_audit_rejects_corrupted_operation_kind(charged_plan):
    mol, atoms, plan = charged_plan
    operation = copy(plan.charge_operations[0])
    # Bypass constructor validation to exercise the independent audit boundary.
    object.__setattr__(operation, "operation_kind", "unproved")
    assert "fusion charge operation has an unsupported operation kind" in _errors(
        mol, atoms, plan, (operation, *plan.charge_operations[1:])
    )
