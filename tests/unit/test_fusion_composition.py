"""Completed-system ownership of independent indicated-H and hydro/oxo sites."""

from dataclasses import replace

import pytest

from openclatura.chains import find_ring_systems
from openclatura.fusion.audit import audit_fusion_plan
from openclatura.fusion.model import AuditStatus, FusionConfirmed, FusionMode, FusionUnsupported
from openclatura.fusion.planner import plan_fusion_parent
from openclatura.graph_io import read_smiles


@pytest.fixture(params=[False, True], ids=["neutral", "cation"])
def composed_parent(request):
    nitrogen = "[NH2+]" if request.param else "N"
    mol = read_smiles(f"{nitrogen}1CCc2c(cc[nH]c2=O)C1")
    atoms = max(find_ring_systems(mol), key=lambda system: len(system.atoms)).atoms
    result = plan_fusion_parent(mol, atoms, mode=FusionMode.AUDITED_PIN)
    assert isinstance(result, FusionConfirmed)
    return mol, atoms, result.plan


def _audit(mol, atoms, plan, **changes):
    arguments = dict(
        ast=plan.ast,
        abstract_parent_graph=plan.abstract_parent_graph,
        numbering=plan.numbering,
        bond_model=plan.bond_model,
        indicated_hydrogens=plan.indicated_hydrogens,
        charge_operations=plan.charge_operations,
        derivative_state=plan.derivative_state,
        mode=FusionMode.AUDITED_PIN,
    )
    arguments.update(changes)
    return audit_fusion_plan(mol, atoms, **arguments)


def test_indicated_h_and_oxo_do_not_suppress_separate_carbon_hydrogenation(composed_parent):
    mol, atoms, plan = composed_parent
    state = plan.derivative_state
    assert tuple(map(str, plan.indicated_hydrogens)) == ("2", "6")
    assert len(state.hydro_operations) == 1
    hydro = state.hydro_operations[0]
    assert hydro.locants == ("3", "4")
    assert len(hydro.bond_ids) == 1
    assert all(mol.atoms[atom].symbol == "C" for atom in hydro.atom_ids)
    assert tuple(operation.locant for operation in state.oxo_operations) == ("5",)
    assert not state.unsaturation_operations
    assert _audit(mol, atoms, plan).status is AuditStatus.CONFIRMED
    assert plan_fusion_parent(mol, atoms, mode=FusionMode.AUDITED_PIN).plan is plan


@pytest.mark.parametrize("field", ["atom_ids", "bond_ids", "locants"])
def test_composed_hydro_operation_still_requires_exact_graph_binding(composed_parent, field):
    mol, atoms, plan = composed_parent
    state = plan.derivative_state
    operation = state.hydro_operations[0]
    corrupted = replace(operation, **{field: getattr(operation, field)[:-1]})
    result = _audit(mol, atoms, plan, derivative_state=replace(state, hydro_operations=(corrupted,)))
    assert result.status is AuditStatus.MISMATCH
    assert "typed hydro operation does not exactly represent hydrogenated parent edges" in result.errors


def test_composed_oxo_operation_still_requires_exact_graph_binding(composed_parent):
    mol, atoms, plan = composed_parent
    state = plan.derivative_state
    corrupted = replace(state.oxo_operations[0], locant="8")
    result = _audit(mol, atoms, plan, derivative_state=replace(state, oxo_operations=(corrupted,)))
    assert result.status is AuditStatus.MISMATCH
    assert "typed oxo operations do not represent every exocyclic parent oxo group" in result.errors


def test_composed_parent_still_requires_indicated_hydrogen_proof(composed_parent):
    mol, atoms, plan = composed_parent
    result = _audit(mol, atoms, plan, indicated_hydrogens=plan.indicated_hydrogens[:1])
    assert result.status is AuditStatus.MISMATCH
    assert "fusion indicated-hydrogen citations omit a graph-required site" in result.errors


def test_multiple_saturated_indicated_h_sites_do_not_bypass_composition_guard():
    mol = read_smiles("N1CCCC2C1C[NH2+]C2")
    result = plan_fusion_parent(mol, mol.atoms, mode=FusionMode.AUDITED_PIN)
    assert isinstance(result, FusionUnsupported)
    assert "combined indicated-hydrogen and bond/oxo derivative fusion grammar is not audited" in result.details
