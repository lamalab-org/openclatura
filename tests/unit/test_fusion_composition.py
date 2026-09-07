"""Completed-system ownership of independent indicated-H and hydro/oxo sites."""

from dataclasses import replace

import pytest

from openclatura.chains import find_ring_systems
from openclatura.fusion.audit import audit_fusion_plan
from openclatura.fusion.mancude import parent_derivative_state
from openclatura.fusion.model import AuditStatus, FusionConfirmed, FusionMode
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


def test_multiple_saturated_indicated_h_sites_preserve_carbon_hydrogenation():
    mol = read_smiles("N1CCCC2C1C[NH2+]C2")
    result = plan_fusion_parent(mol, mol.atoms, mode=FusionMode.AUDITED_PIN)
    assert isinstance(result, FusionConfirmed)
    plan = result.plan
    assert tuple(map(str, plan.indicated_hydrogens)) == ("1", "6")
    assert len(plan.derivative_state.hydro_operations[0].atom_ids) == 6
    assert len(plan.derivative_state.hydro_operations[0].bond_ids) == 3
    assert _audit(mol, mol.atoms, plan).status is AuditStatus.CONFIRMED


def test_uncovered_saturated_carbon_h_sites_use_complete_typed_hydrogenation():
    mol = read_smiles("N1CCCC2C1CC[NH2+]C2")
    result = plan_fusion_parent(mol, mol.atoms, mode=FusionMode.AUDITED_PIN)
    assert isinstance(result, FusionConfirmed)
    plan = result.plan
    assert not plan.indicated_hydrogens
    operation = plan.derivative_state.hydro_operations[0]
    assert len(operation.atom_ids) == 10
    assert len(operation.bond_ids) == 5
    assert set(operation.atom_ids) == set(mol.atoms)
    assert _audit(mol, mol.atoms, plan).status is AuditStatus.CONFIRMED


def test_incomplete_nh_constrained_state_still_fails_composition_audit():
    mol = read_smiles("N1CCCC2C1CC[NH2+]C2")
    plan = plan_fusion_parent(mol, mol.atoms, mode=FusionMode.AUDITED_PIN).plan
    sites = {atom for atom in mol.atoms if mol.atoms[atom].symbol == "N"}
    locants = dict(plan.numbering.input_locant_maps[0])
    state = parent_derivative_state(mol, mol.atoms, plan.bond_model, locants, indicated_hydrogen_atom_ids=sites)
    result = _audit(
        mol, mol.atoms, plan, derivative_state=state, indicated_hydrogens=tuple(locants[atom] for atom in sites)
    )
    assert result.status is AuditStatus.ABSTAIN
    assert "combined indicated-hydrogen and bond/oxo derivative fusion grammar is not audited" in result.errors


@pytest.mark.parametrize("field", ["atom_ids", "bond_ids", "locants"])
@pytest.mark.parametrize("smiles", ["N1CCCC2C1C[NH2+]C2", "N1CCCC2C1CC[NH2+]C2"])
def test_saturated_composition_rejects_corrupted_hydro_binding(field, smiles):
    mol = read_smiles(smiles)
    plan = plan_fusion_parent(mol, mol.atoms, mode=FusionMode.AUDITED_PIN).plan
    state = plan.derivative_state
    operation = state.hydro_operations[0]
    corrupted = replace(operation, **{field: getattr(operation, field)[:-1]})
    result = _audit(mol, mol.atoms, plan, derivative_state=replace(state, hydro_operations=(corrupted,)))
    assert result.status is AuditStatus.MISMATCH
    assert "typed hydro operation does not exactly represent hydrogenated parent edges" in result.errors


@pytest.mark.parametrize("smiles", ["N1CCCC2C1C[NH2+]C2", "N1CCCC2C1CC[NH2+]C2"])
def test_saturated_composition_requires_charge_operation(smiles):
    mol = read_smiles(smiles)
    plan = plan_fusion_parent(mol, mol.atoms, mode=FusionMode.AUDITED_PIN).plan
    result = _audit(mol, mol.atoms, plan, charge_operations=())
    assert result.status is AuditStatus.MISMATCH
    assert "fusion charge operations do not exactly represent parent formal-charge changes" in result.errors


def test_full_hydro_cannot_suppress_nh_proof_after_losing_a_pair():
    mol = read_smiles("N1CCCC2C1CC[NH2+]C2")
    plan = plan_fusion_parent(mol, mol.atoms, mode=FusionMode.AUDITED_PIN).plan
    state = plan.derivative_state
    delta = replace(state.bond_delta, hydrogenated_edges=state.bond_delta.hydrogenated_edges[:-1])
    result = _audit(mol, mol.atoms, plan, derivative_state=replace(state, bond_delta=delta))
    assert result.status is not AuditStatus.CONFIRMED
