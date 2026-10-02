"""Net hydrogenation requires exact redistribution and implicit-state proofs."""

from dataclasses import replace

import pytest
from rdkit import Chem

from openclatura.fusion.audit import _audit_derivative_state
from openclatura.fusion.mancude import (
    _single_site_parent_model,
    compare_actual_parent_to_implied_parent,
    parent_derivative_state,
)
from openclatura.fusion.model import BondAssignment, FusionConfirmed, FusionGraph, FusionGraphAtom, FusionGraphBond
from openclatura.fusion.numbering import parent_bond_model
from openclatura.fusion.planner import plan_fusion_parent
from openclatura.graph_io import read_rdkit_mol, read_smiles

SMILES = "C1=CC2=C3C4=C(C2C=C1)C1C=CC=CC1=C4c1ccccc13"


def _model(mol, fixed_single=None):
    return parent_bond_model(
        FusionGraph(
            atoms=tuple(FusionGraphAtom(atom, value.symbol) for atom, value in mol.atoms.items()),
            bonds=tuple(
                FusionGraphBond(
                    tuple(sorted((bond.u, bond.v))),
                    "single" if tuple(sorted((bond.u, bond.v))) == fixed_single else "mancude",
                )
                for bond in mol.bonds.values()
            ),
        )
    )


@pytest.fixture
def polycarbon_plan():
    mol = read_smiles(SMILES)
    result = plan_fusion_parent(mol, mol.atoms, mode="general")
    assert isinstance(result, FusionConfirmed)
    return mol, result.plan


def test_polycarbon_keeps_raw_delta_and_proves_only_net_hydro_endpoints(polycarbon_plan):
    mol, plan = polycarbon_plan
    state = plan.derivative_state
    raw = state.bond_delta
    assert raw == compare_actual_parent_to_implied_parent(
        mol, mol.atoms, plan.bond_model, atom_to_locant=dict(plan.numbering.input_locant_maps[0])
    )
    assert len(raw.hydrogenated_edges) == 3
    assert len(raw.additional_multiple_bond_ids) == 2
    proof = state.pi_redistribution
    assert proof is not None
    assert proof.hydrogenated_atom_ids == frozenset({6, 9})
    assert set(state.hydro_operations[0].atom_ids) == {6, 9}
    assert not state.unsaturation_operations
    assert state.hydro_operations[0].bond_ids == tuple(sorted(proof.removed_bond_ids | proof.added_bond_ids))
    assert len(proof.removed_bond_ids) - len(proof.added_bond_ids) == 1

    replayed = dict(raw.assignment.orders)
    for bond_id, change in [
        *((bond, -1) for bond in proof.removed_bond_ids),
        *((bond, 1) for bond in proof.added_bond_ids),
    ]:
        bond = mol.bonds[bond_id]
        replayed[tuple(sorted((bond.u, bond.v)))] += change
    assert replayed == dict(proof.final_assignment.orders)
    assert replayed == {tuple(sorted((bond.u, bond.v))): bond.order for bond in mol.bonds.values()}
    implicit = _single_site_parent_model(plan.bond_model, proof.hydrogenated_atom_ids)
    assert proof.final_assignment in implicit.allowed_kekule_assignments
    assert plan.bond_model.maximum_non_cumulative_double_bonds == 11
    assert implicit.maximum_non_cumulative_double_bonds == 10


@pytest.mark.parametrize(
    "field", ["removed_bond_ids", "added_bond_ids", "hydrogenated_atom_ids", "final_assignment", "missing"]
)
def test_audit_rejects_corrupted_redistribution_certificate(polycarbon_plan, field):
    mol, plan = polycarbon_plan
    proof = plan.derivative_state.pi_redistribution
    assert proof is not None
    if field == "missing":
        corrupted = None
    elif field == "final_assignment":
        orders = proof.final_assignment.orders
        corrupted = replace(proof, final_assignment=BondAssignment(((orders[0][0], 3 - orders[0][1]), *orders[1:])))
    else:
        corrupted = replace(proof, **{field: frozenset()})
    errors = []
    _audit_derivative_state(
        mol,
        frozenset(mol.atoms),
        plan.numbering,
        plan.bond_model,
        plan.indicated_hydrogens,
        replace(plan.derivative_state, pi_redistribution=corrupted),
        errors,
    )
    assert any("pi redistribution" in error for error in errors)


def test_audit_still_reconstructs_raw_deleted_and_added_edges(polycarbon_plan):
    mol, plan = polycarbon_plan
    errors = []
    corrupted = replace(plan.derivative_state.bond_delta, additional_multiple_bond_ids=frozenset())
    _audit_derivative_state(
        mol,
        frozenset(mol.atoms),
        plan.numbering,
        plan.bond_model,
        plan.indicated_hydrogens,
        replace(plan.derivative_state, bond_delta=corrupted),
        errors,
    )
    assert "typed derivative state does not carry the selected parent bond delta" in errors


@pytest.mark.parametrize("field", ["hydro_operations", "unsaturation_operations"])
def test_redistribution_cannot_double_count_the_raw_edge_operations(polycarbon_plan, field):
    mol, plan = polycarbon_plan
    raw_state = parent_derivative_state(
        mol,
        mol.atoms,
        plan.bond_model,
        dict(plan.numbering.input_locant_maps[0]),
        preserve_retained_parent_state=True,
    )
    errors = []
    _audit_derivative_state(
        mol,
        frozenset(mol.atoms),
        plan.numbering,
        plan.bond_model,
        plan.indicated_hydrogens,
        replace(plan.derivative_state, **{field: getattr(raw_state, field)}),
        errors,
    )
    assert errors
    assert any("exactly represent hydrogenated" in error or "unsaturation operations" in error for error in errors)


@pytest.mark.parametrize("smiles", ["C1C=CCC=C1", "C1C=CCc2ccccc21"])
def test_alternating_path_hydrogenation_is_not_specific_to_polycarbon_parent(smiles):
    mol = read_smiles(smiles)
    state = parent_derivative_state(mol, mol.atoms, _model(mol), {atom: str(atom + 1) for atom in mol.atoms})
    assert state.pi_redistribution is not None
    assert state.pi_redistribution.hydrogenated_atom_ids == {0, 3}
    assert state.hydro_operations[0].locants == ("1", "4")
    assert not state.unsaturation_operations


@pytest.mark.parametrize("smiles,fixed", [("C1=CC=C=CC1", None), ("C1C=CCC=C1", (4, 5))])
def test_genuine_unsaturation_and_fixed_single_constraints_cannot_be_erased(smiles, fixed):
    mol = read_smiles(smiles)
    state = parent_derivative_state(mol, mol.atoms, _model(mol, fixed), {atom: str(atom + 1) for atom in mol.atoms})
    assert state.pi_redistribution is None
    assert state.unsaturation_operations
    assert {
        operation.bond_id for operation in state.unsaturation_operations
    } == state.bond_delta.additional_multiple_bond_ids


def test_retained_parent_conventions_do_not_implicitly_enable_redistribution():
    mol = read_smiles("C1C=CCC=C1")
    state = parent_derivative_state(
        mol,
        mol.atoms,
        _model(mol),
        {atom: str(atom + 1) for atom in mol.atoms},
        preserve_retained_parent_state=True,
    )
    assert state.pi_redistribution is None
    assert state.unsaturation_operations


def test_substitution_at_a_hydro_endpoint_preserves_the_parent_pi_deficit(polycarbon_plan):
    _, plan = polycarbon_plan
    graph = Chem.RWMol(Chem.MolFromSmiles(SMILES))
    methyl = graph.AddAtom(Chem.Atom("C"))
    graph.AddBond(6, methyl, Chem.BondType.SINGLE)
    Chem.SanitizeMol(graph)
    mol = read_rdkit_mol(graph)
    locants = dict(plan.numbering.input_locant_maps[0])
    state = parent_derivative_state(mol, frozenset(locants), plan.bond_model, locants)
    assert mol.atoms[6].total_h_count == 0
    assert state.pi_redistribution is not None
    assert state.pi_redistribution.hydrogenated_atom_ids == {6, 9}
    assert set(state.hydro_operations[0].atom_ids) == {6, 9}


def test_redistribution_implicit_matching_respects_search_budget(polycarbon_plan, monkeypatch):
    from openclatura.fusion.numbering import MancudeSearchBudgetExceeded

    mol, plan = polycarbon_plan

    def exhausted(*args, **kwargs):
        raise MancudeSearchBudgetExceeded(1)

    monkeypatch.setattr("openclatura.fusion.mancude._single_site_parent_model", exhausted)
    with pytest.raises(MancudeSearchBudgetExceeded):
        parent_derivative_state(mol, mol.atoms, plan.bond_model, dict(plan.numbering.input_locant_maps[0]))
