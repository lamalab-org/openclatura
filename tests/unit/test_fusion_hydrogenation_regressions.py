"""Fusion N composition must not erase or invent additive hydrogen pairs."""

from dataclasses import replace

import pytest

from openclatura.fusion.audit import _audit_derivative_state
from openclatura.fusion.model import FusionConfirmed
from openclatura.fusion.planner import plan_fusion_parent
from openclatura.graph_io import read_smiles


@pytest.fixture(params=[("O=c1c2ccccc2nc2n1CCC2", ("2", "3")), ("C1=CSC2=NCCN12", ("5", "6"))])
def hydro_plan(request):
    smiles, expected = request.param
    mol = read_smiles(smiles)
    atoms = frozenset(atom for atom in mol.atoms if mol.atoms[atom].symbol != "O")
    result = plan_fusion_parent(mol, atoms, mode="general")
    assert isinstance(result, FusionConfirmed)
    return mol, atoms, result.plan, expected


def test_hydrogenation_has_one_carbon_pair_and_no_false_indicated_h(hydro_plan):
    mol, atoms, plan, expected = hydro_plan
    state = plan.derivative_state
    assert plan.audit.confirmed
    assert plan.indicated_hydrogens == ()
    assert not state.unsaturation_operations
    assert not state.added_hydrogen_operations
    assert len(state.hydro_operations) == 1
    operation = state.hydro_operations[0]
    assert operation.locants == expected
    assert len(state.bond_delta.hydrogenated_edges) == 1
    assert operation.bond_ids == (mol.get_bond(*operation.atom_ids).idx,)
    assert all(mol.atoms[atom].symbol == "C" and not mol.atoms[atom].is_aromatic for atom in operation.atom_ids)
    assert set(operation.atom_ids) <= atoms


@pytest.mark.parametrize("corruption", ["missing", "locants", "atoms", "bonds"])
def test_derivative_audit_rejects_corrupted_hydro_proof(hydro_plan, corruption):
    mol, atoms, plan, _ = hydro_plan
    state = plan.derivative_state
    operation = state.hydro_operations[0]
    changes = {"locants": {"locants": ("99", "100")}, "atoms": {"atom_ids": (-1, -2)}, "bonds": {"bond_ids": (-1,)}}
    operations = () if corruption == "missing" else (replace(operation, **changes[corruption]),)
    errors = []
    _audit_derivative_state(
        mol,
        atoms,
        plan.numbering,
        plan.bond_model,
        plan.indicated_hydrogens,
        replace(state, hydro_operations=operations),
        errors,
    )
    assert errors and any("hydro" in error for error in errors)
