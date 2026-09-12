"""Fusion N composition must not erase or invent additive hydrogen pairs."""

from dataclasses import replace

import pytest

from openclatura.fusion.audit import _audit_derivative_state
from openclatura.fusion.model import FusionConfirmed
from openclatura.fusion.planner import plan_fusion_parent
from openclatura.graph_io import read_smiles


@pytest.fixture(params=[("O=c1c2ccccc2nc2n1CCC2", ("2", "3"), ("1",)), ("C1=CSC2=NCCN12", ("5", "6"), ())])
def hydro_plan(request):
    smiles, expected, added = request.param
    mol = read_smiles(smiles)
    atoms = frozenset(atom for atom in mol.atoms if mol.atoms[atom].symbol != "O")
    result = plan_fusion_parent(mol, atoms, mode="general")
    assert isinstance(result, FusionConfirmed)
    return mol, atoms, result.plan, expected, added


def test_hydrogenation_pair_and_suffix_added_h_have_separate_graph_scopes(hydro_plan):
    mol, atoms, plan, expected, added = hydro_plan
    state = plan.derivative_state
    assert plan.audit.confirmed
    assert plan.indicated_hydrogens == ()
    assert not state.unsaturation_operations
    assert tuple(locant for operation in state.added_hydrogen_operations for locant in operation.locants) == added
    locants = {atom: str(locant) for atom, locant in plan.numbering.input_locant_maps[0]}
    for operation in state.added_hydrogen_operations:
        assert tuple(locants[atom] for atom in operation.atom_ids) == operation.locants
        assert all(mol.atoms[atom].symbol == "C" and mol.atoms[atom].total_h_count == 2 for atom in operation.atom_ids)
        assert not set(operation.atom_ids).intersection(state.bond_delta.hydrogenated_atom_ids)
        assert set(operation.bond_ids) == {
            mol.get_bond(atom, neighbor).idx
            for atom in operation.atom_ids
            for neighbor in mol.get_neighbors(atom)
            if neighbor in atoms
        }
    assert len(state.hydro_operations) == 1
    operation = state.hydro_operations[0]
    assert operation.locants == expected
    assert len(state.bond_delta.hydrogenated_edges) == 1
    assert operation.bond_ids == (mol.get_bond(*operation.atom_ids).idx,)
    assert all(mol.atoms[atom].symbol == "C" and not mol.atoms[atom].is_aromatic for atom in operation.atom_ids)
    assert set(operation.atom_ids) <= atoms


@pytest.mark.parametrize("corruption", ["missing", "locants", "atoms", "bonds"])
def test_derivative_audit_rejects_corrupted_hydro_proof(hydro_plan, corruption):
    mol, atoms, plan, _, _ = hydro_plan
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
