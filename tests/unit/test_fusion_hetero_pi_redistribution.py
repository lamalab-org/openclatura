"""Carbon hydrogenation may redistribute pi bonds beside fixed heteroatoms."""

from dataclasses import replace

import pytest
from rdkit import Chem

from openclatura import name, opsin_available, verify_with_opsin
from openclatura.fusion.audit import _audit_derivative_state
from openclatura.fusion.mancude import (
    compare_actual_parent_to_implied_parent,
    parent_derivative_state,
    prove_pi_redistribution,
)
from openclatura.fusion.model import BondAssignment, FusionConfirmed, FusionGraph, FusionGraphAtom, FusionGraphBond
from openclatura.fusion.numbering import parent_bond_model
from openclatura.fusion.planner import plan_fusion_parent
from openclatura.graph_io import read_rdkit_mol, read_smiles
from openclatura.molecule import Molecule

SMILES = "C1OCC2=C1OC=C2"


@pytest.fixture
def furan_plan():
    mol = read_smiles(SMILES)
    result = plan_fusion_parent(mol, mol.atoms, mode="audited_pin")
    assert isinstance(result, FusionConfirmed), result
    return mol, result.plan


@pytest.mark.parametrize("mode", ["general", "audited_pin"])
@pytest.mark.parametrize("reverse", [False, True])
def test_furan_hydro_uses_carbon_endpoints_not_raw_deleted_edges(mode, reverse):
    graph = Chem.MolFromSmiles(SMILES)
    if reverse:
        graph = Chem.RenumberAtoms(graph, list(reversed(range(graph.GetNumAtoms()))))
    mol = read_rdkit_mol(graph)
    result = plan_fusion_parent(mol, mol.atoms, mode=mode)
    assert isinstance(result, FusionConfirmed), result
    plan = result.plan
    state = plan.derivative_state
    raw = state.bond_delta
    assert len(raw.hydrogenated_edges) == 2
    assert not raw.additional_multiple_bond_ids
    proof = state.pi_redistribution
    assert proof is not None
    assert len(proof.removed_bond_ids) == 2
    assert len(proof.added_bond_ids) == 1
    assert len(state.hydro_operations) == 1
    assert state.hydro_operations[0].locants == ("4", "6")
    assert set(state.hydro_operations[0].atom_ids) == proof.hydrogenated_atom_ids
    assert all(mol.atoms[atom].symbol == "C" for atom in proof.hydrogenated_atom_ids)
    assert not state.unsaturation_operations
    errors = []
    _audit_derivative_state(mol, frozenset(mol.atoms), plan.numbering, plan.bond_model, (), state, errors)
    assert errors == []


def test_furan_public_name_has_exact_opsin_hydrogen_count():
    generated = name(SMILES).name
    assert generated == "4,6-dihydrofuro[3,4-b]furan"
    if opsin_available():
        check = verify_with_opsin(generated, SMILES, standardize_smiles=False)
        assert check.status == "matched", check.to_dict()


@pytest.mark.parametrize(
    "field", ["removed_bond_ids", "added_bond_ids", "hydrogenated_atom_ids", "final_assignment", "missing"]
)
def test_hetero_spectators_do_not_weaken_redistribution_certificate_audit(furan_plan, field):
    mol, plan = furan_plan
    state = plan.derivative_state
    proof = state.pi_redistribution
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
        (),
        replace(state, pi_redistribution=corrupted),
        errors,
    )
    assert any("pi redistribution" in error for error in errors)


def _spectator_ring(symbol, *, fixed_edge=None):
    mol = Molecule()
    for atom in range(7):
        mol.add_atom(symbol if atom == 6 else "C", idx=atom)
    for atom in range(7):
        other = (atom + 1) % 7
        mol.add_bond(atom, other, idx=atom + 10, order=2 if atom in {1, 4} else 1)
    for atom, value in mol.atoms.items():
        bonds = sum(mol.get_bond(atom, other).order for other in mol.get_neighbors(atom))
        mol.update_atom(atom, total_h_count=value.element.standard_valence - bonds)
    graph = FusionGraph(
        atoms=tuple(FusionGraphAtom(atom, value.symbol, forced_single=atom == 6) for atom, value in mol.atoms.items()),
        bonds=tuple(
            FusionGraphBond(tuple(sorted((bond.u, bond.v))), "single" if {bond.u, bond.v} == fixed_edge else "mancude")
            for bond in mol.bonds.values()
        ),
    )
    return mol, parent_bond_model(graph)


@pytest.mark.parametrize("symbol", ["O", "S", "N"])
def test_graph_built_neutral_heteroatom_is_an_unchanged_spectator(symbol):
    mol, model = _spectator_ring(symbol)
    state = parent_derivative_state(mol, mol.atoms, model, {atom: str(atom + 1) for atom in mol.atoms})
    proof = state.pi_redistribution
    assert proof is not None
    assert proof.hydrogenated_atom_ids == {0, 3}
    assert state.hydro_operations[0].locants == ("1", "4")
    assert not state.unsaturation_operations
    expected = dict(state.bond_delta.assignment.orders)
    assert all(
        expected[tuple(sorted((6, other)))] == mol.get_bond(6, other).order == 1 for other in mol.get_neighbors(6)
    )


@pytest.mark.parametrize("corruption", ["charge", "bonding_number", "external_multiple"])
def test_unproved_heteroatom_chemistry_cannot_be_treated_as_a_spectator(corruption):
    mol, model = _spectator_ring("O")
    delta = compare_actual_parent_to_implied_parent(mol, mol.atoms, model)
    assert prove_pi_redistribution(mol, frozenset(mol.atoms), model, delta) is not None
    atoms = frozenset(mol.atoms)
    if corruption == "charge":
        mol.update_atom(6, charge=1)
    elif corruption == "bonding_number":
        mol.update_atom(6, total_h_count=1)
    else:
        mol.add_atom("O", idx=99)
        mol.add_bond(6, 99, order=2, idx=99)
    assert prove_pi_redistribution(mol, atoms, model, delta) is None


def test_spectator_extension_preserves_fixed_single_constraints():
    mol, model = _spectator_ring("O", fixed_edge={1, 2})
    state = parent_derivative_state(mol, mol.atoms, model, {atom: str(atom + 1) for atom in mol.atoms})
    assert state.pi_redistribution is None
    assert state.unsaturation_operations


def test_neutral_amine_and_carbon_are_proved_hydrogenation_endpoints():
    mol = Molecule()
    for atom in range(6):
        mol.add_atom("N" if atom == 0 else "C", idx=atom, total_h_count=1 if atom == 0 else 2 if atom == 3 else 1)
    for atom in range(6):
        mol.add_bond(atom, (atom + 1) % 6, order=2 if atom in {1, 4} else 1)
    graph = FusionGraph(
        atoms=tuple(FusionGraphAtom(atom, value.symbol) for atom, value in mol.atoms.items()),
        bonds=tuple(FusionGraphBond(tuple(sorted((bond.u, bond.v))), "mancude") for bond in mol.bonds.values()),
    )
    model = parent_bond_model(graph)
    delta = compare_actual_parent_to_implied_parent(mol, mol.atoms, model, preserve_retained_parent_state=True)
    assert delta is not None
    proof = prove_pi_redistribution(mol, frozenset(mol.atoms), model, delta)
    assert proof is not None
    assert proof.hydrogenated_atom_ids == {0, 3}
    assert dict(proof.final_assignment.orders) == {
        tuple(sorted((bond.u, bond.v))): bond.order for bond in mol.bonds.values()
    }
    mol.update_atom(0, charge=1)
    assert prove_pi_redistribution(mol, frozenset(mol.atoms), model, delta) is None


def test_indicated_hydrogen_on_unchanged_spectator_preserves_redistribution():
    mol, model = _spectator_ring("N")
    delta = compare_actual_parent_to_implied_parent(mol, mol.atoms, model)
    proof = prove_pi_redistribution(mol, frozenset(mol.atoms), model, delta, indicated_hydrogen_atom_ids={6})
    assert proof is not None
    assert proof.hydrogenated_atom_ids == {0, 3}
    for invalid in ({0}, {99}):
        assert (
            prove_pi_redistribution(mol, frozenset(mol.atoms), model, delta, indicated_hydrogen_atom_ids=invalid)
            is None
        )


@pytest.mark.opsin
@pytest.mark.parametrize("smiles", ["C1OCC2=C1NC=C2", "C1OCC2=C1NC=N2", "C1NCC2=C1OC=N2"])
def test_indicated_nh_with_carbon_redistribution_roundtrips(smiles):
    mol = read_smiles(smiles)
    result = plan_fusion_parent(mol, mol.atoms, mode="audited_pin")
    assert isinstance(result, FusionConfirmed), result
    plan = result.plan
    assert plan.indicated_hydrogens
    assert plan.derivative_state.pi_redistribution is not None
    generated = name(smiles).name
    if not opsin_available():
        pytest.skip("OPSIN and Java are required")
    check = verify_with_opsin(generated, smiles, standardize_smiles=False)
    assert check.status == "matched", check.to_dict()
