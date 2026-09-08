"""Graph-only regression coverage for intrinsic sites and derivative replay."""

from dataclasses import replace

import pytest
from test_fusion_audit import _two_fused_rings

from openclatura.fusion import indicated_hydrogen as intrinsic
from openclatura.fusion.audit import _audit_derivative_state, _has_consistent_derivative_operations
from openclatura.fusion.mancude import parent_derivative_state
from openclatura.fusion.model import FusionGraph, FusionGraphAtom, FusionGraphBond
from openclatura.fusion.numbering import MancudeSearchBudgetExceeded, parent_bond_model
from openclatura.locants import SystemLocant
from openclatura.molecule import Molecule


def _two_carbon_h_domains():
    # Two odd pi domains joined by a fixed single edge. Each maximum matching
    # leaves one CH2; neither site may be obtained by deleting a parent double.
    edges = tuple(tuple(sorted((offset + atom, offset + (atom + 1) % 5))) for offset in (0, 5) for atom in range(5)) + (
        (2, 7),
    )
    doubles = {(1, 2), (3, 4), (6, 7), (8, 9)}
    mol = Molecule()
    for atom in range(10):
        mol.add_atom("C", idx=atom, total_h_count=2 if atom in {0, 5} else 0 if atom in {2, 7} else 1)
    for bond_id, edge in enumerate(edges):
        mol.add_bond(*edge, idx=bond_id, order=2 if edge in doubles else 1)
    graph = FusionGraph(
        atoms=tuple(FusionGraphAtom(atom, "C") for atom in range(10)),
        bonds=tuple(FusionGraphBond(edge, "single" if edge == (2, 7) else "mancude") for edge in edges),
    )
    locants = {atom: SystemLocant(atom + 1) for atom in range(10)}
    return mol, graph, locants


def test_multiple_intrinsic_carbon_sites_preserve_joint_maximum():
    mol, graph, locants = _two_carbon_h_domains()
    original = parent_bond_model(graph)
    model, sites = intrinsic.intrinsic_carbon_parent_model(mol, graph, original, locants, frozenset({0, 5}))
    assert sites == frozenset({0, 5})
    assert model.maximum_non_cumulative_double_bonds == original.maximum_non_cumulative_double_bonds == 4
    assert all(
        order == 1
        for assignment in model.allowed_kekule_assignments
        for edge, order in assignment.orders
        if sites.intersection(edge)
    )
    state = parent_derivative_state(mol, frozenset(locants), model, locants)
    assert state is not None
    assert not state.hydro_operations
    assert not state.unsaturation_operations


def test_unproved_unpaired_carbon_cannot_be_silently_ignored():
    mol, graph, locants = _two_carbon_h_domains()
    original = parent_bond_model(graph)
    model, sites = intrinsic.intrinsic_carbon_parent_model(mol, graph, original, locants, frozenset({0}))
    assert not sites
    assert model == original


def test_carbon_site_constraint_propagates_matching_budget(monkeypatch):
    mol, graph, locants = _two_carbon_h_domains()
    original = parent_bond_model(graph)

    def exhausted(*args, **kwargs):
        raise MancudeSearchBudgetExceeded(1)

    monkeypatch.setattr(intrinsic, "_single_site_parent_model", exhausted)
    with pytest.raises(MancudeSearchBudgetExceeded):
        intrinsic.intrinsic_carbon_parent_model(mol, graph, original, locants, frozenset({0, 5}))


@pytest.mark.parametrize("substituted", (False, True))
def test_shared_lone_pair_helper_preserves_charged_oxo_derivative_donor(substituted):
    mol = Molecule()
    mol.add_atom("N", idx=0, is_aromatic=True, total_h_count=0 if substituted else 1)
    mol.add_atom("C", idx=1, charge=-1, is_aromatic=True)
    mol.add_atom("C", idx=2, is_aromatic=True)
    mol.add_atom("O", idx=3)
    mol.add_bond(0, 1, idx=0)
    mol.add_bond(0, 2, idx=1)
    mol.add_bond(1, 2, idx=2)
    mol.add_bond(2, 3, idx=3, order=2)
    if substituted:
        mol.add_atom("C", idx=4, total_h_count=3)
        mol.add_bond(0, 4, idx=4)
    graph = FusionGraph(
        atoms=(FusionGraphAtom(0, "N"), FusionGraphAtom(1, "C"), FusionGraphAtom(2, "C")),
        bonds=tuple(FusionGraphBond(edge) for edge in ((0, 1), (0, 2), (1, 2))),
    )
    assert not intrinsic.aromatic_nitrogen_hydrogen_atoms(mol, graph)
    assert intrinsic.intrinsic_parent_lone_pair_sites(mol, graph) == frozenset({0})
    mol.update_atom(0, charge=1)
    assert not intrinsic.intrinsic_parent_lone_pair_sites(mol, graph)


def test_operation_consistency_does_not_require_component_owned_hydro():
    case = _two_fused_rings()
    locants = dict(case.numbering.input_locant_maps[0])
    graph = FusionGraph(
        atoms=tuple(FusionGraphAtom(atom, "C") for atom in sorted(case.parent_atoms)),
        bonds=tuple(FusionGraphBond(bond.atoms) for bond in case.graph.bonds),
    )
    state = parent_derivative_state(case.mol, case.parent_atoms, parent_bond_model(graph), locants)
    assert state is not None and state.hydro_operations
    assert _has_consistent_derivative_operations(case.mol, case.ast, {}, case.numbering, (), state)
    operation = state.hydro_operations[0]
    duplicate_owner = replace(state, bond_delta=replace(state.bond_delta, added_hydrogen_operations=(operation,)))
    assert not _has_consistent_derivative_operations(case.mol, case.ast, {}, case.numbering, (), duplicate_owner)


def test_duplicate_oxo_is_rejected_by_independent_replay():
    case = _two_fused_rings()
    mol = case.mol
    parent_atom = 0
    oxygen = max(mol.atoms) + 1
    mol.add_atom("O", idx=oxygen)
    mol.add_bond(parent_atom, oxygen, idx=max(mol.bonds) + 1, order=2)
    state = parent_derivative_state(mol, case.parent_atoms, case.bond_model, dict(case.numbering.input_locant_maps[0]))
    assert state is not None and len(state.oxo_operations) == 1
    errors = []
    _audit_derivative_state(
        mol,
        case.parent_atoms,
        case.numbering,
        case.bond_model,
        (),
        replace(state, oxo_operations=state.oxo_operations * 2),
        errors,
    )
    assert "typed oxo operations duplicate an exocyclic parent oxo group" in errors


def test_intrinsic_carbon_scope_accepts_ortho_peri_but_requires_pi_budgets():
    case = _two_fused_rings(interface_atom_count=3)
    specs = {
        match.occurrence_id: replace(
            case.registry[match.spec_key],
            template=replace(case.registry[match.spec_key].template, mancude_double_bonds=0),
        )
        for match in case.ast.component_occurrences
    }
    assert intrinsic.intrinsic_carbon_fusion_scope(case.ast, specs)
    missing_budget = {**specs, 0: replace(specs[0], template=replace(specs[0].template, mancude_double_bonds=None))}
    assert not intrinsic.intrinsic_carbon_fusion_scope(case.ast, missing_budget)
    assert not intrinsic.intrinsic_carbon_fusion_scope(case.ast, {0: specs[0]})
