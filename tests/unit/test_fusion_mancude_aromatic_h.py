"""Indicated aromatic nitrogen must constrain the parent pi matching."""

import random
from dataclasses import replace

import pytest
from rdkit import Chem

from openclatura import FusionMode, name_mol, opsin_available
from openclatura.fusion.mancude import (
    compare_actual_parent_to_implied_parent,
    indicated_hydrogen_parent_bond_model,
    parent_derivative_state,
)
from openclatura.fusion.model import FusionGraph, FusionGraphAtom, FusionGraphBond
from openclatura.fusion.numbering import MancudeSearchBudgetExceeded, parent_bond_model
from openclatura.molecule import Molecule


def _graph_case():
    mol = Molecule()
    symbols = ("C", "N", "C", "C", "C", "C", "C", "N", "N")
    hydrogens = (3, 0, 2, 2, 0, 0, 1, 0, 1)
    for atom, (symbol, hydrogen) in enumerate(zip(symbols, hydrogens)):
        mol.add_atom(symbol, idx=atom, total_h_count=hydrogen, is_aromatic=atom >= 4)
    edges = ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (1, 5), (4, 8))
    for bond_id, edge in enumerate(edges, start=1):
        mol.add_bond(*edge, idx=bond_id, order=2 if edge in {(4, 5), (6, 7)} else 1)
    atoms = frozenset(range(1, 9))
    graph = FusionGraph(
        atoms=tuple(FusionGraphAtom(atom, symbols[atom]) for atom in sorted(atoms)),
        bonds=tuple(FusionGraphBond(edge) for edge in edges if set(edge) <= atoms),
    )
    locants = dict(zip((8, 7, 6, 5, 1, 2, 3, 4), ("1", "2", "3", "3a", "4", "5", "6", "6a")))
    return mol, atoms, graph, locants


def test_graph_native_rematch_moves_pi_bond_off_indicated_nitrogen():
    mol, atoms, graph, locants = _graph_case()
    model = parent_bond_model(graph)
    original_assignments = model.allowed_kekule_assignments
    assert model.maximum_non_cumulative_double_bonds == 4

    constrained = indicated_hydrogen_parent_bond_model(graph, {8})
    state = parent_derivative_state(mol, atoms, constrained, locants, indicated_hydrogen_atom_ids={8})

    assert state is not None
    assert state.hydro_operations[0].locants == ("5", "6")
    assert state.hydro_operations[0].atom_ids == (2, 3)
    assert state.hydro_operations[0].bond_ids == (mol.get_bond(2, 3).idx,)
    assert not state.unsaturation_operations
    assert not state.oxo_operations
    assert state.bond_delta.hydrogenated_edges == ((2, 3),)
    assert {edge for edge, order in state.bond_delta.assignment.orders if order == 2} == {(2, 3), (4, 5), (6, 7)}
    assert state.bond_delta.implied_multiple_bond_ids == frozenset({5, 7})
    assert model.allowed_kekule_assignments == original_assignments


def test_aromatic_h_rematch_preserves_required_single_and_double_bonds():
    mol, atoms, graph, locants = _graph_case()
    graph = FusionGraph(
        atoms=graph.atoms,
        bonds=tuple(
            FusionGraphBond(
                bond.atoms, "double" if bond.atoms == (6, 7) else "single" if bond.atoms == (1, 2) else "mancude"
            )
            for bond in graph.bonds
        ),
    )
    delta = compare_actual_parent_to_implied_parent(
        mol,
        atoms,
        indicated_hydrogen_parent_bond_model(graph, {8}),
        indicated_hydrogen_atom_ids={8},
        atom_to_locant=locants,
    )
    assert delta is not None and delta.compatible
    assert dict(delta.assignment.orders)[(6, 7)] == 2
    assert dict(delta.assignment.orders)[(1, 2)] == 1
    assert delta.hydrogenated_edges == ((2, 3),)


def test_aromatic_h_conflicting_with_required_double_abstains():
    mol, atoms, graph, locants = _graph_case()
    graph = FusionGraph(
        atoms=graph.atoms,
        bonds=tuple(FusionGraphBond(b.atoms, "double" if b.atoms == (7, 8) else "mancude") for b in graph.bonds),
    )
    with pytest.raises(ValueError, match="required parent double"):
        indicated_hydrogen_parent_bond_model(graph, {8})


def test_aromatic_h_rematch_budget_exhaustion_propagates(monkeypatch):
    mol, atoms, graph, locants = _graph_case()

    def exhausted(*args, **kwargs):
        raise MancudeSearchBudgetExceeded(1)

    monkeypatch.setattr("openclatura.fusion.numbering.parent_bond_model", exhausted)
    with pytest.raises(MancudeSearchBudgetExceeded):
        indicated_hydrogen_parent_bond_model(graph, {8})


def test_intrinsic_h_constraints_preserve_original_component_roles(monkeypatch):
    _, _, graph, _ = _graph_case()
    graph = replace(
        graph,
        atoms=tuple(
            replace(atom, pi_capacity=0, saturated=True)
            if atom.id == 5
            else replace(atom, forced_single=True)
            if atom.id == 1
            else replace(atom, formal_charge=1, indicated_h_site=True)
            if atom.id == 7
            else atom
            for atom in graph.atoms
        ),
    )
    captured = []

    def capture(constrained):
        captured.append(constrained)
        return parent_bond_model(constrained)

    monkeypatch.setattr("openclatura.fusion.numbering.parent_bond_model", capture)
    model = indicated_hydrogen_parent_bond_model(graph, {8})
    assert captured[0].bonds == graph.bonds
    for original, constrained in zip(graph.atoms, captured[0].atoms):
        assert constrained == (replace(original, forced_single=True) if original.id == 8 else original)
    assert all(
        order == 1
        for assignment in model.allowed_kekule_assignments
        for edge, order in assignment.orders
        if {1, 5, 8}.intersection(edge)
    )


def test_intrinsic_h_rejects_unknown_sites():
    _, _, graph, _ = _graph_case()
    with pytest.raises(ValueError, match="outside the parent graph"):
        indicated_hydrogen_parent_bond_model(graph, {100})


def test_empty_intrinsic_h_sites_preserve_parent_model():
    _, _, graph, _ = _graph_case()
    assert indicated_hydrogen_parent_bond_model(graph, set()) == parent_bond_model(graph)


@pytest.mark.parametrize("mode", (FusionMode.GENERAL, FusionMode.AUDITED_PIN))
@pytest.mark.parametrize(
    ("smiles", "expected"),
    (
        ("CN1CCC2=C1C=NN2", "4-methyl-5,6-dihydro-1H-pyrrolo[3,2-c]pyrazole"),
        ("CCN1CCC2=C1C=NN2", "4-ethyl-5,6-dihydro-1H-pyrrolo[3,2-c]pyrazole"),
        ("CN1CC(C)C2=C1C=NN2", "4,6-dimethyl-5,6-dihydro-1H-pyrrolo[3,2-c]pyrazole"),
    ),
)
def test_aromatic_h_fusion_name_is_atom_order_invariant(mode, smiles, expected):
    mol = Chem.MolFromSmiles(smiles)
    original = list(range(mol.GetNumAtoms()))
    orders = [original, original[::-1]]
    for seed in range(6):
        order = original.copy()
        random.Random(seed).shuffle(order)
        orders.append(order)
    for order in orders:
        result = name_mol(Chem.RenumberAtoms(mol, order), fusion_mode=mode, include_trace=True)
        assert result.name == expected, (smiles, order, result.name)
        assert result.parent_nomenclature == "systematic_fusion"


@pytest.mark.skipif(not opsin_available(), reason="py2opsin/Java is unavailable")
@pytest.mark.parametrize(
    "smiles",
    (
        "CN1CCC2=C1C=NN2",
        "CN1CCc2[nH]ncc21",
        "CCN1CCC2=C1C=NN2",
        "CN1CC(C)C2=C1C=NN2",
        "CN1CCCC2=C1C=NN2",
        "N1CCC2=C1C=NN2",
        "CN1CCC2=C1C=NO2",
        "CN1CCC2=C1C=NS2",
    ),
)
def test_aromatic_h_neighborhood_round_trips(smiles):
    mol = Chem.MolFromSmiles(smiles)
    original = list(range(mol.GetNumAtoms()))
    names = []
    for order in (original, original[::-1]):
        result = name_mol(
            Chem.RenumberAtoms(mol, order),
            fusion_mode=FusionMode.GENERAL,
            verify_opsin=True,
            include_trace=True,
        )
        assert result.parent_nomenclature == "systematic_fusion"
        assert result.opsin_check is not None and result.opsin_check.ok, (smiles, order, result.opsin_check)
        names.append(result.name)
    assert names[0] == names[1]
