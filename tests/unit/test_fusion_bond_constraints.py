"""Adversarial parent-hydride valence constraints, independent of naming engines."""

from dataclasses import replace

import pytest

from openclatura.fusion.model import BondAssignment, FusionGraph, FusionGraphAtom, FusionGraphBond
from openclatura.fusion.numbering import parent_bond_model, validate_parent_bond_valence
from openclatura.molecule import Molecule
from openclatura.rules import elements


def _star(symbol="C", degree=4, charge=0, required_double=False):
    return FusionGraph(
        atoms=(FusionGraphAtom(0, symbol, charge),)
        + tuple(FusionGraphAtom(atom, "C") for atom in range(1, degree + 1)),
        bonds=tuple(
            FusionGraphBond((0, atom), "double" if required_double and atom == 1 else "mancude")
            for atom in range(1, degree + 1)
        ),
    )


def _molecule(graph):
    mol = Molecule()
    for atom in graph.atoms:
        mol.add_atom(atom.symbol, idx=atom.id, charge=atom.formal_charge)
    for bond in graph.bonds:
        mol.add_bond(*bond.atoms, order=2 if bond.bond_class == "double" else 1)
    return mol


def _baseline(graph):
    return BondAssignment(tuple((bond.atoms, 2 if bond.bond_class == "double" else 1) for bond in graph.bonds))


@pytest.mark.parametrize("compatibility", (False, True))
@pytest.mark.parametrize("degree, maximum", ((3, 1), (4, 0)))
def test_carbon_capacity_is_shared_by_graph_and_molecule(compatibility, degree, maximum):
    graph = _star(degree=degree)
    parent = _molecule(graph) if compatibility else graph
    ids = (atom.id for atom in graph.atoms) if compatibility else None
    model = parent_bond_model(parent, ids)
    assert model.maximum_non_cumulative_double_bonds == maximum
    for assignment in model.allowed_kekule_assignments:
        assert sum(order for edge, order in assignment.orders if 0 in edge) <= 4
        validate_parent_bond_valence(parent, assignment, range(degree + 1) if compatibility else None)


@pytest.mark.parametrize("compatibility", (False, True))
def test_overloaded_single_bond_skeleton_is_rejected_before_search(compatibility, monkeypatch):
    graph = _star(degree=5)

    def unexpected_search(*args, **kwargs):
        pytest.fail("an infeasible mandatory skeleton must not enter matching search")

    monkeypatch.setattr("openclatura.fusion.numbering._maximum_matchings", unexpected_search)
    with pytest.raises(ValueError, match="bond-order load 5 above fixed-valence limit 4"):
        parent_bond_model(_molecule(graph) if compatibility else graph, range(6) if compatibility else None)


def test_required_double_cannot_bypass_saturated_carbon_guard(monkeypatch):
    graph = _star(degree=4, required_double=True)

    def unexpected_search(*args, **kwargs):
        pytest.fail("an overloaded required double must not enter matching search")

    monkeypatch.setattr("openclatura.fusion.numbering._maximum_matchings", unexpected_search)
    with pytest.raises(ValueError, match="bond-order load 5 above fixed-valence limit 4"):
        parent_bond_model(graph)
    with pytest.raises(ValueError, match="fixed-valence limit 4"):
        validate_parent_bond_valence(graph, _baseline(graph))


def test_feasible_required_double_reserves_endpoints_without_losing_remote_pi_bond():
    star = _star(degree=3, required_double=True)
    graph = replace(star, bonds=star.bonds + (FusionGraphBond((2, 3)),))
    model = parent_bond_model(graph)
    assert model.required_double_bonds == frozenset({(0, 1)})
    assert model.pi_eligible_edges == frozenset({(2, 3)})
    assert model.maximum_non_cumulative_double_bonds == 2
    for assignment in model.allowed_kekule_assignments:
        validate_parent_bond_valence(graph, assignment)


@pytest.mark.parametrize("charge", (-1, 1))
@pytest.mark.parametrize("compatibility", (False, True))
def test_charged_carbon_does_not_inherit_neutral_capacity(charge, compatibility):
    graph = _star(degree=3, charge=charge)
    model = parent_bond_model(_molecule(graph) if compatibility else graph, range(4) if compatibility else None)
    assert model.maximum_non_cumulative_double_bonds == 0
    assert not model.pi_eligible_edges


@pytest.mark.parametrize("charge", (-1, 1))
def test_required_double_load_is_charge_sensitive(charge):
    with pytest.raises(ValueError, match="bond-order load 4 above fixed-valence limit 3"):
        parent_bond_model(_star(degree=3, charge=charge, required_double=True))


@pytest.mark.parametrize(
    "symbol, charge, limit",
    (
        ("B", 0, 3),
        ("B", -1, 4),
        ("B", 1, 2),
        ("C", 0, 4),
        ("C", -1, 3),
        ("C", 1, 3),
        ("O", 0, 2),
        ("O", -1, 1),
        ("O", 1, 3),
        ("F", 0, 1),
        ("F", -1, 0),
        ("F", 1, 2),
    ),
)
@pytest.mark.parametrize("compatibility", (False, True))
def test_explicit_fixed_valence_limits_include_charged_states(symbol, charge, limit, compatibility):
    graph = _star(symbol, degree=limit, charge=charge)
    parent = _molecule(graph) if compatibility else graph
    ids = range(limit + 1) if compatibility else None
    validate_parent_bond_valence(parent, _baseline(graph), ids)
    assert parent_bond_model(parent, ids).maximum_non_cumulative_double_bonds == 0

    overloaded = _star(symbol, degree=limit + 1, charge=charge)
    with pytest.raises(ValueError, match="above fixed-valence limit"):
        parent_bond_model(
            _molecule(overloaded) if compatibility else overloaded,
            range(limit + 2) if compatibility else None,
        )


@pytest.mark.parametrize("compatibility", (False, True))
def test_unknown_fixed_valence_charge_fails_closed(compatibility):
    graph = _star(charge=2, degree=2)
    with pytest.raises(ValueError, match="unsupported fixed-valence charge"):
        parent_bond_model(_molecule(graph) if compatibility else graph, range(3) if compatibility else None)


@pytest.mark.parametrize("charge", (-1, 0, 1))
@pytest.mark.parametrize("compatibility", (False, True))
def test_nitrogen_retains_separate_donor_and_charge_handling(charge, compatibility):
    graph = _star("N", degree=3, charge=charge)
    model = parent_bond_model(_molecule(graph) if compatibility else graph, range(4) if compatibility else None)
    assert model.maximum_non_cumulative_double_bonds == 1
    assert elements.get("N").mancude_limit_for_charge(charge) is None


@pytest.mark.parametrize(
    "site_changes", ({"forced_single": True}, {"pi_capacity": 0}, {"pi_capacity": 0, "saturated": True})
)
def test_explicit_nitrogen_donor_constraints_are_preserved(site_changes):
    graph = _star("N", degree=3)
    graph = replace(graph, atoms=(replace(graph.atoms[0], **site_changes),) + graph.atoms[1:])
    assert parent_bond_model(graph).maximum_non_cumulative_double_bonds == 0


@pytest.mark.parametrize("symbol, degree", (("N", 3), ("P", 4), ("S", 5), ("Se", 5)))
def test_required_bond_load_does_not_replace_lambda_or_nitrogen_proofs(symbol, degree):
    graph = _star(symbol, degree=degree, required_double=True)
    model = parent_bond_model(graph)
    assert model.required_double_bonds == frozenset({(0, 1)})
    for assignment in model.allowed_kekule_assignments:
        validate_parent_bond_valence(graph, assignment)


@pytest.mark.parametrize("compatibility", (False, True))
def test_auditor_helper_rejects_forged_overvalent_assignment(compatibility):
    graph = _star(degree=4)
    assignment = BondAssignment(tuple((bond.atoms, 2 if index == 0 else 1) for index, bond in enumerate(graph.bonds)))
    with pytest.raises(ValueError, match="parent atom 0.*bond-order load 5"):
        validate_parent_bond_valence(
            _molecule(graph) if compatibility else graph, assignment, range(5) if compatibility else None
        )


@pytest.mark.parametrize("orders", ((), (((0, 99), 1),), (((0, 1), 3),), (((0, 1), 1.5),)))
def test_auditor_helper_rejects_incomplete_foreign_or_non_kekule_orders(orders):
    with pytest.raises(ValueError, match="parent valence assignment"):
        validate_parent_bond_valence(_star(degree=1), BondAssignment(orders))


def test_validation_normalizes_edge_orientation_and_keeps_parent_namespace():
    graph = FusionGraph(
        atoms=(FusionGraphAtom(8, "C"), FusionGraphAtom(42, "C")),
        bonds=(FusionGraphBond((42, 8), "double"),),
    )
    validate_parent_bond_valence(graph, BondAssignment((((8, 42), 2),)))


def test_compatibility_path_uses_parent_subset_not_observed_hydrogen_or_external_loads():
    graph = _star(degree=3)
    mol = _molecule(graph)
    mol.update_atom(0, total_h_count=1)
    mol.add_atom("O", idx=10)
    mol.add_bond(0, 10, order=2)
    model = parent_bond_model(mol, iter(range(4)))
    assert model == parent_bond_model(graph)
    for assignment in model.allowed_kekule_assignments:
        validate_parent_bond_valence(mol, assignment, iter(range(4)))


def test_compatibility_observed_double_does_not_become_a_required_parent_double():
    graph = _star(degree=3, required_double=True)
    model = parent_bond_model(_molecule(graph), range(4))
    assert not model.required_double_bonds
    assert len(model.allowed_kekule_assignments) == 3


def test_molecule_validation_requires_explicit_parent_atom_ids():
    mol = _molecule(_star(degree=1))
    with pytest.raises(TypeError, match="atom_ids are required"):
        validate_parent_bond_valence(mol, BondAssignment((((0, 1), 1),)))
