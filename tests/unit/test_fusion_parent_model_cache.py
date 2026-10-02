"""Candidate reuse must not reuse mutable graphs, policies, or search budgets."""

from dataclasses import FrozenInstanceError, replace

import pytest

from openclatura.fusion import numbering
from openclatura.fusion.model import FusionGraph, FusionGraphAtom, FusionGraphBond
from openclatura.molecule import Molecule
from openclatura.rules import elements


@pytest.fixture(autouse=True)
def empty_model_cache():
    numbering._parent_graph_bond_model.cache_clear()
    yield
    numbering._parent_graph_bond_model.cache_clear()


def _cycle(offset=0):
    return FusionGraph(
        tuple(FusionGraphAtom(offset + atom, "C") for atom in range(6)),
        tuple(FusionGraphBond((offset + atom, offset + (atom + 1) % 6)) for atom in range(6)),
    )


def test_repeated_candidate_reuses_an_immutable_model(monkeypatch):
    calls = []
    original = numbering._maximum_matchings

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(numbering, "_maximum_matchings", counted)
    first = numbering.parent_bond_model(_cycle())
    assert numbering.parent_bond_model(_cycle()) is first
    assert len(calls) == 1
    assert first.maximum_non_cumulative_double_bonds == 3
    with pytest.raises(FrozenInstanceError):
        first.maximum_non_cumulative_double_bonds = 0
    assert numbering._parent_graph_bond_model.cache_info().maxsize == 128


@pytest.mark.parametrize(
    "changes",
    (
        {"formal_charge": 1},
        {"symbol": "N"},
        {"pi_capacity": 0},
        {"forced_single": True},
        {"indicated_h_site": True},
        {"saturated": True, "pi_capacity": 0},
    ),
)
def test_every_atom_role_participates_in_the_snapshot(changes):
    graph = _cycle()
    first = numbering.parent_bond_model(graph)
    changed = replace(graph, atoms=(replace(graph.atoms[0], **changes), *graph.atoms[1:]))
    result = numbering.parent_bond_model(changed)
    assert result is not first
    assert numbering.parent_bond_model(changed) is result
    assert numbering._parent_graph_bond_model.cache_info().misses == 2


def test_relabelled_graph_keeps_its_own_atom_ids():
    original = numbering.parent_bond_model(_cycle())
    shifted = numbering.parent_bond_model(_cycle(100))
    assert shifted is not original
    assert all(
        atom >= 100
        for assignment in shifted.allowed_kekule_assignments
        for edge, _ in assignment.orders
        for atom in edge
    )


def test_molecule_updates_produce_fresh_graph_snapshots():
    mol = Molecule()
    for atom in range(4):
        mol.add_atom("C", idx=atom)
    for other in range(1, 4):
        mol.add_bond(0, other)
    neutral = numbering.parent_bond_model(mol, range(4))
    mol.update_atom(0, charge=1)
    charged = numbering.parent_bond_model(mol, range(4))
    assert charged is not neutral
    assert charged.maximum_non_cumulative_double_bonds == 0
    mol.update_atom(0, charge=0)
    assert numbering.parent_bond_model(mol, range(4)) is neutral
    mol.add_bond(1, 2)
    connected = numbering.parent_bond_model(mol, range(4))
    assert connected is not neutral
    assert connected.maximum_non_cumulative_double_bonds == 2


def test_active_element_policy_is_part_of_the_key(monkeypatch):
    graph = FusionGraph(
        (FusionGraphAtom(0, "C", formal_charge=1), *(FusionGraphAtom(atom, "C") for atom in range(1, 4))),
        tuple(FusionGraphBond((0, atom)) for atom in range(1, 4)),
    )
    original = numbering.parent_bond_model(graph)
    carbon = elements.get("C")
    changed = replace(
        carbon,
        mancude_charged_bonding_limits=tuple(
            (charge, 4 if charge == 1 else limit) for charge, limit in carbon.mancude_charged_bonding_limits
        ),
    )
    monkeypatch.setitem(elements.ELEMENTS, "C", changed)
    updated = numbering.parent_bond_model(graph)
    assert updated is not original
    assert original.maximum_non_cumulative_double_bonds == 0
    assert updated.maximum_non_cumulative_double_bonds == 1


def test_warm_cache_does_not_bypass_a_smaller_search_budget():
    graph = _cycle()
    numbering.parent_bond_model(graph, search_budget=100)
    with pytest.raises(numbering.MancudeSearchBudgetExceeded):
        numbering.parent_bond_model(graph, search_budget=1)
    assert numbering._parent_graph_bond_model.cache_info().currsize == 1
