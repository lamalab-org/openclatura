"""Graph-local nitrogen composition constraints and bounded reassignment."""

from dataclasses import replace

from openclatura import FusionMode
from openclatura.fusion.mancude import _single_site_parent_model, compare_actual_parent_to_implied_parent
from openclatura.fusion.model import FusionGraph, FusionGraphAtom, FusionGraphBond, FusionUnsupported
from openclatura.fusion.numbering import MancudeSearchBudgetExceeded, parent_bond_model
from openclatura.fusion.planner import plan_fusion_parent
from openclatura.graph_io import read_smiles


def _mancude_model(mol):
    return parent_bond_model(
        FusionGraph(
            atoms=tuple(FusionGraphAtom(atom.idx, atom.symbol) for atom in mol.atoms.values()),
            bonds=tuple(FusionGraphBond(tuple(sorted((bond.u, bond.v))), "mancude") for bond in mol.bonds.values()),
        )
    )


def test_five_membered_indicated_nh_keeps_both_carbon_hydro_pairs():
    mol = read_smiles("N1CCCC1")
    model = _mancude_model(mol)
    delta = compare_actual_parent_to_implied_parent(mol, mol.atoms, model, indicated_hydrogen_atom_ids={0})
    assert delta.hydrogenated_atom_ids == frozenset({1, 2, 3, 4})
    assert len(delta.hydrogenated_edges) == 2
    assert all(order == 1 for edge, order in delta.assignment.orders if 0 in edge)


def test_six_membered_indicated_nh_preserves_existing_additive_state():
    mol = read_smiles("N1CCCCC1")
    model = _mancude_model(mol)
    delta = compare_actual_parent_to_implied_parent(mol, mol.atoms, model, indicated_hydrogen_atom_ids={0})
    assert delta.assignment in model.allowed_kekule_assignments
    assert len(delta.hydrogenated_edges) == 2


def test_fusion_nitrogen_charge_is_not_treated_as_neutral_lone_pair():
    mol = read_smiles("N12CCNCC1CCNC2")
    model = _mancude_model(mol)
    neutral = compare_actual_parent_to_implied_parent(mol, mol.atoms, model)
    assert len(neutral.hydrogenated_atom_ids) == 8
    assert 0 not in neutral.hydrogenated_atom_ids
    mol.atoms[0] = replace(mol.atoms[0], charge=1)
    charged = compare_actual_parent_to_implied_parent(mol, mol.atoms, model)
    assert len(charged.hydrogenated_atom_ids) == 10
    assert 0 in charged.hydrogenated_atom_ids


def test_single_site_constraint_cannot_override_fixed_component_double():
    mol = read_smiles("N1CCCC1")
    graph = FusionGraph(
        atoms=tuple(FusionGraphAtom(atom.idx, atom.symbol) for atom in mol.atoms.values()),
        bonds=tuple(
            FusionGraphBond(tuple(sorted((bond.u, bond.v))), "double" if bond.u == 0 and bond.v == 1 else "mancude")
            for bond in mol.bonds.values()
        ),
    )
    model = parent_bond_model(graph)
    assert compare_actual_parent_to_implied_parent(mol, mol.atoms, model, indicated_hydrogen_atom_ids={0}) is None


def test_derivative_reassignment_uses_bounded_cache():
    model = _mancude_model(read_smiles("N1CCCC1"))
    _single_site_parent_model.cache_clear()
    first = _single_site_parent_model(model, frozenset({0}))
    assert _single_site_parent_model(model, frozenset({0})) is first
    assert _single_site_parent_model.cache_info().hits == 1
    assert _single_site_parent_model.cache_info().maxsize == 256


def test_derivative_search_exhaustion_is_typed_abstention(monkeypatch):
    def exhausted(*args, **kwargs):
        raise MancudeSearchBudgetExceeded(1)

    monkeypatch.setattr("openclatura.fusion.planner.parent_derivative_state", exhausted)
    mol = read_smiles("O1C2=C(C=C1)CCS2")
    result = plan_fusion_parent(mol, mol.atoms, mode=FusionMode.GENERAL)
    assert isinstance(result, FusionUnsupported)
    assert result.reason == "derivative parent assignment search budget exhausted"
    assert any("budget of 1 states" in detail for detail in result.details)
