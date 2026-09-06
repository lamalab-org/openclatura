from openclatura.fusion.mancude import compare_actual_parent_to_implied_parent
from openclatura.fusion.model import FusionConfirmed, FusionGraph, FusionGraphAtom, FusionGraphBond, FusionMode
from openclatura.fusion.numbering import parent_bond_model
from openclatura.fusion.planner import plan_fusion_parent
from openclatura.graph_io import read_smiles


def _confirmed_plan(smiles: str):
    mol = read_smiles(smiles)
    result = plan_fusion_parent(mol, mol.atoms, mode=FusionMode.GENERAL)
    assert isinstance(result, FusionConfirmed)
    return mol, result.plan


def test_mancude_delta_accepts_an_exact_kekule_form():
    mol, plan = _confirmed_plan("O1C2=C(C=C1)C=CS2")

    delta = compare_actual_parent_to_implied_parent(mol, set(mol.atoms), plan.bond_model)

    assert delta is not None and delta.compatible
    assert not delta.hydrogenated_edges
    assert not delta.additional_multiple_bond_ids
    assert len(delta.implied_multiple_bond_ids) == 3


def test_mancude_delta_identifies_hydrogenation_without_restating_remaining_double_bonds():
    mol, plan = _confirmed_plan("O1C2=C(C=C1)CCS2")

    delta = compare_actual_parent_to_implied_parent(mol, set(mol.atoms), plan.bond_model)

    assert delta is not None and delta.compatible
    assert len(delta.hydrogenated_edges) == 1
    assert delta.hydrogenated_atom_ids == frozenset({5, 6})
    assert len(delta.implied_multiple_bond_ids) == 2
    assert not delta.additional_multiple_bond_ids


def test_mancude_delta_keeps_additional_multiple_bonds_outside_the_implied_set():
    mol, plan = _confirmed_plan("O1C2=C(C=C1)C=CS2")
    bond_id = next(
        bond.idx
        for bond in mol.bonds.values()
        if tuple(sorted((bond.u, bond.v))) in plan.bond_model.required_single_bonds
    )
    mol.update_bond(bond_id, order=2)

    delta = compare_actual_parent_to_implied_parent(mol, set(mol.atoms), plan.bond_model)

    assert delta is not None and delta.compatible
    assert bond_id in delta.additional_multiple_bond_ids
    assert bond_id not in delta.implied_multiple_bond_ids


def test_exact_template_double_bond_is_required_and_reserves_its_endpoints():
    graph = FusionGraph(
        atoms=tuple(FusionGraphAtom(atom, "C") for atom in range(4)),
        bonds=(
            FusionGraphBond((0, 1), "double"),
            FusionGraphBond((1, 2), "mancude"),
            FusionGraphBond((2, 3), "mancude"),
            FusionGraphBond((0, 3), "mancude"),
        ),
    )

    model = parent_bond_model(graph)

    assert model.required_double_bonds == frozenset({(0, 1)})
    assert model.pi_eligible_edges == frozenset({(2, 3)})
    assert all(dict(assignment.orders)[(0, 1)] == 2 for assignment in model.allowed_kekule_assignments)
    assert model.maximum_non_cumulative_double_bonds == 2
