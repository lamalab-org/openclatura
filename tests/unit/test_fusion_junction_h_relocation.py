"""A consumed component H can relocate away from its fused pi junction."""

import pytest
from rdkit import Chem

from openclatura import name_mol, opsin_available, verify_with_opsin
from openclatura.chains import find_ring_systems
from openclatura.fusion.indicated_hydrogen import (
    component_parent_graph,
    intrinsic_carbon_candidate_atoms,
    released_fusion_carbon_sites,
)
from openclatura.fusion.model import FusionConfirmed
from openclatura.fusion.planner import plan_fusion_parent
from openclatura.fusion.registry import fusion_component_registry
from openclatura.graph_io import read_rdkit_mol


def _fused_pyran(alkyl_length, *, saturated=False, small_ring_heteroatom="O"):
    graph = Chem.RWMol()
    for symbol in ("C", "O", "C", "C", small_ring_heteroatom, "C", "C", "C", "C"):
        graph.AddAtom(Chem.Atom(symbol))
    for u, v, order in (
        (0, 1, 1),
        (1, 2, 1),
        (2, 3, 2),
        (3, 4, 1),
        (4, 5, 1),
        (5, 6, 2),
        (6, 2, 1),
        (6, 7, 1),
        (7, 8, 2),
        (8, 0, 1),
    ):
        graph.AddBond(u, v, Chem.BondType.DOUBLE if order == 2 and not saturated else Chem.BondType.SINGLE)
    last = 0
    for _ in range(alkyl_length):
        carbon = graph.AddAtom(Chem.Atom("C"))
        graph.AddBond(last, carbon, Chem.BondType.SINGLE)
        last = carbon
    mol = graph.GetMol()
    Chem.SanitizeMol(mol)
    return mol


@pytest.mark.parametrize("alkyl_length", (0, 1, 2, 4))
@pytest.mark.parametrize("ordering", ("original", "reverse", "kekule"))
def test_component_h_relocation_keeps_completed_parent_pi_budget(alkyl_length, ordering):
    rd_mol = _fused_pyran(alkyl_length)
    order = list(range(rd_mol.GetNumAtoms()))
    if ordering == "reverse":
        order.reverse()
        rd_mol = Chem.RenumberAtoms(rd_mol, order)
    elif ordering == "kekule":
        Chem.Kekulize(rd_mol, clearAromaticFlags=True)
    mol = read_rdkit_mol(rd_mol)
    atoms = max(find_ring_systems(mol), key=lambda ring: len(ring.atoms)).atoms
    result = plan_fusion_parent(mol, atoms, mode="audited_pin")
    assert isinstance(result, FusionConfirmed), result
    plan = result.plan
    assert plan.bond_model.maximum_non_cumulative_double_bonds == 3
    locants = dict(plan.numbering.input_locant_maps[0])
    assert plan.indicated_hydrogens == (locants[order.index(0)],)
    assert not plan.derivative_state.hydro_operations
    assert not plan.derivative_state.unsaturation_operations
    named = name_mol(rd_mol, include_trace=True)
    assert named.error is None
    assert named.parent_nomenclature == "systematic_fusion"
    assert "2H-furo[3,4-b]pyran" in named.name
    if opsin_available():
        check = verify_with_opsin(named.name, Chem.MolToSmiles(rd_mol), standardize_smiles=False)
        assert check.status == "matched", check.to_dict()


@pytest.mark.parametrize("small_ring_heteroatom", ("C", "O"))
@pytest.mark.parametrize("alkyl_length", (0, 2))
@pytest.mark.parametrize("reverse", (False, True))
def test_saturated_junction_uses_completed_not_isolated_component_pi_budget(
    small_ring_heteroatom, alkyl_length, reverse
):
    rd_mol = _fused_pyran(alkyl_length, saturated=True, small_ring_heteroatom=small_ring_heteroatom)
    if reverse:
        rd_mol = Chem.RenumberAtoms(rd_mol, list(reversed(range(rd_mol.GetNumAtoms()))))
    mol = read_rdkit_mol(rd_mol)
    atoms = max(find_ring_systems(mol), key=lambda ring: len(ring.atoms)).atoms
    result = plan_fusion_parent(mol, atoms, mode="audited_pin")
    assert isinstance(result, FusionConfirmed), result
    plan = result.plan
    expected_pi_count = 3 if small_ring_heteroatom == "O" else 4
    assert plan.bond_model.maximum_non_cumulative_double_bonds == expected_pi_count
    assert sum(len(operation.atom_ids) for operation in plan.derivative_state.hydro_operations) == 2 * expected_pi_count
    named = name_mol(rd_mol, include_trace=True)
    assert named.error is None
    assert named.parent_nomenclature == "systematic_fusion"
    if opsin_available():
        check = verify_with_opsin(named.name, Chem.MolToSmiles(rd_mol), standardize_smiles=False)
        assert check.status == "matched", check.to_dict()


@pytest.mark.parametrize("competing_operation", ("charge", "oxo", "missing_component_role"))
def test_hydrogenation_release_does_not_override_other_operation_roles(competing_operation):
    mol = read_rdkit_mol(_fused_pyran(0, saturated=True))
    atoms = max(find_ring_systems(mol), key=lambda ring: len(ring.atoms)).atoms
    result = plan_fusion_parent(mol, atoms, mode="audited_pin")
    assert isinstance(result, FusionConfirmed)
    plan = result.plan
    registry = fusion_component_registry()
    specs = {match.occurrence_id: registry.spec_for_match(match) for match in plan.ast.component_occurrences}
    candidates = intrinsic_carbon_candidate_atoms(plan.ast, specs, mol)
    # The unrelaxed projection, not plan.abstract_parent_graph. Since a
    # component's indicated hydrogen is released at projection (P-25.7.1.3, see
    # component_carbon_h_relocation_scope), the planner hands this molecule a
    # graph with nothing left saturated, so there is no site for this guard to
    # refuse. The state it guards is the one component_parent_graph still builds
    # on request: pyran's C-2 held saturated on the ring fusion.
    graph = component_parent_graph(plan.ast, specs, relocate_carbon_h=False)
    assert released_fusion_carbon_sites(mol, graph, candidates)
    if competing_operation == "missing_component_role":
        candidates = frozenset()
    elif competing_operation == "charge":
        mol.update_atom(0, charge=-1, total_h_count=1)
    else:
        mol.add_atom("O", idx=99)
        mol.add_bond(0, 99, order=2, idx=99)
        mol.update_atom(0, total_h_count=0)
    assert not released_fusion_carbon_sites(mol, graph, candidates)
