"""Explicit donor H and neutral junctions compose before hydro assignment."""

import pytest
from rdkit import Chem

from openclatura import name_mol, opsin_available, verify_with_opsin
from openclatura.chains import find_ring_systems
from openclatura.fusion.mancude import _nitrogen_composition_parent_model
from openclatura.fusion.model import FusionGraph, FusionGraphAtom, FusionGraphBond
from openclatura.fusion.numbering import parent_bond_model
from openclatura.fusion.planner import plan_fusion_parent
from openclatura.graph_io import read_rdkit_mol


def _graph(symbols, edges, doubles=(), ligand_length=0, attachment=0):
    graph = Chem.RWMol()
    for symbol in symbols:
        graph.AddAtom(Chem.Atom(symbol))
    doubles = {frozenset(edge) for edge in doubles}
    for u, v in edges:
        graph.AddBond(u, v, Chem.BondType.DOUBLE if frozenset((u, v)) in doubles else Chem.BondType.SINGLE)
    for _ in range(ligand_length):
        carbon = graph.AddAtom(Chem.Atom("C"))
        graph.AddBond(attachment, carbon, Chem.BondType.SINGLE)
        attachment = carbon
    result = graph.GetMol()
    Chem.SanitizeMol(result)
    return result


@pytest.mark.parametrize("size", range(3, 9))
def test_cited_nitrogen_sigma_valence_is_independent_of_ring_size(size):
    edges = tuple((atom, (atom + 1) % size) for atom in range(size))
    mol = read_rdkit_mol(_graph(("N",) + ("C",) * (size - 1), edges))
    graph = FusionGraph(
        tuple(FusionGraphAtom(atom, mol.atoms[atom].symbol) for atom in range(size)),
        tuple(FusionGraphBond(tuple(sorted(edge)), "mancude") for edge in edges),
    )
    model = _nitrogen_composition_parent_model(mol, frozenset(range(size)), parent_bond_model(graph), {0})
    assert model.maximum_non_cumulative_double_bonds == (size - 1) // 2
    assert all(
        order == 1 for assignment in model.allowed_kekule_assignments for edge, order in assignment.orders if 0 in edge
    )


@pytest.mark.parametrize("terminal", (("N", "C", "C"), ("O", "C", "C"), ("N", "N", "C")))
@pytest.mark.parametrize("ligand_length", (0, 1, 2))
@pytest.mark.parametrize("reverse", (False, True))
def test_cited_small_ring_nh_keeps_fused_branch_hydrogenation(terminal, ligand_length, reverse):
    mol = _graph(
        ("C", "C", "N", "C", "C", "C") + terminal,
        ((0, 1), (0, 5), (1, 2), (1, 3), (2, 3), (3, 4), (4, 5), (4, 8), (5, 6), (6, 7), (7, 8)),
        ((4, 5), (7, 8)),
        ligand_length,
    )
    _assert_fusion_roundtrip(mol, reverse)


@pytest.mark.parametrize("heteroatom", ("N", "O"))
@pytest.mark.parametrize("ligand_length", (0, 1, 2))
@pytest.mark.parametrize("reverse", (False, True))
def test_four_connected_carbon_is_a_single_bond_spectator(heteroatom, ligand_length, reverse):
    mol = _graph(
        ("C", "O", "C", "C", heteroatom, "C", "C", "C", "C"),
        ((0, 1), (0, 8), (1, 2), (2, 3), (3, 4), (3, 5), (3, 8), (4, 5), (5, 6), (6, 7), (7, 8)),
        ((6, 7),),
        ligand_length,
    )
    _assert_fusion_roundtrip(mol, reverse)


def _assert_fusion_roundtrip(rd_mol, reverse):
    if reverse:
        rd_mol = Chem.RenumberAtoms(rd_mol, list(reversed(range(rd_mol.GetNumAtoms()))))
    mol = read_rdkit_mol(rd_mol)
    atoms = max(find_ring_systems(mol), key=lambda ring: len(ring.atoms)).atoms
    plan = plan_fusion_parent(mol, atoms, mode="audited_pin").plan
    assert plan.audit.status.value == "confirmed"
    assert plan.derivative_state.unsaturation_operations == ()
    assert all(mol.atoms[atom].symbol == "C" for op in plan.derivative_state.hydro_operations for atom in op.atom_ids)
    result = name_mol(rd_mol, include_trace=True)
    assert result.error is None
    assert result.parent_nomenclature == "systematic_fusion"
    if opsin_available():
        check = verify_with_opsin(result.name, Chem.MolToSmiles(rd_mol), standardize_smiles=False)
        assert check.status == "matched", check.to_dict()
