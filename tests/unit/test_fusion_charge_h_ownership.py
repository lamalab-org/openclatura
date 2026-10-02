"""Charge-owned nitrogen H must not hide an intrinsic carbon H site."""

import pytest
from rdkit import Chem

from openclatura import name_mol, opsin_available, verify_with_opsin
from openclatura.chains import find_ring_systems
from openclatura.fusion.charges import protonated_pi_nitrogen_atoms
from openclatura.fusion.model import FusionConfirmed
from openclatura.fusion.planner import plan_fusion_parent
from openclatura.graph_io import read_rdkit_mol


def _charged_fused_graph(extra_nitrogen, ligand_length):
    graph = Chem.RWMol()
    for index, symbol in enumerate(("C", "N", "C", "C", "C", "N" if extra_nitrogen else "C", "N", "N", "N")):
        atom = Chem.Atom(symbol)
        if index == 1:
            atom.SetFormalCharge(1)
            atom.SetNumExplicitHs(1)
        elif index == 7:
            atom.SetFormalCharge(-1)
        graph.AddAtom(atom)
    for u, v, order in (
        (0, 1, 2),
        (1, 2, 1),
        (2, 3, 2),
        (3, 4, 1),
        (4, 0, 1),
        (3, 5, 1),
        (5, 6, 2),
        (6, 7, 1),
        (7, 2, 1),
        (0, 8, 1),
    ):
        graph.AddBond(u, v, Chem.BondType.DOUBLE if order == 2 else Chem.BondType.SINGLE)
    last = 8
    for _ in range(ligand_length):
        carbon = graph.AddAtom(Chem.Atom("C"))
        graph.AddBond(last, carbon, Chem.BondType.SINGLE)
        last = carbon
    mol = graph.GetMol()
    Chem.SanitizeMol(mol)
    return mol


@pytest.mark.parametrize("extra_nitrogen", (False, True))
@pytest.mark.parametrize("ligand_length", (0, 1, 2))
@pytest.mark.parametrize("reverse", (False, True))
def test_charge_operations_leave_carbon_indicated_h_independent(extra_nitrogen, ligand_length, reverse):
    rd_mol = _charged_fused_graph(extra_nitrogen, ligand_length)
    order = list(range(rd_mol.GetNumAtoms()))
    if reverse:
        order.reverse()
        rd_mol = Chem.RenumberAtoms(rd_mol, order)
    mol = read_rdkit_mol(rd_mol)
    atoms = max(find_ring_systems(mol), key=lambda ring: len(ring.atoms)).atoms
    result = plan_fusion_parent(mol, atoms, mode="audited_pin")
    assert isinstance(result, FusionConfirmed), result
    plan = result.plan
    assert not plan.derivative_state.hydro_operations
    locants = dict(plan.numbering.input_locant_maps[0])
    assert locants[order.index(4)] in plan.indicated_hydrogens
    assert protonated_pi_nitrogen_atoms(mol, plan.abstract_parent_graph) == {order.index(1)}
    assert {operation.atom_id for operation in plan.charge_operations} == {order.index(1), order.index(7)}
    named = name_mol(rd_mol, include_trace=True)
    assert named.error is None
    assert named.parent_nomenclature == "systematic_fusion"
    if opsin_available():
        check = verify_with_opsin(named.name, Chem.MolToSmiles(rd_mol), standardize_smiles=False)
        assert check.status == "matched", check.to_dict()


@pytest.mark.parametrize("changes", ({"charge": 0}, {"charge": 2}, {"total_h_count": 0}, {"total_h_count": 2}))
def test_protonation_role_requires_the_explicit_single_proton_cation(changes):
    mol = read_rdkit_mol(_charged_fused_graph(False, 0))
    atoms = max(find_ring_systems(mol), key=lambda ring: len(ring.atoms)).atoms
    result = plan_fusion_parent(mol, atoms, mode="audited_pin")
    assert isinstance(result, FusionConfirmed)
    graph = result.plan.abstract_parent_graph
    assert protonated_pi_nitrogen_atoms(mol, graph) == {1}
    mol.update_atom(1, **changes)
    assert not protonated_pi_nitrogen_atoms(mol, graph)
