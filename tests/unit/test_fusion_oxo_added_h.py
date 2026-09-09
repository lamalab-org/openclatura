"""Oxo substitution can relocate a parent pi bond and leave carbon added H."""

import pytest
from rdkit import Chem

from openclatura import name_mol, opsin_available, verify_with_opsin
from openclatura.chains import find_ring_systems
from openclatura.fusion.model import FusionConfirmed
from openclatura.fusion.planner import plan_fusion_parent
from openclatura.graph_io import read_rdkit_mol


def _fused_oxo_graph(heteroatom, alkyl_length):
    graph = Chem.RWMol()
    for symbol in ("C", heteroatom, "C", "C", "C", "C", "C", "N", "O"):
        graph.AddAtom(Chem.Atom(symbol))
    for u, v, order in (
        (0, 1, 1),
        (1, 2, 1),
        (2, 3, 1),
        (3, 4, 2),
        (4, 0, 1),
        (4, 5, 1),
        (5, 6, 2),
        (6, 7, 1),
        (7, 3, 1),
        (0, 8, 2),
    ):
        graph.AddBond(u, v, Chem.BondType.DOUBLE if order == 2 else Chem.BondType.SINGLE)
    last = 7
    for _ in range(alkyl_length):
        carbon = graph.AddAtom(Chem.Atom("C"))
        graph.AddBond(last, carbon, Chem.BondType.SINGLE)
        last = carbon
    mol = graph.GetMol()
    Chem.SanitizeMol(mol)
    return mol


@pytest.mark.parametrize("heteroatom", ("O", "N"))
@pytest.mark.parametrize("alkyl_length", (0, 1, 2))
@pytest.mark.parametrize("reverse", (False, True))
def test_oxo_and_donor_constraints_emit_added_h_not_junction_hydro(heteroatom, alkyl_length, reverse):
    rd_mol = _fused_oxo_graph(heteroatom, alkyl_length)
    order = list(range(rd_mol.GetNumAtoms()))
    if reverse:
        order.reverse()
        rd_mol = Chem.RenumberAtoms(rd_mol, order)
    mol = read_rdkit_mol(rd_mol)
    atoms = max(find_ring_systems(mol), key=lambda ring: len(ring.atoms)).atoms
    result = plan_fusion_parent(mol, atoms, mode="audited_pin")
    assert isinstance(result, FusionConfirmed), result
    state = result.plan.derivative_state
    assert not state.hydro_operations
    assert not state.unsaturation_operations
    assert len(state.oxo_operations) == 1
    assert {atom for operation in state.added_hydrogen_operations for atom in operation.atom_ids} == {order.index(2)}
    named = name_mol(rd_mol, include_trace=True)
    assert named.error is None
    assert named.parent_nomenclature == "systematic_fusion"
    if opsin_available():
        check = verify_with_opsin(named.name, Chem.MolToSmiles(rd_mol), standardize_smiles=False)
        assert check.status == "matched", check.to_dict()
