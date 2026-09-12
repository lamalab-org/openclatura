"""Retained bridge parents preserve H counts while internal pi bonds move."""

import pytest
from rdkit import Chem

from openclatura import name_mol, opsin_available, verify_with_opsin
from openclatura.chains import find_ring_systems
from openclatura.fusion.wrappers import plan_bridged_fusion_wrapper
from openclatura.graph_io import read_rdkit_mol


def _bridged_pentalene(bridge_element, ligand_length):
    graph = Chem.RWMol()
    for symbol in ("C",) * 8 + (bridge_element,):
        graph.AddAtom(Chem.Atom(symbol))
    for u, v in (
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 0),
        (3, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 8),
        (8, 3),
    ):
        graph.AddBond(u, v, Chem.BondType.DOUBLE if (u, v) in {(1, 2), (5, 6)} else Chem.BondType.SINGLE)
    last = 1
    for _ in range(ligand_length):
        carbon = graph.AddAtom(Chem.Atom("C"))
        graph.AddBond(last, carbon, Chem.BondType.SINGLE)
        last = carbon
    mol = graph.GetMol()
    Chem.SanitizeMol(mol)
    return mol


@pytest.mark.parametrize("bridge_element", ("C", "O"))
@pytest.mark.parametrize("ligand_length", (0, 1, 2))
@pytest.mark.parametrize("reverse", (False, True))
def test_retained_bridge_parent_uses_proved_hydrogen_endpoints(bridge_element, ligand_length, reverse):
    rd_mol = _bridged_pentalene(bridge_element, ligand_length)
    if reverse:
        rd_mol = Chem.RenumberAtoms(rd_mol, list(reversed(range(rd_mol.GetNumAtoms()))))
    mol = read_rdkit_mol(rd_mol)
    atoms = max(find_ring_systems(mol), key=lambda ring: len(ring.atoms)).atoms
    plan = plan_bridged_fusion_wrapper(mol, atoms, mode="audited_pin")
    assert plan is not None
    assert plan.parent.fusion_plan is None
    state = plan.derivative_state
    assert state.pi_redistribution is not None
    assert len(state.pi_redistribution.hydrogenated_atom_ids) == 4
    assert sum(len(operation.atom_ids) for operation in state.hydro_operations) == 4
    assert not state.unsaturation_operations
    named = name_mol(rd_mol, include_trace=True)
    assert named.error is None
    assert named.parent_nomenclature == "bridged_fusion"
    assert "pentalene" in named.name
    if opsin_available():
        check = verify_with_opsin(named.name, Chem.MolToSmiles(rd_mol), standardize_smiles=False)
        assert check.status == "matched", check.to_dict()
