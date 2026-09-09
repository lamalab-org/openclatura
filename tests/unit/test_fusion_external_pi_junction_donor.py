"""Junction donors and external pi bonds require one composed parent witness."""

import pytest
from rdkit import Chem

from openclatura import name_mol, opsin_available, verify_with_opsin
from openclatura.chains import find_ring_systems
from openclatura.fusion.model import FusionConfirmed
from openclatura.fusion.planner import plan_fusion_parent
from openclatura.graph_io import read_rdkit_mol


@pytest.mark.parametrize("external", ("O", "N", "C"))
@pytest.mark.parametrize("arene_atom", ("C", "N"))
@pytest.mark.parametrize("side_length", (0, 2))
@pytest.mark.parametrize("reverse", (False, True))
def test_joint_donor_pi_composition_keeps_hydrogenation_off_junction_n(external, arene_atom, side_length, reverse):
    graph = Chem.RWMol()
    for symbol in (external, "C", "C", arene_atom, "C", "C", "C", "C", "N", "C", "N", "C", "C", "C"):
        graph.AddAtom(Chem.Atom(symbol))
    doubles = {(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)}
    edges = tuple((i, i + 1) for i in range(13)) + ((1, 10), (2, 7), (9, 13))
    for edge in edges:
        graph.AddBond(*edge, Chem.BondType.DOUBLE if edge in doubles else Chem.BondType.SINGLE)
    last = 12
    for _ in range(side_length):
        atom = graph.AddAtom(Chem.Atom("C"))
        graph.AddBond(last, atom, Chem.BondType.SINGLE)
        last = atom
    rd_mol = graph.GetMol()
    Chem.SanitizeMol(rd_mol)
    order = list(range(rd_mol.GetNumAtoms()))
    if reverse:
        order.reverse()
        rd_mol = Chem.RenumberAtoms(rd_mol, order)
    mol = read_rdkit_mol(rd_mol)
    atoms = max(find_ring_systems(mol), key=lambda ring: len(ring.atoms)).atoms
    planned = plan_fusion_parent(mol, atoms, mode="audited_pin")
    assert isinstance(planned, FusionConfirmed), planned
    state = planned.plan.derivative_state
    assert state.hydro_operations
    assert state.added_hydrogen_operations
    for operation in state.hydro_operations + state.added_hydrogen_operations:
        assert order.index(10) not in operation.atom_ids
    named = name_mol(rd_mol, include_trace=True)
    assert named.error is None
    assert named.parent_nomenclature == "systematic_fusion"
    if opsin_available():
        check = verify_with_opsin(named.name, Chem.MolToSmiles(rd_mol), standardize_smiles=False)
        assert check.status == "matched", check.to_dict()
