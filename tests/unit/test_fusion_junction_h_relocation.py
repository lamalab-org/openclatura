"""A consumed component H can relocate away from its fused pi junction."""

import pytest
from rdkit import Chem

from openclatura import name_mol, opsin_available, verify_with_opsin
from openclatura.chains import find_ring_systems
from openclatura.fusion.model import FusionConfirmed
from openclatura.fusion.planner import plan_fusion_parent
from openclatura.graph_io import read_rdkit_mol


def _fused_pyran(alkyl_length):
    graph = Chem.RWMol()
    for symbol in ("C", "O", "C", "C", "O", "C", "C", "C", "C"):
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
        graph.AddBond(u, v, Chem.BondType.DOUBLE if order == 2 else Chem.BondType.SINGLE)
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
