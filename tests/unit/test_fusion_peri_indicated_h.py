"""P-58.2.1.1 requires explicit indicated H even when OPSIN accepts omission.

Rule source: https://iupac.qmul.ac.uk/BlueBook/P5.html
"""

import pytest
from rdkit import Chem

from openclatura import name_mol, opsin_available, verify_with_opsin
from openclatura.chains import find_ring_systems
from openclatura.fusion.model import FusionConfirmed
from openclatura.fusion.planner import plan_fusion_parent
from openclatura.graph_io import read_rdkit_mol


@pytest.mark.parametrize("element", ("O", "N", "S"))
@pytest.mark.parametrize("methyl", (False, True))
@pytest.mark.parametrize("reverse", (False, True))
def test_peri_carbon_indicated_h_is_graph_owned(element, methyl, reverse):
    graph = Chem.RWMol()
    for symbol in (element,) + ("C",) * 11:
        graph.AddAtom(Chem.Atom(symbol))
    edges = tuple((i, i + 1) for i in range(11)) + ((6, 11), (2, 7), (0, 8))
    for edge in edges:
        graph.AddBond(
            *edge, Chem.BondType.DOUBLE if edge in {(2, 3), (4, 5), (6, 7), (8, 9), (10, 11)} else Chem.BondType.SINGLE
        )
    if methyl:
        branch = graph.AddAtom(Chem.Atom("C"))
        graph.AddBond(3, branch, Chem.BondType.SINGLE)
    rd_mol = graph.GetMol()
    Chem.SanitizeMol(rd_mol)
    order = list(range(rd_mol.GetNumAtoms()))
    if reverse:
        order.reverse()
        rd_mol = Chem.RenumberAtoms(rd_mol, order)
    mol = read_rdkit_mol(rd_mol)
    atoms = max(find_ring_systems(mol), key=lambda ring: len(ring.atoms)).atoms
    result = plan_fusion_parent(mol, atoms, mode="audited_pin")
    assert isinstance(result, FusionConfirmed), result
    plan = result.plan
    carbon = order.index(1)
    locants = dict(plan.numbering.input_locant_maps[0])
    assert str(locants[carbon]) == "2"
    assert locants[carbon] in plan.indicated_hydrogens
    assert mol.atoms[carbon].total_h_count == 2
    assert all(mol.get_bond(carbon, neighbor).order == 1 for neighbor in mol.get_neighbors(carbon))
    assert all(
        bond_order == 1 for edge, bond_order in plan.derivative_state.bond_delta.assignment.orders if carbon in edge
    )
    named = name_mol(rd_mol, include_trace=True)
    assert named.error is None
    assert named.parent_nomenclature == "systematic_fusion"
    if opsin_available():
        check = verify_with_opsin(named.name, Chem.MolToSmiles(rd_mol), standardize_smiles=False)
        assert check.status == "matched", check.to_dict()
