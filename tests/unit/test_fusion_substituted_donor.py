"""Replacing donor H with an alkyl ligand does not add a parent pi bond."""

import pytest
from rdkit import Chem

from openclatura import name_mol, opsin_available, verify_with_opsin
from openclatura.chains import find_ring_systems
from openclatura.fusion.indicated_hydrogen import aromatic_nitrogen_lone_pair_sites
from openclatura.fusion.model import FusionConfirmed
from openclatura.fusion.planner import plan_fusion_parent
from openclatura.graph_io import read_rdkit_mol


def _alkyl_pyrazole(length, saturated):
    graph = Chem.RWMol()
    for symbol in ("N", "N", "C", "C", "C", "C", "C", "C"):
        graph.AddAtom(Chem.Atom(symbol))
    for u, v, order in (
        (0, 1, 1), (1, 2, 2), (2, 3, 1), (3, 4, 2), (4, 0, 1),
        (4, 5, 1), (5, 6, 1), (6, 7, 1 if saturated else 2), (7, 3, 1),
    ):
        graph.AddBond(u, v, Chem.BondType.DOUBLE if order == 2 else Chem.BondType.SINGLE)
    last = 0
    for _ in range(length):
        carbon = graph.AddAtom(Chem.Atom("C"))
        graph.AddBond(last, carbon, Chem.BondType.SINGLE)
        last = carbon
    mol = graph.GetMol()
    Chem.SanitizeMol(mol)
    return mol


@pytest.mark.parametrize("length", (1, 2, 4))
@pytest.mark.parametrize("saturated", (False, True))
@pytest.mark.parametrize("reverse", (False, True))
def test_graph_built_n_alkyl_donor_preserves_parent_pi_capacity(length, saturated, reverse):
    mol = _alkyl_pyrazole(length, saturated)
    if reverse:
        mol = Chem.RenumberAtoms(mol, list(reversed(range(mol.GetNumAtoms()))))
    graph = read_rdkit_mol(mol)
    atoms = max(find_ring_systems(graph), key=lambda ring: len(ring.atoms)).atoms
    result = plan_fusion_parent(graph, atoms, mode="audited_pin")
    assert isinstance(result, FusionConfirmed), result
    plan = result.plan
    donor = next(i for i in atoms if graph.atoms[i].symbol == "N" and len(graph.get_neighbors(i)) == 3)
    assert not graph.atoms[donor].total_h_count
    assert donor in aromatic_nitrogen_lone_pair_sites(graph, plan.abstract_parent_graph)
    assert all(
        order == 1 for assignment in plan.bond_model.allowed_kekule_assignments
        for edge, order in assignment.orders if donor in edge
    )
    assert dict(plan.numbering.input_locant_maps[0])[donor] not in plan.indicated_hydrogens
    named = name_mol(mol)
    assert named.error is None
    assert "cyclopenta[c]pyrazole" in named.name
    if opsin_available():
        assert verify_with_opsin(named.name, Chem.MolToSmiles(mol), standardize_smiles=False).status == "matched"
