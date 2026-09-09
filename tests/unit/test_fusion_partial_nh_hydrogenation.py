"""Partial hydrogenation owns N-H separately from intrinsic donor citations."""

import pytest
from rdkit import Chem

from openclatura import name_mol, opsin_available, verify_with_opsin
from openclatura.chains import find_ring_systems
from openclatura.fusion.model import FusionConfirmed
from openclatura.fusion.planner import plan_fusion_parent
from openclatura.graph_io import read_rdkit_mol


def _fused_azole(symbol, methyl):
    graph = Chem.RWMol()
    for element in ("C", "N", "C", "C", symbol, "C", "N", "C", "C"):
        graph.AddAtom(Chem.Atom(element))
    for u, v, order in (
        (0, 1, 1),
        (1, 2, 1),
        (2, 3, 2),
        (3, 4, 1),
        (4, 5, 1),
        (5, 6, 2),
        (6, 2, 1),
        (3, 7, 1),
        (7, 8, 2),
        (8, 0, 1),
    ):
        graph.AddBond(u, v, Chem.BondType.DOUBLE if order == 2 else Chem.BondType.SINGLE)
    if methyl:
        graph.AddBond(0, graph.AddAtom(Chem.Atom("C")), Chem.BondType.SINGLE)
    mol = graph.GetMol()
    Chem.SanitizeMol(mol)
    return mol


@pytest.mark.parametrize("symbol", ("N", "O"))
@pytest.mark.parametrize("methyl", (False, True))
@pytest.mark.parametrize("reverse", (False, True))
def test_partially_hydrogenated_ring_nh_is_owned_by_hydro_operation(symbol, methyl, reverse):
    mol = _fused_azole(symbol, methyl)
    if reverse:
        mol = Chem.RenumberAtoms(mol, list(reversed(range(mol.GetNumAtoms()))))
    graph = read_rdkit_mol(mol)
    atoms = max(find_ring_systems(graph), key=lambda ring: len(ring.atoms)).atoms
    planned = plan_fusion_parent(graph, atoms, mode="audited_pin")
    assert isinstance(planned, FusionConfirmed), planned
    plan = planned.plan
    saturated_n = next(i for i, atom in graph.atoms.items() if atom.symbol == "N" and not atom.is_aromatic)
    hydro_atoms = {atom for operation in plan.derivative_state.hydro_operations for atom in operation.atom_ids}
    assert saturated_n in hydro_atoms
    assert len(hydro_atoms) == 2
    locants = dict(plan.numbering.input_locant_maps[0])
    assert locants[saturated_n] not in plan.indicated_hydrogens
    result = name_mol(mol)
    assert result.error is None
    assert "dihydro" in result.name
    assert "bicyclo[" not in result.name
    if opsin_available():
        assert verify_with_opsin(result.name, Chem.MolToSmiles(mol), standardize_smiles=False).status == "matched"


def test_intrinsic_component_donor_does_not_become_adjacent_hydro_pair():
    mol = Chem.MolFromSmiles("C1NCC2=NNN=C12")
    result = name_mol(mol)
    assert result.name == "2H,4H,5H,6H-pyrrolo[3,4-d][1,2,3]triazole"
    if opsin_available():
        assert verify_with_opsin(result.name, Chem.MolToSmiles(mol), standardize_smiles=False).status == "matched"
        assert (
            verify_with_opsin(
                "4,6-dihydro-2H,5H-pyrrolo[3,4-d][1,2,3]triazole",
                Chem.MolToSmiles(mol),
                standardize_smiles=False,
            ).status
            == "matched"
        )
