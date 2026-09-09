"""Relocated carbon H belongs to the completed graph, not one component."""

import random

import pytest
from rdkit import Chem

from openclatura import name_mol, opsin_available
from openclatura.chains import find_ring_systems
from openclatura.fusion.model import FusionConfirmed, FusionMode
from openclatura.fusion.planner import plan_fusion_parent
from openclatura.graph_io import read_rdkit_mol
from openclatura.opsin_verify import verify_with_opsin

SMILES = "c1ccc2c(c1)Cc1cc3c(cc1-2)Cc1cc2c(cc1-3)Cc1cccc3c1N2c1cccnc1C3"


def _molecule(order, ligand):
    mol = Chem.MolFromSmiles(SMILES)
    if ligand:
        # Add a substituent to an available aromatic carbon, keeping its core.
        site = next(
            a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 6 and a.GetIsAromatic() and a.GetTotalNumHs()
        )
        editable = Chem.RWMol(mol)
        atom = editable.AddAtom(Chem.Atom(ligand))
        editable.AddBond(site, atom, Chem.BondType.SINGLE)
        mol = editable.GetMol()
        Chem.SanitizeMol(mol)
    indices = list(range(mol.GetNumAtoms()))
    if order == "reversed":
        indices.reverse()
    elif order == "shuffled":
        random.Random(19).shuffle(indices)
    return Chem.RenumberAtoms(mol, indices)


@pytest.mark.parametrize("order", ("original", "reversed", "shuffled"))
@pytest.mark.parametrize("ligand", (None, "C", "F"))
def test_completed_hydrogen_sites_have_conserved_pi_budget(order, ligand):
    mol = read_rdkit_mol(_molecule(order, ligand))
    atoms = max(find_ring_systems(mol), key=lambda r: len(r.atoms)).atoms
    result = plan_fusion_parent(mol, atoms, mode=FusionMode.AUDITED_PIN)
    assert isinstance(result, FusionConfirmed), result
    plan = result.plan
    assert plan.audit.confirmed
    locants = {atom: str(locant) for atom, locant in plan.numbering.input_locant_maps[0]}
    assert {str(locant) for locant in plan.indicated_hydrogens} == {"5", "9"}
    sites = {atom for atom, locant in locants.items() if locant in {"5", "9"}}
    assert len(sites) == 2
    assert all(mol.atoms[atom].symbol == "C" and mol.atoms[atom].total_h_count == 2 for atom in sites)
    for assignment in plan.bond_model.allowed_kekule_assignments:
        assert all(order == 1 for edge, order in assignment.orders if sites.intersection(edge))
        assert sum(order == 2 for _, order in assignment.orders) == plan.bond_model.maximum_non_cumulative_double_bonds
    hydro = {atom for op in plan.derivative_state.hydro_operations for atom in op.atom_ids}
    assert not hydro.intersection(sites)
    assert not any(mol.atoms[atom].is_aromatic for atom in hydro)


@pytest.mark.skipif(not opsin_available(), reason="OPSIN is unavailable")
@pytest.mark.parametrize("order", ("original", "reversed", "shuffled"))
@pytest.mark.parametrize("ligand", (None, "C", "F"))
def test_cross_component_hydrogen_roundtrips_exactly(order, ligand):
    mol = _molecule(order, ligand)
    result = name_mol(mol, fusion_mode=FusionMode.AUDITED_PIN)
    assert result.error is None
    assert "5H,9H" in result.name
    check = verify_with_opsin(result.name, Chem.MolToSmiles(mol), standardize_smiles=False)
    assert check.status == "matched", (result.name, check)
    assert check.canonical_original == check.canonical_roundtrip
