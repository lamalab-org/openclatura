"""Graph coverage and H-count boundaries of saturated nitrogen composition."""

from dataclasses import replace

import pytest
from rdkit import Chem

from openclatura.chains import find_ring_systems
from openclatura.fusion.audit import _has_complete_hydrogenation
from openclatura.fusion.mancude import _single_site_parent_model, saturated_nitrogen_hydrogen_sites
from openclatura.fusion.model import FusionConfirmed, FusionMode
from openclatura.fusion.planner import plan_fusion_parent
from openclatura.graph_io import read_rdkit_mol, read_smiles


@pytest.mark.parametrize("reverse", [False, True], ids=["original", "reversed"])
@pytest.mark.parametrize(
    ("smiles", "uncovered", "hydro_nitrogens"),
    [
        ("N1CCC2C1C[NH2+]C2", 0, 0),
        ("N1CCCC2C1C[NH2+]C2", 1, 0),
        ("N1CCCC2C1C[NH2+]CC2", 0, 0),
        ("N1CCCC2C1CC[NH2+]C2", 0, 2),
    ],
)
def test_saturated_carbon_coverage_boundary(smiles, uncovered, hydro_nitrogens, reverse):
    rd_mol = Chem.MolFromSmiles(smiles)
    if reverse:
        rd_mol = Chem.RenumberAtoms(rd_mol, list(reversed(range(rd_mol.GetNumAtoms()))))
    mol = read_rdkit_mol(rd_mol)
    result = plan_fusion_parent(mol, mol.atoms, mode=FusionMode.GENERAL)
    assert isinstance(result, FusionConfirmed)
    plan = result.plan
    state = plan.derivative_state
    carbons = {atom.idx for atom in mol.atoms.values() if atom.symbol == "C"}
    covered = state.bond_delta.hydrogenated_atom_ids
    assert len(covered - carbons) == hydro_nitrogens
    assert len(carbons - covered) == uncovered
    assert _has_complete_hydrogenation(mol, plan.numbering, plan.indicated_hydrogens, state)


@pytest.mark.parametrize("alkyl", ["C", "CC"])
def test_saturated_h_proof_counts_external_nitrogen_alkyl_bonds(alkyl):
    mol = read_smiles(f"N1CCCC2C1C[NH+]({alkyl})C2")
    atoms = max(find_ring_systems(mol), key=lambda system: len(system.atoms)).atoms
    nitrogens = frozenset(atom for atom in atoms if mol.atoms[atom].symbol == "N")
    assert saturated_nitrogen_hydrogen_sites(mol, atoms, nitrogens) == nitrogens
    charged = next(atom for atom in nitrogens if mol.atoms[atom].charge)
    assert mol.atoms[charged].total_h_count == mol.atoms[charged].explicit_h_count == 1
    mol.atoms[charged] = replace(mol.atoms[charged], total_h_count=2)
    assert not saturated_nitrogen_hydrogen_sites(mol, atoms, nitrogens)


def test_saturated_nitrogen_constraints_reuse_bounded_assignment_cache():
    _single_site_parent_model.cache_clear()
    for _ in range(2):
        mol = read_smiles("N1CCCC2C1C[NH2+]C2")
        assert isinstance(plan_fusion_parent(mol, mol.atoms, mode=FusionMode.AUDITED_PIN), FusionConfirmed)
    assert _single_site_parent_model.cache_info().hits > 0
    assert _single_site_parent_model.cache_info().maxsize == 256
