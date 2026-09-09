"""Intrinsic junction CH must not become an unrelated nitrogen tautomer."""

import random

import pytest
from rdkit import Chem

from openclatura import name_mol, opsin_available, verify_with_opsin
from openclatura.chains import find_ring_systems
from openclatura.fusion.indicated_hydrogen import is_intrinsic_carbon_h_site
from openclatura.fusion.model import FusionConfirmed, FusionMode
from openclatura.fusion.planner import plan_fusion_parent
from openclatura.graph_io import read_rdkit_mol
from openclatura.molecule import Molecule

CASES = (
    ("Nc1cnn2c1C=CC1C=CC=C12", "5a"),
    ("Nc1cnn2c1C=CC1(C)C=CC=C12", "5a"),
    ("Nc1cnn2c1C=CC1(CC)C=CC=C12", "5a"),
    ("CCOC(=O)c1ncn2c1C(C)=NC(c1ccccc1F)=C1C=C(C#C[Si](C)(C)C)C=CC12", "10a"),
    ("C1=CC2=Cn3cnnc3C=NC2C=C1", "9a"),
)


@pytest.mark.parametrize("smiles,locant", CASES)
@pytest.mark.parametrize("order", ("original", "reversed", "shuffled"))
def test_junction_h_is_proved_without_hydrogenation_and_roundtrips(smiles, locant, order):
    rd_mol = Chem.MolFromSmiles(smiles)
    indices = list(range(rd_mol.GetNumAtoms()))
    if order == "reversed":
        indices.reverse()
    elif order == "shuffled":
        random.Random(19).shuffle(indices)
    rd_mol = Chem.RenumberAtoms(rd_mol, indices)
    mol = read_rdkit_mol(rd_mol)
    atoms = max(find_ring_systems(mol), key=lambda system: len(system.atoms)).atoms
    result = plan_fusion_parent(mol, atoms, mode=FusionMode.AUDITED_PIN)
    assert isinstance(result, FusionConfirmed), result
    plan = result.plan
    assert plan.audit.confirmed
    assert tuple(map(str, plan.indicated_hydrogens)) == (locant,)
    junction = next(atom for atom, value in plan.numbering.input_locant_maps[0] if str(value) == locant)
    assert len(atoms.intersection(mol.get_neighbors(junction))) == 3
    assert is_intrinsic_carbon_h_site(mol, junction, atoms)
    assert not plan.derivative_state.hydro_operations
    assert not plan.derivative_state.added_hydrogen_operations
    assert not plan.derivative_state.unsaturation_operations
    assert all(
        all(bond_order == 1 for edge, bond_order in assignment.orders if junction in edge)
        for assignment in plan.bond_model.allowed_kekule_assignments
    )
    name = name_mol(rd_mol, fusion_mode=FusionMode.AUDITED_PIN)
    assert name.error is None
    assert locant + "H-" in name.name
    if opsin_available():
        check = verify_with_opsin(name.name, smiles, standardize_smiles=False)
        assert check.status == "matched", (name.name, check)
        assert check.canonical_original == check.canonical_roundtrip


@pytest.mark.parametrize("degree,external", ((2, 0), (2, 1), (2, 2), (3, 0), (3, 1)))
def test_graph_built_sigma_carbon_roles_include_replaced_hydrogen(degree, external):
    mol = Molecule()
    mol.add_atom("C", idx=0, total_h_count=4 - degree - external)
    for atom in range(1, degree + external + 1):
        mol.add_atom("C", idx=atom, total_h_count=3)
        mol.add_bond(0, atom)
    atoms = frozenset(range(degree + 1))
    before = mol.atoms.copy(), mol.bonds.copy()
    assert is_intrinsic_carbon_h_site(mol, 0, atoms)
    assert (mol.atoms, mol.bonds) == before


@pytest.mark.parametrize("corruption", ("charge", "aromatic", "double", "bad_h", "degree_four"))
def test_junction_candidate_does_not_bypass_valence_or_pi_constraints(corruption):
    mol = Molecule()
    mol.add_atom("C", idx=0, total_h_count=1)
    for atom in range(1, 4):
        mol.add_atom("C", idx=atom, total_h_count=3)
        mol.add_bond(0, atom)
    atoms = {0, 1, 2, 3}
    if corruption == "charge":
        mol.update_atom(0, charge=1)
    elif corruption == "aromatic":
        mol.update_atom(0, is_aromatic=True)
    elif corruption == "double":
        mol.update_bond(mol.get_bond(0, 1).idx, order=2)
    elif corruption == "bad_h":
        mol.update_atom(0, total_h_count=2)
    else:
        mol.add_atom("C", idx=4, total_h_count=3)
        mol.add_bond(0, 4)
        mol.update_atom(0, total_h_count=0)
        atoms.add(4)
    assert not is_intrinsic_carbon_h_site(mol, 0, atoms)
