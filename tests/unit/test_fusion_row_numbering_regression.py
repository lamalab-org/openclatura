"""Contiguous horizontal fusion rows determine completed-system locants."""

import pytest
from rdkit import Chem

from openclatura import name_mol, opsin_available, verify_with_opsin
from openclatura.fusion.model import FusionConfirmed, FusionMode
from openclatura.fusion.planner import plan_fusion_parent
from openclatura.graph_io import read_smiles

SMILES = "c1ccc2c(c1)ccc1c2ccc2c3c[nH]cc3ncc21"
EXPECTED_NAME = "2H-pyrrolo[3,4-b]phenanthro[2,1-d]pyridine"
DIONE_SMILES = "CCc1cccc2c1[nH]c1c3c(c(C(C)=O)cc12)C(=O)C=CC3=O"
DIONE_NAME = "5-acetyl-10-ethyl-11H-benzo[a]carbazole-1,4-dione"


@pytest.mark.parametrize("offset", (0, 7, 14))
def test_disconnected_collinear_faces_do_not_shift_completed_system_numbering(offset):
    rdkit_mol = Chem.MolFromSmiles(SMILES)
    order = list(reversed(range(rdkit_mol.GetNumAtoms())))
    reordered = Chem.RenumberAtoms(rdkit_mol, order[offset:] + order[:offset])
    mol = read_smiles(Chem.MolToSmiles(reordered, canonical=False))

    result = plan_fusion_parent(mol, mol.atoms, mode=FusionMode.AUDITED_PIN)

    assert isinstance(result, FusionConfirmed)
    plan = result.plan
    assert plan.rendered_base_name == EXPECTED_NAME
    assert plan.numbering.selected_layout.orientation_score[1] == -2
    for items in plan.numbering.input_locant_maps:
        assert {
            str(locant)
            for atom, locant in items
            if mol.atoms[atom].symbol == "N" and mol.atoms[atom].total_h_count == 1
        } == {"2"}


@pytest.mark.opsin
@pytest.mark.parametrize("methyl_site", (None, 14, 19))
def test_row_numbering_roundtrips_parent_and_substituent_locants(methyl_site):
    if not opsin_available():
        pytest.skip("OPSIN and Java are required")
    mol = Chem.RWMol(Chem.MolFromSmiles(SMILES))
    if methyl_site is not None:
        methyl = mol.AddAtom(Chem.Atom("C"))
        mol.AddBond(methyl_site, methyl, Chem.BondType.SINGLE)
    Chem.SanitizeMol(mol)

    result = name_mol(mol, fusion_mode=FusionMode.AUDITED_PIN, include_trace=True)

    assert result.error is None
    assert result.parent_nomenclature == "systematic_fusion"
    if methyl_site is None:
        assert result.name == EXPECTED_NAME
    check = verify_with_opsin(result.name, Chem.MolToSmiles(mol), standardize_smiles=False)
    assert check.status == "matched", check.to_dict()


@pytest.mark.parametrize("offset", (0, 7, 14))
def test_parallel_fusion_sides_preserve_angular_dione_numbering(offset):
    rdkit_mol = Chem.MolFromSmiles(DIONE_SMILES)
    order = list(reversed(range(rdkit_mol.GetNumAtoms())))
    reordered = Chem.RenumberAtoms(rdkit_mol, order[offset:] + order[:offset])
    smiles = Chem.MolToSmiles(reordered, canonical=False)
    mol = read_smiles(smiles)
    serialized = Chem.MolFromSmiles(smiles)
    parent_atoms = {atom.GetIdx() for atom in serialized.GetAtoms() if atom.IsInRing()}

    result = plan_fusion_parent(mol, parent_atoms, mode=FusionMode.GENERAL)

    assert isinstance(result, FusionConfirmed)
    assert result.plan.numbering.selected_layout.orientation_score[1] == -3
    for items in result.plan.numbering.input_locant_maps:
        assert {str(locant) for atom, locant in items if mol.atoms[atom].symbol == "N"} == {"11"}
        oxo_atoms = {
            atom
            for atom in parent_atoms
            if any(mol.atoms[neighbour].symbol == "O" for neighbour in mol.get_neighbors(atom))
        }
        assert {str(locant) for atom, locant in items if atom in oxo_atoms} == {"1", "4"}


@pytest.mark.opsin
@pytest.mark.parametrize("mode", (FusionMode.GENERAL, FusionMode.AUDITED_PIN))
def test_angular_dione_generated_name_roundtrips_with_completed_system_locants(mode):
    if not opsin_available():
        pytest.skip("OPSIN and Java are required")
    mol = Chem.MolFromSmiles(DIONE_SMILES)

    result = name_mol(mol, fusion_mode=mode, include_trace=True)

    assert result.error is None
    assert result.parent_nomenclature == "systematic_fusion"
    assert result.name == DIONE_NAME
    check = verify_with_opsin(result.name, DIONE_SMILES, standardize_smiles=False)
    assert check.status == "matched", check.to_dict()
