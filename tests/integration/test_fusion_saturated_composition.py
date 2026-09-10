"""Saturated fused nitrogen parents retain every hydrogen and formal charge."""

import pytest
from rdkit import Chem

from openclatura import FusionMode, name_mol, opsin_available


@pytest.mark.opsin
@pytest.mark.skipif(not opsin_available(), reason="py2opsin/Java is unavailable")
@pytest.mark.parametrize(
    ("smiles", "expected"),
    [
        ("N1CCCC2C1C[NH2+]C2", "hexahydro-1H,6H-pyrrolo[3,4-b]pyridin-6-ium"),
        ("[NH2+]1CCCC2C1CNC2", "hexahydro-1H,6H-pyrrolo[3,4-b]pyridin-1-ium"),
        ("[NH2+]1CCCC2C1C[NH2+]C2", "hexahydro-1H,6H-pyrrolo[3,4-b]pyridin-1,6-diium"),
        ("N1CCCC2C1CNC2", "hexahydro-1H,6H-pyrrolo[3,4-b]pyridine"),
        ("N1CCC2C1C[NH2+]C2", "octahydropyrrolo[3,4-b]pyrrol-5-ium"),
        ("N1CCCC2C1C[NH2+]CC2", "decahydropyrido[3,4-b]pyridin-7-ium"),
        ("N1CCCC2C1[NH2+]CC2", "hexahydro-1H,7H-pyrrolo[2,3-b]pyridin-1-ium"),
        ("N1CCC(C)C2C1C[NH2+]C2", "4-methylhexahydro-1H,6H-pyrrolo[3,4-b]pyridin-6-ium"),
        ("N1CCCC2C1C[NH+](C)C2", "6-methylhexahydro-1H,6H-pyrrolo[3,4-b]pyridin-6-ium"),
        ("N1CCCC2C1C[NH+](CC)C2", "6-ethylhexahydro-1H,6H-pyrrolo[3,4-b]pyridin-6-ium"),
        ("C[NH+]1CCCC2C1C[NH2+]C2", "1-methylhexahydro-1H,6H-pyrrolo[3,4-b]pyridin-1,6-diium"),
        ("CN1CCCC2C1C[NH2+]C2", "1-methyl-2,3,4,4a,7,7a-hexahydro-6H-pyrrolo[3,4-b]pyridin-6-ium"),
        ("CCN1CCCC2C1C[NH2+]C2", "1-ethyl-2,3,4,4a,7,7a-hexahydro-6H-pyrrolo[3,4-b]pyridin-6-ium"),
        ("N1CCCC2C1CC[NH2+]C2", "decahydropyrido[4,3-b]pyridin-6-ium"),
        ("[NH2+]1CCCC2C1CCNC2", "decahydropyrido[4,3-b]pyridin-1-ium"),
        ("[NH2+]1CCCC2C1CC[NH2+]C2", "decahydropyrido[4,3-b]pyridin-1,6-diium"),
        ("N1CCCC2C1CC[NH+](C)C2", "6-methyldecahydropyrido[4,3-b]pyridin-6-ium"),
        ("C[NH+]1CCCC2C1CC[NH2+]C2", "1-methyldecahydropyrido[4,3-b]pyridin-1,6-diium"),
        ("CN1CCCC2C1CC[NH2+]C2", "1-methyldecahydropyrido[4,3-b]pyridin-6-ium"),
        ("CCN1CCCC2C1CC[NH2+]C2", "1-ethyldecahydropyrido[4,3-b]pyridin-6-ium"),
        ("N1CCC2C(C1)C[NH2+]CC2", "decahydropyrido[3,4-c]pyridin-7-ium"),
        ("N1CCC(C)C2C1CC[NH2+]C2", "4-methyldecahydropyrido[4,3-b]pyridin-6-ium"),
    ],
)
def test_saturated_composition_roundtrips_after_atom_reordering(smiles, expected):
    mol = Chem.MolFromSmiles(smiles)
    for order in (list(range(mol.GetNumAtoms())), list(reversed(range(mol.GetNumAtoms())))):
        result = name_mol(Chem.RenumberAtoms(mol, order), fusion_mode=FusionMode.AUDITED_PIN, verify_opsin=True)
        assert result.name == expected
        assert result.parent_nomenclature == "systematic_fusion"
        assert result.opsin_check is not None and result.opsin_check.status == "matched"
        parsed = Chem.MolFromSmiles(result.opsin_check.opsin_smiles)
        assert Chem.MolToSmiles(parsed) == Chem.MolToSmiles(mol)
        assert sum(atom.GetTotalNumHs() for atom in parsed.GetAtoms()) == sum(
            atom.GetTotalNumHs() for atom in mol.GetAtoms()
        )
        assert Chem.GetFormalCharge(parsed) == Chem.GetFormalCharge(mol)
