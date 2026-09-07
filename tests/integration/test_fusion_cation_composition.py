"""Previously matched charged fusion derivatives must retain fusion naming."""

import pytest
from rdkit import Chem

from openclatura import FusionMode, name, name_mol, opsin_available


@pytest.mark.opsin
@pytest.mark.skipif(not opsin_available(), reason="py2opsin/Java is unavailable")
@pytest.mark.parametrize(
    ("smiles", "expected"),
    [
        (
            "CC(C)(C)/C=C/C(=O)N1CCC[C@@H]2[C@H]1C[NH2+]C2",
            "(2E)-1-((4aS,7aS)-2,3,4,4a,7,7a-hexahydro-6H-pyrrolo[3,4-b]pyridin-6-ium-1-yl)"
            "-4,4-dimethylpent-2-en-1-one",
        ),
        (
            "Cn1c(ccn1)C[NH+]2CCc3c(cc[nH]c3=O)C2",
            "2-((1-methyl-1H-pyrazol-5-yl)methyl)-3,4-dihydro-2H,6H-pyrido[4,3-c]pyridin-2-ium-5-one",
        ),
    ],
)
def test_charged_hydro_fusion_roundtrips_after_atom_reordering(smiles, expected):
    mol = Chem.MolFromSmiles(smiles)
    reordered = Chem.RenumberAtoms(mol, list(reversed(range(mol.GetNumAtoms()))))
    results = (
        name(smiles, fusion_mode=FusionMode.AUDITED_PIN, verify_opsin=True),
        name_mol(reordered, fusion_mode=FusionMode.AUDITED_PIN, verify_opsin=True),
    )
    for result in results:
        assert result.name == expected
        assert result.opsin_check is not None and result.opsin_check.status == "matched"


@pytest.mark.opsin
@pytest.mark.skipif(not opsin_available(), reason="py2opsin/Java is unavailable")
@pytest.mark.parametrize(
    ("smiles", "expected"),
    [
        ("CN1CCCC2C1C[NH2+]C2", "1-methyl-2,3,4,4a,7,7a-hexahydro-6H-pyrrolo[3,4-b]pyridin-6-ium"),
        ("CCN1CCCC2C1C[NH2+]C2", "1-ethyl-2,3,4,4a,7,7a-hexahydro-6H-pyrrolo[3,4-b]pyridin-6-ium"),
        ("CN1CCCC2C1C[NH2+]C2C", "1,5-dimethyl-2,3,4,4a,7,7a-hexahydro-6H-pyrrolo[3,4-b]pyridin-6-ium"),
        ("[NH2+]1CCc2c(cc[nH]c2=O)C1", "3,4-dihydro-2H,6H-pyrido[4,3-c]pyridin-2-ium-5-one"),
        ("C[NH+]1CCc2c(cc[nH]c2=O)C1", "2-methyl-3,4-dihydro-2H,6H-pyrido[4,3-c]pyridin-2-ium-5-one"),
        ("CC[NH+]1CCc2c(cc[nH]c2=O)C1", "2-ethyl-3,4-dihydro-2H,6H-pyrido[4,3-c]pyridin-2-ium-5-one"),
        ("[NH2+]1CCc2cc[nH]c(=O)c2C1", "3,4-dihydro-2H,7H-pyrido[3,4-c]pyridin-2-ium-8-one"),
    ],
)
def test_separate_hydro_and_nitrogen_h_variants_roundtrip(smiles, expected):
    mol = Chem.MolFromSmiles(smiles)
    for order in (list(range(mol.GetNumAtoms())), list(reversed(range(mol.GetNumAtoms())))):
        result = name_mol(Chem.RenumberAtoms(mol, order), fusion_mode=FusionMode.AUDITED_PIN, verify_opsin=True)
        assert result.name == expected
        assert result.parent_nomenclature == "systematic_fusion"
        assert result.opsin_check is not None and result.opsin_check.status == "matched"
