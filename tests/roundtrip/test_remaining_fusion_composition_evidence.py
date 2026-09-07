"""OPSIN witnesses for the five formerly unproved fusion compositions.

These names are round-trip evidence, not assertions of preferred-name status.
Generator promotion belongs in integration coverage after the independent
numbering and atom-level derivative proofs have been completed.
"""

import pytest

from openclatura import opsin_available
from openclatura.opsin_verify import verify_with_opsin

FUSION_COMPOSITION_WITNESSES = (
    (
        "pyran_acetate",
        "CC(=O)OC1Oc2ccc(C)cc2-c2oc(=O)c([Se]c3ccccc3)cc21",
        "9-methyl-2-oxo-3-(phenylselanyl)-2,5-dihydropyrano[5,6-c]benzo[e]pyran-5-yl acetate",
    ),
    (
        "imidazo_benzamide",
        "O=C(Nc1ccc2nc(=O)n3c(c2c1)NCC3)c1cc(Cl)ccc1Cl",
        "2,5-dichloro-N-(5-oxo-2,3-dihydro-1H-imidazo[1,2-c]benzo[e]pyrimidin-9-yl)benzamide",
    ),
    (
        "benzo_carbazole_dione",
        "CCc1cccc2c1[nH]c1c3c(c(C(C)=O)cc12)C(=O)C=CC3=O",
        "5-acetyl-10-ethyl-11H-benzo[a]carbazole-1,4-dione",
    ),
    (
        "saturated_pyrazino_pyrazine",
        "CC1(CO)CN(Cc2ccccc2)CC2CN(Cc3ccc(F)cc3)CCN21",
        "(8-benzyl-2-((4-fluorophenyl)methyl)-6-methyloctahydropyrazino[1,2-a]pyrazin-6-yl)methanol",
    ),
    (
        "spiro_benzofuran_dione",
        "CC1=C2CC3C(C)(C=CC(=O)C34CO4)CC2OC1=O",
        "3,8a-dimethyl-4,4a,8a,9-tetrahydrospiro[benzo[f]1-benzofuran-5,2'-oxirane]-2,6(9aH)-dione",
    ),
)


@pytest.mark.skipif(not opsin_available(), reason="py2opsin/Java is unavailable")
@pytest.mark.parametrize(
    "case_id,smiles,fusion_name",
    FUSION_COMPOSITION_WITNESSES,
    ids=[case[0] for case in FUSION_COMPOSITION_WITNESSES],
)
def test_remaining_fusion_composition_has_opsin_witness(case_id, smiles, fusion_name):
    check = verify_with_opsin(fusion_name, smiles)
    assert check.status == "matched", (case_id, check.to_dict())
