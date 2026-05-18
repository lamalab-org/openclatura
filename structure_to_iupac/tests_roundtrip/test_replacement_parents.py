"""Round-trip tests for test_replacement_parents.py."""

import pytest

from structure_to_iupac.tests_roundtrip.roundtrip_helpers import roundtrip_smiles

SMILES = [
    'OP(=O)(O)O',
    'O=S(=O)(O)O',
    'O=N(=O)O',
    'OBr',
    'C[BH2]',
    'Cl[SiH2]Cl',
    'COP(OC)OC',
    'CP(C)(C)=O',
    'CC(C)(C)O[SiH2]OC(C)(C)C',
    'CC(C)(C)O[PH](OC(C)(C)C)OC(C)(C)C',
    '[BH3]',
    '[NH3]',
    'O',
    '[SiH4]',
    '[PH3]',
    'S',
    '[ClH]',
    '[AsH3]',
    '[SbH3]',
    '[BiH3]',
    'NN',
    'N=N',
    'NN=N',
    '[SiH]#[SiH]',
    'NNNNNNNNN',
    'SSSS',
    '[SiH3][SiH2][SiH2][SiH2][SiH3]',
    'C[SiH2][SiH2]C',
    'Cl[SiH2][SiH2]Cl',
    'CO[SiH2][SiH2]OC',
    'Cl[SiH]=[SiH]Cl',
    'C[Si]#[Si]C',
    'c1ccccc1P(c2ccccc2)c2ccccc2',
    'CO[Si](Cl)(C)c1ccccc1',
    'CP(F)(OC)c1ccccc1',
    'CP(=O)(OC)c1ccccc1',
    'FP(F)(F)(F)F',
]

@pytest.mark.parametrize("smiles", SMILES)
def test_roundtrip(smiles):
    roundtrip_smiles(smiles)

def test_has_smiles():
    if not SMILES:
        pytest.skip("No SMILES literals found.")
