"""Exact regression coverage accompanying the azo/guanidino stereo handoff."""

import pytest
from rdkit import Chem

from openclatura import name_mol, opsin_available, verify_with_opsin

CASES = (
    ("4658", "C=C[Si](C)(C)O[Si](CCCNC(=O)/N=N/C(=O)OCC)(OCC)OCC"),
    ("7159", "O=S(=O)(F)c1ccc(N=Nc2ccc(N3CCN(C(S)S)CC3)cc2)cc1"),
    ("43852", "COC(=O)N(C)/C(=N/C(=O)NCC1CCCC2CCCCC21)N(C)C"),
    ("azo-E", "COC(=O)/N=N/C(=O)NC"),
    ("azo-Z", "COC(=O)/N=N\\C(=O)NC"),
    ("azo-unspecified", "COC(=O)N=NC(=O)NC"),
    ("aryl-azo", "Cc1ccc(N=Nc2ccccc2)cc1"),
    ("guanidino-E", "COC(=O)N(C)/C(=N/C(=O)NC)N(C)C"),
    ("guanidino-Z", "COC(=O)N(C)/C(=N\\C(=O)NC)N(C)C"),
)


@pytest.mark.opsin
@pytest.mark.skipif(not opsin_available(), reason="OPSIN and Java required")
@pytest.mark.parametrize("reverse", (False, True))
@pytest.mark.parametrize("case,smiles", CASES, ids=[case for case, _ in CASES])
def test_azo_and_guanidino_stereo_roundtrip(case, smiles, reverse):
    mol = Chem.MolFromSmiles(smiles)
    if reverse:
        mol = Chem.RenumberAtoms(mol, list(reversed(range(mol.GetNumAtoms()))))
    result = name_mol(mol, include_trace=True)
    assert result.error is None, result.error
    check = verify_with_opsin(result.name, smiles, standardize_smiles=False)
    assert check.status == "matched", (case, result.name, check.to_dict())
    assert check.canonical_original == check.canonical_roundtrip
