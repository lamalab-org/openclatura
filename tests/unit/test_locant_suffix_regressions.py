"""Reported fused locant and bridge-attached suffix failures."""

import pytest

from openclatura import name, opsin_available, verify_with_opsin

CASES = {
    7202: "C=CC(O)N1CCN(c2nc(=O)n3c4c(c(-c5ccc(F)cc5)c(Cl)cc24)SC[C@@H](OC)C3)CC1",
    8519: "COC(=O)[C@H]1[C@H]2CC[C@H]3[C@@H]1C(OS(C)(=O)=O)CCC(C)(C)[C@@H]23",
}


@pytest.mark.parametrize("index", CASES)
def test_reported_locant_failures_generate_names(index):
    result = name(CASES[index])
    assert result.ok, result.error
    if index == 8519:
        assert result.name.startswith("methyl ")
        assert "carboxylate" in result.name


@pytest.mark.opsin
@pytest.mark.parametrize("index", CASES)
def test_reported_locant_failures_opsin_roundtrip(index):
    if not opsin_available():
        pytest.skip("OPSIN requires py2opsin and Java")
    result = name(CASES[index])
    assert result.ok, result.error
    check = verify_with_opsin(result.name, CASES[index], standardize_smiles=False)
    assert check.ok, check.to_dict()
