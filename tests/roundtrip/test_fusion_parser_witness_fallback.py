"""Unproved OPSIN fusion constructions must continue to a safe parent type."""

import pytest

from openclatura import name, opsin_available

CASES = (
    pytest.param(
        "Clc1cccc2c1Cc1ccccc1-c1ccccc1-2",
        "tetracyclo",
        id="multiplied-benzenes-on-seven-membered-parent",
    ),
    pytest.param(
        "Fc1ccc2c(c1)C1C(CC3CCCCCN31)c1ccccc1C2",
        "pentacyclo",
        id="nested-multiplied-benzenes-on-seven-membered-host",
    ),
    pytest.param(
        "C/C=C(\\C)C(=O)Oc1c(OC)c(OC)cc2c1-c1c(cc3c(c1OC)OCO3)[C@H](O)[C@H](C)[C@H](C)C2",
        "tetracyclo",
        id="higher-order-eight-membered-host-without-parser-layout",
    ),
)


@pytest.mark.skipif(not opsin_available(), reason="OPSIN is unavailable")
@pytest.mark.parametrize("smiles,fallback_marker", CASES)
def test_unproved_fusion_construction_uses_roundtrippable_fallback(smiles, fallback_marker):
    result = name(smiles, verify_opsin=True)

    assert result.error is None
    assert fallback_marker in result.name
    assert result.opsin_check.status == "matched", result.opsin_check.to_dict()
