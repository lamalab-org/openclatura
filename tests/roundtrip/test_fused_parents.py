"""Round-trip tests for test_fused_parents.py."""

import json
from collections import Counter
from pathlib import Path

import pytest
from rdkit import Chem

from openclatura import FusionMode, name, opsin_available, verify_with_opsin
from roundtrip.roundtrip_helpers import roundtrip_smiles

PIN_CASE_DATA = json.loads((Path(__file__).parents[1] / "data" / "fusion_pin_cases.json").read_text(encoding="utf-8"))
SUITE_PIN_CASES = tuple(PIN_CASE_DATA["suite_cases"])
OPSIN_GENERATED_PIN_CASES = tuple(PIN_CASE_DATA["opsin_generated_cases"])

SMILES = [
    "c1cnc2cc[nH]c2c1",
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "c1cc2ccc3cccc4ccc(c1)c2c34",
    "c1ccc2ccccc2c1",
    "C1CC2=CC=CC3=C2C1=CC=C3",
    "c1oc2ncccc2c1",
    "c1ccc2cc3ncccc3cc2c1",
    "c1oc2nc3occc3cc2c1",
    "c1ccc2cc3ccccc3cc2c1",
    "c1ccc2c(c1)ccc1ccccc12",
    "c1ccc2ncccc2c1",
    "c1ccc2cnccc2c1",
    "Cc1ccc2ncccc2c1",
    "COc1ccc2ncccc2c1",
    "N#Cc1ccc2ncccc2c1",
    "O=C(O)c1ccc2ncccc2c1",
    "Cc1ccc2cnccc2c1",
    "n1cccc2ncccc12",
    "n1cccc2cnccc12",
    "n1cccc2ccncc12",
    "n1cccc2cccnc12",
    "c1nccc2cnccc12",
    "c1nccc2ccncc12",
    "Cc1nccc2cnccc12",
    "n1cncc2ccccc12",
    "n1ccnc2ccccc12",
    "n1nccc2ccccc12",
    "c1nncc2ccccc12",
    "Cc1ncnc2ccccc12",
    "c1ccc2[nH]ccc2c1",
    "c1ccc2occc2c1",
    "c1ccc2sccc2c1",
    "c1ccc2nc3ccccc3cc2c1",
    "C1=Cc2ccccc2C1",
    "c1ccc2c(c1)CCc1ccccc1-2",
    "c1ccc2cc3c(ccc4ccccc43)cc2c1",
    "c1ccc2cc3cc4ccccc4cc3cc2c1",
    "c1ccc2cc3cc4cc5ccccc5cc4cc3cc2c1",
    "c1ccc2c(c1)-c1ccccc1-2",
    "c1ccc2c(c1)c1ccccc1c1ccccc21",
    "c1ncc2nc[nH]c2n1",
    "c1ccc2[nH]ncc2c1",
    "c1ccc2ncncc2c1",
    "c1ccc2nccnc2c1",
    "c1ccc2cnncc2c1",
    "c1ccc2nnccc2c1",
    "c1ccc2nc3ccccc3nc2c1",
    "c1ccc2c(c1)[nH]c1ccccc12",
    "c1ccc2c(c1)CCOc1ccccc1-2",
    "c1ccc2c(c1)CCSc1ccccc1-2",
    "C1=COc2ccccc2C1",
    "C1=Cc2ccccc2OC1",
    "N1CCCC2=CC=CC=C12",
    "C1CCC2=CC=CC=C2C1",
    "Oc1cccc2c1NCCC2",
    "C1Cc2ncccc2O1",
    "N1CCc2ncccc2C1",
    "C1CNc2ncccc2C1",
    "c1[se]c2ncccc2c1",
    "c1oc2nccnc2c1",
    "c1oc2ncncc2c1",
    "c1nnc2ncccc2c1",
    "c1[nH]c2ncccc2n1",
    "c1ccc2cc3ncncc3cc2c1",
    "c1ccc2cc3nccnc3cc2c1",
    "c1ccc2cc3cnncc3cc2c1",
    "c1ccc2cc3nnccc3cc2c1",
    "c1oc2cc3ncncc3cc2c1",
    "c1sc2cc3ncncc3cc2c1",
    "c1oc2cc3nccnc3cc2c1",
    "c1sc2cc3nccnc3cc2c1",
    "c1oc2cc3cnncc3cc2c1",
    "c1oc2cc3nnccc3cc2c1",
    "c1oc2cc3nc4ccccc4nc3cc2c1",
    "c1sc2cc3nc4ccccc4nc3cc2c1",
    "c1oc2cc3[nH]ncc3cc2c1",
    "c1oc2nc3occc3nc2c1",
    "c1sc2nc3sccc3nc2c1",
    "c1oc2nc3sccc3cc2c1",
    "c1sc2nc3occc3cc2c1",
    "c1ccc2c(c1)ccc1c3ccccc3ccc21",
    "c1ccc2c(c1)ccc1cc3c(ccc4ccccc43)cc12",
    "Cc1cccc2ccccc12",
    "Cc1ccc2ccccc2c1",
    "Oc1cccc2ccccc12",
    "Nc1cccc2ccccc12",
    "Sc1cccc2ccccc12",
    "Oc1cccc2ncccc12",
    "Nc1cccc2[nH]ccc12",
    "Oc1cccc2occc12",
    "Oc1ccc2cc3ccccc3cc2c1",
    "Oc1ccc2c(c1)ccc1ccccc12",
    "Oc1ccc2nc3ccccc3cc2c1",
    "C1=Cc2c(O)cccc2C1",
    "Oc1ccc2c(c1)CCc1ccccc1-2",
    "Oc1cc2ccc3cccc4ccc(c1)c2c34",
    "Oc1ccc2cc3c(ccc4ccccc43)cc2c1",
    "Oc1ccc2[nH]ncc2c1",
    "Oc1ccc2ncncc2c1",
    "Oc1ccc2nccnc2c1",
    "Oc1ccc2cnncc2c1",
    "Oc1ccc2nnccc2c1",
    "Oc1ccc2nc3ccccc3nc2c1",
    "Oc1ccc2c(c1)[nH]c1ccccc12",
    "Oc1ccc2c(c1)CCOc1ccccc1-2",
    "Oc1ccc2c(c1)CCSc1ccccc1-2",
    "C1=COc2c(O)cccc2C1",
    "C1=CSc2c(O)cccc2C1",
    "C1=Cc2c(O)cccc2OC1",
    "C1=Cc2c(O)cccc2SC1",
    "CC1CC2=CC=CC3=C2C1=CC=C3",
    "ClC1CC2=CC=CC3=C2C1=CC=C3",
    "O1CC2=CC=CC3=C2C1=CC=C3",
    "N1CC2=CC=CC3=C2C1=CC=C3",
    "S1CC2=CC=CC3=C2C1=CC=C3",
    "Cc1oc2ncccc2c1",
    "Oc1oc2ncccc2c1",
    "Nc1oc2ncccc2c1",
    "N#Cc1oc2ncccc2c1",
    "O=Cc1oc2ncccc2c1",
    "O=C(O)c1oc2ncccc2c1",
    "NC(=O)c1oc2ncccc2c1",
    "CCOC(=O)c1oc2ncccc2c1",
    "O=C(Cl)c1oc2ncccc2c1",
    "O=S(=O)(O)c1oc2ncccc2c1",
    "OP(=O)(O)c1oc2ncccc2c1",
    "OS(=O)(=O)Oc1oc2ncccc2c1",
    "OP(=O)(O)Oc1oc2ncccc2c1",
    "c1oc2ncccc2c1O[N+](=O)[O-]",
    "c1oc2ncccc2c1ON=O",
    "NS(=O)(=O)c1oc2ncccc2c1",
]


@pytest.mark.parametrize("smiles", SMILES)
def test_roundtrip(smiles):
    roundtrip_smiles(smiles)


def test_has_smiles():
    if not SMILES:
        pytest.skip("No SMILES literals found.")


def _pin_route(result) -> str:
    if result.parent_nomenclature == "systematic_fusion":
        assert result.pin_status == "confirmed"
        assert result.proof_source == "fusion_reconstruction"
        selected = [step for step in result.decisions if step.decision == "selected audited systematic fusion parent"]
        assert len(selected) == 1
        assert "input_graph_identity" in selected[0].data["audit_checks"]
        return "systematic_fusion"

    decisions = {step.decision for step in result.decisions}
    assert result.pin_status is None
    if "used retained parent name" in decisions:
        return "retained_parent"
    if "named structural replacement parent" in decisions:
        return "principal_parent_shortcut"
    assert "systematic fusion fallback" in decisions
    return "unverified_fallback"


def test_fused_parent_suite_pin_claims_are_explicit_and_evidence_backed():
    expected_pin_smiles = {case["smiles"] for case in SUITE_PIN_CASES}
    assert expected_pin_smiles <= set(SMILES)

    routes = Counter()
    observed_pin_smiles = set()
    for smiles in SMILES:
        result = name(smiles, fusion_mode=FusionMode.AUDITED_PIN, include_trace=True)
        assert result.error is None
        route = _pin_route(result)
        routes[route] += 1
        if route == "systematic_fusion":
            observed_pin_smiles.add(smiles)

    assert observed_pin_smiles == expected_pin_smiles
    assert routes == Counter(PIN_CASE_DATA["expected_suite_routes"])


def test_opsin_generated_pin_graphs_are_unique_additions_to_the_suite():
    suite_graphs = {Chem.MolToSmiles(Chem.MolFromSmiles(smiles)) for smiles in SMILES}
    generated_graphs = [Chem.MolToSmiles(Chem.MolFromSmiles(case["smiles"])) for case in OPSIN_GENERATED_PIN_CASES]

    assert len(generated_graphs) == len(set(generated_graphs))
    assert not suite_graphs.intersection(generated_graphs)


@pytest.mark.parametrize(
    "case",
    SUITE_PIN_CASES,
    ids=[case["name"] for case in SUITE_PIN_CASES],
)
def test_fused_parent_suite_confirmed_pin_names(case):
    result = name(case["smiles"], fusion_mode=FusionMode.AUDITED_PIN, include_trace=True)

    assert result.name == case["name"]
    assert _pin_route(result) == "systematic_fusion"


@pytest.mark.parametrize(
    "case",
    OPSIN_GENERATED_PIN_CASES,
    ids=[case["name"] for case in OPSIN_GENERATED_PIN_CASES],
)
def test_opsin_generated_fusion_graphs_are_valid_confirmed_pins(case):
    assert Chem.MolFromSmiles(case["smiles"]) is not None

    result = name(case["smiles"], fusion_mode=FusionMode.AUDITED_PIN, include_trace=True)
    assert result.error is None
    assert result.name == case["name"]
    assert _pin_route(result) == "systematic_fusion"


@pytest.mark.opsin
@pytest.mark.skipif(not opsin_available(), reason="py2opsin/Java is unavailable")
@pytest.mark.parametrize(
    "case",
    OPSIN_GENERATED_PIN_CASES,
    ids=[case["name"] for case in OPSIN_GENERATED_PIN_CASES],
)
def test_opsin_generated_fusion_sources_and_openclatura_names_roundtrip(case):
    source_check = verify_with_opsin(case["source_name"], case["smiles"])
    assert source_check.ok, source_check.to_dict()

    result = name(
        case["smiles"],
        fusion_mode=FusionMode.AUDITED_PIN,
        verify_opsin=True,
    )
    assert result.name == case["name"]
    assert result.opsin_check is not None
    assert result.opsin_check.ok, result.opsin_check.to_dict()
