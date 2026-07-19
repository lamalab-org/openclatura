"""Tests for the whole-graph reconstruction audit."""

import pytest

import openclatura.component_namer as component_namer
from openclatura import analyze_smiles
from openclatura.reconstruction_audit import audit_component_reconstruction


def _trace_audit(smiles: str) -> dict | None:
    audit = None
    for step in analyze_smiles(smiles).decisions:
        data = getattr(step, "data", None) or {}
        if "reconstruction_audit" in data:
            audit = data["reconstruction_audit"]
    return audit


@pytest.mark.parametrize(
    "smiles",
    [
        "CC=CCCCO",  # chain parent + suffix + unsaturation
        "Oc1ccncc1",  # retained heteroaromatic parent
        "C1CC2CCCC2C1",  # von Baeyer bicycle
        "CC(C)CC(=O)O",  # substituent + carboxylic acid
        "CC(=O)Oc1ccccc1C(=O)O",  # multiple characteristic groups
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # fused heterobicycle (caffeine)
        "N=S1C=CC1",  # lambda-convention heteroatom replacement
    ],
)
def test_reconstruction_audit_matches_supported_names(smiles):
    audit = _trace_audit(smiles)
    assert audit is not None
    assert audit["status"] == "matched", audit["issues"]


def test_reconstruction_audit_skips_charged_components():
    audit = _trace_audit("CC[N+](C)(C)C")
    assert audit is not None
    assert audit["status"] == "skipped"
    assert audit["issues"] == ["charged component not modeled"]


@pytest.fixture
def captured_parts(monkeypatch):
    """Name a molecule and hand its (mol, parts) to the test for tampering."""

    captured = {}
    original = component_namer.audit_component_reconstruction

    def capture(mol, parts):
        captured["mol"], captured["parts"] = mol, parts
        return original(mol, parts)

    monkeypatch.setattr(component_namer, "audit_component_reconstruction", capture)

    from openclatura import name_smiles

    name_smiles("CC(C)CC(=O)O")  # 3-methylbutanoic acid
    return captured["mol"], captured["parts"]


def test_reconstruction_audit_flags_wrong_substituent_locant(captured_parts):
    mol, parts = captured_parts
    assert audit_component_reconstruction(mol, parts).status == "matched"

    parts.substituents[0].locants = ["2"]
    result = audit_component_reconstruction(mol, parts)
    assert result.status == "mismatched"
    assert any("claims locants" in issue for issue in result.issues)


def test_reconstruction_audit_flags_phantom_unsaturation(captured_parts):
    mol, parts = captured_parts
    parts.parent_bond_orders_by_locants[("2", "3")] = 2
    result = audit_component_reconstruction(mol, parts)
    assert result.status == "mismatched"
    assert any("differs from input" in issue for issue in result.issues)
