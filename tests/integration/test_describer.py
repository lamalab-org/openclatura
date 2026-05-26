"""Tests for the structure-driven natural-language describer."""

from __future__ import annotations

import json
import subprocess
import sys

from bluenamer import Description, DescriptionFacts, describe


def test_describe_returns_structured_description():
    d = describe("CCO")
    assert isinstance(d, Description)
    assert d.smiles == "CCO"
    assert d.name == "ethanol"
    assert d.text
    assert isinstance(d.facts, tuple)
    assert all(isinstance(f, DescriptionFacts) for f in d.facts)


def test_describe_emits_structural_prose():
    d = describe("CCO")
    # Structure-driven prose: speaks about the parent and substituents.
    assert "parent" in d.text
    assert "hydroxy group" in d.text
    assert "carbon chain" in d.text


def test_describe_benzene_speaks_about_carbocycle_and_double_bonds():
    d = describe("c1ccccc1")
    assert d.name == "benzene"
    assert "6-membered carbocycle" in d.text
    assert "double bond between positions" in d.text


def test_describe_heterocycle_includes_heteroatom_positions():
    d = describe("OCC1OC(O)C(O)C(O)C1O")  # tetrahydropyran sugar
    assert d.name  # the namer should give some non-empty result
    facts = d.facts[0]
    assert any("oxygen" in h for h in facts.heteroatoms)
    assert any("hydroxy" in s for s in facts.substituents)


def test_describe_includes_reconstruction_connectivity():
    d = describe("C1CCCCC1")
    facts = d.facts[0]
    # cyclohexane → 6 bonds in the ring, all single
    assert len(facts.connectivity) == 6
    assert all("single" in c for c in facts.connectivity)


def test_describe_is_deterministic():
    a = describe("CC(=O)Nc1ccccc1")
    b = describe("CC(=O)Nc1ccccc1")
    assert a.text == b.text
    assert a.facts == b.facts


def test_describe_handles_empty_smiles():
    d = describe("")
    assert isinstance(d, Description)
    assert d.text == ""
    assert d.facts == ()


def test_str_of_description_returns_text():
    s = str(describe("CCO"))
    assert isinstance(s, str)
    assert "carbon chain" in s


def test_to_dict_is_json_serialisable():
    payload = describe("CCO").to_dict()
    json.dumps(payload)
    assert payload["name"] == "ethanol"
    assert payload["text"]
    assert isinstance(payload["facts"], list)
    assert payload["facts"][0]["parent_summary"]


def test_cli_describe_subcommand():
    result = subprocess.run(
        [sys.executable, "-m", "bluenamer.cli", "describe", "CCO"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "carbon chain" in result.stdout
    assert "hydroxy group" in result.stdout


def test_cli_describe_json():
    result = subprocess.run(
        [sys.executable, "-m", "bluenamer.cli", "describe", "CCO", "--json"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["name"] == "ethanol"
    assert payload["smiles"] == "CCO"
    assert payload["text"]
    assert isinstance(payload["facts"], list)
