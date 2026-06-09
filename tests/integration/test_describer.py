"""Tests for the natural-language describer added in PR4."""

from __future__ import annotations

import json
import subprocess
import sys

from bluenamer import DescribedComponent, Description, describe


def test_describe_returns_structured_description():
    d = describe("CCO")
    assert isinstance(d, Description)
    assert d.smiles == "CCO"
    assert d.name == "ethanol"
    # The summary line names the result.
    assert "ethanol" in d.summary
    # Multiple paragraphs render as text.
    assert "\n\n" in str(d)
    # rules_hit are extracted.
    assert d.rules_hit, "expected at least one P-XX rule id"
    # Components are typed and phase-tagged.
    assert all(isinstance(c, DescribedComponent) for c in d.components)
    assert any(c.phase == "parent_selection" for c in d.components)


def test_describe_explains_functional_group_selection():
    d = describe("CC(=O)Nc1ccccc1")
    text = str(d)
    assert "N-phenylacetamide" in text
    assert "amide" in text
    # Parent selection is mentioned.
    assert "parent skeleton" in text
    # Trace name pieces appear in the prose.
    assert "Name pieces" in text


def test_describe_handles_benzene_without_functional_group():
    d = describe("c1ccccc1")
    assert d.name == "benzene"
    text = str(d)
    assert "benzene" in text
    # No principal group → that fact is stated.
    assert "no nameable principal groups" in text


def test_str_of_description_returns_text():
    s = str(describe("CCO"))
    assert isinstance(s, str)
    assert "ethanol" in s


def test_describe_is_deterministic():
    a = describe("CC(=O)Nc1ccccc1")
    b = describe("CC(=O)Nc1ccccc1")
    assert str(a) == str(b)
    assert a.rules_hit == b.rules_hit


def test_describe_failure_case_still_returns_description():
    """Even when naming yields no name, describe must not raise."""

    d = describe("")
    assert isinstance(d, Description)
    assert "could not be named" in d.summary or d.name == ""


def test_cli_describe_subcommand():
    result = subprocess.run(
        [sys.executable, "-m", "bluenamer.cli", "describe", "CCO"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "ethanol" in result.stdout
    assert "RDKit parsed" in result.stdout


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
    assert isinstance(payload["paragraphs"], list)
    assert payload["paragraphs"]
    assert isinstance(payload["components"], list)
    assert any(c["phase"] == "parent_selection" for c in payload["components"])
