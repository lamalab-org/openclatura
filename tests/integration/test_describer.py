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


def test_describe_exposes_token_binding_summary():
    d = describe("CCO")
    payload = d.to_dict()

    assert payload["token_summary"]["total"] >= 1
    assert payload["token_summary"]["fallback_tokens"] == []
    assert payload["token_spans"]
    assert any(token["atoms"] for token in payload["token_spans"])
    assert "Name-token graph bindings" in str(d)


def test_describe_renders_nested_substituent_tree_from_metadata():
    d = describe("CC(=O)Nc1ccccc1")
    text = str(d)

    assert d.substituent_tree
    assert "Component and substituent structure" in text
    assert "Substituent" in text
    assert "phenyl" in text
    assert "retained as benzene" in text


def test_describe_orders_main_parent_trace_before_nested_substituent_trace():
    text = str(describe("CC(=O)Nc1ccccc1"))

    assert "The final assembled name is **N-phenylacetamide**." not in text
    assert "Component assembled as **N-phenylacetamide**" not in text
    eth_index = text.index('contributes "eth"')
    amide_index = text.index('contributes "amide"')
    benzene_index = text.index('contributes "benzene"')
    assert eth_index < amide_index < benzene_index


def test_describe_explains_parenthesized_unsaturation_locants():
    text = str(describe("CN1C=NC2=C1C(=O)N(C(=O)N2C)C"))

    assert "Unsaturation: double at 1(6),8" in text
    assert "1(6) means a multiple bond between locants 1 and 6" in text


def test_describe_renders_oxygen_carbonyl_shortcut_substituent_tree():
    text = str(
        describe("C1[C@H]([C@@H]([C@@H](C=C1C(=O)O)OC(=O)/C=C/C2=CC(=C(C=C2)O)O)O)O")
    )

    assert "Substituent at 3: ((1E)-2-(3,4-dihydroxyphenyl)ethenylcarbonyloxy)" in text
    assert "Substituent: ethenyl covers 2 atoms and 1 bond" in text
    assert "Substituent: dihydroxyphenyl covers 8 atoms and 8 bonds" in text
    assert text.count("Substituent: hydroxy covers 1 atom and 1 bond") >= 2


def test_describe_renders_heteroatom_shortcut_ligand_tree():
    text = str(describe("O=S(=O)(Nc1nc(cc(n1)C)C)c2ccc(N)cc2"))

    assert "Substituent at N: (4-aminophenylsulfonyl)" in text
    assert "Substituent: 4-aminophenyl covers 7 atoms and 7 bonds" in text
    assert "Parent: ring parent with 6 atoms retained as benzene" in text
    assert "Substituent at 4: amino covers 1 atom" in text


def test_describe_carbonylamino_shortcut_does_not_render_carbonyl_oxygen_as_hydroxy():
    text = str(describe("CC(C)C[C@@H](C(=O)N[C@@H](CC(C)C)C(=O)N[C@@H](CC(C)C)C(=O)O)N"))

    assert "Substituent: carbonyl covers 2 atoms and 1 bond" in text
    assert "Substituent at 1: (1S)-1-amino-3-methylbutylcarbonylamino covers 9 atoms and 8 bonds" in text
    assert "Substituent: (1S)-1-amino-3-methylbutylcarbonyl covers 8 atoms and 7 bonds" in text
    assert "Substituent: hydroxy covers 1 atom and 1 bond" not in text
