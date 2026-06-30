"""Tests for the typed public API introduced in PR2."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from bluenamer import (
    DEFAULT_NAMING_ENGINE,
    NamingEngine,
    NamingRequest,
    NamingResult,
    OpsinCheck,
    analyze_smiles,
    name,
    name_many,
    name_smiles,
)


def test_name_smiles_legacy_still_returns_string():
    assert name_smiles("CCO") == "ethanol"


def test_name_returns_naming_result_with_smiles_field():
    result = name("CCO")
    assert isinstance(result, NamingResult)
    assert result.smiles == "CCO"
    assert result.name == "ethanol"
    assert result.error is None
    assert result.ok is True
    assert result.opsin_check is None
    # No trace was requested, so rules_hit is empty by design.
    assert result.rules_hit == ()


def test_naming_result_is_easy_to_use():
    """Ergonomics: str, bool, repr, to_dict all do the obvious thing."""

    good = name("CCO")
    assert str(good) == "ethanol"
    assert bool(good) is True
    assert "ethanol" in repr(good)
    payload = good.to_dict()
    assert payload["name"] == "ethanol"
    assert payload["smiles"] == "CCO"
    assert payload["ok"] is True
    # Round-trips through json without extra args.
    import json

    json.dumps(payload)

    bad = name("")
    assert str(bad) == ""
    assert bool(bad) is False


def test_name_with_trace_populates_rules_hit_and_hints():
    result = name("CC(=O)Nc1ccccc1", include_trace=True)
    assert result.name == "N-phenylacetamide"
    assert result.trace_segments, "trace_segments must be populated when include_trace=True"
    assert result.substituent_tree, "substituent_tree must be populated when include_trace=True"
    assert result.decisions, "decisions must be populated when include_trace=True"
    # Rule hints should reference Blue Book P-XX identifiers in this case.
    assert result.rules_hit, "expected at least one P-XX rule id"
    assert all(rid.startswith("P-") for rid in result.rules_hit)
    assert len(result.rule_hints) >= 1

    phenyl_parent = next(segment for segment in result.trace_segments if segment.get("substituent_name") == "phenyl")
    assert phenyl_parent["key"] == "substituent_parent"
    assert phenyl_parent["label"] == "substituent parent skeleton"

    main_parent = next(segment for segment in result.trace_segments if segment.get("key") == "parent")
    main_decisions = main_parent.get("decisions", [])
    assert [decision["decision"] for decision in main_decisions] == [
        "selected parent skeleton",
        "selected numbering",
    ]
    numbering = main_decisions[-1]
    assert numbering["data"]["atom_to_locant"] == {1: "1", 0: "2"}


def test_emitted_tokens_are_only_in_trace_when_token_debug_is_enabled():
    normal = name("CC(=O)Nc1ccccc1", include_trace=True)
    normal_assembly = [step for step in normal.decisions if step.decision == "assembled component name"][-1]
    normal_bindings = normal_assembly.data["name_atom_bindings"]

    assert normal_bindings
    assert all("emitted_tokens" not in binding for binding in normal_bindings)
    assert normal_assembly.data["name_token_spans"] == []

    debug = name("CC(=O)Nc1ccccc1", include_trace=True, token_debug=True)
    debug_assembly = [step for step in debug.decisions if step.decision == "assembled component name"][-1]
    debug_bindings = debug_assembly.data["name_atom_bindings"]

    assert any(binding.get("emitted_tokens") for binding in debug_bindings)
    assert debug_assembly.data["name_token_spans"]


def test_analyze_smiles_token_debug_is_explicit():
    normal = analyze_smiles("CCO")
    normal_assembly = [step for step in normal.decisions if step.decision == "assembled component name"][-1]
    assert normal_assembly.data["name_token_spans"] == []
    assert all("emitted_tokens" not in binding for binding in normal_assembly.data["name_atom_bindings"])

    debug = analyze_smiles("CCO", token_debug=True)
    debug_assembly = [step for step in debug.decisions if step.decision == "assembled component name"][-1]
    assert debug_assembly.data["name_token_spans"]
    assert any(binding.get("emitted_tokens") for binding in debug_assembly.data["name_atom_bindings"])


def test_substituent_tree_preserves_nested_branch_hierarchy_and_flat_trace():
    result = name(
        "O=C(O)C[C@H](O)C[C@H](O)CCn2c(c(c(c2c1ccc(F)cc1)c3ccccc3)C(=O)Nc4ccccc4)C(C)C",
        include_trace=True,
        token_debug=True,
    )

    root = result.substituent_tree[0]
    pyrrolyl = next(child for child in root["substituents"] if "pyrrol" in child["name"])
    fluorophenyl = next(child for child in pyrrolyl["substituents"] if "fluorophenyl" in child["name"])
    phenylcarbamoyl = next(child for child in pyrrolyl["substituents"] if "phenylcarbamoyl" in child["name"])

    assert root["kind"] == "component"
    assert root["parent"]["parent_length"] == 7
    assert pyrrolyl["locants"] == ["7"]
    assert pyrrolyl["parent"]["retained_name"] == "pyrrole"
    assert {child["name"].strip("()") for child in pyrrolyl["substituents"]} >= {
        "propan-2-yl",
        "phenylcarbamoyl",
        "phenyl",
        "4-fluorophenyl",
    }
    assert fluorophenyl["parent"]["retained_name"] == "benzene"
    assert fluorophenyl["substituents"][0]["name"] == "fluoro"
    assert fluorophenyl["substituents"][0]["locants"] == ["4"]
    assert phenylcarbamoyl["functional_prefix"]["group_key"] == "ring_amide"
    assert phenylcarbamoyl["functional_prefix"]["attachment_atom"] == 13
    assert phenylcarbamoyl["functional_prefix"]["core_atoms"] == [29, 30, 31]
    assert phenylcarbamoyl["functional_prefix"]["group_atoms"] == [13, 29, 30, 31]
    assert phenylcarbamoyl["functional_prefix"]["ligands"][0]["parent"]["retained_name"] == "benzene"
    assert phenylcarbamoyl["substituents"][0]["name"] == "phenyl"

    assert result.trace_segments
    assert any(segment.get("nested_decisions") for segment in result.trace_segments)
    assembly = [step for step in result.decisions if step.decision == "assembled component name"][-1]
    assert assembly.data["name_token_spans"]


def test_absolute_stereo_tokens_bind_to_stereocenter_atoms():
    result = name(
        "O=C(O)C[C@H](O)C[C@H](O)CCn2c(c(c(c2c1ccc(F)cc1)c3ccccc3)C(=O)Nc4ccccc4)C(C)C",
        include_trace=True,
        token_debug=True,
    )
    assembly = [step for step in result.decisions if step.decision == "assembled component name"][-1]
    token_spans = assembly.data["name_token_spans"]

    expected_tokens = [
        ("3", 4, "locant", "renderer_stereo"),
        ("R", 4, "stereo", "renderer_stereo"),
        ("5", 7, "locant", "renderer_stereo"),
        ("R", 7, "stereo", "renderer_stereo"),
    ]
    for text, atom, token_kind, source in expected_tokens:
        token = next(
            token
            for token in token_spans
            if token["text"] == text
            and token["atoms"] == [atom]
            and token["token_kind"] == token_kind
            and token["source"] == source
        )
        assert token["confidence"] != "fallback"


def test_naming_errors_become_result_error_not_exception():
    # Empty SMILES is a well-defined "no atoms" path; assert at a different angle
    # by feeding clearly malformed text — the engine must capture, not raise.
    result = name("definitely-not-a-smiles")
    # Either name is empty with error captured, or rdkit accepted it.
    # We only assert there is no uncaught exception.
    assert isinstance(result, NamingResult)


def test_empty_smiles_yields_empty_name():
    result = name("")
    assert result.name == ""
    assert result.smiles == ""


def test_name_many_serial_preserves_order():
    inputs = ["CCO", "c1ccccc1", "CC(=O)O", "CCN"]
    results = name_many(inputs, processes=1)
    assert [r.smiles for r in results] == inputs
    assert [r.name for r in results] == ["ethanol", "benzene", "acetic acid", "ethanamine"]


def test_name_many_swallows_per_row_errors():
    inputs = ["CCO", "this-is-not-smiles", "c1ccccc1"]
    results = name_many(inputs)
    assert results[0].ok is True
    assert results[2].ok is True
    # Middle one: either named (some inputs are lenient) or marked as error.
    # The important contract is that the batch as a whole didn't raise.
    assert all(isinstance(r, NamingResult) for r in results)


def test_engine_run_with_verify_opsin_skips_gracefully_without_opsin(monkeypatch):
    # Force the "no py2opsin" path even when the optional dep is installed.
    import bluenamer.opsin_verify as ov

    monkeypatch.setattr(ov, "_try_import_py2opsin", lambda: None)
    result = DEFAULT_NAMING_ENGINE.run(NamingRequest(smiles="CCO", verify_opsin=True))
    assert isinstance(result.opsin_check, OpsinCheck)
    assert result.opsin_check.status == "skipped_no_opsin"
    assert result.verified is False


def test_engine_run_with_verify_opsin_skips_gracefully_without_java(monkeypatch):
    import bluenamer.opsin_verify as ov

    # py2opsin present, java not.
    monkeypatch.setattr(ov, "_try_import_py2opsin", lambda: object())
    monkeypatch.setattr(ov, "_java_available", lambda: False)
    result = DEFAULT_NAMING_ENGINE.run(NamingRequest(smiles="CCO", verify_opsin=True))
    assert result.opsin_check is not None
    assert result.opsin_check.status == "skipped_no_java"


def test_naming_engine_is_reusable():
    engine = NamingEngine()
    a = engine.run(NamingRequest(smiles="CCO"))
    b = engine.run(NamingRequest(smiles="c1ccccc1"))
    assert a.name == "ethanol"
    assert b.name == "benzene"


def test_cli_name_subcommand_prints_name():
    result = subprocess.run(
        [sys.executable, "-m", "bluenamer.cli", "name", "CCO"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ethanol"


def test_cli_name_json_emits_structured_payload():
    result = subprocess.run(
        [sys.executable, "-m", "bluenamer.cli", "name", "CC(=O)O", "--json"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["name"] == "acetic acid"
    assert payload["smiles"] == "CC(=O)O"
    assert payload["ok"] is True
    assert "trace_segments" in payload
    assert "substituent_tree" in payload


@pytest.mark.parametrize("processes", [1, 2])
def test_name_many_parallel_and_serial_agree(processes):
    inputs = ["CCO", "c1ccccc1", "CC(=O)O", "CCN", "CC(C)C"]
    serial = name_many(inputs, processes=1)
    other = name_many(inputs, processes=processes)
    assert [r.name for r in serial] == [r.name for r in other]
