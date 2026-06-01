"""Owned parser-grammar data used by production gates."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from .naming_data import load_json_table


@dataclass(frozen=True)
class RetainedFusedDerivativeGate:
    production_parent_names: frozenset[str]
    audit_only_parent_names: frozenset[str]
    allowed_principal_keys: frozenset[str | None]
    allowed_group_keys: frozenset[str]
    allowed_substituent_names: frozenset[str]


@dataclass(frozen=True)
class RetainedFusedToken:
    parent_name: str
    parent_stems: tuple[str, ...]
    substituent_stems: tuple[str, ...]
    fusion_stems: tuple[str, ...]
    derivative_status: str
    default_indicated_h: tuple[str, ...]


@lru_cache(maxsize=1)
def local_grammar_snapshot() -> dict[str, Any]:
    """Return the checked-in grammar snapshot after schema validation."""

    snapshot = load_json_table("parser_grammar_snapshot.json")
    _validate_snapshot(snapshot)
    return snapshot


@lru_cache(maxsize=1)
def retained_fused_tokens() -> dict[str, RetainedFusedToken]:
    """Return parser-visible retained fused stems keyed by emitted parent."""

    tokens: dict[str, RetainedFusedToken] = {}
    for parent_name, raw in local_grammar_snapshot()["retained_fused_tokens"].items():
        tokens[str(parent_name)] = RetainedFusedToken(
            parent_name=str(parent_name),
            parent_stems=tuple(str(value) for value in raw.get("parent_stems", ())),
            substituent_stems=tuple(str(value) for value in raw.get("substituent_stems", ())),
            fusion_stems=tuple(str(value) for value in raw.get("fusion_stems", ())),
            derivative_status=str(raw.get("derivative_status", "audit_only")),
            default_indicated_h=tuple(str(value) for value in raw.get("default_indicated_h", ())),
        )
    return tokens


def retained_fused_token(parent_name: str) -> RetainedFusedToken | None:
    """Return one retained fused token row, if present."""

    return retained_fused_tokens().get(parent_name)


@lru_cache(maxsize=1)
def retained_fused_derivative_gate() -> RetainedFusedDerivativeGate:
    """Return the retained-fused derivative production gate.

    The gate is intentionally narrower than the token registry: indicated-H
    and mancude retained parents stay audit-only until derivative numbering
    and round-trip classes are verified.
    """

    grammar = local_grammar_snapshot()
    raw = grammar["retained_fused_derivative_gate"]
    tokens = retained_fused_tokens()
    production_parent_names = frozenset(
        name for name, token in tokens.items() if token.derivative_status == "production_safe"
    )
    audit_only_parent_names = frozenset(
        name for name, token in tokens.items() if token.derivative_status == "audit_only"
    )
    _assert_gate_list_matches_tokens(raw, "production_parent_names", production_parent_names)
    _assert_gate_list_matches_tokens(raw, "audit_only_parent_names", audit_only_parent_names)
    return RetainedFusedDerivativeGate(
        production_parent_names=production_parent_names,
        audit_only_parent_names=audit_only_parent_names,
        allowed_principal_keys=frozenset(None if value is None else str(value) for value in raw["allowed_principal_keys"]),
        allowed_group_keys=frozenset(str(value) for value in raw["allowed_group_keys"]),
        allowed_substituent_names=frozenset(str(value) for value in raw["allowed_substituent_names"]),
    )


def retained_fused_token_status(parent_name: str) -> str | None:
    """Return the production status for a retained fused token."""

    token = retained_fused_token(parent_name)
    if token is None:
        return None
    return token.derivative_status


def oxoacid_ester_suffix_templates() -> dict[str, Any]:
    """Return local suffix templates for oxoacid ester roles."""

    return local_grammar_snapshot()["oxoacid_ester_suffix_templates"]


def _validate_snapshot(snapshot: dict[str, Any]) -> None:
    required_sections = {
        "retained_fused_derivative_gate",
        "retained_fused_tokens",
        "charge_suffixes",
        "hetero_replacement_priority",
        "halogen_oxoacid_common_names",
        "oxoacid_ester_suffix_templates",
    }
    missing = sorted(required_sections - set(snapshot))
    if missing:
        raise ValueError(f"local grammar snapshot is missing required sections: {missing}")

    source = snapshot.get("source", {})
    if source.get("resource_root") or source.get("external_resource_path"):
        raise ValueError("local grammar snapshot must not reference an external resource path")
    if source.get("resource_json_dir") != "parser_xml_resources":
        raise ValueError("local grammar snapshot must point at owned parser_xml_resources JSON data")

    charge_suffixes = snapshot["charge_suffixes"]
    canonical_suffixes = set(charge_suffixes.get("canonical", ()))
    required_suffixes = {"ium", "ide", "ylium", "uide"}
    if not required_suffixes.issubset(canonical_suffixes):
        raise ValueError(
            "local grammar snapshot charge_suffixes.canonical must include "
            f"{sorted(required_suffixes)}; found {sorted(canonical_suffixes)}"
        )

    tokens = snapshot["retained_fused_tokens"]
    if not isinstance(tokens, dict) or not tokens:
        raise ValueError("local grammar snapshot retained_fused_tokens must be a non-empty mapping")
    for parent_name, token in tokens.items():
        status = token.get("derivative_status")
        if status not in {"production_safe", "audit_only"}:
            raise ValueError(f"retained fused token {parent_name!r} has invalid derivative_status {status!r}")
        if not token.get("parent_stems"):
            raise ValueError(f"retained fused token {parent_name!r} must define at least one parent stem")

    gate = snapshot["retained_fused_derivative_gate"]
    for key in ("production_parent_names", "audit_only_parent_names", "allowed_principal_keys", "allowed_group_keys", "allowed_substituent_names"):
        if key not in gate:
            raise ValueError(f"local grammar snapshot retained_fused_derivative_gate is missing {key!r}")

    oxo_templates = snapshot["oxoacid_ester_suffix_templates"]
    for key in ("phosphate", "charge_normalized_halogen_oxoester", "charge_normalized_halogen_peroxy_oxoester"):
        if key not in oxo_templates:
            raise ValueError(f"local grammar snapshot oxoacid_ester_suffix_templates is missing {key!r}")


def _assert_gate_list_matches_tokens(raw: dict[str, Any], key: str, derived_names: frozenset[str]) -> None:
    configured = frozenset(str(value) for value in raw.get(key, ()))
    if configured and configured != derived_names:
        missing = sorted(derived_names - configured)
        extra = sorted(configured - derived_names)
        raise ValueError(
            f"retained fused derivative gate {key} does not match retained_fused_tokens: "
            f"missing={missing}, extra={extra}"
        )
