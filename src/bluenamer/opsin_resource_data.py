"""OPSIN-resource-backed grammar facts used as data gates.

The XML files in OPSIN's resources describe parser-visible stems, charge
suffixes, and common non-carboxylic acid names.  This module keeps those facts
separate from production enablement: a token being parseable by OPSIN does not
mean every derivative of that parent is safe for live naming.
"""

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
def opsin_resource_grammar() -> dict[str, Any]:
    """Return OPSIN-derived grammar facts recorded for local rule gates."""

    return load_json_table("opsin_resource_grammar.json")


@lru_cache(maxsize=1)
def retained_fused_tokens() -> dict[str, RetainedFusedToken]:
    """Return OPSIN-visible retained fused stems keyed by emitted parent name."""

    tokens: dict[str, RetainedFusedToken] = {}
    for parent_name, raw in opsin_resource_grammar()["retained_fused_tokens"].items():
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
    """Return one retained fused OPSIN token row, if present."""

    return retained_fused_tokens().get(parent_name)


@lru_cache(maxsize=1)
def retained_fused_derivative_gate() -> RetainedFusedDerivativeGate:
    """Return the retained-fused derivative production gate.

    The gate is intentionally narrower than the retained-fused token registry:
    indicated-H and mancude retained parents stay audit-only until derivative
    numbering and OPSIN round-trip classes are verified.
    """

    grammar = opsin_resource_grammar()
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


def _assert_gate_list_matches_tokens(raw: dict[str, Any], key: str, derived_names: frozenset[str]) -> None:
    configured = frozenset(str(value) for value in raw.get(key, ()))
    if configured and configured != derived_names:
        missing = sorted(derived_names - configured)
        extra = sorted(configured - derived_names)
        raise ValueError(
            f"retained fused derivative gate {key} does not match retained_fused_tokens: "
            f"missing={missing}, extra={extra}"
        )


def retained_fused_token_status(parent_name: str) -> str | None:
    """Return the OPSIN-resource production status for a retained fused token."""

    token = retained_fused_token(parent_name)
    if token is None:
        return None
    return token.derivative_status


def oxoacid_ester_suffix_templates() -> dict[str, Any]:
    """Return OPSIN-resource-backed suffix templates for oxoacid ester roles."""

    return opsin_resource_grammar()["oxoacid_ester_suffix_templates"]
