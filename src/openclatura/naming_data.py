"""Data-backed nomenclature lookups used by :mod:`openclatura.namer`.

The tables loaded here keep extendable naming vocabulary out of the graph
algorithms.  Each JSON entry carries a Blue Book rule reference so new entries
can be reviewed against the relevant IUPAC recommendation before use.
"""

from __future__ import annotations

import json
from functools import cache, lru_cache
from pathlib import Path
from typing import Any

from .rule_layout import RuleGroupView, rule_groups

DATA_DIR = Path(__file__).with_name("data")


@cache
def load_json_table(filename: str) -> dict[str, Any]:
    """Load an extendable JSON nomenclature table.

    Blue Book references: P-12 (operations in nomenclature), P-13 (operations
    in names), and the chapter-specific rule references stored with each row.
    """

    with (DATA_DIR / filename).open(encoding="utf-8") as handle:
        return json.load(handle)


@lru_cache(maxsize=1)
def namer_rules() -> dict[str, Any]:
    """Return the data table for recursive SMILES-to-name assembly decisions.

    Blue Book references are stored under ``bluebook_rule`` in each JSON section.
    """

    return load_json_table("namer_rules.json")


@lru_cache(maxsize=1)
def grouped_namer_rules() -> dict[str, RuleGroupView]:
    """Return ``namer_rules.json`` as named domain groups."""

    return rule_groups(namer_rules())


def values(section: str) -> list[Any]:
    """Return a list-valued rule section from ``namer_rules.json``.

    Blue Book references: the selected section's ``bluebook_rule`` metadata.
    """

    return namer_rules()[section]["values"]


def mapping(section: str) -> dict[str, Any]:
    """Return a mapping-valued rule section from ``namer_rules.json``.

    Blue Book references: the selected section's ``bluebook_rule`` metadata.
    """

    return namer_rules()[section]["values"]
