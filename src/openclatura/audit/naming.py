"""Atom/token coverage bookkeeping for a generated component name.

The canonical implementations live in :mod:`openclatura.naming_audit` (imported
by the naming pipeline itself); this module re-exports them so the audit package
presents one coherent surface without duplicating the battle-tested logic.
"""

from __future__ import annotations

from ..naming_audit import (
    ChargePairTemplateAudit,
    NamingCoverage,
    UnnamedAtomError,
    assert_component_fully_named,
    audit_charge_pair_templates,
    component_named_atom_coverage,
)

__all__ = [
    "ChargePairTemplateAudit",
    "NamingCoverage",
    "UnnamedAtomError",
    "assert_component_fully_named",
    "audit_charge_pair_templates",
    "component_named_atom_coverage",
]
