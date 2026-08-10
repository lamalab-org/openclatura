"""Post-naming audit subsystem — OPSIN-free verification of generated names.

Groups the modules that verify a generated name against its molecule without any
external dependency (no Java, no OPSIN):

* :mod:`~openclatura.audit.reconstruction` — rebuild the component from the name
  alone and compare it to the input (the main structural audit);
* :mod:`~openclatura.audit.substituent_reconstruction` — recursive,
  structure-independent reconstruction of compositional substituent fragments;
* :mod:`~openclatura.audit.naming` — atom/token coverage bookkeeping;
* :mod:`~openclatura.audit.stereo` — stereochemistry audit;
* :mod:`~openclatura.audit.relative_stereo` — ring ``cis``/``trans`` oracle, for
  names that pin a configuration with a word instead of per-atom ``R``/``S``;
* :mod:`~openclatura.audit.self_audit` — end-to-end SMILES-driven driver.

Two entry points::

    from openclatura.audit import audit, self_audit

    # Given a named component's (mol, parts):
    result = audit(mol, parts)          # -> ReconstructionAudit

    # Or straight from a SMILES (names it, then audits every component):
    result = self_audit("CC(=O)O")      # -> ReconstructionAudit
"""

from __future__ import annotations

from .naming import (
    UnnamedAtomError,
    assert_component_fully_named,
    audit_charge_pair_templates,
    component_named_atom_coverage,
)
from .reconstruction import ReconstructionAudit, audit_component_reconstruction
from .relative_stereo import ring_face_relation
from .self_audit import aggregate_audits, capture_component_audits, self_audit
from .stereo import audit_stereochemistry
from .substituent_reconstruction import resolve_fragment_mol

# Primary entry point: audit a named component by reconstruction.
audit = audit_component_reconstruction

__all__ = [
    "audit",
    "audit_component_reconstruction",
    "ReconstructionAudit",
    "self_audit",
    "capture_component_audits",
    "aggregate_audits",
    "audit_stereochemistry",
    "ring_face_relation",
    "resolve_fragment_mol",
    "UnnamedAtomError",
    "assert_component_fully_named",
    "audit_charge_pair_templates",
    "component_named_atom_coverage",
]
