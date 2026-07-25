"""OPSIN-free end-to-end self-audit driven from a SMILES string.

This ties the reconstruction audit into the naming pipeline without any external
dependency (no Java, no OPSIN).  It installs a capture hook so every top-level
component the namer produces is rebuilt from its name and compared to the input,
then aggregates the per-component verdicts.

Usage::

    from openclatura.audit import self_audit
    result = self_audit("CC(=O)O")     # -> ReconstructionAudit(verdict="confirmed")
"""

from __future__ import annotations

import sys
from contextlib import contextmanager

from .reconstruction import ReconstructionAudit, audit_component_reconstruction


def _impl_code():
    """The code object of the real ``name_component`` implementation.

    Counted rather than the same-named wrapper in ``namer`` so the depth
    reflects genuine recursion into the component namer, not the one-hop
    delegation the wrapper adds.
    """

    from .. import component_namer as cn

    return cn.name_component.__code__


def _naming_recursion_depth() -> int:
    """Count active component-namer frames on the stack.

    The namer recurses into the component namer for substituents, spiro sides,
    and other decompositions; a component is *top-level* only when exactly one
    such frame is active.  Filtering on this keeps the self-audit from comparing
    against the transient intermediate graphs those recursions operate on.
    """

    target = _impl_code()
    depth = 0
    frame = sys._getframe()
    while frame is not None:
        if frame.f_code is target:
            depth += 1
        frame = frame.f_back
    return depth

# Verdict precedence when several components disagree: a single refuted or
# unnamed component condemns the whole molecule; otherwise we can only claim
# confirmation when every component was positively confirmed.
_PRECEDENCE = {"error": 3, "mismatch": 2, "abstained": 1, "confirmed": 0}


@contextmanager
def capture_component_audits():
    """Context manager yielding a list that fills with one
    :class:`ReconstructionAudit` per top-level component named inside the block.

    Installs and restores :data:`openclatura.component_namer.COMPONENT_AUDIT_HOOK`.
    """

    from .. import component_namer as cn
    from .. import graph_io

    collected: list[ReconstructionAudit] = []

    def hook(mol, component_atoms, parts) -> None:
        if getattr(parts, "is_substituent", False):
            return  # substituent components are covered by their parent's graft
        if _naming_recursion_depth() != 1:
            return  # nested recursion works on transient intermediate graphs
        collected.append(audit_component_reconstruction(mol, parts, set(component_atoms)))

    previous = cn.COMPONENT_AUDIT_HOOK
    cn.COMPONENT_AUDIT_HOOK = hook
    # Independent modern-CIP labelling is audit-only overhead; enable it for the
    # duration of the block so the stereo gate has an oracle to verify against.
    previous_cip = graph_io._AUDIT_CIP_ENABLED
    graph_io.set_audit_cip_enabled(True)
    try:
        yield collected
    finally:
        cn.COMPONENT_AUDIT_HOOK = previous
        graph_io.set_audit_cip_enabled(previous_cip)


def aggregate_audits(audits: list[ReconstructionAudit]) -> ReconstructionAudit:
    """Combine per-component audits into a single molecule-level verdict."""

    if not audits:
        return ReconstructionAudit(verdict="abstained", reason="no auditable component produced")
    worst = max(audits, key=lambda a: _PRECEDENCE.get(a.verdict, 1))
    if worst.verdict == "confirmed":
        return worst
    reasons = "; ".join(f"[{a.verdict}] {a.reason}" for a in audits if a.verdict != "confirmed")
    return ReconstructionAudit(
        verdict=worst.verdict,
        reason=reasons,
        reference_smiles=worst.reference_smiles,
        reconstructed_smiles=worst.reconstructed_smiles,
        coverage=worst.coverage,
        stereo=worst.stereo,
        charge_pairs=worst.charge_pairs,
    )


def self_audit(smiles: str) -> ReconstructionAudit:
    """Name ``smiles`` and audit every component by OPSIN-free reconstruction.

    Returns the aggregate :class:`ReconstructionAudit`.  Never raises: a naming
    failure surfaces as ``verdict="error"``.
    """

    import openclatura as oc

    try:
        with capture_component_audits() as collected:
            result = oc.name(smiles)
    except Exception as exc:  # pragma: no cover - naming boundary
        return ReconstructionAudit(verdict="error", reason=f"{type(exc).__name__}: {exc}")
    if not result.ok:
        return ReconstructionAudit(verdict="error", reason=result.error or "naming produced no name")
    return aggregate_audits(list(collected))


__all__ = ["self_audit", "capture_component_audits", "aggregate_audits"]
