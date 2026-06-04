"""Optional OPSIN-based round-trip verification.

`OpsinCheck` carries the outcome of feeding a generated IUPAC name back
through OPSIN and comparing the canonical SMILES to the input. The
helper degrades gracefully (returns a ``skipped_*`` status) when
``py2opsin`` or a Java runtime are unavailable, so callers can request
verification without having to gate on the optional dependency.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from typing import Literal
from bluenamer.utils import standardize_mol

from .resonance_compare import canonical_smiles, equivalent_smiles

OpsinStatus = Literal[
    "matched",
    "mismatched",
    "name_unparseable",
    "name_empty",
    "skipped_no_opsin",
    "skipped_no_java",
    "error",
]


@dataclass(frozen=True)
class OpsinCheck:
    """Outcome of an OPSIN round-trip on a generated IUPAC name."""

    status: OpsinStatus
    name: str
    canonical_original: str | None = None
    opsin_smiles: str | None = None
    canonical_roundtrip: str | None = None
    error_message: str | None = None

    @property
    def ok(self) -> bool:
        """Whether the round-trip matched the input."""

        return self.status == "matched"


def _try_import_py2opsin():
    try:
        import py2opsin
    except Exception:  # pragma: no cover - optional dependency
        return None
    return py2opsin


def _java_available() -> bool:
    if shutil.which("java") is None:
        return False
    try:
        subprocess.run(
            ["java", "-version"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, OSError):
        return False
    return True


def _canonicalize(smiles: str) -> str | None:
    return canonical_smiles(smiles)


def verify_with_opsin(name: str, smiles: str, standardize_smiles: bool = True) -> OpsinCheck:
    """Round-trip ``name`` through OPSIN and compare to ``smiles``.

    Returns an ``OpsinCheck`` with one of the documented ``status`` values.
    Never raises for missing-Java / missing-py2opsin / unparseable name —
    those become explicit, non-``ok`` statuses so the caller can
    introspect.
    """

    if not name:
        return OpsinCheck(status="name_empty", name=name)

    py2opsin = _try_import_py2opsin()
    if py2opsin is None:
        return OpsinCheck(status="skipped_no_opsin", name=name)
    if not _java_available():
        return OpsinCheck(status="skipped_no_java", name=name)

    if standardize_smiles:

        canonical_original = standardize_mol(smiles)
    else:
        canonical_original = _canonicalize(smiles)

    try:
        decoded = py2opsin.py2opsin([name])
    except Exception as exc:  # pragma: no cover - py2opsin internal
        return OpsinCheck(
            status="error",
            name=name,
            canonical_original=canonical_original,
            error_message=str(exc),
        )

    if not decoded or not decoded[0]:
        return OpsinCheck(
            status="name_unparseable",
            name=name,
            canonical_original=canonical_original,
        )

    opsin_smiles = decoded[0]
    canonical_roundtrip = _canonicalize(opsin_smiles)

    if canonical_original is None or canonical_roundtrip is None:
        return OpsinCheck(
            status="error",
            name=name,
            canonical_original=canonical_original,
            opsin_smiles=opsin_smiles,
            canonical_roundtrip=canonical_roundtrip,
            error_message= "Failed to standardize SMILES for comparison." if standardize_smiles else "Failed to canonicalize SMILES for comparison.",
        )

    if equivalent_smiles(smiles, opsin_smiles):
        return OpsinCheck(
            status="matched",
            name=name,
            canonical_original=canonical_original,
            opsin_smiles=opsin_smiles,
            canonical_roundtrip=canonical_roundtrip,
        )

    return OpsinCheck(
        status="mismatched",
        name=name,
        canonical_original=canonical_original,
        opsin_smiles=opsin_smiles,
        canonical_roundtrip=canonical_roundtrip,
    )


__all__ = ["OpsinCheck", "OpsinStatus", "verify_with_opsin"]
