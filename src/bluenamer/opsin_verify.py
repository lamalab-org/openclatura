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

    def __bool__(self) -> bool:
        return self.ok

    def __str__(self) -> str:
        return self.status

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "name": self.name,
            "canonical_original": self.canonical_original,
            "opsin_smiles": self.opsin_smiles,
            "canonical_roundtrip": self.canonical_roundtrip,
            "error_message": self.error_message,
        }


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


def opsin_available() -> bool:
    """``True`` iff both ``py2opsin`` and a Java runtime are usable."""

    return _try_import_py2opsin() is not None and _java_available()


def _canonicalize(smiles: str) -> str | None:
    if not smiles:
        return None
    try:
        from rdkit.Chem import CanonSmiles  # type: ignore
    except Exception:  # pragma: no cover - rdkit always available
        return None
    try:
        return CanonSmiles(smiles)
    except Exception:
        return None


def verify_with_opsin(name: str, smiles: str) -> OpsinCheck:
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
            error_message="Failed to canonicalize SMILES for comparison.",
        )

    if canonical_original == canonical_roundtrip:
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


__all__ = ["OpsinCheck", "OpsinStatus", "opsin_available", "verify_with_opsin"]
