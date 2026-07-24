"""Typed operations used before final name rendering.

These objects are the migration path away from late string rewriting.  They
capture *what* the graph requires, while renderers decide the final spelling
and suffix ordering from data-backed rules.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class NameOperation:
    """Base metadata shared by structured naming operations."""

    key: str
    reason: str = ""


@dataclass(frozen=True)
class ParentSuffixOperation(NameOperation):
    """A suffix operation applied to the selected parent."""

    locants: tuple[str, ...] = ()
    suffix: str = ""
    charge: int | None = None
    atom_symbols: tuple[str, ...] = ()


@dataclass(frozen=True)
class HydroOperation(NameOperation):
    """An additive hydrogen operation tied to parent locants."""

    locants: tuple[str, ...] = ()
    atom_ids: tuple[int, ...] = ()
    operation_kind: str = "indicated_hydrogen"
