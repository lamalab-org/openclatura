"""Typed operations used before final name rendering.

These objects are the migration path away from late string rewriting.  They
capture *what* the graph requires, while renderers decide the final spelling
and suffix ordering from data-backed rules.
"""

from dataclasses import dataclass
from typing import ClassVar


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
    bond_ids: tuple[int, ...] = ()
    operation_kind: str = "indicated_hydrogen"


@dataclass(frozen=True)
class UnsaturationOperation(NameOperation):
    """An observed parent multiple bond not implied by its parent hydride."""

    locants: tuple[str, str] = ("", "")
    atom_ids: tuple[int, int] = (0, 0)
    bond_id: int = 0
    bond_order: int = 2


@dataclass(frozen=True)
class OxoOperation(NameOperation):
    """An exocyclic oxo group attached to a parent-hydride atom."""

    external_atom_symbol: ClassVar[str] = "O"
    locant: str = ""
    parent_atom_id: int = 0
    oxygen_atom_id: int = 0
    bond_id: int = 0

    @property
    def external_atom_id(self) -> int:
        return self.oxygen_atom_id


@dataclass(frozen=True)
class IminoOperation(NameOperation):
    """An exocyclic imino bond, separate from the nitrogen's other ligands."""

    external_atom_symbol: ClassVar[str] = "N"
    locant: str = ""
    parent_atom_id: int = 0
    nitrogen_atom_id: int = 0
    bond_id: int = 0

    @property
    def external_atom_id(self) -> int:
        return self.nitrogen_atom_id


@dataclass(frozen=True)
class AlkylideneOperation(NameOperation):
    """A parent-to-carbon double bond; carbon ligands retain separate ownership."""

    external_atom_symbol: ClassVar[str] = "C"
    locant: str = ""
    parent_atom_id: int = 0
    carbon_atom_id: int = 0
    bond_id: int = 0

    @property
    def external_atom_id(self) -> int:
        return self.carbon_atom_id
