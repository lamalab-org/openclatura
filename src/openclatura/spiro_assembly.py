"""Structured spiro assembly data shared by planning and rendering."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .assembly_parts import AssemblyParts, SubstituentItem


@dataclass(frozen=True)
class SpiroAssembly:
    parent_locant: str
    side_locant: str
    side_parent_name: str
    side_prefixes: tuple[str, ...] = ()
    side_suffixes: tuple[tuple[str, str], ...] = ()
    # (primed locant, descriptor) pairs the side component contributes to the
    # whole name's leading stereo group.
    side_stereo: tuple[tuple[str, str], ...] = ()
    side_substituents: tuple["SubstituentItem", ...] = ()
    # Unprimed component-local proof and typed operations, before projection.
    side_parts: "AssemblyParts | None" = None
    side_prime: str = "'"
    continuation: "SpiroAssembly | None" = None
