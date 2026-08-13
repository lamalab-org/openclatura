"""Structured spiro assembly data shared by planning and rendering."""

from dataclasses import dataclass


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
