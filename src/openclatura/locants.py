"""Locant parsing and atom/bond locant helpers."""

from __future__ import annotations

import re

from .assembly_utils import parse_locant as parse_locant
from .molecule import Molecule

_SYSTEM_LOCANT_RE = re.compile(r"^(?P<base>[1-9][0-9]*)(?P<suffix>[a-z]*)(?:\^(?P<distance>[1-9][0-9]*))?$")
_SUPERSCRIPT_TO_ASCII = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻", "0123456789-")


class DisplayLocant(int):
    """Integer locant with a separate display representation."""

    def __new__(cls, value: int, display: str | None = None):
        obj = int.__new__(cls, value)
        obj.display = str(value) if display is None else str(display)
        return obj


def as_display_locant(locant: int, display: str | None = None) -> DisplayLocant:
    """Return a display-aware locant while preserving existing display text."""

    if isinstance(locant, DisplayLocant) and display is None:
        return locant
    inherited_display = getattr(locant, "display", None)
    return DisplayLocant(int(locant), display if display is not None else inherited_display)


def locant_text(locant: int, display: str | None = None) -> str:
    """Return the rendered text for a locant."""

    if display is not None:
        return str(display)
    inherited_display = getattr(locant, "display", None)
    return str(locant) if inherited_display is None else str(inherited_display)


def get_atom_locants(oriented_path: list[int], target_indices: set[int]) -> list[int]:
    """Return locants for target atoms in an oriented parent path."""

    return sorted(oriented_path.index(i) + 1 for i in target_indices if i in oriented_path)


def get_bond_locants(
    mol: Molecule,
    oriented_path: list[int],
    is_bicycle: bool,
    is_spiro: bool,
    is_polycycle: bool,
) -> tuple[list[int], list[int]]:
    """Return double- and triple-bond locants for an oriented parent path."""

    double_locs = []
    triple_locs = []
    seen_bonds = set()
    for u in oriented_path:
        for v in mol.get_neighbors(u):
            if v in oriented_path:
                bond = mol.get_bond(u, v)
                if bond and bond.order > 1 and bond.idx not in seen_bonds:
                    seen_bonds.add(bond.idx)
                    loc_u = oriented_path.index(u) + 1
                    loc_v = oriented_path.index(v) + 1
                    min_loc, max_loc = min(loc_u, loc_v), max(loc_u, loc_v)

                    if max_loc == min_loc + 1:
                        locant_val = min_loc
                    elif (
                        min_loc == 1 and max_loc == len(oriented_path) and not (is_bicycle or is_spiro or is_polycycle)
                    ):
                        locant_val = max_loc
                    else:
                        locant_val = min_loc

                    if bond.order == 2:
                        double_locs.append(locant_val)
                    elif bond.order == 3:
                        triple_locs.append(locant_val)

    return sorted(double_locs), sorted(triple_locs)


def retained_locant_sort_key(locant: str) -> tuple[int, str]:
    """Sort retained-ring locants numerically first, then by letter suffix (``4a`` after ``4``)."""

    digits = ""
    suffix = ""
    for char in str(locant):
        if char.isdigit() and not suffix:
            digits += char
        else:
            suffix += char
    return (int(digits) if digits else 10_000, suffix)


def parse_system_locant(value: object):
    """Parse a completed fused-system locant into its typed representation.

    Ordinary integer locants, fusion locants such as ``4a``, and explicit
    interior-distance forms such as ``9^2`` are accepted. Component primes are
    intentionally excluded because they belong to fusion descriptors, not the
    completed parent numbering namespace.
    """

    from .fusion.model import SystemLocant

    if isinstance(value, SystemLocant):
        return value
    text = str(value).strip()
    if any(mark in text for mark in ("'", "′")):
        raise ValueError(f"component prime is not valid in a system locant: {text!r}")
    if any(char in text for char in "⁰¹²³⁴⁵⁶⁷⁸⁹"):
        split = next((index for index, char in enumerate(text) if char in "⁰¹²³⁴⁵⁶⁷⁸⁹"), len(text))
        text = f"{text[:split]}^{text[split:].translate(_SUPERSCRIPT_TO_ASCII)}"
    match = _SYSTEM_LOCANT_RE.fullmatch(text)
    if match is None:
        raise ValueError(f"invalid completed-system locant: {value!r}")
    distance = match.group("distance")
    return SystemLocant(
        base=int(match.group("base")),
        fusion_suffix=match.group("suffix"),
        interior_distance=int(distance) if distance else None,
    )


def system_locant_sort_key(value: object) -> tuple[int, int, str, int]:
    """Return the single canonical ordering key for completed-system locants."""

    locant = parse_system_locant(value)
    return (
        locant.base,
        0 if not locant.fusion_suffix else 1,
        locant.fusion_suffix,
        locant.interior_distance or 0,
    )


def canonical_locant_pair(left: object, right: object) -> tuple[str, str]:
    """Return a deterministically ordered pair of completed-system locants."""

    return tuple(sorted((str(left), str(right)), key=system_locant_sort_key))
