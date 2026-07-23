"""Locant parsing and atom/bond locant helpers."""

import re
from collections.abc import Mapping

from .molecule import Molecule


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


def coerce_display_numbering(
    numbering: Mapping[int, int], display_numbering: Mapping[int, str] | None = None
) -> dict[int, DisplayLocant]:
    """Attach display locants to a numeric atom-numbering map."""

    return {
        atom: as_display_locant(locant, display_numbering.get(atom) if display_numbering is not None else None)
        for atom, locant in numbering.items()
    }


def parse_locant(l):
    """Return a sortable representation of a locant string."""

    s = locant_text(l)
    match = re.match(r"^(\d+)([a-zA-Z]*)$", s.split("(")[0])
    if match:
        return (1, float(match.group(1)), match.group(2))
    if any(c.isdigit() for c in s):
        nums = re.findall(r"\d+", s)
        return (1, float(nums[0]) if nums else 0.0, s)
    return (2, 0.0, s)


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
