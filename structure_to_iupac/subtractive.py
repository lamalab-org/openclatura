"""Explicit subtractive feature collection for selected parents."""

from .assembler import AssemblyParts, UnsaturationItem
from .locants import parse_locant
from .molecule import Molecule


def add_unsaturations(
    mol: Molecule,
    parts: AssemblyParts,
    numbered_path: list[int],
    get_loc,
    is_bicycle: bool,
    is_spiro: bool,
    is_polycycle: bool,
) -> None:
    """Add double/triple bond locants to assembly parts."""

    if parts.retained_name:
        return

    seen_bonds = set()
    for u_idx in numbered_path:
        for v_idx in mol.get_neighbors(u_idx):
            if v_idx not in numbered_path:
                continue
            bond = mol.get_bond(u_idx, v_idx)
            if not bond or bond.order <= 1 or bond.idx in seen_bonds:
                continue
            seen_bonds.add(bond.idx)
            bond_key = "double" if bond.order == 2 else "triple"
            loc_u_idx = numbered_path.index(u_idx)
            loc_v_idx = numbered_path.index(v_idx)
            min_idx, max_idx = min(loc_u_idx, loc_v_idx), max(loc_u_idx, loc_v_idx)

            loc_u_str = get_loc(u_idx)
            loc_v_str = get_loc(v_idx)
            min_loc_str, max_loc_str = (
                min(loc_u_str, loc_v_str, key=lambda x: parse_locant(x)),
                max(loc_u_str, loc_v_str, key=lambda x: parse_locant(x)),
            )

            if max_idx == min_idx + 1:
                locant_str = min_loc_str
            elif min_idx == 0 and max_idx == len(numbered_path) - 1 and not (
                is_bicycle or is_spiro or is_polycycle
            ):
                locant_str = max_loc_str
            else:
                locant_str = f"{min_loc_str}({max_loc_str})"

            existing = next((u for u in parts.unsaturations if u.bond_key == bond_key), None)
            if existing:
                existing.locants.append(locant_str)
                existing.atom_ids.update({u_idx, v_idx})
                existing.bond_ids.add(bond.idx)
            else:
                parts.unsaturations.append(
                    UnsaturationItem(
                        bond_key=bond_key,
                        locants=[locant_str],
                        atom_ids={u_idx, v_idx},
                        bond_ids={bond.idx},
                    )
                )
