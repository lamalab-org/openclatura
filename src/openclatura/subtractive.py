"""Explicit subtractive feature collection for selected parents."""

from .assembly_parts import AssemblyParts, UnsaturationItem
from .locants import canonical_locant_pair, parse_locant
from .molecule import Molecule


def _implied_parent_multiple_bonds(parts: AssemblyParts) -> frozenset[int]:
    """Return graph bond ids implied by the selected parent-hydride model."""

    parent = parts.parent_hydride
    if parent is None or parent.bond_model is None:
        return frozenset()
    actual_by_edge = {
        frozenset(locants): order for locants, order in parts.parent_bond_orders_by_locants.items()
    }
    for assignment in parent.bond_model.allowed_kekule_assignments:
        assignment_by_edge = {
            frozenset(
                (
                    _locant_for_atom(parts, edge[0]),
                    _locant_for_atom(parts, edge[1]),
                )
            ): order
            for edge, order in assignment.orders
        }
        if assignment_by_edge != actual_by_edge:
            continue
        return frozenset(
            parts.parent_bond_ids_by_locants[canonical_locant_pair(*locants)]
            for locants, order in assignment_by_edge.items()
            if order > 1 and canonical_locant_pair(*locants) in parts.parent_bond_ids_by_locants
        )
    return frozenset()


def _locant_for_atom(parts: AssemblyParts, atom_id: int) -> str:
    return next(locant for locant, atom in parts.parent_atom_ids_by_locant.items() if atom == atom_id)


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
    implied_multiple_bonds = _implied_parent_multiple_bonds(parts)
    for u_idx in numbered_path:
        for v_idx in mol.get_neighbors(u_idx):
            if v_idx not in numbered_path:
                continue
            bond = mol.get_bond(u_idx, v_idx)
            if (
                not bond
                or bond.order <= 1
                or bond.idx in seen_bonds
                or bond.idx in implied_multiple_bonds
            ):
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
            elif min_idx == 0 and max_idx == len(numbered_path) - 1 and not (is_bicycle or is_spiro or is_polycycle):
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
