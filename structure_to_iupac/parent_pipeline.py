"""Shared parent planning steps for component and subgraph naming."""

from .assembly_parts import AssemblyParts, ParentChargeItem
from .molecule import Molecule
from .namer_config import RETAINED_RING_ELEMENTS
from .naming_context import NamingIntent, ParentAssemblyPlan
from .numbering import choose_parent_numbering
from .parent_selection import ParentSelection
from .rules import retained
from .subgraph_tools import subgraph_locant_getter
from .trace_helpers import bond_ids_within


def resolve_retained_parent(
    mol: Molecule, path: list[int], is_ring: bool, is_bicycle: bool, is_polycycle: bool
) -> tuple[str | None, list[dict[int, str]] | None]:
    """Return a retained parent name and locant maps when valid for a path."""

    temp_retained = retained.get_retained_ring(mol, path) if is_ring else None
    if not temp_retained:
        return None, None
    retained_name, locant_maps = temp_retained
    if any(mol.atoms[idx].symbol not in RETAINED_RING_ELEMENTS for idx in path):
        return None, None
    if locant_maps is None and (is_bicycle or is_polycycle):
        if all(mol.atoms[idx].symbol == "C" and mol.atoms[idx].charge == 0 for idx in path):
            return retained_name, None
        return None, None
    return retained_name, locant_maps


def build_parent_assembly_plan(
    mol: Molecule,
    selection: ParentSelection,
    intent: NamingIntent,
    substituent_mapping: dict[int, list],
    locant_maps,
    retained_name: str | None,
) -> ParentAssemblyPlan:
    """Number a selected parent and create base assembly parts."""

    numbered_path, locant_map = choose_parent_numbering(
        mol,
        selection.paths,
        intent.principal_atoms,
        substituent_mapping,
        locant_maps,
        selection.is_ring,
        selection.is_bicycle,
        selection.is_spiro,
        selection.is_polycycle,
        retained_name,
        fixed_start=intent.fixed_start,
    )
    get_loc = subgraph_locant_getter(numbered_path, locant_map)
    parts = build_parent_parts(
        mol,
        numbered_path,
        get_loc,
        retained_name,
        selection,
        intent,
    )
    return ParentAssemblyPlan(numbered_path=numbered_path, locant_map=locant_map, get_loc=get_loc, parts=parts)


def build_parent_parts(
    mol: Molecule,
    numbered_path: list[int],
    get_loc,
    retained_name: str | None,
    selection: ParentSelection,
    intent: NamingIntent,
) -> AssemblyParts:
    """Create shared parent assembly parts for a naming intent."""

    assembly_overrides = {}
    if intent.is_substituent:
        if intent.root_atom is None:
            raise ValueError("Subgraph naming intent requires a root atom.")
        upstream_order = _upstream_bond_order(mol, intent.root_atom, intent.upstream_atom)
        assembly_overrides.update(
            {
                "is_substituent": True,
                "is_double_attach": upstream_order == 2,
                "is_triple_attach": upstream_order == 3,
                "attachment_locant": get_loc(intent.root_atom),
            }
        )

    parts = AssemblyParts(
        parent_length=len(numbered_path),
        is_ring=selection.is_ring,
        is_bicycle=selection.is_bicycle,
        is_spiro=selection.is_spiro,
        is_polycycle=selection.is_polycycle,
        bicycle_xyz=selection.xyz if selection.is_bicycle else (0, 0, 0),
        spiro_xy=(selection.xyz[0], selection.xyz[1]) if selection.is_spiro else (0, 0),
        polycycle_descriptor=selection.polycycle_descriptor,
        retained_name=retained_name,
        parent_atom_ids=set(numbered_path),
        parent_bond_ids=bond_ids_within(mol, set(numbered_path)),
        **assembly_overrides,
    )
    for atom_idx in numbered_path:
        locant = str(get_loc(atom_idx))
        parts.parent_atom_ids_by_locant[locant] = atom_idx
        parts.parent_atom_symbols_by_locant[locant] = mol.atoms[atom_idx].symbol
        parts.parent_atom_charges_by_locant[locant] = mol.atoms[atom_idx].charge
        if mol.atoms[atom_idx].stereo:
            parts.stereo_features.append((get_loc(atom_idx), mol.atoms[atom_idx].stereo))
        if mol.atoms[atom_idx].charge:
            parts.parent_charges.append(
                ParentChargeItem(
                    locant=locant,
                    symbol=mol.atoms[atom_idx].symbol,
                    charge=mol.atoms[atom_idx].charge,
                    atom_id=atom_idx,
                )
            )
    parent_set = set(numbered_path)
    for atom_idx in numbered_path:
        locant = str(get_loc(atom_idx))
        for neighbor_idx in mol.get_neighbors(atom_idx):
            if neighbor_idx in parent_set and atom_idx < neighbor_idx:
                neighbor_locant = str(get_loc(neighbor_idx))
                bond = mol.get_bond(atom_idx, neighbor_idx)
                if bond is not None:
                    parts.parent_bond_orders_by_locants[tuple(sorted((locant, neighbor_locant)))] = bond.order
    return parts


def _upstream_bond_order(mol: Molecule, start_idx: int, upstream_atom: int | None) -> int:
    if upstream_atom is None:
        return 0
    bond = mol.get_bond(start_idx, upstream_atom)
    return bond.order if bond else 0
