"""Explicit additive/replacement feature collection for selected parents."""

from .assembly_parts import AssemblyParts, SubstituentItem
from .grammar_snapshot_data import retained_fused_token
from .molecule import Molecule
from .name_operations import HydroOperation
from .namer_config import INDICATED_H_RETAINED_NAMES


def add_indicated_hydrogens(mol: Molecule, parts: AssemblyParts, numbered_path: list[int], get_loc) -> None:
    """Add indicated hydrogen locants for retained ring names."""

    if parts.retained_name not in INDICATED_H_RETAINED_NAMES:
        return
    if parts.retained_name == "tetrazole" and any(mol.atoms[idx].charge for idx in numbered_path):
        return
    oxo_derivative = parts.principal_group is not None and parts.principal_group.key == "ketone"
    retained_token = retained_fused_token(parts.retained_name)
    default_indicated_h = set(retained_token.default_indicated_h) if retained_token is not None else set()
    for idx in numbered_path:
        atom = mol.atoms[idx]
        locant = str(get_loc(idx))
        # Saturated carbon atoms elsewhere in a fused ring system are hydro
        # derivatives, not additional indicated-H sites.  Carbon-bound H is
        # emitted only where the retained parent plan declares it (9H-xanthene,
        # indene, fluorene, and similar parent hydrides).
        if retained_token is not None and default_indicated_h and atom.is_carbon and locant not in default_indicated_h:
            continue
        if oxo_derivative:
            if atom.explicit_h_count + atom.total_h_count <= 0:
                continue
            # Hydrogen introduced next to =O is implied by ``-one``/``-dione``
            # and must not become a new indicated-H locant.  Carbon is allowed
            # only when the retained parent itself declares that H locant, as
            # in 9H-xanthene; this excludes the spurious 2H in carbazol-1-one.
            ring_neighbor_count = sum(neighbor in numbered_path for neighbor in mol.get_neighbors(idx))
            if (
                atom.is_carbon
                and locant not in default_indicated_h
                and not (not default_indicated_h and ring_neighbor_count == 3)
            ):
                continue
            if default_indicated_h and locant not in default_indicated_h:
                continue
        if atom.symbol in ["N", "C"]:
            ring_bonds = [mol.get_bond(idx, n) for n in mol.get_neighbors(idx) if n in numbered_path]
            fusion_carbon_h = (
                retained_token is not None
                and not default_indicated_h
                and atom.is_carbon
                and len(ring_bonds) == 3
                and atom.explicit_h_count + atom.total_h_count > 0
            )
            if sum(b.order for b in ring_bonds) == 2 or fusion_carbon_h:
                locant = str(get_loc(idx))
                parts.indicated_hydrogens.append(locant)
                parts.hydro_operations.append(
                    HydroOperation(
                        key="indicated_hydrogen",
                        reason="Retained unsaturated parent requires indicated-hydrogen locant.",
                        locants=(locant,),
                        atom_ids=(idx,),
                        operation_kind="indicated_hydrogen",
                    )
                )


def add_replacement_prefixes(mol: Molecule, parts: AssemblyParts, numbered_path: list[int], get_loc) -> None:
    """Add replacement prefixes and lambda annotations for parent atoms."""

    if parts.retained_name:
        return
    for atom_idx in numbered_path:
        atom = mol.atoms[atom_idx]
        if atom.is_carbon:
            continue
        hw_stem = atom.element.hw_stem
        if not hw_stem:
            continue
        valence = sum(mol.get_bond(atom_idx, n).order for n in mol.get_neighbors(atom_idx))
        loc = get_loc(atom_idx)
        if atom.charge == 0 and valence > atom.element.standard_valence:
            loc = f"{loc}lambda^{valence}"
        parts.a_prefixes.append(SubstituentItem(name=hw_stem, locants=[loc], atom_ids={atom_idx}))
