"""Explicit additive/replacement feature collection for selected parents."""

from .assembly_parts import AssemblyParts, SubstituentItem
from .locants import parse_locant
from .molecule import Molecule
from .name_operations import HydroOperation
from .namer_config import INDICATED_H_RETAINED_NAMES


def _saturated_ring_carbons(mol: Molecule, numbered_path: list[int], get_loc) -> set[str]:
    """Locants of ring carbons whose ring bonds are all single and that hold H."""

    found: set[str] = set()
    for idx in numbered_path:
        atom = mol.atoms[idx]
        if not atom.is_carbon:
            continue
        ring_bonds = [
            bond for n in mol.get_neighbors(idx) if n in numbered_path and (bond := mol.get_bond(idx, n)) is not None
        ]
        if not ring_bonds or sum(bond.order for bond in ring_bonds) != len(ring_bonds):
            continue
        if atom.explicit_h_count + atom.total_h_count > 0:
            found.add(str(get_loc(idx)))
    return found


def add_indicated_hydrogens(mol: Molecule, parts: AssemblyParts, numbered_path: list[int], get_loc) -> None:
    """Add indicated hydrogen locants for retained ring names."""

    if parts.retained_name not in INDICATED_H_RETAINED_NAMES:
        return
    if parts.retained_name == "tetrazole" and any(mol.atoms[idx].charge for idx in numbered_path):
        return
    oxo_derivative = parts.principal_group is not None and parts.principal_group.key == "ketone"
    metadata = parts.retained_parent_metadata
    default_indicated_h = set(metadata.default_indicated_h) if metadata is not None else set()
    # The declared locants say how many indicated hydrogens the parent hydride
    # supports, and where they sit in the unsubstituted parent.  Where they sit
    # in *this* molecule is a property of this molecule: 1H- and 2H-indene are
    # different compounds.  So when the structure presents exactly as many
    # saturated carbons as the parent supports, those are the indicated-H sites
    # and their real locants replace the declared ones.  A different count means
    # the extra positions are hydro derivatives, which are cited separately, and
    # the declared locants stand.
    observed = _saturated_ring_carbons(mol, numbered_path, get_loc)
    if default_indicated_h and len(observed) == len(default_indicated_h):
        default_indicated_h = observed
    fusion_locants = set(metadata.fusion_locants) if metadata is not None else set()
    candidates: list[tuple[str, int]] = []
    for idx in numbered_path:
        atom = mol.atoms[idx]
        locant = str(get_loc(idx))
        # Saturated carbon atoms elsewhere in a fused ring system are hydro
        # derivatives, not additional indicated-H sites.  Carbon-bound H is
        # emitted only where the retained parent plan declares it (9H-xanthene,
        # indene, fluorene, and similar parent hydrides).
        if metadata is not None and default_indicated_h and atom.is_carbon and locant not in default_indicated_h:
            continue
        if oxo_derivative:
            # A ring nitrogen at a saturated position is a hydrogenated position
            # of the parent hydride whether it holds the hydrogen or a
            # substituent, so theophylline is
            # 1,3-dimethyl-3,7-dihydro-1H-purine-2,6-dione: N1 and N3 count even
            # though both are methylated.
            ring_bond_order = sum(
                bond.order for n in mol.get_neighbors(idx) if n in numbered_path and (bond := mol.get_bond(idx, n))
            )
            substituted_ring_nitrogen = atom.symbol == "N" and ring_bond_order == 2
            if atom.explicit_h_count + atom.total_h_count <= 0 and not substituted_ring_nitrogen:
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
                metadata is not None
                and not default_indicated_h
                and atom.is_carbon
                and locant in fusion_locants
                and len(ring_bonds) == 3
                and atom.explicit_h_count + atom.total_h_count > 0
            )
            indicated_h_site = sum(b.order for b in ring_bonds) == 2 and (
                not atom.is_carbon or atom.explicit_h_count + atom.total_h_count > 0
            )
            if indicated_h_site or fusion_carbon_h:
                candidates.append((locant, idx))

    # A hydrogenated fusion carbon changes the retained-parent hydride state.
    # It is expressed together with the other affected locants as one additive
    # hydro operation (for example 1,4-dihydropurine), not as indicated H.
    additive_hydrogen = any(
        mol.atoms[atom_idx].is_carbon and locant in fusion_locants for locant, atom_idx in candidates
    )
    if additive_hydrogen and len(candidates) > 1:
        parts.hydro_operations.append(
            HydroOperation(
                key="additive_hydrogen",
                reason="Retained parent requires an additive hydrogen prefix.",
                locants=tuple(locant for locant, _ in candidates),
                atom_ids=tuple(atom_idx for _, atom_idx in candidates),
                operation_kind="additive_hydrogen",
            )
        )
        return

    # A mancude parent hydride supports a fixed number of indicated hydrogens.
    # Saturated positions beyond that are *added* hydrogen and take a hydro
    # prefix, so xanthine is 3,7-dihydro-1H-purine-2,6-dione rather than a run
    # of indicated-H citations.  The lowest locants stay indicated.
    # A hydro prefix saturates whole double bonds, so it can only absorb an even
    # surplus.  An odd one stays as indicated hydrogen (1H,9H-purin-6-one),
    # which is still a readable name.
    supported = metadata.indicated_hydrogen_count if metadata is not None else len(candidates)
    if metadata is not None and len(candidates) > supported and (len(candidates) - supported) % 2 == 0:
        candidates.sort(key=lambda candidate: parse_locant(candidate[0]))
        surplus = candidates[supported:]
        candidates = candidates[:supported]
        parts.hydro_operations.append(
            HydroOperation(
                key="additive_hydrogen",
                reason="Saturation beyond the parent's indicated hydrogen is added hydrogen.",
                locants=tuple(locant for locant, _ in surplus),
                atom_ids=tuple(atom_idx for _, atom_idx in surplus),
                operation_kind="additive_hydrogen",
            )
        )

    for locant, atom_idx in candidates:
        parts.indicated_hydrogens.append(locant)
        parts.hydro_operations.append(
            HydroOperation(
                key="indicated_hydrogen",
                reason="Retained unsaturated parent requires indicated-hydrogen locant.",
                locants=(locant,),
                atom_ids=(atom_idx,),
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
