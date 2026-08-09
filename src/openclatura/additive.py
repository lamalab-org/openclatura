"""Explicit additive/replacement feature collection for selected parents."""

from .assembly_parts import AssemblyParts, SubstituentItem
from .locants import parse_locant
from .molecule import Molecule
from .name_operations import HydroOperation
from .namer_config import INDICATED_H_ELEMENTS, cites_indicated_hydrogen
from .rules.retained import mancude_monocycle_hydro_plan


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


def _declared_sites_are_unambiguous(mol: Molecule, numbered_path: list[int], get_loc, declared: set[str]) -> bool:
    """True when the declared indicated-H locants are the only place they could sit.

    A declared heteroatom site is ambiguous when the ring system holds another
    heteroatom of the same element that could carry the hydrogen instead: 1H-
    and 2H-indazole are both real parents, so the ``1`` in 1H-indazole-3,5-dione
    is what tells the two apart and cannot be traded for a saturated carbon.
    """

    for idx in numbered_path:
        atom = mol.atoms[idx]
        if str(get_loc(idx)) not in declared or atom.is_carbon:
            continue
        if sum(mol.atoms[other].symbol == atom.symbol for other in numbered_path) > 1:
            return False
    return True


def add_indicated_hydrogens(mol: Molecule, parts: AssemblyParts, numbered_path: list[int], get_loc) -> None:
    """Add indicated hydrogen locants for retained ring names."""

    plan = mancude_monocycle_hydro_plan(mol, numbered_path, parts.retained_name)
    if plan is not None:
        _add_monocycle_hydro(parts, plan, get_loc)
        return
    # A retained fused parent brings its own metadata, which is what says how
    # many saturated positions it supports and how the surplus is cited.  Its
    # name already spells its indicated hydrogen ("2H-1-benzopyran"), so only
    # the surplus is emitted here -- citing it again gives "2H-2H-".
    name_states_indicated_h = not cites_indicated_hydrogen(parts.retained_name)
    metadata = parts.retained_parent_metadata
    if name_states_indicated_h and metadata is None:
        return
    # A saturated parent has no mancude bond to hydrogenate.
    if metadata is not None and metadata.mancude_double_bonds == 0 and name_states_indicated_h:
        return
    if parts.retained_name == "tetrazole" and any(mol.atoms[idx].charge for idx in numbered_path):
        return
    oxo_derivative = parts.principal_group is not None and parts.principal_group.key == "ketone"
    default_indicated_h = set(metadata.default_indicated_h) if metadata is not None else set()
    name_declared_indicated_h = set(default_indicated_h)

    observed = _saturated_ring_carbons(mol, numbered_path, get_loc)
    if (
        default_indicated_h
        and len(observed) == len(default_indicated_h)
        and _declared_sites_are_unambiguous(mol, numbered_path, get_loc, default_indicated_h)
    ):
        default_indicated_h = observed
    fusion_locants = set(metadata.fusion_locants) if metadata is not None else set()
    candidates: list[tuple[str, int]] = []
    hydro_only: list[tuple[str, int]] = []

    for idx in numbered_path:
        atom = mol.atoms[idx]
        locant = str(get_loc(idx))
        # An oxo prefix must cite its saturation; an -one/-dione suffix implies it.
        if not oxo_derivative and metadata is not None and _is_oxo_ring_site(mol, idx, numbered_path):
            hydro_only.append((locant, idx))
            continue

        if metadata is not None and default_indicated_h and atom.is_carbon and locant not in default_indicated_h:
            # Saturated all the same, so it is a hydro position even though it
            # is not an indicated-H site: 2,3-dihydro-1H-indole.
            if _is_saturated_ring_site(mol, idx, numbered_path):
                hydro_only.append((locant, idx))
            continue
        if oxo_derivative:
            ring_bond_order = sum(
                bond.order for n in mol.get_neighbors(idx) if n in numbered_path and (bond := mol.get_bond(idx, n))
            )
            substituted_ring_nitrogen = atom.symbol == "N" and ring_bond_order == 2
            if atom.explicit_h_count + atom.total_h_count <= 0 and not substituted_ring_nitrogen:
                continue

            ring_neighbor_count = sum(neighbor in numbered_path for neighbor in mol.get_neighbors(idx))
            if (
                atom.is_carbon
                and locant not in default_indicated_h
                and not (not default_indicated_h and ring_neighbor_count == 3)
            ):
                continue
            if default_indicated_h and locant not in default_indicated_h:
                continue
        if atom.symbol in INDICATED_H_ELEMENTS:
            ring_bonds = [mol.get_bond(idx, n) for n in mol.get_neighbors(idx) if n in numbered_path]
            fusion_carbon_h = (
                metadata is not None
                and not default_indicated_h
                and atom.is_carbon
                and locant in fusion_locants
                and len(ring_bonds) == 3
                and _is_saturated_ring_site(mol, idx, numbered_path)
            )
            indicated_h_site = sum(b.order for b in ring_bonds) == 2 and (
                not atom.is_carbon or _is_saturated_ring_site(mol, idx, numbered_path)
            )
            if indicated_h_site or fusion_carbon_h:
                candidates.append((locant, idx))

    supported = metadata.indicated_hydrogen_count if metadata is not None else len(candidates)
    additive_hydrogen = any(
        mol.atoms[atom_idx].is_carbon and locant in fusion_locants for locant, atom_idx in candidates
    )
    if additive_hydrogen and len(candidates) > 1:
        # The parent keeps its own indicated hydrogen: octahydro-1H-indole.
        # Only if what is left is still whole double bonds, else 1,4-dihydropurine.
        pool = sorted(candidates + hydro_only, key=lambda item: parse_locant(item[0]))
        held = supported if (len(pool) - supported) % 2 == 0 else 0
        candidates, surplus = pool[:held], pool[held:]
        parts.hydro_operations.append(
            HydroOperation(
                key="additive_hydrogen",
                reason="Retained parent requires an additive hydrogen prefix.",
                locants=tuple(locant for locant, _ in surplus),
                atom_ids=tuple(atom_idx for _, atom_idx in surplus),
                operation_kind="additive_hydrogen",
            )
        )
        if not name_states_indicated_h:
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
        return

    # Saturation beyond the supported indicated hydrogens takes a hydro prefix,
    # which saturates whole bonds and so can only absorb an even surplus.
    surplus_count = len(candidates) + len(hydro_only) - supported
    if metadata is not None and surplus_count > 0 and surplus_count % 2 == 0:
        candidates.sort(key=lambda candidate: parse_locant(candidate[0]))
        declared = _declared_indicated_hydrogen_split(
            candidates + hydro_only,
            name_declared_indicated_h,
            supported,
            name_states_indicated_h=name_states_indicated_h,
        )
        if declared is not None:
            candidates, surplus = declared
        elif len(candidates) < supported:
            # An indicated-H site on a heteroatom never enters `candidates`.
            pool = sorted(candidates + hydro_only, key=lambda item: parse_locant(item[0]))
            candidates, surplus = pool[:supported], pool[supported:]
        else:
            surplus = sorted(candidates[supported:] + hydro_only, key=lambda item: parse_locant(item[0]))
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

    # P-14.7: a mancude parent that the -one saturates cites the extra hydrogen
    # after the suffix locant -- quinolin-4(1H)-one, 1,3-benzoxazol-2(3H)-one.
    added_h_needed = oxo_derivative and supported == 0 and _name_spells_no_hydrogen(parts.retained_name)
    if name_states_indicated_h and not added_h_needed:
        return

    for locant, atom_idx in candidates:
        parts.indicated_hydrogens.append(locant)
        parts.hydro_operations.append(
            HydroOperation(
                key="added_hydrogen" if added_h_needed else "indicated_hydrogen",
                reason="Retained unsaturated parent requires indicated-hydrogen locant.",
                locants=(locant,),
                atom_ids=(atom_idx,),
                operation_kind="indicated_hydrogen",
            )
        )


def _declared_indicated_hydrogen_split(
    pool: list[tuple[str, int]],
    default_indicated_h: set[str],
    supported: int,
    *,
    name_states_indicated_h: bool,
) -> tuple[list[tuple[str, int]], list[tuple[str, int]]] | None:
    """Hold the sites the parent name already spells, hydro the rest.

    Lowest-locant order instead cites 1H-isoindole's position 1 twice, once as
    hydro and once as the ``1H``.
    """

    if not name_states_indicated_h or not default_indicated_h:
        return None
    held = [item for item in pool if item[0] in default_indicated_h]
    if len(held) != supported:
        return None
    surplus = [item for item in pool if item[0] not in default_indicated_h]
    return held, sorted(surplus, key=lambda item: parse_locant(item[0]))


def _name_spells_no_hydrogen(retained_name: str | None) -> bool:
    return bool(retained_name) and "H-" not in retained_name


def _is_saturated_ring_site(mol: Molecule, atom_idx: int, numbered_path: list[int]) -> bool:
    """A ring position a hydro prefix cites: sp3, however substituted.

    2,2-dimethylchromane's C2 carries no hydrogen and is a hydro site all the
    same, so the hydrogen count is not consulted.
    """

    ring_bonds = []
    for neighbor in mol.get_neighbors(atom_idx):
        bond = mol.get_bond(atom_idx, neighbor)
        if bond is None:
            continue
        if neighbor in numbered_path:
            ring_bonds.append(bond)
        elif bond.order > 1:
            return False
    return bool(ring_bonds) and all(bond.order == 1 for bond in ring_bonds)


def _is_oxo_ring_site(mol: Molecule, atom_idx: int, numbered_path: list[int]) -> bool:
    atom = mol.atoms[atom_idx]
    if not atom.is_carbon:
        return False
    ring_bonds = [
        bond for n in mol.get_neighbors(atom_idx) if n in numbered_path and (bond := mol.get_bond(atom_idx, n))
    ]
    if not ring_bonds or any(bond.order != 1 for bond in ring_bonds):
        return False
    return any(
        mol.atoms[n].symbol == "O" and (bond := mol.get_bond(atom_idx, n)) is not None and bond.order == 2
        for n in mol.get_neighbors(atom_idx)
        if n not in numbered_path
    )


def _add_monocycle_hydro(parts: AssemblyParts, plan: tuple[int, list[int]], get_loc) -> None:
    """Cite a partly saturated retained monocycle's saturation."""

    indicated, added, citable = plan
    ordered = sorted(((str(get_loc(idx)), idx) for idx in citable), key=lambda item: parse_locant(item[0]))
    for locant, atom_idx in ordered[: indicated + added]:
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
    hydro = ordered[indicated + added :]
    if hydro:
        parts.hydro_operations.append(
            HydroOperation(
                key="additive_hydrogen",
                reason="Saturation beyond the retained parent is cited as a hydro prefix.",
                locants=tuple(locant for locant, _ in hydro),
                atom_ids=tuple(atom_idx for _, atom_idx in hydro),
                operation_kind="additive_hydrogen",
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
