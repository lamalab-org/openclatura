import re

from .assembly_parts import AssemblyParts, SubstituentItem
from .fusion.mancude import compare_actual_parent_to_implied_parent
from .locants import parse_locant
from .molecule import Molecule
from .name_operations import HydroOperation
from .namer_config import INDICATED_H_ELEMENTS, cites_indicated_hydrogen
from .rules.retained import mancude_monocycle_hydro_plan

_STEM_HYDRO_RE = re.compile(r"^(\d+[a-z]?(?:,\d+[a-z]?)*)-(?:di|tri|tetra|penta|hexa|hepta|octa|nona|deca)hydro-")


def _fold_stem_hydro_prefix(parts: AssemblyParts, numbered_path: list[int], get_loc) -> None:
    """Merge a preferred spelling's own hydro prefix into the added-hydrogen operation.

    ``2,3-dihydro-1H-indene`` plus six more saturated positions is one
    ``octahydro-1H-indene``, not ``hexahydro-2,3-dihydro-1H-indene``.
    """

    match = _STEM_HYDRO_RE.match(parts.retained_name or "")
    if match is None:
        return
    operation = parts.hydro_operations[-1]
    atom_by_locant = {str(get_loc(idx)): idx for idx in numbered_path}
    stem_locants = match.group(1).split(",")
    if any(locant not in atom_by_locant for locant in stem_locants):
        return
    merged = sorted(
        {
            **dict(zip(operation.locants, operation.atom_ids)),
            **{loc: atom_by_locant[loc] for loc in stem_locants},
        }.items(),
        key=lambda item: parse_locant(item[0]),
    )
    parts.retained_name = parts.retained_name[match.end() :]
    parts.hydro_operations[-1] = HydroOperation(
        key=operation.key,
        reason=operation.reason,
        locants=tuple(locant for locant, _ in merged),
        atom_ids=tuple(atom_idx for _, atom_idx in merged),
        operation_kind=operation.operation_kind,
    )


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
        if atom.explicit_h_count + atom.total_h_count > 0 or _is_oxo_ring_site(mol, idx, numbered_path):
            found.add(str(get_loc(idx)))
    return found


def _respell_indicated_hydrogen(retained_name: str, sites: set[str]) -> str:
    """
    Rewrite the indicated-hydrogen locants a retained name spells for itself.
    """

    match = re.match(r"(\d+[a-z]?H(?:,\d+[a-z]?H)*)-", retained_name)
    if match is None:
        return retained_name
    spelled = ",".join(f"{locant}H" for locant in sorted(sites, key=parse_locant))
    return f"{spelled}-{retained_name[match.end() :]}"


def _exocyclic_double_bond_site(mol: Molecule, atom_idx: int, numbered_path: list[int]) -> bool:
    """
    A ring carbon held sp2 only by a double bond leaving the ring.
    """

    atom = mol.atoms[atom_idx]
    if not atom.is_carbon:
        return False
    ring_bonds = [
        bond for n in mol.get_neighbors(atom_idx) if n in numbered_path and (bond := mol.get_bond(atom_idx, n))
    ]
    if not ring_bonds or any(bond.order != 1 for bond in ring_bonds):
        return False
    return any(
        (bond := mol.get_bond(atom_idx, n)) is not None and bond.order == 2
        for n in mol.get_neighbors(atom_idx)
        if n not in numbered_path
    )


def _declared_sites_are_unambiguous(mol: Molecule, numbered_path: list[int], get_loc, declared: set[str]) -> bool:
    """
    True when the declared indicated-H locants are the only place they could sit.
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

    parent = parts.parent_hydride
    if parent is not None and parent.bond_model is not None:
        delta = parts.parent_bond_delta
        if delta is None:
            delta = compare_actual_parent_to_implied_parent(mol, parts.parent_atom_ids, parent.bond_model)
            parts.parent_bond_delta = delta
        if delta is not None and delta.compatible and delta.hydrogenated_edges:
            atom_ids = sorted(
                delta.hydrogenated_atom_ids,
                key=lambda atom_idx: parse_locant(str(get_loc(atom_idx))),
            )
            parts.hydro_operations.append(
                HydroOperation(
                    key="additive_hydrogen",
                    reason="Observed parent bond orders require hydrogenation of the proved mancude parent.",
                    locants=tuple(str(get_loc(atom_idx)) for atom_idx in atom_ids),
                    atom_ids=tuple(atom_ids),
                    operation_kind="additive_hydrogen",
                )
            )
            return

    plan = mancude_monocycle_hydro_plan(mol, numbered_path, parts.retained_name)
    if plan is not None:
        _add_monocycle_hydro(parts, plan, get_loc)
        if parts.principal_group is not None and parts.principal_group.key == "ketone":
            _recast_ring_ketone_hydrogens(mol, parts, numbered_path, get_loc)
        return

    metadata = parts.retained_parent_metadata
    name_states_indicated_h = not cites_indicated_hydrogen(parts.retained_name) and not (
        metadata is not None and metadata.relocated_indicated_h
    )
    if name_states_indicated_h and metadata is None:
        return
    # A saturated parent has no mancude bond to hydrogenate.
    if metadata is not None and metadata.mancude_double_bonds == 0 and name_states_indicated_h:
        return
    if parts.retained_name == "tetrazole" and any(mol.atoms[idx].charge for idx in numbered_path):
        return
    oxo_derivative = parts.principal_group is not None and parts.principal_group.key == "ketone"
    principal_suffix_sites = (
        parts.parent_atom_ids & parts.principal_group.atom_ids if parts.principal_group is not None else set()
    )
    default_indicated_h = set(metadata.default_indicated_h) if metadata is not None else set()
    inherent_saturated_locants = set(metadata.inherent_saturated_locants) if metadata is not None else set()
    name_declared_indicated_h = set(default_indicated_h) or _name_indicated_hydrogen_locants(parts.retained_name)

    observed_saturated = _saturated_ring_carbons(mol, numbered_path, get_loc)
    if (
        default_indicated_h
        and len(observed_saturated) == len(default_indicated_h)
        and _declared_sites_are_unambiguous(mol, numbered_path, get_loc, default_indicated_h)
    ):
        default_indicated_h = observed_saturated

    if parts.retained_name and default_indicated_h:
        cited = _name_indicated_hydrogen_locants(parts.retained_name)
        if cited and cited != default_indicated_h and len(cited) == len(default_indicated_h):
            parts.retained_name = _respell_indicated_hydrogen(parts.retained_name, default_indicated_h)
            name_declared_indicated_h = set(default_indicated_h)
    fusion_locants = set(metadata.fusion_locants) if metadata is not None else set()
    candidates: list[tuple[str, int]] = []
    hydro_only: list[tuple[str, int]] = []

    for idx in numbered_path:
        atom = mol.atoms[idx]
        locant = str(get_loc(idx))
        if locant in inherent_saturated_locants:
            continue
        # An oxo prefix must cite its saturation; an -one/-dione suffix implies it.
        if not oxo_derivative and metadata is not None and _is_oxo_ring_site(mol, idx, numbered_path):
            # A principal suffix already represents an exocyclic multiple bond
            # at its parent site (for example C=N in a ring hydrazone). Only an
            # otherwise unrepresented site contributes another hydro operation.
            if idx not in principal_suffix_sites:
                hydro_only.append((locant, idx))
            continue

        if metadata is not None and default_indicated_h and atom.is_carbon and locant not in default_indicated_h:
            if _is_saturated_ring_site(mol, idx, numbered_path):
                hydro_only.append((locant, idx))
            continue
        if oxo_derivative:
            ring_bond_order = sum(
                bond.order for n in mol.get_neighbors(idx) if n in numbered_path and (bond := mol.get_bond(idx, n))
            )
            substituted_ring_nitrogen = atom.symbol == "N" and ring_bond_order == 2
            oxo_indicated_site = locant in default_indicated_h and _is_oxo_ring_site(mol, idx, numbered_path)
            if (
                atom.explicit_h_count + atom.total_h_count <= 0
                and not substituted_ring_nitrogen
                and not oxo_indicated_site
            ):
                continue

            ring_neighbor_count = sum(neighbor in numbered_path for neighbor in mol.get_neighbors(idx))
            if (
                atom.is_carbon
                and locant not in default_indicated_h
                and not (not default_indicated_h and ring_neighbor_count == 3)
            ):
                # A saturated ring carbon beyond the ketone still has to be spelt: 3,4-dihydro.
                if metadata is not None and _is_saturated_ring_site(mol, idx, numbered_path):
                    hydro_only.append((locant, idx))
                continue
            if default_indicated_h and locant not in default_indicated_h:
                # A saturated ring heteroatom away from the cited site is a hydro position:
                # 2,3-dihydro-1H-isoindol-1-one.
                if (
                    metadata is not None
                    and not metadata.inherent_saturated_locants  # indoline spells its own N-H
                    and not atom.is_carbon
                    and _is_saturated_ring_site(mol, idx, numbered_path)
                ):
                    hydro_only.append((locant, idx))
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
                not atom.is_carbon
                or _is_saturated_ring_site(mol, idx, numbered_path)
                or (locant in default_indicated_h and _exocyclic_double_bond_site(mol, idx, numbered_path))
            )
            if indicated_h_site or fusion_carbon_h:
                candidates.append((locant, idx))

    supported = metadata.indicated_hydrogen_count if metadata is not None else len(candidates)
    if metadata is not None:
        # Template-inherent saturated positions were removed from the observed
        # pool above.  They must also be removed from the parent's supported-H
        # capacity; otherwise they are counted once as part of the retained
        # hydride and a second time when the surplus is calculated.  This is
        # relevant to every partially saturated retained parent, not only a
        # particular spelling (indane, benzodioxole, xanthene, ...).
        supported -= len(default_indicated_h & inherent_saturated_locants)
        supported = max(0, supported)
    additive_hydrogen = any(
        mol.atoms[atom_idx].is_carbon and locant in fusion_locants for locant, atom_idx in candidates
    )
    if additive_hydrogen and len(candidates) > 1:
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
        _fold_stem_hydro_prefix(parts, numbered_path, get_loc)

    # P-14.7: a mancude parent that the -one saturates cites the extra hydrogen
    # after the suffix locant -- quinolin-4(1H)-one, 1,3-benzoxazol-2(3H)-one.
    added_h_needed = oxo_derivative and supported == 0 and _name_spells_no_hydrogen(parts.retained_name)
    if (
        metadata is not None
        and oxo_derivative
        and (not name_states_indicated_h or added_h_needed)
        and surplus_count % 2 == 1
    ):
        candidates = sorted(candidates + hydro_only, key=lambda item: parse_locant(item[0]))
    if name_states_indicated_h and not added_h_needed:
        if metadata is not None and oxo_derivative and surplus_count > 0 and surplus_count % 2 == 1 and hydro_only:
            # 3a,4,7,7a-tetrahydro-1H-isoindole-1,3(2H)-dione: one site beside a ketone is added
            # hydrogen cited with the suffix, the even remainder a hydro prefix.
            pool = sorted(candidates[supported:] + hydro_only, key=lambda item: parse_locant(item[0]))
            ketone_atoms = {idx for idx in numbered_path if _is_oxo_ring_site(mol, idx, numbered_path)}
            beside = [item for item in pool if any(n in ketone_atoms for n in mol.get_neighbors(item[1]))]
            added = (beside or pool)[0]
            rest = [item for item in pool if item != added]
            parts.indicated_hydrogens.append(added[0])
            parts.hydro_operations.append(
                HydroOperation(
                    key="added_hydrogen",
                    reason="P-14.7: the ketone's neighbouring sp3 site is added hydrogen.",
                    locants=(added[0],),
                    atom_ids=(added[1],),
                    operation_kind="indicated_hydrogen",
                )
            )
            if rest:
                parts.hydro_operations.append(
                    HydroOperation(
                        key="additive_hydrogen",
                        reason="Saturation beyond the added hydrogen is a hydro prefix.",
                        locants=tuple(locant for locant, _ in rest),
                        atom_ids=tuple(atom_idx for _, atom_idx in rest),
                        operation_kind="additive_hydrogen",
                    )
                )
        return

    if added_h_needed and len(candidates) >= 3 and (len(candidates) - 1) % 2 == 0:
        # 3,4-dihydroquinolin-2(1H)-one: one added hydrogen, the rest as a hydro prefix.
        candidates = sorted(candidates, key=lambda item: parse_locant(item[0]))
        added, rest = candidates[:1], candidates[1:]
        parts.hydro_operations.append(
            HydroOperation(
                key="additive_hydrogen",
                reason="Saturation beyond the added hydrogen is a hydro prefix.",
                locants=tuple(locant for locant, _ in rest),
                atom_ids=tuple(atom_idx for _, atom_idx in rest),
                operation_kind="additive_hydrogen",
            )
        )
        candidates = added
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
    if oxo_derivative and not added_h_needed:
        _recast_ring_ketone_hydrogens(mol, parts, numbered_path, get_loc)


def _recast_ring_ketone_hydrogens(mol: Molecule, parts: AssemblyParts, numbered_path: list[int], get_loc) -> None:
    """P-31.1.4.2.4: spell a mancude ring ketone's saturation the way the rules do.

    With as many indicated-hydrogen sites as ketones, the hydrogens are *added*
    hydrogen cited with the suffix -- ``pyrimidine-2,4(1H,3H)-dione``.  With more
    NH sites than the single ketone, the ketone carbon carries the indicated
    hydrogen and the NH sites are hydro prefixes -- ``1,9-dihydro-6H-purin-6-one``.
    """

    group = parts.principal_group
    if group is None:
        return
    indicated = [op for op in parts.hydro_operations if op.key == "indicated_hydrogen"]
    if not indicated or any(op.key == "added_hydrogen" for op in parts.hydro_operations):
        return
    ketone_locants = [str(locant) for locant in group.locants]
    nh_locants = [str(op.locants[0]) for op in indicated]
    if set(nh_locants) & set(ketone_locants):
        return
    if len(nh_locants) == len(ketone_locants):
        for op in indicated:
            parts.hydro_operations[parts.hydro_operations.index(op)] = HydroOperation(
                key="added_hydrogen",
                reason="A ring ketone's NH sites are added hydrogen cited with the suffix.",
                locants=op.locants,
                atom_ids=op.atom_ids,
                operation_kind="indicated_hydrogen",
            )
        return
    if len(ketone_locants) == 1 and len(nh_locants) > 1:
        ketone_atom = next((idx for idx in numbered_path if str(get_loc(idx)) == ketone_locants[0]), None)
        if ketone_atom is None:
            return
        for op in indicated:
            parts.hydro_operations.remove(op)
        parts.indicated_hydrogens = [loc for loc in parts.indicated_hydrogens if loc not in nh_locants]
        parts.indicated_hydrogens.append(ketone_locants[0])
        parts.hydro_operations.append(
            HydroOperation(
                key="indicated_hydrogen",
                reason="The ketone carbon carries the ring's indicated hydrogen.",
                locants=(ketone_locants[0],),
                atom_ids=(ketone_atom,),
                operation_kind="indicated_hydrogen",
            )
        )
        parts.hydro_operations.append(
            HydroOperation(
                key="additive_hydrogen",
                reason="The remaining NH sites of a ring ketone are hydro prefixes.",
                locants=tuple(nh_locants),
                atom_ids=tuple(op.atom_ids[0] for op in indicated),
                operation_kind="additive_hydrogen",
            )
        )


def _name_indicated_hydrogen_locants(retained_name: str | None) -> set[str]:
    """The locants a name spells for itself, for parents that declare none.

    ``7H-pyrrolo[2,3-d]pyrimidine`` names its sp3 position in the stem while its
    template leaves ``default_indicated_h`` empty; without this the hydro set
    could take locant 7 as well and cite it twice.
    """

    if not retained_name:
        return set()
    match = re.match(r"(\d+[a-z]?H(?:,\d+[a-z]?H)*)-", retained_name)
    return {locant.removesuffix("H") for locant in match.group(1).split(",")} if match else set()


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
    # An exocyclic =O, =S or =N (ketone, thione, ylidene) makes the ring position sp2 without a
    # ring double bond, so it is a hydro site just as a ketone carbon is.
    return any(
        mol.atoms[n].symbol in {"O", "S", "N", "C"}
        and (bond := mol.get_bond(atom_idx, n)) is not None
        and bond.order == 2
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

    if parts.retained_name or (parts.parent_hydride is not None and parts.parent_hydride.absorbs_skeletal_replacement):
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
