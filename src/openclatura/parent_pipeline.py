"""Shared parent planning steps for component and subgraph naming."""

from .assembly_parts import AssemblyParts, NameAtomBinding, ParentChargeItem, RetainedParentMetadata
from .fusion.model import ParentHydridePlan
from .heteroatom_subgraphs import upstream_bond_order
from .locant_sources import LocantMapSource
from .locants import canonical_locant_pair
from .molecule import Molecule, bond_ids_within
from .name_bindings import ensure_name_atom_binding_tokens
from .namer_config import RETAINED_RING_ELEMENTS
from .naming_context import NamingIntent, ParentAssemblyPlan
from .numbering import choose_parent_numbering
from .parent_selection import ParentSelection
from .ring_renderer import is_von_baeyer_descriptor
from .rules import retained
from .small_ring_stereo import scoped_small_ring_stereo_features
from .subgraph_tools import subgraph_locant_getter


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
    retained_parent_metadata: RetainedParentMetadata | None = None,
    parent_hydride: ParentHydridePlan | None = None,
) -> ParentAssemblyPlan:
    """Number a selected parent and create base assembly parts."""

    if parent_hydride is not None and parent_hydride.locant_maps:
        locant_maps = list(parent_hydride.string_locant_maps())
        locant_map_source = LocantMapSource.PROOF
    else:
        locant_map_source = LocantMapSource.SUPPLIED if locant_maps else LocantMapSource.GENERATED
    if (
        locant_maps is None
        and selection.ring_parent is not None
        and selection.ring_parent.numbering_candidates
        and is_von_baeyer_descriptor(selection.ring_parent.descriptor)
    ):
        audited_maps = [
            numbering.locant_map for numbering in selection.ring_parent.numbering_candidates if numbering.audit_ok
        ]
        locant_maps = audited_maps or None
        if locant_maps:
            locant_map_source = LocantMapSource.PROOF
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
        retained_parent_metadata,
        parent_hydride=parent_hydride,
        locant_map_source=locant_map_source,
    )
    return ParentAssemblyPlan(
        numbered_path=numbered_path,
        locant_map=locant_map,
        locant_map_source=locant_map_source,
        get_loc=get_loc,
        parts=parts,
        parent_hydride=parent_hydride,
    )


def build_parent_parts(
    mol: Molecule,
    numbered_path: list[int],
    get_loc,
    retained_name: str | None,
    selection: ParentSelection,
    intent: NamingIntent,
    retained_parent_metadata: RetainedParentMetadata | None = None,
    *,
    parent_hydride: ParentHydridePlan | None = None,
    locant_map_source: LocantMapSource = LocantMapSource.GENERATED,
) -> AssemblyParts:
    """Create shared parent assembly parts for a naming intent."""

    if retained_parent_metadata is None and retained_name is not None:
        retained_parent_metadata = retained.parent_metadata(retained_name)
    if retained_parent_metadata is None and parent_hydride is not None:
        metadata = parent_hydride.metadata
        retained_parent_metadata = RetainedParentMetadata(
            default_indicated_h=tuple(str(locant) for locant in metadata.default_indicated_h),
            fusion_locants=tuple(str(locant) for locant in metadata.fusion_locants),
            derivative_stem=metadata.derivative_stem,
            indicated_hydrogen_count=len(metadata.default_indicated_h),
            mancude_double_bonds=metadata.mancude_double_bond_count,
            inherent_saturated_locants=tuple(
                str(locant) for locant in metadata.inherent_saturated_locants
            ),
        )
    assembly_overrides = {}
    if intent.is_substituent:
        if intent.root_atom is None:
            raise ValueError("Subgraph naming intent requires a root atom.")
        upstream_order = upstream_bond_order(mol, intent.root_atom, intent.upstream_atom)
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
        retained_parent_metadata=retained_parent_metadata,
        parent_hydride=parent_hydride,
        locant_map_source=locant_map_source,
        omit_redundant_locants=intent.omit_redundant_locants,
        parent_atom_ids=set(numbered_path),
        parent_bond_ids=bond_ids_within(mol, set(numbered_path)),
        **assembly_overrides,
    )
    for atom_idx in numbered_path:
        locant = str(get_loc(atom_idx))
        parts.parent_atom_ids_by_locant[locant] = atom_idx
        parts.parent_atom_symbols_by_locant[locant] = mol.atoms[atom_idx].symbol
        parts.parent_atom_charges_by_locant[locant] = mol.atoms[atom_idx].charge
        parts.parent_atom_isotopes_by_locant[locant] = mol.atoms[atom_idx].isotope
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
    _add_relative_ring_stereo(mol, parts, numbered_path, get_loc)
    parent_set = set(numbered_path)
    for atom_idx in numbered_path:
        locant = str(get_loc(atom_idx))
        for neighbor_idx in mol.get_neighbors(atom_idx):
            if neighbor_idx in parent_set and atom_idx < neighbor_idx:
                neighbor_locant = str(get_loc(neighbor_idx))
                bond = mol.get_bond(atom_idx, neighbor_idx)
                if bond is not None:
                    locant_pair = canonical_locant_pair(locant, neighbor_locant)
                    parts.parent_bond_orders_by_locants[locant_pair] = bond.order
                    parts.parent_bond_ids_by_locants[locant_pair] = bond.idx
    return parts


def _add_accurate_ring_descriptors(mol: Molecule, parts: AssemblyParts, raw_atoms: list[int], get_loc) -> None:
    """Emit per-atom descriptors for ring centres the legacy perception leaves
    unassigned but the accurate labeller does name.

    Every centre must carry a label, and each a numeric parent locant, or nothing
    is emitted: a partial set would describe some of the ring's configuration and
    silently drop the rest."""

    features: list[tuple[str, str]] = []
    for atom_idx in raw_atoms:
        descriptor = mol.accurate_cip.get(atom_idx)
        locant = str(get_loc(atom_idx))
        if descriptor is None or not locant.isdigit():
            return
        features.append((locant, descriptor))
    features.sort(key=lambda item: int(item[0]))
    parts.stereo_features.extend(features)
    parts.name_atom_bindings.append(
        ensure_name_atom_binding_tokens(
            NameAtomBinding(
                stage="assembly",
                role="small_ring_stereo",
                term=",".join(f"{locant}{descriptor}" for locant, descriptor in features),
                atom_ids=set(raw_atoms),
                locants=tuple(locant for locant, _ in features),
            )
        )
    )


def _add_relative_ring_stereo(mol: Molecule, parts: AssemblyParts, numbered_path: list[int], get_loc) -> None:
    """Render unassigned tetrahedral ring stereo as cis/trans for simple rings."""

    if not parts.is_ring or parts.is_bicycle or parts.is_spiro or parts.is_polycycle:
        return
    scoped_features = scoped_small_ring_stereo_features(mol, parts, numbered_path, get_loc)
    if scoped_features:
        parts.stereo_features.extend(scoped_features)
        parts.name_atom_bindings.append(
            ensure_name_atom_binding_tokens(
                NameAtomBinding(
                    stage="assembly",
                    role="small_ring_stereo",
                    term=",".join(f"{locant}{descriptor}" for locant, descriptor in scoped_features),
                    atom_ids={atom_idx for atom_idx in numbered_path if mol.atoms[atom_idx].raw_stereo},
                    locants=tuple(locant for locant, _ in scoped_features),
                )
            )
        )
        return
    raw_atoms = [
        atom_idx for atom_idx in numbered_path if mol.atoms[atom_idx].raw_stereo and not mol.atoms[atom_idx].stereo
    ]
    if len(raw_atoms) != 2:
        return

    first, second = raw_atoms
    first_tag = mol.atoms[first].raw_stereo
    second_tag = mol.atoms[second].raw_stereo
    if first_tag not in {"CW", "CCW"} or second_tag not in {"CW", "CCW"}:
        return
    parent_set = set(numbered_path)
    if any(
        sum(1 for neighbor_idx in mol.get_neighbors(atom_idx) if neighbor_idx not in parent_set) != 1
        for atom_idx in raw_atoms
    ):
        # ``cis``/``trans`` can only relate a pair that each bear exactly one
        # substituent — there is no single face to speak of otherwise — so a ring
        # position carrying two different groups gets its descriptor spelled out
        # instead, from the accurate labeller that does assign these centres.
        _add_accurate_ring_descriptors(mol, parts, raw_atoms, get_loc)
        return

    term = "cis" if first_tag == second_tag else "trans"
    locants = tuple(str(get_loc(atom_idx)) for atom_idx in raw_atoms)
    parts.relative_stereo_prefixes.append(term)
    parts.name_atom_bindings.append(
        ensure_name_atom_binding_tokens(
            NameAtomBinding(
                stage="assembly",
                role="relative_stereo",
                term=term,
                atom_ids=set(raw_atoms),
                locants=locants,
            )
        )
    )
