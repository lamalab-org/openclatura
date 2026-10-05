# openclatura/perception.py

from dataclasses import dataclass, field

from .chains import get_cyclic_atoms
from .chalcogen_roles import (
    AcylLeavingGroup,
    Chalcogen,
    ChalcogenLigand,
    ChalcogenLigandRole,
    DerivativeKind,
    FunctionalFamily,
    FunctionalGroupDescriptor,
    LeavingGroupDescriptor,
    chalcogen_for_symbol,
    classify_chalcogen_ligand,
    classify_peroxide_linkages,
    require_validated_chalcogens,
)
from .chalcogen_vocabulary import (
    chalcogen_citation_rule,
    resolve_acyl_rule,
    resolve_anhydride_rule,
    resolve_central_acid_rule,
    resolve_imide_route_rule,
    resolve_nitrile_chalcogenide_rule,
    resolve_peroxol_rule,
    resolve_peroxy_acyl_rule,
    simple_group_key,
    uses_existing_chalcogen_citation,
)
from .charge_pair_roles import NitrogenChalcogenideKind, charge_pair_roles
from .functional_groups import PERCEPTION_DETECTORS, PERCEPTION_SPECS, PerceptionDetectorSpec, metadata_for_group
from .molecule import (
    AtomBinding,
    BondBinding,
    FunctionalGroupMetadata,
    Molecule,
    bond_ids_within,
    has_non_h_multiple_bond_neighbor,
)
from .nitrogen_roles import amidinohydrazone_tail_atoms, nitrogen_chain_roles
from .nomenclature import (
    RULES,
    ChalcogenCitationContext,
    ChalcogenCitationProjection,
    FunctionalGroupRule,
)


@dataclass
class PerceivedGroup:
    """
    Functional-group perception result bound to graph atoms and metadata.
    """

    key: str
    is_principal_candidate: bool
    attachment_carbon: int
    atoms_involved: set[int]
    metadata: FunctionalGroupMetadata = field(default_factory=FunctionalGroupMetadata)
    atom_bindings: tuple[AtomBinding, ...] = ()
    bond_bindings: tuple[BondBinding, ...] = ()
    decision_reasons: tuple[str, ...] = ()
    variant: str | None = None
    role: str | None = None
    descriptor: FunctionalGroupDescriptor | None = None
    resolved_rule: FunctionalGroupRule | None = None

    @property
    def atom_ids(self) -> set[int]:
        """Return all atoms represented by this group, including attachment."""

        return set(self.atoms_involved) | {self.attachment_carbon}

    @property
    def bond_ids(self) -> set[int]:
        """Return all graph bonds bound to this group."""

        return {bond_id for binding in self.bond_bindings for bond_id in binding.bond_ids}

    @property
    def prefix(self) -> str | None:
        return self.metadata.prefix

    @property
    def suffix(self) -> str | None:
        return self.metadata.suffix

    @property
    def seniority(self) -> int | None:
        return self.metadata.seniority


def _copy_perceived_group(group: PerceivedGroup) -> PerceivedGroup:
    """Copy a cached group; only ``atoms_involved`` is mutated by callers."""

    return PerceivedGroup(
        group.key,
        group.is_principal_candidate,
        group.attachment_carbon,
        set(group.atoms_involved),
        group.metadata,
        group.atom_bindings,
        group.bond_bindings,
        group.decision_reasons,
        group.variant,
        group.role,
        group.descriptor,
        group.resolved_rule,
    )


def _perceive_groups_uncached(mol: Molecule) -> list[PerceivedGroup]:
    groups = []
    for detector in PERCEPTION_DETECTORS:
        groups.extend(detector(mol))
    if PERCEPTION_SPECS:
        specs = tuple(sorted(BUILTIN_PERCEPTION_SPECS + tuple(PERCEPTION_SPECS), key=lambda item: item.priority))
    else:
        specs = BUILTIN_PERCEPTION_SPECS
    for spec in specs:
        groups.extend(spec.detector(mol))
    return _enrich_groups(mol, _demote_zwitterion_cations(mol, groups))


def perceive_groups(mol: Molecule) -> list[PerceivedGroup]:
    """Perceive functional groups, memoized per molecule graph.

    Naming re-perceives the same graph roughly ten times per molecule. The result
    depends only on the graph and the detector registries, so it is cached on the
    molecule (invalidated on mutation) and handed back as fresh copies, matching the
    previous contract that every call returns objects callers may mutate.
    """

    fingerprint = (len(PERCEPTION_DETECTORS), len(PERCEPTION_SPECS))
    cached = mol._perception_cache
    if cached is None or cached[0] != fingerprint:
        cached = (fingerprint, _perceive_groups_uncached(mol))
        mol._perception_cache = cached
    return [_copy_perceived_group(group) for group in cached[1]]


BUILTIN_PERCEPTION_SPECS = (
    PerceptionDetectorSpec(
        key="builtin.functional_groups",
        detector=lambda mol: _builtin_perceive_groups(mol),
        priority=100,
        families=("functional_group",),
        description="Built-in structural functional-group detectors.",
    ),
)


def _closes_ring_back_to(mol: Molecule, carbon: int, hetero: int, double_o: int, cyclic_atoms: set[int]) -> bool:
    """Whether hetero reaches carbon again through the ring, making it a lactone/lactam."""

    visited = {carbon, double_o}
    queue = [hetero]
    while queue:
        curr = queue.pop(0)
        for nxt in mol.get_neighbors(curr):
            if nxt == carbon and curr != hetero:
                return True
            if nxt not in visited and nxt in cyclic_atoms:
                visited.add(nxt)
                queue.append(nxt)
    return False


CATIONIC_SUFFIX_GROUP_KEYS = frozenset({"aminium", "iminium", "diazonio"})
ANIONIC_SUFFIX_GROUP_KEYS = frozenset({"olate", "thiolate", "aminide", "carboxylate", "ring_carboxylate", "sulfonate"})


def _connected_component(mol: Molecule, idx: int) -> set[int]:
    """Return the atoms reachable from ``idx``."""

    seen = {idx}
    stack = [idx]
    while stack:
        current = stack.pop()
        for neighbor in mol.get_neighbors(current):
            if neighbor not in seen:
                seen.add(neighbor)
                stack.append(neighbor)
    return seen


def _demote_zwitterion_cations(mol: Molecule, groups: list[PerceivedGroup]) -> list[PerceivedGroup]:
    """Keep a cationic group out of the suffix slot when it shares a zwitterion.

    A lone cation outranks every neutral group (P-41) and takes the suffix, but
    in a zwitterion the anion keeps it and the cation is cited as a prefix:
    betaine is 2-(trimethylammonio)acetate.  Skeletal ``-ide`` centres are not
    characteristic groups, and a counter-ion is a separate component.
    """

    cations = [group for group in groups if group.key in CATIONIC_SUFFIX_GROUP_KEYS]
    if not cations:
        return groups
    anion_atoms = {group.attachment_carbon for group in groups if group.key in ANIONIC_SUFFIX_GROUP_KEYS}
    if not anion_atoms:
        return groups
    for cation in cations:
        if not cation.is_principal_candidate:
            continue
        component = _connected_component(mol, cation.attachment_carbon)
        if anion_atoms & component:
            cation.is_principal_candidate = False
    return groups


def _builtin_perceive_groups(mol: Molecule) -> list[PerceivedGroup]:
    require_validated_chalcogens(mol)
    groups = _composable_chalcogen_linkage_groups(mol)
    consumed = set()
    cyclic_atoms = get_cyclic_atoms(mol)

    for atom in mol:
        if atom.symbol == "N" and atom.idx not in consumed and atom.idx not in cyclic_atoms:
            oxygens = [n for n in mol.get_neighbors(atom.idx) if mol.atoms[n].symbol == "O"]
            adj_atoms = [n for n in mol.get_neighbors(atom.idx) if n not in oxygens]

            # A nitro nitrogen carries no hydrogen.  One that does is the azinic
            # acid tautomer — ``[O-][NH+](O)Ph`` is N-phenylazinic acid, which
            # differs from nitrobenzene by two hydrogens, so calling it nitro
            # would name a different compound.
            # Nor does either of its oxygens: an N-OH is the acid, so
            # ``C=[N+]([O-])O`` is an azinic acid ylidene rather than a nitro
            # group, and differs from it by a hydrogen.
            hydroxyl_oxygen = any(
                mol.atoms[o].charge == 0
                and (bond := mol.get_bond(atom.idx, o)) is not None
                and bond.order == 1
                and mol.atoms[o].total_h_count > 0
                for o in oxygens
            )
            if len(oxygens) == 2 and len(adj_atoms) == 1 and atom.total_h_count == 0 and not hydroxyl_oxygen:
                has_double_o = any(mol.get_bond(atom.idx, o).order == 2 for o in oxygens)
                if atom.charge == 1 or has_double_o:
                    groups.append(PerceivedGroup("nitro", False, adj_atoms[0], {atom.idx} | set(oxygens)))
                    consumed.update([atom.idx] + oxygens)
            elif len(oxygens) == 1 and len(adj_atoms) == 1:
                if mol.get_bond(atom.idx, oxygens[0]).order == 2:
                    groups.append(PerceivedGroup("nitroso", False, adj_atoms[0], {atom.idx, oxygens[0]}))
                    consumed.update([atom.idx, oxygens[0]])

    for atom in mol:
        imide_group = _cyclic_imide_group(mol, atom.idx, consumed, cyclic_atoms)
        if imide_group is not None:
            groups.append(imide_group)

    # P-66.3.1: acyl hydrazides outrank the hydrazine chain roles that would otherwise consume the N-N.
    # Discover every acyl end before claiming atoms: a diacylhydrazine has two
    # equally real C(=E)-N-N units, and choosing whichever carbon happens to
    # have the lower input index makes both perception and naming depend on
    # SMILES atom order.
    hydrazide_groups: list[PerceivedGroup] = []
    hydrazide_claimed_atoms: set[int] = set()
    for atom in mol:
        if not atom.is_carbon or atom.idx in consumed or atom.idx in cyclic_atoms:
            continue
        double_ligands = [
            ligand
            for neighbor in mol.get_neighbors(atom.idx)
            if neighbor not in consumed
            and (ligand := classify_chalcogen_ligand(mol, atom.idx, neighbor)) is not None
            and ligand.role is ChalcogenLigandRole.DOUBLE_BONDED
        ]
        if len(double_ligands) != 1:
            continue
        double_ligand = double_ligands[0]
        for single_n in [n for n in mol.get_neighbors(atom.idx) if mol.atoms[n].symbol == "N" and n not in consumed]:
            if mol.get_bond(atom.idx, single_n).order != 1:
                continue
            hydrazide_nitrogens = _hydrazide_nitrogens(mol, atom.idx, single_n, cyclic_atoms)
            if hydrazide_nitrogens is None:
                continue
            ring_neighbors = [n for n in mol.get_neighbors(atom.idx) if n in cyclic_atoms]
            target = atom.idx
            external = False
            if len(ring_neighbors) == 1 and mol.get_bond(atom.idx, ring_neighbors[0]).order == 1:
                target = ring_neighbors[0]
                external = True
            descriptor = FunctionalGroupDescriptor(
                family=FunctionalFamily.ACYL,
                derivative=DerivativeKind.HYDRAZIDE,
                centers=(atom.idx,),
                ligands=(double_ligand,),
                attachment_atom=target,
                is_external=external,
            )
            key, rule = resolve_acyl_rule(descriptor)
            group = PerceivedGroup(
                key,
                True,
                target,
                {atom.idx, double_ligand.atom, *hydrazide_nitrogens},
                descriptor=descriptor,
                resolved_rule=rule,
            )
            hydrazide_groups.append(group)
            hydrazide_claimed_atoms.update([double_ligand.atom, *hydrazide_nitrogens])
            break
    groups.extend(hydrazide_groups)
    consumed.update(hydrazide_claimed_atoms)

    # Central-atom hydrazides must likewise claim their N-N unit before the
    # generic hydrazine parent recognizer runs.
    for atom in mol:
        central_group = _central_chalcogen_derivative(mol, atom.idx, consumed, cyclic_atoms)
        if central_group is not None:
            groups.append(central_group)
            consumed.update(central_group.atoms_involved - {central_group.attachment_carbon})

    for atom in mol:
        if not atom.is_carbon or atom.idx in consumed:
            continue
        pseudohalide_group = _acyl_pseudohalide_group(mol, atom.idx, consumed, cyclic_atoms)
        if pseudohalide_group is not None:
            groups.append(pseudohalide_group)
            consumed.update(pseudohalide_group.atoms_involved - {pseudohalide_group.attachment_carbon})

    nitrogen_chains = nitrogen_chain_roles(mol, cyclic_atoms, consumed)
    for role in nitrogen_chains:
        groups.append(
            PerceivedGroup(
                role.key,
                role.is_principal_candidate,
                role.attachment_atom,
                set(role.atom_ids),
                variant=role.variant,
                role="nitrogen_chain",
                decision_reasons=(role.reason,),
            )
        )
        consumed.update(role.atom_ids)

    for atom in mol:
        if atom.symbol != "N" or atom.idx in consumed or atom.idx in cyclic_atoms:
            continue
        neighbors = mol.get_neighbors(atom.idx)
        # A nitrogen double-bonded to a hypervalent centre is an imino group
        # whether or not it also carries a substituent.  Requiring it to be
        # terminal sent `P(=N-tBu)` to the amine catch-all below, which does
        # not look at bond order, and the double bond vanished into a `-amine`
        # suffix.
        double_bonded = [n for n in neighbors if (bond := mol.get_bond(atom.idx, n)) is not None and bond.order == 2]
        if len(double_bonded) != 1:
            continue
        center = double_bonded[0]
        if mol.atoms[center].is_carbon:
            continue
        if any((bond := mol.get_bond(atom.idx, n)) is None or bond.order != 1 for n in neighbors if n != center):
            continue
        # A substituted imino only becomes a prefix on a ring centre, where the
        # ring is the parent regardless.  On an acyclic centre the nitrogen is
        # still a parent candidate itself -- CH3-N=P(CH3)3 is
        # N-(trimethylphosphanylidene)methanamine -- and this must not pre-empt
        # that competition.  A nitrogen carrying an acyl or another nitrogen
        # belongs to a senior group of its own (an amide, a hydrazone), so
        # claiming it here would steal it from that suffix and, for a cyclic
        # sulfoximine, open the ring it is double-bonded into.
        if len(neighbors) != 1 and (center not in cyclic_atoms or _has_senior_nitrogen_ligand(mol, atom.idx, center)):
            continue
        variant = "terminal_heteroatom_imino" if len(neighbors) == 1 else "substituted_heteroatom_imino"
        groups.append(
            PerceivedGroup(
                "imino_prefix",
                False,
                center,
                {atom.idx},
                variant=variant,
                role="chalcogen_imide",
                decision_reasons=(f"Matched terminal N double-bonded to heteroatom {center}.",),
            )
        )
        consumed.add(atom.idx)

    for atom in mol:
        if atom.symbol == "N" and atom.idx not in consumed and atom.idx not in cyclic_atoms:
            nitrogens = [n for n in mol.get_neighbors(atom.idx) if mol.atoms[n].symbol == "N"]
            adj_atoms = [n for n in mol.get_neighbors(atom.idx) if n not in nitrogens]
            if len(adj_atoms) == 1 and len(nitrogens) == 1:
                attach_bond = mol.get_bond(atom.idx, adj_atoms[0])
                if attach_bond is None or attach_bond.order != 1:
                    continue
                n2 = nitrogens[0]
                n2_nitrogens = [n for n in mol.get_neighbors(n2) if mol.atoms[n].symbol == "N" and n != atom.idx]
                if len(n2_nitrogens) == 1:
                    n3 = n2_nitrogens[0]
                    if mol.degree(n3) == 1:
                        groups.append(PerceivedGroup("azido", False, adj_atoms[0], {atom.idx, n2, n3}))
                        consumed.update([atom.idx, n2, n3])

    for group in _chalcogen_anhydride_groups(mol, consumed):
        groups.append(group)
        if group.is_principal_candidate:
            consumed.update(group.atoms_involved - {group.attachment_carbon})

    for atom in mol:
        if atom.symbol == "S" and atom.idx not in consumed:
            oxygens = [n for n in mol.get_neighbors(atom.idx) if mol.atoms[n].symbol == "O"]
            adj_atoms = [n for n in mol.get_neighbors(atom.idx) if n not in oxygens]
            sulfonyl_group = _sulfonyl_derivative(mol, atom.idx, oxygens, adj_atoms, cyclic_atoms)
            if sulfonyl_group is not None:
                key, c_idx, hetero = sulfonyl_group
                groups.append(PerceivedGroup(key, True, c_idx, {atom.idx, *oxygens, hetero}))
                consumed.update([atom.idx, *oxygens, hetero])
                continue
            if len(oxygens) >= 3 and len(adj_atoms) == 1:
                double_o_list = [o for o in oxygens if mol.get_bond(atom.idx, o).order == 2]
                # A sulfonate/sulfonic acid requires the sulfur to be bonded to a
                # carbon (C-SO2-O).  When the lone non-oxygen neighbour is instead a
                # halogen or heteroatom the sulfur centre is a chlorosulfate,
                # sulfamate or the like — not a sulfonate — so leaving it here lets
                # the ester/``…oxy`` prefix machinery name it and, crucially, keeps
                # it from being chosen as a bogus principal group hung off a non-
                # carbon "attachment carbon".
                if len(double_o_list) >= 2 and mol.atoms[adj_atoms[0]].is_carbon:
                    c_idx = adj_atoms[0]
                    single_o_list = [o for o in oxygens if mol.get_bond(atom.idx, o).order == 1]
                    if atom.idx in cyclic_atoms and any(o in cyclic_atoms for o in single_o_list):
                        continue
                    ester_o = next((o for o in single_o_list if mol.degree(o) == 2), None)
                    anion_o = next((o for o in single_o_list if mol.atoms[o].charge == -1), None)

                    if ester_o is not None or anion_o is not None:
                        key = "sulfonate"
                        groups.append(PerceivedGroup(key, True, c_idx, {atom.idx} | set(oxygens)))
                        consumed.update([atom.idx] + oxygens)
                    elif len(single_o_list) > 0:
                        key = "sulfonic_acid"
                        groups.append(PerceivedGroup(key, True, c_idx, {atom.idx} | set(oxygens)))
                        consumed.update([atom.idx] + oxygens)

    for atom in mol:
        if atom.is_carbon and atom.idx not in consumed:
            nitrogens = [n for n in mol.get_neighbors(atom.idx) if mol.atoms[n].symbol == "N" and n not in consumed]
            oxygens = [o for o in mol.get_neighbors(atom.idx) if mol.atoms[o].symbol == "O" and o not in consumed]
            sulfurs = [s for s in mol.get_neighbors(atom.idx) if mol.atoms[s].symbol == "S" and s not in consumed]

            double_n = next((n for n in nitrogens if mol.get_bond(atom.idx, n).order == 2), None)
            double_o = next((o for o in oxygens if mol.get_bond(atom.idx, o).order == 2 and mol.degree(o) == 1), None)
            double_s = next((s for s in sulfurs if mol.get_bond(atom.idx, s).order == 2 and mol.degree(s) == 1), None)

            if double_n is not None:
                if double_s is not None:
                    n_neighbors = [x for x in mol.get_neighbors(double_n) if x != atom.idx]
                    if len(n_neighbors) > 0:
                        groups.append(
                            PerceivedGroup("isothiocyanato", False, n_neighbors[0], {atom.idx, double_n, double_s})
                        )
                        consumed.update([atom.idx, double_n, double_s])
                        continue
                elif double_o is not None:
                    n_neighbors = [x for x in mol.get_neighbors(double_n) if x != atom.idx]
                    if len(n_neighbors) > 0:
                        groups.append(
                            PerceivedGroup("isocyanato", False, n_neighbors[0], {atom.idx, double_n, double_o})
                        )
                        consumed.update([atom.idx, double_n, double_o])
                        continue

            if double_n is not None and atom.idx not in cyclic_atoms:
                amidine = _amidine_group(mol, atom.idx, double_n, nitrogens, cyclic_atoms)
                if amidine is not None:
                    key, attachment, involved = amidine
                    groups.append(PerceivedGroup(key, True, attachment, involved))
                    consumed.update(involved - {attachment})
                    continue

            triple_n = next((n for n in nitrogens if mol.get_bond(atom.idx, n).order == 3), None)
            single_o = next((o for o in oxygens if mol.get_bond(atom.idx, o).order == 1), None)
            single_s = next((s for s in sulfurs if mol.get_bond(atom.idx, s).order == 1), None)

            if triple_n is not None:
                if single_s is not None:
                    s_neighbors = [x for x in mol.get_neighbors(single_s) if x != atom.idx]
                    if len(s_neighbors) > 0:
                        groups.append(
                            PerceivedGroup("thiocyanato", False, s_neighbors[0], {atom.idx, triple_n, single_s})
                        )
                        consumed.update([atom.idx, triple_n, single_s])
                        continue
                elif single_o is not None:
                    o_neighbors = [x for x in mol.get_neighbors(single_o) if x != atom.idx]
                    if len(o_neighbors) > 0:
                        groups.append(PerceivedGroup("cyanato", False, o_neighbors[0], {atom.idx, triple_n, single_o}))
                        consumed.update([atom.idx, triple_n, single_o])
                        continue

    for atom in mol:
        if atom.is_carbon and atom.idx not in consumed:
            nitrogens = [n for n in mol.get_neighbors(atom.idx) if mol.atoms[n].symbol == "N" and n not in consumed]
            triple_n = next((n for n in nitrogens if mol.get_bond(atom.idx, n).order == 3), None)
            if triple_n is not None:
                terminal_chalcogenide = next(
                    (
                        x
                        for x in mol.get_neighbors(triple_n)
                        if x != atom.idx
                        and chalcogen_for_symbol(mol.atoms[x].symbol) is not None
                        and mol.atoms[x].charge == -1
                        and mol.degree(x) == 1
                        and mol.get_bond(triple_n, x).order == 1
                    ),
                    None,
                )
                if terminal_chalcogenide is not None and mol.atoms[triple_n].charge == 1:
                    ring_neighbors = [n for n in mol.get_neighbors(atom.idx) if n in cyclic_atoms]
                    is_exocyclic = False
                    attached_ring_atom = None
                    if atom.idx not in cyclic_atoms and len(ring_neighbors) == 1:
                        attached_ring_atom = ring_neighbors[0]
                        if mol.get_bond(atom.idx, attached_ring_atom).order == 1:
                            is_exocyclic = True
                    target_carbon = attached_ring_atom if is_exocyclic else atom.idx
                    ligand = classify_chalcogen_ligand(mol, triple_n, terminal_chalcogenide)
                    if ligand is None:
                        continue
                    descriptor = FunctionalGroupDescriptor(
                        family=FunctionalFamily.NITRILE_CHALCOGENIDE,
                        derivative=DerivativeKind.ZWITTERION,
                        centers=(atom.idx, triple_n),
                        ligands=(ligand,),
                        attachment_atom=target_carbon,
                        is_external=is_exocyclic,
                    )
                    key, rule = resolve_nitrile_chalcogenide_rule(descriptor)
                    groups.append(
                        PerceivedGroup(
                            key,
                            True,
                            target_carbon,
                            {atom.idx, triple_n, terminal_chalcogenide},
                            descriptor=descriptor,
                            resolved_rule=rule,
                        )
                    )
                    consumed.update([triple_n, terminal_chalcogenide])
                    continue
                n_neighbors = [x for x in mol.get_neighbors(triple_n) if x != atom.idx]
                if len(n_neighbors) > 0:
                    groups.append(PerceivedGroup("isocyano", False, n_neighbors[0], {atom.idx, triple_n}))
                    consumed.update([atom.idx, triple_n])
                else:
                    ring_neighbors = [n for n in mol.get_neighbors(atom.idx) if n in cyclic_atoms]
                    is_exocyclic = False
                    attached_ring_atom = None
                    if atom.idx not in cyclic_atoms and len(ring_neighbors) == 1:
                        attached_ring_atom = ring_neighbors[0]
                        if mol.get_bond(atom.idx, attached_ring_atom).order == 1:
                            is_exocyclic = True
                    target_carbon = attached_ring_atom if is_exocyclic else atom.idx
                    key = "ring_nitrile" if is_exocyclic else "nitrile"
                    groups.append(PerceivedGroup(key, True, target_carbon, {atom.idx, triple_n}))
                    consumed.update([triple_n])

    for atom in mol:
        if atom.is_carbon:
            matched_acyl = _acyl_chalcogen_group(mol, atom.idx, consumed, cyclic_atoms)
            acyl_groups = matched_acyl if isinstance(matched_acyl, tuple) else (matched_acyl,)
            for acyl_group in acyl_groups:
                if acyl_group is None:
                    continue
                groups.append(acyl_group)
                if acyl_group.is_principal_candidate:
                    consumed.update(acyl_group.atoms_involved - {acyl_group.attachment_carbon})

    for atom in mol:
        if atom.symbol == "N" and atom.idx not in consumed and atom.idx not in cyclic_atoms:
            c_neighbors = [n for n in mol.get_neighbors(atom.idx) if mol.atoms[n].is_carbon]
            n_neighbors = [n for n in mol.get_neighbors(atom.idx) if mol.atoms[n].symbol == "N"]

            double_c = next((c for c in c_neighbors if mol.get_bond(atom.idx, c).order == 2), None)

            if double_c is not None:
                # N-nitro/nitroso is a ligand of an imine, not a hydrazone
                # tail. Preserve C=N before the multiple-bond tail guard.
                if any(
                    mol.get_bond(atom.idx, n).order == 1 and _is_nitroso_or_nitro_nitrogen(mol, n, atom.idx)
                    for n in n_neighbors
                ):
                    key = "iminium" if atom.charge > 0 else "imine"
                    groups.append(PerceivedGroup(key, True, double_c, {atom.idx}))
                    consumed.add(atom.idx)
                    continue
                if len(n_neighbors) > 0:
                    n2 = n_neighbors[0]
                    if mol.get_bond(atom.idx, n2).order == 1:
                        amidino_tail_atoms = amidinohydrazone_tail_atoms(mol, n2, {atom.idx})
                        if has_non_h_multiple_bond_neighbor(mol, n2, {atom.idx}) and not amidino_tail_atoms:
                            continue
                        if n2 not in cyclic_atoms:
                            ring_neighbors = [n for n in mol.get_neighbors(double_c) if n in cyclic_atoms]
                            c_of_double_c = [n for n in mol.get_neighbors(double_c) if mol.atoms[n].is_carbon]
                            hydrazone_atoms = {atom.idx, n2} | amidino_tail_atoms
                            if (
                                double_c not in cyclic_atoms
                                and len(ring_neighbors) == 1
                                and len(c_of_double_c) == 1
                                and mol.get_bond(double_c, ring_neighbors[0]).order == 1
                            ):
                                key = (
                                    "ring_aldehyde_amidinohydrazone"
                                    if amidino_tail_atoms
                                    else "ring_aldehyde_hydrazone"
                                )
                                groups.append(
                                    PerceivedGroup(
                                        key,
                                        True,
                                        ring_neighbors[0],
                                        {double_c} | hydrazone_atoms,
                                    )
                                )
                            else:
                                if len(c_of_double_c) <= 1 and double_c not in cyclic_atoms:
                                    key = "aldehyde_amidinohydrazone" if amidino_tail_atoms else "aldehyde_hydrazone"
                                    groups.append(PerceivedGroup(key, True, double_c, hydrazone_atoms))
                                else:
                                    groups.append(PerceivedGroup("hydrazone", True, double_c, hydrazone_atoms))
                            consumed.update(hydrazone_atoms)
                        else:
                            key = "iminium" if atom.charge > 0 else "imine"
                            groups.append(PerceivedGroup(key, True, double_c, {atom.idx}))
                            consumed.update([atom.idx])
                    else:
                        key = "iminium" if atom.charge > 0 else "imine"
                        groups.append(PerceivedGroup(key, True, double_c, {atom.idx}))
                        consumed.update([atom.idx])
                else:
                    key = "iminium" if atom.charge > 0 else "imine"
                    groups.append(PerceivedGroup(key, True, double_c, {atom.idx}))
                    consumed.update([atom.idx])

    for atom in mol:
        if atom.symbol == "N" and atom.idx not in consumed and atom.idx not in cyclic_atoms and not atom.charge:
            c_neighbors = [n for n in mol.get_neighbors(atom.idx) if mol.atoms[n].is_carbon]
            n_neighbors = [
                n for n in mol.get_neighbors(atom.idx) if mol.atoms[n].symbol == "N" and not mol.atoms[n].charge
            ]

            if len(n_neighbors) > 0:
                n2 = n_neighbors[0]
                c_att = c_neighbors[0] if c_neighbors else None
                if c_att:
                    c_att_bond = mol.get_bond(atom.idx, c_att)
                    if (
                        c_att_bond is not None
                        and c_att_bond.order == 1
                        and mol.get_bond(atom.idx, n2).order == 1
                        and not has_non_h_multiple_bond_neighbor(mol, n2, {atom.idx})
                    ):
                        if n2 not in cyclic_atoms:
                            groups.append(
                                PerceivedGroup(
                                    "hydrazine",
                                    False,
                                    c_att,
                                    {atom.idx, n2},
                                    variant="prefix",
                                    role="nitrogen_chain",
                                    decision_reasons=(
                                        f"Matched C-N-N hydrazine fragment at atom {c_att}; render as prefix.",
                                    ),
                                )
                            )
                            consumed.update([atom.idx, n2])
                        else:
                            groups.append(PerceivedGroup("amine", True, c_att, {atom.idx}))
                            consumed.update([atom.idx])

    for terminal_atom in mol:
        if nitrogen_chains:
            break
        if (
            terminal_atom.idx in consumed
            or terminal_atom.idx in cyclic_atoms
            or chalcogen_for_symbol(terminal_atom.symbol) is None
            or mol.degree(terminal_atom.idx) != 1
            or terminal_atom.total_h_count == 0
            or terminal_atom.radical_electrons
        ):
            continue
        first_atom = mol.get_neighbors(terminal_atom.idx)[0]
        if chalcogen_for_symbol(mol.atoms[first_atom].symbol) is None or first_atom in consumed:
            continue
        anchors = [neighbor for neighbor in mol.get_neighbors(first_atom) if neighbor != terminal_atom.idx]
        if len(anchors) != 1 or not mol.atoms[anchors[0]].is_carbon:
            continue
        anchor = anchors[0]
        if any(
            neighbor != first_atom
            and (candidate := classify_chalcogen_ligand(mol, anchor, neighbor)) is not None
            and candidate.role is ChalcogenLigandRole.DOUBLE_BONDED
            for neighbor in mol.get_neighbors(anchor)
        ):
            continue
        first_ligand = classify_chalcogen_ligand(mol, anchor, first_atom)
        terminal_ligand = classify_chalcogen_ligand(mol, first_atom, terminal_atom.idx)
        if (
            first_ligand is None
            or first_ligand.role is not ChalcogenLigandRole.CHALCOGEN_LINK
            or terminal_ligand is None
            or terminal_ligand.role is not ChalcogenLigandRole.HYDROGEN_BEARING
        ):
            continue
        descriptor = FunctionalGroupDescriptor(
            family=FunctionalFamily.PEROXOL,
            derivative=DerivativeKind.ALCOHOL,
            centers=(anchor,),
            ligands=(first_ligand, terminal_ligand),
            linker_paths=((anchor, first_atom, terminal_atom.idx),),
            attachment_atom=anchor,
        )
        if uses_existing_chalcogen_citation(
            ChalcogenCitationContext.PEROXOL,
            (first_ligand.element, terminal_ligand.element),
            bridge_atom_count=2,
        ):
            groups.append(
                PerceivedGroup(
                    FunctionalFamily.PEROXOL.value,
                    False,
                    anchor,
                    {first_atom, terminal_atom.idx},
                    descriptor=descriptor,
                )
            )
            continue
        key, rule = resolve_peroxol_rule(descriptor)
        groups.append(
            PerceivedGroup(
                key,
                True,
                anchor,
                {first_atom, terminal_atom.idx},
                descriptor=descriptor,
                resolved_rule=rule,
            )
        )
        consumed.update({first_atom, terminal_atom.idx})

    for atom in mol:
        if atom.idx in consumed or atom.idx in cyclic_atoms or chalcogen_for_symbol(atom.symbol) is None:
            continue
        if mol.degree(atom.idx) != 1:
            continue
        center = mol.get_neighbors(atom.idx)[0]
        ligand = classify_chalcogen_ligand(mol, center, atom.idx)
        if ligand is None:
            continue
        if ligand.role is ChalcogenLigandRole.DOUBLE_BONDED:
            if not mol.atoms[center].is_carbon:
                continue
            if ligand.element is not Chalcogen.OXYGEN and any(candidate.charge for candidate in mol):
                continue
            if ligand.element is not Chalcogen.OXYGEN and any(
                neighbor != atom.idx and not mol.atoms[neighbor].is_carbon for neighbor in mol.get_neighbors(center)
            ):
                continue
            ring_neighbors = [neighbor for neighbor in mol.get_neighbors(center) if neighbor in cyclic_atoms]
            external = (
                center not in cyclic_atoms
                and mol.atoms[center].total_h_count > 0
                and len(ring_neighbors) == 1
                and mol.get_bond(center, ring_neighbors[0]).order == 1
            )
            derivative = (
                DerivativeKind.ALDEHYDE
                if mol.atoms[center].total_h_count > 0 and center not in cyclic_atoms
                else DerivativeKind.KETONE
            )
            key = simple_group_key(FunctionalFamily.CARBONYL, derivative, ligand.element, external=external)
            if key is None:
                continue
            attachment = ring_neighbors[0] if external else center
            descriptor = FunctionalGroupDescriptor(
                family=FunctionalFamily.CARBONYL,
                derivative=derivative,
                centers=(center,),
                ligands=(ligand,),
                attachment_atom=attachment,
                is_external=external,
            )
            involved = {center, atom.idx} if derivative is DerivativeKind.ALDEHYDE else {atom.idx}
            groups.append(PerceivedGroup(key, True, attachment, involved, descriptor=descriptor))
            consumed.add(atom.idx)
            continue
        if ligand.role not in {ChalcogenLigandRole.HYDROGEN_BEARING, ChalcogenLigandRole.ANIONIC}:
            continue
        if not mol.atoms[center].is_carbon and not (
            ligand.role is ChalcogenLigandRole.ANIONIC
            and ligand.element is not Chalcogen.OXYGEN
            and mol.atoms[center].symbol == "N"
            and mol.atoms[center].charge > 0
        ):
            continue
        derivative = DerivativeKind.ANION if atom.charge < 0 else DerivativeKind.ALCOHOL
        key = simple_group_key(FunctionalFamily.HYDROXY, derivative, ligand.element)
        if key is None:
            continue
        descriptor = FunctionalGroupDescriptor(
            family=FunctionalFamily.HYDROXY,
            derivative=derivative,
            centers=(center,),
            ligands=(ligand,),
            attachment_atom=center,
        )
        groups.append(PerceivedGroup(key, True, center, {atom.idx}, descriptor=descriptor))
        consumed.add(atom.idx)

    for atom in mol:
        if atom.symbol == "N" and atom.idx not in consumed and atom.idx not in cyclic_atoms:
            adj_atoms = mol.get_neighbors(atom.idx)
            if len(adj_atoms) > 0:
                # aminide is the mono-anion suffix; a doubly charged nitrogen
                # would be an azanediide, which nothing here can spell.
                key = "aminium" if atom.charge > 0 else "aminide" if atom.charge == -1 else "amine"

                principal = key != "aminium" or all(mol.get_bond(atom.idx, n).order == 1 for n in adj_atoms)
                for c in adj_atoms:
                    groups.append(PerceivedGroup(key, principal, c, {atom.idx}))
                consumed.add(atom.idx)

    halogen_map = {"F": "fluoro", "Cl": "chloro", "Br": "bromo", "I": "iodo"}
    for atom in mol:
        if atom.symbol in halogen_map and atom.idx not in consumed and atom.idx not in cyclic_atoms:
            if mol.degree(atom.idx) == 1:
                adj_atoms = mol.get_neighbors(atom.idx)
                if len(adj_atoms) == 1:
                    groups.append(PerceivedGroup(halogen_map[atom.symbol], False, adj_atoms[0], {atom.idx}))
                    consumed.add(atom.idx)

    return groups


def _composable_chalcogen_linkage_groups(mol: Molecule) -> list[PerceivedGroup]:
    """Describe nonprincipal E-E and nitrogen-E linkages without assuming uniqueness."""

    groups = [
        PerceivedGroup(
            FunctionalFamily.PEROXIDE.value,
            False,
            descriptor.attachment_atom if descriptor.attachment_atom is not None else descriptor.centers[0],
            set(descriptor.centers),
            descriptor=descriptor,
        )
        for descriptor in classify_peroxide_linkages(mol, set(mol.atoms))
    ]
    for role in charge_pair_roles(mol):
        if role.nitrogen_kind not in {NitrogenChalcogenideKind.AMINE, NitrogenChalcogenideKind.IMINE}:
            continue
        ligand = classify_chalcogen_ligand(mol, role.positive_atom, role.negative_atom)
        if ligand is None:
            continue
        organic_neighbors = tuple(
            neighbor
            for neighbor in mol.get_neighbors(role.positive_atom)
            if neighbor != role.negative_atom and mol.atoms[neighbor].is_carbon
        )
        attachment = organic_neighbors[0] if organic_neighbors else role.positive_atom
        descriptor = FunctionalGroupDescriptor(
            family=FunctionalFamily.NITROGEN_CHALCOGENIDE,
            derivative=DerivativeKind.ZWITTERION,
            centers=(role.positive_atom,),
            ligands=(ligand,),
            attachment_atom=attachment,
        )
        groups.append(
            PerceivedGroup(
                FunctionalFamily.NITROGEN_CHALCOGENIDE.value,
                False,
                attachment,
                set(role.atom_ids),
                descriptor=descriptor,
            )
        )
    return groups


def _acyl_chalcogen_group(
    mol: Molecule,
    carbon: int,
    consumed: set[int],
    cyclic_atoms: set[int],
) -> PerceivedGroup | tuple[PerceivedGroup, ...] | None:
    """Classify one complete C(=E) derivative before simpler fragments."""

    ligands = [
        ligand
        for neighbor in mol.get_neighbors(carbon)
        if neighbor not in consumed and (ligand := classify_chalcogen_ligand(mol, carbon, neighbor)) is not None
    ]
    double_ligands = [ligand for ligand in ligands if ligand.role is ChalcogenLigandRole.DOUBLE_BONDED]
    if len(double_ligands) != 1:
        return None
    double_ligand = double_ligands[0]
    ring_neighbors = [neighbor for neighbor in mol.get_neighbors(carbon) if neighbor in cyclic_atoms]
    external = (
        carbon not in cyclic_atoms and len(ring_neighbors) == 1 and mol.get_bond(carbon, ring_neighbors[0]).order == 1
    )
    attachment = ring_neighbors[0] if external else carbon

    nitrogens = [
        neighbor
        for neighbor in mol.get_neighbors(carbon)
        if neighbor not in consumed and mol.atoms[neighbor].symbol == "N" and mol.get_bond(carbon, neighbor).order == 1
    ]
    halogens = [
        neighbor
        for neighbor in mol.get_neighbors(carbon)
        if neighbor not in consumed
        and mol.atoms[neighbor].symbol in {"F", "Cl", "Br", "I"}
        and mol.get_bond(carbon, neighbor).order == 1
    ]

    single_ligands = [
        ligand
        for ligand in ligands
        if ligand.role
        in {
            ChalcogenLigandRole.HYDROGEN_BEARING,
            ChalcogenLigandRole.ANIONIC,
            ChalcogenLigandRole.CARBON_LINK,
            ChalcogenLigandRole.CHALCOGEN_LINK,
        }
    ]
    # A centre carrying a nitrogen or a typed leaving group is not a plain
    # ester merely because it also has a single-bonded chalcogen.  Defer those
    # competing ligands to their complete derivative routes below.
    if len(single_ligands) == 1 and not nitrogens and not halogens:
        single_ligand = single_ligands[0]
        if single_ligand.role is ChalcogenLigandRole.CHALCOGEN_LINK:
            terminal_atom = single_ligand.attachment_atom
            if terminal_atom is None:
                return None
            if (
                carbon in cyclic_atoms
                and single_ligand.atom in cyclic_atoms
                and terminal_atom in cyclic_atoms
                and _closes_ring_back_to(mol, carbon, single_ligand.atom, double_ligand.atom, cyclic_atoms)
            ):
                return _carbonyl_chalcogen_group(mol, carbon, attachment, double_ligand, DerivativeKind.KETONE)
            terminal_ligand = classify_chalcogen_ligand(mol, single_ligand.atom, terminal_atom)
            if terminal_ligand is None:
                return None
            derivative_by_role = {
                ChalcogenLigandRole.HYDROGEN_BEARING: DerivativeKind.ACID,
                ChalcogenLigandRole.ANIONIC: DerivativeKind.ANION,
                ChalcogenLigandRole.CARBON_LINK: DerivativeKind.ESTER,
            }
            derivative = derivative_by_role.get(terminal_ligand.role)
            if derivative is None:
                return None
            descriptor = FunctionalGroupDescriptor(
                family=FunctionalFamily.ACYL,
                derivative=derivative,
                centers=(carbon,),
                ligands=(double_ligand, single_ligand, terminal_ligand),
                linker_paths=((carbon, single_ligand.atom, terminal_ligand.atom),),
                attachment_atom=attachment,
                is_external=external,
            )
            projected = _established_acyl_projection(descriptor)
            if projected is not None:
                return projected
            key, rule = resolve_peroxy_acyl_rule(descriptor)
            return PerceivedGroup(
                key,
                True,
                attachment,
                {carbon, double_ligand.atom, single_ligand.atom, terminal_ligand.atom},
                descriptor=descriptor,
                resolved_rule=rule,
            )
        if (
            single_ligand.role is ChalcogenLigandRole.CARBON_LINK
            and carbon in cyclic_atoms
            and single_ligand.atom in cyclic_atoms
            and _closes_ring_back_to(mol, carbon, single_ligand.atom, double_ligand.atom, cyclic_atoms)
        ):
            return _carbonyl_chalcogen_group(mol, carbon, attachment, double_ligand, DerivativeKind.KETONE)
        derivative_by_role = {
            ChalcogenLigandRole.HYDROGEN_BEARING: DerivativeKind.ACID,
            ChalcogenLigandRole.ANIONIC: DerivativeKind.ANION,
            ChalcogenLigandRole.CARBON_LINK: DerivativeKind.ESTER,
        }
        derivative = derivative_by_role[single_ligand.role]
        descriptor = FunctionalGroupDescriptor(
            family=FunctionalFamily.ACYL,
            derivative=derivative,
            centers=(carbon,),
            ligands=(double_ligand, single_ligand),
            linker_paths=((carbon, single_ligand.atom, single_ligand.attachment_atom),)
            if single_ligand.attachment_atom is not None
            else (),
            attachment_atom=attachment,
            is_external=external,
        )
        projected = _established_acyl_projection(descriptor)
        if projected is not None:
            return projected
        key, rule = resolve_acyl_rule(descriptor)
        return PerceivedGroup(
            key,
            True,
            attachment,
            {carbon, double_ligand.atom, single_ligand.atom},
            descriptor=descriptor,
            resolved_rule=rule,
        )

    if len(halogens) == 1 and not nitrogens:
        leaving = halogens[0]
        leaving_group = _leaving_group_descriptor(
            mol,
            AcylLeavingGroup(mol.atoms[leaving].symbol),
            carbon,
            (leaving,),
        )
        descriptor = FunctionalGroupDescriptor(
            family=FunctionalFamily.ACYL,
            derivative=DerivativeKind.ACID_HALIDE,
            centers=(carbon,),
            ligands=(double_ligand,),
            attachment_atom=attachment,
            is_external=external,
            leaving_group=leaving_group,
        )
        key, rule = resolve_acyl_rule(descriptor)
        return PerceivedGroup(
            key,
            True,
            attachment,
            {carbon, double_ligand.atom, leaving},
            descriptor=descriptor,
            resolved_rule=rule,
        )

    if not nitrogens:
        return None
    nitrogens.sort(
        key=lambda nitrogen: (
            nitrogen not in cyclic_atoms,
            any(neighbor != carbon and mol.atoms[neighbor].symbol != "H" for neighbor in mol.get_neighbors(nitrogen)),
        ),
        reverse=True,
    )
    nitrogen = nitrogens[0]
    is_lactam = (
        carbon in cyclic_atoms
        and nitrogen in cyclic_atoms
        and _closes_ring_back_to(mol, carbon, nitrogen, double_ligand.atom, cyclic_atoms)
    )
    if is_lactam:
        return _carbonyl_chalcogen_group(mol, carbon, attachment, double_ligand, DerivativeKind.KETONE)
    hydrazide_nitrogens = _hydrazide_nitrogens(mol, carbon, nitrogen, cyclic_atoms)
    if hydrazide_nitrogens is not None:
        derivative = DerivativeKind.HYDRAZIDE
        group_nitrogens = hydrazide_nitrogens
    else:
        urea_nitrogens = _urea_nitrogens(mol, carbon, nitrogens, cyclic_atoms)
        if urea_nitrogens is not None:
            derivative = DerivativeKind.UREA
            group_nitrogens = urea_nitrogens
            attachment = carbon
            external = False
        elif nitrogen not in cyclic_atoms:
            derivative = DerivativeKind.AMIDE
            group_nitrogens = (nitrogen,)
        else:
            return None
    descriptor = FunctionalGroupDescriptor(
        family=FunctionalFamily.ACYL,
        derivative=derivative,
        centers=(carbon,),
        ligands=(double_ligand,),
        attachment_atom=attachment,
        is_external=external,
    )
    key, rule = resolve_acyl_rule(descriptor)
    return PerceivedGroup(
        key,
        True,
        attachment,
        {carbon, double_ligand.atom, *group_nitrogens},
        descriptor=descriptor,
        resolved_rule=rule,
    )


def _established_acyl_projection(
    descriptor: FunctionalGroupDescriptor,
) -> PerceivedGroup | tuple[PerceivedGroup, ...] | None:
    """Project supported O/S acyl paths onto established structural detectors."""

    double = descriptor.ligands[0]
    bridge = descriptor.ligands[1:]
    citation_rule = chalcogen_citation_rule(
        ChalcogenCitationContext.ACYL,
        tuple(ligand.element for ligand in descriptor.ligands),
        bridge_atom_count=len(bridge),
        site_elements=(double.element,),
        bridge_elements=tuple(ligand.element for ligand in bridge),
    )
    if citation_rule is None:
        return None
    center = descriptor.centers[0]
    if citation_rule.projection is ChalcogenCitationProjection.CARBONYL:
        return PerceivedGroup(
            FunctionalFamily.ACYL.value,
            False,
            descriptor.attachment_atom if descriptor.attachment_atom is not None else center,
            set(descriptor.atom_ids),
            descriptor=descriptor,
        )
    if citation_rule.projection is ChalcogenCitationProjection.ESTER:
        key = RULES.chalcogens.standard_acyl_keys[
            (
                DerivativeKind.ESTER.value,
                Chalcogen.OXYGEN.value,
                Chalcogen.OXYGEN.value,
                descriptor.is_external,
            )
        ]
        first_linker = bridge[0]
        attachment = descriptor.attachment_atom if descriptor.attachment_atom is not None else center
        return (
            PerceivedGroup(
                FunctionalFamily.ACYL.value,
                False,
                attachment,
                set(descriptor.atom_ids),
                descriptor=descriptor,
            ),
            PerceivedGroup(
                key,
                True,
                attachment,
                {center, double.atom, first_linker.atom},
            ),
        )
    return None


def _carbonyl_chalcogen_group(
    mol: Molecule,
    carbon: int,
    attachment: int,
    ligand: ChalcogenLigand,
    derivative: DerivativeKind,
) -> PerceivedGroup | None:
    external = attachment != carbon
    key = simple_group_key(FunctionalFamily.CARBONYL, derivative, ligand.element, external=external)
    if key is None:
        return None
    descriptor = FunctionalGroupDescriptor(
        family=FunctionalFamily.CARBONYL,
        derivative=derivative,
        centers=(carbon,),
        ligands=(ligand,),
        attachment_atom=attachment,
        is_external=external,
    )
    return PerceivedGroup(key, True, attachment, {carbon, ligand.atom}, descriptor=descriptor)


def _pseudohalide_leaving_group(
    mol: Molecule,
    center: int,
    double_ligand: int,
    consumed: set[int],
) -> LeavingGroupDescriptor | None:
    """Classify a complete pseudohalide attached directly to an acyl center."""

    for first in mol.get_neighbors(center):
        if first == double_ligand or first in consumed or mol.get_bond(center, first).order != 1:
            continue
        second_neighbors = [neighbor for neighbor in mol.get_neighbors(first) if neighbor != center]
        if len(second_neighbors) != 1:
            continue
        second = second_neighbors[0]
        first_symbol = mol.atoms[first].symbol
        second_symbol = mol.atoms[second].symbol
        first_bond = mol.get_bond(first, second)
        if first_symbol == "C" and second_symbol == "N" and first_bond.order == 3 and mol.degree(second) == 1:
            return _leaving_group_descriptor(mol, AcylLeavingGroup.CYANIDE, center, (first, second))
        if first_symbol != "N":
            continue
        if second_symbol == "C" and first_bond.order == 3 and mol.degree(second) == 1:
            return _leaving_group_descriptor(mol, AcylLeavingGroup.ISOCYANIDE, center, (first, second))
        third_neighbors = [neighbor for neighbor in mol.get_neighbors(second) if neighbor != first]
        if len(third_neighbors) != 1:
            continue
        third = third_neighbors[0]
        second_bond = mol.get_bond(second, third)
        if (
            second_symbol == "N"
            and mol.atoms[third].symbol == "N"
            and mol.atoms[second].charge > 0
            and mol.atoms[third].charge < 0
        ):
            return _leaving_group_descriptor(mol, AcylLeavingGroup.AZIDE, center, (first, second, third))
        terminal_element = chalcogen_for_symbol(mol.atoms[third].symbol)
        if second_symbol != "C" or first_bond.order != 2 or second_bond.order != 2 or terminal_element is None:
            continue
        if terminal_element is Chalcogen.OXYGEN:
            kind = AcylLeavingGroup.ISOCYANATE
        elif terminal_element is Chalcogen.SULFUR:
            kind = AcylLeavingGroup.ISOTHIOCYANATE
        elif terminal_element is Chalcogen.SELENIUM:
            kind = AcylLeavingGroup.ISOSELENOCYANATE
        elif terminal_element is Chalcogen.TELLURIUM:
            kind = AcylLeavingGroup.ISOTELLUROCYANATE
        else:
            continue
        return _leaving_group_descriptor(mol, kind, center, (first, second, third))
    return None


def _acyl_pseudohalide_group(
    mol: Molecule,
    carbon: int,
    consumed: set[int],
    cyclic_atoms: set[int],
) -> PerceivedGroup | None:
    """Bind an acyl center and its complete pseudohalide before simpler detectors."""

    double_ligand = _terminal_double_chalcogen(mol, carbon, consumed)
    if double_ligand is None:
        return None
    leaving_group = _pseudohalide_leaving_group(mol, carbon, double_ligand.atom, consumed)
    if leaving_group is None:
        return None
    owned_neighbors = {double_ligand.atom, leaving_group.attachment_atom}
    if any(
        neighbor not in owned_neighbors and not mol.atoms[neighbor].is_carbon for neighbor in mol.get_neighbors(carbon)
    ):
        return None
    ring_neighbors = [neighbor for neighbor in mol.get_neighbors(carbon) if neighbor in cyclic_atoms]
    external = (
        carbon not in cyclic_atoms and len(ring_neighbors) == 1 and mol.get_bond(carbon, ring_neighbors[0]).order == 1
    )
    attachment = ring_neighbors[0] if external else carbon
    descriptor = FunctionalGroupDescriptor(
        family=FunctionalFamily.ACYL,
        derivative=DerivativeKind.ACYL_PSEUDOHALIDE,
        centers=(carbon,),
        ligands=(double_ligand,),
        linker_paths=((carbon, *leaving_group.atom_ids),),
        attachment_atom=attachment,
        is_external=external,
        leaving_group=leaving_group,
    )
    key, rule = resolve_acyl_rule(descriptor)
    return PerceivedGroup(
        key,
        True,
        attachment,
        {carbon, double_ligand.atom, *leaving_group.atom_ids},
        descriptor=descriptor,
        resolved_rule=rule,
    )


def _cyclic_imide_group(
    mol: Molecule,
    nitrogen: int,
    consumed: set[int],
    cyclic_atoms: set[int],
) -> PerceivedGroup | None:
    """Return an eligible cyclic imide with two acyl centers sharing nitrogen."""

    atom = mol.atoms[nitrogen]
    if atom.symbol != "N" or atom.charge or nitrogen in consumed or nitrogen not in cyclic_atoms:
        return None
    centers = tuple(
        sorted(
            neighbor
            for neighbor in mol.get_neighbors(nitrogen)
            if neighbor in cyclic_atoms
            and mol.atoms[neighbor].is_carbon
            and mol.get_bond(nitrogen, neighbor).order == 1
        )
    )
    if len(centers) != 2:
        return None
    if any(
        neighbor not in centers and mol.get_bond(nitrogen, neighbor).order != 1
        for neighbor in mol.get_neighbors(nitrogen)
    ):
        return None
    ligands = tuple(_terminal_double_chalcogen(mol, center, consumed) for center in centers)
    if any(ligand is None for ligand in ligands):
        return None
    double_ligands = tuple(ligand for ligand in ligands if ligand is not None)
    blocked = {nitrogen, *(ligand.atom for ligand in double_ligands)}
    if not _connected_without_bridge(mol, centers[0], centers[1], blocked):
        return None
    descriptor = FunctionalGroupDescriptor(
        family=FunctionalFamily.ACYL,
        derivative=DerivativeKind.IMIDE,
        centers=centers,
        ligands=double_ligands,
        linker_paths=((centers[0], nitrogen, centers[1]),),
        attachment_atom=centers[0],
        shared_atoms=(nitrogen,),
    )
    key, rule = resolve_imide_route_rule(descriptor)
    involved = {nitrogen, *centers, *(ligand.atom for ligand in double_ligands)}
    return PerceivedGroup(
        key,
        True,
        centers[0],
        involved,
        descriptor=descriptor,
        resolved_rule=rule,
    )


def _leaving_group_descriptor(
    mol: Molecule,
    kind: AcylLeavingGroup,
    center: int,
    atoms: tuple[int, ...],
) -> LeavingGroupDescriptor:
    path = (center, *atoms)
    bonds = tuple(mol.get_bond(left, right).idx for left, right in zip(path, path[1:]))
    return LeavingGroupDescriptor(
        kind=kind,
        attachment_atom=atoms[0],
        atom_ids=atoms,
        bond_ids=bonds,
    )


def _chalcogen_anhydride_groups(mol: Molecule, consumed: set[int]) -> list[PerceivedGroup]:
    """Return acyclic C(=E)-E[-E]-C(=E) groups with ordered bridges."""

    groups = []
    seen: set[tuple[frozenset[int], tuple[int, ...]]] = set()
    for first_center in mol:
        if not first_center.is_carbon or first_center.idx in consumed:
            continue
        first_double = _terminal_double_chalcogen(mol, first_center.idx, consumed)
        if first_double is None:
            continue
        for first_bridge_atom in mol.get_neighbors(first_center.idx):
            if first_bridge_atom in consumed or first_bridge_atom == first_double.atom:
                continue
            first_bridge = classify_chalcogen_ligand(mol, first_center.idx, first_bridge_atom)
            if first_bridge is None or first_bridge.role not in {
                ChalcogenLigandRole.CARBON_LINK,
                ChalcogenLigandRole.CHALCOGEN_LINK,
            }:
                continue
            if first_bridge.role is ChalcogenLigandRole.CARBON_LINK:
                second_center = first_bridge.attachment_atom
                bridge_atoms = (first_bridge_atom,)
                bridge_ligands = (first_bridge,)
            else:
                second_bridge_atom = first_bridge.attachment_atom
                if second_bridge_atom is None:
                    continue
                second_bridge = classify_chalcogen_ligand(mol, first_bridge_atom, second_bridge_atom)
                if second_bridge is None or second_bridge.role is not ChalcogenLigandRole.CARBON_LINK:
                    continue
                second_center = second_bridge.attachment_atom
                bridge_atoms = (first_bridge_atom, second_bridge_atom)
                bridge_ligands = (first_bridge, second_bridge)
            if second_center is None or second_center == first_center.idx or not mol.atoms[second_center].is_carbon:
                continue
            second_double = _terminal_double_chalcogen(mol, second_center, consumed)
            if second_double is None or _connected_without_bridge(
                mol, first_center.idx, second_center, set(bridge_atoms)
            ):
                continue
            identity = (frozenset({first_center.idx, second_center}), tuple(sorted(bridge_atoms)))
            if identity in seen:
                continue
            seen.add(identity)
            if first_center.idx <= second_center:
                centers = (first_center.idx, second_center)
                doubles = (first_double, second_double)
                ordered_bridge_atoms = bridge_atoms
                ordered_bridge_ligands = bridge_ligands
            else:
                centers = (second_center, first_center.idx)
                doubles = (second_double, first_double)
                ordered_bridge_atoms = tuple(reversed(bridge_atoms))
                ordered_bridge_ligands = tuple(reversed(bridge_ligands))
            descriptor = FunctionalGroupDescriptor(
                family=FunctionalFamily.ACYL,
                derivative=DerivativeKind.ANHYDRIDE,
                centers=centers,
                ligands=(*doubles, *ordered_bridge_ligands),
                linker_paths=((centers[0], *ordered_bridge_atoms, centers[1]),),
                attachment_atom=centers[0],
            )
            citation_rule = chalcogen_citation_rule(
                ChalcogenCitationContext.ANHYDRIDE,
                tuple(ligand.element for ligand in descriptor.ligands),
                bridge_atom_count=len(bridge_atoms),
                site_elements=tuple(ligand.element for ligand in doubles),
                bridge_elements=tuple(ligand.element for ligand in ordered_bridge_ligands),
            )
            if citation_rule is not None:
                groups.append(
                    PerceivedGroup(
                        FunctionalFamily.ACYL.value,
                        False,
                        centers[0],
                        set(descriptor.atom_ids),
                        descriptor=descriptor,
                    )
                )
                groups.extend(_established_anhydride_projection(mol, descriptor, citation_rule.projection))
                continue
            key, rule = resolve_anhydride_rule(descriptor)
            groups.append(
                PerceivedGroup(
                    key,
                    True,
                    centers[0],
                    {first_double.atom, second_double.atom, *bridge_atoms},
                    descriptor=descriptor,
                    resolved_rule=rule,
                )
            )
    return groups


def _established_anhydride_projection(
    mol: Molecule,
    descriptor: FunctionalGroupDescriptor,
    projection: ChalcogenCitationProjection,
) -> list[PerceivedGroup]:
    """Project a recognized bridge onto an established ester/carbonyl route."""

    centers = descriptor.centers
    doubles = descriptor.ligands[:2]
    bridge = descriptor.ligands[2:]
    if projection is ChalcogenCitationProjection.PEROXY_ESTER:
        key = RULES.chalcogens.standard_peroxy_acyl_keys[(DerivativeKind.ESTER.value, False)]
        ester_descriptor = FunctionalGroupDescriptor(
            family=FunctionalFamily.ACYL,
            derivative=DerivativeKind.ESTER,
            centers=(centers[0],),
            ligands=(doubles[0], *bridge),
            linker_paths=(descriptor.linker_paths[0],),
            attachment_atom=centers[0],
        )
        return [
            PerceivedGroup(
                key,
                True,
                centers[0],
                {centers[0], doubles[0].atom, *(ligand.atom for ligand in bridge)},
                descriptor=ester_descriptor,
            )
        ]

    if projection is ChalcogenCitationProjection.ESTER_OR_CARBONYL:
        for index, (center, double) in enumerate(zip(centers, doubles, strict=True)):
            adjacent_bridge = bridge[0] if index == 0 else bridge[-1]
            if double.element is not Chalcogen.OXYGEN or adjacent_bridge.element is not Chalcogen.OXYGEN:
                continue
            key = RULES.chalcogens.standard_acyl_keys[
                (DerivativeKind.ESTER.value, Chalcogen.OXYGEN.value, Chalcogen.OXYGEN.value, False)
            ]
            return [PerceivedGroup(key, True, center, {center, double.atom, adjacent_bridge.atom})]

        projected = []
        for center, double in zip(centers, doubles, strict=True):
            key = simple_group_key(FunctionalFamily.CARBONYL, DerivativeKind.KETONE, double.element)
            if key is not None:
                carbonyl_descriptor = FunctionalGroupDescriptor(
                    family=FunctionalFamily.CARBONYL,
                    derivative=DerivativeKind.KETONE,
                    centers=(center,),
                    ligands=(double,),
                    attachment_atom=center,
                )
                projected.append(
                    PerceivedGroup(
                        key,
                        True,
                        center,
                        {center, double.atom},
                        descriptor=carbonyl_descriptor,
                    )
                )
        return projected
    return []


def _terminal_double_chalcogen(mol: Molecule, center: int, consumed: set[int]) -> ChalcogenLigand | None:
    ligands = [
        ligand
        for neighbor in mol.get_neighbors(center)
        if neighbor not in consumed
        and (ligand := classify_chalcogen_ligand(mol, center, neighbor)) is not None
        and ligand.role is ChalcogenLigandRole.DOUBLE_BONDED
    ]
    return ligands[0] if len(ligands) == 1 else None


def _connected_without_bridge(mol: Molecule, start: int, target: int, bridge_atoms: set[int]) -> bool:
    visited = set(bridge_atoms) | {start}
    queue = [start]
    while queue:
        current = queue.pop(0)
        for neighbor in mol.get_neighbors(current):
            if neighbor == target:
                return True
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return False


def _central_chalcogen_derivative(
    mol: Molecule,
    center: int,
    consumed: set[int],
    cyclic_atoms: set[int],
) -> PerceivedGroup | None:
    central_element = chalcogen_for_symbol(mol.atoms[center].symbol)
    if (
        central_element not in {Chalcogen.SULFUR, Chalcogen.SELENIUM, Chalcogen.TELLURIUM}
        or center in consumed
        or center in cyclic_atoms
        or mol.atoms[center].charge != 0
        or mol.atoms[center].stereo is not None
        or mol.atoms[center].raw_stereo is not None
    ):
        return None
    if any(
        chalcogen_for_symbol(mol.atoms[neighbor].symbol) is None and mol.get_bond(center, neighbor).order > 1
        for neighbor in mol.get_neighbors(center)
    ):
        return None
    carbon_anchors = [
        neighbor
        for neighbor in mol.get_neighbors(center)
        if mol.atoms[neighbor].is_carbon and mol.get_bond(center, neighbor).order == 1
    ]
    if len(carbon_anchors) != 1:
        return None
    anchor = carbon_anchors[0]
    ligands = [
        ligand
        for neighbor in mol.get_neighbors(center)
        if neighbor != anchor
        and neighbor not in consumed
        and (ligand := classify_chalcogen_ligand(mol, center, neighbor)) is not None
    ]
    double_ligands = [ligand for ligand in ligands if ligand.role is ChalcogenLigandRole.DOUBLE_BONDED]
    if len(double_ligands) not in {1, 2}:
        return None
    terminal_ligands = [
        ligand
        for ligand in ligands
        if ligand.role
        in {
            ChalcogenLigandRole.HYDROGEN_BEARING,
            ChalcogenLigandRole.ANIONIC,
            ChalcogenLigandRole.CARBON_LINK,
            ChalcogenLigandRole.CHALCOGEN_LINK,
        }
    ]
    nitrogens = [
        neighbor
        for neighbor in mol.get_neighbors(center)
        if neighbor not in consumed
        and mol.atoms[neighbor].symbol == "N"
        and mol.atoms[neighbor].charge == 0
        and mol.get_bond(center, neighbor).order == 1
    ]
    halogens = [
        neighbor
        for neighbor in mol.get_neighbors(center)
        if neighbor not in consumed
        and mol.atoms[neighbor].symbol in {"F", "Cl", "Br", "I"}
        and mol.get_bond(center, neighbor).order == 1
    ]
    derivative = None
    group_atoms = {center, *(ligand.atom for ligand in double_ligands)}
    leaving_group = None
    if len(terminal_ligands) == 1 and not nitrogens and not halogens:
        terminal = terminal_ligands[0]
        if terminal.role is ChalcogenLigandRole.CHALCOGEN_LINK:
            terminal_atom = terminal.attachment_atom
            if terminal_atom is None:
                return None
            terminal_end = classify_chalcogen_ligand(mol, terminal.atom, terminal_atom)
            if terminal_end is None:
                return None
            derivative = {
                ChalcogenLigandRole.HYDROGEN_BEARING: DerivativeKind.ACID,
                ChalcogenLigandRole.ANIONIC: DerivativeKind.ANION,
                ChalcogenLigandRole.CARBON_LINK: DerivativeKind.ESTER,
            }.get(terminal_end.role)
            if derivative is None:
                return None
            descriptor_ligands = (*double_ligands, terminal, terminal_end)
            group_atoms.update({terminal.atom, terminal_end.atom})
        else:
            derivative = {
                ChalcogenLigandRole.HYDROGEN_BEARING: DerivativeKind.ACID,
                ChalcogenLigandRole.ANIONIC: DerivativeKind.ANION,
                ChalcogenLigandRole.CARBON_LINK: DerivativeKind.ESTER,
            }[terminal.role]
            descriptor_ligands = (*double_ligands, terminal)
            group_atoms.add(terminal.atom)
    else:
        if len(nitrogens) == 1:
            if nitrogens[0] in cyclic_atoms:
                return None
            hydrazide_nitrogens = _hydrazide_nitrogens(mol, center, nitrogens[0], cyclic_atoms)
            if hydrazide_nitrogens is not None:
                terminal_nitrogen = hydrazide_nitrogens[1]
                if any(
                    neighbor != nitrogens[0]
                    and mol.atoms[neighbor].is_carbon
                    and any(
                        adjacent != terminal_nitrogen
                        and chalcogen_for_symbol(mol.atoms[adjacent].symbol) is not None
                        and mol.get_bond(neighbor, adjacent).order > 1
                        for adjacent in mol.get_neighbors(neighbor)
                    )
                    for neighbor in mol.get_neighbors(terminal_nitrogen)
                ):
                    return None
            if hydrazide_nitrogens is None and any(
                neighbor != center
                and (mol.atoms[neighbor].symbol not in {"C", "H"} or mol.get_bond(nitrogens[0], neighbor).order != 1)
                for neighbor in mol.get_neighbors(nitrogens[0])
            ):
                return None
            derivative = DerivativeKind.HYDRAZIDE if hydrazide_nitrogens is not None else DerivativeKind.AMIDE
            descriptor_ligands = tuple(double_ligands)
            group_atoms.update(hydrazide_nitrogens or (nitrogens[0],))
        elif len(halogens) == 1:
            derivative = DerivativeKind.ACID_HALIDE
            descriptor_ligands = tuple(double_ligands)
            leaving = halogens[0]
            leaving_group = _leaving_group_descriptor(
                mol,
                AcylLeavingGroup(mol.atoms[leaving].symbol),
                center,
                (leaving,),
            )
            group_atoms.add(leaving)
        else:
            return None
    owned_neighbors = {anchor} | (group_atoms & set(mol.get_neighbors(center)))
    if owned_neighbors != set(mol.get_neighbors(center)):
        return None
    descriptor = FunctionalGroupDescriptor(
        family=FunctionalFamily.CENTRAL_ACID,
        derivative=derivative,
        centers=(center,),
        ligands=tuple(descriptor_ligands),
        attachment_atom=anchor,
        central_element=central_element,
        leaving_group=leaving_group,
    )
    key, rule = resolve_central_acid_rule(descriptor)
    return PerceivedGroup(
        key,
        True,
        anchor,
        group_atoms,
        descriptor=descriptor,
        resolved_rule=rule,
    )


def _enrich_groups(mol: Molecule, groups: list[PerceivedGroup]) -> list[PerceivedGroup]:
    """Attach metadata and graph bindings to perceived groups."""

    for group in groups:
        resolved_rule = group.resolved_rule
        group.metadata = _metadata_for_group(group.key, resolved_rule)
        if resolved_rule is None:
            group.resolved_rule = RULES.functional_groups.by_key.get(group.key)
        group.atom_bindings = _atom_bindings_for_group(group)
        group.bond_bindings = _bond_bindings_for_group(mol, group)
        if not group.decision_reasons:
            group.decision_reasons = (
                f"Matched {group.key.replace('_', ' ')} pattern near atom {group.attachment_carbon}.",
            )
    return groups


def _metadata_for_group(key: str, resolved_rule: FunctionalGroupRule | None = None) -> FunctionalGroupMetadata:
    """Return naming metadata for a perceived group from rule tables."""

    if resolved_rule is None:
        return metadata_for_group(key)
    return FunctionalGroupMetadata(
        prefix=resolved_rule.prefix,
        suffix=resolved_rule.suffix,
        multi_suffix=resolved_rule.multi_suffix,
        suffix_multiplier_positions=resolved_rule.suffix_multiplier_positions,
        seniority=resolved_rule.seniority,
        suffix_with_locant=resolved_rule.suffix_with_locant,
        source="nomenclature.resolved_functional_group",
    )


def _atom_bindings_for_group(group: PerceivedGroup) -> tuple[AtomBinding, ...]:
    """Return role-labelled atom bindings for a group."""

    characteristic_atoms = tuple(sorted(group.atoms_involved))
    attachment_atoms = (group.attachment_carbon,)
    bindings = [
        AtomBinding("attachment", attachment_atoms),
        AtomBinding("characteristic_group", characteristic_atoms),
        AtomBinding("full_group", tuple(sorted(group.atom_ids))),
    ]
    if group.descriptor is not None:
        bindings.append(AtomBinding("characteristic_centers", group.descriptor.centers))
        for ligand in group.descriptor.ligands:
            bindings.append(AtomBinding(ligand.role.value, (ligand.atom,)))
        for path in group.descriptor.linker_paths:
            bindings.append(AtomBinding("ordered_linker_path", path))
    return tuple(bindings)


def _bond_bindings_for_group(mol: Molecule, group: PerceivedGroup) -> tuple[BondBinding, ...]:
    """Return role-labelled bond bindings for a group."""

    characteristic_atoms = set(group.atoms_involved)
    full_atoms = group.atom_ids
    characteristic_bonds = bond_ids_within(mol, characteristic_atoms)
    full_bonds = bond_ids_within(mol, full_atoms)
    attachment_bonds = full_bonds - characteristic_bonds
    return (
        BondBinding("characteristic_group", tuple(sorted(characteristic_bonds))),
        BondBinding("attachment", tuple(sorted(attachment_bonds))),
        BondBinding("full_group", tuple(sorted(full_bonds))),
    )


def _has_senior_nitrogen_ligand(mol: Molecule, nitrogen: int, center: int) -> bool:
    """
    Whether an imino nitrogen's substituent binds it into a senior group.
    """

    for ligand in mol.get_neighbors(nitrogen):
        if ligand == center or mol.atoms[ligand].symbol == "H":
            continue
        if mol.atoms[ligand].symbol == "N":
            return True
        if mol.atoms[ligand].is_carbon and any(
            mol.atoms[far].symbol in {"O", "S", "N"}
            and (bond := mol.get_bond(ligand, far)) is not None
            and bond.order == 2
            for far in mol.get_neighbors(ligand)
        ):
            return True
    return False


def _urea_nitrogens(
    mol: Molecule, carbon: int, single_n_candidates: list[int], cyclic_atoms: set[int]
) -> tuple[int, int] | None:
    """The two amine nitrogens of an acyclic urea/thiourea carbon, or ``None``.

    Hydrazine- and hydroxylamine-bearing nitrogens are left to their own parents
    (hydrazinecarboxamide, N-hydroxyurea), and ring nitrogens keep the ring as the
    parent (piperidine-1-carboxamide).
    """

    if carbon in cyclic_atoms or len(single_n_candidates) != 2 or mol.degree(carbon) != 3:
        return None
    nitrogens = tuple(sorted(single_n_candidates))
    for n in nitrogens:
        atom = mol.atoms[n]
        if n in cyclic_atoms or atom.charge:
            return None
        for neighbor in mol.get_neighbors(n):
            if neighbor == carbon:
                continue
            if mol.atoms[neighbor].symbol in {"N", "O", "S"} and not _is_nitroso_or_nitro_nitrogen(mol, neighbor, n):
                return None
            if neighbor != carbon and mol.get_bond(n, neighbor).order != 1:
                return None
    return nitrogens


def _is_nitroso_or_nitro_nitrogen(mol: Molecule, nitrogen: int, parent: int) -> bool:
    """An N-nitroso / N-nitro substituent nitrogen: only oxygens besides its parent (N-nitrosoureas)."""

    atom = mol.atoms[nitrogen]
    if atom.symbol != "N" or atom.total_h_count:
        return False
    others = [x for x in mol.get_neighbors(nitrogen) if x != parent]
    return bool(others) and all(mol.atoms[x].symbol == "O" and mol.degree(x) == 1 for x in others)


_SULFONYL_HALIDE_KEYS = {
    "F": "sulfonyl_fluoride",
    "Cl": "sulfonyl_chloride",
    "Br": "sulfonyl_bromide",
    "I": "sulfonyl_iodide",
}


def _sulfonyl_derivative(
    mol: Molecule, sulfur: int, oxygens: list[int], adj_atoms: list[int], cyclic_atoms: set[int]
) -> tuple[str, int, int] | None:
    """R-SO2-NR2 / R-SO2-X on a carbon: (group key, attachment carbon, the N or halogen)."""

    if sulfur in cyclic_atoms or len(oxygens) != 2 or len(adj_atoms) != 2:
        return None
    if any(mol.get_bond(sulfur, o).order != 2 or mol.degree(o) != 1 for o in oxygens):
        return None
    carbons = [a for a in adj_atoms if mol.atoms[a].is_carbon]
    others = [a for a in adj_atoms if not mol.atoms[a].is_carbon]
    if len(carbons) != 1 or len(others) != 1:
        return None
    hetero = others[0]
    symbol = mol.atoms[hetero].symbol
    if mol.atoms[sulfur].charge or mol.atoms[hetero].charge:
        return None
    if symbol == "N":
        if hetero in cyclic_atoms:
            return None
        # Sulfonyl hydrazides, hydroxamates and sulfonimides keep their own handling.
        for neighbor in mol.get_neighbors(hetero):
            if neighbor != sulfur and mol.atoms[neighbor].symbol in {"N", "O", "S"}:
                return None
            if neighbor != sulfur and mol.get_bond(hetero, neighbor).order != 1:
                return None
        return "sulfonamide", carbons[0], hetero
    if symbol in _SULFONYL_HALIDE_KEYS and mol.degree(hetero) == 1:
        return _SULFONYL_HALIDE_KEYS[symbol], carbons[0], hetero
    return None


def _plain_amine_nitrogen(mol: Molecule, nitrogen: int, carbon: int, cyclic_atoms: set[int]) -> bool:
    """An acyclic, uncharged nitrogen whose other bonds are single bonds to carbon or hydrogen."""

    atom = mol.atoms[nitrogen]
    if nitrogen in cyclic_atoms or atom.charge:
        return False
    for neighbor in mol.get_neighbors(nitrogen):
        if neighbor == carbon:
            continue
        if mol.get_bond(nitrogen, neighbor).order != 1 or not (
            mol.atoms[neighbor].is_carbon or mol.atoms[neighbor].symbol == "H"
        ):
            return False
        # An acyl or imidoyl group on the nitrogen belongs to a senior amide/amidine.
        if any(
            mol.get_bond(neighbor, x).order == 2 and mol.atoms[x].symbol in {"O", "S", "N"}
            for x in mol.get_neighbors(neighbor)
            if x != nitrogen
        ):
            return False
    return True


def _amidine_group(
    mol: Molecule, carbon: int, double_n: int, nitrogens: list[int], cyclic_atoms: set[int]
) -> tuple[str, int, set[int]] | None:
    """P-66.4.1: C(=NR)NR2 is an amidine; C(=NR)(NR2)NR2 the retained parent guanidine.

    Returns (group key, attachment atom, involved atoms).  The imino nitrogen may
    carry hydroxy (amidoximes) or amino; the amine nitrogens must be plain.
    """

    if mol.atoms[double_n].charge or double_n in cyclic_atoms or mol.degree(carbon) not in (2, 3):
        return None
    for neighbor in mol.get_neighbors(double_n):
        if neighbor != carbon and mol.get_bond(double_n, neighbor).order != 1:
            return None
        if neighbor != carbon and mol.atoms[neighbor].symbol == "N":
            return None  # amidrazones keep their own handling
    single_ns = [n for n in nitrogens if n != double_n and mol.get_bond(carbon, n).order == 1]
    if not single_ns or not all(_plain_amine_nitrogen(mol, n, carbon, cyclic_atoms) for n in single_ns):
        return None
    others = [x for x in mol.get_neighbors(carbon) if x != double_n and x not in single_ns]
    if len(single_ns) == 2 and not others:
        return "guanidine", carbon, {carbon, double_n, *single_ns}
    if len(single_ns) == 1 and not others and mol.degree(carbon) == 2:
        return "amidine", carbon, {carbon, double_n, single_ns[0]}
    if len(single_ns) != 1 or len(others) != 1:
        return None
    other = others[0]
    if not mol.atoms[other].is_carbon:
        return None
    involved = {carbon, double_n, single_ns[0]}
    if other in cyclic_atoms and mol.get_bond(carbon, other).order == 1:
        return "ring_amidine", other, involved
    return "amidine", carbon, involved


def _hydrazide_nitrogens(mol: Molecule, carbon: int, single_n: int, cyclic_atoms: set[int]) -> tuple[int, int] | None:
    """P-66.3.1: ``-C(=O)-N-N`` is a hydrazide; (acyl N, terminal N) or ``None``."""

    if single_n in cyclic_atoms or mol.atoms[single_n].charge:
        return None
    carbon_neighbors = [x for x in mol.get_neighbors(carbon) if x != single_n and mol.atoms[x].symbol not in {"O", "H"}]
    if any(mol.atoms[x].symbol == "N" for x in carbon_neighbors):
        return None  # semicarbazides are ureas/hydrazinecarboxamides, not hydrazides
    terminal = [
        x
        for x in mol.get_neighbors(single_n)
        if x != carbon and mol.atoms[x].symbol == "N" and mol.get_bond(single_n, x).order == 1
    ]
    if len(terminal) != 1:
        return None
    terminal_n = terminal[0]
    if terminal_n in cyclic_atoms or mol.atoms[terminal_n].charge:
        return None
    for x in mol.get_neighbors(terminal_n):
        if x != single_n:
            if mol.atoms[x].symbol not in {"C", "H"} or mol.get_bond(terminal_n, x).order != 1:
                return None
    return single_n, terminal_n
