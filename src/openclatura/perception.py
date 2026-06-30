# openclatura/perception.py

from dataclasses import dataclass, field

from .chains import get_cyclic_atoms
from .functional_groups import PERCEPTION_DETECTORS, PERCEPTION_SPECS, PerceptionDetectorSpec, metadata_for_group
from .molecule import AtomBinding, BondBinding, FunctionalGroupMetadata, Molecule
from .nitrogen_roles import nitrogen_chain_roles


@dataclass
class PerceivedGroup:
    """Functional-group perception result bound to graph atoms and metadata.

    The first four fields are the legacy contract used throughout namer.py.
    The remaining fields make the object self-describing: rule metadata,
    atom-role bindings, bond-role bindings, and short reasons explaining why
    the detector emitted the group.
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


def perceive_groups(mol: Molecule) -> list[PerceivedGroup]:
    groups = []
    for detector in PERCEPTION_DETECTORS:
        groups.extend(detector(mol))
    if PERCEPTION_SPECS:
        specs = tuple(sorted(BUILTIN_PERCEPTION_SPECS + tuple(PERCEPTION_SPECS), key=lambda item: item.priority))
    else:
        specs = BUILTIN_PERCEPTION_SPECS
    for spec in specs:
        groups.extend(spec.detector(mol))
    return _enrich_groups(mol, groups)


BUILTIN_PERCEPTION_SPECS = (
    PerceptionDetectorSpec(
        key="builtin.functional_groups",
        detector=lambda mol: _builtin_perceive_groups(mol),
        priority=100,
        families=("functional_group",),
        description="Built-in structural functional-group detectors.",
    ),
)


def _builtin_perceive_groups(mol: Molecule) -> list[PerceivedGroup]:
    groups = []
    consumed = set()
    cyclic_atoms = get_cyclic_atoms(mol)

    for atom in mol:
        if atom.symbol == "N" and atom.idx not in consumed and atom.idx not in cyclic_atoms:
            oxygens = [n for n in mol.get_neighbors(atom.idx) if mol.atoms[n].symbol == "O"]
            adj_atoms = [n for n in mol.get_neighbors(atom.idx) if n not in oxygens]

            if len(oxygens) == 2 and len(adj_atoms) == 1:
                has_double_o = any(mol.get_bond(atom.idx, o).order == 2 for o in oxygens)
                if atom.charge == 1 or has_double_o:
                    groups.append(PerceivedGroup("nitro", False, adj_atoms[0], {atom.idx} | set(oxygens)))
                    consumed.update([atom.idx] + oxygens)
            elif len(oxygens) == 1 and len(adj_atoms) == 1:
                if mol.get_bond(atom.idx, oxygens[0]).order == 2:
                    groups.append(PerceivedGroup("nitroso", False, adj_atoms[0], {atom.idx, oxygens[0]}))
                    consumed.update([atom.idx, oxygens[0]])

    for role in nitrogen_chain_roles(mol, cyclic_atoms, consumed):
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
        if len(neighbors) != 1:
            continue
        center = neighbors[0]
        bond = mol.get_bond(atom.idx, center)
        if bond is None or bond.order != 2 or mol.atoms[center].is_carbon:
            continue
        groups.append(
            PerceivedGroup(
                "imino_prefix",
                False,
                center,
                {atom.idx},
                variant="terminal_heteroatom_imino",
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

    for atom in mol:
        if atom.symbol == "O" and mol.degree(atom.idx) == 2 and atom.idx not in consumed:
            adj_atoms = mol.get_neighbors(atom.idx)
            if len(adj_atoms) == 2:
                c1, c2 = adj_atoms
                if mol.atoms[c1].is_carbon and mol.atoms[c2].is_carbon:
                    o1 = next(
                        (
                            o
                            for o in mol.get_neighbors(c1)
                            if mol.atoms[o].symbol == "O" and mol.get_bond(c1, o).order == 2
                        ),
                        None,
                    )
                    o2 = next(
                        (
                            o
                            for o in mol.get_neighbors(c2)
                            if mol.atoms[o].symbol == "O" and mol.get_bond(c2, o).order == 2
                        ),
                        None,
                    )
                    if o1 and o2:
                        visited = {atom.idx, c1}
                        q = [c1]
                        is_cyclic = False
                        while q:
                            curr = q.pop(0)
                            for nxt in mol.get_neighbors(curr):
                                if nxt == c2:
                                    is_cyclic = True
                                    break
                                if nxt not in visited:
                                    visited.add(nxt)
                                    q.append(nxt)
                            if is_cyclic:
                                break

                        if not is_cyclic:
                            groups.append(PerceivedGroup("anhydride", True, c1, {atom.idx, o1, o2}))
                            consumed.update([atom.idx, o1, o2])

    for atom in mol:
        if atom.symbol == "S" and atom.idx not in consumed:
            oxygens = [n for n in mol.get_neighbors(atom.idx) if mol.atoms[n].symbol == "O"]
            adj_atoms = [n for n in mol.get_neighbors(atom.idx) if n not in oxygens]
            if len(oxygens) >= 3 and len(adj_atoms) == 1:
                double_o_list = [o for o in oxygens if mol.get_bond(atom.idx, o).order == 2]
                if len(double_o_list) >= 2:
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
                n_oxide = next(
                    (
                        x
                        for x in mol.get_neighbors(triple_n)
                        if x != atom.idx and mol.atoms[x].symbol == "O" and mol.get_bond(triple_n, x).order == 1
                    ),
                    None,
                )
                if n_oxide is not None:
                    ring_neighbors = [n for n in mol.get_neighbors(atom.idx) if n in cyclic_atoms]
                    is_exocyclic = False
                    attached_ring_atom = None
                    if atom.idx not in cyclic_atoms and len(ring_neighbors) == 1:
                        attached_ring_atom = ring_neighbors[0]
                        if mol.get_bond(atom.idx, attached_ring_atom).order == 1:
                            is_exocyclic = True
                    target_carbon = attached_ring_atom if is_exocyclic else atom.idx
                    key = "ring_nitrile_oxide" if is_exocyclic else "nitrile_oxide"
                    groups.append(PerceivedGroup(key, True, target_carbon, {atom.idx, triple_n, n_oxide}))
                    consumed.update([triple_n, n_oxide])
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
            oxygens = [n for n in mol.get_neighbors(atom.idx) if mol.atoms[n].symbol == "O" and n not in consumed]
            nitrogens = [n for n in mol.get_neighbors(atom.idx) if mol.atoms[n].symbol == "N" and n not in consumed]
            sulfurs = [n for n in mol.get_neighbors(atom.idx) if mol.atoms[n].symbol == "S" and n not in consumed]
            halogens = [
                n
                for n in mol.get_neighbors(atom.idx)
                if mol.atoms[n].symbol in ["F", "Cl", "Br", "I"] and n not in consumed
            ]

            double_o = next((o for o in oxygens if mol.get_bond(atom.idx, o).order == 2 and mol.degree(o) == 1), None)
            double_s = next((s for s in sulfurs if mol.get_bond(atom.idx, s).order == 2 and mol.degree(s) == 1), None)
            single_o = next((o for o in oxygens if mol.get_bond(atom.idx, o).order == 1), None)
            single_n_candidates = [n for n in nitrogens if mol.get_bond(atom.idx, n).order == 1]
            single_n_candidates.sort(
                key=lambda n: any(
                    neighbor != atom.idx and mol.atoms[neighbor].symbol != "H" for neighbor in mol.get_neighbors(n)
                ),
                reverse=True,
            )
            single_n = single_n_candidates[0] if single_n_candidates else None
            single_x = next((x for x in halogens if mol.get_bond(atom.idx, x).order == 1), None)

            if double_o is not None:
                ring_neighbors = [n for n in mol.get_neighbors(atom.idx) if n in cyclic_atoms]
                is_exocyclic = False
                attached_ring_atom = None

                if atom.idx not in cyclic_atoms and len(ring_neighbors) == 1:
                    attached_ring_atom = ring_neighbors[0]
                    if mol.get_bond(atom.idx, attached_ring_atom).order == 1:
                        is_exocyclic = True

                target_carbon = attached_ring_atom if is_exocyclic else atom.idx

                if single_o is not None:
                    o_neighbors = [x for x in mol.get_neighbors(single_o) if x != atom.idx]
                    is_peroxy = False
                    peroxy_o = None
                    if len(o_neighbors) == 1 and mol.atoms[o_neighbors[0]].symbol == "O":
                        is_peroxy = True
                        peroxy_o = o_neighbors[0]
                    elif len(o_neighbors) > 1 and any(mol.atoms[x].symbol == "O" for x in o_neighbors):
                        is_peroxy = True
                        peroxy_o = next(x for x in o_neighbors if mol.atoms[x].symbol == "O")

                    if is_peroxy:
                        if target_carbon == atom.idx and single_o in cyclic_atoms and peroxy_o in cyclic_atoms:
                            groups.append(PerceivedGroup("ketone", True, target_carbon, {atom.idx, double_o}))
                            consumed.update([double_o])
                        elif mol.degree(peroxy_o) == 1 or mol.atoms[peroxy_o].charge == -1:
                            if mol.atoms[peroxy_o].charge == -1:
                                key = "ring_peroxy_ester" if is_exocyclic else "peroxy_ester"
                            else:
                                key = "ring_peroxy_acid" if is_exocyclic else "peroxy_acid"
                            groups.append(
                                PerceivedGroup(key, True, target_carbon, {atom.idx, double_o, single_o, peroxy_o})
                            )
                            consumed.update([double_o, single_o, peroxy_o])
                        else:
                            key = "ring_peroxy_ester" if is_exocyclic else "peroxy_ester"
                            groups.append(
                                PerceivedGroup(key, True, target_carbon, {atom.idx, double_o, single_o, peroxy_o})
                            )
                            consumed.update([double_o, single_o, peroxy_o])
                    else:
                        is_lactone = False
                        if target_carbon == atom.idx and single_o in cyclic_atoms:
                            visited = {atom.idx, double_o}
                            q = [single_o]
                            while q:
                                curr = q.pop(0)
                                for nxt in mol.get_neighbors(curr):
                                    if nxt == atom.idx and curr != single_o:
                                        is_lactone = True
                                        break
                                    if nxt not in visited and nxt in cyclic_atoms:
                                        visited.add(nxt)
                                        q.append(nxt)
                                if is_lactone:
                                    break

                        if is_lactone:
                            groups.append(PerceivedGroup("ketone", True, target_carbon, {atom.idx, double_o}))
                            consumed.update([double_o])
                        else:
                            if mol.atoms[single_o].charge == -1:
                                key = "ring_carboxylate" if is_exocyclic else "carboxylate"
                            elif mol.degree(single_o) == 1:
                                key = "ring_carboxylic_acid" if is_exocyclic else "carboxylic_acid"
                            else:
                                key = "ring_carboxylate" if is_exocyclic else "ester"
                            groups.append(PerceivedGroup(key, True, target_carbon, {atom.idx, double_o, single_o}))
                            consumed.update([double_o, single_o])
                elif single_n is not None:
                    is_lactam = False
                    if target_carbon == atom.idx and single_n in cyclic_atoms:
                        visited = {atom.idx, double_o}
                        q = [single_n]
                        while q:
                            curr = q.pop(0)
                            for nxt in mol.get_neighbors(curr):
                                if nxt == atom.idx and curr != single_n:
                                    is_lactam = True
                                    break
                                if nxt not in visited and nxt in cyclic_atoms:
                                    visited.add(nxt)
                                    q.append(nxt)
                            if is_lactam:
                                break

                    if is_lactam:
                        groups.append(PerceivedGroup("ketone", True, target_carbon, {atom.idx, double_o}))
                        consumed.update([double_o])
                    else:
                        if single_n in cyclic_atoms:
                            pass
                        else:
                            key = "ring_amide" if is_exocyclic else "amide"
                            groups.append(PerceivedGroup(key, True, target_carbon, {atom.idx, double_o, single_n}))
                            consumed.update([double_o, single_n])
                elif single_x is not None:
                    sym = mol.atoms[single_x].symbol
                    if is_exocyclic:
                        x_map = {
                            "F": "ring_acid_fluoride",
                            "Cl": "ring_acid_chloride",
                            "Br": "ring_acid_bromide",
                            "I": "ring_acid_iodide",
                        }
                    else:
                        x_map = {"F": "acid_fluoride", "Cl": "acid_chloride", "Br": "acid_bromide", "I": "acid_iodide"}
                    groups.append(PerceivedGroup(x_map[sym], True, target_carbon, {atom.idx, double_o, single_x}))
                    consumed.update([double_o, single_x])
            elif double_s is not None:
                ring_neighbors = [n for n in mol.get_neighbors(atom.idx) if n in cyclic_atoms]
                carbon_neighbors = [n for n in mol.get_neighbors(atom.idx) if mol.atoms[n].is_carbon]
                non_chalcogen_neighbors = [
                    n for n in mol.get_neighbors(atom.idx) if n != double_s and mol.atoms[n].symbol != "H"
                ]
                if single_n is not None and single_n not in cyclic_atoms:
                    target_carbon = atom.idx
                    is_exocyclic = False
                    if atom.idx not in cyclic_atoms and len(ring_neighbors) == 1 and len(carbon_neighbors) == 1:
                        ring_neighbor = ring_neighbors[0]
                        if mol.get_bond(atom.idx, ring_neighbor).order == 1:
                            target_carbon = ring_neighbor
                            is_exocyclic = True
                    key = "ring_thioamide" if is_exocyclic else "thioamide"
                    groups.append(PerceivedGroup(key, True, target_carbon, {atom.idx, double_s, single_n}))
                    consumed.update([double_s, single_n])
                elif (
                    atom.idx not in cyclic_atoms
                    and len(ring_neighbors) == 1
                    and non_chalcogen_neighbors == ring_neighbors
                ):
                    ring_neighbor = ring_neighbors[0]
                    if mol.get_bond(atom.idx, ring_neighbor).order == 1:
                        groups.append(PerceivedGroup("ring_thioaldehyde", True, ring_neighbor, {atom.idx, double_s}))
                        consumed.update([double_s])
                elif (
                    atom.idx not in cyclic_atoms
                    and len(non_chalcogen_neighbors) <= 1
                    and all(mol.atoms[n].is_carbon for n in non_chalcogen_neighbors)
                ):
                    groups.append(PerceivedGroup("thioaldehyde", True, atom.idx, {atom.idx, double_s}))
                    consumed.update([double_s])

    for atom in mol:
        if atom.symbol == "N" and atom.idx not in consumed and atom.idx not in cyclic_atoms:
            c_neighbors = [n for n in mol.get_neighbors(atom.idx) if mol.atoms[n].is_carbon]
            n_neighbors = [n for n in mol.get_neighbors(atom.idx) if mol.atoms[n].symbol == "N"]

            double_c = next((c for c in c_neighbors if mol.get_bond(atom.idx, c).order == 2), None)

            if double_c is not None:
                if len(n_neighbors) > 0:
                    n2 = n_neighbors[0]
                    if mol.get_bond(atom.idx, n2).order == 1:
                        amidino_tail_atoms = _amidinohydrazone_tail_atoms(mol, n2, {atom.idx})
                        if _has_non_h_multiple_bond_neighbor(mol, n2, {atom.idx}) and not amidino_tail_atoms:
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
        if atom.symbol == "N" and atom.idx not in consumed and atom.idx not in cyclic_atoms:
            c_neighbors = [n for n in mol.get_neighbors(atom.idx) if mol.atoms[n].is_carbon]
            n_neighbors = [n for n in mol.get_neighbors(atom.idx) if mol.atoms[n].symbol == "N"]

            if len(n_neighbors) > 0:
                n2 = n_neighbors[0]
                c_att = c_neighbors[0] if c_neighbors else None
                if c_att:
                    c_att_bond = mol.get_bond(atom.idx, c_att)
                    if (
                        c_att_bond is not None
                        and c_att_bond.order == 1
                        and mol.get_bond(atom.idx, n2).order == 1
                        and not _has_non_h_multiple_bond_neighbor(mol, n2, {atom.idx})
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

    for atom in mol:
        if atom.idx not in consumed:
            if atom.symbol == "O" and mol.degree(atom.idx) == 1:
                adj_atoms = mol.get_neighbors(atom.idx)
                if len(adj_atoms) == 1:
                    c_idx = adj_atoms[0]
                    bond = mol.get_bond(atom.idx, c_idx)
                    if bond.order == 2:
                        if mol.atoms[c_idx].is_carbon:
                            if c_idx in cyclic_atoms:
                                groups.append(PerceivedGroup("ketone", True, c_idx, {atom.idx}))
                            elif len(mol.get_neighbors(c_idx)) >= 3:
                                groups.append(PerceivedGroup("ketone", True, c_idx, {atom.idx}))
                            else:
                                ring_neighbors = [n for n in mol.get_neighbors(c_idx) if n in cyclic_atoms]
                                if len(ring_neighbors) == 1 and mol.get_bond(c_idx, ring_neighbors[0]).order == 1:
                                    groups.append(
                                        PerceivedGroup("ring_aldehyde", True, ring_neighbors[0], {c_idx, atom.idx})
                                    )
                                else:
                                    groups.append(PerceivedGroup("aldehyde", True, c_idx, {c_idx, atom.idx}))
                    elif bond.order == 1:
                        if mol.atoms[c_idx].is_carbon:
                            key = "olate" if atom.charge < 0 else "alcohol"
                            groups.append(PerceivedGroup(key, True, c_idx, {atom.idx}))
                    consumed.add(atom.idx)
            elif atom.symbol == "S" and mol.degree(atom.idx) == 1 and atom.idx not in cyclic_atoms:
                adj_atoms = mol.get_neighbors(atom.idx)
                if len(adj_atoms) == 1:
                    c_idx = adj_atoms[0]
                    bond = mol.get_bond(atom.idx, c_idx)
                    if bond.order == 1:
                        if mol.atoms[c_idx].symbol != "H":
                            key = "thiolate" if atom.charge < 0 else "thiol"
                            groups.append(PerceivedGroup(key, True, c_idx, {atom.idx}))
                            consumed.add(atom.idx)

    for atom in mol:
        if atom.symbol == "N" and atom.idx not in consumed and atom.idx not in cyclic_atoms:
            adj_atoms = mol.get_neighbors(atom.idx)
            if len(adj_atoms) > 0:
                key = "aminium" if atom.charge > 0 else "amine"
                for c in adj_atoms:
                    groups.append(PerceivedGroup(key, True, c, {atom.idx}))
                consumed.add(atom.idx)

    for atom in mol:
        if atom.symbol == "O" and atom.idx not in consumed and atom.idx not in cyclic_atoms:
            adj_atoms = mol.get_neighbors(atom.idx)
            if len(adj_atoms) == 2:
                for c in adj_atoms:
                    groups.append(PerceivedGroup("ether", False, c, {atom.idx}))
                consumed.add(atom.idx)

    for atom in mol:
        if atom.symbol == "S" and atom.idx not in consumed and atom.idx not in cyclic_atoms:
            adj_atoms = mol.get_neighbors(atom.idx)
            if len(adj_atoms) == 2:
                for c in adj_atoms:
                    groups.append(PerceivedGroup("thioether", False, c, {atom.idx}))
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


def _enrich_groups(mol: Molecule, groups: list[PerceivedGroup]) -> list[PerceivedGroup]:
    """Attach metadata and graph bindings to perceived groups."""

    for group in groups:
        group.metadata = _metadata_for_group(group.key)
        group.atom_bindings = _atom_bindings_for_group(group)
        group.bond_bindings = _bond_bindings_for_group(mol, group)
        if not group.decision_reasons:
            group.decision_reasons = (
                f"Matched {group.key.replace('_', ' ')} pattern near atom {group.attachment_carbon}.",
            )
    return groups


def _metadata_for_group(key: str) -> FunctionalGroupMetadata:
    """Return naming metadata for a perceived group from rule tables."""

    return metadata_for_group(key)


def _atom_bindings_for_group(group: PerceivedGroup) -> tuple[AtomBinding, ...]:
    """Return role-labelled atom bindings for a group."""

    characteristic_atoms = tuple(sorted(group.atoms_involved))
    attachment_atoms = (group.attachment_carbon,)
    return (
        AtomBinding("attachment", attachment_atoms),
        AtomBinding("characteristic_group", characteristic_atoms),
        AtomBinding("full_group", tuple(sorted(group.atom_ids))),
    )


def _bond_bindings_for_group(mol: Molecule, group: PerceivedGroup) -> tuple[BondBinding, ...]:
    """Return role-labelled bond bindings for a group."""

    characteristic_atoms = set(group.atoms_involved)
    full_atoms = group.atom_ids
    characteristic_bonds = _bond_ids_within(mol, characteristic_atoms)
    full_bonds = _bond_ids_within(mol, full_atoms)
    attachment_bonds = full_bonds - characteristic_bonds
    return (
        BondBinding("characteristic_group", tuple(sorted(characteristic_bonds))),
        BondBinding("attachment", tuple(sorted(attachment_bonds))),
        BondBinding("full_group", tuple(sorted(full_bonds))),
    )


def _bond_ids_within(mol: Molecule, atom_ids: set[int]) -> set[int]:
    """Return IDs for bonds whose endpoints are both in atom_ids."""

    bond_ids = set()
    for atom_idx in atom_ids:
        for neighbor_idx in mol.get_neighbors(atom_idx):
            if neighbor_idx in atom_ids and atom_idx < neighbor_idx:
                bond = mol.get_bond(atom_idx, neighbor_idx)
                if bond is not None:
                    bond_ids.add(bond.idx)
    return bond_ids


def _has_non_h_multiple_bond_neighbor(mol: Molecule, atom_idx: int, allowed: set[int]) -> bool:
    for neighbor in mol.get_neighbors(atom_idx):
        if neighbor in allowed or mol.atoms[neighbor].symbol == "H":
            continue
        bond = mol.get_bond(atom_idx, neighbor)
        if bond is not None and bond.order != 1:
            return True
    return False


def _amidinohydrazone_tail_atoms(mol: Molecule, hydrazone_terminal_n: int, allowed: set[int]) -> set[int]:
    """Return atoms for an N-C(=N)-N amidino tail attached to a hydrazone N."""

    for carbon in mol.get_neighbors(hydrazone_terminal_n):
        if carbon in allowed or not mol.atoms[carbon].is_carbon:
            continue
        n_c_bond = mol.get_bond(hydrazone_terminal_n, carbon)
        if n_c_bond is None or n_c_bond.order != 2:
            continue
        terminal_ns = [
            neighbor
            for neighbor in mol.get_neighbors(carbon)
            if neighbor != hydrazone_terminal_n
            and mol.atoms[neighbor].symbol == "N"
            and (bond := mol.get_bond(carbon, neighbor)) is not None
            and bond.order == 1
        ]
        if len(terminal_ns) == 2:
            return {carbon, *terminal_ns}
    return set()
