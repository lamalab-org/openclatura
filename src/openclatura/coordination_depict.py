"""Textbook-style pictures of coordination compounds and metallocenes.

However the input was drawn -- disconnected components, dative bonds, a haptic
molfile -- the perceived complex is laid out afresh: the metal in the middle,
each ligand on its own site, cyclopentadienyl rings drawn in perspective on the
metal's axis, phenyl groups and carbonyls abbreviated to ``Ph`` and ``CO``.

The picture is an illustration of what was perceived, not a statement of
geometry: sites are spread evenly, and a ligand whose donor atom the input did
not say is attached through the most likely one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .coordination import _perceive, contains_transition_metal, read_molblock

_BOND = 1.5  # ligand bond length, as RDKit lays molecules out
_METAL_BOND = 1.7
_RING_RADIUS = 1.0
_RING_SQUASH = 0.34
_RING_DISTANCE = 1.45
_SCALE = 30.0  # pixels per layout unit
_FONT = 17.0
_DONOR_PREFERENCE = ("P", "N", "S", "O", "Cl", "Br", "I", "F", "C")

Point = tuple[float, float]


class _CannotDraw(Exception):
    """The complex has a shape this layout does not handle."""


@dataclass
class _Atom:
    symbol: str
    hydrogens: int = 0
    charge: int = 0
    tail: str = ""  # rest of an abbreviation, written after the symbol ("O" of CO)
    always_labelled: bool = False
    count: int = 1  # identical counter-ions written once: (Cl-)3


@dataclass
class _Ring:
    atoms: tuple[int, ...]  # in drawing order around the ring
    metal: int
    depth: dict[int, float] = field(default_factory=dict)  # > 0 behind the ring plane's centre


@dataclass
class _Scene:
    atoms: dict[int, _Atom] = field(default_factory=dict)
    bonds: dict[frozenset[int], int] = field(default_factory=dict)  # ligand bonds -> order
    links: list[tuple[int, int]] = field(default_factory=list)  # metal, donor
    rings: list[_Ring] = field(default_factory=list)
    pos: dict[int, Point] = field(default_factory=dict)
    link_style: dict[tuple[int, int], str] = field(default_factory=dict)  # "wedge" or "hash"
    link_order: dict[tuple[int, int], int] = field(default_factory=dict)  # 2 for M=O
    entity: set[int] = field(default_factory=set)  # the centre and its ligands
    entity_charge: int = 0

    def neighbours(self, idx: int) -> list[int]:
        return [next(iter(pair - {idx})) for pair in self.bonds if idx in pair]

    def component(self, start: int, blocked: set[int]) -> set[int]:
        seen = {start}
        stack = [start]
        while stack:
            for neighbour in self.neighbours(stack.pop()):
                if neighbour not in seen and neighbour not in blocked:
                    seen.add(neighbour)
                    stack.append(neighbour)
        return seen


def _unit(angle_degrees: float) -> Point:
    radians = math.radians(angle_degrees)
    return (math.cos(radians), math.sin(radians))


def _normalise(vector: Point) -> Point | None:
    length = math.hypot(*vector)
    return None if length < 1e-6 else (vector[0] / length, vector[1] / length)


def _centroid(points: list[Point]) -> Point:
    return (sum(p[0] for p in points) / len(points), sum(p[1] for p in points) / len(points))


def _rigid(points: dict[int, Point], pivot: Point, source: Point | None, target: Point, to: Point) -> dict[int, Point]:
    """Rotate ``points`` about ``pivot`` so direction ``source`` becomes ``target``, then move the pivot to ``to``."""

    if source is None:
        cos, sin = 1.0, 0.0
    else:
        cos = source[0] * target[0] + source[1] * target[1]
        sin = source[0] * target[1] - source[1] * target[0]
    moved = {}
    for idx, (x, y) in points.items():
        dx, dy = x - pivot[0], y - pivot[1]
        moved[idx] = (to[0] + cos * dx - sin * dy, to[1] + sin * dx + cos * dy)
    return moved


def _build_scene(state) -> _Scene:
    """Copy the detached molecule into a scene, abbreviating phenyl groups and carbonyls."""

    from rdkit import Chem

    rw = state.mol
    kekulised = Chem.Mol(rw)
    Chem.Kekulize(kekulised, clearAromaticFlags=True)
    scene = _Scene()
    metals = set(state.metals)
    donors = set().union(*state.donors_of.values()) if state.donors_of else set()
    for atom in rw.GetAtoms():
        scene.atoms[atom.GetIdx()] = _Atom(
            symbol=atom.GetSymbol(),
            hydrogens=0 if atom.GetIdx() in metals else atom.GetTotalNumHs(),
            charge=atom.GetFormalCharge(),
        )
    for bond in kekulised.GetBonds():
        scene.bonds[frozenset((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))] = int(bond.GetBondTypeAsDouble())

    ring_info = rw.GetRingInfo()
    for ring in ring_info.AtomRings():
        atoms = [rw.GetAtomWithIdx(idx) for idx in ring]
        if len(ring) != 6 or not all(a.GetAtomicNum() == 6 for a in atoms):
            continue
        if all(a.GetIsAromatic() for a in atoms):
            label = "Ph"
        elif all(a.GetHybridization() == Chem.HybridizationType.SP3 for a in atoms):
            label = "Cy"
        else:
            continue
        if any(ring_info.NumAtomRings(idx) != 1 for idx in ring) or donors & set(ring):
            continue
        substituted = [a for a in atoms if a.GetDegree() == 3]
        if len(substituted) != 1 or any(a.GetDegree() not in (2, 3) or a.GetFormalCharge() for a in atoms):
            continue
        ipso = substituted[0].GetIdx()
        for idx in ring:
            if idx != ipso:
                _remove_atom(scene, idx)
        scene.atoms[ipso] = _Atom(symbol=label, always_labelled=True)

    for fragment in Chem.GetMolFrags(rw):
        if len(fragment) != 2:
            continue
        symbols = sorted(rw.GetAtomWithIdx(idx).GetSymbol() for idx in fragment)
        if symbols != ["C", "O"] or scene.bonds.get(frozenset(fragment)) != 3:
            continue
        carbon, oxygen = sorted(fragment, key=lambda idx: rw.GetAtomWithIdx(idx).GetAtomicNum())
        _remove_atom(scene, oxygen)
        scene.atoms[carbon] = _Atom(symbol="C", tail="O", always_labelled=True)
    return scene


def _remove_atom(scene: _Scene, idx: int) -> None:
    del scene.atoms[idx]
    for pair in [pair for pair in scene.bonds if idx in pair]:
        del scene.bonds[pair]


_METAL_KEY = -1  # where the metal sits in a ligand laid out together with its bonds to it


def _free_coordinates(scene: _Scene, atoms: set[int], donors: tuple[int, ...] = ()) -> dict[int, Point]:
    """Lay a connected set of scene atoms out as an ordinary skeletal drawing.

    With ``donors``, a stand-in for the metal is bonded to them first, so a
    chelate comes out as the ring it closes; its place is returned under
    ``_METAL_KEY``.
    """

    from rdkit import Chem
    from rdkit.Chem import AllChem

    order = sorted(atoms)
    if len(order) == 1 and not donors:
        return {order[0]: (0.0, 0.0)}
    index = {idx: position for position, idx in enumerate(order)}
    mol = Chem.RWMol()
    for _ in order:
        mol.AddAtom(Chem.Atom(6))
    bond_types = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE}
    for pair, bond_order in scene.bonds.items():
        if pair <= atoms:
            a, b = tuple(pair)
            mol.AddBond(index[a], index[b], bond_types.get(bond_order, Chem.BondType.SINGLE))
    if donors:
        stand_in = mol.AddAtom(Chem.Atom(6))
        for donor in donors:
            mol.AddBond(stand_in, index[donor], Chem.BondType.SINGLE)
        index = {**index, _METAL_KEY: stand_in}
    mol.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(mol)
    AllChem.Compute2DCoords(mol)
    conformer = mol.GetConformer()
    return {idx: (conformer.GetAtomPosition(at).x, conformer.GetAtomPosition(at).y) for idx, at in index.items()}


def _place_ring(scene: _Scene, ring: tuple[int, ...], metal: int, centre: Point, axis: Point, placed: set[int]) -> None:
    """Draw a face-bound C5 ring in perspective at ``centre`` and hang its substituents off it."""

    ring_set = set(ring)
    view = axis if axis[1] > 1e-6 or (abs(axis[1]) <= 1e-6 and axis[0] > 0) else (-axis[0], -axis[1])
    across = (view[1], -view[0])
    substituted = [idx for idx in ring if any(n not in ring_set for n in scene.neighbours(idx))]
    start = ring.index(substituted[0]) if substituted else 0
    ordered = tuple(ring[(start + offset) % len(ring)] for offset in range(len(ring)))
    mark = _Ring(atoms=ordered, metal=metal)
    angles = {}
    for position, idx in enumerate(ordered):
        angle = 2 * math.pi * position / len(ordered)
        angles[idx] = angle
        flat, depth = _RING_RADIUS * math.cos(angle), _RING_RADIUS * math.sin(angle)
        scene.pos[idx] = (
            centre[0] + flat * across[0] + _RING_SQUASH * depth * view[0],
            centre[1] + flat * across[1] + _RING_SQUASH * depth * view[1],
        )
        mark.depth[idx] = depth
    scene.rings.append(mark)
    placed |= ring_set

    for idx in substituted:
        # Radially out of the ring, leaning away from the metal so that the
        # substituents of two stacked rings do not meet between them.
        outward = _normalise(
            (
                math.cos(angles[idx]) * across[0] + 0.45 * math.sin(angles[idx]) * view[0] + 0.4 * axis[0],
                math.cos(angles[idx]) * across[1] + 0.45 * math.sin(angles[idx]) * view[1] + 0.4 * axis[1],
            )
        )
        for neighbour in scene.neighbours(idx):
            if neighbour in ring_set:
                continue
            branch = scene.component(neighbour, ring_set)
            if branch & placed:
                raise _CannotDraw("a ring substituent joins two rings")
            free = _free_coordinates(scene, branch | {idx})
            direction = _normalise((free[neighbour][0] - free[idx][0], free[neighbour][1] - free[idx][1]))
            moved = _rigid(free, free[idx], direction, outward, scene.pos[idx])
            for atom in branch:
                scene.pos[atom] = moved[atom]
            placed |= branch


def _layout_sandwich(scene: _Scene, metal: int, rings: list[tuple[int, ...]], placed: set[int]) -> set[int]:
    """Lay a metallocene unit out around its metal at the origin; return the atoms placed."""

    before = set(placed)
    scene.pos[metal] = (0.0, 0.0)
    placed.add(metal)
    for ring, direction in zip(rings, (90.0, 270.0), strict=False):
        axis = _unit(direction)
        centre = (axis[0] * _RING_DISTANCE, axis[1] * _RING_DISTANCE)
        _place_ring(scene, ring, metal, centre, axis, placed)
    return placed - before


def _bonds_apart(scene: _Scene, first: int, second: int, limit: int = 4) -> int | None:
    """Number of bonds between two atoms of a ligand, up to ``limit``."""

    frontier, seen = {first}, {first}
    for distance in range(1, limit + 1):
        frontier = {n for idx in frontier for n in scene.neighbours(idx)} - seen
        if second in frontier:
            return distance
        seen |= frontier
    return None


def _chelating_pair(scene: _Scene, atoms: list[int]) -> bool:
    """Two nitrogens three bonds apart (ethane-1,2-diamine, 2,2'-bipyridine) close a five-membered chelate ring."""

    return len(atoms) == 2 and _bonds_apart(scene, atoms[0], atoms[1]) == 3


def _guess_donors(scene: _Scene, candidates: set[int]) -> list[int]:
    """Donor atoms for a ligand the input left unattached.

    Anionic atoms first, with a second donor where it would close a five- or
    six-membered chelate ring (acetylacetonate, oxalate, glycinate); then
    phosphorus, a chelating N,N pair, both carbons of an alkene, else one atom.
    """

    anions = sorted(idx for idx in candidates if scene.atoms[idx].charge < 0)
    if anions:
        first = anions[0]
        partners = [
            idx
            for idx in sorted(candidates)
            if idx != first
            and scene.atoms[idx].symbol in ("O", "N", "S")
            and scene.atoms[idx].charge <= 0
            and _bonds_apart(scene, first, idx) in (3, 4)
        ]
        return [first, *partners[:1]]
    for symbol in _DONOR_PREFERENCE:
        matches = sorted(idx for idx in candidates if scene.atoms[idx].symbol == symbol)
        if not matches:
            continue
        if symbol == "P" or (symbol == "N" and _chelating_pair(scene, matches)):
            return matches
        if symbol == "C":
            alkene = sorted(sorted(pair) for pair, order in scene.bonds.items() if order == 2 and pair <= candidates)
            if alkene:
                return alkene[0]
        return matches[:1]
    return [min(candidates)]


# Six sites drawn as an octahedron seen from the side, in order round the metal:
# two in the plane of the page, two behind it (hashed) and two in front (wedged).
_OCTAHEDRON = {90.0: "plain", 25.0: "hash", 335.0: "wedge", 270.0: "plain", 205.0: "wedge", 155.0: "hash"}


def _site_angles(hapto_count: int, denticities: list[int]) -> tuple[list[float], list[list[float]]]:
    """Directions for the face-bound rings and, ligand by ligand, for the donor atoms."""

    total = sum(denticities)
    if hapto_count == 0 and total == 6 and max(denticities) == 1:
        flat = list(_OCTAHEDRON)
        ring_angles: list[float] = []
    elif hapto_count == 0:
        step = 360.0 / max(total, 1)
        first = 180.0 - (denticities[0] - 1) / 2 * step if denticities else 180.0
        flat = [first + step * position for position in range(total)]
        ring_angles = []
    elif hapto_count == 1:
        span = 0.0 if total <= 1 else min(40.0 * total, 170.0)
        flat = [270.0 - span / 2 + (span * position / (total - 1) if total > 1 else 0.0) for position in range(total)]
        ring_angles = [90.0]
    elif hapto_count == 2:
        span = 0.0 if total <= 1 else min(50.0 * (total - 1), 120.0)
        flat = [-span / 2 + (span * position / (total - 1) if total > 1 else 0.0) for position in range(total)]
        ring_angles = [90.0, 270.0]
    else:
        slots = hapto_count + total
        every = [90.0 + 360.0 * position / slots for position in range(slots)]
        ring_angles, flat = every[:hapto_count], every[hapto_count:]
    per_ligand, cursor = [], 0
    for denticity in denticities:
        per_ligand.append(flat[cursor : cursor + denticity])
        cursor += denticity
    return ring_angles, per_ligand


def _layout(state, scene: _Scene) -> None:
    placed: set[int] = set()
    centre = state.centre
    unit_of = {ring[0]: metal for metal, rings in state.units.items() for ring in rings}

    if centre is not None:
        scene.pos[centre] = (0.0, 0.0)
        placed.add(centre)
        centre_rings = [ring for ring, owner in state.ring_owner.items() if owner == centre]
        # A benzene ring of a hydrocarbon ligand the input left unattached sits
        # on the metal face-on, like a cyclopentadienyl ring.
        mol = state.mol
        for group in state.inner_groups:
            if group & state.donors_of[centre] or any(mol.GetAtomWithIdx(idx).GetAtomicNum() != 6 for idx in group):
                continue
            for ring in mol.GetRingInfo().AtomRings():
                if (
                    len(ring) == 6
                    and set(ring) <= group
                    and all(idx in scene.atoms and mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring)
                ):
                    centre_rings.append(tuple(ring))
                    break
        centre_ring_atoms = {idx for ring in centre_rings for idx in ring}

        ligands: list[tuple[set[int], list[int]]] = []  # atoms, donor atoms
        for group in state.inner_groups:
            atoms = {idx for idx in group if idx in scene.atoms}
            if atoms & centre_ring_atoms:
                continue
            donors = sorted(atoms & state.donors_of[centre])
            if not donors:
                ring_atoms = {idx for ring in state.ring_owner for idx in ring}
                donors = _guess_donors(scene, (atoms - set(state.units) - ring_atoms) or atoms)
            ligands.append((atoms, donors))
            scene.links.extend((centre, donor) for donor in donors)
            scene.entity |= atoms
        scene.entity |= {centre} | {idx for idx in centre_ring_atoms if idx in scene.atoms}
        scene.entity_charge = (
            sum(state.mol.GetAtomWithIdx(idx).GetFormalCharge() for group in state.inner_groups for idx in group)
            + state.mol.GetAtomWithIdx(centre).GetFormalCharge()
        )
        ligands.sort(key=lambda item: (-len(item[1]), min(item[0])))

        ring_angles, ligand_angles = _site_angles(len(centre_rings), [len(donors) for _, donors in ligands])
        for ring, angle in zip(centre_rings, ring_angles, strict=True):
            axis = _unit(angle)
            _place_ring(scene, ring, centre, (axis[0] * _RING_DISTANCE, axis[1] * _RING_DISTANCE), axis, placed)

        crowded = sum(len(donors) for _, donors in ligands) >= 6
        for (atoms, donors), angles in zip(ligands, ligand_angles, strict=True):
            unit_metals = sorted(metal for metal in atoms if metal in state.units)
            if unit_metals:
                if len(unit_metals) > 1:
                    raise _CannotDraw("a ligand with more than one metallocene")
                local_placed: set[int] = set()
                saved = dict(scene.pos)
                done = _layout_sandwich(scene, unit_metals[0], state.units[unit_metals[0]], local_placed)
                if done != atoms:
                    raise _CannotDraw("a metallocene ligand with detached parts")
                local_scene_pos = {idx: scene.pos[idx] for idx in atoms}
                scene.pos = saved
            else:
                # Laid out with its bonds to the metal, so a chelate closes its ring.
                free = _free_coordinates(scene, atoms, tuple(donors))
                metal_at = free.pop(_METAL_KEY)
                middle = _centroid([free[donor] for donor in donors])
                outward = _normalise((middle[0] - metal_at[0], middle[1] - metal_at[1]))
                moved = _rigid(free, metal_at, outward, _unit(sum(angles) / len(angles)), (0.0, 0.0))
                if len(donors) == 1:
                    shift = _unit(angles[0])
                    extra = _METAL_BOND - math.dist(moved[donors[0]], (0.0, 0.0))
                    moved = {idx: (x + shift[0] * extra, y + shift[1] * extra) for idx, (x, y) in moved.items()}
                elif crowded:
                    # Three chelate rings round one metal: draw each a little further
                    # out, so neighbouring donors do not land on top of each other.
                    shift = _unit(sum(angles) / len(angles))
                    moved = {idx: (x + shift[0] * 0.7, y + shift[1] * 0.7) for idx, (x, y) in moved.items()}
                scene.pos.update(moved)
                placed |= atoms
                octahedral = not centre_rings and len(ligands) == 6 and all(len(d) == 1 for _, d in ligands)
                if octahedral and _OCTAHEDRON.get(angles[0], "plain") != "plain":
                    scene.link_style[(centre, donors[0])] = _OCTAHEDRON[angles[0]]
                continue

            # A metallocene ligand keeps its own sandwich layout and is moved as one piece.
            direction = _unit(sum(angles) / len(angles))
            anchor = _centroid([local_scene_pos[donor] for donor in donors])
            rest = [local_scene_pos[idx] for idx in atoms if idx not in donors]
            middle = _centroid(rest)
            body = _normalise((middle[0] - anchor[0], middle[1] - anchor[1]))
            spread = max(math.dist(local_scene_pos[donor], anchor) for donor in donors)
            reach = _METAL_BOND if len(donors) == 1 else max(math.sqrt(max(_METAL_BOND**2 - spread**2, 0.0)), 1.3)
            moved = _rigid(local_scene_pos, anchor, body, direction, (direction[0] * reach, direction[1] * reach))
            scene.pos.update(moved)
            placed |= atoms

    # Whatever is left stands beside the complex: free metallocenes, counter-ions, solvent.
    cursor = (max((p[0] for p in scene.pos.values()), default=0.0) + 2.5) if scene.pos else 0.0
    if scene.entity_charge:
        cursor += 1.6  # room for the bracket and its charge
    # Identical one-atom counter-ions are written once, with a count.
    seen: dict[tuple[str, int], int] = {}
    for idx in sorted(scene.atoms):
        atom = scene.atoms[idx]
        if idx in placed or idx in state.units or scene.neighbours(idx):
            continue
        first = seen.setdefault((atom.symbol, atom.charge), idx)
        if first != idx:
            scene.atoms[first].count += 1
            del scene.atoms[idx]
    for idx in sorted(scene.atoms):
        if idx in placed:
            continue
        if idx in state.units:
            saved = dict(scene.pos)
            scene.pos = {}
            local_placed = set()
            members = _layout_sandwich(scene, idx, state.units[idx], local_placed)
            local = dict(scene.pos)
            scene.pos = saved
        elif any(idx in ring for ring in state.ring_owner if ring[0] in unit_of):
            continue  # placed with its metallocene
        else:
            members = scene.component(idx, set())
            if any(ring[0] in members for ring in state.ring_owner if ring[0] in unit_of):
                continue
            local = _free_coordinates(scene, members)
        left = min(p[0] for p in local.values())
        middle = (min(p[1] for p in local.values()) + max(p[1] for p in local.values())) / 2
        for atom, (x, y) in local.items():
            scene.pos[atom] = (x - left + cursor, y - middle)
        cursor = max(p[0] for p in scene.pos.values()) + 2.0
        placed |= set(local)


def _spread_leaves(scene: _Scene) -> None:
    """Move each donor's single-atom substituents into the space its metal bond leaves free."""

    from itertools import product

    ring_atoms = {idx for ring in scene.rings for idx in ring.atoms}
    metals_of: dict[int, list[int]] = {}
    for metal, donor in scene.links:
        metals_of.setdefault(donor, []).append(metal)
    for donor, metals in metals_of.items():
        if donor in ring_atoms:
            continue
        neighbours = scene.neighbours(donor)
        leaves = [n for n in neighbours if len(scene.neighbours(n)) == 1 and n not in metals_of]
        if not leaves:
            continue
        origin = scene.pos[donor]

        def angle(idx: int, origin: Point = origin) -> float:
            return math.degrees(math.atan2(scene.pos[idx][1] - origin[1], scene.pos[idx][0] - origin[0])) % 360.0

        if len(leaves) == len(neighbours) and len(metals) == 1:
            # Nothing but the metal holds this donor in place: fan its
            # substituents out behind it, clear of the neighbouring ligands.
            away = angle(metals[0]) + 180.0
            span = min(55.0 * (len(leaves) - 1), 150.0)
            for position, leaf in enumerate(sorted(leaves)):
                slot = away - span / 2 + (span * position / (len(leaves) - 1) if len(leaves) > 1 else 0.0)
                length = math.dist(scene.pos[leaf], origin)
                direction = _unit(slot)
                scene.pos[leaf] = (origin[0] + direction[0] * length, origin[1] + direction[1] * length)
            continue

        fixed = sorted(angle(idx) for idx in [n for n in neighbours if n not in leaves] + metals)
        gaps = [(start, (fixed[(i + 1) % len(fixed)] - start) % 360.0 or 360.0) for i, start in enumerate(fixed)]
        best: tuple[float, tuple[int, ...]] | None = None
        for counts in product(range(len(leaves) + 1), repeat=len(gaps)):
            if sum(counts) != len(leaves):
                continue
            tightest = min(width / (count + 1) for (_, width), count in zip(gaps, counts, strict=True) if count)
            if best is None or tightest > best[0] + 1e-6:
                best = (tightest, counts)
        assert best is not None
        slots = [
            start + width * (position + 1) / (count + 1)
            for (start, width), count in zip(gaps, best[1], strict=True)
            for position in range(count)
        ]
        for leaf, slot in zip(sorted(leaves, key=angle), sorted(slots, key=lambda a: a % 360.0), strict=True):
            length = math.dist(scene.pos[leaf], origin)
            direction = _unit(slot)
            scene.pos[leaf] = (origin[0] + direction[0] * length, origin[1] + direction[1] * length)


def _settle_charges(state, scene: _Scene) -> None:
    """Show the charge of each coordination entity on its metal instead of on the donors."""

    for metal in state.metals:
        scene.atoms[metal].always_labelled = True
    for ring, owner in state.ring_owner.items():
        if owner is None:
            continue
        for idx in ring:
            scene.atoms[owner].charge += scene.atoms[idx].charge
            scene.atoms[idx].charge = 0
    for metal, donor in scene.links:
        atom = scene.atoms[donor]
        if atom.tail:  # CO: the zwitterion's charges cancel
            atom.charge = 0
        elif atom.charge < 0:
            if atom.charge == -2:
                scene.link_order[(metal, donor)] = 2  # an oxido or imido ligand: M=O
            scene.atoms[metal].charge += atom.charge
            atom.charge = 0
        side_on = any(other in scene.neighbours(donor) for m, other in scene.links if m == metal)
        if atom.symbol == "H" or (atom.symbol == "C" and not side_on):
            atom.always_labelled = True
    if scene.entity_charge and state.centre is not None:
        scene.atoms[state.centre].charge = 0  # written on the bracket instead


def _labelled(scene: _Scene, idx: int) -> bool:
    atom = scene.atoms[idx]
    return atom.always_labelled or atom.symbol != "C" or atom.charge != 0 or not scene.neighbours(idx)


def _charge_text(charge: int) -> str:
    if not charge:
        return ""
    sign = "+" if charge > 0 else "−"
    return sign if abs(charge) == 1 else f"{abs(charge)}{sign}"


def _text_width(text: str, size: float) -> float:
    return sum(0.72 if ch.isupper() else 0.56 for ch in text) * size


def _render(scene: _Scene) -> str:
    xs = [p[0] for p in scene.pos.values()]
    ys = [p[1] for p in scene.pos.values()]
    margin = 1.6
    min_x, max_y = min(xs) - margin, max(ys) + margin
    width = (max(xs) - min(xs) + 2 * margin) * _SCALE
    height = (max(ys) - min(ys) + 2 * margin) * _SCALE

    def px(point: Point) -> Point:
        return ((point[0] - min_x) * _SCALE, (max_y - point[1]) * _SCALE)

    gap = _FONT * 0.62
    parts: list[str] = []

    def line(a: Point, b: Point, trim_a: float, trim_b: float, width_: float = 1.6, offset: float = 0.0) -> None:
        length = math.dist(a, b)
        if length <= trim_a + trim_b + 0.5:
            return
        ux, uy = (b[0] - a[0]) / length, (b[1] - a[1]) / length
        ox, oy = -uy * offset, ux * offset
        parts.append(
            f'<line x1="{a[0] + ux * trim_a + ox:.1f}" y1="{a[1] + uy * trim_a + oy:.1f}" '
            f'x2="{b[0] - ux * trim_b + ox:.1f}" y2="{b[1] - uy * trim_b + oy:.1f}" stroke-width="{width_}"/>'
        )

    ring_bonds = {
        frozenset((r.atoms[i], r.atoms[(i + 1) % len(r.atoms)])) for r in scene.rings for i in range(len(r.atoms))
    }

    for pair, order in scene.bonds.items():
        if pair in ring_bonds:
            continue
        a, b = tuple(pair)
        trims = [gap if _labelled(scene, idx) else 0.0 for idx in (a, b)]
        offsets = {1: (0.0,), 2: (-2.4, 2.4), 3: (-3.6, 0.0, 3.6)}.get(order, (0.0,))
        for offset in offsets:
            line(px(scene.pos[a]), px(scene.pos[b]), trims[0], trims[1], offset=offset)

    for metal, donor in scene.links:
        start, end = px(scene.pos[metal]), px(scene.pos[donor])
        trim_end = gap * 1.25 if _labelled(scene, donor) else 0.0
        style = scene.link_style.get((metal, donor))
        length = math.dist(start, end)
        if style is None or length <= gap * 2.4 + 2:
            for offset in (-2.4, 2.4) if scene.link_order.get((metal, donor)) == 2 else (0.0,):
                line(start, end, gap * 1.15, gap if _labelled(scene, donor) else 0.0, offset=offset)
            continue
        ux, uy = (end[0] - start[0]) / length, (end[1] - start[1]) / length
        a = (start[0] + ux * gap * 1.15, start[1] + uy * gap * 1.15)
        b = (end[0] - ux * trim_end, end[1] - uy * trim_end)
        nx, ny, wide = -uy, ux, 4.5
        if style == "wedge":
            parts.append(
                f'<polygon points="{a[0]:.1f},{a[1]:.1f} {b[0] + nx * wide:.1f},{b[1] + ny * wide:.1f} '
                f'{b[0] - nx * wide:.1f},{b[1] - ny * wide:.1f}" stroke-width="0.8"/>'
            )
        else:
            steps = 8
            for step in range(1, steps + 1):
                t = step / steps
                cx, cy, half = a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, 0.8 + (wide - 0.8) * t
                parts.append(
                    f'<line x1="{cx + nx * half:.1f}" y1="{cy + ny * half:.1f}" '
                    f'x2="{cx - nx * half:.1f}" y2="{cy - ny * half:.1f}" stroke-width="1.3"/>'
                )

    if scene.entity_charge and scene.entity:
        corners = [px(scene.pos[idx]) for idx in scene.entity if idx in scene.pos]
        pad, lip = _SCALE * 1.15, 9.0
        left, right = min(c[0] for c in corners) - pad, max(c[0] for c in corners) + pad
        top, bottom = min(c[1] for c in corners) - pad * 0.8, max(c[1] for c in corners) + pad * 0.8
        for edge, inward in ((left, lip), (right, -lip)):
            parts.append(
                f'<path d="M{edge + inward:.1f},{top:.1f} H{edge:.1f} V{bottom:.1f} H{edge + inward:.1f}" '
                f'fill="none" stroke-width="2"/>'
            )
        parts.append(
            f'<text x="{right + 5:.1f}" y="{top + _FONT * 0.6:.1f}">{_charge_text(scene.entity_charge)}</text>'
        )

    for ring in scene.rings:
        points = [px(scene.pos[idx]) for idx in ring.atoms]
        centre = _centroid(points)
        metal = px(scene.pos[ring.metal])
        line(metal, centre, gap * 1.15, 0.0)
        count = len(points)
        for position in range(count):
            a, b = ring.atoms[position], ring.atoms[(position + 1) % count]
            front = (ring.depth[a] + ring.depth[b]) / 2 < -0.3
            line(points[position], points[(position + 1) % count], 0.0, 0.0, width_=5.0 if front else 1.6)
        axis = _normalise((centre[0] - metal[0], centre[1] - metal[1])) or (0.0, -1.0)
        tilt = math.degrees(math.atan2(axis[1], axis[0])) - 90.0
        radius = _RING_RADIUS * _SCALE * 0.52
        parts.append(
            f'<ellipse cx="{centre[0]:.1f}" cy="{centre[1]:.1f}" rx="{radius:.1f}" ry="{radius * _RING_SQUASH:.1f}" '
            f'transform="rotate({tilt:.1f} {centre[0]:.1f} {centre[1]:.1f})" fill="none" stroke-width="1.4"/>'
        )

    small = _FONT * 0.68
    for idx, atom in scene.atoms.items():
        if not _labelled(scene, idx):
            continue
        x, y = px(scene.pos[idx])
        partners = scene.neighbours(idx) + [m for m, d in scene.links if d == idx]
        flipped = (
            bool(partners)
            and (atom.hydrogens or atom.tail)
            and all(scene.pos[p][0] > scene.pos[idx][0] + 0.3 for p in partners)
        )
        main_width = _text_width(atom.symbol, _FONT)
        hydrogens = ""
        if atom.hydrogens:
            count = f'<tspan font-size="{small:.1f}" dy="4">{atom.hydrogens}</tspan><tspan dy="-4">​</tspan>'
            hydrogens = "H" + (count if atom.hydrogens > 1 else "")
        charge = _charge_text(atom.charge)
        charge_span = f'<tspan font-size="{small:.1f}" dy="-6">{charge}</tspan>' if charge else ""
        baseline = y + _FONT * 0.36
        if atom.count > 1:
            parts.append(
                f'<text x="{x - main_width / 2 - _FONT * 0.35:.1f}" y="{baseline:.1f}">({atom.symbol}{charge_span}'
                f'<tspan dy="6">)</tspan><tspan font-size="{small:.1f}" dy="4">{atom.count}</tspan></text>'
            )
            continue
        if flipped:
            before = (atom.tail or "") + hydrogens
            parts.append(
                f'<text x="{x + main_width / 2:.1f}" y="{baseline:.1f}" text-anchor="end">{before}{atom.symbol}</text>'
            )
            if charge:
                parts.append(
                    f'<text x="{x + main_width / 2:.1f}" y="{baseline - 6:.1f}" font-size="{small:.1f}">{charge}</text>'
                )
        else:
            parts.append(
                f'<text x="{x - main_width / 2:.1f}" y="{baseline:.1f}">{atom.symbol}{atom.tail}{hydrogens}{charge_span}</text>'
            )

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}" '
        f'viewBox="0 0 {width:.0f} {height:.0f}" stroke="currentColor" fill="currentColor" stroke-linecap="round" '
        f'font-family="Helvetica, Arial, sans-serif" font-size="{_FONT:.0f}">'
        + "".join(parts).replace("<text ", '<text stroke="none" ')
        + "</svg>"
    )


def depict_complex_svg(smiles: str = "", *, molblock: str = "", rdkit_mol=None) -> str | None:
    """Return an SVG of the complex or metallocene in ``smiles``/``molblock``, laid out afresh.

    ``None`` when the input has no transition metal or is not a structure this
    module perceives or can lay out.
    """

    if molblock and rdkit_mol is None:
        rdkit_mol = read_molblock(molblock)
    if not contains_transition_metal(smiles, rdkit_mol):
        return None
    try:
        _, state = _perceive(smiles, rdkit_mol)
        scene = _build_scene(state)
        _layout(state, scene)
        _spread_leaves(scene)
        _settle_charges(state, scene)
        if set(scene.atoms) - set(scene.pos):
            return None
        return _render(scene)
    except (ValueError, _CannotDraw):
        return None


__all__ = ["depict_complex_svg"]
