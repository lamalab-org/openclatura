"""Coordination compounds: one transition-metal centre and its ligands.

Whatever way the input draws the complex -- ligands beside a bare metal,
dative bonds, ordinary bonds to the metal, a haptic bond from a molfile, or
PubChem's radical rings beside a neutral metal -- it is reduced to one model:

    central atom + oxidation state + ligands (each with a count and, when the
    input shows them, its donor atoms)

The reduction detaches every ligand from the metal and moves the bonding
electrons onto the ligand, so ``Cl[Pd]Cl`` becomes Pd(2+) and two chlorides
while ``P->[Pd]`` leaves a neutral phosphane.  The additive name is then the
ligands in alphabetical order followed by the metal and its oxidation number:
``[1,1'-bis(diphenylphosphanyl)ferrocene]tetracarbonylchromium(0)``.

A metal whose only partners are two cyclopentadienide rings is a metallocene,
not a centre: it is handed back in the ionic drawing the metallocene namer
reads, which is how ``dppf`` inside a larger complex gets its ligand name.

Not modelled: geometry (cis/trans, fac/mer, polyhedral symbols), bridging
ligands and more than one metal centre.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from .coordination_names import ATE_NAMES, LIGAND_SMILES
from .metallocenes import METALLOCENE_PARENT_NAMES, UnsupportedMetalError
from .rules import multipliers

# d-block and lanthanides.  Main-group metals keep their existing treatment.
_TRANSITION_METALS = frozenset([*range(21, 31), *range(39, 49), *range(57, 81)])
_METAL_SYMBOLS = (
    "Sc Ti V Cr Mn Fe Co Ni Cu Zn Y Zr Nb Mo Tc Ru Rh Pd Ag Cd La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu "
    "Hf Ta W Re Os Ir Pt Au Hg"
).split()
_METAL_IN_SMILES_RE = re.compile(r"\[\d*(?:" + "|".join(sorted(_METAL_SYMBOLS, key=len, reverse=True)) + r")(?![a-z])")

# When a drawing does not say which metal two free rings belong to, the
# commonest metallocene wins.
_METALLOCENE_PRIORITY = ("Fe", "Ru", "Os", "Co", "Ni", "Cr", "V", "Ti")

# Preferred ligand word for a ligand structure, where it is not a compound name.
_PREFERRED_LIGAND_WORDS = (
    "carbonyl",
    "aqua",
    "ammine",
    "nitrosyl",
    "hydrido",
    "fluorido",
    "chlorido",
    "bromido",
    "iodido",
    "cyanido",
    "hydroxido",
    "oxido",
    "methyl",
    "cyclopentadienyl",
)
_ROMAN = ("0", "I", "II", "III", "IV", "V", "VI", "VII", "VIII")
_DONOR_PROP = "_openclatura_donor"
_RING_SIZE = 5


@dataclass(frozen=True)
class Ligand:
    """One kind of ligand: its free structure, charge, and donor atoms if the input showed them."""

    smiles: str
    charge: int
    count: int = 1
    donors: tuple[str, ...] = ()  # element symbols of the atoms bound to the metal
    hapto: bool = False  # donors are the whole cyclopentadienide ring


@dataclass(frozen=True)
class Complex:
    """A mononuclear coordination entity."""

    metal: str
    oxidation_state: int
    ligands: tuple[Ligand, ...]
    other_components: str = ""  # counter-ions and solvent outside the coordination sphere

    @property
    def charge(self) -> int:
        return self.oxidation_state + sum(ligand.charge * ligand.count for ligand in self.ligands)


@dataclass(frozen=True)
class MetalRoute:
    """How an input containing a transition metal is to be named.

    ``smiles`` is the input redrawn with every ligand detached.  ``complex`` is
    set when there is a metal centre to name additively; otherwise the redrawn
    SMILES goes through ordinary naming (a metallocene on its own).
    """

    smiles: str
    complex: Complex | None = None


@dataclass
class _Reduction:
    """Working state while the ligands are detached from the metals."""

    mol: object
    metals: list[int]
    donors_of: dict[int, set[int]] = field(default_factory=dict)  # metal -> donor atoms
    ring_owner: dict[tuple[int, ...], int | None] = field(default_factory=dict)  # Cp ring -> metal
    radical_rings: list[tuple[int, ...]] = field(default_factory=list)
    had_metal_bonds: bool = False


def contains_transition_metal(smiles: str = "", rdkit_mol=None) -> bool:
    """Cheap test for whether an input needs the coordination route at all."""

    if rdkit_mol is not None:
        return any(atom.GetAtomicNum() in _TRANSITION_METALS for atom in rdkit_mol.GetAtoms())
    return bool(smiles) and _METAL_IN_SMILES_RE.search(smiles) is not None


def read_molblock(molblock: str):
    """Read a molfile, turning haptic (multi-centre) bonds into dative bonds from each ring atom."""

    from rdkit import Chem

    mol = Chem.MolFromMolBlock(molblock, sanitize=False, removeHs=False)
    if mol is None:
        raise ValueError("could not read the molfile")
    mol = Chem.HapticBondsToDative(mol)
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        mol.UpdatePropertyCache(strict=False)
        Chem.AssignRadicals(mol)
    return Chem.RemoveHs(mol, sanitize=False)


def _parse(smiles: str):
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            raise ValueError("could not parse the SMILES")
        mol.UpdatePropertyCache(strict=False)
        Chem.AssignRadicals(mol)
    return mol


def _isoelectronic_valence(atomic_number: int, charge: int) -> int | None:
    from rdkit import Chem

    neighbour = atomic_number - charge
    if not 1 <= neighbour <= 118:
        return None
    return Chem.GetPeriodicTable().GetDefaultValence(neighbour)


def _detach_metals(mol) -> _Reduction:
    """Remove every metal-ligand bond, pinning each donor's hydrogens as drawn."""

    from rdkit import Chem

    rw = Chem.RWMol(mol)
    metals = [atom.GetIdx() for atom in rw.GetAtoms() if atom.GetAtomicNum() in _TRANSITION_METALS]
    metal_set = set(metals)
    state = _Reduction(mol=rw, metals=metals, donors_of={metal: set() for metal in metals})

    records: list[tuple[int, int, bool, int]] = []  # metal, donor, dative, order
    for bond in rw.GetBonds():
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if begin in metal_set and end in metal_set:
            raise UnsupportedMetalError("metal-metal bonds are not supported")
        if begin in metal_set or end in metal_set:
            metal, donor = (begin, end) if begin in metal_set else (end, begin)
            dative = bond.GetBondType() == Chem.BondType.DATIVE
            records.append((metal, donor, dative, 0 if dative else int(round(bond.GetBondTypeAsDouble()))))
    state.had_metal_bonds = bool(records)

    hydrogens = {donor: rw.GetAtomWithIdx(donor).GetTotalNumHs() for _, donor, _, _ in records}
    for metal, donor, _, _ in records:
        rw.RemoveBond(metal, donor)
        state.donors_of[metal].add(donor)
    for donor, count in hydrogens.items():
        atom = rw.GetAtomWithIdx(donor)
        atom.SetNoImplicit(True)
        atom.SetNumExplicitHs(count)
    for metal in metals:
        atom = rw.GetAtomWithIdx(metal)
        atom.SetNoImplicit(True)
        atom.SetNumExplicitHs(0)
        atom.SetNumRadicalElectrons(0)

    _settle_cyclopentadienide_rings(state)
    ring_atoms = {idx for ring in state.ring_owner for idx in ring}

    # An ordinary (non-dative) bond to the metal is a shared pair: give it to
    # the donor when the donor would otherwise be short of its valence.
    for metal, donor, dative, _ in records:
        if dative or donor in ring_atoms:
            continue
        atom = rw.GetAtomWithIdx(donor)
        valence = int(round(sum(b.GetBondTypeAsDouble() for b in atom.GetBonds()))) + atom.GetNumExplicitHs()
        charge = atom.GetFormalCharge()
        for taken in range(4):
            if _isoelectronic_valence(atom.GetAtomicNum(), charge - taken) == valence:
                atom.SetFormalCharge(charge - taken)
                metal_atom = rw.GetAtomWithIdx(metal)
                metal_atom.SetFormalCharge(metal_atom.GetFormalCharge() + taken)
                break

    _canonicalise_carbonyls(state)
    return state


def _settle_cyclopentadienide_rings(state: _Reduction) -> None:
    """Find isolated C5 rings drawn as cyclopentadienyl and redraw them as the aromatic anion."""

    from rdkit import Chem

    rw = state.mol
    rw.ClearComputedProps()
    Chem.FastFindRings(rw)
    ring_info = rw.GetRingInfo()
    for ring in ring_info.AtomRings():
        atoms = [rw.GetAtomWithIdx(idx) for idx in ring]
        if not (
            len(ring) == _RING_SIZE
            and all(atom.GetAtomicNum() == 6 for atom in atoms)
            and all(ring_info.NumAtomRings(idx) == 1 for idx in ring)
            and all(atom.GetDegree() <= 3 for atom in atoms)
        ):
            continue
        owner = next((metal for metal, donors in state.donors_of.items() if donors.issuperset(ring)), None)
        is_anion = sorted(atom.GetFormalCharge() for atom in atoms) == [-1, 0, 0, 0, 0]
        is_radical = all(atom.GetNumRadicalElectrons() >= 1 for atom in atoms)
        if owner is not None:
            if not is_anion:
                _draw_cyclopentadienide(rw, ring)
                metal_atom = rw.GetAtomWithIdx(owner)
                metal_atom.SetFormalCharge(metal_atom.GetFormalCharge() + 1)
            else:
                for atom in atoms:
                    atom.SetNoImplicit(False)
                    atom.SetNumExplicitHs(0)
                _draw_cyclopentadienide(rw, ring)
            state.ring_owner[tuple(ring)] = owner
        elif is_radical and not any(atom.GetFormalCharge() for atom in atoms):
            _draw_cyclopentadienide(rw, ring)
            state.ring_owner[tuple(ring)] = None
            state.radical_rings.append(tuple(ring))
        elif is_anion and not any(set(ring) & donors for donors in state.donors_of.values()):
            double_bonds = sum(
                rw.GetBondBetweenAtoms(ring[i], ring[(i + 1) % _RING_SIZE]).GetBondTypeAsDouble() > 1
                for i in range(_RING_SIZE)
            )
            if all(atom.GetIsAromatic() for atom in atoms) or double_bonds == 2:
                state.ring_owner[tuple(ring)] = None


def _draw_cyclopentadienide(rw, ring: tuple[int, ...]) -> None:
    """Redraw a C5 ring as cyclopentadienide: one anionic carbon and two ring double bonds."""

    from rdkit import Chem

    ring_set = set(ring)
    substituted = [
        idx for idx in ring if any(n.GetIdx() not in ring_set for n in rw.GetAtomWithIdx(idx).GetNeighbors())
    ]
    start = ring.index(substituted[0]) if substituted else 0
    ordered = [ring[(start + offset) % _RING_SIZE] for offset in range(_RING_SIZE)]
    for position, idx in enumerate(ordered):
        atom = rw.GetAtomWithIdx(idx)
        atom.SetFormalCharge(-1 if position == 0 else 0)
        atom.SetNumRadicalElectrons(0)
        atom.SetIsAromatic(False)
        atom.SetNoImplicit(False)
        atom.SetNumExplicitHs(0)
    for position in range(_RING_SIZE):
        bond = rw.GetBondBetweenAtoms(ordered[position], ordered[(position + 1) % _RING_SIZE])
        bond.SetIsAromatic(False)
        bond.SetBondType(Chem.BondType.DOUBLE if position in (1, 3) else Chem.BondType.SINGLE)


def _canonicalise_carbonyls(state: _Reduction) -> None:
    """Draw every detached C-O ligand as ``[C-]#[O+]``, whatever bond orders it came with."""

    from rdkit import Chem

    rw = state.mol
    for fragment in Chem.GetMolFrags(rw):
        if len(fragment) != 2:
            continue
        atoms = [rw.GetAtomWithIdx(idx) for idx in fragment]
        if sorted(atom.GetAtomicNum() for atom in atoms) != [6, 8] or any(atom.GetTotalNumHs() for atom in atoms):
            continue
        carbon, oxygen = sorted(atoms, key=lambda atom: atom.GetAtomicNum())
        surplus = carbon.GetFormalCharge() + oxygen.GetFormalCharge()
        if surplus:
            owner = next((m for m, donors in state.donors_of.items() if donors & set(fragment)), None)
            if owner is None:
                continue
            metal_atom = rw.GetAtomWithIdx(owner)
            metal_atom.SetFormalCharge(metal_atom.GetFormalCharge() + surplus)
        rw.GetBondBetweenAtoms(carbon.GetIdx(), oxygen.GetIdx()).SetBondType(Chem.BondType.TRIPLE)
        for atom, charge in ((carbon, -1), (oxygen, 1)):
            atom.SetFormalCharge(charge)
            atom.SetNumRadicalElectrons(0)
            atom.SetNoImplicit(True)
            atom.SetNumExplicitHs(0)


def _claim_metallocenes(state: _Reduction) -> dict[int, list[tuple[int, ...]]]:
    """Pair metals with two cyclopentadienide rings each; return metal -> its rings."""

    rw = state.mol
    ring_donor_atoms = {idx for ring in state.ring_owner for idx in ring}
    units: dict[int, list[tuple[int, ...]]] = {}

    def priority(metal: int) -> int:
        symbol = rw.GetAtomWithIdx(metal).GetSymbol()
        return _METALLOCENE_PRIORITY.index(symbol) if symbol in _METALLOCENE_PRIORITY else len(_METALLOCENE_PRIORITY)

    for metal in sorted(state.metals, key=priority):
        atom = rw.GetAtomWithIdx(metal)
        if atom.GetSymbol() not in METALLOCENE_PARENT_NAMES or state.donors_of[metal] - ring_donor_atoms:
            continue
        owned = [ring for ring, owner in state.ring_owner.items() if owner == metal]
        if len(owned) == 2 and atom.GetFormalCharge() == 2:
            units[metal] = owned
            continue
        if owned:
            continue
        free = [ring for ring, owner in state.ring_owner.items() if owner is None]
        radical = [ring for ring in free if ring in state.radical_rings]
        ionic = [ring for ring in free if ring not in state.radical_rings]
        if len(radical) == 2 and atom.GetFormalCharge() == 0:
            atom.SetFormalCharge(2)
            chosen = radical
            state.radical_rings = []
        elif len(ionic) == 2 and atom.GetFormalCharge() == 2 and not radical:
            chosen = ionic
        else:
            continue
        for ring in chosen:
            state.ring_owner[ring] = metal
        units[metal] = chosen
    return units


def perceive(smiles: str = "", rdkit_mol=None) -> MetalRoute:
    """Reduce an input containing a transition metal to a complex model or a plain metallocene."""

    from rdkit import Chem

    mol = rdkit_mol if rdkit_mol is not None else _parse(smiles)
    state = _detach_metals(mol)
    rw = state.mol
    units = _claim_metallocenes(state)
    centres = [metal for metal in state.metals if metal not in units]

    # Whether the input itself bonds ligands to the centre; without that, every
    # other component counts as a ligand.
    drawn_bonds = {metal: bool(donors) for metal, donors in state.donors_of.items()}
    if len(centres) == 1 and state.radical_rings:
        # A radical ring left over belongs to the centre (half-sandwich complexes).
        atom = rw.GetAtomWithIdx(centres[0])
        atom.SetFormalCharge(atom.GetFormalCharge() + len(state.radical_rings))
        for ring in state.radical_rings:
            state.ring_owner[ring] = centres[0]
            state.donors_of[centres[0]].update(ring)
        state.radical_rings = []
    if state.radical_rings:
        raise UnsupportedMetalError("could not assign every cyclopentadienyl ring to a metal")

    for metal, donors in state.donors_of.items():
        for donor in donors:
            rw.GetAtomWithIdx(donor).SetIntProp(_DONOR_PROP, metal)
    try:
        Chem.SanitizeMol(rw)
    except Exception as exc:
        raise UnsupportedMetalError(f"ligands are not valid structures once detached from the metal: {exc}") from exc
    normalised = Chem.MolToSmiles(rw)

    if not centres:
        return MetalRoute(smiles=normalised)
    if len(centres) > 1:
        raise UnsupportedMetalError("more than one metal centre is not supported")
    centre = centres[0]
    centre_atom = rw.GetAtomWithIdx(centre)

    fragments = [set(fragment) for fragment in Chem.GetMolFrags(rw) if centre not in fragment]
    groups: list[set[int]] = []
    for metal, rings in units.items():
        unit = {metal}
        for ring in rings:
            unit |= next(fragment for fragment in fragments if ring[0] in fragment)
        groups.append(unit)
    grouped_atoms = set().union(*groups) if groups else set()
    groups.extend(fragment for fragment in fragments if not fragment & grouped_atoms)

    centre_donors = state.donors_of[centre]
    centre_rings = [set(ring) for ring, owner in state.ring_owner.items() if owner == centre]
    disconnected = not drawn_bonds[centre]
    inner: dict[tuple[str, int, tuple[str, ...], bool], int] = {}
    outer: list[str] = []
    for group in groups:
        fragment_smiles = Chem.MolFragmentToSmiles(rw, atomsToUse=sorted(group))
        charge = sum(rw.GetAtomWithIdx(idx).GetFormalCharge() for idx in group)
        bound = sorted(group & centre_donors)
        if not disconnected and not bound:
            outer.append(fragment_smiles)
            continue
        hapto = any(ring <= set(bound) for ring in centre_rings)
        donors = () if hapto else tuple(sorted(rw.GetAtomWithIdx(idx).GetSymbol() for idx in bound))
        key = (Chem.CanonSmiles(fragment_smiles), charge, donors, hapto)
        inner[key] = inner.get(key, 0) + 1

    if not inner:
        raise UnsupportedMetalError(f"{centre_atom.GetSymbol()} has no ligands")
    if disconnected and not any(charge == 0 or _heavy_atoms(smiles_) > 1 for smiles_, charge, _, _ in inner):
        raise UnsupportedMetalError(
            f"{centre_atom.GetSymbol()} with only simple anions is a salt, not a coordination entity this module names"
        )

    ligands = tuple(
        Ligand(smiles=smiles_, charge=charge, count=count, donors=donors, hapto=hapto)
        for (smiles_, charge, donors, hapto), count in inner.items()
    )
    return MetalRoute(
        smiles=normalised,
        complex=Complex(
            metal=centre_atom.GetSymbol(),
            oxidation_state=centre_atom.GetFormalCharge(),
            ligands=ligands,
            other_components=".".join(sorted(outer)),
        ),
    )


def _heavy_atoms(smiles: str) -> int:
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    return mol.GetNumHeavyAtoms() if mol is not None else 0


def _ligand_table() -> dict[str, str]:
    from rdkit import Chem

    return {Chem.CanonSmiles(LIGAND_SMILES[word]): word for word in _PREFERRED_LIGAND_WORDS}


def _ligand_word(ligand: Ligand, name_smiles: Callable[[str], str]) -> tuple[str, bool]:
    """Return the ligand's name inside a complex and whether it is a simple word."""

    word = _ligand_table().get(ligand.smiles)
    if word is not None:
        if ligand.hapto:
            return f"η5-{word}", False
        return word, True

    name = name_smiles(ligand.smiles)
    if not name:
        raise UnsupportedMetalError(f"could not name the ligand {ligand.smiles}")
    if ligand.charge > 0:
        raise UnsupportedMetalError(f"cationic ligand {ligand.smiles} is not supported")
    if ligand.charge < 0:
        for ending, replacement in (("ate", "ato"), ("ide", "ido"), ("ite", "ito")):
            if name.endswith(ending):
                name = name[: -len(ending)] + replacement
                break
        else:
            raise UnsupportedMetalError(f"anionic ligand {ligand.smiles} has no anion name to cite ({name!r})")
    if ligand.donors:
        seen: dict[str, int] = {}
        cited = []
        for symbol in ligand.donors:
            cited.append(symbol + "'" * seen.get(symbol, 0))
            seen[symbol] = seen.get(symbol, 0) + 1
        count = str(len(cited)) if len(cited) > 1 else ""
        name = f"{name}-κ{count}{','.join(cited)}"
    return name, False


def _enclose(word: str) -> str:
    return f"[{word}]" if "(" in word or "[" in word else f"({word})"


def _stock_number(oxidation_state: int) -> str:
    magnitude = abs(oxidation_state)
    if magnitude >= len(_ROMAN):
        raise UnsupportedMetalError(f"oxidation state {oxidation_state:+d} is out of range")
    return ("-" if oxidation_state < 0 else "") + _ROMAN[magnitude]


def _metal_name(symbol: str, anionic: bool) -> str:
    from rdkit import Chem

    table = Chem.GetPeriodicTable()
    name = table.GetElementName(table.GetAtomicNumber(symbol)).lower()
    if not anionic:
        return name
    if name in ATE_NAMES:
        return ATE_NAMES[name]
    for ending in ("ium", "um"):
        if name.endswith(ending):
            return name[: -len(ending)] + "ate"
    return name + "ate"


def name_complex(complex_: Complex, name_smiles: Callable[[str], str]) -> str:
    """Additive name: ligands in alphabetical order, then the central atom and its oxidation number."""

    cited = []
    for ligand in complex_.ligands:
        word, simple = _ligand_word(ligand, name_smiles)
        sort_key = re.sub(r"^(?:[^A-Za-zη]*)(?:η\d-)?", "", word).lower()
        if ligand.count == 1:
            text = word if simple else _enclose(word)
        elif simple:
            text = f"{multipliers.basic(ligand.count)}{word}"
        else:
            text = f"{multipliers.complex_(ligand.count)}{_enclose(word)}"
        cited.append((sort_key, text))
    charge = complex_.charge
    entity = (
        "".join(text for _, text in sorted(cited))
        + _metal_name(complex_.metal, anionic=charge < 0)
        + f"({_stock_number(complex_.oxidation_state)})"
    )
    if not complex_.other_components:
        return entity
    others = name_smiles(complex_.other_components)
    if not others:
        raise UnsupportedMetalError(f"could not name {complex_.other_components} outside the coordination sphere")
    return f"{others} {entity}" if charge < 0 else f"{entity} {others}"


__all__ = [
    "Complex",
    "Ligand",
    "MetalRoute",
    "contains_transition_metal",
    "name_complex",
    "perceive",
    "read_molblock",
]
