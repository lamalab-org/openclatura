"""Hantzsch-Widman monocycle names, derived from the ring graph.

Most of what a naming engine calls a "retained" heterocycle is not retained at
all: `oxolane`, `1,2,4-oxadiazolidine`, `selenophene` and `1,3,2-dioxaborolane`
are all spelled out by one rule -- name the heteroatoms in seniority order, then
add the stem for the ring size and saturation.  A table of such names only ever
covers the rings someone thought to type in, which is why an `azepane` or a
`1,3,5,2,4,6-trioxatriborinane` used to fall back to replacement nomenclature.

This module builds the name instead, and reports the parent's shape (size and
mancude double-bond count) so the indicated-hydrogen and hydro-prefix machinery
can treat a generated parent exactly like a tabulated one.  Genuinely retained
names -- pyridine, furan, morpholine -- stay in the table and are matched first.
"""

from .molecule import Molecule
from .rules import elements as _elements
from .rules import elision, multipliers

MIN_RING_SIZE = 3
MAX_RING_SIZE = 10

# P-22.2.2.1: the three six-membered stem sets, keyed by the class of the least
# senior heteroatom in the ring -- boron makes trioxatriborinane, not -borane.
_SIX_RING_CLASSES = {
    "A": frozenset({"O", "S", "Se", "Te", "Bi"}),
    "B": frozenset({"N", "Si", "Ge", "Sn", "Pb"}),
    "C": frozenset({"B", "F", "Cl", "Br", "I", "P", "As", "Sb"}),
}

_SIX_RING_STEMS = {"A": ("ine", "ane"), "B": ("ine", "inane"), "C": ("inine", "inane")}

# (mancude, saturated) stems by ring size; sizes 3-5 pick by nitrogen presence.
_NITROGEN_STEMS = {3: ("irine", "iridine"), 4: ("ete", "etidine"), 5: ("ole", "olidine")}
_CARBON_STEMS = {3: ("irene", "irane"), 4: ("ete", "etane"), 5: ("ole", "olane")}
_LARGE_STEMS = {
    7: ("epine", "epane"),
    8: ("ocine", "ocane"),
    9: ("onine", "onane"),
    10: ("ecine", "ecane"),
}


def hw_stem(size: int, saturated: bool, symbols: list[str]) -> str | None:
    """The Hantzsch-Widman ring stem for a size, saturation state and atom set."""

    index = 1 if saturated else 0
    if size in _NITROGEN_STEMS:
        table = _NITROGEN_STEMS if "N" in symbols else _CARBON_STEMS
        return table[size][index]
    if size == 6:
        hetero = [symbol for symbol in symbols if symbol != "C"]
        if not hetero:
            return None
        least_senior = max(hetero, key=lambda symbol: _elements.get(symbol).hw_priority or 0)
        ring_class = next((key for key, members in _SIX_RING_CLASSES.items() if least_senior in members), None)
        return _SIX_RING_STEMS[ring_class][index] if ring_class else None
    stems = _LARGE_STEMS.get(size)
    return stems[index] if stems else None


def _join(parts: list[str]) -> str:
    result = parts[0]
    for part in parts[1:]:
        result = elision.elide_terminal_a(result, part)
    return result


def _multiplied(prefix: str, count: int) -> str:
    if count == 1:
        return prefix
    return elision.elide_terminal_a(multipliers.basic(count), prefix)


def hw_name(size: int, saturated: bool, hetero: list[tuple[int, str]]) -> str | None:
    """Spell a Hantzsch-Widman name from ``(locant, symbol)`` heteroatom pairs.

    Heteroatoms are cited in seniority order and all their locants are collected
    into one front-of-name list, which is what makes `1,3,2-dioxaborolane` read
    the way it does rather than as two separately located prefixes.
    """

    if not hetero:
        return None
    symbols = [symbol for _, symbol in hetero]
    stem = hw_stem(size, saturated, symbols)
    if stem is None:
        return None

    groups: dict[int, list[tuple[int, str]]] = {}
    for locant, symbol in hetero:
        element = _elements.get(symbol)
        if element.hw_stem is None or element.hw_priority is None:
            return None
        groups.setdefault(element.hw_priority, []).append((locant, element.hw_stem))

    locants: list[int] = []
    prefixes: list[str] = []
    for priority in sorted(groups):
        group = sorted(groups[priority])
        locants.extend(locant for locant, _ in group)
        prefixes.append(_multiplied(group[0][1], len(group)))

    parent = _join([*prefixes, stem])
    if len(hetero) == 1 and locants == [1]:
        return parent
    return f"{','.join(str(locant) for locant in locants)}-{parent}"


# Divalent at their normal valence, so they take no ring double bond and cut the
# mancude system into segments: 1,2,3,5-oxathiadiazole has one, not two.
FIXED_SATURATED = ("O", "S", "Se", "Te")


def mancude_bond_orders(symbols: list[str]) -> list[int]:
    """Ring bond orders of the mancude parent; entry ``i`` joins atom ``i`` to ``i+1``.

    As many noncumulative double bonds as will fit.  A chalcogen cannot carry
    one, so it splits the cycle into open runs, and each run is filled by pairing
    off its atoms from the start.  With no chalcogen the run is the whole cycle,
    which is why the pairing begins at atom 0 there.
    """

    size = len(symbols)
    orders = [1] * size
    fixed = [pos for pos, symbol in enumerate(symbols) if symbol in FIXED_SATURATED]
    start = fixed[0] if fixed else size - 1
    run: list[int] = []
    for offset in range(1, size + 1):
        pos = (start + offset) % size
        if symbols[pos] in FIXED_SATURATED:
            _pair_run(orders, run, size)
            run = []
        else:
            run.append(pos)
    _pair_run(orders, run, size)
    return orders


def _pair_run(orders: list[int], run: list[int], size: int) -> None:
    for first in range(0, len(run) - 1, 2):
        a, b = run[first], run[first + 1]
        orders[a if (a + 1) % size == b else b] = 2


def mancude_double_bonds(symbols: list[str]) -> int:
    return sum(order == 2 for order in mancude_bond_orders(symbols))


def cites_indicated_hydrogen(size: int, double_bonds: int, symbols: list[str]) -> bool:
    """Whether a mancude monocycle of this shape spells its own indicated hydrogen.

    An odd mancude ring has one position left over.  It is an indicated hydrogen
    only when a carbon or a trivalent heteroatom holds it -- in oxepine the
    leftover is the oxygen, which is single-bonded there in any case.
    """

    if not double_bonds:
        return False
    fixed = sum(1 for symbol in symbols if symbol in FIXED_SATURATED)
    return size - 2 * double_bonds - fixed > 0


def _canonical_hetero(mol: Molecule, path: list[int]) -> list[tuple[int, str]] | None:
    """Heteroatoms as ``(locant, symbol)`` under the ring's own HW numbering.

    The ring arrives in perception order, not locant order, so the seniority
    numbering has to be settled here: the most senior heteroatom takes 1, then
    the heteroatom set as a whole takes the lowest locants, then the individual
    elements do, in seniority order.  Numbering the parent again downstream lands
    on the same answer, so the spelling and the locant map agree.
    """

    size = len(path)
    symbols = [mol.atoms[idx].symbol for idx in path]
    if not any(symbol != "C" for symbol in symbols):
        return None
    priorities = [_elements.get(symbol).hw_priority if symbol != "C" else None for symbol in symbols]
    if any(symbol != "C" and priority is None for symbol, priority in zip(symbols, priorities)):
        return None

    best = None
    for offset in range(size):
        for step in (1, -1):
            order = [symbols[(offset + step * i) % size] for i in range(size)]
            hetero = [(pos, symbol) for pos, symbol in enumerate(order, start=1) if symbol != "C"]
            if _elements.get(hetero[0][1]).hw_priority != min(
                _elements.get(symbol).hw_priority for _, symbol in hetero
            ):
                continue
            by_seniority = tuple(sorted((_elements.get(symbol).hw_priority, locant) for locant, symbol in hetero))
            key = (tuple(locant for locant, _ in hetero), by_seniority)
            if best is None or key < best[0]:
                best = (key, hetero)
    return best[1] if best else None


def hw_spec_for_ring(mol: Molecule, path: list[int]) -> dict | None:
    """A monocycle spec for the ring's Hantzsch-Widman parent, or None.

    The spec describes the *parent* -- mancude or fully saturated -- rather than
    the ring handed in, so a partly saturated ring reports the mancude parent it
    is a hydro form of and the hydro machinery downstream can take it from there.
    """

    size = len(path)
    if not MIN_RING_SIZE <= size <= MAX_RING_SIZE:
        return None
    if any(_needs_lambda(mol, idx) for idx in path):
        return None
    hetero = _canonical_hetero(mol, path)
    if hetero is None:
        return None

    double_bonds = _ring_double_bonds(mol, path)
    if double_bonds is None:
        return None
    saturated = double_bonds == 0
    name = hw_name(size, saturated, hetero)
    if name is None:
        return None

    ordered = ["C"] * size
    counts: dict[str, int] = {}
    for locant, symbol in hetero:
        ordered[locant - 1] = symbol
        counts[symbol] = counts.get(symbol, 0) + 1
    spec = {
        "name": name,
        "size": size,
        "double_bonds": 0 if saturated else mancude_double_bonds(ordered),
        "symbols": counts,
        "ring_symbols": ordered,
        "generated": True,
    }
    _SPECS_BY_NAME[name] = spec
    return spec


# Name -> spec for every parent this module has spelled.  The hydro and
# indicated-hydrogen steps look a parent up by name alone, and a generated name
# is in no table; since the mapping is a pure function of the name, memoising
# what was spelled is enough to answer them.
_SPECS_BY_NAME: dict[str, dict] = {}


def hw_spec_for_name(name: str) -> dict | None:
    return _SPECS_BY_NAME.get(name)


def hw_generated_names() -> list[str]:
    """Every Hantzsch-Widman parent name spelled so far, for stem matching."""

    return list(_SPECS_BY_NAME)


def hw_parent_template(name: str) -> tuple[str, list[str]] | None:
    """A SMILES skeleton and its locant labels, for the audit to rebuild from.

    The audit reconstructs a parent from a template, and a generated name has
    none written down -- so it is built from the same spec that produced the
    name.  A mancude parent gets one valid Kekule placement; the audit moves the
    double bonds itself when the molecule's indicated hydrogen sits elsewhere.
    """

    spec = _SPECS_BY_NAME.get(name)
    if spec is None:
        return None
    symbols = spec["ring_symbols"]
    size = len(symbols)
    orders = mancude_bond_orders(symbols) if spec["double_bonds"] else [1] * size

    smiles = f"{symbols[0]}1"
    for pos in range(1, size):
        smiles += ("=" if orders[pos - 1] == 2 else "") + symbols[pos]
    if orders[size - 1] == 2:
        return None
    smiles += "1"
    return smiles, [str(pos + 1) for pos in range(size)]


def spec_cites_indicated_hydrogen(spec: dict) -> bool:
    """Apply :func:`cites_indicated_hydrogen` to a monocycle spec's shape."""

    symbols = [symbol for symbol, count in spec.get("symbols", {}).items() for _ in range(count)]
    return cites_indicated_hydrogen(spec["size"], spec.get("double_bonds", 0), symbols)


def _needs_lambda(mol: Molecule, atom_idx: int) -> bool:
    """Whether the atom's bonding number departs from its standard valence.

    A Hantzsch-Widman stem has nowhere to hang a lambda, so a ring holding a
    lambda^4 sulfur or lambda^5 phosphorus is left to replacement nomenclature,
    which can say it.
    """

    atom = mol.atoms[atom_idx]
    if atom.is_carbon or atom.charge:
        return False
    bonding = atom.total_h_count or 0
    for neighbor in mol.get_neighbors(atom_idx):
        bond = mol.get_bond(atom_idx, neighbor)
        if bond is None:
            continue
        # An exocyclic oxo is cited as a `dioxo` prefix rather than raising the
        # skeletal atom's lambda, so a sulfone keeps its ring name.
        if bond.order == 2 and mol.atoms[neighbor].symbol == "O":
            continue
        bonding += bond.order
    return bonding != atom.element.standard_valence


def _ring_double_bonds(mol: Molecule, path: list[int]) -> int | None:
    """Ring double bonds, or None when a bond order no HW stem can say is present.

    A ring triple bond has no Hantzsch-Widman spelling -- `azacyclohept-4-yne`
    says it and `azepane` would silently drop it -- so such a ring is refused
    outright rather than counted as saturated.
    """

    total = 0
    for a, b in zip(path, path[1:] + path[:1]):
        bond = mol.get_bond(a, b)
        if bond is None or bond.order not in (1, 2):
            return None
        total += bond.order == 2
    return total
