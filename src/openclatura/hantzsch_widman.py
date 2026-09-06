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

from dataclasses import dataclass

from .molecule import Molecule
from .retained_graph_model import (
    RetainedGraphAtomTemplate,
    RetainedGraphBondTemplate,
    RetainedGraphTemplate,
)
from .rules import elements as _elements
from .rules import elision, multipliers

MIN_RING_SIZE = 3
MAX_RING_SIZE = 10

_FUSION_COMPONENT_KEY_PREFIX = "generated-hw:"


@dataclass(frozen=True, slots=True)
class HWFusionComponent:
    """One graph-derived, independently numbered HW fusion component.

    This is deliberately a value object rather than a registry entry.  Fusion
    matching may derive it from a face and discard it after planning; no global
    table grows with the molecules processed by the naming engine.
    """

    key: str
    parent_name: str
    attached_prefix: str
    template: RetainedGraphTemplate
    atom_to_locant: tuple[tuple[int, str], ...]
    multiplicative_prefix_style: str


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


def hw_fusion_component_key(ring_symbols: tuple[str, ...]) -> str:
    """Return a reversible internal key for a numbered HW component graph."""

    if not MIN_RING_SIZE <= len(ring_symbols) <= MAX_RING_SIZE:
        raise ValueError("HW fusion component ring size is outside the supported range")
    if not _fusion_symbols_supported(ring_symbols):
        raise ValueError("unsupported HW fusion component atom sequence")
    return f"{_FUSION_COMPONENT_KEY_PREFIX}{'.'.join(ring_symbols)}"


def hw_fusion_component_from_key(key: str) -> HWFusionComponent | None:
    """Reconstruct a generated fusion component without a mutable cache."""

    if not key.startswith(_FUSION_COMPONENT_KEY_PREFIX):
        return None
    symbols = tuple(key.removeprefix(_FUSION_COMPONENT_KEY_PREFIX).split("."))
    if not MIN_RING_SIZE <= len(symbols) <= MAX_RING_SIZE:
        return None
    if not _fusion_symbols_supported(symbols):
        return None
    return _hw_fusion_component(symbols, ())


def hw_fusion_components_for_ring(mol: Molecule, path: list[int]) -> tuple[HWFusionComponent, ...]:
    """Derive preferred local numberings for a neutral HW fusion face.

    The operation is linear in the ring size apart from the at-most ``2n``
    orientation comparison required by HW numbering.  It neither consults nor
    mutates the generated-parent name cache used by the general parent pipeline.

    A face may be the mancude component itself or a hydro derivative of it.
    The latter is accepted only when its observed double bonds are a subset of
    at least one valid maximum noncumulative mancude assignment.  The generated
    template remains mancude: the fusion parent's ordinary hydro-operation
    machinery records the missing double bonds after the component graphs have
    been merged and numbered.
    """

    if not MIN_RING_SIZE <= len(path) <= MAX_RING_SIZE or len(path) != len(set(path)):
        return ()
    if any(atom_idx not in mol.atoms for atom_idx in path):
        return ()
    if any(mol.get_bond(left, right) is None for left, right in zip(path, path[1:] + path[:1])):
        return ()
    atoms = tuple(mol.atoms[atom_idx] for atom_idx in path)
    if (
        all(atom.symbol == "C" for atom in atoms)
        or any(atom.charge or atom.isotope is not None for atom in atoms)
        or any(not atom.element.fusion_supported for atom in atoms)
        or any(
            atom.symbol != "C" and (atom.element.hw_stem is None or atom.element.hw_priority is None) for atom in atoms
        )
        or any(_needs_lambda(mol, atom.idx) for atom in atoms)
    ):
        return ()

    orders = _canonical_ring_orders(mol, path)
    if not orders:
        return ()
    symbols = tuple(mol.atoms[atom_idx].symbol for atom_idx in orders[0])
    expected_double_bonds = mancude_double_bonds(list(symbols))
    if expected_double_bonds == 0 or not _is_mancude_or_hydro_derivative(
        mol,
        path,
        expected_double_bonds=expected_double_bonds,
    ):
        return ()

    components = []
    for order in orders:
        atom_to_locant = tuple((atom_idx, str(locant)) for locant, atom_idx in enumerate(order, start=1))
        components.append(_hw_fusion_component(symbols, atom_to_locant))
    return tuple(components)


def _is_mancude_or_hydro_derivative(
    mol: Molecule,
    path: list[int],
    *,
    expected_double_bonds: int,
) -> bool:
    """Whether a ring's pi bonds can belong to its generated mancude parent.

    Merely accepting fewer double bonds is insufficient: it would also admit a
    double bond at a forced-single chalcogen or a pattern that no Kekule form of
    the parent can contain.  Fix the observed double bonds, remove their
    endpoints, and prove that the remaining eligible cycle vertices can still
    complete a maximum matching of the expected size.
    """

    actual_double_bonds = _ring_double_bonds(mol, path)
    if actual_double_bonds is None or actual_double_bonds == 0 or actual_double_bonds > expected_double_bonds:
        return False

    size = len(path)
    unavailable = [mol.atoms[atom_idx].symbol in FIXED_SATURATED for atom_idx in path]
    for position, (left, right) in enumerate(zip(path, path[1:] + path[:1])):
        bond = mol.get_bond(left, right)
        if bond is None or bond.order != 2:
            continue
        following = (position + 1) % size
        if unavailable[position] or unavailable[following]:
            return False
        unavailable[position] = True
        unavailable[following] = True

    available = tuple(not blocked for blocked in unavailable)
    return actual_double_bonds + _maximum_cycle_matching_size(available) == expected_double_bonds


def _maximum_cycle_matching_size(available: tuple[bool, ...]) -> int:
    """Maximum edge matching on a cycle after unavailable vertices are removed."""

    size = len(available)
    if not size or not any(available):
        return 0
    if all(available):
        return size // 2

    # Starting immediately after a blocked vertex turns the residual cycle into
    # independent paths.  A path of n vertices contributes floor(n / 2) edges.
    blocked = available.index(False)
    result = 0
    run_length = 0
    for offset in range(1, size + 1):
        if available[(blocked + offset) % size]:
            run_length += 1
        else:
            result += run_length // 2
            run_length = 0
    return result + run_length // 2


def _hw_fusion_component(
    symbols: tuple[str, ...],
    atom_to_locant: tuple[tuple[int, str], ...],
) -> HWFusionComponent:
    hetero = [(locant, symbol) for locant, symbol in enumerate(symbols, start=1) if symbol != "C"]
    ordinary_name = hw_name(len(symbols), False, hetero)
    if ordinary_name is None:
        raise ValueError("invalid generated HW fusion component")
    parent_name, attached_prefix = _hw_fusion_forms(ordinary_name)
    key = hw_fusion_component_key(symbols)
    locants = tuple(str(locant) for locant in range(1, len(symbols) + 1))
    edges = tuple(zip(locants, locants[1:] + locants[:1], strict=True))
    template = RetainedGraphTemplate(
        name=key,
        pin=True,
        priority=1000,
        aliases=(),
        attached_prefix=attached_prefix,
        derivative_stem=None,
        default_indicated_h=(),
        locants=locants,
        atoms=tuple(
            RetainedGraphAtomTemplate(
                locant=locant,
                symbol=symbol,
                aromatic=symbol == "C",
            )
            for locant, symbol in zip(locants, symbols, strict=True)
        ),
        bonds=tuple(RetainedGraphBondTemplate(locants=edge, bond_class="mancude") for edge in edges),
        rings=(locants,),
        fusion_atoms=(),
        peripheral_atoms=locants,
        interior_atoms=(),
        family="generated_hw_monocycle",
        numbering_policy="hantzsch_widman",
        aromatic_equivalence_policy="neutral_kekule_equivalent",
        charge_policy="exact",
        enforce_mancude_double_bonds=True,
        enabled=True,
        pre_descriptor_selection=True,
        mancude_double_bonds=mancude_double_bonds(list(symbols)),
    )
    return HWFusionComponent(
        key,
        parent_name,
        attached_prefix,
        template,
        atom_to_locant,
        "complex" if len(hetero) > 1 else "basic",
    )


def _fusion_symbols_supported(symbols: tuple[str, ...]) -> bool:
    if not symbols or all(symbol == "C" for symbol in symbols):
        return False
    for symbol in symbols:
        if symbol not in _elements.ELEMENTS:
            return False
        element = _elements.get(symbol)
        if not element.fusion_supported:
            return False
        if symbol != "C" and (element.hw_stem is None or element.hw_priority is None):
            return False
    return True


def _hw_fusion_forms(name: str) -> tuple[str, str]:
    """Return P-25 parent and attached forms of a generated HW name."""

    locants = ""
    base = name
    if "-" in name:
        candidate, remainder = name.split("-", 1)
        if all(part.isdigit() for part in candidate.split(",")):
            locants = candidate
            base = remainder
    parent = f"[{locants}]{base}" if locants else base
    attached_base = f"{base[:-1]}o" if base.endswith("e") else f"{base}o"
    attached = f"[{locants}]{attached_base}" if locants else attached_base
    return parent, attached


# Divalent at their normal valence, so they take no ring double bond and cut the
# mancude system into segments: 1,2,3,5-oxathiadiazole has one, not two.
FIXED_SATURATED = tuple(element.symbol for element in _elements.ELEMENTS.values() if element.mancude_forced_single)


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

    orders = _canonical_ring_orders(mol, path)
    if not orders:
        return None
    return [
        (locant, mol.atoms[atom_idx].symbol)
        for locant, atom_idx in enumerate(orders[0], start=1)
        if mol.atoms[atom_idx].symbol != "C"
    ]


def _canonical_ring_orders(mol: Molecule, path: list[int]) -> tuple[tuple[int, ...], ...]:
    """Return every ring orientation tied by the HW numbering criteria."""

    size = len(path)
    symbols = [mol.atoms[idx].symbol for idx in path]
    if not any(symbol != "C" for symbol in symbols):
        return ()
    priorities = [_elements.get(symbol).hw_priority if symbol != "C" else None for symbol in symbols]
    if any(symbol != "C" and priority is None for symbol, priority in zip(symbols, priorities)):
        return ()

    candidates: list[tuple[tuple, tuple[int, ...]]] = []
    for offset in range(size):
        for step in (1, -1):
            atom_order = tuple(path[(offset + step * index) % size] for index in range(size))
            hetero = [
                (locant, mol.atoms[atom_idx].symbol)
                for locant, atom_idx in enumerate(atom_order, start=1)
                if mol.atoms[atom_idx].symbol != "C"
            ]
            first_priority = _elements.get(hetero[0][1]).hw_priority
            if first_priority != min(_elements.get(symbol).hw_priority for _, symbol in hetero):
                continue
            by_seniority = tuple(sorted((_elements.get(symbol).hw_priority, locant) for locant, symbol in hetero))
            key = (tuple(locant for locant, _ in hetero), by_seniority)
            candidates.append((key, atom_order))
    if not candidates:
        return ()
    best_key = min(key for key, _ in candidates)
    return tuple(dict.fromkeys(order for key, order in candidates if key == best_key))


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
