"""Independent von Baeyer ``bicyclo[a.b.c]`` / ``spiro[a.b]`` name → graph parser.

openclatura represents most fused bicyclic ring systems (including benzo-fused
heterocycles, drawn as fully-unsaturated von Baeyer polyenes) with von Baeyer
nomenclature.  Reconstructing them soundly is the single biggest coverage lever
after the substituent grammar, so this module parses the descriptor, stem,
skeletal replacement (``oxa``/``aza``/…) and unsaturation locants back into a
numbered ring skeleton — entirely from the name, never the input graph.

The two-ring ``bicyclo`` case and monospiro ``spiro[a.b]`` are modelled (both
with skeletal replacement and unsaturation), as are monocycles that state their
own heteroatoms — the ``1-azacyclohexane`` replacement form and the contracted
Hantzsch-Widman one (``1,4-dioxane``, ``azepane``).  Polyspiro (``dispiro``…),
bridges carrying their own atoms, and anything that does not parse cleanly
return ``None`` so the caller abstains.
"""

from __future__ import annotations

import re

from rdkit import Chem

from ..rules import elements as _elements
from ..rules import multipliers as _multipliers
from ..rules import stems as _stems

Numbered = tuple[Chem.RWMol, dict[str, int], int]

# Skeletal-replacement prefixes, read back off the same table the namer writes
# them from, so the parser accepts exactly the set the namer can emit.
_REPLACEMENT_ELEMENTS = _elements.SYMBOLS_BY_HW_STEM
# ``en``/``yn`` unsaturation stems carry the multiplier as a *basic* prefix
# (``dien``, ``trien``), so they are derived from the shared table rather than
# spelled out again here.
_UNSAT_MULT: dict[str, int] = {"en": 1, "yn": 1} | {
    f"{mult.basic}{stem}": mult.count for mult in _multipliers.MULTIPLIERS.values() for stem in ("en", "yn")
}
# Longest first so ``dien`` is not read as ``di`` + a stray ``en``.
_UNSAT_ALTERNATION = "|".join(sorted(_UNSAT_MULT, key=len, reverse=True))

_VONBAEYER_RE = re.compile(r"(?:bi|tri|tetra|penta)cyclo\[([0-9.,^{}]+)\]")
_SECONDARY_RE = re.compile(r"(\d+)\^?\{?(\d+),(\d+)\}?")
# A replacement clause may carry a lambda-convention valence on its locant —
# ``1lambda^6-thia`` is still just "sulfur at position 1"; the non-standard
# valence is realised by the ``dioxo`` prefix the caller grafts, so the
# annotation is matched and discarded rather than being left as an unparsed
# leftover that forces an abstention.
_LAMBDA = r"(?:lambda\^?\{?\d+\}?)?"
_BASIC_ALTERNATION = "|".join(sorted((mult.basic for mult in _multipliers.MULTIPLIERS.values()), key=len, reverse=True))
_REPL_RE = re.compile(
    r"(\d+(?:,\d+)*)"
    + _LAMBDA
    + r"-("
    + _BASIC_ALTERNATION
    + r")?("
    # longest first, so a prefix that starts another one cannot shadow it
    + "|".join(sorted(_REPLACEMENT_ELEMENTS, key=len, reverse=True))
    + r")"
)

# A parsed von Baeyer descriptor: the main bicycle (a,b,c) plus zero or more
# secondary bridges (length, bridgehead-f, bridgehead-g).
Descriptor = tuple[int, int, int, list[tuple[int, str, str]]]


def _parse_descriptor(inner: str) -> Descriptor | None:
    tokens = inner.split(".")
    if len(tokens) < 3:
        return None
    try:
        a, b, c = int(tokens[0]), int(tokens[1]), int(tokens[2])
    except ValueError:
        return None
    secondary: list[tuple[int, str, str]] = []
    for tok in tokens[3:]:
        m = _SECONDARY_RE.fullmatch(tok)
        if m is None:
            return None
        secondary.append((int(m.group(1)), m.group(2), m.group(3)))
    return a, b, c, secondary


def parse_von_baeyer(name: str) -> Numbered | None:
    """Parse a full ``[replacement](bi|tri…)cyclo[…]stem[unsat]-<n>-yl``
    substituent (or the ``…-<n>-ylidene`` / bare parent form)."""

    m = _VONBAEYER_RE.search(name)
    if m is None:
        return None
    descriptor = _parse_descriptor(m.group(1))
    if descriptor is None:
        return None
    a, b, c, secondary = descriptor
    pre, post = name[: m.start()], name[m.end() :]

    total = a + b + c + 2 + sum(length for length, _, _ in secondary)
    rw, locants = _build_skeleton(a, b, c, secondary)
    if rw is None:
        return None

    if not _apply_replacement(rw, locants, pre):
        return None

    attach = _split_and_apply_post(rw, locants, post, total)
    if attach is None:
        return None
    return rw, locants, attach


def build_skeleton(a: int, b: int, c: int) -> tuple[Chem.RWMol | None, dict[str, int]]:
    """Public: build a saturated von Baeyer ``bicyclo[a.b.c]`` carbon skeleton
    with numeric locant labels, for reuse by parent reconstruction."""
    return _build_skeleton(a, b, c)


def build_skeleton_from_descriptor(descriptor: str) -> tuple[Chem.RWMol | None, dict[str, int]]:
    """Public: build a saturated skeleton from a full ``…cyclo[…]`` descriptor
    string (mono-, bi-, or polycyclic), for parent reconstruction."""
    m = _VONBAEYER_RE.search(descriptor)
    if m is None:
        return None, {}
    parsed = _parse_descriptor(m.group(1))
    if parsed is None:
        return None, {}
    a, b, c, secondary = parsed
    return _build_skeleton(a, b, c, secondary)


def _build_skeleton(
    a: int, b: int, c: int, secondary: list[tuple[int, str, str]] = ()
) -> tuple[Chem.RWMol | None, dict[str, int]]:
    if a < b or b < c or b == 0:  # a>=b>=c>=0 and at least two real bridges
        return None, {}
    n = a + b + c + 2
    rw = Chem.RWMol()
    idx = {str(i): rw.AddAtom(Chem.Atom(6)) for i in range(1, n + 1)}

    def bond(i: int, j: int) -> None:
        rw.AddBond(idx[str(i)], idx[str(j)], Chem.BondType.SINGLE)

    bh1, bh2 = 1, a + 2
    # main bridge: 1-2-...-(a+1)-(a+2)
    for i in range(1, a + 2):
        bond(i, i + 1)
    # second bridge: (a+2)-(a+3)-...-(a+b+2)-1
    prev = bh2
    for i in range(a + 3, a + b + 3):
        bond(prev, i)
        prev = i
    bond(prev, bh1)
    # third bridge
    if c == 0:
        bond(bh1, bh2)
    else:
        prev = bh1
        for i in range(a + b + 3, a + b + c + 3):
            bond(prev, i)
            prev = i
        bond(prev, bh2)
    # Secondary bridges, numbered after the main bicycle in citation order
    # (P-23.2.5.1).  A length-0 bridge is a direct bond between two skeleton atoms
    # (the usual fused-aromatic case); a longer one contributes its own atoms,
    # numbered from the end attached to the *higher*-numbered bridgehead.
    next_locant = n + 1
    for length, f, g in secondary:
        if f not in idx or g not in idx:
            return None, {}
        if length == 0:
            if rw.GetBondBetweenAtoms(idx[f], idx[g]) is not None:
                return None, {}
            rw.AddBond(idx[f], idx[g], Chem.BondType.SINGLE)
            continue
        high, low = (f, g) if int(f) > int(g) else (g, f)
        previous = idx[high]
        for _ in range(length):
            atom = rw.AddAtom(Chem.Atom(6))
            idx[str(next_locant)] = atom
            rw.AddBond(previous, atom, Chem.BondType.SINGLE)
            previous = atom
            next_locant += 1
        rw.AddBond(previous, idx[low], Chem.BondType.SINGLE)
    return rw, idx


# --------------------------------------------------------------------------- #
# Monospiro
# --------------------------------------------------------------------------- #
_SPIRO_RE = re.compile(r"spiro\[(\d+)\.(\d+)\]")
# Polyspiro descriptors reuse the ``spiro[`` token with a different numbering
# rule, so they must not be parsed as monospiro.
_POLYSPIRO_RE = re.compile(r"(?:di|tri|tetra|penta)spiro\[")


def parse_spiro(name: str) -> Numbered | None:
    """Parse a full ``[replacement]spiro[a.b]stem[unsat]-<n>-yl`` substituent
    (``2-azaspiro[3.3]heptan-2-yl``, ``5-oxaspiro[3.4]octan-7-yl``)."""

    if _POLYSPIRO_RE.search(name):
        return None
    m = _SPIRO_RE.search(name)
    if m is None:
        return None
    a, b = int(m.group(1)), int(m.group(2))
    pre, post = name[: m.start()], name[m.end() :]

    rw, locants = _build_spiro_skeleton(a, b)
    if rw is None:
        return None
    if not _apply_replacement(rw, locants, pre):
        return None
    attach = _split_and_apply_post(rw, locants, post, a + b + 1)
    if attach is None:
        return None
    return rw, locants, attach


def build_spiro_skeleton(a: int, b: int) -> tuple[Chem.RWMol | None, dict[str, int]]:
    """Public: build a saturated monospiro ``spiro[a.b]`` carbon skeleton with
    numeric locant labels, for reuse by parent reconstruction."""
    return _build_spiro_skeleton(a, b)


def _build_spiro_skeleton(a: int, b: int) -> tuple[Chem.RWMol | None, dict[str, int]]:
    """Number a monospiro system per P-24.2.4.1: start in the *smaller* ring at an
    atom next to the spiro atom, go round that ring — so the spiro atom itself is
    ``a+1`` — then continue through the larger ring back to the spiro atom.

    ``spiro[3.3]heptane`` therefore closes 1-2-3-4(spiro)-1 and 4-5-6-7-4.  The
    descriptor is written smaller-ring-first, so ``a > b`` is a name we do not
    model rather than one we renumber."""

    if a < 2 or b < a:  # each ring needs >= 3 atoms, smaller bridge cited first
        return None, {}
    n = a + b + 1
    rw = Chem.RWMol()
    idx = {str(i): rw.AddAtom(Chem.Atom(6)) for i in range(1, n + 1)}
    spiro = a + 1

    def bond(i: int, j: int) -> None:
        rw.AddBond(idx[str(i)], idx[str(j)], Chem.BondType.SINGLE)

    for i in range(1, spiro):  # 1-2-…-spiro
        bond(i, i + 1)
    bond(spiro, 1)  # close the smaller ring
    prev = spiro
    for i in range(spiro + 1, n + 1):  # spiro-…-n
        bond(prev, i)
        prev = i
    bond(n, spiro)  # close the larger ring
    return rw, idx


def _apply_replacement(rw: Chem.RWMol, locants: dict[str, int], pre: str) -> bool:
    if not pre:
        return True
    consumed = 0
    for mm in _REPL_RE.finditer(pre):
        locs = mm.group(1).split(",")
        mult = _multipliers.count_for(mm.group(2) or "") or 1
        if len(locs) != mult:
            return False
        element = _REPLACEMENT_ELEMENTS[mm.group(3)]
        for loc in locs:
            if loc not in locants:
                return False
            rw.GetAtomWithIdx(locants[loc]).SetAtomicNum(Chem.Atom(element).GetAtomicNum())
        consumed += mm.end() - mm.start()
    # Anything in the prefix we did not consume (odd connectives excluded) -> abstain.
    leftover = _REPL_RE.sub("", pre).replace("-", "")
    return leftover == ""


def _split_and_apply_post(rw: Chem.RWMol, locants: dict[str, int], post: str, total: int) -> int | None:
    # attachment
    ym = re.search(r"-(\d+)-yl(?:idene)?$", post)
    if ym is None:
        return None
    attach = ym.group(1)
    core = post[: ym.start()]
    if attach not in locants:
        return None

    # stem
    sm = re.match(r"^([a-z]+?)(?=$|-|\d)", core)
    if sm is None:
        return None
    stem_word = sm.group(1)
    rest = core[sm.end() :]
    if not _check_stem(stem_word, total):
        return None

    # unsaturation: "-1(6),2,4-trien" style, appearing before the attachment
    if rest:
        if not _apply_unsaturation(rw, locants, rest):
            return None
    return locants[attach]


def _check_stem(stem_word: str, total: int) -> bool:
    return _stem_length(stem_word) == total


def _stem_length(stem_word: str) -> int | None:
    """Ring size for a multiplying stem, with or without its connective.

    The candidates are tried longest-form first — ``nonan`` -> ``non``, ``nona``
    -> ``non`` — but the bare stem is tried *before* any stripping, so a stem that
    itself ends in the connective letter survives: ``non`` (as in
    ``bicyclo[3.3.1]non-6-ene``) is nine carbons, not ``no``."""

    candidates = (stem_word, stem_word.removesuffix("an"), stem_word.removesuffix("a"), stem_word.removesuffix("n"))
    for word in candidates:
        length = next((ln for ln, row in _stems.STEMS.items() if row.stem == word), None)
        if length is not None:
            return length
    return None


_MONO_ATTACH_RE = re.compile(r"-(\d+)-yl(?:idene)?$")


def parse_monocyclic_replacement(name: str) -> Numbered | None:
    """Parse a Hantzsch-Widman replacement *monocycle* substituent —
    ``1-azacycloheptan-1-yl``, ``1-thia-3,4-diazacyclopenta-2,4-dien-2-yl``,
    ``1-azacyclohexa-3,5-dien-3-yl`` — into a numbered ring fragment.

    Front modifiers (``oxo``/``methyl``/…) are handled by the caller's prefix
    machinery; here we consume only the replacement prefix, the ring-size stem,
    the unsaturation and the ``-<n>-yl`` attachment.  The bracketed von Baeyer
    form is left to :func:`parse_von_baeyer`; anything that does not parse cleanly
    returns ``None`` so the caller abstains."""

    if "cyclo[" in name:  # bracketed → von Baeyer, handled separately
        return None
    m = _MONO_ATTACH_RE.search(name)
    if m is None:
        return None
    attach = m.group(1)
    pre, sep, rest = name[: m.start()].partition("cyclo")
    if not sep:
        return None
    # A monocyclic *replacement* name must carry at least one replacement clause;
    # without one it is an ordinary carbocycle handled elsewhere.
    if _REPL_RE.search(pre) is None:
        return None
    sm = re.match(r"^[a-z]+", rest)
    if sm is None:
        return None
    n = _stem_length(sm.group(0))
    unsat = rest[sm.end() :]
    if n is None or n < 3:
        return None
    rw, locants = _build_monocycle(n)
    if attach not in locants:
        return None
    if not _apply_replacement(rw, locants, pre):
        return None
    if unsat and not _apply_unsaturation(rw, locants, unsat):
        return None
    return rw, locants, locants[attach]


# --------------------------------------------------------------------------- #
# Hantzsch-Widman contracted monocycles
# --------------------------------------------------------------------------- #
# Heteroatom prefixes in their elided form — the ``-a`` is dropped before the
# vowel of the size stem, so ``oxa`` + ``ane`` reads ``oxane``.
_HW_ELEMENTS = {"ox": "O", "thi": "S", "az": "N", "sel": "Se", "tellur": "Te"}
# Saturated size stems, longest first so ``olane`` is not read as ``ane``.
# Mancude (unsaturated) rings — ``oxazole``, ``thiazole`` — keep their retained
# spellings elsewhere; only the saturated series is generated here.
_HW_SIZE_STEMS: tuple[tuple[str, int], ...] = (
    ("iridine", 3),
    ("irane", 3),
    ("etidine", 4),
    ("etane", 4),
    ("olidine", 5),
    ("olane", 5),
    ("inane", 6),
    ("epane", 7),
    ("ocane", 8),
    ("onane", 9),
    ("ecane", 10),
    ("ane", 6),
)
_HW_RE = re.compile(r"^(?:(?P<locants>\d+(?:,\d+)*)-)?(?P<prefix>[a-z]+)$")


def parse_hantzsch_widman(name: str) -> Numbered | None:
    """Parse a contracted saturated Hantzsch-Widman monocycle substituent —
    ``1,4-dioxan-2-yl``, ``thian-4-yl``, ``azepan-1-yl``, ``1,3-dithiolan-2-yl``.

    These say their own ring size and heteroatom placement, so the ring is
    derived from the name's morphology rather than looked up.  Names that do not
    parse cleanly return ``None`` so the caller abstains."""

    m = _MONO_ATTACH_RE.search(name)
    if m is None:
        return None
    attach = m.group(1)
    head = _HW_RE.match(name[: m.start()])
    if head is None:
        return None
    for stem, size in _HW_SIZE_STEMS:
        # ``-an-2-yl`` drops the stem's final ``e`` before the locant.
        body = head.group("prefix")
        for spelling in (stem, stem[:-1]):
            if body.endswith(spelling) and len(body) > len(spelling):
                elements = _parse_hw_prefix(body[: -len(spelling)])
                if elements is None:
                    break
                return _build_hw_ring(size, elements, head.group("locants"), attach, stem)
    return None


def _parse_hw_prefix(prefix: str) -> list[str] | None:
    """Expand ``diox`` -> ``["O", "O"]``, ``thiaz`` -> ``["S", "N"]``, in citation
    order.  ``None`` if any token is not a heteroatom prefix."""

    elements: list[str] = []
    while prefix:
        # Multiplied readings first (longest prefix wins), then the bare token —
        # ``oxa`` is one oxygen, ``dioxa`` two.
        for count, rest in (*_multipliers.candidate_splits(prefix), (1, prefix)):
            for word in sorted(_HW_ELEMENTS, key=len, reverse=True):
                if rest.startswith(word):
                    elements.extend([_HW_ELEMENTS[word]] * count)
                    prefix = rest[len(word) :].lstrip("a-")
                    break
            else:
                continue
            break
        else:
            return None
    return elements or None


def _build_hw_ring(size: int, elements: list[str], locants: str | None, attach: str, stem: str) -> Numbered | None:
    """Place ``elements`` at the cited locants (or at position 1 for a lone
    heteroatom) around a ring of ``size`` atoms."""

    if size < 3:
        return None
    # A six-membered ring containing nitrogen takes ``-inane``; plain ``-ane``
    # there would be the parent hydride ``azane``, not a ring at all.
    if "N" in elements and stem == "ane":
        return None
    positions = locants.split(",") if locants else ["1"]
    if len(positions) != len(elements):
        return None
    rw, ring_locants = _build_monocycle(size)
    for position, element in zip(positions, elements):
        if position not in ring_locants:
            return None
        rw.GetAtomWithIdx(ring_locants[position]).SetAtomicNum(Chem.Atom(element).GetAtomicNum())
    if attach not in ring_locants:
        return None
    return rw, ring_locants, ring_locants[attach]


def _build_monocycle(n: int) -> tuple[Chem.RWMol, dict[str, int]]:
    rw = Chem.RWMol()
    idx = {str(i): rw.AddAtom(Chem.Atom(6)) for i in range(1, n + 1)}
    for i in range(1, n):
        rw.AddBond(idx[str(i)], idx[str(i + 1)], Chem.BondType.SINGLE)
    rw.AddBond(idx[str(n)], idx[str(1)], Chem.BondType.SINGLE)
    return rw, idx


def _apply_unsaturation(rw: Chem.RWMol, locants: dict[str, int], rest: str) -> bool:
    rest = rest.strip("-")
    um = re.match(r"^([0-9(),]+)-(" + _UNSAT_ALTERNATION + r")e?$", rest)
    if um is None:
        return False
    tokens = um.group(1).split(",")
    if len(tokens) != _UNSAT_MULT[um.group(2)]:
        return False
    for tok in tokens:
        pm = re.match(r"^(\d+)(?:\((\d+)\))?$", tok)
        if pm is None:
            return False
        lo = pm.group(1)
        hi = pm.group(2) or str(int(lo) + 1)
        if lo not in locants or hi not in locants:
            return False
        bond = rw.GetBondBetweenAtoms(locants[lo], locants[hi])
        if bond is None:
            return False
        bond.SetBondType(Chem.BondType.DOUBLE)
    return True


__all__ = [
    "parse_von_baeyer",
    "parse_hantzsch_widman",
    "parse_spiro",
    "parse_monocyclic_replacement",
    "build_skeleton",
    "build_spiro_skeleton",
]
