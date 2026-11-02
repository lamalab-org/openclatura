"""Request-scoped presentation style for generated names.

A style never changes which structure a name describes; it only selects
between equivalent spellings.  ``NamingStyle()`` reproduces the engine's
historical output byte for byte, so the default path and every benchmark
built on it stay unchanged.

Styles act at two points:

* lexical choices that change alphanumerical ordering or multiplier choice
  (``thiol_prefix``) are resolved where the prefix is generated;
* the enclosing-mark nesting is a length-preserving pass over the finished
  name, so token spans recorded during assembly stay valid.
"""

from __future__ import annotations

import re
from contextvars import ContextVar, Token
from dataclasses import dataclass, fields, replace
from typing import Literal

EnclosingMarks = Literal["parentheses", "nested", "alternating"]
ThiolPrefix = Literal["sulfanyl", "mercapto"]
LocantPlacement = Literal["before_suffix", "before_parent"]
HeterocycleNames = Literal["retained", "hantzsch_widman"]


@dataclass(frozen=True)
class NamingStyle:
    """Spelling options for a naming request.

    ``enclosing_marks``: ``"parentheses"`` nests plain parentheses
    ``((()))``; ``"nested"`` follows the P-16.5.4 order ``{[({[( )]})]}``;
    ``"alternating"`` cycles parentheses and brackets ``([([ ])])``.

    ``thiol_prefix``: prefix for an unsubstituted ``-SH`` group, the
    preferred ``"sulfanyl"`` (P-63.1.5) or the traditional ``"mercapto"``.

    ``locant_placement``: where the parent's first locant set goes,
    ``"before_suffix"`` (P-14.3.2: but-2-ene, pentan-2-ol) or the older
    ``"before_parent"`` (2-butene, 2-pentanol, 2-buten-1-ol).  Only plain
    alphabetic parent stems move; ``bicyclo[2.2.1]heptan-2-one``,
    ``2H-pyran-2-one`` and substituent prefixes keep their spelling.

    ``heterocycle_names``: monocyclic heterocycles take their retained names
    (``"retained"``: furan, pyrrolidine, pyrimidine) or Hantzsch-Widman
    names (``"hantzsch_widman"``: oxole, azolidine, 1,3-diazine).  Pyridine
    and pyran stay retained: ``azine`` names a compound class and ``oxine``
    is 8-hydroxyquinoline, and OPSIN reads neither.  Fusion components are
    unaffected.
    """

    enclosing_marks: EnclosingMarks = "parentheses"
    thiol_prefix: ThiolPrefix = "sulfanyl"
    locant_placement: LocantPlacement = "before_suffix"
    heterocycle_names: HeterocycleNames = "retained"

    def __post_init__(self) -> None:
        if self.enclosing_marks not in ("parentheses", "nested", "alternating"):
            raise ValueError(f"unknown enclosing_marks {self.enclosing_marks!r}")
        if self.thiol_prefix not in ("sulfanyl", "mercapto"):
            raise ValueError(f"unknown thiol_prefix {self.thiol_prefix!r}")
        if self.locant_placement not in ("before_suffix", "before_parent"):
            raise ValueError(f"unknown locant_placement {self.locant_placement!r}")
        if self.heterocycle_names not in ("retained", "hantzsch_widman"):
            raise ValueError(f"unknown heterocycle_names {self.heterocycle_names!r}")


DEFAULT_STYLE = NamingStyle()

PRESETS: dict[str, NamingStyle] = {
    # The engine's historical output.
    "default": DEFAULT_STYLE,
    # IUPAC 2013 recommendations (P-16.5.4 enclosing-mark nesting).
    "iupac2013": NamingStyle(enclosing_marks="nested"),
    # Pre-1993 spellings still common in catalogues and older literature.
    "traditional": NamingStyle(enclosing_marks="nested", thiol_prefix="mercapto", locant_placement="before_parent"),
}

STYLE_OPTIONS = tuple(field.name for field in fields(NamingStyle))


def resolve_style(style: NamingStyle | str | None = None, **overrides: str) -> NamingStyle:
    """Return a style from a preset name or instance plus per-option overrides."""

    if style is None:
        base = DEFAULT_STYLE
    elif isinstance(style, NamingStyle):
        base = style
    else:
        try:
            base = PRESETS[style]
        except KeyError:
            raise ValueError(f"unknown naming style {style!r}; presets: {', '.join(PRESETS)}") from None
    unknown = set(overrides) - set(STYLE_OPTIONS)
    if unknown:
        raise ValueError(f"unknown style option(s): {', '.join(sorted(unknown))}")
    set_overrides = {key: value for key, value in overrides.items() if value is not None}
    return replace(base, **set_overrides) if set_overrides else base


_NAMING_STYLE: ContextVar[NamingStyle] = ContextVar("openclatura_naming_style", default=DEFAULT_STYLE)


def current_naming_style() -> NamingStyle:
    """Return the style for the current naming request."""

    return _NAMING_STYLE.get()


def set_naming_style(style: NamingStyle) -> Token[NamingStyle]:
    """Install a request-local style and return its reset token."""

    return _NAMING_STYLE.set(style)


def reset_naming_style(token: Token[NamingStyle]) -> None:
    """Restore the style that preceded ``token``."""

    _NAMING_STYLE.reset(token)


# Hantzsch-Widman parents never used in place of their retained names.
RETAINED_OVER_HANTZSCH_WIDMAN = frozenset({"azine", "oxine"})


def hantzsch_widman_requested() -> bool:
    """Whether the current style names monocyclic heterocycles by Hantzsch-Widman."""

    return _NAMING_STYLE.get().heterocycle_names == "hantzsch_widman"


def styled_thiol_prefix(prefix: str) -> str:
    """Spell the unsubstituted ``-SH`` prefix in the current style."""

    if prefix == "sulfanyl" and _NAMING_STYLE.get().thiol_prefix == "mercapto":
        return "mercapto"
    return prefix


# A parent word whose stem is plain letters followed by its first locant set:
# ``but-2-ene``, ``pentan-2-ol``, ``benzene-1,2-diol``.
_PARENT_LEADING_LOCANTS = re.compile(r"^([a-z]+)-(\d+[a-z]?'*(?:,\d+[a-z]?'*)*)-(?=[a-z])")


def place_parent_locants(core_name: str) -> str:
    """Move the parent's first locant set in front of it when the style asks.

    ``but-2-en-1-ol`` becomes ``2-buten-1-ol``: only the first set moves,
    later ones stay with their suffix, as in CAS and pre-1993 IUPAC names.
    """

    if _NAMING_STYLE.get().locant_placement != "before_parent":
        return core_name
    match = _PARENT_LEADING_LOCANTS.match(core_name)
    if match is None:
        return core_name
    stem, locants = match.groups()
    return f"{locants}-{stem}{core_name[match.end() :]}"


# Contents of a parenthesis that is a descriptor rather than an enclosing mark:
# stereo (2R,3S) (E) (aR), added hydrogen (2H), charges (1+), isotopes (13C2),
# subtractive locants 1(10).  Enclosed prefixes always hold a word of three or
# more lowercase letters.
_ENCLOSED_WORD = re.compile(r"[a-z]{3,}")
_OPENERS = {"(": ")", "[": "]", "{": "}"}
_NESTED_MARKS = (("(", ")"), ("[", "]"), ("{", "}"))
_ALTERNATING_MARKS = (("(", ")"), ("[", "]"))


def nest_enclosing_marks(name: str, marks: tuple[tuple[str, str], ...] = _NESTED_MARKS) -> str:
    """Re-spell enclosing parentheses by nesting level, cycling through ``marks``.

    The default cycle is the P-16.5.4 order ``{[({[( )]})]}``.

    Each enclosing mark takes the level one above the highest mark it
    contains.  Descriptor parentheses (stereo, added hydrogen) count as
    level 0 and keep their parentheses; brackets that belong to a
    descriptor (``bicyclo[2.2.1]``, ``thieno[2,3-b]``, ``[1,1'-biphenyl]``)
    count as level 1 and are kept as written.  Unbalanced input is returned
    unchanged.
    """

    if "(" not in name:
        return name
    # Parse into (open_index, close_index, children) groups.
    stack: list[tuple[int, list]] = [(-1, [])]
    groups: dict[int, tuple[int, list]] = {}
    for index, char in enumerate(name):
        if char in _OPENERS:
            stack.append((index, []))
        elif char in ")]}":
            if len(stack) == 1:
                return name
            open_index, children = stack.pop()
            if _OPENERS[name[open_index]] != char:
                return name
            groups[open_index] = (index, children)
            stack[-1][1].append(open_index)
    if len(stack) != 1:
        return name

    chars = list(name)

    def level(open_index: int) -> int:
        close_index, children = groups[open_index]
        inner = max((level(child) for child in children), default=-1)
        opener = name[open_index]
        if opener != "(":
            # Descriptor brackets are written as is but still raise the
            # marks around them.
            return max(1, inner + 1)
        own = inner + 1
        if _ENCLOSED_WORD.search(name, open_index + 1, close_index) is None:
            return own if children else 0
        open_mark, close_mark = marks[own % len(marks)]
        chars[open_index] = open_mark
        chars[close_index] = close_mark
        return own

    for top in stack[0][1]:
        level(top)
    return "".join(chars)


def apply_final_style(name: str, style: NamingStyle) -> str:
    """Apply the length-preserving style passes to a finished name."""

    if style.enclosing_marks == "nested":
        name = nest_enclosing_marks(name)
    elif style.enclosing_marks == "alternating":
        name = nest_enclosing_marks(name, _ALTERNATING_MARKS)
    return name
