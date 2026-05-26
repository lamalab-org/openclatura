"""
structure-to-iupac/rules/elision.py

Vowel and consonant elision rules for IUPAC name assembly.

When concatenating name fragments (stems, multipliers, suffixes, etc.),
certain vowels are elided to produce a pronounceable result. These rules
are applied at the assembler stage when joining pieces.

References:
- IUPAC 2013 Recommendations, P-16.3.3 (elision of vowels)
- IUPAC 2013 Recommendations, P-25.3.1.3 (Hantzsch-Widman elision)
"""

VOWELS: frozenset[str] = frozenset("aeiouy")


def elide_terminal_e(stem: str, suffix: str) -> str:
    """Elide a terminal 'e' from `stem` when `suffix` starts with a vowel.

    This is the most common elision rule, used for parent terminal vowel:
        "hexane" + "ol"  -> "hexan" + "ol"  -> "hexanol"
        "hexane" + "oic" -> "hexan" + "oic" -> "hexanoic"
        "hexane" + "amine" -> "hexan" + "amine" -> "hexanamine"
    But:
        "hexane" + "dione" -> "hexanedione"  (suffix starts with 'd', no elision)

    Note: applied to the bare stem+bond combination *before* attaching the
    parent terminal "e". In practice the assembler decides whether to keep
    the "e" based on what follows.
    """
    if not stem or not suffix:
        return stem + suffix
    if stem.endswith("e") and suffix[0] in VOWELS:
        return stem[:-1] + suffix
    return stem + suffix


def elide_terminal_a(prefix: str, following: str) -> str:
    """Elide a terminal 'a' from a Hantzsch-Widman heteroatom prefix
    when followed by a vowel-starting fragment.

    Used in heterocycle naming:
        "oxa" + "azine"  -> "ox" + "azine"  -> "oxazine"
        "oxa" + "irane"  -> "ox" + "irane"  -> "oxirane"
        "thia" + "azole" -> "thi" + "azole" -> "thiazole"
    But:
        "oxa" + "thiane" -> "oxathiane"  (consonant follows, no elision)

    Also used in replacement (skeletal) nomenclature:
        "oxa" + "ane" -> "ox" + "ane" -> ...  (rare; handled per-case)
    """
    if not prefix or not following:
        return prefix + following
    if prefix.endswith("a") and following[0] in VOWELS:
        return prefix[:-1] + following
    return prefix + following


def elide_terminal_o(prefix: str, following: str) -> str:
    """Elide a terminal 'o' from certain prefixes before a vowel.

    Less common than 'a' or 'e' elision, but applies to a few cases like
    "iodo" combinations and some retained-name prefixes. In practice this
    is rare in modern PINs; included for completeness.

    Note: most "o"-ending substituent prefixes (chloro, bromo, fluoro, nitro)
    are NOT elided before vowels — "chloroethane" not "chlorethane".
    Elision is not the default for substituent prefixes; this helper exists
    only for specific named cases that opt in.
    """
    if not prefix or not following:
        return prefix + following
    if prefix.endswith("o") and following[0] in VOWELS:
        return prefix[:-1] + following
    return prefix + following


def join_with_elision(*fragments: str, rule: str = "e") -> str:
    """Join name fragments left-to-right, applying the specified elision rule
    at each junction.

    rule:
        "e" - elide trailing 'e' before vowel-starting fragment (most common)
        "a" - elide trailing 'a' before vowel-starting fragment (HW prefixes)
        "o" - elide trailing 'o' before vowel-starting fragment (rare)
        "none" - no elision, simple concatenation

    Example:
        join_with_elision("hexan", "e")              -> "hexane"
        join_with_elision("hexan", "e", "ol", rule="e") -> wrong; see below
        join_with_elision("hexan", "ol", rule="e")   -> "hexanol"

    Note: this is intentionally simple. Complex multi-fragment names with
    locants (e.g. "hexa-1,3-dien-1-ol") are assembled by the dedicated
    name assembler, not by this helper. Use this only for two- or
    three-fragment joins where the elision rule is clear.
    """
    if not fragments:
        return ""
    if rule == "none":
        return "".join(fragments)
    elider = {
        "e": elide_terminal_e,
        "a": elide_terminal_a,
        "o": elide_terminal_o,
    }.get(rule)
    if elider is None:
        raise ValueError(f"Unknown elision rule: {rule!r}")
    result = fragments[0]
    for frag in fragments[1:]:
        result = elider(result, frag)
    return result


def is_vowel_start(s: str) -> bool:
    """Return True if `s` begins with a vowel (a, e, i, o, u, y)."""
    return bool(s) and s[0].lower() in VOWELS


def is_vowel_end(s: str) -> bool:
    """Return True if `s` ends with a vowel."""
    return bool(s) and s[-1].lower() in VOWELS
