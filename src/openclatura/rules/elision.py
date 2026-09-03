"""
openclatura/rules/elision.py

Vowel and consonant elision rules for IUPAC name assembly.

When concatenating name fragments (stems, multipliers, suffixes, etc.),
certain vowels are elided to produce a pronounceable result. These rules
are applied at the assembler stage when joining pieces.

References:
- IUPAC 2013 Recommendations, P-16.3.3 (elision of vowels)
- IUPAC 2013 Recommendations, P-25.3.1.3 (Hantzsch-Widman elision)
"""

VOWELS: frozenset[str] = frozenset("aeiouy")


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


def elide_terminal_e(stem: str, following: str) -> str:
    """Elide a parent hydride's terminal 'e' before a suffix starting with a vowel (P-16.3.3).

    A locant set between the two does not stop the elision: "thietane" + "-1-ylidene"
    -> "thietan-1-ylidene".
    """
    if not stem.endswith("e") or not following:
        return stem + following
    rest = following.lstrip("-")
    rest = rest.split("-", 1)[1] if rest[:1].isdigit() and "-" in rest else rest
    return (stem[:-1] + following) if is_vowel_start(rest) else stem + following


def is_vowel_start(s: str) -> bool:
    """Return True if `s` begins with a vowel (a, e, i, o, u, y)."""
    return bool(s) and s[0].lower() in VOWELS
