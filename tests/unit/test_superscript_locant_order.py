"""P-14.3.5 orders locants by value, and a superscript is a value.

P-25.3.3.3 numbers an interior carbon from its nearest peripheral atom and
appends the bond distance as a superscript, so a completed fused system spells
locants like 3a and 3a-superscript-one side by side. P-14.3.5 cites the
unsuperscripted locant first and compares the rest by value.

Two ways of getting that wrong are easy and both were live here. Matching only
letters leaves the superscripted form to a fallback that ranks the whole label
against a bare suffix, where a digit beats a letter and the superscripted
locant comes first. Folding the superscript into the suffix text fixes that
pair but still compares distances as characters, which transposes any
two-digit distance.
"""

import pytest

from openclatura.assembly_utils import parse_locant
from openclatura.locants import parse_system_locant, system_locant_sort_key

# P-14.3.5: the unsuperscripted locant first, then ascending distance, and the
# whole group before the next fusion letter and the next numeral.
ASCENDING = ("4a", "5a", "5a¹", "5a²", "5a³", "5a¹⁰", "5b", "6")


def test_superscripted_locants_sort_by_value():
    assert sorted(ASCENDING, key=parse_locant) == list(ASCENDING)


def test_the_distance_is_compared_as_a_number_not_as_text():
    """The case a suffix-text fix still gets wrong: 5a-10 after 5a-2."""

    assert parse_locant("5a²") < parse_locant("5a¹⁰")
    assert parse_locant("5a") < parse_locant("5a¹")


def test_both_spellings_of_a_distance_agree():
    """The ASCII caret form and the rendered superscript are one locant."""

    assert parse_locant("5a^10") == parse_locant("5a¹⁰")
    assert parse_locant("3a^1") == parse_locant("3a¹")


def test_ordinary_locants_are_untouched():
    assert sorted(["10", "2", "2a", "N", "1"], key=parse_locant) == ["N", "1", "2", "2a", "10"]


def test_the_completed_system_key_agrees_with_the_assembly_key():
    """One order, whichever side of the naming layer asks for it."""

    typed = [parse_system_locant(text) for text in ASCENDING]
    assert [str(value) for value in sorted(typed, key=system_locant_sort_key)] == list(ASCENDING)


@pytest.mark.parametrize("locant", ("3a¹", "5a¹⁰", "2a²"))
def test_a_superscripted_locant_round_trips_through_its_typed_form(locant):
    assert str(parse_system_locant(locant)) == locant
