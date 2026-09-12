"""Skeletal replacement preserves vowels; Hantzsch-Widman contracts them."""

import pytest

from openclatura.assembly_parent import apply_replacement_prefix
from openclatura.assembly_parts import AssemblyParts, SubstituentItem
from openclatura.assembly_prefixes import format_replacement_prefixes
from openclatura.rules.elision import elide_terminal_a


@pytest.mark.parametrize(
    "parent", ["ethane", "octane", "undecane", "icosane", "bicyclo[2.2.1]heptane", "octacyclo[26.4.0]"]
)
@pytest.mark.parametrize("prefix", ["1-aza", "1-oxa", "1-thia", "1-sila", "1-azonia", "1,3,5,7-tetraaza"])
def test_skeletal_replacement_does_not_elide_terminal_a(parent, prefix):
    assert apply_replacement_prefix(parent, prefix) == prefix + parent


def test_empty_replacement_prefix_leaves_parent_unchanged():
    assert apply_replacement_prefix("octane", "") == "octane"


def test_replacement_multiplier_and_parent_boundaries_both_preserve_a():
    parts = AssemblyParts(parent_length=11)
    parts.a_prefixes = [SubstituentItem(name="aza", locants=["1", "3", "5", "7"])]
    prefix = format_replacement_prefixes(parts)
    assert prefix == "1,3,5,7-tetraaza"
    assert apply_replacement_prefix("undecane", prefix) == "1,3,5,7-tetraazaundecane"


@pytest.mark.parametrize(
    "prefix, following, expected",
    [("oxa", "irane", "oxirane"), ("thia", "azole", "thiazole"), ("oxa", "thiane", "oxathiane")],
)
def test_hantzsch_widman_elision_remains_separate(prefix, following, expected):
    assert elide_terminal_a(prefix, following) == expected
