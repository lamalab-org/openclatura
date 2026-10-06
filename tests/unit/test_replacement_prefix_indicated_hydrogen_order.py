"""Indicated hydrogen is cited ahead of the skeletal-replacement prefixes.

P-25.7.1.3.2 puts the indicated hydrogen of an ortho- or ortho-and-peri-fused
ring system at the front of that system's name, "including replacement terms,
if any", and prints 6H-1,7-dioxacyclopenta[cd]indene for it. FR-9.3.1 states
the same order with the same example. So the run leads the name rather than
sitting against the ring body, and once it leads there is nothing to separate
the prefixes from the ring: the hyphen P-16.2.4.1(a) requires after a locant
is the one the run already carries.

This is the ring system's own name. It is not a rule about where indicated
hydrogen goes relative to the substituent prefixes of a complete name, which
is why the cases below assert the stem and not a whole substituted molecule.
"""

import pytest

from openclatura.assembly_parent import apply_replacement_prefix

# The published example, as the renderer hands it over: the ring body, the
# replacement prefix run, and the indicated hydrogen that leads the result.
PUBLISHED = ("6H-cyclopenta[cd]indene", "1,7-dioxa", "6H-", "6H-1,7-dioxacyclopenta[cd]indene")

CASES = (
    pytest.param(*PUBLISHED, id="P-25.7.1.3.2-published-example"),
    pytest.param(
        "1H,3H-benzo[de]isochromene",
        "2-aza",
        "1H,3H-",
        "1H,3H-2-azabenzo[de]isochromene",
        id="multiple-hydrogen-run",
    ),
    pytest.param(
        "cyclopenta[cd]indene",
        "1,7-dioxa",
        "",
        "1,7-dioxacyclopenta[cd]indene",
        id="no-indicated-hydrogen",
    ),
    pytest.param(
        "6H-cyclopenta[cd]indene",
        "",
        "6H-",
        "6H-cyclopenta[cd]indene",
        id="no-replacement-prefix",
    ),
)


@pytest.mark.parametrize("stem,prefixes,indicated,expected", CASES)
def test_the_indicated_hydrogen_run_leads_the_replacement_prefixes(stem, prefixes, indicated, expected):
    assert apply_replacement_prefix(stem, prefixes, indicated_hydrogen=indicated) == expected


def test_an_unrecorded_run_is_never_guessed_out_of_the_stem():
    """Without the renderer's own run, the stem is joined as it stands.

    The caller passes what the tokens recorded. A stem that happens to open on
    a locant is not evidence of indicated hydrogen, so nothing is moved on the
    strength of the text alone.
    """

    assert apply_replacement_prefix("6H-cyclopenta[cd]indene", "1,7-dioxa") == "1,7-dioxa6H-cyclopenta[cd]indene"


def test_a_run_the_stem_does_not_carry_is_left_alone():
    assert apply_replacement_prefix("cyclopenta[cd]indene", "1,7-dioxa", indicated_hydrogen="6H-") == (
        "1,7-dioxacyclopenta[cd]indene"
    )
