"""P-24.2.4.1.1: a dispiro system is numbered from its smaller terminal ring."""

import re

import pytest
from rdkit import Chem

from openclatura import name_smiles


def _linear_dispiro(first: int, middle: int, last: int) -> str:
    """Ring A(first) -- spiro -- ring B(middle) -- spiro -- ring C(last)."""

    return "C12(" + "C" * (first - 1) + "1)" + "C" * (middle - 2) + "C23" + "C" * (last - 1) + "3"


def _first_bracket_number(name: str) -> int:
    match = re.search(r"dispiro\[(\d+)\.", name)
    assert match, f"no dispiro descriptor in {name!r}"
    return int(match.group(1))


def test_reported_dispiro_numbering_starts_in_the_three_membered_ring():
    """The case from user feedback: cyclobutane-spiro-cyclopropane-spiro-cyclopropane.

    Numbering used to open in the four-membered ring, which put the gem-dimethyl
    carbon at position 2 instead of 6.
    """

    assert name_smiles("CC1(C)CC2(CC23CC3)C1") == "6,6-dimethyldispiro[2.0.3.1]octane"


@pytest.mark.parametrize(
    "first,middle,last",
    [(a, b, c) for a in range(3, 7) for b in range(3, 7) for c in range(3, 7) if a != c],
)
def test_dispiro_numbering_opens_in_the_smaller_terminal_ring(first, middle, last):
    smiles = _linear_dispiro(first, middle, last)
    molecule = Chem.MolFromSmiles(smiles)
    assert molecule is not None
    assert sorted(len(ring) for ring in molecule.GetRingInfo().AtomRings()) == sorted([first, middle, last])

    name = name_smiles(smiles)
    # The leading bracket number counts the terminal ring's atoms besides its
    # spiro atom, so it identifies which terminal ring the numbering opened in.
    assert _first_bracket_number(name) == min(first, last) - 1
