"""Regression tests for several charge-representation gaps found while
auditing the codebase for other instances of the "charge silently dropped
from the name" bug class (the same class of defect the azanide/aminide fix
earlier this session addressed for amide/amine nitrogen).

Bug 1 -- single_atom_component_name ignored charge sign entirely (FIXED):
``single_atom_cations`` (Na, K, Li, Mg, Ca) and ``single_atom_anions`` (F,
Cl, Br, I) were matched by ELEMENT SYMBOL ALONE, with no check that the
atom's actual charge matched the cation/anion role being named. A neutral
atom of one of these elements, or one with the wrong charge sign (e.g. a
stray "[Cl+]"), was still confidently named "sodium"/"chloride" etc. Two
positively-charged components with no anion at all (a chemically invalid,
charge-imbalanced input) could even both be silently named as if they
formed a valid neutral salt together. Fixed by requiring ``charge > 0``/
``charge < 0`` respectively; a mismatched-charge input now honestly fails
to find a supported parent skeleton instead of naming the wrong thing.

Bug 2 -- the same function's mononuclear-hydride fallback lacked a charge
guard (FIXED): the branch just above it (retained names "ammonia"/"water")
already correctly required ``atom.charge == 0``, but the general hydride
fallback right below it did not, so a genuinely CHARGED metal cation of an
element that also has a covalent hydride name (e.g. Al -- "alumane" is
AlH3) was named using that neutral hydride name regardless of its actual
charge: [Al+3] paired with three chlorides came out as "alumane
trichloride" (confirmed by OPSIN reparsing back to neutral covalent AlCl3,
not the ionic input). Fixed by requiring ``atom.charge == 0`` before using
the hydride-name fallback, matching the sibling branch right above it.
Naively extending the cation table to cover these metals was tried and
rejected: round-tripping candidate names ("aluminum trichloride", "tin
dichloride", "bismuth trichloride", "lead dichloride") through OPSIN shows
most reparse as the wrong (neutral covalent) structure regardless -- these
elements need oxidation-state-aware (Stock/Roman-numeral) naming
machinery this codebase doesn't have yet, so an honest failure is the
correct trade until that exists, not a differently-wrong name.

Bug 3 -- azinic_acid_result was completely broken (FIXED): every single
invocation via name_one(verify_opsin=True) failed a "final assembly audit"
because its core NameAtomBinding (covering the charged nitrogen and its
oxido oxygen) never populated charge_atom_ids, the same binding-metadata
gap already fixed elsewhere this session for phosphinic/phosphonic
esters. This didn't affect the plain name_smiles() path the main test
suite uses (which is why it went unnoticed), but made the feature
unusable under any audited/verified path. Fixed by adding
charge_atom_ids={a for a in core_atoms if mol.atoms[a].charge}, the same
pattern used everywhere else in this file.

Bug 4 -- ring/chain PARENT cations were only spelled with an -ium suffix
for nitrogen and oxygen (FIXED): positive_parent_ium_charges (which drives
the generic "-<locant>-ium" parent-suffix mechanism used by a-prefix
replacement rings such as thiacyclohexatriene, and by retained ring names
like pyridine/phosphinine/arsinine) filtered charge sites to
{"N", "O"} only. A positively-charged ring/chain parent atom of any other
onium-forming element -- S, Se, Te, P, As -- structurally names correctly
(the ring skeleton itself was never in question) but silently LOST its
formal charge from the name entirely: a thiopyrylium analogue named as
"1-thiacyclohexa-1,3,5-triene" with no hint of the +1 charge on sulfur,
and a saturated S+/Se+/Te+ ring (e.g. protonated thiane) named as the
plain neutral parent ("thiane") with the cation dropped completely. Fixed
by widening the filter to the standard onium-forming set: N, O, S, Se, Te,
P, As, Sb, Bi. (Sb/Bi aromatic 6-rings -- stibinine/bismabenzene-type --
turned out to lack even basic STRUCTURAL support in this codebase
independent of charge, confirmed separately and left alone as an
unrelated, pre-existing gap outside this fix's scope.)

Bug 5 -- the hydrogen halides (HF, HCl, HBr, HI) had no retained name at
all, and fixing bug 1 exposed it (FIXED): before bug 1's charge-sign fix,
a neutral, H-bearing halogen atom like "[ClH]" (real HCl) accidentally
matched the SYMBOL-only single_atom_anions lookup and was named
"chloride" -- wrong (a bare anion name for a neutral covalent molecule),
but non-empty, so nothing noticed. Once bug 1 correctly required
``charge < 0`` for that lookup, HCl fell through everything and produced
no name at all, since RETAINED_MONONUCLEAR_HYDRIDE_NAMES only had
"N"/"O" (ammonia/water). Fixed by adding the P-21.1.1.1 retained names
for all four hydrogen halides.

Bug 6 -- neither the retained-hydride-name branch nor the general
mononuclear-hydride fallback checked that the atom actually carries its
full complement of hydrogens (FIXED, found while fixing bug 5): a bare,
disconnected, zero-hydrogen atom like "[N]" or "[Si]" -- not a real,
complete molecule, and not something any of these functions should ever
successfully name -- matched purely on element symbol and charge==0,
producing "ammonia" for "[N]" and "silane" for "[Si]" even though neither
carries the hydrogens those names imply. Fixed by additionally requiring
``atom.total_h_count == atom.element.standard_valence`` in both branches,
so only an atom with its full, real hydrogen complement matches.

Also investigated, found to be dead code rather than a live bug: three
``ParentChargeRule`` entries in ionic_naming.py (PARENT_CHARGE_RULES) have
charge_sign=1 predicates hardcoded to ``site.symbol == "N"``, but the only
caller of apply_parent_charge_names passes charge_signs={-1} -- so those
three positive-charge rules can never actually fire today regardless of
element. Left unchanged: broadening an unreachable predicate has no
observable effect and would just be noise.

Also investigated, NOT reproduced as a live bug: a hypothesised missing
principal-group-competition guard in sulfamic_acid_result. Deliberately
constructing a molecule with both a sulfamic-acid-type group and a more
senior competing principal group elsewhere (a carboxylic acid reached via
an N-alkyl chain) already correctly demotes the sulfamic group to a
"sulfamoylamino" prefix and picks the carboxylic acid as parent --
apparently because sulfamic_acid_result is only reachable, architecturally,
when no carbon-based parent candidate exists in the component at all (its
own detection requires a ring-free nitrogen with no ring atoms and the
whole shape is tried only as a last-resort replacement-parent-hydride
candidate). No concrete failing input was found, so no speculative guard
was added.
"""

from __future__ import annotations

import pytest

from openclatura import name as name_one


def _assert_matched(smiles: str, expected_name: str) -> None:
    result = name_one(smiles, verify_opsin=True)
    assert result.error is None, (smiles, result.error)
    assert result.name == expected_name, (smiles, result.name, expected_name)
    assert result.opsin_check is not None
    assert result.opsin_check.status == "matched", (
        smiles,
        result.name,
        result.opsin_check.status,
        result.opsin_check.opsin_smiles,
    )


# ---------------------------------------------------------------------------
# Bug 1: single-atom cation/anion naming must respect the atom's own charge.
# ---------------------------------------------------------------------------


def test_single_atom_cations_and_anions_still_name_correctly():
    _assert_matched("[Na+].[Cl-]", "sodium chloride")
    _assert_matched("[Mg+2].[Cl-].[Cl-]", "magnesium dichloride")


@pytest.mark.parametrize(
    "smiles",
    [
        "[Na].[Cl-]",  # neutral sodium, not a cation
        "[Na].[Cl]",  # both neutral -- no ions at all
        "[Cl+].[Na+]",  # two cations, no anion -- charge-imbalanced nonsense
    ],
)
def test_wrong_charge_single_atoms_fail_honestly_instead_of_mislabeling(smiles):
    result = name_one(smiles, verify_opsin=True)
    assert result.error is not None, (smiles, result.name)
    assert "sodium" not in result.name
    assert "chloride" not in result.name


def test_single_atom_component_name_checks_charge_sign_directly():
    from openclatura.graph_io import read_smiles
    from openclatura.special_cases import single_atom_component_name

    mol = read_smiles("[Na].[Cl]")
    na_idx = next(idx for idx, atom in mol.atoms.items() if atom.symbol == "Na")
    cl_idx = next(idx for idx, atom in mol.atoms.items() if atom.symbol == "Cl")
    assert single_atom_component_name(mol, {na_idx}) == ""
    assert single_atom_component_name(mol, {cl_idx}) == ""


# ---------------------------------------------------------------------------
# Bug 2: a charged metal atom must not be named as its neutral hydride.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "smiles",
    [
        "[Al+3].[Cl-].[Cl-].[Cl-]",
        "[Sn+2].[Cl-].[Cl-]",
        "[Bi+3].[Cl-].[Cl-].[Cl-]",
        "[Pb+2].[Cl-].[Cl-]",
    ],
)
def test_charged_metal_atoms_are_not_named_as_neutral_hydrides(smiles):
    result = name_one(smiles, verify_opsin=True)
    assert result.error is not None, (smiles, result.name)
    assert "ane" not in result.name or result.name == ""


def test_neutral_group13_hydrides_are_unaffected_by_the_charge_guard():
    """Control: the genuinely neutral hydrides these names belong to must
    still resolve correctly."""

    _assert_matched("[AlH3]", "alumane")
    _assert_matched("[SnH4]", "stannane")


# ---------------------------------------------------------------------------
# Bug 5: the hydrogen halides need their own retained names.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("smiles", "expected_name"),
    [
        ("[FH]", "hydrogen fluoride"),
        ("[ClH]", "hydrogen chloride"),
        ("[BrH]", "hydrogen bromide"),
        ("[IH]", "hydrogen iodide"),
    ],
)
def test_hydrogen_halides_have_retained_names(smiles, expected_name):
    _assert_matched(smiles, expected_name)


# ---------------------------------------------------------------------------
# Bug 6: a bare, hydrogen-less atom must not be named as the full hydride.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("smiles", ["[N]", "[O]", "[Cl]", "[Si]", "[P]"])
def test_bare_hydrogen_less_atoms_are_not_named_as_their_hydride(smiles):
    result = name_one(smiles, verify_opsin=True)
    assert result.error is not None, (smiles, result.name)


# ---------------------------------------------------------------------------
# Bug 3: azinic_acid_result must survive the final assembly audit.
# ---------------------------------------------------------------------------

AZINIC_ACID_CASES = (
    ("n-substituted", "[O-][NH+](O)c1ccccc1", "N-phenylazinic acid"),
    ("methylidene-form", "C=[N+]([O-])O", "(methylidene)azinic acid"),
    ("ethylidene-form", "CC=[N+]([O-])O", "(ethylidene)azinic acid"),
)


@pytest.mark.parametrize(
    ("_case", "smiles", "expected_name"),
    AZINIC_ACID_CASES,
    ids=[c for c, _s, _n in AZINIC_ACID_CASES],
)
def test_azinic_acid_survives_the_final_assembly_audit(_case, smiles, expected_name):
    _assert_matched(smiles, expected_name)


# ---------------------------------------------------------------------------
# Bug 4: ring/chain parent cations of onium-forming elements beyond N/O
# must keep their -ium suffix, not silently drop the charge.
# ---------------------------------------------------------------------------

RING_PARENT_ONIUM_CASES = (
    ("aromatic-sulfur", "c1cc[s+]cc1", "1-thiacyclohexa-1,3,5-trien-1-ium"),
    ("aromatic-selenium", "c1cc[se+]cc1", "1-selenacyclohexa-1,3,5-trien-1-ium"),
    ("aromatic-tellurium", "c1cc[te+]cc1", "1-telluracyclohexa-1,3,5-trien-1-ium"),
    ("aromatic-phosphorus", "c1cc[pH+]cc1", "phosphinin-1-ium"),
    ("aromatic-arsenic", "c1cc[asH+]cc1", "arsinin-1-ium"),
    ("saturated-sulfur", "[SH+]1CCCCC1", "thian-1-ium"),
    ("saturated-selenium", "[SeH+]1CCCCC1", "selenan-1-ium"),
    ("saturated-tellurium", "[TeH+]1CCCCC1", "telluran-1-ium"),
)


@pytest.mark.parametrize(
    ("_case", "smiles", "expected_name"),
    RING_PARENT_ONIUM_CASES,
    ids=[c for c, _s, _n in RING_PARENT_ONIUM_CASES],
)
def test_ring_parent_onium_charge_survives_beyond_nitrogen_and_oxygen(_case, smiles, expected_name):
    _assert_matched(smiles, expected_name)


def test_nitrogen_and_oxygen_ring_parent_cations_are_unaffected_controls():
    """Control: the two elements that already worked must still work."""

    _assert_matched("c1cc[nH+]cc1", "pyridin-1-ium")
    _assert_matched("c1cc[o+]cc1", "1-oxacyclohexa-1,3,5-trien-1-ium")


def test_positive_parent_ium_charges_covers_the_full_onium_element_set():
    from openclatura.assembly_charge import _IUM_SUFFIX_ELEMENTS

    assert _IUM_SUFFIX_ELEMENTS == {"N", "O", "S", "Se", "Te", "P", "As", "Sb", "Bi"}
