"""Regression tests for bugs found and fixed while adversarially testing the
fix-elided-locant-parentheses branch, plus one finding that was retracted
after deeper investigation showed it was never actually a bug.

Both fixed bugs were confirmed pre-existing (not introduced by this branch)
before being fixed here: each reproduced byte-identically against commit
918a9f9 ("update values" -- the last commit before this branch's own
parenthesis-fix commits, e2e9bfd through 54299d2), via a separate git
worktree.

1. Ligand citation-order bug (FIXED): ``format_center_ligands`` (formatting.py)
   is called from many sites -- carbamoyl/amide-N substituents, ammonio,
   iminium, azanidyl/amino, lambda-substituent sulfur/pnictogen/group13-14/
   halogen subgraphs -- most of which passed no ``sort_key``, so ligands were
   ordered with plain Python string comparison. A ligand that arrives already
   parenthesised (because it's independently flagged "complex", e.g. it
   starts with a locant digit) sorted before every bare lowercase-starting
   ligand purely because ``(`` (ASCII 40) is less than every letter,
   regardless of the ligand's TRUE first letter once enclosing marks are
   ignored (the correct IUPAC P-14.5.2 alphanumerical-order rule). Fixed by
   passing ``sort_key=substituent_sort_key`` (which strips enclosing marks
   before comparing, already used correctly by a handful of call sites --
   organoboronic_acid_result, sulfonium_ylide_name, sulfamic_acid_result,
   phosphane_borane_zwitterion_result, and the Group 14 hydride path via
   _grouped_ligand_prefix) at every remaining call site.

2. Bare acyclic azanide charge-drop (FIXED): a bare ACYCLIC azanide anion
   (no ring, no other principal-group parent) silently dropped its formal
   charge and was named as the neutral amine. Root cause:
   ``perception.py``'s whole-molecule nitrogen-group detector only branched
   on ``atom.charge > 0`` ("aminium") vs. else ("amine") -- there was no
   ``atom.charge < 0`` branch at all, unlike the exactly analogous oxygen
   ("olate"/"alcohol") and sulfur ("thiolate"/"thiol") detectors right next
   to it in the same function, which already had one. Fixed by adding the
   missing branch (key "aminide") and registering "aminide" as a proper
   functional-group suffix (namer_rules.json, mirroring "olate"/"thiolate"'s
   shape) and anionic-suffix/exact-charge-renderer membership alongside them.
   The SUBSTITUENT-prefix path ("azanidyl") was already correct and
   untouched -- see test_center_ligand_parenthesization.py's nitrogen cases.

3. Ring-stem "multiplying prefix" lookalike (RETRACTED, not a bug): the
   original review flagged ``_starts_with_multiplier`` (assembly_prefixes.py)
   as a false-positive risk on heterocycle stems like "triazolyl"/
   "tetrazolyl", reasoning that "tri"/"tetra" there were coincidental letter
   overlap with the real multiplying-prefix words. That reasoning was WRONG:
   reading ``hantzsch_widman.py`` shows Hantzsch-Widman heterocycle names are
   built by literally calling the SAME multiplier machinery
   (``multipliers.basic(count)``) to multiply a heteroatom replacement prefix
   ("aza", "oxa", "thia", ...) when a ring has more than one of that
   heteroatom -- e.g. 1,2,4-triazole's "tri" is `_multiplied("aza", 3)`,  a
   REAL multiplying prefix, just applied to a replacement prefix rather than
   to a discrete substituent group. So ``_starts_with_multiplier`` correctly
   recognises it, and the conservative parenthesisation this produces (kept,
   not stripped) is the intended, correct behaviour, not a defect. See the
   retraction test below for the corrected understanding and a positive
   confirmation.
"""

from __future__ import annotations

from openclatura import name as name_one
from openclatura.assembly_prefixes import _starts_with_multiplier

# ---------------------------------------------------------------------------
# Fix 1: ligand citation order no longer depends on incidental parenthesis
# characters introduced by an unrelated "is this ligand complex" check.
# ---------------------------------------------------------------------------


def test_complex_ligand_no_longer_sorts_by_its_own_leading_parenthesis():
    """cyclopropyl ('c') is now correctly cited before a locant-bearing
    2-methylpropyl ('m' once you ignore the locant/parens it's wrapped in
    for being complex) across every family this bug could reach."""

    cases = {
        # carbamoyl N-substituents (functional_prefixes.amide_prefix_from_group)
        "CC(C)(C(=O)N(CC(C)C)C1CC1)NC(=O)OC": ("methyl (2-(cyclopropyl(2-methylpropyl)carbamoyl)propan-2-yl)carbamate"),
        # ammonio substituents (ionic_naming.ammonio_prefix)
        "[N+](C1CC1)(CC(C)C)(C)CC(=O)[O-]": "2-(cyclopropyl(methyl)(2-methylpropyl)ammonio)acetate",
        # phosphine ligand list (heteroatom_subgraphs.name_pnictogen_subgraph);
        # avoid a ring ligand here -- simple_central_parent_hydride_result
        # now correctly refuses to fire when a ligand is part of a ring
        # (P-44.1.1: rings outrank mononuclear hydride parents), so a
        # cyclopropyl ligand routes through cyclopropane-as-parent instead
        # of exercising this ligand-list code at all (see
        # test_group14_ether_ligand_parent_selection.py for that fix)
        "P(CC(C)C)(CC)C": "ethyl(methyl)(2-methylpropyl)phosphane",
    }
    for smiles, expected in cases.items():
        result = name_one(smiles, verify_opsin=True)
        assert result.name == expected, (smiles, result.name, expected)
        assert result.opsin_check.status == "matched", (smiles, result.name, result.opsin_check.status)


# ---------------------------------------------------------------------------
# Fix 2: a bare acyclic nitrogen anion keeps its formal charge.
# ---------------------------------------------------------------------------


def test_bare_acyclic_azanide_anion_keeps_its_charge():
    result = name_one("[N-](C)CC", verify_opsin=True)
    assert result.name == "N-methylethanaminide"
    assert result.opsin_check.status == "matched", (result.name, result.opsin_check.status)


def test_smaller_bare_acyclic_azanide_anion_keeps_its_charge():
    result = name_one("[N-](C)C", verify_opsin=True)
    assert result.name == "N-methylmethanaminide"
    assert result.opsin_check.status == "matched", (result.name, result.opsin_check.status)


def test_aryl_substituted_acyclic_azanide_anion_keeps_its_charge():
    result = name_one("[N-](c1ccccc1)C", verify_opsin=True)
    assert result.name == "N-methylbenzenaminide"
    assert result.opsin_check.status == "matched", (result.name, result.opsin_check.status)


def test_ring_azanide_anion_is_unaffected_by_the_fix():
    """Control: this path was already correct (a different code path
    entirely -- ring parents get their "-ide" suffix elsewhere) and must
    keep behaving exactly as before."""

    result = name_one("[N-]1CCCC1", verify_opsin=True)
    assert result.name == "pyrrolidin-1-ide"
    assert result.opsin_check.status == "matched"


def test_substituent_prefix_azanidyl_form_is_unaffected_by_the_fix():
    """Control: the SUBSTITUENT-prefix spelling ("azanidyl", used when the
    nitrogen anion is attached to something else that outranks it, e.g. a
    carboxylate) was already correct before this fix and must stay so."""

    result = name_one("[N-](C(C)C)CC(=O)[O-]", verify_opsin=True)
    assert result.opsin_check.status == "matched", (result.name, result.opsin_check.status)


# ---------------------------------------------------------------------------
# Retracted finding: ring stems starting with "tri"/"tetra" are NOT a false
# positive of _starts_with_multiplier -- Hantzsch-Widman ring names really do
# construct that "tri"/"tetra" via the same multiplier machinery.
# ---------------------------------------------------------------------------


def test_starts_with_multiplier_correctly_flags_hantzsch_widman_ring_stems():
    """1,2,4-triazole's "tri" is genuinely `_multiplied("aza", 3)` (see
    hantzsch_widman.hw_name), i.e. a real multiplying prefix applied to the
    "aza" replacement prefix three times -- not a coincidental look-alike.
    `_starts_with_multiplier` classifying "triazolyl"/"tetrazolyl" as
    "starts with a multiplying prefix" is therefore CORRECT, and the
    resulting conservative parenthesisation is intended, not a bug."""

    assert _starts_with_multiplier("triazolyl") is True
    assert _starts_with_multiplier("tetrazolyl") is True
    # Contrast with a name where a locant genuinely precedes the ring stem
    # (the normal shape produced by the actual naming pipeline): the string
    # no longer starts with the multiplier word, and is classified correctly.
    assert _starts_with_multiplier("1,2,4-triazol-1-yl") is False
