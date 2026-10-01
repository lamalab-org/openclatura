"""Tests for disubstituted phosphinic acid and phosphinate ester support,
added to ``organophosphinic_acid_result`` (special_cases.py).

Scope and Blue Book references (2013 IUPAC recommendations, confirmed
against the official online text before implementation):

- P-67.1.1.2: phosphinic acid's parent, H2P(=O)(OH), is substituted at the
  central phosphorus's two remaining hydrogens by one or two organyl
  groups. (C2H5)2P(=O)(OH) is "diethylphosphinic acid" -- NOT
  "P,P-diethylphosphinic acid" and not "ethyl(ethyl)phosphinic acid".
  Phosphorus itself never carries a locant in the acid name.
- P-16.5.1.3.1: two DIFFERENT substituents are cited in alphanumerical
  order with the second (and any later) one parenthesised, e.g.
  "methyl(phenyl)phosphinic acid" -- exactly the citation order/enclosure
  ``format_center_ligands`` already implements for every other ligand-list
  family in this codebase.
- P-67.1.3.2: esters replace the acid's single acid-oxygen with an
  O-organyl group and are named the same way phosphonate esters already
  are here -- the O-bound group as a leading separate word, e.g.
  "methyl diethylphosphinate".
- Only phosphorus is in scope for this pass. Phosphinous/phosphonous acid
  (P(III), no P=O), the analogous As/Sb rows (arsinic/arsonic/arsinous/
  arsonous, stibinic/stibonic/stibinous/stibonous per P-67.1.1/P-67.1.2),
  phosphinic amides (P-67.1.2.6, which need P-16.3.3-style N,P element
  locants -- e.g. N,N,P,P-tetramethylphosphinic amide), and bismuth (which
  P-68.3.3 explicitly has no retained acid names for) are deliberately
  deferred to a later pass.

Prior to this fix, only two shapes were recognised: the mono-substituted
acid retaining one P-H (R-P(=O)(OH)H) and phosphonic acid (R-P(=O)(OH)2,
one carbon + two acid oxygens). The fully disubstituted phosphinic acid
shape -- R(R')P(=O)(OH), two carbon substituents and no P-H -- fell through
to the generic "phosphoryl" substituent-prefix fallback on an arbitrarily
chosen carbon parent (e.g. the pre-fix name for
``O=P(O)(C1CC1)CC(C)C`` was ``(hydroxy(2-methylpropyl)phosphoryl)cyclopropane``
instead of ``cyclopropyl(2-methylpropyl)phosphinic acid``).
"""

from __future__ import annotations

import pytest

from openclatura import name as name_one
from openclatura import name_smiles


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
# Disubstituted phosphinic acid: two different substituents (P-16.5.1.3.1).
# ---------------------------------------------------------------------------

DISTINCT_SUBSTITUENT_CASES = (
    (
        "cyclopropyl-and-2-methylpropyl",
        "O=P(O)(C1CC1)CC(C)C",
        "cyclopropyl(2-methylpropyl)phosphinic acid",
    ),
    (
        "methyl-and-phenyl-blue-book-example",
        "CP(=O)(O)c1ccccc1",
        "methyl(phenyl)phosphinic acid",
    ),
    (
        "isopropyl-and-cyclopropyl",
        "CC(C)P(=O)(O)C1CC1",
        "cyclopropyl(propan-2-yl)phosphinic acid",
    ),
    (
        "methyl-and-locant-bearing-chloropropyl",
        "O=P(O)(C)CC(C)Cl",
        "(2-chloropropyl)(methyl)phosphinic acid",
    ),
    (
        "ring-substituent-and-methyl",
        "CC1CCC(C)CC1P(=O)(O)C",
        "(2,5-dimethylcyclohexyl)(methyl)phosphinic acid",
    ),
)


@pytest.mark.parametrize(
    ("_case", "smiles", "expected_name"),
    DISTINCT_SUBSTITUENT_CASES,
    ids=[c for c, _s, _n in DISTINCT_SUBSTITUENT_CASES],
)
def test_disubstituted_phosphinic_acid_distinct_substituents(_case, smiles, expected_name):
    _assert_matched(smiles, expected_name)


# ---------------------------------------------------------------------------
# Disubstituted phosphinic acid: identical substituents -- plain "di-"
# multiplying prefix, no P-locants, no parenthesised repetition.
# ---------------------------------------------------------------------------

IDENTICAL_SUBSTITUENT_CASES = (
    ("diethylphosphinic-acid", "CCP(=O)(O)CC", "diethylphosphinic acid"),
    ("dibutylphosphinic-acid", "CCCCP(=O)(O)CCCC", "dibutylphosphinic acid"),
    ("diphenylphosphinic-acid", "c1ccccc1P(=O)(O)c1ccccc1", "diphenylphosphinic acid"),
)


@pytest.mark.parametrize(
    ("_case", "smiles", "expected_name"),
    IDENTICAL_SUBSTITUENT_CASES,
    ids=[c for c, _s, _n in IDENTICAL_SUBSTITUENT_CASES],
)
def test_disubstituted_phosphinic_acid_identical_substituents(_case, smiles, expected_name):
    _assert_matched(smiles, expected_name)
    assert "P," not in expected_name  # no P,P- element-locant style
    assert "(" not in expected_name  # a plain multiplied ligand needs no parens


# ---------------------------------------------------------------------------
# Mono-substituted phosphinic acid (R-P(=O)(OH)H, one retained P-H): the
# pre-existing case, now additionally handling ligands _alkyl_ligand_name
# alone couldn't (rings, heteroatom-containing chains) via the same
# _phosphorus_ligand_name fallback the disubstituted case uses.
# ---------------------------------------------------------------------------


def test_mono_substituted_phosphinic_acid_simple_alkyl_ligand_is_unchanged():
    _assert_matched("CCP(=O)O", "ethylphosphinic acid")


def test_mono_substituted_phosphinic_acid_now_handles_a_heteroatom_containing_ligand():
    """Previously fell through to '1-fluoro-3-(hydroxyphosphinoyl)propane'
    because the mono-case's ligand namer rejected any non-carbon atom."""

    _assert_matched("FCCCP(=O)O", "(3-fluoropropyl)phosphinic acid")


# ---------------------------------------------------------------------------
# Phosphinate esters (P-67.1.3.2): O-bound group as a leading separate word.
#
# Each of these previously raised FinalAssemblyAuditError under
# name_one(..., verify_opsin=True) ("unnamed atoms" for the ester's O-alkyl
# group -- neither the ligand binding nor the core binding covered it).
# organophosphinic_acid_result now emits a dedicated
# "organophosphinic_ester_modifier" NameAtomBinding for those atoms, so
# _assert_matched (which exercises that stricter audited path) can be used
# directly here instead of falling back to the unaudited name_smiles().
# ---------------------------------------------------------------------------


def test_disubstituted_phosphinate_ester_blue_book_example():
    _assert_matched("CCP(=O)(OC)CC", "methyl diethylphosphinate")


def test_disubstituted_phosphinate_ester_distinct_ligands():
    _assert_matched("O=P(OC)(C1CC1)CC(C)C", "methyl cyclopropyl(2-methylpropyl)phosphinate")


def test_mono_substituted_phosphinate_ester():
    """The ester analogue of the mono-substituted acid: also newly resolved
    (previously '(ethoxyphosphinoyl)benzene', prefix-style fallback)."""

    _assert_matched("CCOP(=O)c1ccccc1", "ethyl phenylphosphinate")


# ---------------------------------------------------------------------------
# The SAME "unnamed atoms" audit gap, pre-existing in
# organophosphonic_acid_result (not something this pass's phosphinic acid
# work introduced, but fixed alongside it since it's the identical pattern):
# phosphonate esters -- mono, identical-diester, and mixed-diester.
# ---------------------------------------------------------------------------


def test_phosphonate_mono_ester_audit_gap_is_fixed():
    _assert_matched("CCOP(=O)(O)c1ccccc1", "ethyl hydrogen phenylphosphonate")


def test_phosphonate_identical_diester_audit_gap_is_fixed():
    _assert_matched("COP(=O)(OC)c1ccccc1", "dimethyl phenylphosphonate")


def test_phosphonate_mixed_diester_audit_gap_is_fixed():
    _assert_matched("COP(=O)(OCC)c1ccccc1", "ethyl methyl phenylphosphonate")


# NOTE: a charge-separated P(+)-O(-) acid-oxygen SMILES (e.g. C[P+]([O-])(O)O
# for what's meant as methylphosphonic acid, rather than the far more common
# neutral CP(=O)(O)O spelling) surfaced a separate "charged atoms not
# represented" audit error while investigating the gap above -- but unlike
# that one, it's not clearly just a binding-metadata omission: OPSIN's own
# structural comparison also flags a mismatch for that input once the audit
# is bypassed, so whether this is a real (if obscure) naming defect or a
# resonance-form-comparison limitation elsewhere is genuinely unclear and
# wasn't pinned down. Left unasserted here rather than guessing.


# ---------------------------------------------------------------------------
# Guard rails: cases that must NOT be claimed by organophosphinic_acid_result.
# ---------------------------------------------------------------------------


def test_cyclic_phosphorus_is_not_claimed_the_ring_stays_the_parent():
    """A ring P(=O)(OH) must keep the ring as parent (P-44: rings outrank
    chains/mononuclear parents) rather than have this shortcut 'open' it."""

    result = name_one("O=P1(O)CCCC1", verify_opsin=True)
    assert result.name == "1-hydroxy-1-oxophospholane"
    assert result.opsin_check.status == "matched"


def test_a_more_senior_acid_elsewhere_still_outranks_phosphinic_acid():
    """When a carboxylic acid is also present, IT is the principal
    characteristic group and the phosphinic-acid-shaped centre is
    correctly demoted back to a substituent prefix."""

    result = name_one("OC(=O)CP(=O)(O)CC", verify_opsin=True)
    assert result.name == "2-(ethyl(hydroxy)phosphoryl)acetic acid"
    assert result.opsin_check.status == "matched"


def test_atom_order_invariance():
    """The same molecule, atoms written in a different order, must produce
    the identical name (ligand ordering must depend on ligand identity,
    not incidental SMILES atom traversal)."""

    a = name_smiles("O=P(O)(C1CC1)CC(C)C")
    b = name_smiles("CC(C)CP(=O)(O)C1CC1")
    assert a == b == "cyclopropyl(2-methylpropyl)phosphinic acid"
