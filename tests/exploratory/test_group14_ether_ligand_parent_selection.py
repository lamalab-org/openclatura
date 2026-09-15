"""Regression tests for parent-selection bugs in
``simple_central_parent_hydride_result`` (special_cases.py), found while
investigating why ``CC[Si](CC)(COC)COC`` (two ethyl + two methoxymethyl on
silicon) named as ``1-(ethylbis(methoxymethyl)silyl)ethane`` instead of
``diethylbis(methoxymethyl)silane``.

Bug 1 -- ether ligands wrongly blocked the shortcut entirely (FIXED):
a Group 14/13/15 hydride centre (silane, germane, borane, phosphane, ...)
bonded only to simple hydrocarbyl/halogen ligands is correctly named with
the central element as parent -- e.g. ``diethyldimethylsilane`` for
Si(CC)(CC)(C)(C). But the moment even ONE ligand contained an ether
oxygen anywhere in it (e.g. methoxymethyl, -CH2-O-CH3), the whole special
case silently bailed out and fell through to a materially worse fallback
name (an arbitrary ethyl ligand promoted to parent instead of silicon).

Root cause: ``_has_principal_group`` (which gates ``hydrocarbyl_allowed``
-- the flag that lets any ligand beyond a bare halogen or lone methyl
recurse through the full branch namer) checked whether a perceived
group's KEY was registered with ``role: "principal"`` in the
functional-groups table, rather than checking that specific instance's
``is_principal_candidate`` flag. "ether" IS registered
``role: "principal"`` (it needs a seniority/suffix entry for other
purposes) but its perception detector (perception.py's
``_builtin_perceive_groups``) always marks every actual ether instance
``is_principal_candidate=False``, since IUPAC substitutive nomenclature
never cites a plain ether as a suffix -- it's always the "oxy"/"alkoxy"
prefix. So any molecule with an ether ligand looked, to
``_has_principal_group``, exactly like it had a competing principal
characteristic group, when it never actually could. Confirmed "ether" is
the ONLY functional-group key with this exact
role="principal"-but-instance-is_principal_candidate=False mismatch:
other groups whose perception detector hardcodes
``is_principal_candidate=False`` (nitro, nitroso, azido, isocyanato,
thiocyanato, cyanato, isocyano, thioether) are all registered
``role: "prefix"`` or unregistered, so this bug never affected them.

Bug 2 -- a ring-ligand guard was ADDED then RETRACTED as based on a false
premise: an earlier draft of this fix set (motivated by
``C[Si](C)(C)C1CC1`` naming as ``cyclopropyltrimethylsilane`` rather than
``(trimethylsilyl)cyclopropane``) assumed P-44.1.1's "a ring is senior to
a chain" extends to a mononuclear parent hydride (silane/germane/borane/
phosphane/...) with a ring SUBSTITUENT, and added a guard making
``simple_central_parent_hydride_result`` refuse to fire whenever any ring
was present anywhere in the component. That premise is WRONG, proven by
extremely well-established, universally-accepted nomenclature it silently
broke: "triphenylphosphine"/"triphenylphosphane" (not
"(diphenylphosphanyl)benzene"), "dimethylphenylphosphine" (not
"(dimethylphosphanyl)benzene") are standard, correct names with the
mononuclear hydride as parent despite the attached ring(s). This is also
consistent with the sibling ``homonuclear_chain_parent_result``, which
keeps a homonuclear element chain (disulfane, disilane, ...) as parent
even with ring ligands on it, confirmed for both a pre-existing symmetric
case ("1,2-diphenyldisulfane") and an asymmetric one
("1-cyclopropyl-2-phenyldisulfane"). P-44.1.1's ring-vs-chain seniority
evidently governs choices among candidate all-carbon parent structures,
not a heteroatom mononuclear/homonuclear element hydride versus a ring
substituent attached to it. The guard was removed; ``cyclopropyltrimethyl
silane`` (chain-as-parent) is the correct, restored behaviour. See the
retracted RING_LIGAND_CASES below, now asserting the original hydride-as-
parent naming instead.

Bug 3 -- a stereocentre on the shortcut's own central atom was silently
dropped (FIXED, unrelated to bug 2): this special case names a mononuclear
hydride centre with its ligands as prefixes -- e.g. "chloro(ethyl)
(isopropoxy)phosphane oxide" -- but has no machinery to express a
stereo-descriptor on the CENTRE itself. When the centre is a genuine
stereocentre (e.g. a chiral phosphine oxide/phosphonochloridate,
CC[P@](=O)(Cl)OC(C)C), firing this shortcut silently drops the
configuration, since the general substituent-prefix pipeline that WOULD
express it never gets a chance to run. Fixed by refusing to fire when the
centre has a stereo/raw_stereo tag, regardless of whether a ring is
anywhere in the component -- confirmed the stereo guard alone (with no
ring-ligand guard at all) is sufficient even when the only ring is
several bonds away through an ether linker (see
test_phosphorus_stereocentre_survives_a_distant_ring_regardless below).
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
# The reported case and its minimal-pair variants: an ether-containing
# ligand (methoxymethyl) alongside plain alkyl ligands on silicon.
# ---------------------------------------------------------------------------

ETHER_LIGAND_CASES = (
    (
        "two-ether-ligands-two-ethyl",
        "CC[Si](CC)(COC)COC",
        "diethylbis(methoxymethyl)silane",
    ),
    (
        "one-ether-ligand-three-ethyl",
        "CC[Si](CC)(COC)CC",
        "triethyl(methoxymethyl)silane",
    ),
    (
        "one-ether-ligand-three-methyl",
        "C[Si](C)(C)COC",
        "(methoxymethyl)trimethylsilane",
    ),
)


@pytest.mark.parametrize(
    ("_case", "smiles", "expected_name"),
    ETHER_LIGAND_CASES,
    ids=[c for c, _s, _n in ETHER_LIGAND_CASES],
)
def test_silicon_with_an_ether_ligand_still_becomes_the_parent(_case, smiles, expected_name):
    _assert_matched(smiles, expected_name)


# ---------------------------------------------------------------------------
# Controls: all-hydrocarbon ligand sets must be completely unaffected by
# the fix (they never triggered the bug, since they have no ether).
# ---------------------------------------------------------------------------

ALL_CARBON_CONTROL_CASES = (
    ("tetraethylsilane", "CC[Si](CC)(CC)CC", "tetraethylsilane"),
    ("diethyldimethylsilane", "CC[Si](CC)(C)C", "diethyldimethylsilane"),
    ("diethyldipropylsilane", "CC[Si](CC)(CCC)CCC", "diethyldipropylsilane"),
    (
        "four-distinct-hydrocarbyl-ligands",
        "[Si](C)(CC)(CCC)CCCC",
        "butyl(ethyl)(methyl)(propyl)silane",
    ),
)


@pytest.mark.parametrize(
    ("_case", "smiles", "expected_name"),
    ALL_CARBON_CONTROL_CASES,
    ids=[c for c, _s, _n in ALL_CARBON_CONTROL_CASES],
)
def test_all_hydrocarbon_silanes_are_unaffected_by_the_fix(_case, smiles, expected_name):
    _assert_matched(smiles, expected_name)


# ---------------------------------------------------------------------------
# The fix must not resurrect ether as a real competing principal group: a
# molecule where something else genuinely IS a principal group (elsewhere
# in the same component) must still correctly demote the hydride centre.
# ---------------------------------------------------------------------------


def test_a_genuine_principal_group_elsewhere_still_blocks_the_hydride_parent():
    """The carboxylic acid here is a REAL competing principal group (unlike
    ether), so silicon must still be demoted to a substituent prefix."""

    result = name_one("OC(=O)C[Si](C)(C)C", verify_opsin=True)
    assert result.opsin_check.status == "matched"
    assert "silane" not in result.name


# ---------------------------------------------------------------------------
# Directly targets the root cause: _has_principal_group must key off the
# per-instance is_principal_candidate flag, not the registry role.
# ---------------------------------------------------------------------------


def test_has_principal_group_ignores_ether_directly():
    from openclatura.graph_io import read_smiles
    from openclatura.special_cases import _has_principal_group

    mol = read_smiles("CC[Si](CC)(COC)COC")
    assert _has_principal_group(mol, set(mol.atoms)) is False


def test_atom_order_invariance():
    a = name_smiles("CC[Si](CC)(COC)COC")
    b = name_smiles("COC[Si](COC)(CC)CC")
    assert a == b == "diethylbis(methoxymethyl)silane"


# ---------------------------------------------------------------------------
# Retracted bug 2: a ring-containing ligand does NOT disqualify the
# mononuclear hydride as parent. These cases previously asserted the
# opposite (ring-as-parent) under the now-disproven premise above; they now
# assert the original, correct, hydride-as-parent naming, confirmed against
# standard nomenclature (triphenylphosphine et al.) and OPSIN round-trip.
# ---------------------------------------------------------------------------

RING_LIGAND_CASES = (
    (
        "plain-cyclopropyl-no-ether",
        "C[Si](C)(C)C1CC1",
        "cyclopropyltrimethylsilane",
    ),
    (
        "cyclopropyl-plus-ether",
        "CO[Si](C)(C)C1CC1",
        "cyclopropyl(methoxy)dimethylsilane",
    ),
    (
        "two-phenyl-ligands-on-silicon",
        "c1ccccc1[Si](c1ccccc1)(C)C",
        "dimethyldiphenylsilane",
    ),
    (
        "three-phenyl-ligands-on-phosphorus",
        "c1ccccc1P(c1ccccc1)c1ccccc1",
        "triphenylphosphane",
    ),
    (
        "two-cyclopropyl-ligands-on-silicon",
        "C1CC1[Si](C1CC1)(C)C",
        "dicyclopropyldimethylsilane",
    ),
    (
        "ring-two-bonds-away-through-an-ether-linker",
        "CP(C)(=O)COCc1ccccc1",
        "((benzyloxy)methyl)dimethylphosphane oxide",
    ),
)


@pytest.mark.parametrize(
    ("_case", "smiles", "expected_name"),
    RING_LIGAND_CASES,
    ids=[c for c, _s, _n in RING_LIGAND_CASES],
)
def test_ring_ligand_does_not_outrank_the_mononuclear_hydride_parent(_case, smiles, expected_name):
    _assert_matched(smiles, expected_name)


def test_phosphorus_stereocentre_survives_a_distant_ring_regardless():
    """A chiral phosphine oxide whose only ring is two bonds away through a
    -CH2-O-CH2- ether linker. The stereo guard alone (no ring-ligand guard)
    must still refuse the shortcut, since the "phosphane oxide" mononuclear-
    parent style has no stereo-descriptor machinery -- (S) must survive via
    the general, ring-agnostic substituent-prefix pipeline instead."""

    result = name_one("CC[C@H](C)C[P@](C)(=O)COCc1ccccc1", verify_opsin=True)
    assert "(S)" in result.name
    assert result.opsin_check.status == "matched", (result.name, result.opsin_check.status)


def test_simple_central_parent_hydride_result_refuses_a_stereocentre_regardless_of_rings():
    from openclatura.graph_io import read_smiles
    from openclatura.special_cases import simple_central_parent_hydride_result

    mol = read_smiles("CC[C@H](C)C[P@](C)(=O)COCc1ccccc1")
    assert simple_central_parent_hydride_result(mol, set(mol.atoms)) is None
