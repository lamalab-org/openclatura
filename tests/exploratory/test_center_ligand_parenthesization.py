"""Adversarial, hand-crafted exploratory tests for the ligand/substituent
parenthesis-boundary logic reworked on branch fix-elided-locant-parentheses.

Scope
-----
The branch touches two closely related pieces of string-level logic:

1. ``openclatura.assembly_prefixes._omit_unlocanted_outer_parentheses`` /
   ``_outer_parentheses_are_redundant`` -- decides whether a *lone*,
   unlocanted substituent's own outer parentheses can be dropped (e.g.
   ``(propan-2-yl)cyclopropane`` keeps its parens because "2" would
   otherwise dangle, but other shapes safely lose theirs).

2. ``openclatura.formatting.format_center_ligands`` -- decides how a list of
   ligands cited on *one* central atom (Si/Ge/Sn/Pb hydrides, P/As/Sb/Bi
   pnictogens including lambda-phosphorus, S/Se/Te chalcogens, N-centered
   amide/ammonio/iminium/hydrazine/azanidyl prefixes, B/halogen "-ate"
   centers) get parenthesised relative to each other, replacing several
   near-duplicate call sites that used to each rewrap this differently.

Every case below is verified two ways: an exact generated-name assertion
(so any change to the parenthesisation itself is caught immediately) and an
independent OPSIN round-trip (name -> structure -> compare to the original
SMILES), which is the ground-truth check that a parenthesis that got
dropped was genuinely redundant rather than load-bearing.

These were designed by directly attacking the specific heuristics in the
functions above (their docstrings and inline comments explain each
mechanism), not derived from any pre-existing failure -- see the module's
sibling file ``test_fixed_parenthesization_bugs.py`` for the handful of real
(but pre-existing, not introduced by this branch) issues that turned up
during that search and were fixed alongside it.
"""

from __future__ import annotations

import pytest

from openclatura import name as name_one
from openclatura import name_smiles


def _assert_balanced_and_matched(smiles: str, expected_name: str | None = None) -> str:
    """Name ``smiles``, assert balanced/non-empty parens, exact name (if given), and OPSIN match."""

    result = name_one(smiles, verify_opsin=True)
    assert result.error is None, (smiles, result.error)
    assert result.name, smiles

    depth = 0
    for ch in result.name:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            assert depth >= 0, (smiles, result.name)
    assert depth == 0, (smiles, result.name)
    assert "()" not in result.name, (smiles, result.name)

    if expected_name is not None:
        assert result.name == expected_name, (smiles, result.name, expected_name)

    assert result.opsin_check is not None
    assert result.opsin_check.status == "matched", (
        smiles,
        result.name,
        result.opsin_check.status,
        result.opsin_check.opsin_smiles,
        result.opsin_check.canonical_original,
        result.opsin_check.canonical_roundtrip,
    )
    return result.name


# ---------------------------------------------------------------------------
# Group 14 hydride centres (Si/Ge/Sn/Pb): format_center_ligands via the
# substituent_sort_key-protected `_grouped_ligand_prefix` wrapper.
# ---------------------------------------------------------------------------

GROUP14_CASES = (
    (
        "four-distinct-including-tert-butyl",
        "[Si](CCl)(CF)(C(C)C)C(C)(C)C",
        "(tert-butyl)(chloromethyl)(fluoromethyl)(propan-2-yl)silane",
    ),
    (
        "singleton-plus-repeated-pair-plus-plain-methyl",
        "[Si](C(Cl)(F)F)(CF)(CF)C",
        "(chlorodifluoromethyl)bis(fluoromethyl)(methyl)silane",
    ),
    (
        "nested-silyl-ligand-one-level-down",
        "C[Si](C)(C)C[Si](C)(C)CCl",
        "chloro(dimethyl((trimethylsilyl)methyl)silyl)methane",
    ),
    (
        "real-tbs-ether-on-chiral-chloropropanol",
        "CC(Cl)CO[Si](C)(C)C(C)(C)C",
        "1-(((tert-butyl)dimethylsilyl)oxy)-2-chloropropane",
    ),
    (
        "germanium-four-distinct-analogue",
        "[Ge](CCl)(CF)(C(C)C)C(C)(C)C",
        "(tert-butyl)(chloromethyl)(fluoromethyl)(propan-2-yl)germane",
    ),
    (
        "tin-four-distinct-analogue",
        "[Sn](CCl)(CF)(C(C)C)C(C)(C)C",
        "(tert-butyl)(chloromethyl)(fluoromethyl)(propan-2-yl)stannane",
    ),
    (
        "two-distinct-repeated-pairs",
        "[Si](CCl)(CCl)(CF)CF",
        "bis(chloromethyl)bis(fluoromethyl)silane",
    ),
    (
        "three-identical-cf3-plus-methyl",
        "[Si](C(F)(F)F)(C(F)(F)F)(C(F)(F)F)C",
        "methyltris(trifluoromethyl)silane",
    ),
    (
        "three-identical-complex-alkoxy-plus-methyl",
        "C[Si](OC(C)(C)C)(OC(C)(C)C)OC(C)(C)C",
        "tris(tert-butoxy)(methyl)silane",
    ),
    (
        "four-distinct-monohalomethyl-alphabetical-order",
        "[Si](CBr)(CI)(CF)CCl",
        "(bromomethyl)(chloromethyl)(fluoromethyl)(iodomethyl)silane",
    ),
)


@pytest.mark.parametrize(
    ("_case", "smiles", "expected_name"),
    GROUP14_CASES,
    ids=[c for c, _s, _n in GROUP14_CASES],
)
def test_group14_hydride_ligand_lists(_case, smiles, expected_name):
    _assert_balanced_and_matched(smiles, expected_name)


# ---------------------------------------------------------------------------
# Group 15 pnictogen centres (P/As/Sb/Bi): format_center_ligands via
# name_pnictogen_subgraph / format_lambda_substituent. These call sites now
# correctly pass sort_key=substituent_sort_key (see
# test_fixed_parenthesization_bugs.py for the ordering bug this fixed).
# Disubstituted phosphinic-acid-shaped molecules (P bonded to two carbons,
# one =O, one -OH/-OR) are matched by organophosphinic_acid_result before
# ever reaching this ligand-list code, so they live in
# test_phosphinic_acid.py instead of here.
# ---------------------------------------------------------------------------

GROUP15_CASES = (
    (
        "trivalent-phosphine-three-distinct",
        "P(C)(CC)CCC",
        "ethyl(methyl)(propyl)phosphane",
    ),
    (
        "phosphonium-triple-repeat-plus-complex-singleton",
        "[P+](C)(C)(C)C(F)(F)Cl",
        "(chlorodifluoromethyl)trimethylphosphanium",
    ),
    ("arsine-three-distinct", "[As](C)(CC)CCC", "ethyl(methyl)(propyl)arsane"),
    ("stibine-three-distinct", "[Sb](C)(CC)CCC", "ethyl(methyl)(propyl)stibane"),
    ("bismuthine-three-distinct", "[Bi](C)(CC)CCC", "ethyl(methyl)(propyl)bismuthane"),
    (
        "phosphinimine-like-p-bearing-imino",
        "CCCP(=N)(C)CC",
        "1-(ethyl(imino)(methyl)phosphanyl)propane",
    ),
    (
        "pentavalent-phosphorane-five-distinct",
        "P(C)(CC)(CCC)(CCCC)F",
        "butyl(ethyl)(fluoro)(methyl)(propyl)-lambda5-phosphane",
    ),
    (
        "phosphonium-triple-repeat-plus-plain-methyl",
        "[P+](CCl)(CCl)(CCl)C",
        "tris(chloromethyl)(methyl)phosphanium",
    ),
)


@pytest.mark.parametrize(
    ("_case", "smiles", "expected_name"),
    GROUP15_CASES,
    ids=[c for c, _s, _n in GROUP15_CASES],
)
def test_group15_pnictogen_ligand_lists(_case, smiles, expected_name):
    _assert_balanced_and_matched(smiles, expected_name)


# ---------------------------------------------------------------------------
# Sulfur / selenium / tellurium centres.
# ---------------------------------------------------------------------------

CHALCOGEN_CASES = (
    (
        "sulfoximine-like-two-carbon-ligands",
        "CCS(=O)(=N)C",
        "1-(methylsulfonimidoyl)ethane",
    ),
    (
        "selenium-lambda-center-three-ligands",
        "[Se](c1ccccc1)(C)CC",
        "(ethyl(methyl)-lambda^3-selanyl)benzene",
    ),
    (
        "tellurium-lambda-center-three-ligands",
        "[Te](c1ccccc1)(C)CC",
        "(ethyl(methyl)-lambda^3-tellanyl)benzene",
    ),
    (
        "sulfamic-acid-two-distinct-n-ligands",
        "CCN(C)S(=O)(=O)O",
        "ethyl(methyl)sulfamic acid",
    ),
    (
        "sulfamic-acid-locant-bearing-n-ligand",
        "CN(CC(C)Cl)S(=O)(=O)O",
        "(2-chloropropyl)(methyl)sulfamic acid",
    ),
    (
        "sulfimide-phenyl-plus-ethyl-plus-imino",
        "N=S(c1ccccc1)CC",
        "(ethyl(imino)sulfanyl)benzene",
    ),
    (
        "sulfimide-two-complex-carbon-ligands",
        "N=S(CC(C)Cl)C(F)Cl",
        "2-chloro-1-((chlorofluoromethyl)(imino)sulfanyl)propane",
    ),
    (
        "n-methyl-sulfoximine-like-with-phenyl",
        "O=S(c1ccccc1)=NC",
        "N-(phenylsulfinylidene)methanamine",
    ),
)


@pytest.mark.parametrize(
    ("_case", "smiles", "expected_name"),
    CHALCOGEN_CASES,
    ids=[c for c, _s, _n in CHALCOGEN_CASES],
)
def test_chalcogen_ligand_lists(_case, smiles, expected_name):
    _assert_balanced_and_matched(smiles, expected_name)


# ---------------------------------------------------------------------------
# Boron centres and hypervalent-halogen "-ate"/lambda centres.
# ---------------------------------------------------------------------------

BORON_HALOGEN_CASES = (
    ("trialkylborane-three-distinct", "B(C)(CC)CCC", "ethyl(methyl)(propyl)borane"),
    (
        "organoboronic-acid-two-branched-locant-bearing-ligands",
        "B(O)(CC(C)Cl)CC(C)CC",
        "(2-chloropropyl)(2-methylbutyl)borinic acid",
    ),
    (
        "hypervalent-iodine-two-identical-chlorines",
        "c1ccccc1[I](Cl)Cl",
        "(dichlorolambda^3-iodanyl)benzene",
    ),
    (
        "hypervalent-iodine-two-distinct-halogens",
        "c1ccccc1[I](Cl)F",
        "(chloro(fluoro)lambda^3-iodanyl)benzene",
    ),
    (
        "phosphaniumyl-boranuide-three-fully-distinct-p-ligands",
        "[BH3-][P+](C)(CC)CCC",
        "(ethyl(methyl)(propyl)phosphaniumyl)boranuide",
    ),
    (
        "borane-methyl-plus-repeated-chloromethyl",
        "B(C)(CCl)CCl",
        "bis(chloromethyl)(methyl)borane",
    ),
    (
        "borinic-acid-methyl-plus-locant-bearing",
        "B(O)(C)CC(C)Cl",
        "(2-chloropropyl)(methyl)borinic acid",
    ),
)


@pytest.mark.parametrize(
    ("_case", "smiles", "expected_name"),
    BORON_HALOGEN_CASES,
    ids=[c for c, _s, _n in BORON_HALOGEN_CASES],
)
def test_boron_and_halogen_center_ligand_lists(_case, smiles, expected_name):
    _assert_balanced_and_matched(smiles, expected_name)


# ---------------------------------------------------------------------------
# Nitrogen centres: carbamoyl/amide-N, ammonio, iminium, hydrazine, urea.
# (a representative subset of the full battery that was actually executed
#  and OPSIN-verified; every one of these matched)
# ---------------------------------------------------------------------------

NITROGEN_CASES = (
    (
        "carbamoyl-locant-bearing-ligand-plus-methyl",
        "CC(C)(C(=O)N(C)C(CC)CCl)NC(=O)OC",
        "methyl (2-((1-chlorobutan-2-yl)(methyl)carbamoyl)propan-2-yl)carbamate",
    ),
    (
        "ammonio-three-distinct-branched-ligands",
        "[N+](C)(CC(C)C)(C(F)Cl)CC(=O)[O-]",
        "2-(chlorofluoromethyl(methyl)(2-methylpropyl)ammonio)acetate",
    ),
    (
        "ammonio-merge-guard-chloromethyl-vs-dimethyl",
        "[N+](C)(C)(CCl)CC(=O)[O-]",
        "2-((chloromethyl)dimethylammonio)acetate",
    ),
    (
        "ammonio-merge-guard-not-independently-complex-singleton",
        "[N+](C(Cl)(F)F)(CF)(CF)CC(=O)[O-]",
        "2-((chlorodifluoromethyl)bis(fluoromethyl)ammonio)acetate",
    ),
    (
        "iminium-one-branched-one-locant-bearing",
        "C=[N+](C(C)C)CCCl",
        "N-(2-chloroethyl)-N-(propan-2-yl)methaniminium",
    ),
    (
        "hydrazine-both-nitrogens-substituted-no-crossbleed",
        "CN(CC)NC(C)CC",
        "2-(butan-2-yl)-1-ethyl-1-methylhydrazine",
    ),
    (
        "hydrazine-repeated-pair-on-one-nitrogen",
        "C(Cl)CN(CCCl)NC",
        "1,1-bis(2-chloroethyl)-2-methylhydrazine",
    ),
    (
        "urea-both-sides-locant-bearing",
        "CC(Cl)CCNC(=O)NC(C)CCl",
        "N-(3-chlorobutyl)-N'-(1-chloropropan-2-yl)urea",
    ),
    (
        "urea-both-sides-doubly-substituted-interleaved-citation",
        "CCN(C)C(=O)N(CC(C)C)CCl",
        "N-(chloromethyl)-N'-ethyl-N'-methyl-N-(2-methylpropyl)urea",
    ),
    (
        "real-drug-lidocaine",
        "CCN(CC)CC(=O)Nc1c(C)cccc1C",
        "2-(diethylamino)-N-(2,6-dimethylphenyl)acetamide",
    ),
    (
        "real-drug-procainamide",
        "CCN(CC)CCNC(=O)c1ccc(N)cc1",
        "4-amino-N-(2-(diethylamino)ethyl)benzamide",
    ),
    (
        "real-benzalkonium-type-surfactant",
        "[N+](C)(C)(Cc1ccccc1)CCCCCCCCCCCC",
        "N-benzyl-N,N-dimethyldodecan-1-aminium",
    ),
)


@pytest.mark.parametrize(
    ("_case", "smiles", "expected_name"),
    NITROGEN_CASES,
    ids=[c for c, _s, _n in NITROGEN_CASES],
)
def test_nitrogen_center_ligand_lists(_case, smiles, expected_name):
    _assert_balanced_and_matched(smiles, expected_name)


# ---------------------------------------------------------------------------
# Multiplying-prefix / ring-stem textual-collision stress: real molecules
# whose names contain ring stems ("triazol-", "tetrazol-") or acyl stems
# ("decanoyl", "nonan-") that happen to start with the same letters as a
# multiplying prefix (di/tri/tetra/deca/nona/...). For the Hantzsch-Widman
# ring stems this "collision" turned out not to be one at all -- see
# test_fixed_parenthesization_bugs.py's retraction test and docstring for
# why "tri"/"tetra" there are genuine multiplying prefixes (of a replacement
# prefix, not a substituent), not a naive string-matching false positive.
# ---------------------------------------------------------------------------

MULTIPLIER_LOOKALIKE_CASES = (
    ("methyltriazole", "Cn1cncn1", "1-methyl-1H-1,2,4-triazole"),
    ("phenyltriazole", "c1ccc(-n2cncn2)cc1", "1-phenyl-1H-1,2,4-triazole"),
    ("methyltetrazole", "Cn1nnnc1", "1-methyl-1H-tetrazole"),
    (
        "spiro-dioxane-fused-ring-system",
        "CC1CCC2(CC1)OCCO2",
        "8-methyl-1,4-dioxaspiro[4.5]decane",
    ),
    ("decanoic-acid-baseline", "O=C(O)CCCCCCCCC", "decanoic acid"),
    (
        "cyclopropyl-nonan-1-one",
        "C1CC1C(=O)CCCCCCCC",
        "1-cyclopropylnonan-1-one",
    ),
)


@pytest.mark.parametrize(
    ("_case", "smiles", "expected_name"),
    MULTIPLIER_LOOKALIKE_CASES,
    ids=[c for c, _s, _n in MULTIPLIER_LOOKALIKE_CASES],
)
def test_multiplier_lookalike_ring_and_acyl_stems(_case, smiles, expected_name):
    _assert_balanced_and_matched(smiles, expected_name)


# ---------------------------------------------------------------------------
# SMILES-order invariance: the same molecule, atoms written in a different
# order, must produce the identical name (guards against the ligand list's
# ordering/wrapping depending on incidental atom traversal rather than the
# ligands' true identities).
# ---------------------------------------------------------------------------

REORDER_INVARIANCE_PAIRS = (
    ("CC(C)(C(=O)N(CC(C)C)C1CC1)NC(=O)OC", "CC(C)(C(=O)N(C1CC1)CC(C)C)NC(=O)OC"),
    ("[Si](CBr)(CI)(CF)CCl", "[Si](CF)(CBr)(CCl)CI"),
    ("[P+](CCl)(CCl)(CCl)C", "[P+](C)(CCl)(CCl)CCl"),
)


@pytest.mark.parametrize("pair", REORDER_INVARIANCE_PAIRS)
def test_ligand_list_naming_is_atom_order_invariant(pair):
    smiles_a, smiles_b = pair
    name_a = name_smiles(smiles_a)
    name_b = name_smiles(smiles_b)
    assert name_a == name_b, (smiles_a, smiles_b, name_a, name_b)
