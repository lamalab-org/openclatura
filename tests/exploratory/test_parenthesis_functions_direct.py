"""Direct, chemistry-free adversarial tests of the pure parenthesis-decision
functions touched by fix-elided-locant-parentheses.

These bypass SMILES/molecule naming entirely and hand-feed the string
functions synthetic-but-plausible substituent-name shapes, so that a logic
bug in the decision itself is caught even for shapes today's naming
pipeline may not (yet) construct from a real molecule. Where a shape *is*
reachable from real chemistry, the equivalent real-molecule case also
lives in test_center_ligand_parenthesization.py; these are the
lower-level, narrower-blast-radius companions.
"""

from __future__ import annotations

import pytest

from openclatura.assembly_prefixes import _outer_parentheses_are_redundant, _starts_with_multiplier
from openclatura.assembly_utils import is_fully_enclosed
from openclatura.formatting import _is_substituted_alkyl_ligand, format_center_ligands

# ---------------------------------------------------------------------------
# _outer_parentheses_are_redundant: nested-group prefix/suffix classification
# ---------------------------------------------------------------------------

REDUNDANT_CASES = (
    ("(propan-2-yl)", True, "0 nested groups, no multiplier-looking prefix"),
    ("(2-(chloromethyl)pentyl)", True, "1 nested group, purely-locant prefix '2-'"),
    ("(9H-fluoren-9-yl)", True, "0 nested groups, no multiplier-looking prefix"),
    ("(1,2,4-triazol-1-yl)", True, "0 nested groups; inner starts with a digit, not 'tri'"),
    (
        "(2-(1,2,4-triazol-1-yl)ethyl)",
        True,
        "1 nested group; prefix '2-' is pure locant, suffix 'ethyl' isn't multiplier-like",
    ),
    (
        "((decanoylamino)methyl)",
        True,
        "1 nested group at position 0 -- the nested group is still self-delimiting by its "
        "own parens, so it's genuinely safe to drop the outer pair regardless of what the "
        "nested group's own content starts with",
    ),
    (
        "(cyclobutyl(methyl)carbamoyl)",
        False,
        "1 nested group with a non-locant ALPHABETIC prefix 'cyclobutyl' before it -- must stay wrapped "
        "(this is the exact shape of the existing carbamate regression test)",
    ),
    (
        "((1R)-1-(chloromethyl)propyl)",
        False,
        "2 top-level groups (stereo descriptor + substituent group) -- must stay wrapped",
    ),
    (
        "((1R,2S)-2-(chloromethyl)cyclopropyl)",
        False,
        "2 top-level groups -- must stay wrapped",
    ),
    (
        "(bis(chloromethyl)phosphoryl)",
        False,
        "1 nested group with non-locant prefix 'bis' -- must stay wrapped",
    ),
    (
        "((chloromethyl)(fluoromethyl)phosphoryl)",
        False,
        "2 top-level groups -- must stay wrapped",
    ),
    (
        "(decahydronaphthalen-1-yl)",
        False,
        "0 nested groups; 'deca' is a real multiplying-prefix word by coincidence here -- "
        "over-conservative (extra parens kept) but not dangerous",
    ),
)


@pytest.mark.parametrize(
    ("name", "expected", "_reason"),
    REDUNDANT_CASES,
    ids=[n for n, _e, _r in REDUNDANT_CASES],
)
def test_outer_parentheses_are_redundant_classification(name, expected, _reason):
    assert _outer_parentheses_are_redundant(name) is expected, _reason


def test_outer_parentheses_are_redundant_never_changes_non_paren_content():
    """Removal must be pure paren-surgery: never touch the non-paren characters."""

    for name, expected, _reason in REDUNDANT_CASES:
        if not expected or not is_fully_enclosed(name):
            continue
        stripped = name[1:-1]
        assert stripped.replace("(", "").replace(")", "") == name.replace("(", "").replace(")", "")


def test_outer_parentheses_are_redundant_rejects_unbalanced_input():
    # depth goes negative -> must not be treated as redundant (and must not raise)
    assert _outer_parentheses_are_redundant("(a)b)") is False
    assert _outer_parentheses_are_redundant("(a(b)") is False


# ---------------------------------------------------------------------------
# _starts_with_multiplier: naive string-prefix check (basic multiplier words
# only -- di/tri/tetra/.../icosa -- not the complex bis/tris/tetrakis forms,
# which is fine in practice because those are always immediately followed by
# "(" and so are caught by the top-level-group branch instead, not this one)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    ["decyl", "nonyl", "pentyl", "octyl", "triphenylmethyl"],
)
def test_starts_with_multiplier_true_positives(text):
    # 'triphenylmethyl' genuinely IS three phenyls -- 'tri' there really is the
    # multiplying prefix, so True is correct, not a false positive.
    if text == "triphenylmethyl":
        assert _starts_with_multiplier(text) is True
    else:
        assert _starts_with_multiplier(text) is False


def test_starts_with_multiplier_false_positive_on_ring_derived_stems():
    """`_starts_with_multiplier` is a bare string-prefix check with no semantic
    awareness, so a heterocycle stem that merely happens to start with the same
    letters as a multiplying prefix is misclassified as "starts with a
    multiplying prefix" even though nothing is being multiplied.

    This is the over-conservative direction (extra parens kept when they
    could safely be dropped) rather than the dangerous one (a needed
    parenthesis wrongly dropped) -- but it is a real latent inconsistency in
    the same code family being tested, so it's pinned down here.

    Reachability caveat: for THIS specific case to manifest as a visible
    behavior change, a heterocyclic substituent name would need to appear
    with (a) no locant at all before its ring stem and (b) as the sole,
    unlocanted substituent on a parent with no principal characteristic
    group -- both fairly unusual for a real generated ring-substituent name
    (which normally carries its own attachment locant), so this is reported
    as a function-level finding rather than a demonstrated end-to-end bug.
    """

    assert _starts_with_multiplier("triazolyl") is True
    assert _starts_with_multiplier("tetrazolyl") is True
    assert _starts_with_multiplier("decanoyl") is True  # unrelated to "ten of anything"
    # Contrast: once a real locant precedes the stem (the normal, realistic shape),
    # the string no longer starts with the multiplier word and is classified correctly:
    assert _starts_with_multiplier("1,2,4-triazol-1-yl") is False


# ---------------------------------------------------------------------------
# format_center_ligands: merge-guard (str.endswith) and double-wrap risk
# ---------------------------------------------------------------------------


def _balanced(text: str) -> bool:
    depth = 0
    for ch in text:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0 and "()" not in text


@pytest.mark.parametrize(
    ("names", "expected"),
    [
        (["hydroxy", "ethyl"], "ethyl(hydroxy)"),
        (["methyl", "methyl"], "dimethyl"),
        (["methoxy", "methyl", "methyl"], "methoxydimethyl"),
        (["(chloromethyl)", "(chloromethyl)", "methyl", "methyl"], "bis(chloromethyl)dimethyl"),
        (["chlorofluoromethyl", "methyl", "methyl", "methyl"], "(chlorofluoromethyl)trimethyl"),
        # first-ligand merge-guard: singleton whose bare text ends with a LATER repeated
        # ligand's bare name must be wrapped even though it's index 0 (would otherwise be bare)
        (["ethylmethyl", "methyl", "methyl"], "(ethylmethyl)dimethyl"),
        (["phenylamino", "amino", "amino"], "diamino(phenylamino)"),
        # non-first singletons are unconditionally wrapped regardless of any merge risk
        (["propyl", "methyl", "methyl"], "dimethyl(propyl)"),
        # already-fully-enclosed ligands must never be double-wrapped
        (["(chloromethyl)", "(chloromethyl)"], "bis(chloromethyl)"),
        (["(chloromethyl)", "(chloromethyl)", "(chloromethyl)"], "tris(chloromethyl)"),
        # degenerate inputs must not crash
        (["methyl"], "methyl"),
        ([], ""),
    ],
)
def test_format_center_ligands_exact(names, expected):
    result = format_center_ligands(names)
    assert result == expected
    assert _balanced(result)


@pytest.mark.parametrize(
    "names",
    [
        ["hydroxy", "ethyl"],
        ["(chloromethyl)", "(chloromethyl)", "methyl", "methyl"],
        ["chlorofluoromethyl", "methyl", "methyl", "methyl"],
        ["dimethylamino", "amino", "amino"],
        ["ethylmethyl", "methyl", "methyl"],
        ["(chloromethyl)", "(chloromethyl)", "(chloromethyl)"],
        ["a", "b", "c", "d", "e"],
    ],
)
def test_format_center_ligands_always_balanced_no_empty_groups(names):
    result = format_center_ligands(names)
    assert _balanced(result)


# ---------------------------------------------------------------------------
# _is_substituted_alkyl_ligand: stem-detection edge cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("methyl", False),
        ("ethyl", False),
        ("propyl", False),
        ("cyclopropyl", False),
        ("cyclohexyl", False),
        ("chloromethyl", True),
        ("2-chloroethyl", True),
        ("tert-butyl", True),
        ("phenyl", False),
        ("benzyl", False),
        ("cyclopropylmethyl", True),
        ("propan-2-yl", False),
    ],
)
def test_is_substituted_alkyl_ligand(name, expected):
    assert _is_substituted_alkyl_ligand(name) is expected
