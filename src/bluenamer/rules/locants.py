"""
bluenamer/rules/locants.py

Locant comparison and selection logic for IUPAC nomenclature.

The "lowest set of locants" rule is applied throughout nomenclature to break
ties when multiple numbering schemes are valid. This module provides the
core comparison primitive used everywhere downstream.

References:
- IUPAC 2013 Recommendations, P-14.5 (lowest locants)
- IUPAC 2013 Recommendations, P-31.1.4.3 (locant assignment priority order)
"""

from __future__ import annotations

from dataclasses import dataclass

# Type alias: a locant set is a list of integers, e.g. [2, 4, 5] for positions
# 2, 4, and 5. Always treated as a sorted sequence when comparing.
LocantSet = list[int]


def normalize(locants: LocantSet) -> LocantSet:
    """Return a sorted copy of the locant set (ascending)."""
    return sorted(locants)


def compare(a: LocantSet, b: LocantSet) -> int:
    """Compare two locant sets per IUPAC "lowest set of locants" rule.

    Returns:
        -1 if `a` is lower (preferred) than `b`
         0 if they are equal
         1 if `a` is higher than `b`

    The rule: compare term-by-term in ascending order. The first point of
    difference determines the winner; the lower number wins.

    Examples:
        compare([2, 5],[3, 4])      -> -1   (2 < 3 at first position)
        compare([2, 4], [2, 5])      -> -1   (tie at first, 4 < 5 at second)
        compare([1, 3, 5], [1, 3, 5]) -> 0
        compare([2, 3], [2])          -> 1    (longer set loses if shorter is prefix;
                                               but in practice equal-length sets compare)
    """
    sa = normalize(a)
    sb = normalize(b)
    for x, y in zip(sa, sb):
        if x < y:
            return -1
        if x > y:
            return 1
    # All compared positions equal; shorter set is "lower" if lengths differ.
    if len(sa) < len(sb):
        return -1
    if len(sa) > len(sb):
        return 1
    return 0


def lowest(*candidates: LocantSet) -> LocantSet:
    """Return the lowest locant set among the candidates.

    Example:
        lowest([2, 5], [3, 4], [2, 4]) -> [2, 4]
    """
    if not candidates:
        raise ValueError("lowest() requires at least one candidate")
    best = candidates[0]
    for c in candidates[1:]:
        if compare(c, best) < 0:
            best = c
    return best


# ---------------------------------------------------------------------------
# Multi-criteria locant assignment
# ---------------------------------------------------------------------------
# IUPAC P-31.1.4.3 specifies that when numbering a parent chain or ring,
# locants are assigned to give the lowest set to the following features,
# applied in this order until a decision is reached:
#
#   1. Principal characteristic group(s) (the suffix-cited group)
#   2. Skeletal heteroatoms (in replacement nomenclature)
#   3. Indicated/added hydrogen
#   4. Detachable prefixes, all together (substituents + unsaturation)
#   5. Double bonds, then triple bonds (separately, double first)
#   6. Detachable prefixes cited in alphabetical order
#
# This is a multi-criteria comparison: each candidate numbering produces a
# tuple of locant sets, one per criterion. The candidate whose tuple is
# lexicographically lowest wins.


@dataclass(frozen=True)
class LocantCriteria:
    """A bundle of locant sets for a candidate numbering, ordered by
    IUPAC priority. Used to select the correct numbering direction
    when multiple are valid.

    Fields are ordered by P-31.1.4.3 priority (most important first).
    None values are skipped during comparison (treated as "not applicable").
    """

    principal_group: LocantSet | None = None
    heteroatoms: LocantSet | None = None
    indicated_hydrogen: LocantSet | None = None
    detachable_prefixes_and_unsaturation: LocantSet | None = None
    double_bonds: LocantSet | None = None
    triple_bonds: LocantSet | None = None
    detachable_prefixes_alphabetical: LocantSet | None = None


def compare_criteria(a: LocantCriteria, b: LocantCriteria) -> int:
    """Compare two LocantCriteria tuples lexicographically by IUPAC priority.

    Returns -1 if `a` wins, 1 if `b` wins, 0 if fully tied.

    A criterion is skipped when either side has no value for it (None).
    For the same molecule under two numbering schemes, both candidates
    should populate the same fields; treating a one-sided None as a tie
    on that criterion (rather than an automatic loss) prevents spurious
    decisions when the caller only fills in some fields.
    """
    fields = (
        "principal_group",
        "heteroatoms",
        "indicated_hydrogen",
        "detachable_prefixes_and_unsaturation",
        "double_bonds",
        "triple_bonds",
        "detachable_prefixes_alphabetical",
    )
    for field in fields:
        va = getattr(a, field)
        vb = getattr(b, field)
        # Skip this criterion if either side has no value to compare.
        # (Previously this branch returned a winner, which contradicted the
        # documented "Tie-break: skip" behavior and produced wrong winners
        # whenever one candidate happened to leave a field unfilled.)
        if va is None or vb is None:
            continue

        if field == "detachable_prefixes_alphabetical":
            # Compare sequentially WITHOUT sorting to preserve alphabetical
            # priority mapping (locant[i] corresponds to the i-th alphabetical
            # prefix; reordering would destroy that mapping).
            for x, y in zip(va, vb):
                if x < y:
                    return -1
                if x > y:
                    return 1
            if len(va) < len(vb):
                return -1
            if len(va) > len(vb):
                return 1
        else:
            result = compare(va, vb)
            if result != 0:
                return result
    return 0


def select_best_numbering(candidates: list[LocantCriteria]) -> int:
    """Given a list of candidate numbering schemes (each represented by its
    LocantCriteria), return the index of the winner.

    Example:
        cands =[
            LocantCriteria(principal_group=[3], double_bonds=[1]),
            LocantCriteria(principal_group=[1], double_bonds=[3]),
        ]
        select_best_numbering(cands) -> 1   # principal group locant 1 < 3
    """
    if not candidates:
        raise ValueError("select_best_numbering() requires at least one candidate")
    best_idx = 0
    for i in range(1, len(candidates)):
        if compare_criteria(candidates[i], candidates[best_idx]) < 0:
            best_idx = i
    return best_idx
