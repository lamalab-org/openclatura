"""Property tests for deeply branched ligand-parenthesis boundaries."""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from openclatura import name as name_one
from openclatura.opsin_verify import opsin_available

_OPSIN_AVAILABLE = opsin_available()

_SIMPLE_LIGANDS = (
    "C",
    "CC",
    "CCC",
    "CCl",
    "C(F)Cl",
    "C(F)F",
    "C(Cl)Cl",
    "C(F)(F)F",
    "CC(F)Cl",
)
_BRANCHED_LIGANDS = (
    "C(C)C",
    "CC(C)C",
    "C(C)(C)C",
    "C(CCl)(C(F)F)C(C)(C)C",
    "CC(CCl)(C(F)F)C(C)C",
)
_LIGANDS = _SIMPLE_LIGANDS + _BRANCHED_LIGANDS

_CENTRAL_HYDRIDES = (
    ("[Si]", 4),
    ("[Ge]", 4),
    ("[Sn]", 4),
    ("[Pb]", 4),
    ("B", 3),
    ("[Al]", 3),
    ("[Ga]", 3),
    ("P", 3),
    ("[As]", 3),
    ("[Sb]", 3),
    ("[Bi]", 3),
)


def _central_graph(center: str, ligands: list[str]) -> str:
    return center + "".join(f"({ligand})" for ligand in ligands)


@st.composite
def branched_central_hydride_pairs(draw: st.DrawFn) -> tuple[str, str]:
    """Generate one central hydride twice with different branch orderings."""

    center, arity = draw(st.sampled_from(_CENTRAL_HYDRIDES))
    ligands = [draw(st.sampled_from(_BRANCHED_LIGANDS))]
    ligands.extend(draw(st.lists(st.sampled_from(_LIGANDS), min_size=arity - 1, max_size=arity - 1)))
    permutation = draw(st.permutations(tuple(range(arity))))
    permuted = [ligands[index] for index in permutation]
    return _central_graph(center, ligands), _central_graph(center, permuted)


_FUNCTIONAL_LIGANDS = st.sampled_from(_LIGANDS)


@st.composite
def branched_functional_center_pairs(draw: st.DrawFn) -> tuple[str, str]:
    """Generate functional-group centers with two order-permuted carbon ligands."""

    left = draw(_FUNCTIONAL_LIGANDS)
    right = draw(_FUNCTIONAL_LIGANDS)
    kind = draw(st.sampled_from(("carbamoyl", "ammonio", "phosphanium", "borinic", "sulfanyl")))
    builders = {
        "carbamoyl": lambda a, b: f"O=C(O)CC(C(=O)N({a}){b})",
        "ammonio": lambda a, b: f"[N+]({a})({b})(C)CC(=O)[O-]",
        "phosphanium": lambda a, b: f"[BH3-][P+]({a})({b})C",
        "borinic": lambda a, b: f"OB({a}){b}",
        "sulfanyl": lambda a, b: f"N=S({a}){b}",
    }
    build = builders[kind]
    return build(left, right), build(right, left)


branched_ligand_graph_pairs = st.one_of(branched_central_hydride_pairs(), branched_functional_center_pairs())


def _assert_balanced_parentheses(name: str) -> None:
    depth = 0
    for character in name:
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            assert depth >= 0, name
    assert depth == 0, name
    assert "()" not in name


_STRUCTURAL_SETTINGS = settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)


@pytest.mark.fuzz
@_STRUCTURAL_SETTINGS
@given(graphs=branched_ligand_graph_pairs)
def test_random_branched_ligand_graphs_have_stable_balanced_names(graphs):
    smiles, reordered_smiles = graphs
    result = name_one(smiles)
    reordered = name_one(reordered_smiles)

    assert result.error is None, (smiles, result.error)
    assert result.name, smiles
    assert reordered.error is None, (reordered_smiles, reordered.error)
    assert reordered.name == result.name
    _assert_balanced_parentheses(result.name)


_ROUNDTRIP_SETTINGS = settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)


@pytest.mark.fuzz
@pytest.mark.opsin
@pytest.mark.skipif(not _OPSIN_AVAILABLE, reason="py2opsin or Java unavailable")
@_ROUNDTRIP_SETTINGS
@given(graphs=branched_ligand_graph_pairs)
def test_random_branched_ligand_graphs_round_trip_through_opsin(graphs):
    smiles, _reordered_smiles = graphs
    result = name_one(smiles, verify_opsin=True)

    assert result.error is None, (smiles, result.error)
    assert result.name, smiles
    assert result.opsin_check is not None
    assert result.opsin_check.status == "matched", (
        smiles,
        result.name,
        result.opsin_check.opsin_smiles,
        result.opsin_check.canonical_original,
        result.opsin_check.canonical_roundtrip,
    )
