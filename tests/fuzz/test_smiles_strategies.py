"""Hypothesis fuzz tests against a controlled SMILES grammar.

Property-based naming engine tests. Each strategy emits SMILES that
RDKit parses cleanly (we validate that as a precondition), then we
assert engine invariants:

- ``bluenamer.name(s)`` never raises (it captures internal errors on
  ``result.error`` instead). Marker: ``fuzz``.
- Outputs are stable across repeated calls for the same input.
- ``name_many([s, s, s])[0].name == name(s).name`` — batch and single
  agree on per-row outputs.

OPSIN round-trip is *not* asserted here — the namer is not yet complete
enough to round-trip arbitrary structures. The integration corpus test
measures round-trip rate on a curated set; fuzz testing focuses on
"never crash, results are consistent" invariants.
"""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from rdkit import Chem

from bluenamer import name as name_one
from bluenamer import name_many


def _rdkit_parses(smiles: str) -> bool:
    try:
        return Chem.MolFromSmiles(smiles) is not None
    except Exception:
        return False


# --- SMILES sub-grammar strategies ---------------------------------------

_HALOGENS = ["F", "Cl", "Br", "I"]
_PRINCIPAL_GROUPS = [
    "",
    "O",  # -OH
    "C(=O)O",  # -COOH
    "C(=O)N",  # -C(=O)NH2
    "C(=O)C",  # ketone tail
    "C(=O)",  # aldehyde tail
    "N",  # -NH2
    "C#N",  # -CN
    "[N+](=O)[O-]",  # -NO2
]


@st.composite
def alkane_chain(draw: st.DrawFn) -> str:
    """A linear or lightly-branched alkane chain of up to 8 carbons."""

    n = draw(st.integers(min_value=1, max_value=8))
    chain = "C" * n
    if n >= 4 and draw(st.booleans()):
        branch_pos = draw(st.integers(min_value=1, max_value=n - 2))
        branch_size = draw(st.integers(min_value=1, max_value=3))
        chain = "C" * branch_pos + "(" + "C" * branch_size + ")" + "C" * (n - branch_pos)
    return chain


@st.composite
def haloalkane(draw: st.DrawFn) -> str:
    chain = draw(alkane_chain())
    halogen = draw(st.sampled_from(_HALOGENS))
    return halogen + chain


@st.composite
def functionalised_chain(draw: st.DrawFn) -> str:
    """Alkane decorated with a single principal group at the head."""

    chain = draw(alkane_chain())
    group = draw(st.sampled_from(_PRINCIPAL_GROUPS))
    return chain + group


@st.composite
def simple_cycle(draw: st.DrawFn) -> str:
    """A monocyclic ring of size 3–7."""

    size = draw(st.integers(min_value=3, max_value=7))
    return "C1" + "C" * (size - 1) + "1"


@st.composite
def benzene_substituted(draw: st.DrawFn) -> str:
    """Benzene with 0–2 simple substituents."""

    subs = draw(st.lists(st.sampled_from(_HALOGENS + ["O", "N", "C"]), min_size=0, max_size=2))
    if not subs:
        return "c1ccccc1"
    if len(subs) == 1:
        return f"c1ccc({subs[0]})cc1"
    return f"c1cc({subs[0]})cc({subs[1]})c1"


smiles_grammar = st.one_of(
    alkane_chain(),
    haloalkane(),
    functionalised_chain(),
    simple_cycle(),
    benzene_substituted(),
)


# --- Fuzz properties -----------------------------------------------------

_FUZZ_SETTINGS = settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
)


@pytest.mark.fuzz
@_FUZZ_SETTINGS
@given(smiles=smiles_grammar)
def test_name_never_raises_on_grammar_smiles(smiles):
    """``bluenamer.name`` must never raise on grammar-valid SMILES.

    Errors must surface as ``result.error`` instead. We pre-filter on
    RDKit parsing so we don't flag the engine for unparseable inputs;
    those are upstream-validation concerns.
    """

    if not _rdkit_parses(smiles):
        return  # not a contract violation for the engine
    result = name_one(smiles)
    assert result is not None
    # If naming failed, the contract is: error captured, name empty.
    if result.error is not None:
        assert result.name == ""


@pytest.mark.fuzz
@_FUZZ_SETTINGS
@given(smiles=smiles_grammar)
def test_name_is_deterministic_for_same_input(smiles):
    if not _rdkit_parses(smiles):
        return
    a = name_one(smiles)
    b = name_one(smiles)
    assert a.name == b.name
    assert a.error == b.error
    assert a.rules_hit == b.rules_hit


@pytest.mark.fuzz
@_FUZZ_SETTINGS
@given(smiles=smiles_grammar)
def test_batch_matches_single_call(smiles):
    if not _rdkit_parses(smiles):
        return
    single = name_one(smiles)
    batch = name_many([smiles, smiles, smiles], processes=1)
    assert all(r.name == single.name for r in batch)


@pytest.mark.fuzz
@_FUZZ_SETTINGS
@given(smiles=st.lists(smiles_grammar, min_size=1, max_size=8))
def test_batch_preserves_order(smiles):
    smiles = [s for s in smiles if _rdkit_parses(s)]
    if not smiles:
        return
    results = name_many(smiles, processes=1)
    assert [r.smiles for r in results] == smiles
