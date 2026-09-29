"""Regression tests for two more bugs found by a large-scale pubchem
corpus comparison (baseline vs. current, 100k real molecules each) after
the previous round of fixes: both are pre-existing defects the earlier
fixes happened to expose by changing which code path a molecule takes.

Bug A -- simple_central_parent_hydride_result dropped stereo descriptors
(FIXED): this special case names a mononuclear hydride centre (silane,
phosphane, ...) with its ligands as prefixes -- e.g. "chloro(ethyl)
(isopropoxy)phosphane oxide" -- but has no machinery to express a
stereo-descriptor on the CENTRE itself. When the centre is a genuine
stereocentre (e.g. a chiral phosphine oxide/phosphonochloridate,
CC[P@](=O)(Cl)OC(C)C), firing this shortcut silently drops the
configuration, since the general substituent-prefix pipeline that WOULD
express it (e.g. "(R)-chloro(ethyl)oxophosphanyl" as a prefix) never gets
a chance to run. Fixed the same way organophosphonic_acid_result already
guards against exactly this ("a configured phosphorus needs a descriptor
this spelling cannot carry"): refuse to fire when the centre has a
stereo/raw_stereo tag.

Bug B -- ring_nitrile and ring_carboxylic_acid missing from
direct_group_prefixes (FIXED): a functional group attached to a RING that
must be expressed as a substituent PREFIX (because something more senior
is the actual principal group elsewhere) is normally recognised directly
by _direct_subgraph_prefix via the direct_group_prefixes table --
"nitrile" maps to "cyano", "carboxylic_acid" maps to "carboxy". Their
ring-attached counterparts, "ring_nitrile" and "ring_carboxylic_acid",
were missing from that table even though perceive_groups correctly
produces those keys for a ring-attached instance (a ring-C#N is
"ring_nitrile", not "nitrile") -- so the direct lookup silently missed,
and the generic recursive branch namer took over instead, treating the
2-3 atom functional-group subgraph as its own tiny "methane"/"methanol"-
style parent with the heteroatom as a prefix on it. For a nitrile this
produced the outright wrong "cyanomethyl"/"nitrilomethyl" (a real,
different substituent -- an extra CH2 that isn't there) instead of
"cyano", which is why it failed OPSIN round-trips. For a carboxylic acid
it produced "hydroxycarbonyl" instead of "carboxy" -- chemically
equivalent and still OPSIN-parseable, but not the standard PIN prefix.

The other five ring_* siblings that could plausibly have the same gap
(ring_amide, ring_acid_fluoride/chloride/bromide/iodide) were checked and
confirmed to already resolve correctly via a separate, working mechanism
("carbamoyl", "chlorocarbonyl", etc. render correctly even without a
direct_group_prefixes entry), so they were deliberately left alone.

CAUTION for anyone tempted to also blanket-replace "nitrilo" with "cyano"
in _finalize_subgraph_name (namer.py) -- an earlier draft of this fix did
exactly that, and it is WRONG: "cyano" and "nitrilo" are not interchangeable
spellings of the same atom count once a chain/locant is involved. "cyano"
always implies its own separate carbon beyond the chain it's cited on
(2-cyanoethyl = 3 carbons total), whereas "nitrilo" (as OPSIN parses it
back) folds the nitrile's carbon INTO the counted chain (2-nitriloethyl =
2 carbons total). Round-tripping both spellings through OPSIN for
O=COCC#N (a genuine 2-carbon alcohol fragment) confirms it: "2-nitriloethyl
formate" matches, "2-cyanoethyl formate" does not (OPSIN reparses it back
with a phantom third carbon). The PIN-correct spelling for that exact
shape is actually "cyanomethyl formate" (one fewer counted chain carbon,
not just a different word) -- but that's a separate, deeper, chain-length
bug in how substituent parent length is computed for chain-terminal
nitriles, deliberately left alone here since this pass only targets the
ISOLATED-nitrile-as-its-own-substituent shape the direct_group_prefixes
fix above addresses. See test_chain_terminal_nitrilo_wording_is_unaffected
below for the guard against re-introducing the blanket-replacement bug.
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
# Bug A: a stereocentre on the mononuclear-hydride shortcut's own central
# atom must fall through to the general (stereo-capable) pipeline.
# ---------------------------------------------------------------------------


def test_chiral_phosphine_oxide_keeps_its_stereo_descriptor():
    """The exact pubchem-corpus regression: a chiral phosphonochloridate
    ester. Before the fix this silently lost the (R) configuration."""

    result = name_one("CC[P@](=O)(Cl)OC(C)C", verify_opsin=True)
    assert "(R)" in result.name
    assert result.opsin_check.status == "matched", (result.name, result.opsin_check.status)


def test_simple_central_parent_hydride_result_refuses_a_stereocentre_directly():
    from openclatura.graph_io import read_smiles
    from openclatura.special_cases import simple_central_parent_hydride_result

    mol = read_smiles("CC[P@](=O)(Cl)OC(C)C")
    assert simple_central_parent_hydride_result(mol, set(mol.atoms)) is None


def test_non_stereo_phosphine_oxide_is_unaffected_by_the_guard():
    """Control: the same shape without a defined stereocentre (identical
    ligands -- ethyl and ethyl -- so there's nothing to configure) must
    still take the mononuclear-hydride shortcut as before."""

    result = name_one("CCP(=O)(Cl)OC(C)C", verify_opsin=True)
    assert "phosphane oxide" in result.name
    assert result.opsin_check.status == "matched"


# ---------------------------------------------------------------------------
# Bug B: nitrile and carboxylic acid attached to a RING, cited as a
# substituent PREFIX (not the principal group), must use the standard
# "cyano"/"carboxy" words -- not a generic decomposed fallback.
# ---------------------------------------------------------------------------

RING_DIRECT_PREFIX_CASES = (
    (
        "ring-nitrile-as-prefix-simple",
        "N#Cc1ccc(CC(=O)O)cc1",
        "2-(4-cyanophenyl)acetic acid",
    ),
    (
        "ring-nitrile-as-prefix-pubchem-corpus-regression-1",
        "Cc1nc(C#N)c(N(CCC(=O)O)C(=O)OC(C)(C)C)o1",
        "3-((tert-butoxycarbonyl)(4-cyano-2-methyl-1,3-oxazol-5-yl)amino)propanoic acid",
    ),
    (
        "ring-carboxylic-acid-as-prefix-simple",
        "OC(=O)Cc1ccccc1C(=O)O",
        "2-(2-carboxyphenyl)acetic acid",
    ),
)


@pytest.mark.parametrize(
    ("_case", "smiles", "expected_name"),
    RING_DIRECT_PREFIX_CASES,
    ids=[c for c, _s, _n in RING_DIRECT_PREFIX_CASES],
)
def test_ring_attached_group_uses_the_standard_prefix_word(_case, smiles, expected_name):
    _assert_matched(smiles, expected_name)


def test_ring_nitrile_as_principal_group_is_unaffected_by_the_fix():
    """Control: when the ring nitrile IS the principal group (nothing
    outranks it), it should keep using the "-carbonitrile" suffix, exactly
    as before -- this path never went through direct_group_prefixes."""

    assert name_smiles("Cc1nc(C#N)co1") == "2-methyl-1,3-oxazole-4-carbonitrile"


def test_direct_group_prefix_lookup_now_covers_both_ring_keys():
    from openclatura.nomenclature import RULES

    assert RULES.functional_groups.direct_subgraph_prefix_for("ring_nitrile") == "cyano"
    assert RULES.functional_groups.direct_subgraph_prefix_for("ring_carboxylic_acid") == "carboxy"


# ---------------------------------------------------------------------------
# Guard against re-introducing a blanket "nitrilo"->"cyano" text
# replacement: chain-terminal nitriles (where OPSIN's own atom-counting
# for "nitrilo" already matches the real structure) must keep their
# current wording untouched by the ring_nitrile/ring_carboxylic_acid fix
# above. See the module docstring's CAUTION note for the full story.
# ---------------------------------------------------------------------------

CHAIN_TERMINAL_NITRILO_CASES = (
    (
        "single-ester-alcohol-fragment",
        "O=COCC#N",
        "2-nitriloethyl formate",
    ),
    (
        "three-carbon-ester-alcohol-fragment",
        "O=COCCC#N",
        "3-nitrilopropyl formate",
    ),
    (
        "repeated-nitriloethyl-diester",
        "CC(C(=O)OCC#N)C(=O)OCC#N",
        "bis(2-nitriloethyl) 2-methylpropanedioate",
    ),
)


@pytest.mark.parametrize(
    ("_case", "smiles", "expected_name"),
    CHAIN_TERMINAL_NITRILO_CASES,
    ids=[c for c, _s, _n in CHAIN_TERMINAL_NITRILO_CASES],
)
def test_chain_terminal_nitrilo_wording_is_unaffected(_case, smiles, expected_name):
    _assert_matched(smiles, expected_name)
