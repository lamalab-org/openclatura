"""The shared nomenclature tables are the single source of truth for the two
directions that use them: the namer *writes* prefixes from them, and the audit's
parsers *read* prefixes back through them.  These tests pin the inverse lookups
to the forward ones so the two can never drift apart again — the failure mode
being an audit that abstains on names the namer is perfectly able to emit.
"""

from __future__ import annotations

import pytest

from openclatura.rules import elements, multipliers


@pytest.mark.parametrize("count", sorted(multipliers.MULTIPLIERS))
def test_every_written_multiplier_reads_back(count: int):
    # Both spellings the namer can emit must round-trip to the same count.
    assert multipliers.count_for(multipliers.basic(count)) == count
    assert multipliers.count_for(multipliers.complex_(count)) == count


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("dimethyl", (2, "methyl")),
        ("bis(2-chloroethyl)", (2, "(2-chloroethyl)")),
        ("tris(hydroxymethyl)", (3, "(hydroxymethyl)")),
        ("hexadecafluoro", (16, "fluoro")),  # longest prefix wins over ``hexa``
        ("methyl", (1, "methyl")),  # no prefix reads as a single occurrence
    ],
)
def test_split_prefix(name: str, expected: tuple[int, str]):
    assert multipliers.split_prefix(name) == expected


def test_candidate_splits_are_longest_first():
    # ``tetradeca`` (14) must be offered before ``tetra`` (4), so a caller that
    # validates the remainder sees the more specific reading first.
    counts = [count for count, _ in multipliers.candidate_splits("tetradecafluoro")]
    assert counts == sorted(counts, reverse=True)
    assert counts[0] == 14


def test_every_replacement_prefix_maps_to_its_element():
    # The inverse table must cover exactly the elements that declare a stem, and
    # agree with them.
    declared = {e.hw_stem: e.symbol for e in elements.ELEMENTS.values() if e.hw_stem}
    assert elements.SYMBOLS_BY_HW_STEM == declared


def test_replacement_stems_are_unique():
    stems = [e.hw_stem for e in elements.ELEMENTS.values() if e.hw_stem]
    assert len(stems) == len(set(stems))


def test_hw_priorities_are_unique_and_ordered_by_seniority():
    # Skeletal-replacement seniority (P-15.4.3.2): halogens, then O, S, Se, Te,
    # then N, P, As, Sb, Bi, then Si, Ge, Sn, Pb, then B, Al, Ga.
    ranked = sorted(
        (e for e in elements.ELEMENTS.values() if e.hw_priority is not None),
        key=lambda e: e.hw_priority,
    )
    assert [e.symbol for e in ranked] == [
        "F", "Cl", "Br", "I", "O", "S", "Se", "Te",
        "N", "P", "As", "Sb", "Bi", "Si", "Ge", "Sn", "Pb", "B", "Al", "Ga",
    ]  # fmt: skip
    priorities = [e.hw_priority for e in ranked]
    assert len(priorities) == len(set(priorities))


def test_audit_parsers_share_the_canonical_tables():
    # The audit reconstructs names independently of the namer's *bindings*, but it
    # must speak the same vocabulary — these are the shared spelling tables, not
    # shared inference.
    from openclatura.audit import reconstruction, von_baeyer_parse

    assert reconstruction._REPLACEMENT_ELEMENTS is elements.SYMBOLS_BY_HW_STEM
    assert von_baeyer_parse._REPLACEMENT_ELEMENTS is elements.SYMBOLS_BY_HW_STEM


def test_audit_models_every_retained_fused_parent_the_namer_emits():
    """The audit abstains on any retained parent it has no template for, so a
    parent enabled for production without one silently stops being verified."""

    from openclatura.audit.reconstruction import _ALL_PARENT_TEMPLATES
    from openclatura.retained_fused_production import PRODUCTION_RETAINED_FUSED_PARENTS

    missing = sorted(name for name in PRODUCTION_RETAINED_FUSED_PARENTS if name not in _ALL_PARENT_TEMPLATES)
    assert missing == []


def test_retained_fused_audit_templates_match_their_graph_templates():
    """Each audit template is a hand-copied projection of the graph template the
    namer matches against.  Rebuild the graph from the copy and compare, so the
    two cannot drift apart."""

    from rdkit import Chem

    from openclatura.audit.reconstruction import _RETAINED_FUSED_PARENT_TEMPLATES
    from openclatura.retained_fused_templates import retained_fused_graph_templates

    templates = {template.name: template for template in retained_fused_graph_templates(include_disabled=True)}
    for name, (smiles, labels) in sorted(_RETAINED_FUSED_PARENT_TEMPLATES.items()):
        template = templates[name]
        frag = Chem.MolFromSmiles(smiles)
        assert frag is not None, f"{name}: template SMILES does not parse"
        assert frag.GetNumAtoms() == len(labels), f"{name}: {len(labels)} labels for {frag.GetNumAtoms()} atoms"

        by_locant = {label: index for index, label in enumerate(labels)}
        assert by_locant.keys() == {atom.locant for atom in template.atoms}, f"{name}: locant sets differ"
        for atom in template.atoms:
            assert frag.GetAtomWithIdx(by_locant[atom.locant]).GetSymbol() == atom.symbol, f"{name}: {atom.locant}"

        expected_bonds = {frozenset(bond.locants) for bond in template.bonds}
        actual_bonds = {
            frozenset((labels[bond.GetBeginAtomIdx()], labels[bond.GetEndAtomIdx()])) for bond in frag.GetBonds()
        }
        assert actual_bonds == expected_bonds, f"{name}: bonds differ"


def test_retained_parent_survives_an_arbitrary_substituent():
    """A substituent is named by its own recursive call, so what it is cannot
    affect whether the parent's locants are right.  A curated allow-list used to
    decide this, and anything off it dropped the retained parent entirely."""

    from openclatura import name_smiles

    for substituent in ("C" * 5, "C" * 6, "C" * 12, "CC(C)CC"):
        name = name_smiles(substituent + "c1ncc2[nH]cnc2n1")
        assert name.endswith("-7H-purine"), name
    assert name_smiles("c1ccc(-c2ncc3[nH]cnc3n2)cc1") == "2-phenyl-7H-purine"


def test_uncitable_added_hydrogen_keeps_the_von_baeyer_parent():
    """4-oxoquinoline-3-carboxamide reads as a different molecule -- the two
    saturated positions have to be cited, and quinoline cannot cite them yet."""

    from openclatura import name_smiles

    name = name_smiles("N#CC[C@H](O)CNC(=O)c1c[nH]c2ccccc2c1=O")
    assert "quinoline" not in name
    assert "bicyclo[4.4.0]" in name
