"""The shared nomenclature tables are the single source of truth for the two
directions that use them: the namer *writes* prefixes from them, and the audit's
parsers *read* prefixes back through them.  These tests pin the inverse lookups
to the forward ones so the two can never drift apart again — the failure mode
being an audit that abstains on names the namer is perfectly able to emit.
"""

from __future__ import annotations

import pytest

from openclatura.rules import bonds, elements, multipliers


@pytest.mark.parametrize("count", sorted(multipliers.MULTIPLIERS))
def test_every_written_multiplier_reads_back(count: int):
    # Both spellings the namer can emit must round-trip to the same count.
    assert multipliers.count_for(multipliers.basic(count)) == count
    assert multipliers.count_for(multipliers.complex_(count)) == count


def test_candidate_splits_are_longest_first():
    # ``tetradeca`` (14) must be offered before ``tetra`` (4), so a caller that
    # validates the remainder sees the more specific reading first.
    counts = [count for count, _ in multipliers.candidate_splits("tetradecafluoro")]
    assert counts == sorted(counts, reverse=True)
    assert counts[0] == 14


@pytest.mark.parametrize(
    ("count", "expected"),
    [
        (1, "en"),
        (2, "adien"),
        (3, "atrien"),
        (4, "atetraen"),
        (9, "anonaen"),
        (10, "adecaen"),
        # Past 10 the bond module used to run out of private table and raise
        # ``KeyError``, refusing names the multiplier table can spell perfectly
        # well -- a long cumulene needs 14.
        (12, "adodecaen"),
        (14, "atetradecaen"),
        (20, "aicosaen"),
    ],
)
def test_unsaturation_infix_spans_the_whole_multiplier_table(count: int, expected: str):
    assert bonds.unsaturation_infix("double", count) == expected


def test_unsaturation_infix_covers_every_multiplier_for_both_bond_orders():
    # Neither bond order may run out before the shared table does.
    for count in multipliers.MULTIPLIERS:
        assert bonds.unsaturation_infix("double", count).endswith("en")
        assert bonds.unsaturation_infix("triple", count).endswith("yn")


def test_a_single_bond_has_no_multiplicity():
    # Saturation is not cited with a count, so asking for one is a caller bug and
    # must say so rather than surfacing a bare lookup failure.
    with pytest.raises(ValueError, match="not cited with a multiplicity"):
        bonds.unsaturation_infix("single", 2)


def test_a_count_beyond_the_table_reports_itself():
    with pytest.raises(ValueError, match="no multiplicative prefix"):
        bonds.unsaturation_infix("double", max(multipliers.MULTIPLIERS) + 1)


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
        "Mg", "Ca", "Li", "Na", "K",
    ]  # fmt: skip
    priorities = [e.hw_priority for e in ranked]
    assert len(priorities) == len(set(priorities))


def test_fusion_priorities_and_mancude_roles_share_typed_element_data():
    from openclatura.fusion.rules import (
        EARLIEST_SPECIAL_HETEROATOM_PRECEDENCE,
        GENERAL_HETEROATOM_COUNT_PRECEDENCE,
    )

    special = sorted(
        (element for element in elements.ELEMENTS.values() if element.fusion_special_priority is not None),
        key=lambda element: element.fusion_special_priority,
    )
    general = sorted(
        (element for element in elements.ELEMENTS.values() if element.fusion_general_priority is not None),
        key=lambda element: element.fusion_general_priority,
    )
    assert tuple(element.symbol for element in special) == EARLIEST_SPECIAL_HETEROATOM_PRECEDENCE
    assert tuple(element.symbol for element in general) == GENERAL_HETEROATOM_COUNT_PRECEDENCE
    assert EARLIEST_SPECIAL_HETEROATOM_PRECEDENCE[:6] == ("N", "F", "Cl", "Br", "I", "O")
    assert GENERAL_HETEROATOM_COUNT_PRECEDENCE[:9] == ("F", "Cl", "Br", "I", "O", "S", "Se", "Te", "N")
    assert EARLIEST_SPECIAL_HETEROATOM_PRECEDENCE != GENERAL_HETEROATOM_COUNT_PRECEDENCE
    assert len({element.fusion_special_priority for element in special}) == len(special)
    assert len({element.fusion_general_priority for element in general}) == len(general)
    assert {
        element.symbol for element in elements.ELEMENTS.values() if element.mancude_forced_single
    } >= {"O", "S", "Se", "Te"}
    assert all(
        element.mancude_pi_capacity == 0
        for element in elements.ELEMENTS.values()
        if element.mancude_forced_single
    )


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


def test_audit_models_every_retained_fused_ring_as_a_substituent():
    """The namer names a ring substituent from the same retained table it uses for
    a parent, so every production parent also has to resolve as a ``-yl`` prefix
    — otherwise the audit abstains on names it used to confirm as von Baeyer."""

    from openclatura.audit.substituent_reconstruction import _RING_STEMS, _match_ring_stem, _ring_yl
    from openclatura.retained_fused_production import PRODUCTION_RETAINED_FUSED_PARENTS

    for name in sorted(PRODUCTION_RETAINED_FUSED_PARENTS):
        stem = name[:-1] if name.endswith("e") else name
        assert _match_ring_stem(stem) == name, name
        _smiles, labels = _RING_STEMS[name]
        for locant in labels:
            if locant.isdigit():
                assert _ring_yl(stem, locant) is not None, f"{stem}-{locant}-yl"


def test_retained_fused_audit_templates_match_their_graph_templates():
    """Each audit template is a hand-copied projection of the graph template the
    namer matches against.  Rebuild the graph from the copy and compare, so the
    two cannot drift apart."""

    from rdkit import Chem

    from openclatura.audit.substituent_reconstruction import _RETAINED_FUSED_RING_STEMS
    from openclatura.retained_fused_templates import retained_fused_graph_templates

    templates = {template.name: template for template in retained_fused_graph_templates(include_disabled=True)}
    for name, (smiles, labels) in sorted(_RETAINED_FUSED_RING_STEMS.items()):
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


def test_added_hydrogen_is_cited_so_the_retained_parent_survives():
    """Bare 4-oxoquinoline-3-carboxamide reads as a different molecule, so this
    used to fall back to von Baeyer; the hydro prefix now cites the saturation."""

    from openclatura import name_smiles

    name = name_smiles("N#CC[C@H](O)CNC(=O)c1c[nH]c2ccccc2c1=O")
    assert name.endswith("-4-oxo-1,4-dihydroquinoline-3-carboxamide"), name


def test_oxadiazoles_and_thiadiazoles_use_their_retained_names():
    """Replacement nomenclature spelt these out as 1-thia-3,4-diazacyclopenta-
    2,4-diene; Hantzsch-Widman retained names cover 5-rings, so they should not."""

    from openclatura import name_smiles

    cases = {
        "c1cnno1": "1,2,3-oxadiazole",
        "c1ncno1": "1,2,4-oxadiazole",
        "c1nonc1": "1,2,5-oxadiazole",
        "c1nnco1": "1,3,4-oxadiazole",
        "c1cnns1": "1,2,3-thiadiazole",
        "c1ncns1": "1,2,4-thiadiazole",
        "c1nsnc1": "1,2,5-thiadiazole",
        "c1nncs1": "1,3,4-thiadiazole",
    }
    for smiles, expected in cases.items():
        assert name_smiles(smiles) == expected, smiles

    # The isomers must stay distinguishable once substituted: a symmetric gap
    # multiset cannot separate 1,2,4- from 1,3,4-, which is what the chalcogen
    # distance criterion is for.
    # C2 and C5 are the ring's only substitutable positions and are equivalent,
    # so locant elision drops the ``2,5-``: there is nothing else the two methyls
    # could be.  The unsymmetrical pair below still needs its locants.
    assert name_smiles("Cc1nnc(C)s1") == "dimethyl-1,3,4-thiadiazole"
    assert name_smiles("Cc1nnc(CC)s1") == "2-ethyl-5-methyl-1,3,4-thiadiazole"
    assert name_smiles("Cc1ncns1") == "5-methyl-1,2,4-thiadiazole"
    assert name_smiles("Cc1nnc(-c2ccccc2)o1") == "2-methyl-5-phenyl-1,3,4-oxadiazole"


def test_audit_models_every_retained_monocycle_the_namer_emits():
    """A retained ring the audit has no template for abstains, so enabling one
    without the matching template silently stops it being verified."""

    from openclatura.audit.reconstruction import _ALL_PARENT_TEMPLATES
    from openclatura.nomenclature import RULES

    emitted = {spec["name"] for spec in RULES.retained.monocycle_specs}
    missing = sorted(name for name in emitted if name not in _ALL_PARENT_TEMPLATES)
    assert missing == []
