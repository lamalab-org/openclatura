"""A retained polycyclic parent has to be offered as a base component.

P-25.3.2.1.3 admits the retained names of Tables 2.7 and 2.8 as fusion
components, parent as well as attached. The planner can only cite what the
component registry offers it, so a retained name missing from that registry is
unreachable however the criteria rank: acridine was known only as a whole
retained ring system, and the planner fell through to the senior registered
nitrogenous component, quinoline.

The ranking itself never needed changing. P-25.3.2.4 settles a tie between two
nitrogenous components by the ring count of the component itself, and
ChemicalComponentSeniorityKey already carries that second. Acridine's three
rings beat quinoline's two as soon as acridine is a candidate.

What is pinned here is three separate things, because a single expected-name
assertion cannot say which of them failed: that the component survives registry
expansion as a parent, that its component-local perimeter and side letters are
reconstructed from its own bonds, and that completed targets render correctly.
The completed system is numbered independently of the component (FR-4.8), so
nothing here asserts that acridine's own 1-10a numbering reaches the name.
"""

import pytest
from rdkit import Chem

from openclatura import name_mol, name_smiles, opsin_available, verify_with_opsin
from openclatura.fusion.registry import _with_ordered_fusion_perimeter, fusion_component_registry
from openclatura.retained_fused_templates import retained_graph_templates

# FR-4.1 letters the sides continuously around the perimeter from 1->2. These
# are component-local edges, not locants of any completed system.
ACRIDINE_PERIMETER = ("1", "2", "3", "4", "4a", "10", "10a", "5", "6", "7", "8", "8a", "9", "9a")
ACRIDINE_SIDES = {"a": ("1", "2"), "e": ("4a", "10"), "g": ("10a", "5"), "j": ("7", "8"), "n": ("9a", "1")}

# FR-4.7 elides no terminal vowel in an ordinary fusion prefix, so the PIN keeps
# the o. The second column is a correct name for the same structure that is not
# the preferred one, and must not come back from the engine.
PREFERRED = {
    "dibenzo[a,j]acridine": "dibenz[a,j]acridine",
    "dibenzo[a,h]acridine": "dibenz[a,h]acridine",
    "dibenzo[c,h]acridine": "dibenz[c,h]acridine",
    "benzo[a]acridine": "benz[a]acridine",
    "benzo[c]acridine": "benz[c]acridine",
}


def _acridine_specs():
    return [spec for spec in fusion_component_registry().specs if (spec.parent_name or spec.key) == "acridine"]


def test_acridine_survives_registry_expansion_as_a_parent():
    """The generic benzoheterocycle family must not downgrade an explicit row."""

    specs = _acridine_specs()
    assert specs, "acridine is not registered as a fusion component"
    for spec in specs:
        assert spec.usable_as_parent
        assert spec.usable_as_attached
        assert spec.attached_prefix == "acridino"
        assert len(spec.rings) == 3


def test_acridine_outranks_quinoline_on_its_own_ring_count():
    """P-25.3.2.4: the component's rings, not how much of the target it covers."""

    registry = fusion_component_registry()
    by_name = {(spec.parent_name or spec.key): spec for spec in registry.specs}
    assert len(by_name["acridine"].rings) > len(by_name["quinoline"].rings)


def test_the_component_perimeter_is_rebuilt_from_its_own_bonds():
    """The stored peripheral array is a membership list, not a walk."""

    template = next(item for item in retained_graph_templates(include_disabled=True) if item.name == "acridine")
    ordered = _with_ordered_fusion_perimeter(template)
    perimeter = tuple(str(atom) for atom in ordered.peripheral_atoms)
    assert perimeter == ACRIDINE_PERIMETER
    bonds = {frozenset(str(locant) for locant in bond.locants) for bond in ordered.bonds}
    for index, atom in enumerate(perimeter):
        assert frozenset((atom, perimeter[(index + 1) % len(perimeter)])) in bonds


def test_the_component_side_letters_follow_the_perimeter():
    """FR-4.1, anchored at 1->2; [a,j] has to mean these two edges."""

    template = next(item for item in retained_graph_templates(include_disabled=True) if item.name == "acridine")
    perimeter = tuple(str(atom) for atom in _with_ordered_fusion_perimeter(template).peripheral_atoms)
    sides = {
        chr(ord("a") + index): (atom, perimeter[(index + 1) % len(perimeter)]) for index, atom in enumerate(perimeter)
    }
    assert {letter: sides[letter] for letter in ACRIDINE_SIDES} == ACRIDINE_SIDES


@pytest.mark.skipif(not opsin_available(), reason="the reference structures come from OPSIN")
@pytest.mark.parametrize("preferred", sorted(PREFERRED))
def test_the_completed_acridine_targets_render_preferred(preferred):
    """Exact name, and the vowel-elided spelling must not come back instead."""

    from py2opsin import py2opsin

    smiles = py2opsin(preferred)
    assert smiles, f"OPSIN could not build {preferred}"
    named = name_smiles(smiles)
    assert named == preferred
    assert named != PREFERRED[preferred]
    assert verify_with_opsin(named, smiles).status == "matched"


@pytest.mark.skipif(not opsin_available(), reason="the reference structure comes from OPSIN")
def test_the_completed_target_is_order_invariant():
    from py2opsin import py2opsin

    mol = Chem.MolFromSmiles(py2opsin("dibenzo[a,j]acridine"))
    count = mol.GetNumAtoms()
    orders = (list(range(count)), list(reversed(range(count))), sorted(range(count), key=lambda i: (i * 7) % count))
    assert len({name_mol(Chem.RenumberAtoms(mol, order)).name for order in orders}) == 1


# FR-4.8 encloses the locants cited as part of a component's name in square
# brackets; P-16.5.2.2 is the Blue Book counterpart. The transformation belongs
# to the fusion citation, not to the standalone parent hydride, which keeps its
# hyphen - so both fusion roles are pinned here against the template's own
# output name.
ENCLOSED_COMPONENT_LOCANTS = {
    "1,5-naphthyridine": ("[1,5]naphthyridine", "[1,5]naphthyridino"),
    "1,6-naphthyridine": ("[1,6]naphthyridine", "[1,6]naphthyridino"),
    "1,7-naphthyridine": ("[1,7]naphthyridine", "[1,7]naphthyridino"),
    "1,8-naphthyridine": ("[1,8]naphthyridine", "[1,8]naphthyridino"),
    "2,6-naphthyridine": ("[2,6]naphthyridine", "[2,6]naphthyridino"),
    "2,7-naphthyridine": ("[2,7]naphthyridine", "[2,7]naphthyridino"),
    "1,7-phenanthroline": ("[1,7]phenanthroline", "[1,7]phenanthrolino"),
    "1,8-phenanthroline": ("[1,8]phenanthroline", "[1,8]phenanthrolino"),
    "1,9-phenanthroline": ("[1,9]phenanthroline", "[1,9]phenanthrolino"),
    "1,10-phenanthroline": ("[1,10]phenanthroline", "[1,10]phenanthrolino"),
    "2,7-phenanthroline": ("[2,7]phenanthroline", "[2,7]phenanthrolino"),
    "2,8-phenanthroline": ("[2,8]phenanthroline", "[2,8]phenanthrolino"),
}


@pytest.mark.parametrize("key", sorted(ENCLOSED_COMPONENT_LOCANTS))
def test_component_locants_are_enclosed_in_both_fusion_roles(key):
    parent_name, attached_prefix = ENCLOSED_COMPONENT_LOCANTS[key]
    component = fusion_component_registry().get(key)

    assert component is not None, f"{key} is not offered as a fusion component"
    assert component.spec.parent_name == parent_name
    assert component.spec.attached_prefix == attached_prefix
    # The template's own output name is what a standalone parent hydride uses
    # and keeps its hyphen; only the fusion forms take enclosing marks.
    assert component.spec.template.output_name == key


# The structure is built from the first name and has to come back as the
# second: the literature spelling of the phenanthroline case drops the
# indicated hydrogen that the preferred name states.
ENCLOSED_BASE_TARGETS = (
    ("dibenzo[b,g][1,8]naphthyridine", "dibenzo[b,g][1,8]naphthyridine"),
    ("benzo[b][1,8]naphthyridine", "benzo[b][1,8]naphthyridine"),
    ("imidazo[4,5-f][1,10]phenanthroline", "1H-imidazo[4,5-f][1,10]phenanthroline"),
)


@pytest.mark.skipif(not opsin_available(), reason="the reference structures come from OPSIN")
@pytest.mark.parametrize("source,expected", ENCLOSED_BASE_TARGETS)
def test_a_locanted_base_component_renders_enclosed(source, expected):
    from py2opsin import py2opsin

    smiles = py2opsin(source)
    assert smiles, f"OPSIN could not build {source}"
    named = name_smiles(smiles)
    assert named == expected
    assert verify_with_opsin(named, smiles).status == "matched"
