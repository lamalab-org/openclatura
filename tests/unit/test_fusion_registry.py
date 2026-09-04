from __future__ import annotations

from copy import deepcopy
from types import MappingProxyType

import pytest

from openclatura.fusion.descriptor import component_sides
from openclatura.fusion.faces import GraphCycle
from openclatura.fusion.registry import (
    FusionComponentRegistry,
    FusionComponentRole,
    fusion_component_registry,
    version,
)
from openclatura.hantzsch_widman import hw_generated_names
from openclatura.molecule import Molecule
from openclatura.naming_data import load_json_table
from openclatura.rules import elements


def _ring(
    symbols: tuple[str, ...],
    *,
    atom_ids: tuple[int, ...],
    nonaromatic_positions: frozenset[int] = frozenset(),
) -> tuple[Molecule, GraphCycle]:
    """Build a ring graph directly, independently of any line notation."""

    mol = Molecule()
    for position, (atom_id, symbol) in enumerate(zip(atom_ids, symbols)):
        mol.add_atom(symbol, idx=atom_id, is_aromatic=position not in nonaromatic_positions)
    for bond_id, (left, right) in enumerate(zip(atom_ids, atom_ids[1:] + atom_ids[:1]), start=100):
        mol.add_bond(left, right, idx=bond_id)
    return mol, GraphCycle.from_atoms(atom_ids)


def _triazole_ring(*, atom_ids: tuple[int, ...] = (0, 1, 2, 3, 4)) -> tuple[Molecule, GraphCycle]:
    """Graph-build a neutral mancude 1,2,4-triazole component."""

    mol = Molecule()
    symbols = ("N", "N", "C", "N", "C")
    for position, (atom_id, symbol) in enumerate(zip(atom_ids, symbols, strict=True)):
        mol.add_atom(symbol, idx=atom_id, is_aromatic=True, total_h_count=position == 0)
    double_edges = {frozenset((atom_ids[1], atom_ids[2])), frozenset((atom_ids[3], atom_ids[4]))}
    for bond_id, (left, right) in enumerate(zip(atom_ids, atom_ids[1:] + atom_ids[:1]), start=200):
        mol.add_bond(
            left,
            right,
            idx=bond_id,
            order=2 if frozenset((left, right)) in double_edges else 1,
        )
    return mol, GraphCycle.from_atoms(atom_ids)


@pytest.mark.parametrize(
    ("key", "symbols", "atom_ids", "nonaromatic", "prefix", "map_count"),
    [
        ("furan", ("O", "C", "C", "C", "C"), (40, 10, 50, 20, 30), {0}, "furo", 2),
        ("thiophene", ("S", "C", "C", "C", "C"), (9, 4, 8, 2, 6), {0}, "thieno", 2),
        ("imidazole", ("N", "C", "N", "C", "C"), (12, 3, 19, 5, 8), {0}, "imidazo", 2),
        ("thiazole", ("S", "C", "N", "C", "C"), (32, 7, 41, 11, 23), {0}, "thiazolo", 1),
        ("phosphole", ("P", "C", "C", "C", "C"), (33, 8, 42, 12, 24), {0}, "phospholo", 2),
        ("arsole", ("As", "C", "C", "C", "C"), (34, 9, 43, 13, 25), {0}, "arsolo", 2),
        ("borole", ("B", "C", "C", "C", "C"), (35, 14, 44, 15, 26), {0}, "borolo", 2),
        ("pyridine", ("N", "C", "C", "C", "C", "C"), (70, 10, 60, 20, 50, 30), set(), "pyrido", 2),
    ],
)
def test_graph_faces_match_every_exact_local_locant_map(
    key: str,
    symbols: tuple[str, ...],
    atom_ids: tuple[int, ...],
    nonaromatic: set[int],
    prefix: str,
    map_count: int,
):
    registry = fusion_component_registry()
    mol, face = _ring(symbols, atom_ids=atom_ids, nonaromatic_positions=frozenset(nonaromatic))

    matches = tuple(match for match in registry.match_faces(mol, (face,), role="attached") if match.spec_key == key)

    assert len(matches) == map_count
    assert {match.covered_face_ids for match in matches} == {frozenset({0})}
    assert len({match.local_to_input_atom for match in matches}) == map_count
    assert all(
        dict(match.local_to_input_atom).keys() == {str(index) for index in range(1, len(symbols) + 1)}
        for match in matches
    )
    assert all(set(dict(match.local_to_input_atom).values()) == set(atom_ids) for match in matches)
    assert all(match.local_to_skeleton_atom == match.local_to_input_atom for match in matches)
    assert all(match.template_name for match in matches)
    assert all(registry.spec_for_match(match).template.name == match.template_name for match in matches)
    component = registry.by_key[key]
    assert component.spec.attached_prefix == prefix
    assert component.spec.usable_as_parent
    assert component.spec.usable_as_attached
    assert component.spec.rule_reference == "P-25.2"


def test_match_faces_preserves_an_explicit_input_to_skeleton_map():
    mol, face = _ring(("O", "C", "C", "C", "C"), atom_ids=(9, 1, 7, 3, 5), nonaromatic_positions=frozenset({0}))
    skeleton = {atom: atom + 100 for atom in mol.atoms}

    matches = fusion_component_registry().match_faces(
        mol,
        (face,),
        role=FusionComponentRole.PARENT,
        input_to_skeleton_atom=skeleton,
    )
    furan = [match for match in matches if match.spec_key == "furan"]

    assert furan
    for match in furan:
        assert dict(match.local_to_skeleton_atom) == {
            locant: skeleton[atom] for locant, atom in match.local_to_input_atom
        }


def test_checked_in_registry_exposes_stable_version_and_unique_policy_keys():
    registry = fusion_component_registry()

    assert version() == registry.version == "2026.1"
    assert len(registry.components) == len(registry.by_key)
    assert len({name for component in registry.components for name in component.template_names}) == sum(
        len(component.template_names) for component in registry.components
    )
    furan = registry.by_key["furan"].spec
    assert furan.template is registry.by_key["furan"].templates[0]
    oxygen = next(atom for atom in furan.atoms if atom.symbol == "O")
    assert oxygen is next(atom for atom in furan.template.atoms if atom.symbol == "O")
    assert elements.get("O").mancude_forced_single
    assert registry.by_key["naphthalene"].spec.horizontal_ring_count == 2
    assert isinstance(registry.by_key, MappingProxyType)
    assert registry.by_key is registry.by_key
    assert registry.get("furan") is registry.by_key["furan"]


def test_every_registered_component_reuses_a_complete_locanted_parent_graph():
    registry = fusion_component_registry()

    for component in registry.components:
        assert component.template_names
        assert component.spec.rule_reference.startswith("P-")
        assert component.spec.template is component.templates[0]
        for template in component.templates:
            atom_locants = {atom.locant for atom in template.atoms}
            assert atom_locants == set(template.locants), component.spec.key
            assert len(atom_locants) == len(template.atoms)
            assert all(set(bond.locants) <= atom_locants for bond in template.bonds)
            assert all(set(ring) <= atom_locants for ring in template.rings)
            assert set(template.peripheral_atoms) <= atom_locants
            assert set(template.fusion_atoms) <= atom_locants
            assert set(template.interior_atoms) <= atom_locants
        sides = component_sides(component.spec)
        assert len(sides) == len(component.spec.peripheral_order)
        assert tuple(side.letter for side in sides) == tuple(
            chr(ord("a") + index) for index in range(len(sides))
        )
        if component.spec.usable_as_attached:
            assert component.spec.attached_prefix
        assert component.spec.usable_as_parent or component.spec.usable_as_attached


def test_generated_carbocycles_use_shared_numbered_graph_templates():
    registry = fusion_component_registry()
    component = registry.by_key["cyclopentadiene"]

    assert component.spec.attached_prefix == "cyclopenta"
    assert not component.spec.usable_as_parent
    assert component.spec.usable_as_attached
    assert component.spec.template.family == "generated_monocycle"
    assert component.spec.locants == ("1", "2", "3", "4", "5")
    assert len(component.spec.atoms) == len(component.spec.bonds) == 5
    assert {bond.bond_class for bond in component.spec.bonds} == {"aromatic"}


def test_unlisted_hw_monocycle_is_derived_as_a_systematic_component():
    registry = fusion_component_registry()
    mol, face = _triazole_ring(atom_ids=(40, 10, 50, 20, 30))
    before_keys = tuple(registry.by_key)
    before_hw_names = tuple(hw_generated_names())

    matches = tuple(
        match
        for match in registry.match_faces(mol, (face,), role=FusionComponentRole.ATTACHED)
        if match.spec_key.startswith("generated-hw:")
    )

    assert len(matches) == 2
    assert {registry.spec_for_match(match).parent_name for match in matches} == {"[1,2,4]triazole"}
    assert {registry.spec_for_match(match).attached_prefix for match in matches} == {"[1,2,4]triazolo"}
    assert all(registry.spec_for_match(match).template.family == "generated_hw_monocycle" for match in matches)
    assert all(registry.spec_for_match(match).template.charge_policy == "exact" for match in matches)
    assert all(set(dict(match.local_to_input_atom).values()) == set(mol.atoms) for match in matches)
    assert tuple(registry.by_key) == before_keys
    assert tuple(hw_generated_names()) == before_hw_names


def test_systematic_hw_numbering_is_independent_of_face_cycle_orientation():
    registry = fusion_component_registry()
    mol, face = _triazole_ring()
    rotated = GraphCycle.from_atoms((2, 3, 4, 0, 1))
    reversed_cycle = GraphCycle.from_atoms(tuple(reversed(face.atoms)))

    def maps(cycle):
        return {
            match.local_to_input_atom
            for match in registry.match_faces(mol, (cycle,))
            if match.spec_key.startswith("generated-hw:")
        }

    assert maps(face) == maps(rotated) == maps(reversed_cycle)


def test_explicit_retained_component_precedes_systematic_hw_derivation():
    mol, face = _ring(
        ("O", "C", "C", "C", "C"),
        atom_ids=(40, 10, 50, 20, 30),
        nonaromatic_positions=frozenset({0}),
    )
    ring_edges = tuple(zip(face.atoms, face.atoms[1:] + face.atoms[:1]))
    for position, edge in enumerate(ring_edges):
        if position in {1, 3}:
            bond = mol.get_bond(*edge)
            assert bond is not None
            mol.update_bond(bond.idx, order=2)

    matches = fusion_component_registry().match_faces(mol, (face,))

    assert {match.spec_key for match in matches} == {"furan"}


@pytest.mark.parametrize("invalid_kind", ("charged", "nonstandard_valence", "saturated"))
def test_systematic_hw_component_rejects_out_of_scope_ring_chemistry(invalid_kind):
    mol, face = _triazole_ring()
    if invalid_kind == "charged":
        mol.update_atom(0, charge=1)
    elif invalid_kind == "nonstandard_valence":
        mol.add_atom("C", idx=9)
        mol.add_bond(0, 9, idx=999)
    else:
        for bond_id in tuple(mol.bonds):
            mol.update_bond(bond_id, order=1)

    matches = fusion_component_registry().match_faces(mol, (face,))

    assert not any(match.spec_key.startswith("generated-hw:") for match in matches)


@pytest.mark.parametrize(
    ("key", "ring_size", "prefix"),
    [
        ("cyclopentadiene", 5, "cyclopenta"),
        ("cycloheptatriene", 7, "cyclohepta"),
        ("cyclooctatetraene", 8, "cycloocta"),
    ],
)
def test_generated_carbocycle_policy_is_separate_from_shared_graph_construction(key, ring_size, prefix):
    component = fusion_component_registry().by_key[key]

    assert component.spec.template.family == "generated_monocycle"
    assert component.spec.template.locants == tuple(str(index) for index in range(1, ring_size + 1))
    assert component.spec.attached_prefix == prefix
    assert component.omit_attached_locants
    assert component.spec.rule_reference == "P-25.2.2"


def test_registration_rejects_duplicate_keys_and_template_names():
    data = load_json_table("fusion_components.json")
    row = deepcopy(next(item for item in data["components"] if item["key"] == "benzene"))
    registry = FusionComponentRegistry("test")
    registry.register(row)

    with pytest.raises(ValueError, match="duplicate fusion component key"):
        registry.register(row)

    duplicate_template = deepcopy(next(item for item in data["components"] if item["key"] == "naphthalene"))
    duplicate_template["template_names"] = [row["key"]]
    with pytest.raises(ValueError, match="duplicate fusion component template name"):
        registry.register(duplicate_template)


def test_registration_rejects_unknown_template_references():
    row = deepcopy(
        next(
            item
            for item in load_json_table("fusion_components.json")["components"]
            if item["key"] == "benzene"
        )
    )
    row["template_names"] = ["not-a-registered-template"]

    with pytest.raises(ValueError, match="references unknown templates"):
        FusionComponentRegistry("test").register(row)


def test_registration_inherits_names_from_the_shared_retained_template():
    data = load_json_table("fusion_components.json")
    row = deepcopy(next(item for item in data["components"] if item["key"] == "naphthalene"))
    row.pop("template_names", None)
    row.pop("parent_name", None)
    row.pop("attached_prefix", None)

    registry = FusionComponentRegistry("test")
    spec = registry.register(row)

    assert spec.parent_name == spec.template.output_name == "naphthalene"
    assert spec.attached_prefix == spec.template.attached_prefix == "naphtho"


def test_polycyclic_component_requires_explicit_horizontal_ring_count():
    data = load_json_table("fusion_components.json")
    row = deepcopy(next(item for item in data["components"] if item["key"] == "naphthalene"))
    row.pop("horizontal_ring_count")

    with pytest.raises(ValueError, match="requires an explicit horizontal_ring_count"):
        FusionComponentRegistry("test").register(row)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"allow_as_parent": False, "allow_as_attached": False}, "not eligible for any role"),
        ({"attached_prefix": ""}, "attached_prefix"),
        ({"rule": ""}, "rule"),
    ],
)
def test_registration_validates_role_prefix_and_rule_provenance(change: dict[str, object], message: str):
    row = deepcopy(load_json_table("fusion_components.json")["components"][5])
    row.update(change)

    with pytest.raises(ValueError, match=message):
        FusionComponentRegistry("test").register(row)


def test_registry_rejects_unknown_schema_versions():
    data = deepcopy(load_json_table("fusion_components.json"))
    data["schema_version"] = 999

    with pytest.raises(ValueError, match="unsupported fusion component schema version"):
        FusionComponentRegistry.from_data(data)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("generator", "text_lookup", "unsupported fusion component generator"),
        ("ring_size", 2, "ring_size must be at least three"),
        ("bond_class", "guessed", "unsupported monocyclic graph bond class"),
        ("pin_component", "yes", "pin_component must be a boolean"),
    ],
)
def test_generated_component_rows_are_strictly_validated(field, value, message):
    data = deepcopy(load_json_table("fusion_components.json"))
    data["generated_components"][0][field] = value

    with pytest.raises((TypeError, ValueError), match=message):
        FusionComponentRegistry.from_data(data)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"generator": "name_table"}, "unsupported systematic fusion component generator"),
        ({"allow_as_parent": "yes"}, "allow_as_parent must be a boolean"),
        ({"allow_as_parent": False, "allow_as_attached": False}, "not eligible for any role"),
        ({"rule": ""}, "rule must be a non-empty string"),
    ],
)
def test_systematic_component_generator_policy_is_strictly_validated(change, message):
    data = deepcopy(load_json_table("fusion_components.json"))
    data["systematic_component_generators"][0].update(change)

    with pytest.raises(ValueError, match=message):
        FusionComponentRegistry.from_data(data)


def test_attached_locant_omission_is_strictly_validated():
    data = deepcopy(load_json_table("fusion_components.json"))
    data["components"][0]["omit_attached_locants"] = "yes"

    with pytest.raises(ValueError, match="omit_attached_locants"):
        FusionComponentRegistry.from_data(data)


def test_match_faces_rejects_a_cycle_that_is_not_in_the_molecular_graph():
    mol, _ = _ring(("O", "C", "C", "C", "C"), atom_ids=(0, 1, 2, 3, 4), nonaromatic_positions=frozenset({0}))

    with pytest.raises(ValueError, match="non-bonded atom pairs"):
        fusion_component_registry().match_faces(mol, ((0, 1, 3, 2, 4),))
