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


@pytest.mark.parametrize(
    ("key", "symbols", "atom_ids", "nonaromatic", "prefix", "map_count"),
    [
        ("furan", ("O", "C", "C", "C", "C"), (40, 10, 50, 20, 30), {0}, "furo", 2),
        ("thiophene", ("S", "C", "C", "C", "C"), (9, 4, 8, 2, 6), {0}, "thieno", 2),
        ("imidazole", ("N", "C", "N", "C", "C"), (12, 3, 19, 5, 8), {0}, "imidazo", 2),
        ("thiazole", ("S", "C", "N", "C", "C"), (32, 7, 41, 11, 23), {0}, "thiazolo", 1),
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


def test_attached_locant_omission_is_strictly_validated():
    data = deepcopy(load_json_table("fusion_components.json"))
    data["components"][0]["omit_attached_locants"] = "yes"

    with pytest.raises(ValueError, match="omit_attached_locants"):
        FusionComponentRegistry.from_data(data)


def test_match_faces_rejects_a_cycle_that_is_not_in_the_molecular_graph():
    mol, _ = _ring(("O", "C", "C", "C", "C"), atom_ids=(0, 1, 2, 3, 4), nonaromatic_positions=frozenset({0}))

    with pytest.raises(ValueError, match="non-bonded atom pairs"):
        fusion_component_registry().match_faces(mol, ((0, 1, 3, 2, 4),))
