from __future__ import annotations

from copy import deepcopy

import pytest

from openclatura.fusion.faces import GraphCycle
from openclatura.fusion.registry import (
    FusionComponentRegistry,
    FusionComponentRole,
    fusion_component_registry,
    version,
)
from openclatura.molecule import Molecule
from openclatura.naming_data import load_json_table


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


def test_registration_rejects_duplicate_keys_and_template_names():
    data = load_json_table("fusion_components.json")
    row = deepcopy(data["components"][0])
    registry = FusionComponentRegistry("test")
    registry.register(row)

    with pytest.raises(ValueError, match="duplicate fusion component key"):
        registry.register(row)

    duplicate_template = deepcopy(data["components"][1])
    duplicate_template["template_names"] = row["template_names"]
    with pytest.raises(ValueError, match="duplicate fusion component template name"):
        registry.register(duplicate_template)


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


def test_match_faces_rejects_a_cycle_that_is_not_in_the_molecular_graph():
    mol, _ = _ring(("O", "C", "C", "C", "C"), atom_ids=(0, 1, 2, 3, 4), nonaromatic_positions=frozenset({0}))

    with pytest.raises(ValueError, match="non-bonded atom pairs"):
        fusion_component_registry().match_faces(mol, ((0, 1, 3, 2, 4),))
