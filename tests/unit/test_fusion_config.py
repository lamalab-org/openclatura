from copy import deepcopy

import pytest

from openclatura.fusion.config import (
    fusion_nomenclature_config,
    fusion_nomenclature_config_from_data,
)
from openclatura.naming_data import load_json_table


def test_checked_in_fusion_configuration_is_complete_and_data_backed():
    config = fusion_nomenclature_config()

    assert config.graph_source == "retained_graph_templates"
    assert config.numbering.interior_carbon_profile == "legacy_cas_letter_continuation"
    assert config.numbering.rule_reference == "FR-5.5.2"
    assert config.numbering.opsin_compatibility_version == "2.9.0"
    assert config.rules.planner_tier == "bounded-polycomponent-v3"
    assert config.rules.support.cover_kinds == ("tree",)
    assert config.rules.support.join_kinds == ("ortho", "ortho_peri", "higher_order")
    assert config.rules.support.multiparent_parents
    assert config.rules.support.maximum_tree_component_occurrences == 16
    assert config.rules.support.maximum_multiparent_component_occurrences == 3
    assert config.rules.support.maximum_multiparent_interparent_faces == 1
    assert config.rules.support.charged_parents
    assert config.rules.support.nonstandard_valence
    assert config.rules.support.interior_atoms
    assert config.rules.support.maximum_indicated_hydrogens is None
    assert config.rules.minimum_ring_size == 5
    assert config.rules.minimum_ring_count == 2
    assert {shape.ring_size for shape in config.ring_shapes} == set(range(3, 9))
    assert config.search.maximum_faces == 24
    assert not hasattr(config.search, "mancude_states")
    assert not hasattr(config.search, "interior_numbering_states")


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("graph_source",), "independent_templates", "shared retained"),
        (("search_limits", "maximum_faces"), 0, "must be positive"),
        (("ring_shapes",), [], "ring_shapes"),
    ],
)
def test_invalid_fusion_configuration_is_rejected(path, value, message):
    data = deepcopy(load_json_table("fusion_components.json"))
    target = data
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(ValueError, match=message):
        fusion_nomenclature_config_from_data(data)


@pytest.mark.parametrize(("field", "value"), [("cover_kinds", ["multiparent"])])
def test_configuration_cannot_enable_an_unimplemented_proof_tier(field, value):
    data = deepcopy(load_json_table("fusion_components.json"))
    data["rules"]["support"][field] = value

    with pytest.raises(ValueError, match="unimplemented|exceed the implemented"):
        fusion_nomenclature_config_from_data(data)


def test_face_limit_is_independent_of_component_grammar_support():
    data = deepcopy(load_json_table("fusion_components.json"))
    data["search_limits"]["maximum_faces"] = 19

    config = fusion_nomenclature_config_from_data(data)

    assert config.search.maximum_faces == 19
