"""Retained fusion components inherit graph and vocabulary from one registry."""

from dataclasses import replace

import pytest

from openclatura.fusion.registry import (
    FusionComponentRegistry,
    _is_retained_benzoheterocycle,
    fusion_component_registry,
)
from openclatura.retained_fused_templates import retained_graph_templates


def test_retained_benzoheterocycle_attachment_reuses_template_metadata():
    registry = fusion_component_registry()
    for template in retained_graph_templates():
        if not _is_retained_benzoheterocycle(template):
            continue
        matches = [component for component in registry.components if template.name in component.template_names]
        assert len(matches) == 1, template.name
        spec = matches[0].spec_for_template(template.name)
        assert spec.usable_as_attached
        assert spec.template.atoms == template.atoms
        assert spec.template.bonds == template.bonds
        assert spec.template.default_indicated_h == template.default_indicated_h


@pytest.mark.parametrize("attribute, value", [("charge", 1), ("saturated", True)])
def test_retained_family_rejects_charged_or_intrinsically_saturated_templates(attribute, value):
    template = next(t for t in retained_graph_templates() if _is_retained_benzoheterocycle(t))
    atom = replace(template.atoms[0], **{attribute: value})
    assert not _is_retained_benzoheterocycle(replace(template, atoms=(atom, *template.atoms[1:])))


def test_retained_family_requires_the_shared_edge_not_just_two_shared_vertices():
    template = next(t for t in retained_graph_templates() if _is_retained_benzoheterocycle(t))
    shared = frozenset(template.rings[0]) & frozenset(template.rings[1])
    broken = replace(template, bonds=tuple(b for b in template.bonds if frozenset(b.locants) != shared))
    assert not _is_retained_benzoheterocycle(broken)


def test_unknown_retained_component_generator_is_rejected():
    with pytest.raises(ValueError, match="unsupported retained fusion-component family"):
        FusionComponentRegistry.from_data(
            {
                "schema_version": 1,
                "registry_version": "test",
                "components": [],
                "generated_components": [],
                "retained_component_generators": [{"family": "unknown"}],
            }
        )
