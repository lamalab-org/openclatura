"""Public family-neutral API for graph-backed retained parents.

The implementation currently lives in :mod:`retained_fused_templates` so
existing imports remain compatible. New parent families should depend on this
module and register templates through the shared data registry.
"""

from .retained_fused_templates import (
    RetainedGraphAtomTemplate,
    RetainedGraphBondTemplate,
    RetainedGraphTemplate,
    RetainedGraphTemplateMatch,
    match_retained_graph_template_maps,
    match_retained_graph_templates,
    molecule_graph_topology_key,
    retained_graph_template_from_data,
    retained_graph_template_topology_key,
    retained_graph_templates,
    retained_parent_metadata,
    template_molecule,
    validate_retained_fused_template,
)

validate_retained_graph_template = validate_retained_fused_template


def validate_retained_graph_family_partition() -> None:
    """Assert that provider families occupy disjoint topology buckets.

    Production matching may stop after the first family with an exact match;
    this audit keeps that optimization valid as new providers are registered.
    """

    owners: dict[tuple, tuple[str, str]] = {}
    for template in retained_graph_templates(include_disabled=True):
        key = retained_graph_template_topology_key(template)
        owner = owners.get(key)
        if owner is not None and owner[0] != template.family:
            raise ValueError(
                f"Retained graph topology is shared by provider families: "
                f"{owner[0]}:{owner[1]} and {template.family}:{template.name}."
            )
        owners[key] = (template.family, template.name)

__all__ = (
    "RetainedGraphAtomTemplate",
    "RetainedGraphBondTemplate",
    "RetainedGraphTemplate",
    "RetainedGraphTemplateMatch",
    "match_retained_graph_template_maps",
    "match_retained_graph_templates",
    "molecule_graph_topology_key",
    "retained_graph_template_from_data",
    "retained_graph_template_topology_key",
    "retained_graph_templates",
    "retained_parent_metadata",
    "template_molecule",
    "validate_retained_graph_template",
    "validate_retained_graph_family_partition",
)
