"""Grouped layout for data-backed namer rules.  The flat on-disk table is loaded through domain
groups so related vocabulary stays discoverable."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RuleGroupSpec:
    key: str
    sections: tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class RuleGroupView:
    spec: RuleGroupSpec
    table: dict[str, Any]

    def values(self, section: str) -> list[Any]:
        self._require(section)
        return self.table[section]["values"]

    def mapping(self, section: str) -> dict[str, Any]:
        self._require(section)
        return self.table[section]["values"]

    def _require(self, section: str) -> None:
        if section not in self.spec.sections:
            raise KeyError(f"Section {section!r} does not belong to rule group {self.spec.key!r}.")


RULE_GROUPS: tuple[RuleGroupSpec, ...] = (
    RuleGroupSpec(
        key="parent_policy",
        sections=(
            "parent_selection_criteria",
            "parent_seniority_criteria_available",
            "numbering_criteria",
        ),
        reason="Parent choice and numbering preference are both seniority policy.",
    ),
    RuleGroupSpec(
        key="retained_parents",
        sections=(
            "retained_ring_elements",
            "retained_substituent_stems",
            "retained_monocycle_specs",
            "retained_fused_polycycle_specs",
        ),
        reason="Retained parent recognition, locants, and substituent stems must evolve together.",
    ),
    RuleGroupSpec(
        key="simple_components",
        sections=(
            "single_atom_cations",
            "single_atom_anions",
            "salt_metal_names",
            "mononuclear_parent_hydrides",
            "retained_homonuclear_chain_names",
            "replacement_parent_oxoacid_specs",
        ),
        reason="Disconnected/simple component names are resolved before parent assembly.",
    ),
    RuleGroupSpec(
        key="charges",
        sections=(
            "retained_ionic_n_parents",
            "saturated_n_ring_ionic_parents",
            "parent_charge_suffixes",
            "replacement_charge_prefixes",
            "heteroatom_charge_prefixes",
            "anion_suffix_placements",
        ),
        reason="Formal charge spelling, ionic retained parents, and suffix placement share locant rules.",
    ),
    RuleGroupSpec(
        key="substituent_vocabulary",
        sections=(
            "alkyl_oxy_prefixes",
            "simple_sulfanyl_prefixes",
            "simple_selanyl_prefixes",
            "retained_sulfonyl_group_names",
            "retained_sulfonyl_ligand_contractions",
            "halogen_prefixes",
            "halogen_lambda_suffixes",
            "direct_group_prefixes",
            "direct_prefix_groups",
            "acid_halide_prefixes",
            "amide_prefix_bases",
        ),
        reason="Reusable prefix vocabulary and contractions used by substituent construction.",
    ),
    RuleGroupSpec(
        key="functional_groups",
        sections=(
            "functional_groups",
            "chain_external_carbonyl_groups",
            "prefix_groups_to_skip",
            "ester_like_prefix_groups",
            "peroxy_ester_groups",
            "amide_like_prefix_groups",
            "carboxy_prefix_groups",
            "cyano_prefix_groups",
            "peroxy_acid_prefix_groups",
            "sulfonyl_prefix_groups",
            "front_modifier_principal_groups",
            "hydrazone_principal_groups",
        ),
        reason="Functional-group rows and their derived behavior families are one extension surface.",
    ),
    RuleGroupSpec(
        key="ring_descriptors",
        sections=(
            "ring_descriptor_templates",
            "polycycle_prefixes",
        ),
        reason="Ring descriptor vocabulary is shared by spiro, bicyclo, and polycycle renderers.",
    ),
    RuleGroupSpec(
        key="assembly_grammar",
        sections=(
            "replacement_prefix_order",
            "unsaturation_order",
            "acid_halide_suffix_keys",
            "substituent_sort",
            "substituent_attachment_suffixes",
            "ambiguous_connection_substituent_stems",
            "suffix_nitrogen_markers",
        ),
        reason="Name assembly ordering, sorting, and connection-boundary grammar are rendering policy.",
    ),
    RuleGroupSpec(
        key="postprocessing",
        sections=("postprocess_literal_replacements",),
        reason="Compatibility rewrites and postprocessing inventories need one owner.",
    ),
)


def rule_groups(table: dict[str, Any]) -> dict[str, RuleGroupView]:
    return {spec.key: RuleGroupView(spec=spec, table=table) for spec in RULE_GROUPS}


def section_group_map() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for spec in RULE_GROUPS:
        for section in spec.sections:
            if section in mapping:
                raise ValueError(f"Rule section {section!r} is assigned to multiple groups.")
            mapping[section] = spec.key
    return mapping
