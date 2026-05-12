"""Central data registry for nomenclature lookup tables.

This module is the integration point for data-backed naming vocabulary.  Code
that needs lookup data should prefer ``RULES.<group>.<field>`` over importing
individual module constants.
"""

from dataclasses import dataclass
from functools import lru_cache

from .naming_data import mapping, values


@dataclass(frozen=True)
class RetainedNameRules:
    indicated_hydrogen_names: set[str]
    ring_elements: set[str]
    substituent_stems: dict[str, tuple[str, str]]


@dataclass(frozen=True)
class HeteroatomRules:
    alkyl_oxy_prefixes: dict[str, str]
    simple_sulfanyl_prefixes: set[str]
    simple_selanyl_prefixes: set[str]
    halogen_prefixes: dict[str, str]
    halogen_lambda_suffixes: dict[str, str]


@dataclass(frozen=True)
class PrefixRules:
    direct_group_prefixes: dict[str, str]
    skip_groups: set[str]
    ester_like_groups: set[str]
    peroxy_ester_groups: set[str]
    amide_like_groups: set[str]
    amide_bases: dict[str, str]
    carboxy_groups: set[str]
    cyano_groups: set[str]
    acid_halide_prefixes: dict[str, str]
    peroxy_acid_groups: set[str]
    sulfonyl_groups: set[str]
    direct_prefixes: dict[str, str]


@dataclass(frozen=True)
class ComponentRules:
    chain_external_carbonyl_groups: set[str]
    front_modifier_principal_groups: set[str]
    n_substituent_principal_groups: set[str]
    hydrazone_principal_groups: set[str]
    special_names: dict[str, str]
    salt_metal_names: set[str]


@dataclass(frozen=True)
class IonRules:
    single_atom_cations: set[str]
    single_atom_anions: dict[str, str]


@dataclass(frozen=True)
class AssemblyRules:
    replacement_prefix_order: dict[str, int]
    unsaturation_order: dict[str, int]
    acid_halide_suffix_keys: set[str]
    substituent_sort_prefix_pattern: str


@dataclass(frozen=True)
class RegexReplacement:
    pattern: str
    replacement: str


@dataclass(frozen=True)
class PostprocessRules:
    literal_replacements: tuple[tuple[str, str], ...]
    regex_replacements: tuple[RegexReplacement, ...]
    exact_replacements: dict[str, str]
    acyl_amido_terms: tuple[str, ...]


@dataclass(frozen=True)
class FunctionalGroupRule:
    key: str
    role: str
    prefix: str | None = None
    suffix: str | None = None
    multi_suffix: str | None = None
    seniority: int | None = None
    suffix_with_locant: bool = False
    needs_locant: bool = True
    perception_handler: str | None = None
    prefix_handler: str | None = None
    component_flags: tuple[str, ...] = ()
    postprocess_tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class FunctionalGroupRules:
    by_key: dict[str, FunctionalGroupRule]


@dataclass(frozen=True)
class NomenclatureRegistry:
    retained: RetainedNameRules
    heteroatoms: HeteroatomRules
    prefixes: PrefixRules
    components: ComponentRules
    ions: IonRules
    assembly: AssemblyRules
    functional_groups: FunctionalGroupRules
    postprocess: PostprocessRules


def _tuple_mapping(section: str) -> dict[str, tuple[str, str]]:
    return {key: tuple(value) for key, value in mapping(section).items()}


def _functional_group_rules() -> FunctionalGroupRules:
    groups = {}
    from .rules import substituents, suffixes

    for key, rule in suffixes.GROUPS.items():
        groups[key] = FunctionalGroupRule(
            key=key,
            role="principal",
            prefix=rule.prefix,
            suffix=rule.suffix,
            multi_suffix=rule.multi_suffix,
            seniority=rule.seniority,
            suffix_with_locant=rule.suffix_with_locant,
            needs_locant=True,
        )
    for key, rule in substituents.SUBSTITUENTS.items():
        groups[key] = FunctionalGroupRule(
            key=key,
            role="prefix",
            prefix=rule.prefix,
            needs_locant=rule.needs_locant,
        )
    for key, item in mapping("functional_groups").items():
        groups[key] = FunctionalGroupRule(
            key=key,
            role=item["role"],
            prefix=item.get("prefix"),
            suffix=item.get("suffix"),
            multi_suffix=item.get("multi_suffix"),
            seniority=item.get("seniority"),
            suffix_with_locant=bool(item.get("suffix_with_locant", False)),
            needs_locant=bool(item.get("needs_locant", True)),
            perception_handler=item.get("perception_handler"),
            prefix_handler=item.get("prefix_handler"),
            component_flags=tuple(item.get("component_flags", [])),
            postprocess_tags=tuple(item.get("postprocess_tags", [])),
        )
    return FunctionalGroupRules(by_key=groups)


def _postprocess_rules() -> PostprocessRules:
    return PostprocessRules(
        literal_replacements=tuple(tuple(item) for item in values("postprocess_literal_replacements")),
        regex_replacements=tuple(
            RegexReplacement(pattern=item["pattern"], replacement=item["replacement"])
            for item in values("postprocess_regex_replacements")
        ),
        exact_replacements=mapping("postprocess_exact_replacements"),
        acyl_amido_terms=tuple(values("postprocess_acyl_amido_terms")),
    )


@lru_cache(maxsize=1)
def registry() -> NomenclatureRegistry:
    """Return the grouped nomenclature lookup registry."""

    return NomenclatureRegistry(
        retained=RetainedNameRules(
            indicated_hydrogen_names=set(values("indicated_hydrogen_retained_names")),
            ring_elements=set(values("retained_ring_elements")),
            substituent_stems=_tuple_mapping("retained_substituent_stems"),
        ),
        heteroatoms=HeteroatomRules(
            alkyl_oxy_prefixes=mapping("alkyl_oxy_prefixes"),
            simple_sulfanyl_prefixes=set(values("simple_sulfanyl_prefixes")),
            simple_selanyl_prefixes=set(values("simple_selanyl_prefixes")),
            halogen_prefixes=mapping("halogen_prefixes"),
            halogen_lambda_suffixes=mapping("halogen_lambda_suffixes"),
        ),
        prefixes=PrefixRules(
            direct_group_prefixes=mapping("direct_group_prefixes"),
            skip_groups=set(values("prefix_groups_to_skip")),
            ester_like_groups=set(values("ester_like_prefix_groups")),
            peroxy_ester_groups=set(values("peroxy_ester_groups")),
            amide_like_groups=set(values("amide_like_prefix_groups")),
            amide_bases=mapping("amide_prefix_bases"),
            carboxy_groups=set(values("carboxy_prefix_groups")),
            cyano_groups=set(values("cyano_prefix_groups")),
            acid_halide_prefixes=mapping("acid_halide_prefixes"),
            peroxy_acid_groups=set(values("peroxy_acid_prefix_groups")),
            sulfonyl_groups=set(values("sulfonyl_prefix_groups")),
            direct_prefixes=mapping("direct_prefix_groups"),
        ),
        components=ComponentRules(
            chain_external_carbonyl_groups=set(values("chain_external_carbonyl_groups")),
            front_modifier_principal_groups=set(values("front_modifier_principal_groups")),
            n_substituent_principal_groups=set(values("n_substituent_principal_groups")),
            hydrazone_principal_groups=set(values("hydrazone_principal_groups")),
            special_names=mapping("special_component_names"),
            salt_metal_names=set(values("salt_metal_names")),
        ),
        ions=IonRules(
            single_atom_cations=set(values("single_atom_cations")),
            single_atom_anions=mapping("single_atom_anions"),
        ),
        assembly=AssemblyRules(
            replacement_prefix_order={key: int(value) for key, value in mapping("replacement_prefix_order").items()},
            unsaturation_order={key: int(value) for key, value in mapping("unsaturation_order").items()},
            acid_halide_suffix_keys=set(values("acid_halide_suffix_keys")),
            substituent_sort_prefix_pattern=mapping("substituent_sort")["prefix_pattern"],
        ),
        functional_groups=_functional_group_rules(),
        postprocess=_postprocess_rules(),
    )


RULES = registry()
