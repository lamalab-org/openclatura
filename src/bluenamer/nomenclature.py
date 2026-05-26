"""Central data registry for nomenclature lookup tables.

This module is the integration point for data-backed naming vocabulary.  Code
that needs lookup data should prefer ``RULES.<group>.<field>`` over importing
individual module constants.
"""

from dataclasses import dataclass
from functools import lru_cache

from .naming_data import grouped_namer_rules


@dataclass(frozen=True)
class RetainedNameRules:
    indicated_hydrogen_names: set[str]
    ring_elements: set[str]
    substituent_stems: dict[str, tuple[str, str]]
    monocycle_specs: tuple[dict, ...]
    fused_polycycle_specs: tuple[dict, ...]


@dataclass(frozen=True)
class HeteroatomRules:
    alkyl_oxy_prefixes: dict[str, str]
    simple_sulfanyl_prefixes: set[str]
    simple_selanyl_prefixes: set[str]
    halogen_prefixes: dict[str, str]
    halogen_lambda_suffixes: dict[str, str]


@dataclass(frozen=True)
class RingRules:
    descriptor_templates: dict[str, str]
    polycycle_prefixes: dict[int, str]


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
    mononuclear_parent_hydrides: dict[str, str]
    replacement_parent_oxoacid_specs: tuple[dict, ...]


@dataclass(frozen=True)
class IonRules:
    single_atom_cations: set[str]
    single_atom_anions: dict[str, str]


@dataclass(frozen=True)
class ParentChargeSuffixRule:
    suffix: str
    reason: str


@dataclass(frozen=True)
class AnionSuffixPlacementRule:
    key: str
    suffix_pattern: str
    placement: str
    reason: str
    atom_symbols: tuple[str, ...] = ("*",)


@dataclass(frozen=True)
class ChargeRules:
    retained_ionic_n_parents: dict[str, str]
    saturated_n_ring_ionic_parents: dict[int, str]
    parent_charge_suffixes: dict[str, ParentChargeSuffixRule]
    replacement_charge_prefixes: dict[str, str]
    heteroatom_charge_prefixes: dict[str, str]
    anion_suffix_placements: tuple[AnionSuffixPlacementRule, ...]


@dataclass(frozen=True)
class AssemblyRules:
    replacement_prefix_order: dict[str, int]
    unsaturation_order: dict[str, int]
    acid_halide_suffix_keys: set[str]
    substituent_sort_prefix_pattern: str
    ambiguous_connection_substituent_stems: set[str]
    connection_boundary_parent_stems: tuple[str, ...]


@dataclass(frozen=True)
class RegexReplacement:
    pattern: str
    replacement: str
    category: str = "migration"
    reason: str = ""


@dataclass(frozen=True)
class LiteralReplacement:
    pattern: str
    replacement: str
    category: str = "migration"
    reason: str = ""


@dataclass(frozen=True)
class PostprocessRules:
    literal_replacements: tuple[LiteralReplacement, ...]
    regex_replacements: tuple[RegexReplacement, ...]
    exact_replacements: tuple[LiteralReplacement, ...]
    acyl_amido_terms: tuple[str, ...]
    n_substituted_functional_suffixes: tuple[str, ...]


@dataclass(frozen=True)
class MultiSuffixTemplate:
    """Template for rendering multiple principal characteristic groups."""

    multiplier_positions: tuple[int, ...] = (0,)


@dataclass(frozen=True)
class FunctionalGroupRule:
    key: str
    role: str
    prefix: str | None = None
    suffix: str | None = None
    multi_suffix: MultiSuffixTemplate | None = None
    suffix_multiplier_positions: tuple[int, ...] = (0,)
    seniority: int | None = None
    suffix_with_locant: bool = False
    needs_locant: bool = True
    perception_handler: str | None = None
    prefix_handler: str | None = None
    families: tuple[str, ...] = ()
    component_flags: tuple[str, ...] = ()
    postprocess_tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class FunctionalGroupRules:
    by_key: dict[str, FunctionalGroupRule]

    def has(self, key: str) -> bool:
        return key in self.by_key

    def get(self, key: str) -> FunctionalGroupRule:
        return self.by_key[key]

    def prefix_for(self, key: str) -> str | None:
        rule = self.by_key.get(key)
        return rule.prefix if rule else None

    def cited_prefix_for(self, key: str) -> str | None:
        prefix = self.prefix_for(key)
        if prefix is None:
            return None
        rule = self.by_key[key]
        if "acid_halide" in rule.families:
            return f"({prefix})"
        return prefix

    def direct_subgraph_prefix_for(self, key: str) -> str | None:
        rule = self.by_key.get(key)
        if rule is None:
            return None
        if rule.role == "prefix" or "direct_group" in rule.families:
            return rule.prefix
        return None

    def principal_keys(self) -> set[str]:
        return {key for key, rule in self.by_key.items() if rule.role == "principal"}

    def prefix_keys(self) -> set[str]:
        return {key for key, rule in self.by_key.items() if rule.prefix}

    def most_senior(self, keys: list[str]) -> FunctionalGroupRule:
        principal_rules = [
            self.by_key[key] for key in keys if key in self.by_key and self.by_key[key].seniority is not None
        ]
        if not principal_rules:
            raise KeyError(f"No seniority metadata for functional-group keys: {keys!r}")
        return min(principal_rules, key=lambda rule: rule.seniority)

    def keys_with_family(self, family: str) -> set[str]:
        return {key for key, rule in self.by_key.items() if family in rule.families}


@dataclass(frozen=True)
class NomenclatureRegistry:
    retained: RetainedNameRules
    heteroatoms: HeteroatomRules
    rings: RingRules
    prefixes: PrefixRules
    components: ComponentRules
    ions: IonRules
    charges: ChargeRules
    assembly: AssemblyRules
    functional_groups: FunctionalGroupRules
    postprocess: PostprocessRules


def _group_tuple_mapping(group_key: str, section: str) -> dict[str, tuple[str, str]]:
    return {key: tuple(value) for key, value in grouped_namer_rules()[group_key].mapping(section).items()}


def _functional_group_rules() -> FunctionalGroupRules:
    groups = {}

    for key, item in grouped_namer_rules()["functional_groups"].mapping("functional_groups").items():
        families = tuple(item.get("families", _derived_functional_group_families(key)))
        suffix = item.get("suffix")
        multi_suffix = item.get("multi_suffix")
        groups[key] = FunctionalGroupRule(
            key=key,
            role=item["role"],
            prefix=item.get("prefix"),
            suffix=suffix,
            multi_suffix=_multi_suffix_template(suffix, multi_suffix),
            suffix_multiplier_positions=_suffix_multiplier_positions(suffix, multi_suffix),
            seniority=item.get("seniority"),
            suffix_with_locant=bool(item.get("suffix_with_locant", False)),
            needs_locant=bool(item.get("needs_locant", True)),
            perception_handler=item.get("perception_handler"),
            prefix_handler=item.get("prefix_handler"),
            families=families,
            component_flags=tuple(item.get("component_flags", [])),
            postprocess_tags=tuple(item.get("postprocess_tags", [])),
        )
    return FunctionalGroupRules(by_key=groups)


def _multi_suffix_template(suffix: str | None, multi_suffix) -> MultiSuffixTemplate | None:
    if multi_suffix is None:
        return None
    return MultiSuffixTemplate(multiplier_positions=_suffix_multiplier_positions(suffix, multi_suffix))


def _suffix_multiplier_positions(suffix: str | None, multi_suffix) -> tuple[int, ...]:
    """Return which words in a suffix phrase take multiplicative prefixes.

    Built-in rows store ``multi_suffix`` as a template object.  String support
    remains as an external compatibility path for older override data.
    """

    if not suffix:
        return (0,)
    if isinstance(multi_suffix, dict):
        return tuple(int(position) for position in multi_suffix.get("multiplier_positions", [0]))
    words = suffix.split()
    if len(words) == 1:
        return (0,)
    if not multi_suffix:
        return (0,)
    multi_words = multi_suffix.split()
    if len(multi_words) != len(words):
        return (0,)
    positions = []
    for idx, (word, multi_word) in enumerate(zip(words, multi_words, strict=True)):
        if multi_word == f"di{word}":
            positions.append(idx)
    return tuple(positions) or (0,)


def _derived_functional_group_families(key: str) -> tuple[str, ...]:
    families = set()
    functional_group_rules = grouped_namer_rules()["functional_groups"]
    substituent_vocabulary = grouped_namer_rules()["substituent_vocabulary"]
    family_sections = {
        "chain_external_carbonyl": "chain_external_carbonyl_groups",
        "prefix_skip": "prefix_groups_to_skip",
        "ester_like": "ester_like_prefix_groups",
        "peroxy_ester": "peroxy_ester_groups",
        "amide_like": "amide_like_prefix_groups",
        "carboxy_prefix": "carboxy_prefix_groups",
        "cyano_prefix": "cyano_prefix_groups",
        "peroxy_acid": "peroxy_acid_prefix_groups",
        "sulfonyl": "sulfonyl_prefix_groups",
        "front_modifier": "front_modifier_principal_groups",
        "n_substitutable": "n_substituent_principal_groups",
        "hydrazone": "hydrazone_principal_groups",
    }
    for family, section in family_sections.items():
        if key in functional_group_rules.values(section):
            families.add(family)
    if key in substituent_vocabulary.mapping("acid_halide_prefixes"):
        families.add("acid_halide")
    if key in substituent_vocabulary.mapping("direct_prefix_groups"):
        families.add("direct_prefix")
    if key in substituent_vocabulary.mapping("direct_group_prefixes"):
        families.add("direct_group")
    return tuple(sorted(families))


def _postprocess_rules() -> PostprocessRules:
    group = grouped_namer_rules()["postprocessing"]
    return PostprocessRules(
        literal_replacements=tuple(
            LiteralReplacement(
                pattern=item["pattern"],
                replacement=item["replacement"],
                category=item["category"],
                reason=item["reason"],
            )
            for item in group.values("postprocess_literal_replacements")
        ),
        regex_replacements=tuple(
            RegexReplacement(
                pattern=item["pattern"],
                replacement=item["replacement"],
                category=item["category"],
                reason=item["reason"],
            )
            for item in group.values("postprocess_regex_replacements")
        ),
        exact_replacements=tuple(
            LiteralReplacement(
                pattern=item["pattern"],
                replacement=item["replacement"],
                category=item["category"],
                reason=item["reason"],
            )
            for item in group.values("postprocess_exact_replacements")
        ),
        acyl_amido_terms=tuple(group.values("postprocess_acyl_amido_terms")),
        n_substituted_functional_suffixes=tuple(group.values("postprocess_n_substituted_functional_suffixes")),
    )


def _charge_rules() -> ChargeRules:
    group = grouped_namer_rules()["charges"]
    return ChargeRules(
        retained_ionic_n_parents=group.mapping("retained_ionic_n_parents"),
        saturated_n_ring_ionic_parents={
            int(key): value for key, value in group.mapping("saturated_n_ring_ionic_parents").items()
        },
        parent_charge_suffixes={
            key: ParentChargeSuffixRule(
                suffix=value["suffix"],
                reason=value["reason"],
            )
            for key, value in group.mapping("parent_charge_suffixes").items()
        },
        replacement_charge_prefixes=group.mapping("replacement_charge_prefixes"),
        heteroatom_charge_prefixes=group.mapping("heteroatom_charge_prefixes"),
        anion_suffix_placements=tuple(
            AnionSuffixPlacementRule(
                key=item["key"],
                suffix_pattern=item.get("suffix_pattern", ""),
                placement=item["placement"],
                reason=item["reason"],
                atom_symbols=tuple(item.get("atom_symbols", ["*"])),
            )
            for item in group.values("anion_suffix_placements")
        ),
    )


@lru_cache(maxsize=1)
def registry() -> NomenclatureRegistry:
    """Return the grouped nomenclature lookup registry."""

    groups = grouped_namer_rules()
    retained = groups["retained_parents"]
    simple_components = groups["simple_components"]
    substituent_vocabulary = groups["substituent_vocabulary"]
    functional_group_rules = groups["functional_groups"]
    ring_descriptors = groups["ring_descriptors"]
    assembly_grammar = groups["assembly_grammar"]
    return NomenclatureRegistry(
        retained=RetainedNameRules(
            indicated_hydrogen_names=set(retained.values("indicated_hydrogen_retained_names")),
            ring_elements=set(retained.values("retained_ring_elements")),
            substituent_stems=_group_tuple_mapping("retained_parents", "retained_substituent_stems"),
            monocycle_specs=tuple(retained.values("retained_monocycle_specs")),
            fused_polycycle_specs=tuple(retained.values("retained_fused_polycycle_specs")),
        ),
        heteroatoms=HeteroatomRules(
            alkyl_oxy_prefixes=substituent_vocabulary.mapping("alkyl_oxy_prefixes"),
            simple_sulfanyl_prefixes=set(substituent_vocabulary.values("simple_sulfanyl_prefixes")),
            simple_selanyl_prefixes=set(substituent_vocabulary.values("simple_selanyl_prefixes")),
            halogen_prefixes=substituent_vocabulary.mapping("halogen_prefixes"),
            halogen_lambda_suffixes=substituent_vocabulary.mapping("halogen_lambda_suffixes"),
        ),
        rings=RingRules(
            descriptor_templates=ring_descriptors.mapping("ring_descriptor_templates"),
            polycycle_prefixes={
                int(key): value for key, value in ring_descriptors.mapping("polycycle_prefixes").items()
            },
        ),
        prefixes=PrefixRules(
            direct_group_prefixes=substituent_vocabulary.mapping("direct_group_prefixes"),
            skip_groups=set(functional_group_rules.values("prefix_groups_to_skip")),
            ester_like_groups=set(functional_group_rules.values("ester_like_prefix_groups")),
            peroxy_ester_groups=set(functional_group_rules.values("peroxy_ester_groups")),
            amide_like_groups=set(functional_group_rules.values("amide_like_prefix_groups")),
            amide_bases=substituent_vocabulary.mapping("amide_prefix_bases"),
            carboxy_groups=set(functional_group_rules.values("carboxy_prefix_groups")),
            cyano_groups=set(functional_group_rules.values("cyano_prefix_groups")),
            acid_halide_prefixes=substituent_vocabulary.mapping("acid_halide_prefixes"),
            peroxy_acid_groups=set(functional_group_rules.values("peroxy_acid_prefix_groups")),
            sulfonyl_groups=set(functional_group_rules.values("sulfonyl_prefix_groups")),
            direct_prefixes=substituent_vocabulary.mapping("direct_prefix_groups"),
        ),
        components=ComponentRules(
            chain_external_carbonyl_groups=set(functional_group_rules.values("chain_external_carbonyl_groups")),
            front_modifier_principal_groups=set(functional_group_rules.values("front_modifier_principal_groups")),
            n_substituent_principal_groups=set(functional_group_rules.values("n_substituent_principal_groups")),
            hydrazone_principal_groups=set(functional_group_rules.values("hydrazone_principal_groups")),
            special_names=simple_components.mapping("special_component_names"),
            salt_metal_names=set(simple_components.values("salt_metal_names")),
            mononuclear_parent_hydrides=simple_components.mapping("mononuclear_parent_hydrides"),
            replacement_parent_oxoacid_specs=tuple(simple_components.values("replacement_parent_oxoacid_specs")),
        ),
        ions=IonRules(
            single_atom_cations=set(simple_components.values("single_atom_cations")),
            single_atom_anions=simple_components.mapping("single_atom_anions"),
        ),
        charges=_charge_rules(),
        assembly=AssemblyRules(
            replacement_prefix_order={
                key: int(value) for key, value in assembly_grammar.mapping("replacement_prefix_order").items()
            },
            unsaturation_order={
                key: int(value) for key, value in assembly_grammar.mapping("unsaturation_order").items()
            },
            acid_halide_suffix_keys=set(assembly_grammar.values("acid_halide_suffix_keys")),
            substituent_sort_prefix_pattern=assembly_grammar.mapping("substituent_sort")["prefix_pattern"],
            ambiguous_connection_substituent_stems=set(
                assembly_grammar.values("ambiguous_connection_substituent_stems")
            ),
            connection_boundary_parent_stems=tuple(assembly_grammar.values("connection_boundary_parent_stems")),
        ),
        functional_groups=_functional_group_rules(),
        postprocess=_postprocess_rules(),
    )


RULES = registry()
