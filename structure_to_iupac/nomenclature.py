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
    ambiguous_connection_substituent_stems: set[str]
    connection_boundary_parent_stems: tuple[str, ...]


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
    n_substituted_functional_suffixes: tuple[str, ...]


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
        principal_rules = [self.by_key[key] for key in keys if key in self.by_key and self.by_key[key].seniority is not None]
        if not principal_rules:
            raise KeyError(f"No seniority metadata for functional-group keys: {keys!r}")
        return min(principal_rules, key=lambda rule: rule.seniority)

    def keys_with_family(self, family: str) -> set[str]:
        return {key for key, rule in self.by_key.items() if family in rule.families}


PRINCIPAL_FUNCTIONAL_GROUP_ROWS: tuple[dict, ...] = (
    {"key": "olate", "seniority": 15, "suffix": "olate", "suffix_with_locant": True, "prefix": "oxido", "multi_suffix": "diolate"},
    {"key": "thiolate", "seniority": 16, "suffix": "thiolate", "suffix_with_locant": True, "prefix": "sulfido", "multi_suffix": "dithiolate"},
    {"key": "carboxylic_acid", "seniority": 20, "suffix": "oic acid", "suffix_with_locant": False, "prefix": "carboxy", "multi_suffix": "dioic acid"},
    {"key": "carboxylate", "seniority": 21, "suffix": "oate", "suffix_with_locant": False, "prefix": "carboxylato", "multi_suffix": "dioate"},
    {"key": "ring_carboxylic_acid", "seniority": 20, "suffix": "carboxylic acid", "suffix_with_locant": True, "prefix": "carboxy", "multi_suffix": "dicarboxylic acid"},
    {"key": "ring_carboxylate", "seniority": 21, "suffix": "carboxylate", "suffix_with_locant": True, "prefix": "carboxylato", "multi_suffix": "dicarboxylate"},
    {"key": "peroxy_acid", "seniority": 22, "suffix": "peroxoic acid", "suffix_with_locant": False, "prefix": "carboperoxy", "multi_suffix": "diperoxoic acid"},
    {"key": "ring_peroxy_acid", "seniority": 22, "suffix": "carboperoxoic acid", "suffix_with_locant": True, "prefix": "carboperoxy", "multi_suffix": "dicarboperoxoic acid"},
    {"key": "peroxy_ester", "seniority": 45, "suffix": "peroxoate", "suffix_with_locant": False, "prefix": "oxycarbonyl", "multi_suffix": None},
    {"key": "ring_peroxy_ester", "seniority": 45, "suffix": "carboperoxoate", "suffix_with_locant": True, "prefix": "oxycarbonyl", "multi_suffix": None},
    {"key": "sulfonic_acid", "seniority": 25, "suffix": "sulfonic acid", "suffix_with_locant": True, "prefix": "sulfo", "multi_suffix": "disulfonic acid"},
    {"key": "sulfonate", "seniority": 26, "suffix": "sulfonate", "suffix_with_locant": True, "prefix": "sulfonato", "multi_suffix": "disulfonate"},
    {"key": "anhydride", "seniority": 30, "suffix": "oic anhydride", "suffix_with_locant": False, "prefix": None, "multi_suffix": None},
    {"key": "ester", "seniority": 40, "suffix": "oate", "suffix_with_locant": False, "prefix": "oxycarbonyl", "multi_suffix": None},
    {"key": "acid_fluoride", "seniority": 50, "suffix": "oyl fluoride", "suffix_with_locant": False, "prefix": "fluorocarbonyl", "multi_suffix": "dioyl difluoride"},
    {"key": "acid_chloride", "seniority": 51, "suffix": "oyl chloride", "suffix_with_locant": False, "prefix": "chlorocarbonyl", "multi_suffix": "dioyl dichloride"},
    {"key": "acid_bromide", "seniority": 52, "suffix": "oyl bromide", "suffix_with_locant": False, "prefix": "bromocarbonyl", "multi_suffix": "dioyl dibromide"},
    {"key": "acid_iodide", "seniority": 53, "suffix": "oyl iodide", "suffix_with_locant": False, "prefix": "iodocarbonyl", "multi_suffix": "dioyl diiodide"},
    {"key": "ring_acid_fluoride", "seniority": 50, "suffix": "carbonyl fluoride", "suffix_with_locant": True, "prefix": "fluorocarbonyl", "multi_suffix": "dicarbonyl difluoride"},
    {"key": "ring_acid_chloride", "seniority": 51, "suffix": "carbonyl chloride", "suffix_with_locant": True, "prefix": "chlorocarbonyl", "multi_suffix": "dicarbonyl dichloride"},
    {"key": "ring_acid_bromide", "seniority": 52, "suffix": "carbonyl bromide", "suffix_with_locant": True, "prefix": "bromocarbonyl", "multi_suffix": "dicarbonyl dibromide"},
    {"key": "ring_acid_iodide", "seniority": 53, "suffix": "carbonyl iodide", "suffix_with_locant": True, "prefix": "iodocarbonyl", "multi_suffix": "dicarbonyl diiodide"},
    {"key": "amide", "seniority": 60, "suffix": "amide", "suffix_with_locant": False, "prefix": "carbamoyl", "multi_suffix": "diamide"},
    {"key": "ring_amide", "seniority": 60, "suffix": "carboxamide", "suffix_with_locant": True, "prefix": "carbamoyl", "multi_suffix": "dicarboxamide"},
    {"key": "thioamide", "seniority": 65, "suffix": "thioamide", "suffix_with_locant": False, "prefix": "carbamothioyl", "multi_suffix": "dithioamide"},
    {"key": "ring_thioamide", "seniority": 65, "suffix": "carbothioamide", "suffix_with_locant": True, "prefix": "carbamothioyl", "multi_suffix": "dicarbothioamide"},
    {"key": "nitrile", "seniority": 70, "suffix": "nitrile", "suffix_with_locant": False, "prefix": "cyano", "multi_suffix": "dinitrile"},
    {"key": "ring_nitrile", "seniority": 70, "suffix": "carbonitrile", "suffix_with_locant": True, "prefix": "cyano", "multi_suffix": "dicarbonitrile"},
    {"key": "aldehyde", "seniority": 80, "suffix": "al", "suffix_with_locant": False, "prefix": "oxo", "multi_suffix": "dial"},
    {"key": "ring_aldehyde", "seniority": 80, "suffix": "carbaldehyde", "suffix_with_locant": True, "prefix": "formyl", "multi_suffix": "dicarbaldehyde"},
    {"key": "ketone", "seniority": 90, "suffix": "one", "suffix_with_locant": True, "prefix": "oxo", "multi_suffix": "dione"},
    {"key": "hydrazone", "seniority": 95, "suffix": "one hydrazone", "suffix_with_locant": True, "prefix": "hydrazono", "multi_suffix": "dione dihydrazone"},
    {"key": "aldehyde_hydrazone", "seniority": 95, "suffix": "al hydrazone", "suffix_with_locant": False, "prefix": "hydrazono", "multi_suffix": "dial dihydrazone"},
    {"key": "ring_aldehyde_hydrazone", "seniority": 95, "suffix": "carbaldehyde hydrazone", "suffix_with_locant": True, "prefix": "hydrazonomethyl", "multi_suffix": "dicarbaldehyde dihydrazone"},
    {"key": "alcohol", "seniority": 100, "suffix": "ol", "suffix_with_locant": True, "prefix": "hydroxy", "multi_suffix": "diol"},
    {"key": "thiol", "seniority": 105, "suffix": "thiol", "suffix_with_locant": True, "prefix": "sulfanyl", "multi_suffix": "dithiol"},
    {"key": "amine", "seniority": 110, "suffix": "amine", "suffix_with_locant": True, "prefix": "amino", "multi_suffix": "diamine"},
    {"key": "aminium", "seniority": 109, "suffix": "aminium", "suffix_with_locant": True, "prefix": "ammonio", "multi_suffix": "diaminium"},
    {"key": "imine", "seniority": 112, "suffix": "imine", "suffix_with_locant": True, "prefix": "imino", "multi_suffix": "diimine"},
    {"key": "iminium", "seniority": 111, "suffix": "iminium", "suffix_with_locant": True, "prefix": "iminio", "multi_suffix": "diiminium"},
    {"key": "hydrazine", "seniority": 115, "suffix": "hydrazine", "suffix_with_locant": True, "prefix": "hydrazinyl", "multi_suffix": "dihydrazine"},
    {"key": "ether", "seniority": 200, "suffix": "ether", "suffix_with_locant": False, "prefix": "oxy", "multi_suffix": None},
)


PREFIX_FUNCTIONAL_GROUP_ROWS: tuple[dict, ...] = (
    {"key": "fluoro", "prefix": "fluoro", "needs_locant": True},
    {"key": "chloro", "prefix": "chloro", "needs_locant": True},
    {"key": "bromo", "prefix": "bromo", "needs_locant": True},
    {"key": "iodo", "prefix": "iodo", "needs_locant": True},
    {"key": "astato", "prefix": "astato", "needs_locant": True},
    {"key": "nitro", "prefix": "nitro", "needs_locant": True},
    {"key": "nitroso", "prefix": "nitroso", "needs_locant": True},
    {"key": "azido", "prefix": "azido", "needs_locant": True},
    {"key": "diazo", "prefix": "diazo", "needs_locant": True},
    {"key": "diazonio", "prefix": "diazonio", "needs_locant": True},
    {"key": "isocyano", "prefix": "isocyano", "needs_locant": True},
    {"key": "cyanato", "prefix": "cyanato", "needs_locant": True},
    {"key": "isocyanato", "prefix": "isocyanato", "needs_locant": True},
    {"key": "thiocyanato", "prefix": "thiocyanato", "needs_locant": True},
    {"key": "isothiocyanato", "prefix": "isothiocyanato", "needs_locant": True},
    {"key": "hydroperoxy", "prefix": "hydroperoxy", "needs_locant": True},
    {"key": "peroxy", "prefix": "peroxy", "needs_locant": True},
    {"key": "sulfanyl", "prefix": "sulfanyl", "needs_locant": True},
    {"key": "silyl", "prefix": "silyl", "needs_locant": True},
    {"key": "phosphanyl", "prefix": "phosphanyl", "needs_locant": True},
    {"key": "phosphoryl", "prefix": "phosphoryl", "needs_locant": True},
    {"key": "boryl", "prefix": "boryl", "needs_locant": True},
)


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

    for item in PRINCIPAL_FUNCTIONAL_GROUP_ROWS:
        key = item["key"]
        groups[key] = FunctionalGroupRule(
            key=key,
            role="principal",
            prefix=item["prefix"],
            suffix=item["suffix"],
            multi_suffix=item["multi_suffix"],
            seniority=item["seniority"],
            suffix_with_locant=item["suffix_with_locant"],
            needs_locant=True,
            families=_derived_functional_group_families(key),
        )
    for item in PREFIX_FUNCTIONAL_GROUP_ROWS:
        key = item["key"]
        groups[key] = FunctionalGroupRule(
            key=key,
            role="prefix",
            prefix=item["prefix"],
            needs_locant=item["needs_locant"],
            families=_derived_functional_group_families(key),
        )
    for key, item in mapping("functional_groups").items():
        families = tuple(item.get("families", _derived_functional_group_families(key)))
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
            families=families,
            component_flags=tuple(item.get("component_flags", [])),
            postprocess_tags=tuple(item.get("postprocess_tags", [])),
        )
    return FunctionalGroupRules(by_key=groups)


def _derived_functional_group_families(key: str) -> tuple[str, ...]:
    families = set()
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
        if key in values(section):
            families.add(family)
    if key in mapping("acid_halide_prefixes"):
        families.add("acid_halide")
    if key in mapping("direct_prefix_groups"):
        families.add("direct_prefix")
    if key in mapping("direct_group_prefixes"):
        families.add("direct_group")
    return tuple(sorted(families))


def _postprocess_rules() -> PostprocessRules:
    return PostprocessRules(
        literal_replacements=tuple(tuple(item) for item in values("postprocess_literal_replacements")),
        regex_replacements=tuple(
            RegexReplacement(pattern=item["pattern"], replacement=item["replacement"])
            for item in values("postprocess_regex_replacements")
        ),
        exact_replacements=mapping("postprocess_exact_replacements"),
        acyl_amido_terms=tuple(values("postprocess_acyl_amido_terms")),
        n_substituted_functional_suffixes=tuple(values("postprocess_n_substituted_functional_suffixes")),
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
            ambiguous_connection_substituent_stems=set(values("ambiguous_connection_substituent_stems")),
            connection_boundary_parent_stems=tuple(values("connection_boundary_parent_stems")),
        ),
        functional_groups=_functional_group_rules(),
        postprocess=_postprocess_rules(),
    )


RULES = registry()
