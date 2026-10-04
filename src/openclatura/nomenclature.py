"""Central data registry for nomenclature lookup tables.
Prefer ``RULES.<group>.<field>`` over importing individual module constants."""

from dataclasses import dataclass
from functools import lru_cache

from .naming_data import grouped_namer_rules


@dataclass(frozen=True)
class RetainedChainParentRule:
    name: str
    substituted_name: str | None = None
    allows_carbon_substitution: bool = True


@dataclass(frozen=True)
class RetainedNameRules:
    ring_elements: set[str]
    substituent_stems: dict[str, tuple[str, str]]
    equivalent_substituent_attachment_parents: frozenset[str]
    functional_parents: dict[tuple[str, str], str]
    positive_nitrogen_parent_names: dict[tuple[str, str], str]
    chain_substituent_stems: dict[tuple[int, str], tuple[str, str]]
    chain_parents: dict[tuple[int, str, int], RetainedChainParentRule]
    aldehyde_chain_words: dict[int, str]
    substituted_chain_parents: dict[tuple[int, str, int, str, str], str]
    chain_branch_endings: dict[tuple[int, str, int, str, str], str]
    acyl_branch_endings: dict[tuple[str, str], str]
    substituents: dict[tuple[int, str, str, str, str], str]
    monocycle_specs: tuple[dict, ...]
    fused_polycycle_specs: tuple[dict, ...]


@dataclass(frozen=True)
class HeteroatomRules:
    alkyl_oxy_prefixes: dict[str, str]
    simple_sulfanyl_prefixes: set[str]
    simple_selanyl_prefixes: set[str]
    retained_sulfonyl_groups: dict[tuple[str, str], str]
    retained_sulfonyl_group_acids: dict[str, str]
    sulfonyl_ligand_contractions: tuple[tuple[str, str, str], ...]
    halogen_prefixes: dict[str, str]
    halogen_lambda_suffixes: dict[str, str]


@dataclass(frozen=True)
class RingRules:
    descriptor_templates: dict[str, str]
    polycycle_prefixes: dict[int, str]


@dataclass(frozen=True)
class PrefixRules:
    peroxy_ester_groups: set[str]
    amide_like_groups: set[str]


@dataclass(frozen=True)
class ComponentRules:
    salt_metal_names: set[str]
    mononuclear_parent_hydrides: dict[str, str]
    retained_mononuclear_hydride_names: dict[str, str]
    retained_homonuclear_chain_names: dict[str, str]
    retained_component_graph_names: dict[tuple[tuple[tuple[str, int], ...], tuple[tuple[str, str, int], ...]], str]
    replacement_parent_oxoacid_specs: tuple[dict, ...]


@dataclass(frozen=True)
class IonRules:
    single_atom_cations: set[str]
    single_atom_anions: dict[str, str]
    mononuclear_hydride_ions: dict[tuple[str, int, int], str]


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
    replacement_charge_states: dict[tuple[str, int, int], str]
    heteroatom_charge_prefixes: dict[str, str]
    heteroatom_prefix_states: dict[str, tuple[str, int, int]]
    anion_suffix_placements: tuple[AnionSuffixPlacementRule, ...]


@dataclass(frozen=True)
class AssemblyRules:
    replacement_prefix_order: dict[str, int]
    unsaturation_order: dict[str, int]
    acid_halide_suffix_keys: set[str]
    substituent_sort_prefix_pattern: str
    substituent_attachment_suffixes: dict[str, str]
    ambiguous_connection_substituent_stems: set[str]
    suffix_nitrogen_markers: tuple[str, ...]
    isotope_prefixes: dict[int, str]
    compound_nitrogen_prefix_units: tuple[str, ...]
    heteroatom_substituent_suffixes: tuple[str, ...]
    heteroatom_ligand_prefixes: tuple[str, ...]


@dataclass(frozen=True)
class LiteralReplacement:
    pattern: str
    replacement: str
    category: str = "migration"
    reason: str = ""


@dataclass(frozen=True)
class PostprocessRules:
    literal_replacements: tuple[LiteralReplacement, ...]


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
    positive_nitrogen_suffix: str | None = None
    multi_suffix: MultiSuffixTemplate | None = None
    suffix_multiplier_positions: tuple[int, ...] = (0,)
    seniority: int | None = None
    suffix_with_locant: bool = False
    needs_locant: bool = True
    families: tuple[str, ...] = ()
    uses_front_modifier: bool = False
    is_peroxy_acid: bool = False
    is_peroxy_ester: bool = False
    suffix_carbon_is_exocyclic: bool = False


@dataclass(frozen=True)
class FunctionalGroupRules:
    by_key: dict[str, FunctionalGroupRule]

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
    functional_group_data = grouped_namer_rules()["functional_groups"]

    for key, item in functional_group_data.mapping("functional_groups").items():
        families = tuple(item.get("families", _derived_functional_group_families(key)))
        suffix = item.get("suffix")
        multi_suffix = item.get("multi_suffix")
        groups[key] = FunctionalGroupRule(
            key=key,
            role=item["role"],
            prefix=item.get("prefix"),
            suffix=suffix,
            positive_nitrogen_suffix=item.get("positive_nitrogen_suffix"),
            multi_suffix=_multi_suffix_template(suffix, multi_suffix),
            suffix_multiplier_positions=_suffix_multiplier_positions(suffix, multi_suffix),
            seniority=item.get("seniority"),
            suffix_with_locant=bool(item.get("suffix_with_locant", False)),
            needs_locant=bool(item.get("needs_locant", True)),
            families=families,
            uses_front_modifier=key in functional_group_data.values("front_modifier_principal_groups"),
            is_peroxy_acid=key in functional_group_data.values("peroxy_acid_prefix_groups"),
            is_peroxy_ester=key in functional_group_data.values("peroxy_ester_groups"),
            suffix_carbon_is_exocyclic=bool(item.get("suffix_carbon_is_exocyclic", False)),
        )
    return FunctionalGroupRules(by_key=groups)


def _multi_suffix_template(suffix: str | None, multi_suffix) -> MultiSuffixTemplate | None:
    if multi_suffix is None:
        return None
    return MultiSuffixTemplate(multiplier_positions=_suffix_multiplier_positions(suffix, multi_suffix))


def _suffix_multiplier_positions(suffix: str | None, multi_suffix) -> tuple[int, ...]:
    """Return which words in a suffix phrase take multiplicative prefixes.  Built-in rows store
    ``multi_suffix`` as a template object; string support remains for older override data."""

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
    )


def _charge_rules() -> ChargeRules:
    group = grouped_namer_rules()["charges"]
    heteroatom_charge_prefixes = group.mapping("heteroatom_charge_prefixes")
    bond_orders = {"single": 1, "double": 2, "triple": 3}
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
        replacement_charge_states={
            (row["symbol"], row["charge"], row["valence"]): row["prefix"]
            for row in group.values("replacement_charge_states")
        },
        heteroatom_charge_prefixes=heteroatom_charge_prefixes,
        heteroatom_prefix_states={
            prefix: (
                state.split(":")[0],
                1 if state.split(":")[1] == "+" else -1,
                bond_orders[state.split(":")[2]],
            )
            for state, prefix in heteroatom_charge_prefixes.items()
        },
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
    retained_substituent_stems = retained.mapping("retained_substituent_stems")
    return NomenclatureRegistry(
        retained=RetainedNameRules(
            ring_elements=set(retained.values("retained_ring_elements")),
            substituent_stems={
                name: (item["stem"], item["terminal"]) for name, item in retained_substituent_stems.items()
            },
            equivalent_substituent_attachment_parents=frozenset(
                name
                for name, item in retained_substituent_stems.items()
                if item.get("equivalent_attachment_locants", False)
            ),
            functional_parents={
                (row["parent"], row["group"]): row["name"] for row in retained.values("retained_functional_parents")
            },
            positive_nitrogen_parent_names={
                (row["name"], row["group"]): row["positive_nitrogen_name"]
                for section in ("retained_functional_parents", "retained_chain_parents")
                for row in retained.values(section)
                if row.get("positive_nitrogen_name")
            },
            chain_substituent_stems={
                (int(row["length"]), row["attachment"]): (row["stem"], row["ending"])
                for row in retained.values("retained_chain_substituent_stems")
            },
            chain_parents={
                (int(row["length"]), row["group"], int(row["count"])): RetainedChainParentRule(
                    name=row["name"],
                    substituted_name=row.get("substituted_name"),
                    allows_carbon_substitution=bool(row.get("allows_carbon_substitution", True)),
                )
                for row in retained.values("retained_chain_parents")
            },
            aldehyde_chain_words={
                int(length): name for length, name in retained.mapping("retained_aldehyde_chain_words").items()
            },
            substituted_chain_parents={
                (
                    int(row["length"]),
                    row["group"],
                    int(row["count"]),
                    row["branch"],
                    str(row["locant"]),
                ): row["name"]
                for row in retained.values("retained_substituted_chain_parents")
            },
            chain_branch_endings={
                (
                    int(row["length"]),
                    row["group"],
                    int(row["count"]),
                    row["ending"],
                    str(row["locant"]),
                ): row["name"]
                for row in retained.values("retained_chain_branch_endings")
            },
            acyl_branch_endings={
                (row["ending"], row["attachment"]): row["name"]
                for row in retained.values("retained_acyl_branch_endings")
            },
            substituents={
                (
                    int(row["length"]),
                    str(row["attachment_locant"]),
                    row["branch"],
                    str(row["branch_locant"]),
                    row["attachment"],
                ): row["name"]
                for row in retained.values("retained_substituents")
            },
            monocycle_specs=tuple(retained.values("retained_monocycle_specs")),
            fused_polycycle_specs=tuple(retained.values("retained_fused_polycycle_specs")),
        ),
        heteroatoms=HeteroatomRules(
            alkyl_oxy_prefixes=substituent_vocabulary.mapping("alkyl_oxy_prefixes"),
            simple_sulfanyl_prefixes=set(substituent_vocabulary.values("simple_sulfanyl_prefixes")),
            simple_selanyl_prefixes=set(substituent_vocabulary.values("simple_selanyl_prefixes")),
            retained_sulfonyl_groups={
                (row["branch"], row["suffix"]): row["name"]
                for row in substituent_vocabulary.values("retained_sulfonyl_group_names")
            },
            retained_sulfonyl_group_acids={
                row["acid"]: row["name"]
                for row in substituent_vocabulary.values("retained_sulfonyl_group_names")
                if row.get("acid")
            },
            sulfonyl_ligand_contractions=tuple(
                (row["ligand_ending"], row["suffix"], row["name"])
                for row in substituent_vocabulary.values("retained_sulfonyl_ligand_contractions")
            ),
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
            peroxy_ester_groups=set(functional_group_rules.values("peroxy_ester_groups")),
            amide_like_groups=set(functional_group_rules.values("amide_like_prefix_groups")),
        ),
        components=ComponentRules(
            salt_metal_names=set(simple_components.values("salt_metal_names")),
            mononuclear_parent_hydrides=simple_components.mapping("mononuclear_parent_hydrides"),
            retained_mononuclear_hydride_names=simple_components.mapping("retained_mononuclear_hydride_names"),
            retained_homonuclear_chain_names=simple_components.mapping("retained_homonuclear_chain_names"),
            retained_component_graph_names={
                (
                    tuple(
                        sorted(
                            (str(symbol), int(charge))
                            for symbol, charge, count in row["atoms"]
                            for _ in range(int(count))
                        )
                    ),
                    tuple(
                        sorted(
                            (min(str(left), str(right)), max(str(left), str(right)), int(order))
                            for left, right, order, count in row["bonds"]
                            for _ in range(int(count))
                        )
                    ),
                ): row["name"]
                for row in simple_components.values("retained_component_graph_names")
            },
            replacement_parent_oxoacid_specs=tuple(simple_components.values("replacement_parent_oxoacid_specs")),
        ),
        ions=IonRules(
            single_atom_cations=set(simple_components.values("single_atom_cations")),
            single_atom_anions=simple_components.mapping("single_atom_anions"),
            mononuclear_hydride_ions={
                (row["element"], int(row["charge"]), int(row["hydrogens"])): row["name"]
                for row in simple_components.values("mononuclear_hydride_ions")
            },
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
            substituent_attachment_suffixes=assembly_grammar.mapping("substituent_attachment_suffixes"),
            ambiguous_connection_substituent_stems=set(
                assembly_grammar.values("ambiguous_connection_substituent_stems")
            ),
            suffix_nitrogen_markers=tuple(assembly_grammar.values("suffix_nitrogen_markers")),
            isotope_prefixes={
                int(mass): prefix for mass, prefix in assembly_grammar.mapping("isotope_prefixes").items()
            },
            compound_nitrogen_prefix_units=tuple(assembly_grammar.values("compound_nitrogen_prefix_units")),
            heteroatom_substituent_suffixes=tuple(assembly_grammar.values("heteroatom_substituent_suffixes")),
            heteroatom_ligand_prefixes=tuple(assembly_grammar.values("heteroatom_ligand_prefixes")),
        ),
        functional_groups=_functional_group_rules(),
        postprocess=_postprocess_rules(),
    )


RULES = registry()
