"""Rule identifiers for graph-classified chalcogen functional groups.

This module is the narrow boundary between structural descriptors and the
existing functional-group rule registry.  Chemistry code selects with typed
roles; complete nomenclature keys are table values, never assembled by editing
another group's name.
"""

from .chalcogen_roles import (
    Chalcogen,
    ChalcogenLigandRole,
    DerivativeKind,
    FunctionalFamily,
    FunctionalGroupDescriptor,
)
from .nomenclature import FunctionalGroupRule, MultiSuffixTemplate


_SIMPLE_KEYS: dict[tuple[FunctionalFamily, DerivativeKind, Chalcogen, bool], str] = {
    (FunctionalFamily.HYDROXY, DerivativeKind.ALCOHOL, Chalcogen.OXYGEN, False): "alcohol",
    (FunctionalFamily.HYDROXY, DerivativeKind.ALCOHOL, Chalcogen.SULFUR, False): "thiol",
    (FunctionalFamily.HYDROXY, DerivativeKind.ALCOHOL, Chalcogen.SELENIUM, False): "selenol",
    (FunctionalFamily.HYDROXY, DerivativeKind.ALCOHOL, Chalcogen.TELLURIUM, False): "tellurol",
    (FunctionalFamily.HYDROXY, DerivativeKind.ANION, Chalcogen.OXYGEN, False): "olate",
    (FunctionalFamily.HYDROXY, DerivativeKind.ANION, Chalcogen.SULFUR, False): "thiolate",
    (FunctionalFamily.HYDROXY, DerivativeKind.ANION, Chalcogen.SELENIUM, False): "selenolate",
    (FunctionalFamily.HYDROXY, DerivativeKind.ANION, Chalcogen.TELLURIUM, False): "tellurolate",
    (FunctionalFamily.CARBONYL, DerivativeKind.ALDEHYDE, Chalcogen.OXYGEN, False): "aldehyde",
    (FunctionalFamily.CARBONYL, DerivativeKind.ALDEHYDE, Chalcogen.SULFUR, False): "thioaldehyde",
    (FunctionalFamily.CARBONYL, DerivativeKind.ALDEHYDE, Chalcogen.SELENIUM, False): "selenoaldehyde",
    (FunctionalFamily.CARBONYL, DerivativeKind.ALDEHYDE, Chalcogen.TELLURIUM, False): "telluroaldehyde",
    (FunctionalFamily.CARBONYL, DerivativeKind.ALDEHYDE, Chalcogen.OXYGEN, True): "ring_aldehyde",
    (FunctionalFamily.CARBONYL, DerivativeKind.ALDEHYDE, Chalcogen.SULFUR, True): "ring_thioaldehyde",
    (FunctionalFamily.CARBONYL, DerivativeKind.ALDEHYDE, Chalcogen.SELENIUM, True): "ring_selenoaldehyde",
    (FunctionalFamily.CARBONYL, DerivativeKind.ALDEHYDE, Chalcogen.TELLURIUM, True): "ring_telluroaldehyde",
    (FunctionalFamily.CARBONYL, DerivativeKind.KETONE, Chalcogen.OXYGEN, False): "ketone",
    (FunctionalFamily.CARBONYL, DerivativeKind.KETONE, Chalcogen.SULFUR, False): "thioketone",
    (FunctionalFamily.CARBONYL, DerivativeKind.KETONE, Chalcogen.SELENIUM, False): "selenoketone",
    (FunctionalFamily.CARBONYL, DerivativeKind.KETONE, Chalcogen.TELLURIUM, False): "telluroketone",
    (FunctionalFamily.CHALCOGEN_ETHER, DerivativeKind.NEUTRAL_LINK, Chalcogen.OXYGEN, False): "ether",
    (FunctionalFamily.CHALCOGEN_ETHER, DerivativeKind.NEUTRAL_LINK, Chalcogen.SULFUR, False): "thioether",
    (FunctionalFamily.CHALCOGEN_ETHER, DerivativeKind.NEUTRAL_LINK, Chalcogen.SELENIUM, False): "selenoether",
    (FunctionalFamily.CHALCOGEN_ETHER, DerivativeKind.NEUTRAL_LINK, Chalcogen.TELLURIUM, False): "telluroether",
}


def simple_group_key(
    family: FunctionalFamily,
    derivative: DerivativeKind,
    element: Chalcogen,
    *,
    external: bool = False,
) -> str | None:
    """Resolve a validated simple-family descriptor to a registry key."""

    return _SIMPLE_KEYS.get((family, derivative, element, external))


_ELEMENT_RANK = {
    Chalcogen.OXYGEN: 0,
    Chalcogen.SULFUR: 1,
    Chalcogen.SELENIUM: 2,
    Chalcogen.TELLURIUM: 3,
}

_REPLACEMENT_INFIX = {
    Chalcogen.SULFUR: "thio",
    Chalcogen.SELENIUM: "seleno",
    Chalcogen.TELLURIUM: "telluro",
}

_STANDARD_ACYL_KEYS = {
    (DerivativeKind.ACID, Chalcogen.OXYGEN, Chalcogen.OXYGEN, False): "carboxylic_acid",
    (DerivativeKind.ACID, Chalcogen.OXYGEN, Chalcogen.OXYGEN, True): "ring_carboxylic_acid",
    (DerivativeKind.ANION, Chalcogen.OXYGEN, Chalcogen.OXYGEN, False): "carboxylate",
    (DerivativeKind.ANION, Chalcogen.OXYGEN, Chalcogen.OXYGEN, True): "ring_carboxylate",
    (DerivativeKind.ESTER, Chalcogen.OXYGEN, Chalcogen.OXYGEN, False): "ester",
    (DerivativeKind.ESTER, Chalcogen.OXYGEN, Chalcogen.OXYGEN, True): "ring_carboxylate",
    (DerivativeKind.AMIDE, Chalcogen.OXYGEN, None, False): "amide",
    (DerivativeKind.AMIDE, Chalcogen.OXYGEN, None, True): "ring_amide",
    (DerivativeKind.AMIDE, Chalcogen.SULFUR, None, False): "thioamide",
    (DerivativeKind.AMIDE, Chalcogen.SULFUR, None, True): "ring_thioamide",
    (DerivativeKind.HYDRAZIDE, Chalcogen.OXYGEN, None, False): "hydrazide",
    (DerivativeKind.HYDRAZIDE, Chalcogen.OXYGEN, None, True): "ring_hydrazide",
    (DerivativeKind.UREA, Chalcogen.OXYGEN, None, False): "urea",
    (DerivativeKind.UREA, Chalcogen.SULFUR, None, False): "thiourea",
    (DerivativeKind.UREA, Chalcogen.SELENIUM, None, False): "selenourea",
    (DerivativeKind.UREA, Chalcogen.TELLURIUM, None, False): "tellurourea",
}

_HALIDE_WORD = {"F": "fluoride", "Cl": "chloride", "Br": "bromide", "I": "iodide"}
_HALIDE_PREFIX = {"F": "fluoro", "Cl": "chloro", "Br": "bromo", "I": "iodo"}
_HALIDE_RANK = {symbol: rank for rank, symbol in enumerate(_HALIDE_WORD)}
_STANDARD_ACID_HALIDE_KEYS = {
    ("F", False): "acid_fluoride",
    ("Cl", False): "acid_chloride",
    ("Br", False): "acid_bromide",
    ("I", False): "acid_iodide",
    ("F", True): "ring_acid_fluoride",
    ("Cl", True): "ring_acid_chloride",
    ("Br", True): "ring_acid_bromide",
    ("I", True): "ring_acid_iodide",
}


def _replacement_infix(elements: tuple[Chalcogen, ...]) -> str:
    replacements = [element for element in elements if element is not Chalcogen.OXYGEN]
    if not replacements:
        return ""
    if len(replacements) == 2 and replacements[0] is replacements[1]:
        return f"di{_REPLACEMENT_INFIX[replacements[0]]}"
    return "".join(sorted((_REPLACEMENT_INFIX[element] for element in replacements)))


def _variant_key(
    derivative: DerivativeKind,
    double_element: Chalcogen,
    single_element: Chalcogen | None,
    external: bool,
    leaving_symbol: str | None,
) -> str:
    sites = double_element.value
    if single_element is not None:
        sites = f"{sites}_{single_element.value}"
    if leaving_symbol is not None:
        sites = f"{sites}_{leaving_symbol}"
    scope = "external" if external else "chain"
    return f"acyl_{derivative.value}_{sites}_{scope}"


def resolve_acyl_rule(
    descriptor: FunctionalGroupDescriptor,
    *,
    leaving_symbol: str | None = None,
) -> tuple[str, FunctionalGroupRule | None]:
    """Resolve an acyl descriptor without deriving chemistry from a group name."""

    double_ligand = descriptor.ligands_with_role(ChalcogenLigandRole.DOUBLE_BONDED)[0]
    single_ligands = tuple(ligand for ligand in descriptor.ligands if ligand is not double_ligand)
    single_element = single_ligands[0].element if single_ligands else None
    if descriptor.derivative is DerivativeKind.ACID_HALIDE and double_ligand.element is Chalcogen.OXYGEN:
        return _STANDARD_ACID_HALIDE_KEYS[(leaving_symbol, descriptor.is_external)], None
    standard = _STANDARD_ACYL_KEYS.get(
        (descriptor.derivative, double_ligand.element, single_element, descriptor.is_external)
    )
    if standard is not None:
        return standard, None

    key = _variant_key(
        descriptor.derivative,
        double_ligand.element,
        single_element,
        descriptor.is_external,
        leaving_symbol,
    )
    elements = (double_ligand.element,) if single_element is None else (double_ligand.element, single_element)
    infix = _replacement_infix(elements)
    carbo = "carbo" if descriptor.is_external else ""
    suffix_with_locant = descriptor.is_external
    multi_suffix = MultiSuffixTemplate((0,))
    families: tuple[str, ...]

    if descriptor.derivative is DerivativeKind.ACID:
        site = f" {single_element.value}-acid" if single_element is not double_ligand.element else " acid"
        suffix = f"{carbo}{infix}ic{site}"
        seniority = (7, *(_ELEMENT_RANK[element] for element in elements))
        prefix = f"{infix}carboxy"
        families = ("carboxy_prefix", "chain_external_carbonyl")
    elif descriptor.derivative is DerivativeKind.ANION:
        suffix = f"{carbo}{infix}ate"
        seniority = (4, *(_ELEMENT_RANK[element] for element in elements))
        prefix = f"{infix}carboxylato"
        families = ("carboxy_prefix", "chain_external_carbonyl")
    elif descriptor.derivative is DerivativeKind.ESTER:
        suffix = f"{carbo}{infix}ate"
        seniority = (9, *(_ELEMENT_RANK[element] for element in elements))
        prefix = f"{infix}oxycarbonyl"
        families = ("ester_like", "front_modifier", "chain_external_carbonyl")
        multi_suffix = None
    elif descriptor.derivative is DerivativeKind.AMIDE:
        suffix = f"{carbo}{infix}amide"
        seniority = (11, _ELEMENT_RANK[double_ligand.element])
        prefix = f"carbamo{infix}yl"
        families = ("amide_like", "chain_external_carbonyl")
    elif descriptor.derivative is DerivativeKind.HYDRAZIDE:
        suffix = f"{carbo}{infix}hydrazide"
        seniority = (12, _ELEMENT_RANK[double_ligand.element])
        prefix = f"hydrazinecarb{infix}oyl"
        families = ("amide_like", "hydrazide", "chain_external_carbonyl")
    elif descriptor.derivative is DerivativeKind.UREA:
        suffix = f"{infix}urea"
        seniority = (11, _ELEMENT_RANK[double_ligand.element])
        prefix = f"carbamo{infix}ylamino"
        families = ("amide_like", "urea")
    elif descriptor.derivative is DerivativeKind.ACID_HALIDE and leaving_symbol is not None:
        suffix = f"{carbo}{infix}yl {_HALIDE_WORD[leaving_symbol]}"
        seniority = (10, _ELEMENT_RANK[double_ligand.element], _HALIDE_RANK[leaving_symbol])
        prefix = f"{_HALIDE_PREFIX[leaving_symbol]}{infix}carbonyl"
        families = ("acid_halide", "chain_external_carbonyl")
    else:
        raise ValueError(f"Unsupported acyl derivative descriptor: {descriptor.derivative.value}")

    return key, FunctionalGroupRule(
        key=key,
        role="principal",
        prefix=prefix,
        suffix=suffix,
        multi_suffix=multi_suffix,
        suffix_multiplier_positions=(0,),
        seniority=seniority,
        suffix_with_locant=suffix_with_locant,
        needs_locant=True,
        families=families,
    )
