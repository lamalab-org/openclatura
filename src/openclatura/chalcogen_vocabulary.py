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

_CHALCOGENIDE_CLASS_NAMES = {
    Chalcogen.OXYGEN: "oxide",
    Chalcogen.SULFUR: "sulfide",
    Chalcogen.SELENIUM: "selenide",
    Chalcogen.TELLURIUM: "telluride",
}

_CHALCOGENIDE_PREFIXES = {
    Chalcogen.OXYGEN: "oxido",
    Chalcogen.SULFUR: "sulfido",
    Chalcogen.SELENIUM: "selenido",
    Chalcogen.TELLURIUM: "tellurido",
}

_PEROXIDE_CLASS_NAMES = {
    frozenset((Chalcogen.OXYGEN,)): "peroxide",
    frozenset((Chalcogen.SULFUR,)): "disulfide",
    frozenset((Chalcogen.SELENIUM,)): "diselenide",
    frozenset((Chalcogen.TELLURIUM,)): "ditelluride",
    frozenset((Chalcogen.OXYGEN, Chalcogen.SULFUR)): "thioperoxide",
    frozenset((Chalcogen.OXYGEN, Chalcogen.SELENIUM)): "selenoperoxide",
    frozenset((Chalcogen.OXYGEN, Chalcogen.TELLURIUM)): "telluroperoxide",
    frozenset((Chalcogen.SULFUR, Chalcogen.SELENIUM)): "selenothioperoxide",
    frozenset((Chalcogen.SULFUR, Chalcogen.TELLURIUM)): "tellurothioperoxide",
    frozenset((Chalcogen.SELENIUM, Chalcogen.TELLURIUM)): "selenotelluroperoxide",
}


def chalcogenide_class_name(element: Chalcogen) -> str:
    """Return the validated functional-class name for an anionic chalcogen ligand."""

    return _CHALCOGENIDE_CLASS_NAMES[element]


def chalcogenide_prefix(element: Chalcogen) -> str:
    """Return the additive prefix for an anionic chalcogen ligand."""

    return _CHALCOGENIDE_PREFIXES[element]


def peroxide_class_name(elements: tuple[Chalcogen, Chalcogen]) -> str:
    """Return the P-63.3 functional-class term for an E-E linkage."""

    return _PEROXIDE_CLASS_NAMES[frozenset(elements)]


def resolve_nitrile_chalcogenide_rule(
    descriptor: FunctionalGroupDescriptor,
) -> tuple[str, FunctionalGroupRule]:
    """Resolve R-C#N+-E- while retaining the terminal chalcogen identity."""

    ligand = descriptor.ligands_with_role(ChalcogenLigandRole.ANIONIC)[0]
    class_name = chalcogenide_class_name(ligand.element)
    external = descriptor.is_external
    key = f"{'ring_' if external else ''}nitrile_{ligand.element.value}"
    carbon_text = "carbo" if external else ""
    return key, FunctionalGroupRule(
        key=key,
        role="principal",
        prefix=f"cyano{chalcogenide_prefix(ligand.element)}",
        suffix=f"{carbon_text}nitrile {class_name}",
        multi_suffix=MultiSuffixTemplate((0,)),
        suffix_multiplier_positions=(0,),
        seniority=(5, _ELEMENT_RANK[ligand.element]),
        suffix_with_locant=external,
        needs_locant=True,
        families=("nitrile_chalcogenide",),
    )


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
    return "".join(sorted(_REPLACEMENT_INFIX[element] for element in replacements))


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
        seniority = (20, *(_ELEMENT_RANK[element] for element in elements))
        prefix = f"{infix}carboxy"
        families = ("carboxy_prefix", "chain_external_carbonyl")
    elif descriptor.derivative is DerivativeKind.ANION:
        suffix = f"{carbo}{infix}ate"
        seniority = (21, *(_ELEMENT_RANK[element] for element in elements))
        prefix = f"{infix}carboxylato"
        families = ("carboxy_prefix", "chain_external_carbonyl")
    elif descriptor.derivative is DerivativeKind.ESTER:
        suffix = f"{carbo}{infix}ate"
        seniority = (40, *(_ELEMENT_RANK[element] for element in elements))
        prefix = f"{infix}oxycarbonyl"
        families = ("ester_like", "front_modifier", "chain_external_carbonyl")
        multi_suffix = None
    elif descriptor.derivative is DerivativeKind.AMIDE:
        suffix = f"{carbo}{infix}amide"
        seniority = (60 + _ELEMENT_RANK[double_ligand.element],)
        prefix = f"carbamo{infix}yl"
        families = ("amide_like", "chain_external_carbonyl")
    elif descriptor.derivative is DerivativeKind.HYDRAZIDE:
        suffix = f"{carbo}{infix}hydrazide"
        seniority = (68, _ELEMENT_RANK[double_ligand.element])
        prefix = f"hydrazinecarb{infix}oyl"
        families = ("amide_like", "hydrazide", "chain_external_carbonyl")
    elif descriptor.derivative is DerivativeKind.UREA:
        suffix = f"{infix}urea"
        seniority = (60, _ELEMENT_RANK[double_ligand.element])
        prefix = f"carbamo{infix}ylamino"
        families = ("amide_like", "urea")
    elif descriptor.derivative is DerivativeKind.ACID_HALIDE and leaving_symbol is not None:
        suffix = f"{carbo}{infix}yl {_HALIDE_WORD[leaving_symbol]}"
        seniority = (50, _ELEMENT_RANK[double_ligand.element], _HALIDE_RANK[leaving_symbol])
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


_PEROXOL_SUFFIXES = {
    (Chalcogen.OXYGEN, Chalcogen.OXYGEN): "peroxol",
    (Chalcogen.SULFUR, Chalcogen.OXYGEN): "SO-thioperoxol",
    (Chalcogen.SELENIUM, Chalcogen.OXYGEN): "SeO-selenoperoxol",
    (Chalcogen.TELLURIUM, Chalcogen.OXYGEN): "TeO-telluroperoxol",
    (Chalcogen.OXYGEN, Chalcogen.SULFUR): "OS-thioperoxol",
    (Chalcogen.OXYGEN, Chalcogen.SELENIUM): "OSe-selenoperoxol",
    (Chalcogen.OXYGEN, Chalcogen.TELLURIUM): "OTe-telluroperoxol",
    (Chalcogen.SULFUR, Chalcogen.SULFUR): "dithioperoxol",
    (Chalcogen.SELENIUM, Chalcogen.SULFUR): "SeS-selenothioperoxol",
    (Chalcogen.TELLURIUM, Chalcogen.SULFUR): "TeS-tellurothioperoxol",
    (Chalcogen.SULFUR, Chalcogen.SELENIUM): "SSe-selenothioperoxol",
    (Chalcogen.SULFUR, Chalcogen.TELLURIUM): "STe-tellurothioperoxol",
    (Chalcogen.SELENIUM, Chalcogen.SELENIUM): "diselenoperoxol",
    (Chalcogen.TELLURIUM, Chalcogen.SELENIUM): "TeSe-selenotelluroperoxol",
    (Chalcogen.SELENIUM, Chalcogen.TELLURIUM): "SeTe-selenotelluroperoxol",
    (Chalcogen.TELLURIUM, Chalcogen.TELLURIUM): "ditelluroperoxol",
}

_PEROXOL_PRIORITY = {pair: rank for rank, pair in enumerate(_PEROXOL_SUFFIXES)}


def resolve_peroxol_rule(descriptor: FunctionalGroupDescriptor) -> tuple[str, FunctionalGroupRule]:
    """Resolve the ordered R-E1-E2-H path using Blue Book Table 6.1."""

    first, terminal = descriptor.ligands
    pair = (first.element, terminal.element)
    suffix = _PEROXOL_SUFFIXES[pair]
    key = f"peroxol_{first.element.value}_{terminal.element.value}"
    element_locanted = first.element is not terminal.element or terminal.element is Chalcogen.OXYGEN
    families = (
        ("peroxol", "element_locanted_suffix")
        if element_locanted
        and pair
        != (
            Chalcogen.OXYGEN,
            Chalcogen.OXYGEN,
        )
        else ("peroxol",)
    )
    return key, FunctionalGroupRule(
        key=key,
        role="principal",
        prefix=None,
        suffix=suffix,
        multi_suffix=MultiSuffixTemplate((0,)),
        suffix_multiplier_positions=(0,),
        seniority=(105, _PEROXOL_PRIORITY[pair]),
        suffix_with_locant=True,
        needs_locant=True,
        families=families,
    )


def _peroxo_infix(first: Chalcogen, second: Chalcogen) -> str:
    if first is Chalcogen.OXYGEN and second is Chalcogen.OXYGEN:
        return "peroxo"
    replacements = tuple(element for element in (first, second) if element is not Chalcogen.OXYGEN)
    if len(replacements) == 2 and replacements[0] is replacements[1]:
        return f"di{_REPLACEMENT_INFIX[replacements[0]]}peroxo"
    replacement = "".join(sorted(_REPLACEMENT_INFIX[element] for element in replacements))
    return f"{replacement}peroxo"


def resolve_peroxy_acyl_rule(descriptor: FunctionalGroupDescriptor) -> tuple[str, FunctionalGroupRule | None]:
    """Resolve C(=E1)-E2-E3-X while retaining the ordered E2-E3 path."""

    double_ligand, first_linker, terminal = descriptor.ligands
    standard_path = (
        double_ligand.element,
        first_linker.element,
        terminal.element,
    ) == (Chalcogen.OXYGEN, Chalcogen.OXYGEN, Chalcogen.OXYGEN)
    if standard_path:
        standard = {
            (DerivativeKind.ACID, False): "peroxy_acid",
            (DerivativeKind.ACID, True): "ring_peroxy_acid",
            (DerivativeKind.ESTER, False): "peroxy_ester",
            (DerivativeKind.ESTER, True): "ring_peroxy_ester",
        }.get((descriptor.derivative, descriptor.is_external))
        if standard is not None:
            return standard, None

    key = _variant_key(
        descriptor.derivative,
        double_ligand.element,
        first_linker.element,
        descriptor.is_external,
        terminal.element.value,
    )
    carbo = "carbo" if descriptor.is_external else ""
    acid_infix = _replacement_infix((double_ligand.element,))
    linker_infix = _peroxo_infix(first_linker.element, terminal.element)
    ordinary_peroxo = first_linker.element is Chalcogen.OXYGEN and terminal.element is Chalcogen.OXYGEN
    if ordinary_peroxo:
        acid_suffix = f"{carbo}{linker_infix}{acid_infix}ic"
        ester_suffix = f"{carbo}{linker_infix}{acid_infix}ate"
    else:
        acid_suffix = f"{carbo}{acid_infix}({linker_infix}ic)"
        ester_suffix = f"{carbo}{acid_infix}({linker_infix}ate)"
    site = (
        f" {first_linker.element.value}{terminal.element.value}-acid"
        if first_linker.element is not terminal.element
        else " acid"
    )
    if descriptor.derivative is DerivativeKind.ACID:
        suffix = f"{acid_suffix}{site}"
        seniority = (
            22,
            _ELEMENT_RANK[double_ligand.element],
            _ELEMENT_RANK[first_linker.element],
            _ELEMENT_RANK[terminal.element],
        )
        families = ("peroxy_acid", "chain_external_carbonyl")
    elif descriptor.derivative is DerivativeKind.ANION:
        suffix = ester_suffix
        seniority = (
            21,
            _ELEMENT_RANK[double_ligand.element],
            _ELEMENT_RANK[first_linker.element],
            _ELEMENT_RANK[terminal.element],
        )
        families = ("peroxy_ester", "chain_external_carbonyl")
    elif descriptor.derivative is DerivativeKind.ESTER:
        suffix = ester_suffix
        seniority = (
            45,
            _ELEMENT_RANK[double_ligand.element],
            _ELEMENT_RANK[first_linker.element],
            _ELEMENT_RANK[terminal.element],
        )
        families = ("ester_like", "peroxy_ester", "front_modifier", "chain_external_carbonyl")
    else:
        raise ValueError(f"Unsupported peroxy acyl derivative: {descriptor.derivative.value}")
    return key, FunctionalGroupRule(
        key=key,
        role="principal",
        prefix=f"{acid_infix}{linker_infix}carbonyl",
        suffix=suffix,
        multi_suffix=None,
        suffix_multiplier_positions=(0,),
        seniority=seniority,
        suffix_with_locant=descriptor.is_external,
        needs_locant=True,
        families=families,
    )


def anhydride_class_name(descriptor: FunctionalGroupDescriptor) -> str:
    """Return the functional-class term for an ordered anhydride bridge."""

    bridge_elements = tuple(ligand.element for ligand in descriptor.ligands[2:])
    if bridge_elements == (Chalcogen.OXYGEN,):
        return "anhydride"
    if len(bridge_elements) == 1:
        return f"{_REPLACEMENT_INFIX[bridge_elements[0]]}anhydride"
    if bridge_elements == (Chalcogen.OXYGEN, Chalcogen.OXYGEN):
        return "peroxyanhydride"
    replacements = tuple(element for element in bridge_elements if element is not Chalcogen.OXYGEN)
    if len(replacements) == 1:
        return f"{_REPLACEMENT_INFIX[replacements[0]]}peroxyanhydride"
    if replacements[0] is replacements[1]:
        return f"di{_REPLACEMENT_INFIX[replacements[0]]}peroxyanhydride"
    replacement = "".join(sorted(_REPLACEMENT_INFIX[element] for element in replacements))
    return f"{replacement}peroxyanhydride"


def resolve_anhydride_rule(descriptor: FunctionalGroupDescriptor) -> tuple[str, FunctionalGroupRule | None]:
    """Resolve one- and two-chalcogen bridges between two acyl centers."""

    bridge_elements = tuple(ligand.element for ligand in descriptor.ligands[2:])
    if (
        descriptor.ligands[0].element is Chalcogen.OXYGEN
        and descriptor.ligands[-1].element is Chalcogen.OXYGEN
        and bridge_elements == (Chalcogen.OXYGEN,)
    ):
        return "anhydride", None
    sites = "_".join(ligand.element.value for ligand in descriptor.ligands)
    key = f"anhydride_{sites}"
    return key, FunctionalGroupRule(
        key=key,
        role="principal",
        prefix=None,
        suffix=anhydride_class_name(descriptor),
        multi_suffix=None,
        suffix_multiplier_positions=(0,),
        seniority=(30, *(_ELEMENT_RANK[element] for element in bridge_elements)),
        suffix_with_locant=False,
        needs_locant=True,
        families=("anhydride",),
    )


_CENTRAL_ACID_ORIGINS = {
    (Chalcogen.SULFUR, 2): ("sulfon", 25, "sulfonic_acid", "sulfonate", "sulfonamide"),
    (Chalcogen.SULFUR, 1): ("sulfin", 27, None, None, None),
    (Chalcogen.SELENIUM, 2): ("selenon", 28, None, None, None),
    (Chalcogen.SELENIUM, 1): ("selenin", 29, None, None, None),
    (Chalcogen.TELLURIUM, 2): ("telluron", 29, None, None, None),
    (Chalcogen.TELLURIUM, 1): ("tellurin", 29, None, None, None),
}


def resolve_central_acid_rule(
    descriptor: FunctionalGroupDescriptor,
    *,
    leaving_symbol: str | None = None,
) -> tuple[str, FunctionalGroupRule | None]:
    """Resolve R-Q(=E)n-E-X for Q = S, Se, or Te."""

    double_ligands = descriptor.ligands_with_role(ChalcogenLigandRole.DOUBLE_BONDED)
    terminal_ligands = tuple(ligand for ligand in descriptor.ligands if ligand not in double_ligands)
    central_element = descriptor.central_element
    if central_element is None:
        raise ValueError("A central-acid descriptor requires its central element")
    origin_stem, origin_rank, acid_key, ester_key, amide_key = _CENTRAL_ACID_ORIGINS[
        (central_element, len(double_ligands))
    ]
    if len(terminal_ligands) == 2:
        first_linker, terminal = terminal_ligands
        double_infix = _replacement_infix(tuple(ligand.element for ligand in double_ligands))
        linker_infix = _peroxo_infix(first_linker.element, terminal.element)
        origin_prefix = f"{origin_stem}{'o' if double_infix else ''}{double_infix}"
        ordinary_peroxo = first_linker.element is Chalcogen.OXYGEN and terminal.element is Chalcogen.OXYGEN
        if ordinary_peroxo:
            acid_suffix = f"{origin_prefix}operoxoic acid"
            ester_suffix = f"{origin_prefix}operoxoate"
        else:
            site = f" {first_linker.element.value}{terminal.element.value}-acid"
            acid_suffix = f"{origin_prefix}o({linker_infix}ic){site}"
            ester_suffix = f"{origin_prefix}o({linker_infix}ate)"
        sites = "_".join(ligand.element.value for ligand in descriptor.ligands)
        key = f"central_{central_element.value}_{descriptor.derivative.value}_{sites}"
        if descriptor.derivative is DerivativeKind.ACID:
            suffix = acid_suffix
            seniority = (
                origin_rank,
                *(_ELEMENT_RANK[ligand.element] for ligand in descriptor.ligands),
            )
            families = ("central_acid", "peroxy_acid")
        elif descriptor.derivative is DerivativeKind.ANION:
            suffix = ester_suffix
            seniority = (26, origin_rank)
            families = ("central_acid", "peroxy_ester")
        elif descriptor.derivative is DerivativeKind.ESTER:
            suffix = ester_suffix
            seniority = (45, origin_rank)
            families = ("central_acid", "ester_like", "peroxy_ester", "front_modifier")
        else:
            raise ValueError(f"Unsupported central peroxy derivative: {descriptor.derivative.value}")
        return key, FunctionalGroupRule(
            key=key,
            role="principal",
            prefix=f"{origin_prefix}operoxyl",
            suffix=suffix,
            multi_suffix=None,
            suffix_multiplier_positions=(0,),
            seniority=seniority,
            suffix_with_locant=True,
            needs_locant=True,
            families=families,
        )
    terminal = terminal_ligands[0] if terminal_ligands else None
    all_oxygen = all(ligand.element is Chalcogen.OXYGEN for ligand in descriptor.ligands)
    standard_key = {
        DerivativeKind.ACID: acid_key,
        DerivativeKind.ANION: ester_key,
        DerivativeKind.ESTER: ester_key,
        DerivativeKind.AMIDE: amide_key,
    }.get(descriptor.derivative)
    if all_oxygen and standard_key is not None:
        return standard_key, None

    expected_oxygen_count = len(descriptor.ligands)
    replacements = tuple(ligand.element for ligand in descriptor.ligands if ligand.element is not Chalcogen.OXYGEN)
    infix = _replacement_infix(
        tuple(Chalcogen.OXYGEN for _ in range(expected_oxygen_count - len(replacements))) + replacements
    )
    key_sites = "_".join(ligand.element.value for ligand in descriptor.ligands)
    key = f"central_{central_element.value}_{descriptor.derivative.value}_{key_sites}"
    suffix_stem = f"{origin_stem}{'o' if infix else ''}{infix}"
    families = ("central_acid",)
    if descriptor.derivative is DerivativeKind.ACID:
        site = f" {terminal.element.value}-acid" if terminal is not None and replacements else " acid"
        suffix = f"{suffix_stem}ic{site}"
        seniority = (origin_rank, *(_ELEMENT_RANK[ligand.element] for ligand in descriptor.ligands))
        prefix = f"{origin_stem}o{infix}o"
    elif descriptor.derivative is DerivativeKind.ANION:
        suffix = f"{suffix_stem}ate"
        seniority = (26, origin_rank, *(_ELEMENT_RANK[ligand.element] for ligand in descriptor.ligands))
        prefix = f"{origin_stem}ato"
    elif descriptor.derivative is DerivativeKind.ESTER:
        suffix = f"{suffix_stem}ate"
        seniority = (40, origin_rank, *(_ELEMENT_RANK[ligand.element] for ligand in descriptor.ligands))
        prefix = f"{origin_stem}yl"
        families += ("ester_like", "front_modifier")
    elif descriptor.derivative is DerivativeKind.AMIDE:
        suffix = f"{suffix_stem}amide"
        seniority = (66, origin_rank, *(_ELEMENT_RANK[ligand.element] for ligand in double_ligands))
        prefix = f"{origin_stem}amoyl"
        families += ("amide_like",)
    elif descriptor.derivative is DerivativeKind.HYDRAZIDE:
        suffix = f"{suffix_stem}hydrazide" if infix else f"{origin_stem}ohydrazide"
        seniority = (68, origin_rank, *(_ELEMENT_RANK[ligand.element] for ligand in double_ligands))
        prefix = f"hydrazine{origin_stem}yl"
        families += ("amide_like", "hydrazide")
    elif descriptor.derivative is DerivativeKind.ACID_HALIDE and leaving_symbol is not None:
        suffix = f"{suffix_stem}yl {_HALIDE_WORD[leaving_symbol]}"
        seniority = (55, origin_rank, _HALIDE_RANK[leaving_symbol])
        prefix = f"{_HALIDE_PREFIX[leaving_symbol]}{origin_stem}yl"
        families += ("acid_halide",)
    else:
        raise ValueError(f"Unsupported central-acid derivative: {descriptor.derivative.value}")
    return key, FunctionalGroupRule(
        key=key,
        role="principal",
        prefix=prefix,
        suffix=suffix,
        multi_suffix=MultiSuffixTemplate((0,)),
        suffix_multiplier_positions=(0,),
        seniority=seniority,
        suffix_with_locant=True,
        needs_locant=True,
        families=families,
    )
