"""Resolve graph-classified chalcogen descriptors through the rule registry.

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
    NitrogenChalcogenideCitation,
)
from .nomenclature import (
    RULES,
    ChalcogenCitationContext,
    ChalcogenCitationRoute,
    FunctionalGroupCapability,
    FunctionalGroupRule,
    MultiSuffixTemplate,
    PrincipalCitationMode,
)


def chalcogen_citation_rule(
    context: ChalcogenCitationContext,
    elements: tuple[Chalcogen, ...],
    *,
    bridge_atom_count: int,
    site_elements: tuple[Chalcogen, ...] = (),
    bridge_elements: tuple[Chalcogen, ...] = (),
):
    """Return the first data rule matching typed structural citation axes."""

    values = tuple(element.value for element in elements)
    site_values = tuple(element.value for element in site_elements)
    bridge_values = tuple(element.value for element in bridge_elements)
    return next(
        (
            rule
            for rule in RULES.chalcogens.citation_routes
            if rule.context is context and rule.matches(values, bridge_atom_count, site_values, bridge_values)
        ),
        None,
    )


def uses_existing_chalcogen_citation(
    context: ChalcogenCitationContext,
    elements: tuple[Chalcogen, ...],
    *,
    bridge_atom_count: int,
) -> bool:
    """Whether this topology should reuse the established naming pipeline."""

    rule = chalcogen_citation_rule(context, elements, bridge_atom_count=bridge_atom_count)
    return rule is not None and rule.route is ChalcogenCitationRoute.EXISTING


def simple_group_key(
    family: FunctionalFamily,
    derivative: DerivativeKind,
    element: Chalcogen,
    *,
    external: bool = False,
) -> str | None:
    """Resolve a validated simple-family descriptor to a registry key."""

    return RULES.chalcogens.simple_group_keys.get((family.value, derivative.value, element.value, external))


def _element_data(element: Chalcogen) -> dict:
    return RULES.chalcogens.elements[element.value]


def _element_rank(element: Chalcogen) -> int:
    return int(_element_data(element)["rank"])


def chalcogenide_class_name(element: Chalcogen) -> str:
    """Return the validated functional-class name for an anionic chalcogen ligand."""

    return RULES.chalcogens.chalcogenide_class_names[element.value]


def nitrogen_chalcogenide_citation(element: Chalcogen) -> NitrogenChalcogenideCitation:
    """Return the data-selected citation route for an N+-E- group."""

    return NitrogenChalcogenideCitation(_element_data(element)["nitrogen_chalcogenide_citation"])


def chalcogenide_prefix(element: Chalcogen) -> str:
    """Return the additive prefix for an anionic chalcogen ligand."""

    return _element_data(element)["additive_prefix"]


def peroxide_class_name(elements: tuple[Chalcogen, Chalcogen]) -> str:
    """Return the P-63.3 functional-class term for an E-E linkage."""

    return RULES.chalcogens.peroxide_class_names[frozenset(element.value for element in elements)]


def _capabilities(policy: dict) -> frozenset[FunctionalGroupCapability]:
    return frozenset(FunctionalGroupCapability(value) for value in policy.get("capabilities", ()))


def _derivative_policy(section: str, derivative: DerivativeKind) -> dict:
    return RULES.chalcogens.policies[section][derivative.value]


def resolve_nitrile_chalcogenide_rule(
    descriptor: FunctionalGroupDescriptor,
) -> tuple[str, FunctionalGroupRule]:
    """Resolve R-C#N+-E- while retaining the terminal chalcogen identity."""

    ligand = descriptor.ligands_with_role(ChalcogenLigandRole.ANIONIC)[0]
    class_name = chalcogenide_class_name(ligand.element)
    policy = RULES.chalcogens.policies["nitrile_chalcogenide"]
    external = descriptor.is_external
    key = f"{'ring_' if external else ''}nitrile_{ligand.element.value}"
    carbon_text = "carbo" if external else ""
    return key, FunctionalGroupRule(
        key=key,
        role="principal",
        prefix=policy["prefix"].format(additive_prefix=chalcogenide_prefix(ligand.element)),
        suffix=policy["suffix"].format(carbo=carbon_text, class_name=class_name),
        multi_suffix=MultiSuffixTemplate((0,)),
        suffix_multiplier_positions=(0,),
        seniority=(int(policy["seniority"]), _element_rank(ligand.element)),
        suffix_with_locant=external,
        needs_locant=True,
        families=tuple(policy["families"]),
        capabilities=_capabilities(policy),
    )


def _replacement_infix(elements: tuple[Chalcogen, ...]) -> str:
    replacements = [element for element in elements if element is not Chalcogen.OXYGEN]
    if not replacements:
        return ""
    if len(replacements) == 2 and replacements[0] is replacements[1]:
        return f"di{_element_data(replacements[0])['replacement_infix']}"
    return "".join(sorted(_element_data(element)["replacement_infix"] for element in replacements))


def _variant_key(
    derivative: DerivativeKind,
    double_element: Chalcogen,
    single_element: Chalcogen | None,
    external: bool,
    qualifier: str | None,
) -> str:
    sites = double_element.value
    if single_element is not None:
        sites = f"{sites}_{single_element.value}"
    if qualifier is not None:
        sites = f"{sites}_{qualifier}"
    scope = "external" if external else "chain"
    return f"acyl_{derivative.value}_{sites}_{scope}"


def resolve_acyl_rule(descriptor: FunctionalGroupDescriptor) -> tuple[str, FunctionalGroupRule | None]:
    """Resolve an acyl descriptor without deriving chemistry from a group name."""

    double_ligand = descriptor.ligands_with_role(ChalcogenLigandRole.DOUBLE_BONDED)[0]
    single_ligands = tuple(ligand for ligand in descriptor.ligands if ligand is not double_ligand)
    single_element = single_ligands[0].element if single_ligands else None
    leaving_group = descriptor.leaving_group.kind if descriptor.leaving_group is not None else None
    leaving_data = RULES.chalcogens.acyl_leaving_groups.get(leaving_group.value) if leaving_group is not None else None
    acyl_leaving_derivatives = {DerivativeKind.ACID_HALIDE, DerivativeKind.ACYL_PSEUDOHALIDE}
    if descriptor.derivative in acyl_leaving_derivatives and double_ligand.element is Chalcogen.OXYGEN:
        standard_key_field = "external_key" if descriptor.is_external else "chain_key"
        if leaving_data is not None and standard_key_field in leaving_data:
            return leaving_data[standard_key_field], None
    standard = RULES.chalcogens.standard_acyl_keys.get(
        (
            descriptor.derivative.value,
            double_ligand.element.value,
            single_element.value if single_element else None,
            descriptor.is_external,
        )
    )
    if standard is not None:
        return standard, None

    key = _variant_key(
        descriptor.derivative,
        double_ligand.element,
        single_element,
        descriptor.is_external,
        leaving_group.value if leaving_group is not None else None,
    )
    elements = (double_ligand.element,) if single_element is None else (double_ligand.element, single_element)
    infix = _replacement_infix(elements)
    carbo = "carbo" if descriptor.is_external else ""
    suffix_with_locant = descriptor.is_external
    multi_suffix = MultiSuffixTemplate((0,))
    families: tuple[str, ...]

    try:
        policy = _derivative_policy("acyl", descriptor.derivative)
    except KeyError:
        raise ValueError(f"Unsupported acyl derivative descriptor: {descriptor.derivative.value}") from None
    if descriptor.derivative is DerivativeKind.ACID:
        site_template = "acid_site_element" if single_element is not double_ligand.element else "acid_site_same"
        site = RULES.chalcogens.templates[site_template].format(element=single_element.value)
    else:
        site = ""
    if descriptor.derivative in acyl_leaving_derivatives and leaving_data is None:
        raise ValueError("An acyl leaving-group descriptor requires typed leaving-group data")
    format_values = {
        "carbo": carbo,
        "infix": infix,
        "site": site,
        "halide_word": leaving_data["word"] if leaving_data else "",
        "halide_prefix": leaving_data["prefix"] if leaving_data else "",
    }
    suffix = policy["suffix"].format(**format_values)
    prefix = policy["prefix"].format(**format_values)
    rank = int(policy["seniority"])
    if policy.get("ranked_seniority"):
        seniority = (rank + _element_rank(double_ligand.element),)
    elif descriptor.derivative in acyl_leaving_derivatives:
        seniority = (rank, _element_rank(double_ligand.element), int(leaving_data["rank"]))
    else:
        seniority = (rank, *(_element_rank(element) for element in elements))
    families = tuple(policy["families"])
    if policy.get("multi_suffix") is False:
        multi_suffix = None

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
        capabilities=_capabilities(policy),
        citation_suffixes=(
            (
                PrincipalCitationMode.ANHYDRIDE_HALF,
                policy["citation_suffix"].format(**format_values),
            ),
        )
        if "citation_suffix" in policy
        else (),
    )


def resolve_peroxol_rule(descriptor: FunctionalGroupDescriptor) -> tuple[str, FunctionalGroupRule]:
    """Resolve the ordered R-E1-E2-H path using Blue Book Table 6.1."""

    first, terminal = descriptor.ligands
    policy = RULES.chalcogens.peroxol_rules[(first.element.value, terminal.element.value)]
    family_policy = RULES.chalcogens.policies["peroxol"]
    key = f"peroxol_{first.element.value}_{terminal.element.value}"
    return key, FunctionalGroupRule(
        key=key,
        role="principal",
        prefix=None,
        suffix=policy["suffix"],
        multi_suffix=MultiSuffixTemplate((0,)),
        suffix_multiplier_positions=(0,),
        seniority=(int(family_policy["seniority"]), int(policy["rank"])),
        suffix_with_locant=True,
        needs_locant=True,
        families=tuple(family_policy["families"]),
        capabilities=_capabilities(policy),
    )


def _peroxo_infix(first: Chalcogen, second: Chalcogen) -> str:
    if first is Chalcogen.OXYGEN and second is Chalcogen.OXYGEN:
        return RULES.chalcogens.templates["peroxo"]
    replacements = tuple(element for element in (first, second) if element is not Chalcogen.OXYGEN)
    if len(replacements) == 2 and replacements[0] is replacements[1]:
        return RULES.chalcogens.templates["doubly_replaced_peroxo"].format(
            replacement=_element_data(replacements[0])["replacement_infix"]
        )
    replacement = "".join(sorted(_element_data(element)["replacement_infix"] for element in replacements))
    return RULES.chalcogens.templates["replaced_peroxo"].format(replacement=replacement)


def resolve_peroxy_acyl_rule(descriptor: FunctionalGroupDescriptor) -> tuple[str, FunctionalGroupRule | None]:
    """Resolve C(=E1)-E2-E3-X while retaining the ordered E2-E3 path."""

    double_ligand, first_linker, terminal = descriptor.ligands
    standard_path = (
        double_ligand.element,
        first_linker.element,
        terminal.element,
    ) == (Chalcogen.OXYGEN, Chalcogen.OXYGEN, Chalcogen.OXYGEN)
    if standard_path:
        standard = RULES.chalcogens.standard_peroxy_acyl_keys.get((descriptor.derivative.value, descriptor.is_external))
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
        acid_suffix = RULES.chalcogens.templates["ordinary_peroxy_acid"].format(
            carbo=carbo, linker=linker_infix, acid_infix=acid_infix
        )
        ester_suffix = RULES.chalcogens.templates["ordinary_peroxy_ester"].format(
            carbo=carbo, linker=linker_infix, acid_infix=acid_infix
        )
    else:
        acid_suffix = RULES.chalcogens.templates["mixed_peroxy_acid"].format(
            carbo=carbo, linker=linker_infix, acid_infix=acid_infix
        )
        ester_suffix = RULES.chalcogens.templates["mixed_peroxy_ester"].format(
            carbo=carbo, linker=linker_infix, acid_infix=acid_infix
        )
    site_template = "acid_site_pair" if first_linker.element is not terminal.element else "acid_site_same"
    site = RULES.chalcogens.templates[site_template].format(
        first=first_linker.element.value, second=terminal.element.value
    )
    try:
        policy = _derivative_policy("peroxy_acyl", descriptor.derivative)
    except KeyError:
        raise ValueError(f"Unsupported peroxy acyl derivative: {descriptor.derivative.value}") from None
    if descriptor.derivative is DerivativeKind.ACID:
        suffix = f"{acid_suffix}{site}"
    else:
        suffix = ester_suffix
    seniority = (
        int(policy["seniority"]),
        _element_rank(double_ligand.element),
        _element_rank(first_linker.element),
        _element_rank(terminal.element),
    )
    return key, FunctionalGroupRule(
        key=key,
        role="principal",
        prefix=RULES.chalcogens.templates["peroxy_prefix"].format(acid_infix=acid_infix, linker=linker_infix),
        suffix=suffix,
        multi_suffix=None,
        suffix_multiplier_positions=(0,),
        seniority=seniority,
        suffix_with_locant=descriptor.is_external,
        needs_locant=True,
        families=tuple(policy["families"]),
        capabilities=_capabilities(policy),
    )


def anhydride_class_name(descriptor: FunctionalGroupDescriptor) -> str:
    """Return the functional-class term for an ordered anhydride bridge."""

    bridge_elements = tuple(ligand.element for ligand in descriptor.ligands[2:])
    if bridge_elements == (Chalcogen.OXYGEN,):
        return RULES.chalcogens.templates["anhydride"]
    if len(bridge_elements) == 1:
        return RULES.chalcogens.templates["single_replacement_anhydride"].format(
            replacement=_element_data(bridge_elements[0])["replacement_infix"]
        )
    if bridge_elements == (Chalcogen.OXYGEN, Chalcogen.OXYGEN):
        return RULES.chalcogens.templates["peroxyanhydride"]
    replacements = tuple(element for element in bridge_elements if element is not Chalcogen.OXYGEN)
    if len(replacements) == 1:
        return RULES.chalcogens.templates["single_replacement_peroxyanhydride"].format(
            replacement=_element_data(replacements[0])["replacement_infix"]
        )
    if replacements[0] is replacements[1]:
        return RULES.chalcogens.templates["double_replacement_peroxyanhydride"].format(
            replacement=_element_data(replacements[0])["replacement_infix"]
        )
    replacement = "".join(sorted(_element_data(element)["replacement_infix"] for element in replacements))
    return RULES.chalcogens.templates["mixed_replacement_peroxyanhydride"].format(replacement=replacement)


def resolve_anhydride_rule(descriptor: FunctionalGroupDescriptor) -> tuple[str, FunctionalGroupRule | None]:
    """Resolve one- and two-chalcogen bridges between two acyl centers."""

    bridge_elements = tuple(ligand.element for ligand in descriptor.ligands[2:])
    if (
        descriptor.ligands[0].element is Chalcogen.OXYGEN
        and descriptor.ligands[-1].element is Chalcogen.OXYGEN
        and bridge_elements == (Chalcogen.OXYGEN,)
    ):
        return RULES.chalcogens.policies["anhydride"]["standard_key"], None
    sites = "_".join(ligand.element.value for ligand in descriptor.ligands)
    key = f"anhydride_{sites}"
    return key, FunctionalGroupRule(
        key=key,
        role="principal",
        prefix=None,
        suffix=anhydride_class_name(descriptor),
        multi_suffix=None,
        suffix_multiplier_positions=(0,),
        seniority=(
            int(RULES.chalcogens.policies["anhydride"]["seniority"]),
            *(_element_rank(element) for element in bridge_elements),
        ),
        suffix_with_locant=False,
        needs_locant=True,
        families=tuple(RULES.chalcogens.policies["anhydride"]["families"]),
        capabilities=_capabilities(RULES.chalcogens.policies["anhydride"]),
    )


def resolve_imide_route_rule(descriptor: FunctionalGroupDescriptor) -> tuple[str, FunctionalGroupRule]:
    """Return the data-defined priority marker for an imide's member groups."""

    if descriptor.derivative is not DerivativeKind.IMIDE:
        raise ValueError(f"Expected an imide descriptor, got: {descriptor.derivative.value}")
    policy = RULES.chalcogens.policies["imide_route"]
    key = policy["key"]
    return key, FunctionalGroupRule(
        key=key,
        role="principal",
        seniority=int(policy["seniority"]),
        families=tuple(policy["families"]),
        capabilities=_capabilities(policy),
    )


def resolve_central_acid_rule(descriptor: FunctionalGroupDescriptor) -> tuple[str, FunctionalGroupRule | None]:
    """Resolve R-Q(=E)n-E-X for Q = S, Se, or Te."""

    double_ligands = descriptor.ligands_with_role(ChalcogenLigandRole.DOUBLE_BONDED)
    terminal_ligands = tuple(ligand for ligand in descriptor.ligands if ligand not in double_ligands)
    central_element = descriptor.central_element
    if central_element is None:
        raise ValueError("A central-acid descriptor requires its central element")
    leaving_group = descriptor.leaving_group.kind if descriptor.leaving_group is not None else None
    leaving_data = RULES.chalcogens.acyl_leaving_groups.get(leaving_group.value) if leaving_group is not None else None
    origin = RULES.chalcogens.central_acid_origins[(central_element.value, len(double_ligands))]
    origin_stem = origin["stem"]
    origin_rank = int(origin["rank"])
    acid_key = origin.get("acid_key")
    ester_key = origin.get("ester_key")
    amide_key = origin.get("amide_key")
    if len(terminal_ligands) == 2:
        first_linker, terminal = terminal_ligands
        double_infix = _replacement_infix(tuple(ligand.element for ligand in double_ligands))
        linker_infix = _peroxo_infix(first_linker.element, terminal.element)
        origin_prefix = (
            RULES.chalcogens.templates["central_origin_with_infix"].format(stem=origin_stem, infix=double_infix)
            if double_infix
            else origin_stem
        )
        ordinary_peroxo = first_linker.element is Chalcogen.OXYGEN and terminal.element is Chalcogen.OXYGEN
        if ordinary_peroxo:
            acid_suffix = RULES.chalcogens.templates["central_peroxy_acid"].format(origin=origin_prefix)
            ester_suffix = RULES.chalcogens.templates["central_peroxy_ester"].format(origin=origin_prefix)
        else:
            site = RULES.chalcogens.templates["acid_site_pair"].format(
                first=first_linker.element.value, second=terminal.element.value
            )
            acid_suffix = RULES.chalcogens.templates["central_mixed_peroxy_acid"].format(
                origin=origin_prefix, linker=linker_infix, site=site
            )
            ester_suffix = RULES.chalcogens.templates["central_mixed_peroxy_ester"].format(
                origin=origin_prefix, linker=linker_infix
            )
        sites = "_".join(ligand.element.value for ligand in descriptor.ligands)
        key = f"central_{central_element.value}_{descriptor.derivative.value}_{sites}"
        try:
            policy = _derivative_policy("central_peroxy", descriptor.derivative)
        except KeyError:
            raise ValueError(f"Unsupported central peroxy derivative: {descriptor.derivative.value}") from None
        if descriptor.derivative is DerivativeKind.ACID:
            suffix = acid_suffix
            seniority = (
                origin_rank,
                *(_element_rank(ligand.element) for ligand in descriptor.ligands),
            )
        elif descriptor.derivative is DerivativeKind.ANION:
            suffix = ester_suffix
            seniority = (int(policy["seniority"]), origin_rank)
        elif descriptor.derivative is DerivativeKind.ESTER:
            suffix = ester_suffix
            seniority = (int(policy["seniority"]), origin_rank)
        return key, FunctionalGroupRule(
            key=key,
            role="principal",
            prefix=RULES.chalcogens.templates["central_peroxy_prefix"].format(origin=origin_prefix),
            suffix=suffix,
            multi_suffix=None,
            suffix_multiplier_positions=(0,),
            seniority=seniority,
            suffix_with_locant=True,
            needs_locant=True,
            families=tuple(policy["families"]),
            capabilities=_capabilities(policy),
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
    suffix_stem = RULES.chalcogens.templates["central_suffix_stem"].format(
        stem=origin_stem, connector="o" if infix else "", infix=infix
    )
    policy = _derivative_policy("central", descriptor.derivative)
    families = tuple(policy["families"])
    if descriptor.derivative is DerivativeKind.ACID:
        site_template = "acid_site_element" if terminal is not None and replacements else "acid_site_same"
        site = RULES.chalcogens.templates[site_template].format(element=terminal.element.value if terminal else "")
        suffix = RULES.chalcogens.templates["central_acid_suffix"].format(stem=suffix_stem, site=site)
        seniority = (origin_rank, *(_element_rank(ligand.element) for ligand in descriptor.ligands))
        prefix = RULES.chalcogens.templates["central_acid_prefix"].format(stem=origin_stem, infix=infix)
    elif descriptor.derivative is DerivativeKind.ANION:
        suffix = RULES.chalcogens.templates["central_anion_suffix"].format(stem=suffix_stem)
        seniority = (
            int(policy["seniority"]),
            origin_rank,
            *(_element_rank(ligand.element) for ligand in descriptor.ligands),
        )
        prefix = RULES.chalcogens.templates["central_anion_prefix"].format(stem=origin_stem)
    elif descriptor.derivative is DerivativeKind.ESTER:
        suffix = RULES.chalcogens.templates["central_ester_suffix"].format(stem=suffix_stem)
        seniority = (
            int(policy["seniority"]),
            origin_rank,
            *(_element_rank(ligand.element) for ligand in descriptor.ligands),
        )
        prefix = RULES.chalcogens.templates["central_ester_prefix"].format(stem=origin_stem)
    elif descriptor.derivative is DerivativeKind.AMIDE:
        suffix = RULES.chalcogens.templates["central_amide_suffix"].format(stem=suffix_stem)
        seniority = (
            int(policy["seniority"]),
            origin_rank,
            *(_element_rank(ligand.element) for ligand in double_ligands),
        )
        prefix = RULES.chalcogens.templates["central_amide_prefix"].format(stem=origin_stem)
    elif descriptor.derivative is DerivativeKind.HYDRAZIDE:
        template = "central_hydrazide_suffix" if infix else "central_plain_hydrazide_suffix"
        suffix = RULES.chalcogens.templates[template].format(stem=suffix_stem if infix else origin_stem)
        seniority = (
            int(policy["seniority"]),
            origin_rank,
            *(_element_rank(ligand.element) for ligand in double_ligands),
        )
        prefix = RULES.chalcogens.templates["central_hydrazide_prefix"].format(stem=origin_stem)
    elif descriptor.derivative is DerivativeKind.ACID_HALIDE and leaving_data is not None:
        suffix = RULES.chalcogens.templates["central_acid_halide_suffix"].format(
            stem=suffix_stem, halide_word=leaving_data["word"]
        )
        seniority = (int(policy["seniority"]), origin_rank, int(leaving_data["rank"]))
        prefix = RULES.chalcogens.templates["central_acid_halide_prefix"].format(
            halide_prefix=leaving_data["prefix"], stem=origin_stem
        )
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
        capabilities=_capabilities(policy),
    )
