"""Charge-aware parent assembly helpers."""

from dataclasses import dataclass, replace

from .assembly_parts import AssemblyParts, ParentChargeItem, SubstituentItem
from .assembly_utils import parse_locant
from .name_operations import ParentSuffixOperation
from .nomenclature import RULES
from .suffix_stack import suffix_operation_spelling


@dataclass(frozen=True)
class ParentChargeOperation:
    locants: tuple[str, ...]
    suffix: str
    reason: str


def positive_parent_n_charges(parts: AssemblyParts) -> list[ParentChargeItem]:
    return [charge for charge in parts.parent_charges if charge.symbol == "N" and charge.charge > 0]


_IUM_SUFFIX_ELEMENTS = frozenset({"N", "O", "S", "Se", "Te", "P", "As", "Sb", "Bi"})


def positive_parent_ium_charges(parts: AssemblyParts) -> list[ParentChargeItem]:
    """Return parent atoms whose positive charge is represented by an ium suffix."""

    return [charge for charge in parts.parent_charges if charge.symbol in _IUM_SUFFIX_ELEMENTS and charge.charge > 0]


def has_ionic_retained_parent(parts: AssemblyParts) -> bool:
    return bool(parts.retained_name in RULES.charges.retained_ionic_n_parents and positive_parent_n_charges(parts))


def has_retained_like_parent(parts: AssemblyParts) -> bool:
    return bool(
        parts.retained_name
        or inferred_ionic_retained_parent(parts)
        or (parts.parent_hydride is not None and parts.parent_hydride.is_fusion_parent)
    )


def inferred_ionic_retained_parent(parts: AssemblyParts) -> str | None:
    if parts.retained_name or not parts.is_ring or parts.is_bicycle or parts.is_spiro or parts.is_polycycle:
        return None
    if parts.unsaturations or len(positive_parent_n_charges(parts)) != 1:
        return None
    aza_locs = [str(loc) for item in parts.a_prefixes if item.name == "aza" for loc in item.locants]
    if len(aza_locs) != 1:
        return None
    return RULES.charges.saturated_n_ring_ionic_parents.get(parts.parent_length)


def single_charged_replacement_locants(parts: AssemblyParts) -> set[str]:
    if parts.retained_name or inferred_ionic_retained_parent(parts):
        return set()
    positive_n_locs = {charge.locant for charge in positive_parent_n_charges(parts)}
    if not positive_n_locs:
        return set()
    aza_locs = [str(loc) for item in parts.a_prefixes if item.name == "aza" for loc in item.locants]
    if len(aza_locs) == 1 and aza_locs[0] in positive_n_locs:
        return {aza_locs[0]}
    return set()


def parent_charge_suffix_locs(parts: AssemblyParts) -> list[str]:
    if has_ionic_retained_parent(parts) or inferred_ionic_retained_parent(parts):
        return []
    represented_as_azonia = single_charged_replacement_locants(parts)
    return sorted(
        [charge.locant for charge in positive_parent_ium_charges(parts) if charge.locant not in represented_as_azonia],
        key=parse_locant,
    )


def parent_charge_name_operations(parts: AssemblyParts) -> list[ParentSuffixOperation]:
    fusion_operations = fusion_parent_charge_name_operations(parts)
    if fusion_operations is not None:
        return fusion_operations
    represented_as_azonia = single_charged_replacement_locants(parts)
    retained_positive_charge = has_ionic_retained_parent(parts) or bool(inferred_ionic_retained_parent(parts))
    retained_parent_positive = bool(parts.retained_name and positive_parent_ium_charges(parts))
    positive_carbon = any(charge.symbol == "C" and charge.charge > 0 for charge in parts.parent_charges)
    positive_charge = any(charge.charge > 0 for charge in parts.parent_charges)
    positive_locants = {str(charge.locant) for charge in parts.parent_charges if charge.charge > 0}
    negative_locants = {str(charge.locant) for charge in parts.parent_charges if charge.charge < 0}
    multiple_negative_sites = len(negative_locants) > 1
    retained_charge_pair = bool(
        parts.retained_name
        and any(
            set(edge) & positive_locants and set(edge) & negative_locants
            for edge in parts.parent_bond_orders_by_locants
        )
    )
    rendered: list[ParentSuffixOperation] = []
    grouped_sites: dict[tuple[int, str], list[ParentChargeItem]] = {}
    for charge in parts.parent_charges:
        if str(charge.locant) in represented_as_azonia:
            continue
        if charge.charge > 0 and retained_positive_charge:
            continue
        if charge.charge < 0:
            early_positive_stack = positive_charge and (not parts.retained_name or retained_charge_pair)
            if (
                parts.principal_group is not None
                and parts.principal_group.key == "ring_aldehyde"
                and len(parts.principal_group.locants) > 1
            ):
                early_positive_stack = False
            if (
                not early_positive_stack
                and not (
                    parts.is_substituent
                    and not parts.retained_name
                    and (parts.parent_length > 1 or not parts.is_double_attach)
                )
                and not (multiple_negative_sites and not parts.retained_name)
            ):
                continue
            rule_key = "*:-"
        elif charge.symbol == "C":
            rule_key = "C:+"
        elif charge.symbol in _IUM_SUFFIX_ELEMENTS:
            rule_key = f"{charge.symbol}:+"
            if rule_key not in RULES.charges.parent_charge_suffixes:
                rule_key = "N:+"
        else:
            continue
        grouped_sites.setdefault((charge.charge, rule_key), []).append(charge)

    sign_order = (
        (1, -1)
        if positive_carbon or retained_positive_charge or (retained_parent_positive and retained_charge_pair)
        else (-1, 1)
        if positive_charge or parts.is_substituent
        else (-1,)
        if any(group_sign < 0 for group_sign, _rule_key in grouped_sites)
        else (1,)
    )
    for sign in sign_order:
        for (group_sign, rule_key), sites in grouped_sites.items():
            if group_sign != sign:
                continue
            rule = RULES.charges.parent_charge_suffixes[rule_key]
            symbols = tuple(sorted({charge.symbol for charge in sites}))
            reasons = tuple(
                dict.fromkeys(
                    RULES.charges.parent_charge_suffixes.get(f"{symbol}:{'+' if sign > 0 else '-'}", rule).reason
                    for symbol in symbols
                )
            )
            rendered.append(
                ParentSuffixOperation(
                    key="parent-anion-suffix" if sign < 0 else "parent-cation-suffix",
                    locants=tuple(sorted((str(charge.locant) for charge in sites), key=parse_locant)),
                    suffix=rule.suffix,
                    reason=" ".join(reasons),
                    charge=sign,
                    atom_symbols=symbols,
                )
            )
    return rendered


def fusion_parent_charge_name_operations(parts: AssemblyParts) -> list[ParentSuffixOperation] | None:
    """Render audited fusion charge deltas before characteristic/branch suffixes."""

    parent = parts.parent_hydride
    if (
        parent is None
        or not parent.is_fusion_parent
        or parent.fusion_plan is None
        or not parent.fusion_plan.audit.confirmed
    ):
        return None
    operations = parent.fusion_plan.charge_operations
    if not any(operation.observed_charge == -1 for operation in operations):
        return None
    # Assembly can select a symmetry-equivalent numbering after the charge
    # audit. Rebind by graph identity only within the proved complete maps.
    selected = {atom_id: locant for locant, atom_id in parts.parent_atom_ids_by_locant.items()}
    approved = next(
        (
            dict(mapping)
            for mapping in parent.fusion_plan.numbering.input_locant_maps
            if {atom_id: str(locant) for atom_id, locant in mapping} == selected
        ),
        None,
    )
    if approved is None:
        raise ValueError("fusion parent charge spelling requires an audited complete numbering")
    operations = tuple(replace(operation, locant=approved[operation.atom_id]) for operation in operations)
    expected = {
        (operation.atom_id, str(operation.locant), operation.symbol, operation.observed_charge)
        for operation in operations
    }
    observed = {(charge.atom_id, str(charge.locant), charge.symbol, charge.charge) for charge in parts.parent_charges}
    if expected != observed:
        raise ValueError("fusion parent charge spelling does not match its graph-bound operations")
    grouped = {}
    for operation in operations:
        sign = operation.observed_charge
        grouped.setdefault(sign, []).append(operation)
    # Nitrogen zwitterions use ide-ium; carbon anions follow the parent ium.
    nitrogen_anion = any(operation.symbol == "N" and operation.observed_charge == -1 for operation in operations)
    rendered = []
    for sign in (-1, 1) if nitrogen_anion else (1, -1):
        sites = grouped.get(sign, ())
        if not sites:
            continue
        rule = RULES.charges.parent_charge_suffixes["N:+" if sign > 0 else "*:-"]
        rendered.append(
            ParentSuffixOperation(
                key="fusion-parent-charge-suffix",
                locants=tuple(sorted((str(site.locant) for site in sites), key=parse_locant)),
                suffix=rule.suffix,
                reason=rule.reason,
                charge=sign,
                atom_symbols=tuple(sorted({site.symbol for site in sites})),
            )
        )
    return rendered


def prepare_fusion_charge_assembly(parts: AssemblyParts) -> bool:
    """Resolve early charged-parent operations before suffixes or branches render."""

    operations = fusion_parent_charge_name_operations(parts)
    if operations is None:
        operations = parent_charge_name_operations(parts)
    if not any(operation.charge < 0 for operation in operations):
        return False
    negative_c = {charge.atom_id for charge in parts.parent_charges if charge.symbol == "C" and charge.charge == -1}
    consumed = [
        operation
        for operation in parts.hydro_operations
        if operation.key == "added_hydrogen" and operation.atom_ids and set(operation.atom_ids) <= negative_c
    ]
    consumed_locants = {locant for operation in consumed for locant in operation.locants}
    parts.hydro_operations = [operation for operation in parts.hydro_operations if operation not in consumed]
    parts.indicated_hydrogens = [locant for locant in parts.indicated_hydrogens if locant not in consumed_locants]
    group = parts.principal_group
    if group is not None and group.key == "ketone":
        parts.substituents.append(
            SubstituentItem(
                name=RULES.functional_groups.get(group.key).prefix,
                locants=list(group.locants),
                atom_ids=set(group.atom_ids),
                bond_ids=set(group.bond_ids),
                charge_atom_ids=set(group.charge_atom_ids),
            )
        )
        parts.principal_group = None
        return True
    return bool(consumed)


def append_charge_suffixes_to_terminal(parts: AssemblyParts, terminal_e: str) -> str:
    operations = parent_charge_name_operations(parts)
    if not operations:
        return terminal_e
    if (
        parts.is_substituent
        and parts.parent_length == 1
        and terminal_e in {"yl", "-1-yl"}
        and len(operations) == 1
        and operations[0].locants == ("1",)
        and operations[0].suffix == "ylium"
    ):
        return "yliumyl"
    return (
        "".join(f"-{','.join(operation.locants)}-{suffix_operation_spelling(operation)}" for operation in operations)
        + terminal_e
    )
