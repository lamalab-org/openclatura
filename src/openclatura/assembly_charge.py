"""Charge-aware parent assembly helpers."""

from dataclasses import dataclass

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


def positive_parent_ium_charges(parts: AssemblyParts) -> list[ParentChargeItem]:
    """Return parent atoms whose positive charge is represented by an ium suffix."""

    return [charge for charge in parts.parent_charges if charge.symbol in {"N", "O"} and charge.charge > 0]


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
    suffix_locs = tuple(parent_charge_suffix_locs(parts))
    if not suffix_locs:
        return []
    rule = RULES.charges.parent_charge_suffixes["N:+"]
    return [
        ParentSuffixOperation(
            key="parent-n-cation-suffix",
            locants=suffix_locs,
            suffix=rule.suffix,
            reason=rule.reason,
            charge=1,
            atom_symbols=tuple(sorted({charge.symbol for charge in positive_parent_ium_charges(parts)})),
        )
    ]


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
    """Resolve charged parent operations before suffixes or branches render."""

    operations = fusion_parent_charge_name_operations(parts)
    if operations is None:
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
    return (
        "".join(f"-{','.join(operation.locants)}-{suffix_operation_spelling(operation)}" for operation in operations)
        + terminal_e
    )
