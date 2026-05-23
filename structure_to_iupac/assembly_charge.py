"""Charge-aware parent assembly helpers."""

from dataclasses import dataclass

from .assembly_parts import AssemblyParts, ParentChargeItem
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


def has_ionic_retained_parent(parts: AssemblyParts) -> bool:
    return bool(parts.retained_name in RULES.charges.retained_ionic_n_parents and positive_parent_n_charges(parts))


def has_retained_like_parent(parts: AssemblyParts) -> bool:
    return bool(parts.retained_name or inferred_ionic_retained_parent(parts))


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
        [
            charge.locant
            for charge in positive_parent_n_charges(parts)
            if charge.locant not in represented_as_azonia
        ],
        key=parse_locant,
    )


def parent_charge_operations(parts: AssemblyParts) -> list[ParentChargeOperation]:
    return [
        ParentChargeOperation(
            locants=operation.locants,
            suffix=operation.suffix,
            reason=operation.reason,
        )
        for operation in parent_charge_name_operations(parts)
    ]


def parent_charge_name_operations(parts: AssemblyParts) -> list[ParentSuffixOperation]:
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
            atom_symbols=("N",),
        )
    ]


def append_charge_suffixes_to_terminal(parts: AssemblyParts, terminal_e: str) -> str:
    operations = parent_charge_name_operations(parts)
    if not operations:
        return terminal_e
    return "".join(
        f"-{','.join(operation.locants)}-{suffix_operation_spelling(operation)}" for operation in operations
    ) + terminal_e
