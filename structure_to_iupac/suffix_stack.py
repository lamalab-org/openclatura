"""Structured parent suffix rendering helpers."""

from dataclasses import dataclass, field
import re

from .name_operations import ParentSuffixOperation
from .nomenclature import RULES
from .rules import multipliers


@dataclass
class SuffixStack:
    """A parent tail plus ordered suffix operations.

    The current assembler still renders most of the parent word directly.
    This adapter centralizes charge-suffix placement so the old regex helpers
    can shrink into data-backed placement rules.
    """

    stem: str
    suffix: str = ""
    operations: list[ParentSuffixOperation] = field(default_factory=list)
    append_unplaced: bool = True

    def render(self) -> str:
        text = self.stem + self.suffix
        for operation in self.operations:
            text = place_parent_suffix_operation(text, operation, append_unplaced=self.append_unplaced)
        return text


def place_parent_suffix_operation(
    text: str,
    operation: ParentSuffixOperation,
    *,
    append_unplaced: bool = True,
) -> str:
    """Place one parent suffix operation using registry placement policy."""

    operation_text = _operation_text(operation)
    if not operation_text or operation_text in text:
        return text
    for rule in RULES.charges.anion_suffix_placements:
        if operation.suffix != "ide" or rule.placement != "before_suffix":
            continue
        if not rule.suffix_pattern:
            continue
        pattern = re.compile(rf"(?P<stem>.+?)(?P<suffix>{rule.suffix_pattern})")
        match = pattern.match(text)
        if match:
            return f"{match.group('stem')}{operation_text}{match.group('suffix')}"
    if not append_unplaced:
        return text
    return f"{text}{operation_text}"


def place_anion_suffix_operations(
    text: str,
    operations: list[ParentSuffixOperation],
    *,
    rule_keys: set[str] | None = None,
    append_unplaced: bool = False,
) -> str:
    """Apply data-backed anion suffix-placement rules to rendered parent text."""

    updated = text
    for operation in operations:
        if operation.suffix != "ide":
            continue
        if _operation_text(operation) in updated:
            continue
        for rule in RULES.charges.anion_suffix_placements:
            if rule_keys is not None and rule.key not in rule_keys:
                continue
            if not _rule_accepts_operation(rule.atom_symbols, operation):
                continue
            candidate = _apply_anion_suffix_rule(updated, operation, rule.key)
            if candidate != updated:
                updated = candidate
                break
        else:
            if append_unplaced:
                updated = place_parent_suffix_operation(updated, operation)
    return updated


def _rule_accepts_operation(rule_symbols: tuple[str, ...], operation: ParentSuffixOperation) -> bool:
    if "*" in rule_symbols:
        return True
    return any(symbol in rule_symbols for symbol in operation.atom_symbols)


def _apply_anion_suffix_rule(text: str, operation: ParentSuffixOperation, key: str) -> str:
    locant = operation.locants[0] if operation.locants else ""
    if not locant:
        return text
    if key == "ketone_to_oxo_ide":
        return _apply_ketone_to_oxo_ide(text, locant)
    if key == "aldehyde_to_ide_al":
        return re.sub(
            r"(?P<stem>[A-Za-z0-9,\-\[\]\^\{\}]+?an)al\b",
            lambda match: f"{match.group('stem')}-{locant}-ide-1-al",
            text,
            count=1,
        )
    if key == "carbaldehyde_suffix_ide":
        return re.sub(
            r"(?P<stem>[A-Za-z0-9,\-\[\]\^\{\}\.'](?:[A-Za-z0-9,\-\[\]\^\{\}\.']|\(\d+\))*?)"
            r"-(?P<suffix_locant>\d+)-carbaldehyde\b",
            lambda match: f"{explicit_implicit_cation_locant(match.group('stem'))}-{locant}-ide-{match.group('suffix_locant')}-carbaldehyde",
            text,
            count=1,
        )
    if key == "dinitrile_suffix_ide":
        return re.sub(
            r"(?P<stem>[A-Za-z0-9,\-\[\]\^\{\}]+?ane)dinitrile\b",
            lambda match: f"{match.group('stem')}-{locant}-ide-1,3-dinitrile",
            text,
            count=1,
        )
    if key == "locanted_carbonitrile_suffix_ide":
        return re.sub(
            r"(?P<stem>[A-Za-z0-9,\-\[\]\^\{\}]+?)-(?P<suffix_locant>\d+)-carbonitrile\b",
            lambda match: f"{match.group('stem')}-{locant}-ide-{match.group('suffix_locant')}-carbonitrile",
            text,
            count=1,
        )
    if key == "terminal_carbonitrile_suffix_ide":
        return re.sub(
            r"(?P<stem>[A-Za-z0-9,\-\[\]\^\{\}]+?)carbonitrile\b",
            lambda match: f"{match.group('stem')}-{locant}-ide-carbonitrile",
            text,
            count=1,
        )
    if key == "terminal_characteristic_suffix_ide":
        return SuffixStack(text, operations=[operation], append_unplaced=False).render()
    return text


def _apply_ketone_to_oxo_ide(text: str, locant: str) -> str:
    pattern = re.compile(
        r"(?<![A-Za-z0-9])(?:\d+H-)?"
        r"(?P<stem>[A-Za-z0-9,\[\]\^\{\}](?:[A-Za-z0-9,\-\[\]\^\{\}\.']|\(\d+\))*?)"
        r"-(?P<oxo_locant>\d+)-one(?![A-Za-z])"
    )

    def repl(match: re.Match) -> str:
        stem = explicit_implicit_cation_locant(match.group("stem"))
        oxo_locant = match.group("oxo_locant")
        separator = "-" if match.start() > 0 and match.string[match.start() - 1] not in "- " else ""
        return f"{separator}{oxo_locant}-oxo-{stem}-{locant}-ide"

    return pattern.sub(repl, text)


def explicit_implicit_cation_locant(stem: str) -> str:
    """Make implicit retained N-cation stems explicit before adding anion suffixes."""

    for ionic_name, neutral_stem in IMPLICIT_RETAINED_CATION_STEMS.items():
        if stem.endswith(ionic_name):
            return stem[: -len(ionic_name)] + f"{neutral_stem}-1-ium"
    return stem


def _operation_text(operation: ParentSuffixOperation) -> str:
    if not operation.locants or not operation.suffix:
        return ""
    return f"-{','.join(operation.locants)}-{suffix_operation_spelling(operation)}"


def suffix_operation_spelling(operation: ParentSuffixOperation) -> str:
    """Return the suffix spelling, including multiplicative prefixes."""

    count = len(operation.locants)
    if count <= 1:
        return operation.suffix
    return f"{multipliers.basic(count)}{operation.suffix}"


IMPLICIT_RETAINED_CATION_STEMS = {
    ionic_name: retained_stem_name[:-1] if retained_stem_name.endswith("e") else retained_stem_name
    for retained_stem_name, ionic_name in RULES.charges.retained_ionic_n_parents.items()
}
