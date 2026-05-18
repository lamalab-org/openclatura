"""Charge-preserving naming helpers.

The core parent/substituent pipeline names connectivity first.  These helpers
apply small, graph-backed ionic spelling changes where OPSIN needs the formal
charge to be explicit in the generated name.
"""

import re
from dataclasses import dataclass
from typing import Callable

from .charge_specs import IONIC_RETAINED_N_PARENTS
from .molecule import Molecule


IONIC_PARENT_SUFFIX_PATTERN = (
    r"(?P<suffix>-\d+-carboxylate|carboxylate|-\d+-carboxamide|-\d+-carbonitrile|"
    r"-\d+-carbaldehyde|-\d+-ol|-\d+-one|-\d+-ylidynyl|-\d+-ylidyne|-\d+-ylidene|"
    r"-\d+-ylmethyl|-\d+-yl|$)"
)
IMPLICIT_RETAINED_CATION_STEMS = {
    ionic_name: retained_stem_name[:-1] if retained_stem_name.endswith("e") else retained_stem_name
    for retained_stem_name, ionic_name in IONIC_RETAINED_N_PARENTS.items()
}


@dataclass(frozen=True)
class ParentChargeSite:
    """A charged atom that belongs to the numbered parent."""

    atom_idx: int
    locant: str
    symbol: str
    charge: int


@dataclass(frozen=True)
class ParentChargeContext:
    """Name and parent metadata available to charge spelling rules."""

    retained_name: str | None = None
    allow_retained_stem_inference: bool = False


@dataclass(frozen=True)
class ParentChargeRule:
    """One graph-backed spelling rule for charged parent atoms."""

    key: str
    charge_sign: int
    applies_to: Callable[[ParentChargeSite], bool]
    apply: Callable[[str, ParentChargeSite, ParentChargeContext], str]


@dataclass(frozen=True)
class AnionicSuffixOperation:
    locant: str
    atom_symbol: str
    operation: str
    reason: str


def ammonio_prefix(branches: list[str]) -> str:
    """Return an ammonio-style prefix for a protonated nitrogen substituent."""

    if not branches:
        return "ammonio"
    counts: dict[str, int] = {}
    order: list[str] = []
    for branch in branches:
        if branch not in counts:
            order.append(branch)
        counts[branch] = counts.get(branch, 0) + 1
    parts = []
    for branch in order:
        count = counts[branch]
        if count == 1:
            parts.append(branch)
        elif count == 2:
            parts.append(f"di{branch}")
        elif count == 3:
            parts.append(f"tri{branch}")
        else:
            parts.append(f"{count}{branch}")
    return f"({''.join(parts)}ammonio)"


def apply_parent_charge_names(
    name: str,
    mol: Molecule,
    numbered_path: list[int],
    get_loc,
    retained_name: str | None = None,
    allow_retained_stem_inference: bool = False,
    charge_signs: set[int] | None = None,
) -> str:
    """Apply graph-backed charge spelling rules to the selected parent."""

    context = ParentChargeContext(
        retained_name=retained_name,
        allow_retained_stem_inference=allow_retained_stem_inference,
    )
    for site in parent_charge_sites(mol, numbered_path, get_loc):
        if charge_signs is not None and charge_sign(site.charge) not in charge_signs:
            continue
        for rule in PARENT_CHARGE_RULES:
            if rule.charge_sign != charge_sign(site.charge) or not rule.applies_to(site):
                continue
            updated = rule.apply(name, site, context)
            if updated != name:
                name = updated
                break
    return name


def apply_parent_ion_suffixes(
    name: str,
    mol: Molecule,
    numbered_path: list[int],
    get_loc,
    retained_name: str | None = None,
    allow_retained_stem_inference: bool = False,
) -> str:
    """Compatibility wrapper for parent charge spelling."""

    return apply_parent_charge_names(
        name,
        mol,
        numbered_path,
        get_loc,
        retained_name,
        allow_retained_stem_inference,
        charge_signs={1},
    )


def apply_anionic_parent_names(
    name: str,
    mol: Molecule,
    numbered_path: list[int],
    get_loc,
    retained_name: str | None = None,
) -> str:
    """Compatibility wrapper for parent charge spelling."""

    return apply_parent_charge_names(name, mol, numbered_path, get_loc, retained_name, charge_signs={-1})


def parent_charge_sites(mol: Molecule, numbered_path: list[int], get_loc) -> list[ParentChargeSite]:
    sites = []
    for atom_idx in numbered_path:
        atom = mol.atoms[atom_idx]
        if atom.charge == 0:
            continue
        sites.append(
            ParentChargeSite(
                atom_idx=atom_idx,
                locant=str(get_loc(atom_idx)),
                symbol=atom.symbol,
                charge=atom.charge,
            )
        )
    return sites


def charge_sign(charge: int) -> int:
    return 1 if charge > 0 else -1


def apply_ionic_retained_parent_name(
    name: str,
    retained_name: str | None,
    cation_locant: str,
    allow_stem_inference: bool = False,
) -> str:
    """Convert the selected retained N parent without touching substituents."""

    retained_names = [retained_name] if retained_name else []
    if allow_stem_inference and not retained_names:
        retained_names = list(IONIC_RETAINED_N_PARENTS)
    for candidate in retained_names:
        updated = apply_one_ionic_retained_parent_name(name, candidate)
        if updated != name:
            return updated
    return name


def apply_one_ionic_retained_parent_name(name: str, retained_name: str | None) -> str:
    ionic_name = IONIC_RETAINED_N_PARENTS.get(retained_name or "")
    if not ionic_name or not retained_name:
        return name
    stem = retained_stem(retained_name)
    terminal = "e?" if retained_name and retained_name.endswith("e") else ""
    pattern = re.compile(rf"(?:\d+H-)?{re.escape(stem)}{terminal}{IONIC_PARENT_SUFFIX_PATTERN}")
    matches = [match for match in pattern.finditer(name) if is_top_level_span(name, match.start())]
    if not matches:
        return name
    match = matches[-1]
    return name[: match.start()] + ionic_name + match.group("suffix") + name[match.end() :]


def is_top_level_span(name: str, position: int) -> bool:
    depth = 0
    for char in name[:position]:
        if char == "(":
            depth += 1
        elif char == ")" and depth:
            depth -= 1
    return depth == 0


def apply_cation_retained_parent_rule(name: str, site: ParentChargeSite, context: ParentChargeContext) -> str:
    return apply_ionic_retained_parent_name(
        name,
        context.retained_name,
        site.locant,
        context.allow_retained_stem_inference,
    )


def apply_replacement_prefix_cation_rule(name: str, site: ParentChargeSite, context: ParentChargeContext) -> str:
    return name.replace(f"{site.locant}-aza", f"{site.locant}-azonia")


def apply_parent_ium_rule(name: str, site: ParentChargeSite, context: ParentChargeContext) -> str:
    updated = apply_retained_parent_ium(name, context.retained_name, site.locant)
    if updated != name:
        return updated
    return insert_parent_ium_suffix(name, site.locant)


def apply_retained_parent_ium(name: str, retained_name: str | None, cation_locant: str) -> str:
    stem = retained_stem(retained_name)
    if not stem:
        return name
    terminal = "e" if retained_name and retained_name.endswith("e") else ""
    pattern = re.compile(rf"(?:\d+H-)?{re.escape(stem)}{terminal}?(?!ium|-\d+-ium)")
    return pattern.sub(f"{stem}-{cation_locant}-ium", name)


def insert_parent_ium_suffix(name: str, locant: str) -> str:
    """Insert an ium suffix into the parent token for cationic ring nitrogens."""

    if f"-{locant}-ium" in name or f"{locant}-azonia" in name:
        return name
    pattern = re.compile(
        r"(?P<parent>(?:\d+(?:,\d+)*-)?(?:\d+H-)?[A-Za-z][A-Za-z0-9,\-\[\]\^\{\}]*?)"
        + IONIC_PARENT_SUFFIX_PATTERN
    )

    def repl(match: re.Match) -> str:
        parent = match.group("parent")
        if parent.endswith("e"):
            parent = parent[:-1]
        return f"{parent}-{locant}-ium{match.group('suffix')}"

    matches = list(pattern.finditer(name))
    if not matches:
        return name
    match = matches[-1]
    return name[: match.start()] + repl(match) + name[match.end() :]


def charge_locants(sites: list[ParentChargeSite], charge: int) -> dict[str, set[str]]:
    locants: dict[str, set[str]] = {}
    sign = charge_sign(charge)
    for site in sites:
        if charge_sign(site.charge) != sign:
            continue
        locants.setdefault(site.locant, set()).add(site.symbol)
    return locants


def charged_locants(mol: Molecule, numbered_path: list[int], get_loc, charge: int) -> dict[str, set[str]]:
    return charge_locants(parent_charge_sites(mol, numbered_path, get_loc), charge)


def anionic_suffix_operations(negative_locants: dict[str, set[str]]) -> list[AnionicSuffixOperation]:
    operations = []
    for locant in sorted(negative_locants, key=lambda value: (not value.isdigit(), value)):
        for symbol in sorted(negative_locants[locant]):
            operations.append(
                AnionicSuffixOperation(
                    locant=locant,
                    atom_symbol=symbol,
                    operation="ide",
                    reason="Negative parent atoms are rendered as parent anion suffix operations.",
                )
            )
    return operations


def _replace_stem_with_ide(name: str, neutral_stem: str, charged_stem: str, anion_locant: str) -> str:
    escaped = re.escape(neutral_stem)
    charged_stem = charged_stem[:-1] if charged_stem.endswith("e") else charged_stem
    pattern = re.compile(rf"(?<![A-Za-z0-9])(?:\d+H-)?{escaped}e?(?!-\d+-ide)")
    return pattern.sub(f"{charged_stem}-{anion_locant}-ide", name)


def retained_stem(retained_name: str | None) -> str:
    if not retained_name:
        return ""
    return retained_name[:-1] if retained_name.endswith("e") else retained_name


def apply_retained_parent_ide(name: str, retained_name: str | None, negative_locants: dict[str, set[str]]) -> str:
    stem = retained_stem(retained_name)
    if not stem:
        return name
    updated = name
    for locant in sorted(negative_locants, key=lambda value: (not value.isdigit(), value)):
        updated = _replace_stem_with_ide(updated, stem, stem, locant)
    return updated


def apply_one_suffix_ide(name: str, negative_locants: dict[str, set[str]]) -> str:
    """Convert generated ketone suffix parents to oxo/ide form by grammar."""

    updated = name
    for operation in anionic_suffix_operations(negative_locants):
        anion_locant = operation.locant
        pattern = re.compile(
            r"(?<![A-Za-z0-9])(?:\d+H-)?"
            r"(?P<stem>[A-Za-z0-9,\[\]\^\{\}](?:[A-Za-z0-9,\-\[\]\^\{\}\.']|\(\d+\))*?)"
            r"-(?P<oxo_locant>\d+)-one(?![A-Za-z])"
        )

        def repl(match: re.Match) -> str:
            stem = explicit_implicit_cation_locant(match.group("stem"))
            oxo_locant = match.group("oxo_locant")
            separator = "-" if match.start() > 0 and match.string[match.start() - 1] not in "- " else ""
            return f"{separator}{oxo_locant}-oxo-{stem}-{anion_locant}-ide"

        updated = pattern.sub(repl, updated)
    return updated


def explicit_implicit_cation_locant(stem: str) -> str:
    """Make implicit retained N-cation stems explicit before adding anion suffixes."""

    for ionic_name, neutral_stem in IMPLICIT_RETAINED_CATION_STEMS.items():
        if stem.endswith(ionic_name):
            return stem[: -len(ionic_name)] + f"{neutral_stem}-1-ium"
    return stem


def apply_aldehyde_suffix_ide(name: str, negative_locants: dict[str, set[str]]) -> str:
    """Convert acyclic aldehyde suffix names to locanted carbanion aldehydes."""

    carbon_locants = [
        operation.locant for operation in anionic_suffix_operations(negative_locants) if operation.atom_symbol == "C"
    ]
    updated = name
    for anion_locant in sorted(carbon_locants, key=lambda value: (not value.isdigit(), value)):
        pattern = re.compile(r"(?P<stem>[A-Za-z0-9,\-\[\]\^\{\}]+?an)al\b")
        updated = pattern.sub(lambda match: f"{match.group('stem')}-{anion_locant}-ide-1-al", updated, count=1)
    return updated


def apply_nitrile_suffix_ide(name: str, negative_locants: dict[str, set[str]]) -> str:
    """Insert carbanion ``-ide`` operations before nitrile suffixes.

    OPSIN treats names such as ``propanedinitrile`` and ``cyclo...carbonitrile``
    as neutral unless the carbanion site is represented in the parent suffix
    stack.  This rule is graph-backed by the charged parent locant and only
    applies to carbon anions.
    """

    carbon_locants = [
        operation.locant for operation in anionic_suffix_operations(negative_locants) if operation.atom_symbol == "C"
    ]
    updated = name
    for anion_locant in sorted(carbon_locants, key=lambda value: (not value.isdigit(), value)):
        if f"-{anion_locant}-ide" in updated:
            continue
        updated = re.sub(
            r"(?P<stem>[A-Za-z0-9,\-\[\]\^\{\}]+?ane)dinitrile\b",
            lambda match: f"{match.group('stem')}-{anion_locant}-ide-1,3-dinitrile",
            updated,
            count=1,
        )
        before_carbonitrile = updated
        updated = re.sub(
            r"(?P<stem>[A-Za-z0-9,\-\[\]\^\{\}]+?)-(?P<suffix_locant>\d+)-carbonitrile\b",
            lambda match: f"{match.group('stem')}-{anion_locant}-ide-{match.group('suffix_locant')}-carbonitrile",
            updated,
            count=1,
        )
        if updated != before_carbonitrile:
            continue
        updated = re.sub(
            r"(?P<stem>[A-Za-z0-9,\-\[\]\^\{\}]+?)carbonitrile\b",
            lambda match: f"{match.group('stem')}-{anion_locant}-ide-carbonitrile",
            updated,
            count=1,
        )
    return updated


def apply_parent_anion_rule(name: str, site: ParentChargeSite, context: ParentChargeContext) -> str:
    negative_locants = {site.locant: {site.symbol}}
    if not negative_locants:
        return name

    updated = apply_one_suffix_ide(name, negative_locants)
    updated = apply_aldehyde_suffix_ide(updated, negative_locants)
    updated = apply_nitrile_suffix_ide(updated, negative_locants)
    if updated != name:
        return updated
    updated = apply_retained_parent_ide(name, context.retained_name, negative_locants)
    return updated


PARENT_CHARGE_RULES: tuple[ParentChargeRule, ...] = (
    ParentChargeRule(
        key="cation-retained-n-parent",
        charge_sign=1,
        applies_to=lambda site: site.symbol == "N",
        apply=apply_cation_retained_parent_rule,
    ),
    ParentChargeRule(
        key="cation-replacement-prefix",
        charge_sign=1,
        applies_to=lambda site: site.symbol == "N",
        apply=apply_replacement_prefix_cation_rule,
    ),
    ParentChargeRule(
        key="cation-parent-ium-suffix",
        charge_sign=1,
        applies_to=lambda site: site.symbol == "N",
        apply=apply_parent_ium_rule,
    ),
    ParentChargeRule(
        key="anion-parent-ide",
        charge_sign=-1,
        applies_to=lambda site: True,
        apply=apply_parent_anion_rule,
    ),
)


def apply_cationic_imino_names(name: str, mol: Molecule) -> str:
    """Make cationic C=N fragments explicit when they appear in nested names."""

    has_cationic_imine = False
    for atom_idx, atom in mol.atoms.items():
        if atom.symbol != "N" or atom.charge <= 0:
            continue
        for neighbor in mol.get_neighbors(atom_idx):
            bond = mol.get_bond(atom_idx, neighbor)
            if bond and bond.order == 2 and mol.atoms[neighbor].is_carbon:
                has_cationic_imine = True
                break
        if has_cationic_imine:
            break
    if not has_cationic_imine:
        return name
    name = name.replace("(imino)methyl", "(iminio)methyl")
    name = name.replace("iminomethyl", "iminiomethyl")
    return name
