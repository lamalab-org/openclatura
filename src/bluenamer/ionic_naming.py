"""Charge-preserving naming helpers.

The core parent/substituent pipeline names connectivity first.  These helpers
apply small, graph-backed ionic spelling changes where OPSIN needs the formal
charge to be explicit in the generated name.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass

from .formatting import format_counted_prefixes
from .molecule import Molecule
from .name_operations import ParentSuffixOperation
from .nomenclature import RULES
from .suffix_stack import place_anion_suffix_operations

IONIC_PARENT_SUFFIX_PATTERN = (
    r"(?P<suffix>-\d+-carboxylate|carboxylate|-\d+-carboxamide|-\d+-carbonitrile|"
    r"-\d+-carbaldehyde|-\d+-ol|-\d+-one|-\d+-ylidynyl|-\d+-ylidyne|-\d+-ylidene|"
    r"-\d+-ylmethyl|-\d+-yl|$)"
)


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
    if any(re.match(r"^[A-Z][a-z]?-", branch) for branch in branches):
        return f"({''.join(branches)}ammonio)"
    return f"({format_counted_prefixes(branches)}ammonio)"


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

    name = apply_ring_parent_nitrogen_zwitterion_stack(name, mol, numbered_path, get_loc)
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


def apply_fused_heteroaromatic_nitrogen_zwitterion(name: str, mol: Molecule, numbered_path: list[int], get_loc) -> str:
    """Compatibility wrapper for the ring N-zwitterion parent stack."""

    return apply_ring_parent_nitrogen_zwitterion_stack(name, mol, numbered_path, get_loc)


def apply_ring_parent_nitrogen_zwitterion_stack(name: str, mol: Molecule, numbered_path: list[int], get_loc) -> str:
    """Spell graph-proven ring-parent adjacent/conjugated ``[N-]``/``[NH+]``.

    OPSIN accepts ``...-<N->-ide-<NH+>-ium`` for unsaturated hetero ring
    parents where both charged nitrogens are part of the parent skeleton.  This
    is not a terminal ``locant-ide`` fallback: it only runs when a parent N
    cation is already represented as ``-<locant>-ium`` and the graph proves a
    cyclic unsaturated parent with exactly one anionic N and one cationic N.
    """

    if "-ium" not in name or "-ide" in name:
        return name
    parent_set = set(numbered_path)
    if not _is_unsaturated_ring_parent(mol, parent_set):
        return name
    sites = parent_charge_sites(mol, numbered_path, get_loc)
    negative_nitrogens = [site for site in sites if site.symbol == "N" and site.charge < 0]
    positive_nitrogens = [site for site in sites if site.symbol == "N" and site.charge > 0]
    if len(negative_nitrogens) != 1 or len(positive_nitrogens) != 1:
        return name
    negative = negative_nitrogens[0]
    positive = positive_nitrogens[0]
    if not _name_has_parent_ium_locant(name, positive.locant):
        return name
    if not _charged_nitrogens_are_adjacent_or_conjugated(mol, parent_set, negative.atom_idx, positive.atom_idx):
        return name
    if f"-{negative.locant}-ide-{positive.locant}-ium" in name:
        return name
    return re.sub(
        rf"-{re.escape(positive.locant)}-ium\b",
        f"-{negative.locant}-ide-{positive.locant}-ium",
        name,
        count=1,
    )


def _is_unsaturated_ring_parent(mol: Molecule, parent_set: set[int]) -> bool:
    if len(parent_set) < 3:
        return False
    internal_edge_count = 0
    has_unsaturation = False
    for atom_idx in parent_set:
        for neighbor in mol.get_neighbors(atom_idx):
            if neighbor not in parent_set or atom_idx >= neighbor:
                continue
            internal_edge_count += 1
            bond = mol.get_bond(atom_idx, neighbor)
            if bond is not None and bond.order > 1:
                has_unsaturation = True
    return internal_edge_count >= len(parent_set) and has_unsaturation


def _name_has_parent_ium_locant(name: str, locant: str) -> bool:
    match = re.search(rf"-{re.escape(locant)}-ium\b", name)
    if match is None:
        return False
    suffix_tail = name[match.end() :]
    return suffix_tail == ""


def _charged_nitrogens_are_adjacent_or_conjugated(
    mol: Molecule, parent_set: set[int], anion_idx: int, cation_idx: int
) -> bool:
    if cation_idx in mol.get_neighbors(anion_idx):
        return True
    queue: list[tuple[int, int]] = [(anion_idx, 0)]
    visited = {anion_idx}
    while queue:
        atom_idx, distance = queue.pop(0)
        if distance >= 3:
            continue
        for neighbor in mol.get_neighbors(atom_idx):
            if neighbor not in parent_set or neighbor in visited:
                continue
            bond = mol.get_bond(atom_idx, neighbor)
            if bond is None or bond.order not in {1, 2}:
                continue
            if neighbor == cation_idx:
                return True
            visited.add(neighbor)
            queue.append((neighbor, distance + 1))
    return False


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
        retained_names = list(RULES.charges.retained_ionic_n_parents)
    for candidate in retained_names:
        updated = apply_one_ionic_retained_parent_name(name, candidate)
        if updated != name:
            return updated
    return name


def apply_one_ionic_retained_parent_name(name: str, retained_name: str | None) -> str:
    ionic_name = RULES.charges.retained_ionic_n_parents.get(retained_name or "")
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
        r"(?P<parent>(?:\d+(?:,\d+)*-)?(?:\d+H-)?[A-Za-z][A-Za-z0-9,\-\[\]\^\{\}]*?)" + IONIC_PARENT_SUFFIX_PATTERN
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
    rule = RULES.charges.parent_charge_suffixes["*:-"]
    for locant in sorted(negative_locants, key=lambda value: (not value.isdigit(), value)):
        for symbol in sorted(negative_locants[locant]):
            operations.append(
                AnionicSuffixOperation(
                    locant=locant,
                    atom_symbol=symbol,
                    operation=rule.suffix,
                    reason=rule.reason,
                )
            )
    return operations


def parent_suffix_operations(negative_locants: dict[str, set[str]]) -> list[ParentSuffixOperation]:
    return [
        ParentSuffixOperation(
            key="parent-anion-suffix",
            locants=(operation.locant,),
            suffix=operation.operation,
            reason=operation.reason,
            charge=-1,
            atom_symbols=(operation.atom_symbol,),
        )
        for operation in anionic_suffix_operations(negative_locants)
    ]


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

    return place_anion_suffix_operations(
        name,
        parent_suffix_operations(negative_locants),
        rule_keys={"ketone_to_oxo_ide"},
    )


def apply_aldehyde_suffix_ide(name: str, negative_locants: dict[str, set[str]]) -> str:
    """Convert acyclic aldehyde suffix names to locanted carbanion aldehydes."""

    return place_anion_suffix_operations(
        name,
        parent_suffix_operations(negative_locants),
        rule_keys={"aldehyde_to_ide_al"},
    )


def apply_carbaldehyde_suffix_ide(name: str, negative_locants: dict[str, set[str]]) -> str:
    """Insert an anion suffix before ring carbaldehyde suffixes.

    OPSIN treats ``...-carbaldehyde`` parents as neutral unless the anionic
    ring atom is represented on the parent stem before the characteristic-group
    suffix.
    """

    return place_anion_suffix_operations(
        name,
        parent_suffix_operations(negative_locants),
        rule_keys={"carbaldehyde_suffix_ide"},
    )


def apply_nitrile_suffix_ide(name: str, negative_locants: dict[str, set[str]]) -> str:
    """Insert carbanion ``-ide`` operations before nitrile suffixes.

    OPSIN treats names such as ``propanedinitrile`` and ``cyclo...carbonitrile``
    as neutral unless the carbanion site is represented in the parent suffix
    stack.  This rule is graph-backed by the charged parent locant and only
    applies to carbon anions.
    """

    updated = place_anion_suffix_operations(
        name,
        parent_suffix_operations(negative_locants),
        rule_keys={"dinitrile_suffix_ide"},
    )
    updated = place_anion_suffix_operations(
        updated,
        parent_suffix_operations(negative_locants),
        rule_keys={"locanted_carbonitrile_suffix_ide"},
    )
    return place_anion_suffix_operations(
        updated,
        parent_suffix_operations(negative_locants),
        rule_keys={"terminal_carbonitrile_suffix_ide"},
    )


def apply_parent_anion_rule(name: str, site: ParentChargeSite, context: ParentChargeContext) -> str:
    negative_locants = {site.locant: {site.symbol}}
    if not negative_locants:
        return name

    updated = apply_one_suffix_ide(name, negative_locants)
    updated = apply_aldehyde_suffix_ide(updated, negative_locants)
    updated = apply_carbaldehyde_suffix_ide(updated, negative_locants)
    updated = apply_nitrile_suffix_ide(updated, negative_locants)
    updated = apply_terminal_characteristic_suffix_ide(updated, negative_locants)
    if updated != name:
        return updated
    updated = apply_retained_parent_ide(name, context.retained_name, negative_locants)
    if updated != name:
        return updated
    return apply_terminal_parent_ide(name, negative_locants)


def apply_terminal_parent_ide(name: str, negative_locants: dict[str, set[str]]) -> str:
    """Do not append unclassified locant-ide suffixes.

    Generic terminal ``-<locant>-ide`` emission is not grammar-safe: it can land
    after suffixes, substituent phrases, or parent words whose anion grammar is
    class-specific.  Keep the parseable neutral/zwitterionic spelling until a
    dedicated anion class installs a verified template.
    """

    return name


INVALID_LOCANT_IDE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"-\d+-ide\b(?!-\d+-ium)"),
    re.compile(r"yl\)-\d+-ide\b"),
    re.compile(r"carbonitrile-\d+-ide\b"),
    re.compile(r"carbaldehyde-\d+-ide\b"),
    re.compile(r"amide-\d+-ide\b"),
    re.compile(r"nitrile-\d+-ide\b"),
)


def contains_invalid_locant_ide(name: str) -> bool:
    """Return whether a generated name contains a known-invalid locant-ide form."""

    return any(pattern.search(name) for pattern in INVALID_LOCANT_IDE_PATTERNS)


def apply_terminal_characteristic_suffix_ide(name: str, negative_locants: dict[str, set[str]]) -> str:
    """Place parent ``-ide`` before terminal suffixes that cannot follow it.

    OPSIN accepts suffix stacks such as ``...ene-5-ide-2,6-dione`` and
    ``...-6-ide-6-aminium``.  Appending ``-ide`` after those suffixes describes
    an unparsable suffix-on-suffix stack, so this rule inserts the anion
    operation at the parent/suffix boundary using the graph-backed anion
    locant.
    """

    return place_anion_suffix_operations(
        name,
        parent_suffix_operations(negative_locants),
        rule_keys={"terminal_characteristic_suffix_ide"},
    )


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
    has_cationic_imidamide = False
    for atom_idx, atom in mol.atoms.items():
        if atom.symbol != "N" or atom.charge <= 0:
            continue
        for neighbor in mol.get_neighbors(atom_idx):
            bond = mol.get_bond(atom_idx, neighbor)
            if bond and bond.order == 2 and mol.atoms[neighbor].is_carbon:
                has_cationic_imine = True
                has_cationic_imidamide = carbon_has_two_hetero_substituents(mol, neighbor, atom_idx)
                break
        if has_cationic_imine:
            break
    if not has_cationic_imine:
        return name
    if has_cationic_imidamide:
        name = normalize_cationic_methylideneammonio_substituents(name)
    name = name.replace("(imino)methyl", "(iminio)methyl")
    name = name.replace("iminomethyl", "iminiomethyl")
    return name


def carbon_has_two_hetero_substituents(mol: Molecule, carbon_idx: int, imino_n_idx: int) -> bool:
    hetero_neighbors = [
        neighbor
        for neighbor in mol.get_neighbors(carbon_idx)
        if neighbor != imino_n_idx and mol.atoms[neighbor].symbol in {"N", "O", "S"}
    ]
    return len(hetero_neighbors) >= 2


def normalize_cationic_methylideneammonio_substituents(name: str) -> str:
    """Bind imidamide/imino-ether cation charge to the exocyclic N atom.

    ``((amino)(alkoxy)methylideneammonio)`` is parsed by OPSIN with the
    positive charge on the substituent heteroatom.  The ``N-(...)ammonio`` form
    preserves a graph pattern where the cationic nitrogen is double-bonded to
    the central carbon.
    """

    pattern = re.compile(
        r"\(\(\((?P<left>[A-Za-z0-9,\-\[\]\^\{\}']+)\)"
        r"\((?P<right>[A-Za-z0-9,\-\[\]\^\{\}']+)\)"
        r"methylidene\)ammonio\)"
    )
    return pattern.sub(
        lambda match: f"(N-(({match.group('left')})({match.group('right')})methylidene)ammonio)",
        name,
    )
