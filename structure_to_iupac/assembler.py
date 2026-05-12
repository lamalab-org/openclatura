# structure-to-iupac/assembler.py

import re
from dataclasses import dataclass, field
from typing import Optional
from .name_postprocessing import apply_acyl_amido_postprocessing, apply_data_postprocessing
from .nomenclature import RULES
from .rules import bonds, elision, multipliers, stems, suffixes


def parse_locant(l):
    s = str(l)
    match = re.match(r"^(\d+)([a-zA-Z]*)$", s.split("(")[0])
    if match:
        return (1, float(match.group(1)), match.group(2))
    if any(c.isdigit() for c in s):
        nums = re.findall(r"\d+", s)
        return (1, float(nums[0]) if nums else 0.0, s)
    return (2, 0.0, s)


@dataclass
class SubstituentItem:
    name: str
    locants: list[str]
    atom_ids: set[int] = field(default_factory=set)
    bond_ids: set[int] = field(default_factory=set)
    trace_segments: list[dict] = field(default_factory=list)


@dataclass
class UnsaturationItem:
    bond_key: str
    locants: list[str]
    atom_ids: set[int] = field(default_factory=set)
    bond_ids: set[int] = field(default_factory=set)


@dataclass
class PrincipalGroupItem:
    key: str
    locants: list[str]
    atom_ids: set[int] = field(default_factory=set)
    bond_ids: set[int] = field(default_factory=set)


@dataclass
class AssemblyParts:
    parent_length: int
    is_ring: bool = False
    is_bicycle: bool = False
    is_spiro: bool = False
    is_polycycle: bool = False
    bicycle_xyz: tuple[int, int, int] = (0, 0, 0)
    spiro_xy: tuple[int, int] = (0, 0)
    tricyclo_xyzw: tuple[int, int, int, int] = (0, 0, 0, 0)
    polycycle_descriptor: str | None = None
    is_substituent: bool = False
    is_double_attach: bool = False
    is_triple_attach: bool = False
    attachment_locant: int | str = 1
    retained_name: str | None = None
    front_modifiers: list[str] = field(default_factory=list)
    a_prefixes: list[SubstituentItem] = field(default_factory=list)
    principal_group: Optional[PrincipalGroupItem] = None
    unsaturations: list[UnsaturationItem] = field(default_factory=list)
    substituents: list[SubstituentItem] = field(default_factory=list)
    stereo_features: list[tuple[str, str]] = field(default_factory=list)
    indicated_hydrogens: list[str] = field(default_factory=list)
    parent_atom_ids: set[int] = field(default_factory=set)
    parent_bond_ids: set[int] = field(default_factory=set)


def needs_hyphen(left: str, right: str) -> bool:
    if not left or not right:
        return False
    if (
        right[0].isdigit()
        or right.startswith("N,")
        or right.startswith("N-")
        or right.startswith("N',")
        or right.startswith("N'-")
    ):
        return True
    if left[-1].isdigit():
        return True
    return False


def is_fully_enclosed(s: str) -> bool:
    if not s.startswith("(") or not s.endswith(")"):
        return False
    depth = 0
    for i, c in enumerate(s):
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        if depth == 0 and i < len(s) - 1:
            return False
    return depth == 0


def _post_process_name(name: str) -> str:
    name = apply_data_postprocessing(name)
    name = re.sub(r"(?<![a-zA-Z0-9])1-azacyclopent-2-ene(?![a-zA-Z])", "4,5-dihydro-1H-pyrrole", name)
    name = re.sub(r"(?<![a-zA-Z0-9])1-azacyclopent-3-ene(?![a-zA-Z])", "2,5-dihydro-1H-pyrrole", name)
    name = re.sub(
        r"(?<![a-zA-Z0-9])1-azacyclopent-2-en-(\d+)-(yl|ylidene|ylidyne|ylidynyl)(?![a-zA-Z])",
        r"4,5-dihydro-1H-pyrrol-\1-\2",
        name,
    )
    name = re.sub(
        r"(?<![a-zA-Z0-9])1-azacyclopent-3-en-(\d+)-(yl|ylidene|ylidyne|ylidynyl)(?![a-zA-Z])",
        r"2,5-dihydro-1H-pyrrol-\1-\2",
        name,
    )
    name = re.sub(r"(?<![a-zA-Z0-9])1-azacyclopenta-2,4-diene(?![a-zA-Z])", "1H-pyrrole", name)
    name = re.sub(
        r"(?<![a-zA-Z0-9])1-azacyclopenta-2,4-dien-(\d+)-(yl|ylidene|ylidyne|ylidynyl)(?![a-zA-Z])",
        r"1H-pyrrol-\1-\2",
        name,
    )

    name = re.sub(r"-1-formate\b", "-formate", name)
    name = re.sub(r"\bmethyl 1-\(", "methyl (", name)
    name = re.sub(r"\b(\S+yl) 1-\(", r"\1 (", name)
    name = re.sub(r"\)1-\(", ")(", name)
    name = re.sub(r" 1-\(", " (", name)

    name = re.sub(r"\((.*?)\)\((?<!thi)oxo\)methyl\b", r"(\1)carbonyl", name)
    name = re.sub(r"\((?<!thi)oxo\)\((.*?)\)methyl\b", r"(\1)carbonyl", name)
    name = name.replace("(oxo)methyl", "formyl")
    name = re.sub(r"(?<!thi)oxomethyl\b", "formyl", name)
    name = name.replace("thioxomethyl", "carbonothioyl")

    name = name.replace("1-oxacyclopropan", "oxiran")
    name = name.replace("1-oxacyclopropane", "oxirane")
    name = name.replace("1-thiacyclopropan", "thiiran")
    name = name.replace("1-thiacyclopropane", "thiirane")
    name = name.replace("1-azacyclopropan", "aziridin")
    name = name.replace("1-azacyclopropane", "aziridine")

    name = name.replace("1,3-dioxacyclopentan", "1,3-dioxolan")
    name = name.replace("1,3-dioxacyclopentane", "1,3-dioxolane")
    name = name.replace("1,3-dioxacyclohexan", "1,3-dioxan")
    name = name.replace("1,3-dioxacyclohexane", "1,3-dioxane")
    name = name.replace("1,4-dioxacyclohexan", "1,4-dioxan")
    name = name.replace("1,4-dioxacyclohexane", "1,4-dioxane")
    name = name.replace("1,3-oxathiolan", "1,3-oxathiolan")
    name = name.replace("1,3-oxazolidine", "oxazolidine")
    name = name.replace("1,3-thiazolidine", "thiazolidine")

    name = name.replace("aminoiminomethyl", "carbamimidoyl")
    name = name.replace("amino(imino)methyl", "carbamimidoyl")
    name = name.replace("iminoamino", "diazenyl")
    name = name.replace("aminoimino", "hydrazono")

    name = re.sub(r"\b1-\((.*?)amino\)methanenitrile\b", r"\1cyanamide", name)
    name = re.sub(r"\b1-([a-zA-Z0-9\-\[\]\(\)\,]+?)aminomethanenitrile\b", r"\1cyanamide", name)
    name = name.replace("aminomethanenitrile", "cyanamide")

    replacements =[
        ("1-hydroxymethanoic acid", "carbonic acid"),
        ("1-hydroxymethanoate", "carbonate"),
        ("1-hydroxymethanamide", "carbamic acid"),
        ("1-hydroxymethanenitrile", "cyanic acid"),
        ("1-hydroxymethanoyl", "carboxy"),
        ("1-hydroxymethanoic anhydride", "dicarbonic acid"),
        ("hydroxymethanoic acid", "carbonic acid"),
        ("hydroxymethanoate", "carbonate"),
        ("hydroxymethanamide", "carbamic acid"),
        ("hydroxymethanenitrile", "cyanic acid"),
        ("hydroxymethanoyl", "carboxy"),
        ("hydroxymethanoic anhydride", "dicarbonic acid"),
        ("benzene-1-carboxylic acid", "benzoic acid"),
        ("benzene-1-carboxamide", "benzamide"),
        ("benzene-1-carboxylate", "benzoate"),
        ("benzene-1-carbonitrile", "benzonitrile"),
        ("benzene-1-carbaldehyde", "benzaldehyde"),
        ("benzene-1-carbonyl", "benzoyl"),
        ("benzenecarboxylic acid", "benzoic acid"),
        ("benzenecarcarboxamide", "benzamide"),
        ("benzenecarboxylate", "benzoate"),
        ("benzenecarcarbonitrile", "benzonitrile"),
        ("benzenecarbaldehyde", "benzaldehyde"),
        ("methanoic acid", "formic acid"),
        ("methanamide", "formamide"),
        ("methanoate", "formate"),
        ("methanoyl", "formyl"),
        ("ethanoic acid", "acetic acid"),
        ("ethanamide", "acetamide"),
        ("ethanenitrile", "acetonitrile"),
        ("ethanoate", "acetate"),
        ("ethanoyl", "acetyl"),
        ("propanenitrile", "propionitrile"),
        ("butanenitrile", "butyronitrile"),
        ("2-methylpropan-2-yl", "tert-butyl"),
        ("1,1-dimethylethyl", "tert-butyl"),
        ("(1,1-dimethylethyl)oxy", "tert-butoxy"),
        ("(tert-butyl)oxy", "tert-butoxy"),
        ("tert-butyloxy", "tert-butoxy"),
        ("methylcarbonyloxy", "acetoxy"),
        ("methylcarbonyl", "acetyl"),
        ("ethylcarbonyl", "propionyl"),
        ("propylcarbonyl", "butyryl"),
        ("phenylcarbonyl", "benzoyl"),
        ("(methylcarbonyl)oxy", "acetoxy"),
        ("(ethylcarbonyl)oxy", "propionyloxy"),
        ("(phenylcarbonyl)oxy", "benzoyloxy"),
        ("aminocarbonothioyl", "carbamothioyl"),
        ("ethan-1-ol", "ethanol"),
        ("methan-1-ol", "methanol"),
        ("ethan-1-amine", "ethanamine"),
        ("methan-1-amine", "methanamine"),
        ("(phenylsulfonyl)amino", "benzenesulfonamido"),
        ("phenylsulfonylamino", "benzenesulfonamido"),
        ("(benzenesulfonyl)amino", "benzenesulfonamido"),
        ("benzenesulfonylamino", "benzenesulfonamido"),
        ("(methanesulfonyl)amino", "methanesulfonamido"),
        ("methanesulfonylamino", "methanesulfonamido"),
        ("(ethanesulfonyl)amino", "ethanesulfonamido"),
        ("ethanesulfonylamino", "ethanesulfonamido"),
        ("(chloromethyl)carbonyl", "chloroacetyl"),
        ("((chloromethyl)carbonyl)amino", "2-chloroacetamido"),
        ("(prop-2-enoyl)amino", "acrylamido"),
        ("prop-2-enoylamino", "acrylamido"),
        ("prop-2-enoyloxy", "acryloyloxy"),
        ("1-oxoethan-1-yl", "acetyl"),
        ("1-oxopropan-1-yl", "propionyl"),
        ("1-oxobutan-1-yl", "butyryl"),
        ("1-oxopentan-1-yl", "pentanoyl"),
        ("1-oxohexan-1-yl", "hexanoyl"),
        ("1-oxoethyl", "acetyl"),
        ("1-oxopropyl", "propionyl"),
        ("1-oxobutyl", "butyryl"),
        ("1-oxopentyl", "pentanoyl"),
        ("1-oxohexyl", "hexanoyl"),
        ("benzene-1-sulfonic acid", "benzenesulfonic acid"),
        ("benzene-1-sulfonamide", "benzenesulfonamide"),
        ("benzene-1-thiol", "benzenethiol"),
        ("ethanedioic acid", "oxalic acid"),
        ("propanedioic acid", "malonic acid"),
        ("butanedioic acid", "succinic acid"),
        ("pentanedioic acid", "glutaric acid"),
        ("hexanedioic acid", "adipic acid"),
        ("phenylmethyl", "benzyl"),
        ("benzylcarbonyl", "phenylacetyl"),
        ("phenylmethoxy", "benzyloxy"),
        ("methanehydrazine", "methylhydrazine"),
        ("ethanehydrazine", "ethylhydrazine"),
        ("propanehydrazine", "propylhydrazine"),
        ("benzenehydrazine", "phenylhydrazine"),
        ("fluoroethanoate", "fluoroacetate"),
        ("chloroethanoate", "chloroacetate"),
        ("bromoethanoate", "bromoacetate"),
        ("iodoethanoate", "iodoacetate"),
        ("fluoroethanoic acid", "fluoroacetic acid"),
        ("chloroethanoic acid", "chloroacetic acid"),
        ("bromoethanoic acid", "bromoacetic acid"),
        ("iodoethanoic acid", "iodoacetic acid"),
        ("fluoroethanoyl", "fluoroacetyl"),
        ("chloroethanoyl", "chloroacetyl"),
        ("bromoethanoyl", "bromoacetyl"),
        ("iodoethanoyl", "iodoacetyl"),
        ("1-azacyclobutane", "azetidine"),
        ("1-azacyclobutan-", "azetidin-"),
        ("1-azacyclopentane", "pyrrolidine"),
        ("1-azacyclopentan-", "pyrrolidin-"),
        ("1-azacyclohexane", "piperidine"),
        ("1-azacyclohexan-", "piperidin-"),
        ("1-oxacyclopentane", "oxolane"),
        ("1-oxacyclopentan-", "oxolan-"),
        ("1-oxacyclohexane", "oxane"),
        ("1-oxacyclohexan-", "oxan-"),
        ("1-thiacyclopentane", "thiolane"),
        ("1-thiacyclopentan-", "thiolan-"),
        ("1-thiacyclohexane", "thiane"),
        ("1-thiacyclohexan-", "thian-"),
    ]
    for old, new in replacements:
        if old in["2-methylpropan-2-yl", "1,1-dimethylethyl"]:
            name = re.sub(rf"(?<![a-zA-Z0-9\-,]){re.escape(old)}(?![a-zA-Z])", new, name)
        else:
            if old in["1-azacyclobutane", "1-azacyclopentane", "1-azacyclohexane", "1-oxacyclopentane", "1-oxacyclohexane", "1-thiacyclopentane", "1-thiacyclohexane"]:
                name = re.sub(rf"-{re.escape(old)}(?![a-zA-Z])", new, name)
            name = re.sub(rf"(?<![a-zA-Z]){re.escape(old)}(?![a-zA-Z])", new, name)

    name = apply_data_postprocessing(name)

    name = re.sub(r"(?<!m)ethanoic acid\b", "acetic acid", name)
    name = re.sub(r"(?<!m)ethanamide\b", "acetamide", name)
    name = re.sub(r"(?<!m)ethanenitrile\b", "acetonitrile", name)
    name = re.sub(r"(?<!m)ethanoate\b", "acetate", name)
    name = re.sub(r"(?<!m)ethanoyl\b", "acetyl", name)

    name = apply_acyl_amido_postprocessing(name)

    name = re.sub(r"\b1-methanethiol\b", "methanethiol", name)
    name = re.sub(r"-1-methanethiol\b", "methanethiol", name)

    name = re.sub(r"\b([a-z]+o)iminomethyl\b", r"\1carbonimidoyl", name)
    name = re.sub(r"\bimino\(([a-z]+o)\)methyl\b", r"\1carbonimidoyl", name)
    name = re.sub(r"\bimino([a-z]+o)methyl\b", r"\1carbonimidoyl", name)

    name = re.sub(r"ane-(\d+(?:,\d+)*)-((?:di|tri)?yl(?:idene|idyne)?)\b", r"an-\1-\2", name)
    name = re.sub(r"ene-(\d+(?:,\d+)*)-((?:di|tri)?yl(?:idene|idyne)?)\b", r"en-\1-\2", name)
    name = re.sub(r"yne-(\d+(?:,\d+)*)-((?:di|tri)?yl(?:idene|idyne)?)\b", r"yn-\1-\2", name)

    name = re.sub(r"one-(\d+(?:,\d+)*)-((?:di|tri)?yl(?:idene|idyne)?)\b", r"on-\1-\2", name)
    name = name.replace("one-diyl", "on-diyl")
    name = name.replace("one-yl", "on-yl")

    name = apply_data_postprocessing(name)

    name = name.replace("methan-1-one hydrazone", "formaldehyde hydrazone")
    name = name.replace("methanone hydrazone", "formaldehyde hydrazone")
    name = name.replace("methanal hydrazone", "formaldehyde hydrazone")
    name = name.replace("ethanal hydrazone", "acetaldehyde hydrazone")

    name = name.replace("oxo(hydroxy)aminooxy", "nitrooxy")
    name = name.replace("(oxo(hydroxy)amino)oxy", "nitrooxy")
    name = name.replace("(oxo)(oxido)aminooxy", "nitrooxy")
    name = name.replace("((oxo)(oxido)amino)oxy", "nitrooxy")

    name = re.sub(r"\b(amino|imino)([a-z]+oxy)methyl\b", r"\1(\2)methyl", name)
    name = re.sub(r"\b(amino|imino)([a-z]+oxy)methylidene\b", r"\1(\2)methylidene", name)
    
    name = re.sub(r"ane-(\d+(?:,\d+)*)-hydrazine\b", r"an-\1-ylhydrazine", name)
    name = re.sub(r"ene-(\d+(?:,\d+)*)-hydrazine\b", r"en-\1-ylhydrazine", name)
    name = re.sub(r"yne-(\d+(?:,\d+)*)-hydrazine\b", r"yn-\1-ylhydrazine", name)
    
    name = re.sub(r"\b(amino|imino)\(", r"(\1)(", name)
    name = apply_data_postprocessing(name)

    return name


SPIRO_SUBSTITUENT_RE = re.compile(r"^\[SPIRO\]-(\d+)-(.*)$")
SUBSTITUENT_SORT_PREFIX_RE = re.compile(RULES.assembly.substituent_sort_prefix_pattern)
A_PREFIX_ORDER = RULES.assembly.replacement_prefix_order
UNSATURATION_ORDER = RULES.assembly.unsaturation_order
RETAINED_SUBSTITUENT_STEMS = RULES.retained.substituent_stems
ACID_HALIDE_SUFFIX_KEYS = RULES.assembly.acid_halide_suffix_keys


def _split_spiro_substituents(parts: AssemblyParts) -> list[tuple[str, str, str]]:
    spiro_subs =[]
    normal_subs =[]
    for sub in parts.substituents:
        match = SPIRO_SUBSTITUENT_RE.match(sub.name)
        if match:
            spiro_subs.append((str(sub.locants[0]), match.group(1), match.group(2)))
        else:
            normal_subs.append(sub)
    parts.substituents = normal_subs
    return spiro_subs


def _substituent_sort_key(name: str) -> str:
    s = name.lower()
    s = re.sub(r"^[\(\[\{\)]+", "", s)
    while True:
        match = SUBSTITUENT_SORT_PREFIX_RE.match(s)
        if not match:
            return s
        s = s[match.end() :]
        s = re.sub(r"^[\(\[\{\)]+", "", s)


def _group_substituents(substituents: list[SubstituentItem]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for sub in substituents:
        grouped.setdefault(sub.name, []).extend(sub.locants)
    return grouped


def _substituent_locant_string(parts: AssemblyParts, locs: list[str], grouped_count: int, spiro_subs) -> str:
    if parts.parent_length == 1 and all(str(l).isdigit() for l in locs) and not parts.principal_group and not parts.a_prefixes:
        return ""
    simple_one_locant = (
        len(locs) == 1
        and str(locs[0]) == "1"
        and parts.is_ring
        and not parts.principal_group
        and not parts.unsaturations
        and not parts.is_substituent
        and not parts.a_prefixes
        and grouped_count == 1
        and not parts.is_bicycle
        and not parts.is_spiro
        and not parts.is_polycycle
        and not spiro_subs
    )
    return "" if simple_one_locant else ",".join(map(str, locs))


def _format_substituent_prefixes(parts: AssemblyParts, spiro_subs) -> str:
    if not parts.substituents:
        return ""
    grouped = _group_substituents(parts.substituents)
    prefix_parts =[]
    for name in sorted(grouped.keys(), key=_substituent_sort_key):
        locs = sorted(grouped[name], key=parse_locant)
        attachments_per_group = 2 if ("diyl" in name and "ylidene" not in name) else 1
        count_raw = len(locs) if locs else len([s for s in parts.substituents if s.name == name])
        count = max(1, count_raw // attachments_per_group)
        is_complex = "(" in name or name[0].isdigit() or "-" in name or " " in name
        mult = (multipliers.complex_(count) if is_complex else multipliers.basic(count)) if count > 1 else ""
        loc_str = _substituent_locant_string(parts, locs, len(grouped), spiro_subs)

        name_to_use = name
        if is_complex and not is_fully_enclosed(name):
            if count > 1 or loc_str:
                name_to_use = f"({name})"
        elif not loc_str and len(grouped) > 1 and not is_fully_enclosed(name):
            if name not in["fluoro", "chloro", "bromo", "iodo"]:
                name_to_use = f"({name})"
        prefix_parts.append(f"{loc_str}-{mult}{name_to_use}" if loc_str else f"{mult}{name_to_use}")

    prefix_str = prefix_parts[0]
    for p in prefix_parts[1:]:
        prefix_str += f"-{p}" if needs_hyphen(prefix_str, p) else p
    return prefix_str


def _format_replacement_prefixes(parts: AssemblyParts) -> str:
    if not parts.a_prefixes:
        return ""
    grouped_a = _group_substituents(parts.a_prefixes)
    a_parts =[]
    for name in sorted(grouped_a.keys(), key=lambda n: A_PREFIX_ORDER.get(n, 99)):
        locs = sorted(grouped_a[name], key=parse_locant)
        loc_str = ",".join(map(str, locs))
        count = len(locs)
        mult = multipliers.basic(count) if count > 1 else ""
        a_parts.append(f"{loc_str}-{mult}{name}")
    return "-".join(a_parts)


def _promote_benzene_retained_name(parts: AssemblyParts) -> None:
    if parts.is_ring and not parts.is_bicycle and not parts.is_spiro and parts.parent_length == 6:
        if (
            len(parts.unsaturations) == 1
            and parts.unsaturations[0].bond_key == "double"
            and len(parts.unsaturations[0].locants) == 3
        ):
            locs = sorted([parse_locant(l)[1] for l in parts.unsaturations[0].locants])
            if locs ==[1.0, 3.0, 5.0]:
                if not parts.a_prefixes:
                    parts.retained_name = "benzene"
                    parts.unsaturations =[]


def _parent_stem_and_terminal(parts: AssemblyParts) -> tuple[str, str]:
    terminal_e = bonds.PARENT_TERMINAL_VOWEL

    if parts.retained_name:
        if parts.is_substituent and parts.retained_name in RETAINED_SUBSTITUENT_STEMS:
            stem_str, terminal_e = RETAINED_SUBSTITUENT_STEMS[parts.retained_name]
        else:
            if parts.retained_name.endswith("e"):
                stem_str = parts.retained_name[:-1]
                terminal_e = "e"
            else:
                stem_str = parts.retained_name
                terminal_e = ""
    else:
        stem_str = stems.stem_for(parts.parent_length)
        if parts.is_bicycle:
            x, y, z = parts.bicycle_xyz
            stem_str = f"bicyclo[{x}.{y}.{z}]" + stem_str
        elif parts.is_spiro:
            x, y = parts.spiro_xy
            stem_str = f"spiro[{x}.{y}]" + stem_str
        elif parts.is_polycycle:
            if parts.polycycle_descriptor:
                stem_str = parts.polycycle_descriptor + stem_str
            else:
                v = parts.parent_length
                if v >= 4:
                    a = max(1, (v - 2) - 2)
                    b = 1
                    c = 1
                    if a + b + c + 2 != v:
                        a = v - 2 - b - c
                    stem_str = f"tricyclo[{a}.{b}.{c}.0^{{1,3}}]" + stem_str
                else:
                    stem_str = f"tricyclo[1.1.0.0^{{1,3}}]" + stem_str
        elif parts.is_ring:
            stem_str = "cyclo" + stem_str
    return stem_str, terminal_e


def _apply_replacement_prefix(stem_str: str, a_prefix_str: str) -> str:
    if a_prefix_str:
        if elision.is_vowel_start(stem_str) and a_prefix_str.endswith("a"):
            a_prefix_str = a_prefix_str[:-1]
        stem_str = a_prefix_str + stem_str
    return stem_str


def _format_unsaturations(parts: AssemblyParts, stem_str: str) -> tuple[str, str]:
    sorted_unsats = sorted(parts.unsaturations, key=lambda u: UNSATURATION_ORDER.get(u.bond_key, 99))
    unsat_parts =[]
    base_infixes =[]
    for u in sorted_unsats:
        count = len(u.locants) or 1
        infix = bonds.unsaturation_infix(u.bond_key, count)
        base_infix = infix[1:] if infix.startswith("a") else infix
        base_infixes.append((u, base_infix))
    if base_infixes and not elision.is_vowel_start(base_infixes[0][1]):
        stem_str += "a"
    for u, base_infix in base_infixes:
        if u.locants:
            loc_str = ",".join(sorted(u.locants, key=parse_locant))
            unsat_parts.append(f"-{loc_str}-{base_infix}")
        else:
            unsat_parts.append(base_infix)
    return stem_str, "".join(unsat_parts)


def _substituent_suffix_word(parts: AssemblyParts) -> str:
    if parts.is_triple_attach:
        return "ylidyne"
    if parts.is_double_attach:
        return "ylidene"
    return "yl"


def _always_print_substituent_locant(parts: AssemblyParts) -> bool:
    if parts.parent_length == 1:
        return False
    return bool(parts.is_ring and (parts.a_prefixes or (parts.retained_name and parts.retained_name != "benzene")))


def _format_substituent_tail(parts: AssemblyParts, stem_str: str, terminal_e: str) -> tuple[str, str, str, str]:
    suffix_yl = _substituent_suffix_word(parts)
    always_print_locant = _always_print_substituent_locant(parts)
    if parts.retained_name == "benzene":
        terminal_e = "yl"
    elif (str(parts.attachment_locant) != "1" or parts.unsaturations or always_print_locant) and parts.parent_length > 1:
        terminal_e = f"-{parts.attachment_locant}-{suffix_yl}"
    else:
        terminal_e = suffix_yl

    unsat_str = ""
    if not parts.retained_name and parts.unsaturations:
        stem_str, unsat_str = _format_unsaturations(parts, stem_str)
    elif not parts.retained_name and not parts.unsaturations:
        if parts.parent_length > 1 and (
            str(parts.attachment_locant) != "1"
            or parts.is_bicycle
            or parts.is_spiro
            or parts.is_polycycle
            or always_print_locant
        ):
            unsat_str = "an"
    return stem_str, unsat_str, terminal_e, ""


def _format_principal_suffix(parts: AssemblyParts, terminal_e: str, spiro_subs) -> tuple[str, str]:
    if not parts.principal_group:
        return terminal_e, ""
    group = suffixes.get(parts.principal_group.key)
    locs = sorted(parts.principal_group.locants, key=parse_locant)
    has_spiro_subs = bool(spiro_subs)
    omit_locant = parts.parent_length == 1
    if not omit_locant and len(locs) == 1 and str(locs[0]) == "1":
        if not group.suffix_with_locant:
            omit_locant = True
        elif (
            parts.is_ring
            and not parts.unsaturations
            and not parts.is_bicycle
            and not parts.is_spiro
            and not parts.is_polycycle
            and not has_spiro_subs
            and not parts.retained_name
        ):
            omit_locant = True

    if group.key in ACID_HALIDE_SUFFIX_KEYS:
        if len(locs) > 1:
            parts_suffix = group.suffix.split()
            suffix_text = (
                multipliers.basic(len(locs))
                + parts_suffix[0]
                + " "
                + multipliers.basic(len(locs))
                + parts_suffix[1]
            )
        else:
            suffix_text = group.suffix
    else:
        suffix_text = multipliers.basic(len(locs)) + group.suffix if len(locs) > 1 else group.suffix

    if elision.is_vowel_start(suffix_text):
        terminal_e = ""
    if group.suffix_with_locant and locs and not omit_locant:
        return terminal_e, f"-{','.join(map(str, locs))}-{suffix_text}"
    return terminal_e, suffix_text


def _format_parent_tail(parts: AssemblyParts, stem_str: str, terminal_e: str, spiro_subs) -> tuple[str, str, str, str]:
    unsat_str = ""
    if not parts.retained_name:
        if not parts.unsaturations:
            unsat_str = bonds.get("single").saturated_suffix
        else:
            stem_str, unsat_str = _format_unsaturations(parts, stem_str)
    terminal_e, suffix_str = _format_principal_suffix(parts, terminal_e, spiro_subs)
    return stem_str, unsat_str, terminal_e, suffix_str


def _format_spiro_core(stem_str: str, unsat_str: str, terminal_e: str, spiro_subs) -> tuple[str, str]:
    if not spiro_subs:
        return stem_str + unsat_str + terminal_e, terminal_e
    core_name = stem_str + unsat_str + "e"
    for p_loc, s_loc, s_name in spiro_subs:
        if "-" in s_name or re.search(r"\d", s_name):
            s_name_str = f"({s_name})"
        else:
            s_name_str = s_name
        core_name = f"spiro[{core_name}-{p_loc},{s_loc}'-{s_name_str}]"

    if terminal_e and terminal_e != "e":
        if ("yl" in terminal_e or elision.is_vowel_start(terminal_e.lstrip("-0123456789,"))) and core_name.endswith("e"):
            core_name = core_name[:-1]
        core_name += terminal_e
    return core_name, ""


def _add_indicated_hydrogen_prefix(parts: AssemblyParts, core_name: str) -> str:
    if not parts.indicated_hydrogens:
        return core_name
    ih_str = ",".join(sorted(parts.indicated_hydrogens, key=parse_locant)) + "H-"
    return ih_str + core_name


def _add_stereo_prefix(parts: AssemblyParts, final_word: str) -> str:
    if not parts.stereo_features:
        return final_word
    unique_stereo =[]
    seen = set()
    for feature in parts.stereo_features:
        if feature not in seen:
            seen.add(feature)
            unique_stereo.append(feature)
    sorted_stereo = sorted(unique_stereo, key=lambda f: parse_locant(f[0]))
    stereo_str = "(" + ",".join(f"{loc}{st}" for loc, st in sorted_stereo) + ")-"
    return stereo_str + final_word


def _add_front_modifiers(parts: AssemblyParts, final_word: str) -> str:
    if not parts.front_modifiers:
        return final_word
    counts = {}
    for mod in parts.front_modifiers:
        counts[mod] = counts.get(mod, 0) + 1
    front_words =[multipliers.basic(c) + m if c > 1 else m for m, c in sorted(counts.items())]
    return f"{' '.join(front_words)} {final_word}"


def assemble_name(parts: AssemblyParts) -> str:
    spiro_subs = _split_spiro_substituents(parts)
    prefix_str = _format_substituent_prefixes(parts, spiro_subs)
    a_prefix_str = _format_replacement_prefixes(parts)
    _promote_benzene_retained_name(parts)
    stem_str, terminal_e = _parent_stem_and_terminal(parts)
    stem_str = _apply_replacement_prefix(stem_str, a_prefix_str)
    if parts.is_substituent:
        stem_str, unsat_str, terminal_e, suffix_str = _format_substituent_tail(parts, stem_str, terminal_e)
    else:
        stem_str, unsat_str, terminal_e, suffix_str = _format_parent_tail(parts, stem_str, terminal_e, spiro_subs)

    core_name, terminal_e = _format_spiro_core(stem_str, unsat_str, terminal_e, spiro_subs)
    core_name = _add_indicated_hydrogen_prefix(parts, core_name)
    core_name += suffix_str
    final_word = (
        prefix_str + "-" + core_name if prefix_str and needs_hyphen(prefix_str, core_name) else prefix_str + core_name
    )
    final_word = _add_stereo_prefix(parts, final_word)
    final_word = _add_front_modifiers(parts, final_word)
    return _post_process_name(final_word)
