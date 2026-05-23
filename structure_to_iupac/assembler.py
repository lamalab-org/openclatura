# structure-to-iupac/assembler.py

import re
from .assembly_parts import AssemblyParts, ParentChargeItem, PrincipalGroupItem, SubstituentItem, UnsaturationItem
from .assembly_charge import (
    positive_parent_n_charges,
)
from .assembly_utils import needs_hyphen, parse_locant
from .assembly_spiro import format_spiro_core, split_spiro_substituents
from .assembly_prefixes import format_replacement_prefixes, format_substituent_prefixes
from .assembly_parent import (
    apply_replacement_prefix,
    format_parent_tail,
    format_substituent_tail,
    parent_stem_and_terminal,
    promote_benzene_retained_name,
)
from .formatting import ensure_stereo_descriptor_boundary
from .name_postprocessing import (
    apply_acyl_amido_postprocessing,
    apply_connection_boundary_postprocessing,
    apply_data_postprocessing,
)
from .nomenclature import RULES
from .rules import multipliers


def _post_process_name(name: str) -> str:
    name = apply_data_postprocessing(name)
    name = ensure_stereo_descriptor_boundary(name)
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
    name = apply_connection_boundary_postprocessing(name)

    return name


def _add_indicated_hydrogen_prefix(parts: AssemblyParts, core_name: str) -> str:
    indicated_hydrogens = [
        locant
        for operation in parts.hydro_operations
        if operation.operation_kind == "indicated_hydrogen"
        for locant in operation.locants
    ] or parts.indicated_hydrogens
    if not indicated_hydrogens:
        return core_name
    if positive_parent_n_charges(parts):
        return core_name
    ih_str = ",".join(sorted(indicated_hydrogens, key=parse_locant)) + "H-"
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


def post_process_name(name: str) -> str:
    return _post_process_name(name)


def assemble_name_raw(parts: AssemblyParts) -> str:
    spiro_subs = split_spiro_substituents(parts)
    prefix_str = format_substituent_prefixes(parts, spiro_subs)
    a_prefix_str = format_replacement_prefixes(parts)
    promote_benzene_retained_name(parts)
    stem_str, terminal_e = parent_stem_and_terminal(parts)
    stem_str = apply_replacement_prefix(stem_str, a_prefix_str)
    if parts.is_substituent:
        stem_str, unsat_str, terminal_e, suffix_str = format_substituent_tail(
            parts, stem_str, terminal_e, spiro_subs
        )
    else:
        stem_str, unsat_str, terminal_e, suffix_str = format_parent_tail(parts, stem_str, terminal_e, spiro_subs)

    core_name, terminal_e = format_spiro_core(stem_str, unsat_str, terminal_e, spiro_subs)
    core_name = _add_indicated_hydrogen_prefix(parts, core_name)
    core_name += suffix_str
    parent_needs_prefix_hyphen = bool(
        prefix_str and positive_parent_n_charges(parts) and parts.retained_name and parts.indicated_hydrogens
    )
    final_word = (
        prefix_str + "-" + core_name
        if prefix_str and (needs_hyphen(prefix_str, core_name) or parent_needs_prefix_hyphen)
        else prefix_str + core_name
    )
    final_word = _add_stereo_prefix(parts, final_word)
    final_word = _add_front_modifiers(parts, final_word)
    return final_word


def assemble_name(parts: AssemblyParts) -> str:
    return post_process_name(assemble_name_raw(parts))
