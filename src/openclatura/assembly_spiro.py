"""Spiro-specific assembly formatting."""

import re
from dataclasses import dataclass

from .assembly_parts import AssemblyParts, SubstituentItem
from .formatting import strip_outer_parentheses
from .nomenclature import RULES
from .rules import elision, multipliers, stems
from .spiro_assembly import SpiroAssembly

SPIRO_SUBSTITUENT_RE = re.compile(r"^\[SPIRO\]-(\d+)-(.*)$")
AMBIGUOUS_CONNECTION_SUBSTITUENT_STEMS = RULES.assembly.ambiguous_connection_substituent_stems


_SIDE_STEREO_RE = re.compile(r"^\((?P<body>\d+[A-Za-z']*(?:[RS]|[EZ])(?:,\d+[A-Za-z']*(?:[RS]|[EZ]))*)\)-?")
_SIDE_STEREO_TERM_RE = re.compile(r"^(?P<locant>\d+[a-z']*)(?P<descriptor>[RSEZ])$")


def split_spiro_substituents(parts: AssemblyParts) -> list[SpiroAssembly]:
    spiro_subs = []
    normal_subs = []
    for sub in parts.substituents:
        if sub.spiro is not None:
            spiro_subs.append(_normalize_spiro_assembly(sub.spiro))
            continue
        match = SPIRO_SUBSTITUENT_RE.match(sub.name)
        if match:
            side_prefixes, side_parent_name, side_suffixes, side_stereo = extract_spiro_side_prefixes(match.group(2))
            spiro_subs.append(
                _normalize_spiro_assembly(
                    SpiroAssembly(
                        parent_locant=str(sub.locants[0]),
                        side_locant=match.group(1),
                        side_parent_name=side_parent_name,
                        side_prefixes=tuple(side_prefixes),
                        side_suffixes=tuple(side_suffixes),
                        side_stereo=side_stereo,
                    )
                )
            )
        else:
            normal_subs.append(sub)
    parts.substituents = normal_subs
    spiro_subs = [_hoist_side_substituent_prefixes(parts, spiro) for spiro in spiro_subs]
    # The side component's descriptors belong in the whole name's leading
    # stereo group; only the assembler can put them there.
    for spiro in spiro_subs:
        # The spiro atom itself is cited once, at its parent locant; the side
        # component's own descriptor for it would name the same centre twice.
        junction = f"{spiro.side_locant}'"
        for feature in spiro.side_stereo:
            if feature[0] != junction and feature not in parts.stereo_features:
                parts.stereo_features.append(feature)
    return spiro_subs


_SIDE_LOCANTS_RE = re.compile(r"^[0-9]+[a-z]*'+(?:,[0-9]+[a-z]*'+)*$")
_MULTIPLIER_PREFIXES = ("di", "tri", "tetra", "penta", "hexa")


def _is_replacement_prefix(name: str) -> bool:
    """Whether a side prefix names a skeletal replacement rather than a group."""

    stem = name
    for multiplier in _MULTIPLIER_PREFIXES:
        if stem.startswith(multiplier):
            stem = stem[len(multiplier) :]
            break
    return stem in RULES.assembly.replacement_prefix_order


def _split_side_prefix_run(text: str) -> list[tuple[str, str]]:
    """Split a rendered run of primed prefixes into (locants, name) pairs.

    Only a top-level locant segment starts a prefix; the ones inside a nested
    substituent number that substituent's own skeleton.
    """

    segments = text.split("-")
    starts = []
    depth = 0
    for index, segment in enumerate(segments):
        was_nested = depth > 0
        depth += segment.count("(") - segment.count(")")
        if was_nested or depth > 0 or not _SIDE_LOCANTS_RE.fullmatch(segment):
            continue
        following = segments[index + 1] if index + 1 < len(segments) else ""
        if following.startswith(_WITHIN_NAME_AFTER_LOCANT):
            continue
        starts.append(index)
    if not starts or starts[0] != 0:
        return [("", text)]
    bounds = starts + [len(segments)]
    prefixes = []
    for start, end in zip(bounds, bounds[1:]):
        prefixes.append((segments[start], "-".join(segments[start + 1 : end])))
    return prefixes


def _unmultiplied_prefix_name(name: str, count: int) -> str:
    """Strip the multiplier a rendered prefix already carries for its locants.

    The prefix group multiplies the name again from the locant count, so
    leaving ``dimethyl`` in place renders ``bis(dimethyl)``.
    """

    if count > 1:
        basic = multipliers.basic(count)
        complex_ = multipliers.complex_(count)
        if name.startswith(basic):
            return strip_outer_parentheses(name[len(basic) :])
        if name.startswith(complex_):
            return strip_outer_parentheses(name[len(complex_) :])
    return strip_outer_parentheses(name)


def _hoist_side_substituent_prefixes(parts: AssemblyParts, spiro: SpiroAssembly) -> SpiroAssembly:
    """Move the side component's detachable prefixes into the prefix group.

    All detachable prefixes of a spiro name are cited together, so a side-ring
    methyl groups with the parent ring's -- ``1,1'-dimethyl``, not
    ``1-methyl-1'-methyl``.  Replacement prefixes name the ring and stay put.
    """

    kept = []
    hoisted = False
    for prefix in spiro.side_prefixes:
        for locants, name in _split_side_prefix_run(prefix):
            if not locants or _is_replacement_prefix(name):
                kept.append(f"{locants}-{name}" if locants else name)
                continue
            locant_list = locants.split(",")
            parts.substituents.append(
                SubstituentItem(name=_unmultiplied_prefix_name(name, len(locant_list)), locants=locant_list)
            )
            hoisted = True
    if not hoisted:
        return spiro
    return SpiroAssembly(
        parent_locant=spiro.parent_locant,
        side_locant=spiro.side_locant,
        side_parent_name=spiro.side_parent_name,
        side_prefixes=tuple(kept),
        side_suffixes=spiro.side_suffixes,
        side_stereo=spiro.side_stereo,
    )


def _normalize_spiro_assembly(spiro: SpiroAssembly) -> SpiroAssembly:
    """Extract side-component prefixes/suffixes before spiro rendering."""

    side_prefixes, side_parent_name, side_suffixes, side_stereo = extract_spiro_side_prefixes(spiro.side_parent_name)
    if not side_prefixes and side_parent_name == spiro.side_parent_name and not side_suffixes and not side_stereo:
        return spiro
    return SpiroAssembly(
        parent_locant=spiro.parent_locant,
        side_locant=spiro.side_locant,
        side_parent_name=side_parent_name,
        side_prefixes=tuple(spiro.side_prefixes) + tuple(side_prefixes),
        side_suffixes=tuple(spiro.side_suffixes) + tuple(side_suffixes),
        side_stereo=tuple(spiro.side_stereo) + tuple(side_stereo),
    )


def format_spiro_core(
    stem_str: str, unsat_str: str, terminal_e: str, spiro_subs: list[SpiroAssembly], suffix_str: str = ""
) -> tuple[str, str, str]:
    if not spiro_subs:
        return stem_str + unsat_str + terminal_e, terminal_e, suffix_str
    core_name = stem_str + unsat_str + ("" if stem_str.endswith("ium") else "e")
    if len(spiro_subs) == 2 and not core_name.startswith("spiro["):
        return _format_dispiro_core(core_name, terminal_e, spiro_subs), "", suffix_str
    side_prefixes = []
    side_suffixes = []
    for spiro in spiro_subs:
        s_name = spiro.side_parent_name
        side_prefixes.extend(spiro.side_prefixes)
        side_suffixes.extend(_prime_side_suffixes(spiro.side_suffixes, "'"))
        extracted_prefixes, extracted_parent, extracted_suffixes, _extracted_stereo = extract_spiro_side_prefixes(
            s_name
        )
        if extracted_prefixes or extracted_parent != s_name or extracted_suffixes:
            side_prefixes.extend(extracted_prefixes)
            side_suffixes.extend(_prime_side_suffixes(extracted_suffixes, "'"))
            s_name = extracted_parent
            spiro = SpiroAssembly(
                parent_locant=spiro.parent_locant,
                side_locant=spiro.side_locant,
                side_parent_name=s_name,
                side_prefixes=spiro.side_prefixes,
                side_suffixes=spiro.side_suffixes,
                side_stereo=spiro.side_stereo,
            )
        if core_name.startswith("spiro["):
            continue
        if _spiro_side_parent_needs_parentheses(s_name):
            s_name_str = f"({s_name})"
        else:
            s_name_str = s_name
        core_name = f"spiro[{core_name}-{spiro.parent_locant},{_spiro_side_locant(spiro)}'-{s_name_str}]"

    if terminal_e and terminal_e != "e":
        if ("yl" in terminal_e or elision.is_vowel_start(terminal_e.lstrip("-0123456789,"))) and core_name.endswith(
            "e"
        ):
            core_name = core_name[:-1]
        core_name += _merge_terminal_and_side_suffixes(terminal_e, side_suffixes)
    elif side_suffixes:
        # One suffix set names the whole spiro system, so a side ``-2'-one``
        # beside the parent's ``-2-one`` is cited once as ``-2,2'-dione``.
        suffix_str, side_suffixes = _merge_principal_and_side_suffixes(suffix_str, side_suffixes)
        core_name += _format_side_suffixes(side_suffixes)
    if side_prefixes:
        side_prefixes = _prime_replacement_prefixes_for_primed_component(core_name, side_prefixes)
        core_name = "-".join(side_prefixes) + core_name
    core_name = _prime_inline_replacement_prefixes_for_primed_component(core_name)
    return core_name, "", suffix_str


def _format_dispiro_core(core_name: str, terminal_e: str, spiro_subs: list[SpiroAssembly]) -> str:
    first, second = sorted(spiro_subs, key=lambda spiro: (int(spiro.parent_locant), spiro.side_parent_name))
    first_side = _spiro_side_name(first.side_parent_name)
    second_side = _spiro_side_name(second.side_parent_name)
    core = (
        f"dispiro[{first_side}-{_spiro_side_locant(first)},{first.parent_locant}'-"
        f"{core_name}-{second.parent_locant}',{_spiro_side_locant(second)}''-{second_side}]"
    )
    side_prefixes = []
    side_suffixes = []
    side_prefixes.extend(_reprime_side_prefixes(first.side_prefixes, "'"))
    side_prefixes.extend(_reprime_side_prefixes(second.side_prefixes, "''"))
    side_suffixes.extend(_prime_side_suffixes(first.side_suffixes, "'"))
    side_suffixes.extend(_prime_side_suffixes(second.side_suffixes, "''"))
    if terminal_e and terminal_e != "e":
        if ("yl" in terminal_e or elision.is_vowel_start(terminal_e.lstrip("-0123456789,"))) and core.endswith("e"):
            core = core[:-1]
        core += _merge_terminal_and_side_suffixes(terminal_e, side_suffixes)
    elif side_suffixes:
        core += _format_side_suffixes(side_suffixes)
    if side_prefixes:
        side_prefixes = _prime_replacement_prefixes_for_primed_component(core, side_prefixes)
        core = "-".join(side_prefixes) + core
    return core


def _spiro_side_name(side_parent_name: str) -> str:
    if _spiro_side_parent_needs_parentheses(side_parent_name):
        return f"({side_parent_name})"
    return side_parent_name


def _spiro_side_locant(spiro: SpiroAssembly) -> str:
    """Return the displayed side-component spiro locant."""

    retained = _retained_saturated_n_ring_info(spiro.side_parent_name)
    if spiro.side_locant == retained.n_locant and retained.ring_size:
        return str(retained.opposite_carbon_locant)
    return spiro.side_locant


def _reprime_side_prefixes(prefixes: tuple[str, ...], prime: str) -> list[str]:
    if prime == "'":
        return list(prefixes)
    return [prefix.replace("'", prime) for prefix in prefixes]


def _prime_replacement_prefixes_for_primed_component(core_name: str, prefixes: list[str]) -> list[str]:
    """Prime replacement prefixes that describe the side ring.

    The side component is always the primed one, and these prefixes are always
    the side ring's, so the priming does not depend on which ring it happens to
    be.  Guarding on ``spiro[cyclopropane-`` meant every other side ring kept
    unprimed locants: spiro[indoline-3,4'-cyclopentane] carrying two ring
    nitrogens came out as ``1,3-diaza`` and read back with the nitrogens on the
    indoline instead.
    """

    if not _spiro_names_its_components(core_name):
        return prefixes
    return [
        re.sub(
            r"^([0-9,]+)-((?:di|tri|tetra|penta)?(?:oxa|aza|thia|selena|tellura|phospha|sila|bora|germa|stanna|magnesa|calca|litha|natra|potassa))$",
            lambda match: f"{','.join(f'{locant}' + chr(39) for locant in match.group(1).split(','))}-{match.group(2)}",
            prefix,
        )
        for prefix in prefixes
    ]


def _prime_inline_replacement_prefixes_for_primed_component(core_name: str) -> str:
    """
    Prime a replacement prefix that ended up in front of the spiro core.
    """

    def prime(match: re.Match) -> str:
        if "'" not in match.group(4):
            return match.group(0)
        locants = ",".join(f"{locant}'" for locant in match.group(2).split(","))
        return f"{match.group(1)}{locants}-{match.group(3)}spiro[{match.group(4)}"

    return re.sub(
        r"(^|-)([0-9,]+)-((?:di|tri|tetra|penta)?(?:oxa|aza|thia|selena|tellura|phospha|sila|bora|germa|stanna|magnesa|calca|litha|natra|potassa))"
        r"spiro\[([^\]]*)",
        prime,
        core_name,
    )


def _spiro_names_its_components(core_name: str) -> bool:
    """
    Whether the spiro descriptor names its rings rather than counting them.
    """

    return bool(re.search(r"spiro\[[^\]]*'", core_name))


def _prime_side_suffixes(suffixes: tuple[tuple[str, str], ...], prime: str) -> list[tuple[str, str]]:
    return [(f"{locant}{prime}", suffix) for locant, suffix in suffixes]


def _merge_terminal_and_side_suffixes(terminal_e: str, side_suffixes: list[tuple[str, str]]) -> str:
    if not side_suffixes:
        return terminal_e
    match = re.fullmatch(r"-([0-9,']+)-(ol|one)", terminal_e)
    if not match:
        return terminal_e + _format_side_suffixes(side_suffixes)
    main_locants = match.group(1).split(",")
    main_suffix = match.group(2)
    same_suffix = [(locant, suffix) for locant, suffix in side_suffixes if suffix == main_suffix]
    other_suffix = [(locant, suffix) for locant, suffix in side_suffixes if suffix != main_suffix]
    if not same_suffix:
        return terminal_e + _format_side_suffixes(side_suffixes)
    locants = main_locants + [locant for locant, _ in same_suffix]
    multiplier = multipliers.basic(len(locants)) if len(locants) > 1 else ""
    merged_suffix = f"{multiplier}{main_suffix}" if multiplier else main_suffix
    return f"-{','.join(locants)}-{merged_suffix}" + _format_side_suffixes(other_suffix)


def _merge_principal_and_side_suffixes(
    suffix_str: str, side_suffixes: list[tuple[str, str]]
) -> tuple[str, list[tuple[str, str]]]:
    """
    Fold side-component suffixes into the parent's suffix of the same kind.
    """

    match = re.fullmatch(r"-([0-9,']+)-(?:di|tri|tetra)?(ol|one)", suffix_str)
    if not match:
        return suffix_str, side_suffixes
    word = match.group(2)
    same = [(locant, suffix) for locant, suffix in side_suffixes if suffix == word]
    if not same:
        return suffix_str, side_suffixes
    other = [(locant, suffix) for locant, suffix in side_suffixes if suffix != word]
    locants = match.group(1).split(",") + [locant for locant, _ in same]
    multiplier = multipliers.basic(len(locants)) if len(locants) > 1 else ""
    return f"-{','.join(locants)}-{multiplier}{word}", other


def _format_side_suffixes(side_suffixes: list[tuple[str, str]]) -> str:
    return "".join(f"-{locant}-{suffix}" for locant, suffix in side_suffixes)


def extract_spiro_side_prefixes(
    side_name: str,
) -> tuple[list[str], str, tuple[tuple[str, str], ...], tuple[tuple[str, str], ...]]:
    """Move simple side-ring substituents to primed spiro prefixes.

    Returns the primed prefixes, the side parent name, its suffixes, and the
    stereo features the side component contributes to the whole name.
    """

    side_name = side_name.strip()
    if side_name.startswith("(") and side_name.endswith(")"):
        side_name = side_name[1:-1]
    parent_aliases = (
        *_retained_n_ring_replacement_aliases(),
        ("1-azacyclopropane", "aziridine"),
        ("1-oxacyclopropane", "oxirane"),
        ("1-thiacyclopropane", "thiirane"),
    )
    parent_names = (
        *_retained_n_ring_parent_names(),
        "indoline",
        "indane",
        "1,3-benzoxazole",
        "1,3-benzothiazole",
        "benzimidazole",
        "benzene",
        "tetracyclo",
        "tricyclo",
        "bicyclo",
        *_replacement_heterocycle_parent_names(),
        "aziridine",
        "oxirane",
        "thiirane",
        *_cycloalkane_parent_names(),
        "cyclobutane",
        "cyclopropane",
    )
    normalized = side_name.replace("-aziridine", "aziridine")
    normalized, side_suffixes = _extract_side_suffixes(normalized)
    parent = None
    prefix_text = ""
    for alias, retained_parent in parent_aliases:
        if normalized.endswith(alias):
            parent = retained_parent
            prefix_text = normalized[: -len(alias)].rstrip("-")
            break
    if parent is None:
        parent = _extract_polycyclic_parent(normalized)
        if parent is not None and normalized.endswith(parent):
            prefix_text = normalized[: -len(parent)].rstrip("-")
    if parent is None:
        parent = next((candidate for candidate in parent_names if normalized.endswith(candidate)), None)
        if parent is None:
            return [], normalized, tuple(side_suffixes), ()
        prefix_text = normalized[: -len(parent)].rstrip("-")
    elif normalized != parent and normalized.endswith(parent):
        prefix_text = normalized[: -len(parent)].rstrip("-")

    prefix_text, side_stereo = _extract_side_stereo(prefix_text)
    if not prefix_text:
        return [], _spiro_side_parent_name(parent), tuple(side_suffixes), side_stereo
    return (
        [_prime_side_prefix_locants(prefix_text)],
        _spiro_side_parent_name(parent),
        tuple(side_suffixes),
        side_stereo,
    )


def _extract_side_stereo(prefix_text: str) -> tuple[str, tuple[tuple[str, str], ...]]:
    """Split a leading side-component stereo descriptor off its prefixes."""

    match = _SIDE_STEREO_RE.match(prefix_text)
    if match is None:
        return prefix_text, ()
    features = []
    for term in match.group("body").split(","):
        term_match = _SIDE_STEREO_TERM_RE.match(term)
        if term_match is None:
            # An unrecognised term would be dropped silently otherwise; keeping
            # the descriptor out of the name is better than misplacing it.
            return prefix_text[match.end() :], ()
        locant = term_match.group("locant")
        features.append((f"{locant}'" if not locant.endswith("'") else locant, term_match.group("descriptor")))
    return prefix_text[match.end() :], tuple(features)


_WITHIN_NAME_AFTER_LOCANT = ("en", "yn", "yl", "ylidene", "ylidyne", "ol", "one", "al", "amine", "oic", "carbo")


def _prime_side_prefix_locants(prefix_text: str) -> str:
    """Prime every locant that introduces a side-ring prefix, and only those.

    The side ring is the primed component, so each of its substituent locants
    carries a prime.  Priming only the first left the rest reading as the other
    ring's positions.
    """

    segments = prefix_text.split("-")
    depth = 0
    for index, segment in enumerate(segments):
        # Only locants at the top level position a prefix on the side ring.
        # Inside a nested substituent -- `(2,6-difluoro-3-methylphenyl)` -- they
        # number that substituent's own skeleton and must be left alone.
        opening, closing = segment.count("("), segment.count(")")
        was_nested = depth > 0
        depth += opening - closing
        if was_nested or depth > 0 or not re.fullmatch(r"[0-9,]+", segment):
            continue
        following = segments[index + 1] if index + 1 < len(segments) else ""
        if following.startswith(_WITHIN_NAME_AFTER_LOCANT):
            continue
        segments[index] = ",".join(f"{locant}'" for locant in segment.split(","))
    return "-".join(segments)


def _extract_side_suffixes(side_name: str) -> tuple[str, list[tuple[str, str]]]:
    match = re.fullmatch(r"(.+?)an-([0-9,]+)-(ol|one)", side_name)
    if not match:
        match = re.fullmatch(r"(.+?)(?:en|yn)-([0-9,]+)-(ol|one)", side_name)
    if not match:
        return side_name, []
    stem, locants, suffix = match.groups()
    if side_name.startswith(stem + "an-"):
        parent = stem + "ane"
    elif side_name.startswith(stem + "en-"):
        parent = stem + "ene"
    elif side_name.startswith(stem + "yn-"):
        parent = stem + "yne"
    else:
        parent = stem
    return parent, [(locant, suffix) for locant in locants.split(",")]


def _spiro_side_parent_name(parent: str) -> str:
    """Render retained ionic side parents in an explicit locanted form."""

    return _retained_saturated_n_ring_info(parent).explicit_parent or parent


def _retained_n_ring_parent_names() -> tuple[str, ...]:
    """Return retained saturated N-ring parent names known to the charge registry."""

    names = set(RULES.charges.retained_ionic_n_parents)
    names.update(RULES.charges.retained_ionic_n_parents.values())
    return tuple(sorted(names, key=len, reverse=True))


def _retained_n_ring_replacement_aliases() -> tuple[tuple[str, str], ...]:
    """Return replacement-parent aliases for retained saturated N-rings."""

    aliases = []
    neutral_by_ionic = {ionic: neutral for neutral, ionic in RULES.charges.retained_ionic_n_parents.items()}
    for ring_size, ionic_name in RULES.charges.saturated_n_ring_ionic_parents.items():
        neutral = neutral_by_ionic.get(ionic_name)
        if not neutral:
            continue
        aliases.append((f"1-azacyclo{stems.stem_for(ring_size)}ane", neutral))
    return tuple(sorted(aliases, key=lambda item: len(item[0]), reverse=True))


def _replacement_heterocycle_parent_names() -> tuple[str, ...]:
    """Return simple replacement heterocycle parent names supported here."""

    names = []
    for ring_size in RULES.charges.saturated_n_ring_ionic_parents:
        stem = stems.stem_for(ring_size)
        names.extend((f"1-azacyclo{stem}ane", f"1-oxacyclo{stem}ane", f"1-thiacyclo{stem}ane"))
    return tuple(sorted(names, key=len, reverse=True))


def _cycloalkane_parent_names() -> tuple[str, ...]:
    """Return simple cycloalkane side parents supported by stem data."""

    return tuple(f"cyclo{stems.stem_for(size)}ane" for size in range(3, 9))


def _retained_saturated_n_ring_info(parent: str) -> "_RetainedSaturatedNRingInfo":
    ionic_by_neutral = RULES.charges.retained_ionic_n_parents
    neutral_by_ionic = {ionic: neutral for neutral, ionic in ionic_by_neutral.items()}
    neutral = neutral_by_ionic.get(parent, parent if parent in ionic_by_neutral else "")
    if not neutral:
        neutral = next(
            (
                retained_parent
                for retained_parent in ionic_by_neutral
                if parent == f"{retained_parent[:-1] if retained_parent.endswith('e') else retained_parent}-1-ium"
            ),
            "",
        )
    ionic = ionic_by_neutral.get(neutral)
    if not neutral or not ionic:
        return _RetainedSaturatedNRingInfo()
    ring_size = next(
        (size for size, name in RULES.charges.saturated_n_ring_ionic_parents.items() if name == ionic),
        0,
    )
    stem = neutral[:-1] if neutral.endswith("e") else neutral
    return _RetainedSaturatedNRingInfo(
        ring_size=ring_size,
        n_locant="1",
        explicit_parent=f"{stem}-1-ium" if parent == ionic else parent,
    )


@dataclass(frozen=True)
class _RetainedSaturatedNRingInfo:
    """Derived display metadata for retained saturated N-ring cations."""

    ring_size: int = 0
    n_locant: str = ""
    explicit_parent: str = ""

    @property
    def opposite_carbon_locant(self) -> int:
        return self.ring_size // 2 + 1 if self.ring_size else 0


def _extract_polycyclic_parent(name: str) -> str | None:
    for marker in ("tetracyclo[", "tricyclo[", "bicyclo["):
        idx = name.rfind(marker)
        if idx > 0 and name[idx - 1] == "-":
            return name[idx:]
        if idx > 0 and re.fullmatch(
            r"(?:[0-9,]+-)?(?:oxa|aza|thia|selena|tellura|phospha|sila|bora|germa|stanna|magnesa|calca|litha|natra|potassa)",
            name[:idx],
        ):
            return name[idx:]
    return None


def _spiro_side_parent_needs_parentheses(side_parent_name: str) -> bool:
    if re.search(r"(?:^|\d+-[a-z]+)(?:bi|tri|tetra)cyclo\[", side_parent_name):
        return False
    return "-" in side_parent_name or bool(re.search(r"\d", side_parent_name))
