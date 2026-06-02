"""Name-term to graph-atom binding helpers."""

from .assembly_parts import AssemblyParts, NameAtomBinding
from .nomenclature import RULES
from .rules import bonds


def refresh_name_atom_bindings(parts: AssemblyParts) -> list[NameAtomBinding]:
    """Populate structured bindings for the current assembly parts."""

    bindings: list[NameAtomBinding] = []
    if parts.parent_atom_ids:
        bindings.append(
            NameAtomBinding(
                stage="parent",
                role="parent",
                term=parts.retained_name or _parent_term(parts),
                atom_ids=set(parts.parent_atom_ids),
                bond_ids=set(parts.parent_bond_ids),
            )
        )
    if parts.front_modifiers:
        bindings.append(
            NameAtomBinding(
                stage="modifier",
                role="front_modifier",
                term=" ".join(parts.front_modifiers),
                atom_ids=set(parts.front_modifier_atom_ids),
                charge_atom_ids=set(parts.front_modifier_charge_atom_ids),
            )
        )
    for item in parts.principal_suffix_modifiers:
        bindings.append(
            NameAtomBinding(
                stage="modifier",
                role="principal_suffix_modifier",
                term=item.name,
                atom_ids=set(item.atom_ids),
                bond_ids=set(item.bond_ids),
            )
        )
    for item in parts.a_prefixes:
        bindings.append(
            NameAtomBinding(
                stage="replacement",
                role="replacement_prefix",
                term=item.name,
                atom_ids=set(item.atom_ids),
                bond_ids=set(item.bond_ids),
                charge_atom_ids=_replacement_charge_atom_ids(parts, item),
                locants=tuple(str(locant) for locant in item.locants),
            )
        )
    for item in parts.substituents:
        role = "spiro_substituent" if item.spiro is not None else "substituent"
        term = item.spiro.side_parent_name if item.spiro is not None else item.name
        bindings.append(
            NameAtomBinding(
                stage="prefix",
                role=role,
                term=term,
                atom_ids=set(item.atom_ids),
                bond_ids=set(item.bond_ids),
                charge_atom_ids=set(item.charge_atom_ids)
                or _exact_charge_renderer_atom_ids(item.name, set(item.atom_ids)),
                locants=tuple(str(locant) for locant in item.locants),
            )
        )
    for item in parts.unsaturations:
        bindings.append(
            NameAtomBinding(
                stage="unsaturation",
                role=item.bond_key,
                term=bonds.get(item.bond_key).suffix,
                atom_ids=set(item.atom_ids),
                bond_ids=set(item.bond_ids),
                locants=tuple(str(locant) for locant in item.locants),
            )
        )
    if parts.principal_group is not None:
        bindings.append(
            NameAtomBinding(
                stage="suffix",
                role=parts.principal_group.key,
                term=parts.principal_group.key,
                atom_ids=set(parts.principal_group.atom_ids),
                bond_ids=set(parts.principal_group.bond_ids),
                charge_atom_ids=set(parts.principal_group.charge_atom_ids)
                or _exact_charge_renderer_atom_ids(
                    parts.principal_group.key, set(parts.principal_group.atom_ids)
                ),
                locants=tuple(str(locant) for locant in parts.principal_group.locants),
            )
        )
    for charge in parts.parent_charges:
        if charge.atom_id is None:
            continue
        bindings.append(
            NameAtomBinding(
                stage="charge",
                role="parent_charge",
                term=f"{charge.symbol}{charge.charge:+d}",
                atom_ids={charge.atom_id},
                charge_atom_ids={charge.atom_id},
                locants=(str(charge.locant),),
            )
        )
    parts.name_atom_bindings = bindings
    return bindings


def binding_trace_data(bindings: list[NameAtomBinding]) -> list[dict]:
    """Return JSON-friendly binding data for decision traces."""

    return [
        {
            "stage": binding.stage,
            "role": binding.role,
            "term": binding.term,
            "locants": list(binding.locants),
            "atoms": sorted(binding.atom_ids),
            "bonds": sorted(binding.bond_ids),
            "charge_atoms": sorted(binding.charge_atom_ids),
        }
        for binding in bindings
    ]


def postprocess_name_atom_bindings(
    bindings: list[NameAtomBinding],
    postprocess_term,
    final_name: str | None = None,
) -> list[NameAtomBinding]:
    """Apply final name post-processing to binding terms."""

    final_text = _normalise_name_text(final_name or "")
    processed = [
        NameAtomBinding(
            stage=binding.stage,
            role=binding.role,
            term=_contextual_postprocessed_binding_term(
                binding.term,
                postprocess_term(binding.term),
                final_text,
            ),
            atom_ids=set(binding.atom_ids),
            bond_ids=set(binding.bond_ids),
            charge_atom_ids=set(binding.charge_atom_ids),
            locants=tuple(binding.locants),
        )
        for binding in bindings
    ]
    return processed


def _contextual_postprocessed_binding_term(original_term: str, rewritten_term: str, final_text: str) -> str:
    """Map terms absorbed by contextual final-name post-processing."""

    original = _normalise_name_text(original_term)
    rewritten = _normalise_name_text(rewritten_term)
    for before, after in _contextual_postprocess_replacements():
        normalised_before = _normalise_name_text(before)
        normalised_after = _normalise_name_text(after)
        if normalised_after in final_text and (
            original in normalised_before or rewritten in normalised_before or rewritten == normalised_after
        ):
            return after
    return rewritten_term


def _contextual_postprocess_replacements() -> tuple[tuple[str, str], ...]:
    from .assembler import LEGACY_POSTPROCESS_LITERAL_REPLACEMENTS

    return (
        tuple((rule.pattern, rule.replacement) for rule in RULES.postprocess.literal_replacements)
        + tuple(LEGACY_POSTPROCESS_LITERAL_REPLACEMENTS)
    )


def _normalise_name_text(text: str) -> str:
    return text.lower().replace(" ", "").replace("(", "").replace(")", "")


def _replacement_charge_atom_ids(parts: AssemblyParts, item) -> set[int]:
    """Return charged replacement-prefix atoms represented by this binding."""

    charged_locants = {
        str(locant)
        for locant in item.locants
        if parts.parent_atom_charges_by_locant.get(str(locant), 0) != 0
    }
    if not charged_locants:
        return set()
    return {
        atom_id
        for locant, atom_id in parts.parent_atom_ids_by_locant.items()
        if str(locant) in charged_locants and atom_id in item.atom_ids
    } or set(item.atom_ids)


def _exact_charge_renderer_atom_ids(key: str, atom_ids: set[int]) -> set[int]:
    """Return atoms represented by an exact charge-bearing renderer key."""

    return set(atom_ids) if key in _EXACT_CHARGE_RENDERER_KEYS else set()


_EXACT_CHARGE_RENDERER_KEYS = frozenset(
    {
        "aminium",
        "ammonio",
        "azido",
        "azonia",
        "azanidyl",
        "carboxylate",
        "chlorophosphoryloxy",
        "diazo",
        "imino",
        "iminio",
        "isocyano",
        "nitrile_oxide",
        "nitro",
        "olate",
        "oxide",
        "oxido",
        "peroxy",
        "phosphoryl",
        "ring_nitrile_oxide",
        "selenido",
        "thiolate",
    }
)


def _parent_term(parts: AssemblyParts) -> str:
    if parts.is_spiro:
        return "spiro parent"
    if parts.is_bicycle:
        return "bicyclo parent"
    if parts.is_polycycle:
        return parts.polycycle_descriptor or "polycyclic parent"
    if parts.is_ring:
        return "cyclic parent"
    return "chain parent"
