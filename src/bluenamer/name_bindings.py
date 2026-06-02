"""Name-term to graph-atom binding helpers."""

import re

from .assembly_parts import AssemblyParts, NameAtomBinding, NameTokenBinding
from .nomenclature import RULES
from .rules import bonds, stems


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
                emitted_tokens=_parent_emitted_tokens(parts),
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
                emitted_tokens=tuple(item.emitted_tokens),
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
    bindings = [_with_default_emitted_tokens(binding) for binding in bindings]
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
            "emitted_tokens": [
                {
                    "text": token.text,
                    "locants": list(token.locants),
                    "atoms": sorted(token.atom_ids),
                    "bonds": sorted(token.bond_ids),
                    "charge_atoms": sorted(token.charge_atom_ids),
                }
                for token in binding.emitted_tokens
            ],
        }
        for binding in bindings
    ]


def ensure_name_atom_binding_tokens(binding: NameAtomBinding) -> NameAtomBinding:
    """Return a binding with renderer-style emitted token metadata."""

    return _with_default_emitted_tokens(binding)


def postprocess_name_atom_bindings(
    bindings: list[NameAtomBinding],
    postprocess_term,
    final_name: str | None = None,
) -> list[NameAtomBinding]:
    """Apply final name post-processing to binding terms."""

    final_text = _normalise_name_text(final_name or "")
    processed = []
    for binding in bindings:
        processed_term = _contextual_postprocessed_binding_term(
            binding.term,
            postprocess_term(binding.term),
            final_text,
        )
        processed_tokens = tuple(
            NameTokenBinding(
                text=_contextual_postprocessed_binding_term(
                    token.text,
                    postprocess_term(token.text),
                    final_text,
                ),
                atom_ids=set(token.atom_ids),
                bond_ids=set(token.bond_ids),
                charge_atom_ids=set(token.charge_atom_ids),
                locants=tuple(token.locants),
            )
            for token in binding.emitted_tokens
        )
        processed.append(
            _with_default_emitted_tokens(
                NameAtomBinding(
                    stage=binding.stage,
                    role=binding.role,
                    term=processed_term,
                    atom_ids=set(binding.atom_ids),
                    bond_ids=set(binding.bond_ids),
                    charge_atom_ids=set(binding.charge_atom_ids),
                    locants=tuple(binding.locants),
                    emitted_tokens=processed_tokens,
                )
            )
        )
    return processed


def _with_default_emitted_tokens(binding: NameAtomBinding) -> NameAtomBinding:
    """Ensure every binding carries renderer-style token metadata."""

    if binding.emitted_tokens:
        return binding
    token_bindings = _operation_emitted_tokens(binding)
    return NameAtomBinding(
        stage=binding.stage,
        role=binding.role,
        term=binding.term,
        atom_ids=set(binding.atom_ids),
        bond_ids=set(binding.bond_ids),
        charge_atom_ids=set(binding.charge_atom_ids),
        locants=tuple(binding.locants),
        emitted_tokens=token_bindings,
    )


def _operation_emitted_tokens(binding: NameAtomBinding) -> tuple[NameTokenBinding, ...]:
    """Emit operation-aware token metadata from a graph binding."""

    if binding.stage == "replacement":
        return _locanted_operation_tokens(binding, charge_from_binding=True)
    if binding.stage == "unsaturation":
        return _locanted_operation_tokens(binding, bond_focused=True)
    if binding.stage == "suffix":
        return _locanted_operation_tokens(binding, charge_from_binding=True)
    if binding.stage == "charge" or binding.role == "parent_charge":
        return _charge_operation_tokens(binding)
    if binding.stage == "assembly" and "stereo" in binding.role:
        return _locanted_operation_tokens(binding)
    if binding.stage == "modifier":
        return _modifier_operation_tokens(binding)
    if binding.stage == "parent":
        return _parent_operation_tokens(binding)
    return _whole_scope_tokens(binding)


def _parent_emitted_tokens(parts: AssemblyParts) -> tuple[NameTokenBinding, ...]:
    token = parts.retained_name or _parent_display_token(parts)
    if not token:
        return ()
    return (
        NameTokenBinding(
            text=token,
            atom_ids=set(parts.parent_atom_ids),
            bond_ids=set(parts.parent_bond_ids),
        ),
    )


def _parent_display_token(parts: AssemblyParts) -> str:
    if parts.polycycle_descriptor:
        return parts.polycycle_descriptor
    if parts.is_spiro:
        return "spiro"
    if parts.is_bicycle:
        return "bicyclo"
    stem = stems.stem_for(parts.parent_length)
    if not stem:
        return ""
    return f"{'cyclo' if parts.is_ring else ''}{stem}ane"


def _locanted_operation_tokens(
    binding: NameAtomBinding,
    *,
    bond_focused: bool = False,
    charge_from_binding: bool = False,
) -> tuple[NameTokenBinding, ...]:
    tokens: list[NameTokenBinding] = []
    if binding.locants:
        tokens.append(
            NameTokenBinding(
                text=",".join(str(locant) for locant in binding.locants),
                atom_ids=set(binding.atom_ids),
                bond_ids=set(binding.bond_ids),
                charge_atom_ids=set(binding.charge_atom_ids) if charge_from_binding else set(),
                locants=tuple(binding.locants),
            )
        )
    for token in _binding_term_tokens(binding.term):
        tokens.append(
            NameTokenBinding(
                text=token,
                atom_ids=set() if bond_focused and binding.bond_ids else set(binding.atom_ids),
                bond_ids=set(binding.bond_ids),
                charge_atom_ids=set(binding.charge_atom_ids) if charge_from_binding else set(),
                locants=tuple(binding.locants),
            )
        )
    return tuple(tokens)


def _charge_operation_tokens(binding: NameAtomBinding) -> tuple[NameTokenBinding, ...]:
    tokens: list[NameTokenBinding] = []
    if binding.locants:
        tokens.append(
            NameTokenBinding(
                text=",".join(str(locant) for locant in binding.locants),
                atom_ids=set(binding.atom_ids),
                charge_atom_ids=set(binding.charge_atom_ids or binding.atom_ids),
                locants=tuple(binding.locants),
            )
        )
    charge_atoms = set(binding.charge_atom_ids or binding.atom_ids)
    for token in _charge_operation_text_tokens(binding):
        tokens.append(
            NameTokenBinding(
                text=token,
                atom_ids=set(binding.atom_ids),
                charge_atom_ids=set(charge_atoms),
                locants=tuple(binding.locants),
            )
        )
    return tuple(tokens)


def _charge_operation_text_tokens(binding: NameAtomBinding) -> tuple[str, ...]:
    if binding.term.endswith("+1") or binding.term.endswith("+"):
        return ("ium",)
    if binding.term.endswith("-1") or binding.term.endswith("-"):
        return ("ide",)
    return _binding_term_tokens(binding.term)


def _modifier_operation_tokens(binding: NameAtomBinding) -> tuple[NameTokenBinding, ...]:
    return tuple(
        NameTokenBinding(
            text=token,
            atom_ids=set(binding.atom_ids),
            bond_ids=set(binding.bond_ids),
            charge_atom_ids=set(binding.charge_atom_ids),
            locants=tuple(binding.locants),
        )
        for token in _binding_term_tokens(binding.term)
    )


def _parent_operation_tokens(binding: NameAtomBinding) -> tuple[NameTokenBinding, ...]:
    tokens = []
    for token in _binding_term_tokens(binding.term):
        if token.lower() in {"parent", "chain", "ring", "polycyclic"}:
            continue
        tokens.append(
            NameTokenBinding(
                text=token,
                atom_ids=set(binding.atom_ids),
                bond_ids=set(binding.bond_ids),
                charge_atom_ids=set(binding.charge_atom_ids),
                locants=tuple(binding.locants),
            )
        )
    return tuple(tokens)


def _whole_scope_tokens(binding: NameAtomBinding) -> tuple[NameTokenBinding, ...]:
    return tuple(
        NameTokenBinding(
            text=token,
            atom_ids=set(binding.atom_ids),
            bond_ids=set(binding.bond_ids),
            charge_atom_ids=set(binding.charge_atom_ids),
            locants=tuple(binding.locants),
        )
        for token in _binding_term_tokens(binding.term)
    )


def _binding_term_tokens(term: str) -> tuple[str, ...]:
    """Return typed token candidates represented by a renderer term."""

    if not term.strip() or "_" in term:
        return ()
    return tuple(match.group(0) for match in _BINDING_TOKEN_RE.finditer(term) if match.group(0).lower() != "parent")


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
        if normalised_before in final_text and (
            original in normalised_after or rewritten in normalised_after or rewritten == normalised_before
        ):
            return before
    return rewritten_term


def _contextual_postprocess_replacements() -> tuple[tuple[str, str], ...]:
    from .assembler import LEGACY_POSTPROCESS_LITERAL_REPLACEMENTS

    return (
        tuple((rule.pattern, rule.replacement) for rule in RULES.postprocess.literal_replacements)
        + tuple(LEGACY_POSTPROCESS_LITERAL_REPLACEMENTS)
    )


def _normalise_name_text(text: str) -> str:
    return text.lower().replace(" ", "").replace("(", "").replace(")", "")


_BINDING_TOKEN_RE = re.compile(r"[A-Za-z]+(?:\^[0-9]+)?|\d+(?:,\d+)*(?:'\")?|[0-9]+(?:\([0-9,]+\))?")


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
