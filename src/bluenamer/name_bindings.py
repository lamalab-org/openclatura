"""Name-term to graph-atom binding helpers."""

import re

from .assembly_parent import parent_stem_and_terminal
from .assembly_parts import AssemblyParts, NameAtomBinding, NameTokenBinding
from .nomenclature import RULES
from .principal_suffixes import render_principal_suffix
from .rules import bonds, stems
from .stereo_descriptors import ABSOLUTE_STEREO_DESCRIPTORS, BOND_STEREO_DESCRIPTORS


def refresh_name_atom_bindings(parts: AssemblyParts) -> list[NameAtomBinding]:
    """Populate structured bindings for the current assembly parts."""

    preserved_assembly_stereo = [
        binding
        for binding in parts.name_atom_bindings
        if binding.stage == "assembly" and binding.role in {"relative_stereo", "small_ring_stereo"}
    ]
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
    for operation in parts.hydro_operations:
        if operation.operation_kind != "indicated_hydrogen":
            continue
        bindings.append(
            NameAtomBinding(
                stage="hydro",
                role=operation.operation_kind,
                term="indicated hydrogen",
                atom_ids=set(operation.atom_ids),
                locants=tuple(str(locant) for locant in operation.locants),
                emitted_tokens=_hydro_operation_tokens(operation),
            )
        )
    for locant, descriptor in parts.stereo_features:
        if descriptor not in ABSOLUTE_STEREO_DESCRIPTORS or not locant:
            continue
        atom_idx = parts.parent_atom_ids_by_locant.get(str(locant))
        if atom_idx is None:
            continue
        bindings.append(
            NameAtomBinding(
                stage="assembly",
                role="absolute_stereo",
                term=f"{locant}{descriptor}",
                atom_ids={atom_idx},
                locants=(str(locant),),
                emitted_tokens=_absolute_stereo_tokens(str(locant), descriptor, atom_idx),
            )
        )
    for locant, descriptor in parts.stereo_features:
        if descriptor not in BOND_STEREO_DESCRIPTORS or not locant:
            continue
        _bond_locants, atom_ids, bond_ids = _bond_stereo_graph_scope(parts, str(locant))
        if not atom_ids:
            continue
        bindings.append(
            NameAtomBinding(
                stage="assembly",
                role="bond_stereo",
                term=descriptor,
                atom_ids=atom_ids,
                bond_ids=bond_ids,
                locants=(str(locant),),
                emitted_tokens=_bond_stereo_tokens(str(locant), descriptor, atom_ids, bond_ids),
            )
        )
    bindings.extend(preserved_assembly_stereo)
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
        prefix_tokens = tuple(item.emitted_tokens) or _rendered_term_tokens(
            term,
            token_kind="prefix",
            grammar_role=role,
            binding_key=f"prefix:{role}",
            atom_ids=set(item.atom_ids),
            bond_ids=set(item.bond_ids),
            charge_atom_ids=set(item.charge_atom_ids),
        )
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
                emitted_tokens=_locanted_emitted_tokens(
                    prefix_tokens,
                    tuple(str(locant) for locant in item.locants),
                    token_kind="prefix",
                    grammar_role=role,
                    binding_key=f"prefix:{role}",
                    atom_ids=set(item.atom_ids),
                    bond_ids=set(item.bond_ids),
                    charge_atom_ids=set(item.charge_atom_ids),
                ),
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
                or _exact_charge_renderer_atom_ids(parts.principal_group.key, set(parts.principal_group.atom_ids)),
                locants=tuple(str(locant) for locant in parts.principal_group.locants),
                emitted_tokens=_principal_suffix_emitted_tokens(parts.principal_group),
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
    *,
    rewrite_source: str = "typed_rewrite",
    rewritten_token_ownership: str | None = None,
) -> list[NameAtomBinding]:
    """Apply final name post-processing to binding terms."""

    final_text = _normalise_name_text(final_name or "")
    processed = []
    for binding in bindings:
        processed_term = _contextual_postprocessed_binding_term(
            binding.term,
            postprocess_term(binding.term),
            final_text,
            allow_suffix_context=binding.stage not in {"prefix", "replacement"},
        )
        processed_tokens = tuple(
            _postprocess_token_binding(
                token,
                postprocess_term,
                final_text,
                rewrite_source=rewrite_source,
                rewritten_token_ownership=rewritten_token_ownership,
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


def _postprocess_token_binding(
    token: NameTokenBinding,
    postprocess_term,
    final_text: str,
    *,
    rewrite_source: str,
    rewritten_token_ownership: str | None,
) -> NameTokenBinding:
    processed_text = _contextual_postprocessed_binding_term(
        token.text,
        postprocess_term(token.text),
        final_text,
        allow_suffix_context=token.token_kind not in {"prefix", "replacement"},
    )
    token_changed = processed_text != token.text
    return NameTokenBinding(
        text=processed_text,
        token_kind=token.token_kind,
        ownership=rewritten_token_ownership if token_changed and rewritten_token_ownership else token.ownership,
        confidence="derived" if token.confidence == "exact" else token.confidence,
        source=rewrite_source if token_changed or token.source in {"renderer", "default_binding"} else token.source,
        grammar_role=token.grammar_role,
        binding_key=token.binding_key,
        atom_ids=set(token.atom_ids),
        bond_ids=set(token.bond_ids),
        charge_atom_ids=set(token.charge_atom_ids),
        locants=tuple(token.locants),
    )


def _with_default_emitted_tokens(binding: NameAtomBinding) -> NameAtomBinding:
    """Ensure every binding carries renderer-style token metadata."""

    if binding.emitted_tokens:
        return _with_required_locant_tokens(binding)
    token_bindings = _operation_emitted_tokens(binding)
    return _with_required_locant_tokens(
        NameAtomBinding(
            stage=binding.stage,
            role=binding.role,
            term=binding.term,
            atom_ids=set(binding.atom_ids),
            bond_ids=set(binding.bond_ids),
            charge_atom_ids=set(binding.charge_atom_ids),
            locants=tuple(binding.locants),
            emitted_tokens=token_bindings,
        )
    )


def _with_required_locant_tokens(binding: NameAtomBinding) -> NameAtomBinding:
    if not binding.locants:
        return binding
    locant_text = ",".join(str(locant) for locant in binding.locants)
    required_tokens = []
    if not any(token.text == locant_text for token in binding.emitted_tokens):
        required_tokens.append(locant_text)
    required_tokens.extend(
        str(locant)
        for locant in binding.locants
        if len(binding.locants) > 1 and not any(token.text == str(locant) for token in binding.emitted_tokens)
    )
    if not required_tokens:
        return binding
    return NameAtomBinding(
        stage=binding.stage,
        role=binding.role,
        term=binding.term,
        atom_ids=set(binding.atom_ids),
        bond_ids=set(binding.bond_ids),
        charge_atom_ids=set(binding.charge_atom_ids),
        locants=tuple(binding.locants),
        emitted_tokens=(
            *(
                NameTokenBinding(
                    text=token_text,
                    token_kind="locant",
                    source="default_binding",
                    grammar_role=binding.role,
                    binding_key=f"{binding.stage}:{binding.role}",
                    atom_ids=set(binding.atom_ids),
                    bond_ids=set(binding.bond_ids),
                    charge_atom_ids=set(binding.charge_atom_ids),
                    locants=tuple(binding.locants),
                )
                for token_text in required_tokens
            ),
            *binding.emitted_tokens,
        ),
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
    if binding.stage == "assembly" and binding.role == "relative_stereo":
        return _relative_stereo_tokens(binding)
    if binding.stage == "assembly" and "stereo" in binding.role:
        return _locanted_operation_tokens(binding)
    if binding.stage == "modifier":
        return _modifier_operation_tokens(binding)
    if binding.stage == "parent":
        return _parent_operation_tokens(binding)
    return _whole_scope_tokens(binding)


def _parent_emitted_tokens(parts: AssemblyParts) -> tuple[NameTokenBinding, ...]:
    tokens = _parent_display_tokens(parts)
    return tuple(
        NameTokenBinding(
            text=token,
            token_kind="parent",
            source="default_binding",
            grammar_role="parent",
            binding_key="parent:parent",
            atom_ids=set(parts.parent_atom_ids),
            bond_ids=set(parts.parent_bond_ids),
        )
        for token in tokens
        if token
    )


def _parent_display_tokens(parts: AssemblyParts) -> tuple[str, ...]:
    tokens: list[str] = []
    if parts.retained_name:
        tokens.extend(_parent_token_variants(parts.retained_name))
    stem_str, terminal_e = parent_stem_and_terminal(parts)
    tokens.extend(_parent_token_variants(stem_str))
    if terminal_e == "e":
        tokens.extend(_parent_token_variants(stem_str + "ane"))
    elif terminal_e:
        tokens.extend(_parent_token_variants(stem_str + terminal_e))
    if parts.polycycle_descriptor:
        tokens.append(parts.polycycle_descriptor)
    if parts.is_spiro:
        tokens.append("spiro")
    if parts.is_bicycle:
        tokens.append("bicyclo")
    if (
        parts.is_ring
        and not parts.retained_name
        and not parts.is_polycycle
        and not parts.is_spiro
        and not parts.is_bicycle
    ):
        tokens.append("cyclo" + stems.stem_for(parts.parent_length))
    stem = stems.stem_for(parts.parent_length)
    if stem:
        tokens.extend(_parent_token_variants(stem))
        tokens.extend(_parent_token_variants(f"{stem}ane"))
        if parts.is_ring:
            tokens.extend(_parent_token_variants(f"cyclo{stem}ane"))
    return tuple(dict.fromkeys(token for token in tokens if token))


def _parent_token_variants(token: str) -> tuple[str, ...]:
    variants = [token]
    if token.endswith("ane"):
        variants.extend((token[:-3], token[:-1]))
    elif token.endswith("ene") or token.endswith("yne"):
        variants.extend((token[:-3], token[:-1]))
    elif token.endswith("e"):
        variants.append(token[:-1])
    if token and not token.endswith(("a", "e")):
        variants.append(token + "a")
    return tuple(dict.fromkeys(variant for variant in variants if variant))


def _hydro_operation_tokens(operation) -> tuple[NameTokenBinding, ...]:
    atoms = set(operation.atom_ids)
    locants = tuple(str(locant) for locant in operation.locants)
    tokens: list[NameTokenBinding] = []
    if locants:
        tokens.append(
            NameTokenBinding(
                text=",".join(locants),
                token_kind="locant",
                source="default_binding",
                grammar_role=operation.operation_kind,
                binding_key=f"hydro:{operation.operation_kind}",
                atom_ids=set(atoms),
                locants=locants,
            )
        )
    tokens.append(
        NameTokenBinding(
            text="H",
            token_kind="hydro",
            source="default_binding",
            grammar_role=operation.operation_kind,
            binding_key=f"hydro:{operation.operation_kind}",
            atom_ids=set(atoms),
            locants=locants,
        )
    )
    return tuple(tokens)


def _absolute_stereo_tokens(locant: str, descriptor: str, atom_idx: int) -> tuple[NameTokenBinding, ...]:
    """Return native tokens for an absolute stereochemical descriptor."""

    return (
        NameTokenBinding(
            text=locant,
            token_kind="locant",
            ownership="exact",
            confidence="exact",
            source="renderer_stereo",
            grammar_role="absolute_stereo",
            binding_key="assembly:absolute_stereo",
            atom_ids={atom_idx},
            locants=(locant,),
        ),
        NameTokenBinding(
            text=descriptor,
            token_kind="stereo",
            ownership="exact",
            confidence="exact",
            source="renderer_stereo",
            grammar_role="absolute_stereo",
            binding_key="assembly:absolute_stereo",
            atom_ids={atom_idx},
            locants=(locant,),
        ),
    )


def _bond_stereo_graph_scope(parts: AssemblyParts, locant: str) -> tuple[tuple[str, ...], set[int], set[int]]:
    """Return the graph scope for an E/Z descriptor rendered at a parent locant."""

    for locant_pair, order in parts.parent_bond_orders_by_locants.items():
        if locant not in locant_pair or order != 2:
            continue
        atom_ids = {
            atom_idx
            for pair_locant in locant_pair
            if (atom_idx := parts.parent_atom_ids_by_locant.get(str(pair_locant))) is not None
        }
        if len(atom_ids) != 2:
            continue
        bond_id = parts.parent_bond_ids_by_locants.get(locant_pair)
        return (
            tuple(str(pair_locant) for pair_locant in locant_pair),
            atom_ids,
            {bond_id} if bond_id is not None else set(),
        )
    atom_idx = parts.parent_atom_ids_by_locant.get(locant)
    return (locant,), {atom_idx} if atom_idx is not None else set(), set()


def _bond_stereo_tokens(
    locant: str,
    descriptor: str,
    atom_ids: set[int],
    bond_ids: set[int],
) -> tuple[NameTokenBinding, ...]:
    """Return native tokens for an E/Z bond stereochemical descriptor."""

    return (
        NameTokenBinding(
            text=locant,
            token_kind="locant",
            ownership="exact",
            confidence="exact",
            source="renderer_stereo",
            grammar_role="bond_stereo",
            binding_key="assembly:bond_stereo",
            atom_ids=atom_ids,
            bond_ids=bond_ids,
            locants=(locant,),
        ),
        NameTokenBinding(
            text=descriptor,
            token_kind="stereo",
            ownership="exact",
            confidence="exact",
            source="renderer_stereo",
            grammar_role="bond_stereo",
            binding_key="assembly:bond_stereo",
            atom_ids=atom_ids,
            bond_ids=bond_ids,
            locants=(locant,),
        ),
    )


def _relative_stereo_tokens(binding: NameAtomBinding) -> tuple[NameTokenBinding, ...]:
    """Return native token metadata for relative cis/trans descriptors."""

    return (
        NameTokenBinding(
            text=binding.term,
            token_kind="stereo",
            ownership="exact",
            confidence="exact",
            source="renderer_stereo",
            grammar_role=binding.role,
            binding_key=f"{binding.stage}:{binding.role}",
            atom_ids=set(binding.atom_ids),
            bond_ids=set(binding.bond_ids),
            charge_atom_ids=set(binding.charge_atom_ids),
            locants=tuple(binding.locants),
        ),
    )


def _locanted_emitted_tokens(
    emitted_tokens: tuple[NameTokenBinding, ...],
    locants: tuple[str, ...],
    *,
    token_kind: str,
    grammar_role: str,
    binding_key: str,
    atom_ids: set[int],
    bond_ids: set[int],
    charge_atom_ids: set[int],
) -> tuple[NameTokenBinding, ...]:
    if not locants:
        return emitted_tokens
    locant_text = ",".join(locants)
    tokens = list(emitted_tokens)
    if not any(token.text == locant_text for token in tokens):
        tokens.insert(
            0,
            NameTokenBinding(
                text=locant_text,
                token_kind="locant",
                source="default_binding",
                grammar_role=grammar_role,
                binding_key=binding_key,
                atom_ids=set(atom_ids),
                bond_ids=set(bond_ids),
                charge_atom_ids=set(charge_atom_ids),
                locants=locants,
            ),
        )
    for locant in reversed(locants):
        if len(locants) <= 1:
            continue
        if any(token.text == locant for token in tokens):
            continue
        tokens.insert(
            0,
            NameTokenBinding(
                text=locant,
                token_kind="locant",
                source="default_binding",
                grammar_role=grammar_role,
                binding_key=binding_key,
                atom_ids=set(atom_ids),
                bond_ids=set(bond_ids),
                charge_atom_ids=set(charge_atom_ids),
                locants=locants,
            ),
        )
    return tuple(tokens)


def _principal_suffix_emitted_tokens(group) -> tuple[NameTokenBinding, ...]:
    """Emit suffix tokens from the functional-group renderer, not a word list."""

    rule = RULES.functional_groups.get(group.key)
    locants = tuple(str(locant) for locant in group.locants)
    rendered_suffix = render_principal_suffix(rule, len(locants) or 1)
    return _rendered_term_tokens(
        rendered_suffix,
        token_kind="suffix",
        grammar_role=group.key,
        binding_key=f"suffix:{group.key}",
        atom_ids=set(group.atom_ids),
        bond_ids=set(group.bond_ids),
        charge_atom_ids=set(group.charge_atom_ids),
        locants=locants,
        source="renderer_suffix",
    )


def _rendered_term_tokens(
    term: str,
    *,
    token_kind: str,
    grammar_role: str,
    binding_key: str,
    atom_ids: set[int],
    bond_ids: set[int],
    charge_atom_ids: set[int],
    locants: tuple[str, ...] = (),
    source: str = "renderer_term",
) -> tuple[NameTokenBinding, ...]:
    """Emit tokens from a renderer term with the term's graph ownership."""

    tokens: list[NameTokenBinding] = []
    for token_text in (*_binding_term_tokens(term), *_data_backed_prefix_subtokens(term)):
        tokens.append(
            NameTokenBinding(
                text=token_text,
                token_kind=token_kind,
                source=source,
                grammar_role=grammar_role,
                binding_key=binding_key,
                atom_ids=set(atom_ids),
                bond_ids=set(bond_ids),
                charge_atom_ids=set(charge_atom_ids),
                locants=locants,
            )
        )
    return tuple(tokens)


def _data_backed_prefix_subtokens(term: str) -> tuple[str, ...]:
    """Return subtokens from registered functional-prefix spellings."""

    normalised_term = _normalise_name_text(term)
    if not normalised_term:
        return ()
    subtokens: list[str] = []
    for rule in RULES.functional_groups.by_key.values():
        prefix = rule.prefix
        if not prefix:
            continue
        for token in _binding_term_tokens(prefix):
            token_norm = _normalise_name_text(token)
            if token_norm and token_norm != normalised_term and token_norm in normalised_term:
                subtokens.append(token)
    return tuple(dict.fromkeys(subtokens))


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
                token_kind="locant",
                source="default_binding",
                grammar_role=binding.role,
                binding_key=f"{binding.stage}:{binding.role}",
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
                token_kind=_token_kind_for_binding(binding),
                source="default_binding",
                grammar_role=binding.role,
                binding_key=f"{binding.stage}:{binding.role}",
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
                token_kind="locant",
                source="default_binding",
                grammar_role=binding.role,
                binding_key=f"{binding.stage}:{binding.role}",
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
                token_kind="charge",
                source="default_binding",
                grammar_role=binding.role,
                binding_key=f"{binding.stage}:{binding.role}",
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
            token_kind="modifier",
            source="default_binding",
            grammar_role=binding.role,
            binding_key=f"{binding.stage}:{binding.role}",
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
                token_kind="parent",
                source="default_binding",
                grammar_role=binding.role,
                binding_key=f"{binding.stage}:{binding.role}",
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
            token_kind=_token_kind_for_binding(binding),
            source="default_binding",
            grammar_role=binding.role,
            binding_key=f"{binding.stage}:{binding.role}",
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


def _token_kind_for_binding(binding: NameAtomBinding) -> str:
    if binding.stage in {"parent", "prefix", "suffix", "replacement", "unsaturation", "charge"}:
        return binding.stage
    if binding.role == "parent_charge":
        return "charge"
    return "structural"


def _contextual_postprocessed_binding_term(
    original_term: str,
    rewritten_term: str,
    final_text: str,
    *,
    allow_suffix_context: bool = True,
) -> str:
    """Map terms absorbed by contextual final-name post-processing."""

    original = _normalise_name_text(original_term)
    rewritten = _normalise_name_text(rewritten_term)
    for before, after in _contextual_postprocess_replacements():
        normalised_before = _normalise_name_text(before)
        normalised_after = _normalise_name_text(after)
        if _normalised_rendered_term_occurs(normalised_after, final_text) and (
            _contextual_match_scope(original, normalised_before, normalised_after, allow_suffix_context)
            or _contextual_match_scope(rewritten, normalised_before, normalised_after, allow_suffix_context)
            or rewritten == normalised_after
        ):
            return after
        if _normalised_rendered_term_occurs(normalised_before, final_text) and (
            _contextual_match_scope(original, normalised_before, normalised_after, allow_suffix_context)
            or _contextual_match_scope(rewritten, normalised_before, normalised_after, allow_suffix_context)
            or rewritten == normalised_before
        ):
            return before
    return rewritten_term


def _contextual_match_scope(text: str, before: str, after: str, allow_suffix_context: bool = True) -> bool:
    """Return true when ``text`` is specific enough for contextual rewrites."""

    return (
        text == before
        or text == after
        or (
            len(text) >= 4
            and (_normalised_rendered_term_occurs(text, before) or _normalised_rendered_term_occurs(text, after))
        )
        or (allow_suffix_context and len(text) >= 6 and (before.endswith(text) or after.endswith(text)))
    )


def _normalised_rendered_term_occurs(term: str, final_text: str) -> bool:
    """Return true when a normalized rendered term is not embedded mid-word."""

    if not term:
        return False
    pos = 0
    while True:
        found = final_text.find(term, pos)
        if found < 0:
            return False
        before = final_text[found - 1] if found > 0 else ""
        if not before or not before.isalpha():
            return True
        pos = found + 1


def _contextual_postprocess_replacements() -> tuple[tuple[str, str], ...]:
    from .assembler import LEGACY_POSTPROCESS_LITERAL_REPLACEMENTS

    return tuple((rule.pattern, rule.replacement) for rule in RULES.postprocess.literal_replacements) + tuple(
        LEGACY_POSTPROCESS_LITERAL_REPLACEMENTS
    )


def _normalise_name_text(text: str) -> str:
    return text.lower().replace(" ", "").replace("(", "").replace(")", "")


_BINDING_TOKEN_RE = re.compile(r"[A-Za-z]+(?:\^[0-9]+)?|\d+(?:,\d+)*(?:'\")?|[0-9]+(?:\([0-9,]+\))?")


def _replacement_charge_atom_ids(parts: AssemblyParts, item) -> set[int]:
    """Return charged replacement-prefix atoms represented by this binding."""

    charged_locants = {
        str(locant) for locant in item.locants if parts.parent_atom_charges_by_locant.get(str(locant), 0) != 0
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
