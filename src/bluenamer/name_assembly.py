"""Typed name assembly objects and metadata-preserving rewrites."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from .assembly_parts import AssemblyParts, NameAtomBinding, NameTokenBinding
from .molecule import Molecule
from .name_bindings import ensure_name_atom_binding_tokens, postprocess_name_atom_bindings


@dataclass(frozen=True)
class GraphRole:
    """Detected structural role with no naming decision attached."""

    key: str
    atom_ids: frozenset[int] = frozenset()
    bond_ids: frozenset[int] = frozenset()
    charges_by_atom: dict[int, int] = field(default_factory=dict)
    locants_by_atom: dict[int, str] = field(default_factory=dict)
    evidence: tuple[str, ...] = ()


@dataclass(frozen=True)
class RendererTemplate:
    """A supported rendering path for a graph role in a grammar context."""

    key: str
    role_key: str
    context: str
    grammar: str
    supported: bool = False
    verified: bool = False
    preserves_formal_charges: bool = False
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class NameFragment:
    """One emitted word or operation and the graph metadata it represents."""

    text: str
    bindings: tuple[NameAtomBinding, ...] = ()
    role: GraphRole | None = None
    template: RendererTemplate | None = None

    @classmethod
    def from_binding(cls, binding: NameAtomBinding) -> "NameFragment":
        return cls(text=binding.term, bindings=(binding,))


@dataclass(frozen=True)
class NameRewriteOperation:
    """A text rewrite that also transforms fragment/binding metadata."""

    name: str
    before: str
    after: str
    binding_count: int
    changed_binding_count: int
    token_count: int = 0
    changed_token_count: int = 0

    @classmethod
    def apply(
        cls,
        name: str,
        bindings: tuple[NameAtomBinding, ...],
        *,
        operation_name: str,
        rewrite: Callable[[str], str],
    ) -> tuple[str, tuple[NameAtomBinding, ...], "NameRewriteOperation"]:
        """Apply a rewrite to final text and every bound term."""

        rewritten_name = rewrite(name)
        rewritten_bindings = tuple(postprocess_name_atom_bindings(list(bindings), rewrite, final_name=rewritten_name))
        changed_bindings = sum(
            1
            for before_binding, after_binding in zip(bindings, rewritten_bindings, strict=False)
            if before_binding.term != after_binding.term
        )
        before_tokens = [token.text for binding in bindings for token in binding.emitted_tokens]
        after_tokens = [token.text for binding in rewritten_bindings for token in binding.emitted_tokens]
        changed_tokens = sum(
            1
            for before_token, after_token in zip(before_tokens, after_tokens, strict=False)
            if before_token != after_token
        ) + abs(len(before_tokens) - len(after_tokens))
        return (
            rewritten_name,
            rewritten_bindings,
            cls(
                name=operation_name,
                before=name,
                after=rewritten_name,
                binding_count=len(bindings),
                changed_binding_count=changed_bindings,
                token_count=len(after_tokens),
                changed_token_count=changed_tokens,
            ),
        )


@dataclass(frozen=True)
class NameTokenSpan:
    """One final-name token span with graph bindings."""

    text: str
    start: int
    end: int
    binding_indices: tuple[int, ...] = ()
    atom_ids: frozenset[int] = frozenset()
    bond_ids: frozenset[int] = frozenset()
    charge_atom_ids: frozenset[int] = frozenset()
    locants: tuple[str, ...] = ()


@dataclass(frozen=True)
class NameAssemblyResult:
    """Final name text plus the graph metadata that survived assembly."""

    raw_text: str
    text: str
    fragments: tuple[NameFragment, ...]
    bindings: tuple[NameAtomBinding, ...]
    rewrite_history: tuple[NameRewriteOperation, ...] = ()
    token_spans: tuple[NameTokenSpan, ...] = ()

    @property
    def atom_ids(self) -> set[int]:
        atoms: set[int] = set()
        for binding in self.bindings:
            atoms.update(binding.atom_ids)
        return atoms

    @property
    def bond_ids(self) -> set[int]:
        bonds: set[int] = set()
        for binding in self.bindings:
            bonds.update(binding.bond_ids)
        return bonds

    @property
    def charged_atom_ids(self) -> set[int]:
        atoms: set[int] = set()
        for binding in self.bindings:
            atoms.update(binding.charge_atom_ids)
            if binding.stage == "charge" or binding.role == "parent_charge":
                atoms.update(binding.atom_ids)
        return atoms

    @classmethod
    def from_raw_name(
        cls,
        raw_text: str,
        bindings: list[NameAtomBinding] | tuple[NameAtomBinding, ...],
        *,
        postprocess: Callable[[str], str],
    ) -> "NameAssemblyResult":
        """Build a final assembly result while keeping binding metadata in sync."""

        return cls.from_rewrite_pipeline(
            raw_text,
            bindings,
            rewrites=(("post_process_name", postprocess),),
        )

    @classmethod
    def from_rewrite_pipeline(
        cls,
        raw_text: str,
        bindings: list[NameAtomBinding] | tuple[NameAtomBinding, ...],
        *,
        rewrites: tuple[tuple[str, Callable[[str], str]], ...],
    ) -> "NameAssemblyResult":
        """Build a final result by applying named rewrites to text and bindings."""

        text = raw_text
        binding_tuple = _ensure_emitted_token_bindings(tuple(bindings))
        history: list[NameRewriteOperation] = []
        for operation_name, rewrite in rewrites:
            text, binding_tuple, operation = NameRewriteOperation.apply(
                text,
                binding_tuple,
                operation_name=operation_name,
                rewrite=rewrite,
            )
            history.append(operation)
        return cls(
            raw_text=raw_text,
            text=text,
            fragments=tuple(NameFragment.from_binding(binding) for binding in binding_tuple),
            bindings=binding_tuple,
            rewrite_history=tuple(history),
            token_spans=build_name_token_spans(text, binding_tuple),
        )

    @classmethod
    def from_final_name(
        cls,
        text: str,
        bindings: list[NameAtomBinding] | tuple[NameAtomBinding, ...],
        *,
        rewrite_history: tuple[NameRewriteOperation, ...] = (),
    ) -> "NameAssemblyResult":
        """Build a result for callers that already finalized text and bindings."""

        binding_tuple = _ensure_emitted_token_bindings(tuple(bindings))
        return cls(
            raw_text=text,
            text=text,
            fragments=tuple(NameFragment.from_binding(binding) for binding in binding_tuple),
            bindings=binding_tuple,
            rewrite_history=rewrite_history,
            token_spans=build_name_token_spans(text, binding_tuple),
        )


@dataclass(frozen=True)
class FinalAssemblyAudit:
    """Final metadata audit for an assembled component name."""

    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.errors


class FinalAssemblyAuditError(ValueError):
    """Raised when final name text and final metadata disagree."""


def _ensure_emitted_token_bindings(bindings: tuple[NameAtomBinding, ...]) -> tuple[NameAtomBinding, ...]:
    return tuple(ensure_name_atom_binding_tokens(binding) for binding in bindings)


def token_span_trace_data(result: NameAssemblyResult) -> list[dict]:
    """Return JSON-friendly final-token to graph-binding metadata."""

    return [
        {
            "text": token.text,
            "start": token.start,
            "end": token.end,
            "binding_indices": list(token.binding_indices),
            "atoms": sorted(token.atom_ids),
            "bonds": sorted(token.bond_ids),
            "charge_atoms": sorted(token.charge_atom_ids),
            "locants": list(token.locants),
        }
        for token in result.token_spans
    ]


def audit_final_name_assembly(
    mol: Molecule,
    component_atoms: set[int],
    parts: AssemblyParts,
    result: NameAssemblyResult,
) -> FinalAssemblyAudit:
    """Check the final emitted name after all post-processing has run."""

    errors: list[str] = []
    warnings: list[str] = []
    explicit_component_atoms = {idx for idx in component_atoms if mol.atoms[idx].symbol != "H"}
    unnamed_atoms = explicit_component_atoms - result.atom_ids
    if unnamed_atoms:
        errors.append(_format_atom_error("unnamed atoms", mol, unnamed_atoms))

    expected_charged_atoms = {
        idx
        for idx in explicit_component_atoms
        if idx in mol.atoms and mol.atoms[idx].charge != 0
    }
    missing_charged_atoms = expected_charged_atoms - result.charged_atom_ids
    if missing_charged_atoms:
        errors.append(_format_atom_error("charged atoms not represented", mol, missing_charged_atoms))

    consumed_bonds = _consumed_bond_ids(parts)
    missing_bonds = consumed_bonds - result.bond_ids
    if missing_bonds:
        errors.append(f"consumed bonds not represented: {sorted(missing_bonds)}")

    invalid_bindings = [
        idx
        for idx, binding in enumerate(result.bindings)
        if not binding.term.strip() or (not binding.atom_ids and not binding.bond_ids)
    ]
    if invalid_bindings:
        errors.append(f"invalid final name bindings: {invalid_bindings}")

    missing_terms = _missing_concrete_binding_terms(result)
    if missing_terms:
        errors.append(f"binding terms absent from final name: {missing_terms}")
    unbound_tokens = _unbound_name_tokens(result)
    if unbound_tokens:
        errors.append(f"final name tokens without graph binding: {unbound_tokens}")

    return FinalAssemblyAudit(tuple(errors), tuple(warnings))


def assert_final_name_assembly(
    mol: Molecule,
    component_atoms: set[int],
    parts: AssemblyParts,
    result: NameAssemblyResult,
) -> None:
    """Raise if the final name has lost required graph metadata."""

    audit = audit_final_name_assembly(mol, component_atoms, parts, result)
    if not audit.ok:
        raise FinalAssemblyAuditError(f"Generated name {result.text!r} failed final metadata audit: {'; '.join(audit.errors)}")


def _consumed_bond_ids(parts: AssemblyParts) -> set[int]:
    bonds: set[int] = set(parts.parent_bond_ids)
    for item in parts.principal_suffix_modifiers:
        bonds.update(item.bond_ids)
    for item in parts.a_prefixes:
        bonds.update(item.bond_ids)
    for item in parts.substituents:
        bonds.update(item.bond_ids)
    for item in parts.unsaturations:
        bonds.update(item.bond_ids)
    if parts.principal_group is not None:
        bonds.update(parts.principal_group.bond_ids)
    for binding in parts.name_atom_bindings:
        bonds.update(binding.bond_ids)
    return bonds


def _format_atom_error(label: str, mol: Molecule, atom_ids: set[int]) -> str:
    details = ", ".join(f"{idx}:{mol.atoms[idx].symbol}" for idx in sorted(atom_ids))
    return f"{label}: {details}"


def build_name_token_spans(text: str, bindings: tuple[NameAtomBinding, ...]) -> tuple[NameTokenSpan, ...]:
    """Bind every lexical final-name token to graph metadata where possible."""

    native_spans = _native_token_spans(text, bindings)
    direct_spans = _direct_binding_spans(text, bindings)
    tokens = []
    for match in _LEXICAL_TOKEN_RE.finditer(text):
        start, end = match.span()
        token_text = match.group(0)
        native_matches = [
            (span_start, span_end, binding_idx, token_binding)
            for span_start, span_end, binding_idx, token_binding in native_spans
            if _spans_overlap(start, end, span_start, span_end)
        ]
        if native_matches:
            tokens.extend(_token_spans_from_native_matches(text, start, end, native_matches))
            continue
        binding_indices = {
            binding_idx
            for span_start, span_end, binding_idx in direct_spans
            if _spans_overlap(start, end, span_start, span_end)
        }
        if not binding_indices:
            binding_indices.update(_fallback_token_binding_indices(token_text, start, end, text, bindings))
        tokens.append(_token_span_from_binding_indices(token_text, start, end, tuple(sorted(binding_indices)), bindings))
    return tuple(tokens)


def _native_token_spans(
    text: str,
    bindings: tuple[NameAtomBinding, ...],
) -> list[tuple[int, int, int, NameTokenBinding]]:
    spans: list[tuple[int, int, int, NameTokenBinding]] = []
    search_text = text.lower()
    for binding_idx, binding in enumerate(bindings):
        ordered_spans = _ordered_native_token_spans(search_text, binding_idx, binding)
        if ordered_spans:
            spans.extend(ordered_spans)
            continue
        for token_binding in binding.emitted_tokens:
            token = token_binding.text.strip().lower()
            if not _native_token_is_searchable(token):
                continue
            pos = 0
            while True:
                found = search_text.find(token, pos)
                if found < 0:
                    break
                spans.append((found, found + len(token), binding_idx, token_binding))
                pos = found + 1
    return spans


def _ordered_native_token_spans(
    search_text: str,
    binding_idx: int,
    binding: NameAtomBinding,
) -> list[tuple[int, int, int, NameTokenBinding]]:
    """Place renderer-emitted tokens once, in renderer order."""

    placed: list[tuple[int, int, int, NameTokenBinding]] = []
    cursor = 0
    for token_binding in binding.emitted_tokens:
        token = token_binding.text.strip().lower()
        if not _native_token_is_searchable(token):
            continue
        found = search_text.find(token, cursor)
        if found < 0:
            return []
        placed.append((found, found + len(token), binding_idx, token_binding))
        cursor = found + len(token)
    return placed


def _native_token_is_searchable(token: str) -> bool:
    return bool(token) and (len(token) >= 2 or token.isdigit() or "," in token or token in _ELEMENT_LOCANT_TOKENS)


def _direct_binding_spans(text: str, bindings: tuple[NameAtomBinding, ...]) -> list[tuple[int, int, int]]:
    spans: list[tuple[int, int, int]] = []
    search_text = text.lower()
    for binding_idx, binding in enumerate(bindings):
        terms = _binding_search_terms(binding)
        for term in terms:
            if not term:
                continue
            pos = 0
            while True:
                found = search_text.find(term, pos)
                if found < 0:
                    break
                spans.append((found, found + len(term), binding_idx))
                pos = found + 1
    return spans


def _binding_search_terms(binding: NameAtomBinding) -> set[str]:
    term = binding.term.strip().lower()
    if not term or term.endswith(" parent") or "_" in term:
        return set()
    terms = {term, term.replace(" ", "")}
    if binding.stage == "parent":
        terms.update(_retained_parent_stem_variants(re.sub(r"-\d+(?:,\d+)*-(?:ide|ium)$", "", term)))
    return {item for item in terms if len(item) >= 2}


def _fallback_token_binding_indices(
    token: str,
    start: int,
    end: int,
    text: str,
    bindings: tuple[NameAtomBinding, ...],
) -> set[int]:
    token_norm = _normalise_name_text(token)
    indices: set[int] = set()
    if token_norm.isdigit() or "," in token_norm:
        indices.update(_locant_binding_indices(token_norm, bindings))
    if not indices:
        indices.update(_charge_suffix_binding_indices(token_norm, start, text, bindings))
    if not indices:
        indices.update(_role_token_binding_indices(token_norm, bindings))
    if not indices and _is_structural_suffix_token(token_norm, text, start, end):
        indices.update(_suffix_binding_indices(bindings))
    if not indices and _is_parent_like_token(token_norm):
        indices.update(_parent_binding_indices(bindings))
    if not indices and _is_prefix_like_token(token_norm):
        indices.update(_prefix_binding_indices(bindings))
    if not indices and (_is_plausible_chemical_token(token_norm) or len(_graph_bearing_binding_indices(bindings)) > 1):
        indices.update(_graph_bearing_binding_indices(bindings))
    return indices


def _locant_binding_indices(token: str, bindings: tuple[NameAtomBinding, ...]) -> set[int]:
    locants = {part for part in re.split(r",", token) if part}
    if not locants:
        return set()
    return {
        idx
        for idx, binding in enumerate(bindings)
        if binding.locants and {str(locant).strip("'") for locant in binding.locants} <= locants
    }


def _charge_suffix_binding_indices(
    token: str,
    start: int,
    text: str,
    bindings: tuple[NameAtomBinding, ...],
) -> set[int]:
    """Bind charge suffix words to adjacent locanted charge operations."""

    if token not in _CHARGE_SUFFIX_TOKENS:
        return set()
    locant_match = re.search(r"(\d+(?:,\d+)*)-$", text[:start])
    locants = set(locant_match.group(1).split(",")) if locant_match else set()
    charge_indices = {
        idx
        for idx, binding in enumerate(bindings)
        if binding.stage == "charge" or binding.role == "parent_charge" or binding.charge_atom_ids
    }
    if not locants:
        return charge_indices
    return {
        idx
        for idx in charge_indices
        if locants <= {str(locant).strip("'") for locant in bindings[idx].locants}
    } or charge_indices


def _role_token_binding_indices(token: str, bindings: tuple[NameAtomBinding, ...]) -> set[int]:
    indices: set[int] = set()
    for idx, binding in enumerate(bindings):
        role = binding.role.replace("_", "").lower()
        if token == role or token in _ROLE_TOKEN_ALIASES.get(role, ()):
            indices.add(idx)
    return indices


def _suffix_binding_indices(bindings: tuple[NameAtomBinding, ...]) -> set[int]:
    return {idx for idx, binding in enumerate(bindings) if binding.stage == "suffix"}


def _parent_binding_indices(bindings: tuple[NameAtomBinding, ...]) -> set[int]:
    return {idx for idx, binding in enumerate(bindings) if binding.stage == "parent"}


def _prefix_binding_indices(bindings: tuple[NameAtomBinding, ...]) -> set[int]:
    return {idx for idx, binding in enumerate(bindings) if binding.stage in {"prefix", "modifier", "replacement"}}


def _graph_bearing_binding_indices(bindings: tuple[NameAtomBinding, ...]) -> set[int]:
    """Return conservative owners for tokens not yet assignable to one role."""

    return {
        idx
        for idx, binding in enumerate(bindings)
        if binding.atom_ids or binding.bond_ids or binding.charge_atom_ids
    }


def _is_structural_suffix_token(token: str, text: str, start: int, end: int) -> bool:
    return token in _STRUCTURAL_SUFFIX_TOKENS or any(token.endswith(suffix) for suffix in _STRUCTURAL_SUFFIX_ENDINGS)


def _is_parent_like_token(token: str) -> bool:
    return token in _PARENT_DESCRIPTOR_TOKENS or any(token.endswith(ending) for ending in _PARENT_TOKEN_ENDINGS)


def _is_prefix_like_token(token: str) -> bool:
    return any(token.endswith(ending) for ending in _PREFIX_TOKEN_ENDINGS)


def _is_plausible_chemical_token(token: str) -> bool:
    return (
        token.isdigit()
        or token in _CHEMISTRY_GRAMMAR_TOKENS
        or _is_parent_like_token(token)
        or _is_prefix_like_token(token)
        or _is_structural_suffix_token(token, "", 0, 0)
    )


def _token_span_from_binding_indices(
    text: str,
    start: int,
    end: int,
    binding_indices: tuple[int, ...],
    bindings: tuple[NameAtomBinding, ...],
) -> NameTokenSpan:
    atoms: set[int] = set()
    bonds: set[int] = set()
    charges: set[int] = set()
    locants: list[str] = []
    for idx in binding_indices:
        binding = bindings[idx]
        atoms.update(binding.atom_ids)
        bonds.update(binding.bond_ids)
        charges.update(binding.charge_atom_ids)
        locants.extend(str(locant) for locant in binding.locants)
    return NameTokenSpan(
        text=text,
        start=start,
        end=end,
        binding_indices=binding_indices,
        atom_ids=frozenset(atoms),
        bond_ids=frozenset(bonds),
        charge_atom_ids=frozenset(charges),
        locants=tuple(locants),
    )


def _token_spans_from_native_matches(
    text: str,
    start: int,
    end: int,
    matches: list[tuple[int, int, int, NameTokenBinding]],
) -> list[NameTokenSpan]:
    exact_matches = [
        (binding_idx, token_binding)
        for span_start, span_end, binding_idx, token_binding in matches
        if span_start == start and span_end == end
    ]
    if exact_matches:
        return [_token_span_from_native_binding_group(text[start:end], start, end, exact_matches)]

    spans = []
    sorted_matches = sorted(matches, key=lambda item: (item[0], item[1], item[2]))
    grouped_matches: dict[tuple[int, int], list[tuple[int, NameTokenBinding]]] = {}
    for span_start, span_end, binding_idx, token_binding in sorted_matches:
        clipped_start = max(start, span_start)
        clipped_end = min(end, span_end)
        if clipped_start < clipped_end:
            grouped_matches.setdefault((clipped_start, clipped_end), []).append((binding_idx, token_binding))
    cursor = start
    for clipped_start, clipped_end in sorted(grouped_matches):
        if cursor < clipped_start:
            gap_text = text[cursor:clipped_start]
            if _normalise_name_text(gap_text):
                spans.append(
                    _token_span_from_native_binding_group(
                        gap_text,
                        cursor,
                        clipped_start,
                        [(match_binding_idx, match_token_binding) for _, _, match_binding_idx, match_token_binding in sorted_matches],
                    )
                )
        spans.append(
            _token_span_from_native_binding_group(
                text[clipped_start:clipped_end],
                clipped_start,
                clipped_end,
                grouped_matches[(clipped_start, clipped_end)],
            )
        )
        cursor = max(cursor, clipped_end)
    if cursor < end:
        gap_text = text[cursor:end]
        if _normalise_name_text(gap_text):
            spans.append(
                _token_span_from_native_binding_group(
                    gap_text,
                    cursor,
                    end,
                    [(match_binding_idx, match_token_binding) for _, _, match_binding_idx, match_token_binding in sorted_matches],
                )
            )
    return spans


def _token_span_from_native_binding_group(
    text: str,
    start: int,
    end: int,
    matches: list[tuple[int, NameTokenBinding]],
) -> NameTokenSpan:
    atoms: set[int] = set()
    bonds: set[int] = set()
    charges: set[int] = set()
    locants: list[str] = []
    binding_indices: set[int] = set()
    for binding_idx, token_binding in matches:
        binding_indices.add(binding_idx)
        atoms.update(token_binding.atom_ids)
        bonds.update(token_binding.bond_ids)
        charges.update(token_binding.charge_atom_ids)
        locants.extend(str(locant) for locant in token_binding.locants)
    return NameTokenSpan(
        text=text,
        start=start,
        end=end,
        binding_indices=tuple(sorted(binding_indices)),
        atom_ids=frozenset(atoms),
        bond_ids=frozenset(bonds),
        charge_atom_ids=frozenset(charges),
        locants=tuple(locants),
    )


def _unbound_name_tokens(result: NameAssemblyResult) -> list[dict]:
    return [
        {"text": token.text, "start": token.start, "end": token.end}
        for token in result.token_spans
        if not token.binding_indices and _token_requires_graph_binding(token.text)
    ]


def _token_requires_graph_binding(token: str) -> bool:
    return bool(_normalise_name_text(token))


def _spans_overlap(left_start: int, left_end: int, right_start: int, right_end: int) -> bool:
    return left_start < right_end and right_start < left_end


def _missing_concrete_binding_terms(result: NameAssemblyResult) -> list[dict]:
    """Return concrete binding terms that no longer occur in final text.

    Some bindings are intentionally abstract graph certificates, for example
    ``chain parent`` or ``N+1`` parent-charge markers. Those are audited by
    atom/bond/charge coverage instead of literal text presence. Concrete name
    words must still survive post-processing into the final rendered string.
    """

    final_text = _normalise_name_text(result.text)
    missing: list[dict] = []
    for idx, binding in enumerate(result.bindings):
        if not _binding_term_requires_text_presence(binding):
            continue
        term = _normalise_name_text(binding.term)
        if term and not _binding_term_occurs_in_final_name(binding, term, final_text):
            if _binding_is_subsumed_by_present_binding(binding, result.bindings, final_text):
                continue
            missing.append(
                {
                    "index": idx,
                    "stage": binding.stage,
                    "role": binding.role,
                    "term": binding.term,
                }
            )
    return missing


def _binding_is_subsumed_by_present_binding(
    binding: NameAtomBinding,
    bindings: tuple[NameAtomBinding, ...],
    final_text: str,
) -> bool:
    """Return whether another emitted term covers this binding's graph scope."""

    for other in bindings:
        if other is binding or not _binding_term_requires_text_presence(other):
            continue
        other_term = _normalise_name_text(other.term)
        if not other_term or not _binding_term_occurs_in_final_name(other, other_term, final_text):
            continue
        atoms_covered = not binding.atom_ids or binding.atom_ids <= other.atom_ids
        bonds_covered = not binding.bond_ids or binding.bond_ids <= other.bond_ids
        charges_covered = not binding.charge_atom_ids or binding.charge_atom_ids <= other.charge_atom_ids
        if atoms_covered and bonds_covered and charges_covered:
            return True
    return False


def _binding_term_requires_text_presence(binding: NameAtomBinding) -> bool:
    term = binding.term.strip()
    if not term:
        return False
    if binding.stage == "charge" or binding.role == "parent_charge":
        return False
    if binding.stage == "unsaturation":
        return False
    if binding.stage == "suffix" and binding.role == term:
        return False
    if binding.role == "replacement_prefix" and term in _REPLACEMENT_PREFIX_CERTIFICATE_TERMS:
        return False
    if term.endswith(" parent"):
        return False
    if binding.stage == "prefix" and ("(" in term or ")" in term):
        return False
    if "_" in term:
        return False
    if term in _ABSTRACT_BINDING_TERMS:
        return False
    return True


def _normalise_name_text(text: str) -> str:
    return text.lower().replace(" ", "").replace("(", "").replace(")", "")


def _term_occurs_in_final_name(term: str, final_text: str) -> bool:
    if term in final_text:
        return True
    return term.endswith("e") and term[:-1] in final_text


def _binding_term_occurs_in_final_name(binding: NameAtomBinding, term: str, final_text: str) -> bool:
    if _term_occurs_in_final_name(term, final_text):
        return True
    role = binding.role.replace("_", "").lower()
    if any(_term_occurs_in_final_name(_normalise_name_text(alias), final_text) for alias in _ROLE_TOKEN_ALIASES.get(role, ())):
        return True
    if binding.stage == "parent" and _ionic_parent_stem_occurs(term, final_text):
        return True
    if binding.stage == "modifier" and binding.role == "front_modifier":
        term_parts = [_normalise_name_text(part) for part in binding.term.split()]
        return bool(term_parts) and all(_term_occurs_in_final_name(part, final_text) for part in term_parts)
    if binding.stage == "parent" and term == "benzene":
        return "benz" in final_text
    return False


def _ionic_parent_stem_occurs(term: str, final_text: str) -> bool:
    """Return whether a retained parent stem survived with different ion suffixing."""

    stem = re.sub(r"-\d+(?:,\d+)*-(?:ide|ium)$", "", term)
    if stem == term or len(stem) < 4:
        return False
    return any(variant in final_text for variant in _retained_parent_stem_variants(stem))


def _retained_parent_stem_variants(stem: str) -> set[str]:
    """Return audit spellings for retained parent stems after ionic suffix rewrites."""

    variants = {stem}
    unlocanted = re.sub(r"^\d+(?:,\d+)*-", "", stem)
    variants.add(unlocanted)
    for item in tuple(variants):
        if item.endswith("in"):
            variants.add(f"{item}e")
    return {variant for variant in variants if len(variant) >= 4}


_ABSTRACT_BINDING_TERMS = frozenset(
    {
        "alcohol",
        "aldehyde",
        "amide",
        "amine",
        "carboxylic acid",
        "ester",
        "ether",
        "ketone",
        "nitrile",
        "thiol",
    }
)

_REPLACEMENT_PREFIX_CERTIFICATE_TERMS = frozenset({"aza", "oxa", "thia"})

_LEXICAL_TOKEN_RE = re.compile(r"[A-Za-z]+(?:\^[0-9]+)?|\d+(?:,\d+)*(?:'\")?|[0-9]+(?:\([0-9,]+\))?")

_ROLE_TOKEN_ALIASES = {
    "ketone": ("one", "dione", "trione", "oxo"),
    "aldehyde": ("al", "formyl", "carbaldehyde"),
    "carboxylate": ("carboxylate", "oate", "ate"),
    "ester": ("oate", "ate", "formate", "carbonate"),
    "amide": ("amide", "carbamic", "formamide"),
    "nitrile": ("nitrile", "carbonitrile", "cyano", "cyanide", "cyanamide"),
    "ringnitrile": ("carbonitrile", "nitrile", "cyano"),
    "alcohol": ("ol", "hydroxy"),
    "amine": ("amine", "amino"),
    "thiol": ("thiol", "sulfanyl"),
    "replacementprefix": ("aza", "oxa", "thia"),
}

_STRUCTURAL_SUFFIX_TOKENS = frozenset(
    {
        "acid",
        "amide",
        "amine",
        "amino",
        "carbonitrile",
        "carbaldehyde",
        "carbamic",
        "carbonate",
        "dione",
        "ide",
        "ium",
        "nitrile",
        "ol",
        "one",
        "oxo",
        "yl",
        "ylidene",
        "ylidyne",
    }
)

_STRUCTURAL_SUFFIX_ENDINGS = ("one", "dione", "trione", "ol", "al", "amide", "nitrile", "oate", "ate", "ide", "ium")
_CHARGE_SUFFIX_TOKENS = frozenset({"ide", "ium", "aminium", "ylium", "uide"})
_PARENT_DESCRIPTOR_TOKENS = frozenset(
    {
        "bicyclo",
        "cyclo",
        "dispiro",
        "pentacyclo",
        "spiro",
        "tetracyclo",
        "tricyclo",
        "trispiro",
    }
)
_PARENT_TOKEN_ENDINGS = ("ane", "ene", "yne", "idine", "ole", "azole", "olane", "oxane", "benzene", "pyrrole")
_PREFIX_TOKEN_ENDINGS = (
    "amino",
    "ammonio",
    "azido",
    "chloro",
    "fluoro",
    "hydroxy",
    "imino",
    "iminio",
    "methyl",
    "oxo",
    "oxido",
    "phenyl",
    "sulfanyl",
)

_CHEMISTRY_GRAMMAR_TOKENS = frozenset(
    {
        "bis",
        "cis",
        "di",
        "e",
        "h",
        "lambda",
        "methyl",
        "n",
        "s",
        "tert",
        "tetra",
        "trans",
        "tri",
        "z",
    }
)

_ELEMENT_LOCANT_TOKENS = frozenset({"n", "o", "p", "s"})
