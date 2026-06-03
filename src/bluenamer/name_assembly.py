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
    def from_binding(cls, binding: NameAtomBinding) -> NameFragment:
        return cls(text=binding.term, bindings=(binding,))


@dataclass(frozen=True)
class NameRewriteRule:
    """A text rewrite with declared graph-ownership semantics."""

    name: str
    rewrite: Callable[[str], str]
    ownership: str = "preserve_all"
    source: str = "typed_rewrite"
    reason: str = ""
    pattern: str = ""
    replacement: str = ""
    match_kind: str = "callable"

    @classmethod
    def literal(
        cls,
        name: str,
        pattern: str,
        replacement: str,
        *,
        ownership: str = "replace_span",
        source: str = "typed_rewrite",
        reason: str = "",
    ) -> NameRewriteRule:
        """Build a rule whose replaced spans can carry graph ownership forward."""

        return cls(
            name=name,
            rewrite=lambda text: text.replace(pattern, replacement),
            ownership=ownership,
            source=source,
            reason=reason,
            pattern=pattern,
            replacement=replacement,
            match_kind="literal",
        )

    @classmethod
    def regex(
        cls,
        name: str,
        pattern: str,
        replacement: str,
        *,
        ownership: str = "regex_changed_span",
        source: str = "typed_rewrite",
        reason: str = "",
    ) -> NameRewriteRule:
        """Build a regex rule whose capture references carry source ownership."""

        compiled = re.compile(pattern)
        return cls(
            name=name,
            rewrite=lambda text: compiled.sub(replacement, text),
            ownership=ownership,
            source=source,
            reason=reason,
            pattern=pattern,
            replacement=replacement,
            match_kind="regex",
        )

    def apply_to_text(self, text: str) -> tuple[str, tuple[NameRewriteEdit, ...]]:
        """Apply the rule and return concrete edit spans when the rule supports them."""

        if self.match_kind == "literal" and self.pattern:
            return _apply_literal_rewrite(text, self.pattern, self.replacement, self.ownership)
        if self.match_kind == "regex" and self.pattern:
            return _apply_regex_rewrite(text, self.pattern, self.replacement, self.ownership)
        return self.rewrite(text), ()


@dataclass(frozen=True)
class NameRewriteEdit:
    """One concrete text edit made by a typed rewrite rule."""

    before_start: int
    before_end: int
    after_start: int
    after_end: int
    before_text: str
    after_text: str
    segments: tuple[NameRewriteSegment, ...] = ()


@dataclass(frozen=True)
class NameRewriteSegment:
    """One ownership-carrying subspan inside a concrete rewrite edit."""

    before_start: int
    before_end: int
    after_start: int
    after_end: int
    before_text: str
    after_text: str
    ownership: str
    group: str = ""


@dataclass(frozen=True)
class NameRewriteOperation:
    """A text rewrite that also transforms fragment/binding metadata."""

    name: str
    before: str
    after: str
    binding_count: int
    changed_binding_count: int
    ownership: str = "preserve_all"
    source: str = "typed_rewrite"
    token_count: int = 0
    changed_token_count: int = 0
    edits: tuple[NameRewriteEdit, ...] = ()

    @classmethod
    def apply(
        cls,
        name: str,
        bindings: tuple[NameAtomBinding, ...],
        *,
        rule: NameRewriteRule,
    ) -> tuple[str, tuple[NameAtomBinding, ...], NameRewriteOperation]:
        """Apply a rewrite to final text and every bound term."""

        rewritten_name, edits = rule.apply_to_text(name)
        rewritten_bindings = tuple(
            postprocess_name_atom_bindings(
                list(bindings),
                rule.rewrite,
                final_name=rewritten_name,
                rewrite_source=rule.source,
                rewritten_token_ownership=rule.ownership if edits else None,
            )
        )
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
                name=rule.name,
                before=name,
                after=rewritten_name,
                ownership=rule.ownership,
                source=rule.source,
                binding_count=len(bindings),
                changed_binding_count=changed_bindings,
                token_count=len(after_tokens),
                changed_token_count=changed_tokens,
                edits=edits,
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
    token_kind: str = "structural"
    ownership: str = "exact"
    confidence: str = "exact"
    source: str = "renderer"
    grammar_role: str = ""
    binding_key: str = ""


@dataclass(frozen=True)
class TokenBindingResolution:
    """How a final lexical token was associated with graph bindings."""

    binding_indices: tuple[int, ...] = ()
    token_kind: str = "structural"
    ownership: str = "exact"
    confidence: str = "exact"
    source: str = "renderer"
    grammar_role: str = ""
    binding_key: str = ""


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
    ) -> NameAssemblyResult:
        """Build a final assembly result while keeping binding metadata in sync."""

        return cls.from_rewrite_pipeline(
            raw_text,
            bindings,
            rewrites=(
                NameRewriteRule(
                    name="post_process_name",
                    rewrite=postprocess,
                    ownership="preserve_all",
                    source="typed_rewrite",
                    reason="Compatibility post-processing keeps graph ownership unless a later typed rule declares merge/split semantics.",
                ),
            ),
        )

    @classmethod
    def from_rewrite_pipeline(
        cls,
        raw_text: str,
        bindings: list[NameAtomBinding] | tuple[NameAtomBinding, ...],
        *,
        rewrites: tuple[NameRewriteRule | tuple[str, Callable[[str], str]], ...],
    ) -> NameAssemblyResult:
        """Build a final result by applying named rewrites to text and bindings."""

        text = raw_text
        binding_tuple = _ensure_emitted_token_bindings(tuple(bindings))
        token_spans = build_name_token_spans(text, binding_tuple)
        history: list[NameRewriteOperation] = []
        for raw_rule in rewrites:
            rule = _coerce_rewrite_rule(raw_rule)
            before_text = text
            text, binding_tuple, operation = NameRewriteOperation.apply(
                text,
                binding_tuple,
                rule=rule,
            )
            if operation.edits:
                token_spans = _rewrite_token_spans(before_text, text, token_spans, operation.edits)
            else:
                token_spans = build_name_token_spans(text, binding_tuple)
            history.append(operation)
        token_spans = _complete_token_spans(text, binding_tuple, token_spans)
        return cls(
            raw_text=raw_text,
            text=text,
            fragments=tuple(NameFragment.from_binding(binding) for binding in binding_tuple),
            bindings=binding_tuple,
            rewrite_history=tuple(history),
            token_spans=token_spans,
        )

    @classmethod
    def from_final_name(
        cls,
        text: str,
        bindings: list[NameAtomBinding] | tuple[NameAtomBinding, ...],
        *,
        rewrite_history: tuple[NameRewriteOperation, ...] = (),
    ) -> NameAssemblyResult:
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


def _coerce_rewrite_rule(rule: NameRewriteRule | tuple[str, Callable[[str], str]]) -> NameRewriteRule:
    if isinstance(rule, NameRewriteRule):
        return rule
    name, rewrite = rule
    return NameRewriteRule(name=name, rewrite=rewrite)


def _apply_literal_rewrite(
    text: str,
    pattern: str,
    replacement: str,
    ownership: str,
) -> tuple[str, tuple[NameRewriteEdit, ...]]:
    """Apply a literal rewrite while preserving before/after span coordinates."""

    if not pattern:
        return text, ()
    pieces: list[str] = []
    edits: list[NameRewriteEdit] = []
    cursor = 0
    after_cursor = 0
    while True:
        found = text.find(pattern, cursor)
        if found < 0:
            tail = text[cursor:]
            pieces.append(tail)
            after_cursor += len(tail)
            break
        unchanged = text[cursor:found]
        pieces.append(unchanged)
        after_cursor += len(unchanged)
        pieces.append(replacement)
        edits.append(
            NameRewriteEdit(
                before_start=found,
                before_end=found + len(pattern),
                after_start=after_cursor,
                after_end=after_cursor + len(replacement),
                before_text=pattern,
                after_text=replacement,
                segments=(
                    NameRewriteSegment(
                        before_start=found,
                        before_end=found + len(pattern),
                        after_start=after_cursor,
                        after_end=after_cursor + len(replacement),
                        before_text=pattern,
                        after_text=replacement,
                        ownership=ownership,
                    ),
                ),
            )
        )
        after_cursor += len(replacement)
        cursor = found + len(pattern)
    if not edits:
        return text, ()
    return "".join(pieces), tuple(edits)


def _apply_regex_rewrite(
    text: str,
    pattern: str,
    replacement: str,
    default_ownership: str,
) -> tuple[str, tuple[NameRewriteEdit, ...]]:
    """Apply a regex rewrite and map replacement capture references to source spans."""

    compiled = re.compile(pattern)
    replacement_parts = _parse_replacement_parts(replacement)
    pieces: list[str] = []
    edits: list[NameRewriteEdit] = []
    cursor = 0
    after_cursor = 0
    for match in compiled.finditer(text):
        unchanged = text[cursor : match.start()]
        pieces.append(unchanged)
        after_cursor += len(unchanged)
        rendered, segments = _render_regex_replacement_segments(
            match,
            replacement_parts,
            replacement,
            after_cursor,
            default_ownership,
        )
        pieces.append(rendered)
        edits.append(
            NameRewriteEdit(
                before_start=match.start(),
                before_end=match.end(),
                after_start=after_cursor,
                after_end=after_cursor + len(rendered),
                before_text=match.group(0),
                after_text=rendered,
                segments=segments,
            )
        )
        after_cursor += len(rendered)
        cursor = match.end()
    if not edits:
        return text, ()
    tail = text[cursor:]
    pieces.append(tail)
    return "".join(pieces), tuple(edits)


def _parse_replacement_parts(replacement: str) -> tuple[tuple[str, str | int], ...]:
    """Parse the subset of Python replacement syntax needed for ownership mapping."""

    parts: list[tuple[str, str | int]] = []
    literal: list[str] = []
    idx = 0
    while idx < len(replacement):
        char = replacement[idx]
        if char != "\\":
            literal.append(char)
            idx += 1
            continue
        if idx + 1 >= len(replacement):
            literal.append(char)
            idx += 1
            continue
        next_char = replacement[idx + 1]
        if next_char.isdigit():
            if literal:
                parts.append(("literal", "".join(literal)))
                literal = []
            parts.append(("group", int(next_char)))
            idx += 2
            continue
        if replacement.startswith("\\g<", idx):
            close = replacement.find(">", idx + 3)
            if close >= 0:
                if literal:
                    parts.append(("literal", "".join(literal)))
                    literal = []
                group_name = replacement[idx + 3 : close]
                parts.append(("group", int(group_name) if group_name.isdigit() else group_name))
                idx = close + 1
                continue
        literal.append(next_char)
        idx += 2
    if literal:
        parts.append(("literal", "".join(literal)))
    return tuple(parts)


def _render_regex_replacement_segments(
    match: re.Match,
    replacement_parts: tuple[tuple[str, str | int], ...],
    replacement: str,
    after_start: int,
    default_ownership: str,
) -> tuple[str, tuple[NameRewriteSegment, ...]]:
    if not replacement_parts:
        rendered = match.expand(replacement)
        return rendered, _literal_regex_segments(match, rendered, after_start, default_ownership, ())

    rendered_parts: list[str] = []
    segments: list[NameRewriteSegment] = []
    referenced_ranges: list[tuple[int, int]] = []
    seen_groups: set[str] = set()
    after_cursor = after_start
    for kind, value in replacement_parts:
        if kind == "group":
            group_text = match.group(value) or ""
            group_start, group_end = match.span(value)
            group_key = str(value)
            ownership = "cloned_capture" if group_key in seen_groups else "capture_reference"
            seen_groups.add(group_key)
            rendered_parts.append(group_text)
            if group_start >= 0 and group_text:
                referenced_ranges.append((group_start, group_end))
                segments.append(
                    NameRewriteSegment(
                        before_start=group_start,
                        before_end=group_end,
                        after_start=after_cursor,
                        after_end=after_cursor + len(group_text),
                        before_text=group_text,
                        after_text=group_text,
                        ownership=ownership,
                        group=group_key,
                    )
                )
            after_cursor += len(group_text)
            continue
        literal_text = str(value)
        rendered_parts.append(literal_text)
        if literal_text:
            segments.extend(
                _literal_regex_segments(match, literal_text, after_cursor, default_ownership, tuple(referenced_ranges))
            )
            after_cursor += len(literal_text)
    rendered = "".join(rendered_parts)
    segments.extend(_absorbed_regex_segments(match, tuple(referenced_ranges), after_start))
    return rendered, tuple(segments)


def _literal_regex_segments(
    match: re.Match,
    literal_text: str,
    after_start: int,
    ownership: str,
    referenced_ranges: tuple[tuple[int, int], ...],
) -> tuple[NameRewriteSegment, ...]:
    source_ranges = _unreferenced_match_ranges(match.start(), match.end(), referenced_ranges)
    if not source_ranges:
        source_ranges = ((match.start(), match.end()),)
    first_start, first_end = source_ranges[0]
    return (
        NameRewriteSegment(
            before_start=first_start,
            before_end=first_end,
            after_start=after_start,
            after_end=after_start + len(literal_text),
            before_text=match.string[first_start:first_end],
            after_text=literal_text,
            ownership=ownership,
        ),
    )


def _absorbed_regex_segments(
    match: re.Match,
    referenced_ranges: tuple[tuple[int, int], ...],
    after_start: int,
) -> tuple[NameRewriteSegment, ...]:
    return tuple(
        NameRewriteSegment(
            before_start=start,
            before_end=end,
            after_start=after_start,
            after_end=after_start,
            before_text=match.string[start:end],
            after_text="",
            ownership="absorbed",
        )
        for start, end in _unreferenced_match_ranges(match.start(), match.end(), referenced_ranges)
        if start < end
    )


def _unreferenced_match_ranges(
    start: int,
    end: int,
    referenced_ranges: tuple[tuple[int, int], ...],
) -> tuple[tuple[int, int], ...]:
    ranges = sorted((max(start, item_start), min(end, item_end)) for item_start, item_end in referenced_ranges)
    ranges = [(item_start, item_end) for item_start, item_end in ranges if item_start < item_end]
    if not ranges:
        return ((start, end),)
    result: list[tuple[int, int]] = []
    cursor = start
    for item_start, item_end in ranges:
        if cursor < item_start:
            result.append((cursor, item_start))
        cursor = max(cursor, item_end)
    if cursor < end:
        result.append((cursor, end))
    return tuple(result)


def _rewrite_token_spans(
    before_text: str,
    after_text: str,
    token_spans: tuple[NameTokenSpan, ...],
    edits: tuple[NameRewriteEdit, ...],
) -> tuple[NameTokenSpan, ...]:
    rewritten: list[NameTokenSpan] = []
    for token in token_spans:
        edit = _covering_edit(token, edits)
        if edit is None:
            rewritten.append(_shift_token_span(token, _rewrite_delta_before(token.start, edits), after_text))
            continue
        rewritten.extend(_token_spans_from_edit_segments(token, edit, after_text))
    rewritten.sort(key=lambda token: (token.start, token.end, token.text))
    return tuple(rewritten)


def _covering_edit(token: NameTokenSpan, edits: tuple[NameRewriteEdit, ...]) -> NameRewriteEdit | None:
    for edit in edits:
        if _spans_overlap(token.start, token.end, edit.before_start, edit.before_end):
            return edit
    return None


def _rewrite_delta_before(position: int, edits: tuple[NameRewriteEdit, ...]) -> int:
    delta = 0
    for edit in edits:
        if edit.before_end <= position:
            delta += (edit.after_end - edit.after_start) - (edit.before_end - edit.before_start)
    return delta


def _shift_token_span(token: NameTokenSpan, delta: int, after_text: str) -> NameTokenSpan:
    start = token.start + delta
    end = token.end + delta
    return _copy_token_span(token, start, end, after_text[start:end], token.ownership, token.source, token.confidence)


def _token_spans_from_edit_segments(
    token: NameTokenSpan,
    edit: NameRewriteEdit,
    after_text: str,
) -> tuple[NameTokenSpan, ...]:
    spans: list[NameTokenSpan] = []
    matched_segment = False
    for segment in edit.segments:
        if not _spans_overlap(token.start, token.end, segment.before_start, segment.before_end):
            continue
        matched_segment = True
        if segment.after_start == segment.after_end:
            continue
        spans.append(
            _copy_token_span(
                token,
                segment.after_start,
                segment.after_end,
                after_text[segment.after_start : segment.after_end],
                segment.ownership,
                "typed_rewrite",
                "derived",
            )
        )
    if spans:
        return tuple(spans)
    if matched_segment:
        return ()
    return (
        _copy_token_span(
            token,
            edit.after_start,
            edit.after_end,
            after_text[edit.after_start : edit.after_end],
            "regex_changed_span",
            "typed_rewrite",
            "derived",
        ),
    )


def _copy_token_span(
    token: NameTokenSpan,
    start: int,
    end: int,
    text: str,
    ownership: str,
    source: str,
    confidence: str,
) -> NameTokenSpan:
    return NameTokenSpan(
        text=text,
        start=start,
        end=end,
        binding_indices=token.binding_indices,
        atom_ids=token.atom_ids,
        bond_ids=token.bond_ids,
        charge_atom_ids=token.charge_atom_ids,
        locants=token.locants,
        token_kind=token.token_kind,
        ownership=ownership,
        confidence=confidence,
        source=source,
        grammar_role=token.grammar_role,
        binding_key=token.binding_key,
    )


def _complete_token_spans(
    text: str,
    bindings: tuple[NameAtomBinding, ...],
    propagated: tuple[NameTokenSpan, ...],
) -> tuple[NameTokenSpan, ...]:
    built = build_name_token_spans(text, bindings)
    complete = list(propagated)
    covered_positions = {pos for token in propagated if token.binding_indices for pos in range(token.start, token.end)}
    for token in built:
        token_positions = set(range(token.start, token.end))
        if token_positions and token_positions <= covered_positions:
            continue
        complete.append(token)
    complete.sort(key=lambda token: (token.start, token.end, token.text))
    return tuple(complete)


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
            "token_kind": token.token_kind,
            "ownership": token.ownership,
            "confidence": token.confidence,
            "source": token.source,
            "grammar_role": token.grammar_role,
            "binding_key": token.binding_key,
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
        idx for idx in explicit_component_atoms if idx in mol.atoms and mol.atoms[idx].charge != 0
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
        raise FinalAssemblyAuditError(
            f"Generated name {result.text!r} failed final metadata audit: {'; '.join(audit.errors)}"
        )


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
        if binding_indices:
            resolution = TokenBindingResolution(
                binding_indices=tuple(sorted(binding_indices)),
                ownership="preserves_binding",
                confidence="derived",
                source="direct_text_match",
                grammar_role="direct_binding_term",
            )
        else:
            resolution = _fallback_token_binding_resolution(token_text, start, end, text, bindings)
        tokens.append(_token_span_from_binding_indices(token_text, start, end, resolution, bindings))
    return tuple(tokens)


def _native_token_spans(
    text: str,
    bindings: tuple[NameAtomBinding, ...],
) -> list[tuple[int, int, int, NameTokenBinding]]:
    spans: list[tuple[int, int, int, NameTokenBinding]] = []
    search_text = text.lower()
    for binding_idx, binding in enumerate(bindings):
        if binding.stage == "hydro" and binding.role == "indicated_hydrogen":
            spans.extend(_indicated_hydrogen_native_token_spans(search_text, binding_idx, binding))
            continue
        if not any(_is_locant_search_token(token_binding) for token_binding in binding.emitted_tokens):
            ordered_spans = _ordered_native_token_spans(text, search_text, binding_idx, binding)
            if ordered_spans:
                spans.extend(ordered_spans)
                continue
        for token_binding in binding.emitted_tokens:
            token = token_binding.text.strip().lower()
            if not _native_token_is_searchable(token):
                continue
            pos = 0
            for found in _native_token_occurrences(text, search_text, token_binding, pos):
                spans.append((found, found + len(token), binding_idx, token_binding))
                pos = found + 1
    return spans


def _indicated_hydrogen_native_token_spans(
    search_text: str,
    binding_idx: int,
    binding: NameAtomBinding,
) -> list[tuple[int, int, int, NameTokenBinding]]:
    locant_token = next((token for token in binding.emitted_tokens if token.token_kind == "locant"), None)
    hydrogen_token = next((token for token in binding.emitted_tokens if token.token_kind == "hydro"), None)
    if locant_token is None or hydrogen_token is None:
        return []
    locant_text = locant_token.text.lower()
    pattern = f"{locant_text}h"
    spans: list[tuple[int, int, int, NameTokenBinding]] = []
    pos = 0
    while True:
        found = search_text.find(pattern, pos)
        if found < 0:
            break
        spans.append((found, found + len(locant_text), binding_idx, locant_token))
        spans.append((found + len(locant_text), found + len(pattern), binding_idx, hydrogen_token))
        pos = found + 1
    return spans


def _ordered_native_token_spans(
    text: str,
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
        found = next(iter(_native_token_occurrences(text, search_text, token_binding, cursor)), -1)
        if found < 0:
            return []
        placed.append((found, found + len(token), binding_idx, token_binding))
        cursor = found + len(token)
    return placed


def _native_token_occurrences(
    text: str,
    search_text: str,
    token_binding: NameTokenBinding,
    start: int,
) -> tuple[int, ...]:
    raw_token = token_binding.text.strip()
    token = raw_token.lower()
    if _is_locant_search_token(token_binding):
        return tuple(
            match.start()
            for match in re.finditer(re.escape(raw_token), text)
            if match.start() >= start and _is_standalone_locant_span(text, match.start(), match.end())
        )
    positions: list[int] = []
    pos = start
    while True:
        found = search_text.find(token, pos)
        if found < 0:
            break
        positions.append(found)
        pos = found + 1
    return tuple(positions)


def _is_locant_search_token(token_binding: NameTokenBinding) -> bool:
    return token_binding.token_kind == "locant" and bool(
        re.fullmatch(r"(?:[0-9]+|N|O|P|S)(?:,(?:[0-9]+|N|O|P|S))*", token_binding.text)
    )


def _is_standalone_locant_span(text: str, start: int, end: int) -> bool:
    before = text[start - 1] if start > 0 else ""
    after = text[end] if end < len(text) else ""
    return (not before or before in "-,( ") and (not after or after in "-,)' ")


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


def _fallback_token_binding_resolution(
    token: str,
    start: int,
    end: int,
    text: str,
    bindings: tuple[NameAtomBinding, ...],
) -> TokenBindingResolution:
    token_norm = _normalise_name_text(token)
    indices: set[int] = set()
    if token_norm.isdigit() or "," in token_norm:
        indices.update(_locant_binding_indices(token_norm, bindings))
        if indices:
            return TokenBindingResolution(
                tuple(sorted(indices)),
                token_kind="locant",
                ownership="exact",
                confidence="derived",
                source="locant_fallback",
                grammar_role="locant",
            )
    if not indices:
        indices.update(_charge_suffix_binding_indices(token_norm, start, text, bindings))
        if indices:
            return TokenBindingResolution(
                tuple(sorted(indices)),
                token_kind="charge",
                ownership="exact",
                confidence="derived",
                source="charge_suffix_fallback",
                grammar_role="charge",
            )
    if not indices:
        indices.update(_indicated_hydrogen_binding_indices(token_norm, start, text, bindings))
        if indices:
            return TokenBindingResolution(
                tuple(sorted(indices)),
                token_kind="hydro",
                ownership="locanted_hydro",
                confidence="derived",
                source="indicated_hydrogen_fallback",
                grammar_role="indicated_hydrogen",
            )
    if not indices:
        indices.update(_dihydro_locant_binding_indices(token_norm, start, end, text, bindings))
        if indices:
            return TokenBindingResolution(
                tuple(sorted(indices)),
                token_kind="hydro",
                ownership="locanted_hydro",
                confidence="derived",
                source="dihydro_locant_fallback",
                grammar_role="dihydro",
            )
    if not indices:
        indices.update(_primed_component_locant_binding_indices(token_norm, start, end, text, bindings))
        if indices:
            return TokenBindingResolution(
                tuple(sorted(indices)),
                token_kind="locant",
                ownership="component_locant",
                confidence="derived",
                source="primed_component_locant_fallback",
                grammar_role="component_locant",
            )
    if not indices:
        indices.update(_role_token_binding_indices(token_norm, bindings))
        if indices:
            return TokenBindingResolution(
                tuple(sorted(indices)),
                token_kind="structural",
                ownership="role_alias",
                confidence="derived",
                source="role_alias_fallback",
                grammar_role=token_norm,
            )
    if not indices:
        indices.update(_retained_alias_context_binding_indices(token_norm, start, end, text, bindings))
        if indices:
            return TokenBindingResolution(
                tuple(sorted(indices)),
                token_kind="retained_alias",
                ownership="retained_alias_context",
                confidence="derived",
                source="retained_alias_context",
                grammar_role="retained_alias",
            )
    if not indices and _is_structural_suffix_token(token_norm, text, start, end):
        indices.update(_suffix_binding_indices(bindings))
        if indices:
            return TokenBindingResolution(
                tuple(sorted(indices)),
                token_kind="suffix",
                ownership="stage_alias",
                confidence="derived",
                source="stage_fallback",
                grammar_role="suffix",
            )
    if not indices and _is_parent_like_token(token_norm):
        indices.update(_parent_binding_indices(bindings))
        if indices:
            return TokenBindingResolution(
                tuple(sorted(indices)),
                token_kind="parent",
                ownership="stage_alias",
                confidence="derived",
                source="stage_fallback",
                grammar_role="parent",
            )
    if not indices and _is_prefix_like_token(token_norm):
        indices.update(_prefix_binding_indices(bindings))
        if indices:
            return TokenBindingResolution(
                tuple(sorted(indices)),
                token_kind="prefix",
                ownership="stage_alias",
                confidence="derived",
                source="stage_fallback",
                grammar_role="prefix",
            )
    if not indices and token_norm in _NON_STRUCTURAL_GRAMMAR_TOKENS:
        return TokenBindingResolution(
            tuple(sorted(_graph_bearing_binding_indices(bindings))),
            token_kind="grammar",
            ownership="grammar_scope",
            confidence="derived",
            source="grammar_token",
            grammar_role=token_norm,
        )
    if not indices:
        operation_scope = _operation_scope_binding_indices(token_norm, start, end, text, bindings)
        if operation_scope:
            return TokenBindingResolution(
                tuple(sorted(operation_scope)),
                token_kind=_operation_scope_token_kind(operation_scope, bindings),
                ownership="operation_scope",
                confidence="derived",
                source="operation_trace",
                grammar_role="operation_scope",
            )
    if not indices and (_is_plausible_chemical_token(token_norm) or len(_graph_bearing_binding_indices(bindings)) > 1):
        indices.update(_graph_bearing_binding_indices(bindings))
        if indices:
            return TokenBindingResolution(
                tuple(sorted(indices)),
                token_kind="structural",
                ownership="ambiguous",
                confidence="fallback",
                source="broad_fallback",
                grammar_role="chemical_token",
            )
    return TokenBindingResolution(
        token_kind="structural",
        ownership="unbound",
        confidence="fallback",
        source="unresolved",
        grammar_role=token_norm,
    )


def _locant_binding_indices(token: str, bindings: tuple[NameAtomBinding, ...]) -> set[int]:
    locants = {_normalise_locant_token(part) for part in re.split(r",", token) if part}
    if not locants:
        return set()
    return {
        idx
        for idx, binding in enumerate(bindings)
        if binding.locants and locants & {_normalise_locant_token(str(locant)) for locant in binding.locants}
    }


def _normalise_locant_token(locant: str) -> str:
    return locant.strip().strip("'").strip('"')


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
        idx for idx in charge_indices if locants <= {str(locant).strip("'") for locant in bindings[idx].locants}
    } or charge_indices


def _indicated_hydrogen_binding_indices(
    token: str,
    start: int,
    text: str,
    bindings: tuple[NameAtomBinding, ...],
) -> set[int]:
    if token != "h":
        return set()
    locant_match = re.search(r"(\d+(?:,\d+)*)$", text[:start])
    if not locant_match:
        return set()
    locants = set(locant_match.group(1).split(","))
    return {
        idx
        for idx, binding in enumerate(bindings)
        if binding.locants and locants <= {str(locant).strip("'") for locant in binding.locants}
    } or _parent_binding_indices(bindings)


def _dihydro_locant_binding_indices(
    token: str,
    start: int,
    end: int,
    text: str,
    bindings: tuple[NameAtomBinding, ...],
) -> set[int]:
    if not re.fullmatch(r"\d+(?:,\d+)+", token):
        return set()
    if text[end : end + len("-dihydro")].lower() != "-dihydro":
        return set()
    if start > 0 and text[start - 1] not in "-( ":
        return set()
    return {
        idx
        for idx, binding in enumerate(bindings)
        if binding.stage in {"parent", "replacement"} and (binding.atom_ids or binding.bond_ids)
    } or _parent_binding_indices(bindings)


def _primed_component_locant_binding_indices(
    token: str,
    start: int,
    end: int,
    text: str,
    bindings: tuple[NameAtomBinding, ...],
) -> set[int]:
    """Bind split numeric tokens such as ``3'`` to the primed component scope."""

    if not re.fullmatch(r"\d+(?:,\d+)*", token):
        return set()
    if not _is_followed_by_prime_marker(text, end):
        return set()
    component_scope = _primed_component_scope_binding_indices(bindings)
    if not component_scope:
        return set()
    following_word = _following_alpha_token(text, end)
    if following_word is None:
        return component_scope
    word_start, word_end, word = following_word
    word_scope = _operation_scope_binding_indices(word.lower(), word_start, word_end, text, bindings)
    return (word_scope & component_scope) or component_scope


def _role_token_binding_indices(token: str, bindings: tuple[NameAtomBinding, ...]) -> set[int]:
    indices: set[int] = set()
    for idx, binding in enumerate(bindings):
        role = binding.role.replace("_", "").lower()
        if token == role or token in _ROLE_TOKEN_ALIASES.get(role, ()):
            indices.add(idx)
    return indices


def _retained_alias_context_binding_indices(
    token: str,
    start: int,
    end: int,
    text: str,
    bindings: tuple[NameAtomBinding, ...],
) -> set[int]:
    """Bind retained-alias words to an adjacent graph-bound role word."""

    if not token.isalpha() or not _is_free_text_word(text, start, end):
        return set()
    following_word = _following_alpha_token(text, end)
    if following_word is None:
        return set()
    _, _, following_token = following_word
    role_scope = _role_token_binding_indices(following_token.lower(), bindings)
    if role_scope:
        return role_scope
    return set()


def _suffix_binding_indices(bindings: tuple[NameAtomBinding, ...]) -> set[int]:
    return {idx for idx, binding in enumerate(bindings) if binding.stage == "suffix"}


def _parent_binding_indices(bindings: tuple[NameAtomBinding, ...]) -> set[int]:
    return {idx for idx, binding in enumerate(bindings) if binding.stage == "parent"}


def _prefix_binding_indices(bindings: tuple[NameAtomBinding, ...]) -> set[int]:
    return {idx for idx, binding in enumerate(bindings) if binding.stage in {"prefix", "modifier", "replacement"}}


def _operation_scope_binding_indices(
    token: str,
    start: int,
    end: int,
    text: str,
    bindings: tuple[NameAtomBinding, ...],
) -> set[int]:
    """Return a local operation scope for unmatched renderer text."""

    if not token or token in _NON_STRUCTURAL_GRAMMAR_TOKENS:
        return set()
    if _is_inside_parentheses(text, start, end):
        prefix_scope = _prefix_binding_indices(bindings)
        if len(prefix_scope) == 1:
            return prefix_scope
    previous_locant = _previous_locant(text, start)
    if previous_locant:
        locanted = _locant_binding_indices(previous_locant, bindings)
        if locanted:
            if _looks_like_parent_morphology(token, text, start, end):
                return locanted | _parent_binding_indices(bindings)
            return locanted
        if _previous_locant_is_primed(text, start):
            component_scope = _primed_component_scope_binding_indices(bindings)
            if component_scope:
                return component_scope
    if _is_before_suffix_boundary(text, end):
        suffix_scope = _suffix_binding_indices(bindings)
        if len(suffix_scope) == 1:
            return suffix_scope
    parent_scope = _parent_binding_indices(bindings) | {
        idx for idx, binding in enumerate(bindings) if binding.stage == "replacement"
    }
    if parent_scope and _looks_like_parent_morphology(token, text, start, end):
        return parent_scope
    graph_scope = _graph_bearing_binding_indices(bindings)
    if len(graph_scope) == 1 and not _is_free_text_word(text, start, end):
        return graph_scope
    return set()


def _operation_scope_token_kind(indices: set[int], bindings: tuple[NameAtomBinding, ...]) -> str:
    stages = {bindings[idx].stage for idx in indices}
    if len(stages) == 1:
        stage = next(iter(stages))
        if stage in {"parent", "prefix", "suffix", "replacement", "unsaturation", "charge"}:
            return stage
    return "structural"


def _is_inside_parentheses(text: str, start: int, end: int) -> bool:
    return text.rfind("(", 0, start) > text.rfind(")", 0, start) and text.find(")", end) >= 0


def _previous_locant(text: str, start: int) -> str:
    match = re.search(r"(?:(?:^|[-(])((?:\d+|N|O|P|S)(?:,(?:\d+|N|O|P|S))*)(?:'+)?-)$", text[:start])
    return match.group(1) if match else ""


def _previous_locant_is_primed(text: str, start: int) -> bool:
    return bool(re.search(r"(?:(?:^|[-(])(?:\d+|N|O|P|S)(?:,(?:\d+|N|O|P|S))*'+-)$", text[:start]))


def _is_followed_by_prime_marker(text: str, end: int) -> bool:
    return end < len(text) and text[end] == "'"


def _following_alpha_token(text: str, end: int) -> tuple[int, int, str] | None:
    cursor = end
    while cursor < len(text) and text[cursor] in "'- ":
        cursor += 1
    match = re.match(r"[A-Za-z]+", text[cursor:])
    if match is None:
        return None
    return cursor, cursor + match.end(), match.group(0)


def _primed_component_scope_binding_indices(bindings: tuple[NameAtomBinding, ...]) -> set[int]:
    """Return bindings that own non-primary component text in primed locant grammar."""

    prefix_scope = {
        idx
        for idx, binding in enumerate(bindings)
        if binding.stage in {"prefix", "modifier"} and (binding.atom_ids or binding.bond_ids)
    }
    if prefix_scope:
        return prefix_scope
    return {
        idx
        for idx, binding in enumerate(bindings)
        if binding.stage == "replacement" and (binding.atom_ids or binding.bond_ids)
    }


def _is_before_suffix_boundary(text: str, end: int) -> bool:
    return end < len(text) and text[end] == "-" and bool(re.match(r"-\d", text[end:]))


def _looks_like_parent_morphology(token: str, text: str, start: int, end: int) -> bool:
    if not token.isalpha():
        return False
    previous_char = text[start - 1] if start > 0 else ""
    next_char = text[end] if end < len(text) else ""
    if (previous_char and previous_char in "([-") or (next_char and next_char in "])-"):
        return True
    return any(token.endswith(ending) for ending in _PARENT_TOKEN_ENDINGS) or token.endswith(("olan", "idin", "iran"))


def _is_free_text_word(text: str, start: int, end: int) -> bool:
    before = text[start - 1] if start > 0 else ""
    after = text[end] if end < len(text) else ""
    return (not before or before.isspace()) and (not after or after.isspace())


def _graph_bearing_binding_indices(bindings: tuple[NameAtomBinding, ...]) -> set[int]:
    """Return conservative owners for tokens not yet assignable to one role."""

    return {
        idx for idx, binding in enumerate(bindings) if binding.atom_ids or binding.bond_ids or binding.charge_atom_ids
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
    resolution: TokenBindingResolution,
    bindings: tuple[NameAtomBinding, ...],
) -> NameTokenSpan:
    atoms: set[int] = set()
    bonds: set[int] = set()
    charges: set[int] = set()
    locants: list[str] = []
    for idx in resolution.binding_indices:
        binding = bindings[idx]
        atoms.update(binding.atom_ids)
        bonds.update(binding.bond_ids)
        charges.update(binding.charge_atom_ids)
        locants.extend(str(locant) for locant in binding.locants)
    return NameTokenSpan(
        text=text,
        start=start,
        end=end,
        binding_indices=resolution.binding_indices,
        atom_ids=frozenset(atoms),
        bond_ids=frozenset(bonds),
        charge_atom_ids=frozenset(charges),
        locants=tuple(locants),
        token_kind=resolution.token_kind,
        ownership=resolution.ownership,
        confidence=resolution.confidence,
        source=resolution.source,
        grammar_role=resolution.grammar_role,
        binding_key=resolution.binding_key,
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
    ordered_keys = sorted(grouped_matches)
    cursor = start
    previous_key: tuple[int, int] | None = None
    for index, (clipped_start, clipped_end) in enumerate(ordered_keys):
        if cursor < clipped_start:
            gap_text = text[cursor:clipped_start]
            if _normalise_name_text(gap_text):
                next_key = (clipped_start, clipped_end)
                spans.append(
                    _token_span_for_native_gap(
                        gap_text,
                        cursor,
                        clipped_start,
                        text,
                        _adjacent_native_gap_matches(grouped_matches, previous_key, next_key),
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
        previous_key = (clipped_start, clipped_end)
    if cursor < end:
        gap_text = text[cursor:end]
        if _normalise_name_text(gap_text):
            spans.append(
                _token_span_for_native_gap(
                    gap_text,
                    cursor,
                    end,
                    text,
                    _adjacent_native_gap_matches(grouped_matches, previous_key, None),
                )
            )
    return spans


def _adjacent_native_gap_matches(
    grouped_matches: dict[tuple[int, int], list[tuple[int, NameTokenBinding]]],
    previous_key: tuple[int, int] | None,
    next_key: tuple[int, int] | None,
) -> list[tuple[int, NameTokenBinding]]:
    matches: list[tuple[int, NameTokenBinding]] = []
    if previous_key is not None:
        matches.extend(grouped_matches.get(previous_key, ()))
    if next_key is not None:
        matches.extend(grouped_matches.get(next_key, ()))
    return matches


def _token_span_for_native_gap(
    text: str,
    start: int,
    end: int,
    full_text: str,
    adjacent_matches: list[tuple[int, NameTokenBinding]] | None = None,
) -> NameTokenSpan:
    token_norm = _normalise_name_text(text)
    if token_norm in _COMPOUND_CONNECTIVE_TOKENS and adjacent_matches:
        bridged = _token_span_from_native_binding_group(text, start, end, adjacent_matches)
        return NameTokenSpan(
            text=bridged.text,
            start=bridged.start,
            end=bridged.end,
            binding_indices=bridged.binding_indices,
            atom_ids=bridged.atom_ids,
            bond_ids=bridged.bond_ids,
            charge_atom_ids=bridged.charge_atom_ids,
            locants=bridged.locants,
            token_kind="grammar",
            ownership="multiplier_scope",
            confidence="derived",
            source="compound_gap_bridge",
            grammar_role=token_norm,
            binding_key=bridged.binding_key,
        )
    if token_norm in _NON_STRUCTURAL_GRAMMAR_TOKENS or token_norm in _COMPOUND_CONNECTIVE_TOKENS:
        return NameTokenSpan(
            text=text,
            start=start,
            end=end,
            token_kind="grammar",
            ownership="grammar_scope",
            confidence="derived",
            source="compound_gap_token",
            grammar_role=token_norm,
        )
    if adjacent_matches:
        bridged = _token_span_from_native_binding_group(text, start, end, adjacent_matches)
        return NameTokenSpan(
            text=bridged.text,
            start=bridged.start,
            end=bridged.end,
            binding_indices=bridged.binding_indices,
            atom_ids=bridged.atom_ids,
            bond_ids=bridged.bond_ids,
            charge_atom_ids=bridged.charge_atom_ids,
            locants=bridged.locants,
            token_kind=bridged.token_kind,
            ownership="morphology_gap",
            confidence="derived",
            source="compound_gap_bridge",
            grammar_role=bridged.grammar_role or token_norm,
            binding_key=bridged.binding_key,
        )
    return NameTokenSpan(
        text=text,
        start=start,
        end=end,
        token_kind="structural",
        ownership="unbound",
        confidence="fallback",
        source="compound_gap_unresolved",
        grammar_role=token_norm,
    )


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
    token_kinds: list[str] = []
    ownerships: list[str] = []
    confidences: list[str] = []
    sources: list[str] = []
    grammar_roles: list[str] = []
    binding_keys: list[str] = []
    for binding_idx, token_binding in matches:
        binding_indices.add(binding_idx)
        atoms.update(token_binding.atom_ids)
        bonds.update(token_binding.bond_ids)
        charges.update(token_binding.charge_atom_ids)
        locants.extend(str(locant) for locant in token_binding.locants)
        token_kinds.append(token_binding.token_kind)
        ownerships.append(token_binding.ownership)
        confidences.append(token_binding.confidence)
        sources.append(token_binding.source)
        if token_binding.grammar_role:
            grammar_roles.append(token_binding.grammar_role)
        if token_binding.binding_key:
            binding_keys.append(token_binding.binding_key)
    return NameTokenSpan(
        text=text,
        start=start,
        end=end,
        binding_indices=tuple(sorted(binding_indices)),
        atom_ids=frozenset(atoms),
        bond_ids=frozenset(bonds),
        charge_atom_ids=frozenset(charges),
        locants=tuple(locants),
        token_kind=_collapse_metadata(token_kinds, default="structural"),
        ownership=_collapse_metadata(ownerships, default="exact"),
        confidence=_weakest_confidence(confidences),
        source=_collapse_metadata(sources, default="renderer"),
        grammar_role=_collapse_metadata(grammar_roles, default=""),
        binding_key=_collapse_metadata(binding_keys, default=""),
    )


def _collapse_metadata(values: list[str], *, default: str) -> str:
    values = [value for value in values if value]
    if not values:
        return default
    unique = sorted(set(values))
    return unique[0] if len(unique) == 1 else "mixed"


def _weakest_confidence(values: list[str]) -> str:
    order = {"exact": 0, "derived": 1, "fallback": 2}
    values = [value for value in values if value]
    if not values:
        return "exact"
    return max(values, key=lambda value: order.get(value, 99))


def _unbound_name_tokens(result: NameAssemblyResult) -> list[dict]:
    return [
        {"text": token.text, "start": token.start, "end": token.end}
        for token in result.token_spans
        if not token.binding_indices and _token_requires_graph_binding(token)
    ]


def _token_requires_graph_binding(token: NameTokenSpan) -> bool:
    if token.token_kind == "grammar" or token.ownership == "grammar_scope":
        return False
    return bool(_normalise_name_text(token.text))


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
    if any(
        _term_occurs_in_final_name(_normalise_name_text(alias), final_text)
        for alias in _ROLE_TOKEN_ALIASES.get(role, ())
    ):
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

_NON_STRUCTURAL_GRAMMAR_TOKENS = frozenset(
    {
        "bis",
        "cis",
        "di",
        "dihydro",
        "e",
        "lambda",
        "tert",
        "tetra",
        "trans",
        "tri",
        "tris",
        "z",
    }
)

_COMPOUND_CONNECTIVE_TOKENS = frozenset({"di", "tri", "tetra"})

_ELEMENT_LOCANT_TOKENS = frozenset({"n", "o", "p", "s"})
