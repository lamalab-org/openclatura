"""Natural-language descriptions backed by naming metadata.

The describer is intentionally not a second naming engine. It calls the
normal namer with tracing enabled and renders prose from the resulting
decision trace and substituent tree. Optional debug mode also exposes
graph-bound token spans. That keeps the description aligned with the name
that was actually generated without showing experimental token metadata by
default.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from .engine import DEFAULT_NAMING_ENGINE, NamingRequest, NamingResult, _extract_rules_hit
from .molecule import TracePhase, TraceStep


@dataclass(frozen=True)
class DescribedComponent:
    """One phase-tagged human-readable line."""

    phase: str
    text: str


@dataclass(frozen=True)
class DescriptionTokenSummary:
    """Aggregate view of final-name token binding metadata."""

    total: int = 0
    by_kind: dict[str, int] = field(default_factory=dict)
    by_ownership: dict[str, int] = field(default_factory=dict)
    by_confidence: dict[str, int] = field(default_factory=dict)
    by_source: dict[str, int] = field(default_factory=dict)
    fallback_tokens: tuple[dict, ...] = ()

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "by_kind": dict(self.by_kind),
            "by_ownership": dict(self.by_ownership),
            "by_confidence": dict(self.by_confidence),
            "by_source": dict(self.by_source),
            "fallback_tokens": list(self.fallback_tokens),
        }


@dataclass(frozen=True)
class Description:
    """Result of :func:`describe`. ``str(d)`` returns the rendered prose."""

    smiles: str
    name: str
    paragraphs: tuple[str, ...]
    components: tuple[DescribedComponent, ...] = ()
    rules_hit: tuple[str, ...] = ()
    rule_hints: tuple[str, ...] = ()
    token_summary: DescriptionTokenSummary = field(default_factory=DescriptionTokenSummary)
    token_spans: tuple[dict, ...] = ()
    substituent_tree: tuple[dict, ...] = ()
    debugging_tokens: bool = False

    def __str__(self) -> str:
        return "\n\n".join(self.paragraphs)

    @property
    def summary(self) -> str:
        return self.paragraphs[0] if self.paragraphs else ""

    def to_dict(self, *, debugging_tokens: bool | None = None) -> dict:
        include_tokens = self.debugging_tokens if debugging_tokens is None else debugging_tokens
        payload = {
            "smiles": self.smiles,
            "name": self.name,
            "summary": self.summary,
            "paragraphs": list(self.paragraphs),
            "components": [{"phase": c.phase, "text": c.text} for c in self.components],
            "rules_hit": list(self.rules_hit),
            "rule_hints": list(self.rule_hints),
            "substituent_tree": list(self.substituent_tree),
        }
        if include_tokens:
            payload["token_summary"] = self.token_summary.to_dict()
            payload["token_spans"] = list(self.token_spans)
        return payload


def describe(
    smiles: str,
    *,
    debugging_tokens: bool = False,
) -> Description:
    """Render a deterministic metadata-backed description for ``smiles``."""

    result = DEFAULT_NAMING_ENGINE.run(NamingRequest(smiles=smiles, include_trace=True))
    token_spans = tuple(_collect_token_spans(result)) if debugging_tokens else ()
    token_summary = _summarize_tokens(token_spans) if debugging_tokens else DescriptionTokenSummary()

    if result.error:
        summary = f"The structure {smiles} could not be named: {result.error}."
    elif not result.name:
        summary = f"The structure {smiles} could not be named by the current ruleset."
    else:
        summary = f"The molecule {smiles} is named **{result.name}**."

    components = tuple(_decision_components(result))
    paragraphs: list[str] = [summary]
    if components:
        paragraphs.append("\n".join(c.text for c in components))

    tree_lines = _substituent_tree_lines(result.substituent_tree)
    if tree_lines:
        paragraphs.append("Component and substituent structure:\n" + "\n".join(tree_lines))

    token_lines = _token_binding_lines(token_summary, token_spans) if debugging_tokens else []
    if debugging_tokens and token_lines:
        paragraphs.append("Name-token graph bindings:\n" + "\n".join(token_lines))

    seg_lines = _trace_segment_lines(result.trace_segments, result.substituent_tree)
    if seg_lines:
        paragraphs.append("Name pieces contributed by the trace:\n" + "\n".join(seg_lines))

    rules, hints = _extract_rules_hit(result.trace_segments)
    if rules:
        paragraphs.append("IUPAC Blue Book rules applied: " + ", ".join(rules) + ".")

    return Description(
        smiles=smiles,
        name=result.name,
        paragraphs=tuple(paragraphs),
        components=components,
        rules_hit=rules,
        rule_hints=hints,
        token_summary=token_summary,
        token_spans=token_spans,
        substituent_tree=tuple(result.substituent_tree),
        debugging_tokens=debugging_tokens,
    )


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def _decision_components(result: NamingResult) -> list[DescribedComponent]:
    components: list[DescribedComponent] = []
    seen: set[tuple[str, str]] = set()
    for step in result.decisions:
        text = _render_decision_step(step)
        if not text:
            continue
        key = (step.phase.value, text)
        if key in seen:
            continue
        seen.add(key)
        components.append(DescribedComponent(phase=step.phase.value, text=text))
    return components


def _render_decision_step(step: TraceStep) -> str | None:
    if step.phase == TracePhase.PARSE:
        return _render_parse(step)
    if step.phase == TracePhase.COMPONENT:
        return _render_component(step)
    if step.phase == TracePhase.PERCEPTION:
        return _render_perception(step)
    if step.phase == TracePhase.PRIORITY:
        return _render_priority(step)
    if step.phase == TracePhase.PARENT_SELECTION:
        return _render_parent_selection(step)
    if step.phase == TracePhase.NUMBERING:
        return _render_numbering(step)
    if step.phase == TracePhase.ASSEMBLY:
        return _render_assembly(step)
    return None


def _render_parse(step: TraceStep) -> str | None:
    atoms = step.data.get("atom_count")
    if atoms is None:
        return None
    bonds = step.data.get("bond_count", 0)
    return f"RDKit parsed the SMILES into a molecular graph with {_plural(atoms, 'atom')} and {_plural(bonds, 'bond')}."


def _render_component(step: TraceStep) -> str | None:
    components = step.data.get("components") or []
    if not components:
        return None
    if len(components) == 1:
        return "The structure is a single connected component, named in one piece."
    sizes = ", ".join(str(len(component)) for component in components)
    return f"The structure splits into {len(components)} connected components with atom counts {sizes}."


def _render_perception(step: TraceStep) -> str | None:
    groups = step.data.get("groups") or []
    if not groups:
        return "Perception found no nameable principal groups."
    names = sorted({str(g.get("key", "?")) for g in groups})
    principals = sorted({str(g.get("key", "?")) for g in groups if g.get("principal_candidate")})
    if not principals:
        return f"Perception identified non-principal groups: {', '.join(names)}."
    return (
        f"Perception identified {_plural(len(groups), 'functional group')} "
        f"({', '.join(names)}); principal candidate{'s' if len(principals) > 1 else ''}: {', '.join(principals)}."
    )


def _render_priority(step: TraceStep) -> str | None:
    key = step.data.get("principal_key")
    if not key:
        return None
    role = "treated as a substituent" if step.data.get("is_substituent") else "selected as the principal group"
    return f"The {key} group is {role} by the registered seniority order."


def _render_parent_selection(step: TraceStep) -> str | None:
    parent = step.data.get("parent_atoms") or []
    if not parent:
        return None
    kind = _parent_kind(step.data)
    descriptor = step.data.get("polycycle_descriptor")
    descriptor_text = f" with descriptor {descriptor}" if descriptor else ""
    return f"The parent skeleton is a {len(parent)}-atom {kind}{descriptor_text} (atoms {sorted(parent)})."


def _render_numbering(step: TraceStep) -> str | None:
    locants = step.data.get("locants") or {}
    if not locants:
        return None
    sample = sorted(locants.items(), key=lambda kv: str(kv[1]))[:6]
    sample_str = ", ".join(f"atom {atom}->{loc}" for atom, loc in sample)
    return f"Parent numbering was selected from the final atom-to-locant map ({sample_str})."


def _render_assembly(step: TraceStep) -> str | None:
    name = step.data.get("name")
    if name is None:
        return None
    if "components" in step.data:
        components = step.data.get("components") or []
        if len(components) <= 1:
            return None
        return f"The final assembled name is **{name}** from {len(components)} named components."
    return None


def _parent_kind(data: dict) -> str:
    if data.get("is_spiro"):
        return "spiro ring system"
    if data.get("is_bicycle"):
        return "bicyclic ring system"
    if data.get("is_polycycle"):
        return "polycyclic ring system"
    if data.get("is_ring"):
        return "ring"
    return "acyclic chain"


def _collect_token_spans(result: NamingResult) -> list[dict]:
    tokens: list[dict] = []
    seen: set[tuple] = set()
    for step in result.decisions:
        for token in step.data.get("name_token_spans") or []:
            if not isinstance(token, dict):
                continue
            key = (
                token.get("start"),
                token.get("end"),
                token.get("text"),
                tuple(token.get("atoms") or ()),
                tuple(token.get("bonds") or ()),
            )
            if key in seen:
                continue
            seen.add(key)
            tokens.append(dict(token))
    for token in _walk_tree_tokens(result.substituent_tree):
        key = (
            token.get("start"),
            token.get("end"),
            token.get("text"),
            tuple(token.get("atoms") or ()),
            tuple(token.get("bonds") or ()),
        )
        if key not in seen:
            seen.add(key)
            tokens.append(dict(token))
    return sorted(tokens, key=lambda t: (int(t.get("start", 10**9)), int(t.get("end", 10**9)), str(t.get("text", ""))))


def _walk_tree_tokens(nodes: Iterable[dict]) -> Iterable[dict]:
    for node in nodes:
        if not isinstance(node, dict):
            continue
        for token in node.get("name_token_spans") or ():
            if isinstance(token, dict):
                yield token
        for segment in node.get("trace_segments") or ():
            if isinstance(segment, dict):
                for token in segment.get("name_token_spans") or ():
                    if isinstance(token, dict):
                        yield token
        for key in ("substituents", "replacement_prefixes"):
            yield from _walk_tree_tokens(node.get(key) or ())
        principal = node.get("principal_group")
        if isinstance(principal, dict):
            yield from _walk_tree_tokens((principal,))
        parent = node.get("parent")
        if isinstance(parent, dict):
            yield from _walk_tree_tokens((parent,))


def _summarize_tokens(tokens: tuple[dict, ...] | list[dict]) -> DescriptionTokenSummary:
    by_kind = Counter(str(t.get("token_kind", "unknown")) for t in tokens)
    by_ownership = Counter(str(t.get("ownership", "unknown")) for t in tokens)
    by_confidence = Counter(str(t.get("confidence", "unknown")) for t in tokens)
    by_source = Counter(str(t.get("source", "unknown")) for t in tokens)
    fallback_tokens = tuple(dict(t) for t in tokens if _is_fallback_token(t))
    return DescriptionTokenSummary(
        total=len(tokens),
        by_kind=dict(sorted(by_kind.items())),
        by_ownership=dict(sorted(by_ownership.items())),
        by_confidence=dict(sorted(by_confidence.items())),
        by_source=dict(sorted(by_source.items())),
        fallback_tokens=fallback_tokens,
    )


def _is_fallback_token(token: dict) -> bool:
    confidence = str(token.get("confidence", "")).lower()
    ownership = str(token.get("ownership", "")).lower()
    source = str(token.get("source", "")).lower()
    return (
        "fallback" in confidence
        or "fallback" in ownership
        or "fallback" in source
        or confidence in {"ambiguous", "unbound"}
        or ownership in {"ambiguous", "unbound"}
    )


def _token_binding_lines(summary: DescriptionTokenSummary, tokens: tuple[dict, ...]) -> list[str]:
    if summary.total == 0:
        return ["- No final-name token spans were emitted."]
    lines = [f"- {summary.total} final-name tokens carry graph-binding metadata."]
    for token in tokens[:12]:
        lines.append(f"- {_token_line(token)}")
    if len(tokens) > 12:
        lines.append(f"- ... {len(tokens) - 12} more tokens are available in `token_spans`.")
    if summary.by_confidence:
        lines.append("- Confidence: " + _counter_text(summary.by_confidence) + ".")
    if summary.fallback_tokens:
        names = ", ".join(str(t.get("text", "")) for t in summary.fallback_tokens[:8])
        more = "" if len(summary.fallback_tokens) <= 8 else f", plus {len(summary.fallback_tokens) - 8} more"
        lines.append(f"- Fallback or ambiguous tokens: {names}{more}.")
    else:
        lines.append("- No fallback or ambiguous token bindings were needed.")
    return lines


def _token_line(token: dict) -> str:
    text = token.get("text", "")
    kind = token.get("token_kind", "unknown")
    atoms = _atom_text(token.get("atoms") or [])
    bonds = _bond_text(token.get("bonds") or [])
    confidence = token.get("confidence", "unknown")
    source = token.get("source", "unknown")
    locants = token.get("locants") or []
    locant_text = f"; locants {','.join(str(locant) for locant in locants)}" if locants else ""
    return f"\"{text}\" [{kind}] -> {atoms}; {bonds}{locant_text}; confidence={confidence}; source={source}."


def _counter_text(counter: dict[str, int]) -> str:
    return ", ".join(f"{key}={value}" for key, value in sorted(counter.items()))


def _substituent_tree_lines(trees: list[dict]) -> list[str]:
    lines: list[str] = []
    for idx, tree in enumerate(trees, start=1):
        lines.extend(_describe_tree_node(tree, depth=0, label=f"Component {idx}"))
    return lines


def _describe_tree_node(node: dict, *, depth: int, label: str | None = None) -> list[str]:
    if not isinstance(node, dict):
        return []
    indent = "  " * depth
    name = str(node.get("name") or "(unnamed)")
    atoms = node.get("atoms") or []
    bonds = node.get("bonds") or []
    prefix = f"{label}: " if label else ""
    lines = [f"{indent}- {prefix}{name} covers {_plural(len(atoms), 'atom')} and {_plural(len(bonds), 'bond')}."]

    parent = node.get("parent")
    if isinstance(parent, dict):
        parent_line = _parent_tree_line(parent)
        if parent_line:
            lines.append(f"{indent}  {parent_line}")

    principal = node.get("principal_group")
    if isinstance(principal, dict):
        group = principal.get("key")
        locants = _locant_text(principal.get("locants") or [])
        atoms_text = _atom_text(principal.get("atoms") or [])
        lines.append(f"{indent}  Principal group: {group}{locants} ({atoms_text}).")

    for item in node.get("replacement_prefixes") or []:
        lines.append(f"{indent}  Replacement prefix: {_named_item_text(item)}.")
    for item in node.get("unsaturations") or []:
        locants = _unsaturation_locant_text(item.get("locants") or [])
        bonds_text = _bond_text(item.get("bonds") or [])
        lines.append(f"{indent}  Unsaturation: {item.get('bond_key')}{locants} ({bonds_text}).")
    for item in node.get("hydro_operations") or []:
        locants = _locant_text(item.get("locants") or [])
        atoms_text = _atom_text(item.get("atom_ids") or [])
        lines.append(f"{indent}  Hydro operation: {item.get('operation_kind')}{locants} ({atoms_text}).")
    for item in node.get("parent_charges") or []:
        locant = item.get("locant")
        atom = item.get("atom_id")
        charge = item.get("charge")
        lines.append(f"{indent}  Parent charge: {item.get('symbol')} {charge:+d} at locant {locant} on atom {atom}.")

    substituents = node.get("substituents") or []
    for child in substituents:
        locants = _locant_text(child.get("locants") or [])
        child_label = f"Substituent{locants}"
        lines.extend(_describe_tree_node(child, depth=depth + 1, label=child_label))
    return lines


def _parent_tree_line(parent: dict) -> str:
    atoms = parent.get("atoms") or []
    retained = parent.get("retained_name")
    kind = "parent"
    if parent.get("is_spiro"):
        kind = "spiro parent"
    elif parent.get("is_bicycle"):
        kind = "bicyclic parent"
    elif parent.get("is_polycycle"):
        kind = "polycyclic parent"
    elif parent.get("is_ring"):
        kind = "ring parent"
    else:
        kind = "chain parent"
    descriptor = ""
    if parent.get("bicycle_descriptor"):
        descriptor = f" descriptor {parent.get('bicycle_descriptor')}"
    elif parent.get("spiro_descriptor"):
        descriptor = f" descriptor {parent.get('spiro_descriptor')}"
    elif parent.get("polycycle_descriptor"):
        descriptor = f" descriptor {parent.get('polycycle_descriptor')}"
    retained_text = f" retained as {retained}" if retained else ""
    locant_map = parent.get("atom_ids_by_locant") or {}
    locants = ", ".join(f"{loc}->{atom}" for loc, atom in sorted(locant_map.items(), key=lambda item: str(item[0]))[:8])
    map_text = f"; locants {locants}" if locants else ""
    return f"Parent: {kind} with {_plural(len(atoms), 'atom')}{descriptor}{retained_text}{map_text}."


def _named_item_text(item: dict) -> str:
    name = item.get("name") or item.get("key") or "(unnamed)"
    locants = _locant_text(item.get("locants") or [])
    atoms = _atom_text(item.get("atoms") or item.get("atom_ids") or [])
    return f"{name}{locants} ({atoms})"


def _locant_text(locants: list[Any]) -> str:
    return "" if not locants else " at " + ",".join(str(locant) for locant in locants)


def _unsaturation_locant_text(locants: list[Any]) -> str:
    if not locants:
        return ""
    displayed = ",".join(str(locant) for locant in locants)
    explanations = []
    for locant in locants:
        text = str(locant)
        match = re.fullmatch(r"([^()]+)\(([^()]+)\)", text)
        if match:
            explanations.append(f"{text} means a multiple bond between locants {match.group(1)} and {match.group(2)}")
    if not explanations:
        return f" at {displayed}"
    return f" at {displayed}; {'; '.join(explanations)}"


def _atom_text(atoms: list[Any]) -> str:
    return "atoms " + ",".join(str(atom) for atom in atoms) if atoms else "no atom metadata"


def _bond_text(bonds: list[Any]) -> str:
    return "bonds " + ",".join(str(bond) for bond in bonds) if bonds else "no bond metadata"


def _trace_segment_lines(trace_segments: list[dict], trees: list[dict]) -> list[str]:
    tree_lines = _tree_trace_segment_lines(trees)
    if tree_lines:
        return tree_lines
    lines = []
    for seg in trace_segments:
        line = _trace_segment_line(seg, depth=0)
        if line:
            lines.append(line)
    return lines


def _tree_trace_segment_lines(trees: list[dict]) -> list[str]:
    lines: list[str] = []
    seen: set[tuple] = set()
    for tree in trees:
        lines.extend(_node_trace_segment_lines(tree, depth=0, seen=seen))
    return lines


def _node_trace_segment_lines(node: dict, *, depth: int, seen: set[tuple]) -> list[str]:
    if not isinstance(node, dict):
        return []
    lines: list[str] = []
    segments = [seg for seg in node.get("trace_segments") or [] if isinstance(seg, dict)]
    children = [child for child in node.get("substituents") or [] if isinstance(child, dict)]
    child_atoms = set().union(*(set(child.get("atoms") or []) for child in children)) if children else set()

    parent = node.get("parent") if isinstance(node.get("parent"), dict) else {}
    parent_atoms = set(parent.get("atoms") or [])
    principal = node.get("principal_group") if isinstance(node.get("principal_group"), dict) else {}
    principal_atoms = set(principal.get("atoms") or [])

    ordered: list[dict] = []
    ordered.extend(_pop_segments(segments, lambda seg: (seg.get("label") == "parent skeleton") and _same_atoms(seg, parent_atoms)))
    ordered.extend(_pop_segments(segments, lambda seg: bool(principal_atoms) and _same_atoms(seg, principal_atoms)))
    ordered.extend(_pop_segments(segments, lambda seg: _is_local_segment(seg, node, child_atoms)))

    for seg in ordered:
        if not isinstance(seg, dict):
            continue
        key = (
            seg.get("key"),
            tuple(seg.get("atoms") or ()),
            tuple(seg.get("bonds") or ()),
            tuple(seg.get("name_terms") or ()),
        )
        if key in seen:
            continue
        seen.add(key)
        line = _trace_segment_line(seg, depth=depth)
        if line:
            lines.append(line)
    for child in children:
        lines.extend(_node_trace_segment_lines(child, depth=depth + 1, seen=seen))
    return lines


def _pop_segments(segments: list[dict], predicate) -> list[dict]:
    selected: list[dict] = []
    remaining: list[dict] = []
    for seg in segments:
        if predicate(seg):
            selected.append(seg)
        else:
            remaining.append(seg)
    segments[:] = remaining
    return selected


def _same_atoms(seg: dict, atoms: set[int]) -> bool:
    return bool(atoms) and set(seg.get("atoms") or []) == atoms


def _is_local_segment(seg: dict, node: dict, child_atoms: set[int]) -> bool:
    atoms = set(seg.get("atoms") or [])
    if not atoms:
        return False
    node_atoms = set(node.get("atoms") or [])
    if node_atoms and not atoms <= node_atoms:
        return False
    return not atoms <= child_atoms


def _trace_segment_line(seg: dict, *, depth: int) -> str | None:
    if not isinstance(seg, dict):
        return None
    label = seg.get("label") or seg.get("key")
    terms = seg.get("name_terms") or []
    if not label or not terms:
        return None
    atoms = seg.get("atoms") or []
    bonds = seg.get("bonds") or []
    graph = []
    if atoms:
        graph.append(_atom_text(atoms))
    if bonds:
        graph.append(_bond_text(bonds))
    suffix = f" ({'; '.join(graph)})" if graph else ""
    indent = "  " * depth
    return f"{indent}- {label}: contributes \"{terms[0]}\"{suffix}."


__all__ = [
    "DescribedComponent",
    "Description",
    "DescriptionTokenSummary",
    "describe",
]
