"""Natural-language description of how a structure was named.

`describe(smiles)` returns a `Description` that bundles a deterministic
multi-line prose explanation plus the structured components it was
rendered from. The prose is built directly from the `DecisionTrace` and
`trace_segments` that the engine already emits, so the output is
faithful to what the namer actually did — not a separate model.

Same input → same output. Safe to use in dataset pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass

from .engine import DEFAULT_NAMING_ENGINE, NamingRequest, _extract_rules_hit
from .molecule import TracePhase, TraceStep


@dataclass(frozen=True)
class DescribedComponent:
    """One phase-tagged human-readable line."""

    phase: str
    text: str


@dataclass(frozen=True)
class Description:
    """Result of `describe`. ``str(d)`` returns the rendered prose."""

    smiles: str
    name: str
    paragraphs: tuple[str, ...]
    components: tuple[DescribedComponent, ...] = ()
    rules_hit: tuple[str, ...] = ()
    rule_hints: tuple[str, ...] = ()

    def __str__(self) -> str:
        return "\n\n".join(self.paragraphs)

    @property
    def summary(self) -> str:
        return self.paragraphs[0] if self.paragraphs else ""

    def to_dict(self) -> dict:
        return {
            "smiles": self.smiles,
            "name": self.name,
            "summary": self.summary,
            "paragraphs": list(self.paragraphs),
            "components": [{"phase": c.phase, "text": c.text} for c in self.components],
            "rules_hit": list(self.rules_hit),
            "rule_hints": list(self.rule_hints),
        }


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


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
    return f"The structure splits into {len(components)} connected components, named independently and joined."


def _render_perception(step: TraceStep) -> str | None:
    groups = step.data.get("groups") or []
    if not groups:
        return "Perception found no nameable principal groups."
    names = sorted({g.get("key", "?") for g in groups})
    principals = sorted({g.get("key", "?") for g in groups if g.get("principal_candidate")})
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
    role = (
        "treated as a substituent"
        if step.data.get("is_substituent")
        else "selected as the principal characteristic group"
    )
    return f"The {key} group is {role}, per the registry seniority order."


def _render_parent_selection(step: TraceStep) -> str | None:
    parent = step.data.get("parent_atoms") or []
    if not parent:
        return None
    if step.data.get("is_bicycle"):
        kind = "bicyclic ring system"
    elif step.data.get("is_ring"):
        kind = "ring"
    else:
        kind = "acyclic chain"
    return f"The parent skeleton is a {len(parent)}-atom {kind} (atoms {sorted(parent)})."


def _render_numbering(step: TraceStep) -> str | None:
    locants = step.data.get("locants") or {}
    if not locants:
        return None
    sample = sorted(locants.items(), key=lambda kv: kv[1])[:4]
    sample_str = ", ".join(f"atom {a}→{loc}" for a, loc in sample)
    return f"Numbering minimizes locants for the principal group and substituents (e.g. {sample_str})."


def _render_assembly(step: TraceStep) -> str | None:
    name = step.data.get("name")
    if name is None:
        return None
    components = step.data.get("components")
    if components is not None:
        if len(components) == 1:
            return f"The final assembled name is **{name}**."
        return f"The final assembled name is **{name}** ({len(components)} pieces joined)."
    sub_count = step.data.get("substituent_count")
    if sub_count is None:
        return None
    return f"Component assembled as **{name}** with {_plural(sub_count, 'substituent')}."


_RENDERERS: dict[TracePhase, callable] = {
    TracePhase.PARSE: _render_parse,
    TracePhase.COMPONENT: _render_component,
    TracePhase.PERCEPTION: _render_perception,
    TracePhase.PRIORITY: _render_priority,
    TracePhase.PARENT_SELECTION: _render_parent_selection,
    TracePhase.NUMBERING: _render_numbering,
    TracePhase.ASSEMBLY: _render_assembly,
}


def _trace_segment_lines(trace_segments: list[dict]) -> list[str]:
    lines = []
    for seg in trace_segments:
        if not isinstance(seg, dict):
            continue
        label = seg.get("label") or seg.get("key")
        terms = seg.get("name_terms") or []
        if not label or not terms:
            continue
        atoms = seg.get("atoms") or []
        suffix = f" (atoms {atoms})" if atoms else ""
        lines.append(f"- {label}: contributes “{terms[0]}”{suffix}.")
    return lines


def describe(smiles: str) -> Description:
    """Render a natural-language explanation of how ``smiles`` is named."""

    result = DEFAULT_NAMING_ENGINE.run(NamingRequest(smiles=smiles, include_trace=True))

    if not result.name:
        summary = f"The structure {smiles} could not be named by the current ruleset."
    else:
        summary = f"The molecule {smiles} is named **{result.name}**."

    components: list[DescribedComponent] = []
    seen: set[tuple[str, str]] = set()
    for step in result.decisions:
        renderer = _RENDERERS.get(step.phase)
        if renderer is None:
            continue
        text = renderer(step)
        if not text:
            continue
        key = (step.phase.value, text)
        if key in seen:
            continue
        seen.add(key)
        components.append(DescribedComponent(phase=step.phase.value, text=text))

    paragraphs: list[str] = [summary]
    if components:
        paragraphs.append("\n".join(c.text for c in components))

    seg_lines = _trace_segment_lines(result.trace_segments)
    if seg_lines:
        paragraphs.append("Name pieces contributed by the trace:\n" + "\n".join(seg_lines))

    rules, hints = _extract_rules_hit(result.trace_segments)
    if rules:
        paragraphs.append("IUPAC Blue Book rules applied: " + ", ".join(rules) + ".")

    return Description(
        smiles=smiles,
        name=result.name,
        paragraphs=tuple(paragraphs),
        components=tuple(components),
        rules_hit=rules,
        rule_hints=hints,
    )


__all__ = ["DescribedComponent", "Description", "describe"]
