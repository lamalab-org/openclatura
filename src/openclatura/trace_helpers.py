"""Trace and explainability helpers for generated names."""

from dataclasses import replace

from .assembly_parts import AssemblyParts, NameTokenBinding, SubstituentItem
from .formatting import strip_outer_parentheses
from .molecule import DecisionTrace, Molecule, TracePhase
from .nomenclature import RULES
from .perception import PerceivedGroup
from .principal_suffixes import principal_suffix_terms
from .rules import bonds, multipliers, stems


def decision_trace_data(trace: DecisionTrace | list | tuple | None) -> list[dict]:
    """Return JSON-safe decision trace data for nested name fragments."""

    if trace is None:
        return []
    steps = trace.steps if isinstance(trace, DecisionTrace) else trace
    data = []
    for step in steps:
        phase = step.phase.value if hasattr(step.phase, "value") else str(step.phase)
        data.append(
            {
                "phase": phase,
                "decision": step.decision,
                "reason": step.reason,
                "atoms": sorted(step.atoms),
                "bonds": sorted(step.bonds),
                "data": step.data,
            }
        )
    return data


def trace_decision(
    trace: DecisionTrace | None,
    phase: TracePhase,
    decision: str,
    reason: str,
    *,
    atoms=None,
    bonds=None,
    data: dict | None = None,
) -> None:
    """Append a decision trace step when tracing is enabled."""

    if trace is None:
        return
    trace.add(phase, decision, reason, atoms=atoms or (), bonds=bonds or (), data=data)


def functional_group_trace_data(groups: list[PerceivedGroup]) -> list[dict]:
    """Return compact functional-group metadata for decision traces."""

    return [
        {
            "key": group.key,
            "principal_candidate": group.is_principal_candidate,
            "attachment_atom": group.attachment_carbon,
            "atoms": sorted(group.atom_ids),
            "bonds": sorted(group.bond_ids),
            "prefix": group.prefix,
            "suffix": group.suffix,
            "seniority": group.seniority,
            "metadata_source": group.metadata.source,
        }
        for group in groups
    ]


def bond_ids_within(mol: Molecule, atom_ids: set[int]) -> set[int]:
    """Return bond IDs whose endpoints are both in atom_ids."""

    bond_ids = set()
    for atom_idx in atom_ids:
        for neighbor_idx in mol.get_neighbors(atom_idx):
            if neighbor_idx in atom_ids and atom_idx < neighbor_idx:
                bond = mol.get_bond(atom_idx, neighbor_idx)
                if bond:
                    bond_ids.add(bond.idx)
    return bond_ids


def add_substituent_trace(
    parts: AssemblyParts,
    name: str,
    locant: str,
    atom_ids=None,
    bond_ids=None,
    charge_atom_ids=None,
    trace_segments=None,
    nested_decisions=None,
    emitted_tokens: tuple[NameTokenBinding, ...] = (),
    substituent_tree=None,
    spiro=None,
) -> None:
    """Append or merge a substituent while preserving trace atom/bond IDs."""

    atom_ids = set(atom_ids or ())
    bond_ids = set(bond_ids or ())
    charge_atom_ids = set(charge_atom_ids or ())
    trace_segments = list(trace_segments or ())
    nested_decisions = list(nested_decisions or ())
    if spiro is not None:
        spiro = replace(spiro, parent_locant=str(locant))
    existing = next((s for s in parts.substituents if s.name == name and s.spiro == spiro), None)
    if existing:
        existing.locants.append(locant)
        existing.atom_ids.update(atom_ids)
        existing.bond_ids.update(bond_ids)
        existing.charge_atom_ids.update(charge_atom_ids)
        existing.trace_segments.extend(trace_segments)
        existing.nested_decisions.extend(nested_decisions)
        if substituent_tree:
            existing.substituent_tree = _merge_substituent_tree_instances(
                existing.substituent_tree,
                substituent_tree,
                existing.name,
            )
        existing.emitted_tokens = existing.emitted_tokens + tuple(emitted_tokens)
    else:
        parts.substituents.append(
            SubstituentItem(
                name=name,
                locants=[locant],
                atom_ids=atom_ids,
                bond_ids=bond_ids,
                charge_atom_ids=charge_atom_ids,
                emitted_tokens=tuple(emitted_tokens),
                trace_segments=trace_segments,
                nested_decisions=nested_decisions,
                substituent_tree=substituent_tree,
                spiro=spiro,
            )
        )


def multiplied_name_terms(name: str, count: int) -> list[str]:
    """Return possible assembled text terms for repeated name fragments."""

    clean = strip_outer_parentheses(name)
    terms = [clean]
    if count > 1:
        is_complex = "(" in clean or clean[:1].isdigit() or "-" in clean or " " in clean
        mult = multipliers.complex_(count) if is_complex else multipliers.basic(count)
        terms.insert(0, mult + clean)
    return terms


def assembly_parent_terms(parts: AssemblyParts) -> list[str]:
    """Return parent-name terms from the actual assembly configuration."""

    if parts.retained_name:
        return [
            parts.retained_name,
            parts.retained_name[:-1] if parts.retained_name.endswith("e") else parts.retained_name,
        ]

    stem = stems.stem_for(parts.parent_length)
    terms = [stem]
    if parts.is_bicycle:
        terms.insert(0, f"bicyclo[{parts.bicycle_xyz[0]}.{parts.bicycle_xyz[1]}.{parts.bicycle_xyz[2]}]{stem}")
        terms.append("bicyclo")
    elif parts.is_spiro:
        terms.insert(0, f"spiro[{parts.spiro_xy[0]}.{parts.spiro_xy[1]}]{stem}")
        terms.append("spiro")
    elif parts.is_polycycle and parts.polycycle_descriptor:
        terms.insert(0, parts.polycycle_descriptor + stem)
    elif parts.is_ring:
        terms.insert(0, "cyclo" + stem)
    return terms


def assembly_trace_segments(parts: AssemblyParts) -> list[dict]:
    """Convert populated AssemblyParts metadata into visualizer annotations."""

    segments = []
    if parts.substituents:
        grouped: dict[str, SubstituentItem] = {}
        for item in parts.substituents:
            target = grouped.setdefault(
                item.name,
                SubstituentItem(name=item.name, locants=[], atom_ids=set(), bond_ids=set()),
            )
            target.locants.extend(item.locants)
            target.atom_ids.update(item.atom_ids)
            target.bond_ids.update(item.bond_ids)
            target.charge_atom_ids.update(item.charge_atom_ids)
            target.trace_segments.extend(item.trace_segments)
            target.nested_decisions.extend(item.nested_decisions)
            if item.substituent_tree:
                target.substituent_tree = _merge_substituent_tree_instances(
                    target.substituent_tree,
                    item.substituent_tree,
                    target.name,
                )
        for item in grouped.values():
            if item.trace_segments and strip_outer_parentheses(item.name) != "methyl":
                for segment in item.trace_segments:
                    segment = dict(segment)
                    if item.nested_decisions:
                        segment.setdefault("substituent_name", item.name)
                        segment.setdefault("nested_decisions", item.nested_decisions)
                    segments.append(segment)
                continue
            segment = {
                "key": f"substituent:{item.name}",
                "label": f"{strip_outer_parentheses(item.name)} substituent",
                "atoms": sorted(item.atom_ids),
                "bonds": sorted(item.bond_ids),
                "name_terms": multiplied_name_terms(item.name, max(1, len(item.locants))),
                "rule_hint": "Detachable prefixes: Blue Book P-14.2, P-16.5, and P-61.",
            }
            if item.nested_decisions:
                segment["nested_decisions"] = item.nested_decisions
            segments.append(segment)

    if parts.a_prefixes:
        grouped_a: dict[str, SubstituentItem] = {}
        for item in parts.a_prefixes:
            target = grouped_a.setdefault(
                item.name,
                SubstituentItem(name=item.name, locants=[], atom_ids=set(), bond_ids=set()),
            )
            target.locants.extend(item.locants)
            target.atom_ids.update(item.atom_ids)
            target.bond_ids.update(item.bond_ids)
            target.charge_atom_ids.update(item.charge_atom_ids)
        for item in grouped_a.values():
            segments.append(
                {
                    "key": f"replacement:{item.name}",
                    "label": f"{item.name} replacement",
                    "atoms": sorted(item.atom_ids),
                    "bonds": sorted(item.bond_ids),
                    "name_terms": multiplied_name_terms(item.name, max(1, len(item.locants))),
                    "rule_hint": "Replacement prefixes in parent structures: Blue Book P-51 and P-52.",
                }
            )

    for item in parts.unsaturations:
        count = max(1, len(item.locants))
        try:
            infix = bonds.unsaturation_infix(item.bond_key, count)
        except KeyError:
            infix = bonds.get(item.bond_key).suffix
        base_infix = infix[1:] if infix.startswith("a") else infix
        segments.append(
            {
                "key": f"unsaturation:{item.bond_key}",
                "label": f"{item.bond_key} bonds",
                "atoms": sorted(item.atom_ids),
                "bonds": sorted(item.bond_ids),
                "name_terms": [base_infix, bonds.get(item.bond_key).suffix],
                "rule_hint": "Unsaturation infixes: Blue Book P-31.1 and P-44.",
            }
        )

    simple_methyl_substituent = (
        parts.is_substituent
        and parts.parent_length == 1
        and not parts.substituents
        and not parts.a_prefixes
        and not parts.unsaturations
        and not parts.principal_group
    )
    if parts.parent_atom_ids and simple_methyl_substituent:
        segments.append(
            {
                "key": "substituent:methyl",
                "label": "methyl substituent",
                "atoms": sorted(parts.parent_atom_ids),
                "bonds": sorted(parts.parent_bond_ids),
                "name_terms": ["methyl"],
                "rule_hint": "Detachable hydrocarbon prefixes: Blue Book P-29 and P-61.",
            }
        )
    elif parts.parent_atom_ids:
        parent_key = "substituent_parent" if parts.is_substituent else "parent"
        parent_label = "substituent parent skeleton" if parts.is_substituent else "parent skeleton"
        segments.append(
            {
                "key": parent_key,
                "label": parent_label,
                "atoms": sorted(parts.parent_atom_ids),
                "bonds": sorted(parts.parent_bond_ids),
                "name_terms": assembly_parent_terms(parts),
                "rule_hint": "Parent hydride / parent structure: Blue Book P-44 and P-45.",
            }
        )

    if parts.principal_group:
        group = RULES.functional_groups.get(parts.principal_group.key)
        terms = list(principal_suffix_terms(group, (1, 2, 3)))
        if group.prefix:
            terms.append(group.prefix)
        segments.append(
            {
                "key": f"principal:{parts.principal_group.key}",
                "label": parts.principal_group.key.replace("_", " "),
                "atoms": sorted(parts.principal_group.atom_ids),
                "bonds": sorted(parts.principal_group.bond_ids),
                "name_terms": terms,
                "rule_hint": "Principal characteristic groups: Blue Book P-41, P-44, and P-61-P-67.",
            }
        )
    return segments


def attach_main_parent_decisions(
    trace_segments: list[dict], decisions: DecisionTrace | list | tuple | None
) -> list[dict]:
    """Attach main-component parent decisions to matching main parent segments.

    Recursive substituent parent decisions already travel as ``nested_decisions``
    on their substituent trace segments.  Main parent segments are generated from
    ``AssemblyParts`` and need this small join against the decision trace so
    parent selection and numbering are visible in the same trace object.
    """

    serialized = decision_trace_data(decisions)
    if not serialized:
        return trace_segments
    parent_decisions = [
        step
        for step in serialized
        if step.get("decision") in {"selected parent skeleton", "used retained parent name", "selected numbering"}
    ]
    if not parent_decisions:
        return trace_segments
    enriched: list[dict] = []
    for segment in trace_segments:
        item = dict(segment)
        if item.get("key") == "parent":
            atom_set = set(item.get("atoms") or ())
            decisions_for_segment = [step for step in parent_decisions if set(step.get("atoms") or ()) == atom_set]
            if decisions_for_segment:
                item.setdefault("decisions", decisions_for_segment)
        enriched.append(item)
    return enriched


def build_naming_tree_node(
    *,
    kind: str,
    name: str,
    atom_ids=None,
    bond_ids=None,
    parent=None,
    principal_group=None,
    substituents=None,
    replacement_prefixes=None,
    unsaturations=None,
    trace_segments=None,
    nested_decisions=None,
    metadata: dict | None = None,
) -> dict:
    """Build the invariant portion of every component/substituent tree node."""

    node = {
        "kind": kind,
        "name": name,
        "atoms": sorted(atom_ids or ()),
        "bonds": sorted(bond_ids or ()),
        "parent": parent,
        "principal_group": principal_group,
        "substituents": list(substituents or ()),
        "replacement_prefixes": list(replacement_prefixes or ()),
        "unsaturations": list(unsaturations or ()),
        "trace_segments": list(trace_segments or ()),
        "nested_decisions": list(nested_decisions or ()),
    }
    if metadata:
        overlapping_keys = node.keys() & metadata.keys()
        if overlapping_keys:
            raise ValueError(f"Tree metadata cannot replace invariant fields: {sorted(overlapping_keys)}")
        node.update(metadata)
    return node


def build_shortcut_tree_node(
    *,
    kind: str,
    name: str,
    atom_ids,
    bond_ids=None,
    decisions=None,
    name_atom_bindings: list[dict] | None = None,
    name_token_spans: list[dict] | None = None,
) -> dict:
    """Build a schema-complete tree node for a shortcut-rendered name."""

    metadata = {}
    if name_atom_bindings is not None:
        metadata["name_atom_bindings"] = list(name_atom_bindings)
    if name_token_spans is not None:
        metadata["name_token_spans"] = list(name_token_spans)
    return build_naming_tree_node(
        kind=kind,
        name=name,
        atom_ids=atom_ids,
        bond_ids=bond_ids,
        nested_decisions=decisions,
        metadata=metadata,
    )


def assembly_substituent_tree(
    parts: AssemblyParts,
    *,
    name: str,
    atom_ids=None,
    bond_ids=None,
    decisions=None,
    trace_segments=None,
) -> dict:
    """Return a nested substituent tree from the graph-bound assembly parts."""

    if trace_segments is None:
        trace_segments = assembly_trace_segments(parts)
    component_atoms = set(atom_ids or parts.parent_atom_ids)
    component_bonds = set(bond_ids or parts.parent_bond_ids)
    return build_naming_tree_node(
        kind="fragment",
        name=name,
        atom_ids=component_atoms,
        bond_ids=component_bonds,
        parent=_parent_tree_node(parts),
        principal_group=_principal_group_tree_node(parts),
        substituents=_substituent_tree_nodes(parts.substituents),
        replacement_prefixes=_simple_item_tree_nodes(parts.a_prefixes, "replacement_prefix"),
        unsaturations=[
            {
                "kind": "unsaturation",
                "bond_key": item.bond_key,
                "locants": list(item.locants),
                "atoms": sorted(item.atom_ids),
                "bonds": sorted(item.bond_ids),
            }
            for item in parts.unsaturations
        ],
        trace_segments=trace_segments,
        nested_decisions=decisions,
        metadata={
            "stereo_features": [
                {"descriptor": descriptor, "locant": locant} for descriptor, locant in parts.stereo_features
            ],
            "indicated_hydrogens": list(parts.indicated_hydrogens),
            "hydro_operations": [
                {
                    "key": operation.key,
                    "reason": operation.reason,
                    "locants": list(operation.locants),
                    "atom_ids": sorted(operation.atom_ids),
                    "operation_kind": operation.operation_kind,
                }
                for operation in parts.hydro_operations
            ],
            "parent_charges": [
                {
                    "locant": item.locant,
                    "symbol": item.symbol,
                    "charge": item.charge,
                    "atom_id": item.atom_id,
                }
                for item in parts.parent_charges
            ],
        },
    )


def _merge_substituent_tree_instances(existing: dict | None, new: dict, name: str) -> dict:
    """Preserve all tree instances when same-name substituents are grouped."""

    if existing is None:
        return new
    if existing == new:
        merged = dict(existing)
        merged["instance_count"] = int(merged.get("instance_count", 1)) + 1
        return merged
    if existing.get("kind") == "grouped_substituent_instances":
        merged = dict(existing)
        merged["instances"] = [*existing.get("instances", ()), new]
        return merged
    return {
        "kind": "grouped_substituent_instances",
        "name": name,
        "instances": [existing, new],
    }


def _parent_tree_node(parts: AssemblyParts) -> dict:
    node = {
        "kind": "parent",
        "retained_name": parts.retained_name,
        "parent_length": parts.parent_length,
        "is_ring": parts.is_ring,
        "is_bicycle": parts.is_bicycle,
        "is_spiro": parts.is_spiro,
        "is_polycycle": parts.is_polycycle,
        "bicycle_descriptor": list(parts.bicycle_xyz) if parts.is_bicycle else [],
        "spiro_descriptor": list(parts.spiro_xy) if parts.is_spiro else [],
        "polycycle_descriptor": parts.polycycle_descriptor,
        "atoms": sorted(parts.parent_atom_ids),
        "bonds": sorted(parts.parent_bond_ids),
        "atom_ids_by_locant": dict(parts.parent_atom_ids_by_locant),
        "atom_symbols_by_locant": dict(parts.parent_atom_symbols_by_locant),
        "atom_charges_by_locant": dict(parts.parent_atom_charges_by_locant),
    }
    suffix_data = _substituent_suffix_tree_node(parts)
    if suffix_data:
        node["substituent_suffix"] = suffix_data
    return node


def _substituent_suffix_tree_node(parts: AssemblyParts) -> dict | None:
    if not parts.is_substituent:
        return None
    locant = str(parts.attachment_locant)
    atom_id = parts.parent_atom_ids_by_locant.get(locant)
    if atom_id is None:
        return None
    suffix = "ylidyne" if parts.is_triple_attach else "ylidene" if parts.is_double_attach else "yl"
    return {
        "locant": locant,
        "suffix": suffix,
        "atom_id": atom_id,
        "is_double_attach": bool(parts.is_double_attach),
        "is_triple_attach": bool(parts.is_triple_attach),
    }


def _principal_group_tree_node(parts: AssemblyParts) -> dict | None:
    if parts.principal_group is None:
        return None
    return {
        "kind": "principal_group",
        "key": parts.principal_group.key,
        "locants": list(parts.principal_group.locants),
        "atoms": sorted(parts.principal_group.atom_ids),
        "bonds": sorted(parts.principal_group.bond_ids),
        "charge_atoms": sorted(parts.principal_group.charge_atom_ids),
    }


def _simple_item_tree_nodes(items: list[SubstituentItem], kind: str) -> list[dict]:
    return [
        {
            "kind": kind,
            "name": item.name,
            "locants": list(item.locants),
            "atoms": sorted(item.atom_ids),
            "bonds": sorted(item.bond_ids),
            "charge_atoms": sorted(item.charge_atom_ids),
            "trace_segments": list(item.trace_segments),
            "nested_decisions": list(item.nested_decisions),
        }
        for item in items
    ]


def _substituent_tree_nodes(items: list[SubstituentItem]) -> list[dict]:
    nodes = []
    for item in items:
        child = dict(item.substituent_tree or {})
        if child:
            child.setdefault("kind", "substituent")
            child.setdefault("name", item.name)
            child.setdefault("atoms", sorted(item.atom_ids))
            child.setdefault("bonds", sorted(item.bond_ids))
            child.setdefault("trace_segments", list(item.trace_segments))
            child.setdefault("nested_decisions", list(item.nested_decisions))
        else:
            child = {
                "kind": "substituent",
                "name": item.name,
                "atoms": sorted(item.atom_ids),
                "bonds": sorted(item.bond_ids),
                "trace_segments": list(item.trace_segments),
                "nested_decisions": list(item.nested_decisions),
                "substituents": [],
            }
        child["locants"] = list(item.locants)
        child["charge_atoms"] = sorted(item.charge_atom_ids)
        if item.spiro is not None:
            child["spiro"] = {
                "parent_locant": item.spiro.parent_locant,
                "side_locant": item.spiro.side_locant,
                "side_parent_name": item.spiro.side_parent_name,
                "prefixes": list(item.spiro.prefixes),
            }
        nodes.append(child)
    return nodes
