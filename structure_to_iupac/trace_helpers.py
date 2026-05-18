"""Trace and explainability helpers for generated names."""

from dataclasses import replace

from .assembly_parts import AssemblyParts, SubstituentItem
from .formatting import strip_outer_parentheses
from .molecule import DecisionTrace, Molecule, TracePhase
from .nomenclature import RULES
from .perception import PerceivedGroup
from .rules import bonds, multipliers, stems


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
    trace_segments=None,
    spiro=None,
) -> None:
    """Append or merge a substituent while preserving trace atom/bond IDs."""

    atom_ids = set(atom_ids or ())
    bond_ids = set(bond_ids or ())
    trace_segments = list(trace_segments or ())
    if spiro is not None:
        spiro = replace(spiro, parent_locant=str(locant))
    existing = next((s for s in parts.substituents if s.name == name and s.spiro == spiro), None)
    if existing:
        existing.locants.append(locant)
        existing.atom_ids.update(atom_ids)
        existing.bond_ids.update(bond_ids)
        existing.trace_segments.extend(trace_segments)
    else:
        parts.substituents.append(
            SubstituentItem(
                name=name,
                locants=[locant],
                atom_ids=atom_ids,
                bond_ids=bond_ids,
                trace_segments=trace_segments,
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
        return [parts.retained_name, parts.retained_name[:-1] if parts.retained_name.endswith("e") else parts.retained_name]

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
            target.trace_segments.extend(item.trace_segments)
        for item in grouped.values():
            if item.trace_segments and strip_outer_parentheses(item.name) != "methyl":
                segments.extend(item.trace_segments)
                continue
            segments.append(
                {
                    "key": f"substituent:{item.name}",
                    "label": f"{strip_outer_parentheses(item.name)} substituent",
                    "atoms": sorted(item.atom_ids),
                    "bonds": sorted(item.bond_ids),
                    "name_terms": multiplied_name_terms(item.name, max(1, len(item.locants))),
                    "rule_hint": "Detachable prefixes: Blue Book P-14.2, P-16.5, and P-61.",
                }
            )

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
        segments.append(
            {
                "key": "parent",
                "label": "parent skeleton",
                "atoms": sorted(parts.parent_atom_ids),
                "bonds": sorted(parts.parent_bond_ids),
                "name_terms": assembly_parent_terms(parts),
                "rule_hint": "Parent hydride / parent structure: Blue Book P-44 and P-45.",
            }
        )

    if parts.principal_group:
        group = RULES.functional_groups.get(parts.principal_group.key)
        terms = [group.suffix]
        if group.multi_suffix:
            terms.insert(0, group.multi_suffix)
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
