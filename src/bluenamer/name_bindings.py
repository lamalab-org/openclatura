"""Name-term to graph-atom binding helpers."""

from .assembly_parts import AssemblyParts, NameAtomBinding
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
        }
        for binding in bindings
    ]


def postprocess_name_atom_bindings(
    bindings: list[NameAtomBinding],
    postprocess_term,
) -> list[NameAtomBinding]:
    """Apply final name post-processing to binding terms."""

    processed = [
        NameAtomBinding(
            stage=binding.stage,
            role=binding.role,
            term=postprocess_term(binding.term),
            atom_ids=set(binding.atom_ids),
            bond_ids=set(binding.bond_ids),
            locants=tuple(binding.locants),
        )
        for binding in bindings
    ]
    return processed


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
