"""Stereochemistry metadata audit helpers."""

from dataclasses import dataclass
import re

from .assembly_parts import AssemblyParts
from .molecule import Molecule


@dataclass(frozen=True)
class StereochemistryAudit:
    """Audit result for assembled stereochemical descriptors."""

    checked_features: int
    issues: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.issues


def audit_stereochemistry(mol: Molecule, parts: AssemblyParts) -> StereochemistryAudit:
    """Check that emitted stereo descriptors map back to parent graph metadata."""

    locant_to_atom = _locant_to_atom(parts)
    checked = 0
    issues: list[str] = []
    for locant, descriptor in parts.stereo_features:
        locant = str(locant)
        if descriptor in {"R", "S"}:
            checked += 1
            atom_idx = locant_to_atom.get(locant)
            if atom_idx is None:
                issues.append(f"{locant}{descriptor}: locant is not in parent map")
            elif mol.atoms[atom_idx].stereo != descriptor:
                issues.append(f"{locant}{descriptor}: atom stereo metadata is {mol.atoms[atom_idx].stereo!r}")
        elif descriptor in {"E", "Z"}:
            checked += 1
            if not _has_matching_bond_stereo(mol, parts, locant, descriptor, locant_to_atom):
                issues.append(f"{locant}{descriptor}: no matching parent bond stereo metadata")
    binding_checked, binding_issues = audit_bound_stereochemistry(mol, parts)
    checked += binding_checked
    issues.extend(binding_issues)
    return StereochemistryAudit(checked_features=checked, issues=tuple(issues))


def audit_bound_stereochemistry(mol: Molecule, parts: AssemblyParts) -> tuple[int, list[str]]:
    """Check stereo coverage in all named atom-binding terms."""

    checked = 0
    issues: list[str] = []
    for binding in parts.name_atom_bindings:
        if binding.role == "parent":
            continue
        stereo_atom_count = sum(1 for atom_id in binding.atom_ids if atom_id in mol.atoms and mol.atoms[atom_id].stereo)
        if stereo_atom_count:
            checked += stereo_atom_count
            descriptor_count = _rs_descriptor_count(binding.term)
            if descriptor_count < stereo_atom_count:
                issues.append(
                    f"{binding.role}:{binding.term}: {descriptor_count} R/S descriptors for {stereo_atom_count} stereo atoms"
                )
        stereo_bond_count = sum(
            1
            for bond_id in binding.bond_ids
            for bond in [mol.bonds.get(bond_id)]
            if bond is not None and bond.stereo in {"E", "Z"}
        )
        if stereo_bond_count:
            checked += stereo_bond_count
            descriptor_count = _ez_descriptor_count(binding.term)
            if descriptor_count < stereo_bond_count:
                issues.append(
                    f"{binding.role}:{binding.term}: {descriptor_count} E/Z descriptors for {stereo_bond_count} stereo bonds"
                )
    return checked, issues


def _rs_descriptor_count(term: str) -> int:
    return len(re.findall(r"\d+[A-Za-z]*[RS](?=[,\)])", term))


def _ez_descriptor_count(term: str) -> int:
    return len(re.findall(r"(?:^|[(,])(?:\d+[A-Za-z]*)?[EZ](?=[,\)])", term))


def _locant_to_atom(parts: AssemblyParts) -> dict[str, int]:
    return dict(parts.parent_atom_ids_by_locant)


def _has_matching_bond_stereo(
    mol: Molecule,
    parts: AssemblyParts,
    locant: str,
    descriptor: str,
    locant_to_atom: dict[str, int],
) -> bool:
    start_atom = locant_to_atom.get(locant)
    if start_atom is None:
        return False
    parent_atoms = parts.parent_atom_ids
    for neighbor in mol.get_neighbors(start_atom):
        if neighbor not in parent_atoms:
            continue
        bond = mol.get_bond(start_atom, neighbor)
        if bond and bond.stereo == descriptor:
            return True
    return False
