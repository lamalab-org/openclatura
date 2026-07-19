"""Stereochemistry metadata audit helpers."""

import re
from dataclasses import dataclass

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
    raw_checked, raw_issues = audit_raw_stereo_completeness(mol, parts)
    checked += raw_checked
    issues.extend(raw_issues)
    return StereochemistryAudit(checked_features=checked, issues=tuple(issues))


def audit_raw_stereo_completeness(mol: Molecule, parts: AssemblyParts) -> tuple[int, list[str]]:
    """Check that every drawn-but-CIP-unassigned center is rendered in the name.

    Atoms carrying only ``raw_stereo`` (a drawn wedge on a ring-symmetry-
    dependent center) are invisible to the descriptor-count audits above:
    dropping them silently produces a stereochemically incomplete name. They
    must be covered by a scoped descriptor group (``small_ring_stereo``) or a
    cis/trans prefix (``relative_stereo``) binding.
    """

    covered: set[int] = set()
    component_atoms: set[int] = set(parts.parent_atom_ids)
    for binding in parts.name_atom_bindings:
        binding_atoms = set(binding.atom_ids)
        component_atoms |= binding_atoms
        if binding.role in {"small_ring_stereo", "relative_stereo"}:
            covered |= binding_atoms
            continue
        term = binding.term or ""
        raw_ids = {
            idx
            for idx in binding_atoms
            if idx in mol.atoms and mol.atoms[idx].raw_stereo in {"CW", "CCW"} and not mol.atoms[idx].stereo
        }
        if not raw_ids:
            continue
        # Scoped descriptor groups are folded into substituent terms (e.g.
        # "((1R,3s)-3-..."); the raw centers count as rendered when the term
        # carries a descriptor for every assigned and raw center it names.
        assigned = sum(1 for idx in binding_atoms if idx in mol.atoms and mol.atoms[idx].stereo)
        if term in {"cis", "trans"} or _rs_any_descriptor_count(term) >= assigned + len(raw_ids):
            covered |= raw_ids
    checked = 0
    issues: list[str] = []
    for atom_idx in sorted(component_atoms):
        atom = mol.atoms.get(atom_idx)
        if atom is None or atom.stereo or atom.raw_stereo not in {"CW", "CCW"}:
            continue
        checked += 1
        if atom_idx not in covered:
            issues.append(f"atom {atom_idx}: drawn stereocenter ({atom.raw_stereo}) is not rendered in the name")
    return checked, issues


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
    return len(re.findall(r"\d+[A-Za-z]*[RS](?=[,\)]|$)", term))


def _rs_any_descriptor_count(term: str) -> int:
    """Count locanted stereodescriptors including pseudoasymmetric r/s."""

    return len(re.findall(r"\d+[A-Za-z]*[RSrs](?=[,\)]|$)", term))


def _ez_descriptor_count(term: str) -> int:
    return len(re.findall(r"(?:^|[(,])(?:\d+[A-Za-z]*)?[EZ](?=[,\)]|$)", term))


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
