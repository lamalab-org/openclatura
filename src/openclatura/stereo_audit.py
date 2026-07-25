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


def audit_stereochemistry(
    mol: Molecule, parts: AssemblyParts, component_atoms: set[int] | None = None
) -> StereochemistryAudit:
    """Independently verify every emitted stereo descriptor, and prove that no
    stereocentre or stereo bond in the component was left unverified.

    The name's descriptors are the namer's *legacy*-CIP perception; confirming
    them against that same perception is circular, so each is adjudicated against
    the independently-computed modern-CIP oracle (:attr:`Atom.cip` /
    :attr:`Bond.cip`).  We positionally verify the descriptors we *can* map to a
    specific atom/bond — parent ``stereo_features`` (via the locant map) and the
    single-centre ``absolute_stereo`` / ``bond_stereo`` bindings (via their
    ``atom_ids`` / ``bond_ids``) — then require that *every* real stereo feature
    in the component ended up verified.  Anything left over (descriptors embedded
    in a multi-atom substituent term we do not rebuild, an absent oracle, a
    sulfur convention we do not replicate) surfaces as an issue so the caller
    abstains rather than confirm on untrusted evidence."""

    locant_to_atom = _locant_to_atom(parts)
    scope = set(component_atoms) if component_atoms is not None else set(mol.atoms)
    checked = 0
    issues: list[str] = []
    verified_atoms: set[int] = set()
    verified_bonds: set[int] = set()

    # 1. Parent descriptors, positionally mapped through the parent locant map.
    for locant, descriptor in parts.stereo_features:
        locant = str(locant)
        if descriptor in {"R", "S"}:
            checked += 1
            atom_idx = locant_to_atom.get(locant)
            if atom_idx is None:
                issues.append(f"{locant}{descriptor}: locant is not in parent map")
                continue
            issue = _absolute_stereo_issue(mol, atom_idx, locant, descriptor)
            issues.append(issue) if issue else verified_atoms.add(atom_idx)
        elif descriptor in {"E", "Z"}:
            checked += 1
            bond = _parent_stereo_bond(mol, parts, locant, locant_to_atom)
            issue = _bond_stereo_issue(bond, f"{locant}{descriptor}", descriptor)
            issues.append(issue) if issue else verified_bonds.add(bond.idx)

    # 2. Dedicated single-feature stereo bindings, positionally mapped through
    #    their own atom/bond ids.
    for binding in parts.name_atom_bindings:
        if binding.role == "absolute_stereo":
            stereo_atoms = [a for a in binding.atom_ids if a in mol.atoms and mol.atoms[a].stereo]
            descriptors = _descriptors(binding.term, "RSrs")
            if len(stereo_atoms) == 1 and len(descriptors) == 1:
                checked += 1
                issue = _absolute_stereo_issue(mol, stereo_atoms[0], binding.term, descriptors[0])
                issues.append(issue) if issue else verified_atoms.add(stereo_atoms[0])
        elif binding.role == "bond_stereo":
            stereo_bonds = [mol.bonds[b] for b in binding.bond_ids if b in mol.bonds and mol.bonds[b].stereo]
            descriptors = _descriptors(binding.term, "EZ")
            if len(stereo_bonds) == 1 and len(descriptors) == 1:
                checked += 1
                issue = _bond_stereo_issue(stereo_bonds[0], binding.term, descriptors[0])
                issues.append(issue) if issue else verified_bonds.add(stereo_bonds[0].idx)

    # 3. Soundness backstop: every real stereo feature in the component must have
    #    been positionally verified above, else we cannot confirm it.
    for aid in scope:
        atom = mol.atoms.get(aid)
        if atom is not None and (atom.stereo or atom.cip) and aid not in verified_atoms:
            checked += 1
            issues.append(f"absolute_stereo:atom-{aid}: stereocentre not independently verified")
    for bond in mol.bonds.values():
        if bond.u in scope and bond.v in scope and (bond.stereo in {"E", "Z"} or bond.cip) and bond.idx not in verified_bonds:
            checked += 1
            issues.append(f"bond_stereo:bond-{bond.idx}: stereo bond not independently verified")

    return StereochemistryAudit(checked_features=checked, issues=tuple(issues))


def _descriptors(term: str, letters: str) -> list[str]:
    return re.findall(rf"[{letters}](?=$|[,)])", term)


def _parent_stereo_bond(mol: Molecule, parts: AssemblyParts, locant: str, locant_to_atom: dict[str, int]):
    """The stereo-bearing parent double bond incident to ``locant`` (or ``None``)."""
    start_atom = locant_to_atom.get(locant)
    if start_atom is None:
        return None
    parent_atoms = parts.parent_atom_ids
    for neighbor in mol.get_neighbors(start_atom):
        if neighbor not in parent_atoms:
            continue
        bond = mol.get_bond(start_atom, neighbor)
        if bond and bond.stereo in {"E", "Z"}:
            return bond
    return None


def _bond_stereo_issue(bond, label: str, descriptor: str) -> str | None:
    """Verify an emitted ``E``/``Z`` descriptor against the independent bond-CIP
    oracle, mirroring :func:`_absolute_stereo_issue` for double bonds."""
    if bond is None:
        return f"{label}: no matching parent stereo bond"
    if bond.cip is None:
        return f"{label}: no independent CIP label to verify against"
    if bond.cip != descriptor:
        return f"{label}: independent bond CIP is {bond.cip!r}"
    return None


def _absolute_stereo_issue(mol: Molecule, atom_idx: int, locant: str, descriptor: str) -> str | None:
    """Return an issue string if the emitted ``R``/``S`` descriptor cannot be
    *independently* confirmed against the modern-CIP oracle, else ``None``.

    The emitted descriptor is the namer's legacy-CIP perception; confirming it
    against that same perception would be circular, so we compare against the
    independently-computed :attr:`Atom.cip`.  Anything we cannot adjudicate this
    way (oracle unavailable, or a 3-coordinate sulfur whose namer-convention
    flip we do not replicate here) yields an issue so the caller abstains rather
    than confirm on untrusted evidence."""

    atom = mol.atoms[atom_idx]
    if atom.symbol == "S" and mol.degree(atom_idx) == 3:
        return f"{locant}{descriptor}: sulfur stereo not independently verified"
    if atom.cip is None:
        return f"{locant}{descriptor}: no independent CIP label to verify against"
    if atom.cip != descriptor:
        return f"{locant}{descriptor}: independent CIP is {atom.cip!r}"
    return None


def _locant_to_atom(parts: AssemblyParts) -> dict[str, int]:
    return dict(parts.parent_atom_ids_by_locant)
