"""Reusable atom-role selectors for perceived functional groups."""

from .molecule import Molecule
from .nomenclature import RULES
from .perception import PerceivedGroup


def bridge_oxygen(mol: Molecule, group: PerceivedGroup) -> int | None:
    return next((o for o in group.atoms_involved if mol.degree(o) == 2 and mol.atoms[o].symbol == "O"), None)


def ester_single_oxygen(mol: Molecule, group: PerceivedGroup) -> int | None:
    return next(
        (
            o
            for o in group.atoms_involved
            if mol.atoms[o].symbol == "O" and (mol.degree(o) == 2 or mol.atoms[o].charge == -1)
        ),
        None,
    )


def peroxy_ester_single_oxygen(mol: Molecule, group: PerceivedGroup) -> int | None:
    return next(
        (
            o
            for o in group.atoms_involved
            if mol.atoms[o].symbol == "O" and mol.get_bond(o, group.attachment_carbon) is None and mol.degree(o) == 2
        ),
        None,
    )


def ester_or_peroxy_single_oxygen(mol: Molecule, group: PerceivedGroup) -> int | None:
    if group.key in RULES.prefixes.peroxy_ester_groups:
        return peroxy_ester_single_oxygen(mol, group)
    return ester_single_oxygen(mol, group)


def sulfonyl_sulfur(mol: Molecule, group: PerceivedGroup) -> int | None:
    return next((s for s in group.atoms_involved if mol.atoms[s].symbol == "S"), None)


def amide_nitrogen(mol: Molecule, group: PerceivedGroup) -> int | None:
    return next((n for n in group.atoms_involved if mol.atoms[n].symbol == "N"), None)
