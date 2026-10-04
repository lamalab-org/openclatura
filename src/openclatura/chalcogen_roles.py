"""Element-aware graph roles shared by chalcogen functional groups.

The objects in this module deliberately contain no nomenclature strings.  They
record what the molecular graph says, leaving selection and rendering to the
rule layer.
"""

from dataclasses import dataclass
from enum import StrEnum

from .molecule import Molecule


class Chalcogen(StrEnum):
    OXYGEN = "O"
    SULFUR = "S"
    SELENIUM = "Se"
    TELLURIUM = "Te"
    POLONIUM = "Po"


CHALCOGENS = frozenset(Chalcogen)


class ChalcogenLigandRole(StrEnum):
    DOUBLE_BONDED = "double_bonded"
    HYDROGEN_BEARING = "hydrogen_bearing"
    ANIONIC = "anionic"
    ORGANIC_LINK = "organic_link"
    CHALCOGEN_LINK = "chalcogen_link"


class FunctionalFamily(StrEnum):
    HYDROXY = "hydroxy"
    CARBONYL = "carbonyl"
    ACYL = "acyl"
    CHALCOGEN_ETHER = "chalcogen_ether"
    PEROXOL = "peroxol"
    PEROXIDE = "peroxide"
    CENTRAL_ACID = "central_acid"
    NITROGEN_CHALCOGENIDE = "nitrogen_chalcogenide"
    NITRILE_CHALCOGENIDE = "nitrile_chalcogenide"


class DerivativeKind(StrEnum):
    ALCOHOL = "alcohol"
    ANION = "anion"
    ALDEHYDE = "aldehyde"
    KETONE = "ketone"
    ACID = "acid"
    ESTER = "ester"
    AMIDE = "amide"
    HYDRAZIDE = "hydrazide"
    UREA = "urea"
    ANHYDRIDE = "anhydride"
    ACID_HALIDE = "acid_halide"
    NEUTRAL_LINK = "neutral_link"
    ZWITTERION = "zwitterion"


@dataclass(frozen=True)
class ChalcogenLigand:
    """One graph-bound chalcogen ligand and its structural role."""

    atom: int
    element: Chalcogen
    role: ChalcogenLigandRole
    center: int
    bond_id: int
    attachment_atom: int | None = None


@dataclass(frozen=True)
class FunctionalGroupDescriptor:
    """Structure-only description retained throughout functional-group naming."""

    family: FunctionalFamily
    derivative: DerivativeKind
    centers: tuple[int, ...]
    ligands: tuple[ChalcogenLigand, ...]
    linker_paths: tuple[tuple[int, ...], ...] = ()
    attachment_atom: int | None = None
    is_external: bool = False
    central_element: Chalcogen | None = None

    @property
    def atom_ids(self) -> frozenset[int]:
        atoms = set(self.centers)
        for ligand in self.ligands:
            atoms.add(ligand.atom)
            if ligand.attachment_atom is not None:
                atoms.add(ligand.attachment_atom)
        for path in self.linker_paths:
            atoms.update(path)
        return frozenset(atoms)

    def ligands_with_role(self, role: ChalcogenLigandRole) -> tuple[ChalcogenLigand, ...]:
        return tuple(ligand for ligand in self.ligands if ligand.role is role)


def chalcogen_for_symbol(symbol: str) -> Chalcogen | None:
    """Return the typed chalcogen for an element symbol, if supported."""

    try:
        return Chalcogen(symbol)
    except ValueError:
        return None


def classify_chalcogen_ligand(
    mol: Molecule,
    center: int,
    ligand_atom: int,
) -> ChalcogenLigand | None:
    """Classify a chalcogen directly attached to ``center`` from graph facts."""

    atom = mol.atoms[ligand_atom]
    element = chalcogen_for_symbol(atom.symbol)
    bond = mol.get_bond(center, ligand_atom)
    if element is None or bond is None:
        return None
    if bond.order == 2 and mol.degree(ligand_atom) == 1:
        role = ChalcogenLigandRole.DOUBLE_BONDED
        attachment = None
    elif bond.order != 1:
        return None
    else:
        side = [neighbor for neighbor in mol.get_neighbors(ligand_atom) if neighbor != center]
        if atom.charge < 0 and not side:
            role = ChalcogenLigandRole.ANIONIC
            attachment = None
        elif atom.total_h_count > 0 and not side and atom.radical_electrons == 0:
            role = ChalcogenLigandRole.HYDROGEN_BEARING
            attachment = None
        elif len(side) == 1 and chalcogen_for_symbol(mol.atoms[side[0]].symbol) is not None:
            role = ChalcogenLigandRole.CHALCOGEN_LINK
            attachment = side[0]
        elif len(side) == 1:
            role = ChalcogenLigandRole.ORGANIC_LINK
            attachment = side[0]
        else:
            return None
    return ChalcogenLigand(
        atom=ligand_atom,
        element=element,
        role=role,
        center=center,
        bond_id=bond.idx,
        attachment_atom=attachment,
    )
