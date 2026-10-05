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
VALIDATED_NOMENCLATURE_CHALCOGENS = frozenset(
    {Chalcogen.OXYGEN, Chalcogen.SULFUR, Chalcogen.SELENIUM, Chalcogen.TELLURIUM}
)


class UnsupportedChalcogenNomenclatureError(ValueError):
    """The graph contains a chalcogen whose organic vocabulary is not validated."""


def require_validated_chalcogens(mol: Molecule) -> None:
    """Reject unvalidated chalcogens explicitly instead of silently omitting them."""

    unsupported = {
        element
        for atom in mol
        if (element := chalcogen_for_symbol(atom.symbol)) is not None
        and element not in VALIDATED_NOMENCLATURE_CHALCOGENS
    }
    if unsupported:
        symbols = ", ".join(sorted(element.value for element in unsupported))
        raise UnsupportedChalcogenNomenclatureError(
            f"Organic functional-group nomenclature is not validated for: {symbols}"
        )


class ChalcogenLigandRole(StrEnum):
    DOUBLE_BONDED = "double_bonded"
    HYDROGEN_BEARING = "hydrogen_bearing"
    ANIONIC = "anionic"
    CARBON_LINK = "carbon_link"
    HETEROATOM_LINK = "heteroatom_link"
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
        elif len(side) == 1:
            attachment = side[0]
            if chalcogen_for_symbol(mol.atoms[attachment].symbol) is not None:
                role = ChalcogenLigandRole.CHALCOGEN_LINK
            elif mol.atoms[attachment].is_carbon:
                role = ChalcogenLigandRole.CARBON_LINK
            else:
                role = ChalcogenLigandRole.HETEROATOM_LINK
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


def classify_peroxide_linkages(
    mol: Molecule,
    component_atoms: set[int],
) -> tuple[FunctionalGroupDescriptor, ...]:
    """Return every acyclic carbon-E-E-carbon linkage in a component."""

    candidates = []
    for bond in mol.bonds.values():
        if bond.u not in component_atoms or bond.v not in component_atoms or bond.order != 1:
            continue
        if chalcogen_for_symbol(mol.atoms[bond.u].symbol) is None:
            continue
        if chalcogen_for_symbol(mol.atoms[bond.v].symbol) is None:
            continue
        left_sides = [neighbor for neighbor in mol.get_neighbors(bond.u) if neighbor != bond.v]
        right_sides = [neighbor for neighbor in mol.get_neighbors(bond.v) if neighbor != bond.u]
        if (
            len(left_sides) != 1
            or len(right_sides) != 1
            or not mol.atoms[left_sides[0]].is_carbon
            or not mol.atoms[right_sides[0]].is_carbon
        ):
            continue
        blocked = {bond.u, bond.v}
        seen = {left_sides[0]}
        stack = [left_sides[0]]
        while stack:
            current = stack.pop()
            for neighbor in mol.get_neighbors(current):
                if neighbor in blocked or neighbor in seen:
                    continue
                seen.add(neighbor)
                stack.append(neighbor)
        if right_sides[0] in seen:
            continue
        if any(
            candidate != linker
            and (ligand := classify_chalcogen_ligand(mol, attachment, candidate)) is not None
            and ligand.role is ChalcogenLigandRole.DOUBLE_BONDED
            for attachment, linker in ((left_sides[0], bond.u), (right_sides[0], bond.v))
            for candidate in mol.get_neighbors(attachment)
        ):
            continue
        candidates.append((bond.u, bond.v, left_sides[0], right_sides[0]))
    return tuple(
        FunctionalGroupDescriptor(
            family=FunctionalFamily.PEROXIDE,
            derivative=DerivativeKind.NEUTRAL_LINK,
            centers=(left, right),
            ligands=(),
            linker_paths=((left_attachment, left, right, right_attachment),),
            attachment_atom=left_attachment,
        )
        for left, right, left_attachment, right_attachment in candidates
    )


def classify_peroxide_linkage(
    mol: Molecule,
    component_atoms: set[int],
) -> FunctionalGroupDescriptor | None:
    """Return a linkage only when the component has one unambiguous instance."""

    linkages = classify_peroxide_linkages(mol, component_atoms)
    return linkages[0] if len(linkages) == 1 else None
