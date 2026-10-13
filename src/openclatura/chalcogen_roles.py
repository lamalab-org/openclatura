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


class NitrogenChalcogenideCitation(StrEnum):
    """Preferred citation route for an N+-E- functional group."""

    SUBSTITUTIVE = "substitutive"
    FUNCTIONAL_CLASS = "functional_class"


CHALCOGENS = frozenset(Chalcogen)
VALIDATED_NOMENCLATURE_CHALCOGENS = frozenset(
    {Chalcogen.OXYGEN, Chalcogen.SULFUR, Chalcogen.SELENIUM, Chalcogen.TELLURIUM}
)


class UnsupportedChalcogenNomenclatureError(ValueError):
    """The graph contains a chalcogen whose organic vocabulary is not validated."""


class AcylDescriptorValidationError(ValueError):
    """A declared acyl descriptor is inconsistent with its molecular graph."""


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
    IMIDE = "imide"
    ANHYDRIDE = "anhydride"
    ACID_HALIDE = "acid_halide"
    ACYL_PSEUDOHALIDE = "acyl_pseudohalide"
    NEUTRAL_LINK = "neutral_link"
    ZWITTERION = "zwitterion"


class AcylLeavingGroup(StrEnum):
    FLUORIDE = "F"
    CHLORIDE = "Cl"
    BROMIDE = "Br"
    IODIDE = "I"
    AZIDE = "N3"
    CYANIDE = "CN"
    ISOCYANIDE = "NC"
    ISOCYANATE = "NCO"
    ISOTHIOCYANATE = "NCS"
    ISOSELENOCYANATE = "NCSe"
    ISOTELLUROCYANATE = "NCTe"


@dataclass(frozen=True)
class LeavingGroupDescriptor:
    """One complete leaving-group unit attached to a functional center."""

    kind: AcylLeavingGroup
    attachment_atom: int
    atom_ids: tuple[int, ...]
    bond_ids: tuple[int, ...]


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
    leaving_group: LeavingGroupDescriptor | None = None
    shared_atoms: tuple[int, ...] = ()

    @property
    def atom_ids(self) -> frozenset[int]:
        atoms = set(self.centers)
        for ligand in self.ligands:
            atoms.add(ligand.atom)
            if ligand.attachment_atom is not None:
                atoms.add(ligand.attachment_atom)
        for path in self.linker_paths:
            atoms.update(path)
        if self.leaving_group is not None:
            atoms.update(self.leaving_group.atom_ids)
        atoms.update(self.shared_atoms)
        return frozenset(atoms)

    def ligands_with_role(self, role: ChalcogenLigandRole) -> tuple[ChalcogenLigand, ...]:
        return tuple(ligand for ligand in self.ligands if ligand.role is role)


@dataclass(frozen=True)
class AcylLinkageView:
    """Validated, path-oriented view of a two-centre acyl linkage."""

    left_center: int
    right_center: int
    left_double: ChalcogenLigand
    right_double: ChalcogenLigand
    bridge_ligands: tuple[ChalcogenLigand, ...]
    path: tuple[int, ...]


def double_ligand_for_center(
    descriptor: FunctionalGroupDescriptor,
    center: int,
) -> ChalcogenLigand:
    """Return the unique double-bonded ligand belonging to ``center``."""

    matches = tuple(
        ligand for ligand in descriptor.ligands_with_role(ChalcogenLigandRole.DOUBLE_BONDED) if ligand.center == center
    )
    if len(matches) != 1:
        raise AcylDescriptorValidationError(
            f"Expected exactly one double ligand at acyl center {center}; found {len(matches)}"
        )
    return matches[0]


def is_complete_acyl_end(
    mol: Molecule,
    center: int,
    double_ligand: ChalcogenLigand,
    linkage_atom: int,
) -> bool:
    """Return whether one linkage end is a complete formyl/carbyl acyl site.

    The bounded carboxylic-anhydride route normally requires hydrogen or one
    carbon substituent.  A heteroatom attachment can also anchor an established
    parent when the component beyond it has parent-defining ring or multiple-
    bond topology.  A simple heteroalkyl ligand instead identifies another
    derivative family, such as a carbonate or carbamate.
    """

    atom = mol.atoms[center]
    if (
        not atom.is_carbon
        or atom.charge != 0
        or atom.radical_electrons != 0
        or atom.is_aromatic
        or double_ligand.center != center
        or double_ligand.role is not ChalcogenLigandRole.DOUBLE_BONDED
    ):
        return False

    double_bond = mol.get_bond(center, double_ligand.atom)
    linkage_bond = mol.get_bond(center, linkage_atom)
    if double_bond is None or double_bond.order != 2 or linkage_bond is None or linkage_bond.order != 1:
        return False

    remaining = set(mol.get_neighbors(center)) - {double_ligand.atom, linkage_atom}
    if not remaining:
        return atom.total_h_count == 1
    if len(remaining) != 1 or atom.total_h_count != 0:
        return False
    substituent = next(iter(remaining))
    substituent_bond = mol.get_bond(center, substituent)
    if substituent_bond is None or substituent_bond.order != 1:
        return False
    if mol.atoms[substituent].is_carbon:
        return True

    component = {substituent}
    frontier = [substituent]
    while frontier:
        current = frontier.pop()
        for neighbor in mol.get_neighbors(current):
            if neighbor == center or neighbor in component:
                continue
            component.add(neighbor)
            frontier.append(neighbor)

    component_edges = {
        mol.get_bond(left, right).idx
        for left in component
        for right in mol.get_neighbors(left)
        if right in component and left < right
    }
    has_cycle = len(component_edges) >= len(component)
    has_multiple_bond = any(mol.bonds[bond_id].order > 1 for bond_id in component_edges)
    return has_cycle or has_multiple_bond


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


def acyl_linkage_view(
    mol: Molecule,
    descriptor: FunctionalGroupDescriptor,
) -> AcylLinkageView:
    """Validate and orient a one- or two-chalcogen anhydride bridge."""

    if descriptor.family is not FunctionalFamily.ACYL or descriptor.derivative is not DerivativeKind.ANHYDRIDE:
        raise AcylDescriptorValidationError("Expected an acyl anhydride descriptor")
    if len(descriptor.centers) != 2 or descriptor.centers[0] == descriptor.centers[1]:
        raise AcylDescriptorValidationError("An anhydride requires two distinct acyl centers")
    if len(descriptor.linker_paths) != 1:
        raise AcylDescriptorValidationError("An anhydride requires one ordered linker path")

    path = descriptor.linker_paths[0]
    if len(path) not in {3, 4}:
        raise AcylDescriptorValidationError("An anhydride bridge must contain one or two chalcogen atoms")
    if set((path[0], path[-1])) != set(descriptor.centers):
        raise AcylDescriptorValidationError("The linker path must terminate at both acyl centers")

    left_center, right_center = path[0], path[-1]
    left_double = double_ligand_for_center(descriptor, left_center)
    right_double = double_ligand_for_center(descriptor, right_center)
    bridge_ligands = []
    for position in range(1, len(path) - 1):
        previous_atom = path[position - 1]
        bridge_atom = path[position]
        next_atom = path[position + 1]
        bond = mol.get_bond(previous_atom, bridge_atom)
        next_bond = mol.get_bond(bridge_atom, next_atom)
        atom = mol.atoms[bridge_atom]
        if bond is None or next_bond is None or bond.order != 1 or next_bond.order != 1:
            raise AcylDescriptorValidationError("Every anhydride bridge bond must be single")
        if (
            chalcogen_for_symbol(atom.symbol) is None
            or atom.charge != 0
            or atom.radical_electrons != 0
            or atom.total_h_count != 0
            or set(mol.get_neighbors(bridge_atom)) != {previous_atom, next_atom}
        ):
            raise AcylDescriptorValidationError(f"Bridge atom {bridge_atom} is not a complete neutral chalcogen linker")
        ligand = classify_chalcogen_ligand(mol, previous_atom, bridge_atom)
        if ligand is None or ligand.attachment_atom != next_atom:
            raise AcylDescriptorValidationError(
                f"Bridge atom {bridge_atom} does not agree with the ordered linker path"
            )
        bridge_ligands.append(ligand)

    return AcylLinkageView(
        left_center=left_center,
        right_center=right_center,
        left_double=left_double,
        right_double=right_double,
        bridge_ligands=tuple(bridge_ligands),
        path=path,
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
        if any(mol.atoms[idx].charge or mol.atoms[idx].radical_electrons for idx in (bond.u, bond.v)):
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
