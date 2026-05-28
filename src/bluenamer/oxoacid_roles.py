"""Graph roles for central-atom oxoacid families."""

from dataclasses import dataclass
from enum import StrEnum

from .molecule import Molecule


class OxoLigandRole(StrEnum):
    """Oxygen ligand role around an oxoacid central atom."""

    OXO = "oxo"
    OXIDO = "oxido"
    HYDROXY = "hydroxy"
    ALKOXY = "alkoxy"
    PEROXY = "peroxy"


@dataclass(frozen=True)
class CentralOxoLigand:
    """One central-atom oxygen ligand with graph role metadata."""

    oxygen: int
    role: OxoLigandRole
    attachment_atom: int | None = None


@dataclass(frozen=True)
class CentralOxoRole:
    """A central oxoacid-like graph classified by oxygen ligand roles."""

    central: int
    central_symbol: str
    ligands: tuple[CentralOxoLigand, ...]

    @property
    def ligand_atoms(self) -> set[int]:
        atoms = {ligand.oxygen for ligand in self.ligands}
        atoms.update(ligand.attachment_atom for ligand in self.ligands if ligand.attachment_atom is not None)
        atoms.discard(None)
        return atoms

    @property
    def oxygen_atoms(self) -> list[int]:
        return [ligand.oxygen for ligand in self.ligands]

    def count(self, role: OxoLigandRole) -> int:
        return sum(1 for ligand in self.ligands if ligand.role == role)

    def spec_counts(self) -> tuple[int, int]:
        """Return table-compatible single/double oxygen counts."""

        single_o = self.count(OxoLigandRole.HYDROXY) + self.count(OxoLigandRole.ALKOXY)
        single_o += self.count(OxoLigandRole.OXIDO) + self.count(OxoLigandRole.PEROXY)
        return single_o, self.count(OxoLigandRole.OXO)

    def has_organic_ester(self) -> bool:
        return any(ligand.role == OxoLigandRole.ALKOXY for ligand in self.ligands)

    def has_peroxy(self) -> bool:
        return any(ligand.role == OxoLigandRole.PEROXY for ligand in self.ligands)

    def has_anion(self) -> bool:
        return any(ligand.role == OxoLigandRole.OXIDO for ligand in self.ligands)



def central_oxo_role(mol: Molecule, component_atoms: set[int], central: int) -> CentralOxoRole | None:
    """Classify an oxoacid-like central atom and its oxygen ligands."""

    if central not in component_atoms:
        return None
    oxygen_neighbors = []
    for neighbor in mol.get_neighbors(central):
        if neighbor not in component_atoms or mol.atoms[neighbor].symbol != "O":
            return None
        oxygen_neighbors.append(neighbor)
    if not oxygen_neighbors:
        return None

    ligands = []
    for oxygen in oxygen_neighbors:
        ligand = _classify_oxygen_ligand(mol, component_atoms, central, oxygen)
        if ligand is None:
            return None
        ligands.append(ligand)
    return CentralOxoRole(central=central, central_symbol=mol.atoms[central].symbol, ligands=tuple(ligands))


def central_oxo_substituent_role(mol: Molecule, component_atoms: set[int], central: int) -> CentralOxoRole | None:
    """Classify oxygen ligands on a heteroatom substituent center.

    Unlike `central_oxo_role`, this allows non-oxygen ligand branches. It only
    binds the central atom's oxygen ligands so recursive substituent naming can
    count oxo/oxido/hydroxy/alkoxy/peroxy roles consistently before falling
    back to element-specific wording.
    """

    if central not in component_atoms:
        return None
    ligands = []
    for neighbor in mol.get_neighbors(central):
        if neighbor not in component_atoms or mol.atoms[neighbor].symbol != "O":
            continue
        ligand = _classify_oxygen_ligand(mol, component_atoms, central, neighbor)
        if ligand is None:
            return None
        ligands.append(ligand)
    if not ligands:
        return None
    return CentralOxoRole(central=central, central_symbol=mol.atoms[central].symbol, ligands=tuple(ligands))


def central_oxo_roles(mol: Molecule, component_atoms: set[int]) -> list[CentralOxoRole]:
    """Return all central oxoacid-like roles in a component."""

    return [
        role
        for atom_idx in sorted(component_atoms)
        if (role := central_oxo_role(mol, component_atoms, atom_idx)) is not None
    ]


def _classify_oxygen_ligand(
    mol: Molecule,
    component_atoms: set[int],
    central: int,
    oxygen: int,
) -> CentralOxoLigand | None:
    bond = mol.get_bond(central, oxygen)
    if bond is None:
        return None
    if bond.order == 2 or _is_charge_normalized_oxo_ligand(mol, central, oxygen):
        return CentralOxoLigand(oxygen=oxygen, role=OxoLigandRole.OXO)

    noncentral = [n for n in mol.get_neighbors(oxygen) if n != central]
    if not noncentral:
        role = OxoLigandRole.OXIDO if mol.atoms[oxygen].charge < 0 else OxoLigandRole.HYDROXY
        return CentralOxoLigand(oxygen=oxygen, role=role)
    if len(noncentral) != 1:
        return None

    attachment = noncentral[0]
    if attachment not in component_atoms:
        return None
    if mol.atoms[attachment].is_carbon:
        return CentralOxoLigand(oxygen=oxygen, role=OxoLigandRole.ALKOXY, attachment_atom=attachment)
    if mol.atoms[attachment].symbol == "O":
        return CentralOxoLigand(oxygen=oxygen, role=OxoLigandRole.PEROXY, attachment_atom=attachment)
    return None


def _is_charge_normalized_oxo_ligand(mol: Molecule, central: int, oxygen: int) -> bool:
    """Return true for single-bonded O-minus that encodes an oxo ligand."""

    central_symbol = mol.atoms[central].symbol
    if central_symbol not in {"Cl", "Br", "I"}:
        return False
    bond = mol.get_bond(central, oxygen)
    return (
        bond is not None
        and bond.order == 1
        and mol.atoms[oxygen].charge < 0
        and sum(1 for _ in mol.get_neighbors(oxygen)) == 1
    )
