"""Graph roles for hypervalent heteroatom centers."""

from dataclasses import dataclass
from enum import StrEnum

from .molecule import Molecule
from .role_certificate import RoleCertificate, RoleCertificateAudit, RoleProjection, audit_role_certificate


class HypervalentLigandRole(StrEnum):
    """Ligand roles around P/S/Se/halogen centers."""

    OXO = "oxo"
    THIOXO = "thioxo"
    IMINO = "imino"
    IMIDO = "imido"
    OXIDO = "oxido"
    HYDROXY = "hydroxy"
    ALKOXY = "alkoxy"
    PEROXY = "peroxy"
    CHARGE_PAIR = "charge_pair"
    SIGMA = "sigma"


@dataclass(frozen=True)
class HypervalentLigand:
    """One ligand attached to a hypervalent center."""

    atom: int
    role: HypervalentLigandRole
    bond_id: int
    attachment_atom: int | None = None

    @property
    def atom_ids(self) -> set[int]:
        atoms = {self.atom}
        if self.attachment_atom is not None:
            atoms.add(self.attachment_atom)
        return atoms


@dataclass(frozen=True)
class HypervalentCenterRole:
    """A graph-bound hypervalent center and its classified ligands."""

    center: int
    center_symbol: str
    ligands: tuple[HypervalentLigand, ...]

    @property
    def atom_ids(self) -> set[int]:
        atoms = {self.center}
        for ligand in self.ligands:
            atoms.update(ligand.atom_ids)
        return atoms

    @property
    def bond_ids(self) -> set[int]:
        return {ligand.bond_id for ligand in self.ligands}

    def count(self, role: HypervalentLigandRole) -> int:
        return sum(1 for ligand in self.ligands if ligand.role == role)

    def certificate(self, mol: Molecule) -> RoleCertificate:
        """Return an auditable certificate for this hypervalent role."""

        atoms = frozenset(self.atom_ids)
        return RoleCertificate(
            key=f"hypervalent:{self.center_symbol}",
            projections=(
                RoleProjection(
                    stage="hypervalent_role",
                    role=self.center_symbol,
                    term=self.center_symbol,
                    atom_ids=atoms,
                    bond_ids=frozenset(self.bond_ids),
                    charges_by_atom={
                        atom_id: mol.atoms[atom_id].charge
                        for atom_id in atoms
                        if atom_id in mol.atoms and mol.atoms[atom_id].charge != 0
                    },
                ),
            ),
            decision_reasons=(f"Classified {self.center_symbol} center ligands from graph bonds and charges.",),
        )

    def template_audit(self, mol: Molecule) -> RoleCertificateAudit:
        """Audit the certificate required by a hypervalent renderer template."""

        return audit_role_certificate(
            mol,
            self.certificate(mol),
            expected_atoms=frozenset(self.atom_ids),
            expected_bonds=frozenset(self.bond_ids),
            require_charged_atoms=True,
        )


def hypervalent_center_role(
    mol: Molecule,
    component_atoms: set[int],
    center: int,
) -> HypervalentCenterRole | None:
    """Classify ligand roles around one supported hypervalent center."""

    if center not in component_atoms or mol.atoms[center].symbol not in {"P", "S", "Se", "Cl", "Br", "I"}:
        return None

    ligands: list[HypervalentLigand] = []
    for neighbor in mol.get_neighbors(center):
        if neighbor not in component_atoms:
            continue
        ligand = _classify_ligand(mol, component_atoms, center, neighbor)
        if ligand is None:
            return None
        ligands.append(ligand)
    if not ligands:
        return None
    high_risk = any(ligand.role != HypervalentLigandRole.SIGMA for ligand in ligands)
    charged = mol.atoms[center].charge != 0 or any(mol.atoms[ligand.atom].charge for ligand in ligands)
    if not high_risk and not charged:
        return None
    return HypervalentCenterRole(
        center=center,
        center_symbol=mol.atoms[center].symbol,
        ligands=tuple(sorted(ligands, key=lambda ligand: (ligand.role.value, ligand.atom))),
    )


def hypervalent_center_roles(mol: Molecule, component_atoms: set[int]) -> list[HypervalentCenterRole]:
    """Return all supported hypervalent center roles in a component."""

    return [
        role
        for atom_id in sorted(component_atoms)
        if (role := hypervalent_center_role(mol, component_atoms, atom_id)) is not None
    ]


def _classify_ligand(
    mol: Molecule,
    component_atoms: set[int],
    center: int,
    ligand_atom: int,
) -> HypervalentLigand | None:
    bond = mol.get_bond(center, ligand_atom)
    if bond is None:
        return None
    atom = mol.atoms[ligand_atom]
    if atom.symbol == "O":
        if bond.order == 2 or _is_charge_normalized_oxo(mol, center, ligand_atom):
            return HypervalentLigand(ligand_atom, HypervalentLigandRole.OXO, bond.idx)
        side = [n for n in mol.get_neighbors(ligand_atom) if n != center]
        if not side:
            role = HypervalentLigandRole.OXIDO if atom.charge < 0 else HypervalentLigandRole.HYDROXY
            return HypervalentLigand(ligand_atom, role, bond.idx)
        if len(side) != 1 or side[0] not in component_atoms:
            return None
        if mol.atoms[side[0]].symbol == "O":
            return HypervalentLigand(ligand_atom, HypervalentLigandRole.PEROXY, bond.idx, side[0])
        if mol.atoms[side[0]].is_carbon:
            return HypervalentLigand(ligand_atom, HypervalentLigandRole.ALKOXY, bond.idx, side[0])
        return HypervalentLigand(ligand_atom, HypervalentLigandRole.SIGMA, bond.idx, side[0])
    if atom.charge and mol.atoms[center].charge:
        return HypervalentLigand(ligand_atom, HypervalentLigandRole.CHARGE_PAIR, bond.idx)
    if atom.symbol == "S" and bond.order == 2:
        return HypervalentLigand(ligand_atom, HypervalentLigandRole.THIOXO, bond.idx)
    if atom.symbol == "N":
        if bond.order == 2:
            return HypervalentLigand(ligand_atom, HypervalentLigandRole.IMINO, bond.idx)
        if bond.order == 1 and atom.charge < 0:
            return HypervalentLigand(ligand_atom, HypervalentLigandRole.IMIDO, bond.idx)
    return HypervalentLigand(ligand_atom, HypervalentLigandRole.SIGMA, bond.idx)


def _is_charge_normalized_oxo(mol: Molecule, center: int, oxygen: int) -> bool:
    bond = mol.get_bond(center, oxygen)
    return (
        bond is not None
        and bond.order == 1
        and mol.atoms[oxygen].charge < 0
        and mol.degree(oxygen) == 1
        and mol.atoms[center].symbol in {"P", "S", "Se", "Cl", "Br", "I"}
        and mol.atoms[center].charge > 0
    )
