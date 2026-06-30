"""Graph roles for peroxy carbonyl and carbonate-like groups."""

from dataclasses import dataclass
from enum import StrEnum

from .molecule import Molecule
from .role_certificate import RoleCertificate, RoleCertificateAudit, RoleProjection, audit_role_certificate


class PeroxyCarbonylKind(StrEnum):
    """Carbonyl role class around O-O carbonyl fragments."""

    HYDROPEROXIDE = "hydroperoxide"
    PEROXOATE = "peroxoate"
    CYCLIC_PEROXY_ESTER = "cyclic_peroxy_ester"
    CARBONATE_LIKE = "carbonate_like"


@dataclass(frozen=True)
class PeroxyCarbonylRole:
    """A carbonyl C(=O)-O-O role with graph-bound metadata."""

    carbonyl: int
    carbonyl_oxygen: int
    proximal_oxygen: int
    distal_oxygen: int
    kind: PeroxyCarbonylKind
    attachment_atom: int | None = None

    @property
    def atom_ids(self) -> set[int]:
        atoms = {self.carbonyl, self.carbonyl_oxygen, self.proximal_oxygen, self.distal_oxygen}
        if self.attachment_atom is not None:
            atoms.add(self.attachment_atom)
        return atoms

    def bond_ids(self, mol: Molecule) -> set[int]:
        pairs = [
            (self.carbonyl, self.carbonyl_oxygen),
            (self.carbonyl, self.proximal_oxygen),
            (self.proximal_oxygen, self.distal_oxygen),
        ]
        if self.attachment_atom is not None:
            pairs.append((self.distal_oxygen, self.attachment_atom))
        return {bond.idx for left, right in pairs if (bond := mol.get_bond(left, right)) is not None}

    def certificate(self, mol: Molecule) -> RoleCertificate:
        atoms = frozenset(self.atom_ids)
        return RoleCertificate(
            key=f"peroxy_carbonyl:{self.kind.value}",
            projections=(
                RoleProjection(
                    stage="peroxy_carbonyl_role",
                    role=self.kind.value,
                    term=self.kind.value,
                    atom_ids=atoms,
                    bond_ids=frozenset(self.bond_ids(mol)),
                    charges_by_atom={
                        atom_id: mol.atoms[atom_id].charge
                        for atom_id in atoms
                        if atom_id in mol.atoms and mol.atoms[atom_id].charge != 0
                    },
                ),
            ),
            decision_reasons=("Classified carbonyl peroxide role from C=O and O-O graph connectivity.",),
        )

    def template_audit(self, mol: Molecule) -> RoleCertificateAudit:
        """Audit the certificate required by a peroxy-carbonyl renderer template."""

        return audit_role_certificate(
            mol,
            self.certificate(mol),
            expected_atoms=frozenset(self.atom_ids),
            expected_bonds=frozenset(self.bond_ids(mol)),
            require_charged_atoms=True,
        )


def peroxy_carbonyl_roles(mol: Molecule, component_atoms: set[int]) -> list[PeroxyCarbonylRole]:
    """Return graph-bound peroxy carbonyl roles in a component."""

    roles: list[PeroxyCarbonylRole] = []
    for carbon in sorted(component_atoms):
        if not mol.atoms[carbon].is_carbon:
            continue
        carbonyl_oxygen = _double_oxygen(mol, component_atoms, carbon)
        if carbonyl_oxygen is None:
            continue
        for oxygen in mol.get_neighbors(carbon):
            if oxygen == carbonyl_oxygen or oxygen not in component_atoms or mol.atoms[oxygen].symbol != "O":
                continue
            bond = mol.get_bond(carbon, oxygen)
            if bond is None or bond.order != 1:
                continue
            distal_oxygens = [
                atom_id
                for atom_id in mol.get_neighbors(oxygen)
                if atom_id != carbon and atom_id in component_atoms and mol.atoms[atom_id].symbol == "O"
            ]
            if len(distal_oxygens) != 1:
                continue
            distal = distal_oxygens[0]
            role = _classify_peroxy_carbonyl(mol, component_atoms, carbon, carbonyl_oxygen, oxygen, distal)
            if role is not None:
                roles.append(role)
    return roles


def _classify_peroxy_carbonyl(
    mol: Molecule,
    component_atoms: set[int],
    carbonyl: int,
    carbonyl_oxygen: int,
    proximal_oxygen: int,
    distal_oxygen: int,
) -> PeroxyCarbonylRole | None:
    distal_neighbors = [atom for atom in mol.get_neighbors(distal_oxygen) if atom != proximal_oxygen]
    if not distal_neighbors:
        kind = PeroxyCarbonylKind.PEROXOATE if mol.atoms[distal_oxygen].charge < 0 else PeroxyCarbonylKind.HYDROPEROXIDE
        return PeroxyCarbonylRole(carbonyl, carbonyl_oxygen, proximal_oxygen, distal_oxygen, kind)
    if len(distal_neighbors) != 1:
        return None
    attachment = distal_neighbors[0]
    if attachment not in component_atoms:
        return None
    if attachment == carbonyl:
        return PeroxyCarbonylRole(
            carbonyl,
            carbonyl_oxygen,
            proximal_oxygen,
            distal_oxygen,
            PeroxyCarbonylKind.CYCLIC_PEROXY_ESTER,
            attachment,
        )
    if mol.atoms[attachment].is_carbon:
        kind = (
            PeroxyCarbonylKind.CARBONATE_LIKE
            if _has_second_single_oxygen(mol, component_atoms, carbonyl, proximal_oxygen)
            else PeroxyCarbonylKind.PEROXOATE
        )
        return PeroxyCarbonylRole(carbonyl, carbonyl_oxygen, proximal_oxygen, distal_oxygen, kind, attachment)
    return None


def _double_oxygen(mol: Molecule, component_atoms: set[int], carbon: int) -> int | None:
    oxygens = [
        atom_id
        for atom_id in mol.get_neighbors(carbon)
        if atom_id in component_atoms
        and mol.atoms[atom_id].symbol == "O"
        and (bond := mol.get_bond(carbon, atom_id)) is not None
        and bond.order == 2
    ]
    return oxygens[0] if len(oxygens) == 1 else None


def _has_second_single_oxygen(
    mol: Molecule,
    component_atoms: set[int],
    carbonyl: int,
    proximal_oxygen: int,
) -> bool:
    return any(
        atom_id != proximal_oxygen
        and atom_id in component_atoms
        and mol.atoms[atom_id].symbol == "O"
        and (bond := mol.get_bond(carbonyl, atom_id)) is not None
        and bond.order == 1
        for atom_id in mol.get_neighbors(carbonyl)
    )
