"""Graph projections for high-risk nomenclature roles.

Role certificates are deliberately smaller than full naming objects.  They
record the part of the molecular graph that a perceived group, assembled name
term, or future role renderer claims to represent.  Audits can then reason
about atoms, bonds, charges, and locants without parsing final strings.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .assembly_parts import AssemblyParts, NameAtomBinding
from .molecule import Molecule
from .name_bindings import refresh_name_atom_bindings

if TYPE_CHECKING:  # pragma: no cover - imported for type checkers only
    from .perception import PerceivedGroup


@dataclass(frozen=True)
class RoleProjection:
    """One graph-bound name role projection."""

    stage: str
    role: str
    term: str = ""
    atom_ids: frozenset[int] = frozenset()
    bond_ids: frozenset[int] = frozenset()
    locants: tuple[str, ...] = ()
    charges_by_atom: dict[int, int] = field(default_factory=dict)

    @classmethod
    def from_name_binding(cls, mol: Molecule, binding: NameAtomBinding) -> "RoleProjection":
        """Build a projection from an emitted name binding."""

        atom_ids = frozenset(binding.atom_ids)
        return cls(
            stage=binding.stage,
            role=binding.role,
            term=binding.term,
            atom_ids=atom_ids,
            bond_ids=frozenset(binding.bond_ids),
            locants=tuple(binding.locants),
            charges_by_atom=_charges_for_atoms(mol, atom_ids),
        )

    @classmethod
    def from_perceived_group(cls, mol: Molecule, group: "PerceivedGroup") -> "RoleProjection":
        """Build a projection from functional-group perception data."""

        atom_ids = frozenset(group.atom_ids)
        return cls(
            stage="perception",
            role=group.role or group.key,
            term=group.key,
            atom_ids=atom_ids,
            bond_ids=frozenset(group.bond_ids),
            locants=(),
            charges_by_atom=_charges_for_atoms(mol, atom_ids),
        )


@dataclass(frozen=True)
class RoleCertificate:
    """Auditable graph projection for one or more related naming roles."""

    key: str
    projections: tuple[RoleProjection, ...]
    decision_reasons: tuple[str, ...] = ()

    @property
    def represented_atoms(self) -> set[int]:
        atoms: set[int] = set()
        for projection in self.projections:
            atoms.update(projection.atom_ids)
        return atoms

    @property
    def represented_bonds(self) -> set[int]:
        bonds: set[int] = set()
        for projection in self.projections:
            bonds.update(projection.bond_ids)
        return bonds

    @property
    def represented_charges(self) -> dict[int, int]:
        charges: dict[int, int] = {}
        for projection in self.projections:
            charges.update(projection.charges_by_atom)
        return charges

    @property
    def locants_by_atom(self) -> dict[int, str]:
        """Return atom locants when a projection binds one atom per locant."""

        locants: dict[int, str] = {}
        for projection in self.projections:
            if len(projection.atom_ids) != len(projection.locants):
                continue
            for atom_id, locant in zip(sorted(projection.atom_ids), projection.locants, strict=False):
                locants[atom_id] = locant
        return locants

    def missing_charged_atoms(self, mol: Molecule, expected_atoms: set[int]) -> set[int]:
        """Return charged expected atoms not covered by this certificate."""

        expected_charged = {
            atom_id
            for atom_id in expected_atoms
            if atom_id in mol.atoms and mol.atoms[atom_id].charge != 0
        }
        return expected_charged - set(self.represented_charges)


@dataclass(frozen=True)
class RoleCertificateAudit:
    """Result of checking a role certificate against a renderer template."""

    certificate: RoleCertificate
    expected_atoms: frozenset[int]
    expected_bonds: frozenset[int] = frozenset()
    audit_errors: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.audit_errors


def audit_role_certificate(
    mol: Molecule,
    certificate: RoleCertificate,
    *,
    expected_atoms: set[int] | frozenset[int],
    expected_bonds: set[int] | frozenset[int] = frozenset(),
    require_charged_atoms: bool = True,
) -> RoleCertificateAudit:
    """Check that a renderer role certificate covers the graph it claims.

    High-risk templates use this at the rendering boundary so they cannot emit
    a charge-pair, hypervalent, or peroxy-carbonyl name unless the graph
    projection explicitly covers the expected atoms, bonds, and charges.
    """

    expected_atom_set = frozenset(expected_atoms)
    expected_bond_set = frozenset(expected_bonds)
    errors: list[str] = []
    missing_atoms = expected_atom_set - certificate.represented_atoms
    extra_atoms = certificate.represented_atoms - expected_atom_set
    if missing_atoms or extra_atoms:
        errors.append(f"certificate atom coverage mismatch: missing={sorted(missing_atoms)} extra={sorted(extra_atoms)}")
    missing_bonds = expected_bond_set - certificate.represented_bonds
    if missing_bonds:
        errors.append(f"certificate bond coverage mismatch: missing={sorted(missing_bonds)}")
    if require_charged_atoms:
        missing_charges = certificate.missing_charged_atoms(mol, set(expected_atom_set))
        if missing_charges:
            errors.append(f"certificate charge coverage mismatch: missing={sorted(missing_charges)}")
    return RoleCertificateAudit(
        certificate=certificate,
        expected_atoms=expected_atom_set,
        expected_bonds=expected_bond_set,
        audit_errors=tuple(errors),
    )


def certificate_from_perceived_group(mol: Molecule, group: "PerceivedGroup") -> RoleCertificate:
    """Create a role certificate from a perceived functional group."""

    return RoleCertificate(
        key=group.key,
        projections=(RoleProjection.from_perceived_group(mol, group),),
        decision_reasons=tuple(group.decision_reasons),
    )


def certificates_from_assembly(mol: Molecule, parts: AssemblyParts) -> list[RoleCertificate]:
    """Create role certificates from the current assembled name bindings."""

    bindings = parts.name_atom_bindings or refresh_name_atom_bindings(parts)
    return [
        RoleCertificate(
            key=f"{binding.stage}:{binding.role}",
            projections=(RoleProjection.from_name_binding(mol, binding),),
        )
        for binding in bindings
    ]


def _charges_for_atoms(mol: Molecule, atom_ids: frozenset[int]) -> dict[int, int]:
    return {
        atom_id: mol.atoms[atom_id].charge
        for atom_id in atom_ids
        if atom_id in mol.atoms and mol.atoms[atom_id].charge != 0
    }
