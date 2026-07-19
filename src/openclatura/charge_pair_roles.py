"""Graph roles for charge-separated fragments.

This module classifies formal charge pairs before any naming template is
allowed to render them.  The classifier is intentionally graph-first: a role is
defined by charged atoms, the bonds that connect or conjugate them, and the
atoms represented by that charge pair.  Renderers can then opt into specific
role keys instead of applying a global anion suffix.
"""

from dataclasses import dataclass

from .molecule import Molecule
from .role_certificate import RoleCertificate, RoleCertificateAudit, RoleProjection, audit_role_certificate

SUPPORTED_TEMPLATE_ROLES = frozenset(
    {
        "diazonium_azanide",
        "n_oxide",
        "phosphane_borane_zwitterion",
        "sulfonium_ylide_single_bond",
        "terminal_chalcogenide_heteroarenium",
    }
)


@dataclass(frozen=True)
class ChargePairRole:
    """A graph-bound formal charge-pair role."""

    key: str
    positive_atom: int
    negative_atom: int
    atom_ids: frozenset[int]
    bond_ids: frozenset[int] = frozenset()
    template_supported: bool = False
    reason: str = ""

    @property
    def charge_pattern(self) -> tuple[tuple[int, int], tuple[int, int]]:
        return ((self.positive_atom, 1), (self.negative_atom, -1))

    def certificate(self, mol: Molecule) -> RoleCertificate:
        """Return an auditable certificate for this charge pair."""

        return RoleCertificate(
            key=self.key,
            projections=(
                RoleProjection(
                    stage="charge_pair",
                    role=self.key,
                    term=self.key,
                    atom_ids=self.atom_ids,
                    bond_ids=self.bond_ids,
                    charges_by_atom={
                        atom_id: mol.atoms[atom_id].charge
                        for atom_id in self.atom_ids
                        if atom_id in mol.atoms and mol.atoms[atom_id].charge != 0
                    },
                ),
            ),
            decision_reasons=(self.reason,),
        )

    def template_audit(self, mol: Molecule) -> RoleCertificateAudit:
        """Audit the certificate required by a charge-pair renderer template."""

        return audit_role_certificate(
            mol,
            self.certificate(mol),
            expected_atoms=self.atom_ids,
            expected_bonds=self.bond_ids,
            require_charged_atoms=True,
        )


def charge_pair_roles(mol: Molecule, component_atoms: set[int] | None = None) -> list[ChargePairRole]:
    """Classify supported and high-risk formal charge pairs in a component."""

    atoms = set(component_atoms) if component_atoms is not None else set(mol.atoms)
    roles: list[ChargePairRole] = []
    positives = [idx for idx in atoms if mol.atoms[idx].charge > 0]
    negatives = [idx for idx in atoms if mol.atoms[idx].charge < 0]
    seen: set[tuple[str, int, int]] = set()

    for pos in positives:
        for neg in negatives:
            role = _classify_charge_pair(mol, atoms, pos, neg)
            if role is None:
                continue
            identity = (role.key, role.positive_atom, role.negative_atom)
            if identity in seen:
                continue
            seen.add(identity)
            roles.append(role)
    return roles


def unsupported_charge_pair_roles(mol: Molecule, component_atoms: set[int] | None = None) -> list[ChargePairRole]:
    """Return charge-pair roles that should not use generic fallback spelling."""

    return [role for role in charge_pair_roles(mol, component_atoms) if not role.template_supported]


def _classify_charge_pair(
    mol: Molecule,
    component_atoms: set[int],
    positive: int,
    negative: int,
) -> ChargePairRole | None:
    pos_atom = mol.atoms[positive]
    neg_atom = mol.atoms[negative]
    bond = mol.get_bond(positive, negative)

    if pos_atom.symbol == "S" and neg_atom.symbol == "C" and bond is not None:
        if bond.order == 1:
            return _role(
                mol,
                "sulfonium_ylide_single_bond",
                positive,
                negative,
                {positive, negative},
                {bond.idx},
                "Matched explicit charge-separated S+--C- ylide.",
            )
        return _role(
            mol,
            "sulfur_carbanion_resonance_charge_pair",
            positive,
            negative,
            {positive, negative},
            {bond.idx},
            "Matched S+/C- charge pair on a non-single bond; no safe OPSIN template is registered.",
            supported=False,
        )

    if pos_atom.symbol == "N" and neg_atom.symbol == "O" and bond is not None and bond.order == 1:
        return _role(
            mol,
            "n_oxide",
            positive,
            negative,
            {positive, negative},
            {bond.idx},
            "Matched N+-O- oxide charge pair.",
        )

    if pos_atom.symbol == "N" and neg_atom.symbol == "N" and bond is not None:
        return _role(
            mol,
            "diazonium_azanide",
            positive,
            negative,
            {positive, negative},
            {bond.idx},
            "Matched adjacent N+/N- diazonium-azanide charge pair.",
        )

    if pos_atom.symbol in {"N", "O"} and neg_atom.symbol in {"S", "Se"}:
        path = _short_ring_conjugation_path(mol, component_atoms, positive, negative, max_depth=6)
        if path:
            bond_ids = _path_bond_ids(mol, path)
            return _role(
                mol,
                "terminal_chalcogenide_heteroarenium",
                positive,
                negative,
                set(path),
                bond_ids,
                "Matched terminal chalcogenide paired with a charged heteroarenium atom.",
            )

    if (
        pos_atom.symbol == "P"
        and neg_atom.symbol == "B"
        and bond is not None
        and _is_boranuide_phosphanium_pair(mol, component_atoms, positive, negative)
    ):
        return _role(
            mol,
            "phosphane_borane_zwitterion",
            positive,
            negative,
            {positive, negative},
            {bond.idx},
            "Matched B(-)(H)3-P(+) phosphane-borane zwitterion.",
        )

    if pos_atom.symbol in {"P", "B"} or neg_atom.symbol in {"P", "B"}:
        atom_ids = {positive, negative}
        bond_ids = {bond.idx} if bond is not None else set()
        return _role(
            mol,
            "pnictogen_boron_charge_pair",
            positive,
            negative,
            atom_ids,
            bond_ids,
            "Matched P/B-containing charge pair without a registered safe template.",
            supported=False,
        )

    if bond is not None:
        return _role(
            mol,
            "adjacent_formal_charge_pair",
            positive,
            negative,
            {positive, negative},
            {bond.idx},
            "Matched adjacent formal charge pair without a registered safe template.",
            supported=False,
        )
    return None


def _is_boranuide_phosphanium_pair(
    mol: Molecule,
    component_atoms: set[int],
    phosphorus: int,
    boron: int,
) -> bool:
    if mol.atoms[boron].symbol != "B" or mol.atoms[boron].charge >= 0 or mol.atoms[boron].total_h_count != 3:
        return False
    if mol.atoms[phosphorus].symbol != "P" or mol.atoms[phosphorus].charge <= 0:
        return False
    p_ligands = [n for n in mol.get_neighbors(phosphorus) if n in component_atoms and n != boron]
    return len(p_ligands) == 3


def _role(
    mol: Molecule,
    key: str,
    positive: int,
    negative: int,
    atom_ids: set[int],
    bond_ids: set[int],
    reason: str,
    *,
    supported: bool | None = None,
) -> ChargePairRole:
    return ChargePairRole(
        key=key,
        positive_atom=positive,
        negative_atom=negative,
        atom_ids=frozenset(atom_ids),
        bond_ids=frozenset(bond_ids),
        template_supported=key in SUPPORTED_TEMPLATE_ROLES if supported is None else supported,
        reason=reason,
    )


def _short_ring_conjugation_path(
    mol: Molecule,
    component_atoms: set[int],
    start: int,
    target: int,
    *,
    max_depth: int,
) -> tuple[int, ...]:
    queue: list[tuple[int, ...]] = [(start,)]
    while queue:
        path = queue.pop(0)
        if len(path) > max_depth + 1:
            continue
        current = path[-1]
        for neighbor in mol.get_neighbors(current):
            if neighbor not in component_atoms or neighbor in path:
                continue
            bond = mol.get_bond(current, neighbor)
            if bond is None:
                continue
            if neighbor == target:
                return tuple(path + (neighbor,))
            if not (mol.atoms[neighbor].is_aromatic or bond.order in {1, 2}):
                continue
            queue.append(path + (neighbor,))
    return ()


def _path_bond_ids(mol: Molecule, path: tuple[int, ...]) -> set[int]:
    bond_ids: set[int] = set()
    for left, right in zip(path, path[1:], strict=False):
        bond = mol.get_bond(left, right)
        if bond is not None:
            bond_ids.add(bond.idx)
    return bond_ids
