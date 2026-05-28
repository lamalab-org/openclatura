"""Graph roles for nitrogen-chain functional groups.

This module classifies N-N fragments before the naming layer decides whether
they are prefixes or principal suffix groups. The role is derived only from
graph topology, bond order, charge, and ring membership.
"""

from dataclasses import dataclass

from .molecule import Molecule


@dataclass(frozen=True)
class NitrogenChainRole:
    """A graph-bound nitrogen-chain naming role."""

    key: str
    is_principal_candidate: bool
    attachment_atom: int
    atom_ids: frozenset[int]
    variant: str
    reason: str


def nitrogen_chain_roles(mol: Molecule, cyclic_atoms: set[int], consumed: set[int] | None = None) -> list[NitrogenChainRole]:
    """Return azido/diazo/diazonio/hydrazone/hydrazine roles in priority order."""

    blocked = consumed or set()
    roles: list[NitrogenChainRole] = []
    roles.extend(_diazo_roles(mol, cyclic_atoms, blocked))
    roles.extend(_azido_roles(mol, cyclic_atoms, blocked | _role_atoms(roles)))
    roles.extend(_hydrazone_roles(mol, cyclic_atoms, blocked | _role_atoms(roles)))
    roles.extend(_hydrazine_roles(mol, cyclic_atoms, blocked | _role_atoms(roles)))
    return _dedupe_roles(roles)


def _role_atoms(roles: list[NitrogenChainRole]) -> set[int]:
    return {atom_idx for role in roles for atom_idx in role.atom_ids}


def _dedupe_roles(roles: list[NitrogenChainRole]) -> list[NitrogenChainRole]:
    result = []
    seen_atoms: set[int] = set()
    for role in roles:
        if seen_atoms & set(role.atom_ids):
            continue
        result.append(role)
        seen_atoms.update(role.atom_ids)
    return result


def _diazo_roles(mol: Molecule, cyclic_atoms: set[int], blocked: set[int]) -> list[NitrogenChainRole]:
    roles = []
    for carbon in mol:
        if not carbon.is_carbon:
            continue
        for n1 in mol.get_neighbors(carbon.idx):
            if n1 in cyclic_atoms or n1 in blocked or mol.atoms[n1].symbol != "N":
                continue
            c_n_bond = mol.get_bond(carbon.idx, n1)
            if c_n_bond is None or c_n_bond.order not in {1, 2}:
                continue
            n2_candidates = [
                n
                for n in mol.get_neighbors(n1)
                if n not in blocked and n != carbon.idx and n not in cyclic_atoms and mol.atoms[n].symbol == "N"
            ]
            if len(n2_candidates) != 1:
                continue
            n2 = n2_candidates[0]
            if _other_non_h_neighbors(mol, n2, {n1}):
                continue
            n_n_bond = mol.get_bond(n1, n2)
            if n_n_bond is None or n_n_bond.order < 2:
                continue
            key = "diazonio" if c_n_bond.order == 1 and (mol.atoms[n1].charge > 0 or mol.atoms[n2].charge > 0) else "diazo"
            roles.append(
                NitrogenChainRole(
                    key=key,
                    is_principal_candidate=False,
                    attachment_atom=carbon.idx,
                    atom_ids=frozenset({n1, n2}),
                    variant="carbon_bound",
                    reason=f"Matched carbon-bound {key} N-N fragment at atom {carbon.idx}.",
                )
            )
    return roles


def _azido_roles(mol: Molecule, cyclic_atoms: set[int], blocked: set[int]) -> list[NitrogenChainRole]:
    roles = []
    for n_attach in mol:
        if n_attach.symbol != "N" or n_attach.idx in cyclic_atoms or n_attach.idx in blocked:
            continue
        external = _single_external_attachment(mol, n_attach.idx, cyclic_atoms)
        if external is None:
            continue
        ext_atom, ext_bond_order = external
        if ext_bond_order != 1:
            continue
        n2_candidates = [
            n
            for n in mol.get_neighbors(n_attach.idx)
            if n not in blocked and n not in cyclic_atoms and mol.atoms[n].symbol == "N"
        ]
        if len(n2_candidates) != 1:
            continue
        n2 = n2_candidates[0]
        n3_candidates = [
            n
            for n in mol.get_neighbors(n2)
            if n not in blocked and n != n_attach.idx and n not in cyclic_atoms and mol.atoms[n].symbol == "N"
        ]
        if len(n3_candidates) != 1:
            continue
        n3 = n3_candidates[0]
        if _other_non_h_neighbors(mol, n3, {n2}):
            continue
        first_bond = mol.get_bond(n_attach.idx, n2)
        second_bond = mol.get_bond(n2, n3)
        key = "azido"
        if (
            first_bond is not None
            and second_bond is not None
            and mol.atoms[n_attach.idx].charge == 0
            and mol.atoms[n2].charge == 0
            and mol.atoms[n3].charge == 0
        ):
            if first_bond.order == 1 and second_bond.order == 2:
                key = "diazenylamino"
            elif first_bond.order == 2 and second_bond.order == 1:
                key = "aminodiazenyl"
            elif first_bond.order == 1 and second_bond.order == 1:
                key = "hydrazinylamino"
        roles.append(
            NitrogenChainRole(
                key=key,
                is_principal_candidate=False,
                attachment_atom=ext_atom,
                atom_ids=frozenset({n_attach.idx, n2, n3}),
                variant="linear_n3",
                reason=f"Matched singly attached terminal N3 {key} fragment at atom {ext_atom}.",
            )
        )
    return roles


def _hydrazone_roles(mol: Molecule, cyclic_atoms: set[int], blocked: set[int]) -> list[NitrogenChainRole]:
    roles = []
    for n1 in mol:
        if n1.symbol != "N" or n1.idx in cyclic_atoms or n1.idx in blocked:
            continue
        carbon = next(
            (
                c
                for c in mol.get_neighbors(n1.idx)
                if mol.atoms[c].is_carbon and (bond := mol.get_bond(n1.idx, c)) is not None and bond.order == 2
            ),
            None,
        )
        if carbon is None:
            continue
        n2_candidates = [
            n
            for n in mol.get_neighbors(n1.idx)
            if n not in blocked and n != carbon and n not in cyclic_atoms and mol.atoms[n].symbol == "N"
        ]
        if len(n2_candidates) != 1:
            continue
        n2 = n2_candidates[0]
        n1_n2_bond = mol.get_bond(n1.idx, n2)
        if n1_n2_bond is None or n1_n2_bond.order != 1:
            continue
        if _has_non_h_multiple_bond_neighbor(mol, n2, {n1.idx}) and not _has_terminal_imino_substituent(
            mol, n2, {n1.idx}
        ):
            continue
        key = _hydrazone_key(mol, carbon, cyclic_atoms)
        roles.append(
            NitrogenChainRole(
                key=key,
                is_principal_candidate=True,
                attachment_atom=_hydrazone_attachment(mol, carbon, cyclic_atoms),
                atom_ids=frozenset({n1.idx, n2}),
                variant="carbon_nitrogen_double_bond",
                reason=f"Matched C=N-N hydrazone fragment at atom {carbon}.",
            )
        )
    return roles


def _hydrazone_key(mol: Molecule, carbon: int, cyclic_atoms: set[int]) -> str:
    ring_neighbors = [n for n in mol.get_neighbors(carbon) if n in cyclic_atoms]
    carbon_neighbors = [n for n in mol.get_neighbors(carbon) if mol.atoms[n].is_carbon]
    non_h_neighbors = [n for n in mol.get_neighbors(carbon) if mol.atoms[n].symbol != "H"]
    if carbon not in cyclic_atoms and len(ring_neighbors) == 1 and len(carbon_neighbors) == 1:
        bond = mol.get_bond(carbon, ring_neighbors[0])
        if bond is not None and bond.order == 1 and len(non_h_neighbors) == 2:
            return "ring_aldehyde_hydrazone"
    if len(carbon_neighbors) <= 1 and carbon not in cyclic_atoms:
        return "aldehyde_hydrazone"
    return "hydrazone"


def _hydrazone_attachment(mol: Molecule, carbon: int, cyclic_atoms: set[int]) -> int:
    if carbon not in cyclic_atoms:
        ring_neighbors = [n for n in mol.get_neighbors(carbon) if n in cyclic_atoms]
        carbon_neighbors = [n for n in mol.get_neighbors(carbon) if mol.atoms[n].is_carbon]
        if len(ring_neighbors) == 1 and len(carbon_neighbors) == 1:
            bond = mol.get_bond(carbon, ring_neighbors[0])
            if bond is not None and bond.order == 1:
                return ring_neighbors[0]
    return carbon


def _hydrazine_roles(mol: Molecule, cyclic_atoms: set[int], blocked: set[int]) -> list[NitrogenChainRole]:
    roles = []
    for n1 in mol:
        if n1.symbol != "N" or n1.idx in cyclic_atoms or n1.idx in blocked:
            continue
        n2_candidates = [
            n
            for n in mol.get_neighbors(n1.idx)
            if n not in blocked and n not in cyclic_atoms and mol.atoms[n].symbol == "N"
        ]
        if len(n2_candidates) != 1:
            continue
        n2 = n2_candidates[0]
        bond = mol.get_bond(n1.idx, n2)
        if bond is None or bond.order != 1:
            continue
        c_neighbors = [n for n in mol.get_neighbors(n1.idx) if mol.atoms[n].is_carbon]
        if len(c_neighbors) != 1:
            continue
        c_bond = mol.get_bond(n1.idx, c_neighbors[0])
        if c_bond is None or c_bond.order != 1:
            continue
        if _has_non_h_multiple_bond_neighbor(mol, n2, {n1.idx}):
            continue
        attached_to_ring_parent = c_neighbors[0] in cyclic_atoms
        roles.append(
            NitrogenChainRole(
                key="hydrazine",
                is_principal_candidate=False,
                attachment_atom=c_neighbors[0],
                atom_ids=frozenset({n1.idx, n2}),
                variant="prefix",
                reason=f"Matched C-N-N hydrazine fragment at atom {c_neighbors[0]}.",
            )
        )
    return roles


def _single_external_attachment(
    mol: Molecule,
    nitrogen: int,
    cyclic_atoms: set[int],
) -> tuple[int, int] | None:
    external = [
        n
        for n in mol.get_neighbors(nitrogen)
        if mol.atoms[n].symbol != "N" and mol.atoms[n].symbol != "H"
    ]
    if len(external) != 1:
        return None
    atom_idx = external[0]
    bond = mol.get_bond(nitrogen, atom_idx)
    if bond is None:
        return None
    return atom_idx, bond.order


def _other_non_h_neighbors(mol: Molecule, atom_idx: int, allowed: set[int]) -> list[int]:
    return [
        n
        for n in mol.get_neighbors(atom_idx)
        if n not in allowed and mol.atoms[n].symbol != "H"
    ]


def _has_non_h_multiple_bond_neighbor(mol: Molecule, atom_idx: int, allowed: set[int]) -> bool:
    for neighbor in mol.get_neighbors(atom_idx):
        if neighbor in allowed or mol.atoms[neighbor].symbol == "H":
            continue
        bond = mol.get_bond(atom_idx, neighbor)
        if bond is not None and bond.order != 1:
            return True
    return False


def _has_terminal_imino_substituent(mol: Molecule, atom_idx: int, allowed: set[int]) -> bool:
    for neighbor in mol.get_neighbors(atom_idx):
        if neighbor in allowed or mol.atoms[neighbor].symbol not in {"C", "N"}:
            continue
        bond = mol.get_bond(atom_idx, neighbor)
        if bond is None or bond.order != 2:
            continue
        if mol.atoms[neighbor].symbol == "N" and _other_non_h_neighbors(mol, neighbor, {atom_idx}):
            continue
        return True
    return False
