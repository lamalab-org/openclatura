"""Graph roles for nitrogen-chain functional groups.

This module classifies N-N fragments before the naming layer decides whether
they are prefixes or principal suffix groups. The role is derived only from
graph topology, bond order, charge, and ring membership.
"""

from dataclasses import dataclass

from .molecule import Molecule


@dataclass(frozen=True)
class NitrogenChainSegment:
    """One ordered bond in a nitrogen-chain role."""

    start_atom: int
    end_atom: int
    bond_order: int
    start_charge: int
    end_charge: int


@dataclass(frozen=True)
class HydrazoneSideMetadata:
    """Graph-side data needed to render hydrazone and amidinohydrazone names."""

    carbon_atom: int
    nitrogen_atom: int
    attachment_atom: int
    parent_kind: str
    amidino_tail_atoms: frozenset[int] = frozenset()


@dataclass(frozen=True)
class AzineSideMetadata:
    """One C=N side of an azine role."""

    carbon_atom: int
    nitrogen_atom: int
    side_atoms: frozenset[int]


@dataclass(frozen=True)
class AzineRole:
    """A graph-bound C=N-N=C azine role with ordered side metadata."""

    atom_ids: frozenset[int]
    ordered_atoms: tuple[int, ...]
    segments: tuple[NitrogenChainSegment, ...]
    left: AzineSideMetadata
    right: AzineSideMetadata
    reason: str


@dataclass(frozen=True)
class NitrogenChainRole:
    """A graph-bound nitrogen-chain naming role."""

    key: str
    is_principal_candidate: bool
    attachment_atom: int
    atom_ids: frozenset[int]
    variant: str
    reason: str
    ordered_atoms: tuple[int, ...] = ()
    segments: tuple[NitrogenChainSegment, ...] = ()
    charge_pattern: tuple[int, ...] = ()
    bond_orders: tuple[int, ...] = ()
    hydrazone_side: HydrazoneSideMetadata | None = None


@dataclass(frozen=True)
class NitrogenChainTemplate:
    """Declarative graph template for an ordered nitrogen-chain role."""

    key: str
    bond_orders: tuple[int | frozenset[int], ...]
    charges: tuple[int | frozenset[int] | None, ...]
    variant: str

    def matches(self, bond_orders: tuple[int, ...], charges: tuple[int, ...]) -> bool:
        if len(bond_orders) != len(self.bond_orders) or len(charges) != len(self.charges):
            return False
        return all(_template_value_matches(expected, actual) for expected, actual in zip(self.bond_orders, bond_orders)) and all(
            _template_value_matches(expected, actual) for expected, actual in zip(self.charges, charges)
        )


CARBON_BOUND_N2_TEMPLATES: tuple[NitrogenChainTemplate, ...] = (
    NitrogenChainTemplate("diazenyl", (1, 2), (0, 0, 0), "carbon_bound_neutral_diazene"),
    NitrogenChainTemplate("diazonio", (1, frozenset({2, 3})), (0, frozenset({0, 1}), frozenset({0, 1})), "carbon_bound_diazonium"),
    NitrogenChainTemplate("diazo", (2, frozenset({2, 3})), (None, None, None), "carbon_bound_diazo"),
)

TERMINAL_N3_TEMPLATES: tuple[NitrogenChainTemplate, ...] = (
    NitrogenChainTemplate("azido", (1, 2, 2), (None, 0, 0, 0), "terminal_cumulene_azide"),
    NitrogenChainTemplate("azido", (1, 2, 2), (None, 0, 1, -1), "terminal_charge_separated_azide"),
    NitrogenChainTemplate("azido", (2, 2, 2), (None, 0, 1, -1), "terminal_charge_separated_azide"),
    NitrogenChainTemplate("azido", (1, 2, 2), (None, -1, 1, 0), "terminal_charge_separated_azide"),
    NitrogenChainTemplate("azido", (2, 2, 2), (None, -1, 1, 0), "terminal_charge_separated_azide"),
    NitrogenChainTemplate("diazenylamino", (1, 1, 2), (None, 0, 0, 0), "neutral_diazenylamino"),
    NitrogenChainTemplate("aminodiazenyl", (1, 2, 1), (None, 0, 0, 0), "neutral_aminodiazenyl"),
    NitrogenChainTemplate("hydrazinylamino", (1, 1, 1), (None, 0, 0, 0), "neutral_hydrazinylamino"),
)


def _template_value_matches(expected, actual: int) -> bool:
    if expected is None:
        return True
    if isinstance(expected, frozenset):
        return actual in expected
    return actual == expected


def _match_template(
    templates: tuple[NitrogenChainTemplate, ...],
    bond_orders: tuple[int, ...],
    charges: tuple[int, ...],
) -> NitrogenChainTemplate | None:
    return next((template for template in templates if template.matches(bond_orders, charges)), None)


def nitrogen_chain_roles(mol: Molecule, cyclic_atoms: set[int], consumed: set[int] | None = None) -> list[NitrogenChainRole]:
    """Return azido/diazo/diazonio/hydrazone/hydrazine roles in priority order."""

    blocked = consumed or set()
    roles: list[NitrogenChainRole] = []
    roles.extend(_diazo_roles(mol, cyclic_atoms, blocked))
    roles.extend(_azido_roles(mol, cyclic_atoms, blocked | _role_atoms(roles)))
    roles.extend(_hydrazone_roles(mol, cyclic_atoms, blocked | _role_atoms(roles)))
    roles.extend(_hydrazine_roles(mol, cyclic_atoms, blocked | _role_atoms(roles)))
    return _dedupe_roles(roles)


def acid_derived_hydrazone_roles(
    mol: Molecule,
    cyclic_atoms: set[int],
    consumed: set[int] | None = None,
) -> list[NitrogenChainRole]:
    """Return acid-derived hydrazone-family roles for audit/template gating."""

    return _dedupe_roles(_acid_derived_hydrazone_roles(mol, cyclic_atoms, consumed or set()))


def _role_atoms(roles: list[NitrogenChainRole]) -> set[int]:
    return {atom_idx for role in roles for atom_idx in role.atom_ids}


def _make_role(
    mol: Molecule,
    *,
    key: str,
    is_principal_candidate: bool,
    attachment_atom: int,
    atom_ids: frozenset[int],
    variant: str,
    reason: str,
    ordered_atoms: tuple[int, ...],
    hydrazone_side: HydrazoneSideMetadata | None = None,
) -> NitrogenChainRole:
    return NitrogenChainRole(
        key=key,
        is_principal_candidate=is_principal_candidate,
        attachment_atom=attachment_atom,
        atom_ids=atom_ids,
        variant=variant,
        reason=reason,
        ordered_atoms=ordered_atoms,
        segments=_chain_segments(mol, ordered_atoms),
        charge_pattern=tuple(mol.atoms[idx].charge for idx in ordered_atoms),
        bond_orders=_chain_bond_orders(mol, ordered_atoms),
        hydrazone_side=hydrazone_side,
    )


def _chain_segments(mol: Molecule, ordered_atoms: tuple[int, ...]) -> tuple[NitrogenChainSegment, ...]:
    segments: list[NitrogenChainSegment] = []
    for start, end in zip(ordered_atoms, ordered_atoms[1:]):
        bond = mol.get_bond(start, end)
        if bond is None:
            continue
        segments.append(
            NitrogenChainSegment(
                start_atom=start,
                end_atom=end,
                bond_order=bond.order,
                start_charge=mol.atoms[start].charge,
                end_charge=mol.atoms[end].charge,
            )
        )
    return tuple(segments)


def _chain_bond_orders(mol: Molecule, ordered_atoms: tuple[int, ...]) -> tuple[int, ...]:
    orders: list[int] = []
    for start, end in zip(ordered_atoms, ordered_atoms[1:]):
        bond = mol.get_bond(start, end)
        orders.append(bond.order if bond is not None else 0)
    return tuple(orders)


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
            template = _match_template(
                CARBON_BOUND_N2_TEMPLATES,
                (c_n_bond.order, n_n_bond.order),
                (mol.atoms[carbon.idx].charge, mol.atoms[n1].charge, mol.atoms[n2].charge),
            )
            if template is None:
                continue
            roles.append(
                _make_role(
                    mol,
                    key=template.key,
                    is_principal_candidate=False,
                    attachment_atom=carbon.idx,
                    atom_ids=frozenset({n1, n2}),
                    variant=template.variant,
                    reason=f"Matched carbon-bound {template.key} N-N fragment at atom {carbon.idx}.",
                    ordered_atoms=(carbon.idx, n1, n2),
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
        if first_bond is None or second_bond is None:
            continue
        template = _match_template(
            TERMINAL_N3_TEMPLATES,
            (ext_bond_order, first_bond.order, second_bond.order),
            (
                mol.atoms[ext_atom].charge,
                mol.atoms[n_attach.idx].charge,
                mol.atoms[n2].charge,
                mol.atoms[n3].charge,
            ),
        )
        if template is None:
            continue
        roles.append(
            _make_role(
                mol,
                key=template.key,
                is_principal_candidate=False,
                attachment_atom=ext_atom,
                atom_ids=frozenset({n_attach.idx, n2, n3}),
                variant=template.variant,
                reason=f"Matched singly attached terminal N3 {template.key} fragment at atom {ext_atom}.",
                ordered_atoms=(ext_atom, n_attach.idx, n2, n3),
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
        amidino_tail_atoms = _amidinohydrazone_tail_atoms(mol, n2, {n1.idx})
        if (
            _has_non_h_multiple_bond_neighbor(mol, n2, {n1.idx})
            and not _has_terminal_imino_substituent(mol, n2, {n1.idx})
            and not amidino_tail_atoms
        ):
            continue
        key = _hydrazone_key(mol, carbon, cyclic_atoms, amidino=bool(amidino_tail_atoms))
        attachment = _hydrazone_attachment(mol, carbon, cyclic_atoms)
        ordered_tail = tuple(sorted(amidino_tail_atoms))
        roles.append(
            _make_role(
                mol,
                key=key,
                is_principal_candidate=True,
                attachment_atom=attachment,
                atom_ids=frozenset({n1.idx, n2} | amidino_tail_atoms),
                variant="carbon_nitrogen_double_bond",
                reason=f"Matched C=N-N hydrazone fragment at atom {carbon}.",
                ordered_atoms=(carbon, n1.idx, n2, *ordered_tail),
                hydrazone_side=HydrazoneSideMetadata(
                    carbon_atom=carbon,
                    nitrogen_atom=n1.idx,
                    attachment_atom=attachment,
                    parent_kind=key,
                    amidino_tail_atoms=frozenset(amidino_tail_atoms),
                ),
            )
        )
    return roles


def _acid_derived_hydrazone_roles(
    mol: Molecule,
    cyclic_atoms: set[int],
    blocked: set[int],
) -> list[NitrogenChainRole]:
    """Classify amidrazone, imidohydrazide, and thiohydrazide graph families.

    These are acid-derived C/N/S systems and must not be consumed by the
    ordinary aldehyde/ketone hydrazone detector.
    """

    roles: list[NitrogenChainRole] = []
    for carbon in mol:
        if not carbon.is_carbon or carbon.idx in cyclic_atoms or carbon.idx in blocked:
            continue
        single_n = _single_bonded_nitrogens(mol, carbon.idx, blocked | cyclic_atoms)
        double_n = _double_bonded_nitrogens(mol, carbon.idx, blocked | cyclic_atoms)
        double_s = _double_bonded_atoms(mol, carbon.idx, "S", blocked | cyclic_atoms)

        if len(single_n) >= 2 and len(double_n) == 1:
            hydrazide_n = _nitrogen_with_terminal_n(mol, single_n, {carbon.idx}, blocked | cyclic_atoms)
            amino_n = next((n for n in single_n if n != hydrazide_n), None)
            if hydrazide_n is not None and amino_n is not None:
                terminal_n = _terminal_nitrogen_neighbor(mol, hydrazide_n, {carbon.idx}, blocked | cyclic_atoms)
                roles.append(
                    _make_role(
                        mol,
                        key="hydrazonamide",
                        is_principal_candidate=True,
                        attachment_atom=carbon.idx,
                        atom_ids=frozenset({carbon.idx, double_n[0], hydrazide_n, terminal_n, amino_n}),
                        variant="acid_derived_amidrazone",
                        reason=f"Matched C(=N)(N)(N-N) hydrazonamide fragment at atom {carbon.idx}.",
                        ordered_atoms=(double_n[0], carbon.idx, hydrazide_n, terminal_n),
                    )
                )
                continue

        if len(single_n) == 1 and len(double_n) == 1:
            terminal_n = _terminal_nitrogen_neighbor(mol, single_n[0], {carbon.idx}, blocked | cyclic_atoms)
            if terminal_n is not None:
                roles.append(
                    _make_role(
                        mol,
                        key="imidohydrazide",
                        is_principal_candidate=True,
                        attachment_atom=carbon.idx,
                        atom_ids=frozenset({carbon.idx, double_n[0], single_n[0], terminal_n}),
                        variant="acid_derived_imidohydrazide",
                        reason=f"Matched C(=N)(N-N) imidohydrazide fragment at atom {carbon.idx}.",
                        ordered_atoms=(double_n[0], carbon.idx, single_n[0], terminal_n),
                    )
                )
                continue

        if len(single_n) == 1 and len(double_s) == 1:
            terminal_n = _terminal_nitrogen_neighbor(mol, single_n[0], {carbon.idx}, blocked | cyclic_atoms)
            if terminal_n is not None:
                roles.append(
                    _make_role(
                        mol,
                        key="thiohydrazide",
                        is_principal_candidate=True,
                        attachment_atom=carbon.idx,
                        atom_ids=frozenset({carbon.idx, double_s[0], single_n[0], terminal_n}),
                        variant="acid_derived_thiohydrazide",
                        reason=f"Matched C(=S)(N-N) thiohydrazide fragment at atom {carbon.idx}.",
                        ordered_atoms=(double_s[0], carbon.idx, single_n[0], terminal_n),
                    )
                )
    return roles


def _single_bonded_nitrogens(mol: Molecule, carbon: int, blocked: set[int]) -> list[int]:
    return [
        neighbor
        for neighbor in mol.get_neighbors(carbon)
        if neighbor not in blocked
        and mol.atoms[neighbor].symbol == "N"
        and (bond := mol.get_bond(carbon, neighbor)) is not None
        and bond.order == 1
    ]


def _double_bonded_nitrogens(mol: Molecule, carbon: int, blocked: set[int]) -> list[int]:
    return _double_bonded_atoms(mol, carbon, "N", blocked)


def _double_bonded_atoms(mol: Molecule, carbon: int, symbol: str, blocked: set[int]) -> list[int]:
    return [
        neighbor
        for neighbor in mol.get_neighbors(carbon)
        if neighbor not in blocked
        and mol.atoms[neighbor].symbol == symbol
        and (bond := mol.get_bond(carbon, neighbor)) is not None
        and bond.order == 2
    ]


def _nitrogen_with_terminal_n(
    mol: Molecule,
    nitrogens: list[int],
    excluded: set[int],
    blocked: set[int],
) -> int | None:
    matches = [nitrogen for nitrogen in nitrogens if _terminal_nitrogen_neighbor(mol, nitrogen, excluded, blocked) is not None]
    return matches[0] if len(matches) == 1 else None


def _terminal_nitrogen_neighbor(
    mol: Molecule,
    nitrogen: int,
    excluded: set[int],
    blocked: set[int],
) -> int | None:
    candidates = [
        neighbor
        for neighbor in mol.get_neighbors(nitrogen)
        if neighbor not in excluded
        and neighbor not in blocked
        and mol.atoms[neighbor].symbol == "N"
        and (bond := mol.get_bond(nitrogen, neighbor)) is not None
        and bond.order == 1
    ]
    return candidates[0] if len(candidates) == 1 else None


def _hydrazone_key(mol: Molecule, carbon: int, cyclic_atoms: set[int], *, amidino: bool = False) -> str:
    ring_neighbors = [n for n in mol.get_neighbors(carbon) if n in cyclic_atoms]
    carbon_neighbors = [n for n in mol.get_neighbors(carbon) if mol.atoms[n].is_carbon]
    non_h_neighbors = [n for n in mol.get_neighbors(carbon) if mol.atoms[n].symbol != "H"]
    if carbon not in cyclic_atoms and len(ring_neighbors) == 1 and len(carbon_neighbors) == 1:
        bond = mol.get_bond(carbon, ring_neighbors[0])
        if bond is not None and bond.order == 1 and len(non_h_neighbors) == 2:
            if amidino:
                return "ring_aldehyde_amidinohydrazone"
            return "ring_aldehyde_hydrazone"
    if len(carbon_neighbors) <= 1 and carbon not in cyclic_atoms:
        if amidino:
            return "aldehyde_amidinohydrazone"
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
        roles.append(
            _make_role(
                mol,
                key="hydrazine",
                is_principal_candidate=False,
                attachment_atom=c_neighbors[0],
                atom_ids=frozenset({n1.idx, n2}),
                variant="prefix",
                reason=f"Matched C-N-N hydrazine fragment at atom {c_neighbors[0]}.",
                ordered_atoms=(c_neighbors[0], n1.idx, n2),
            )
        )
    return roles


def terminal_n3_substituent_role(
    mol: Molecule,
    start_idx: int,
    exclude_atoms: set[int],
    upstream_atom: int | None,
) -> NitrogenChainRole | None:
    """Return an ordered role for a terminal N3 substituent fragment."""

    if upstream_atom is None or mol.atoms[start_idx].symbol != "N":
        return None
    first_bond = mol.get_bond(start_idx, upstream_atom)
    if first_bond is None:
        return None
    n2_candidates = [
        n
        for n in mol.get_neighbors(start_idx)
        if n != upstream_atom and n not in exclude_atoms and mol.atoms[n].symbol == "N"
    ]
    if len(n2_candidates) != 1:
        return None
    n2 = n2_candidates[0]
    n3_candidates = [
        n
        for n in mol.get_neighbors(n2)
        if n != start_idx and n not in exclude_atoms and mol.atoms[n].symbol == "N"
    ]
    if len(n3_candidates) != 1:
        return None
    n3 = n3_candidates[0]
    if _other_non_h_neighbors(mol, n3, {n2}):
        return None
    n1_n2 = mol.get_bond(start_idx, n2)
    n2_n3 = mol.get_bond(n2, n3)
    if n1_n2 is None or n2_n3 is None:
        return None
    charges = (mol.atoms[start_idx].charge, mol.atoms[n2].charge, mol.atoms[n3].charge)
    orders = (first_bond.order, n1_n2.order, n2_n3.order)
    template = _match_template(
        tuple(template for template in TERMINAL_N3_TEMPLATES if template.key == "azido"),
        orders,
        (mol.atoms[upstream_atom].charge, *charges),
    )
    if template is None:
        return None
    return _make_role(
        mol,
        key=template.key,
        is_principal_candidate=False,
        attachment_atom=upstream_atom,
        atom_ids=frozenset({start_idx, n2, n3}),
        variant=template.variant,
        reason=f"Matched terminal charge-separated N3 substituent at atom {upstream_atom}.",
        ordered_atoms=(upstream_atom, start_idx, n2, n3),
    )


def azine_roles(mol: Molecule, component_atoms: set[int]) -> list[AzineRole]:
    """Return ordered C=N-N=C azine roles with side-component metadata."""

    roles: list[AzineRole] = []
    for n1 in component_atoms:
        if mol.atoms[n1].symbol != "N":
            continue
        for n2 in mol.get_neighbors(n1):
            if n2 <= n1 or n2 not in component_atoms or mol.atoms[n2].symbol != "N":
                continue
            n_n_bond = mol.get_bond(n1, n2)
            if n_n_bond is None or n_n_bond.order != 1:
                continue
            c1 = _double_bonded_carbon(mol, n1, {n2})
            c2 = _double_bonded_carbon(mol, n2, {n1})
            if c1 is None or c2 is None:
                continue
            side1 = _component_atoms_until_blocked(mol, component_atoms, c1, {n1, n2})
            side2 = _component_atoms_until_blocked(mol, component_atoms, c2, {n1, n2})
            if not side1 or not side2 or side1 & side2:
                continue
            if side1 | side2 | {n1, n2} != component_atoms:
                continue
            ordered_atoms = (c1, n1, n2, c2)
            roles.append(
                AzineRole(
                    atom_ids=frozenset(component_atoms),
                    ordered_atoms=ordered_atoms,
                    segments=_chain_segments(mol, ordered_atoms),
                    left=AzineSideMetadata(c1, n1, frozenset(side1)),
                    right=AzineSideMetadata(c2, n2, frozenset(side2)),
                    reason=f"Matched C=N-N=C azine role through atoms {c1}-{n1}-{n2}-{c2}.",
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
        if mol.atoms[n].symbol != "H" and (mol.atoms[n].symbol != "N" or n in cyclic_atoms)
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


def _double_bonded_carbon(mol: Molecule, nitrogen: int, blocked: set[int]) -> int | None:
    candidates = [
        n
        for n in mol.get_neighbors(nitrogen)
        if n not in blocked
        and mol.atoms[n].is_carbon
        and (bond := mol.get_bond(nitrogen, n)) is not None
        and bond.order == 2
    ]
    return candidates[0] if len(candidates) == 1 else None


def _component_atoms_until_blocked(
    mol: Molecule,
    component_atoms: set[int],
    root: int,
    blocked: set[int],
) -> set[int]:
    atoms = set()
    queue = [root]
    while queue:
        atom_idx = queue.pop(0)
        if atom_idx in atoms:
            continue
        if atom_idx not in component_atoms or atom_idx in blocked:
            return set()
        atoms.add(atom_idx)
        for neighbor in mol.get_neighbors(atom_idx):
            if neighbor in blocked:
                continue
            if neighbor in component_atoms:
                queue.append(neighbor)
    return atoms


def _amidinohydrazone_tail_atoms(mol: Molecule, hydrazone_terminal_n: int, allowed: set[int]) -> set[int]:
    """Return atoms for an N-C(=N)-N amidino tail attached to a hydrazone N."""

    for carbon in mol.get_neighbors(hydrazone_terminal_n):
        if carbon in allowed or not mol.atoms[carbon].is_carbon:
            continue
        n_c_bond = mol.get_bond(hydrazone_terminal_n, carbon)
        if n_c_bond is None or n_c_bond.order != 2:
            continue
        terminal_ns = [
            neighbor
            for neighbor in mol.get_neighbors(carbon)
            if neighbor != hydrazone_terminal_n
            and mol.atoms[neighbor].symbol == "N"
            and (bond := mol.get_bond(carbon, neighbor)) is not None
            and bond.order == 1
        ]
        if len(terminal_ns) == 2:
            return {carbon, *terminal_ns}
    return set()
