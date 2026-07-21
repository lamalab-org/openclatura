"""Whole-graph reconstruction audit for assembled component names.

The coverage audits (``naming_audit``, ``name_assembly``) prove that every
atom was claimed by a name binding and every name token is bound — they
validate the bookkeeping, not its meaning. This module closes the loop by
rebuilding a molecule from the *nomenclature-level* facts in
``AssemblyParts`` and comparing it against the input component with RDKit:

- the parent skeleton is rebuilt purely from the claimed locant maps
  (``parent_atom_symbols_by_locant`` + ``parent_bond_orders_by_locants``);
- substituent and characteristic-group subtrees are grafted from their
  claimed atom sets, with their stated attachment locants checked against
  the actual graph;
- unsaturation items, heteroatom symbols, and ring-closure counts are
  cross-checked against the claimed parent bonds.

The audit is diagnostic: it returns a status instead of raising, and bails
to ``skipped`` for features it does not model yet (charged components,
spiro assemblies, front modifiers).
"""

from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem
from rdkit.rdBase import BlockLogs

from .assembly_parts import AssemblyParts, SubstituentItem
from .molecule import Molecule

_BOND_TYPES = {
    1: Chem.rdchem.BondType.SINGLE,
    2: Chem.rdchem.BondType.DOUBLE,
    3: Chem.rdchem.BondType.TRIPLE,
}

_UNSATURATION_ORDERS = {"double": 2, "triple": 3}


@dataclass(frozen=True)
class ReconstructionAudit:
    """Outcome of rebuilding a component from its assembly parts."""

    status: str  # 'matched' | 'mismatched' | 'skipped'
    issues: tuple[str, ...] = ()
    reference_smiles: str | None = None
    reconstructed_smiles: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == "matched"


def audit_component_reconstruction(mol: Molecule, parts: AssemblyParts) -> ReconstructionAudit:
    """Rebuild the component claimed by ``parts`` and compare it to ``mol``."""

    try:
        return _audit(mol, parts)
    except Exception as exc:  # diagnostic path must never break naming
        return ReconstructionAudit(status="skipped", issues=(f"audit error: {exc}",))


def _audit(mol: Molecule, parts: AssemblyParts) -> ReconstructionAudit:
    skip_reason = _unsupported_feature(mol, parts)
    if skip_reason:
        return ReconstructionAudit(status="skipped", issues=(skip_reason,))

    issues: list[str] = []
    issues.extend(_check_parent_skeleton(parts))
    issues.extend(_check_unsaturations(parts))
    issues.extend(_check_heteroatom_symbols(parts))

    atom_to_locant = {atom_idx: locant for locant, atom_idx in parts.parent_atom_ids_by_locant.items()}
    issues.extend(_check_attachment_locants(mol, parts, atom_to_locant))

    reference = _reference_rdkit_mol(mol, _component_atoms(parts))
    reconstructed = _reconstructed_rdkit_mol(mol, parts)
    if reference is None or reconstructed is None:
        issues.append("could not build comparison molecules")
        return ReconstructionAudit(status="skipped", issues=tuple(issues))

    with BlockLogs():
        reference_smiles = _canonical(reference)
        reconstructed_smiles = _canonical(reconstructed)
    if reference_smiles is None or reconstructed_smiles is None:
        issues.append("could not canonicalize comparison molecules")
        return ReconstructionAudit(status="skipped", issues=tuple(issues))

    if reference_smiles != reconstructed_smiles:
        issues.append(f"reconstructed graph {reconstructed_smiles!r} differs from input {reference_smiles!r}")

    return ReconstructionAudit(
        status="mismatched" if issues else "matched",
        issues=tuple(issues),
        reference_smiles=reference_smiles,
        reconstructed_smiles=reconstructed_smiles,
    )


def _unsupported_feature(mol: Molecule, parts: AssemblyParts) -> str | None:
    if parts.is_substituent:
        return "substituent scope"
    if parts.front_modifiers or parts.front_modifier_atom_ids:
        return "front modifiers not modeled"
    if parts.parent_charges or any(mol.atoms[idx].charge for idx in _component_atoms(parts) if idx in mol.atoms):
        return "charged component not modeled"
    if any(item.spiro is not None for item in parts.substituents):
        return "spiro substituent assembly not modeled"
    if parts.is_double_attach or parts.is_triple_attach:
        return "multi-attachment scope"
    if not parts.parent_atom_ids_by_locant:
        return "no parent locant map"
    return None


def _component_atoms(parts: AssemblyParts) -> set[int]:
    atoms: set[int] = set(parts.parent_atom_ids)
    for item in _attachment_items(parts):
        atoms |= set(item.atom_ids)
    for binding in parts.name_atom_bindings:
        atoms |= set(binding.atom_ids)
    return atoms


def _attachment_items(parts: AssemblyParts) -> list[SubstituentItem]:
    items = list(parts.substituents) + list(parts.principal_suffix_modifiers) + list(parts.a_prefixes)
    if parts.principal_group is not None:
        items.append(
            SubstituentItem(
                name=parts.principal_group.key,
                locants=list(parts.principal_group.locants),
                atom_ids=set(parts.principal_group.atom_ids),
                bond_ids=set(parts.principal_group.bond_ids),
            )
        )
    return items


def _check_parent_skeleton(parts: AssemblyParts) -> list[str]:
    issues: list[str] = []
    locants = set(parts.parent_atom_ids_by_locant)
    if len(parts.parent_atom_ids_by_locant) != parts.parent_length:
        issues.append(
            f"parent locant map has {len(parts.parent_atom_ids_by_locant)} entries for parent_length {parts.parent_length}"
        )
    if all(locant.isdigit() for locant in locants) and locants:
        expected = {str(i) for i in range(1, parts.parent_length + 1)}
        if locants != expected:
            issues.append(f"parent locants {sorted(locants)} are not 1..{parts.parent_length}")

    ring_count = _claimed_ring_count(parts)
    if ring_count is not None:
        expected_bonds = parts.parent_length - 1 + ring_count
        if len(parts.parent_bond_orders_by_locants) != expected_bonds:
            issues.append(
                f"parent claims {len(parts.parent_bond_orders_by_locants)} bonds; "
                f"a {'ring system' if ring_count else 'chain'} of {parts.parent_length} atoms needs {expected_bonds}"
            )
    return issues


def _claimed_ring_count(parts: AssemblyParts) -> int | None:
    if parts.is_polycycle:
        return None  # descriptor-dependent; not modeled yet
    if parts.is_bicycle or parts.is_spiro:
        return 2
    if parts.is_ring:
        return 1
    return 0


def _check_unsaturations(parts: AssemblyParts) -> list[str]:
    issues: list[str] = []
    for item in parts.unsaturations:
        order = _UNSATURATION_ORDERS.get(item.bond_key)
        if order is None:
            continue
        for locant in item.locants:
            pair = _unsaturation_locant_pair(parts, locant)
            if pair is None:
                issues.append(f"{item.bond_key} bond locant {locant!r} has no matching parent bond")
                continue
            claimed = parts.parent_bond_orders_by_locants.get(pair)
            if claimed != order:
                issues.append(f"{item.bond_key} bond claimed at {'-'.join(pair)} but parent bond order is {claimed}")
    return issues


def _unsaturation_locant_pair(parts: AssemblyParts, locant: str) -> tuple[str, str] | None:
    if "(" in locant:  # compound locant, e.g. '1(6)'
        first, second = locant.rstrip(")").split("(", 1)
        candidates = [tuple(sorted((first, second)))]
    else:
        candidates = []
        if locant.isdigit():
            candidates.append(tuple(sorted((locant, str(int(locant) + 1)))))
        candidates.extend(pair for pair in parts.parent_bond_orders_by_locants if pair[0] == locant or pair[1] == locant)
    for pair in candidates:
        if pair in parts.parent_bond_orders_by_locants:
            return pair
    return None


def _check_heteroatom_symbols(parts: AssemblyParts) -> list[str]:
    if parts.retained_name:
        return []  # heteroatoms implied by the retained parent
    replaced: dict[str, str] = {}
    for item in parts.a_prefixes:
        for locant in item.locants:
            # lambda-convention locants ('1lambda^4') address plain locant '1'
            replaced[str(locant).split("lambda", 1)[0]] = item.name
    issues: list[str] = []
    for locant, symbol in parts.parent_atom_symbols_by_locant.items():
        if symbol != "C" and locant not in replaced:
            issues.append(f"parent atom at locant {locant} is {symbol} but no replacement prefix claims it")
    return issues


def _check_attachment_locants(mol: Molecule, parts: AssemblyParts, atom_to_locant: dict[int, str]) -> list[str]:
    issues: list[str] = []
    for item in parts.substituents:
        if not item.locants or not all(str(locant).isdigit() for locant in item.locants):
            continue
        actual: list[str] = []
        for atom_idx in item.atom_ids:
            for neighbor in mol.get_neighbors(atom_idx):
                if neighbor in parts.parent_atom_ids and neighbor not in item.atom_ids:
                    actual.append(atom_to_locant.get(neighbor, "?"))
        if actual and sorted(actual) != sorted(str(locant) for locant in item.locants):
            issues.append(f"substituent {item.name!r} claims locants {sorted(item.locants)} but attaches at {sorted(actual)}")
    return issues


def _reference_rdkit_mol(mol: Molecule, component_atoms: set[int]) -> Chem.Mol | None:
    editable = Chem.RWMol()
    mapping: dict[int, int] = {}
    for atom_idx in sorted(component_atoms):
        atom = mol.atoms.get(atom_idx)
        if atom is None:
            return None
        mapping[atom_idx] = editable.AddAtom(Chem.Atom(atom.symbol))
    for bond in mol.bonds.values():
        if bond.u in mapping and bond.v in mapping:
            bond_type = _BOND_TYPES.get(bond.order)
            if bond_type is None:
                return None
            editable.AddBond(mapping[bond.u], mapping[bond.v], bond_type)
    return editable.GetMol()


def _reconstructed_rdkit_mol(mol: Molecule, parts: AssemblyParts) -> Chem.Mol | None:
    editable = Chem.RWMol()
    mapping: dict[int, int] = {}

    # Parent skeleton purely from the claimed locant maps.
    for locant, atom_idx in parts.parent_atom_ids_by_locant.items():
        symbol = parts.parent_atom_symbols_by_locant.get(locant)
        if symbol is None:
            return None
        mapping[atom_idx] = editable.AddAtom(Chem.Atom(symbol))
    for (loc_u, loc_v), order in parts.parent_bond_orders_by_locants.items():
        u = parts.parent_atom_ids_by_locant.get(loc_u)
        v = parts.parent_atom_ids_by_locant.get(loc_v)
        bond_type = _BOND_TYPES.get(order)
        if u is None or v is None or bond_type is None:
            return None
        editable.AddBond(mapping[u], mapping[v], bond_type)

    # Graft each claimed subtree; bonds inside a subtree and its attachment
    # bonds come from the graph, so a wrong claimed atom set or a dropped
    # attachment surfaces as a canonical SMILES difference.
    added_bonds: set[tuple[int, int]] = set()
    for item in _attachment_items(parts):
        extra = set(item.atom_ids) - parts.parent_atom_ids
        for atom_idx in sorted(extra):
            if atom_idx in mapping:
                continue
            atom = mol.atoms.get(atom_idx)
            if atom is None:
                return None
            mapping[atom_idx] = editable.AddAtom(Chem.Atom(atom.symbol))
        for atom_idx in sorted(extra):
            for neighbor in mol.get_neighbors(atom_idx):
                if neighbor not in mapping:
                    continue
                bond = mol.get_bond(atom_idx, neighbor)
                if bond is None:
                    continue
                key = (min(atom_idx, neighbor), max(atom_idx, neighbor))
                if key in added_bonds:
                    continue
                bond_type = _BOND_TYPES.get(bond.order)
                if bond_type is None:
                    return None
                editable.AddBond(mapping[atom_idx], mapping[neighbor], bond_type)
                added_bonds.add(key)
    return editable.GetMol()


def _canonical(rd_mol: Chem.Mol) -> str | None:
    try:
        Chem.SanitizeMol(rd_mol)
        return Chem.MolToSmiles(rd_mol)
    except Exception:
        return None


__all__ = ["ReconstructionAudit", "audit_component_reconstruction"]
