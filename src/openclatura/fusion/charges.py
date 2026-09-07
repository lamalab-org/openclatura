"""Graph-bound formal-charge operations on completed fusion parents."""

from copy import copy
from dataclasses import replace

from ..locants import system_locant_sort_key
from ..molecule import Molecule
from .model import FusionChargeOperation, FusionChargeOperationKind, FusionGraph, FusionNumberingProof


def fusion_component_charge_parent(mol: Molecule, atoms: frozenset[int]) -> Molecule:
    """Restore removed protons for component recognition, preserving graph IDs.

    Only the matching view is neutralized. The original molecule must remain
    the input to numbering, charge-operation construction and reconstruction.
    """

    restored = {}
    for atom_id in atoms:
        atom = mol.atoms[atom_id]
        if atom.charge != -1 or atom.symbol not in {"C", "N"}:
            continue
        _require_deprotonation_valence(mol, atom_id)
        restored[atom_id] = replace(
            atom, charge=0, total_h_count=atom.total_h_count + 1, explicit_h_count=atom.explicit_h_count + 1
        )
    if not restored:
        return mol
    parent = copy(mol)
    parent.atoms = {**mol.atoms, **restored}
    parent._retained_fused_cache = {}
    parent._fusion_plan_cache = {}
    parent._invalidate_graph_caches()
    return parent


def _require_deprotonation_valence(mol: Molecule, atom_id: int) -> None:
    atom = mol.atoms[atom_id]
    valence = sum(mol.get_bond(atom_id, neighbor).order for neighbor in mol.get_neighbors(atom_id))
    valence += atom.total_h_count
    if valence != {"C": 3, "N": 2}.get(atom.symbol):
        raise ValueError("fusion deprotonation requires a proton-removal C/N valence")


def fusion_charge_lone_pair_sites(mol: Molecule, graph: FusionGraph) -> frozenset[int]:
    """Preserve neutral aromatic N donors when a charged derivative has oxo.

    Deprotonation and external oxo do not turn a neutral aromatic N lone pair
    into a parent pi-bond site. NH and N-substitution have the same valence
    constraint; charged nitrogen H remains separate.
    """

    if not any(mol.atoms[atom.id].charge == -1 for atom in graph.atoms):
        return frozenset()
    return frozenset(
        atom.id
        for atom in graph.atoms
        if mol.atoms[atom.id].symbol == "N"
        and mol.atoms[atom.id].charge == 0
        and mol.atoms[atom.id].is_aromatic
        and len(mol.get_neighbors(atom.id)) + mol.atoms[atom.id].total_h_count == 3
        and all(mol.get_bond(atom.id, neighbor).order == 1 for neighbor in mol.get_neighbors(atom.id))
    )


def fusion_charge_operations(
    mol: Molecule,
    graph: FusionGraph,
    numbering: FusionNumberingProof,
) -> tuple[FusionChargeOperation, ...]:
    """Describe supported charge deltas without changing component identities.

    An anionic C/N site must have one less bond-order unit (including H)
    than its neutral valence. This excludes electron-addition and unusual
    valence states from the proton-removal operation.
    """

    locants = dict(numbering.input_locant_maps[0])
    operations = []
    for base in graph.atoms:
        observed = mol.atoms[base.id]
        if observed.symbol != base.symbol:
            raise ValueError("fusion charge operation cannot change the parent element")
        if observed.charge == base.formal_charge:
            continue
        if base.id not in locants:
            raise ValueError(f"charged atom {base.id} has no completed-system locant")
        kind = FusionChargeOperationKind.HETEROATOM_CATIONIZATION
        if observed.charge < 0:
            kind = FusionChargeOperationKind.DEPROTONATION
            _require_deprotonation_valence(mol, base.id)
        operations.append(
            FusionChargeOperation(
                atom_id=base.id,
                locant=locants[base.id],
                symbol=observed.symbol,
                base_charge=base.formal_charge,
                observed_charge=observed.charge,
                operation_kind=kind,
            )
        )
    return tuple(sorted(operations, key=lambda operation: system_locant_sort_key(operation.locant)))
