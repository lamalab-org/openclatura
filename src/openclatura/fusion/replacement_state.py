"""Joint pi and hydrogen proof after replacement of a carbon fusion parent."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from ..assembly_parts import NameTokenBinding
from ..locants import system_locant_sort_key
from .mancude import ParentDerivativeState, indicated_hydrogen_parent_bond_model, parent_derivative_state
from .model import FusionGraph, ParentBondModel
from .numbering import parent_bond_model

if TYPE_CHECKING:
    from ..molecule import Molecule
    from .model import FusionParentPlan


@dataclass(frozen=True)
class ReplacementHydrogenBalance:
    atom_id: int
    parent_hydrogens: int
    additive_hydrogens: int
    external_bond_order: int
    observed_hydrogens: int

    @property
    def exact(self) -> bool:
        return (
            min(self.parent_hydrogens, self.additive_hydrogens, self.external_bond_order, self.observed_hydrogens) >= 0
            and self.parent_hydrogens + self.additive_hydrogens - self.external_bond_order == self.observed_hydrogens
        )


@dataclass(frozen=True)
class ReplacementFusionState:
    """A restored-element parent and its independently checked H accounting."""

    graph: FusionGraph
    bond_model: ParentBondModel
    derivative_state: ParentDerivativeState
    indicated_hydrogens: tuple[str, ...]
    hydrogen_balances: tuple[ReplacementHydrogenBalance, ...]
    rendered_parts: tuple[NameTokenBinding, ...]
    audit_checks: tuple[str, ...]

    @property
    def rendered_name(self) -> str:
        return "".join(part.text for part in self.rendered_parts)

    @property
    def audit_ok(self) -> bool:
        return (
            bool(self.audit_checks)
            and self.derivative_state.bond_delta.compatible
            and {balance.atom_id for balance in self.hydrogen_balances} == {atom.id for atom in self.graph.atoms}
            and all(balance.exact for balance in self.hydrogen_balances)
        )


def saturated_replacement_scope(mol: Molecule, atoms: frozenset[int]) -> bool:
    """The supported composition domain has standard, neutral sigma valence."""
    for atom_id in atoms:
        atom = mol.atoms[atom_id]
        neighbors = mol.get_neighbors(atom_id)
        if atom.charge or atom.is_aromatic or any(mol.get_bond(atom_id, other).order != 1 for other in neighbors):
            return False
        if atom.total_h_count != atom.element.standard_valence - len(neighbors):
            return False
        if atom.is_carbon:
            continue
        ring_degree = len(atoms.intersection(neighbors))
        if not (
            (atom.element.mancude_forced_single and ring_degree == 2)
            or (atom.symbol == "N" and ring_degree == 3 and len(neighbors) == 3)
        ):
            return False
    return True


def prove_replacement_state(mol: Molecule, carbon_plan: FusionParentPlan) -> ReplacementFusionState | None:
    """Reassign carbon pi/H only after restoring the skeletal heteroatoms.

    The carbon AST proves topology and numbering, not the replaced parent's
    intrinsic H or pi capacity. Preserve explicit component bond constraints,
    then solve all restored-element and carbon-H constraints together.
    """
    locants = carbon_plan.numbering.string_input_locant_maps()[0]
    atoms = frozenset(locants)
    if not carbon_plan.audit.confirmed or not saturated_replacement_scope(mol, atoms):
        return None
    graph = replace(
        carbon_plan.abstract_parent_graph,
        atoms=tuple(
            replace(
                atom,
                symbol=mol.atoms[atom.id].symbol,
                pi_capacity=atom.pi_capacity if mol.atoms[atom.id].is_carbon else 0,
                forced_single=atom.forced_single or not mol.atoms[atom.id].is_carbon,
            )
            for atom in carbon_plan.abstract_parent_graph.atoms
        ),
    )
    initial = parent_bond_model(graph)
    candidates = []
    for assignment in initial.allowed_kekule_assignments:
        paired = {atom for edge, order in assignment.orders if order == 2 for atom in edge}
        intrinsic = {atom for atom in atoms - paired if mol.atoms[atom].is_carbon}
        if any(len(atoms.intersection(mol.get_neighbors(atom))) != 2 for atom in intrinsic):
            continue
        model = indicated_hydrogen_parent_bond_model(graph, intrinsic)
        if model.maximum_non_cumulative_double_bonds != initial.maximum_non_cumulative_double_bonds:
            continue
        state = parent_derivative_state(
            mol, atoms, model, locants, indicated_hydrogen_atom_ids=intrinsic, preserve_retained_parent_state=True
        )
        if state is None or state.unsaturation_operations or state.external_pi_operations:
            continue
        balances = _hydrogen_balances(mol, atoms, state)
        if balances is None:
            continue
        indicated = tuple(sorted((locants[atom] for atom in intrinsic), key=system_locant_sort_key))
        rank = (
            tuple(map(system_locant_sort_key, indicated)),
            tuple(system_locant_sort_key(locant) for op in state.hydro_operations for locant in op.locants),
        )
        candidates.append((rank, model, state, indicated, balances))
    if not candidates:
        return None
    _, model, state, indicated, balances = min(candidates, key=lambda candidate: candidate[0])
    # Typed component tokens retain their original graph ownership. Only the
    # intrinsic-H citation is replaced by the completed replacement proof.
    core = tuple(
        part
        for part in carbon_plan.rendered_parts
        if part.grammar_role not in {"fusion_indicated_hydrogen", "fusion_indicated_hydrogen_separator"}
    )
    atom_by_locant = {locant: atom for atom, locant in locants.items()}
    hydrogen = tuple(
        part
        for index, locant in enumerate(indicated)
        for part in (
            NameTokenBinding(
                text=f"{locant}H",
                token_kind="hydro",
                source="replacement_fusion_renderer",
                grammar_role="replacement_indicated_hydrogen",
                binding_key=f"replacement:H:{locant}",
                atom_ids={atom_by_locant[locant]},
                locants=(locant,),
                match_priority=100,
            ),
            NameTokenBinding(
                text="," if index + 1 < len(indicated) else "-",
                token_kind="grammar",
                source="replacement_fusion_renderer",
                grammar_role="replacement_indicated_hydrogen_separator",
            ),
        )
    )
    return ReplacementFusionState(
        graph,
        model,
        state,
        indicated,
        balances,
        hydrogen + core,
        (
            "postreplacement_pi_capacity",
            "joint_carbon_pi_hydrogen_state",
            "exact_restored_bonds",
            "exact_atom_hydrogen_balance",
        ),
    )


def _hydrogen_balances(
    mol: Molecule, atoms: frozenset[int], state: ParentDerivativeState
) -> tuple[ReplacementHydrogenBalance, ...] | None:
    delta = state.bond_delta
    assignment = dict(delta.assignment.orders)
    hydro = set(delta.hydrogenated_edges)
    if hydro != {edge for edge, order in assignment.items() if order == 2}:
        return None
    if len(delta.hydrogenated_atom_ids) != 2 * len(hydro):
        return None
    if {atom for op in state.hydro_operations for atom in op.atom_ids} != set(delta.hydrogenated_atom_ids):
        return None
    if any(
        mol.get_bond(*edge) is None or mol.get_bond(*edge).order != (1 if edge in hydro else order)
        for edge, order in assignment.items()
    ):
        return None
    balances = []
    for atom in sorted(atoms):
        parent_valence = sum(order for edge, order in assignment.items() if atom in edge)
        external = sum(mol.get_bond(atom, other).order for other in mol.get_neighbors(atom) if other not in atoms)
        balance = ReplacementHydrogenBalance(
            atom,
            mol.atoms[atom].element.standard_valence - parent_valence,
            int(atom in delta.hydrogenated_atom_ids),
            external,
            mol.atoms[atom].total_h_count,
        )
        if not balance.exact:
            return None
        balances.append(balance)
    return tuple(balances)
