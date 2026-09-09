"""Exact oxo reconstruction for retained parents with unresolved carbon H."""

from dataclasses import dataclass

from .fusion.exocyclic import is_neutral_external_pi_ligand
from .fusion.model import BondAssignment, ParentBondModel
from .locants import retained_locant_sort_key
from .molecule import Molecule
from .name_operations import HydroOperation
from .retained_graph_model import RetainedGraphTemplate


@dataclass(frozen=True)
class RetainedOxoHydrogenProof:
    indicated_h: tuple[str, ...]
    atom_ids: tuple[int, ...]
    oxo_atom_ids: frozenset[int]
    model: ParentBondModel
    assignment: BondAssignment
    added_hydrogen: HydroOperation


def prove_retained_oxo_carbon_hydrogen(
    mol: Molecule, template: RetainedGraphTemplate, locant_to_atom: dict[str, int]
) -> RetainedOxoHydrogenProof | None:
    """Locate intrinsic carbon H only when oxo substitution explains every delta.

    No hydro endpoints or extra unsaturation may be inferred by counting sites.
    One allowed, capacity-preserving retained tautomer must reconstruct every
    observed ring bond after each external oxygen consumes one internal pi bond.
    """

    from .fusion.numbering import RetainedParentBondCapacityError, retained_template_parent_bond_model

    if template.default_indicated_h or template.indicated_hydrogen_count <= 0:
        return None
    if set(locant_to_atom) != set(template.locants):
        return None
    atoms = frozenset(locant_to_atom.values())
    if len(atoms) != len(template.locants) or not atoms <= mol.atoms.keys():
        return None
    observed = {tuple(sorted((b.u, b.v))): b.order for b in mol.bonds.values() if b.u in atoms and b.v in atoms}
    edges = {tuple(sorted(locant_to_atom[locant] for locant in b.locants)) for b in template.bonds}
    if set(observed) != edges or any(order not in {1, 2} for order in observed.values()):
        return None
    oxo_sites = set()
    for locant, index in locant_to_atom.items():
        atom = mol.atoms[index]
        neighbors = mol.get_neighbors(index)
        if (
            atom.symbol != template.atom_by_locant[locant].symbol
            or atom.charge
            or atom.total_h_count + sum(mol.get_bond(index, n).order for n in neighbors)
            != atom.element.standard_valence
        ):
            return None
        for neighbor in neighbors:
            if neighbor in atoms or mol.get_bond(index, neighbor).order == 1:
                continue
            if (
                mol.atoms[neighbor].symbol != "O"
                or not is_neutral_external_pi_ligand(mol, index, neighbor)
                or index in oxo_sites
            ):
                return None
            oxo_sites.add(index)
    if not oxo_sites:
        return None
    indicated = tuple(
        sorted(
            (
                locant
                for locant, index in locant_to_atom.items()
                if mol.atoms[index].is_carbon
                and index not in oxo_sites
                and template.atom_by_locant[locant].resolved_pi_capacity
                and not template.atom_by_locant[locant].fusion
                and all(order == 1 for edge, order in observed.items() if index in edge)
            ),
            key=retained_locant_sort_key,
        )
    )
    if len(indicated) != template.indicated_hydrogen_count:
        return None
    try:
        model = retained_template_parent_bond_model(template, locant_to_atom, indicated_h=indicated)
    except RetainedParentBondCapacityError:
        return None
    for assignment in model.allowed_kekule_assignments:
        consumed = {index: 0 for index in oxo_sites}
        added_sites = set()
        consumed_bonds = set()
        for edge, expected in assignment.orders:
            actual = observed[edge]
            if actual == expected:
                continue
            suffix_sites = oxo_sites.intersection(edge)
            if expected != 2 or actual != 1 or len(suffix_sites) != 1:
                break
            consumed[next(iter(suffix_sites))] += 1
            added_sites.update(set(edge) - suffix_sites)
            consumed_bonds.add(mol.get_bond(*edge).idx)
        else:
            if all(count == 1 for count in consumed.values()):
                added_locants = tuple(
                    sorted(
                        (locant for locant, index in locant_to_atom.items() if index in added_sites),
                        key=retained_locant_sort_key,
                    )
                )
                return RetainedOxoHydrogenProof(
                    indicated,
                    tuple(locant_to_atom[locant] for locant in indicated),
                    frozenset(oxo_sites),
                    model,
                    assignment,
                    HydroOperation(
                        key="added_hydrogen",
                        reason="The oxo suffix consumes a proved parent pi bond; its other endpoint receives added H.",
                        locants=added_locants,
                        atom_ids=tuple(locant_to_atom[locant] for locant in added_locants),
                        bond_ids=tuple(sorted(consumed_bonds)),
                        operation_kind="indicated_hydrogen",
                    ),
                )
    return None
