"""Graph-backed adaptation of substituted fused spiro side components."""

from dataclasses import replace

from .assembly_parts import AssemblyParts, SubstituentItem, rendered_substituent_text
from .fusion.model import FusionMode
from .fusion.wrappers import plan_fusion_spiro_side
from .graph_kernel import biconnected_edge_components
from .locants import retained_locant_sort_key
from .molecule import Molecule, edges_within_atoms
from .spiro_assembly import SpiroAssembly
from .subgraph_tools import subgraph_component


def plan_substituted_fusion_spiro_side(
    mol: Molecule, side_atoms: set[int], junction: int, *, mode: FusionMode | str
) -> SpiroAssembly | None:
    """Separate singly attached branches before requesting a fusion proof.

    The proof describes the ring skeleton, not its detachable substituents.
    Keep component-local hydrogenation inside the component name and export
    branch and stereo locants to the primed spiro namespace.
    """

    from .assembler import _add_indicated_hydrogen_prefix
    from .namer import name_subgraph

    edges = edges_within_atoms(mol, set(side_atoms))
    cores = []
    for block in biconnected_edge_components(side_atoms, edges):
        atoms = {atom for edge in block for atom in edge}
        if junction in atoms and len(block) > len(atoms):
            cores.append(atoms)
    if len(cores) != 1:
        return None
    core = cores[0]
    side_mol = mol.subgraph(side_atoms)
    plan = plan_fusion_spiro_side(side_mol, core, junction, mode=mode)
    if plan is None:
        return None
    state = plan.parent.fusion_plan.derivative_state
    if state is not None and (
        state.oxo_operations
        or state.unsaturation_operations
        or state.pi_redistribution
        or state.added_hydrogen_operations
        or state.intrinsic_hydro_operations
    ):
        return None
    assembly = plan.to_spiro_assembly()
    parent_name = assembly.side_parent_name
    if state is not None:
        if any(operation.operation_kind != "additive_hydrogen" for operation in state.hydro_operations):
            return None
        parts = AssemblyParts(
            parent_length=len(core),
            parent_hydride=plan.parent.hydride,
            parent_bond_delta=state.bond_delta,
            hydro_operations=list(state.hydro_operations),
        )
        parent_name = _add_indicated_hydrogen_prefix(parts, parent_name)

    locants = dict(plan.parent.locant_maps[0])
    remaining = set(side_atoms) - core
    branches = []
    while remaining:
        branch = subgraph_component(mol, min(remaining), set(mol.atoms) - remaining)
        attachments = [(atom, neighbor) for atom in core for neighbor in mol.get_neighbors(atom) if neighbor in branch]
        if len(attachments) != 1:
            return None
        atom, neighbor = attachments[0]
        bond = mol.get_bond(atom, neighbor)
        if bond.order != 1:
            return None
        branches.append((atom, neighbor, branch))
        remaining -= branch

    prefixes = []
    substituents = []
    for atom, neighbor, branch in sorted(branches, key=lambda item: retained_locant_sort_key(locants[item[0]])):
        rendered, trace, tree = name_subgraph(
            mol,
            neighbor,
            set(mol.atoms) - branch,
            upstream_atom=atom,
            return_trace=True,
            return_tree=True,
        )
        name = rendered_substituent_text(rendered)
        if not name:
            return None
        prefixes.append(f"{locants[atom]}'-{name}")
        substituents.append(
            SubstituentItem(
                name=name,
                locants=[f"{locants[atom]}'"],
                atom_ids=set(branch),
                bond_ids={
                    bond.idx for bond in mol.bonds.values() if bond.u in branch | {atom} and bond.v in branch | {atom}
                },
                charge_atom_ids={idx for idx in branch if mol.atoms[idx].charge},
                trace_segments=trace,
                substituent_tree=tree,
            )
        )
    stereo = tuple(
        (f"{locants[atom]}'", mol.atoms[atom].stereo)
        for atom in sorted(core, key=lambda atom: retained_locant_sort_key(locants[atom]))
        if atom != junction and mol.atoms[atom].stereo
    )
    return replace(
        assembly,
        side_parent_name=parent_name,
        side_prefixes=assembly.side_prefixes + tuple(prefixes),
        side_stereo=assembly.side_stereo + stereo,
        side_substituents=assembly.side_substituents + tuple(substituents),
    )
