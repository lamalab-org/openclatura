"""Graph-backed adaptation of numbered component parents to spiro sides."""

from collections.abc import Callable
from copy import deepcopy

from .assembly_parts import AssemblyParts, rendered_substituent_text
from .fusion.context import reset_fusion_mode, set_fusion_mode
from .fusion.model import FusionMode
from .graph_kernel import biconnected_edge_components
from .molecule import Molecule, edges_within_atoms
from .spiro_assembly import SpiroAssembly


def plan_graph_spiro_side(
    mol: Molecule, side_atoms: set[int], junction: int, *, required_core: set[int] | None = None
) -> SpiroAssembly | None:
    """Use ordinary parent selection, numbering, and feature perception.

    build_parent_assembly_plan selects the complete numbered fusion variant,
    including its bond model and derivative state. No rendered name is used
    to recover the junction or branch locants.
    """

    from .assembly_spiro import spiro_assembly_from_parts
    from .component_namer import name_component
    from .namer import _assemble_parent_name, _spiro_subgraph_assembly, name_subgraph

    if junction not in side_atoms:
        return None

    captured: AssemblyParts | None = None
    render_parent: Callable[[AssemblyParts], str] | None = None

    def capture(component_mol, parts, path, get_loc, **kwargs):
        nonlocal captured, render_parent
        # Assembly mutates parts. Preserve one snapshot, scoped to the actual
        # side component rather than any recursively assembled acid fragment.
        if set(component_mol.atoms) == side_atoms:
            captured = deepcopy(parts)

            def render_parent(local):
                return rendered_substituent_text(
                    _assemble_parent_name(component_mol, local, path, get_loc, emit_metadata=False)
                )

        return _assemble_parent_name(component_mol, parts, path, get_loc, **kwargs)

    side_mol = mol.subgraph(side_atoms)
    name_component(
        side_mol,
        set(side_atoms),
        # Fused side groups are detachable prefixes in the whole spiro name.
        is_substituent=required_core is not None or _polycyclic_side_core(side_mol, side_atoms, junction) is not None,
        name_subgraph=name_subgraph,
        name_spiro_subgraph=_spiro_subgraph_assembly,
        assemble_parent_name=capture,
        return_tree=True,
        omit_redundant_locants=False,
    )
    parts = captured
    # A shortcut without structured numbering is not a junction-locant proof.
    if parts is None:
        return None
    if required_core is not None:
        if parts.parent_atom_ids != required_core:
            return None
        if parts.parent_hydride is None or not parts.parent_hydride.is_systematic_fusion:
            return None
    locants = parts.parent_atom_ids_by_locant
    if set(locants.values()) != parts.parent_atom_ids or len(locants) != len(parts.parent_atom_ids):
        return None
    junction_locant = next((locant for locant, atom in locants.items() if atom == junction), None)
    if junction_locant is None:
        return None
    return spiro_assembly_from_parts(parts, junction_locant, render_parent=render_parent)


def _polycyclic_side_core(mol: Molecule, side_atoms: set[int], junction: int) -> set[int] | None:
    edges = edges_within_atoms(mol, set(side_atoms))
    cores = []
    for block in biconnected_edge_components(side_atoms, edges):
        atoms = {atom for edge in block for atom in edge}
        if junction in atoms and len(block) > len(atoms):
            cores.append(atoms)
    return cores[0] if len(cores) == 1 else None


def plan_substituted_fusion_spiro_side(
    mol: Molecule, side_atoms: set[int], junction: int, *, mode: FusionMode | str
) -> SpiroAssembly | None:
    """Accept a complete fusion proof selected by the shared parent pipeline."""

    if FusionMode(mode) in {FusionMode.DISABLED, FusionMode.LEGACY}:
        return None
    core = _polycyclic_side_core(mol, side_atoms, junction)
    if core is None:
        return None
    token = set_fusion_mode(mode)
    try:
        return plan_graph_spiro_side(mol, side_atoms, junction, required_core=core)
    finally:
        reset_fusion_mode(token)
