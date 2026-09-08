"""Graph-backed adaptation of numbered component parents to spiro sides."""

from collections.abc import Callable
from copy import deepcopy
from dataclasses import replace

from .assembly_parts import AssemblyParts, SubstituentItem, rendered_substituent_text
from .fusion.context import reset_fusion_mode, set_fusion_mode
from .fusion.model import FusionMode
from .graph_kernel import biconnected_edge_components
from .molecule import Molecule, edges_within_atoms
from .spiro_assembly import SpiroAssembly


class _UnavailableSpiroParent(Exception):
    """The selected component cannot prove the requested spiro junction."""


def _plan_spiro_parent(mol, selection, intent, substituents, *, junction, required_core, **parent_options):
    from .component_namer import build_component_parent_plan
    from .fusion.context import current_fusion_mode
    from .fusion.wrappers import plan_fusion_spiro_side
    from .parent_pipeline import build_parent_assembly_plan

    core = selection.atom_set
    if junction not in core:
        raise _UnavailableSpiroParent
    if required_core is not None:
        if core != required_core:
            raise _UnavailableSpiroParent
        proof = plan_fusion_spiro_side(mol, core, junction, mode=current_fusion_mode())
        if proof is None:
            raise _UnavailableSpiroParent
        return build_parent_assembly_plan(mol, selection, intent, substituents, parent_hydride=proof.parent.hydride)

    # The junction outranks detachable groups. Include the latter in the
    # tie-break without adding fictitious substituents to the assembled parts.
    numbering_substituents = {atom: list(items) for atom, items in substituents.items()}
    for atom in intent.principal_atoms:
        numbering_substituents.setdefault(atom, []).append(SubstituentItem(name="", locants=[]))
    intent = replace(intent, principal_atoms=(junction,))
    # Saturated multiheteroatom sides cite primed skeletal replacement prefixes
    # on the carbon ring, instead of an independently named heterocycle.
    if (
        selection.is_ring
        and not (selection.is_bicycle or selection.is_polycycle or selection.is_spiro)
        and sum(not mol.atoms[atom].is_carbon for atom in core) > 1
        and all(bond.order == 1 for bond in mol.bonds.values() if bond.u in core and bond.v in core)
    ):
        parent_options.update(retained_name=None, locant_maps=None, retained_parent_metadata=None)
    return build_component_parent_plan(mol, selection, intent, numbering_substituents, **parent_options)


def plan_graph_spiro_side(
    mol: Molecule, side_atoms: set[int], junction: int, *, required_core: set[int] | None = None
) -> SpiroAssembly | None:
    """Project one shared component pipeline with junction-aware numbering."""
    from .assembly_spiro import spiro_assembly_from_parts
    from .component_namer import name_component
    from .namer import _assemble_parent_name, _select_subgraph_parent, _spiro_subgraph_assembly, name_subgraph

    if junction not in side_atoms:
        return None

    captured: AssemblyParts | None = None
    render_parent: Callable[[AssemblyParts], str] | None = None

    def select_parent(component_mol, exclude_atoms, principal_atoms):
        return _select_subgraph_parent(component_mol, junction, set(component_mol.atoms), exclude_atoms)

    def plan_parent(component_mol, selection, intent, substituents, **options):
        return _plan_spiro_parent(
            component_mol,
            selection,
            intent,
            substituents,
            junction=junction,
            required_core=required_core,
            **options,
        )

    def capture(component_mol, parts, path, get_loc, **kwargs):
        nonlocal captured, render_parent
        # Assembly mutates parts. Keep the component-local proof before any
        # rendering, not the parts of a recursively assembled acid fragment.
        if set(component_mol.atoms) == side_atoms:
            captured = deepcopy(parts)

            def render_parent(local):
                return rendered_substituent_text(
                    _assemble_parent_name(component_mol, local, path, get_loc, emit_metadata=False)
                )

        return _assemble_parent_name(component_mol, parts, path, get_loc, **kwargs)

    side_mol = mol.subgraph(side_atoms)
    try:
        name_component(
            side_mol,
            set(side_atoms),
            is_substituent=required_core is not None
            or _polycyclic_side_core(side_mol, side_atoms, junction) is not None,
            name_subgraph=name_subgraph,
            name_spiro_subgraph=_spiro_subgraph_assembly,
            assemble_parent_name=capture,
            parent_plan_builder=plan_parent,
            parent_selector=select_parent,
            return_tree=True,
            omit_redundant_locants=False,
        )
    except _UnavailableSpiroParent:
        return None
    parts = captured
    # A shortcut without structured numbering is not a junction-locant proof.
    if parts is None:
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
