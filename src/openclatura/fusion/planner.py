"""Audited planner for the bounded systematic-fusion production tier."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from ..molecule import Molecule
from .audit import audit_fusion_plan
from .descriptor import FusionDescriptorError, build_fusion_name_ast, render_fusion_name
from .faces import FaceSearchBudgetExceeded, select_bounded_face_model
from .layout import LayoutSearchBudgetExceeded, preferred_intrinsic_layouts
from .model import (
    AuditStatus,
    Face,
    FaceModel,
    FusionAuditFailed,
    FusionConfirmed,
    FusionGraph,
    FusionGraphAtom,
    FusionGraphBond,
    FusionMode,
    FusionNotApplicable,
    FusionNumberingProof,
    FusionParentPlan,
    FusionPlanningResult,
    FusionRuleDecision,
    FusionUnsupported,
    PinStatus,
)
from .numbering import completed_system_numbering_selection, parent_bond_model
from .registry import fusion_component_registry
from .rules import explain_component_comparison, fusion_mode_allows_planning, pin_ring_size_gate

PLANNER_TIER = "ortho-tree-v1"


def plan_fusion_parent(
    mol: Molecule,
    parent_atom_ids: Iterable[int],
    *,
    mode: FusionMode | str,
) -> FusionPlanningResult:
    """Return a proven fusion parent or a typed reason for safe fallback."""

    policy = FusionMode(mode)
    atoms = frozenset(parent_atom_ids)
    cache_key = ("systematic_fusion", PLANNER_TIER, policy.value, tuple(sorted(atoms)))
    cached = mol._fusion_plan_cache.get(cache_key)
    if cached is not None:
        return cached
    result = _plan_uncached(mol, atoms, policy)
    mol._fusion_plan_cache[cache_key] = result
    return result


def _plan_uncached(mol: Molecule, atoms: frozenset[int], mode: FusionMode) -> FusionPlanningResult:
    if not fusion_mode_allows_planning(mode):
        return FusionNotApplicable(f"fusion mode {mode.value!r} does not enable systematic planning")
    if len(atoms) < 6:
        return FusionNotApplicable("selected parent is too small to contain an ortho-fused system")
    if atoms - mol.atoms.keys():
        return FusionUnsupported("selected parent contains unknown graph atoms")
    if any(mol.atoms[atom].charge for atom in atoms):
        return FusionUnsupported("charged fused parents are outside the bounded production tier")
    if not _standard_valence_parent(mol, atoms):
        return FusionUnsupported("nonstandard-valence fused parents are outside the bounded production tier")

    try:
        bounded = select_bounded_face_model(mol, atoms)
    except FaceSearchBudgetExceeded as exc:
        return FusionUnsupported("bounded-face search budget exhausted", (str(exc),))
    if bounded is None or bounded.cycle_rank < 2:
        return FusionNotApplicable("selected parent has no audited multi-face fused model")
    if set(bounded.outer_boundary.atoms) != set(atoms):
        return FusionUnsupported("interior-atom fused systems require the later numbering tier")
    ring_sizes = tuple(len(face.atoms) for face in bounded.faces)
    if mode is FusionMode.AUDITED_PIN and not pin_ring_size_gate(ring_sizes):
        return FusionUnsupported("PIN ring-size gate requires at least two rings of size five or larger")

    registry = fusion_component_registry()
    matches = registry.match_faces(mol, bounded)
    try:
        ast = build_fusion_name_ast(mol, matches, registry)
    except FusionDescriptorError as exc:
        return FusionUnsupported("no supported audited fusion-component decomposition", (str(exc),))

    face_model = _typed_face_model(mol, bounded)
    try:
        layouts = preferred_intrinsic_layouts(face_model)
    except LayoutSearchBudgetExceeded as exc:
        return FusionUnsupported("intrinsic fused-layout search budget exhausted", (str(exc),))
    if not layouts:
        return FusionUnsupported("no consistent audited intrinsic fused-ring layout")
    numbering_selection = completed_system_numbering_selection(
        mol,
        bounded,
        face_model=face_model,
        layouts=layouts,
    )
    numberings = numbering_selection.accepted
    if not numberings:
        return FusionUnsupported("no layout-derived peripheral system numbering was proven")
    selected = numberings[0]
    if selected.layout_index is None:
        return FusionUnsupported("completed-system numbering lacks intrinsic-layout provenance")
    layout = layouts[selected.layout_index]
    numbering = FusionNumberingProof(
        selected_face_model=face_model,
        selected_layout=layout,
        orientation_score=(layout.orientation_score, numberings[0].score),
        abstract_atom_to_locant=numberings[0].atom_to_locant,
        input_locant_maps=tuple(numbering.atom_to_locant for numbering in numberings),
        rejected_numberings=numbering_selection.rejected,
    )
    graph = _abstract_graph(ast, registry)
    bond_model = parent_bond_model(mol, atoms)
    indicated_h = tuple(
        locant
        for atom, locant in numbering.abstract_atom_to_locant
        if mol.atoms[atom].symbol != "C" and mol.atoms[atom].total_h_count > 0
    )
    if len(indicated_h) > 1:
        return FusionUnsupported("multiple indicated-hydrogen fusion parents require a later additive tier")
    rendered = render_fusion_name(ast, registry)
    audit = audit_fusion_plan(
        mol,
        atoms,
        ast=ast,
        abstract_parent_graph=graph,
        numbering=numbering,
        bond_model=bond_model,
        mode=mode,
        registry=registry,
    )
    if audit.status is AuditStatus.ABSTAIN:
        return FusionUnsupported("fusion nomenclature audit abstained", audit.errors)
    if not audit.confirmed:
        return FusionAuditFailed("fusion reconstruction audit rejected the candidate", audit.errors)
    root_id = ast.parent_occurrences[0]
    root_match = next(match for match in ast.component_occurrences if match.occurrence_id == root_id)
    specs_by_key = {key: value.spec for key, value in registry.by_key.items()}
    seniority_trace = tuple(
        explain_component_comparison(root_match, match, specs_by_key)
        for match in ast.component_occurrences
        if match.occurrence_id != root_id
    )
    plan = FusionParentPlan(
        ast=ast,
        rendered_base_name=rendered,
        abstract_parent_graph=graph,
        numbering=numbering,
        bond_model=bond_model,
        indicated_hydrogens=indicated_h,
        pin_status=PinStatus.CONFIRMED if mode is FusionMode.AUDITED_PIN else PinStatus.VALID_GENERAL_NAME,
        rule_trace=seniority_trace
        + (
            FusionRuleDecision(
                rule="P-25",
                criterion="bounded_ortho_fusion",
                outcome="confirmed",
                reason="A complete graph-backed component cover, descriptor, numbering, and reconstruction audit passed.",
            ),
        ),
        audit=audit,
    )
    return FusionConfirmed(plan)


def _standard_valence_parent(mol: Molecule, atoms: frozenset[int]) -> bool:
    allowed = {"B", "C", "N", "O", "P", "S", "Se", "Si", "Te"}
    return all(
        mol.atoms[atom].symbol in allowed
        and sum((mol.get_bond(atom, neighbor).order for neighbor in mol.get_neighbors(atom)), 0) <= 4
        for atom in atoms
    )


def _typed_face_model(mol: Molecule, bounded) -> FaceModel:
    owners: dict[int, list[int]] = defaultdict(list)
    faces = []
    for face_id, cycle in enumerate(bounded.faces):
        edge_cycle = tuple(mol.get_bond(left, right).idx for left, right in _cycle_pairs(cycle.atoms))
        faces.append(Face(face_id, cycle.atoms, edge_cycle, len(cycle.atoms)))
        for edge_id in edge_cycle:
            owners[edge_id].append(face_id)
    perimeter = frozenset(edge for edge, face_ids in owners.items() if len(face_ids) == 1)
    fusion = frozenset(edge for edge, face_ids in owners.items() if len(face_ids) == 2)
    adjacency = []
    for edge, face_ids in sorted(owners.items()):
        if len(face_ids) == 2:
            adjacency.append((face_ids[0], face_ids[1], edge))
    return FaceModel(
        faces=tuple(faces),
        edge_to_faces=tuple(sorted((edge, tuple(sorted(face_ids))) for edge, face_ids in owners.items())),
        perimeter_edges=perimeter,
        fusion_edges=fusion,
        outer_boundary=bounded.outer_boundary.atoms,
        face_adjacency=tuple(sorted(adjacency)),
    )


def _abstract_graph(ast, registry) -> FusionGraph:
    labels: dict[int, tuple[str, int]] = {}
    edges: dict[tuple[int, int], str] = {}
    for match in ast.component_occurrences:
        spec = registry.by_key[match.spec_key].spec
        local_map = match.input_atom_by_locant
        for atom in spec.atoms:
            labels[local_map[atom.locant]] = (atom.symbol, atom.formal_charge)
        for bond in spec.bonds:
            left, right = (local_map[locant] for locant in bond.locants)
            edge = (left, right) if left < right else (right, left)
            previous = edges.setdefault(edge, bond.bond_class)
            if previous != bond.bond_class:
                raise ValueError("component bond classes disagree on a shared fusion edge")
    return FusionGraph(
        atoms=tuple(FusionGraphAtom(atom, *labels[atom]) for atom in sorted(labels)),
        bonds=tuple(FusionGraphBond(edge, edges[edge]) for edge in sorted(edges)),
    )


def _cycle_pairs(cycle: tuple[int, ...]):
    return zip(cycle, cycle[1:] + cycle[:1])
