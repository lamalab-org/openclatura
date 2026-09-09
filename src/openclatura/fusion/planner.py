"""Audited planner for the bounded systematic-fusion production tier."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace

from ..assembly_parts import NameTokenBinding
from ..locants import SystemLocant, system_locant_sort_key
from ..molecule import Molecule
from ..polycycle_topology import ring_system_topology
from .audit import audit_fusion_plan
from .charges import fusion_charge_operations as _fusion_charge_operations
from .charges import fusion_component_charge_parent
from .config import fusion_nomenclature_config
from .descriptor import FusionDescriptorError, iter_fusion_name_asts, render_fusion_name_parts
from .faces import BoundedFaceModel, FaceSearchBudgetExceeded, cached_bounded_face_model
from .faces import typed_face_model as _typed_face_model
from .indicated_hydrogen import (
    component_carbon_h_relocation_scope,
    component_h_locants,
    component_parent_atoms,
    component_parent_graph,
    intrinsic_carbon_candidate_atoms,
    intrinsic_carbon_parent_model,
    intrinsic_parent_lone_pair_sites,
)
from .layout import LayoutSearchBudgetExceeded, preferred_intrinsic_layouts
from .mancude import (
    has_complete_saturated_hydrogenation,
    indicated_hydrogen_parent_bond_model,
    parent_derivative_state,
    saturated_nitrogen_hydrogen_sites,
)
from .model import (
    AuditStatus,
    FaceModel,
    FusedLayout,
    FusionAuditFailed,
    FusionComponentSpec,
    FusionConfirmed,
    FusionGraph,
    FusionMode,
    FusionNameAst,
    FusionNotApplicable,
    FusionNumberingProof,
    FusionParentPlan,
    FusionPlanningResult,
    FusionRuleDecision,
    FusionUnsupported,
    ParentBondModel,
)
from .numbering import (
    CompletedNumberingSelection,
    MancudeSearchBudgetExceeded,
    completed_system_numbering_selection,
    indicated_hydrogen_candidate_atoms,
    observed_parent_matches_bond_model,
    parent_bond_model,
)
from .registry import FusionComponentRegistry, fusion_component_registry
from .rules import explain_component_comparison, fusion_mode_allows_planning, fusion_ring_size_gate
from .valence import fusion_lambda_descriptors

PLANNER_TIER = fusion_nomenclature_config().rules.planner_tier
SUPPORT = fusion_nomenclature_config().rules.support

# Preserve the planner-level test/extension seam while sharing face proofs
# between ordinary and P-25.5 planning.
select_bounded_face_model = cached_bounded_face_model


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
    if not SUPPORT.charged_parents and any(mol.atoms[atom].charge for atom in atoms):
        return FusionUnsupported("charged fused parents are outside the configured production tier")
    if any(not mol.atoms[atom].element.fusion_supported for atom in atoms):
        return FusionUnsupported("fused parent contains an unsupported skeletal element")
    if not SUPPORT.nonstandard_valence and not _standard_valence_parent(mol, atoms):
        return FusionUnsupported("nonstandard-valence fused parents are outside the bounded production tier")

    try:
        bounded = select_bounded_face_model(mol, atoms)
    except FaceSearchBudgetExceeded as exc:
        return FusionUnsupported("bounded-face search budget exhausted", (str(exc),))
    if bounded is None:
        return _classify_unmodelled_ring_system(mol, atoms)
    if bounded.cycle_rank < 2:
        return FusionNotApplicable("selected parent has no audited multi-face fused model")
    if bounded.interior_atoms:
        topology = ring_system_topology(mol, atoms)
        if topology.classification == "bicyclic":
            return FusionUnsupported("bridged fused systems are outside the ordinary ortho-fusion tier")
    if not SUPPORT.interior_atoms and bounded.interior_atoms:
        return FusionUnsupported("interior-atom fused systems require the later numbering tier")
    ring_sizes = tuple(len(face.atoms) for face in bounded.faces)
    if not fusion_ring_size_gate(ring_sizes):
        return FusionUnsupported("fusion ring-size eligibility requires at least two rings of size five or larger")

    registry = fusion_component_registry()
    try:
        matching_parent = fusion_component_charge_parent(mol, atoms)
    except ValueError as exc:
        return FusionUnsupported("fused-parent charge operation is outside the audited production tier", (str(exc),))
    matches = registry.match_faces(matching_parent, bounded)
    face_model = _typed_face_model(mol, bounded)
    try:
        layouts = preferred_intrinsic_layouts(face_model)
    except LayoutSearchBudgetExceeded as exc:
        return FusionUnsupported("intrinsic fused-layout search budget exhausted", (str(exc),))
    if not layouts:
        return FusionUnsupported("no consistent audited intrinsic fused-ring layout")
    rejected = []
    numbering_cache: dict[bool, CompletedNumberingSelection] = {}
    try:
        for ast in iter_fusion_name_asts(mol, matches, registry):
            result = _plan_numbered_candidate(
                mol, atoms, mode, ast, registry, bounded, face_model, layouts, numbering_cache=numbering_cache
            )
            if isinstance(result, FusionConfirmed):
                return result
            rejected.append(result)
    except FusionDescriptorError as exc:
        return FusionUnsupported("no supported audited fusion-component decomposition", (str(exc),))
    if rejected:
        if all(result == rejected[0] for result in rejected):
            return rejected[0]
        reasons = []
        for result in rejected:
            reasons.append(result.reason)
            if isinstance(result, FusionUnsupported):
                reasons.extend(result.details)
            elif isinstance(result, FusionAuditFailed):
                reasons.extend(result.candidate_summary)
        details = tuple(dict.fromkeys(reasons))
        if isinstance(rejected[0], FusionAuditFailed):
            return FusionAuditFailed("ranked fusion candidates failed reconstruction", details)
        return FusionUnsupported("ranked fusion candidates have no supported chemical plan", details)
    return FusionUnsupported("no supported audited fusion-component decomposition")


def _plan_numbered_candidate(
    mol: Molecule,
    atoms: frozenset[int],
    mode: FusionMode,
    ast: FusionNameAst,
    registry: FusionComponentRegistry,
    bounded: BoundedFaceModel,
    face_model: FaceModel,
    layouts: tuple[FusedLayout, ...],
    *,
    numbering_cache: dict[bool, CompletedNumberingSelection] | None = None,
) -> FusionPlanningResult:
    """Keep chemistry local to each numbering, sharing topology discovery."""
    specs = {match.occurrence_id: registry.spec_for_match(match) for match in ast.component_occurrences}
    try:
        prove_carbon_h = bool(intrinsic_carbon_candidate_atoms(ast, specs, mol)) or (
            any(component_parent_atoms(spec) != spec.atoms for spec in specs.values())
            and all(atom.symbol == "C" and atom.charge == 0 for spec in specs.values() for atom in spec.atoms)
        )
    except MancudeSearchBudgetExceeded as exc:
        return FusionUnsupported("component hydrogen assignment search budget exhausted", (str(exc),))
    except ValueError as exc:
        return FusionAuditFailed("component hydrogen constraints are inconsistent", (str(exc),))
    numbering_selection = numbering_cache.get(prove_carbon_h) if numbering_cache is not None else None
    if numbering_selection is None:
        numbering_selection = completed_system_numbering_selection(
            mol,
            bounded,
            face_model=face_model,
            layouts=layouts,
            defer_indicated_hydrogen=prove_carbon_h,
        )
        if numbering_cache is not None:
            numbering_cache[prove_carbon_h] = numbering_selection
    numberings = numbering_selection.accepted
    if not numberings:
        return FusionUnsupported("no layout-derived peripheral system numbering was proven")
    try:
        graph = _abstract_graph(ast, registry)
    except MancudeSearchBudgetExceeded as exc:
        return FusionUnsupported("mancude assignment search budget exhausted", (str(exc),))
    except ValueError as exc:
        return FusionAuditFailed("fusion component graphs could not be merged consistently", (str(exc),))
    try:
        intrinsic_sites = intrinsic_parent_lone_pair_sites(mol, graph)
        initial_model = (
            indicated_hydrogen_parent_bond_model(graph, intrinsic_sites)
            if intrinsic_sites
            else parent_bond_model(graph)
        )
    except MancudeSearchBudgetExceeded as exc:
        return FusionUnsupported("mancude assignment search budget exhausted", (str(exc),))
    except ValueError as exc:
        return FusionAuditFailed("fusion component constraints are inconsistent", (str(exc),))
    alternatives = []
    for candidate in numberings:
        if candidate.layout_index is None:
            return FusionUnsupported("completed-system numbering lacks intrinsic-layout provenance")
        layout = layouts[candidate.layout_index]
        proof = FusionNumberingProof(
            selected_face_model=face_model,
            selected_layout=layout,
            orientation_score=(layout.orientation_score, candidate.score),
            abstract_atom_to_locant=candidate.atom_to_locant,
            input_locant_maps=(candidate.atom_to_locant,),
            rejected_numberings=numbering_selection.rejected,
        )
        result = _complete_fusion_plan(
            mol, atoms, mode, ast, registry, proof, graph=graph, initial_bond_model=initial_model
        )
        if isinstance(result, FusionConfirmed):
            alternatives.append(result.plan)
    if not alternatives:
        return result
    if prove_carbon_h:
        best_h = min(tuple(map(system_locant_sort_key, plan.indicated_hydrogens)) for plan in alternatives)
        preferred = tuple(
            plan for plan in alternatives if tuple(map(system_locant_sort_key, plan.indicated_hydrogens)) == best_h
        )
    else:
        preferred = tuple(alternatives)
    plan = replace(
        preferred[0],
        numbering=replace(
            preferred[0].numbering,
            input_locant_maps=tuple(plan.numbering.input_locant_maps[0] for plan in preferred),
        ),
        numbering_variants=preferred,
    )
    return FusionConfirmed(plan)


def _complete_fusion_plan(
    mol: Molecule,
    atoms: frozenset[int],
    mode: FusionMode,
    ast: FusionNameAst,
    registry: FusionComponentRegistry,
    numbering: FusionNumberingProof,
    *,
    graph: FusionGraph | None = None,
    initial_bond_model: ParentBondModel | None = None,
) -> FusionPlanningResult:
    """Prove chemistry and rendering against one specific numbered parent."""

    try:
        if graph is None:
            graph = _abstract_graph(ast, registry)
    except MancudeSearchBudgetExceeded as exc:
        return FusionUnsupported("mancude assignment search budget exhausted", (str(exc),))
    except ValueError as exc:
        return FusionAuditFailed(
            "fusion component graphs could not be merged consistently",
            (str(exc),),
        )
    try:
        charge_operations = _fusion_charge_operations(mol, graph, numbering)
    except ValueError as exc:
        return FusionUnsupported("fused-parent charge operation is outside the audited production tier", (str(exc),))
    try:
        intrinsic_n_h = intrinsic_parent_lone_pair_sites(mol, graph)
        bond_model = (
            initial_bond_model
            if initial_bond_model is not None
            else (
                indicated_hydrogen_parent_bond_model(graph, intrinsic_n_h)
                if intrinsic_n_h
                else parent_bond_model(graph)
            )
        )
        specs = {match.occurrence_id: registry.spec_for_match(match) for match in ast.component_occurrences}
        initial_citations = set(_cited_indicated_hydrogens(mol, ast, registry, numbering, bond_model))
        cited_n_h = frozenset(
            atom
            for atom, locant in numbering.input_locant_maps[0]
            if locant in initial_citations
            and mol.atoms[atom].symbol == "N"
            and not mol.atoms[atom].charge
            and mol.atoms[atom].total_h_count == 1
        )
        bond_model, intrinsic_carbon_h = intrinsic_carbon_parent_model(
            mol,
            graph,
            bond_model,
            dict(numbering.input_locant_maps[0]),
            intrinsic_carbon_candidate_atoms(ast, specs, mol),
            intrinsic_hydrogen_atom_ids=intrinsic_n_h,
            cited_nitrogen_hydrogen_atom_ids=cited_n_h,
        )
    except MancudeSearchBudgetExceeded as exc:
        return FusionUnsupported("mancude assignment search budget exhausted", (str(exc),))
    except ValueError as exc:
        return FusionUnsupported("intrinsic hydrogen conflicts with the fused parent bond model", (str(exc),))
    indicated_h = _cited_indicated_hydrogens(mol, ast, registry, numbering, bond_model)
    input_locants = dict(numbering.input_locant_maps[0])
    indicated_h = tuple(
        sorted(set(indicated_h) | {input_locants[atom] for atom in intrinsic_carbon_h}, key=system_locant_sort_key)
    )
    input_atom_by_locant = {locant: atom for atom, locant in numbering.input_locant_maps[0]}
    indicated_h_atoms = {input_atom_by_locant[locant] for locant in indicated_h}
    try:
        derivative_state = parent_derivative_state(
            mol,
            atoms,
            bond_model,
            dict(numbering.input_locant_maps[0]),
            indicated_hydrogen_atom_ids=indicated_h_atoms,
        )
        if (
            derivative_state is not None
            and indicated_h_atoms
            and len(atoms) % 2 == 0
            and not any(spec.template.default_indicated_h for spec in specs.values())
            and (
                len(atoms - indicated_h_atoms - derivative_state.bond_delta.hydrogenated_atom_ids) > 1
                or any(mol.atoms[atom].symbol == "N" for atom in derivative_state.bond_delta.hydrogenated_atom_ids)
            )
            and saturated_nitrogen_hydrogen_sites(mol, atoms, {atom for atom in atoms if mol.atoms[atom].symbol == "N"})
        ):
            # Multiple saturated N-H citations can strand carbon positions.
            # A complete pi-pair cover instead owns all H through hydro ops.
            complete_state = parent_derivative_state(mol, atoms, bond_model, input_locants)
            if complete_state is not None and has_complete_saturated_hydrogenation(mol, atoms, complete_state):
                derivative_state = complete_state
                indicated_h = ()
    except MancudeSearchBudgetExceeded as exc:
        return FusionUnsupported("derivative parent assignment search budget exhausted", (str(exc),))
    if derivative_state is None:
        return FusionUnsupported("observed bond state cannot be expressed from the fused parent hydride")
    if SUPPORT.maximum_indicated_hydrogens is not None and len(indicated_h) > SUPPORT.maximum_indicated_hydrogens:
        return FusionUnsupported("multiple indicated-hydrogen fusion parents require a later additive tier")
    rendered_parts = render_fusion_name_parts(ast, registry, mol=mol)
    rendered_core_name = "".join(part.text for part in rendered_parts)
    lambda_descriptors = fusion_lambda_descriptors(
        mol,
        atoms,
        dict(numbering.input_locant_maps[0]),
    )
    if lambda_descriptors:
        rendered_parts = (
            NameTokenBinding(
                text=f"{','.join(descriptor.text for descriptor in lambda_descriptors)}-",
                token_kind="replacement",
                source="fusion_renderer",
                grammar_role="fusion_lambda_descriptor",
                binding_key="fusion:lambda_descriptor",
                atom_ids={descriptor.atom_id for descriptor in lambda_descriptors},
                locants=tuple(str(descriptor.locant) for descriptor in lambda_descriptors),
                match_priority=100,
            ),
            *rendered_parts,
        )
    if indicated_h:
        input_atom_by_locant = {locant: atom for atom, locant in numbering.input_locant_maps[0]}
        hydrogen_parts: list[NameTokenBinding] = []
        for index, locant in enumerate(indicated_h):
            hydrogen_parts.append(
                NameTokenBinding(
                    text=str(locant),
                    token_kind="locant",
                    source="fusion_renderer",
                    grammar_role="fusion_indicated_hydrogen",
                    binding_key=f"fusion:indicated_hydrogen:{locant}",
                    atom_ids={input_atom_by_locant[locant]},
                    locants=(str(locant),),
                    match_priority=100,
                )
            )
            hydrogen_parts.append(
                NameTokenBinding(
                    text="H",
                    token_kind="hydro",
                    source="fusion_renderer",
                    grammar_role="fusion_indicated_hydrogen",
                    binding_key=f"fusion:indicated_hydrogen:{locant}",
                    atom_ids={input_atom_by_locant[locant]},
                    locants=(str(locant),),
                    match_priority=100,
                )
            )
            hydrogen_parts.append(
                NameTokenBinding(
                    text="," if index + 1 < len(indicated_h) else "-",
                    token_kind="grammar",
                    source="fusion_renderer",
                    grammar_role="fusion_indicated_hydrogen_separator",
                    binding_key="fusion:indicated_hydrogen:separator",
                )
            )
        rendered_parts = (*hydrogen_parts, *rendered_parts)
    rendered = "".join(part.text for part in rendered_parts)
    audit = audit_fusion_plan(
        mol,
        atoms,
        ast=ast,
        abstract_parent_graph=graph,
        numbering=numbering,
        bond_model=bond_model,
        mode=mode,
        registry=registry,
        lambda_descriptors=lambda_descriptors,
        indicated_hydrogens=indicated_h,
        charge_operations=charge_operations,
        derivative_state=derivative_state,
        rendered_core_name=rendered_core_name,
    )
    if audit.status is AuditStatus.ABSTAIN:
        return FusionUnsupported("fusion nomenclature audit abstained", audit.errors)
    if not audit.confirmed:
        return FusionAuditFailed("fusion reconstruction audit rejected the candidate", audit.errors)
    root_id = ast.parent_occurrences[0]
    root_match = next(match for match in ast.component_occurrences if match.occurrence_id == root_id)
    seniority_trace = tuple(
        explain_component_comparison(root_match, match, registry)
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
        pin_eligibility="fusion_rules_satisfied",
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
        derivative_state=derivative_state,
        charge_operations=charge_operations,
        lambda_descriptors=lambda_descriptors,
        rendered_parts=rendered_parts,
    )
    return FusionConfirmed(plan)


def _cited_indicated_hydrogens(
    mol: Molecule,
    ast: FusionNameAst,
    registry: FusionComponentRegistry,
    numbering: FusionNumberingProof,
    bond_model: ParentBondModel,
) -> tuple[SystemLocant, ...]:
    """Select citations from component roles and the completed-system map."""

    locants = dict(numbering.abstract_atom_to_locant)
    candidates = indicated_hydrogen_candidate_atoms(mol, locants)
    candidate_atoms = set(candidates)
    unpaired = set()
    for assignment in bond_model.allowed_kekule_assignments:
        paired = {atom for edge, order in assignment.orders if order == 2 for atom in edge}
        unpaired.update(candidate_atoms - paired)
    component_h_roles = set()
    for match in ast.component_occurrences:
        spec = registry.spec_for_match(match)
        component_h_roles.update(match.input_atom_by_locant[locant] for locant in component_h_locants(spec, "N"))
    cited_atoms = {
        atom
        for atom in candidates
        if mol.atoms[atom].symbol != "C"
        and (mol.atoms[atom].is_aromatic or mol.atoms[atom].charge or atom in unpaired or atom in component_h_roles)
    }
    cited_atoms.update(
        saturated_nitrogen_hydrogen_sites(
            mol, frozenset(locants), {atom for atom in candidates if mol.atoms[atom].symbol == "N"}
        )
    )

    root_ids = set(ast.parent_occurrences)
    roots = [match for match in ast.component_occurrences if match.occurrence_id in root_ids]
    attached = [match for match in ast.component_occurrences if match.occurrence_id not in root_ids]
    has_attached_carbocycle = any(
        _is_attached_carbocycle_component(registry.spec_for_match(match)) for match in attached
    )
    has_polycyclic_parent = any(len(registry.spec_for_match(match).rings) > 1 for match in roots)
    if has_attached_carbocycle and has_polycyclic_parent and observed_parent_matches_bond_model(mol, bond_model):
        cited_atoms.update(atom for atom in candidates if mol.atoms[atom].symbol == "C")

    return tuple(sorted((locants[atom] for atom in cited_atoms), key=system_locant_sort_key))


def _is_attached_carbocycle_component(spec: FusionComponentSpec) -> bool:
    """Return whether a component is a monocyclic all-carbon attached prefix."""

    return (
        len(spec.rings) == 1
        and spec.usable_as_attached
        and not spec.usable_as_parent
        and all(atom.symbol == "C" for atom in spec.atoms)
    )


def _classify_unmodelled_ring_system(mol: Molecule, atoms: frozenset[int]) -> FusionPlanningResult:
    """Explain why a completed bounded-face proof was unavailable.

    This deliberately reuses the package's existing topology classifier and is
    called only on an already rejected candidate, so ordinary and successfully
    fused naming paths pay no additional graph-search cost.
    """

    topology = ring_system_topology(mol, atoms)
    if topology.classification in {"monospiro", "linear_dispiro", "complex_spiro"}:
        return FusionNotApplicable("spiro-only ring systems do not use fusion nomenclature")
    if topology.cycle_rank < 2:
        return FusionNotApplicable("selected parent has no audited multi-face fused model")
    if topology.fused_edges:
        return FusionUnsupported("fused ring system has no bounded face model within the configured ring-size tier")
    return FusionUnsupported("bridged or non-ortho polycycles are outside the ordinary fusion tier")


def _standard_valence_parent(mol: Molecule, atoms: frozenset[int]) -> bool:
    for atom_id in atoms:
        atom = mol.atoms[atom_id]
        if not atom.element.fusion_supported:
            return False
        bonding_number = sum(mol.get_bond(atom_id, neighbor).order for neighbor in mol.get_neighbors(atom_id))
        bonding_number += atom.total_h_count or atom.explicit_h_count
        if bonding_number > atom.element.standard_valence:
            return False
    return True


def _abstract_graph(ast, registry) -> FusionGraph:
    specs = {match.occurrence_id: registry.spec_for_match(match) for match in ast.component_occurrences}
    return component_parent_graph(ast, specs, relocate_carbon_h=component_carbon_h_relocation_scope(ast, specs))
