"""Component identity must not depend on an isolated indicated-H tautomer."""

from dataclasses import replace

import pytest
from rdkit import Chem

from openclatura import name_mol, opsin_available, verify_with_opsin
from openclatura.fusion.audit import audit_fusion_plan
from openclatura.fusion.component_hydrogen import (
    _component_hydrogen_consumption,
    component_hydrogen_consumption,
)
from openclatura.fusion.descriptor import build_fusion_name_ast, render_fusion_name
from openclatura.fusion.faces import GraphCycle, cached_bounded_face_model
from openclatura.fusion.model import AuditStatus, FusionConfirmed, FusionMode
from openclatura.fusion.planner import plan_fusion_parent
from openclatura.fusion.registry import _component_matching_template, fusion_component_registry
from openclatura.graph_io import read_rdkit_mol
from openclatura.retained_fused_templates import match_retained_graph_template_maps, retained_graph_templates


def _fused_pyran(nitrogens=(), *, methyl=False):
    """Build the 5/6-ring skeleton shared by QM9 24473--24477."""
    graph = Chem.RWMol()
    for atom in range(9):
        graph.AddAtom(Chem.Atom("O" if atom == 0 else "N" if atom in nitrogens else "C"))
    for atom in range(9):
        graph.AddBond(
            atom,
            (atom + 1) % 9,
            Chem.BondType.DOUBLE if atom in {1, 3, 5, 7} else Chem.BondType.SINGLE,
        )
    graph.AddBond(3, 7, Chem.BondType.SINGLE)
    if methyl:
        added = graph.AddAtom(Chem.Atom("C"))
        graph.AddBond(1, added, Chem.BondType.SINGLE)
    Chem.SanitizeMol(graph)
    return graph.GetMol()


@pytest.mark.parametrize("role", [None, "parent", "attached"])
def test_explicit_component_precedes_generated_hw_without_losing_maps(role):
    mol = read_rdkit_mol(_fused_pyran())
    face = GraphCycle.from_atoms((0, 1, 2, 3, 7, 8))
    registry = fusion_component_registry()
    spec = registry.get("pyran").spec
    source = spec.template
    assert source.atom_by_locant["2"].saturated
    assert not match_retained_graph_template_maps(mol, set(face.atoms), source, allow_nonaromatic=True)
    skeleton = {atom: atom + 100 for atom in mol.atoms}
    matches = registry.match_faces(mol, (face,), role=role, input_to_skeleton_atom=skeleton)
    assert len(matches) == 2
    assert {match.spec_key for match in matches} == {"pyran"}
    assert len({match.local_to_input_atom for match in matches}) == 2
    for match in matches:
        assert dict(match.local_to_input_atom)["1"] == 0
        assert set(dict(match.local_to_input_atom).values()) == set(face.atoms)
        assert dict(match.local_to_skeleton_atom) == {
            locant: skeleton[atom] for locant, atom in match.local_to_input_atom
        }
        assert registry.spec_for_match(match).template is source
    view = _component_matching_template(spec)
    assert not view.atom_by_locant["2"].saturated
    assert view.default_indicated_h == source.default_indicated_h == ("2",)
    assert view.mancude_double_bonds == source.mancude_double_bonds == 2
    assert view.bonds == source.bonds
    assert view.rings == source.rings
    assert source.atom_by_locant["2"].saturated
    retained = next(template for template in retained_graph_templates() if template.name == "pyran")
    assert retained.atom_by_locant["2"].saturated


@pytest.mark.parametrize("constraint", ["forced_single", "pi_capacity", "exact_bonds"])
def test_component_view_cannot_relax_inherent_saturation(constraint):
    spec = fusion_component_registry().get("pyran").spec
    template = spec.template
    if constraint == "exact_bonds":
        template = replace(template, bonds=tuple(replace(bond, bond_class="single") for bond in template.bonds))
    else:
        changes = {"forced_single": True} if constraint == "forced_single" else {"pi_capacity": 0}
        template = replace(
            template,
            atoms=tuple(replace(atom, **changes) if atom.locant == "2" else atom for atom in template.atoms),
        )
    assert _component_matching_template(replace(spec, template=template)) is template


def test_components_without_movable_carbon_h_keep_original_template():
    registry = fusion_component_registry()
    for key in ("furan", "pyridine", "imidazole"):
        spec = registry.get(key).spec
        assert _component_matching_template(spec) is spec.template


def test_generated_hw_remains_available_without_an_explicit_component():
    graph = Chem.RWMol()
    for atom in range(7):
        graph.AddAtom(Chem.Atom("O" if atom == 0 else "C"))
    for atom in range(7):
        graph.AddBond(atom, (atom + 1) % 7, Chem.BondType.DOUBLE if atom in {1, 3, 5} else Chem.BondType.SINGLE)
    Chem.SanitizeMol(graph)
    mol = read_rdkit_mol(graph.GetMol())
    matches = fusion_component_registry().match_faces(mol, (GraphCycle.from_atoms(tuple(range(7))),))
    assert matches
    assert all(match.spec_key.startswith("generated-hw:") for match in matches)


CASES = [
    (24473, (), "cyclopenta[c]pyran"),
    (24474, (6,), "pyrano[3,4-b]pyrrole"),
    (24475, (5,), "pyrano[3,4-c]pyrrole"),
    (24476, (4,), "pyrano[4,3-b]pyrrole"),
    (24477, (4, 6), "pyrano[3,4-d]imidazole"),
]


@pytest.mark.opsin
@pytest.mark.parametrize("index,nitrogens,citation", CASES)
def test_report_family_citations_and_public_names_roundtrip_exactly(index, nitrogens, citation):
    if not opsin_available():
        pytest.skip("OPSIN and Java are required")
    graph = _fused_pyran(nitrogens)
    original = Chem.MolToSmiles(graph)
    registry = fusion_component_registry()
    public_names = []
    for order in (list(range(9)), list(reversed(range(9)))):
        reordered = Chem.RenumberAtoms(graph, order)
        mol = read_rdkit_mol(reordered)
        faces = cached_bounded_face_model(mol, set(mol.atoms))
        matches = registry.match_faces(mol, faces)
        ast = build_fusion_name_ast(mol, matches, registry)
        rendered = render_fusion_name(ast, registry)
        assert rendered == citation
        planned = plan_fusion_parent(mol, mol.atoms, mode=FusionMode.AUDITED_PIN)
        assert isinstance(planned, FusionConfirmed), (index, planned)
        assert planned.plan.rendered_base_name == citation
        assert not planned.plan.indicated_hydrogens
        assert not planned.plan.derivative_state.hydro_operations
        assert not planned.plan.derivative_state.unsaturation_operations
        assert planned.plan.bond_model.maximum_non_cumulative_double_bonds == 4
        result = name_mol(reordered)
        assert result, (index, result)
        assert result.name == citation
        for text in (rendered, result.name):
            check = verify_with_opsin(text, original, standardize_smiles=False)
            assert check.ok, (index, check.to_dict())
            assert check.canonical_roundtrip == original, (index, check.to_dict())
        public_names.append(result.name)
    assert public_names[0] == public_names[1]


@pytest.mark.opsin
def test_graph_built_substituted_variant_roundtrips_without_name_replacement():
    if not opsin_available():
        pytest.skip("OPSIN and Java are required")
    graph = _fused_pyran(methyl=True)
    original = Chem.MolToSmiles(graph)
    result = name_mol(graph)
    assert result, result
    assert "cyclopenta[c]pyran" in result.name
    check = verify_with_opsin(result.name, original, standardize_smiles=False)
    assert check.ok, check.to_dict()
    assert check.canonical_roundtrip == original, check.to_dict()


@pytest.fixture
def carbon_h_composition():
    mol = read_rdkit_mol(_fused_pyran())
    registry = fusion_component_registry()
    faces = cached_bounded_face_model(mol, mol.atoms)
    ast = build_fusion_name_ast(mol, registry.match_faces(mol, faces), registry)
    specs = {match.occurrence_id: registry.spec_for_match(match) for match in ast.component_occurrences}
    return mol, ast, specs


def test_movable_component_h_is_consumed_at_other_fusion_locants(carbon_h_composition):
    _mol, ast, specs = carbon_h_composition
    proof = component_hydrogen_consumption(ast, specs)
    assert proof is not None
    junctions = {atom for join in ast.joins for atom in join.shared_input_atoms}
    for match in ast.component_occurrences:
        for locant in specs[match.occurrence_id].template.default_indicated_h:
            assert match.input_atom_by_locant[locant] not in junctions
            assert (match.occurrence_id, locant) in proof.default_sites
    assert {atom for _, atom in proof.junction_sites} == junctions
    assert proof.parent_model.maximum_non_cumulative_double_bonds == 4


@pytest.mark.parametrize("keep", [0, 1])
def test_consumption_requires_every_unpaired_local_site_to_be_a_junction(carbon_h_composition, keep):
    _mol, ast, specs = carbon_h_composition
    junctions = sorted({atom for join in ast.joins for atom in join.shared_input_atoms})
    components = tuple((match, specs[match.occurrence_id]) for match in ast.component_occurrences)
    assert _component_hydrogen_consumption(components, frozenset(junctions[:keep])) is None


@pytest.mark.parametrize("constraint", ["pi_budget", "forced_single", "charge", "heteroatom_h", "fixed_bond"])
def test_consumption_cannot_waive_unproved_component_constraints(carbon_h_composition, constraint):
    _mol, ast, specs = carbon_h_composition
    occurrence = next(key for key, spec in specs.items() if spec.key == "pyran")
    spec = specs[occurrence]
    template = spec.template
    if constraint == "pi_budget":
        template = replace(template, mancude_double_bonds=1)
    elif constraint == "heteroatom_h":
        template = replace(template, default_indicated_h=("1", "2"))
    elif constraint == "fixed_bond":
        template = replace(template, bonds=tuple(replace(bond, bond_class="single") for bond in template.bonds))
    else:
        changes = {"forced_single": True} if constraint == "forced_single" else {"charge": 1}
        template = replace(
            template,
            atoms=tuple(replace(atom, **changes) if atom.locant == "2" else atom for atom in template.atoms),
        )
    changed = {**specs, occurrence: replace(spec, template=template)}
    assert component_hydrogen_consumption(ast, changed) is None


def test_residual_completed_parent_carbon_h_is_not_consumed():
    graph = Chem.RWMol(_fused_pyran())
    Chem.Kekulize(graph, clearAromaticFlags=True)
    graph.RemoveBond(5, 6)
    added = graph.AddAtom(Chem.Atom("C"))
    graph.AddBond(5, added, Chem.BondType.DOUBLE)
    graph.AddBond(added, 6, Chem.BondType.SINGLE)
    Chem.SanitizeMol(graph)
    mol = read_rdkit_mol(graph.GetMol())
    registry = fusion_component_registry()
    faces = cached_bounded_face_model(mol, mol.atoms)
    ast = build_fusion_name_ast(mol, registry.match_faces(mol, faces), registry)
    specs = {match.occurrence_id: registry.spec_for_match(match) for match in ast.component_occurrences}
    assert component_hydrogen_consumption(ast, specs) is None


@pytest.mark.parametrize("field", ["abstract_parent_graph", "bond_model"])
def test_consumption_does_not_bypass_independent_parent_audit(carbon_h_composition, field):
    mol, _ast, _specs = carbon_h_composition
    result = plan_fusion_parent(mol, mol.atoms, mode=FusionMode.AUDITED_PIN)
    assert isinstance(result, FusionConfirmed)
    plan = result.plan
    arguments = dict(
        ast=plan.ast,
        abstract_parent_graph=plan.abstract_parent_graph,
        numbering=plan.numbering,
        bond_model=plan.bond_model,
        indicated_hydrogens=plan.indicated_hydrogens,
        derivative_state=plan.derivative_state,
        mode=FusionMode.AUDITED_PIN,
    )
    assert audit_fusion_plan(mol, mol.atoms, **arguments).status is AuditStatus.CONFIRMED
    if field == "abstract_parent_graph":
        graph = plan.abstract_parent_graph
        carbon = next(atom.id for atom in graph.atoms if atom.symbol == "C")
        arguments[field] = replace(
            graph, atoms=tuple(replace(atom, pi_capacity=0) if atom.id == carbon else atom for atom in graph.atoms)
        )
    else:
        assignment = plan.bond_model.allowed_kekule_assignments[0]
        dropped = next(edge for edge, order in assignment.orders if order == 2)
        changed = replace(
            assignment, orders=tuple((edge, 1 if edge == dropped else order) for edge, order in assignment.orders)
        )
        arguments[field] = replace(
            plan.bond_model, allowed_kekule_assignments=(changed,), maximum_non_cumulative_double_bonds=3
        )
    assert audit_fusion_plan(mol, mol.atoms, **arguments).status is not AuditStatus.CONFIRMED
