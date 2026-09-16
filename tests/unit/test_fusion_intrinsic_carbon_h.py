"""Intrinsic carbon H constrains a parent; it does not consume hydro pairs."""

from dataclasses import replace

import pytest
from rdkit import Chem

from openclatura import FusionMode, name, name_mol, opsin_available
from openclatura.chains import find_ring_systems
from openclatura.fusion import indicated_hydrogen as intrinsic
from openclatura.fusion.audit import audit_fusion_plan
from openclatura.fusion.model import AuditStatus, FusionConfirmed
from openclatura.fusion.numbering import parent_bond_model
from openclatura.fusion.planner import plan_fusion_parent
from openclatura.fusion.registry import fusion_component_registry
from openclatura.graph_io import read_smiles

CASES = (
    (
        "CCOC(=O)c1c(NC(=O)C(CC)Sc2cccc(N)c2)sc2c1CCCCC2",
        "ethyl 2-(2-((3-aminophenyl)sulfanyl)butanamido)-5,6,7,8-tetrahydro-4H-cyclohepta[b]thiophene-3-carboxylate",
        "4",
        ("5", "6", "7", "8"),
    ),
    ("C1CC2OCC=CC2O1", "2,3,3a,7a-tetrahydro-5H-furo[3,2-b]pyran", "5", ("2", "3", "3a", "7a")),
    ("C1N=COC2=NON=C12", "7H-[1,2,5]oxadiazolo[3,4-e][1,3]oxazine", "7", ()),
)


def _plan(smiles):
    mol = read_smiles(smiles)
    atoms = max(find_ring_systems(mol), key=lambda system: len(system.atoms)).atoms
    result = plan_fusion_parent(mol, atoms, mode=FusionMode.AUDITED_PIN)
    assert isinstance(result, FusionConfirmed), result
    return mol, atoms, result.plan


def _audit(mol, atoms, plan, **changes):
    arguments = dict(
        ast=plan.ast,
        abstract_parent_graph=plan.abstract_parent_graph,
        numbering=plan.numbering,
        bond_model=plan.bond_model,
        indicated_hydrogens=plan.indicated_hydrogens,
        derivative_state=plan.derivative_state,
        charge_operations=plan.charge_operations,
        lambda_descriptors=plan.lambda_descriptors,
    )
    arguments.update(changes)
    return audit_fusion_plan(mol, atoms, **arguments)


@pytest.mark.parametrize("smiles,expected,h,hydro", CASES)
def test_intrinsic_carbon_h_is_separate_from_hydrogenation(smiles, expected, h, hydro):
    mol, atoms, plan = _plan(smiles)
    assert tuple(map(str, plan.indicated_hydrogens)) == (h,)
    locants = dict(plan.numbering.input_locant_maps[0])
    site = next(atom for atom, locant in locants.items() if str(locant) == h)
    assert all(
        order == 1
        for assignment in plan.bond_model.allowed_kekule_assignments
        for edge, order in assignment.orders
        if site in edge
    )
    state = plan.derivative_state
    assert tuple(locant for operation in state.hydro_operations for locant in operation.locants) == hydro
    assert not state.unsaturation_operations
    assert not state.oxo_operations
    assert _audit(mol, atoms, plan).confirmed
    omitted = _audit(mol, atoms, plan, indicated_hydrogens=())
    assert omitted.status is AuditStatus.MISMATCH
    assert "fusion indicated-hydrogen citations omit a graph-required site" in omitted.errors
    unconstrained = _audit(mol, atoms, plan, bond_model=parent_bond_model(plan.abstract_parent_graph))
    assert unconstrained.status is AuditStatus.MISMATCH
    assert "parent bond model does not preserve the proved intrinsic-hydrogen sites" in unconstrained.errors


@pytest.mark.parametrize("smiles,expected,h,hydro", CASES)
def test_exact_intrinsic_carbon_name_is_atom_order_invariant(smiles, expected, h, hydro):
    mol = Chem.MolFromSmiles(smiles)
    for order in (list(range(mol.GetNumAtoms())), list(reversed(range(mol.GetNumAtoms())))):
        result = name_mol(Chem.RenumberAtoms(mol, order), fusion_mode=FusionMode.AUDITED_PIN, include_trace=True)
        assert result.name == expected
        assert result.parent_nomenclature == "systematic_fusion"
        assert result.pin_status == "confirmed"


@pytest.mark.skipif(not opsin_available(), reason="py2opsin/Java is unavailable")
@pytest.mark.parametrize(
    "smiles,expected",
    [(case[0], case[1]) for case in CASES]
    + [
        ("CN1CCC2=C1C=NN2", "4-methyl-5,6-dihydro-1H-pyrrolo[3,2-c]pyrazole"),
    ],
)
def test_production_intrinsic_h_names_roundtrip(smiles, expected):
    result = name(smiles, verify_opsin=True)
    assert result.name == expected
    assert result.opsin_check is not None and result.opsin_check.ok


def test_declared_carbon_h_can_move_without_changing_component_mancude_capacity():
    spec = fusion_component_registry().by_key["pyran"].spec
    atoms = intrinsic.component_parent_atoms(spec)
    assert next(atom for atom in spec.atoms if atom.locant == "2").saturated
    assert not next(atom for atom in atoms if atom.locant == "2").saturated
    assert intrinsic.component_parent_atoms(spec) is atoms
    assert parent_bond_model(intrinsic._component_graph(spec, atoms)).maximum_non_cumulative_double_bonds == 2


@pytest.mark.parametrize(
    "constraint", ["unmarked", "forced_single", "explicit_zero_pi", "wrong_mancude_count", "fixed_bond"]
)
def test_component_saturation_is_not_globally_released(constraint):
    spec = fusion_component_registry().by_key["pyran"].spec
    template = spec.template
    if constraint == "unmarked":
        template = replace(template, default_indicated_h=())
    elif constraint == "wrong_mancude_count":
        template = replace(template, mancude_double_bonds=3)
    elif constraint == "fixed_bond":
        template = replace(template, bonds=(replace(template.bonds[0], bond_class="single"), *template.bonds[1:]))
    else:
        field = {"forced_single": {"forced_single": True}, "explicit_zero_pi": {"pi_capacity": 0}}[constraint]
        template = replace(
            template, atoms=tuple(replace(atom, **field) if atom.locant == "2" else atom for atom in template.atoms)
        )
    spec = replace(spec, template=template)
    assert intrinsic.component_parent_atoms(spec) == spec.atoms


def test_no_ch2_skips_component_candidate_model_search(monkeypatch):
    mol, atoms, plan = _plan("c1cc2ccsc2o1")
    registry = fusion_component_registry()
    specs = {match.occurrence_id: registry.spec_for_match(match) for match in plan.ast.component_occurrences}

    def unexpected(*args, **kwargs):
        pytest.fail("aromatic parents without CH2 must not prepare carbon-H component models")

    monkeypatch.setattr(intrinsic, "_component_carbon_h_locants", unexpected)
    monkeypatch.setattr(intrinsic, "_component_carbon_pi_locants", unexpected)
    assert not intrinsic.intrinsic_carbon_candidate_atoms(plan.ast, specs, mol)
