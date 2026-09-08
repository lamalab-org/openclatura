"""Reported fusion failures requiring carbon indicated H alongside nitrogen H."""

import pytest
from rdkit import Chem

from openclatura import name_mol, opsin_available, verify_with_opsin
from openclatura.fusion.audit import audit_fusion_plan
from openclatura.fusion.model import FusionConfirmed, FusionMode
from openclatura.fusion.planner import plan_fusion_parent
from openclatura.graph_io import read_smiles

CORES = (
    pytest.param("C1C=CC2=C1N=CN2", id="5651"),
    pytest.param("C1CC2=C(C1)C=CN2", id="5684"),
    pytest.param("C1C2CC1C1=CNC=C21", id="24212"),
)
ORDERINGS = ("original", "reverse", "rotate")


@pytest.fixture(scope="module", params=CORES)
def core_smiles(request):
    return request.param


@pytest.fixture(scope="module", params=("core", "C0-methyl"))
def case(core_smiles, request):
    graph = Chem.RWMol(Chem.MolFromSmiles(core_smiles))
    if request.param == "C0-methyl":
        site = graph.GetAtomWithIdx(0)
        assert site.GetSymbol() == "C"
        assert site.GetHybridization() == Chem.HybridizationType.SP3
        assert site.GetTotalNumHs() == 2
        assert all(bond.GetBondType() == Chem.BondType.SINGLE for bond in site.GetBonds())
        methyl = graph.AddAtom(Chem.Atom("C"))
        graph.AddBond(0, methyl, Chem.BondType.SINGLE)
    Chem.SanitizeMol(graph)
    # Fix the reference before renumbering, independently of the generated name.
    reference_smiles = core_smiles if request.param == "core" else Chem.MolToSmiles(graph)
    return graph.GetMol(), reference_smiles


def _atom_order(graph, ordering):
    order = list(range(graph.GetNumAtoms()))
    if ordering == "reverse":
        order.reverse()
    elif ordering == "rotate":
        order = order[3:] + order[:3]
    return Chem.RenumberAtoms(graph, order)


@pytest.fixture(scope="module", params=ORDERINGS)
def ordered_case(case, request):
    graph, reference_smiles = case
    return _atom_order(graph, request.param), reference_smiles


@pytest.fixture(scope="module")
def named_case(ordered_case):
    graph, _ = ordered_case
    return name_mol(graph, include_trace=True)


def test_public_naming_requires_fusion_not_von_baeyer(named_case):
    assert named_case.error is None, named_case
    assert named_case.name
    assert named_case.parent_nomenclature in {"systematic_fusion", "bridged_fusion"}, named_case


def test_public_name_is_atom_order_invariant(case):
    graph, _ = case
    names = []
    for ordering in ORDERINGS:
        result = name_mol(_atom_order(graph, ordering), include_trace=True)
        assert result.error is None, result
        assert result.name
        names.append(result.name)
    assert names == [names[0]] * len(names), names


@pytest.mark.opsin
def test_generated_name_roundtrips_exactly(ordered_case, named_case):
    if not opsin_available():
        pytest.skip("OPSIN and Java are required")
    _, reference_smiles = ordered_case
    assert named_case.error is None, named_case
    assert named_case.name
    check = verify_with_opsin(named_case.name, reference_smiles, standardize_smiles=False)
    assert check.status == "matched", (named_case.name, check.to_dict())
    assert check.canonical_roundtrip == check.canonical_original


@pytest.mark.parametrize("damage", (None, "omit_carbon", "hydro_site"))
def test_independent_audit_checks_mixed_h_citations(damage):
    mol = read_smiles("C1CC2=C(C1)C=CN2")
    result = plan_fusion_parent(mol, mol.atoms, mode=FusionMode.AUDITED_PIN)
    assert isinstance(result, FusionConfirmed)
    plan = result.plan
    atom_by_locant = {locant: atom for atom, locant in plan.numbering.input_locant_maps[0]}
    citations = plan.indicated_hydrogens
    assert sorted(mol.atoms[atom_by_locant[locant]].symbol for locant in citations) == ["C", "N"]
    hydro_atoms = {atom for operation in plan.derivative_state.hydro_operations for atom in operation.atom_ids}
    assert not hydro_atoms.intersection(atom_by_locant[locant] for locant in citations)
    if damage:
        citations = tuple(locant for locant in citations if mol.atoms[atom_by_locant[locant]].symbol == "N")
        if damage == "hydro_site":
            citations += (next(locant for locant, atom in atom_by_locant.items() if atom in hydro_atoms),)
    audit = audit_fusion_plan(
        mol,
        mol.atoms,
        ast=plan.ast,
        abstract_parent_graph=plan.abstract_parent_graph,
        numbering=plan.numbering,
        bond_model=plan.bond_model,
        derivative_state=plan.derivative_state,
        indicated_hydrogens=citations,
        mode=FusionMode.AUDITED_PIN,
    )
    assert audit.confirmed is (damage is None), audit
