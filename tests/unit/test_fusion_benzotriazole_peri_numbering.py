"""Peri-fused unequal ring sizes must retain both allowed numbering orientations."""

import pytest
from rdkit import Chem

from openclatura import name_mol, opsin_available, verify_with_opsin
from openclatura.fusion.model import FusionConfirmed
from openclatura.fusion.planner import plan_fusion_parent
from openclatura.graph_io import read_rdkit_mol

CORE_SMILES = "n1nc2cccc3c2n1CCCS3"
CASES = (
    pytest.param((None, CORE_SMILES), id="core"),
    pytest.param((9, "CC1CCSc2cccc3nnn1c23"), id="N-adjacent-methyl"),
    pytest.param((10, "CC1CSc2cccc3nnn(c23)C1"), id="middle-methyl"),
    pytest.param((11, "CC1CCn2nnc3cccc(c32)S1"), id="S-adjacent-methyl"),
)


@pytest.fixture(scope="module", params=CASES)
def case(request):
    methyl_site, reference_smiles = request.param
    graph = Chem.RWMol(Chem.MolFromSmiles(CORE_SMILES))
    if methyl_site is not None:
        site = graph.GetAtomWithIdx(methyl_site)
        assert site.GetSymbol() == "C" and site.GetTotalNumHs() == 2
        methyl = graph.AddAtom(Chem.Atom("C"))
        graph.AddBond(methyl_site, methyl, Chem.BondType.SINGLE)
    Chem.SanitizeMol(graph)
    return graph.GetMol(), reference_smiles


def _atom_order(graph, ordering):
    order = list(range(graph.GetNumAtoms()))
    if ordering == "reverse":
        order.reverse()
    elif ordering == "rotate":
        order = order[5:] + order[:5]
    return Chem.RenumberAtoms(graph, order)


@pytest.fixture(scope="module", params=("original", "reverse", "rotate"))
def ordered_case(case, request):
    graph, reference_smiles = case
    return _atom_order(graph, request.param), reference_smiles


@pytest.fixture(scope="module")
def named_case(ordered_case):
    graph, _ = ordered_case
    return name_mol(graph, include_trace=True)


def test_graph_built_variants_match_diagnostic(case):
    graph, reference_smiles = case
    assert Chem.MolToSmiles(graph) == Chem.MolToSmiles(Chem.MolFromSmiles(reference_smiles))


@pytest.mark.parametrize("symbol,expected", (("N", {"1", "2", "3"}), ("S", {"7"})))
def test_completed_system_heteroatom_locants(ordered_case, symbol, expected):
    graph, _ = ordered_case
    mol = read_rdkit_mol(graph)
    parent_atoms = {atom.GetIdx() for atom in graph.GetAtoms() if atom.IsInRing()}
    planned = plan_fusion_parent(mol, parent_atoms, mode="general")

    assert isinstance(planned, FusionConfirmed), planned
    assert planned.plan.audit.confirmed
    for items in planned.plan.numbering.input_locant_maps:
        actual = {str(locant) for atom, locant in items if mol.atoms[atom].symbol == symbol}
        assert actual == expected


def test_public_naming_uses_systematic_fusion(named_case):
    assert named_case.error is None, named_case
    assert named_case.name
    assert named_case.parent_nomenclature == "systematic_fusion"


def test_public_name_is_atom_order_invariant(case):
    graph, _ = case
    names = []
    for ordering in ("original", "reverse", "rotate"):
        result = name_mol(_atom_order(graph, ordering), include_trace=True)
        assert result.error is None, result
        assert result.name
        assert result.parent_nomenclature == "systematic_fusion"
        names.append(result.name)
    assert names == [names[0]] * len(names), names


@pytest.mark.opsin
def test_generated_name_roundtrips_exactly(ordered_case, named_case):
    if not opsin_available():
        pytest.skip("OPSIN and Java are required")
    _, reference_smiles = ordered_case
    assert named_case.error is None, named_case
    assert named_case.name
    assert named_case.parent_nomenclature == "systematic_fusion"
    check = verify_with_opsin(named_case.name, reference_smiles, standardize_smiles=False)
    assert check.status == "matched", check.to_dict()
    assert check.canonical_roundtrip == check.canonical_original


@pytest.mark.opsin
def test_diagnostic_core_name_has_exact_opsin_witness():
    if not opsin_available():
        pytest.skip("OPSIN and Java are required")
    # An independent witness, not a required spelling for the generated name.
    check = verify_with_opsin(
        "4,5-dihydro[1,4]thiazepino[4,3,2-cd]benzotriazole",
        CORE_SMILES,
        standardize_smiles=False,
    )
    assert check.status == "matched", check.to_dict()
    assert check.canonical_roundtrip == check.canonical_original
