"""Small fused faces must share the accurate lattice of their larger neighbors."""

import pytest
from rdkit import Chem

from openclatura import name_mol, opsin_available, verify_with_opsin
from openclatura.fusion.layout import RING_SHAPE_TEMPLATES
from openclatura.fusion.model import FusionConfirmed, FusionMode
from openclatura.fusion.planner import plan_fusion_parent
from openclatura.graph_io import read_rdkit_mol


def _aziridine_fused_bicycle(element="O", methyl=False, reverse=False):
    graph = Chem.RWMol()
    for symbol in ("C", "C", "C", "C", "C", "N", "C", element, "N"):
        graph.AddAtom(Chem.Atom(symbol))
    for left, right in ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 3), (4, 8), (8, 0), (8, 1)):
        graph.AddBond(left, right, Chem.BondType.DOUBLE if (left, right) in {(3, 4), (5, 6)} else Chem.BondType.SINGLE)
    if methyl:
        graph.AddBond(0, graph.AddAtom(Chem.Atom("C")), Chem.BondType.SINGLE)
    Chem.SanitizeMol(graph)
    return Chem.RenumberAtoms(graph, list(reversed(range(graph.GetNumAtoms())))) if reverse else graph


def test_equilateral_triangle_shares_the_five_and_six_ring_lattice():
    shape = next(shape for shape in RING_SHAPE_TEMPLATES if shape.shape_id == "triangle-eisenstein")
    assert shape.coordinate_system == "eisenstein"
    assert shape.distortion_rank == 0
    lengths = []
    for left, right in zip(shape.vertices, shape.vertices[1:] + shape.vertices[:1]):
        dx, dy = right[0] - left[0], right[1] - left[1]
        lengths.append(dx * dx + dx * dy + dy * dy)
    assert lengths == [16, 16, 16]


def test_three_five_five_fusion_uses_a_complete_linear_row_and_correct_locants():
    mol = read_rdkit_mol(_aziridine_fused_bicycle())
    result = plan_fusion_parent(mol, mol.atoms, mode=FusionMode.AUDITED_PIN)
    assert isinstance(result, FusionConfirmed), result
    plan = result.plan
    assert plan.numbering.selected_layout.orientation_score[:2] == (0, -3)
    locants = plan.numbering.string_input_locant_maps()[0]
    assert {atom: locants[atom] for atom in (7, 5, 8, 1)} == {7: "1", 5: "3", 8: "4", 1: "5a"}
    assert plan.audit.confirmed


@pytest.mark.opsin
@pytest.mark.parametrize("element", ("O", "S"))
@pytest.mark.parametrize("methyl", (False, True))
@pytest.mark.parametrize("reverse", (False, True))
def test_triangle_fused_heterocycles_and_branches_roundtrip(element, methyl, reverse):
    if not opsin_available():
        pytest.skip("OPSIN and Java are required")
    graph = _aziridine_fused_bicycle(element, methyl, reverse)
    result = name_mol(graph)
    assert result.error is None, result.error
    assert "azirino[" in result.name
    assert "pyrrolo[" in result.name
    check = verify_with_opsin(result.name, Chem.MolToSmiles(graph), standardize_smiles=False)
    assert check.status == "matched", (result.name, check.to_dict())
    assert Chem.MolToSmiles(Chem.MolFromSmiles(check.opsin_smiles)) == Chem.MolToSmiles(graph)
