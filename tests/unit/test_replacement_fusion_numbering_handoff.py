"""Replacement numbering must select the matching graph-bound derivative plan."""

import pytest
from rdkit import Chem

from openclatura import name, name_mol, opsin_available, parent_pipeline


def _replacement_parent(branch_length):
    graph = Chem.RWMol()
    for symbol in ("C", "C", "N", "C", "C", "C", "C", "N", "C", "N", "C", "C", "N", "O", "C", "N"):
        graph.AddAtom(Chem.Atom(symbol))
    edges = (
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 8),
        (8, 9),
        (9, 10),
        (10, 11),
        (11, 6),
        (10, 12),
        (12, 0),
        (11, 2),
        (1, 13),
        (12, 14),
        (8, 15),
    )
    double = {(6, 7), (8, 9), (10, 11), (1, 13)}
    for edge in edges:
        graph.AddBond(*edge, Chem.BondType.DOUBLE if edge in double else Chem.BondType.SINGLE)
    last = 0
    for _ in range(branch_length):
        atom = graph.AddAtom(Chem.Atom("C"))
        graph.AddBond(last, atom, Chem.BondType.SINGLE)
        last = atom
    mol = graph.GetMol()
    Chem.SanitizeMol(mol)
    return mol


@pytest.mark.parametrize("branch_length", (0, 1, 2))
def test_selected_replacement_map_and_derivative_plan_agree(monkeypatch, branch_length):
    original = parent_pipeline.build_parent_parts
    captured = []

    def capture(mol, numbered_path, get_loc, *args, **kwargs):
        parent = kwargs.get("parent_hydride")
        if parent is not None and parent.is_skeletal_replacement_fusion:
            selected_map = {atom: str(get_loc(atom)) for atom in parent.atoms}
            assert selected_map == parent.fusion_plan.numbering.string_input_locant_maps()[0]
            captured.append(parent)
        return original(mol, numbered_path, get_loc, *args, **kwargs)

    monkeypatch.setattr(parent_pipeline, "build_parent_parts", capture)
    graph = _replacement_parent(branch_length)
    names = set()
    for order in (list(range(graph.GetNumAtoms())), list(reversed(range(graph.GetNumAtoms())))):
        result = name_mol(Chem.RenumberAtoms(graph, order), include_trace=True, token_debug=True)
        assert result.error is None
        assert result.parent_nomenclature == "skeletal_replacement_fusion"
        names.add(result.name)
    assert len(captured) == 2
    assert len(names) == 1


@pytest.mark.skipif(not opsin_available(), reason="OPSIN and Java are required")
@pytest.mark.parametrize("branch_length", (0, 1, 2))
def test_replacement_derivatives_roundtrip(branch_length):
    result = name_mol(_replacement_parent(branch_length), verify_opsin=True)
    assert result.error is None
    assert result.parent_nomenclature == "skeletal_replacement_fusion"
    assert result.opsin_check.status == "matched", result.opsin_check.to_dict()


@pytest.mark.skipif(not opsin_available(), reason="OPSIN and Java are required")
def test_reported_replacement_parent_with_stereochemical_sidechain_roundtrips():
    result = name(
        "CCC1C(=O)N2CCCc3nc(NC/C(C=NCc4ccc(C(F)(F)F)nc4)=C/N)nc(c32)N1C",
        verify_opsin=True,
    )
    assert result.error is None
    assert result.parent_nomenclature == "skeletal_replacement_fusion"
    assert result.opsin_check.status == "matched", result.opsin_check.to_dict()
