"""Exact charged fusion production names and graph-bound operation coverage."""

from copy import deepcopy
from dataclasses import replace

import pytest
from rdkit import Chem

from openclatura import name_mol, opsin_available
from openclatura.fusion import planner
from openclatura.fusion.charges import fusion_charge_operations, fusion_component_charge_parent
from openclatura.fusion.model import FusionChargeOperation, FusionChargeOperationKind
from openclatura.graph_io import read_smiles
from openclatura.ionic_naming import apply_terminal_parent_ide
from openclatura.locants import SystemLocant
from openclatura.opsin_verify import verify_with_opsin

WITNESSES = [
    ("[N-]1[NH+]=CC=C2C=CN=C12", "pyrrolo[2,3-c]pyridazin-1-ide-2-ium"),
    ("[N-]1[NH+]=CC=C2N=CC=C12", "pyrrolo[3,2-c]pyridazin-1-ide-2-ium"),
    ("[N-]1[NH+]=CC=C2N=CN=C12", "imidazo[4,5-c]pyridazin-1-ide-2-ium"),
    ("[N-]1[NH+]=CN=C2C=CN=C12", "pyrrolo[3,2-e][1,2,4]triazin-1-ide-2-ium"),
    ("[N-]1[NH+]=CN=C2N=CN=C12", "imidazo[4,5-e][1,2,4]triazin-1-ide-2-ium"),
    ("[N-]1[NH+]=NC=C2N=CN=C12", "imidazo[4,5-d][1,2,3]triazin-1-ide-2-ium"),
    ("[NH3+][C-]1C=CC2=C1N=NO2", "cyclopenta[d][1,2,3]oxadiazol-4-ide-4-aminium"),
    ("O=C1[CH-]NC2=C1C[NH2+]C2", "3-oxo-1,4,5,6-tetrahydropyrrolo[3,4-b]pyrrol-5-ium-2-ide"),
]


def _assert_exact_charged_graph(decoded_smiles, original):
    decoded = Chem.MolFromSmiles(decoded_smiles)
    assert decoded is not None
    assert Chem.MolToSmiles(decoded) == Chem.MolToSmiles(original)
    assert sorted(
        (atom.GetSymbol(), atom.GetFormalCharge(), atom.GetTotalNumHs())
        for atom in decoded.GetAtoms()
        if atom.GetFormalCharge()
    ) == sorted(
        (atom.GetSymbol(), atom.GetFormalCharge(), atom.GetTotalNumHs())
        for atom in original.GetAtoms()
        if atom.GetFormalCharge()
    )


@pytest.mark.opsin
@pytest.mark.skipif(not opsin_available(), reason="py2opsin/Java is unavailable")
@pytest.mark.parametrize(("smiles", "expected"), WITNESSES)
@pytest.mark.parametrize("reverse", [False, True], ids=["original-order", "reversed-order"])
def test_charged_fusion_witness_is_opsin_graph_exact(smiles, expected, reverse):
    mol = Chem.MolFromSmiles(smiles)
    if reverse:
        mol = Chem.RenumberAtoms(mol, list(reversed(range(mol.GetNumAtoms()))))
    check = verify_with_opsin(expected, Chem.MolToSmiles(mol, canonical=False))
    assert check.status == "matched", check
    _assert_exact_charged_graph(check.opsin_smiles, mol)


@pytest.mark.opsin
@pytest.mark.skipif(not opsin_available(), reason="py2opsin/Java is unavailable")
@pytest.mark.parametrize(
    ("smiles", "expected"),
    WITNESSES
    + [
        ("[N-]1[NH+]=C(C)C=C2C=CN=C12", "3-methylpyrrolo[2,3-c]pyridazin-1-ide-2-ium"),
        ("[N-]1[NH+]=C(C)C=C2N=CC=C12", "3-methylpyrrolo[3,2-c]pyridazin-1-ide-2-ium"),
        ("[N-]1[NH+]=C(C)C=C2N=CN=C12", "3-methylimidazo[4,5-c]pyridazin-1-ide-2-ium"),
        ("[NH2+](C)[C-]1C=CC2=C1N=NO2", "N-methylcyclopenta[d][1,2,3]oxadiazol-4-ide-4-aminium"),
        (
            "O=C1[CH-]N(C)C2=C1C[NH2+]C2",
            "1-methyl-3-oxo-1,2,4,6-tetrahydro-5H-pyrrolo[3,4-b]pyrrol-5-ium-2-ide",
        ),
        (
            "O=C1[CH-]NC2=C1C[NH+](C)C2",
            "5-methyl-3-oxo-1,4,5,6-tetrahydropyrrolo[3,4-b]pyrrol-5-ium-2-ide",
        ),
    ],
)
def test_default_charge_parent_preserves_graph_after_reordering(smiles, expected):
    mol = Chem.MolFromSmiles(smiles)
    for order in (list(range(mol.GetNumAtoms())), list(reversed(range(mol.GetNumAtoms())))):
        result = name_mol(Chem.RenumberAtoms(mol, order), verify_opsin=True, include_trace=True)
        assert result.name == expected
        assert result.parent_nomenclature == "systematic_fusion"
        assert result.opsin_check.status == "matched"
        _assert_exact_charged_graph(result.opsin_check.opsin_smiles, mol)
        parent = result.substituent_tree[0]["parent"]
        selected = next(
            step for step in result.decisions if step.decision == "selected audited systematic fusion parent"
        )
        for operation in selected.data["charge_operations"]:
            locant = operation["locant"]
            assert operation["atom_id"] == parent["atom_ids_by_locant"][locant]
            assert operation["symbol"] == parent["atom_symbols_by_locant"][locant]
            assert operation["observed_charge"] == parent["atom_charges_by_locant"][locant]
        assembled = next(step for step in result.decisions if step.decision == "assembled component name")
        rewrite = next(
            item for item in assembled.data["name_rewrite_history"] if item["name"] == "apply_anionic_parent_names"
        )
        assert rewrite["before"] == rewrite["after"] == expected


@pytest.mark.opsin
@pytest.mark.skipif(not opsin_available(), reason="py2opsin/Java is unavailable")
def test_charged_fusion_parent_is_rendered_before_branch_conversion():
    mol = Chem.MolFromSmiles("[N-]1[NH+]=C(NC(C)=O)C=C2C=CN=C12")
    expected = "N-(pyrrolo[2,3-c]pyridazin-1-ide-2-ium-3-yl)acetamide"
    for order in (list(range(mol.GetNumAtoms())), list(reversed(range(mol.GetNumAtoms())))):
        result = name_mol(Chem.RenumberAtoms(mol, order), verify_opsin=True)
        assert result.name == expected
        assert result.opsin_check.status == "matched"
        _assert_exact_charged_graph(result.opsin_check.opsin_smiles, mol)


@pytest.mark.opsin
@pytest.mark.skipif(not opsin_available(), reason="py2opsin/Java is unavailable")
def test_protonated_furo_pyridine_has_no_duplicate_indicated_hydrogen():
    smiles = "O1C=CC=2C1=[NH+]C=CC2"
    mol = Chem.MolFromSmiles(smiles)
    for order in (list(range(mol.GetNumAtoms())), list(reversed(range(mol.GetNumAtoms())))):
        result = name_mol(Chem.RenumberAtoms(mol, order), verify_opsin=True)
        assert result.name == "furo[2,3-b]pyridin-7-ium"
        assert result.opsin_check.status == "matched"
        _assert_exact_charged_graph(result.opsin_check.opsin_smiles, mol)


def test_component_charge_view_preserves_original_atoms_and_caches():
    mol = read_smiles(WITNESSES[0][0])
    before = dict(mol.atoms)
    mol._fusion_plan_cache["sentinel"] = object()
    mol._retained_fused_cache["sentinel"] = ()
    parent = fusion_component_charge_parent(mol, frozenset(mol.atoms))
    assert mol.atoms == before
    assert "sentinel" in mol._fusion_plan_cache and "sentinel" in mol._retained_fused_cache
    assert not parent._fusion_plan_cache and not parent._retained_fused_cache
    assert parent.bonds == mol.bonds
    for atom_id, atom in mol.atoms.items():
        if atom.charge == -1:
            assert parent.atoms[atom_id].charge == 0
            assert parent.atoms[atom_id].total_h_count == atom.total_h_count + 1
        else:
            assert parent.atoms[atom_id] == atom


@pytest.mark.parametrize("smiles", [WITNESSES[0][0], WITNESSES[-1][0]])
def test_charge_helper_rejects_invalid_graph_metadata(monkeypatch, smiles):
    captured = []

    def capture(mol, graph, numbering):
        captured.append((mol, graph, numbering))
        return fusion_charge_operations(mol, graph, numbering)

    monkeypatch.setattr(planner, "_fusion_charge_operations", capture)
    mol = read_smiles(smiles)
    parent_atoms = frozenset(atom for atom in mol.atoms if len(mol.get_neighbors(atom)) > 1)
    planner.plan_fusion_parent(mol, parent_atoms, mode="audited_pin")
    assert captured
    mol, graph, numbering = captured[0]
    site = next(atom.id for atom in graph.atoms if mol.atoms[atom.id].charge == -1)
    invalid = deepcopy(mol)
    invalid.atoms[site] = replace(invalid.atoms[site], total_h_count=invalid.atoms[site].total_h_count + 1)
    with pytest.raises(ValueError, match="proton-removal"):
        fusion_charge_operations(invalid, graph, numbering)
    changed_graph = replace(
        graph, atoms=tuple(replace(atom, symbol="O") if atom.id == site else atom for atom in graph.atoms)
    )
    with pytest.raises(ValueError, match="parent element"):
        fusion_charge_operations(mol, changed_graph, numbering)
    missing = replace(
        numbering,
        input_locant_maps=(
            tuple(
                (max(mol.atoms) + 1 if atom == site else atom, locant)
                for atom, locant in numbering.input_locant_maps[0]
            ),
        ),
    )
    with pytest.raises(ValueError, match="no completed-system locant"):
        fusion_charge_operations(mol, graph, missing)


@pytest.mark.parametrize("symbol", ["C", "N"])
def test_deprotonation_operation_rejects_other_charge_deltas(symbol):
    operation = FusionChargeOperation(0, SystemLocant(1), symbol, 0, -1, FusionChargeOperationKind.DEPROTONATION)
    for changes in ({"observed_charge": -2}, {"base_charge": 1}, {"symbol": "O"}, {"operation_kind": "unknown"}):
        with pytest.raises(ValueError):
            replace(operation, **changes)


@pytest.mark.parametrize("phrase", ["ethyl acetate", "N-ethylbenzamide", "(pyridin-2-yl)methyl", "ethanol"])
@pytest.mark.parametrize("symbol", ["C", "N"])
def test_negative_atom_does_not_authorize_arbitrary_terminal_ide(phrase, symbol):
    assert apply_terminal_parent_ide(phrase, {"2": {symbol}}) == phrase
