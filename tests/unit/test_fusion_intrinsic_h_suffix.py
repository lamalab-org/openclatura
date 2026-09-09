"""Intrinsic fusion-parent H must not be cited again as suffix-added H."""

import pytest
from rdkit import Chem

from openclatura import name_mol, opsin_available, verify_with_opsin
from openclatura.chains import find_ring_systems
from openclatura.fusion.model import FusionConfirmed
from openclatura.fusion.planner import plan_fusion_parent
from openclatura.graph_io import read_rdkit_mol


def _fused_ketone(branch):
    graph = Chem.RWMol()
    for symbol in ("O", "C", "C", "C", "C", "C", "C", "C", "N"):
        graph.AddAtom(Chem.Atom(symbol))
    for u, v, order in (
        (0, 1, 2),
        (1, 2, 1),
        (2, 3, 2),
        (3, 4, 1),
        (4, 5, 2),
        (5, 1, 1),
        (5, 6, 1),
        (6, 7, 2),
        (7, 8, 1),
        (8, 4, 1),
    ):
        graph.AddBond(u, v, Chem.BondType.DOUBLE if order == 2 else Chem.BondType.SINGLE)
    if branch:
        carbon = graph.AddAtom(Chem.Atom("C"))
        graph.AddBond(6, carbon, Chem.BondType.SINGLE)
    mol = graph.GetMol()
    Chem.SanitizeMol(mol)
    return mol


@pytest.mark.parametrize("branch", (False, True))
@pytest.mark.parametrize("reverse", (False, True))
def test_graph_built_fusion_ketone_keeps_intrinsic_h_before_parent(branch, reverse):
    mol = _fused_ketone(branch)
    if reverse:
        mol = Chem.RenumberAtoms(mol, list(reversed(range(mol.GetNumAtoms()))))
    graph = read_rdkit_mol(mol)
    atoms = max(find_ring_systems(graph), key=lambda ring: len(ring.atoms)).atoms
    planned = plan_fusion_parent(graph, atoms, mode="audited_pin")
    assert isinstance(planned, FusionConfirmed)
    state = planned.plan.derivative_state
    assert planned.plan.indicated_hydrogens
    assert state.oxo_operations
    assert not state.added_hydrogen_operations
    result = name_mol(mol)
    assert result.error is None
    assert "1H-cyclopenta[b]pyrrol-4-one" in result.name
    assert "(1H)" not in result.name
    if opsin_available():
        assert verify_with_opsin(result.name, Chem.MolToSmiles(mol), standardize_smiles=False).status == "matched"


@pytest.mark.parametrize(
    "smiles,expected",
    (
        ("O=C1C=CC2=C1NC=C2", "1H-cyclopenta[b]pyrrol-6-one"),
        ("O=C1NC2=C(O1)C=CO2", "3H-furo[2,3-d][1,3]oxazol-2-one"),
        ("O=C1NC2=C(OC=C2)O1", "1H-furo[3,2-d][1,3]oxazol-2-one"),
        ("O=C1OC2=C(O1)C=CN2", "4H-[1,3]dioxolo[4,5-b]pyrrol-2-one"),
    ),
)
def test_fused_heteroketone_does_not_duplicate_indicated_h(smiles, expected):
    result = name_mol(Chem.MolFromSmiles(smiles))
    assert result.name == expected
    if opsin_available():
        assert verify_with_opsin(result.name, smiles, standardize_smiles=False).status == "matched"
