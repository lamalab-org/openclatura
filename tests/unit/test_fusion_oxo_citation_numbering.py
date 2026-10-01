"""An oxo citation must use the parser-compatible numbered fusion graph."""

import random

import pytest
from rdkit import Chem

from openclatura import name_mol, opsin_available
from openclatura.chains import find_ring_systems
from openclatura.fusion.model import FusionConfirmed
from openclatura.fusion.planner import plan_fusion_parent
from openclatura.graph_io import read_rdkit_mol
from openclatura.opsin_verify import verify_with_opsin


@pytest.mark.skipif(not opsin_available(), reason="OPSIN and Java are required")
@pytest.mark.parametrize("urea_oxygen,extend_methyl", [(False, False), (True, False), (False, True)])
def test_peri_fusion_oxo_locant_cannot_target_the_shared_nitrogen(urea_oxygen, extend_methyl):
    graph = Chem.RWMol(Chem.MolFromSmiles("CC1CCc2cccc3c2N1C(=O)c1cc(NC(=S)Nc2ccccc2)ccc1O3"))
    if urea_oxygen:
        next(atom for atom in graph.GetAtoms() if atom.GetAtomicNum() == 16).SetAtomicNum(8)
    if extend_methyl:
        methyl = next(atom.GetIdx() for atom in graph.GetAtoms() if atom.GetAtomicNum() == 6 and atom.GetDegree() == 1)
        graph.AddBond(methyl, graph.AddAtom(Chem.Atom(6)), Chem.BondType.SINGLE)
    Chem.SanitizeMol(graph)
    smiles = Chem.MolToSmiles(graph)
    original = list(range(graph.GetNumAtoms()))
    orders = [original, original[::-1]]
    for seed in (7, 73, 19452, 20483):
        order = original.copy()
        random.Random(seed).shuffle(order)
        orders.append(order)
    names = set()
    for order in orders:
        permuted = Chem.RenumberAtoms(graph, order)
        mol = read_rdkit_mol(permuted)
        atoms = max(find_ring_systems(mol), key=lambda system: len(system.atoms)).atoms
        planned = plan_fusion_parent(mol, atoms, mode="audited_pin")
        assert isinstance(planned, FusionConfirmed)
        assert planned.plan.audit.confirmed
        locants = planned.plan.numbering.string_input_locant_maps()[0]
        assert {locants[atom] for atom in atoms if mol.atoms[atom].symbol == "N"} == {"13"}
        assert {locants[atom] for atom in atoms if mol.atoms[atom].symbol == "O"} == {"7"}
        assert tuple(op.locant for op in planned.plan.derivative_state.oxo_operations) == ("12",)
        result = name_mol(permuted, include_trace=True, verify_self=True)
        assert result.error is None
        assert not result.self_audit.coverage.unnamed_atoms
        check = verify_with_opsin(result.name, smiles, standardize_smiles=False)
        assert check.status == "matched", check.to_dict()
        assert check.canonical_original == check.canonical_roundtrip
        names.add(result.name)
    assert len(names) == 1
