"""Component-local maximum matchings need a common completed resonance witness."""

import pytest
from rdkit import Chem

from openclatura import name_mol, opsin_available, verify_with_opsin
from openclatura.chains import find_ring_systems
from openclatura.fusion.component_hydrogen import component_hydrogen_consumption
from openclatura.fusion.model import FusionConfirmed
from openclatura.fusion.planner import plan_fusion_parent
from openclatura.fusion.registry import fusion_component_registry
from openclatura.graph_io import read_rdkit_mol


def _graph(branch_length, halogen):
    graph = Chem.RWMol()
    for symbol in ("N", "C", "C", "C", "C", "C", "C", "C", "O", "C", "C", "O", "C", "C", "C", "C"):
        graph.AddAtom(Chem.Atom(symbol))
    edges = (
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 2),
        (7, 8),
        (8, 9),
        (9, 10),
        (10, 15),
        (15, 0),
        (10, 11),
        (11, 12),
        (12, 13),
        (13, 14),
        (14, 15),
    )
    for edge in edges:
        graph.AddBond(*edge, Chem.BondType.DOUBLE if edge in {(2, 3), (4, 5), (6, 7)} else Chem.BondType.SINGLE)
    oxygen = graph.AddAtom(Chem.Atom("O"))
    graph.AddBond(1, oxygen, Chem.BondType.DOUBLE)
    methyl = graph.AddAtom(Chem.Atom("C"))
    graph.AddBond(0, methyl, Chem.BondType.SINGLE)
    last = 12
    for _ in range(branch_length):
        atom = graph.AddAtom(Chem.Atom("C"))
        graph.AddBond(last, atom, Chem.BondType.SINGLE)
        last = atom
    if halogen:
        atom = graph.AddAtom(Chem.Atom(halogen))
        graph.AddBond(4, atom, Chem.BondType.SINGLE)
    Chem.SanitizeMol(graph)
    return graph.GetMol()


@pytest.mark.parametrize("branch_length", (0, 1, 2))
@pytest.mark.parametrize("halogen", (None, "F"))
def test_completed_parent_uses_a_common_component_matching_witness(branch_length, halogen):
    graph = _graph(branch_length, halogen)
    original = Chem.MolToSmiles(graph)
    for order in (list(range(graph.GetNumAtoms())), list(reversed(range(graph.GetNumAtoms())))):
        reordered = Chem.RenumberAtoms(graph, order)
        mol = read_rdkit_mol(reordered)
        atoms = max(find_ring_systems(mol), key=lambda ring: len(ring.atoms)).atoms
        planned = plan_fusion_parent(mol, atoms, mode="audited_pin")
        assert isinstance(planned, FusionConfirmed), planned
        plan = planned.plan
        registry = fusion_component_registry()
        specs = {match.occurrence_id: registry.spec_for_match(match) for match in plan.ast.component_occurrences}
        proof = component_hydrogen_consumption(plan.ast, specs)
        assert proof is not None
        assert proof.parent_model.maximum_non_cumulative_double_bonds == 7
        assert proof.parent_model.allowed_kekule_assignments
        result = name_mol(reordered, include_trace=True)
        assert result.error is None
        assert "pyrano[2,3-c]benzo[g][1,5]oxazocin" in result.name
        if opsin_available():
            check = verify_with_opsin(result.name, original, standardize_smiles=False)
            assert check.ok, check.to_dict()
