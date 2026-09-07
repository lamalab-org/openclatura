"""Generator coverage for intrinsic NH and neutral fusion-N composition."""

import pytest
from rdkit import Chem

from openclatura import FusionMode, name_mol, opsin_available
from openclatura.chains import find_ring_systems
from openclatura.fusion.model import FusionConfirmed
from openclatura.fusion.planner import plan_fusion_parent
from openclatura.graph_io import read_smiles

CASES = (
    (
        "O=C(Nc1ccc2nc(=O)n3c(c2c1)NCC3)c1cc(Cl)ccc1Cl",
        (
            "2,5-dichloro-N-(5-oxo-2,3-dihydro-1H-imidazo[1,2-c]benzo[e]pyrimidin-9-yl)benzamide",
            "2,5-dichloro-N-(5-oxo-2,3-dihydro-1H-imidazo[1,2-c]quinazolin-9-yl)benzamide",
        ),
        2,
    ),
    (
        "CC1(CO)CN(Cc2ccccc2)CC2CN(Cc3ccc(F)cc3)CCN21",
        ("(8-benzyl-2-((4-fluorophenyl)methyl)-6-methyloctahydropyrazino[1,2-a]pyrazin-6-yl)methanol",),
        8,
    ),
)


@pytest.mark.parametrize("smiles,expected,hydrogen_count", CASES)
@pytest.mark.parametrize("reverse", (False, True))
@pytest.mark.parametrize("mode", (FusionMode.GENERAL, FusionMode.AUDITED_PIN))
def test_generator_composes_intrinsic_nitrogen_and_hydro(smiles, expected, hydrogen_count, reverse, mode):
    mol = Chem.MolFromSmiles(smiles)
    if reverse:
        mol = Chem.RenumberAtoms(mol, list(reversed(range(mol.GetNumAtoms()))))
    result = name_mol(mol, fusion_mode=mode, verify_opsin=opsin_available())
    assert result.name in expected
    if reverse:
        assert result.name == name_mol(Chem.MolFromSmiles(smiles), fusion_mode=mode).name
    assert result.error is None
    if opsin_available():
        assert result.opsin_check.status == "matched"

    graph = read_smiles(Chem.MolToSmiles(mol, canonical=False))
    atoms = max(find_ring_systems(graph), key=lambda system: len(system.atoms)).atoms
    planned = plan_fusion_parent(graph, atoms, mode=mode)
    assert isinstance(planned, FusionConfirmed)
    state = planned.plan.derivative_state
    assert len(state.hydro_operations) == 1
    assert len(state.hydro_operations[0].locants) == hydrogen_count
    assert not state.unsaturation_operations
    junction_n = {
        atom
        for atom in atoms
        if graph.atoms[atom].symbol == "N"
        and graph.atoms[atom].charge == 0
        and len(atoms.intersection(graph.get_neighbors(atom))) == 3
    }
    assert junction_n
    assert not junction_n.intersection(state.hydro_operations[0].atom_ids)
    assert all(order == 1 for edge, order in state.bond_delta.assignment.orders if junction_n.intersection(edge))


@pytest.mark.parametrize("alkyl", ("C", "CC"))
def test_fully_saturated_fusion_nitrogen_variants_round_trip(alkyl):
    smiles = f"{alkyl}N1CCN2CCN({alkyl})CC2C1"
    result = name_mol(Chem.MolFromSmiles(smiles), fusion_mode=FusionMode.GENERAL, verify_opsin=opsin_available())
    assert result.error is None
    assert "octahydropyrazino[1,2-a]pyrazine" in result.name
    assert "decahydro" not in result.name
    if opsin_available():
        assert result.opsin_check.status == "matched"
