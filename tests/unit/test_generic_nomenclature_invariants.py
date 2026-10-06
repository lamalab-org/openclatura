from random import Random

import pytest
from rdkit import Chem

from openclatura import RULES, name_mol


def _permuted_names(smiles: str, *, seed: int, count: int = 6) -> set[str]:
    graph = Chem.MolFromSmiles(smiles)
    assert graph is not None
    random = Random(seed)
    names = set()
    for _ in range(count):
        order = random.sample(range(graph.GetNumAtoms()), graph.GetNumAtoms())
        result = name_mol(Chem.RenumberAtoms(graph, order))
        assert result.ok, result.error
        names.add(result.name)
    return names


@pytest.mark.parametrize(
    "smiles,expected",
    [
        ("CC([O-])[C+]=C=O", "4-oxobut-3-en-3-ylium-2-olate"),
        ("[C+]#CC#CC#C[CH2-]", "hepta-1,3,5-triyn-1-ylium-7-ide"),
        (
            "c1ccc([C+]2C[C-](c3ccccc3)C3CCCC23)cc1",
            "2,4-diphenylbicyclo[3.3.0]octan-4-ylium-2-ide",
        ),
    ],
)
def test_charged_parent_numbering_is_atom_order_invariant(smiles, expected):
    assert _permuted_names(smiles, seed=712) == {expected}


@pytest.mark.parametrize(
    "smiles,expected",
    [
        (
            "C1Oc2c3c(c(c4c2C4)O1)C3",
            "9,11-dioxatetracyclo[3.3.3.0^{6,8}.0^{2,4}]undeca-1,4,6(8)-triene",
        ),
        (
            "C12CCC(CC3CCC(CC4CCC(C2)CC4)CC3)CC1",
            "tetracyclo[11.2.2.2^{8,11}.2^{3,6}]henicosane",
        ),
    ],
)
def test_audited_polycycle_decomposition_is_atom_order_invariant(smiles, expected):
    assert _permuted_names(smiles, seed=913) == {expected}


def test_nitrogen_template_constraints_are_well_formed_and_uniquely_identified():
    templates = RULES.nitrogen.chain_templates
    identities = [(template.scope, template.bond_orders, template.charges) for template in templates]

    assert len(identities) == len(set(identities))
    assert all(len(template.charges) == len(template.bond_orders) + 1 for template in templates)
    assert all(template.name and template.group_key for template in templates)
