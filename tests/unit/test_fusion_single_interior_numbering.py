"""A unique interior vertex follows the rule for its own element.

An interior CARBON takes P-25.3.3.3: the locant of its nearest peripheral atom,
the lowest where several are equally near, with the bond distance as a
superscript. Uniqueness is not an exception - phenalene has one interior carbon
and is printed 3a-superscript-one, and P-52.2.4.4.2.1 carries that locant
through skeletal replacement in 1,3a-superscript-one,4,9-tetraazaphenalene.

An interior HETEROATOM reached through heterocyclic fusion components is a
separate procedure and continues the integer sequence. Both rules live here so
that collapsing one into the other shows up immediately.

The pinned parser continues fusion letters for interior carbons instead, so it
cannot read a name citing the superscript. That divergence is asserted, not
waived: a parse returning some other structure would still be a failure.
"""

import random

import pytest
from rdkit import Chem

from openclatura import name_mol, opsin_available
from openclatura.chains import find_ring_systems
from openclatura.fusion.faces import select_bounded_face_model
from openclatura.fusion.model import FusionMode
from openclatura.fusion.numbering import _number_completed_system
from openclatura.fusion.third_component import plan_third_component_fusion_parent
from openclatura.graph_io import read_rdkit_mol
from openclatura.molecule import Molecule
from openclatura.opsin_verify import verify_with_opsin


@pytest.mark.parametrize("symbol,expected", (("C", "2a\u00b9"), ("N", "7")))
@pytest.mark.parametrize("reverse_ids", (False, True))
def test_unique_interior_vertex_follows_the_rule_for_its_element(symbol, expected, reverse_ids):
    index = {atom: 9 - atom if reverse_ids else atom for atom in range(10)}
    cycles = ((0, 1, 2, 3, 4), (0, 4, 5, 6, 7), (0, 7, 8, 9, 1))
    mol = Molecule()
    for atom in range(10):
        mol.add_atom(symbol if atom == 0 else "C", idx=index[atom])
    edges = {tuple(sorted((a, b))) for cycle in cycles for a, b in zip(cycle, cycle[1:] + cycle[:1])}
    for edge, (a, b) in enumerate(sorted(edges)):
        mol.add_bond(index[a], index[b], idx=edge)
    bounded = select_bounded_face_model(mol, mol.atoms)
    assert bounded is not None
    assert bounded.interior_atoms == frozenset({index[0]})
    perimeter = tuple(index[a] for a in (2, 3, 4, 5, 6, 7, 8, 9, 1))
    locants = _number_completed_system(mol, bounded, perimeter, {index[a] for a in (0, 1, 4, 7)})
    assert locants is not None
    assert set(locants) == set(mol.atoms)
    assert len(set(locants.values())) == len(mol.atoms)
    assert str(locants[index[0]]) == expected
    # Only the interior carbon carries a distance; the heteroatom keeps its
    # integer, and no peripheral atom is superscripted either way.
    carried = {atom for atom, locant in locants.items() if locant.interior_distance is not None}
    assert carried == ({index[0]} if symbol == "C" else set())
    assert [str(locants[atom]) for atom in perimeter] == ["1", "2", "2a", "3", "4", "4a", "5", "6", "6a"]


CASES = (
    pytest.param("COC(=O)C1(C(=O)OC)CC2=C(C)C(=O)[C@H]3CC[C@@H]1[C@@H]23", id="pubchem-15044"),
    pytest.param("Cc1nc2cccc3cnc1n32", id="pubchem-33015"),
    pytest.param("Cc1ccc(S(=O)(=O)O[C@H]2C[C@@H]3C=C[C@@H]4C=C[C@@H]2[C@H]43)cc1", id="pubchem-72233"),
)


@pytest.mark.skipif(not opsin_available(), reason="OPSIN is unavailable")
@pytest.mark.parametrize("smiles", CASES)
@pytest.mark.parametrize("ordering", ("original", "reversed", "shuffled"))
def test_single_interior_completed_names_roundtrip_exactly(smiles, ordering):
    graph = Chem.MolFromSmiles(smiles)
    order = list(range(graph.GetNumAtoms()))
    if ordering == "reversed":
        order.reverse()
    elif ordering == "shuffled":
        random.Random(73).shuffle(order)
    graph = Chem.RenumberAtoms(graph, order)
    result = name_mol(graph, fusion_mode=FusionMode.AUDITED_PIN)
    assert result.error is None
    assert "cyclo[" not in result.name
    check = verify_with_opsin(result.name, smiles, standardize_smiles=False)
    if any(mark in result.name for mark in "\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079"):
        # The name cites an interior carbon, which the pinned parser cannot
        # read. Asserting the rejection keeps the record honest and makes it
        # go stale the day the parser learns the locant.
        assert check.status == "name_unparseable", (result.name, check)
    else:
        assert check.status == "matched", (result.name, check)


def test_skeletal_replacement_keeps_all_single_interior_maps_under_permutations():
    graph = Chem.MolFromSmiles("CCc1c(C(CC)OC)c2n3c1CCC[C@@H]3CC2")
    order = list(range(graph.GetNumAtoms()))
    shuffled = order.copy()
    random.Random(73).shuffle(shuffled)
    results = []
    for indices in (order, list(reversed(order)), shuffled):
        mol = read_rdkit_mol(Chem.RenumberAtoms(graph, indices))
        atoms = max(find_ring_systems(mol), key=lambda system: len(system.atoms)).atoms
        result = plan_third_component_fusion_parent(mol, atoms, mode=FusionMode.AUDITED_PIN)
        assert result is not None
        maps = {
            tuple(sorted((indices[atom], str(locant)) for atom, locant in locants.items()))
            for locants in result.parent.proof_locant_maps
        }
        assert len(maps) == 2
        # P-52.2.4.4.2.1: replacement prefixes do not renumber the hydrocarbon
        # framework, so the interior carbon keeps the locant that framework
        # gave it - the 1,3a-superscript-one,4,9-tetraazaphenalene pattern.
        assert all(dict(locants)[10] == "2a\u00b9" for locants in maps)
        results.append(maps)
    assert results[0] == results[1] == results[2]
