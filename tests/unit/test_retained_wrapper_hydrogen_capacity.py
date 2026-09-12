"""Retained bridge parents must keep hydrogen spelling and pi proofs aligned."""

import random
from dataclasses import replace

import pytest
from rdkit import Chem

from openclatura import name_mol, opsin_available, verify_with_opsin
from openclatura.fusion.numbering import RetainedParentBondCapacityError, retained_template_parent_bond_model
from openclatura.fusion.wrappers import _retained_wrapper_parent
from openclatura.graph_io import read_rdkit_mol
from openclatura.retained_fused_templates import retained_graph_templates


@pytest.fixture
def template():
    return next(value for value in retained_graph_templates() if value.name == "4H-quinolizine")


@pytest.mark.parametrize("locant,valid", [(str(i), i % 2 == 0) for i in (1, 2, 3, 4, 6, 7, 8, 9)])
def test_relocated_carbon_hydrogen_must_preserve_parent_capacity(template, locant, valid):
    mapping = {locant: i for i, locant in enumerate(template.locants)}
    if not valid:
        with pytest.raises(RetainedParentBondCapacityError):
            retained_template_parent_bond_model(template, mapping, indicated_h=(locant,))
    else:
        model = retained_template_parent_bond_model(template, mapping, indicated_h=(locant,))
        assert model.maximum_non_cumulative_double_bonds == template.mancude_double_bonds
        assert all(
            order == 1
            for assignment in model.allowed_kekule_assignments
            for edge, order in assignment.orders
            if mapping[locant] in edge
        )


@pytest.mark.parametrize("reverse", (False, True))
@pytest.mark.parametrize("side_length", (0, 2))
def test_hydrogenated_parent_retries_original_hydrogen_state(template, reverse, side_length):
    graph = Chem.RWMol()
    mapping = {atom.locant: graph.AddAtom(Chem.Atom(atom.symbol)) for atom in template.atoms}
    for bond in template.bonds:
        graph.AddBond(*(mapping[locant] for locant in bond.locants), Chem.BondType.SINGLE)
    parent_count = graph.GetNumAtoms()
    previous = mapping["1"]
    for _ in range(side_length):
        atom = graph.AddAtom(Chem.Atom("C"))
        graph.AddBond(previous, atom, Chem.BondType.SINGLE)
        previous = atom
    Chem.SanitizeMol(graph)
    order = list(range(graph.GetNumAtoms()))
    if reverse:
        order.reverse()
    mol = read_rdkit_mol(Chem.RenumberAtoms(graph, order))
    parent = _retained_wrapper_parent(mol, frozenset(order.index(i) for i in range(parent_count)))
    assert parent is not None
    metadata = parent.hydride.hydride_metadata
    assert metadata.default_indicated_h == template.default_indicated_h
    assert not metadata.relocated_indicated_h
    assert parent.hydride.parent_name == template.name
    for entries, model in zip(parent.locant_maps, parent.bond_models, strict=True):
        assert model.maximum_non_cumulative_double_bonds == template.mancude_double_bonds
        h_atom = next(atom for atom, locant in entries if locant == template.default_indicated_h[0])
        assert all(
            order == 1
            for assignment in model.allowed_kekule_assignments
            for edge, order in assignment.orders
            if h_atom in edge
        )


def test_invalid_default_template_capacity_is_still_an_error(template):
    mapping = {locant: i for i, locant in enumerate(template.locants)}
    with pytest.raises(RetainedParentBondCapacityError):
        retained_template_parent_bond_model(replace(template, mancude_double_bonds=99), mapping)


@pytest.mark.parametrize("trace", (False, True))
@pytest.mark.parametrize("ordering", ("original", "reversed", "shuffled"))
def test_benchmark_bridged_parent_names_and_preserves_stereochemistry(trace, ordering):
    source = Chem.MolFromSmiles("C[C@]12CCC[C@]34CCC[C@@H](C=C[C@H]1CC3)N24")
    order = list(range(source.GetNumAtoms()))
    if ordering == "reversed":
        order.reverse()
    elif ordering == "shuffled":
        random.Random(53).shuffle(order)
    source = Chem.RenumberAtoms(source, order)
    result = name_mol(source, include_trace=trace)
    assert result.error is None
    assert result.name == ("(1R,4S,6R,9aS)-9a-methyl-1,6,7,8,9,9a-hexahydro-1,6-ethano-4,6-propano-4H-quinolizine")
    if trace:
        assert result.parent_nomenclature == "bridged_fusion"
    if opsin_available():
        check = verify_with_opsin(result.name, Chem.MolToSmiles(source), standardize_smiles=False)
        assert check.status == "matched", check.to_dict()
