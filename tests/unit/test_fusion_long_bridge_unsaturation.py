"""Long neutral bridges use graph-locanted dehydrogenation, not finite prefix grammar."""

import pytest
from rdkit import Chem

from openclatura import name_mol, opsin_available, verify_with_opsin
from openclatura.fusion.numbering import retained_template_parent_bond_model
from openclatura.retained_fused_templates import retained_graph_templates


def _graph(parent, length):
    template = next(t for t in retained_graph_templates() if t.name == parent)
    graph = Chem.RWMol()
    atoms = {a.locant: graph.AddAtom(Chem.Atom(a.symbol)) for a in template.atoms}
    model = retained_template_parent_bond_model(template, atoms)
    for (left, right), order in model.allowed_kekule_assignments[0].orders:
        graph.AddBond(left, right, Chem.BondType.DOUBLE if order == 2 else Chem.BondType.SINGLE)
    bridge = [graph.AddAtom(Chem.Atom("C")) for _ in range(length)]
    path = [atoms["2"], *bridge, atoms["5"]]
    for i, (left, right) in enumerate(zip(path, path[1:])):
        graph.AddBond(left, right, Chem.BondType.DOUBLE if i == 2 else Chem.BondType.SINGLE)
    Chem.SanitizeMol(graph)
    return graph.GetMol()


@pytest.mark.opsin
@pytest.mark.parametrize("parent", ("naphthalene", "quinoline"))
@pytest.mark.parametrize("length", (5, 8, 12))
def test_long_bridge_double_bond_roundtrips_on_carbon_and_hetero_parents(parent, length):
    if not opsin_available():
        pytest.skip("OPSIN and Java are required")
    graph = _graph(parent, length)
    original = Chem.MolToSmiles(graph)
    for order in (list(range(graph.GetNumAtoms())), list(reversed(range(graph.GetNumAtoms())))):
        result = name_mol(Chem.RenumberAtoms(graph, order), include_trace=True)
        assert result.error is None
        assert result.parent_nomenclature == "bridged_fusion"
        assert "didehydro" in result.name
        check = verify_with_opsin(result.name, original, standardize_smiles=False)
        assert check.ok, check.to_dict()


@pytest.mark.opsin
def test_substituted_stereogenic_octeno_bridge_uses_completed_system_locants():
    if not opsin_available():
        pytest.skip("OPSIN and Java are required")
    graph = Chem.MolFromSmiles("C=C1CC[C@@H]2CCC3=C(C2(C)C)[C@](O)(C/C(C)=C\\[C@@H](O)C1)N(CC)C3=O")
    # The graph-built homologues above exercise arbitrary lengths; this
    # derivative additionally checks nonaromatic parent and bond stereochemistry.
    assert graph is not None
    result = name_mol(graph, include_trace=True)
    assert result.error is None
    assert result.parent_nomenclature == "bridged_fusion"
    assert "13,14-didehydro-1,6-octano" in result.name
    check = verify_with_opsin(result.name, Chem.MolToSmiles(graph), standardize_smiles=False)
    assert check.ok, check.to_dict()
