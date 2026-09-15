"""Bridge parents keep fixed hydro states separate from movable carbon H."""

import pytest
from rdkit import Chem

from openclatura import name_mol, opsin_available, verify_with_opsin
from openclatura.retained_fused_templates import retained_graph_templates


def _graph(bridge_length, oxo, methyls):
    template = next(t for t in retained_graph_templates() if t.name == "1H-indene")
    graph = Chem.RWMol()
    atoms = {a.locant: graph.AddAtom(Chem.Atom(a.symbol)) for a in template.atoms}
    doubles = {frozenset(("1", "2"))} if oxo else {frozenset(("2", "3")), frozenset(("5", "6"))}
    for bond in template.bonds:
        graph.AddBond(
            *(atoms[l] for l in bond.locants),
            Chem.BondType.DOUBLE if frozenset(bond.locants) in doubles else Chem.BondType.SINGLE,
        )
    last = atoms["4"]
    for _ in range(bridge_length):
        carbon = graph.AddAtom(Chem.Atom("C"))
        graph.AddBond(last, carbon, Chem.BondType.SINGLE)
        last = carbon
    graph.AddBond(last, atoms["7"], Chem.BondType.SINGLE)
    if oxo:
        oxygen = graph.AddAtom(Chem.Atom("O"))
        graph.AddBond(atoms["3"], oxygen, Chem.BondType.DOUBLE)
    for _ in range(methyls):
        carbon = graph.AddAtom(Chem.Atom("C"))
        graph.AddBond(atoms["5"] if oxo else atoms["1"], carbon, Chem.BondType.SINGLE)
    Chem.SanitizeMol(graph)
    return graph.GetMol()


@pytest.mark.opsin
@pytest.mark.parametrize("bridge_length", (1, 2))
@pytest.mark.parametrize("oxo", (False, True))
@pytest.mark.parametrize("methyls", (0, 2))
def test_bridged_parent_hydrogen_state_roundtrips_graph_variants(bridge_length, oxo, methyls):
    if not opsin_available():
        pytest.skip("OPSIN and Java are required")
    graph = _graph(bridge_length, oxo, methyls)
    expected = Chem.MolToSmiles(graph)
    for order in (list(range(graph.GetNumAtoms())), list(reversed(range(graph.GetNumAtoms())))):
        result = name_mol(Chem.RenumberAtoms(graph, order), include_trace=True)
        assert result.error is None
        assert result.parent_nomenclature == "bridged_fusion"
        check = verify_with_opsin(result.name, expected, standardize_smiles=False)
        assert check.ok, check.to_dict()


@pytest.mark.opsin
@pytest.mark.parametrize(
    "smiles",
    (
        "C=CCC1(CC=C)C=CC2C3C=CC(C3)C21",
        "CCc1ccc(-c2ccc(Cl)c(C(F)(F)F)c2)cc1C1=C(O)[C@H]2C3CCC(CC3)C2C1=O",
        "CC1=CC2C(C1=O)[C@]1(C)O[C@H]2C(=O)C12CC2",
    ),
)
def test_reported_bridged_hydrogen_and_oxo_composition(smiles):
    if not opsin_available():
        pytest.skip("OPSIN and Java are required")
    graph = Chem.MolFromSmiles(smiles)
    for order in (list(range(graph.GetNumAtoms())), list(reversed(range(graph.GetNumAtoms())))):
        result = name_mol(Chem.RenumberAtoms(graph, order), include_trace=True)
        assert result.error is None
        assert "cyclo[" not in result.name
        check = verify_with_opsin(result.name, smiles, standardize_smiles=False)
        assert check.ok, check.to_dict()
