"""Oxo-added H constrains the matching domain before bridge hydro is proved."""

import pytest
from rdkit import Chem

from openclatura import name_mol, opsin_available, verify_with_opsin
from openclatura.retained_fused_templates import retained_graph_templates


def _graph(sidechain_length):
    template = next(t for t in retained_graph_templates() if t.name == "anthracene")
    graph = Chem.RWMol()
    atoms = {atom.locant: graph.AddAtom(Chem.Atom(atom.symbol)) for atom in template.atoms}
    doubles = {frozenset(("4a", "9a")), frozenset(("6", "7"))}
    for bond in template.bonds:
        graph.AddBond(
            *(atoms[locant] for locant in bond.locants),
            Chem.BondType.DOUBLE if frozenset(bond.locants) in doubles else Chem.BondType.SINGLE,
        )
    for left, right in (("1", "4"), ("9", "10")):
        bridge = graph.AddAtom(Chem.Atom("C"))
        graph.AddBond(atoms[left], bridge, Chem.BondType.SINGLE)
        graph.AddBond(bridge, atoms[right], Chem.BondType.SINGLE)
    for locant in ("5", "8"):
        oxygen = graph.AddAtom(Chem.Atom("O"))
        graph.AddBond(atoms[locant], oxygen, Chem.BondType.DOUBLE)
    last = atoms["2"]
    for _ in range(sidechain_length):
        carbon = graph.AddAtom(Chem.Atom("C"))
        graph.AddBond(last, carbon, Chem.BondType.SINGLE)
        last = carbon
    for locant in ("1", "4", "8a", "9", "10", "10a"):
        graph.GetAtomWithIdx(atoms[locant]).SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
    Chem.SanitizeMol(graph)
    return graph.GetMol()


@pytest.mark.opsin
@pytest.mark.parametrize("sidechain_length", (0, 1, 2))
def test_oxo_endpoint_constraints_precede_alternating_pi_proof(sidechain_length):
    if not opsin_available():
        pytest.skip("OPSIN and Java are required")
    graph = _graph(sidechain_length)
    expected = Chem.MolToSmiles(graph)
    for order in (list(range(graph.GetNumAtoms())), list(reversed(range(graph.GetNumAtoms())))):
        result = name_mol(Chem.RenumberAtoms(graph, order), include_trace=True)
        assert result.error is None
        assert result.parent_nomenclature == "bridged_fusion"
        assert "(8aH,10aH)" in result.name
        check = verify_with_opsin(result.name, expected, standardize_smiles=False)
        assert check.ok, check.to_dict()
