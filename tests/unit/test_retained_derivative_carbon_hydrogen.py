"""Retained oxo derivatives locate carbon H from an exact parent bond proof."""

import random
from copy import deepcopy
from dataclasses import replace

import pytest
from rdkit import Chem

from openclatura import name_mol, opsin_available, verify_with_opsin
from openclatura.additive import _recast_ring_ketone_hydrogens
from openclatura.assembly_parts import AssemblyParts, PrincipalGroupItem
from openclatura.fusion.numbering import retained_template_parent_bond_model
from openclatura.graph_io import read_rdkit_mol
from openclatura.name_operations import HydroOperation
from openclatura.retained_derivative_hydrogen import prove_retained_oxo_carbon_hydrogen
from openclatura.retained_fused_templates import match_retained_graph_template_maps, retained_graph_templates

REPORT_SMILES = "COC1=Nc2ccccc2C(=O)N(Cc2ccccc2)C1"
EXPECTED = "4-benzyl-2-methoxy-3H-1,4-benzodiazepin-5(4H)-one"


def _template(name="1H-1,4-benzodiazepine"):
    return next(t for t in retained_graph_templates() if t.name == name)


def _graph(parent="1H-1,4-benzodiazepine", substitution=None):
    template = _template(parent)
    graph = Chem.RWMol()
    mapping = {a.locant: graph.AddAtom(Chem.Atom(a.symbol)) for a in template.atoms}
    oxo, nitrogen = ("5", "4") if parent == "1H-1,4-benzodiazepine" else ("4", "5")
    model = retained_template_parent_bond_model(template, mapping, indicated_h=("3",))
    consumed = tuple(sorted((mapping[oxo], mapping[nitrogen])))
    assignment = next(a for a in model.allowed_kekule_assignments if dict(a.orders)[consumed] == 2)
    for edge, order in assignment.orders:
        graph.AddBond(*edge, Chem.BondType.DOUBLE if order == 2 and edge != consumed else Chem.BondType.SINGLE)
    oxygen = graph.AddAtom(Chem.Atom("O"))
    graph.AddBond(mapping[oxo], oxygen, Chem.BondType.DOUBLE)
    branches = {
        "N-methyl": [(nitrogen, "C")],
        "fluoro": [("6", "F")],
        "C-methyl": [("3", "C")],
        "C-dimethyl": [("3", "C"), ("3", "C")],
        "methoxy": [("2", "O")],
    }
    for locant, symbol in branches.get(substitution, []):
        added = graph.AddAtom(Chem.Atom(symbol))
        graph.AddBond(mapping[locant], added, Chem.BondType.SINGLE)
        if substitution == "methoxy":
            methyl = graph.AddAtom(Chem.Atom("C"))
            graph.AddBond(added, methyl, Chem.BondType.SINGLE)
    Chem.SanitizeMol(graph)
    return graph.GetMol(), template, mapping


def _orders(graph):
    original = list(range(graph.GetNumAtoms()))
    shuffled = original.copy()
    random.Random(2737).shuffle(shuffled)
    return original, original[::-1], shuffled


@pytest.mark.parametrize("branch", (None, "1", "4", "6"))
def test_declared_parent_hydrogen_and_oxo_added_hydrogen_have_separate_proofs(branch):
    template = _template("1H-pyrrolo[3,2-b]pyridine")
    graph = Chem.RWMol()
    mapping = {a.locant: graph.AddAtom(Chem.Atom(a.symbol)) for a in template.atoms}
    model = retained_template_parent_bond_model(template, mapping, indicated_h=("1",))
    consumed = tuple(sorted((mapping["4"], mapping["5"])))
    assignment = next(a for a in model.allowed_kekule_assignments if dict(a.orders)[consumed] == 2)
    for edge, order in assignment.orders:
        graph.AddBond(*edge, Chem.BondType.DOUBLE if order == 2 and edge != consumed else Chem.BondType.SINGLE)
    oxygen = graph.AddAtom(Chem.Atom("O"))
    graph.AddBond(mapping["5"], oxygen, Chem.BondType.DOUBLE)
    if branch:
        methyl = graph.AddAtom(Chem.Atom("C"))
        graph.AddBond(mapping[branch], methyl, Chem.BondType.SINGLE)
    Chem.SanitizeMol(graph)
    for order in _orders(graph):
        mol = read_rdkit_mol(Chem.RenumberAtoms(graph, order))
        locants = {loc: order.index(atom) for loc, atom in mapping.items()}
        proof = prove_retained_oxo_carbon_hydrogen(mol, template, locants, declared_indicated_h=("1",))
        assert proof is not None
        assert proof.indicated_h == ("1",)
        assert proof.added_hydrogen.locants == ("4",)
        assert proof.added_hydrogen.atom_ids == (locants["4"],)
        assert prove_retained_oxo_carbon_hydrogen(mol, template, locants, declared_indicated_h=("99",)) is None


@pytest.mark.opsin
def test_declared_fused_donor_and_oxo_suffix_roundtrip():
    if not opsin_available():
        pytest.skip("OPSIN and Java are required")
    graph = Chem.MolFromSmiles("O=c1ccc2c(ccn2Cc2nnc3ccc(-c4ccccc4)nn23)[nH]1")
    original = Chem.MolToSmiles(graph)
    for order in _orders(graph):
        result = name_mol(Chem.RenumberAtoms(graph, order), include_trace=True)
        assert result.error is None
        assert "1H-pyrrolo[3,2-b]pyridin-5(4H)-one" in result.name
        assert verify_with_opsin(result.name, original, standardize_smiles=False).ok


@pytest.mark.parametrize("parent", ["1H-1,4-benzodiazepine", "1H-1,5-benzodiazepine"])
@pytest.mark.parametrize("substitution", [None, "N-methyl", "fluoro", "C-methyl", "C-dimethyl", "methoxy"])
def test_graph_proof_preserves_capacity_and_reconstructs_all_bonds(parent, substitution):
    graph, template, mapping = _graph(parent, substitution)
    for order in _orders(graph):
        mol = read_rdkit_mol(Chem.RenumberAtoms(graph, order))
        locants = {loc: order.index(index) for loc, index in mapping.items()}
        before = deepcopy((mol.atoms, mol.bonds))
        proof = prove_retained_oxo_carbon_hydrogen(mol, template, locants)
        assert proof is not None
        assert proof.indicated_h == ("3",)
        assert proof.atom_ids == (locants["3"],)
        assert proof.model.maximum_non_cumulative_double_bonds == template.mancude_double_bonds
        expected = dict(proof.assignment.orders)
        observed = {
            tuple(sorted((b.u, b.v))): b.order
            for b in mol.bonds.values()
            if b.u in locants.values() and b.v in locants.values()
        }
        changed = [edge for edge in expected if expected[edge] != observed[edge]]
        assert len(changed) == len(proof.oxo_atom_ids)
        assert all(
            expected[edge] == 2 and observed[edge] == 1 and proof.oxo_atom_ids.intersection(edge) for edge in changed
        )
        assert (mol.atoms, mol.bonds) == before
        matches = match_retained_graph_template_maps(mol, set(locants.values()), template, allow_nonaromatic=True)
        assert matches
        # Topological component recognition remains separate from derivative H.
        assert all(not m.indicated_h for m in matches)


@pytest.mark.parametrize(
    "change", ["missing_oxo", "extra_hydro", "charged", "wrong_element", "missing_map", "duplicate_map", "capacity"]
)
def test_unproved_derivative_hydrogen_is_rejected(change):
    graph, template, mapping = _graph()
    mol = read_rdkit_mol(graph)
    if change == "missing_oxo":
        oxygen = next(i for i in mol.atoms if i not in mapping.values())
        bond = mol.get_bond(mapping["5"], oxygen)
        mol.update_bond(bond.idx, order=1)
        mol.update_atom(oxygen, total_h_count=1)
        mol.update_atom(mapping["5"], total_h_count=1)
    elif change == "extra_hydro":
        bond = mol.get_bond(mapping["1"], mapping["2"])
        mol.update_bond(bond.idx, order=1)
        for locant in ("1", "2"):
            atom = mol.atoms[mapping[locant]]
            mol.update_atom(atom.idx, total_h_count=atom.total_h_count + 1)
    elif change == "charged":
        mol.update_atom(mapping["4"], charge=1)
    elif change == "wrong_element":
        mol.update_atom(mapping["1"], symbol="C")
    elif change == "missing_map":
        mapping.pop("3")
    elif change == "duplicate_map":
        mapping["3"] = mapping["2"]
    else:
        template = replace(template, mancude_double_bonds=99)
    assert prove_retained_oxo_carbon_hydrogen(mol, template, mapping) is None
    if change in {"missing_oxo", "extra_hydro"}:
        matches = match_retained_graph_template_maps(mol, set(mapping.values()), template, allow_nonaromatic=True)
        assert matches
        assert all(not match.indicated_h for match in matches)


def test_existing_nh_recast_is_preserved():
    graph, _, mapping = _graph()
    mol = read_rdkit_mol(graph)
    site = mapping["4"]
    operation = HydroOperation(
        key="indicated_hydrogen",
        reason="test parent H",
        locants=("4",),
        atom_ids=(site,),
        operation_kind="indicated_hydrogen",
    )
    parts = AssemblyParts(
        parent_length=len(mapping),
        principal_group=PrincipalGroupItem("ketone", ["5"]),
        hydro_operations=[operation],
    )
    inverse = {index: locant for locant, index in mapping.items()}
    _recast_ring_ketone_hydrogens(mol, parts, list(mapping.values()), inverse.__getitem__)
    assert parts.hydro_operations[0].key == "added_hydrogen"
    assert parts.hydro_operations[0].atom_ids == (site,)


def test_proved_carbon_hydrogen_bypasses_count_based_recast(monkeypatch):
    import openclatura.additive as additive

    def reject_recast(*args, **kwargs):
        pytest.fail("proved intrinsic carbon H reached the NH count recast")

    monkeypatch.setattr(additive, "_recast_ring_ketone_hydrogens", reject_recast)
    result = name_mol(Chem.MolFromSmiles(REPORT_SMILES), verify_opsin=False)
    assert result.error is None, result.error
    assert result.name == EXPECTED


@pytest.mark.opsin
def test_report2737_exact_opsin_and_permutations():
    if not opsin_available():
        pytest.skip("OPSIN and Java are required")
    graph = Chem.MolFromSmiles(REPORT_SMILES)
    original = Chem.MolToSmiles(graph)
    for order in _orders(graph):
        result = name_mol(Chem.RenumberAtoms(graph, order), include_trace=True, verify_opsin=False)
        assert result.error is None, result.error
        assert result.name == EXPECTED
        check = verify_with_opsin(result.name, original, standardize_smiles=False)
        assert check.ok, check.to_dict()
        assert check.canonical_roundtrip == original


@pytest.mark.opsin
@pytest.mark.parametrize("parent", ["1H-1,4-benzodiazepine", "1H-1,5-benzodiazepine"])
@pytest.mark.parametrize("substitution", [None, "N-methyl", "fluoro", "C-methyl", "C-dimethyl", "methoxy"])
def test_graph_built_derivatives_exact_opsin_and_permutations(parent, substitution):
    if not opsin_available():
        pytest.skip("OPSIN and Java are required")
    graph, _, _ = _graph(parent, substitution)
    original = Chem.MolToSmiles(graph)
    names = []
    for order in _orders(graph):
        result = name_mol(Chem.RenumberAtoms(graph, order), include_trace=True, verify_opsin=False)
        assert result.error is None, result.error
        assert "benzodiazepin" in result.name
        check = verify_with_opsin(result.name, original, standardize_smiles=False)
        assert check.ok, check.to_dict()
        assert check.canonical_roundtrip == original, check.to_dict()
        names.append(result.name)
    assert len(set(names)) == 1
