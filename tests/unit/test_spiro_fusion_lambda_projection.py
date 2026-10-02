"""Fusion lambda descriptors use the completed spiro locant namespace."""

import random
from copy import deepcopy
from dataclasses import replace

import pytest
from rdkit import Chem

from openclatura import name_mol, opsin_available, verify_with_opsin
from openclatura.assembly_parts import AssemblyParts, SubstituentItem
from openclatura.assembly_spiro import split_spiro_substituents

SMILES = "[C-]#[N+]c1cccc(-c2ccc3c(c2)C2(CC(c4ccccc4)S3(=O)=O)N=C(N)N(C)C2=O)c1"
EXPECTED = (
    "2-amino-6'-(3-isocyanophenyl)-1-methyl-1',1'-dioxo-2'-phenyl-"
    "1,4-dihydro-5H-1'lambda^6-spiro[imidazole-4,4'-(3,4-dihydro-2H-benzo[b]thiine)]-5-one"
)


def _permuted(graph, seed):
    order = list(range(graph.GetNumAtoms()))
    if seed is None:
        return graph
    random.Random(seed).shuffle(order)
    graph = Chem.RenumberAtoms(graph, order)
    rebuilt = Chem.RWMol()
    for atom in graph.GetAtoms():
        rebuilt.AddAtom(Chem.Atom(atom))
    for bond in reversed(list(graph.GetBonds())):
        rebuilt.AddBond(bond.GetEndAtomIdx(), bond.GetBeginAtomIdx(), bond.GetBondType())
    Chem.SanitizeMol(rebuilt)
    return rebuilt.GetMol()


def _exact(name, graph):
    if not opsin_available():
        pytest.skip("OPSIN is unavailable")
    result = verify_with_opsin(name, Chem.MolToSmiles(graph), standardize_smiles=False)
    assert result.status == "matched", result
    assert result.canonical_original == result.canonical_roundtrip


@pytest.mark.opsin
@pytest.mark.parametrize("seed", [None, 0, 17])
@pytest.mark.parametrize("mode", [None, "general"])
def test_2579_default_fusion_lambda_scope_exact_under_graph_permutations(seed, mode):
    graph = _permuted(Chem.MolFromSmiles(SMILES), seed)
    result = name_mol(graph, **({} if mode is None else {"fusion_mode": mode}))
    assert result.error is None
    assert result.name == EXPECTED
    _exact(result.name, graph)


@pytest.mark.opsin
@pytest.mark.parametrize("symbol", ["S", "Se"])
@pytest.mark.parametrize("oxo_count", [1, 2])
@pytest.mark.parametrize("seed", [None, 17])
def test_graph_modified_chalcogen_oxides_keep_fusion_and_correct_lambda_scope(symbol, oxo_count, seed):
    graph = Chem.RWMol(Chem.MolFromSmiles(SMILES))
    center = next(atom for atom in graph.GetAtoms() if atom.GetSymbol() == "S")
    center.SetAtomicNum(Chem.GetPeriodicTable().GetAtomicNumber(symbol))
    if oxo_count == 1:
        oxygen = next(atom.GetIdx() for atom in center.GetNeighbors() if atom.GetSymbol() == "O")
        graph.RemoveAtom(oxygen)
    Chem.SanitizeMol(graph)
    graph = _permuted(graph.GetMol(), seed)
    result = name_mol(graph)
    assert result.error is None
    assert f"1'lambda^{2 + 2 * oxo_count}-spiro[" in result.name
    assert "benzo[b]" in result.name
    assert "bicyclo[" not in result.name
    _exact(result.name, graph)


def test_projection_keeps_local_graph_proof_and_hoists_only_typed_lambda(monkeypatch):
    import openclatura.assembly_spiro as assembly

    original = assembly.spiro_assembly_from_parts
    captured = []

    def capture(parts, junction_locant, **kwargs):
        before = deepcopy(parts)
        side = original(parts, junction_locant, **kwargs)
        assert parts == before
        if side is not None and parts.parent_hydride is not None and parts.parent_hydride.is_systematic_fusion:
            captured.append(side)
        return side

    def reject_parse(*args, **kwargs):
        pytest.fail("typed fusion lambda projection must not parse rendered prefix names")

    monkeypatch.setattr(assembly, "spiro_assembly_from_parts", capture)
    monkeypatch.setattr(assembly, "_split_side_prefix_run", reject_parse)
    result = name_mol(Chem.MolFromSmiles(SMILES))
    assert result.name == EXPECTED
    side = next(item for item in captured if item.side_parts.parent_hydride.fusion_plan.lambda_descriptors)
    parts = side.side_parts
    plan = parts.parent_hydride.fusion_plan
    assert plan.audit.confirmed
    (descriptor,) = plan.lambda_descriptors
    assert str(descriptor.locant) == "1"
    assert parts.parent_atom_ids_by_locant["1"] == descriptor.atom_id
    assert descriptor.bonding_number == 6
    assert "1lambda^6-" in plan.rendered_base_name
    assert side.side_parent_name == "3,4-dihydro-2H-benzo[b]thiine"
    assert side.side_prefixes[-1] == "1'lambda^6-"
    assert len(plan.derivative_state.oxo_operations) == 2
    assert {item.parent_atom_id for item in plan.derivative_state.oxo_operations} == {descriptor.atom_id}
    assert not plan.derivative_state.added_hydrogen_operations
    local_h = deepcopy(parts.hydro_operations)
    wrapper = AssemblyParts(parent_length=5, substituents=[SubstituentItem("", ["4"], spiro=side)])
    (projected,) = split_spiro_substituents(wrapper)
    assert projected.side_prefixes == ("1'lambda^6-",)
    assert parts.hydro_operations == local_h
    inconsistent = deepcopy(parts)
    inconsistent.parent_hydride = replace(parts.parent_hydride, parent_name="unproved parent spelling")
    assert original(inconsistent, side.side_locant) is None


@pytest.mark.parametrize("seed", [None, 0, 17])
def test_rendered_tokens_keep_primed_sulfur_oxo_and_unprimed_suffix_ownership(seed):
    graph = _permuted(Chem.MolFromSmiles(SMILES), seed)
    result = name_mol(graph, include_trace=True, token_debug=True)
    assert result.error is None
    assert result.name == EXPECTED
    assembly = [step for step in result.decisions if step.decision == "assembled component name"][-1]
    assert assembly.data["name"] == result.name
    center = next(atom for atom in graph.GetAtoms() if atom.GetSymbol() == "S")
    oxygen_ids = {atom.GetIdx() for atom in center.GetNeighbors() if atom.GetSymbol() == "O"}
    oxo_bonds = {graph.GetBondBetweenAtoms(center.GetIdx(), atom).GetIdx() for atom in oxygen_ids}
    bindings = assembly.data["name_atom_bindings"]
    (lambda_binding,) = [binding for binding in bindings if binding["role"] == "fusion_lambda_descriptor"]
    assert lambda_binding["stage"] == "replacement"
    assert lambda_binding["atoms"] == [center.GetIdx()]
    assert lambda_binding["locants"] == ["1'"]
    assert {token["text"] for token in lambda_binding["emitted_tokens"]} == {"1", "1'", "lambda^6"}
    spans = assembly.data["name_token_spans"]
    lambda_spans = [span for span in spans if span["grammar_role"] == "fusion_lambda_descriptor"]
    assert {span["text"] for span in lambda_spans} == {"1", "lambda^6"}
    for span in lambda_spans:
        assert span["atoms"] == [center.GetIdx()]
        assert span["locants"] == ["1'"]
        assert span["ownership"] == "exact"
        assert span["source"] == "spiro_renderer"
        assert result.name[span["start"] : span["end"]] == span["text"]
    (oxo,) = [binding for binding in bindings if binding["term"] == "oxo"]
    assert oxo["locants"] == ["1'", "1'"]
    assert set(oxo["atoms"]) == oxygen_ids
    # The reader normalizes bond IDs; compare suffix and prefix ownership
    # with its graph rather than assuming RDKit insertion numbering.
    from openclatura.graph_io import read_rdkit_mol

    mol = read_rdkit_mol(graph)
    assert len(oxo_bonds) == 2
    assert set(oxo["bonds"]) == {mol.get_bond(center.GetIdx(), atom).idx for atom in oxygen_ids}
    (suffix,) = [binding for binding in bindings if binding["stage"] == "suffix"]
    assert suffix["role"] == "ketone"
    assert suffix["locants"] == ["5"]
    assert set(suffix["atoms"]).isdisjoint({center.GetIdx(), *oxygen_ids})
    assert len(suffix["bonds"]) == 1
    bond = mol.bonds[suffix["bonds"][0]]
    assert bond.order == 2
    assert {bond.u, bond.v} == set(suffix["atoms"])
    assert {mol.atoms[atom].symbol for atom in suffix["atoms"]} == {"C", "O"}


def test_raw_and_structured_assembly_agree_on_fusion_lambda_projection(monkeypatch):
    import openclatura.namer as namer
    from openclatura.assembler import assemble_name_raw, assemble_name_result

    original = namer._assemble_parent_name
    captured = []

    def capture(mol, parts, *args, **kwargs):
        if len(mol.atoms) == Chem.MolFromSmiles(SMILES).GetNumAtoms():
            captured.append(deepcopy(parts))
        return original(mol, parts, *args, **kwargs)

    monkeypatch.setattr(namer, "_assemble_parent_name", capture)
    result = name_mol(Chem.MolFromSmiles(SMILES))
    assert result.name == EXPECTED
    parts = next(parts for parts in captured if any(item.spiro is not None for item in parts.substituents))
    assert assemble_name_raw(deepcopy(parts)) == EXPECTED
    rendered = assemble_name_result(deepcopy(parts))
    assert rendered.text == EXPECTED
    (binding,) = [binding for binding in rendered.bindings if binding.role == "fusion_lambda_descriptor"]
    assert binding.locants == ("1'",)
    assert len(binding.atom_ids) == 1
    assert any(token.text == "lambda^6" for token in binding.emitted_tokens)
