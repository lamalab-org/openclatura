"""External-pi replacement must retain construction-compatible numbering."""

import random

import pytest
from rdkit import Chem

from openclatura import name_mol, opsin_available
from openclatura.fusion import planner
from openclatura.fusion.model import FusionUnsupported
from openclatura.opsin_verify import verify_with_opsin


@pytest.mark.skipif(not opsin_available(), reason="OPSIN and Java are required")
@pytest.mark.parametrize("extend_sidechain", (False, True))
def test_external_pi_replacement_has_exact_public_roundtrip(extend_sidechain, monkeypatch):
    rejected = []
    plan_candidate = planner._plan_numbered_candidate

    def record_candidate(*args, **kwargs):
        result = plan_candidate(*args, **kwargs)
        if isinstance(result, FusionUnsupported):
            rejected.append(result.reason)
        return result

    monkeypatch.setattr(planner, "_plan_numbered_candidate", record_candidate)
    smiles = "C/C=C1/C(=O)O[C@@]23C=C[C@@H]4C[C@@H](O)O[C@@H](O[C@@H]12)[C@H]43"
    graph = Chem.MolFromSmiles(smiles)
    if extend_sidechain:
        editable = Chem.RWMol(graph)
        terminal = editable.AddAtom(Chem.Atom("C"))
        editable.AddBond(0, terminal, Chem.BondType.SINGLE)
        graph = editable.GetMol()
        Chem.SanitizeMol(graph)
        smiles = Chem.MolToSmiles(graph, isomericSmiles=True)
    original = list(range(graph.GetNumAtoms()))
    orders = [original, list(reversed(original))]
    for seed in (7, 73, 19119, 40352):
        shuffled = original.copy()
        random.Random(seed).shuffle(shuffled)
        orders.append(shuffled)
    names = set()
    for order in orders:
        rejected.clear()
        result = name_mol(Chem.RenumberAtoms(graph, order), include_trace=True, verify_self=True)
        assert result.error is None
        assert result.parent_nomenclature == "bridged_fusion"
        assert "tied fusion numberings identify different locant-labelled parent graphs" in rejected
        assert not result.self_audit.coverage.unnamed_atoms
        check = verify_with_opsin(result.name, smiles, standardize_smiles=False)
        assert check.status == "matched", check.to_dict()
        assert check.canonical_original == check.canonical_roundtrip
        names.add(result.name)
    assert len(names) == 1
