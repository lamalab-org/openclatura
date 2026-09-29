import importlib.util
import sys
from pathlib import Path


def _load_builder():
    path = Path(__file__).parents[2] / "benchmarks" / "build_pubchem_hf_corpus.py"
    spec = importlib.util.spec_from_file_location("build_pubchem_hf_corpus", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BUILDER = _load_builder()


class FakeDataset:
    def __init__(self, smiles):
        self.smiles = smiles

    def __len__(self):
        return len(self.smiles)

    def select(self, indices):
        return {"smiles": [self.smiles[index] for index in indices]}


def test_sample_candidates_is_deterministic_unique_and_excludes_canonical_smiles():
    dataset = FakeDataset(["CCO", "OCC", "CCN", "CCC", "not-smiles", "CCCl"])
    excluded = {"CCN"}

    first = BUILDER.sample_candidates(dataset, size=3, seed=17, excluded=excluded)
    second = BUILDER.sample_candidates(dataset, size=3, seed=17, excluded=excluded)

    assert first == second
    assert {row["canonical_smiles"] for row in first}.isdisjoint(excluded)
    assert len({row["canonical_smiles"] for row in first}) == 3
