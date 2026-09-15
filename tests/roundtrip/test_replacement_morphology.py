"""Graph-derived regressions for replacement vowels before octacyclo.

These check fallback correctness, not whether fusion nomenclature is selected.
The independent fusion stress suite retains that stricter requirement.
"""

import pytest
from rdkit import Chem

from openclatura import name_mol, opsin_available, verify_with_opsin
from roundtrip.random_fusion_helpers import random_fusion_cases, validate_case

EIGHT_RING_CASES = tuple(case for case in random_fusion_cases() if len(case.faces) == 8)


@pytest.mark.opsin
@pytest.mark.parametrize("case", EIGHT_RING_CASES, ids=lambda case: case.id)
def test_eight_ring_replacement_roundtrips_exact_graph(case):
    if not opsin_available():
        pytest.skip("OPSIN requires py2opsin and Java")
    validate_case(case)
    mol = Chem.Mol(case.binary)
    result = name_mol(mol)
    assert result.name and not result.error, result
    check = verify_with_opsin(result.name, Chem.MolToSmiles(mol), standardize_smiles=False)
    assert check.status == "matched", check.to_dict()
