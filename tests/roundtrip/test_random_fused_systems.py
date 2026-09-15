"""Strict two-seed fusion stress suite; run with -m 'slow and opsin'.

Inputs are generated from graphs, never selected according to naming success.
Failures expose coverage gaps rather than being xfailed or silently discarded.
Each naming/OPSIN call is isolated to bound pathological search/runtime costs.
"""

import json
import subprocess
import sys
from collections import Counter

import pytest
from rdkit import Chem

from openclatura import opsin_available
from roundtrip.random_fusion_helpers import random_fusion_cases, validate_case

CASES = random_fusion_cases()
ADDITIONAL_CASES = random_fusion_cases(count=30, seed=20260908)
WORKER = """
import json, sys
from rdkit import Chem
from openclatura import name_mol, verify_with_opsin
mol = Chem.Mol(bytes.fromhex(sys.argv[1]))
result = name_mol(mol, include_trace=True)
check = (verify_with_opsin(result.name, Chem.MolToSmiles(mol), standardize_smiles=False)
         if result.name else None)
print(json.dumps(dict(name=result.name, error=str(result.error) if result.error else None,
                     parent_nomenclature=result.parent_nomenclature,
                     opsin=check.to_dict() if check is not None else None)))
"""


def test_random_fusion_generator_is_neutral_unique_and_reproducible():
    assert len(CASES) == len({case.smiles for case in CASES}) == 100
    assert Counter(len(case.faces) for case in CASES) == {5: 17, 6: 17, 7: 17, 8: 17, 9: 16, 10: 16}
    assert CASES == random_fusion_cases()
    assert {a.GetSymbol() for case in CASES for a in Chem.Mol(case.binary).GetAtoms()} == {"C", "N", "O", "S"}
    assert {len(face) for case in CASES for face in case.faces} == {5, 6}
    # Exercise branched face trees as well as linear chains.
    assert any(
        sum(len(set(face) & set(other)) == 2 for other in case.faces if other != face) >= 3
        for case in CASES
        for face in case.faces
    )
    for case in CASES:
        validate_case(case)


def test_second_seed_adds_independent_neutral_graphs():
    assert ADDITIONAL_CASES == random_fusion_cases(count=30, seed=20260908)
    assert len({case.smiles for case in ADDITIONAL_CASES}) == 30
    assert not {case.smiles for case in CASES}.intersection(case.smiles for case in ADDITIONAL_CASES)
    for case in ADDITIONAL_CASES:
        validate_case(case)


@pytest.mark.slow
@pytest.mark.opsin
@pytest.mark.parametrize(
    "case",
    [*CASES, *(pytest.param(case, id=f"seed-20260908-{case.id}") for case in ADDITIONAL_CASES)],
    ids=lambda case: case.id,
)
def test_random_fused_system_uses_fusion_and_roundtrips(case, record_property):
    if not opsin_available():
        pytest.skip("OPSIN requires py2opsin and Java")
    record_property("input_smiles", case.smiles)
    record_property("ring_count", len(case.faces))
    try:
        completed = subprocess.run(
            [sys.executable, "-c", WORKER, case.binary.hex()],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(f"{case.id}: naming/OPSIN exceeded 30 seconds; input={case.smiles}")
    assert completed.returncode == 0, f"{case.smiles}\n{completed.stdout}\n{completed.stderr}"
    outcome = json.loads(completed.stdout.strip().splitlines()[-1])
    record_property("naming_outcome", json.dumps(outcome))
    errors = []
    if not outcome["name"] or outcome["error"]:
        errors.append("no name generated")
    if outcome["parent_nomenclature"] not in {"systematic_fusion", "skeletal_replacement_fusion", "bridged_fusion"}:
        errors.append("fusion nomenclature not used")
    if not outcome["opsin"] or outcome["opsin"]["status"] != "matched":
        errors.append("OPSIN exact roundtrip failed")
    assert not errors, f"{case.smiles}\n{'; '.join(errors)}\n{json.dumps(outcome, indent=2)}"
