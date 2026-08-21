"""Configurable constitutional locant-elision regressions."""

from openclatura import DEFAULT_NAMING_ENGINE, NamingRequest, name_many, name_smiles
from openclatura.assembly_parts import AssemblyParts, SubstituentItem
from openclatura.locant_elision import apply_redundant_locant_elision


def _name(smiles: str, *, trace: bool = False):
    return DEFAULT_NAMING_ENGINE.run(
        NamingRequest(
            smiles=smiles,
            include_trace=trace,
            omit_redundant_locants=True,
        )
    )


def _name_without_elision(smiles: str):
    return DEFAULT_NAMING_ENGINE.run(NamingRequest(smiles=smiles, omit_redundant_locants=False))


def test_hexamethylbenzene_existing_elision_is_preserved_by_default_policy():
    smiles = "Cc1c(C)c(C)c(C)c(C)c1C"

    assert name_smiles(smiles) == "hexamethylbenzene"
    assert _name(smiles).name == "hexamethylbenzene"


def test_broader_elision_defaults_on_and_can_be_disabled():
    assert NamingRequest().omit_redundant_locants is True
    assert name_smiles("C1=CCCCC1") == "cyclohexene"
    assert _name_without_elision("C1=CCCCC1").name == "cyclohex-1-ene"


def test_batch_api_forwards_default_and_opt_out_policy():
    assert name_many(["C1=CCCCC1"])[0].name == "cyclohexene"
    assert name_many(["C1=CCCCC1"], omit_redundant_locants=False)[0].name == "cyclohex-1-ene"


def test_trace_records_symmetry_proof_without_inventing_rule_id():
    result = _name("C1=CCCCC1", trace=True)
    assembly = [step for step in result.decisions if step.decision == "assembled component name"][-1]

    assert assembly.data["locant_elisions"] == [
        {
            "category": "unsaturation",
            "key": "double",
            "locants": ["1-2"],
            "reason": "placement is unique under exact parent-graph symmetry",
        }
    ]
    assert not any("rule" in decision for decision in assembly.data["locant_elisions"])


def test_distinct_constitutional_placements_keep_locants():
    assert _name("Cc1c(C)c(C)ccc1").name == "1,2,3-trimethylbenzene"
    assert _name("Cc1cccc(C)c1").name == "1,3-dimethylbenzene"
    assert _name("CC=CC").name == "but-2-ene"
    assert _name("CCC(O)").name == "propan-1-ol"


def test_simple_ring_and_chain_unsaturation_can_elide_unique_locants():
    assert _name("C1=CCCCC1").name == "cyclohexene"
    assert _name("CC=C").name == "propene"


def test_nonconstitutional_and_unsupported_parent_locants_are_unchanged():
    unchanged = {
        "C/C=C\\C": "(2Z)-but-2-ene",
        "[N-]=[N+]=P1=CC=CC=C1": "1-diazo-1lambda^5-phosphacyclohexa-1,3,5-triene",
        "C1CC2CCC1C2": "bicyclo[2.2.1]heptane",
    }
    for smiles, expected in unchanged.items():
        assert name_smiles(smiles) == expected
        assert _name(smiles).name == expected


def test_equivalent_smiles_atom_orderings_are_deterministic():
    names = {
        _name(smiles).name
        for smiles in (
            "Cc1c(C)c(C)c(C)c(C)c1C",
            "c1(C)c(C)c(C)c(C)c(C)c1C",
        )
    }
    assert names == {"hexamethylbenzene"}


def test_candidate_limit_falls_back_to_printing_locants():
    parts = AssemblyParts(parent_length=20, omit_redundant_locants=True)
    for index in range(1, 21):
        locant = str(index)
        parts.parent_atom_symbols_by_locant[locant] = "C"
        parts.parent_atom_charges_by_locant[locant] = 0
        if index > 1:
            edge = (str(index - 1), locant)
            parts.parent_bond_orders_by_locants[edge] = 1
    parts.substituents = [SubstituentItem(name="methyl", locants=[str(index) for index in range(1, 11)])]

    apply_redundant_locant_elision(parts)

    assert parts.elided_substituent_locants == set()
    assert parts.locant_elision_decisions == []
