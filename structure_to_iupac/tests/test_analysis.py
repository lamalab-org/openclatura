from structure_to_iupac import TracePhase, analyze_smiles, name_smiles
from structure_to_iupac.chains import find_all_carbon_paths, find_ring_systems
from structure_to_iupac.namer import read_smiles
from structure_to_iupac.parent_selection import ParentSelection, select_principal_parent
from structure_to_iupac.perception import perceive_groups


def test_name_smiles_stays_plain_fast_api():
    assert name_smiles("CCO") == "ethanol"


def test_functional_groups_carry_metadata_and_graph_bindings():
    mol = read_smiles("CC(=O)O")
    groups = perceive_groups(mol)

    acid = next(group for group in groups if group.key == "carboxylic_acid")
    assert acid.suffix == "oic acid"
    assert acid.prefix == "carboxy"
    assert acid.seniority == 20
    assert acid.atom_ids == {1, 2, 3}
    assert acid.bond_ids == {2, 3}
    assert {binding.role for binding in acid.atom_bindings} == {
        "attachment",
        "characteristic_group",
        "full_group",
    }


def test_analyze_smiles_exposes_decision_trace():
    analysis = analyze_smiles("CCO")

    assert analysis.name == "ethanol"
    assert analysis.trace_segments
    phases = [step.phase for step in analysis.decisions]
    assert TracePhase.PARSE in phases
    assert TracePhase.PERCEPTION in phases
    assert TracePhase.PRIORITY in phases
    assert TracePhase.PARENT_SELECTION in phases
    assert TracePhase.NUMBERING in phases
    assert TracePhase.ASSEMBLY in phases


def test_parent_selection_has_named_shape():
    mol = read_smiles("CCO")
    selection = select_principal_parent(
        mol,
        find_all_carbon_paths(mol, exclude_atoms={2}),
        find_ring_systems(mol, exclude_atoms={2}),
        principal_carbons=[1],
    )

    assert isinstance(selection, ParentSelection)
    assert selection.primary_path == [0, 1]
    assert selection.atom_set == {0, 1}
    assert selection.is_ring is False
    assert selection.polycycle_descriptor is None


def test_public_api_golden_names():
    cases = {
        "": "",
        "C": "methane",
        "CC": "ethane",
        "CCC": "propane",
        "C=C": "eth-1-ene",
        "C#C": "eth-1-yne",
        "CCO": "ethanol",
        "CC(=O)O": "acetic acid",
        "C1CCCCC1": "cyclohexane",
        "[Na+].[Cl-]": "sodium chloride",
        "c1ccccc1O": "benzen-1-ol",
        "CC(=O)Oc1ccccc1C(=O)O": "2-(acetoxy)benzoic acid",
    }

    for smiles, expected in cases.items():
        assert name_smiles(smiles) == expected


def test_trace_segment_schema_for_functionalized_molecules():
    for smiles in ["CCO", "CC(=O)O", "CC(=O)Oc1ccccc1C(=O)O"]:
        analysis = analyze_smiles(smiles)
        assert analysis.trace_segments
        for segment in analysis.trace_segments:
            assert {"key", "label", "atoms", "bonds", "name_terms", "rule_hint"} <= set(segment)
            assert isinstance(segment["atoms"], list)
            assert isinstance(segment["bonds"], list)
            assert isinstance(segment["name_terms"], list)
