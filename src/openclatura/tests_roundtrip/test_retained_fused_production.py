"""OPSIN-audited production tests for retained fused parent plans."""

from __future__ import annotations

import shutil
import warnings
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from itertools import combinations
from random import Random

import pytest

from openclatura import name_many
from openclatura.retained_fused_templates import retained_fused_graph_templates
from openclatura.utils import standardize_mol

try:
    import py2opsin
except Exception:  # pragma: no cover - optional test dependency
    py2opsin = None


PARENT_CASES = (
    ("phenazine", "phenazine"),
    ("1,4-phenanthroline", "1,4-phenanthroline"),
    ("1,5-phenanthroline", "1,5-phenanthroline"),
    ("1,6-phenanthroline", "1,6-phenanthroline"),
    ("1,7-phenanthroline", "1,7-phenanthroline"),
    ("1,8-phenanthroline", "1,8-phenanthroline"),
    ("1,9-phenanthroline", "1,9-phenanthroline"),
    ("1,10-phenanthroline", "1,10-phenanthroline"),
    ("2,7-phenanthroline", "2,7-phenanthroline"),
    ("2,8-phenanthroline", "2,8-phenanthroline"),
    ("3,5-phenanthroline", "3,5-phenanthroline"),
    ("3,6-phenanthroline", "3,6-phenanthroline"),
    ("4,5-phenanthroline", "4,5-phenanthroline"),
    ("acridine", "acridine"),
    ("9H-carbazole", "9H-carbazole"),
    ("purine", "7H-purine"),
    ("1H-indazole", "1H-indazole"),
    ("9H-xanthene", "9H-xanthene"),
)

OXO_CASES = (
    ("phenazin-1-one", "phenazin-1-one"),
    ("phenazine-1,6-dione", "phenazine-1,6-dione"),
    ("1,10-phenanthrolin-5-one", "1,10-phenanthrolin-5-one"),
    ("1,10-phenanthroline-5,6-dione", "1,10-phenanthroline-5,6-dione"),
    ("acridin-9-one", "acridin-9-one"),
    ("9H-carbazol-1-one", "9H-carbazol-1-one"),
    ("purine-2,6-dione", "1H-purine-2,6-dione"),
    ("2,8-diamino-1,4-dihydropurin-6-one", "2,8-diamino-1,4-dihydropurin-6-one"),
    ("1,3,7-trimethylpurine-2,6-dione", "1,3,7-trimethylpurine-2,6-dione"),
    ("1H-indazol-3-one", "1H-indazol-3-one"),
    ("xanthen-9-one", "xanthen-9-one"),
)

_CML_NS = {"cml": "http://www.xml-cml.org/schema"}
_COMBINATION_SAMPLE_SIZE = 100
_COMBINATION_SAMPLE_SEED = 42


@dataclass(frozen=True)
class OxoProbe:
    parent: str
    name: str
    root: str
    occupied_locants: tuple[str, ...]
    carbon_locants: tuple[str, ...]
    smiles: str


def _require_opsin() -> None:
    if py2opsin is None:
        pytest.skip("py2opsin is not available")
    if shutil.which("java") is None:
        pytest.skip("Java runtime not found (OPSIN requires Java)")


def _opsin(names: list[str], output_format: str = "SMILES") -> list[str]:
    _require_opsin()
    # py2opsin 2.9 splits batched CML at newlines instead of at molecules.
    # Single-name calls preserve each XML document intact.
    if output_format == "CML":
        return [py2opsin.py2opsin(name, output_format=output_format) for name in names]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*OPSIN raised the following error.*", category=RuntimeWarning)
        return list(py2opsin.py2opsin(names, output_format=output_format))


def _cml_plan(cml: str) -> tuple[dict[str, str], set[frozenset[str]], set[str]]:
    root = ET.fromstring(cml)
    atom_rows: dict[str, tuple[str, str | None]] = {}
    for atom in root.findall(".//cml:atom", _CML_NS):
        labels = atom.findall("cml:label", _CML_NS)
        locant = next((label.get("value") for label in labels if label.get("value", "")[0].isdigit()), None)
        atom_rows[atom.get("id", "")] = (atom.get("elementType", ""), locant)

    elements = {locant: element for element, locant in atom_rows.values() if locant is not None}
    edges: set[frozenset[str]] = set()
    hydrogenated: set[str] = set()
    for bond in root.findall(".//cml:bond", _CML_NS):
        left_id, right_id = bond.get("atomRefs2", "").split()
        left_element, left_locant = atom_rows[left_id]
        right_element, right_locant = atom_rows[right_id]
        if left_locant is not None and right_locant is not None:
            edges.add(frozenset((left_locant, right_locant)))
        elif left_element == "H" and right_locant is not None:
            hydrogenated.add(right_locant)
        elif right_element == "H" and left_locant is not None:
            hydrogenated.add(left_locant)
    return elements, edges, hydrogenated


def _normalized_pairs_match(left: list[str], right: list[str]) -> bool:
    for left_smiles, right_smiles in zip(left, right, strict=True):
        standardized_left = standardize_mol(left_smiles)
        standardized_right = standardize_mol(right_smiles)
        if standardized_left is None or standardized_right is None or standardized_left != standardized_right:
            return False
    return True


def test_normalized_pairs_match_requires_parseable_equivalent_structures():
    assert _normalized_pairs_match(["C(C)O"], ["CCO"])
    assert not _normalized_pairs_match(["CCO"], ["CC"])
    assert not _normalized_pairs_match(["not-a-smiles"], ["CCO"])


@pytest.fixture(scope="module")
def parent_cml_rows() -> list[str]:
    return _opsin([name for name, _ in PARENT_CASES], output_format="CML")


@pytest.fixture(scope="module")
def valid_oxo_rows(parent_cml_rows) -> list[OxoProbe]:
    candidates: list[OxoProbe] = []
    templates = {template.name: template for template in retained_fused_graph_templates(include_disabled=True)}
    for (parent, _), cml in zip(PARENT_CASES, parent_cml_rows, strict=True):
        elements, _, hydrogenated = _cml_plan(cml)
        carbon_locants = tuple(
            sorted(
                (loc for loc in hydrogenated if elements[loc] == "C"),
                key=lambda value: int(value),
            )
        )
        template_name = parent.removeprefix("1H-").removeprefix("7H-").removeprefix("9H-")
        stem = templates[template_name].derivative_stem
        assert stem is not None
        root = "phenanthrol" if "phenanthroline" in parent else stem.removeprefix("1H-").removeprefix("9H-")
        candidates.extend(
            OxoProbe(parent, f"{stem}-{locant}-one", root, (locant,), carbon_locants, "") for locant in carbon_locants
        )
        candidates.extend(
            OxoProbe(
                parent,
                f"{parent}-{left},{right}-dione",
                root,
                (left, right),
                carbon_locants,
                "",
            )
            for left, right in combinations(carbon_locants, 2)
        )

    candidate_smiles = _opsin([probe.name for probe in candidates])
    return [
        OxoProbe(
            probe.parent,
            probe.name,
            probe.root,
            probe.occupied_locants,
            probe.carbon_locants,
            smiles,
        )
        for probe, smiles in zip(candidates, candidate_smiles, strict=True)
        if smiles
    ]


def _substituted_derivative_name(base_name: str, substituents: tuple[tuple[str, str], ...]) -> str:
    grouped: dict[str, list[str]] = {}
    for locant, substituent in substituents:
        grouped.setdefault(substituent, []).append(locant)
    prefix_parts = []
    for substituent in sorted(grouped):
        locants = sorted(grouped[substituent], key=int)
        multiplier = {1: "", 2: "di"}[len(locants)]
        prefix_parts.append(f"{','.join(locants)}-{multiplier}{substituent}")
    separator = "-" if base_name[0].isdigit() else ""
    return f"{'-'.join(prefix_parts)}{separator}{base_name}"


def _stratified_combination_sample(
    candidates: list[tuple[str, str, str, str]],
) -> list[tuple[str, str, str, str]]:
    """Select a seeded sample covering every retained parent and probe class."""

    rng = Random(_COMBINATION_SAMPLE_SEED)
    shuffled = list(dict.fromkeys(candidates))
    rng.shuffle(shuffled)
    selected: list[tuple[str, str, str, str]] = []

    for parent in sorted({row[2] for row in candidates}):
        selected.append(next(row for row in shuffled if row[2] == parent))
    for probe_class in sorted({row[3] for row in candidates}):
        if not any(row[3] == probe_class for row in selected):
            selected.append(next(row for row in shuffled if row[3] == probe_class))

    selected_set = set(selected)
    selected.extend(row for row in shuffled if row not in selected_set)
    return selected[:_COMBINATION_SAMPLE_SIZE]


@pytest.mark.opsin
def test_opsin_cml_locants_and_graphs_are_the_source_of_truth_for_production_plans(parent_cml_rows):
    """Every template atom, element, edge, and locant must equal OPSIN CML."""

    templates = {template.name: template for template in retained_fused_graph_templates(include_disabled=True)}

    for (opsin_name, expected_name), cml in zip(PARENT_CASES, parent_cml_rows, strict=True):
        template_name = expected_name.removeprefix("7H-").removeprefix("9H-").removeprefix("1H-")
        template = templates[template_name]
        elements, edges, _ = _cml_plan(cml)
        template_edges = {frozenset(bond.locants) for bond in template.bonds}

        assert set(elements) == set(template.locants), opsin_name
        assert elements == {locant: atom.symbol for locant, atom in template.atom_by_locant.items()}, opsin_name
        assert edges == template_edges, opsin_name


@pytest.mark.opsin
def test_opsin_generated_parent_smiles_use_the_audited_retained_names():
    opsin_smiles = _opsin([name for name, _ in PARENT_CASES])
    generated = [result.name for result in name_many(opsin_smiles, processes=1)]

    assert generated == [expected for _, expected in PARENT_CASES]


@pytest.mark.opsin
def test_every_opsin_hydrogen_bearing_carbon_locant_survives_methyl_substitution(parent_cml_rows):
    """Probe every substitutable carbon locant, including symmetry-equivalent sites."""

    opsin_names = [name for name, _ in PARENT_CASES]
    probes: list[str] = []
    for parent, cml in zip(opsin_names, parent_cml_rows, strict=True):
        elements, _, hydrogenated = _cml_plan(cml)
        for locant in sorted((loc for loc in hydrogenated if elements[loc] == "C"), key=lambda value: int(value)):
            separator = "-" if parent[0].isdigit() else ""
            probes.append(f"{locant}-methyl{separator}{parent}")

    substituted_smiles = _opsin(probes)
    assert len(probes) >= 130
    assert substituted_smiles and all(substituted_smiles)
    generated_names = [result.name for result in name_many(substituted_smiles, processes=1)]
    regenerated_smiles = _opsin(generated_names)

    assert all(generated_names)
    assert all("methyl" in name for name in generated_names)
    assert _normalized_pairs_match(substituted_smiles, regenerated_smiles)


@pytest.mark.opsin
def test_all_opsin_valid_oxo_and_dione_positions_keep_the_retained_parent(valid_oxo_rows):
    """Exercise every carbonyl position and pair accepted by OPSIN."""

    assert len(valid_oxo_rows) >= 400

    generated_names = [result.name for result in name_many([row.smiles for row in valid_oxo_rows], processes=1)]
    regenerated_smiles = _opsin(generated_names)

    assert all(
        row.root.lower() in generated.lower() for row, generated in zip(valid_oxo_rows, generated_names, strict=True)
    )
    failures = [
        (row.name, generated, standardize_mol(row.smiles), standardize_mol(regenerated))
        for row, generated, regenerated in zip(valid_oxo_rows, generated_names, regenerated_smiles, strict=True)
        if standardize_mol(row.smiles) != standardize_mol(regenerated)
    ]
    assert not failures, failures[:20]


@pytest.mark.opsin
def test_oxo_dione_amino_methyl_combinations_preserve_hydride_state(valid_oxo_rows):
    """Cross principal groups with substituents that can move indicated H."""

    representative_rows: list[OxoProbe] = []
    for parent, _ in PARENT_CASES:
        parent_rows = [probe for probe in valid_oxo_rows if probe.parent == parent]
        mono_rows = [probe for probe in parent_rows if len(probe.occupied_locants) == 1]
        dione_rows = [probe for probe in parent_rows if len(probe.occupied_locants) == 2]
        # One mono-oxo and one dione per retained plan exercise every family
        # without repeating symmetry-equivalent cross products. Purine keeps
        # the complete positional matrix because fusion-carbon hydride shifts
        # were observed there in the PubChem regression corpus.
        representative_rows.extend(parent_rows if parent == "purine" else [mono_rows[0], dione_rows[0]])

    candidate_names: list[tuple[str, str, str, str]] = []
    for probe in representative_rows:
        derivative_class = "mono" if len(probe.occupied_locants) == 1 else "dione"
        free_locants = tuple(locant for locant in probe.carbon_locants if locant not in probe.occupied_locants)
        for locant in free_locants:
            candidate_names.append(
                (
                    _substituted_derivative_name(probe.name, ((locant, "amino"),)),
                    probe.root,
                    probe.parent,
                    f"{derivative_class}:amino",
                )
            )
            candidate_names.append(
                (
                    _substituted_derivative_name(probe.name, ((locant, "methyl"),)),
                    probe.root,
                    probe.parent,
                    f"{derivative_class}:methyl",
                )
            )

        # Pair substitutions are the important hydride-state stressor. Apply
        # every pair to mono-oxo representatives; diones retain complete
        # single-substitution coverage without a second quadratic expansion.
        if len(probe.occupied_locants) == 1:
            for left, right in combinations(free_locants, 2):
                candidate_names.append(
                    (
                        _substituted_derivative_name(probe.name, ((left, "amino"), (right, "amino"))),
                        probe.root,
                        probe.parent,
                        "mono:diamino",
                    )
                )
                candidate_names.append(
                    (
                        _substituted_derivative_name(probe.name, ((left, "amino"), (right, "methyl"))),
                        probe.root,
                        probe.parent,
                        "mono:amino_methyl",
                    )
                )
                candidate_names.append(
                    (
                        _substituted_derivative_name(probe.name, ((left, "methyl"), (right, "amino"))),
                        probe.root,
                        probe.parent,
                        "mono:amino_methyl",
                    )
                )

    candidate_names = _stratified_combination_sample(candidate_names)
    assert {row[2] for row in candidate_names} == {parent for parent, _ in PARENT_CASES}
    assert {row[3] for row in candidate_names} == {
        "mono:amino",
        "mono:methyl",
        "mono:diamino",
        "mono:amino_methyl",
        "dione:amino",
        "dione:methyl",
    }
    candidate_smiles = _opsin([name for name, _, _, _ in candidate_names])
    valid_rows = [
        (name, root, smiles)
        for (name, root, _, _), smiles in zip(candidate_names, candidate_smiles, strict=True)
        if smiles
    ]
    assert len(valid_rows) == _COMBINATION_SAMPLE_SIZE

    generated_names = [result.name for result in name_many([smiles for _, _, smiles in valid_rows], processes=1)]
    regenerated_smiles = _opsin(generated_names)
    failures = [
        (source_name, generated, standardize_mol(source), standardize_mol(regenerated))
        for (source_name, _, source), generated, regenerated in zip(
            valid_rows, generated_names, regenerated_smiles, strict=True
        )
        if not generated or standardize_mol(source) != standardize_mol(regenerated)
    ]

    assert all(root.lower() in generated.lower() for (_, root, _), generated in zip(valid_rows, generated_names))
    assert not failures, failures[:20]


@pytest.mark.opsin
def test_oxo_and_dione_derivative_classes_are_opsin_exact_and_roundtrip():
    opsin_smiles = _opsin([opsin_name for opsin_name, _ in OXO_CASES])
    generated_names = [result.name for result in name_many(opsin_smiles, processes=1)]
    regenerated_smiles = _opsin(generated_names)

    assert generated_names == [expected for _, expected in OXO_CASES]
    assert _normalized_pairs_match(opsin_smiles, regenerated_smiles)


@pytest.mark.opsin
def test_hydro_oxo_analogs_do_not_false_match_mancude_production_plans():
    opsin_names = [
        "3,4-dihydrophenazin-1-one",
        "3,4-dihydroacridin-9-one",
        "1,2-dihydroxanthen-9-one",
    ]
    opsin_smiles = _opsin(opsin_names)
    generated_names = [result.name for result in name_many(opsin_smiles, processes=1)]
    regenerated_smiles = _opsin(generated_names)

    assert "phenazin" not in generated_names[0]
    assert "acridin" not in generated_names[1]
    assert "xanthen" not in generated_names[2]
    assert _normalized_pairs_match(opsin_smiles, regenerated_smiles)
