"""Construction resolves peripheral symmetry without relaxing graph identity."""

import xml.etree.ElementTree as ET
from dataclasses import replace
from random import Random

import pytest
from interior_crosswalk_support import to_iupac_edges, to_iupac_labels
from rdkit import Chem

from openclatura import name_mol, opsin_available, verify_with_opsin
from openclatura.assembly_utils import parse_locant
from openclatura.chains import find_ring_systems
from openclatura.fusion import numbering, planner
from openclatura.fusion.citation_numbering import numbered_parent_graphs_agree
from openclatura.fusion.model import FusionConfirmed
from openclatura.graph_io import read_rdkit_mol

SMILES = "C1=CC=C2C=CC3=CC=CC4=CC=C1C2=C34"
BASE = "benzo[1,2,3,4-def]phenanthrene"
_SUPERSCRIPT = str.maketrans("0123456789", "\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079")


def _plan(graph):
    mol = read_rdkit_mol(graph)
    atoms = max(find_ring_systems(mol), key=lambda ring: len(ring.atoms)).atoms
    return mol, atoms, planner.plan_fusion_parent(mol, atoms, mode="general")


def _orders(graph):
    order = list(range(graph.GetNumAtoms()))
    yield order
    yield order[::-1]
    for seed in (7, 73, 20483):
        shuffled = order[:]
        Random(seed).shuffle(shuffled)
        yield shuffled


def _labelled_opsin_graph(tmp_path):
    from py2opsin import py2opsin

    root = ET.fromstring(py2opsin(BASE, output_format="CML", tmp_fpath=str(tmp_path / "parent.txt")))
    ns = {"c": "http://www.xml-cml.org/schema"}
    labels = {
        atom.attrib["id"]: atom.find("c:label", ns).attrib["value"]
        for atom in root.findall(".//c:atom", ns)
        if atom.attrib["elementType"] != "H"
    }
    edges = {
        frozenset(labels[atom] for atom in bond.attrib["atomRefs2"].split())
        for bond in root.findall(".//c:bond", ns)
        if all(atom in labels for atom in bond.attrib["atomRefs2"].split())
    }
    assert len(labels) == 16
    return set(labels.values()), edges


@pytest.mark.skipif(not opsin_available(), reason="OPSIN and Java are required")
@pytest.mark.parametrize("substitution", (None, (0, "C", "methyl"), (9, "Cl", "chloro")))
def test_every_selected_interior_map_matches_raw_opsin_graph(substitution, tmp_path):
    graph = Chem.RWMol(Chem.MolFromSmiles(SMILES))
    if substitution:
        parent, symbol, _ = substitution
        graph.AddBond(parent, graph.AddAtom(Chem.Atom(symbol)), Chem.BondType.SINGLE)
        Chem.SanitizeMol(graph)
    expected_vertices, expected_edges = _labelled_opsin_graph(tmp_path)
    for order in _orders(graph):
        mol, atoms, result = _plan(Chem.RenumberAtoms(graph, order))
        assert isinstance(result, FusionConfirmed), result
        plan = result.plan
        assert plan.audit.confirmed
        assert plan.rendered_base_name == BASE
        maps = plan.numbering.string_input_locant_maps()
        assert maps
        assert numbered_parent_graphs_agree(mol, maps)
        # The parser numbers interior carbons by continuing the fusion letters
        # and this engine applies P-25.3.3.3, so the two graphs are related by
        # the reviewed crosswalk - not compared raw, and not by a relabelling
        # discovered here, which could absorb a peripheral error.
        iupac_vertices = to_iupac_labels(BASE, expected_vertices)
        iupac_edges = to_iupac_edges(BASE, expected_edges)
        assert iupac_vertices is not None, f"no reviewed crosswalk covers {BASE}"
        for mapping in maps:
            assert set(mapping.values()) == iupac_vertices
            assert {
                frozenset((mapping[bond.u], mapping[bond.v]))
                for bond in mol.bonds.values()
                if bond.u in atoms and bond.v in atoms
            } == iupac_edges
            name = BASE
            if substitution:
                parent, _, prefix = substitution
                name = f"{mapping[order.index(parent)]}-{prefix}{BASE}"
            check = verify_with_opsin(name, Chem.MolToSmiles(graph), standardize_smiles=False)
            assert check.status == "matched", check.to_dict()
            assert check.canonical_original == check.canonical_roundtrip


def test_removing_provider_priority_does_not_create_ambiguity():
    """The replacement for "construction provenance is required to resolve the tie".

    It no longer is. Pyrene's peripheral symmetry produces several atom-to-locant
    maps, but under P-25.3.3.3 they are several maps of ONE numbered parent
    graph, so there is nothing for a tie-break to decide. The provider that
    found a candidate is not a nomenclatural criterion and must not rank it.
    """

    mol, atoms, result = _plan(Chem.MolFromSmiles(SMILES))
    assert isinstance(result, FusionConfirmed), result
    maps = result.plan.numbering.string_input_locant_maps()
    assert len(maps) > 1, "the symmetry really does yield several input mappings"
    assert numbered_parent_graphs_agree(mol, maps), "and they are one numbered parent graph"


@pytest.mark.parametrize(
    "smiles,mappings",
    (
        pytest.param(SMILES, 4, id="pyrene"),
        pytest.param("C1=CC2=CC=CC3=CC=CC(=C1)C23", 6, id="phenalene"),
    ),
)
def test_symmetry_related_maps_complete_to_one_numbered_graph(smiles, mappings):
    """The regression observation worth keeping: N maps, one graph.

    Numbering every interior carbon by the same rule is what collapses these.
    Fixed-order continuation labels do not exchange with the perimeter under a
    symmetry operation, which made one parent look like two.
    """

    from openclatura.fusion.faces import cached_bounded_face_model, typed_face_model
    from openclatura.fusion.layout import preferred_intrinsic_layouts
    from openclatura.fusion.numbering import _fusion_atoms, _numbering_from_layout
    from openclatura.graph_io import read_rdkit_mol

    mol = read_rdkit_mol(Chem.MolFromSmiles(smiles))
    atoms = frozenset(max(find_ring_systems(mol), key=lambda ring: len(ring.atoms)).atoms)
    bounded = cached_bounded_face_model(mol, atoms)
    face_model = typed_face_model(mol, bounded)
    fusion_atoms = _fusion_atoms(bounded)
    cache, derived = {}, []
    for index, layout in enumerate(preferred_intrinsic_layouts(face_model)):
        numbering = _numbering_from_layout(mol, bounded, face_model, layout, index, fusion_atoms, perimeter_cache=cache)
        if numbering is not None:
            derived.append(dict(numbering.atom_to_locant))
    assert len(derived) == mappings
    assert len({tuple(sorted((atom, str(locant)) for atom, locant in m.items())) for m in derived}) == mappings
    assert numbered_parent_graphs_agree(mol, derived)


def test_every_accepted_map_satisfies_the_numbering_rules(tmp_path):
    """The replacement for "a construction witness must control the numbering".

    What an accepted map has to satisfy is the rules, whichever provider found
    it: every interior carbon takes its nearest peripheral anchor, lowest first,
    with the bond distance as a superscript.
    """

    from collections import deque

    mol, atoms, result = _plan(Chem.MolFromSmiles(SMILES))
    assert isinstance(result, FusionConfirmed)
    for mapping in result.plan.numbering.string_input_locant_maps():
        interior = {atom: locant for atom, locant in mapping.items() if "\u00b9" in locant or "\u00b2" in locant}
        assert interior, "pyrene has interior carbons"
        perimeter = {atom for atom in mapping if atom not in interior}
        for atom, locant in interior.items():
            distance, queue = {atom: 0}, deque([atom])
            while queue:
                current = queue.popleft()
                for neighbour in mol.get_neighbors(current):
                    if neighbour in atoms and neighbour not in distance:
                        distance[neighbour] = distance[current] + 1
                        queue.append(neighbour)
            reach = {other: distance[other] for other in perimeter if other in distance}
            shortest = min(reach.values())
            closest = sorted(
                (mapping[other] for other, value in reach.items() if value == shortest),
                key=parse_locant,
            )
            assert locant == f"{closest[0]}{str(shortest).translate(_SUPERSCRIPT)}"


def test_public_retained_parent_precedence_is_unchanged():
    result = name_mol(Chem.MolFromSmiles(SMILES))
    assert result.error is None
    assert result.name == "pyrene"


@pytest.mark.skipif(not opsin_available(), reason="OPSIN and Java are required")
@pytest.mark.parametrize("parent", ("perylene", "benzo[a]pyrene", "benzo[e]pyrene", "benzo[ghi]perylene"))
def test_neighboring_hexagonal_parents_match_every_raw_labelled_map(parent, tmp_path):
    from py2opsin import py2opsin

    from openclatura.retained_fused_templates import retained_graph_templates

    template = next(template for template in retained_graph_templates(include_disabled=True) if template.name == parent)
    graph = Chem.RWMol()
    ids = {}
    for atom in template.atoms:
        rd_atom = Chem.Atom(atom.symbol)
        rd_atom.SetIsAromatic(True)
        ids[atom.locant] = graph.AddAtom(rd_atom)
    for bond in template.bonds:
        graph.AddBond(*(ids[locant] for locant in bond.locants), Chem.BondType.AROMATIC)
    Chem.SanitizeMol(graph)
    expected_name = None
    for order in _orders(graph):
        mol, atoms, result = _plan(Chem.RenumberAtoms(graph, order))
        assert isinstance(result, FusionConfirmed), result
        name = result.plan.rendered_base_name
        if expected_name is None:
            expected_name = name
        assert name == expected_name
        root = ET.fromstring(py2opsin(name, output_format="CML", tmp_fpath=str(tmp_path / "neighbor.txt")))
        ns = {"c": "http://www.xml-cml.org/schema"}
        labels = {
            atom.attrib["id"]: atom.find("c:label", ns).attrib["value"]
            for atom in root.findall(".//c:atom", ns)
            if atom.attrib["elementType"] != "H"
        }
        expected = {
            frozenset(labels[atom] for atom in bond.attrib["atomRefs2"].split())
            for bond in root.findall(".//c:bond", ns)
            if all(atom in labels for atom in bond.attrib["atomRefs2"].split())
        }
        iupac_vertices = to_iupac_labels(expected_name, set(labels.values()))
        iupac_edges = to_iupac_edges(expected_name, expected)
        if iupac_vertices is None:
            pytest.skip(f"no reviewed interior crosswalk for {expected_name}")
        for mapping in result.plan.numbering.string_input_locant_maps():
            assert set(mapping.values()) == iupac_vertices
            assert {
                frozenset((mapping[bond.u], mapping[bond.v]))
                for bond in mol.bonds.values()
                if bond.u in atoms and bond.v in atoms
            } == iupac_edges
        check = verify_with_opsin(name, Chem.MolToSmiles(graph), standardize_smiles=False)
        assert check.status == "matched", check.to_dict()


@pytest.mark.parametrize("withheld", ("exhausted", "generated", "missing"))
def test_withholding_construction_metadata_does_not_create_ambiguity(withheld, monkeypatch):
    """The replacement for three "preserves ambiguity" assertions.

    Each used to withhold the construction order and require the plan to fail,
    on the premise that nothing else could resolve pyrene. The rules resolve it:
    the construction order may reproduce a candidate or help diagnose candidate
    generation, but withholding it must not change the answer.
    """

    from openclatura.fusion import construction_order

    original = construction_order.ordered_hexagonal_construction

    def withhold(ast, specs, model, positions, search_budget):
        if withheld == "exhausted":
            return original(ast, specs, model, positions, 0)
        if withheld == "missing":
            specs = {occurrence: replace(spec, construction_order=None) for occurrence, spec in specs.items()}
        else:
            specs = {
                occurrence: replace(
                    spec,
                    template=replace(spec.template, numbering_policy="generated_acene_series"),
                    construction_order=None,
                )
                for occurrence, spec in specs.items()
            }
        return original(ast, specs, model, positions, search_budget)

    reference_mol, _, reference = _plan(Chem.MolFromSmiles(SMILES))
    assert isinstance(reference, FusionConfirmed)
    expected = reference.plan.numbering.string_input_locant_maps()

    monkeypatch.setattr(construction_order, "ordered_hexagonal_construction", withhold)
    mol, _, result = _plan(Chem.MolFromSmiles(SMILES))
    assert isinstance(result, FusionConfirmed), result
    assert result.plan.rendered_base_name == BASE
    assert numbered_parent_graphs_agree(mol, [*expected, *result.plan.numbering.string_input_locant_maps()])


def test_a_genuinely_unresolved_input_still_yields_nothing():
    """The honest failure case, kept with an input that really is unresolved.

    Pyrene is no longer an artificial "unresolvable without provenance"
    fixture, so the guard is exercised where candidate generation genuinely
    supplies nothing: with no layouts there is no perimeter to number, and the
    selection has to come back empty rather than invent one.
    """

    from openclatura.fusion.faces import cached_bounded_face_model, typed_face_model

    mol, atoms, _ = _plan(Chem.MolFromSmiles(SMILES))
    bounded = cached_bounded_face_model(mol, frozenset(atoms))
    selection = numbering.completed_system_numbering_selection(
        mol, bounded, face_model=typed_face_model(mol, bounded), layouts=()
    )
    assert selection.accepted == ()


@pytest.mark.parametrize("seed", (7, 73))
@pytest.mark.parametrize("parent", ("pyrene", "benzo[ghi]perylene"))
def test_reordering_component_records_does_not_change_the_numbered_graph(seed, parent):
    """The replacement for "stable construction enumeration selects the graph".

    What must hold is that the answer does not depend on the order records
    happen to arrive in. Shuffling a component template's atoms and bonds, and
    reversing its bond locants, is a pure re-presentation of the same component,
    so the selected numbered parent graph has to come back identical.
    """

    from openclatura.retained_fused_templates import retained_graph_templates

    template = next(item for item in retained_graph_templates(include_disabled=True) if item.name == parent)

    def build(atoms, bonds):
        graph = Chem.RWMol()
        ids = {}
        for atom in atoms:
            rd_atom = Chem.Atom(atom.symbol)
            rd_atom.SetIsAromatic(True)
            ids[atom.locant] = graph.AddAtom(rd_atom)
        for bond in bonds:
            graph.AddBond(*(ids[locant] for locant in bond.locants), Chem.BondType.AROMATIC)
        Chem.SanitizeMol(graph)
        return graph

    reference_mol, _, reference = _plan(build(template.atoms, template.bonds))
    assert isinstance(reference, FusionConfirmed), reference

    atoms = list(template.atoms)
    bonds = [replace(bond, locants=tuple(reversed(bond.locants))) for bond in template.bonds]
    Random(seed).shuffle(atoms)
    Random(seed).shuffle(bonds)
    shuffled_mol, _, shuffled = _plan(build(atoms, bonds))
    assert isinstance(shuffled, FusionConfirmed), shuffled

    assert shuffled.plan.rendered_base_name == reference.plan.rendered_base_name

    # The atom ids differ between the two builds, so compare the labelled
    # graphs rather than the maps: same vertices, same locant-labelled edges.
    def labelled(mol, result):
        mapping = result.plan.numbering.string_input_locant_maps()[0]
        return (
            frozenset(mapping.values()),
            frozenset(
                frozenset((mapping[bond.u], mapping[bond.v]))
                for bond in mol.bonds.values()
                if bond.u in mapping and bond.v in mapping
            ),
        )

    assert labelled(shuffled_mol, shuffled) == labelled(reference_mol, reference)


@pytest.mark.parametrize("defect", ("missing_edge", "extra_vertex", "missing_source", "missing_version"))
def test_invalid_component_construction_metadata_is_rejected(defect):
    from openclatura.fusion.registry import fusion_component_registry

    spec = next(item.spec for item in fusion_component_registry().components if item.spec.key == "phenanthrene")
    order = spec.construction_order
    assert order is not None
    with pytest.raises(ValueError):
        if defect == "missing_edge":
            order = replace(order, directed_bond_locants=order.directed_bond_locants[:-1])
        elif defect == "extra_vertex":
            order = replace(order, atom_locants=(*order.atom_locants, "100"))
        elif defect == "missing_source":
            order = replace(order, source="")
        else:
            order = replace(order, version="")
        replace(spec, construction_order=order)


def test_registry_construction_provenance_is_explicit_and_immutable():
    from dataclasses import FrozenInstanceError

    from openclatura.fusion.registry import fusion_component_registry

    certified = {}
    for component in fusion_component_registry().components:
        spec = component.spec
        order = spec.construction_order
        if order is None:
            continue
        assert order.covers(spec.template)
        assert order.version == "2.9.0"
        assert order.source.endswith("/fusionComponents.xml")
        certified[spec.key] = order
    assert set(certified) == {"benzene", "naphthalene", "anthracene", "phenanthrene"}
    with pytest.raises(FrozenInstanceError):
        certified["benzene"].source = "changed"
