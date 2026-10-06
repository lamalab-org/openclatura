"""A terminal three-membered ring opens a row of its own (P-25.3.2.3.3).

Three-membered rings are not categorically assigned to a higher row: the
permitted triangle of P-25.3.2.3.1 may point left or right, and a terminal
triangle whose common bond is vertical shares its host's row. It does not share
it here. In phenanthro[1',2':1,2]cyclopenta[2,3-b]oxirene the cyclopenta's two
fusion bonds are adjacent, so they cannot both be the vertical common bonds of
one horizontal row; criterion (a) ties at two rings either way and criterion
(b) decides. Taking the two lower phenanthrene rings as the row carries
C -> cyclopenta -> oxirene into the upper right, which the alternative loses.
The oxirene is then the sole uppermost ring, and P-25.3.3.1.1 starts the
numbering at its only nonfusion atom, the oxygen.

OPSIN builds a direction-table ring map rather than applying the criteria to a
drawing, and for the systems recorded below it numbers from the pentagon
instead. That is an implementation convention: these are not among the systems
FR-5.1 concedes fall outside its rules, since no distorted ring is needed and
(a) then (b) reach a decision. So OPSIN's labelled graph is the oracle for
every other fused parent (see test_fusion_construction_numbering) but not for
these, and what is pinned here is that each divergence is a pure relabelling of
the same ring system - never a different one.
"""

import xml.etree.ElementTree as ET

import pytest
from rdkit import Chem

from openclatura import name_mol, name_smiles, opsin_available, verify_with_opsin
from openclatura.fusion import planner
from openclatura.fusion.model import FusionConfirmed

_CML = {"c": "http://www.xml-cml.org/schema"}

# Same motif as the divergent family - a terminal three-membered ring on a
# pentagon that carries a second fusion bond - where our numbering and OPSIN's
# already agree. cyclopenta[1,2]phenanthro[8a,9-b]oxirene is the ring system of
# a published CAS index name whose lettered locants are 4a,5a,6a,6b,9a,11a,11b.
AGREEING = (
    "naphtho[1',2':1,2]cyclopenta[2,3-b]oxirene",
    "phenanthro[1',2':1,2]cyclopenta[2,3-b]furan",
    "cyclopenta[1,2]phenanthro[8a,9-b]oxirene",
)

# Our locants, which the orientation above selects. OPSIN numbers these from
# the pentagon and so reaches 2a,3a,5a,5b,9a,11a,11b instead of 1a,3a,3b,...
#
# phenanthro[9',10':4,5]indeno[3,3a-b]oxirene and its thiirene were recorded
# here too. They no longer diverge: triphenylene is an admitted retained
# component, so those systems are now named on it -
# triphenyleno[1',2':1,2]cyclopenta[1,5-b]oxirene - and our numbering of that
# parent agrees with OPSIN atom for atom. A divergence that disappears because
# a better component became available is not evidence either way about the
# orientation rule; the three below still turn on it.
DIVERGENT = {
    "phenanthro[1',2':1,2]cyclopenta[2,3-b]oxirene": "1 1a 2 3 3a 3b 4 5 5a 6 7 8 9 9a 9b 10 11 11a",
    "phenanthro[1',2':1,2]cyclopenta[2,3-b]thiirene": "1 1a 2 3 3a 3b 4 5 5a 6 7 8 9 9a 9b 10 11 11a",
}


def _opsin_labelled_graph(base, tmp_path):
    """OPSIN's locant-labelled skeleton for a fusion parent, from its CML."""

    from py2opsin import py2opsin

    for name in (base, "perhydro" + base):
        cml = py2opsin(name, output_format="CML", tmp_fpath=str(tmp_path / "reference.txt"))
        if not cml:
            continue
        root = ET.fromstring(cml)
        atoms = root.findall(".//c:atom", _CML)
        labels = {
            atom.attrib["id"]: atom.find("c:label", _CML).attrib["value"]
            for atom in atoms
            if atom.find("c:label", _CML) is not None
        }
        if not labels:
            continue
        elements = {
            labels[atom.attrib["id"]]: atom.attrib["elementType"] for atom in atoms if atom.attrib["id"] in labels
        }
        edges = {
            frozenset(labels[atom] for atom in bond.attrib["atomRefs2"].split())
            for bond in root.findall(".//c:bond", _CML)
            if all(atom in labels for atom in bond.attrib["atomRefs2"].split())
        }
        return elements, edges
    raise AssertionError(f"OPSIN gave no labelled skeleton for {base}")


def _our_labelled_graph(seed, monkeypatch):
    """Name the skeleton OPSIN built and keep our own locant-labelled parent.

    The parent we name is what has to be compared, not the name it was seeded
    from: the two differ whenever our component citation differs from the seed.
    """

    from py2opsin import py2opsin

    smiles = py2opsin(seed) or py2opsin("perhydro" + seed)
    assert smiles, f"OPSIN could not build {seed}"
    captured = {}
    original = planner._complete_fusion_plan

    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        if isinstance(result, FusionConfirmed) and result.plan.rendered_base_name and not captured:
            plan = result.plan
            locants = dict(plan.numbering.abstract_atom_to_locant)
            graph = plan.abstract_parent_graph
            captured["base"] = plan.rendered_base_name
            captured["elements"] = {str(locants[atom.id]): atom.symbol for atom in graph.atoms}
            captured["edges"] = {frozenset(str(locants[atom]) for atom in bond.atoms) for bond in graph.bonds}
        return result

    monkeypatch.setattr(planner, "_complete_fusion_plan", capture)
    name_mol(Chem.MolFromSmiles(smiles))
    assert captured, f"no fusion plan was built for {seed}"
    return captured["base"], (captured["elements"], captured["edges"])


def _is_relabelling(ours, reference):
    """Whether two labelled graphs are the same ring system, differently named."""

    our_elements, our_edges = ours
    ref_elements, ref_edges = reference
    our_adjacent = {locant: set() for locant in our_elements}
    for edge in our_edges:
        left, right = sorted(edge)
        our_adjacent[left].add(right)
        our_adjacent[right].add(left)
    ref_adjacent = {locant: set() for locant in ref_elements}
    for edge in ref_edges:
        left, right = sorted(edge)
        ref_adjacent[left].add(right)
        ref_adjacent[right].add(left)
    order = sorted(our_elements, key=lambda locant: (-len(our_adjacent[locant]), locant))
    mapping: dict[str, str] = {}

    def place(index):
        if index == len(order):
            return True
        ours_here = order[index]
        for theirs in ref_elements:
            if theirs in mapping.values() or ref_elements[theirs] != our_elements[ours_here]:
                continue
            if len(ref_adjacent[theirs]) != len(our_adjacent[ours_here]):
                continue
            if any(mapping[other] not in ref_adjacent[theirs] for other in our_adjacent[ours_here] if other in mapping):
                continue
            mapping[ours_here] = theirs
            if place(index + 1):
                return True
            del mapping[ours_here]
        return False

    return place(0)


@pytest.mark.skipif(not opsin_available(), reason="the numbering oracle needs java and py2opsin")
@pytest.mark.parametrize("base", AGREEING)
def test_the_motif_itself_does_not_move_the_numbering(base, monkeypatch, tmp_path):
    """One fewer hexagon, a five-membered terminal ring, or the CAS case."""

    named, ours = _our_labelled_graph(base, monkeypatch)
    assert ours == _opsin_labelled_graph(named, tmp_path)


@pytest.mark.skipif(not opsin_available(), reason="the numbering oracle needs java and py2opsin")
@pytest.mark.parametrize("base", sorted(DIVERGENT))
def test_the_recorded_divergence_is_a_relabelling_and_nothing_more(base, monkeypatch, tmp_path):
    """The ring system has to be identical; only the locants on it differ."""

    named, ours = _our_labelled_graph(base, monkeypatch)
    reference = _opsin_labelled_graph(named, tmp_path)
    assert sorted(ours[0]) == sorted(DIVERGENT[base].split())
    assert ours != reference
    assert _is_relabelling(ours, reference)


@pytest.mark.skipif(not opsin_available(), reason="the numbering oracle needs java and py2opsin")
def test_the_oxirene_oxygen_opens_the_numbering(monkeypatch):
    """The operative consequence: the sole uppermost ring is the oxirene, and
    its only nonfusion atom, the oxygen, therefore takes locant 1."""

    _, (elements, _) = _our_labelled_graph("phenanthro[1',2':1,2]cyclopenta[2,3-b]oxirene", monkeypatch)
    assert [locant for locant, symbol in elements.items() if symbol != "C"] == ["1"]


# The 13,17-epoxide steroid below is the one molecule in a 1208-molecule
# benchmark sample whose name OPSIN cannot read, because its parent is the
# first entry of DIVERGENT. Translating every parent locant through the relabelling - 1 -> 3,
# 1a -> 2a, 3a -> 11b, 3b -> 11a, 4 -> 11, 5 -> 10, 5a -> 9a, 6 -> 9, 7 -> 8,
# 8 -> 7, 9 -> 6, 9a -> 5b, 9b -> 5a, 10 -> 5, 11 -> 4, 11a -> 3a - leaves the
# name otherwise untouched and OPSIN then reads it back to the same structure.
# That is what proves the rest of the name: the fusion name, the decahydro
# allocation, both added hydrogens, the two substituent locants and all six
# stereodescriptors.
EPOXIDE = "C[C@]12CCC(=O)C=C1CC[C@@H]1[C@@H]2C(=O)C[C@]23O[C@]2(C(=O)CO)CC[C@@H]13"
EPOXIDE_NAME = (
    "(1aR,3aS,3bS,9aR,9bS,11aR)-1a-(2-hydroxyacetyl)-9a-methyl-"
    "2,3,3a,3b,4,5,8,9,9a,9b-decahydrophenanthro[1',2':1,2]"
    "cyclopenta[2,3-b]oxirene-7,10(1aH,11H)-dione"
)
EPOXIDE_NAME_IN_OPSIN_LOCANTS = (
    "(2aR,3aR,5aS,5bR,11aS,11bS)-2a-(2-hydroxyacetyl)-5b-methyl-"
    "1,2,5a,5b,6,7,10,11,11a,11b-decahydrophenanthro[1',2':1,2]"
    "cyclopenta[2,3-b]oxirene-5,8(2aH,4H)-dione"
)


def test_the_divergent_parent_still_names_its_molecule():
    assert name_smiles(EPOXIDE) == EPOXIDE_NAME


@pytest.mark.skipif(not opsin_available(), reason="OPSIN round trip needs java and py2opsin")
def test_everything_but_the_locants_is_verifiable():
    """The same name under OPSIN's locants reads back to the same structure."""

    assert verify_with_opsin(EPOXIDE_NAME_IN_OPSIN_LOCANTS, EPOXIDE).status == "matched"
