"""A reviewed crosswalk must be proved, not discovered.

The reference parser numbers an interior carbon by continuing the fusion
letters; P-25.3.3.3 numbers it from its nearest peripheral atom with the bond
distance as a superscript. Where that is the whole of the difference, the two
numbered graphs are related by a relabelling, and the relabelling is what these
fixtures record.

Two separate things have to hold, and conflating them is how a real defect
would slip through.

Correspondence says which atom in one graph is which atom in the other. That is
necessary but nowhere near sufficient: an implementation that labelled
phenalene's interior carbon 6a-superscript-one instead of 3a-superscript-one
would still produce an element-preserving, adjacency-preserving relabelling
that moves no peripheral locant at all.

So each conversion also carries a certificate, computed from the parser graph
and its fixed perimeter alone and never from the engine's own choice of anchor:
the shortest distance to the periphery, every peripheral atom at that distance,
and the lowest of them. Pyrene is the published control, FR-5.5.2 printing
10b to 3a-superscript-one and 10c to 5a-superscript-one. The deliberately wrong
phenalene conversion is kept as a negative control, because a check that cannot
fail is not a check.
"""

import json
import pathlib
import re
from collections import deque

import pytest

FIXTURE = json.loads(pathlib.Path("tests/data/interior_locant_crosswalk.json").read_text())
SYSTEMS = [pytest.param(record, id=record["base_name"][:44]) for record in FIXTURE["systems"]]
SUPERSCRIPT = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")


def _locant_key(label):
    match = re.match(r"^(\d+)([a-z]*)$", label)
    return (int(match.group(1)), len(match.group(2)), match.group(2)) if match else (10**6, 0, label)


def _adjacency(edges):
    adjacency = {}
    for left, right in edges:
        adjacency.setdefault(left, set()).add(right)
        adjacency.setdefault(right, set()).add(left)
    return adjacency


def _nearest_anchor(record, interior_locant):
    """P-25.3.3.3.1-2, from the parser graph and its fixed perimeter alone."""

    adjacency = _adjacency(record["parser_graph"]["edges"])
    perimeter = set(record["peripheral_locants"])
    distance = {interior_locant: 0}
    queue = deque([interior_locant])
    while queue:
        current = queue.popleft()
        for neighbour in adjacency[current]:
            if neighbour not in distance:
                distance[neighbour] = distance[current] + 1
                queue.append(neighbour)
    reachable = {label: distance[label] for label in perimeter if label in distance}
    shortest = min(reachable.values())
    closest = sorted([label for label, value in reachable.items() if value == shortest], key=_locant_key)
    return shortest, closest


@pytest.mark.parametrize("record", SYSTEMS)
def test_the_crosswalk_carries_one_graph_exactly_onto_the_other(record):
    """Bijective, element- and edge-preserving, and pointwise fixed outside it."""

    crosswalk = record["crosswalk"]
    parser, iupac = record["parser_graph"], record["iupac_graph"]
    assert len(set(crosswalk.values())) == len(crosswalk), "a crosswalk must be injective"
    relabel = {label: crosswalk.get(label, label) for label in parser["elements"]}
    assert len(set(relabel.values())) == len(relabel), "relabelling must stay bijective"

    # Every locant outside the crosswalk is fixed pointwise - no peripheral drift.
    assert all(relabel[label] == label for label in parser["elements"] if label not in crosswalk)

    assert {relabel[label]: symbol for label, symbol in parser["elements"].items()} == iupac["elements"]
    assert {tuple(sorted((relabel[left], relabel[right]))) for left, right in parser["edges"]} == {
        tuple(sorted(edge)) for edge in iupac["edges"]
    }


@pytest.mark.parametrize("record", SYSTEMS)
def test_only_interior_carbons_are_converted(record):
    """An interior heteroatom keeps its integer; a peripheral atom never moves."""

    interior = set(record["interior_locants"])
    elements = record["parser_graph"]["elements"]
    assert set(record["crosswalk"]) <= interior
    assert all(elements[label] == "C" for label in record["crosswalk"])
    unconverted = [label for label in interior if label not in record["crosswalk"]]
    assert all(elements[label] != "C" for label in unconverted), (
        "an interior carbon left unconverted would keep a superseded continuation label"
    )


@pytest.mark.parametrize("record", SYSTEMS)
def test_every_conversion_carries_its_own_anchor_proof(record):
    """Recompute the rule from the frozen graph; do not trust the recorded anchor."""

    for certificate in record["certificates"]:
        parser_locant = certificate["parser_locant"]
        shortest, closest = _nearest_anchor(record, parser_locant)
        assert shortest == certificate["shortest_distance_to_periphery"]
        assert closest == certificate["all_closest_peripheral_locants"]
        expected = f"{closest[0]}{str(shortest).translate(SUPERSCRIPT)}"
        assert expected == certificate["iupac_locant"] == record["crosswalk"][parser_locant]


def test_the_published_pyrene_conversion_is_reproduced():
    """FR-5.5.2 prints this one, so it is the positive control."""

    record = next(r for r in FIXTURE["systems"] if r["base_name"] == "benzo[1,2,3,4-def]phenanthrene")
    assert record["crosswalk"] == {"10b": "3a¹", "10c": "5a¹"}
    assert _nearest_anchor(record, "10b")[1] == ["3a", "10a"]
    assert _nearest_anchor(record, "10c")[1] == ["5a", "8a"]


def test_a_wrong_anchor_is_rejected_even_though_the_graphs_still_correspond():
    """The negative control. Correspondence alone cannot catch this."""

    phenalene = {
        "parser_graph": {
            "elements": dict.fromkeys(("1", "2", "3", "3a", "4", "5", "6", "6a", "7", "8", "9", "9a", "9b"), "C"),
            "edges": [
                ("1", "2"),
                ("2", "3"),
                ("3", "3a"),
                ("3a", "4"),
                ("4", "5"),
                ("5", "6"),
                ("6", "6a"),
                ("6a", "7"),
                ("7", "8"),
                ("8", "9"),
                ("9", "9a"),
                ("9a", "1"),
                ("3a", "9b"),
                ("6a", "9b"),
                ("9a", "9b"),
            ],
        },
        "peripheral_locants": ["1", "2", "3", "3a", "4", "5", "6", "6a", "7", "8", "9", "9a"],
    }
    shortest, closest = _nearest_anchor(phenalene, "9b")
    assert (shortest, closest) == (1, ["3a", "6a", "9a"])
    rule_gives = f"{closest[0]}{str(shortest).translate(SUPERSCRIPT)}"
    assert rule_gives == "3a¹"
    assert rule_gives != "6a¹", "the equally near 3a is lower, so 6a-superscript-one is wrong"


def test_the_fixture_records_the_parser_it_was_reviewed_against():
    assert FIXTURE["parser"]["jar"] and FIXTURE["parser"]["jar_sha256_prefix"]
    assert FIXTURE["rule"].startswith("P-25.3.3.3")
