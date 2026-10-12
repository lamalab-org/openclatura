"""Shared access to the reviewed interior-carbon locant crosswalks.

The reference parser continues fusion letters into the interior; P-25.3.3.3
numbers an interior carbon from its nearest peripheral atom with the bond
distance as a superscript. Tests that compare the two numberings consume the
reviewed correspondence from tests/data/interior_locant_crosswalk.json rather
than deriving one at run time, so a peripheral numbering error can never be
absorbed by a relabelling discovered to make the comparison succeed.

test_interior_locant_crosswalk.py is what proves those fixtures: bijective,
element- and edge-preserving, pointwise fixed outside the crosswalk, and each
conversion carrying an anchor certificate recomputed from the parser graph.
"""

import json
import pathlib
import re

_FIXTURE = json.loads(pathlib.Path("tests/data/interior_locant_crosswalk.json").read_text())
_BY_BASE = {record["base_name"]: record for record in _FIXTURE["systems"]}

# A base name may be cited with or without its indicated hydrogen, and the
# crosswalk describes the ring system either way.
_INDICATED_HYDROGEN = re.compile(r"^(?:\d+[a-z]*[\u00b9\u00b2\u00b3]?H,)*\d+[a-z]*[\u00b9\u00b2\u00b3]?H-")
_BY_SKELETON = {}
for _record in _FIXTURE["systems"]:
    _BY_SKELETON.setdefault(_INDICATED_HYDROGEN.sub("", _record["base_name"]), _record)


def reviewed_crosswalk(base_name):
    """Return {parser locant: IUPAC locant} for a base name, or None."""

    record = _BY_BASE.get(base_name) or _BY_SKELETON.get(_INDICATED_HYDROGEN.sub("", base_name))
    return dict(record["crosswalk"]) if record else None


def to_iupac_labels(base_name, parser_labels):
    """Relabel a parser-numbered collection into this engine's numbering.

    Returns None when no reviewed crosswalk covers the system, so a caller can
    keep the case failing rather than comparing against a guess.
    """

    crosswalk = reviewed_crosswalk(base_name)
    if crosswalk is None:
        return None
    return {crosswalk.get(label, label) for label in parser_labels}


def to_iupac_edges(base_name, parser_edges):
    """Relabel parser-numbered edges, as a set of frozensets."""

    crosswalk = reviewed_crosswalk(base_name)
    if crosswalk is None:
        return None
    return {frozenset(crosswalk.get(label, label) for label in edge) for edge in parser_edges}


def covered_base_names():
    return frozenset(_BY_BASE)
