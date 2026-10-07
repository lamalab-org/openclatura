"""P-58.2.1.2 chooses the reference realization, not just how one is spelled.

A hydrogenated target can often be derived from more than one mancude
realization of the same retained parent, each preserving the reference maximum
and each reproducing the target:

    4H-indene  + hydro at 3a,5
    3aH-indene + hydro at 4,5

Both describe C=1C=CC2CCC=CC12, and they differ in which part of the molecule's
saturation belongs to the reference and which the hydro prefix states. The rule
prefers the lowest NONFUSION position for indicated hydrogen, and that
preference comes before the locant's own value - so 3a,5-dihydro-4H-indene is
preferred although 3a is the lower locant. Selecting a realization first and
treating it as authoritative stops the rule from operating at all, which is why
these tests assert the committed state and not only the rendered name.

A fusion site is not inadmissible, it merely follows: where only a fusion site
can hold the citation, it keeps it. The unsubstituted control below pins that,
so the preference cannot harden into a prohibition.
"""

import re

import pytest
from rdkit import Chem

from openclatura import name_mol, opsin_available

pytestmark = pytest.mark.skipif(not opsin_available(), reason="the reference structures come from OPSIN")

# The selected realization is observable in the name: the indicated-hydrogen
# run and the hydro locants together say which reference was used. The two
# targets reach that selection by different routes - indene's template declares
# an indicated hydrogen and carries the choice in its metadata, while
# carbazole's declares none and the choice is made when the citation is
# relocated - so the preference has to hold on both paths, not one.
CASES = (
    pytest.param("3a,5-dihydro-4H-indene", {"4"}, {"3a", "5"}, ("4",), id="nonfusion-beats-the-lower-locant"),
    pytest.param("2,3,4,9-tetrahydro-1H-carbazole", {"1"}, {"2", "3", "4", "9"}, None, id="site-class-ties"),
    pytest.param("3aH-indene", {"3a"}, set(), ("3a",), id="a-fusion-site-keeps-it-when-alone"),
)


def _allocation(name):
    """Split a name into its indicated-hydrogen run and its hydro locants."""

    indicated = set(re.findall(r"(\d+[a-z]?)H", name))
    hydro = set()
    match = re.search(r"([\d,a-z]+)-(?:di|tri|tetra|penta|hexa|hepta|octa|deca|dodeca)?hydro", name)
    if match:
        hydro = {part for part in match.group(1).split(",") if part}
    return indicated, hydro


def _committed(target, monkeypatch):
    """Return (metadata indicated H, final name, smiles) for the committed parent."""

    from py2opsin import py2opsin

    import openclatura.subgraph_tools as subgraph_tools

    smiles = py2opsin(target)
    assert smiles, f"OPSIN could not build {target}"
    original = subgraph_tools.add_replacement_prefixes
    captured = {}

    def capture(mol, parts, numbered_path, get_loc):
        metadata = parts.retained_parent_metadata
        captured.setdefault("h", getattr(metadata, "default_indicated_h", None))
        return original(mol, parts, numbered_path, get_loc)

    monkeypatch.setattr(subgraph_tools, "add_replacement_prefixes", capture)
    result = name_mol(Chem.MolFromSmiles(smiles))
    return captured.get("h"), result.name, smiles


@pytest.mark.parametrize("target,indicated,hydro,metadata_h", CASES)
def test_the_selected_realization_is_the_preferred_one(target, indicated, hydro, metadata_h, monkeypatch):
    committed, rendered, _ = _committed(target, monkeypatch)

    assert rendered == target
    assert _allocation(rendered) == (indicated, hydro)
    if metadata_h is not None:
        # Where the template declares the citation, the metadata has to record
        # the realization that was chosen - not the one offered first.
        assert committed == metadata_h


@pytest.mark.parametrize("target,indicated,hydro,metadata_h", CASES)
def test_the_preference_does_not_depend_on_enumeration_order(target, indicated, hydro, metadata_h, monkeypatch):
    """Reversing the input atom order must not change the winner.

    A comparator that takes whichever candidate was generated first passes the
    case above and fails here.
    """

    _, rendered, smiles = _committed(target, monkeypatch)
    graph = Chem.MolFromSmiles(smiles)
    reversed_order = list(reversed(range(graph.GetNumAtoms())))

    assert name_mol(Chem.RenumberAtoms(graph, reversed_order)).name == rendered
