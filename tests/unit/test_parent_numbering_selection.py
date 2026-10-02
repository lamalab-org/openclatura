import pytest

import openclatura.numbering as numbering
from openclatura.graph_io import read_smiles


def _choose(mol, maps):
    return numbering.choose_parent_numbering(
        mol,
        candidate_paths=[],
        principal_atoms={2},
        substituent_mapping={3: ["methyl"]},
        locant_maps=maps,
        is_ring=True,
        is_bicycle=False,
        is_spiro=False,
        is_polycycle=False,
        retained_name=None,
    )


@pytest.mark.parametrize("container", [list, tuple])
def test_single_map_matches_scored_selection_and_preserves_identity_and_order(monkeypatch, container):
    mol = read_smiles("N1CCCC1")
    locants = {3: "4", 1: "2", 4: "5", 0: "1", 2: "3"}
    expected = _choose(mol, [locants, dict(locants)])

    def unexpected_score(*args, **kwargs):
        raise AssertionError("a single map must not be scored")

    monkeypatch.setattr(numbering, "parse_locant", unexpected_score)
    monkeypatch.setattr(numbering, "_is_saturated_ring_site", unexpected_score)
    monkeypatch.setattr(numbering, "_indicated_hydrogen_like_atoms", unexpected_score)
    actual = _choose(mol, container([locants]))

    assert actual == expected
    assert actual[0] == [3, 1, 4, 0, 2]
    assert actual[1] is locants


@pytest.mark.parametrize("reverse", [False, True])
def test_multiple_maps_still_select_by_locant_preference(reverse):
    mol = read_smiles("N1CCCC1")
    preferred = {atom: str(atom + 1) for atom in mol.atoms}
    other = {atom: str(5 - atom) for atom in mol.atoms}
    maps = [other, preferred]
    if reverse:
        maps.reverse()

    path, selected = _choose(mol, maps)

    assert selected is preferred
    assert path == list(preferred)


def test_empty_maps_still_use_normal_numbering(monkeypatch):
    mol = read_smiles("N1CCCC1")
    expected = [0, 4, 3, 2, 1]
    calls = []

    def numbered(*args, **kwargs):
        calls.append(args)
        return expected

    monkeypatch.setattr(numbering, "number_parent", numbered)

    assert _choose(mol, []) == (expected, None)
    assert len(calls) == 1
