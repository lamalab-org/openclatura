"""Read additive coordination names that OPSIN cannot parse as a whole.

``[1,1'-bis(diphenylphosphino)ferrocene]tetracarbonylchromium(0)`` is a list of
ligands followed by the central atom and its oxidation number.  OPSIN has no
grammar for that, but it reads the two hard parts on their own: an organic
ligand written as a compound name, and ``chromium(0)``.  This module splits the
name, sends those parts to OPSIN, takes ligand-only words (``carbonyl``,
``chlorido``, ``aqua``) from a small table, and returns the complex as
disconnected components -- the form in which such complexes are usually drawn
in SMILES, without metal-ligand bonds.

The result therefore states *composition* (which metal, which oxidation state,
which ligands, how many), not which atoms bind the metal or the geometry.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass

# Ligand words that are not compound names.  Charged ligands carry their charge.
LIGAND_SMILES: dict[str, str] = {
    "carbonyl": "[C-]#[O+]",
    "aqua": "O",
    "ammine": "N",
    "nitrosyl": "[N]=O",
    "hydrido": "[H-]",
    "fluoro": "[F-]",
    "fluorido": "[F-]",
    "chloro": "[Cl-]",
    "chlorido": "[Cl-]",
    "bromo": "[Br-]",
    "bromido": "[Br-]",
    "iodo": "[I-]",
    "iodido": "[I-]",
    "cyano": "[C-]#N",
    "cyanido": "[C-]#N",
    "hydroxo": "[OH-]",
    "hydroxido": "[OH-]",
    "oxo": "[O-2]",
    "oxido": "[O-2]",
    "methyl": "[CH3-]",
    "cyclopentadienyl": "c1cc[cH-]c1",
}

_SIMPLE_MULTIPLIERS = {"di": 2, "tri": 3, "tetra": 4, "penta": 5, "hexa": 6, "hepta": 7, "octa": 8}
_COMPLEX_MULTIPLIERS = {"bis": 2, "tris": 3, "tetrakis": 4, "pentakis": 5, "hexakis": 6}
_BRACKETS = {"(": ")", "[": "]", "{": "}"}
_ROMAN = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7, "VIII": 8}

# Central-atom names in an anionic complex, where they are not element name + "ate".
ATE_NAMES: dict[str, str] = {
    "iron": "ferrate",
    "copper": "cuprate",
    "silver": "argentate",
    "gold": "aurate",
    "nickel": "nickelate",
    "cobalt": "cobaltate",
    "zinc": "zincate",
    "manganese": "manganate",
    "mercury": "mercurate",
    "tungsten": "tungstate",
    "molybdenum": "molybdate",
    "platinum": "platinate",
    "tantalum": "tantalate",
    "lanthanum": "lanthanate",
}

_OXIDATION_RE = re.compile(r"\((0|-?[IV]+|\d[+-]|[+-]\d?)\)$")
# Binding descriptors say how a ligand is attached, which the disconnected
# result does not record: ``η5-``, ``μ-``, ``-κ2P,P'``.
_HAPTO_RE = re.compile(r"[ημ]\d*-")
_KAPPA_RE = re.compile(r"-κ\d*[A-Za-z,']*")

OpsinParser = Callable[[Sequence[str]], Sequence[str]]


@dataclass(frozen=True)
class CoordinationName:
    """A coordination name split into its central atom and ligands."""

    metal: str
    oxidation_state: int | None
    ligands: tuple[tuple[str, int], ...]  # (ligand name as written, count)
    smiles: str

    @property
    def charge(self) -> int:
        from rdkit import Chem

        mol = Chem.MolFromSmiles(self.smiles)
        return sum(atom.GetFormalCharge() for atom in mol.GetAtoms())


def _metal_names() -> dict[str, str]:
    """Central-atom spellings -> element symbol, including the ``-ate`` forms of anionic complexes."""

    from rdkit import Chem

    table = Chem.GetPeriodicTable()
    names = {table.GetElementName(z).lower(): table.GetElementSymbol(z) for z in range(3, 104)}
    for name, symbol in list(names.items()):
        if name in ATE_NAMES:
            names[ATE_NAMES[name]] = symbol
        elif name.endswith("ium"):
            names[name[:-3] + "ate"] = symbol
        elif name.endswith("um"):
            names[name[:-2] + "ate"] = symbol
    return names


def _normalise(name: str) -> str:
    text = name.strip().replace("′", "'").replace("’", "'").replace("‐", "-").replace("–", "-")
    text = _HAPTO_RE.sub("", text)
    return _KAPPA_RE.sub("", text)


def _split_central_atom(text: str) -> tuple[str, str, int | None] | None:
    """Split ``<ligands><metal>[(oxidation number)]`` at the central atom."""

    oxidation: int | None = None
    match = _OXIDATION_RE.search(text)
    if match:
        token = match.group(1)
        if token.lstrip("-") in _ROMAN:
            oxidation = _ROMAN[token.lstrip("-")] * (-1 if token.startswith("-") else 1)
        elif token == "0":
            oxidation = 0
        else:
            digits = token.strip("+-") or "1"
            oxidation = int(digits) * (-1 if "-" in token else 1)
        text = text[: match.start()]
    lowered = text.lower()
    for metal_name, symbol in sorted(_metal_names().items(), key=lambda item: -len(item[0])):
        if lowered.endswith(metal_name):
            return text[: -len(metal_name)], symbol, oxidation
    return None


def _matching_close(text: str, start: int) -> int | None:
    stack = []
    for index in range(start, len(text)):
        char = text[index]
        if char in _BRACKETS:
            stack.append(_BRACKETS[char])
        elif stack and char == stack[-1]:
            stack.pop()
            if not stack:
                return index
    return None


def _leading_table_ligand(lowered: str) -> str | None:
    for word in sorted(LIGAND_SMILES, key=len, reverse=True):
        if lowered.startswith(word):
            return word
    return None


def _tokenise_ligands(text: str) -> list[tuple[str, int]] | None:
    """Read ``[ligand]tetracarbonyl...`` into ``(ligand, count)`` pairs, left to right."""

    ligands: list[tuple[str, int]] = []
    position = 0
    while position < len(text):
        if text[position] in "- ":
            position += 1
            continue
        rest = text[position:]
        lowered = rest.lower()

        if rest[0] in _BRACKETS:
            close = _matching_close(text, position)
            if close is None:
                return None
            ligands.append((text[position + 1 : close], 1))
            position = close + 1
            continue

        word = _leading_table_ligand(lowered)
        if word:
            ligands.append((word, 1))
            position += len(word)
            continue

        multiplied = False
        for multipliers in (_COMPLEX_MULTIPLIERS, _SIMPLE_MULTIPLIERS):
            for prefix, count in sorted(multipliers.items(), key=lambda item: -len(item[0])):
                if not lowered.startswith(prefix):
                    continue
                after = position + len(prefix)
                if after < len(text) and text[after] in _BRACKETS:
                    close = _matching_close(text, after)
                    if close is None:
                        return None
                    ligands.append((text[after + 1 : close], count))
                    position = close + 1
                    multiplied = True
                    break
                word = _leading_table_ligand(lowered[len(prefix) :])
                if word:
                    ligands.append((word, count))
                    position = after + len(word)
                    multiplied = True
                    break
            if multiplied:
                break
        if not multiplied:
            return None
    return ligands


def _opsin_candidates(ligand: str) -> list[str]:
    """Names to try in OPSIN for a ligand: as written, then its anion spelling (``acetato`` -> ``acetate``)."""

    candidates = [ligand]
    lowered = ligand.lower()
    if lowered.endswith("ato"):
        candidates.append(ligand[:-1] + "e")
    elif lowered.endswith("ido"):
        candidates.append(ligand[:-1] + "e")
    return candidates


def _default_opsin_parser(names: Sequence[str]) -> Sequence[str]:
    import tempfile
    import warnings
    from pathlib import Path

    from py2opsin import py2opsin

    with tempfile.TemporaryDirectory(prefix="openclatura_opsin_") as tmpdir, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return py2opsin(list(names), tmp_fpath=str(Path(tmpdir) / "input.txt"))


def parse_coordination_name(name: str, *, opsin: OpsinParser | None = None) -> CoordinationName | None:
    """Split a coordination name and return the complex as disconnected SMILES.

    Returns ``None`` when the name is not of the form ligands + central atom, or
    a ligand can be read neither from the table nor by OPSIN.
    """

    split = _split_central_atom(_normalise(name))
    if split is None:
        return None
    ligand_text, metal, oxidation = split
    ligands = _tokenise_ligands(ligand_text)
    if not ligands:
        return None

    resolved: dict[str, str] = {
        ligand: LIGAND_SMILES[ligand.lower()] for ligand, _ in ligands if ligand.lower() in LIGAND_SMILES
    }
    unresolved = [ligand for ligand, _ in ligands if ligand not in resolved]
    if unresolved:
        queries = [candidate for ligand in unresolved for candidate in _opsin_candidates(ligand)]
        answers = dict(zip(queries, (opsin or _default_opsin_parser)(queries), strict=True))
        for ligand in unresolved:
            smiles = next((answers[c] for c in _opsin_candidates(ligand) if answers.get(c)), None)
            if not smiles:
                return None
            resolved[ligand] = smiles

    metal_smiles = f"[{metal}]" if not oxidation else f"[{metal}{'+' if oxidation > 0 else '-'}{abs(oxidation)}]"
    components = [metal_smiles]
    for ligand, count in ligands:
        components.extend([resolved[ligand]] * count)
    return CoordinationName(
        metal=metal,
        oxidation_state=oxidation,
        ligands=tuple(ligands),
        smiles=".".join(components),
    )


def _counter_ion_smiles(word: str, opsin: OpsinParser) -> str | None:
    """Read one counter-ion or solvent word, allowing a multiplier OPSIN rejects (``trichloride``)."""

    candidates = [(word, 1)]
    lowered = word.lower()
    for prefix, count in _SIMPLE_MULTIPLIERS.items():
        if lowered.startswith(prefix) and len(word) > len(prefix):
            candidates.append((word[len(prefix) :], count))
    answers = opsin([candidate for candidate, _ in candidates])
    for (_, count), smiles in zip(candidates, answers, strict=True):
        if smiles:
            return ".".join([smiles] * count)
    return None


def parse_coordination_compound(name: str, *, opsin: OpsinParser | None = None) -> str | None:
    """Read a name whose words are one coordination entity plus ordinary counter-ion or solvent names.

    ``hexaamminecobalt(III) trichloride`` is the complex word and a word OPSIN
    reads on its own.  Returns disconnected SMILES, or ``None``.
    """

    words = _normalise(name).split()
    for index, word in enumerate(words):
        entity = parse_coordination_name(word, opsin=opsin)
        if entity is None:
            continue
        components = [entity.smiles]
        for other in words[:index] + words[index + 1 :]:
            other_smiles = _counter_ion_smiles(other, opsin or _default_opsin_parser)
            if other_smiles is None:
                return None
            components.append(other_smiles)
        return ".".join(components)
    return None


def _composition_key(smiles: str) -> Counter | None:
    """Canonical multiset of components, with drawing conventions for metals removed.

    A cyclopentadienide ring is drawn either as an aromatic anion beside a metal
    cation or, PubChem-style, as five radical carbons beside a neutral metal.
    Both are reduced to the neutral drawing: the ring loses its charge and its
    double bonds, and bare metal atoms lose their charge.
    """

    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    ring_info = mol.GetRingInfo()
    for ring in ring_info.AtomRings():
        atoms = [mol.GetAtomWithIdx(idx) for idx in ring]
        if not (
            len(ring) == 5
            and all(atom.GetAtomicNum() == 6 and atom.GetIsAromatic() for atom in atoms)
            and all(ring_info.NumAtomRings(idx) == 1 for idx in ring)
            and sorted(atom.GetFormalCharge() for atom in atoms) == [-1, 0, 0, 0, 0]
        ):
            continue
        for atom in atoms:
            hydrogens = atom.GetTotalNumHs()
            atom.SetFormalCharge(0)
            atom.SetIsAromatic(False)
            atom.SetNoImplicit(True)
            atom.SetNumExplicitHs(hydrogens)
            atom.SetNumRadicalElectrons(1)
        for position, idx in enumerate(ring):
            bond = mol.GetBondBetweenAtoms(idx, ring[(position + 1) % 5])
            bond.SetBondType(Chem.BondType.SINGLE)
            bond.SetIsAromatic(False)
    for atom in mol.GetAtoms():
        if atom.GetDegree() == 0 and atom.GetTotalNumHs() == 0 and atom.GetFormalCharge() > 0:
            atom.SetFormalCharge(0)
    try:
        rendered = Chem.MolToSmiles(mol)
        reparsed = Chem.MolFromSmiles(rendered)
        if reparsed is None:
            return None
        return Counter(Chem.MolToSmiles(reparsed).split("."))
    except Exception:
        return None


def composition_equivalent(smiles_a: str, smiles_b: str) -> bool:
    """Whether two disconnected drawings contain the same metal and ligand components."""

    key_a, key_b = _composition_key(smiles_a), _composition_key(smiles_b)
    return key_a is not None and key_a == key_b


__all__ = [
    "ATE_NAMES",
    "LIGAND_SMILES",
    "CoordinationName",
    "composition_equivalent",
    "parse_coordination_compound",
    "parse_coordination_name",
]
