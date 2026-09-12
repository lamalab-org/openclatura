"""Independent, reproducible graph inputs for fusion stress tests.

These are catacondensed edge-fused systems, not a sampler of every polycycle.
RDKit validity does not imply stability or synthetic accessibility.
"""

from collections import Counter
from dataclasses import dataclass
from random import Random

from rdkit import Chem, rdBase


@dataclass(frozen=True)
class RandomFusionCase:
    index: int
    faces: tuple[tuple[int, ...], ...]
    binary: bytes
    smiles: str

    @property
    def id(self) -> str:
        return f"case-{self.index:03d}-{len(self.faces)}-rings"


def _graph(rng: Random, ring_count: int):
    graph = Chem.RWMol()

    def atom():
        value = Chem.Atom("C")
        value.SetIsAromatic(True)
        return graph.AddAtom(value)

    def edge(a, b):
        graph.AddBond(a, b, Chem.BondType.AROMATIC)

    faces = [tuple(atom() for _ in range(6))]
    for a, b in zip(faces[0], faces[0][1:] + faces[0][:1], strict=True):
        edge(a, b)
    while len(faces) < ring_count:
        # Sharing only a peripheral C-C edge with degree-two endpoints prevents
        # spiro junctions, overlapping rings and over-coordinated fusion atoms.
        available = [
            (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
            for bond in graph.GetBonds()
            if all(a.GetDegree() == 2 and a.GetSymbol() == "C" for a in (bond.GetBeginAtom(), bond.GetEndAtom()))
        ]
        if not available:
            return None
        a, b = rng.choice(available)
        size = rng.choice((5, 6))
        new = [atom() for _ in range(size - 2)]
        if size == 5:
            donor = graph.GetAtomWithIdx(new[1])
            donor.SetAtomicNum(rng.choice((7, 8, 16)))
            if donor.GetAtomicNum() == 7:
                donor.SetNumExplicitHs(1)
        path = [a, *new, b]
        for left, right in zip(path, path[1:]):
            edge(left, right)
        faces.append(tuple(path))

    candidates = [a.GetIdx() for a in graph.GetAtoms() if a.GetSymbol() == "C" and a.GetDegree() == 2]
    for idx in rng.sample(candidates, rng.randint(1, max(1, len(candidates) // 4))):
        graph.GetAtomWithIdx(idx).SetAtomicNum(7)
    mol = graph.GetMol()
    with rdBase.BlockLogs():
        if Chem.SanitizeMol(mol, catchErrors=True) != Chem.SanitizeFlags.SANITIZE_NONE:
            return None
    if any(a.GetFormalCharge() or a.GetNumRadicalElectrons() for a in mol.GetAtoms()):
        return None
    return mol, tuple(faces)


def random_fusion_cases(count=100, seed=20260907):
    """Keep unique neutral graphs, without querying the namer or OPSIN."""
    rng = Random(seed)
    seen = set()
    result = []
    for _ in range(count * 100):
        rings = 5 + len(result) % 6
        built = _graph(rng, rings)
        if built is None:
            continue
        mol, faces = built
        smiles = Chem.MolToSmiles(mol)
        if smiles in seen:
            continue
        seen.add(smiles)
        result.append(RandomFusionCase(len(result), faces, mol.ToBinary(), smiles))
        if len(result) == count:
            return tuple(result)
    raise AssertionError(f"Generated only {len(result)}/{count} unique neutral fused graphs")


def validate_case(case):
    mol = Chem.Mol(case.binary)
    Chem.SanitizeMol(mol)
    assert len(Chem.GetMolFrags(mol)) == 1
    assert all(a.GetFormalCharge() == 0 and a.GetNumRadicalElectrons() == 0 for a in mol.GetAtoms())
    assert any(a.GetAtomicNum() != 6 for a in mol.GetAtoms())
    assert mol.GetNumBonds() - mol.GetNumAtoms() + 1 == len(case.faces)
    assert 5 <= len(case.faces) <= 10
    edges = Counter()
    for face in case.faces:
        assert len(face) in (5, 6)
        for a, b in zip(face, face[1:] + face[:1], strict=True):
            assert mol.GetBondBetweenAtoms(a, b) is not None
            edges[tuple(sorted((a, b)))] += 1
    assert max(edges.values()) == 2
    assert sum(n == 2 for n in edges.values()) == len(case.faces) - 1
    assert len(edges) == mol.GetNumBonds()
