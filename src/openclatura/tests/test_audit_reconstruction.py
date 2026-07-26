"""Tests for the OPSIN-free reconstruction self-audit (``openclatura.audit``)."""

from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path

import pytest

import openclatura as oc
import openclatura.audit as audit
import openclatura.component_namer as component_namer
from openclatura.audit import ReconstructionAudit, audit_component_reconstruction, self_audit
from openclatura.audit.relative_stereo import ring_face_relation
from openclatura.audit.substituent_reconstruction import resolve_fragment_mol


def _canonical(mol) -> str | None:
    from rdkit import Chem

    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


# --------------------------------------------------------------------------- #
# Substituent grammar: compositional operators, unlocanted prefixes, von Baeyer
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "name,expected",
    [
        ("benzyl", "*Cc1ccccc1"),
        ("tert-butyl", "*C(C)(C)C"),
        ("phenoxy", "*Oc1ccccc1"),
        ("(benzyloxy)", "*OCc1ccccc1"),
        ("(chloromethyl)", "*CCl"),
        ("(difluoromethyl)", "*C(F)F"),
        ("(ethylamino)", "*NCC"),
        ("(diethylamino)", "*N(CC)CC"),
        ("(phenylsulfonyl)", "*S(=O)(=O)c1ccccc1"),
        ("(4-methylphenylsulfonyl)", "*S(=O)(=O)c1ccc(C)cc1"),
        ("(ethoxycarbonyl)", "*C(=O)OCC"),
        ("(3-chloro-4-methoxyphenyl)", "*c1ccc(OC)c(Cl)c1"),
        ("naphthalen-1-yl", "*c1cccc2ccccc12"),
        # N-substituted azoles: substituting the pyrrole-type N consumes its H
        ("(1-methyl-1H-pyrazol-4-yl)", "*c1cnn(C)c1"),
        ("(1-phenyl-1H-pyrazol-4-yl)", "*c1cnn(-c2ccccc2)c1"),
        ("(1-methyl-1H-imidazol-2-yl)", "*c1nccn1C"),
        # multiplied non-leaf base + amide leaf
        ("(diphenylmethyl)", "*C(c1ccccc1)c1ccccc1"),
        ("benzamido", "*NC(=O)c1ccccc1"),
        # von Baeyer fused-ring substituents (validated against OPSIN)
        ("bicyclo[2.2.1]heptan-2-yl", "*C1CC2CCC1C2"),
        ("(7,9-dioxabicyclo[4.3.0]nona-1(6),2,4-trien-3-yl)", "*c1ccc2c(c1)OCO2"),
        ("(2-azabicyclo[4.4.0]deca-1,3,5,7,9-pentaen-10-yl)", "*c1cccc2cccnc12"),
        # oxo/methyl-decorated von Baeyer cores (front modifiers peeled + grafted)
        ("(7,9-dioxo-8-azabicyclo[4.3.0]nona-1,3,5-trien-8-yl)", "*N1C(=O)c2ccccc2C1=O"),
        # Hantzsch-Widman replacement monocycles (validated against OPSIN)
        ("1-azacycloheptan-1-yl", "*N1CCCCCC1"),
        ("1-oxa-4-azacyclohexan-4-yl", "*N1CCOCC1"),
        ("(5-ethyl-1-thia-3,4-diazacyclopenta-2,4-dien-2-yl)", "*c1nnc(CC)s1"),
        ("(4,6-dimethyl-2-oxo-1-azacyclohexa-3,5-dien-3-yl)", "*c1c(C)cc(C)[nH]c1=O"),
        # <acyl-ring>carboxamido = parent-NH-C(=O)-ring (extra carbonyl carbon)
        ("(furan-2-carboxamido)", "*NC(=O)c1ccco1"),
        ("(pyridine-3-carboxamido)", "*NC(=O)c1cccnc1"),
        ("cyclopropanecarboxamido", "*NC(=O)C1CC1"),
        ("(2-fluorobenzene-1-carboxamido)", "*NC(=O)c1ccccc1F"),
        ("(3,4-dichlorobenzene-1-carboxamido)", "*NC(=O)c1ccc(Cl)c(Cl)c1"),
        # nitrilo (terminal nitrile) + isocyanide/isocyanate leaves
        ("(nitrilomethyl)", "*C#N"),
        ("(3-nitrilopropyl)", "*CCC#N"),
        ("(3-(nitrilomethyl)phenyl)", "*c1cccc(C#N)c1"),
        ("isocyano", "*[N+]#[C-]"),
        ("isocyanato", "*N=C=O"),
        # ylidene: attachment via a double bond (…yl fragment, bond promoted)
        ("(phenylmethylidene)", "*=Cc1ccccc1"),
        ("(propan-2-ylidene)", "*=C(C)C"),
        ("(diphenylmethylidene)", "*=C(c1ccccc1)c1ccccc1"),
        ("(diaminomethylidene)", "*=C(N)N"),
        ("((furan-2-yl)methylidene)", "*=Cc1ccco1"),
        ("((phenylmethylidene)amino)", "*N=Cc1ccccc1"),  # Schiff base
        # silyl / phosphoryl / boryl / thio leaves (+ composed oxy/methyl)
        ("(trimethylsilyl)", "*[Si](C)(C)C"),
        ("(((tert-butyl)dimethylsilyl)oxy)", "*O[Si](C)(C)C(C)(C)C"),
        ("(dimethoxyphosphoryl)", "*P(=O)(OC)OC"),
        ("(dihydroxyboryl)", "*B(O)O"),
        ("(phenylselanyl)", "*[Se]c1ccccc1"),
        # disubstituted amino (two distinct ligands), carbamoyl, imino operators
        ("((methyl)(phenyl)amino)", "*N(C)c1ccccc1"),
        ("((benzyl)(methyl)amino)", "*N(C)Cc1ccccc1"),
        ("(phenylcarbamoyl)", "*C(=O)Nc1ccccc1"),
        ("(ethylimino)", "*=NCC"),
        ("((amino)(imino)methyl)", "*C(N)=N"),  # amidine via (A)(B)methyl
        ("(cyclohex-1-en-1-yl)", "*C1=CCCCC1"),  # cycloalkenyl
        ("(cyclohexa-2,4-dien-1-yl)", "*C1C=CC=CC1"),
        ("(propionyl)", "*C(=O)CC"),
        # chain acylamino (<alkanoyl>amido) with front-modifier grafting
        ("propanamido", "*NC(=O)CC"),
        ("(2,2-dimethylpropanamido)", "*NC(=O)C(C)(C)C"),  # pivalamido
        ("(2-phenylacetamido)", "*NC(=O)Cc1ccccc1"),
        ("(prop-2-enamido)", "*NC(=O)C=C"),
        # adamantane cage (bridgehead 1-yl and bridge 2-yl)
        ("adamantan-1-yl", "*C12CC3CC(CC(C3)C1)C2"),
        ("adamantan-2-yl", "*C1C2CC3CC(C2)CC1C3"),
        # chain acyl: systematic -oyl, retained -yl, and unsaturated forms
        ("(hexanoyl)", "*C(=O)CCCCC"),
        ("(butyryl)", "*C(=O)CCC"),
        ("(valeryl)", "*C(=O)CCCC"),
        ("(2-methylpropanoyl)", "*C(=O)C(C)C"),
        ("(prop-2-enoyl)", "*C(=O)C=C"),
        # …with front modifiers grafted onto the acid chain (never onto C1)
        ("(2-(4-chlorophenyl)-acetyl)", "*C(=O)Cc1ccc(Cl)cc1"),
        ("(2-ethylbutyryl)", "*C(=O)C(CC)CC"),
        ("(1-butyrylpiperidin-4-yl)", "*C1CCN(C(=O)CCC)CC1"),
        ("(1-(2-phenylacetyl)piperidin-4-yl)", "*C1CCN(C(=O)Cc2ccccc2)CC1"),
        ("(3-(2-phenylacetyl)-3-azabicyclo[3.1.0]hexan-6-yl)", "*C1C2CN(C(=O)Cc3ccccc3)CC12"),
        # <ring>carboxamido where the acid stem keeps its terminal "e" before the locant
        ("(piperidine-4-carboxamido)", "*NC(=O)C1CCNCC1"),
        ("(bicyclo[3.1.0]hexane-6-carboxamido)", "*NC(=O)C1C2CCCC12"),
        ("(3-methyl-3-azabicyclo[3.1.0]hexane-6-carboxamido)", "*NC(=O)C1C2CN(C)CC12"),
        # N-substituted amide prefixes: the extra group lands on the amide nitrogen
        ("(N-methylacetamido)", "*N(C)C(C)=O"),
        ("(N-methylbenzamido)", "*N(C)C(=O)c1ccccc1"),
        ("(N-methylpyridine-3-carboxamido)", "*N(C)C(=O)c1cccnc1"),
        ("(N-methyl-3-methylbutanamido)", "*N(C)C(=O)CC(C)C"),
        # monospiro rings: numbering runs round the smaller ring first
        ("(spiro[3.3]heptan-2-yl)", "*C1CC2(CCC2)C1"),
        ("(2-azaspiro[3.3]heptan-2-yl)", "*N1CC2(CCC2)C1"),
        ("(5-azaspiro[2.4]heptan-5-yl)", "*N1CCC2(CC2)C1"),
        ("(5,8-dioxaspiro[3.5]nonan-2-yl)", "*C1CC2(C1)COCCO2"),
        ("(2-oxaspiro[3.3]heptan-6-yl)", "*C1CC2(C1)COC2"),
        # contracted Hantzsch-Widman monocycles, derived from the name's own
        # morphology (ring size from the stem, heteroatoms from the prefix)
        ("(thian-4-yl)", "*C1CCSCC1"),
        ("(1,4-dioxan-2-yl)", "*C1COCCO1"),
        ("(1,3-dioxolan-2-yl)", "*C1OCCO1"),
        ("((4R)-2,2-dimethyl-1,3-dioxolan-4-yl)", "*C1COC(C)(C)O1"),
        ("(1,4-dithian-2-yl)", "*C1CSCCS1"),
        ("(azepan-1-yl)", "*N1CCCCCC1"),
        ("(1,4-diazepan-1-yl)", "*N1CCCNCC1"),
        ("(oxepan-2-yl)", "*C1CCCCCO1"),
        ("(azocan-1-yl)", "*N1CCCCCCC1"),
        ("(2-methyl-1,3-dioxolan-2-yl)", "*C1(C)OCCO1"),
        # N-substituted amides whose acyl half opens with a stereo descriptor
        ("(N-methyl(2S,3R)-2-fluoro-3-(trifluoromethyl)butanamido)", "*N(C)C(=O)C(F)C(C)C(F)(F)F"),
        ("(N-methyl(1R,3S,5S)-bicyclo[3.1.0]hexane-3-carboxamido)", "*N(C)C(=O)C1CC2CC2C1"),
        ("((3S)-3-(N-methyl(2S,3R)-2-fluoro-3-(trifluoromethyl)butanamido)butyl)", "*CCC(C)N(C)C(=O)C(F)C(C)C(F)(F)F"),
        # lambda-convention valences: the annotation rides on the locant and the
        # extra bonds come from the oxo prefixes
        ("(1,1-dioxo-1lambda^6-thiacyclohexan-3-yl)", "*C1CCCS(=O)(=O)C1"),
        ("(3,3-dioxo-3lambda^6-thia-7-azabicyclo[3.3.0]octan-1-yl)", "*C12CNCC1CS(=O)(=O)C2"),
        # indicated hydrogen away from position 1 is a different N-H tautomer
        ("(2H-tetrazol-5-yl)", "*c1nn[nH]n1"),
        ("(1H-tetrazol-5-yl)", "*c1nnn[nH]1"),
        ("(2H-indazol-5-yl)", "*c1ccc2n[nH]cc2c1"),
        ("(4H-1,2,4-triazol-3-yl)", "*c1nnc[nH]1"),
        # heteroatom hubs carrying a ligand list
        ("((ethoxy)(methyl)phosphoryl)", "*P(=O)(OCC)C"),
        ("(dimethyloxophosphanyl)", "*P(C)(C)=O"),
        ("((2-(formyl)phenyl)(4-chlorophenyl)(methyl)silyl)", "*[Si](C)(c1ccc(Cl)cc1)c1ccccc1C=O"),
        # multiplied unsaturation, whose locant count must match the multiplier
        ("(deca-1,8-dien-1-yl)", "*C=CCCCCCC=CC"),
        ("(nona-2,4,6,8-tetraen-1-yl)", "*CC=CC=CC=CC=C"),
        ("(hexa-1,2,5-trien-1-yl)", "*C=C=CCC=C"),
        ("(1-oxohexa-3,5-dien-1-yl)", "*C(=O)CC=CC=C"),
    ],
)
def test_resolve_fragment_grammar(name, expected):
    from rdkit.Chem import CanonSmiles

    assert _canonical(resolve_fragment_mol(name)) == CanonSmiles(expected)


@pytest.mark.parametrize(
    "name",
    [
        "(dispiro[3.1.3.1]decan-2-yl)",  # polyspiro: a different numbering rule
        "(azane)",  # 'az' + 'ane' is ammonia, not a six-membered ring
        "(deca-1,8-trien-1-yl)",  # multiplier disagrees with the cited locants
        # ``phenylacetyl`` puts the phenyl on C2, but an unlocanted prefix would
        # otherwise be placed on C1 — the acyl base withdraws C1 so this abstains
        # rather than reconstructing the wrong graph.
        "(phenylacetyl)",
    ],
)
def test_unresolvable_substituent_returns_none(name):
    # A construct outside the modelled grammar must abstain (None), never guess.
    assert resolve_fragment_mol(name) is None


# --------------------------------------------------------------------------- #
# Relative ring stereo: the cis/trans oracle
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "smiles,expected",
    [
        # 1,3-cyclobutane and 1,3-cyclopentane
        ("C[C@H]1C[C@@H](C)C1", "cis"),
        ("C[C@H]1C[C@H](C)C1", "trans"),
        ("C[C@H]1CC[C@@H](C)C1", "cis"),
        ("C[C@H]1CC[C@H](C)C1", "trans"),
        # 1,4-cyclohexane: a cis pair sits axial/equatorial in a chair, which is
        # why a mean-ring-plane test gets these wrong and parity does not
        ("C[C@H]1CC[C@@H](C)CC1", "cis"),
        ("C[C@H]1CC[C@H](C)CC1", "trans"),
        # 1,2-cyclohexane: cis is the meso (R,S) diastereomer
        ("C[C@H]1CCCC[C@H]1C", "cis"),
        ("C[C@H]1CCCC[C@@H]1C", "trans"),
        # heteroatom substituents, and a heteroatom in the ring
        ("N[C@H]1CC[C@@H](O)CC1", "cis"),
        ("N[C@H]1CC[C@H](O)CC1", "trans"),
        ("CS(=O)(=O)[C@H]1C[C@@H](N)C1", "cis"),
        ("CS(=O)(=O)[C@H]1C[C@H](N)C1", "trans"),
    ],
)
def test_ring_face_relation(smiles, expected):
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    ring = next(r for r in mol.GetRingInfo().AtomRings())
    pair = [
        a
        for a in ring
        if sum(1 for n in mol.GetAtomWithIdx(a).GetNeighbors() if n.GetIdx() not in ring and n.GetAtomicNum() > 1) == 1
    ]
    assert ring_face_relation(mol, pair[0], pair[1]) == expected


def test_ring_face_relation_agrees_with_cip_on_meso():
    # Independent cross-check of the two 1,2-dimethylcyclohexanes: the cis isomer
    # is meso, so modern CIP labels it (R,S) and the trans one (R,R)/(S,S).
    from rdkit import Chem
    from rdkit.Chem import rdCIPLabeler

    for smiles in ("C[C@H]1CCCC[C@H]1C", "C[C@H]1CCCC[C@@H]1C"):
        mol = Chem.MolFromSmiles(smiles)
        rdCIPLabeler.AssignCIPLabels(mol)
        codes = sorted(a.GetProp("_CIPCode") for a in mol.GetAtoms() if a.HasProp("_CIPCode"))
        ring = mol.GetRingInfo().AtomRings()[0]
        pair = [
            a
            for a in ring
            if any(n.GetIdx() not in ring and n.GetAtomicNum() > 1 for n in mol.GetAtomWithIdx(a).GetNeighbors())
        ]
        relation = ring_face_relation(mol, pair[0], pair[1])
        assert relation == ("cis" if codes == ["R", "S"] else "trans")


def test_ring_face_relation_abstains_without_parity():
    # No assigned parity -> not determinable, never a guess.
    from rdkit import Chem

    mol = Chem.MolFromSmiles("CC1CCC(C)CC1")
    ring = mol.GetRingInfo().AtomRings()[0]
    pair = [
        a
        for a in ring
        if any(n.GetIdx() not in ring and n.GetAtomicNum() > 1 for n in mol.GetAtomWithIdx(a).GetNeighbors())
    ]
    assert ring_face_relation(mol, pair[0], pair[1]) is None


@pytest.mark.parametrize(
    "smiles",
    [
        "C[C@H]1C[C@@H](C)C1",
        "N[C@H]1CC[C@@H](O)CC1",
        "CS(=O)(=O)[C@H]1C[C@H](N)C1",
        "COC(=O)[C@H]1CC[C@@H](CC(=O)O)CC1",
    ],
)
def test_ring_face_relation_inverts_with_one_centre(smiles):
    # Inverting exactly one centre must flip the relation; inverting both must
    # leave it alone.  These hold for any correct face oracle, so they pin the
    # parity arithmetic without needing external ground truth.
    from rdkit import Chem

    flip = {
        Chem.ChiralType.CHI_TETRAHEDRAL_CW: Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.ChiralType.CHI_TETRAHEDRAL_CCW: Chem.ChiralType.CHI_TETRAHEDRAL_CW,
    }
    mol = Chem.MolFromSmiles(smiles)
    ring = mol.GetRingInfo().AtomRings()[0]
    a, b = [
        idx
        for idx in ring
        if sum(1 for n in mol.GetAtomWithIdx(idx).GetNeighbors() if n.GetIdx() not in ring and n.GetAtomicNum() > 1)
        == 1
    ]
    base = ring_face_relation(mol, a, b)
    assert base is not None

    def inverted(*centres):
        rw = Chem.RWMol(mol)
        for centre in centres:
            atom = rw.GetAtomWithIdx(centre)
            atom.SetChiralTag(flip[atom.GetChiralTag()])
        return ring_face_relation(rw.GetMol(), a, b)

    assert inverted(a) != base
    assert inverted(b) != base
    assert inverted(a, b) == base


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _capture_top_level(smiles: str):
    """Name ``smiles`` and return (mol, component_atoms, parts) for the last
    top-level component named (enough for these single-component fixtures).

    The independent-CIP oracle and the retained source molecule are audit-only
    overhead the namer does not switch on by itself, so this enables them for the
    duration exactly as :func:`capture_component_audits` does — without them a
    captured component carries no stereo evidence to audit against."""

    from openclatura import graph_io

    captured: list[tuple] = []
    original = component_namer.assert_component_fully_named

    def spy(mol, atoms, parts, name):
        captured.append((mol, set(atoms), parts))
        return original(mol, atoms, parts, name)

    component_namer.assert_component_fully_named = spy
    previous_cip = graph_io._AUDIT_CIP_ENABLED
    graph_io.set_audit_cip_enabled(True)
    try:
        oc.name(smiles)
    finally:
        component_namer.assert_component_fully_named = original
        graph_io.set_audit_cip_enabled(previous_cip)
    assert captured, f"no component captured for {smiles!r}"
    return captured[-1]


# --------------------------------------------------------------------------- #
# Package surface
# --------------------------------------------------------------------------- #
def test_package_exposes_expected_api():
    for symbol in (
        "audit",
        "self_audit",
        "audit_component_reconstruction",
        "ReconstructionAudit",
        "resolve_fragment_mol",
        "audit_stereochemistry",
        "capture_component_audits",
        "aggregate_audits",
    ):
        assert hasattr(audit, symbol), symbol
    # The primary callable is the reconstruction audit.
    assert audit.audit is audit_component_reconstruction


# --------------------------------------------------------------------------- #
# Confirmation: a broad set of names the reconstructor fully models
# --------------------------------------------------------------------------- #
CONFIRMED_SMILES = [
    "C",  # methane
    "CCCCCC",  # hexane
    "CC(C)CC",  # 2-methylbutane
    "CC=CC",  # but-2-ene
    "CC#CC",  # but-2-yne
    "C=CC=C",  # buta-1,3-diene
    "CCO",  # ethanol
    "OCCO",  # ethane-1,2-diol
    "CC(=O)O",  # acetic acid
    "CCC#N",  # propionitrile
    "CC=O",  # ethanal
    "CC(=O)C",  # propanone
    "CCN",  # ethanamine
    "CC(=O)N",  # acetamide
    "CCS",  # ethanethiol
    "CCCCl",  # 1-chloropropane
    "ClCCBr",  # 1-bromo-2-chloroethane
    "BrC(Br)Br",  # tribromomethane
    "CCOCC",  # diethyl ether
    "c1ccccc1",  # benzene
    "Cc1ccccc1O",  # 2-methylphenol
    "Oc1ccccc1",  # phenol
    "c1ccc(Cl)cc1",  # chlorobenzene
    "FC(F)(F)c1ccccc1",  # (trifluoromethyl)benzene
    "c1ccncc1",  # pyridine
    "c1ccncc1C",  # methylpyridine
    "c1ccoc1",  # furan
    "c1cc[nH]c1",  # pyrrole
    "C1CCCCC1",  # cyclohexane
    "O=C1CCCCC1",  # cyclohexanone
    "C1CCNCC1",  # piperidine
    "OCC1CCCCC1",  # cyclohexylmethanol
    # compositional operators + retained stems + N-locants
    "OCc1ccccc1",  # benzyl (retained stem)
    "CCOc1ccccc1",  # phenoxy / -oxy operator
    "c1ccc(Oc2ccccc2)cc1",  # diphenyl ether
    "O=S(=O)(c1ccccc1)N",  # phenylsulfonyl operator
    "CC(C)Nc1ccccc1",  # anilino / amino operator
    "CNC(C)=O",  # N-methylacetamide (italic-N locant)
    "CN(C)C(C)=O",  # N,N-dimethylacetamide
    "CCC(C)C",  # 2-methylbutane via retained-ish
    "CC(=O)OC",  # methyl acetate (ester front-modifier)
    "CCOC(=O)c1ccccc1",  # ethyl benzoate (ring ester)
    # net-neutral charge separation (nitro / N-oxide) reconstructs with charges
    "O=[N+]([O-])c1ccccc1",  # nitrobenzene
    "Cc1ccc([N+](=O)[O-])cc1",  # 4-nitrotoluene
    # …while a sulfonyl written charge-separated denotes the same group as the
    # hypervalent spelling the reconstruction builds, so it must not read as a
    # disagreement (this refuted a correct name before the spellings converged)
    "CC(C)CCN1CC(CN[S+](=O)([O-])c2ccccc2)C2(C1)CN(c1ccccc1)C2",
    # fused retained + von Baeyer polycyclic parents
    "c1ccc2ccccc2c1",  # naphthalene (retained fused parent)
    "C1c2ccccc2-c2ccccc21",  # fluorene (tricyclo/polycyclic parent)
    "O=C1c2ccccc2-c2ccccc21",  # fluorenone
    "c1ccc2[nH]ccc2c1",  # indole (retained fused parent)
    # stereochemistry verified independently against the modern-CIP oracle
    "C[C@H](N)C(=O)O",  # (2S)-alanine (parent stereocentre)
    "C[C@@H](O)CC",  # (2R)-butan-2-ol (functional group on stereocentre)
    "C/C=C/C(=O)O",  # (2E)-but-2-enoic acid (parent E double bond)
    "N[C@@H](Cc1ccccc1)C(=O)O",  # phenylalanine (stereocentre beside a substituent)
    # substituent-embedded stereo verified via constitution isomorphism vs input CIP
    "CC[C@@H](C)c1ccc(S(=O)(=O)Nc2c(C)cc(C)cc2C)cc1",  # (2R)-butan-2-yl inside an N-substituent
    "Cc1cc(F)ccc1-c1noc([C@@H]2CCCCN2)n1",  # (2S)-piperidin-2-yl inside a ring substituent
    "CS(=O)(=O)OC[C@]1(c2ccccn2)CCNC[C@@H]1O",  # (3R,4R) piperidinylmethyl ester group
    "COCCNCc1cc(F)ccc1OC/C=C/Cl",  # (2E)-3-chloroprop-2-en-1-yl: substituent-embedded E/Z bond
    # cyclohexadienyl-SULFONYL: the trailing "yl" of "sulfonyl" must not be
    # swallowed as the ring attachment (would drop the SO2 and refute a correct name)
    "O=S(=O)(O)c1ccc(CS(=O)(=O)C2CC=C(F)C=C2F)cc1",
    # multi-amine principal groups: N / N' / N'' italic nitrogen locants
    "CNCCCNC",  # N,N'-dimethylpropane-1,3-diamine (symmetric)
    "CN(C)CCCN(C)C",  # N,N,N',N'-tetramethylpropane-1,3-diamine
    "CCNc1ccnc(NC)n1",  # N'-ethyl-N-methylpyrimidine-2,4-diamine (asymmetric: guards N/N' order)
    # retained saturated heterocycle / benzo-fused parents (substituted forms guard locants)
    "CC1NC(=O)NC1=O",  # 5-methylimidazolidine-2,4-dione
    "CN1CCSC1",  # 3-methyl-1,3-thiazolidine
    "O=C1CNC(=O)N1",  # imidazolidine-2,4-dione (hydantoin)
    "CC1Cc2ccccc2C1",  # 2-methylindane
    "CN1CCc2ccccc21",  # 1-methylindoline
    "CC12CC3CC(CC(C3)C1)C2",  # 1-methyladamantane
    # hub oxo-acids and acid halides (principal groups)
    "O=S(=O)(O)c1ccccc1",  # benzenesulfonic acid
    "O=C(Cl)c1ccccc1",  # benzoyl chloride
    "CCC(=O)Cl",  # propanoyl chloride
    # monospiro parents, with skeletal replacement and unsaturation
    "C1CC2(C1)CCC2",  # spiro[3.3]heptane
    "C1CC2(CC1)CCCCC2",  # spiro[4.5]decane
    "O=C(O)C1CC2(C1)CCOCC2",  # 7-oxaspiro[3.5]nonane-2-carboxylic acid
    "CC1=NOC2(C1)CCCC2",  # 3-methyl-1-oxa-2-azaspiro[4.4]non-2-ene
    "C1COCC11CNC1",  # 6-oxa-2-azaspiro[3.4]octane
    "O=C(O)C1CN(C(=O)C2CC3(CCC3)C2)C1",  # spiro[3.3]heptan-2-yl as a substituent
    # spiro *assemblies*: two named ring systems sharing one atom, cited with
    # primed locants for the side ring
    "OC1CC2OC1C21CC1",  # spiro[5-oxabicyclo[2.1.1]hexane-6,1'-cyclopropane]-2-ol
    "CC1C2C(=O)C1C21CN1",  # side ring from a retained template
    "CC1C2(C)NC2C11CN1",
    "C[C@H]1C(=O)[C@H]2CC[C@H]1CC21OCCO1",  # primed replacement prefixes on the side ring
    "CC[C@H]1C(=O)[C@@]2(c3ccccc31)N(c1ccccc1)C(=O)C2(C)C",  # primed prefixes + a side suffix
    # relative ring stereo: the name pins the configuration with a cis/trans word
    # rather than per-atom R/S, verified against the input's tetrahedral parities
    "C#CCCCOCCOCCC(=O)N[C@@H](C)CNC(=O)[C@H]1C[C@@H](S(C)(=O)=O)C1",  # cis-cyclobutane, via …carboxamido
    "CC[C@@H](O)CC(=O)N[C@H]1C[C@@H](C(=O)NCC(C)(C)F)C1",  # cis-cyclobutyl beside a parent (3R)
    "COC1(C(=O)N[C@H]2CC[C@H](NC(=O)[C@@H]3CCS(=O)(=O)N3)CC2)CCOCC1",  # trans-cyclohexyl
    # …and the same word on the *parent*, which the namer binds to its two ring
    # atoms directly rather than embedding in a substituent term
    "O=C(O)[C@H]1C[C@@H](c2cc(Cl)cc(-c3cnc4c(F)cccc4c3)c2)C1",  # cis-cyclobutanecarboxylic acid
    "C#CCCN(C)CC(=O)N1CCC[C@@H](N(C)C(=O)[C@H]2CC[C@H](CC)CC2)C1",  # trans-cyclohexanecarboxamide
    "C[C@@H](C[C@@H](C)NC(=O)[C@H]1C[C@@H](O)C1)NC(=O)CCc1cn[nH]c1",  # cis, beside parent (2R,4S)
    # Fused / small-ring centres the legacy CIP perception mislabelled; the
    # accurate labeller names them correctly, so they now confirm
    "CC[C@@H]1CC1C[C@@H]1CC1C[C@H]1C[C@H]1CCCCCCCC(C)=O",  # cyclopropane chain
    "CC(=O)[C@]1(C)CC[C@]2(C)CC[C@]3(C)C4=CCc5c(cc(O)c(O)c5C)[C@]4(C)CC[C@@]3(C)[C@@H]2C1",  # pentacyclic steroid
]


@pytest.mark.parametrize("smiles", CONFIRMED_SMILES)
def test_self_audit_confirms_modelled_names(smiles):
    result = self_audit(smiles)
    assert result.verdict == "confirmed", f"{smiles}: {result.verdict} ({result.reason})"
    assert result.ok
    assert bool(result) is True


@pytest.mark.parametrize(
    "smiles",
    [
        # Fused and small-ring centres, where the legacy CIP perception the namer
        # used to emit disagreed with the accurate labeller.  They now carry the
        # correct descriptors, so they confirm — and stay here as the regression
        # guard for that.
        "CC[C@@H]1CC1C[C@@H]1CC1C[C@H]1C[C@H]1CCCCCCCC(C)=O",
        "CC(=O)[C@]1(C)CC[C@]2(C)CC[C@]3(C)C4=CCc5c(cc(O)c(O)c5C)[C@]4(C)CC[C@@]3(C)[C@@H]2C1",
        "C[C@H](N)C(=O)O",
    ],
)
def test_corrupted_stereo_descriptor_is_not_confirmed(smiles):
    # Soundness: whatever the descriptors are derived from, one that does not
    # describe the atom it is bound to must block confirmation — never a
    # false-confirm, and never a false-mismatch either, since a descriptor
    # disagreement is not proof the constitution is wrong.
    mol, atoms, parts = _capture_top_level(smiles)
    assert audit_component_reconstruction(mol, parts, atoms).verdict == "confirmed"

    swap = {"R": "S", "S": "R", "r": "s", "s": "r"}
    bad = copy.deepcopy(parts)
    flipped = False
    for i, (locant, descriptor) in enumerate(bad.stereo_features):
        if descriptor in swap:
            bad.stereo_features[i] = (locant, swap[descriptor])
            flipped = True
            break
    if not flipped:
        for i, binding in enumerate(bad.name_atom_bindings):
            if binding.role == "absolute_stereo" and binding.term[-1] in swap:
                bad.name_atom_bindings[i] = replace(binding, term=binding.term[:-1] + swap[binding.term[-1]])
                flipped = True
                break
    if not flipped:
        # Descriptors embedded in a substituent term, e.g. "((2R)-butan-2-yl)".
        import re as _re

        for substituent in bad.substituents:
            replaced, count = _re.subn(
                r"(\d+)([RSrs])(?=[,)])",
                lambda m: m.group(1) + swap[m.group(2)],
                substituent.name,
                count=1,
            )
            if count:
                substituent.name = replaced
                flipped = True
                break
    assert flipped, f"no stereo descriptor to corrupt for {smiles}"
    assert audit_component_reconstruction(mol, bad, atoms).verdict == "abstained"


# --------------------------------------------------------------------------- #
# Abstention: honestly declines on constructs it does not model
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "smiles",
    [
        "COC(=O)N1N=CN=N1",  # 2H-tetrazole: indicated-hydrogen position not modelled
        "[Na+].[Cl-]",  # ionic: no auditable component
        # The name drops the isotopic label, so confirming it would certify a name
        # that does not denote the input.
        "[13CH4]",
        "[2H]C([2H])([2H])O",
    ],
)
def test_self_audit_abstains_on_unmodelled(smiles):
    result = self_audit(smiles)
    assert result.verdict == "abstained", f"{smiles}: {result.verdict}"
    assert not result.ok


# --------------------------------------------------------------------------- #
# Soundness: corrupting the name provably flips the verdict to mismatch
# --------------------------------------------------------------------------- #
def test_mutated_substituent_identity_is_caught():
    mol, atoms, parts = _capture_top_level("c1ccc(Cl)cc1")
    assert audit_component_reconstruction(mol, parts, atoms).verdict == "confirmed"
    bad = copy.deepcopy(parts)
    bad.substituents[0].name = "bromo"
    assert audit_component_reconstruction(mol, bad, atoms).verdict == "mismatch"


def test_mutated_substituent_locant_is_caught():
    mol, atoms, parts = _capture_top_level("Cc1ccccc1O")
    bad = copy.deepcopy(parts)
    bad.substituents[0].locants = ["4"]
    assert audit_component_reconstruction(mol, bad, atoms).verdict == "mismatch"


def test_dropped_substituent_is_caught():
    mol, atoms, parts = _capture_top_level("ClCCBr")
    bad = copy.deepcopy(parts)
    bad.substituents = bad.substituents[:1]
    assert audit_component_reconstruction(mol, bad, atoms).verdict == "mismatch"


@pytest.mark.parametrize(
    "smiles",
    [
        "C#CCCCOCCOCCC(=O)N[C@@H](C)CNC(=O)[C@H]1C[C@@H](S(C)(=O)=O)C1",
        "CC[C@@H](O)CC(=O)N[C@H]1C[C@@H](C(=O)NCC(C)(C)F)C1",
        "COC1(C(=O)N[C@H]2CC[C@H](NC(=O)[C@@H]3CCS(=O)(=O)N3)CC2)CCOCC1",
    ],
)
def test_flipped_relative_stereo_word_is_not_confirmed(smiles):
    # Non-vacuity for the cis/trans verification: swapping the word in the name
    # asserts the other diastereomer, which the input's parities contradict, so
    # the audit must stop confirming.
    mol, atoms, parts = _capture_top_level(smiles)
    assert audit_component_reconstruction(mol, parts, atoms).verdict == "confirmed"

    bad = copy.deepcopy(parts)
    swapped = 0
    for substituent in bad.substituents:
        if "cis-" in substituent.name:
            substituent.name = substituent.name.replace("cis-", "trans-")
            swapped += 1
        elif "trans-" in substituent.name:
            substituent.name = substituent.name.replace("trans-", "cis-")
            swapped += 1
    assert swapped == 1, f"expected exactly one cis/trans word to flip, flipped {swapped}"
    assert audit_component_reconstruction(mol, bad, atoms).verdict != "confirmed"


@pytest.mark.parametrize(
    "smiles",
    [
        "O=C(O)[C@H]1C[C@@H](c2cc(Cl)cc(-c3cnc4c(F)cccc4c3)c2)C1",
        "C#CCCN(C)CC(=O)N1CCC[C@@H](N(C)C(=O)[C@H]2CC[C@H](CC)CC2)C1",
    ],
)
def test_flipped_parent_relative_stereo_binding_is_not_confirmed(smiles):
    # Same non-vacuity check for the parent-level cis/trans binding.
    mol, atoms, parts = _capture_top_level(smiles)
    assert audit_component_reconstruction(mol, parts, atoms).verdict == "confirmed"

    bad = copy.deepcopy(parts)
    swap = {"cis": "trans", "trans": "cis"}
    flipped = 0
    for i, binding in enumerate(bad.name_atom_bindings):
        if binding.role == "relative_stereo" and binding.term in swap:
            bad.name_atom_bindings[i] = replace(binding, term=swap[binding.term])
            flipped += 1
    assert flipped == 1, f"expected one relative_stereo binding, found {flipped}"
    assert audit_component_reconstruction(mol, bad, atoms).verdict != "confirmed"


def test_spiro_side_that_is_not_a_ring_abstains():
    # A spiro union joins two rings.  When the cited side resolves to a chain the
    # name is not describing a rebuildable assembly, and inventing a structure to
    # compare against would manufacture a disagreement — so it must abstain.
    from rdkit import Chem

    from openclatura.assembly_parts import SubstituentItem
    from openclatura.audit.reconstruction import _Abstain, _apply_spiro_substituent
    from openclatura.spiro_assembly import SpiroAssembly

    rw = Chem.RWMol(Chem.MolFromSmiles("C1CCCCC1"))
    item = SubstituentItem(
        name="",
        locants=["1"],
        spiro=SpiroAssembly(parent_locant="1", side_locant="2", side_parent_name="propane"),
    )
    with pytest.raises(_Abstain):
        _apply_spiro_substituent(rw, {"1": 0}, item)


def test_wrong_principal_group_is_caught():
    mol, atoms, parts = _capture_top_level("CCO")
    bad = copy.deepcopy(parts)
    bad.principal_group = replace(bad.principal_group, key="thiol")
    assert audit_component_reconstruction(mol, bad, atoms).verdict == "mismatch"


def test_wrong_parent_length_is_caught():
    mol, atoms, parts = _capture_top_level("CCCCCC")
    bad = copy.deepcopy(parts)
    bad.parent_length = 5
    assert audit_component_reconstruction(mol, bad, atoms).verdict == "mismatch"


def test_wrong_unsaturation_locant_is_caught():
    mol, atoms, parts = _capture_top_level("CC=CC")
    bad = copy.deepcopy(parts)
    bad.unsaturations[0].locants = ["1"]
    assert audit_component_reconstruction(mol, bad, atoms).verdict == "mismatch"


# --------------------------------------------------------------------------- #
# Non-vacuity on a real defect class: a triple bond cannot live in a ring the
# namer calls "piperidine" (a saturated ring). The audit must refute it.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "smiles",
    [
        "CC1(C)C#CC(O)C(C)(C)N1CCO",
        "CC(C)C1CCC#CN1C(C)C",
    ],
)
def test_ring_alkyne_named_as_saturated_ring_is_caught(smiles):
    assert self_audit(smiles).verdict == "mismatch"


# --------------------------------------------------------------------------- #
# Charge-separated spellings converge; obligatory charge separation does not
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "written,hypervalent",
    [
        ("CN[S+](=O)([O-])c1ccccc1", "CNS(=O)(=O)c1ccccc1"),  # sulfonamide
        ("C[S+](C)[O-]", "CS(C)=O"),  # sulfoxide
    ],
)
def test_optional_charge_separation_collapses(written, hypervalent):
    from rdkit import Chem

    from openclatura.audit.reconstruction import _collapse_charge_separation

    def collapsed(smiles):
        return Chem.MolToSmiles(_collapse_charge_separation(Chem.MolFromSmiles(smiles)))

    assert collapsed(written) == collapsed(hypervalent)


@pytest.mark.parametrize(
    "smiles",
    [
        "O=[N+]([O-])c1ccccc1",  # nitro: nitrogen cannot take the extra bond
        "C[N+](C)(C)[O-]",  # N-oxide
        "N=[N+]=[N-]",  # azide
    ],
)
def test_obligatory_charge_separation_is_preserved(smiles):
    # These have no neutral spelling, so their charges must survive — otherwise a
    # genuine difference in charge placement could be normalised away.
    from rdkit import Chem

    from openclatura.audit.reconstruction import _collapse_charge_separation

    collapsed = _collapse_charge_separation(Chem.MolFromSmiles(smiles))
    assert any(atom.GetFormalCharge() != 0 for atom in collapsed.GetAtoms())


# --------------------------------------------------------------------------- #
# Reference / reconstruction plumbing
# --------------------------------------------------------------------------- #
def test_confirmed_result_exposes_matching_smiles():
    result = self_audit("CC(=O)O")
    assert result.reference_smiles == result.reconstructed_smiles
    assert result.reference_smiles is not None


def test_error_never_raises_on_broken_parts():
    mol, atoms, parts = _capture_top_level("CCO")
    bad = copy.deepcopy(parts)
    bad.parent_length = -3  # nonsensical
    result = audit_component_reconstruction(mol, bad, atoms)
    assert isinstance(result, ReconstructionAudit)
    assert result.verdict in {"abstained", "mismatch", "error"}


# --------------------------------------------------------------------------- #
# Engine / public-API integration
# --------------------------------------------------------------------------- #
def test_verify_self_populates_result():
    result = oc.name("CCO", verify_self=True)
    assert result.self_audit is not None
    assert result.self_verified is True
    assert result.self_audit.verdict == "confirmed"
    assert result.to_dict()["self_audit"]["verdict"] == "confirmed"


def test_verify_self_is_off_by_default():
    result = oc.name("CCO")
    assert result.self_audit is None
    assert result.self_verified is False
    assert "self_audit" not in result.to_dict()


def test_verify_self_flags_bad_name():
    result = oc.name("CC1(C)C#CC(O)C(C)(C)N1CCO", verify_self=True)
    assert result.self_verified is False
    assert result.self_audit.verdict == "mismatch"


def test_name_many_carries_self_audit():
    results = oc.name_many(["CCO", "CC(=O)O", "c1ccccc1"], verify_self=True)
    assert all(r.self_audit is not None for r in results)
    assert all(r.self_verified for r in results)


# --------------------------------------------------------------------------- #
# No false mismatch on the golden corpus (strongest soundness signal)
# --------------------------------------------------------------------------- #
def _corpus_smiles() -> list[str]:
    corpus = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "diverse_corpus.csv"
    if not corpus.exists():
        return []
    import csv

    with corpus.open() as handle:
        return [row["smiles"] for row in csv.DictReader(handle)]


def test_no_false_mismatch_on_golden_corpus():
    smiles = _corpus_smiles()
    if not smiles:
        pytest.skip("diverse_corpus.csv fixture not available")
    offenders = [s for s in smiles if self_audit(s).verdict == "mismatch"]
    assert offenders == [], f"self-audit refuted known-good names: {offenders}"
