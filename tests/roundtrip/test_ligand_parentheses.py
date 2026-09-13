"""Stress tests for boundaries between branched central-atom ligands."""

from __future__ import annotations

import pytest

from openclatura import name_smiles

from .roundtrip_helpers import roundtrip_smiles

LIGAND_BOUNDARY_CASES = (
    (
        "substituted-methyl-before-trimethyl",
        "C[Si](C)(C)C(F)Cl",
        "(chlorofluoromethyl)trimethylsilane",
    ),
    (
        "substituted-methyl-before-triethyl",
        "CC[Si](CC)(CC)C(F)Cl",
        "(chlorofluoromethyl)triethylsilane",
    ),
    (
        "substituted-ethyl-before-triethyl",
        "CC(F)(Cl)[Si](CC)(CC)CC",
        "(1-chloro-1-fluoroethyl)triethylsilane",
    ),
    (
        "repeated-substituted-methyl",
        "C[Si](C)(CCl)CCl",
        "bis(chloromethyl)dimethylsilane",
    ),
    (
        "two-repeated-simple-ligands",
        "CO[Si](OC)(C)(C)",
        "dimethoxydimethylsilane",
    ),
    (
        "repeated-branched-alkyl",
        "C[Si](C)(CC(C)C)CC(C)C",
        "dimethylbis(2-methylpropyl)silane",
    ),
    (
        "repeated-complex-alkoxy",
        "CO[Si](OC)(OC(C)(C)C)OC(C)(C)C",
        "bis(tert-butoxy)dimethoxysilane",
    ),
    (
        "four-distinct-linear-ligands",
        "[Si](C)(CC)(CCC)CCCC",
        "butyl(ethyl)(methyl)(propyl)silane",
    ),
    (
        "four-distinct-branched-ligands",
        "[Si](CC(C)C)(CC(C)(C)C)(CCl)C(F)F",
        "(chloromethyl)(difluoromethyl)(2,2-dimethylpropyl)(2-methylpropyl)silane",
    ),
    (
        "four-distinct-halogenated-ligands",
        "[Si](CCl)(C(F)F)(C(Cl)Cl)C(F)(F)F",
        "(chloromethyl)(dichloromethyl)(difluoromethyl)(trifluoromethyl)silane",
    ),
    (
        "nested-methoxymethyl-ligands",
        "CC[Si](CC)(COC)COC",
        "1-(bis(methoxymethyl)(ethyl)silyl)ethane",
    ),
    (
        "repeated-substituted-germanium-ligands",
        "C[Ge](C)(CCl)CCl",
        "bis(chloromethyl)dimethylgermane",
    ),
    (
        "repeated-substituted-tin-ligands",
        "C[Sn](C)(CCl)CCl",
        "bis(chloromethyl)dimethylstannane",
    ),
    (
        "repeated-substituted-phosphorus-ligands",
        "CP(C)(CCl)CCl",
        "bis(chloromethyl)dimethyl-lambda5-phosphane",
    ),
    (
        "repeated-substituted-boron-ligands",
        "B(C)(CCl)CCl",
        "bis(chloromethyl)(methyl)borane",
    ),
    (
        "substituted-carbamoyl-functional-prefix",
        "O=C(O)CC(C(=O)N(CCl)CCl)",
        "3-(bis(chloromethyl)carbamoyl)propanoic acid",
    ),
    (
        "substituted-ammonio-functional-prefix",
        "[N+](CCl)(CCl)(C)CC(=O)[O-]",
        "2-(bis(chloromethyl)(methyl)ammonio)acetate",
    ),
    (
        "substituted-phosphanium-functional-prefix",
        "[BH3-][P+](CCl)(CCl)C",
        "(bis(chloromethyl)(methyl)phosphaniumyl)boranuide",
    ),
    (
        "substituted-borinic-functional-parent",
        "OB(CCl)CCl",
        "bis(chloromethyl)borinic acid",
    ),
    (
        "mixed-sulfanyl-functional-prefix",
        "N=S(CCl)C(F)Cl",
        "chloro((chloromethyl)(imino)sulfanyl)fluoromethane",
    ),
    (
        "substituted-sulfamic-functional-parent",
        "CN(C)S(=O)(=O)O",
        "dimethylsulfamic acid",
    ),
)


@pytest.mark.parametrize(
    ("_case", "smiles", "expected_name"),
    LIGAND_BOUNDARY_CASES,
    ids=[case for case, _smiles, _expected_name in LIGAND_BOUNDARY_CASES],
)
def test_branched_ligand_parentheses_are_exact(_case, smiles, expected_name):
    assert name_smiles(smiles) == expected_name


@pytest.mark.parametrize(
    ("_case", "smiles", "_expected_name"),
    LIGAND_BOUNDARY_CASES,
    ids=[case for case, _smiles, _expected_name in LIGAND_BOUNDARY_CASES],
)
def test_branched_ligand_parentheses_round_trip_through_opsin(_case, smiles, _expected_name):
    roundtrip_smiles(smiles)
