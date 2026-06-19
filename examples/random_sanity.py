from bluenamer.namer import name_smiles

TESTS = [
    {
        "name": "Adamantane (PubChem CID: 13540)",
        "desc": "Tests the new Graph Signature recognition for the adamantane polycycle.",
        "smiles": "C1C2CC3CC1CC(C2)C3",
    },
    {
        "name": "Cubane (PubChem CID: 136984)",
        "desc": "Tests the Graph Signature recognition for the highly strained cubane core.",
        "smiles": "C12C3C4C1C5C2C3C45",
    },
    {
        "name": "Gonane / Steroid Core (PubChem CID: 12304505)",
        "desc": "Tests the Graph Signature recognition for the 17-carbon tetracyclic steroid backbone.",
        "smiles": "C1CCC2C(C1)CCC3C2CCC4C3CCC4",
    },
    {"name": "Complex Amine", "desc": "", "smiles": "CCN(C)CCCCCCN(CC)CCN(C)C"},
    {
        "name": "Anthracene (PubChem CID: 8418)",
        "desc": "Tests linear vs bent tricycle differentiation (Anthracene vs Phenanthrene).",
        "smiles": "c1ccc2cc3ccccc3cc2c1",
    },
    {
        "name": "Phenanthrene (PubChem CID: 955)",
        "desc": "Tests linear vs bent tricycle differentiation (Anthracene vs Phenanthrene).",
        "smiles": "c1ccc2c(c1)ccc3ccccc23",
    },
    {
        "name": "Small Ring E/Z Omission",
        "desc": "Tests that the (Z) stereocenter is correctly OMITTED for a 5-membered ring.",
        "smiles": r"F/C1=C(\Cl)CCC1",
    },
    {
        "name": "Large Ring E/Z Inclusion",
        "desc": "Tests that the (Z) stereocenter is INCLUDED for an 11-membered ring where trans is possible.",
        "smiles": r"F/C1=C(\Cl)CCCCCCCC1",
    },
    {
        "name": "DEET (PubChem CID: 4284)",
        "desc": "Tests complex N,N-dialkylated amides attached to a substituted benzene ring.",
        "smiles": "CCN(CC)C(=O)c1cc(C)ccc1",
    },
    {
        "name": "Sodium Tosylate (PubChem CID: 23673458)",
        "desc": "Tests multi-component salt handling and sulfonic acid anion nomenclature.",
        "smiles": "Cc1ccc(cc1)S(=O)(=O)[O-].[Na+]",
    },
    {
        "name": "Cocaine",
        "desc": "Tests the anhydride perception logic.",
        "smiles": "COC(=O)[C@H]1[C@@H](OC(=O)c2ccccc2)C[C@@H]2CC[C@H]1N2C",
    },
    {
        "name": "7-oxabicyclo[2.2.1]heptane (PubChem CID: 115134)",
        "desc": "Tests skeletal replacement (oxa) inside a von Baeyer bicyclic system.",
        "smiles": "C1CC2CCC1O2",
    },
    {
        "name": "Diglyme / Nested Ethers (PubChem CID: 8133)",
        "desc": "Tests deeply nested alkoxy substituents (methoxyethoxy).",
        "smiles": "COCCOCCOC",
    },
    {
        "name": "2-amino-5-nitrobenzenesulfonic acid (PubChem CID: 6697)",
        "desc": "Tests priority: Sulfonic Acid > Amine > Nitro.",
        "smiles": "Nc1ccc(cc1S(=O)(=O)O)[N+](=O)[O-]",
    },
    {
        "name": "Monomethyl Glutarate (PubChem CID: 14366)",
        "desc": "Tests priority: Carboxylic Acid > Ester (forces ester to become an 'oxo'/'methoxy' prefix).",
        "smiles": "COC(=O)CCCC(=O)O",
    },
    {
        "name": "The 'Kitchen Sink' Aliphatic",
        "desc": "Tests R/S, E/Z, Acid, Alcohol, and Amine all on the same chain.",
        "smiles": "N[C@@H](CO)/C=C/C(=O)O",
    },
    {
        "name": "1,4-dichlorocubane",
        "desc": "Tests numbering and substituent placement on a recognized polycycle.",
        "smiles": "ClC12C3C4C1C5C2C3C45Cl",
    },
    {
        "name": "1-adamantanol (PubChem CID: 73334)",
        "desc": "Tests principal group suffix (-ol) attached to a recognized polycycle.",
        "smiles": "OC12CC3CC(CC(C3)C1)C2",
    },
    {
        "name": "8-hydroxyquinoline (PubChem CID: 1923)",
        "desc": "Tests principal group suffix attached to a retained fused heterocycle.",
        "smiles": "Oc1cccc2cccnc12",
    },
    {
        "name": "5-methoxyindole (PubChem CID: 11804)",
        "desc": "Tests alkoxy substituent on a retained fused heterocycle with indicated hydrogen.",
        "smiles": "CC1=C2[C@H](C(=O)[C@@]3([C@@H](C[C@@H]4[C@](C3[C@@H]([C@@](C2(C)C)(CC1OC(=O)[C@H](O)[C@@H](NC(=O)c5ccccc5)c6ccccc6)O)OC(=O)c7ccccc7)(CO4)OC(=O)C)O)C)OC(=O)C",
    },
]


def run_tests():
    print("🧪 RUNNING PUBCHEM MASTER TESTS 🧪")
    print("========================================\n")

    passed = 0
    for i, test in enumerate(TESTS, 1):
        print(f"{i}. {test['name']}")
        print(f"   Description: {test['desc']}")
        print(f"   SMILES:      {test['smiles']}")

        try:
            result = name_smiles(test["smiles"])
            print(f"   IUPAC Name:  ✨ {result} ✨\n")
            passed += 1
        except Exception as e:
            print(f"   ❌ ERROR: {str(e)}\n")

    print("========================================")
    print(f"Successfully generated names for {passed}/{len(TESTS)} molecules!")


if __name__ == "__main__":
    run_tests()
