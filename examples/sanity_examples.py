from bluenamer.namer import name_smiles

tests = {
    "1. The 'Everything' Acyclic Chain": {
        "smiles": r"O=C(O)/C=C/[C@H](Cl)C(=O)C",
        "desc": "Tests E/Z alkenes, R/S chirality, and priority sorting (Acid > Ketone > Halogen).",
    },
    "2. Deeply Nested Recursive Substituents": {
        "smiles": "CCCCC(C(C)C(Cl)C)CCCC",
        "desc": "Tests a branch (propyl) that has its own branches (chloro and methyl), attached to a nonane backbone.",
    },
    "3. Complex Ester with a Ring Modifier": {
        "smiles": "CC(Cl)C(=O)OC1CCC(Br)CC1",
        "desc": "Tests chain-breaking ester logic where the front modifier is a substituted ring.",
    },
    "4. Complex Amide with N-locants and Nested Branches": {
        "smiles": "CCC(=O)N(CC)C(C)C(Cl)C",
        "desc": "Tests alphabetical sorting of N-locants alongside a complex recursive N-substituent.",
    },
    "5. Heavily Substituted Spiro Ring": {
        "smiles": "FC1CCC2(C1)CCC(Cl)C(Br)C2",
        "desc": "Tests spiro[x.y] numbering and alphabetical sorting of multiple halogens.",
    },
    "6. Bicyclo with Skeletal Replacement and Stereochemistry": {
        "smiles": "C[C@H]1CC[C@@H]2CC[C@H]1O2",
        "desc": "Tests 7-oxabicyclo[2.2.1]heptane with 3D stereocenters and an alkyl branch.",
    },
    "7. Naphthalenol (Retained Name + Principal Group)": {
        "smiles": "OC1=CC=C2C=C(Br)C(Cl)=CC2=C1",
        "desc": "Tests appending a principal suffix (-ol) to a retained fused ring (naphthalene).",
    },
    "8. Multi-Stereocenter Macrocycle": {
        "smiles": "Cl[C@H]1CC[C@@H](F)CC[C@H](Br)CC1",
        "desc": "Tests a 9-membered ring with three distinct R/S chiral centers.",
    },
    "9. THE FINAL BOSS (Chiral Bicyclic Ester)": {
        "smiles": r"O=C(OC1CC2CCC1C2)/C(Cl)=C(\Br)C",
        "desc": "Tests a bicyclic front modifier attached to an E/Z stereochemical alkene acid chain!",
    },
    "10. Complex Ethers (Alkoxy Prefixes)": {
        "smiles": "COCC(Cl)C(OC)C",
        "desc": "Tests multiple ether groups (methoxy) on a chain.",
    },
    "11. Complex Amines (Principal vs Substituent)": {
        "smiles": "CN(C)C(C)CC(N)C",
        "desc": "Tests primary amine (principal) and tertiary amine (substituent) on the same chain.",
    },
    "12. Ether and Amine with Stereochemistry": {
        "smiles": "C[C@H](OC)[C@@H](N(C)C)C",
        "desc": "Tests stereochemistry with methoxy and dimethylamino groups.",
    },
    "13. The Ultimate Heterocycle Test": {
        "smiles": "C1CCOC1",
        "desc": "Tests that ring heteroatoms are kept in the skeleton (oxacyclopentane) instead of breaking the chain.",
    },
}

print("🧪 RUNNING EXTREME IUPAC TESTS 🧪\n" + "=" * 40)

for name, data in tests.items():
    print(f"\n{name}")
    print(f"   Description: {data['desc']}")
    print(f"   SMILES:      {data['smiles']}")
    try:
        result = name_smiles(data["smiles"])
        print(f"   IUPAC Name:  ✨ {result} ✨")
    except Exception as e:
        print(f"   Error:       ❌ {e}")

print("\n" + "=" * 40)
