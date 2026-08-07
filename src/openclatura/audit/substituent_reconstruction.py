"""Recursive, structure-independent reconstruction of substituent fragments.

Compositional substituent names such as ``(4-methoxyphenyl)`` or
``((4-methoxyphenyl)methyl)`` decompose into a small set of *base* blocks
(alkyl chains, phenyl, cycloalkyls, common heteroaryls) decorated with
sub-substituents at locants — so we resolve them by recursion instead of
enumerating every name.

Everything here is built from the **name only** (never the input graph), so a
fragment it produces can be compared against the input to certify a name
without any risk of circularity — consistent with the reconstruction audit's
soundness contract. Anything the grammar does not fully model returns ``None``
(the caller then abstains), never a guess.
"""

from __future__ import annotations

import re

from rdkit import Chem

from ..hantzsch_widman import hw_generated_names, hw_parent_template
from ..rules import multipliers as _multipliers
from ..rules import stems as _stems
from .von_baeyer_parse import parse_hantzsch_widman as _parse_hantzsch_widman
from .von_baeyer_parse import parse_monocyclic_replacement as _parse_monocyclic_replacement
from .von_baeyer_parse import parse_spiro as _parse_spiro
from .von_baeyer_parse import parse_von_baeyer as _parse_von_baeyer

# A numbered fragment: an editable molecule, a map from IUPAC locant -> atom
# index, and the atom that bonds outward to whatever bears the substituent.
Numbered = tuple[Chem.RWMol, dict[str, int], int]

# Leaf sub-substituents: name -> SMILES of the *added* fragment, whose first
# atom bonds (with the encoded order) to the base atom at the stated locant.
_LEAF_SMILES: dict[str, str] = {
    "fluoro": "F",
    "chloro": "Cl",
    "bromo": "Br",
    "iodo": "I",
    "hydroxy": "O",
    "oxo": "=O",
    "thioxo": "=S",
    "sulfanyl": "S",
    "mercapto": "S",
    "amino": "N",
    "imino": "=N",
    "nitro": "[N+](=O)[O-]",
    "nitroso": "N=O",
    "cyano": "C#N",
    "nitrilo": "#N",  # terminal nitrile expressed as a ≡N on the carbon it decorates
    "isocyano": "[N+]#[C-]",
    "isocyanato": "N=C=O",
    "isothiocyanato": "N=C=S",
    # …and the ``O``/``S``-bonded pair the ``iso`` forms are distinguished from
    "cyanato": "OC#N",
    "thiocyanato": "SC#N",
    "azido": "N=[N+]=[N-]",
    "hydroperoxy": "OO",
    "diazenyl": "N=N",
    # Triazene / triazane chains (N1[=/-]N2[-/=]N3), attached at N1. ``triaz-1-en-
    # 1-yl`` and its retained synonym ``aminodiazenyl`` are the same -N=N-NH2.
    "triaz-1-en-1-yl": "N=NN",
    "aminodiazenyl": "N=NN",
    "triaz-2-en-1-yl": "NN=N",
    "triazan-1-yl": "NNN",
    "triazanyl": "NNN",
    "silyl": "[SiH3]",
    "methoxy": "OC",
    "ethoxy": "OCC",
    "propoxy": "OCCC",
    "methylsulfanyl": "SC",
    "ethylsulfanyl": "SCC",
    "formyl": "C=O",
    "formamido": "NC=O",
    # ``oxido`` is the charge-separated spelling of an oxo on a hub that carries
    # the matching ``+``; modelled as the oxo itself so it agrees with an input
    # written either way.  On a centre that cannot take the extra bond (a
    # quaternary N-oxide) the result fails to sanitise and the caller abstains.
    "oxido": "=O",
    "carboxy": "C(=O)O",
    "carbamoyl": "C(N)=O",
    "acetyl": "C(C)=O",
    "propionyl": "C(=O)CC",
    "propanoyl": "C(=O)CC",
    "butanoyl": "C(=O)CCC",
    "pentanoyl": "C(=O)CCCC",
    "methylcarbonyl": "C(C)=O",
    "acetyloxy": "OC(C)=O",
    "methylcarbonyloxy": "OC(C)=O",
    "methylcarbonylamino": "NC(C)=O",
    "trifluoromethyl": "C(F)(F)F",
    "hydroxymethyl": "CO",
    "aminomethyl": "CN",
    "methoxymethyl": "COC",
    "cyanomethyl": "CC#N",
    "methylsulfonyl": "S(=O)(=O)C",
    "aminosulfonyl": "S(N)(=O)=O",
    "sulfamoyl": "S(N)(=O)=O",
    "sulfo": "S(=O)(=O)O",
    "dimethylamino": "N(C)C",
    "methylamino": "NC",
    "hydrazinyl": "NN",
    "hydrazino": "NN",
    "hydroxyimino": "=NO",
    "methylimino": "=NC",
    "methoxyimino": "=NOC",
    "ethoxyimino": "=NOCC",
    "acetamido": "NC(C)=O",
    "acetylamino": "NC(C)=O",
    "benzamido": "NC(=O)c1ccccc1",
    "methylcarbamoyl": "C(=O)NC",
    "dimethylcarbamoyl": "C(=O)N(C)C",
    "difluoromethyl": "C(F)F",
    # silyl / phosphoryl / boryl / thio groups
    "trimethylsilyl": "[Si](C)(C)C",
    "(tert-butyl)dimethylsilyl": "[Si](C)(C)C(C)(C)C",
    "triethylsilyl": "[Si](CC)(CC)CC",
    "dimethoxyphosphoryl": "P(=O)(OC)OC",
    "diethoxyphosphoryl": "P(=O)(OCC)OCC",
    "dihydroxyphosphoryl": "P(=O)(O)O",
    "phosphono": "P(=O)(O)O",
    "phosphanyl": "P",
    "dihydroxyboryl": "B(O)O",
    "carbamothioyl": "C(=S)N",
    "diazo": "=[N+]=[N-]",
    "methyl": "C",
    "ethyl": "CC",
    "propyl": "CCC",
    "butyl": "CCCC",
    # retained / contracted alkyl and aralkyl substituents
    "isopropyl": "C(C)C",
    "isobutyl": "CC(C)C",
    "sec-butyl": "C(C)CC",
    "tert-butyl": "C(C)(C)C",
    "isopentyl": "CCC(C)C",
    "isoamyl": "CCC(C)C",
    "neopentyl": "CC(C)(C)C",
    "tert-pentyl": "C(C)(C)CC",
    "vinyl": "C=C",
    "ethenyl": "C=C",
    "allyl": "CC=C",
    "benzyl": "Cc1ccccc1",
    "phenethyl": "CCc1ccccc1",
    "styryl": "C=Cc1ccccc1",
    "benzhydryl": "C(c1ccccc1)c1ccccc1",
    "trityl": "C(c1ccccc1)(c1ccccc1)c1ccccc1",
    "benzoyl": "C(=O)c1ccccc1",
    "phenoxy": "Oc1ccccc1",
    "anilino": "Nc1ccccc1",
    "methylidene": "=C",
    "ethylidene": "=CC",
}

# Heteroaryl / saturated-heterocycle stems: name -> (SMILES, locant labels).
# Labels are the IUPAC ring locants in SMILES-atom order.
_RING_STEMS: dict[str, tuple[str, list[str]]] = {
    "phenyl": ("c1ccccc1", ["1", "2", "3", "4", "5", "6"]),
    "pyridine": ("n1ccccc1", ["1", "2", "3", "4", "5", "6"]),
    "pyrimidine": ("n1cnccc1", ["1", "2", "3", "4", "5", "6"]),
    "pyrazine": ("n1ccncc1", ["1", "2", "3", "4", "5", "6"]),
    "pyridazine": ("n1ncccc1", ["1", "2", "3", "4", "5", "6"]),
    "furan": ("o1cccc1", ["1", "2", "3", "4", "5"]),
    "thiophene": ("s1cccc1", ["1", "2", "3", "4", "5"]),
    "pyrrole": ("[nH]1cccc1", ["1", "2", "3", "4", "5"]),
    "oxazole": ("o1cncc1", ["1", "2", "3", "4", "5"]),
    "1,3-oxazole": ("o1cncc1", ["1", "2", "3", "4", "5"]),
    "isoxazole": ("o1nccc1", ["1", "2", "3", "4", "5"]),
    "1,2-oxazole": ("o1nccc1", ["1", "2", "3", "4", "5"]),
    "thiazole": ("s1cncc1", ["1", "2", "3", "4", "5"]),
    "1,3-thiazole": ("s1cncc1", ["1", "2", "3", "4", "5"]),
    "isothiazole": ("s1nccc1", ["1", "2", "3", "4", "5"]),
    "1,2-thiazole": ("s1nccc1", ["1", "2", "3", "4", "5"]),
    "imidazole": ("[nH]1cncc1", ["1", "2", "3", "4", "5"]),
    "pyrazole": ("[nH]1nccc1", ["1", "2", "3", "4", "5"]),
    "naphthalene": (
        "c1cccc2ccccc12",
        ["1", "2", "3", "4", "4a", "5", "6", "7", "8", "8a"],
    ),
    # linearly and angularly fused tricyclics, whose numbering runs round the
    # periphery from the ring furthest from the fusion
    "anthracene": (
        "c1cccc2cc3ccccc3cc12",
        ["1", "2", "3", "4", "4a", "10", "10a", "5", "6", "7", "8", "8a", "9", "9a"],
    ),
    "phenanthrene": (
        "c1cccc2c3ccccc3ccc12",
        ["1", "2", "3", "4", "4a", "4b", "5", "6", "7", "8", "8a", "9", "10", "10a"],
    ),
    # five-membered, multi-heteroatom azoles (heteroatom order fixes numbering)
    "1,2,3-triazole": ("[nH]1nncc1", ["1", "2", "3", "4", "5"]),
    "1,2,4-triazole": ("[nH]1ncnc1", ["1", "2", "3", "4", "5"]),
    "tetrazole": ("[nH]1nnnc1", ["1", "2", "3", "4", "5"]),
    "1,2,3-oxadiazole": ("o1nncc1", ["1", "2", "3", "4", "5"]),
    "1,2,4-oxadiazole": ("o1ncnc1", ["1", "2", "3", "4", "5"]),
    "1,2,5-oxadiazole": ("o1nccn1", ["1", "2", "3", "4", "5"]),
    "1,3,4-oxadiazole": ("o1cnnc1", ["1", "2", "3", "4", "5"]),
    "1,2,3-thiadiazole": ("s1nncc1", ["1", "2", "3", "4", "5"]),
    "1,2,4-thiadiazole": ("s1ncnc1", ["1", "2", "3", "4", "5"]),
    "1,2,5-thiadiazole": ("s1nccn1", ["1", "2", "3", "4", "5"]),
    "1,3,4-thiadiazole": ("s1cnnc1", ["1", "2", "3", "4", "5"]),
    # benzo-fused five-membered heterocycles (labels 1,2,3,3a,4,5,6,7,7a)
    "indole": ("[nH]1ccc2ccccc21", ["1", "2", "3", "3a", "4", "5", "6", "7", "7a"]),
    "indazole": ("[nH]1ncc2ccccc21", ["1", "2", "3", "3a", "4", "5", "6", "7", "7a"]),
    "benzimidazole": ("[nH]1cnc2ccccc21", ["1", "2", "3", "3a", "4", "5", "6", "7", "7a"]),
    "benzofuran": ("o1ccc2ccccc21", ["1", "2", "3", "3a", "4", "5", "6", "7", "7a"]),
    "1-benzofuran": ("o1ccc2ccccc21", ["1", "2", "3", "3a", "4", "5", "6", "7", "7a"]),
    "benzothiophene": ("s1ccc2ccccc21", ["1", "2", "3", "3a", "4", "5", "6", "7", "7a"]),
    "1-benzothiophene": ("s1ccc2ccccc21", ["1", "2", "3", "3a", "4", "5", "6", "7", "7a"]),
    "benzothiazole": ("s1cnc2ccccc21", ["1", "2", "3", "3a", "4", "5", "6", "7", "7a"]),
    "1,3-benzothiazole": ("s1cnc2ccccc21", ["1", "2", "3", "3a", "4", "5", "6", "7", "7a"]),
    "benzoxazole": ("o1cnc2ccccc21", ["1", "2", "3", "3a", "4", "5", "6", "7", "7a"]),
    "1,3-benzoxazole": ("o1cnc2ccccc21", ["1", "2", "3", "3a", "4", "5", "6", "7", "7a"]),
    # benzo-fused six-membered heterocycles (labels 1,2,3,4,4a,5,6,7,8,8a)
    "quinoline": ("n1cccc2ccccc21", ["1", "2", "3", "4", "4a", "5", "6", "7", "8", "8a"]),
    "isoquinoline": ("c1nccc2ccccc21", ["1", "2", "3", "4", "4a", "5", "6", "7", "8", "8a"]),
    "quinazoline": ("n1cncc2ccccc21", ["1", "2", "3", "4", "4a", "5", "6", "7", "8", "8a"]),
    "quinoxaline": ("n1ccnc2ccccc21", ["1", "2", "3", "4", "4a", "5", "6", "7", "8", "8a"]),
    # saturated / partially-saturated heterocycles (heteroatom gets locant 1)
    "piperidine": ("N1CCCCC1", ["1", "2", "3", "4", "5", "6"]),
    "piperazine": ("N1CCNCC1", ["1", "2", "3", "4", "5", "6"]),
    "morpholine": ("O1CCNCC1", ["1", "2", "3", "4", "5", "6"]),
    "thiomorpholine": ("S1CCNCC1", ["1", "2", "3", "4", "5", "6"]),
    "pyrrolidine": ("N1CCCC1", ["1", "2", "3", "4", "5"]),
    "oxolane": ("O1CCCC1", ["1", "2", "3", "4", "5"]),
    "tetrahydrofuran": ("O1CCCC1", ["1", "2", "3", "4", "5"]),
    "thiolane": ("S1CCCC1", ["1", "2", "3", "4", "5"]),
    "oxane": ("O1CCCCC1", ["1", "2", "3", "4", "5", "6"]),
    "thiane": ("S1CCCCC1", ["1", "2", "3", "4", "5", "6"]),
    "tetrahydropyran": ("O1CCCCC1", ["1", "2", "3", "4", "5", "6"]),
    "azetidine": ("N1CCC1", ["1", "2", "3", "4"]),
    "oxetane": ("O1CCC1", ["1", "2", "3", "4"]),
    "aziridine": ("N1CC1", ["1", "2", "3"]),
    "oxirane": ("O1CC1", ["1", "2", "3"]),
    "thiirane": ("S1CC1", ["1", "2", "3"]),
    # saturated five-membered multi-heteroatom rings (heteroatom order fixes numbering)
    "imidazolidine": ("N1CNCC1", ["1", "2", "3", "4", "5"]),
    "pyrazolidine": ("N1NCCC1", ["1", "2", "3", "4", "5"]),
    "1,3-thiazolidine": ("S1CNCC1", ["1", "2", "3", "4", "5"]),
    "1,3-oxazolidine": ("O1CNCC1", ["1", "2", "3", "4", "5"]),
    "isothiazolidine": ("S1NCCC1", ["1", "2", "3", "4", "5"]),
    "1,2-thiazolidine": ("S1NCCC1", ["1", "2", "3", "4", "5"]),
    "isoxazolidine": ("O1NCCC1", ["1", "2", "3", "4", "5"]),
    "1,2-oxazolidine": ("O1NCCC1", ["1", "2", "3", "4", "5"]),
    # six-membered triazines
    "1,3,5-triazine": ("n1cncnc1", ["1", "2", "3", "4", "5", "6"]),
    "1,2,3-triazine": ("n1nnccc1", ["1", "2", "3", "4", "5", "6"]),
    "1,2,4-triazine": ("n1ncncc1", ["1", "2", "3", "4", "5", "6"]),
    # benzo-fused carbocycle / 2,3-dihydro azole (labels 1,2,3,3a,4,5,6,7,7a)
    "indoline": ("N1CCc2ccccc21", ["1", "2", "3", "3a", "4", "5", "6", "7", "7a"]),
    "indane": ("C1CCc2ccccc21", ["1", "2", "3", "3a", "4", "5", "6", "7", "7a"]),
    "indan": ("C1CCc2ccccc21", ["1", "2", "3", "3a", "4", "5", "6", "7", "7a"]),
    # adamantane (tricyclo[3.3.1.1^{3,7}]decane); locants follow the SMILES atom order
    "adamantane": ("C12CC3CC(CC(C1)C3)C2", ["1", "2", "3", "4", "5", "6", "7", "8", "10", "9"]),
}

# Retained *fused* ring systems the stems above do not cover.  The namer emits
# each of these both as a parent (``phenazin-2-ol``) and, since it names a ring
# substituent from the same retained table, as a ``-yl`` prefix
# (``(phenazin-2-yl)methanol``), so one table serves both.  Each entry is a
# hand-copied projection of the graph template the namer matches
# against (`retained_fused_graph_templates`), so the locants below are the
# template's own -- they are not in ascending order, and need not be: the lookup
# pairs each label with the SMILES atom at the same position.  Parents whose
# pyrrole-type N-H can sit on more than one nitrogen (7H-/9H-purine) carry it at
# the first position their aliases cite; `move_indicated_hydrogen` relocates it
# when the name asks for the other tautomer.
# fmt: off
_RETAINED_FUSED_RING_STEMS: dict[str, tuple[str, list[str]]] = {
    "1H-indole": ("c1ccc2[nH]ccc2c1", ["5", "6", "7", "7a", "1", "2", "3", "3a", "4"]),
    "1H-perimidine": ("C1=Nc2cccc3cccc(c23)N1", ["2", "3", "3a", "4", "5", "6", "6a", "7", "8", "9", "9a", "9b", "1"]),
    "2H-isoindole": ("c1ccc2c[nH]cc2c1", ["5", "6", "7", "7a", "1", "2", "3", "3a", "4"]),
    "acenaphthylene": ("C1=Cc2cccc3cccc1c23", ["1", "2", "2a", "3", "4", "5", "5a", "6", "7", "8", "8a", "8b"]),
    "azulene": ("c1ccc2cccc-2cc1", ["6", "7", "8", "8a", "1", "2", "3", "3a", "4", "5"]),
    "fluoranthene": ("c1ccc2c(c1)-c1cccc3cccc-2c13", ["9", "8", "7", "6b", "10a", "10", "10b", "1", "2", "3", "3a", "4", "5", "6", "6a", "10c"]),
    "indolizine": ("c1ccn2cccc2c1", ["7", "6", "5", "4", "3", "2", "1", "8a", "8"]),
    "pteridine": ("c1cnc2ncncc2n1", ["6", "7", "8", "8a", "1", "2", "3", "4", "4a", "5"]),
    "1,10-phenanthroline": ("c1cnc2c(c1)ccc1cccnc12", ["3", "2", "1", "10b", "4a", "4", "5", "6", "6a", "7", "8", "9", "10", "10a"]),
    "1,4-phenanthroline": ("c1ccc2c(c1)ccc1nccnc12", ["8", "9", "10", "10a", "6a", "7", "6", "5", "4a", "4", "3", "2", "1", "10b"]),
    "1,5-naphthyridine": ("c1cnc2cccnc2c1", ["3", "2", "1", "8a", "8", "7", "6", "5", "4a", "4"]),
    "1,5-phenanthroline": ("c1ccc2c(c1)cnc1cccnc12", ["8", "9", "10", "10a", "6a", "7", "6", "5", "4a", "4", "3", "2", "1", "10b"]),
    "1,6-naphthyridine": ("c1cnc2ccncc2c1", ["3", "2", "1", "8a", "8", "7", "6", "5", "4a", "4"]),
    "1,6-phenanthroline": ("c1cnc2c(c1)cnc1ccccc12", ["3", "2", "1", "10b", "4a", "4", "5", "6", "6a", "7", "8", "9", "10", "10a"]),
    "1,7-naphthyridine": ("c1cnc2cnccc2c1", ["3", "2", "1", "8a", "8", "7", "6", "5", "4a", "4"]),
    "1,7-phenanthroline": ("c1cnc2c(c1)ccc1ncccc12", ["3", "2", "1", "10b", "4a", "4", "5", "6", "6a", "7", "8", "9", "10", "10a"]),
    "1,8-naphthyridine": ("c1cnc2ncccc2c1", ["3", "2", "1", "8a", "8", "7", "6", "5", "4a", "4"]),
    "1,8-phenanthroline": ("c1cnc2c(c1)ccc1cnccc12", ["3", "2", "1", "10b", "4a", "4", "5", "6", "6a", "7", "8", "9", "10", "10a"]),
    "1,9-phenanthroline": ("c1cnc2c(c1)ccc1ccncc12", ["3", "2", "1", "10b", "4a", "4", "5", "6", "6a", "7", "8", "9", "10", "10a"]),
    "2,6-naphthyridine": ("c1cc2cnccc2cn1", ["3", "4", "4a", "5", "6", "7", "8", "8a", "1", "2"]),
    "2,7-naphthyridine": ("c1cc2ccncc2cn1", ["3", "4", "4a", "5", "6", "7", "8", "8a", "1", "2"]),
    "2,7-phenanthroline": ("c1cnc2ccc3ccncc3c2c1", ["9", "8", "7", "6a", "6", "5", "4a", "4", "3", "2", "1", "10b", "10a", "10"]),
    "2,8-phenanthroline": ("c1cc2c(ccc3ccncc32)cn1", ["9", "10", "10a", "6a", "6", "5", "4a", "4", "3", "2", "1", "10b", "7", "8"]),
    "3,5-phenanthroline": ("c1ccc2c(c1)cnc1cnccc12", ["8", "9", "10", "10a", "6a", "7", "6", "5", "4a", "4", "3", "2", "1", "10b"]),
    "3,6-phenanthroline": ("c1ccc2c(c1)ncc1cnccc12", ["8", "9", "10", "10a", "6a", "7", "6", "5", "4a", "4", "3", "2", "1", "10b"]),
    "4,5-phenanthroline": ("c1ccc2c(c1)cnc1ncccc12", ["8", "9", "10", "10a", "6a", "7", "6", "5", "4a", "4", "3", "2", "1", "10b"]),
    "acridine": ("c1ccc2nc3ccccc3cc2c1", ["2", "3", "4", "4a", "10", "10a", "5", "6", "7", "8", "8a", "9", "9a", "1"]),
    "carbazole": ("c1ccc2c(c1)[nH]c1ccccc12", ["2", "3", "4", "4a", "9a", "1", "9", "8a", "8", "7", "6", "5", "4b"]),
    "cinnoline": ("c1ccc2nnccc2c1", ["6", "7", "8", "8a", "1", "2", "3", "4", "4a", "5"]),
    "phenazine": ("c1ccc2nc3ccccc3nc2c1", ["3", "2", "1", "10a", "10", "9a", "9", "8", "7", "6", "5a", "5", "4a", "4"]),
    "phthalazine": ("c1ccc2cnncc2c1", ["6", "7", "8", "8a", "1", "2", "3", "4", "4a", "5"]),
    "purine": ("c1ncc2[nH]cnc2n1", ["2", "1", "6", "5", "7", "8", "9", "4", "3"]),
    "xanthene": ("c1ccc2c(c1)Cc1ccccc1O2", ["2", "3", "4", "4a", "9a", "1", "9", "8a", "8", "7", "6", "5", "10a", "10"]),
}
# fmt: on

# Folded in so the ``-yl`` lookups below reach them; kept as their own table
# because the parent side and the graph-template drift check both cite it.
_RING_STEMS.update(_RETAINED_FUSED_RING_STEMS)

# Retained "-o" substituent aliases -> canonical ring-yl form.
_RING_ALIASES: dict[str, str] = {
    "morpholino": "morpholin-4-yl",
    "piperidino": "piperidin-1-yl",
    "pyrrolidino": "pyrrolidin-1-yl",
}

# Base substituent names longer than a bare stem, matched as a suffix first.
# The stem may itself carry a locant set (``1,2,4-triazol``) and the attachment
# locant may be a ring-fusion locant (``3a``).
_LOCANT_YL_RE = re.compile(r"^(?P<stem>[a-z0-9,\[\]-]+?)-(?P<loc>\d+[a-z]?)-yl$")

# Every ring template here places its indicated hydrogen at position 1, so a
# Ring stems are stored in their ``1H-`` tautomer (``pyrrole`` -> ``[nH]1cccc1``),
# so a cited indicated hydrogen is peeled off the name here and, when it names a
# different position (``2H-``, ``4H-``, ``9H-``), realised afterwards by moving
# the template's N-H — those are genuinely different tautomers, so the ring must
# be rebuilt rather than the annotation ignored.
_INDICATED_H_RE = re.compile(r"^(\d+)H-")


# Leaf prefixes longest first, so a greedy walk cannot let ``chloro`` shadow a
# longer leaf that starts the same way.
_LEAVES_LONGEST_FIRST: tuple[str, ...] = tuple(sorted(_LEAF_SMILES, key=len, reverse=True))


def move_indicated_hydrogen(rw: Chem.RWMol, locants: dict[str, int], position) -> bool:
    """Relocate a ring's single pyrrole-type N-H to the cited locant.

    ``False`` (so the caller abstains) when there is no unique N-H to move or the
    destination cannot carry one — the tautomer is then outside what the stored
    template can express."""

    target = locants.get(str(position))
    if target is None:
        return False
    destination = rw.GetAtomWithIdx(target)
    if _saturated_ring_centre_is_sole(rw, destination):
        # A mancude ring whose one saturated position is a carbon (9H-xanthene)
        # cites that carbon, and the template already carries it there — there is
        # no N-H to move.  Being the ring's only such centre makes the citation
        # unambiguous: there is no second position the name could have meant.
        return True
    if destination.GetAtomicNum() != 7:
        return False
    donors = [a for a in rw.GetAtoms() if a.GetAtomicNum() == 7 and a.GetNumExplicitHs() == 1]
    if len(donors) != 1:
        return False
    donor = donors[0]
    if donor.GetIdx() == target:
        return True
    donor.SetNumExplicitHs(0)
    donor.SetNoImplicit(True)
    destination.SetNumExplicitHs(1)
    destination.SetNoImplicit(True)
    return True


def _saturated_ring_centre_is_sole(rw: Chem.RWMol, destination: Chem.Atom) -> bool:
    """Whether ``destination`` is the template's only hydrogen-bearing saturated
    ring carbon — the position an ``nH`` citation on such a ring must mean."""

    if destination.GetAtomicNum() != 6 or destination.GetIsAromatic():
        return False
    saturated = [
        atom
        for atom in rw.GetAtoms()
        if atom.GetAtomicNum() == 6 and not atom.GetIsAromatic() and atom.IsInRing() and atom.GetTotalNumHs() > 0
    ]
    return len(saturated) == 1 and saturated[0].GetIdx() == destination.GetIdx()


def resolve_fragment_mol(name: str) -> Chem.Mol | None:
    """Return an RDKit mol with exactly one dummy ``*`` at the attachment point,
    or ``None`` if the name is not fully modeled."""
    # Capture the leading stereo descriptors before they are stripped, so we can
    # tag the resolved fragment with what the name asserts (verified later against
    # the input): per-locant R/S, and a whole-fragment cis/trans.
    stereo_map, relative = _leading_stereo_map(name)
    mol = _resolve_fragment(name, stereo_map)
    # The relative word is tagged on the *finished* fragment: which ring it speaks
    # about is a property of the assembled structure, and operator-wrapped names
    # (``…carboxamido``) put that ring behind an added carbonyl, so there is no
    # single locant map to anchor it to.
    if mol is not None and relative is not None:
        _tag_relative_stereo(mol, relative)
    return mol


def _resolve_fragment(name: str, stereo_map: dict[str, str]) -> Chem.Mol | None:
    name = _strip_outer_parens(name.strip())
    if name in _LEAF_SMILES:
        mol = Chem.MolFromSmiles("*" + _LEAF_SMILES[name], sanitize=False)
        if mol is None:
            return None
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            return None
        return mol
    # An ``…ylidene`` attaches through a *double* bond: resolve the corresponding
    # ``…yl`` fragment and promote its single attachment bond to a double one
    # (``phenylmethylidene`` = ``=CH-phenyl`` from ``phenylmethyl`` = benzyl,
    # ``propan-2-ylidene`` = ``=C(CH3)2`` from ``propan-2-yl``).
    if name.endswith("ylidene") and name not in _LEAF_SMILES:
        base = resolve_fragment_mol(name[: -len("ylidene")] + "yl")
        if base is not None:
            return _promote_to_double_attachment(base, stereo_map)
        return None
    # von Baeyer / spiro ring substituents carry a replacement prefix that is not
    # a graftable sub-substituent, so parse the whole unit up front. If the parse
    # fails (e.g. the ring is only the inner base of ``(…-yl)oxy`` or
    # ``(…-yl)methyl``), fall through to the recursive grammar and operators.
    if name.endswith("yl") and ("cyclo[" in name or "spiro[" in name):
        vb = _parse_von_baeyer(name) if "cyclo[" in name else _parse_spiro(name)
        if vb is not None:
            rw, _lc, attach = vb
            _tag_name_stereo(rw, _lc, stereo_map)
            dummy = rw.AddAtom(Chem.Atom(0))
            rw.AddBond(dummy, attach, Chem.BondType.SINGLE)
            mol = rw.GetMol()
            try:
                Chem.SanitizeMol(mol)
            except Exception:
                return None
            return mol
    numbered = _resolve(name)
    if numbered is not None:
        rw, _locants, attach = numbered
        _tag_name_stereo(rw, _locants, stereo_map)
        # Substituting at a pyrrole-type NH consumes its hydrogen.
        attach_atom = rw.GetAtomWithIdx(attach)
        if attach_atom.GetNumExplicitHs() > 0:
            attach_atom.SetNumExplicitHs(attach_atom.GetNumExplicitHs() - 1)
        dummy = rw.AddAtom(Chem.Atom(0))
        rw.AddBond(dummy, attach, Chem.BondType.SINGLE)
        mol = rw.GetMol()
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            return None
        return mol
    # Compositional operators that wrap a recursively-resolved base:
    # ``phenylsulfonyl`` = S(=O)(=O) on phenyl, ``ethylamino`` = N on ethyl, etc.
    return _resolve_operator(name, stereo_map)


# --------------------------------------------------------------------------- #
# Compositional operators (…oxy / …amino / …sulfanyl / …sulfonyl / …carbonyl)
# --------------------------------------------------------------------------- #
# Two-port wrapper: SMILES with exactly two dummies — one bonds outward to the
# parent, the other to the recursively-resolved inner substituent.
_TWO_PORT_WRAPPERS: dict[str, str] = {
    "sulfonyl": "*S(=O)(=O)*",
    "sulfinyl": "*S(=O)*",
    "sulfanyl": "*S*",
    "selanyl": "*[Se]*",
    "carbonyl": "*C(=O)*",
    "oxy": "*O*",
}


def _resolvable(name: str) -> bool:
    return resolve_fragment_mol(name) is not None


def _normalize_yl(stem: str) -> str:
    """Turn a contracted operator stem back into a substituent name: ``phen`` ->
    ``phenyl``, ``but`` -> ``butyl``; anything already ending in ``yl`` (or a
    parenthesised group) is returned untouched."""
    stem = stem.strip().rstrip("-")
    if stem.endswith("yl") or stem.endswith(")"):
        return stem
    return stem + "yl"


def _resolve_operator(name: str, stereo_map: dict[str, str] | None = None) -> Chem.Mol | None:
    frag = _resolve_substituted_hydrazinyl(name)
    if frag is not None:
        return frag
    if name.startswith("N-") and name.endswith("amido"):
        frag = _resolve_n_substituted_amido(name[2:])
        if frag is not None:
            return frag
    if name.endswith("carboxamido") and len(name) > len("carboxamido"):
        frag = _resolve_carboxamido(name[: -len("carboxamido")], stereo_map)
        if frag is not None:
            return frag
    if name.endswith("amino") and len(name) > len("amino"):
        frag = _resolve_amino(name[: -len("amino")])
        if frag is not None:
            return frag
    frag = _resolve_liganded_hub(name)
    if frag is not None:
        return frag
    frag = _resolve_disubstituted_amide_hub(name)
    if frag is not None:
        return frag
    # ``Xcarbamoyl`` = parent-C(=O)-NH-X (an added carbonyl); ``Xsulfamoyl`` =
    # parent-S(=O)(=O)-NH-X; ``Ximino`` = parent=N-X (attached through a double bond).
    for suffix, wrapper in (
        ("carbamoyl", "*C(=O)N*"),
        ("sulfamoyl", "*S(=O)(=O)N*"),
        ("imino", "*=N*"),
        ("formyl", "*C(=O)*"),
    ):
        if name.endswith(suffix) and len(name) > len(suffix):
            stem = name[: -len(suffix)].rstrip("-")
            if suffix in _AMIDE_N_HUBS:
                stem = _strip_n_locants(stem)
            inner = _resolve_operator_inner(stem, stereo_map)
            if inner is not None:
                joined = _join_two_port(wrapper, inner)
                if joined is not None:
                    return joined
    for suffix in sorted(_TWO_PORT_WRAPPERS, key=len, reverse=True):
        if name.endswith(suffix) and len(name) > len(suffix):
            stem = name[: -len(suffix)].rstrip("-")
            inner = _resolve_operator_inner(stem, stereo_map)
            if inner is None:
                continue
            joined = _join_two_port(_TWO_PORT_WRAPPERS[suffix], inner, (stereo_map or {}).get(_UNLOCANTED))
            if joined is not None:
                return joined
    return None


def _resolve_operator_inner(stem: str, stereo_map: dict[str, str] | None) -> Chem.Mol | None:
    """Resolve an operator's inner group.

    The stem may already be a full substituent name (``ethoxycarbonyl`` ->
    ``ethoxy``) or a contracted one needing ``yl`` (``phenoxy`` -> ``phen`` ->
    ``phenyl``), so both are tried.  A leading stereo prefix consumed before the
    operator split numbers this inner group, so it is pushed back on first —
    otherwise the inner is rebuilt untagged and its descriptors go unverified."""

    prefix = _stereo_prefix(stereo_map)
    for candidate in (stem, _normalize_yl(stem)):
        for spelling in (prefix + candidate, candidate) if prefix else (candidate,):
            inner = resolve_fragment_mol(spelling)
            if inner is not None:
                return inner
    return None


def _promote_to_double_attachment(frag: Chem.Mol, stereo_map: dict[str, str] | None = None) -> Chem.Mol | None:
    """Turn a ``…yl`` fragment (single ``*`` attachment) into the ``…ylidene``
    form by promoting that attachment bond to a double bond and freeing a
    hydrogen on the attachment atom so the valence balances."""

    rw = Chem.RWMol(frag)
    dummy = next((a for a in rw.GetAtoms() if a.GetAtomicNum() == 0), None)
    if dummy is None:
        return None
    neighbors = list(dummy.GetNeighbors())
    if len(neighbors) != 1:
        return None
    attach = neighbors[0]
    bond = rw.GetBondBetweenAtoms(dummy.GetIdx(), attach.GetIdx())
    if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
        return None
    bond.SetBondType(Chem.BondType.DOUBLE)
    # An ``ylidene``'s own ``E``/``Z`` describes the bond it attaches *through* —
    # the one just promoted.  The ``…yl`` base had only a single bond there, so
    # nothing could carry the descriptor and it was dropped, leaving the
    # hydrazone/oxime C=N unverifiable.  Claim it here, but only when the name
    # pins exactly one such bond and the base did not already tag one, so a name
    # that also fixes an internal double bond abstains instead of guessing which
    # descriptor belongs to which bond.
    descriptors = [d for d in (stereo_map or {}).values() if d in {"E", "Z"}]
    if len(descriptors) == 1 and not any(b.HasProp(_NAME_CIP) for b in rw.GetBonds()):
        bond.SetProp(_NAME_CIP, descriptors[0])
    # Freeing a bond order costs one hydrogen; drop an explicit one if the atom
    # carries them, otherwise let RDKit re-derive the implicit count on sanitize.
    if attach.GetNumExplicitHs() > 0:
        attach.SetNumExplicitHs(attach.GetNumExplicitHs() - 1)
    attach.SetNoImplicit(False)
    mol = rw.GetMol()
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    return mol


def _resolve_n_substituted_amido(rest: str) -> Chem.Mol | None:
    """``N-<substituent><…amido>`` — an amide prefix whose nitrogen carries an
    extra group: ``N-methylacetamido`` = ``parent-N(CH3)-C(=O)CH3``,
    ``N-methylpyridine-3-carboxamido`` = ``parent-N(CH3)-C(=O)-pyridin-3-yl``.

    ``rest`` is the name with the leading ``N-`` removed.  Where the italic-N
    ligand ends is not marked, so every depth-0 split is tried and the first one
    whose *both* halves resolve wins; a wrong split leaves an unresolvable half
    and is rejected, keeping the guess-free contract."""

    for cut in range(1, len(rest)):
        if rest[:cut].count("(") != rest[:cut].count(")"):
            continue  # never split inside a parenthesised group
        ligand_name, amide_name = rest[:cut], rest[cut:].lstrip("-")
        if not amide_name.endswith("amido"):
            continue
        ligand = resolve_fragment_mol(ligand_name)
        if ligand is None:
            continue
        amide = resolve_fragment_mol(amide_name)
        if amide is None:
            continue
        substituted = _substitute_attachment(amide, ligand, element=7)
        if substituted is not None:
            return substituted
    return None


def _substitute_attachment(frag: Chem.Mol, ligand: Chem.Mol, element: int | None = None) -> Chem.Mol | None:
    """Graft ``ligand`` onto ``frag``'s own attachment atom (the one bonded to its
    dummy), leaving the dummy in place so the result is still a one-port fragment.
    ``element`` guards the atomic number of that atom when the caller knows it
    (the amide nitrogen for an ``N-``-substituted amide prefix)."""

    rw = Chem.RWMol(frag)
    dummy = next((a for a in rw.GetAtoms() if a.GetAtomicNum() == 0), None)
    if dummy is None:
        return None
    neighbors = list(dummy.GetNeighbors())
    if len(neighbors) != 1:
        return None
    attach = neighbors[0]
    if element is not None and attach.GetAtomicNum() != element:
        return None
    if not _graft_onto(rw, attach.GetIdx(), ligand):
        return None
    mol = rw.GetMol()
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    return mol


def _resolve_carboxamido(stem: str, stereo_map: dict[str, str] | None = None) -> Chem.Mol | None:
    """``<acyl-ring>carboxamido`` = ``parent-NH-C(=O)-<ring>``.  The ``carbox``
    contributes an extra carbonyl carbon, so we resolve the ring as a ``-yl``
    fragment and wrap it in ``*NC(=O)*`` (the N is the outward port to the
    parent, the carbonyl C bonds the ring).  ``furan-2-carboxamido`` ->
    ``NH-C(=O)-furan-2-yl``, ``cyclopropanecarboxamido`` -> ``NH-C(=O)-cyclopropyl``."""

    stem = stem.rstrip("-")
    # The leading stereo prefix was consumed before the operator split, but its
    # locants number the *ring*, so it is put back on the ring candidate — the
    # recursive call then tags those centres as it would for a bare ``-yl``.
    prefix = _stereo_prefix(stereo_map)
    for candidate in _acyl_ring_variants(stem):
        inner = resolve_fragment_mol(prefix + candidate) or resolve_fragment_mol(candidate)
        if inner is not None:
            return _join_two_port("*NC(=O)*", inner)
    return None


def _stereo_prefix(stereo_map: dict[str, str] | None) -> str:
    """Re-render a captured ``{locant: descriptor}`` map as the name prefix it
    came from, so it can be pushed onto an inner ring name."""

    if not stereo_map:
        return ""
    ordered = sorted(stereo_map.items(), key=lambda item: (not item[0].isdigit(), item[0].isdigit() and int(item[0])))
    return "(" + ",".join(f"{locant}{descriptor}" for locant, descriptor in ordered) + ")-"


def _acyl_ring_variants(stem: str):
    """Yield ``-yl`` spellings of an acid ring stem: ``furan-2`` -> ``furan-2-yl``,
    ``cyclopropane`` -> ``cyclopropyl``, ``benzene-1`` -> ``phenyl``."""
    yield stem
    yield _normalize_yl(stem)
    if "-" in stem and stem.rsplit("-", 1)[1].isdigit():
        yield stem + "-yl"  # furan-2 -> furan-2-yl
    # ``…benzene-1`` is a substituted benzoic-acid ring: with the acyl carbon at
    # position 1 the remaining substituent locants already read from that point,
    # so the ``-yl`` form is the same prefixes on ``phenyl``.
    bm = re.match(r"^(?P<pre>.*?)benzene-1$", stem)
    if bm is not None:
        yield bm.group("pre") + "phenyl"
    if stem.endswith("ane"):
        yield stem[:-3] + "yl"  # cyclopropane -> cyclopropyl
    # A locanted ring stem keeps its terminal ``e`` in the acid name but drops it
    # in the substituent form: ``bicyclo[3.1.0]hexane-6`` ->
    # ``bicyclo[3.1.0]hexan-6-yl``, ``piperidine-4`` -> ``piperidin-4-yl``.
    lm = re.match(r"^(?P<core>.*[a-z])e-(?P<loc>\d+)$", stem)
    if lm is not None:
        yield f"{lm.group('core')}-{lm.group('loc')}-yl"


def _resolve_amino(rest: str) -> Chem.Mol | None:
    """Build an ``…amino`` nitrogen bearing its organyl groups: ``ethylamino`` ->
    NH-ethyl, ``diethylamino`` -> N(ethyl)2, and two *different* groups written as
    consecutive parenthesised clauses ``(methyl)(phenyl)amino`` -> N(methyl)(phenyl)."""
    rest = rest.strip().strip("-")
    if not rest:
        return None

    # Two (or more) distinct parenthesised ligands: graft each once.
    groups = _top_level_groups(rest)
    if len(groups) >= 2 and all(g.startswith("(") for g in groups):
        frags = [resolve_fragment_mol(g) for g in groups]
        if all(f is not None for f in frags):
            return _amino_from_ligands(frags)

    count, base = _multiplied_ligand(rest)
    inner = resolve_fragment_mol(base)
    if inner is None:
        return None
    return _amino_from_ligands([inner] * count)


def _multiplied_ligand(rest: str) -> tuple[int, str]:
    """Split a ligand spec into ``(count, single-ligand name)``.

    A leading multiplier only counts when ``rest`` is *not itself* the name of one
    substituent: ``diethyl`` is not a substituent, so ``diethylamino`` is two
    ethyls, but ``diethylaminosulfonyl`` is one ligand (the ``di`` belongs to the
    inner ``diethylamino``) and must not be read as two ``ethylaminosulfonyl``.
    Preferring the whole-name reading keeps the multiplier bound to the innermost
    prefix it can attach to, which is where the name put it.
    """

    if _resolvable(rest):
        return 1, rest
    for count, base in _multipliers.candidate_splits(rest):
        if _resolvable(base):
            return count, base
    return 1, rest


def _amino_from_ligands(frags: list[Chem.Mol]) -> Chem.Mol | None:
    rw = Chem.RWMol()
    nitrogen = rw.AddAtom(Chem.Atom(7))
    dummy = rw.AddAtom(Chem.Atom(0))
    rw.AddBond(dummy, nitrogen, Chem.BondType.SINGLE)
    for frag in frags:
        if not _graft_onto(rw, nitrogen, frag):
            return None
    mol = rw.GetMol()
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    return mol


# Hub groups that bond outward through a heteroatom carrying its own oxo load and
# a list of parenthesised ligands: ``(ethoxy)(methyl)phosphoryl`` is
# ``-P(=O)(OEt)(Me)``.  Value: (element, number of ``=O``).
_LIGANDED_HUBS: dict[str, tuple[str, int]] = {
    "phosphoryl": ("P", 1),
    "oxophosphanyl": ("P", 1),
    "phosphanyl": ("P", 0),
    "silyl": ("Si", 0),
    "boryl": ("B", 0),
}


# Hubs that bond outward to the parent through a core carrying an amide-type
# nitrogen, and it is *that nitrogen* the cited ligands sit on:
# ``(A)(B)carbamoyl`` -> ``-C(=O)N(A)(B)``, ``(A)(B)sulfamoyl`` ->
# ``-S(=O)(=O)N(A)(B)``.  Value: (hub element, number of ``=O``).
_AMIDE_N_HUBS: dict[str, tuple[str, int]] = {
    "carbamoyl": ("C", 1),
    "sulfamoyl": ("S", 2),
}

_N_LOCANT_RE = re.compile(r"^N'*(?:,N'*)*-")


def _strip_n_locants(head: str) -> str:
    """Drop a leading italic-``N`` locant set (``N-``, ``N,N-``, ``N,N'-``).

    On an amide-nitrogen hub the ligands go on that nitrogen either way, so the
    locant adds no placement information the rebuild needs."""

    return _N_LOCANT_RE.sub("", head)


# A ``hydrazinyl``'s two nitrogens, as the italic locants cite them: ``N`` is the
# one bonded to the parent, ``N'`` the far one.
_HYDRAZINYL_SITES: tuple[str, str] = ("N", "N'")

_N_CLAUSE_RE = re.compile(r"N'*(?:,N'*)*-")


def _resolve_substituted_hydrazinyl(name: str) -> Chem.Mol | None:
    """``<N-locanted clauses>hydrazinyl`` -> ``parent-N(…)-N(…)``.

    A single-nitrogen hub can discard its italic locants (see
    :func:`_strip_n_locants`) because its ligands have only one place to go.  A
    hydrazine's two nitrogens are distinguishable and the primes are what tell
    them apart: ``N'-acetylhydrazinyl`` acylates the far nitrogen,
    ``N-acetylhydrazinyl`` the one bonded to the parent.  Dropping the locant
    would rebuild the wrong graph, so anything that does not parse as a clean
    ``N``/``N'`` clause returns ``None`` and lets the audit abstain.
    """

    if not name.endswith("hydrazinyl") or len(name) == len("hydrazinyl"):
        return None
    clauses = _parse_n_locant_clauses(name[: -len("hydrazinyl")].rstrip("-"))
    if clauses is None:
        return None
    rw = Chem.RWMol()
    dummy = rw.AddAtom(Chem.Atom(0))
    near = rw.AddAtom(Chem.Atom(7))
    far = rw.AddAtom(Chem.Atom(7))
    rw.AddBond(dummy, near, Chem.BondType.SINGLE)
    rw.AddBond(near, far, Chem.BondType.SINGLE)
    sites = dict(zip(_HYDRAZINYL_SITES, (near, far)))
    for locs, subname in clauses:
        frag = resolve_fragment_mol(subname)
        if frag is None:
            return None
        for loc in locs:
            if not _graft_onto(rw, sites[loc], frag):
                return None
    mol = rw.GetMol()
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    return mol


def _parse_n_locant_clauses(head: str) -> list[tuple[list[str], str]] | None:
    """Split ``N'-acetyl``, ``N,N'-dimethyl``, ``N',N'-dimethyl`` into
    ``(locants, ligand-name)`` pairs, the italic-``N`` counterpart of
    :func:`_parse_clauses`.

    Splits only on locant groups at parenthesis depth 0, so an ``N`` inside a
    parenthesised ligand stays part of that ligand."""

    starts = [
        m.start() for m in _N_CLAUSE_RE.finditer(head) if head[: m.start()].count("(") == head[: m.start()].count(")")
    ]
    if not starts or starts[0] != 0:
        return None
    clauses: list[tuple[list[str], str]] = []
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(head)
        m = re.match(r"^(N'*(?:,N'*)*)-(.*)$", head[start:end], re.DOTALL)
        if m is None:
            return None
        locs = m.group(1).split(",")
        if any(loc not in _HYDRAZINYL_SITES for loc in locs):
            return None  # a third nitrogen: not a hydrazine
        body = m.group(2).rstrip("-")
        leaf = _leading_multiplier(body)
        if leaf is not None:
            count, body = leaf
            if len(locs) != count:
                return None
        elif len(locs) > 1:
            # As in :func:`_parse_clauses`: the locants already pin the count, so
            # a ``bis``/``tris`` spelling over a parenthesised ligand is readable
            # whenever it agrees with them.
            for count, rest in _multipliers.candidate_splits(body):
                if count == len(locs) and _resolvable(rest):
                    body = rest
                    break
        if not body:
            return None
        clauses.append((locs, body))
    return clauses


def _resolve_disubstituted_amide_hub(name: str) -> Chem.Mol | None:
    """``(A)(B)carbamoyl`` / ``(A)(B)sulfamoyl`` — an amide or sulfonamide whose
    nitrogen carries two cited ligands rather than the single one the plain
    ``<X>carbamoyl`` / ``<X>sulfamoyl`` operator covers.  Two ligands are
    required, so a single-ligand name still falls through to that operator."""

    for word, (element, oxo_count) in _AMIDE_N_HUBS.items():
        if not name.endswith(word) or len(name) <= len(word):
            continue
        frags = _hub_ligands(_strip_n_locants(name[: -len(word)].rstrip("-")))
        if frags is None or len(frags) < 2:
            continue
        rw = Chem.RWMol()
        hub = rw.AddAtom(Chem.Atom(element))
        dummy = rw.AddAtom(Chem.Atom(0))
        rw.AddBond(dummy, hub, Chem.BondType.SINGLE)
        for _ in range(oxo_count):
            oxo = rw.AddAtom(Chem.Atom(8))
            rw.AddBond(hub, oxo, Chem.BondType.DOUBLE)
        nitrogen = rw.AddAtom(Chem.Atom(7))
        rw.AddBond(hub, nitrogen, Chem.BondType.SINGLE)
        if not all(_graft_onto(rw, nitrogen, frag) for frag in frags):
            continue
        mol = rw.GetMol()
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            continue
        return mol
    return None


def _resolve_liganded_hub(name: str) -> Chem.Mol | None:
    """``(A)(B)phosphoryl`` / ``(A)(B)(C)silyl`` — a heteroatom hub bonded outward
    to the parent, carrying its oxo load plus each cited ligand.

    The ligands are either a list of parenthesised clauses or a single multiplied
    one (``dimethyloxophosphanyl`` -> two methyls); a head that is neither falls
    through rather than being guessed at."""

    for word, (element, oxo_count) in _LIGANDED_HUBS.items():
        if not name.endswith(word) or len(name) <= len(word):
            continue
        head = name[: -len(word)].rstrip("-")
        frags = _hub_ligands(head)
        if frags is None:
            continue
        rw = Chem.RWMol()
        hub = rw.AddAtom(Chem.Atom(element))
        dummy = rw.AddAtom(Chem.Atom(0))
        rw.AddBond(dummy, hub, Chem.BondType.SINGLE)
        for _ in range(oxo_count):
            oxo = rw.AddAtom(Chem.Atom(8))
            rw.AddBond(hub, oxo, Chem.BondType.DOUBLE)
        if not all(_graft_onto(rw, hub, frag) for frag in frags):
            continue
        mol = rw.GetMol()
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            continue
        return mol
    return None


def _hub_ligands(head: str) -> list[Chem.Mol] | None:
    """Resolve a hub's ligand list: several parenthesised clauses, or one clause
    under a multiplier.  ``None`` if the head is neither."""

    groups = _top_level_groups(head)
    if len(groups) >= 2 and all(g.startswith("(") and g.endswith(")") for g in groups):
        frags = [resolve_fragment_mol(g) for g in groups]
        return None if any(f is None for f in frags) else frags
    count, base = _multiplied_ligand(head)
    if count > 1:
        # ``tris(propan-2-yl)`` — a bare multiplier in front of one ligand — says
        # outright how many copies there are, so it is not the ambiguous mixed
        # list guarded against below.
        inner = resolve_fragment_mol(base)
        return None if inner is None else [inner] * count
    if len(groups) > 1 and not groups[0].startswith("("):
        # ``dimethyl(phenyl)`` — a multiplied *bare* ligand followed by
        # parenthesised ones.  The bare head is read on its own and the rest are
        # each their own clause, so nothing here is ambiguous; the mixed list the
        # guard below rejects is the reverse order, where a parenthesised clause
        # leads and could instead be modifying what follows it.
        lead_count, lead_base = _multiplied_ligand(groups[0])
        lead = resolve_fragment_mol(lead_base)
        rest = [resolve_fragment_mol(g) for g in groups[1:]]
        if lead is not None and all(g.startswith("(") for g in groups[1:]) and all(rest):
            return [lead] * lead_count + rest
        return None
    if len(groups) != 1:
        # A mixed list such as ``((…)oxy)ethenyl`` is ambiguous — the leading
        # parenthesised clause could be a sibling ligand on the hub or a modifier
        # of what follows it — so abstain rather than pick one reading.
        return None
    if groups[0].startswith("("):
        # One clause, enclosed and covering the whole head: nothing follows it
        # inside the head, so it can only be the hub's single ligand —
        # ``((iodo(quinolin-7-yl)methyl)phosphanyl)`` is that carbon on the P.
        # The enclosure is itself the namer's mark that the ligand is complex;
        # a simple one is spelt bare (``methylphosphanyl``) and read below.
        inner = resolve_fragment_mol(groups[0])
        return None if inner is None else [inner]
    inner = resolve_fragment_mol(base)
    return None if inner is None else [inner] * count


def _top_level_groups(s: str) -> list[str]:
    """Split a string into consecutive depth-0 tokens, keeping parenthesised
    groups intact: ``(methyl)(phenyl)`` -> ``['(methyl)', '(phenyl)']``,
    ``methyl`` -> ``['methyl']``."""
    groups: list[str] = []
    depth = 0
    cur = ""
    for ch in s:
        if ch == "(":
            if depth == 0 and cur:
                groups.append(cur)
                cur = ""
            depth += 1
            cur += ch
        elif ch == ")":
            depth -= 1
            cur += ch
            if depth == 0:
                groups.append(cur)
                cur = ""
        else:
            cur += ch
    if cur:
        groups.append(cur)
    return [g for g in groups if g]


def _join_two_port(wrapper_smiles: str, inner: Chem.Mol, hub_stereo: str | None = None) -> Chem.Mol | None:
    wrap = Chem.MolFromSmiles(wrapper_smiles, sanitize=False)
    if wrap is None:
        return None
    try:
        Chem.SanitizeMol(wrap)
    except Exception:
        return None
    dummies = [a.GetIdx() for a in wrap.GetAtoms() if a.GetAtomicNum() == 0]
    if len(dummies) != 2:
        return None
    graft_dummy = dummies[1]  # dummies[0] stays as the outward (parent) port
    port_neighbors = [n.GetIdx() for n in wrap.GetAtomWithIdx(graft_dummy).GetNeighbors()]
    if len(port_neighbors) != 1:
        return None
    port = port_neighbors[0]
    rw = Chem.RWMol()
    old_to_new: dict[int, int] = {}
    for atom in wrap.GetAtoms():
        if atom.GetIdx() == graft_dummy:
            continue
        new_atom = Chem.Atom(atom.GetAtomicNum())
        new_atom.SetFormalCharge(atom.GetFormalCharge())
        old_to_new[atom.GetIdx()] = rw.AddAtom(new_atom)
    for bond in wrap.GetBonds():
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if graft_dummy in (u, v):
            continue
        rw.AddBond(old_to_new[u], old_to_new[v], bond.GetBondType())
    if not _graft_onto(rw, old_to_new[port], inner):
        return None
    # An unlocanted ``(R)``/``(S)`` on an operator names the operator's own
    # stereogenic atom — the sulfur of ``(R)-methylsulfinyl``.  There is no
    # locant to map it through, so the wrapper identifies it: the atom the
    # fragment bonds outward from.  A hub that is not really stereogenic simply
    # fails to match the oracle later, so this cannot manufacture a confirmation.
    if hub_stereo is not None:
        rw.GetAtomWithIdx(old_to_new[port]).SetProp(_NAME_CIP, hub_stereo)
    mol = rw.GetMol()
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    return mol


# --------------------------------------------------------------------------- #
# Recursion
# --------------------------------------------------------------------------- #
# A skeletal-replacement clause: ``7-aza``, ``2,4-dioxa``, ``3lambda^6-thia``.
_REPLACEMENT_CLAUSE_RE = re.compile(
    r"\d+(?:,\d+)*(?:lambda\^?\{?\d+\}?)?-(?:di|tri|tetra|penta|hexa)?"
    r"(?:oxa|aza|thia|selena|tellura|phospha|arsa|sila|germa|stanna|bora)-"
)
# The ring token a replacement prefix belongs to: bracketed von Baeyer/spiro, or
# the bare ``cyclo`` of a replacement monocycle.
_RING_DESCRIPTOR_RE = re.compile(r"(?:bi|tri|tetra|penta)?(?:cyclo|spiro)\[?")


def _hoist_replacement_prefixes(name: str) -> str:
    """Move skeletal-replacement clauses next to the ring token they modify.

    A ring's replacement prefixes and its ordinary substituent prefixes are cited
    in one alphanumeric sequence — ``7-methyl-7-aza-8-oxo-bicyclo[4.3.0]…`` — but
    they are consumed by different machinery: replacement belongs to the ring
    parser, the rest to the prefix grafting.  No single split of that sequence
    separates them while they are interleaved, so the replacement clauses are
    gathered and re-emitted immediately before the ring token, leaving a
    contiguous ordinary prefix in front."""

    ring = _RING_DESCRIPTOR_RE.search(name)
    if ring is None:
        return name
    head, tail = name[: ring.start()], name[ring.start() :]
    clauses = _REPLACEMENT_CLAUSE_RE.findall(head)
    if not clauses:
        return name
    remainder = _REPLACEMENT_CLAUSE_RE.sub("", head)
    return remainder + "".join(clauses) + tail


def _resolve(name: str) -> Numbered | None:
    name = _strip_outer_parens(name.strip())
    if not name:
        return None
    name = _RING_ALIASES.get(name, name)
    name = _hoist_replacement_prefixes(name)
    # Peel a base off the end: try every start position, longest base first, and
    # accept the first whose remaining prefix fully parses. A wrong split fails
    # prefix parsing and is rejected, so this backtracking stays sound.
    for start in _base_start_positions(name):
        base = name[start:]
        numbered = _build_base(base)
        if numbered is None:
            continue
        rw, locants, attach = numbered
        prefix = name[:start]
        if not prefix:
            return rw, locants, attach
        if _apply_prefix(rw, locants, prefix):
            return rw, locants, attach
    return None


# Atom/bond property holding the R/S (or E/Z) descriptor the *name* asserts for a
# reconstructed atom/bond, to be checked against the input's independent CIP.
_NAME_CIP = "nStereo"
# Atom property marking the two ring atoms a leading ``cis``/``trans`` word
# relates, to be checked against the input's tetrahedral parities.
_NAME_RELATIVE = "nRelStereo"
# Relative-configuration words we can verify.  The rest of ``_STEREO_WORD_RE``
# (``rel``, ``rac``, ``syn``, ``endo``…) stays unverifiable and so stays peeled
# and discarded, leaving the caller to abstain on those centres.
_RELATIVE_WORDS = {"cis", "trans"}
# Key under which a descriptor cited *without* a locant is kept — ``(R)-methyl-
# sulfinyl`` names its operator's own stereogenic atom rather than a numbered
# position, so there is no locant to map it through.  Not a valid locant, so the
# locant-driven taggers pass over it and only the operator claims it.
_UNLOCANTED = ""


def _leading_stereo_map(name: str) -> tuple[dict[str, str], str | None]:
    """Extract ``({locant: descriptor}, relative-word)`` from a leading stereo
    prefix, peeling outer parens/stereo-words exactly as
    :func:`_strip_outer_parens` does so the captured locants line up with the
    fragment that will be built.

    The relative word is returned without its locants: which ring positions a
    ``cis``/``trans`` relates is read off the built fragment instead (the two
    ring atoms actually bearing substituents), which avoids having to reproduce
    the namer's locant conventions to find them."""

    s = name.strip()
    result: dict[str, str] = {}
    relative: str | None = None
    changed = True
    while changed:
        changed = False
        while s.startswith("(") and s.endswith(")") and _balanced(s[1:-1]):
            s = s[1:-1].strip()
            changed = True
        m = _STEREO_GROUP_RE.match(s)
        if m and re.search(r"[RSrsEZ]", m.group(1)):
            pairs = re.findall(r"(\d+)([RSrsEZ])", m.group(1))
            for loc, desc in pairs:
                result[loc] = desc
            if not pairs:
                # No locant at all: the descriptor speaks about the fragment's
                # own single stereogenic atom, which the operator that builds it
                # identifies.  Only a lone R/S qualifies — anything richer is
                # left unclaimed so the caller abstains.
                bare = re.fullmatch(r"\s*([RSrs])\s*", m.group(1))
                if bare:
                    result[_UNLOCANTED] = bare.group(1)
            s = s[m.end() :].strip()
            changed = True
            continue
        m2 = _STEREO_WORD_RE.match(s)
        if m2:
            word = m2.group(1)
            if word in _RELATIVE_WORDS:
                # Two different relative words on one fragment would each need
                # their own atom pair to be meaningful; we model only one.
                if relative is not None and relative != word:
                    return result, None
                relative = word
            s = s[m2.end() :].strip()
            changed = True
    return result, relative


def _tag_relative_stereo(mol: Chem.Mol, relative: str) -> None:
    """Mark the two ring atoms a ``cis``/``trans`` word relates.

    Which ring the word speaks about is not spelled out, so it is identified
    structurally.  A ``cis``/``trans`` is a relation between exactly two
    substituted ring positions, so the ring it refers to necessarily has exactly
    two atoms bearing exactly one exocyclic substituent each (the attachment
    dummy counting as one).  Among the rings that look like that we take:

    * the one containing the attachment atom, since the word qualifies the
      substituent's own base skeleton and that is the ring the parent hangs off;
    * failing that (an operator-wrapped name like ``…carboxamido`` puts the ring
      behind an added carbonyl, so the attachment is outside it) the sole
      candidate, if there is exactly one.

    Otherwise the relation stays unpinned — nothing is tagged and the caller
    abstains rather than adjudicate the wrong ring."""

    candidates = [ring for ring in mol.GetRingInfo().AtomRings() if _relative_pair(mol, ring) is not None]
    if not candidates:
        return
    attach = _attachment_atom(mol)
    anchored = [ring for ring in candidates if attach is not None and attach in ring]
    if len(anchored) == 1:
        ring = anchored[0]
    elif not anchored and len(candidates) == 1:
        ring = candidates[0]
    else:
        return
    for atom_idx in _relative_pair(mol, ring):
        mol.GetAtomWithIdx(atom_idx).SetProp(_NAME_RELATIVE, relative)


def _attachment_atom(mol: Chem.Mol) -> int | None:
    """The atom bonded to the fragment's dummy, i.e. what the parent bonds to."""

    dummy = next((a for a in mol.GetAtoms() if a.GetAtomicNum() == 0), None)
    if dummy is None:
        return None
    neighbors = list(dummy.GetNeighbors())
    return neighbors[0].GetIdx() if len(neighbors) == 1 else None


def _relative_pair(mol: Chem.Mol, ring: tuple[int, ...]) -> tuple[int, int] | None:
    """The ring's two singly-substituted atoms, or ``None`` if it does not have
    exactly two."""

    marked = [
        atom_idx
        for atom_idx in ring
        if sum(1 for n in mol.GetAtomWithIdx(atom_idx).GetNeighbors() if n.GetIdx() not in ring) == 1
    ]
    return (marked[0], marked[1]) if len(marked) == 2 else None


def _tag_name_stereo(rw: Chem.RWMol, locants: dict[str, int], stereo_map: dict[str, str]) -> None:
    """Tag the atom (R/S) or double bond (E/Z) at each mapped locant with the
    descriptor the name asserts."""
    for loc, desc in stereo_map.items():
        if desc in "RSrs" and loc in locants:
            rw.GetAtomWithIdx(locants[loc]).SetProp(_NAME_CIP, desc)
        elif desc in "EZ" and loc in locants:
            _tag_double_bond(rw, locants, loc, desc)


def _tag_double_bond(rw: Chem.RWMol, locants: dict[str, int], loc: str, desc: str) -> None:
    """Tag the ``loc``→``loc+1`` double bond (the usual chain/ring stereo bond),
    else any acyclic double bond incident to ``loc``."""
    a = locants.get(loc)
    if a is None:
        return
    b = locants.get(str(int(loc) + 1)) if loc.isdigit() else None
    if b is not None:
        bond = rw.GetBondBetweenAtoms(a, b)
        if bond is not None and bond.GetBondType() == Chem.BondType.DOUBLE:
            bond.SetProp(_NAME_CIP, desc)
            return
    for bond in rw.GetAtomWithIdx(a).GetBonds():
        if bond.GetBondType() == Chem.BondType.DOUBLE and not bond.IsInRing():
            bond.SetProp(_NAME_CIP, desc)
            return


def _base_start_positions(name: str) -> list[int]:
    """Candidate indices where the base substituent could begin: the start, and
    every position following a depth-0 locant hyphen or a lowercase letter.
    Earliest (longest base) first."""
    positions = [0]
    for i in range(1, len(name) - 1):
        if name[i - 1] in "-)" or name[i - 1].isalpha():
            positions.append(i)
    return positions


# --------------------------------------------------------------------------- #
# Base builders
# --------------------------------------------------------------------------- #
def _build_base(base: str) -> Numbered | None:
    indicated = _INDICATED_H_RE.match(base)
    if indicated is not None:
        base = base[indicated.end() :]
    numbered = _build_base_core(base)
    if numbered is None or indicated is None:
        return numbered
    rw, locants, attach = numbered
    return numbered if move_indicated_hydrogen(rw, locants, indicated.group(1)) else None


def _build_base_core(base: str) -> Numbered | None:
    # The retained ``benz`` acyl bases, before the generic ``…yl`` readings below
    # get a chance to mis-split them.
    benz = _benz_acyl_base(base)
    if benz is not None:
        return benz
    # A von Baeyer ring core (``…cyclo[…]…-N-yl``) as a decoratable base, so that
    # front modifiers like ``7,9-dioxo`` / ``4,6-dimethyl`` can be peeled off by
    # ``_resolve`` and grafted through the ordinary locanted-clause machinery.
    if base.endswith("yl") and "cyclo[" in base:
        vb = _parse_von_baeyer(base)
        if vb is not None:
            return vb
    # Likewise a monospiro ring core (``2-azaspiro[3.3]heptan-2-yl``).
    if base.endswith("yl") and "spiro[" in base:
        sp = _parse_spiro(base)
        if sp is not None:
            return sp
    # A Hantzsch-Widman replacement monocycle core (``1-azacyclohexa-3,5-dien-3-yl``),
    # likewise decoratable via peeled front modifiers.
    if base.endswith("yl") and "cyclo" in base and "cyclo[" not in base:
        mono = _parse_monocyclic_replacement(base)
        if mono is not None:
            return mono
    # A contracted Hantzsch-Widman monocycle (``1,4-dioxan-2-yl``, ``thian-4-yl``),
    # likewise decoratable via peeled front modifiers.
    if base.endswith("yl"):
        hw = _parse_hantzsch_widman(base)
        if hw is not None:
            return hw
    # phenyl / heteroaryl-yl
    if base == "phenyl":
        return _ring_from_stem("phenyl")
    m = _LOCANT_YL_RE.match(base)
    if m:
        ring = _ring_yl(m.group("stem"), m.group("loc"))
        if ring is not None:  # else fall through: it may be a chain like propan-2-yl
            return ring

    # cycloalkenyl (unsaturated carbocycle): cyclohex-1-en-1-yl, cyclohexa-1,3-dien-1-yl
    if base.startswith("cyclo") and base.endswith("yl") and re.search(r"\d+-(?:en|yn|dien|trien)", base):
        numbered = _cycloalkenyl_base(base)
        if numbered is not None:
            return numbered
    # cycloalkyl, with or without an explicit attachment locant
    # (``cyclopentyl``, ``cyclopentan-2-yl``)
    if base.startswith("cyclo") and base.endswith("yl"):
        core = base[len("cyclo") : -2].rstrip("-")
        attach = "1"
        located = re.search(r"-(\d+)$", core)
        if located is not None:
            attach = located.group(1)
            core = core[: located.start()]
        n = _stem_length(core)
        if n is None and core.endswith("an"):
            n = _stem_length(core[:-2])
        if n is None or n < 3:
            return None
        rw, locants, _ = _carbocycle(n)
        if attach not in locants:
            return None
        return rw, locants, locants[attach]

    # linear alkyl
    if base.endswith("yl"):
        n = _stem_length(base[:-2])
        if n is not None:
            return _alkyl_chain(n)
    # chain acyl (``hexanoyl``, ``prop-2-enoyl``, retained ``acetyl``/``butyryl``)
    # and chain acylamino (``propanamido``, ``prop-2-enamido``): both are
    # decoratable bases, so front modifiers (``2,2-dimethyl`` -> pivaloyl /
    # pivalamido) graft onto the acid carbons through the ordinary clause machinery.
    if base.endswith("oyl") or base.endswith(_RETAINED_ACYL_YL):
        acyl = _acyl_chain_base(base)
        if acyl is not None:
            return acyl
    if base.endswith("amido"):
        acyl = _acylamino_chain_base(base)
        if acyl is not None:
            return acyl
    # general acyclic chain: internal attachment and/or unsaturation
    return _parse_chain_base(base)


# Retained acyl stems: classical acid name -> carbon count of the acyl chain
# (the carbonyl carbon is C1). ``acet``ic = 2 C, ``propion``ic = 3, ``butyr``ic
# = 4, ``valer``ic = 5.
_RETAINED_ACYL_STEMS: dict[str, int] = {"acet": 2, "propion": 3, "butyr": 4, "valer": 5}
# The ``-yl`` spellings of those stems, used to recognise a retained acyl base.
_RETAINED_ACYL_YL = tuple(stem + "yl" for stem in _RETAINED_ACYL_STEMS)


def _acyl_chain_skeleton(core: str) -> tuple[Chem.RWMol, dict[str, int]] | None:
    """Build the carbon skeleton of an acyl chain from the stem the acyl suffix
    left behind — ``hexan`` from ``hexanoyl``, ``prop-2-en`` from ``prop-2-enoyl``,
    ``butyr`` from ``butyryl``, ``propan`` from ``propanamido`` — with C1 (the
    carbonyl carbon) already bearing its ``=O``.  Returns the skeleton and its
    locant map; callers decide what C1 bonds outward to."""

    core = core.rstrip("-")
    stem_len: int | None = None
    rest: str | None = None
    for retained, length in _RETAINED_ACYL_STEMS.items():
        if core.startswith(retained):
            stem_len, rest = length, core[len(retained) :]
            break
    if stem_len is None:
        for length, row in sorted(_stems.STEMS.items(), key=lambda kv: -len(kv[1].stem)):
            if core.startswith(row.stem):
                stem_len, rest = length, core[len(row.stem) :]
                break
        if stem_len is None:
            return None
    parsed = _parse_unsaturation(rest)
    if parsed is None:
        return None
    unsats, residue = parsed
    if residue:  # unmodelled leftover tokens -> abstain
        return None

    rw = Chem.RWMol()
    locants = {str(i): rw.AddAtom(Chem.Atom(6)) for i in range(1, stem_len + 1)}
    for i in range(1, stem_len):
        rw.AddBond(locants[str(i)], locants[str(i + 1)], Chem.BondType.SINGLE)
    for loc, order in unsats:
        if str(loc + 1) not in locants:
            return None
        bond = rw.GetBondBetweenAtoms(locants[str(loc)], locants[str(loc + 1)])
        bond.SetBondType(Chem.BondType.DOUBLE if order == 2 else Chem.BondType.TRIPLE)
    oxo = rw.AddAtom(Chem.Atom(8))
    rw.AddBond(locants["1"], oxo, Chem.BondType.DOUBLE)
    return rw, locants


def _acyl_chain_base(base: str) -> Numbered | None:
    """``<alkan>oyl`` and the retained ``acetyl``/``propionyl``/``butyryl``/
    ``valeryl`` = ``parent-C(=O)-<chain>``, attached through the carbonyl carbon
    (``2-(4-chlorophenyl)acetyl``, ``2-ethylbutyryl``, ``prop-2-enoyl``)."""

    core = base[:-3] if base.endswith("oyl") else base[:-2]
    skeleton = _acyl_chain_skeleton(core)
    if skeleton is None:
        return None
    rw, locants = skeleton
    return rw, locants, _hide_acyl_carbon(locants)


# The retained ``benz`` acyl bases, and whether the ring's acyl carbon carries an
# amide nitrogen beyond it (``benzamido``) or bonds outward itself (``benzoyl``).
_BENZ_ACYL_BASES: dict[str, bool] = {"benzoyl": False, "benzamido": True}


def _benz_acyl_base(base: str) -> Numbered | None:
    """``benzoyl`` / ``benzamido`` — the retained benzene acyl and acylamino bases.

    Both are also flat leaves, which is enough while they are bare; built here as
    a *decoratable* ring core instead, their ring substituents graft through the
    ordinary locanted-clause machinery.  ``4-methylbenzamido`` then reads as the
    methyl on ring C4, C1 being the ring carbon carrying the acyl.

    That C1 already holds two ring bonds and the acyl, so nothing can substitute
    there; it is withdrawn from the exposed locant map for the same reason
    :func:`_hide_acyl_carbon` withdraws a chain's C1.
    """

    with_nitrogen = _BENZ_ACYL_BASES.get(base)
    if with_nitrogen is None:
        return None
    smiles, labels = _RING_STEMS["phenyl"]
    numbered = _numbered_from_smiles(smiles, labels, attach_locant="1")
    if numbered is None:
        return None
    rw, locants, ring_c1 = numbered
    carbonyl = rw.AddAtom(Chem.Atom(6))
    oxygen = rw.AddAtom(Chem.Atom(8))
    rw.AddBond(ring_c1, carbonyl, Chem.BondType.SINGLE)
    rw.AddBond(carbonyl, oxygen, Chem.BondType.DOUBLE)
    attach = carbonyl
    if with_nitrogen:
        nitrogen = rw.AddAtom(Chem.Atom(7))
        rw.AddBond(carbonyl, nitrogen, Chem.BondType.SINGLE)
        attach = nitrogen
    locants.pop("1")
    return rw, locants, attach


def _acylamino_chain_base(base: str) -> Numbered | None:
    """``<alkanoyl>amido`` = ``parent-NH-C(=O)-<chain>`` with the chain's C1 as the
    carbonyl carbon (``propanamido`` -> ``NH-C(=O)-CH2-CH3``).  Chain locants are
    exposed so front modifiers graft onto the acid carbons; the attachment is the
    amide nitrogen."""

    skeleton = _acyl_chain_skeleton(base[: -len("amido")])
    if skeleton is None:
        return None
    rw, locants = skeleton
    carbonyl = _hide_acyl_carbon(locants)
    nitrogen = rw.AddAtom(Chem.Atom(7))
    rw.AddBond(carbonyl, nitrogen, Chem.BondType.SINGLE)
    return rw, locants, nitrogen


def _hide_acyl_carbon(locants: dict[str, int]) -> int:
    """Pop C1 out of the exposed locant map and return its index.

    An acyl C1 carries its ``=O``, the outward bond and the rest of the chain, so
    nothing can substitute there.  Withdrawing it stops :func:`_apply_prefix` from
    treating an *unlocanted* front modifier as a C1 substituent — ``phenylacetyl``
    puts the phenyl on C2, so guessing C1 would silently reconstruct the wrong
    graph; with C1 hidden the prefix fails to place and the audit abstains."""

    return locants.pop("1")


# Multiplied unsaturation suffixes: how many locants each must cite.
_UNSAT_COUNTS = {
    "en": 1,
    "dien": 2,
    "trien": 3,
    "tetraen": 4,
    "pentaen": 5,
    "yn": 1,
    "diyn": 2,
    "triyn": 3,
}
_UNSAT_RE = re.compile(r"(\d+(?:,\d+)*)-(" + "|".join(sorted(_UNSAT_COUNTS, key=len, reverse=True)) + r")")


def _parse_unsaturation(rest: str) -> tuple[list[tuple[int, int]], str] | None:
    """Read a chain's unsaturation clauses into ``(locant, bond order)`` pairs,
    returning them with whatever text was left unconsumed (the caller abstains if
    that is non-empty).

    Handles the multiplied forms — ``deca-1,8-dien``, ``nona-2,4,6,8-tetraen`` —
    by requiring the cited locant count to match the multiplier, so a malformed
    clause is rejected instead of silently under-building the chain.  ``None``
    means exactly that rejection."""

    unsats: list[tuple[int, int]] = []
    for match in _UNSAT_RE.finditer(rest):
        locants = match.group(1).split(",")
        if len(locants) != _UNSAT_COUNTS[match.group(2)]:
            return None
        order = 3 if match.group(2).endswith("yn") else 2
        unsats.extend((int(locant), order) for locant in locants)
    residue = _UNSAT_RE.sub("", rest).replace("an", "").strip("-")
    # Unlocanted unsaturation (``propenyl`` = prop-1-en-1-yl) starts at locant 1.
    if residue in ("en", "yn"):
        unsats.append((1, 2 if residue == "en" else 3))
        residue = ""
    # The elision vowel a multiplied suffix takes (``dec`` + ``a`` + ``-1,8-dien``).
    return unsats, residue.strip("-a")


def _parse_chain_base(base: str) -> Numbered | None:
    """Parse acyclic chain substituents like ``propan-2-yl``, ``prop-2-en-1-yl``,
    ``but-2-yn-1-yl`` into a numbered fragment."""
    if not base.endswith("yl"):
        return None
    core = base[:-2].rstrip("-")
    stem_len = None
    rest = None
    for length, row in sorted(_stems.STEMS.items(), key=lambda kv: -len(kv[1].stem)):
        if core.startswith(row.stem):
            stem_len, rest = length, core[len(row.stem) :]
            break
    if stem_len is None:
        return None

    attach = "1"
    m = re.search(r"-(\d+)$", rest)
    if m:
        attach = m.group(1)
        rest = rest[: m.start()]
    parsed = _parse_unsaturation(rest)
    if parsed is None:
        return None
    unsats, residue = parsed
    if residue:  # leftover tokens we did not model -> abstain
        return None
    if str(attach) not in {str(i) for i in range(1, stem_len + 1)}:
        return None

    rw = Chem.RWMol()
    locants = {str(i): rw.AddAtom(Chem.Atom(6)) for i in range(1, stem_len + 1)}
    for i in range(1, stem_len):
        rw.AddBond(locants[str(i)], locants[str(i + 1)], Chem.BondType.SINGLE)
    for loc, order in unsats:
        if str(loc + 1) not in locants:
            return None
        bond = rw.GetBondBetweenAtoms(locants[str(loc)], locants[str(loc + 1)])
        bond.SetBondType(Chem.BondType.DOUBLE if order == 2 else Chem.BondType.TRIPLE)
    return rw, locants, locants[attach]


def _stem_length(stem: str) -> int | None:
    for length, row in _stems.STEMS.items():
        if row.stem == stem:
            return length
    return None


def _alkyl_chain(n: int) -> Numbered:
    rw = Chem.RWMol()
    locants: dict[str, int] = {}
    prev = None
    for i in range(1, n + 1):
        idx = rw.AddAtom(Chem.Atom(6))
        locants[str(i)] = idx
        if prev is not None:
            rw.AddBond(prev, idx, Chem.BondType.SINGLE)
        prev = idx
    return rw, locants, locants["1"]


def _cycloalkenyl_base(base: str) -> Numbered | None:
    """Parse an unsaturated carbocycle substituent (``cyclohex-1-en-1-yl``,
    ``cyclohexa-1,3-dien-1-yl``) into a numbered ring fragment."""
    core = base[len("cyclo") : -2].rstrip("-")
    sm = re.match(r"^([a-z]+)", core)
    if sm is None:
        return None
    stem = sm.group(1)
    n = _stem_length(stem) or _stem_length(stem.rstrip("a"))
    if n is None or n < 3:
        return None
    rest = core[sm.end() :]
    # The attachment locant must sit at the very end (``…-1``): otherwise this is
    # not a bare cycloalkenyl but a larger unit like ``…-1-ylsulfonyl`` whose
    # trailing ``yl`` we would wrongly swallow, silently dropping the operator.
    am = re.search(r"-(\d+)$", rest)
    if am is None:
        return None
    attach = am.group(1)
    rest = rest[: am.start()]
    unsats: list[tuple[int, int]] = []
    for m in re.finditer(r"(\d+(?:,\d+)*)-(en|dien|trien|tetraen|yn|diyn)", rest):
        order = 3 if m.group(2).endswith("yn") else 2
        unsats.extend((int(loc), order) for loc in m.group(1).split(","))
    # Everything between stem and attachment must be accounted for by unsaturation.
    if re.sub(r"(\d+(?:,\d+)*)-(?:en|dien|trien|tetraen|yn|diyn)", "", rest).strip("-"):
        return None
    if not unsats:
        return None
    rw, locants, _ = _carbocycle(n)
    for loc, order in unsats:
        hi = str(loc + 1) if loc < n else "1"  # last ring bond wraps back to locant 1
        a, b = locants.get(str(loc)), locants.get(hi)
        if a is None or b is None:
            return None
        bond = rw.GetBondBetweenAtoms(a, b)
        if bond is None:
            return None
        bond.SetBondType(Chem.BondType.DOUBLE if order == 2 else Chem.BondType.TRIPLE)
    if attach not in locants:
        return None
    return rw, locants, locants[attach]


def _carbocycle(n: int) -> Numbered:
    rw = Chem.RWMol()
    locants: dict[str, int] = {}
    idxs = []
    for i in range(1, n + 1):
        idx = rw.AddAtom(Chem.Atom(6))
        locants[str(i)] = idx
        idxs.append(idx)
    for a, b in zip(idxs, idxs[1:]):
        rw.AddBond(a, b, Chem.BondType.SINGLE)
    rw.AddBond(idxs[-1], idxs[0], Chem.BondType.SINGLE)
    return rw, locants, locants["1"]


def _ring_from_stem(name: str) -> Numbered | None:
    entry = _RING_STEMS.get("phenyl") if name == "phenyl" else _RING_STEMS.get(name)
    if entry is None:
        return None
    smiles, labels = entry
    return _numbered_from_smiles(smiles, labels, attach_locant="1")


def _ring_yl(stem: str, loc: str) -> Numbered | None:
    ring = _match_ring_stem(stem)
    if ring is None:
        return None
    entry = _ring_stem_template(ring)
    if entry is None:
        return None
    smiles, labels = entry
    if loc not in labels:
        return None
    return _numbered_from_smiles(smiles, labels, attach_locant=loc)


def _ring_stem_template(ring: str) -> tuple[str, list[str]] | None:
    """The ring's skeleton, from the table or built from its Hantzsch-Widman spec."""

    return _RING_STEMS.get(ring) or hw_parent_template(ring)


def _match_ring_stem(stem: str) -> str | None:
    for ring in (*_RING_STEMS, *hw_generated_names()):
        base = ring[:-1] if ring.endswith("e") else ring
        if stem == base or stem == ring:
            return ring
    return None


def _numbered_from_smiles(smiles: str, labels: list[str], attach_locant: str) -> Numbered | None:
    frag = Chem.MolFromSmiles(smiles)
    if frag is None or frag.GetNumAtoms() != len(labels):
        return None
    rw = Chem.RWMol(frag)
    locants = {label: i for i, label in enumerate(labels)}
    if attach_locant not in locants:
        return None
    return rw, locants, locants[attach_locant]


# --------------------------------------------------------------------------- #
# Prefix (sub-substituent) application
# --------------------------------------------------------------------------- #
_CLAUSE_RE = re.compile(r"(\d+(?:,\d+)*)-")


def _apply_prefix(rw: Chem.RWMol, locants: dict[str, int], prefix: str) -> bool:
    # A single leading parenthesised sub-substituent with no locant attaches at
    # position 1, e.g. "((4-methoxyphenyl)methyl)" -> phenyl-CH2- at C1.
    if prefix.startswith("(") and prefix.endswith(")") and _balanced(prefix[1:-1]) and "1" in locants:
        frag = resolve_fragment_mol(prefix)
        if frag is not None:
            return _graft_onto(rw, locants["1"], frag)
        return False
    # Several unlocanted ligands on the same atom, e.g. ``(amino)(imino)methyl``
    # -> C1 bears both -> amidine, or ``(oxo)cyclopropylmethyl`` -> an acyl.  The
    # base has already been peeled off the end, so what remains here is a ligand
    # list; each one must resolve on its own for the split to be accepted.
    if prefix.startswith("(") and "1" in locants:
        groups = _top_level_groups(prefix)
        # With a trailing *unparenthesised* group the split is ambiguous, and both
        # readings often resolve.  The leading clause's attachment bond settles it:
        # ``(oxo)pyrrolidin-1-ylmethyl`` leads with a double-bond ligand, which can
        # only describe C1's own bonding, so those are siblings (an acyl); a
        # single-bond lead as in ``(2,2,2-trifluoroethyl)sulfamoylaminomethyl`` is
        # the inner ligand of the operator that follows it, so the whole prefix is
        # one substituent.
        if not all(group.startswith("(") for group in groups) and not _attaches_by_double_bond(
            resolve_fragment_mol(groups[0])
        ):
            whole = resolve_fragment_mol(prefix)
            # The ``whole`` reading is only legitimate when the trailing group is
            # an *operator* that absorbs the leading ligand at one of its own
            # internal sites (``(trifluoroethyl)sulfamoylamino`` -> the ethyl on
            # the distal N).  When the trailing group is instead a complete ring-
            # or chain-yl, an unlocanted leading ligand has nowhere legitimate to
            # go and ``resolve_fragment_mol`` piles it onto the attachment atom
            # itself (``(aryl)indan-1-yl`` -> aryl on indane C1).  That over-
            # substitution of the attachment atom marks the reading as spurious,
            # so those are siblings on the base instead.
            if whole is not None and not _ligand_on_attachment_atom(whole, groups[-1]):
                return _graft_onto(rw, locants["1"], whole)
        if len(groups) >= 2:
            frags = [resolve_fragment_mol(group) for group in groups]
            if all(frag is not None for frag in frags):
                for frag in frags:
                    if not _graft_onto(rw, locants["1"], frag):
                        return False
                return True
    # A prefix that does not *begin* with a depth-0 locant is a single unlocanted
    # sub-substituent binding at position 1 — either a leaf (``chloromethyl`` ->
    # Cl on C1) or a whole ring-yl (``piperazin-1-ylmethyl`` -> piperazin-1-yl on
    # C1, whose own ``1`` is an internal ring locant, not a base-attachment one).
    starts = _clause_starts(prefix)
    if (not starts or starts[0] != 0) and not prefix.startswith("("):
        # A bare unlocanted substituent (leaf or whole ring-yl). A parenthesised
        # lead (e.g. ``(oxo)pyrrolidin-1-yl``) is a concatenation of several
        # sub-substituents we do not split here, so it falls through and abstains.
        return _apply_unlocanted_prefix(rw, locants, prefix)
    clauses = _parse_clauses(prefix)
    if clauses is None:
        return False
    for locs, subname in clauses:
        frag = resolve_fragment_mol(subname)
        if frag is None:
            return False
        for loc in locs:
            base_idx = locants.get(loc)
            if base_idx is None:
                return False
            if not _graft_onto(rw, base_idx, frag):
                return False
    return True


def _attaches_by_double_bond(frag: Chem.Mol | None) -> bool:
    """Whether ``frag`` bonds to its parent through a double bond (``oxo``,
    ``imino``, any ``…ylidene``).  ``None`` counts as ``False``: an unresolvable
    clause tells us nothing about how it would have attached."""

    if frag is None:
        return False
    dummy = next((a for a in frag.GetAtoms() if a.GetAtomicNum() == 0), None)
    if dummy is None:
        return False
    return any(b.GetBondType() == Chem.BondType.DOUBLE for b in dummy.GetBonds())


def _attachment_heavy_degree(frag: Chem.Mol | None) -> int | None:
    """Heavy-atom degree of ``frag``'s attachment atom (the dummy's neighbour),
    excluding the dummy itself, or ``None`` if there is no single such atom."""

    if frag is None:
        return None
    idx = _attachment_atom(frag)
    if idx is None:
        return None
    return sum(1 for n in frag.GetAtomWithIdx(idx).GetNeighbors() if n.GetAtomicNum() != 0)


def _ligand_on_attachment_atom(whole: Chem.Mol, trailing: str) -> bool:
    """Whether the leading ligand(s) in a ``whole`` reading landed on the trailing
    group's own attachment atom rather than an internal site.

    Compares the attachment atom's heavy-degree in ``whole`` against the trailing
    group resolved on its own: a strict increase means the ligand piled onto the
    attachment carbon (an unlocanted substitution a complete substituent cannot
    take), so the ``whole`` reading is spurious.  An indeterminate comparison is
    treated as *not* piled on, leaving the existing behaviour unchanged."""

    bare = _attachment_heavy_degree(resolve_fragment_mol(trailing))
    combined = _attachment_heavy_degree(whole)
    if bare is None or combined is None:
        return False
    return combined > bare


def _apply_unlocanted_prefix(rw: Chem.RWMol, locants: dict[str, int], prefix: str) -> bool:
    if "1" not in locants:
        return False
    leaf = _leading_multiplier(prefix)
    if leaf is not None:
        count, body = leaf
    else:
        # Otherwise the multiplier may belong to a group *inside* the prefix
        # rather than counting copies of it: ``diphenylmethyl`` is two phenyls on
        # C1, but ``dihydroxyphosphorylmethyl`` is one phosphoryl wearing two
        # hydroxys, not two ``hydroxyphosphoryl``s.
        count, body = _multiplied_ligand(prefix)
    frag = resolve_fragment_mol(body)
    if frag is None:
        return _apply_leaf_run(rw, locants, prefix) or _apply_leaf_led_sibling_run(rw, locants, prefix)
    for _ in range(count):
        if not _graft_onto(rw, locants["1"], frag):
            return False
    return True


def _apply_leaf_led_sibling_run(rw: Chem.RWMol, locants: dict[str, int], prefix: str) -> bool:
    """Graft a leaf run followed by parenthesised ligands, all onto position 1 —
    ``iodo(methylphosphanyl)methyl`` is an iodine *and* a phosphanyl on the same
    carbon.

    Only a *leading* leaf run is read this way.  A leaf hosts no ligands of its
    own, so nothing that follows it can be its argument and the pieces can only
    be siblings; a leaf run *after* a parenthesised group is instead left to
    :func:`_apply_prefix`, which has to weigh it against the reading where the
    trailing text is an operator absorbing the group before it.
    """

    groups = _top_level_groups(prefix)
    if len(groups) < 2 or groups[0].startswith("("):
        return False
    if not all(group.startswith("(") for group in groups[1:]):
        return False
    frags = [resolve_fragment_mol(group) for group in groups[1:]]
    if any(frag is None for frag in frags):
        return False
    if not _apply_leaf_run(rw, locants, groups[0]):
        return False
    return all(_graft_onto(rw, locants["1"], frag) for frag in frags)


def _split_leaf_run(prefix: str) -> list[tuple[int, str]] | None:
    """Split a *run* of leaf prefixes — ``chlorodifluoro`` -> one chlorine and two
    fluorines — or ``None`` if the string is not exactly such a run.

    Only leaves qualify.  A leaf takes no ligands of its own, so each piece is
    unambiguous once matched and a greedy longest-first walk cannot strand a
    valid parse.  Requiring the walk to consume the *whole* string is what keeps
    a compound prefix from being shredded into fragments that merely look like
    leaves: any leftover means the caller abstains instead.
    """

    grafts: list[tuple[int, str]] = []
    rest = prefix
    while rest:
        for count, tail in (*_multipliers.candidate_splits(rest), (1, rest)):
            leaf = next((w for w in _LEAVES_LONGEST_FIRST if tail.startswith(w)), None)
            if leaf is not None:
                grafts.append((count, leaf))
                rest = tail[len(leaf) :]
                break
        else:
            return None
    return grafts or None


def _apply_leaf_run(rw: Chem.RWMol, locants: dict[str, int], prefix: str) -> bool:
    grafts = _split_leaf_run(prefix)
    if grafts is None:
        return False
    for count, leaf in grafts:
        frag = resolve_fragment_mol(leaf)
        if frag is None:
            return False
        for _ in range(count):
            if not _graft_onto(rw, locants["1"], frag):
                return False
    return True


def _parse_clauses(prefix: str) -> list[tuple[list[str], str]] | None:
    """Parse ``4-methoxy``, ``3,4-dimethoxy``, ``2-(pyrrolidin-1-yl)`` … into
    (locants, substituent-name) pairs. Requires explicit locants, and only
    splits on locant hyphens at parenthesis depth 0 so nested names stay
    intact."""
    starts = _clause_starts(prefix)
    if not starts or starts[0] != 0:
        return None
    clauses: list[tuple[list[str], str]] = []
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(prefix)
        segment = prefix[start:end]
        m = re.match(r"^(\d+(?:,\d+)*)-(.*)$", segment, re.DOTALL)
        if not m:
            return None
        locs = m.group(1).split(",")
        body = m.group(2).rstrip("-")  # drop the separator hyphen before the next clause
        leaf = _leading_multiplier(body)
        if leaf is not None:
            count, body = leaf
            if len(locs) != count:
                return None
        elif len(locs) > 1:
            # Inside a locanted clause the count is already pinned by the
            # locants, so a multiplier can be stripped whenever it agrees with
            # them.  That is what makes the ``bis``/``tris`` spellings readable:
            # they take a *parenthesised* argument rather than the bare leaf
            # ``_leading_multiplier`` insists on, as in ``3,5-bis(trifluoro-
            # methyl)phenyl``.
            for count, rest in _multipliers.candidate_splits(body):
                if count == len(locs) and _resolvable(rest):
                    body = rest
                    break
        clauses.append((locs, body))
    return clauses


def _clause_starts(prefix: str) -> list[int]:
    """Indices where a depth-0 locant group (``\\d+(,\\d+)*-``) begins."""
    starts: list[int] = []
    for m in re.finditer(r"\d+(?:,\d+)*-", prefix):
        # depth at the match start
        d = prefix[: m.start()].count("(") - prefix[: m.start()].count(")")
        if d == 0:
            starts.append(m.start())
    return starts


def _leading_multiplier(body: str) -> tuple[int, str] | None:
    """Split a multiplied *leaf* — ``difluoro`` -> ``(2, "fluoro")`` — else ``None``.

    A leaf takes no ligands of its own, so a multiplier in front of one can only
    be counting copies.  That makes this reading unambiguous, unlike a multiplier
    over a compound name (see :func:`_multiplied_ligand`)."""

    for count, rest in _multipliers.candidate_splits(body):
        if rest in _LEAF_SMILES:
            return count, rest
    return None


def _graft_onto(rw: Chem.RWMol, base_idx: int, frag: Chem.Mol) -> bool:
    dummy = next((a for a in frag.GetAtoms() if a.GetAtomicNum() == 0), None)
    if dummy is None:
        return False
    frag_to_new: dict[int, int] = {}
    for atom in frag.GetAtoms():
        if atom.GetIdx() == dummy.GetIdx():
            continue
        frag_to_new[atom.GetIdx()] = rw.AddAtom(_clone_atom(atom))
    for bond in frag.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if dummy.GetIdx() in (a, b):
            other = b if a == dummy.GetIdx() else a
            _consume_hydrogen(rw.GetAtomWithIdx(base_idx), bond.GetBondType())
            rw.AddBond(base_idx, frag_to_new[other], bond.GetBondType())
            _copy_bond_stereo_tag(bond, rw.GetBondBetweenAtoms(base_idx, frag_to_new[other]))
        else:
            rw.AddBond(frag_to_new[a], frag_to_new[b], bond.GetBondType())
            _copy_bond_stereo_tag(bond, rw.GetBondBetweenAtoms(frag_to_new[a], frag_to_new[b]))
    return True


def _copy_bond_stereo_tag(src: Chem.Bond, dst: Chem.Bond | None) -> None:
    if dst is not None and src.HasProp(_NAME_CIP):
        dst.SetProp(_NAME_CIP, src.GetProp(_NAME_CIP))


def _clone_atom(atom: Chem.Atom) -> Chem.Atom:
    """Copy an atom's element/charge/H state and its name-asserted stereo tag, so
    the tag survives grafting into a larger fragment."""
    na = Chem.Atom(atom.GetAtomicNum())
    na.SetFormalCharge(atom.GetFormalCharge())
    na.SetNumExplicitHs(atom.GetNumExplicitHs())
    na.SetNoImplicit(atom.GetNoImplicit())
    for prop in (_NAME_CIP, _NAME_RELATIVE):
        if atom.HasProp(prop):
            na.SetProp(prop, atom.GetProp(prop))
    return na


def _consume_hydrogen(atom: Chem.Atom, bond_type: Chem.BondType) -> None:
    """Substituting at an atom with an explicit hydrogen (e.g. a pyrrole-type NH)
    consumes that hydrogen; implicit-H atoms are left for RDKit to rebalance."""
    explicit = atom.GetNumExplicitHs()
    if explicit <= 0:
        return
    order = 2 if bond_type == Chem.BondType.DOUBLE else 3 if bond_type == Chem.BondType.TRIPLE else 1
    atom.SetNumExplicitHs(max(0, explicit - order))


# A parenthesised stereo-descriptor group: only stereo characters inside, with
# at least one CIP/EZ letter, e.g. "(1s,4s)", "(R)", "(2R,3S)", "(E)".
_STEREO_GROUP_RE = re.compile(r"^\(([0-9RSrsEZ,\s*'\-]+)\)-")
_STEREO_WORD_RE = re.compile(r"^(rel|rac|cis|trans|syn|anti|endo|exo|\(\+/?-\)|\(±\))-")


def _strip_outer_parens(name: str) -> str:
    name = name.strip()
    changed = True
    while changed:
        changed = False
        while name.startswith("(") and name.endswith(")") and _balanced(name[1:-1]):
            name = name[1:-1].strip()
            changed = True
        stripped = _strip_leading_stereo(name)
        if stripped != name:
            name = stripped
            changed = True
    return name


def _strip_leading_stereo(name: str) -> str:
    """Remove a leading stereo-descriptor prefix. Stereo does not affect the
    constitution fragment (it is verified separately), so dropping it is safe."""
    m = _STEREO_GROUP_RE.match(name)
    if m and re.search(r"[RSrsEZ]", m.group(1)):
        return name[m.end() :].strip()
    m = _STEREO_WORD_RE.match(name)
    if m:
        return name[m.end() :].strip()
    return name


def _balanced(s: str) -> bool:
    depth = 0
    for ch in s:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0
