# structure-to-iupac/rules/suffixes.py
from dataclasses import dataclass

@dataclass(frozen=True)
class CharacteristicGroup:
    key: str
    seniority: int
    suffix: str
    suffix_with_locant: bool
    prefix: str | None
    multi_suffix: str | None

GROUPS: dict[str, CharacteristicGroup] = {
    "olate": CharacteristicGroup("olate", 15, "olate", True, "oxido", "diolate"),
    "thiolate": CharacteristicGroup("thiolate", 16, "thiolate", True, "sulfido", "dithiolate"),
    
    "carboxylic_acid": CharacteristicGroup("carboxylic_acid", 20, "oic acid", False, "carboxy", "dioic acid"),
    "carboxylate": CharacteristicGroup("carboxylate", 21, "oate", False, "carboxylato", "dioate"),
    
    "ring_carboxylic_acid": CharacteristicGroup("ring_carboxylic_acid", 20, "carboxylic acid", True, "carboxy", "dicarboxylic acid"),
    "ring_carboxylate": CharacteristicGroup("ring_carboxylate", 21, "carboxylate", True, "carboxylato", "dicarboxylate"),
    
    "peroxy_acid": CharacteristicGroup("peroxy_acid", 22, "peroxoic acid", False, "carboperoxy", "diperoxoic acid"),
    "ring_peroxy_acid": CharacteristicGroup("ring_peroxy_acid", 22, "carboperoxoic acid", True, "carboperoxy", "dicarboperoxoic acid"),
    "peroxy_ester": CharacteristicGroup("peroxy_ester", 45, "peroxoate", False, "oxycarbonyl", None),
    "ring_peroxy_ester": CharacteristicGroup("ring_peroxy_ester", 45, "carboperoxoate", True, "oxycarbonyl", None),
    
    "sulfonic_acid": CharacteristicGroup("sulfonic_acid", 25, "sulfonic acid", True, "sulfo", "disulfonic acid"),
    "sulfonate": CharacteristicGroup("sulfonate", 26, "sulfonate", True, "sulfonato", "disulfonate"),
    "anhydride": CharacteristicGroup("anhydride", 30, "oic anhydride", False, None, None),
    
    "ester": CharacteristicGroup("ester", 40, "oate", False, "oxycarbonyl", None),
    
    "acid_fluoride": CharacteristicGroup("acid_fluoride", 50, "oyl fluoride", False, "fluorocarbonyl", "dioyl difluoride"),
    "acid_chloride": CharacteristicGroup("acid_chloride", 51, "oyl chloride", False, "chlorocarbonyl", "dioyl dichloride"),
    "acid_bromide": CharacteristicGroup("acid_bromide", 52, "oyl bromide", False, "bromocarbonyl", "dioyl dibromide"),
    "acid_iodide": CharacteristicGroup("acid_iodide", 53, "oyl iodide", False, "iodocarbonyl", "dioyl diiodide"),
    
    "ring_acid_fluoride": CharacteristicGroup("ring_acid_fluoride", 50, "carbonyl fluoride", True, "fluorocarbonyl", "dicarbonyl difluoride"),
    "ring_acid_chloride": CharacteristicGroup("ring_acid_chloride", 51, "carbonyl chloride", True, "chlorocarbonyl", "dicarbonyl dichloride"),
    "ring_acid_bromide": CharacteristicGroup("ring_acid_bromide", 52, "carbonyl bromide", True, "bromocarbonyl", "dicarbonyl dibromide"),
    "ring_acid_iodide": CharacteristicGroup("ring_acid_iodide", 53, "carbonyl iodide", True, "iodocarbonyl", "dicarbonyl diiodide"),
    
    "amide": CharacteristicGroup("amide", 60, "amide", False, "carbamoyl", "diamide"),
    "ring_amide": CharacteristicGroup("ring_amide", 60, "carboxamide", True, "carbamoyl", "dicarboxamide"),
    
    "thioamide": CharacteristicGroup("thioamide", 65, "thioamide", False, "carbamothioyl", "dithioamide"),
    "ring_thioamide": CharacteristicGroup("ring_thioamide", 65, "carbothioamide", True, "carbamothioyl", "dicarbothioamide"),
    
    "nitrile": CharacteristicGroup("nitrile", 70, "nitrile", False, "cyano", "dinitrile"),
    "ring_nitrile": CharacteristicGroup("ring_nitrile", 70, "carbonitrile", True, "cyano", "dicarbonitrile"),
    
    "aldehyde": CharacteristicGroup("aldehyde", 80, "al", False, "oxo", "dial"),
    "ring_aldehyde": CharacteristicGroup("ring_aldehyde", 80, "carbaldehyde", True, "formyl", "dicarbaldehyde"),
    
    "ketone": CharacteristicGroup("ketone", 90, "one", True, "oxo", "dione"),
    "hydrazone": CharacteristicGroup("hydrazone", 95, "one hydrazone", True, "hydrazono", "dione dihydrazone"),
    "aldehyde_hydrazone": CharacteristicGroup("aldehyde_hydrazone", 95, "al hydrazone", False, "hydrazono", "dial dihydrazone"),
    "ring_aldehyde_hydrazone": CharacteristicGroup("ring_aldehyde_hydrazone", 95, "carbaldehyde hydrazone", True, "hydrazonomethyl", "dicarbaldehyde dihydrazone"),
    
    "alcohol": CharacteristicGroup("alcohol", 100, "ol", True, "hydroxy", "diol"),
    "thiol": CharacteristicGroup("thiol", 105, "thiol", True, "sulfanyl", "dithiol"),
    
    "amine": CharacteristicGroup("amine", 110, "amine", True, "amino", "diamine"),
    "aminium": CharacteristicGroup("aminium", 109, "aminium", True, "ammonio", "diaminium"),
    "imine": CharacteristicGroup("imine", 112, "imine", True, "imino", "diimine"),
    "iminium": CharacteristicGroup("iminium", 111, "iminium", True, "iminio", "diiminium"),
    "hydrazine": CharacteristicGroup("hydrazine", 115, "hydrazine", True, "hydrazinyl", "dihydrazine"),

    "ether": CharacteristicGroup("ether", 200, "ether", False, "oxy", None),
}

def get(key: str) -> CharacteristicGroup: return GROUPS[key]
def most_senior(keys: list[str]) -> CharacteristicGroup: return min((GROUPS[k] for k in keys), key=lambda g: g.seniority)
