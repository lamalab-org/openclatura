# structure-to-iupac/rules/substituents.py

from dataclasses import dataclass


@dataclass(frozen=True)
class Substituent:
    key: str             # Internal identifier, e.g. "chloro"
    prefix: str          # Name as cited in the final IUPAC name, e.g. "chloro"
    needs_locant: bool   # True if locant is required (e.g. on a chain); False for trivial cases


# Always-prefix substituents.
# Halogens use simple "halo-" forms (fluoro, chloro, bromo, iodo).
# Other prefix-only groups: nitro, nitroso, azido, diazo, isocyano, etc.
SUBSTITUENTS: dict[str, Substituent] = {
    # --- Halogens ---
    "fluoro":    Substituent("fluoro",    "fluoro",    needs_locant=True),
    "chloro":    Substituent("chloro",    "chloro",    needs_locant=True),
    "bromo":     Substituent("bromo",     "bromo",     needs_locant=True),
    "iodo":      Substituent("iodo",      "iodo",      needs_locant=True),
    "astato":    Substituent("astato",    "astato",    needs_locant=True),

    # --- Nitrogen-based prefix-only groups ---
    "nitro":     Substituent("nitro",     "nitro",     needs_locant=True),   # -NO2
    "nitroso":   Substituent("nitroso",   "nitroso",   needs_locant=True),   # -N=O
    "azido":     Substituent("azido",     "azido",     needs_locant=True),   # -N3
    "diazo":     Substituent("diazo",     "diazo",     needs_locant=True),   # =N2
    "diazonio":  Substituent("diazonio",  "diazonio",  needs_locant=True),   # -N2+
    "isocyano":  Substituent("isocyano",  "isocyano",  needs_locant=True),   # -NC
    "cyanato":   Substituent("cyanato",   "cyanato",   needs_locant=True),   # -OCN
    "isocyanato":Substituent("isocyanato","isocyanato",needs_locant=True),   # -NCO
    "thiocyanato":   Substituent("thiocyanato",   "thiocyanato",   needs_locant=True),  # -SCN
    "isothiocyanato":Substituent("isothiocyanato","isothiocyanato",needs_locant=True),  # -NCS

    # --- Oxygen / chalcogen prefix-only groups (when not principal) ---
    # Note: hydroxy, oxo, etc. live in suffixes.py because they CAN be principal.
    # These ones cannot be principal characteristic groups under PIN rules.
    "hydroperoxy":   Substituent("hydroperoxy",   "hydroperoxy",   needs_locant=True),  # -OOH
    "peroxy":        Substituent("peroxy",        "peroxy",        needs_locant=True),  # -OO- (linker form)

    # --- Sulfur-based prefix-only groups ---
    # Note: -SH (sulfanyl/thiol) and -SO3H (sulfo) CAN be principal in PINs;
    # they belong in suffixes.py if/when you add them.
    "sulfanyl":  Substituent("sulfanyl",  "sulfanyl",  needs_locant=True),   # -SH as substituent (also: "mercapto" in older lit)

    # --- Silicon / phosphorus / boron simple substituents ---
    "silyl":     Substituent("silyl",     "silyl",     needs_locant=True),   # -SiH3
    "phosphanyl":Substituent("phosphanyl","phosphanyl",needs_locant=True),   # -PH2 (older: "phosphino")
    "phosphoryl":Substituent("phosphoryl","phosphoryl",needs_locant=True),   # -P(=O)H2
    "boryl":     Substituent("boryl",     "boryl",     needs_locant=True),   # -BH2
}


def get(key: str) -> Substituent:
    """Look up a substituent by key."""
    return SUBSTITUENTS[key]


def is_known(key: str) -> bool:
    return key in SUBSTITUENTS