# openclatura/rules/elements.py
from dataclasses import dataclass


@dataclass(frozen=True)
class Element:
    symbol: str
    name: str
    atomic_number: int
    standard_valence: int
    hw_stem: str | None
    hw_priority: int | None
    substituent_prefix: str | None


ELEMENTS: dict[str, Element] = {
    "H": Element("H", "hydrogen", 1, 1, None, None, "hydro"),
    "B": Element("B", "boron", 5, 3, "bora", 18, "boryl"),
    "C": Element("C", "carbon", 6, 4, None, None, None),
    "N": Element("N", "nitrogen", 7, 3, "aza", 9, "amino"),
    "O": Element("O", "oxygen", 8, 2, "oxa", 5, "oxy"),
    "F": Element("F", "fluorine", 9, 1, "fluora", 1, "fluoro"),
    "Si": Element("Si", "silicon", 14, 4, "sila", 14, "silyl"),
    "P": Element("P", "phosphorus", 15, 3, "phospha", 10, "phosphanyl"),
    "S": Element("S", "sulfur", 16, 2, "thia", 6, "sulfanyl"),
    "Cl": Element("Cl", "chlorine", 17, 1, "chlora", 2, "chloro"),
    "As": Element("As", "arsenic", 33, 3, "arsa", 11, "arsanyl"),
    "Se": Element("Se", "selenium", 34, 2, "selena", 7, "selanyl"),
    "Br": Element("Br", "bromine", 35, 1, "broma", 3, "bromo"),
    "Sb": Element("Sb", "antimony", 51, 3, "stiba", 12, "stibanyl"),
    "Te": Element("Te", "tellurium", 52, 2, "tellura", 8, "tellanyl"),
    "I": Element("I", "iodine", 53, 1, "ioda", 4, "iodo"),
    "Bi": Element("Bi", "bismuth", 83, 3, "bisma", 13, "bismuthanyl"),
    "Li": Element("Li", "lithium", 3, 1, None, None, None),
    "Na": Element("Na", "sodium", 11, 1, None, None, None),
    "K": Element("K", "potassium", 19, 1, None, None, None),
    "Mg": Element("Mg", "magnesium", 12, 2, None, None, None),
    "Ca": Element("Ca", "calcium", 20, 2, None, None, None),
}


def get(symbol: str) -> Element:
    return ELEMENTS[symbol]


def is_known(symbol: str) -> bool:
    return symbol in ELEMENTS
