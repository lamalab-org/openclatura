from .hantzsch_widman import hw_spec_for_name, spec_cites_indicated_hydrogen
from .naming_data import load_json_table
from .nomenclature import RULES


def _indicated_h_retained_names() -> frozenset[str]:
    """
    Retained parents that carry an indicated hydrogen, read off the templates.
    """

    return frozenset(
        row["indicated_hydrogen_name"]
        for row in load_json_table("retained_fused_graph_templates.json").get("parents", ())
        if "indicated_hydrogen_name" in row
    )


INDICATED_H_RETAINED_NAMES = _indicated_h_retained_names()


def cites_indicated_hydrogen(retained_name: str | None) -> bool:
    """
    Whether a parent name spells its own indicated hydrogen ("1H-azepine").
    """

    if retained_name is None:
        return False
    spec = hw_spec_for_name(retained_name) or _monocycle_spec(retained_name)
    if spec is not None:
        return spec_cites_indicated_hydrogen(spec)
    return retained_name in INDICATED_H_RETAINED_NAMES


def _monocycle_spec(name: str) -> dict | None:
    return next((spec for spec in RULES.retained.monocycle_specs if spec["name"] == name), None)


INDICATED_H_ELEMENTS = frozenset({"C", "N", "P", "As", "Sb", "Bi", "B", "Si", "Ge", "Sn", "Pb"})
ALKYL_OXY_PREFIXES = RULES.heteroatoms.alkyl_oxy_prefixes
SIMPLE_SULFANYL_PREFIXES = RULES.heteroatoms.simple_sulfanyl_prefixes
SIMPLE_SELANYL_PREFIXES = RULES.heteroatoms.simple_selanyl_prefixes
HALOGEN_PREFIXES = RULES.heteroatoms.halogen_prefixes
HALOGEN_LAMBDA_SUFFIXES = RULES.heteroatoms.halogen_lambda_suffixes
RETAINED_RING_ELEMENTS = RULES.retained.ring_elements
SALT_METAL_NAMES = RULES.components.salt_metal_names
