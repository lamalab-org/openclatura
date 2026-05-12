"""Compatibility aliases for the central nomenclature registry.

New code should import ``RULES`` from :mod:`structure_to_iupac.nomenclature`.
These names remain for existing modules and external imports.
"""

from .nomenclature import RULES


INDICATED_H_RETAINED_NAMES = RULES.retained.indicated_hydrogen_names
ALKYL_OXY_PREFIXES = RULES.heteroatoms.alkyl_oxy_prefixes
SIMPLE_SULFANYL_PREFIXES = RULES.heteroatoms.simple_sulfanyl_prefixes
SIMPLE_SELANYL_PREFIXES = RULES.heteroatoms.simple_selanyl_prefixes
HALOGEN_PREFIXES = RULES.heteroatoms.halogen_prefixes
HALOGEN_LAMBDA_SUFFIXES = RULES.heteroatoms.halogen_lambda_suffixes
RETAINED_RING_ELEMENTS = RULES.retained.ring_elements
DIRECT_GROUP_PREFIXES = RULES.prefixes.direct_group_prefixes
CHAIN_EXTERNAL_CARBONYL_GROUPS = RULES.components.chain_external_carbonyl_groups
PREFIX_GROUPS_TO_SKIP = RULES.prefixes.skip_groups
ESTER_LIKE_PREFIX_GROUPS = RULES.prefixes.ester_like_groups
PEROXY_ESTER_GROUPS = RULES.prefixes.peroxy_ester_groups
AMIDE_LIKE_PREFIX_GROUPS = RULES.prefixes.amide_like_groups
AMIDE_PREFIX_BASES = RULES.prefixes.amide_bases
CARBOXY_PREFIX_GROUPS = RULES.prefixes.carboxy_groups
CYANO_PREFIX_GROUPS = RULES.prefixes.cyano_groups
ACID_HALIDE_PREFIXES = RULES.prefixes.acid_halide_prefixes
PEROXY_ACID_PREFIX_GROUPS = RULES.prefixes.peroxy_acid_groups
SULFONYL_PREFIX_GROUPS = RULES.prefixes.sulfonyl_groups
DIRECT_PREFIX_GROUPS = RULES.prefixes.direct_prefixes
FRONT_MODIFIER_PRINCIPAL_GROUPS = RULES.components.front_modifier_principal_groups
N_SUBSTITUENT_PRINCIPAL_GROUPS = RULES.components.n_substituent_principal_groups
HYDRAZONE_PRINCIPAL_GROUPS = RULES.components.hydrazone_principal_groups
SPECIAL_COMPONENT_NAMES = RULES.components.special_names
SINGLE_ATOM_CATIONS = RULES.ions.single_atom_cations
SINGLE_ATOM_ANIONS = RULES.ions.single_atom_anions
SALT_METAL_NAMES = RULES.components.salt_metal_names
