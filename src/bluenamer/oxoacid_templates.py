"""OPSIN-checked rendering templates for central oxoacid roles."""

from dataclasses import dataclass
from enum import StrEnum

from .molecule import Molecule
from .grammar_snapshot_data import oxoacid_ester_suffix_templates
from .oxoacid_roles import CentralOxoRole, OxoLigandRole


class OxoacidTemplateKind(StrEnum):
    """How a central-oxo role is rendered."""

    ESTER_SUFFIX = "ester_suffix"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class OxoacidRoleTemplate:
    """A graph-role rendering template with verification metadata."""

    key: str
    kind: OxoacidTemplateKind
    central_symbols: frozenset[str]
    counts: dict[OxoLigandRole, int]
    suffix: str = ""
    center_charge: int | None = None
    opsin_verified: bool = False
    preserves_formal_charges: bool = False
    reason: str = ""

    def matches(self, mol: Molecule, role: CentralOxoRole) -> bool:
        if role.central_symbol not in self.central_symbols:
            return False
        if self.center_charge is not None and mol.atoms[role.central].charge != self.center_charge:
            return False
        for ligand_role in OxoLigandRole:
            expected = self.counts.get(ligand_role, 0)
            if role.count(ligand_role) != expected:
                return False
        return True


_SUFFIX_TEMPLATES = oxoacid_ester_suffix_templates()
_PHOSPHATE_SUFFIXES = _SUFFIX_TEMPLATES["phosphate"]
_HALOGEN_PEROXY_SUFFIXES = _SUFFIX_TEMPLATES["charge_normalized_halogen_peroxy_oxoester"]

OXOACID_ROLE_TEMPLATES: tuple[OxoacidRoleTemplate, ...] = (
    OxoacidRoleTemplate(
        key="phosphate_monoester_neutral",
        kind=OxoacidTemplateKind.ESTER_SUFFIX,
        central_symbols=frozenset({"P"}),
        counts={OxoLigandRole.OXO: 1, OxoLigandRole.HYDROXY: 2, OxoLigandRole.ALKOXY: 1},
        suffix=_PHOSPHATE_SUFFIXES["neutral_monoester"],
        center_charge=0,
        opsin_verified=True,
        preserves_formal_charges=True,
        reason="OPSIN parses e.g. methyl dihydrogen phosphate as P(=O)(OC)(O)O.",
    ),
    OxoacidRoleTemplate(
        key="phosphate_monoester_monoanion",
        kind=OxoacidTemplateKind.ESTER_SUFFIX,
        central_symbols=frozenset({"P"}),
        counts={
            OxoLigandRole.OXO: 1,
            OxoLigandRole.HYDROXY: 1,
            OxoLigandRole.OXIDO: 1,
            OxoLigandRole.ALKOXY: 1,
        },
        suffix=_PHOSPHATE_SUFFIXES["monoanion_monoester"],
        center_charge=0,
        opsin_verified=True,
        preserves_formal_charges=True,
        reason="OPSIN parses e.g. methyl hydrogen phosphate as P(=O)(OC)(O)[O-].",
    ),
    OxoacidRoleTemplate(
        key="phosphate_monoester_dianion",
        kind=OxoacidTemplateKind.ESTER_SUFFIX,
        central_symbols=frozenset({"P"}),
        counts={OxoLigandRole.OXO: 1, OxoLigandRole.OXIDO: 2, OxoLigandRole.ALKOXY: 1},
        suffix=_PHOSPHATE_SUFFIXES["dianion_monoester"],
        center_charge=0,
        opsin_verified=True,
        preserves_formal_charges=True,
        reason="OPSIN parses e.g. methyl phosphate as P(=O)(OC)([O-])[O-].",
    ),
    OxoacidRoleTemplate(
        key="charge_normalized_halogen_oxoester",
        kind=OxoacidTemplateKind.ESTER_SUFFIX,
        central_symbols=frozenset({"Cl", "Br", "I"}),
        counts={OxoLigandRole.OXO: 2, OxoLigandRole.ALKOXY: 1},
        center_charge=2,
        opsin_verified=True,
        preserves_formal_charges=False,
        reason=(
            "OPSIN accepts chlorate/bromate/iodate ester grammar, but normalizes "
            "charge-separated halogen oxo ligands to neutral X=O bonds."
        ),
    ),
    OxoacidRoleTemplate(
        key="charge_normalized_chlorine_peroxy_oxoester",
        kind=OxoacidTemplateKind.ESTER_SUFFIX,
        central_symbols=frozenset({"Cl"}),
        counts={OxoLigandRole.OXO: 2, OxoLigandRole.PEROXY: 1},
        suffix=_HALOGEN_PEROXY_SUFFIXES["Cl"],
        center_charge=2,
        opsin_verified=True,
        preserves_formal_charges=True,
        reason=(
            "OPSIN parses methyl peroxychlorate as Cl(=O)(=O)OOC; RDKit "
            "canonicalization preserves the charge-normalized [Cl+2]([O-])2 graph."
        ),
    ),
    OxoacidRoleTemplate(
        key="charge_normalized_bromine_peroxy_oxoester",
        kind=OxoacidTemplateKind.ESTER_SUFFIX,
        central_symbols=frozenset({"Br"}),
        counts={OxoLigandRole.OXO: 2, OxoLigandRole.PEROXY: 1},
        suffix=_HALOGEN_PEROXY_SUFFIXES["Br"],
        center_charge=2,
        opsin_verified=True,
        preserves_formal_charges=True,
        reason="Common peroxyhalate ester template, parallel to methyl peroxychlorate.",
    ),
    OxoacidRoleTemplate(
        key="charge_normalized_iodine_peroxy_oxoester",
        kind=OxoacidTemplateKind.ESTER_SUFFIX,
        central_symbols=frozenset({"I"}),
        counts={OxoLigandRole.OXO: 2, OxoLigandRole.PEROXY: 1},
        suffix=_HALOGEN_PEROXY_SUFFIXES["I"],
        center_charge=2,
        opsin_verified=True,
        preserves_formal_charges=True,
        reason="Common peroxyhalate ester template, parallel to methyl peroxychlorate.",
    ),
)


def oxoacid_role_template(mol: Molecule, role: CentralOxoRole) -> OxoacidRoleTemplate | None:
    """Return the most specific template for a central oxo role."""

    for template in OXOACID_ROLE_TEMPLATES:
        if template.matches(mol, role):
            return template
    return None


def unsupported_oxoacid_role_template(mol: Molecule, role: CentralOxoRole) -> OxoacidRoleTemplate | None:
    """Return an unsupported template certificate for high-risk oxo roles."""

    template = oxoacid_role_template(mol, role)
    if template is not None and template.kind == OxoacidTemplateKind.UNSUPPORTED:
        return template
    return None
