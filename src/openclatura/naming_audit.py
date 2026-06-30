"""Graph coverage checks for generated component names."""

from dataclasses import dataclass

from .assembly_parts import AssemblyParts
from .charge_pair_roles import ChargePairRole, charge_pair_roles
from .molecule import Molecule
from .name_bindings import refresh_name_atom_bindings


@dataclass(frozen=True)
class NamingCoverage:
    """Atom coverage for a named component."""

    named_atoms: set[int]
    unnamed_atoms: set[int]


@dataclass(frozen=True)
class ChargePairTemplateAudit:
    """Whether every formal charge pair has an allowed template path."""

    roles: tuple[ChargePairRole, ...]

    @property
    def unsupported_roles(self) -> tuple[ChargePairRole, ...]:
        return tuple(role for role in self.roles if not role.template_supported)

    @property
    def ok(self) -> bool:
        return not self.unsupported_roles


class UnnamedAtomError(ValueError):
    """Raised when a component name leaves graph atoms unnamed."""


def component_named_atom_coverage(mol: Molecule, component_atoms: set[int], parts: AssemblyParts) -> NamingCoverage:
    """Return the atoms represented by the assembled component name."""

    bindings = parts.name_atom_bindings or refresh_name_atom_bindings(parts)
    named_atoms: set[int] = set()
    for binding in bindings:
        named_atoms.update(binding.atom_ids)

    explicit_component_atoms = {idx for idx in component_atoms if mol.atoms[idx].symbol != "H"}
    return NamingCoverage(
        named_atoms=named_atoms & explicit_component_atoms,
        unnamed_atoms=explicit_component_atoms - named_atoms,
    )


def assert_component_fully_named(mol: Molecule, component_atoms: set[int], parts: AssemblyParts, name: str) -> None:
    """Raise if a generated component name did not consume all graph atoms."""

    coverage = component_named_atom_coverage(mol, component_atoms, parts)
    if coverage.unnamed_atoms:
        details = ", ".join(f"{idx}:{mol.atoms[idx].symbol}" for idx in sorted(coverage.unnamed_atoms))
        raise UnnamedAtomError(f"Generated name {name!r} left unnamed atoms: {details}")


def audit_charge_pair_templates(mol: Molecule, component_atoms: set[int]) -> ChargePairTemplateAudit:
    """Return graph-classified charge-pair template support for a component."""

    return ChargePairTemplateAudit(tuple(charge_pair_roles(mol, component_atoms)))
