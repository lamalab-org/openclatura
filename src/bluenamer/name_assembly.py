"""Typed name assembly objects and metadata-preserving rewrites."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from .assembly_parts import AssemblyParts, NameAtomBinding
from .molecule import Molecule
from .name_bindings import postprocess_name_atom_bindings


@dataclass(frozen=True)
class GraphRole:
    """Detected structural role with no naming decision attached."""

    key: str
    atom_ids: frozenset[int] = frozenset()
    bond_ids: frozenset[int] = frozenset()
    charges_by_atom: dict[int, int] = field(default_factory=dict)
    locants_by_atom: dict[int, str] = field(default_factory=dict)
    evidence: tuple[str, ...] = ()


@dataclass(frozen=True)
class RendererTemplate:
    """A supported rendering path for a graph role in a grammar context."""

    key: str
    role_key: str
    context: str
    grammar: str
    supported: bool = False
    verified: bool = False
    preserves_formal_charges: bool = False
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class NameFragment:
    """One emitted word or operation and the graph metadata it represents."""

    text: str
    bindings: tuple[NameAtomBinding, ...] = ()
    role: GraphRole | None = None
    template: RendererTemplate | None = None

    @classmethod
    def from_binding(cls, binding: NameAtomBinding) -> "NameFragment":
        return cls(text=binding.term, bindings=(binding,))


@dataclass(frozen=True)
class NameRewriteOperation:
    """A text rewrite that also transforms fragment/binding metadata."""

    name: str
    before: str
    after: str
    binding_count: int
    changed_binding_count: int

    @classmethod
    def apply(
        cls,
        name: str,
        bindings: tuple[NameAtomBinding, ...],
        *,
        operation_name: str,
        rewrite: Callable[[str], str],
    ) -> tuple[str, tuple[NameAtomBinding, ...], "NameRewriteOperation"]:
        """Apply a rewrite to final text and every bound term."""

        rewritten_name = rewrite(name)
        rewritten_bindings = tuple(postprocess_name_atom_bindings(list(bindings), rewrite))
        changed_bindings = sum(
            1
            for before_binding, after_binding in zip(bindings, rewritten_bindings, strict=False)
            if before_binding.term != after_binding.term
        )
        return (
            rewritten_name,
            rewritten_bindings,
            cls(
                name=operation_name,
                before=name,
                after=rewritten_name,
                binding_count=len(bindings),
                changed_binding_count=changed_bindings,
            ),
        )


@dataclass(frozen=True)
class NameAssemblyResult:
    """Final name text plus the graph metadata that survived assembly."""

    raw_text: str
    text: str
    fragments: tuple[NameFragment, ...]
    bindings: tuple[NameAtomBinding, ...]
    rewrite_history: tuple[NameRewriteOperation, ...] = ()

    @property
    def atom_ids(self) -> set[int]:
        atoms: set[int] = set()
        for binding in self.bindings:
            atoms.update(binding.atom_ids)
        return atoms

    @property
    def bond_ids(self) -> set[int]:
        bonds: set[int] = set()
        for binding in self.bindings:
            bonds.update(binding.bond_ids)
        return bonds

    @property
    def charged_atom_ids(self) -> set[int]:
        atoms: set[int] = set()
        for binding in self.bindings:
            if binding.stage == "charge" or "charge" in binding.role or binding.role.endswith(("_ium", "_ide")):
                atoms.update(binding.atom_ids)
        return atoms

    @classmethod
    def from_raw_name(
        cls,
        raw_text: str,
        bindings: list[NameAtomBinding] | tuple[NameAtomBinding, ...],
        *,
        postprocess: Callable[[str], str],
    ) -> "NameAssemblyResult":
        """Build a final assembly result while keeping binding metadata in sync."""

        binding_tuple = tuple(bindings)
        final_text, final_bindings, rewrite = NameRewriteOperation.apply(
            raw_text,
            binding_tuple,
            operation_name="post_process_name",
            rewrite=postprocess,
        )
        return cls(
            raw_text=raw_text,
            text=final_text,
            fragments=tuple(NameFragment.from_binding(binding) for binding in final_bindings),
            bindings=final_bindings,
            rewrite_history=(rewrite,),
        )

    @classmethod
    def from_final_name(
        cls,
        text: str,
        bindings: list[NameAtomBinding] | tuple[NameAtomBinding, ...],
        *,
        rewrite_history: tuple[NameRewriteOperation, ...] = (),
    ) -> "NameAssemblyResult":
        """Build a result for callers that already finalized text and bindings."""

        binding_tuple = tuple(bindings)
        return cls(
            raw_text=text,
            text=text,
            fragments=tuple(NameFragment.from_binding(binding) for binding in binding_tuple),
            bindings=binding_tuple,
            rewrite_history=rewrite_history,
        )


@dataclass(frozen=True)
class FinalAssemblyAudit:
    """Final metadata audit for an assembled component name."""

    errors: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.errors


class FinalAssemblyAuditError(ValueError):
    """Raised when final name text and final metadata disagree."""


def audit_final_name_assembly(
    mol: Molecule,
    component_atoms: set[int],
    parts: AssemblyParts,
    result: NameAssemblyResult,
) -> FinalAssemblyAudit:
    """Check the final emitted name after all post-processing has run."""

    errors: list[str] = []
    explicit_component_atoms = {idx for idx in component_atoms if mol.atoms[idx].symbol != "H"}
    unnamed_atoms = explicit_component_atoms - result.atom_ids
    if unnamed_atoms:
        errors.append(_format_atom_error("unnamed atoms", mol, unnamed_atoms))

    expected_charged_atoms = {
        idx
        for idx in explicit_component_atoms
        if idx in mol.atoms and mol.atoms[idx].charge != 0
    }
    missing_charged_atoms = expected_charged_atoms - result.atom_ids
    if missing_charged_atoms:
        errors.append(_format_atom_error("charged atoms not represented", mol, missing_charged_atoms))

    consumed_bonds = _consumed_bond_ids(parts)
    missing_bonds = consumed_bonds - result.bond_ids
    if missing_bonds:
        errors.append(f"consumed bonds not represented: {sorted(missing_bonds)}")

    invalid_bindings = [
        idx
        for idx, binding in enumerate(result.bindings)
        if not binding.term.strip() or (not binding.atom_ids and not binding.bond_ids)
    ]
    if invalid_bindings:
        errors.append(f"invalid final name bindings: {invalid_bindings}")

    return FinalAssemblyAudit(tuple(errors))


def assert_final_name_assembly(
    mol: Molecule,
    component_atoms: set[int],
    parts: AssemblyParts,
    result: NameAssemblyResult,
) -> None:
    """Raise if the final name has lost required graph metadata."""

    audit = audit_final_name_assembly(mol, component_atoms, parts, result)
    if not audit.ok:
        raise FinalAssemblyAuditError(f"Generated name {result.text!r} failed final metadata audit: {'; '.join(audit.errors)}")


def _consumed_bond_ids(parts: AssemblyParts) -> set[int]:
    bonds: set[int] = set(parts.parent_bond_ids)
    for item in parts.principal_suffix_modifiers:
        bonds.update(item.bond_ids)
    for item in parts.a_prefixes:
        bonds.update(item.bond_ids)
    for item in parts.substituents:
        bonds.update(item.bond_ids)
    for item in parts.unsaturations:
        bonds.update(item.bond_ids)
    if parts.principal_group is not None:
        bonds.update(parts.principal_group.bond_ids)
    for binding in parts.name_atom_bindings:
        bonds.update(binding.bond_ids)
    return bonds


def _format_atom_error(label: str, mol: Molecule, atom_ids: set[int]) -> str:
    details = ", ".join(f"{idx}:{mol.atoms[idx].symbol}" for idx in sorted(atom_ids))
    return f"{label}: {details}"
