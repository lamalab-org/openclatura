"""Typed name assembly objects and metadata-preserving rewrites."""

from __future__ import annotations

import re
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
        rewritten_bindings = tuple(postprocess_name_atom_bindings(list(bindings), rewrite, final_name=rewritten_name))
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
            atoms.update(binding.charge_atom_ids)
            if binding.stage == "charge" or binding.role == "parent_charge":
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

        return cls.from_rewrite_pipeline(
            raw_text,
            bindings,
            rewrites=(("post_process_name", postprocess),),
        )

    @classmethod
    def from_rewrite_pipeline(
        cls,
        raw_text: str,
        bindings: list[NameAtomBinding] | tuple[NameAtomBinding, ...],
        *,
        rewrites: tuple[tuple[str, Callable[[str], str]], ...],
    ) -> "NameAssemblyResult":
        """Build a final result by applying named rewrites to text and bindings."""

        text = raw_text
        binding_tuple = tuple(bindings)
        history: list[NameRewriteOperation] = []
        for operation_name, rewrite in rewrites:
            text, binding_tuple, operation = NameRewriteOperation.apply(
                text,
                binding_tuple,
                operation_name=operation_name,
                rewrite=rewrite,
            )
            history.append(operation)
        return cls(
            raw_text=raw_text,
            text=text,
            fragments=tuple(NameFragment.from_binding(binding) for binding in binding_tuple),
            bindings=binding_tuple,
            rewrite_history=tuple(history),
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
    warnings: tuple[str, ...] = ()

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
    warnings: list[str] = []
    explicit_component_atoms = {idx for idx in component_atoms if mol.atoms[idx].symbol != "H"}
    unnamed_atoms = explicit_component_atoms - result.atom_ids
    if unnamed_atoms:
        errors.append(_format_atom_error("unnamed atoms", mol, unnamed_atoms))

    expected_charged_atoms = {
        idx
        for idx in explicit_component_atoms
        if idx in mol.atoms and mol.atoms[idx].charge != 0
    }
    missing_charged_atoms = expected_charged_atoms - result.charged_atom_ids
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

    missing_terms = _missing_concrete_binding_terms(result)
    if missing_terms:
        errors.append(f"binding terms absent from final name: {missing_terms}")

    return FinalAssemblyAudit(tuple(errors), tuple(warnings))


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


def _missing_concrete_binding_terms(result: NameAssemblyResult) -> list[dict]:
    """Return concrete binding terms that no longer occur in final text.

    Some bindings are intentionally abstract graph certificates, for example
    ``chain parent`` or ``N+1`` parent-charge markers. Those are audited by
    atom/bond/charge coverage instead of literal text presence. Concrete name
    words must still survive post-processing into the final rendered string.
    """

    final_text = _normalise_name_text(result.text)
    missing: list[dict] = []
    for idx, binding in enumerate(result.bindings):
        if not _binding_term_requires_text_presence(binding):
            continue
        term = _normalise_name_text(binding.term)
        if term and not _binding_term_occurs_in_final_name(binding, term, final_text):
            if _binding_is_subsumed_by_present_binding(binding, result.bindings, final_text):
                continue
            missing.append(
                {
                    "index": idx,
                    "stage": binding.stage,
                    "role": binding.role,
                    "term": binding.term,
                }
            )
    return missing


def _binding_is_subsumed_by_present_binding(
    binding: NameAtomBinding,
    bindings: tuple[NameAtomBinding, ...],
    final_text: str,
) -> bool:
    """Return whether another emitted term covers this binding's graph scope."""

    for other in bindings:
        if other is binding or not _binding_term_requires_text_presence(other):
            continue
        other_term = _normalise_name_text(other.term)
        if not other_term or not _binding_term_occurs_in_final_name(other, other_term, final_text):
            continue
        atoms_covered = not binding.atom_ids or binding.atom_ids <= other.atom_ids
        bonds_covered = not binding.bond_ids or binding.bond_ids <= other.bond_ids
        charges_covered = not binding.charge_atom_ids or binding.charge_atom_ids <= other.charge_atom_ids
        if atoms_covered and bonds_covered and charges_covered:
            return True
    return False


def _binding_term_requires_text_presence(binding: NameAtomBinding) -> bool:
    term = binding.term.strip()
    if not term:
        return False
    if binding.stage == "charge" or binding.role == "parent_charge":
        return False
    if binding.stage == "unsaturation":
        return False
    if binding.stage == "suffix" and binding.role == term:
        return False
    if binding.role == "replacement_prefix" and term in _REPLACEMENT_PREFIX_CERTIFICATE_TERMS:
        return False
    if term.endswith(" parent"):
        return False
    if binding.stage == "prefix" and ("(" in term or ")" in term):
        return False
    if "_" in term:
        return False
    if term in _ABSTRACT_BINDING_TERMS:
        return False
    return True


def _normalise_name_text(text: str) -> str:
    return text.lower().replace(" ", "").replace("(", "").replace(")", "")


def _term_occurs_in_final_name(term: str, final_text: str) -> bool:
    if term in final_text:
        return True
    return term.endswith("e") and term[:-1] in final_text


def _binding_term_occurs_in_final_name(binding: NameAtomBinding, term: str, final_text: str) -> bool:
    if _term_occurs_in_final_name(term, final_text):
        return True
    if binding.stage == "parent" and _ionic_parent_stem_occurs(term, final_text):
        return True
    if binding.stage == "modifier" and binding.role == "front_modifier":
        term_parts = [_normalise_name_text(part) for part in binding.term.split()]
        return bool(term_parts) and all(_term_occurs_in_final_name(part, final_text) for part in term_parts)
    if binding.stage == "parent" and term == "benzene":
        return "benz" in final_text
    return False


def _ionic_parent_stem_occurs(term: str, final_text: str) -> bool:
    """Return whether a retained parent stem survived with different ion suffixing."""

    stem = re.sub(r"-\d+(?:,\d+)*-(?:ide|ium)$", "", term)
    if stem == term or len(stem) < 4:
        return False
    return any(variant in final_text for variant in _retained_parent_stem_variants(stem))


def _retained_parent_stem_variants(stem: str) -> set[str]:
    """Return audit spellings for retained parent stems after ionic suffix rewrites."""

    variants = {stem}
    unlocanted = re.sub(r"^\d+(?:,\d+)*-", "", stem)
    variants.add(unlocanted)
    for item in tuple(variants):
        if item.endswith("in"):
            variants.add(f"{item}e")
    return {variant for variant in variants if len(variant) >= 4}


_ABSTRACT_BINDING_TERMS = frozenset(
    {
        "alcohol",
        "aldehyde",
        "amide",
        "amine",
        "carboxylic acid",
        "ester",
        "ether",
        "ketone",
        "nitrile",
        "thiol",
    }
)

_REPLACEMENT_PREFIX_CERTIFICATE_TERMS = frozenset({"aza", "oxa", "thia"})
