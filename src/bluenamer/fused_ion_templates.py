"""Audit registry for charged retained fused parent spellings.

The registry records graph-operation templates for future charged fused
renderers.  It intentionally does not render production names: a parser-visible
``ium`` or ``ide`` suffix is only vocabulary until the row has a graph delta
certificate and OPSIN round-trip proof.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from .assembly_parts import AssemblyParts, NameAtomBinding
from .naming_data import load_json_table
from .grammar_snapshot_data import local_grammar_snapshot, retained_fused_token_status


@dataclass(frozen=True)
class FusedIonTemplate:
    template_id: str
    base_parent_class: str
    base_parent_name: str
    base_parent_aliases: tuple[str, ...]
    allowed_charge_operation: str
    allowed_locants: tuple[str, ...]
    suffix_or_prefix_form: str
    requires_indicated_h: bool
    requires_lambda: bool
    opsin_grammar_name: str
    opsin_compatible_spelling: str
    roundtrip_expected_graph_delta: str
    examples: tuple[str, ...]
    production_ready: bool

    @property
    def base_parent_token_status(self) -> str | None:
        return retained_fused_token_status(self.base_parent_name)


@dataclass(frozen=True)
class FusedIonTemplateRegistry:
    templates: tuple[FusedIonTemplate, ...]

    @property
    def by_id(self) -> dict[str, FusedIonTemplate]:
        return {template.template_id: template for template in self.templates}

    def for_parent(self, parent_name: str) -> tuple[FusedIonTemplate, ...]:
        return tuple(
            template
            for template in self.templates
            if template.base_parent_name == parent_name or parent_name in template.base_parent_aliases
        )

    def production_templates(self) -> tuple[FusedIonTemplate, ...]:
        return tuple(template for template in self.templates if template.production_ready)

    def for_operation(
        self,
        parent_name: str,
        operation: str,
        locants: tuple[str, ...],
    ) -> FusedIonTemplate | None:
        for template in self.for_parent(parent_name):
            if (
                template.allowed_charge_operation == operation
                and template.allowed_locants == locants
            ):
                return template
        return None


@dataclass(frozen=True)
class FusedIonOperationCandidate:
    """Graph-derived charged fused-parent operation before text rendering."""

    operation: str
    parent_name: str
    locants: tuple[str, ...]
    represented_atom_ids: frozenset[int]
    represented_bond_ids: frozenset[int]
    template: FusedIonTemplate | None
    consumed_substituent_names: tuple[str, ...] = ()
    audit_errors: tuple[str, ...] = ()

    @property
    def production_ready(self) -> bool:
        return self.template is not None and self.template.production_ready and not self.audit_errors

    @property
    def rendered_name(self) -> str | None:
        if not self.production_ready or self.template is None:
            return None
        return self.template.opsin_grammar_name


@lru_cache(maxsize=1)
def fused_ion_template_registry() -> FusedIonTemplateRegistry:
    """Return audit-only charged fused template rows."""

    raw = load_json_table("fused_ion_templates.json")
    templates = tuple(_template_from_row(row) for row in raw.get("templates", ()))
    _validate_charge_suffix_vocabulary(templates)
    return FusedIonTemplateRegistry(templates=templates)


def _template_from_row(row: dict[str, Any]) -> FusedIonTemplate:
    return FusedIonTemplate(
        template_id=str(row["template_id"]),
        base_parent_class=str(row["base_parent_class"]),
        base_parent_name=str(row["base_parent_name"]),
        base_parent_aliases=tuple(str(value) for value in row.get("base_parent_aliases", ())),
        allowed_charge_operation=str(row["allowed_charge_operation"]),
        allowed_locants=tuple(str(value) for value in row.get("allowed_locants", ())),
        suffix_or_prefix_form=str(row["suffix_or_prefix_form"]),
        requires_indicated_h=bool(row.get("requires_indicated_h", False)),
        requires_lambda=bool(row.get("requires_lambda", False)),
        opsin_grammar_name=str(row["opsin_grammar_name"]),
        opsin_compatible_spelling=str(row["opsin_compatible_spelling"]),
        roundtrip_expected_graph_delta=str(row["roundtrip_expected_graph_delta"]),
        examples=tuple(str(value) for value in row.get("examples", ())),
        production_ready=bool(row.get("production_ready", False)),
    )


def _validate_charge_suffix_vocabulary(templates: tuple[FusedIonTemplate, ...]) -> None:
    allowed = set(local_grammar_snapshot()["charge_suffixes"]["canonical"]) | {"oxide"}
    invalid = sorted(
        {
            template.suffix_or_prefix_form
            for template in templates
            if template.suffix_or_prefix_form not in allowed
        }
    )
    if invalid:
        raise ValueError(f"fused ion template uses unsupported charge suffixes: {invalid}")


def select_fused_ion_operation(parts: AssemblyParts) -> FusedIonOperationCandidate | None:
    """Return the first production-ready retained fused ion operation."""

    for candidate in fused_ion_operation_candidates(parts):
        if candidate.production_ready:
            return candidate
    return None


def consume_fused_ion_operation(parts: AssemblyParts, candidate: FusedIonOperationCandidate) -> None:
    """Consume name fragments represented by a fused ion operation.

    The operation replaces generic ``parent_charge`` / ``oxido`` terms with a
    graph-bound fused-ion binding.  Unrelated substituents stay on the parts
    object so derivative names can still be assembled normally.
    """

    if not candidate.production_ready:
        return
    if candidate.consumed_substituent_names:
        parts.substituents = [
            item
            for item in parts.substituents
            if not (
                item.name in candidate.consumed_substituent_names
                and tuple(str(locant) for locant in item.locants) == candidate.locants
                and set(item.atom_ids).issubset(set(candidate.represented_atom_ids))
            )
        ]
    parts.parent_charges = [
        charge for charge in parts.parent_charges if str(charge.locant) not in set(candidate.locants)
    ]
    _replace_fused_ion_bindings(parts, candidate)


def render_fused_ion_template_name(parts: AssemblyParts) -> str | None:
    """Render graph-certified retained fused ion names.

    This is a production renderer only for generic graph operations whose
    represented atoms are already present in ``AssemblyParts``.  It does not
    inspect the assembled string.  New ion classes should add a graph operation
    renderer here, then opt into template rows through the registry.
    """

    candidate = select_fused_ion_operation(parts)
    if candidate is not None:
        consume_fused_ion_operation(parts, candidate)
        return candidate.rendered_name
    return None


def fused_ion_operation_candidates(parts: AssemblyParts) -> tuple[FusedIonOperationCandidate, ...]:
    """Return graph-derived retained fused ion operations for assembly parts."""

    candidates = []
    n_oxide_candidate = _retained_fused_n_oxide_operation(parts)
    if n_oxide_candidate is not None:
        candidates.append(n_oxide_candidate)
    charge_candidate = _retained_fused_ring_n_charge_operation(parts)
    if charge_candidate is not None:
        candidates.append(charge_candidate)
    return tuple(candidates)


def _retained_fused_ring_n_charge_operation(parts: AssemblyParts) -> FusedIonOperationCandidate | None:
    if parts.is_substituent or not parts.retained_name:
        return None
    if parts.principal_group or parts.a_prefixes or parts.front_modifiers:
        return None
    if parts.unsaturations or parts.hydro_operations:
        return None
    if len(parts.parent_charges) != 1:
        return None

    charge = parts.parent_charges[0]
    locant = str(charge.locant)
    if charge.symbol != "N" or parts.parent_atom_symbols_by_locant.get(locant) != "N":
        return None
    if parts.parent_atom_charges_by_locant.get(locant) != charge.charge:
        return None

    if charge.charge > 0:
        operation = "ring_n_ium"
    elif charge.charge < 0:
        operation = "ring_n_deprotonation"
    else:
        return None
    represented_atoms = frozenset({charge.atom_id} if charge.atom_id is not None else set())
    template = fused_ion_template_registry().for_operation(parts.retained_name, operation, (locant,))
    return FusedIonOperationCandidate(
        operation=operation,
        parent_name=parts.retained_name,
        locants=(locant,),
        represented_atom_ids=represented_atoms,
        represented_bond_ids=frozenset(),
        template=template,
    )


def _retained_fused_n_oxide_operation(
    parts: AssemblyParts,
) -> FusedIonOperationCandidate | None:
    if parts.is_substituent or not parts.retained_name:
        return None
    if parts.principal_group or parts.a_prefixes or parts.front_modifiers:
        return None
    if parts.unsaturations or parts.hydro_operations or parts.indicated_hydrogens:
        return None

    oxido_items = [item for item in parts.substituents if item.name == "oxido" and len(item.locants) == 1]
    if len(oxido_items) != 1:
        return None
    oxido_item = oxido_items[0]
    locant = str(oxido_item.locants[0])

    charged_n_locs = [
        charge.locant
        for charge in parts.parent_charges
        if charge.symbol == "N" and charge.charge > 0 and str(charge.locant) == locant
    ]
    if len(charged_n_locs) != 1 or len(parts.parent_charges) != 1:
        return None

    if parts.parent_atom_symbols_by_locant.get(locant) != "N":
        return None
    if parts.parent_atom_charges_by_locant.get(locant) != 1:
        return None

    template = fused_ion_template_registry().for_operation(parts.retained_name, "ring_n_oxide", (locant,))
    charge_atoms = {
        charge.atom_id
        for charge in parts.parent_charges
        if charge.atom_id is not None and str(charge.locant) == locant
    }
    represented_atoms = frozenset(set(oxido_item.atom_ids) | {atom for atom in charge_atoms if atom is not None})
    return FusedIonOperationCandidate(
        operation="ring_n_oxide",
        parent_name=parts.retained_name,
        locants=(locant,),
        represented_atom_ids=represented_atoms,
        represented_bond_ids=frozenset(oxido_item.bond_ids),
        consumed_substituent_names=("oxido",),
        template=template,
    )


def _replace_fused_ion_bindings(parts: AssemblyParts, candidate: FusedIonOperationCandidate) -> None:
    if candidate.template is None:
        return
    template = candidate.template
    parts.name_atom_bindings = [
        binding
        for binding in parts.name_atom_bindings
        if not (
            (binding.role == "substituent" and binding.term == "oxido" and binding.locants == template.allowed_locants)
            or (binding.role == "parent_charge" and binding.locants == template.allowed_locants)
        )
    ]
    if candidate.operation == "ring_n_oxide":
        role = "fused_n_oxide"
    elif candidate.operation == "ring_n_ium":
        role = "fused_ring_n_ium"
    elif candidate.operation == "ring_n_deprotonation":
        role = "fused_ring_n_ide"
    else:
        role = "fused_ion_operation"
    parts.name_atom_bindings.append(
        NameAtomBinding(
            stage="charge",
            role=role,
            term=template.opsin_grammar_name,
            atom_ids=set(candidate.represented_atom_ids),
            bond_ids=set(candidate.represented_bond_ids),
            charge_atom_ids=set(candidate.represented_atom_ids),
            locants=template.allowed_locants,
        )
    )
