"""Graph-backed retained tetrapyrrole macrocycle parents.

The checked-in templates are locant graphs generated offline from OPSIN CML.
Production recognition is structure-only: cheap graph invariants select a
small candidate bucket, followed by exact labelled-graph isomorphism.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from .molecule import Molecule
from .naming_data import load_json_table
from .retained_fused_templates import (
    RetainedFusedGraphTemplate,
    RetainedFusedTemplateMatch,
    match_retained_fused_template,
    molecule_graph_topology_key,
    retained_fused_template_from_data,
    retained_graph_template_topology_key,
)


@dataclass(frozen=True)
class RetainedMacrocycleMatch:
    """A retained macrocycle parent and its conventional locant map."""

    template: RetainedFusedGraphTemplate
    atom_to_locant: dict[int, str]
    locant_to_atom: dict[str, int]
    matched_atoms: frozenset[int]

    @property
    def name(self) -> str:
        return self.template.output_name


@lru_cache(maxsize=1)
def retained_macrocycle_templates() -> tuple[RetainedFusedGraphTemplate, ...]:
    """Return validated retained macrocycle templates in policy order."""

    rows = load_json_table("retained_macrocycle_templates.json").get("parents", ())
    templates = tuple(retained_fused_template_from_data(dict(row)) for row in rows)
    return tuple(sorted(templates, key=lambda template: (template.priority, template.name)))


@lru_cache(maxsize=1)
def _macrocycles_by_topology() -> dict[tuple, tuple[RetainedFusedGraphTemplate, ...]]:
    """Index the small retained family before exact graph matching."""

    index: dict[tuple, list[RetainedFusedGraphTemplate]] = {}
    for template in retained_macrocycle_templates():
        index.setdefault(retained_graph_template_topology_key(template), []).append(template)
    return {key: tuple(values) for key, values in index.items()}


@lru_cache(maxsize=1)
def _macrocycle_sizes() -> frozenset[int]:
    """Heavy-atom counts that can possibly enter macrocycle matching."""

    return frozenset(len(template.atoms) for template in retained_macrocycle_templates())


def match_retained_macrocycle(mol: Molecule, atom_ids: set[int]) -> RetainedMacrocycleMatch | None:
    """Match an exact retained macrocycle component without text keys."""

    atom_set = set(atom_ids)
    if len(atom_set) not in _macrocycle_sizes():
        return None
    candidates = _macrocycles_by_topology().get(molecule_graph_topology_key(mol, atom_set), ())
    for template in candidates:
        if not _mancude_bond_count_matches(mol, atom_set, template):
            continue
        match = match_retained_fused_template(mol, atom_set, template, allow_nonaromatic=True)
        if match is not None:
            return _macrocycle_match(match)
    return None


def _mancude_bond_count_matches(
    mol: Molecule, atom_ids: set[int], template: RetainedFusedGraphTemplate
) -> bool:
    """Reject hydro derivatives while allowing equivalent Kekule placement."""

    expected = template.mancude_double_bonds
    if expected is None:
        return True
    observed = sum(
        bond.order == 2 and bond.u in atom_ids and bond.v in atom_ids for bond in mol.bonds.values()
    )
    return observed == expected


def _macrocycle_match(match: RetainedFusedTemplateMatch) -> RetainedMacrocycleMatch:
    return RetainedMacrocycleMatch(
        template=match.template,
        atom_to_locant=dict(match.atom_to_locant),
        locant_to_atom=dict(match.locant_to_atom),
        matched_atoms=match.matched_atoms,
    )
