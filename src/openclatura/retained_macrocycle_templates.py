"""Graph-backed retained tetrapyrrole macrocycle parents.

The checked-in templates are locant graphs generated offline from OPSIN CML.
Production recognition is structure-only: cheap graph invariants select a
small candidate bucket, followed by exact labelled-graph isomorphism.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

from .assembly_parts import RetainedParentMetadata
from .molecule import Molecule
from .retained_graph_templates import (
    RetainedGraphTemplate,
    RetainedGraphTemplateMatch,
    match_retained_graph_templates,
    retained_graph_templates,
)


@dataclass(frozen=True)
class RetainedMacrocycleMatch:
    """A retained macrocycle parent and its conventional locant map."""

    template: RetainedGraphTemplate
    atom_to_locant: dict[int, str]
    locant_to_atom: dict[str, int]
    matched_atoms: frozenset[int]

    @property
    def name(self) -> str:
        return self.template.output_name


def retained_macrocycle_templates() -> tuple[RetainedGraphTemplate, ...]:
    """Return the macrocycle view of the shared retained graph registry."""

    return retained_graph_templates(families=frozenset({"macrocycle"}))


def match_retained_macrocycle(mol: Molecule, atom_ids: set[int]) -> RetainedMacrocycleMatch | None:
    """Match an exact retained macrocycle component without text keys."""

    matches = match_retained_macrocycles(mol, atom_ids)
    return matches[0] if matches else None


def match_retained_macrocycles(mol: Molecule, atom_ids: set[int]) -> list[RetainedMacrocycleMatch]:
    """Return all conventional locant maps for the best macrocycle parent."""

    atom_set = set(atom_ids)
    has_external_attachment = any(
        neighbor not in atom_set for atom_idx in atom_set for neighbor in mol.get_neighbors(atom_idx)
    )
    matches = match_retained_graph_templates(
        mol,
        atom_set,
        allow_nonaromatic=True,
        families=frozenset({"macrocycle"}),
    )
    return [
        _macrocycle_match(match)
        for match in matches
        if not has_external_attachment or match.template.derivative_production_enabled
    ]


def _macrocycle_match(match: RetainedGraphTemplateMatch) -> RetainedMacrocycleMatch:
    return RetainedMacrocycleMatch(
        template=match.template,
        atom_to_locant=dict(match.atom_to_locant),
        locant_to_atom=dict(match.locant_to_atom),
        matched_atoms=match.matched_atoms,
    )


@cache
def retained_macrocycle_parent_metadata(parent_name: str) -> RetainedParentMetadata | None:
    """Return assembly metadata for one exact retained macrocycle spelling."""

    template = next((item for item in retained_macrocycle_templates() if item.name == parent_name), None)
    if template is None:
        return None
    return RetainedParentMetadata(
        default_indicated_h=template.default_indicated_h,
        fusion_locants=(),
        derivative_stem=template.derivative_stem,
        indicated_hydrogen_count=template.indicated_hydrogen_count,
        mancude_double_bonds=template.mancude_double_bonds or 0,
        inherent_saturated_locants=tuple(atom.locant for atom in template.atoms if atom.saturated),
    )
