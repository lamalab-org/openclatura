"""Retained parent metadata used by parent assembly and attachment policies."""

from dataclasses import dataclass

from .nomenclature import RULES


@dataclass(frozen=True)
class AttachmentLocantPolicy:
    print_substituent_locant: bool = False
    use_parent_attachment_equivalence: bool = False
    reason: str = ""


@dataclass(frozen=True)
class RetainedParentSpec:
    name: str
    substituent_stem: str | None = None
    substituent_terminal: str | None = None
    attachment_policy: AttachmentLocantPolicy = AttachmentLocantPolicy()


def retained_parent_spec(name: str | None) -> RetainedParentSpec | None:
    if not name:
        return None
    stem_data = RULES.retained.substituent_stems.get(name)
    stem = stem_data[0] if stem_data else None
    terminal = stem_data[1] if stem_data else None
    normalized_stem = (stem or name).lower().rstrip("e")
    needs_locant = any(
        normalized_stem.endswith(policy_stem)
        for policy_stem in RULES.assembly.ambiguous_connection_substituent_stems
    )
    return RetainedParentSpec(
        name=name,
        substituent_stem=stem,
        substituent_terminal=terminal,
        attachment_policy=AttachmentLocantPolicy(
            print_substituent_locant=needs_locant,
            use_parent_attachment_equivalence=True,
            reason="Retained parent radical attachment is ambiguous without an explicit locant."
            if needs_locant
            else "",
        ),
    )
