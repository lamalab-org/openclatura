"""Provenance labels for parent locant maps."""

from enum import StrEnum


class LocantMapSource(StrEnum):
    """Where the parent atom-to-locant map came from."""

    GENERATED = "generated"
    SUPPLIED = "supplied"
    PROOF = "proof"
