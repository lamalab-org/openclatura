"""Shared stereochemical descriptor tokens used by renderers and metadata."""

ABSOLUTE_STEREO_DESCRIPTORS = frozenset({"R", "S"})
BOND_STEREO_DESCRIPTORS = frozenset({"E", "Z"})
RELATIVE_STEREO_DESCRIPTORS = frozenset({"cis", "trans"})

SEARCHABLE_STEREO_TOKENS = frozenset(
    descriptor.lower()
    for descriptor in (
        *ABSOLUTE_STEREO_DESCRIPTORS,
        *BOND_STEREO_DESCRIPTORS,
        *RELATIVE_STEREO_DESCRIPTORS,
    )
)


def is_searchable_stereo_token(text: str) -> bool:
    """Return whether a renderer-emitted stereo token may be matched directly."""

    return text.lower() in SEARCHABLE_STEREO_TOKENS
