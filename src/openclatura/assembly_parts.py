"""Structured inputs for name assembly."""

from dataclasses import dataclass, field

from .locant_sources import LocantMapSource
from .name_operations import HydroOperation
from .spiro_assembly import SpiroAssembly


@dataclass(frozen=True)
class NameTokenBinding:
    """Renderer-emitted token metadata before final string positioning."""

    text: str
    token_kind: str = "structural"
    ownership: str = "exact"
    confidence: str = "exact"
    source: str = "renderer"
    grammar_role: str = ""
    binding_key: str = ""
    atom_ids: set[int] = field(default_factory=set)
    bond_ids: set[int] = field(default_factory=set)
    charge_atom_ids: set[int] = field(default_factory=set)
    locants: tuple[str, ...] = ()
    render_order: int | None = None
    match_priority: int = 0
    left_context: str = ""
    right_context: str = ""


@dataclass(frozen=True)
class RenderedSubstituentName:
    """Rendered text plus construction-time parenthesis-boundary metadata."""

    text: str
    outer_parentheses_optional: bool = False

    def __str__(self) -> str:
        return self.text


RenderedSubstituentText = str | RenderedSubstituentName


def split_rendered_substituent_name(name: RenderedSubstituentText) -> tuple[str, bool]:
    """Return plain text and explicit boundary metadata for a rendered name."""

    if isinstance(name, RenderedSubstituentName):
        return name.text, name.outer_parentheses_optional
    return name, False


def rendered_substituent_text(name: RenderedSubstituentText) -> str:
    """Return plain text when boundary metadata is no longer needed."""

    return name.text if isinstance(name, RenderedSubstituentName) else name


@dataclass
class SubstituentItem:
    name: str
    locants: list[str]
    atom_ids: set[int] = field(default_factory=set)
    bond_ids: set[int] = field(default_factory=set)
    charge_atom_ids: set[int] = field(default_factory=set)
    emitted_tokens: tuple[NameTokenBinding, ...] = ()
    trace_segments: list[dict] = field(default_factory=list)
    nested_decisions: list[dict] = field(default_factory=list)
    substituent_tree: dict | None = None
    spiro: SpiroAssembly | None = None
    outer_parentheses_optional: bool = False


@dataclass
class UnsaturationItem:
    bond_key: str
    locants: list[str]
    atom_ids: set[int] = field(default_factory=set)
    bond_ids: set[int] = field(default_factory=set)


@dataclass
class PrincipalGroupItem:
    key: str
    locants: list[str]
    atom_ids: set[int] = field(default_factory=set)
    bond_ids: set[int] = field(default_factory=set)
    charge_atom_ids: set[int] = field(default_factory=set)


@dataclass(frozen=True)
class ParentChargeItem:
    locant: str
    symbol: str
    charge: int
    atom_id: int | None = None


@dataclass(frozen=True)
class NameAtomBinding:
    """Mapping from one emitted name term/operation to graph atoms and bonds."""

    stage: str
    role: str
    term: str
    atom_ids: set[int] = field(default_factory=set)
    bond_ids: set[int] = field(default_factory=set)
    charge_atom_ids: set[int] = field(default_factory=set)
    locants: tuple[str, ...] = ()
    emitted_tokens: tuple[NameTokenBinding, ...] = ()


@dataclass(frozen=True)
class RetainedParentMetadata:
    """Naming metadata carried from a matched retained-parent template."""

    default_indicated_h: tuple[str, ...] = ()
    fusion_locants: tuple[str, ...] = ()
    derivative_stem: str | None = None
    # How many indicated hydrogens the mancude parent hydride itself supports.
    # Saturated positions beyond this many are *added* hydrogen and are cited as
    # a hydro prefix: xanthine is 3,7-dihydro-1H-purine-2,6-dione, never
    # 1H,3H,7H-purine-2,6-dione.
    indicated_hydrogen_count: int = 0
    # Monocycle family states may declare hydrogenation directly from their
    # graph template. These locants are converted to graph-bound
    # ``HydroOperation`` objects after the final retained locant map is chosen.
    additive_hydrogen_locants: tuple[str, ...] = ()
    indicated_hydrogen_locants: tuple[str, ...] = ()
    bare_parent_only: bool = False
    derivative_name: str | None = None


@dataclass
class AssemblyParts:
    parent_length: int
    is_ring: bool = False
    is_bicycle: bool = False
    is_spiro: bool = False
    is_polycycle: bool = False
    bicycle_xyz: tuple[int, int, int] = (0, 0, 0)
    spiro_xy: tuple[int, int] = (0, 0)
    tricyclo_xyzw: tuple[int, int, int, int] = (0, 0, 0, 0)
    polycycle_descriptor: str | None = None
    is_substituent: bool = False
    is_double_attach: bool = False
    is_triple_attach: bool = False
    attachment_locant: int | str = 1
    retained_name: str | None = None
    # Set when ``retained_name`` is one of the retained names that spells the
    # principal characteristic group as well as the ring -- ``phenol`` is benzene
    # *and* its ``-ol``.  The suffix must then not be rendered a second time, and
    # the parent binding owns the group's atoms.
    retained_absorbs_principal_group: bool = False
    # Set when the whole substituent word is a retained name that spells its own
    # branch as well as its skeleton -- ``benzyl`` is the methylene *and* the
    # phenyl on it.  The absorbed branches are moved out of ``substituents`` so
    # they are not also cited as prefixes, and are kept here so the parent
    # binding can claim the atoms the retained word now names.
    retained_substituent_name: str | None = None
    retained_absorbed_substituents: list[SubstituentItem] = field(default_factory=list)
    retained_parent_metadata: RetainedParentMetadata | None = None
    front_modifiers: list[str] = field(default_factory=list)
    front_modifier_locants: list[str | None] = field(default_factory=list)
    front_modifier_atom_ids: set[int] = field(default_factory=set)
    front_modifier_charge_atom_ids: set[int] = field(default_factory=set)
    principal_suffix_modifiers: list[SubstituentItem] = field(default_factory=list)
    a_prefixes: list[SubstituentItem] = field(default_factory=list)
    principal_group: PrincipalGroupItem | None = None
    unsaturations: list[UnsaturationItem] = field(default_factory=list)
    substituents: list[SubstituentItem] = field(default_factory=list)
    stereo_features: list[tuple[str, str]] = field(default_factory=list)
    relative_stereo_prefixes: list[str] = field(default_factory=list)
    indicated_hydrogens: list[str] = field(default_factory=list)
    hydro_operations: list[HydroOperation] = field(default_factory=list)
    parent_charges: list[ParentChargeItem] = field(default_factory=list)
    parent_atom_ids: set[int] = field(default_factory=set)
    parent_bond_ids: set[int] = field(default_factory=set)
    parent_atom_ids_by_locant: dict[str, int] = field(default_factory=dict)
    parent_atom_symbols_by_locant: dict[str, str] = field(default_factory=dict)
    parent_atom_charges_by_locant: dict[str, int] = field(default_factory=dict)
    parent_bond_orders_by_locants: dict[tuple[str, str], int] = field(default_factory=dict)
    parent_bond_ids_by_locants: dict[tuple[str, str], int] = field(default_factory=dict)
    locant_map_source: LocantMapSource = LocantMapSource.GENERATED
    name_atom_bindings: list[NameAtomBinding] = field(default_factory=list)
    name_token_spans: list[dict] = field(default_factory=list)
    name_rewrite_history: list[dict] = field(default_factory=list)
    stereo_audit_issues: list[str] = field(default_factory=list)
