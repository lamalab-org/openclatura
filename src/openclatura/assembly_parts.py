"""Structured inputs for name assembly."""

from dataclasses import dataclass, field

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
    front_modifiers: list[str] = field(default_factory=list)
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
    name_atom_bindings: list[NameAtomBinding] = field(default_factory=list)
    name_token_spans: list[dict] = field(default_factory=list)
    name_rewrite_history: list[dict] = field(default_factory=list)
    stereo_audit_issues: list[str] = field(default_factory=list)
    reconstruction_audit_status: str | None = None
    reconstruction_audit_issues: list[str] = field(default_factory=list)
