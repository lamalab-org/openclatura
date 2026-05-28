"""Data-shaped heteroatom substituent prefix rules."""

from dataclasses import dataclass

from .formatting import is_complex_prefix, strip_outer_parentheses
from .naming_data import load_json_table


@dataclass(frozen=True)
class HeteroatomSubstituentSpec:
    """Prefix names for a heteroatom substituent family."""

    unsubstituted_prefixes: dict[int, str]
    ligand_prefixes: dict[int, str]
    ligand_join_mode: str = "concat"


def heteroatom_substituent_specs() -> dict[str, HeteroatomSubstituentSpec]:
    """Return configured heteroatom substituent specs keyed by element symbol."""

    data = load_json_table("heteroatom_substituents.json")["heteroatom_substituents"]
    return {
        symbol: HeteroatomSubstituentSpec(
            unsubstituted_prefixes={int(count): prefix for count, prefix in spec["unsubstituted_prefixes"].items()},
            ligand_prefixes={int(count): prefix for count, prefix in spec["ligand_prefixes"].items()},
            ligand_join_mode=spec.get("ligand_join_mode", "concat"),
        )
        for symbol, spec in data.items()
    }


def spec_for_symbol(symbol: str) -> HeteroatomSubstituentSpec | None:
    """Return the configured prefix spec for an element symbol."""

    return heteroatom_substituent_specs().get(symbol)


def unsubstituted_prefix(symbol: str, oxo_count: int = 0) -> str | None:
    """Return an unsubstituted heteroatom prefix from data."""

    spec = spec_for_symbol(symbol)
    return None if spec is None else spec.unsubstituted_prefixes.get(oxo_count)


def ligand_prefix(symbol: str, ligand_name: str, oxo_count: int = 0) -> str | None:
    """Return a ligand-bearing heteroatom prefix from data."""

    spec = spec_for_symbol(symbol)
    if spec is None:
        return None
    base = spec.ligand_prefixes.get(oxo_count)
    if base is None:
        return None
    ligand = _apply_join_mode(strip_outer_parentheses(ligand_name), spec.ligand_join_mode)
    if ligand.startswith("("):
        return f"({ligand}{base})"
    return f"({ligand}{base})" if is_complex_prefix(ligand) else f"{ligand}{base}"


def _apply_join_mode(name: str, join_mode: str) -> str:
    """Apply data-configured ligand joining."""

    if join_mode == "drop_yl" and name.endswith("yl"):
        return name[:-2]
    return f"({name})" if is_complex_prefix(name) else name
