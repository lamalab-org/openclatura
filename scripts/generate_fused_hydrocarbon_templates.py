"""Generate locant-keyed PAH templates from OPSIN CML.

This development utility is intentionally outside production naming.  Runtime
matching consumes only the checked-in atom/edge data and never uses names,
SMILES, or SMARTS as structure keys.
"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import py2opsin

from openclatura.retained_fused_templates import _smallest_ring_basis

_CML_NS = {"cml": "http://www.xml-cml.org/schema"}
_PARENTS = (
    ("benz[a]anthracene", 361, "benz[a]anthracen", "benz[a]anthraceno"),
    ("benzo[a]pyrene", 362, "benzo[a]pyren", "benzo[a]pyreno"),
    ("benzo[b]fluoranthene", 363, "benzo[b]fluoranthen", "benzo[b]fluorantheno"),
    ("benzo[k]fluoranthene", 364, "benzo[k]fluoranthen", "benzo[k]fluorantheno"),
    ("indeno[1,2,3-cd]pyrene", 365, "indeno[1,2,3-cd]pyren", "indeno[1,2,3-cd]pyreno"),
    ("benzo[ghi]perylene", 366, "benzo[ghi]perylen", "benzo[ghi]peryleno"),
    ("tetraphenylene", 371, "tetraphenylen", "tetraphenyleno"),
    ("rubicene", 372, "rubicen", "rubiceno"),
    ("trinaphthylene", 373, "trinaphthylen", "trinaphthyleno"),
    ("pyranthrene", 374, "pyranthren", "pyranthreno"),
    ("ovalene", 375, "ovalen", "ovaleno"),
    ("benzo[e]pyrene", 376, "benzo[e]pyren", "benzo[e]pyreno"),
    ("benzo[j]fluoranthene", 377, "benzo[j]fluoranthen", "benzo[j]fluorantheno"),
    ("cyclopenta[cd]pyrene", 378, "cyclopenta[cd]pyren", "cyclopenta[cd]pyreno"),
)


def _parent_row(name: str, priority: int, derivative_stem: str, attached_prefix: str) -> dict:
    root = ET.fromstring(py2opsin.py2opsin(name, output_format="CML"))
    atoms_by_id: dict[str, dict[str, str]] = {}
    locants: list[str] = []
    for atom in root.findall(".//cml:atom", _CML_NS):
        labels = atom.findall("cml:label", _CML_NS)
        locant = next((label.get("value") for label in labels if label.get("value", "")[:1].isdigit()), None)
        if locant is None:
            continue
        atoms_by_id[atom.get("id", "")] = {
            "locant": locant,
            "symbol": atom.get("elementType", "C"),
            "hydrogen_count": atom.get("hydrogenCount", ""),
        }
        locants.append(locant)

    edges: list[tuple[str, str]] = []
    bond_orders: dict[frozenset[str], str] = {}
    hydrogenated: set[str] = set()
    all_atoms = {atom.get("id", ""): atom.get("elementType", "") for atom in root.findall(".//cml:atom", _CML_NS)}
    for bond in root.findall(".//cml:bond", _CML_NS):
        left_id, right_id = bond.get("atomRefs2", "").split()
        left = atoms_by_id.get(left_id)
        right = atoms_by_id.get(right_id)
        if left is not None and right is not None:
            edge = (left["locant"], right["locant"])
            edges.append(edge)
            bond_orders[frozenset(edge)] = bond.get("order", "S")
        elif all_atoms.get(left_id) == "H" and right is not None:
            hydrogenated.add(right["locant"])
        elif all_atoms.get(right_id) == "H" and left is not None:
            hydrogenated.add(left["locant"])

    neighbors = {locant: [] for locant in locants}
    for left, right in edges:
        neighbors[left].append(right)
        neighbors[right].append(left)
    fusion_atoms = [locant for locant in locants if not locant.isdigit()]
    atom_rows = []
    for atom in atoms_by_id.values():
        locant = atom["locant"]
        saturated = len(neighbors[locant]) == 2 and all(
            bond_orders[frozenset((locant, neighbor))] == "S" for neighbor in neighbors[locant]
        )
        row: dict[str, object] = {"locant": locant}
        if atom["symbol"] != "C":
            row["symbol"] = atom["symbol"]
        if locant in fusion_atoms:
            row["fusion"] = True
        if saturated:
            row.update({"aromatic": False, "saturated": True})
        atom_rows.append(row)

    edge_tuple = tuple(edges)
    return {
        "name": name,
        "pin": True,
        "priority": priority,
        "aliases": [],
        "attached_prefix": attached_prefix,
        "derivative_stem": derivative_stem,
        "template": {
            "atoms": atom_rows,
            "bonds": [{"locants": list(edge)} for edge in edges],
            "locants": locants,
            "rings": [list(ring) for ring in _smallest_ring_basis(tuple(locants), edge_tuple)],
            "fusion_atoms": fusion_atoms,
            "peripheral_atoms": [locant for locant in locants if locant in hydrogenated or locant.isdigit()],
            "interior_atoms": [],
            "default_indicated_h": [],
            "mancude_double_bonds": sum(order == "D" for order in bond_orders.values()),
            "enabled": True,
            "derivative_production_enabled": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    payload = {
        "bluebook_rule": "P-25.1.1 retained fused hydrocarbon parents and P-25.3 fusion names.",
        "source": "Generated offline from OPSIN CML locants; runtime matching is graph-only.",
        "parents": [_parent_row(*spec) for spec in _PARENTS],
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
