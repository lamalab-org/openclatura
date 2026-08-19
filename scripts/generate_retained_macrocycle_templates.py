"""Generate locant-keyed retained macrocycle templates from OPSIN CML.

This development utility is intentionally outside production naming. Runtime
matching consumes only checked-in atom/edge data and never uses names, SMILES,
or SMARTS as structure keys.
"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import py2opsin

_CML_NS = {"cml": "http://www.xml-cml.org/schema"}
_PARENTS = (
    ("porphyrin", 100, ("porphine", "21H,23H-porphine")),
    ("corrin", 110, ()),
)


def _parent_row(name: str, priority: int, aliases: tuple[str, ...]) -> dict:
    root = ET.fromstring(py2opsin.py2opsin(name, output_format="CML"))
    all_atoms = {
        atom.get("id", ""): atom.get("elementType", "") for atom in root.findall(".//cml:atom", _CML_NS)
    }
    atoms_by_id: dict[str, dict[str, object]] = {}
    locants: list[str] = []
    for atom in root.findall(".//cml:atom", _CML_NS):
        locant = next(
            (
                label.get("value")
                for label in atom.findall("cml:label", _CML_NS)
                if label.get("value", "")[:1].isdigit()
            ),
            None,
        )
        if locant is None:
            continue
        atoms_by_id[atom.get("id", "")] = {
            "locant": locant,
            "symbol": atom.get("elementType", "C"),
            "hydrogen_count": int(atom.get("hydrogenCount", "0") or 0),
        }
        locants.append(locant)

    edges: list[tuple[str, str]] = []
    bond_orders: dict[frozenset[str], str] = {}
    hydrogenated: set[str] = set()
    for bond in root.findall(".//cml:bond", _CML_NS):
        left_id, right_id = bond.get("atomRefs2", "").split()
        left = atoms_by_id.get(left_id)
        right = atoms_by_id.get(right_id)
        if left is not None and right is not None:
            edge = (str(left["locant"]), str(right["locant"]))
            edges.append(edge)
            bond_orders[frozenset(edge)] = bond.get("order", "S")
        elif all_atoms.get(left_id) == "H" and right is not None:
            hydrogenated.add(str(right["locant"]))
        elif all_atoms.get(right_id) == "H" and left is not None:
            hydrogenated.add(str(left["locant"]))

    neighbors = {locant: [] for locant in locants}
    for left, right in edges:
        neighbors[left].append(right)
        neighbors[right].append(left)

    atom_rows = []
    for atom in atoms_by_id.values():
        locant = str(atom["locant"])
        saturated = all(bond_orders[frozenset((locant, neighbor))] == "S" for neighbor in neighbors[locant])
        # ``aromatic`` in the shared graph-template kernel means a mancude
        # (not fully saturated) site and deliberately accepts Kekule input.
        row: dict[str, object] = {"locant": locant}
        if atom["symbol"] != "C":
            row["symbol"] = atom["symbol"]
        if saturated:
            row.update({"aromatic": False, "saturated": True})
        if int(atom["hydrogen_count"]) > 0 or locant in hydrogenated:
            row["default_h"] = True
        atom_rows.append(row)

    return {
        "name": name,
        "preferred_name": name,
        "pin": True,
        "priority": priority,
        "aliases": list(aliases),
        "template": {
            "atoms": atom_rows,
            "bonds": [
                {
                    "locants": list(edge),
                    "bond_class": "double" if bond_orders[frozenset(edge)] == "D" else "single",
                }
                for edge in edges
            ],
            "locants": locants,
            # Macrocycles are not fused-parent face systems. Their complete
            # locanted edge graph is the structural proof; ring decomposition
            # remains intentionally outside the fused-ring template model.
            "rings": [],
            "fusion_atoms": [],
            "peripheral_atoms": locants,
            "interior_atoms": [],
            "default_indicated_h": [],
            "mancude_double_bonds": sum(order == "D" for order in bond_orders.values()),
            "numbering_policy": "retained_macrocycle_template",
            "aromatic_equivalence_policy": "neutral_kekule_equivalent",
            "enabled": True,
            "derivative_production_enabled": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    payload = {
        "source": "Generated offline from OPSIN CML locants; runtime matching is graph-only.",
        "parents": [_parent_row(*spec) for spec in _PARENTS],
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
