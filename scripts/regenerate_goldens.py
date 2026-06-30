"""Regenerate the corpus golden file from the current namer + rdkit.

Run when the namer or rdkit version intentionally changes the expected
output for a corpus entry::

    python scripts/regenerate_goldens.py

Always run after the change is reviewed: the goldens are a *contract*
that next CI run pins. Inspect the diff before committing.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import rdkit

from openclatura import name_smiles

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "tests/fixtures/diverse_corpus.csv"
GOLDEN = ROOT / "tests/fixtures/diverse_corpus.golden.json"


def main() -> int:
    rows = []
    with CORPUS.open() as fh:
        for row in csv.DictReader(fh):
            smiles = row["smiles"].strip()
            if not smiles:
                continue
            rows.append(
                {
                    "smiles": smiles,
                    "name": name_smiles(smiles),
                    "category": row.get("category", "").strip(),
                }
            )
    GOLDEN.write_text(json.dumps(rows, indent=2) + "\n")
    named = sum(1 for r in rows if r["name"])
    print(f"rdkit {rdkit.__version__}: wrote {len(rows)} entries, named={named}/{len(rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
