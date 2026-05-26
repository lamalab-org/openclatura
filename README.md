# structure-to-iupac

A deterministic SMILES → IUPAC name generator implementing the rules of the
IUPAC Blue Book (2013 recommendations).

The package walks the molecular graph (parsed via RDKit), perceives functional
groups and ring systems, selects a principal parent, assigns locants, and
assembles the substitutive name. Every step is recorded in an inspectable
decision trace so the *why* of a name is recoverable, not just the *what*.

> **Status:** alpha. The naming engine handles a broad slice of organic
> structures (alkanes/alkenes/alkynes, common functional groups, simple
> heterocycles, fused/spiro/bridged systems, retained names from the Blue
> Book). PubChem/QM9 coverage is being measured; see `examples/`.

## Install

```bash
pip install structure_to_iupac
```

Optional extras:

| extra        | adds                                                  |
| ------------ | ----------------------------------------------------- |
| `[opsin]`    | `py2opsin` for OPSIN-based round-trip verification    |
| `[datasets]` | `datasets` + `tqdm` for PubChem/QM9-style evaluations |
| `[web]`      | FastAPI + uvicorn for the HTTP service                |
| `[dev]`      | pytest, ruff, pre-commit, hypothesis, py2opsin        |

```bash
pip install "structure_to_iupac[opsin,datasets]"
```

## Quick start

```python
from structure_to_iupac import name_smiles

name_smiles("CCO")          # 'ethanol'
name_smiles("c1ccccc1")     # 'benzene'
name_smiles("CC(=O)O")      # 'acetic acid'
```

For an explainable result with the full decision trace:

```python
from structure_to_iupac import analyze_smiles

analysis = analyze_smiles("CC(=O)Nc1ccccc1")
print(analysis.name)        # 'N-phenylacetamide'
for step in analysis.decisions:
    print(step.phase, step.decision, step.reason)
```

## Development

```bash
git clone https://github.com/lamalab-org/iupac-name-generator
cd iupac-name-generator
pip install -e ".[dev]"

# run the unit + round-trip tests
pytest

# run only fast tests
pytest -m "not slow and not dataset"

# lint and format
ruff check --fix
ruff format
```

Java is required for the OPSIN-based round-trip checks (see `py2opsin`).

## License

MIT. See `LICENSE`.
