# bluenamer

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
pip install bluenamer
```

Optional extras:

| extra        | adds                                                  |
| ------------ | ----------------------------------------------------- |
| `[opsin]`    | `py2opsin` for OPSIN-based round-trip verification    |
| `[datasets]` | `datasets` + `tqdm` for PubChem/QM9-style evaluations |
| `[web]`      | FastAPI + uvicorn for the HTTP service                |
| `[dev]`      | pytest, ruff, pre-commit, hypothesis, py2opsin        |

```bash
pip install "bluenamer[opsin,datasets]"
```

## Quick start

```python
from bluenamer import name_smiles

name_smiles("CCO")          # 'ethanol'
name_smiles("c1ccccc1")     # 'benzene'
name_smiles("CC(=O)O")      # 'acetic acid'
```

### Typed result with rules hit + OPSIN round-trip

For everything richer than the bare string, use `bluenamer.name`:

```python
from bluenamer import name

result = name("CC(=O)Nc1ccccc1", include_trace=True, verify_opsin=True)

result.name           # 'N-phenylacetamide'
result.smiles         # 'CC(=O)Nc1ccccc1'
result.ok             # True
result.rules_hit      # ('P-44', 'P-45', 'P-41', 'P-61', 'P-67', ...)
result.rule_hints     # ('Parent hydride / parent structure: Blue Book P-44 ...',)
result.opsin_check.status   # 'matched' | 'mismatched' | 'skipped_no_java' | ...
result.verified       # True when opsin_check is matched
```

Errors do not raise — they are captured on `result.error`, which makes
the batch API safe to point at noisy datasets:

```python
from bluenamer import name_many

results = name_many(
    ["CCO", "c1ccccc1", "definitely-not-a-smiles"],
    processes="auto",       # or an integer, or 1 for in-process
    verify_opsin=False,
)
[r.name for r in results if r.ok]
```

For the full decision trace (one `TraceStep` per phase: parse, perception,
parent selection, numbering, assembly, …):

```python
from bluenamer import analyze_smiles

analysis = analyze_smiles("CC(=O)Nc1ccccc1")
for step in analysis.decisions:
    print(step.phase, step.decision, step.reason)
```

### Natural-language description (`describe`)

`bluenamer.describe(smiles)` walks the same trace and renders a
deterministic, multi-paragraph explanation of how the name is built.
Useful for explainability views and for generating (SMILES, name,
description) training tuples:

```python
from bluenamer import describe

d = describe("CC(=O)Nc1ccccc1")
print(d)            # multi-paragraph prose
d.rules_hit         # ('P-44', 'P-45', 'P-41', 'P-61', 'P-67')
d.components[0]     # DescribedComponent(phase='parse', text='RDKit parsed ...')
```

Same input → same output. No LLM in the loop.

### CLI

```bash
bluenamer name "CC(=O)Nc1ccccc1"            # → N-phenylacetamide
bluenamer name "CC(=O)Nc1ccccc1" --json     # JSON with trace + rules
bluenamer batch smiles.txt --output names.jsonl --processes auto
```

## Development

```bash
git clone https://github.com/lamalab-org/openclatura
cd openclatura
pip install -e ".[dev]"

# run the unit + round-trip tests
pytest

# run only fast tests
pytest -m "not slow and not dataset and not golden"

# strict RDKit-version regression suite (also runs in the rdkit-compat CI job)
pytest -m golden

# lint and format
ruff check --fix src/bluenamer
ruff format src/bluenamer
```

Java is required for the OPSIN-based round-trip checks (see `py2opsin`).

## HTTP service (Docker)

The `[web]` extra ships a FastAPI app with `name`, `batch`, `describe`
and `healthz` endpoints. The bundled `Dockerfile` includes a headless JRE
so `verify_opsin=True` works out of the box.

```bash
# build + run
docker build -t bluenamer:local .
docker run --rm -p 8000:8000 bluenamer:local

# or via compose
docker compose -f docker/compose.yaml up --build
```

Call the API:

```bash
curl -X POST localhost:8000/name -H 'content-type: application/json' \
     -d '{"smiles":"CC(=O)Nc1ccccc1","include_trace":true,"verify_opsin":true}'

curl -X POST localhost:8000/batch -H 'content-type: application/json' \
     -d '{"smiles":["CCO","c1ccccc1","CC(=O)O"],"processes":1}'

curl -X POST localhost:8000/describe -H 'content-type: application/json' \
     -d '{"smiles":"CC(=O)Nc1ccccc1"}'
```

OpenAPI docs are served at `http://localhost:8000/docs`.


## Human-like description

OpenBlue can generate uncanny human-like descriptions.
```python

from bluenamer import describe_human

d = describe_human("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")
print(d.text)

""" Processed SMILES: Cn1cnc2c1c(=O)n(C)c(=O)n2C
Atom ids in that SMILES: C{0}n{1}1c{2}n{3}c{4}2c{5}1c{6}(=O{7})n{8}(C{13})c{9}(=O{10})n{11}2C{12}

The molecule is named 2,4,7-trimethyl-2,4,7,9-tetraazabicyclo[4.3.0]nona-1(6),8-diene-3,5-dione.

The molecule is built around a 9-membered bicyclic [4.3.0] heteroskeleton.
Within that parent framework, there is nitrogen at positions 2 (atom id 11), 4 (atom id 8), 7 (atom id 1), and 9 (atom id 3).
Within that parent framework, there is a double bond between position 1 (atom id 4) and position 6 (atom id 5) and a double bond between position 8 (atom id 2) and position 9 (atom id 3).
The principal characteristic feature is oxo groups at positions 3 (atom id 9) and 5 (atom id 6).
Attached to this framework are methyl groups at positions 2 (atom id 11), 4 (atom id 8), and 7 (atom id 1). """

```



## Debugging

Token binding metadata is currently available through the assembly decision trace when `include_trace=True`.

```python
from blunamer import name

result = name("C(C1C(C(C(C(O1)O)O)O)O)O", include_trace=True)

print(result.name)
# 6-(hydroxymethyl)oxane-2,3,4,5-tetraol
```

To inspect token spans:

```python
def name_token_spans(result):
    for step in reversed(result.decisions):
        if isinstance(step.data, dict) and "name_token_spans" in step.data:
            return step.data["name_token_spans"]
    return []


for token in name_token_spans(result):
    print(
        token["text"],
        "atoms=", token["atoms"],
        "bonds=", token["bonds"],
        "kind=", token["token_kind"],
        "confidence=", token["confidence"],
        "source=", token["source"],
    )
```

Example output:

```text
6 atoms= [0, 11] bonds= [1, 11] kind= locant confidence= derived source= typed_rewrite
hydroxymethyl atoms= [0, 11] bonds= [11] kind= prefix confidence= derived source= substituent_renderer
oxane atoms= [1, 2, 3, 4, 5, 6] bonds= [2, 3, 4, 5, 6, 12] kind= parent confidence= derived source= typed_rewrite
2,3,4,5 atoms= [2, 3, 4, 5, 7, 8, 9, 10] bonds= [3, 4, 5, 7, 8, 9, 10] kind= locant confidence= derived source= typed_rewrite
tetraol atoms= [2, 3, 4, 5, 7, 8, 9, 10] bonds= [3, 4, 5, 7, 8, 9, 10] kind= suffix confidence= derived source= renderer_suffix
```

The token metadata is split into `token_kind`, `ownership`, `confidence`, and `source`.

### `token_kind`

What grammar role the token plays.

| Value            | Meaning / example                                                           |
| ---------------- | --------------------------------------------------------------------------- |
| `parent`         | Parent skeleton token, e.g. `ethan`, `benzene`, `spiro[...]`.               |
| `prefix`         | Prefix/substituent token, e.g. `chloro`, `methyl`, `hydroxy`.               |
| `suffix`         | Principal suffix token, e.g. `acid`, `ol`, `one`, `nitrile`.                |
| `locant`         | Locant token, e.g. `2`, `1,3`, `N`, `4a`.                                   |
| `charge`         | Charge-bearing name part, e.g. `ium`, `oxide`, `ammonio`.                   |
| `hydro`          | Indicated hydrogen or hydro operation, e.g. `1H`, `dihydro`.                |
| `replacement`    | Replacement prefix token, e.g. `oxa`, `aza`, `thia`.                        |
| `unsaturation`   | Unsaturation token, e.g. `en`, `yn`, `diene`.                               |
| `modifier`       | Front or suffix modifier token, e.g. stereo, hydro, or functional modifier. |
| `grammar`        | Pure grammar token, e.g. `di`, `bis`, or parentheses-bridging particles.    |
| `structural`     | Structural token that does not fit a narrower kind.                         |
| `retained_alias` | Token matched as a retained-name alias or context term.                     |

### `ownership`

How the token claims graph atoms.

| Value                    | Meaning / example                                                                          |
| ------------------------ | ------------------------------------------------------------------------------------------ |
| `exact`                  | Token was intentionally emitted for these atoms. Best case.                                |
| `preserves_binding`      | Text matched a known binding directly after assembly.                                      |
| `preserve_all`           | Rewrite preserved all previous atom ownership.                                             |
| `locanted_hydro`         | Hydro token owns atoms through locants, e.g. `1H`.                                         |
| `component_locant`       | Locant belongs to a component namespace, often primed spiro/fused components.              |
| `role_alias`             | Token is an alias for a graph role.                                                        |
| `retained_alias_context` | Token matched retained parent alias context.                                               |
| `stage_alias`            | Token inferred from all bindings of a stage, e.g. generic suffix token.                    |
| `grammar_scope`          | Grammar token applies to nearby graph-bound terms, not its own atom.                       |
| `multiplier_scope`       | Multiplier token, e.g. `di`, scopes over repeated graph-bound terms.                       |
| `operation_scope`        | Token recovered from a named operation trace.                                              |
| `morphology_gap`         | Token is a morphology bridge inside a compound token.                                      |
| `ambiguous`              | Best-effort broad binding. Diagnostic only.                                                |
| `unbound`                | No reliable graph binding found. Should be treated as a problem.                           |
| `absorbed`               | Token was absorbed by a rewrite into another token.                                        |
| rewrite-specific values  | Values such as `retained_replace` or `merge_replaced_span`; these come from rewrite rules. |

### `confidence`

How strong the assignment is.

| Value      | Meaning                                                                                                                              |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| `exact`    | Renderer emitted this token with atom metadata directly.                                                                             |
| `derived`  | Recovered from locants, rewrite history, context, or operation trace. |
| `fallback` | Best-effort binding. Useful for debugging, but not proof of correct graph naming.                                                    |

### `source`

Where the binding came from.

| Value                              | Meaning / example                                                            |
| ---------------------------------- | ---------------------------------------------------------------------------- |
| `renderer`                         | Direct renderer-emitted token. Strongest source.                             |
| `renderer_suffix`                  | Principal suffix renderer emitted it.                                        |
| `functional_prefix_renderer`       | Functional-prefix renderer emitted it.                                       |
| `substituent_renderer`             | Substituent renderer emitted it.                                             |
| `default_binding`                  | Built from a broader `NameAtomBinding` when no finer token metadata existed. |
| `typed_rewrite`                    | Came through a typed post-processing rewrite; atom metadata was propagated.  |
| `direct_text_match`                | Token text directly matched an existing binding.                             |
| `locant_fallback`                  | Locant was matched back to a binding by locant metadata.                     |
| `charge_suffix_fallback`           | Charge token inferred from charge suffix context.                            |
| `indicated_hydrogen_fallback`      | `H` or indicated hydrogen inferred from hydro metadata.                      |
| `dihydro_locant_fallback`          | Dihydro locants inferred from hydro operation metadata.                      |
| `primed_component_locant_fallback` | Primed/component locant inferred from component scope.                       |
| `role_alias_fallback`              | Token matched a known role alias.                                            |
| `retained_alias_context`           | Token matched retained-name alias context.                                   |
| `stage_fallback`                   | Token assigned to all bindings of a stage, e.g. parent, prefix, or suffix.   |
| `grammar_token`                    | Pure grammar token.                                                          |
| `operation_trace`                  | Binding recovered from recorded naming operation.                            |
| `broad_fallback`                   | Last-resort plausible chemical token binding. Diagnostic only.               |
| `unresolved`                       | No binding found.                                                            |
| `compound_gap_bridge`              | Token bridges adjacent bound tokens inside a compound word.                  |
| `compound_gap_token`               | Grammar-like token found inside a compound gap.                              |
| `compound_gap_unresolved`          | Unresolved token inside a compound gap.                                      |
| dynamic rewrite names              | Any named rewrite can appear as a source if it changed the token.            |

## License

MIT. See `LICENSE`.
