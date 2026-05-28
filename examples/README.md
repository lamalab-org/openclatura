# Examples

Standalone scripts and sanity tools. They are not part of the test suite —
several require optional dependencies (`py2opsin`, `datasets`, network access).

## Files

| script | purpose | extras needed |
| --- | --- | --- |
| `sanity_examples.py` | Hand-picked extreme structures, prints IUPAC names | none |
| `random_sanity.py` | Random spot-check of the naming engine | none |
| `eval_via_opsin.py` | Sample N molecules from PubChem and measure round-trip accuracy with OPSIN | `[opsin,datasets]` |
| `find_small_failures.py` | Filter PubChem for small molecules and find round-trip failures | `[opsin,datasets]` |
| `test_opsin_mac.py` | macOS-friendly variant of `eval_via_opsin.py` (forces `spawn` start method) | `[opsin,datasets]` |

Install the extras:

```bash
pip install -e .[opsin,datasets]
```

Run, e.g.:

```bash
python examples/sanity_examples.py
python examples/eval_via_opsin.py
```
