## Default test suite

Run the default test suite, excluding slow and dataset-dependent tests:

```bash
pytest -m "not slow and not dataset"
```

## Fuzz tests

Run the Hypothesis-based fuzz tests:

```bash
pytest tests/fuzz/test_smiles_strategies.py -v
```

## Optional dataset tests

Install the optional dataset dependencies first:

```bash
pip install -e ".[datasets]"
```

Then run all dataset tests:

```bash
pytest -m dataset -v -s
```

You can also run the dataset tests individually:

```bash
pytest tests/datasets/test_pubchem_sample.py -v -s
pytest tests/datasets/test_qm9_sample.py -v -s
```

The sample size and random seed can be adjusted with environment variables:

```bash
OPENCLATURA_DATASET_SAMPLE_N=50 OPENCLATURA_DATASET_SEED=123 pytest -m dataset -v -s
```

## Testing OPSIN-specific behavior

Run OPSIN-specific tests with:

```bash
pytest -m opsin -v -s
```

Note that OPSIN tests require OPSIN/Java to be available. 