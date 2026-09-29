# Performance benchmarks

The CI performance gate compares the pull request with its target commit on
the same GitHub Actions runner. It measures 5,000 calls through `name_many`
with `processes=1`, after a 100-molecule warm-up. Three adjacent base/PR pairs
are measured, with the order reversed for the middle pair to reduce runner
warm-up and drift bias.

The gate fails only when the median paired result is both more than 15% and
more than one second slower. Raw measurements are written to separate
`performance-report-*.json` files and uploaded together as a workflow artifact.

The original tracked corpus was selected deterministically from
`test_100000.csv`. A second corpus is selected from a new deterministic
200,000-molecule sample of the pinned PubChem Hugging Face dataset. Both
contain only structures whose generated openclatura name round-tripped to the
input through OPSIN. Selection metadata and checksums are recorded in their
adjacent manifest files.

To rebuild the corpus locally with the optional OPSIN dependency and Java
available:

```bash
python benchmarks/build_corpus.py \
  /path/to/test_100000.csv \
  benchmarks/data/opsin_verified_5000.csv
```

To regenerate the PubChem verified pool and its 5,000-molecule speed corpus:

```bash
python benchmarks/build_pubchem_hf_corpus.py
```

This uses pinned revision
`cd46fb8a3110cc12d6a669ad0d8e260b3acf6d0b`, sampling seed `20260929`,
and benchmark-selection seed `20260930`. The full successful pool is retained
as `data/pubchem_hf_seed20260929_verified.csv`; CI benchmarks the deterministic
5,000-row subset so both three-pair comparisons fit within the job timeout.

To compare two local checkouts:

```bash
python benchmarks/single_thread_benchmark.py \
  --base-src /path/to/base/src \
  --head-src /path/to/head/src \
  --corpus benchmarks/data/opsin_verified_5000.csv
```
