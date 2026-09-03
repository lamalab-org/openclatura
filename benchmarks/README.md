# Performance benchmarks

The CI performance gate compares the pull request with its target commit on
the same GitHub Actions runner. It measures 5,000 calls through `name_many`
with `processes=1`, after a 100-molecule warm-up. Three adjacent base/PR pairs
are measured, with the order reversed for the middle pair to reduce runner
warm-up and drift bias.

The gate fails only when the median paired result is both more than 15% and
more than one second slower. Raw measurements are written to
`performance-report.json` and uploaded as a workflow artifact.

The tracked corpus was selected deterministically from `test_100000.csv` and
contains only structures whose generated OpenClatura name round-tripped to the
input through OPSIN. Its selection metadata and checksum are recorded in
`data/opsin_verified_5000.manifest.json`.

To rebuild the corpus locally with the optional OPSIN dependency and Java
available:

```bash
python benchmarks/build_corpus.py \
  /path/to/test_100000.csv \
  benchmarks/data/opsin_verified_5000.csv
```

To compare two local checkouts:

```bash
python benchmarks/single_thread_benchmark.py \
  --base-src /path/to/base/src \
  --head-src /path/to/head/src \
  --corpus benchmarks/data/opsin_verified_5000.csv
```
