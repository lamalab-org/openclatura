from argparse import Namespace
from pathlib import Path

import pytest
from benchmarks import single_thread_benchmark as benchmark


def _measurement(seconds: float) -> dict[str, object]:
    return {
        "version": "test",
        "rows": 5000,
        "elapsed_seconds": seconds,
        "rows_per_second": 5000 / seconds,
        "names_sha256": "checksum",
    }


@pytest.mark.parametrize(
    ("head_seconds", "expected_exit", "expected_passed"),
    [(11.0, 0, True), (12.0, 1, False)],
)
def test_compare_applies_relative_and_absolute_thresholds(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    head_seconds: float,
    expected_exit: int,
    expected_passed: bool,
) -> None:
    monkeypatch.setattr(
        benchmark,
        "_run_measurement",
        lambda source, _corpus, _warmup: _measurement(10.0 if source == Path("base") else head_seconds),
    )
    report_path = tmp_path / "report.json"
    args = Namespace(
        base_src=Path("base"),
        head_src=Path("head"),
        corpus=tmp_path / "corpus.csv",
        repetitions=3,
        warmup_rows=10,
        threshold_percent=15.0,
        minimum_absolute_seconds=1.0,
        report=report_path,
    )

    assert benchmark.compare(args) == expected_exit
    assert f'"passed": {str(expected_passed).lower()}' in report_path.read_text()


def test_load_smiles_requires_nonempty_corpus(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.csv"
    corpus.write_text("smiles\n", encoding="utf-8")

    with pytest.raises(ValueError, match="no SMILES"):
        benchmark._load_smiles(corpus)
