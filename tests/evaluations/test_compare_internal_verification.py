import importlib.util
import sys
from pathlib import Path


def _load_comparison():
    path = Path(__file__).parents[2] / "evaluations" / "compare_internal_verification.py"
    spec = importlib.util.spec_from_file_location("compare_internal_verification", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


COMPARISON = _load_comparison()


def test_comparison_categories_distinguish_catches_misses_and_abstentions():
    assert COMPARISON.category("matched", "confirmed") == "agree_valid"
    assert COMPARISON.category("matched", "mismatch") == "internal_rejected_opsin_match"
    assert COMPARISON.category("failed", "mismatch") == "caught_opsin_failure"
    assert COMPARISON.category("failed", "confirmed") == "missed_opsin_failure"
    assert COMPARISON.category("failed", "abstained") == "internal_abstained_on_opsin_failure"


def test_metrics_treat_only_positive_mismatches_as_caught_failures():
    matrix = {
        "matched": {"confirmed": 80, "mismatch": 2, "abstained": 8, "error": 0},
        "failed": {"confirmed": 3, "mismatch": 4, "abstained": 2, "error": 1},
    }

    metrics = COMPARISON._metrics(matrix)

    assert metrics["total"] == 100
    assert metrics["decisive"] == 89
    assert metrics["opsin_failures_caught"] == 4
    assert metrics["opsin_failures_missed"] == 3
    assert metrics["failure_detection_recall_percent"] == 40.0
    assert metrics["mismatch_precision_percent"] == 66.666667
    assert metrics["opsin_matches_rejected"] == 2


def test_reason_groups_collapse_name_and_locant_specific_abstentions():
    assert COMPARISON._reason_group("[abstained] substituent '(4-fluorophenyl)' not modelled") == (
        "substituent not modelled"
    )
    assert COMPARISON._reason_group("[abstained] indicated hydrogen 7a not placeable") == (
        "indicated hydrogen not placeable"
    )


def test_name_changed_category_is_separate_from_opsin_confusion_matrix():
    assert COMPARISON.NAME_CHANGED_CATEGORY == "name_changed_during_internal_audit"
