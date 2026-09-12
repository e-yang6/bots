import json
from pathlib import Path

from src.evaluate import match_and_score

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name):
    with open(FIXTURES / name) as f:
        return json.load(f)


def test_perfect_match_gives_f1_one_and_zero_errors():
    pred = _load("perfect_pred.json")
    ref = _load("perfect_ref.json")

    scores = match_and_score(pred, ref)

    assert scores["precision"] == 1.0
    assert scores["recall"] == 1.0
    assert scores["f1"] == 1.0
    assert scores["true_positives"] == 2
    assert scores["false_positives"] == 0
    assert scores["false_negatives"] == 0
    assert scores["mean_ostium_error_mm"] == 0.0
    assert scores["mean_seed_error_mm"] == 0.0
    assert scores["mean_direction_error_deg"] == 0.0
    assert scores["mean_radius_error_mm"] == 0.0


def test_one_false_positive_and_one_false_negative():
    pred = _load("fp_fn_pred.json")
    ref = _load("fp_fn_ref.json")

    scores = match_and_score(pred, ref)

    assert scores["true_positives"] == 1
    assert scores["false_positives"] == 1
    assert scores["false_negatives"] == 1
    assert scores["precision"] == 0.5
    assert scores["recall"] == 0.5
    assert scores["f1"] == 0.5
    # matched pair is the coincident (branch_001, branch_001) ostium
    assert len(scores["matches"]) == 1
    pred_idx, ref_idx, distance = scores["matches"][0]
    assert distance == 0.0


def test_prediction_outside_distance_threshold_is_unmatched():
    pred = _load("outside_threshold_pred.json")
    ref = _load("outside_threshold_ref.json")

    scores = match_and_score(pred, ref, distance_threshold_mm=10)

    assert scores["matches"] == []
    assert scores["true_positives"] == 0
    assert scores["false_positives"] == 1
    assert scores["false_negatives"] == 1
    assert scores["precision"] == 0.0
    assert scores["recall"] == 0.0
    assert scores["f1"] == 0.0

    # the same pair matches once the threshold is widened past the 15mm gap
    scores_wide = match_and_score(pred, ref, distance_threshold_mm=20)
    assert len(scores_wide["matches"]) == 1


def test_empty_predictions_vs_nonempty_references_recall_zero_no_crash():
    pred = _load("empty_pred.json")
    ref = _load("nonempty_ref.json")

    scores = match_and_score(pred, ref)

    assert scores["recall"] == 0.0
    assert scores["precision"] == 0.0
    assert scores["f1"] == 0.0
    assert scores["true_positives"] == 0
    assert scores["false_positives"] == 0
    assert scores["false_negatives"] == 2
    assert scores["matches"] == []
