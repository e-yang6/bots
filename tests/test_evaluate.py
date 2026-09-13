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


def _daughter(instance_id, ostium, radius):
    return {
        "instance_id": instance_id, "parent_instance_id": "aorta",
        "ostium_xyz_mm": ostium, "seed_xyz_mm": [ostium[0] + 5.0, ostium[1], ostium[2]],
        "radius_mm": radius, "direction_xyz": [1.0, 0.0, 0.0],
    }


def test_null_reference_radius_is_skipped_for_radius_error_only():
    # draft EVAL_SET references carry radius_mm: null when unmeasurable
    pred = {"daughters": [_daughter("branch_001", [0.0, 0.0, 0.0], 2.5),
                          _daughter("branch_002", [0.0, 50.0, 0.0], 1.0)]}
    ref = {"daughters": [_daughter("branch_001", [1.0, 0.0, 0.0], 2.0),
                         _daughter("branch_002", [0.0, 50.0, 2.0], None)]}

    scores = match_and_score(pred, ref)

    # both still match and count fully for detection and the other errors
    assert scores["true_positives"] == 2
    assert scores["f1"] == 1.0
    assert scores["mean_ostium_error_mm"] == 1.5
    # radius error comes from the measured pair alone
    assert scores["radius_scored_matches"] == 1
    assert scores["mean_radius_error_mm"] == 0.5


def test_all_null_reference_radii_give_no_radius_error_rather_than_zero():
    pred = {"daughters": [_daughter("branch_001", [0.0, 0.0, 0.0], 2.5)]}
    ref = {"daughters": [_daughter("branch_001", [0.0, 0.0, 0.0], None)]}

    scores = match_and_score(pred, ref)

    assert scores["true_positives"] == 1
    assert scores["radius_scored_matches"] == 0
    assert scores["mean_radius_error_mm"] is None


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
