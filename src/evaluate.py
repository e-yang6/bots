"""Match predicted daughter branches to reference annotations and score them.

CLI:
    python -m src.evaluate --predictions pred.json --references ref.json
"""

import argparse
import json
import math

import numpy as np
from scipy.optimize import linear_sum_assignment


def _extract_daughters(data):
    """Accept either a full prediction/reference JSON dict (with a "daughters"
    key) or a bare list of daughter dicts, and return the list of daughters.
    """
    if isinstance(data, dict):
        return data.get("daughters", [])
    return list(data)


def _euclidean(a, b):
    return math.dist(a, b)


def _direction_angle_deg(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    cos_theta = np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0)
    return math.degrees(math.acos(cos_theta))


def match_and_score(predictions, references, distance_threshold_mm=10):
    """Match predicted daughters to reference daughters and compute scores.

    predictions / references: either a full JSON dict with a "daughters" key,
    or a bare list of daughter dicts. Each daughter dict must have
    ostium_xyz_mm, seed_xyz_mm, radius_mm, direction_xyz.

    Matching is one-to-one via the Hungarian algorithm on pairwise Euclidean
    ostium distance, only accepting a match if distance <= distance_threshold_mm.
    Unmatched predictions are false positives; unmatched references are false
    negatives.

    Returns a dict with precision, recall, f1, mean_ostium_error_mm,
    mean_seed_error_mm, mean_direction_error_deg, mean_radius_error_mm,
    true_positives, false_positives, false_negatives, and matches (a list of
    (pred_index, ref_index, distance_mm) tuples for accepted matches).
    """
    preds = _extract_daughters(predictions)
    refs = _extract_daughters(references)

    n_pred = len(preds)
    n_ref = len(refs)

    matches = []

    if n_pred > 0 and n_ref > 0:
        cost = np.zeros((n_pred, n_ref))
        for i, p in enumerate(preds):
            for j, r in enumerate(refs):
                cost[i, j] = _euclidean(p["ostium_xyz_mm"], r["ostium_xyz_mm"])

        row_ind, col_ind = linear_sum_assignment(cost)
        for i, j in zip(row_ind, col_ind):
            d = cost[i, j]
            if d <= distance_threshold_mm:
                matches.append((int(i), int(j), float(d)))

    matched_pred_idx = {i for i, _, _ in matches}
    matched_ref_idx = {j for _, j, _ in matches}

    true_positives = len(matches)
    false_positives = n_pred - len(matched_pred_idx)
    false_negatives = n_ref - len(matched_ref_idx)

    precision = true_positives / n_pred if n_pred > 0 else 0.0
    recall = true_positives / n_ref if n_ref > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    ostium_errors = []
    seed_errors = []
    direction_errors = []
    radius_errors = []

    for i, j, d in matches:
        p, r = preds[i], refs[j]
        ostium_errors.append(d)
        seed_errors.append(_euclidean(p["seed_xyz_mm"], r["seed_xyz_mm"]))
        direction_errors.append(_direction_angle_deg(p["direction_xyz"], r["direction_xyz"]))
        radius_errors.append(abs(p["radius_mm"] - r["radius_mm"]))

    def _mean(values):
        return float(np.mean(values)) if values else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_ostium_error_mm": _mean(ostium_errors),
        "mean_seed_error_mm": _mean(seed_errors),
        "mean_direction_error_deg": _mean(direction_errors),
        "mean_radius_error_mm": _mean(radius_errors),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "matches": matches,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Score predicted daughter branches against reference annotations."
    )
    parser.add_argument("--predictions", required=True, help="Path to predictions JSON.")
    parser.add_argument("--references", required=True, help="Path to reference annotations JSON.")
    parser.add_argument(
        "--distance-threshold-mm",
        type=float,
        default=10,
        help="Maximum ostium distance (mm) for a valid match (default: 10).",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    with open(args.predictions) as f:
        predictions = json.load(f)
    with open(args.references) as f:
        references = json.load(f)

    scores = match_and_score(predictions, references, distance_threshold_mm=args.distance_threshold_mm)
    printable = {k: v for k, v in scores.items() if k != "matches"}
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()
