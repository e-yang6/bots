"""Sweep the two most consequential tunables -- confidence threshold and
flood budget -- over the real dev-set cases, with no reference labels
needed, to choose an operating point defensibly.

There is nothing here to maximize: without labels there is no F1 curve, only
a question of where the pipeline's OWN behaviour is stable. A threshold (or
budget) sitting in the middle of a broad, flat stretch of "detections per
case" and "detections per 10cm of aorta" is a safe choice -- a small error in
where exactly to draw the line barely changes what gets reported. A
threshold sitting on a cliff, where a tiny nudge changes the count sharply,
is a bad choice even if it happens to look good on today's cohort, because
it is one dataset shift away from landing on the wrong side of the cliff.
See scripts/stability.py for the complementary per-detection view (does THIS
finding survive a nudge); this script asks the aggregate, per-parameter
version of the same question, and scripts/validate_phantom.py for the only
place accuracy numbers are honest at all.

Usage:
    python -m scripts.sweep [--data-dir PATH] [--cases subject001 ...]
                            [--report sweep_report.json]
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run import analyze_volumes, build_daughters  # noqa: E402
from scripts.inspect_pipeline import find_real_data_dir, list_real_cases  # noqa: E402
from src.evaluate import match_and_score  # noqa: E402
from src.floodfill import DEFAULT_BUDGET_MM  # noqa: E402
from src.geometry import crop_to_mask_bbox, resample_isotropic  # noqa: E402
from src.io_utils import load_case  # noqa: E402
from src.rules import CONFIDENCE_THRESHOLD  # noqa: E402

# Threshold sweep only re-scores already-traced instances (build_daughters
# doesn't touch the flood), so this can afford to be fine-grained.
THRESHOLD_GRID = tuple(round(t, 3) for t in np.arange(0.05, 0.96, 0.05))

# A small nudge either side of each threshold, used only to ask "does the
# accepted set change under a tiny move" -- the local-flatness signal, not a
# claim about correctness.
THRESHOLD_LOCAL_DELTA = 0.02

# Budget sweep re-runs the flood itself, so this stays coarser to keep
# runtime bounded (see the elapsed_s figures in the report).
BUDGET_FRACTION_GRID = (0.5, 0.7, 0.85, 1.0, 1.3, 1.6, 2.0)

# Same-case identity match: an accepted candidate's ostium is identical
# regardless of the threshold that accepted it (only the accept/reject
# decision moves, not the geometry), so a tight radius is a correspondence
# check, not a tolerance.
IDENTITY_MATCH_RADIUS_MM = 0.5
CROSS_RUN_MATCH_RADIUS_MM = 5.0

CM_MM = 100.0  # per-10cm normalisation


def _aortic_length_mm(context):
    total = 0.0
    for component in context["centerline"]["kept"]:
        points = component["points_mm"]
        if len(points) < 2:
            continue
        total += float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
    return total


def _local_stability(context, threshold):
    lo, _ = build_daughters(context, threshold=max(0.0, threshold - THRESHOLD_LOCAL_DELTA))
    hi, _ = build_daughters(context, threshold=min(1.0, threshold + THRESHOLD_LOCAL_DELTA))
    if not lo and not hi:
        return 1.0
    score = match_and_score(hi, lo, distance_threshold_mm=IDENTITY_MATCH_RADIUS_MM)
    agree = score["true_positives"]
    return agree / max(len(lo), len(hi))


def _threshold_sweep(context, aortic_length_mm):
    rows = []
    for threshold in THRESHOLD_GRID:
        daughters, _ = build_daughters(context, threshold=threshold)
        n = len(daughters)
        per_10cm = n / (aortic_length_mm / CM_MM) if aortic_length_mm > 0 else None
        rows.append({
            "threshold": threshold, "n_daughters": n, "detections_per_10cm": per_10cm,
            "local_stability": _local_stability(context, threshold),
        })
    return rows


def _budget_sweep(resampled_image, resampled_mask, original_grid, aortic_length_mm):
    rows = []
    for fraction in BUDGET_FRACTION_GRID:
        budget = DEFAULT_BUDGET_MM * fraction
        context = analyze_volumes(resampled_image, resampled_mask, original_grid=original_grid,
                                  flood_budget_mm=budget)
        daughters, _ = build_daughters(context)
        n = len(daughters)
        per_10cm = n / (aortic_length_mm / CM_MM) if aortic_length_mm > 0 else None
        rows.append({"budget_mm": budget, "budget_fraction": fraction, "n_daughters": n,
                     "detections_per_10cm": per_10cm})
    return rows


def run_sweep(data_dir, case_names=None, verbose=True):
    cases = list_real_cases(data_dir)
    if case_names:
        cases = [c for c in cases if c[0] in case_names]

    per_case = []
    for case_id, image_path, mask_path in cases:
        t0 = time.time()
        image, mask = load_case(image_path, mask_path)
        cropped_image, cropped_mask = crop_to_mask_bbox(image, mask, margin_mm=15)
        resampled_image, resampled_mask, original_grid = resample_isotropic(cropped_image, cropped_mask)

        baseline_context = analyze_volumes(resampled_image, resampled_mask, original_grid=original_grid)
        aortic_length_mm = _aortic_length_mm(baseline_context)

        threshold_rows = _threshold_sweep(baseline_context, aortic_length_mm)
        budget_rows = _budget_sweep(resampled_image, resampled_mask, original_grid, aortic_length_mm)

        elapsed = time.time() - t0
        per_case.append({
            "case_id": case_id, "aortic_length_mm": aortic_length_mm,
            "threshold_sweep": threshold_rows, "budget_sweep": budget_rows, "elapsed_s": elapsed,
        })
        if verbose:
            n_at_default = next(r["n_daughters"] for r in threshold_rows if abs(r["threshold"] - CONFIDENCE_THRESHOLD) < 1e-6) \
                if any(abs(r["threshold"] - CONFIDENCE_THRESHOLD) < 1e-6 for r in threshold_rows) else None
            print(f"{case_id:12s} aortic_length={aortic_length_mm:6.1f}mm "
                  f"n_at_default_threshold={n_at_default} [{elapsed:.1f}s]", file=sys.stderr)

    aggregate_threshold = _aggregate_axis(per_case, "threshold_sweep", "threshold")
    aggregate_budget = _aggregate_axis(per_case, "budget_sweep", "budget_fraction")

    chosen_threshold, plateau = _choose_operating_point(aggregate_threshold, "threshold")

    return {
        "cases": per_case,
        "aggregate_threshold_sweep": aggregate_threshold,
        "aggregate_budget_sweep": aggregate_budget,
        "chosen_threshold": chosen_threshold,
        "chosen_threshold_plateau": plateau,
        "current_default_threshold": CONFIDENCE_THRESHOLD,
    }


def _aggregate_axis(per_case, sweep_key, x_key):
    """Mean detections/case, mean detections/10cm, mean local_stability (if
    present) at each x value across every case that has it.
    """
    by_x = {}
    for case in per_case:
        for row in case[sweep_key]:
            by_x.setdefault(row[x_key], []).append(row)
    aggregate = []
    for x in sorted(by_x):
        rows = by_x[x]
        entry = {
            x_key: x,
            "mean_n_daughters": float(np.mean([r["n_daughters"] for r in rows])),
            "total_n_daughters": int(np.sum([r["n_daughters"] for r in rows])),
            "mean_detections_per_10cm": float(np.mean(
                [r["detections_per_10cm"] for r in rows if r["detections_per_10cm"] is not None]
            )),
        }
        if "local_stability" in rows[0]:
            entry["mean_local_stability"] = float(np.mean([r["local_stability"] for r in rows]))
        aggregate.append(entry)
    return aggregate


# A "cliff" is a single step whose relative change in total detections
# exceeds this fraction of the step's own starting count -- a LOCAL test,
# not "how far has this drifted from some reference point". A global,
# window-max-relative version of this rejected almost the whole low end of
# the threshold grid as "not flat enough" for exactly the wrong reason: no
# candidate in this cohort scores between roughly 0.05 and 0.45 at all, so
# that whole stretch is trivially flat (there is nothing there to change),
# which is a very different claim from "this is a validated safe region".
LOCAL_STEP_TOLERANCE = 0.20
PLATEAU_STABILITY_FLOOR = 0.9


def _choose_operating_point(aggregate_threshold, x_key):
    """Whether the CURRENT default sits on a cliff or in a graded-but-safe
    region, and how far that region extends either side of it.

    Grows outward from the grid point nearest rules.CONFIDENCE_THRESHOLD
    while consecutive steps stay under LOCAL_STEP_TOLERANCE and local
    stability stays at or above PLATEAU_STABILITY_FLOOR -- this asks "is the
    existing default defensible", which is what a threshold sweep with no
    labels can actually answer, rather than hunting for a different
    'optimal' value nothing here has grounds to prefer.
    """
    xs = [row[x_key] for row in aggregate_threshold]
    counts = [row["total_n_daughters"] for row in aggregate_threshold]
    stabilities = [row.get("mean_local_stability", 1.0) for row in aggregate_threshold]
    n = len(xs)

    anchor = min(range(n), key=lambda i: abs(xs[i] - CONFIDENCE_THRESHOLD))

    def _step_ok(i, j):
        if counts[i] == 0 and counts[j] == 0:
            return True
        reference = max(counts[i], counts[j], 1)
        return abs(counts[i] - counts[j]) / reference <= LOCAL_STEP_TOLERANCE

    start = end = anchor
    while start - 1 >= 0 and _step_ok(start - 1, start) and stabilities[start - 1] >= PLATEAU_STABILITY_FLOOR:
        start -= 1
    while end + 1 < n and _step_ok(end, end + 1) and stabilities[end + 1] >= PLATEAU_STABILITY_FLOOR:
        end += 1

    on_a_cliff = start == end == anchor and n > 1

    plateau = {
        "lower": xs[start], "upper": xs[end],
        "width": xs[end] - xs[start],
        "contains_current_default": True,
        "on_a_cliff": on_a_cliff,
        "mean_total_n_daughters": float(np.mean(counts[start:end + 1])),
        "mean_local_stability": float(np.mean(stabilities[start:end + 1])),
    }
    # The default itself, not some other point in the plateau: the question
    # this answers is "is the existing value defensible", not "what value
    # would this sweep have picked with no prior default to check".
    return CONFIDENCE_THRESHOLD, plateau


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=None, help="Path to the TORALIS CHALLENGE dev-set folder.")
    parser.add_argument("--cases", nargs="*", default=None, help="Restrict to these case ids (e.g. subject001).")
    parser.add_argument("--report", default="sweep_report.json")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    data_dir = find_real_data_dir(args.data_dir)
    if data_dir is None:
        print("No TORALIS CHALLENGE data folder found; nothing to do.", file=sys.stderr)
        return 1

    report = run_sweep(data_dir, case_names=args.cases)
    with open(args.report, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print()
    print("=== aggregate threshold sweep (total across all cases) ===")
    for row in report["aggregate_threshold_sweep"]:
        stab = row.get("mean_local_stability")
        print(f"  T={row['threshold']:.2f}  total_n={row['total_n_daughters']:3d}  "
              f"mean/case={row['mean_n_daughters']:.2f}  per10cm={row['mean_detections_per_10cm']:.2f}"
              + (f"  local_stability={stab:.2f}" if stab is not None else ""))

    print()
    print("=== aggregate flood-budget sweep (total across all cases) ===")
    for row in report["aggregate_budget_sweep"]:
        print(f"  budget={row['budget_fraction']:.2f}x  total_n={row['total_n_daughters']:3d}  "
              f"mean/case={row['mean_n_daughters']:.2f}  per10cm={row['mean_detections_per_10cm']:.2f}")

    print()
    plateau = report["chosen_threshold_plateau"]
    print(f"current default CONFIDENCE_THRESHOLD = {report['current_default_threshold']:.2f} -- kept as-is")
    if plateau["on_a_cliff"]:
        print(f"  ON A CLIFF: neighbouring grid points already differ by more than "
              f"{LOCAL_STEP_TOLERANCE:.0%} in total detections -- reconsider this default.")
    else:
        print(f"  sits inside a graded-but-safe region [{plateau['lower']:.2f}, {plateau['upper']:.2f}] "
              f"(width {plateau['width']:.2f}) where no neighbouring step changes total detections by "
              f"more than {LOCAL_STEP_TOLERANCE:.0%}, mean_local_stability={plateau['mean_local_stability']:.2f}")
    print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
