"""Run the pipeline over the synthetic phantom suite and score it against
exact, analytically-known ground truth.

This is the ONLY place in the project that can honestly report precision,
recall, F1, or a mean error in mm -- because it is the only place ground
truth exists. It proves the geometry and measurement chain (thresholding,
flood, tracing, ostium/seed/radius/direction extraction) is correct on
vessels shaped like the real ones. It does NOT prove detection succeeds on
real patients: real anatomy, real noise and real segmentation quirks are
not fully captured by any phantom. See scripts/stability.py,
scripts/sweep.py and scripts/plausibility.py for what stands in for that
on the real 25 cases, and README.md for how these are meant to be read
together.

Usage:
    python -m scripts.validate_phantom [--out-dir phantom_suite] [--seed 0]
                                       [--report validate_phantom_report.json]
"""

import argparse
import json
import sys
import time

import numpy as np

sys.path.insert(0, ".")
from run import analyze_case, build_daughters  # noqa: E402
from scripts.make_phantom import build_suite, write_phantom_case  # noqa: E402
from src.evaluate import match_and_score  # noqa: E402

# How close a daughter's ostium must be to a reference point to count as
# "found there" for the structural checks below -- match_and_score's own
# matching radius, reused for consistency rather than inventing a second one.
MATCH_RADIUS_MM = 10.0

# scripts/make_phantom.py's close pair is CLOSE_PAIR_GAP_MM (4mm) apart, so
# a radius comfortably smaller than that avoids one ostium's search picking
# up its neighbour while still being generous about localisation error.
CLOSE_PAIR_SEARCH_RADIUS_MM = 3.0

SEED_ERROR_1_5MM_SPACING_LIMIT_MM = 1.0


def _run_pipeline(image_path, mask_path):
    context = analyze_case(image_path, mask_path, verbose=False)
    daughters, scored = build_daughters(context, verbose=False)
    return daughters, context, scored


def _nearest_daughter_distance(point_mm, daughters):
    if not daughters:
        return np.inf, None
    ostia = np.array([d["ostium_xyz_mm"] for d in daughters])
    distances = np.linalg.norm(ostia - np.asarray(point_mm), axis=1)
    index = int(np.argmin(distances))
    return float(distances[index]), daughters[index]


def check_structural_claims(case_id, meta, daughters):
    """The brief's specific pass/fail assertions -- independent of the
    aggregate Hungarian scoring, because "one instance not two" or "zero
    daughters near this point" are claims about a COUNT, not an error in mm.
    """
    checks = []

    def record(name, passed, detail):
        checks.append({"case": case_id, "check": name, "passed": bool(passed), "detail": detail})

    if meta["truncate"]:
        distance, nearest = _nearest_daughter_distance(meta["truncation_point_mm"], daughters)
        record(
            "truncation_no_branch_at_cut_face", distance > MATCH_RADIUS_MM,
            f"nearest daughter {distance:.1f}mm from the cut face (need > {MATCH_RADIUS_MM:.0f}mm)"
            + (f", instance {nearest['instance_id']}" if nearest else ""),
        )

    if "leak_blob" in meta.get("distractors", {}):
        distance, nearest = _nearest_daughter_distance(meta["distractors"]["leak_blob"]["centre"], daughters)
        record(
            "leak_bridge_not_reported", distance > MATCH_RADIUS_MM,
            f"nearest daughter {distance:.1f}mm from the bridged blob (need > {MATCH_RADIUS_MM:.0f}mm)"
            + (f", instance {nearest['instance_id']}" if nearest else ""),
        )

    if "ivc" in meta.get("distractors", {}):
        ivc = meta["distractors"]["ivc"]
        midpoint = (np.asarray(ivc["start"]) + np.asarray(ivc["end"])) / 2.0
        distance, nearest = _nearest_daughter_distance(midpoint, daughters)
        record(
            "ivc_not_reported_at_midpoint", distance > MATCH_RADIUS_MM,
            f"nearest daughter {distance:.1f}mm from the IVC analogue's midpoint"
            + (f", instance {nearest['instance_id']}" if nearest else ""),
        )

    thin_branch = next((b for b in meta["branches"] if b["label"] == "thin_branch"), None)
    if thin_branch is not None:
        distance, nearest = _nearest_daughter_distance(thin_branch["ostium"], daughters)
        record(
            "thin_branch_not_reported", distance > MATCH_RADIUS_MM,
            f"nearest daughter {distance:.1f}mm from the {thin_branch['radius_mm']:.1f}mm-radius "
            f"thin branch's ostium (need > {MATCH_RADIUS_MM:.0f}mm)"
            + (f", instance {nearest['instance_id']}" if nearest else ""),
        )

    trunk = next((b for b in meta["branches"] if b["label"] == "bifurcation_trunk"), None)
    if trunk is not None:
        within = [
            d for d in daughters
            if np.linalg.norm(np.asarray(d["ostium_xyz_mm"]) - trunk["ostium"]) <= MATCH_RADIUS_MM
        ]
        record(
            "bifurcation_reported_as_one_instance", len(within) == 1,
            f"{len(within)} daughters within {MATCH_RADIUS_MM:.0f}mm of the trunk's ostium (need exactly 1)",
        )

    pair = [b for b in meta["branches"] if b["label"].startswith("close_pair_")]
    if len(pair) == 2:
        matches = []
        for branch in pair:
            distance, nearest = _nearest_daughter_distance(branch["ostium"], daughters)
            matches.append(nearest["instance_id"] if distance <= CLOSE_PAIR_SEARCH_RADIUS_MM else None)
        record(
            "close_pair_stays_two_instances",
            all(m is not None for m in matches) and matches[0] != matches[1],
            f"matches: {matches} (need two distinct, non-None instances)",
        )

    if meta["non_contrast"]:
        record("non_contrast_returns_empty", len(daughters) == 0, f"{len(daughters)} daughters (need 0)")

    return checks


def _spacing_tag(spacing):
    sx, sy, sz = spacing
    return "iso" if sx == sy == sz else "aniso"


def summarize(rows, key):
    groups = {}
    for row in rows:
        groups.setdefault(row[key], []).append(row)
    summary = {}
    for value, group in sorted(groups.items(), key=lambda kv: str(kv[0])):
        summary[str(value)] = {
            metric: float(np.mean([r[metric] for r in group if r[metric] is not None]))
            for metric in ("precision", "recall", "f1", "mean_ostium_error_mm", "mean_seed_error_mm",
                          "mean_direction_error_deg", "mean_radius_error_mm")
            if any(r[metric] is not None for r in group)
        }
        summary[str(value)]["n_cases"] = len(group)
    return summary


def run_validation(out_dir="phantom_suite", seed=0, verbose=True):
    cases = build_suite(base_seed=seed)
    rows, all_checks = [], []

    for image, mask, reference, meta in cases:
        paths = write_phantom_case(image, mask, reference, out_dir)
        t0 = time.time()
        daughters, _context, _scored = _run_pipeline(paths["image"], paths["mask"])
        elapsed = time.time() - t0

        prediction = {"case_id": meta["case_id"], "parent": reference["parent"], "daughters": daughters}
        score = match_and_score(prediction, reference, distance_threshold_mm=MATCH_RADIUS_MM)

        row = {
            "case_id": meta["case_id"], "spacing": meta["spacing"], "spacing_tag": _spacing_tag(meta["spacing"]),
            "lumen_hu": meta["lumen_hu"], "n_branches": meta["n_branches"], "n_reference": len(reference["daughters"]),
            "n_predicted": len(daughters), "elapsed_s": elapsed, **score,
        }
        rows.append(row)
        all_checks.extend(check_structural_claims(meta["case_id"], meta, daughters))

        if verbose:
            if row["n_reference"] == 0:
                status = "OK" if row["n_predicted"] == 0 else "LOW-F1"
            else:
                status = "OK" if score["f1"] >= 0.5 else "LOW-F1"
            print(f"{meta['case_id']:55s} ref={row['n_reference']:2d} pred={row['n_predicted']:2d} "
                  f"f1={score['f1']:.2f} ostium_err={score['mean_ostium_error_mm']:.2f}mm "
                  f"[{elapsed:.1f}s] {status}")

    # The seed-error regression check from the brief: at 1.5mm isotropic
    # spacing, seed error should stay under ~1mm -- this is sensitive to any
    # Euclidean-vs-geodesic arc-length mixup (src.tracing._point_at_arc uses
    # geodesic distance directly, never a re-measured Euclidean one; a
    # regression there shows up as seed error growing with radius/curvature).
    iso_1_5_rows = [r for r in rows if r["spacing"] == (1.5, 1.5, 1.5) and r["mean_seed_error_mm"] is not None]
    if iso_1_5_rows:
        mean_seed_error = float(np.mean([r["mean_seed_error_mm"] for r in iso_1_5_rows]))
        all_checks.append({
            "case": "iso_1.5_aggregate", "check": "seed_error_under_1mm_at_1.5mm_spacing",
            "passed": mean_seed_error < SEED_ERROR_1_5MM_SPACING_LIMIT_MM,
            "detail": f"mean seed error {mean_seed_error:.2f}mm across {len(iso_1_5_rows)} cases "
                     f"(need < {SEED_ERROR_1_5MM_SPACING_LIMIT_MM:.1f}mm)",
        })

    return rows, all_checks


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", default="phantom_suite")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--report", default="validate_phantom_report.json")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    rows, checks = run_validation(out_dir=args.out_dir, seed=args.seed)

    # Phantoms with zero reference daughters (the non-contrast case) have no
    # meaningful precision/recall/F1 -- "correctly predicted nothing" is a
    # structural check (see check_structural_claims), not a point on an
    # accuracy curve, and would silently drag the aggregate F1 toward 0.
    scoring_rows = [r for r in rows if r["n_reference"] > 0]

    overall = {
        metric: float(np.mean([r[metric] for r in scoring_rows if r[metric] is not None]))
        for metric in ("precision", "recall", "f1", "mean_ostium_error_mm", "mean_seed_error_mm",
                      "mean_direction_error_deg", "mean_radius_error_mm")
        if any(r[metric] is not None for r in scoring_rows)
    }

    report = {
        "overall": overall,
        "by_spacing": summarize(scoring_rows, "spacing_tag"),
        "by_lumen_hu": summarize(scoring_rows, "lumen_hu"),
        "by_n_branches": summarize(scoring_rows, "n_branches"),
        "cases": rows,
        "structural_checks": checks,
    }
    with open(args.report, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print()
    print("=== overall (phantom suite; NOT a real-case accuracy figure) ===")
    for key, value in overall.items():
        print(f"  {key}: {value:.3f}")

    print()
    print("=== structural checks ===")
    failed = [c for c in checks if not c["passed"]]
    for check in checks:
        mark = "PASS" if check["passed"] else "FAIL"
        print(f"  [{mark}] {check['case']}: {check['check']} -- {check['detail']}")

    print()
    print(f"wrote {args.report}")
    if failed:
        print(f"{len(failed)}/{len(checks)} structural checks FAILED")
        return 1
    print(f"all {len(checks)} structural checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
