"""Unlabelled sanity checks across the real dev-set cases.

No reference labels exist for these cases, so nothing here can say a
detection is right or wrong. What it CAN do is compare every case's own
detections against the distribution the rest of the cohort produces:
branch density (detections per 10cm of aortic mask), radius, angle to the
aortic centerline, and how many detections sit near a mask cap. A case that
sits far outside the cohort's own spread on any of these is worth a manual
look -- either the case is unusual, or the pipeline is misbehaving on it --
and scripts/visualize_case.py is where that look actually happens.

An outlier flag here is a pointer to go look, not a verdict.

Usage:
    python -m scripts.plausibility [--data-dir PATH] [--cases subject001 ...]
                                   [--report plausibility_report.json]
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run import analyze_case, build_daughters  # noqa: E402
from scripts.inspect_pipeline import find_real_data_dir, list_real_cases  # noqa: E402
from scripts.sweep import _aortic_length_mm  # noqa: E402

CM_MM = 100.0

# Modified z-score (0.6745 * (x - median) / MAD) beyond this magnitude is
# the classic Iglewicz & Hoaglin outlier rule of thumb.
OUTLIER_Z_THRESHOLD = 3.5

# A radius outside this range is physically implausible for a DIRECT aortic
# daughter regardless of cohort spread: too small to trace reliably, or
# large enough to be a segment of the aorta itself rather than a branch.
IMPLAUSIBLE_RADIUS_MM = (0.5, 15.0)

NEAR_CAP_MM = 10.0


def _match_features_to_daughters(daughters, scored, tolerance_mm=1e-3):
    """scored holds every traced instance (features, instance dict); daughters
    holds only what build_daughters actually kept. Both were built from the
    same ostium_mm values without modification, so matching on that
    coordinate (not identity/index, since merges can drop entries) recovers
    the feature record behind each reported daughter.
    """
    lookup = []
    for entry in scored:
        point = np.asarray(entry["ostium"]["ostium_mm"])
        lookup.append((point, entry))

    matched = []
    for daughter in daughters:
        target = np.asarray(daughter["ostium_xyz_mm"])
        best = min(lookup, key=lambda pair: np.linalg.norm(pair[0] - target))
        matched.append(best[1] if np.linalg.norm(best[0] - target) <= tolerance_mm else None)
    return matched


def _modified_z_scores(values):
    values = np.asarray(values, dtype=float)
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    if mad == 0:
        return np.zeros_like(values)
    return 0.6745 * (values - median) / mad


def collect_case_metrics(data_dir, case_names=None, verbose=True):
    cases = list_real_cases(data_dir)
    if case_names:
        cases = [c for c in cases if c[0] in case_names]

    per_case = []
    all_detections = []  # flat list of {case_id, radius_mm, angle_deg, touches_cap, distance_to_cap_mm}

    for case_id, image_path, mask_path in cases:
        context = analyze_case(image_path, mask_path)
        daughters, scored = build_daughters(context)
        matched_features = _match_features_to_daughters(daughters, scored)

        aortic_length_mm = _aortic_length_mm(context)
        detections = []
        for daughter, entry in zip(daughters, matched_features):
            features = entry["features"] if entry is not None else {}
            record = {
                "case_id": case_id,
                "instance_id": daughter["instance_id"],
                "radius_mm": daughter["radius_mm"],
                "angle_to_centerline_deg": features.get("angle_to_centerline_deg"),
                "touches_cap": bool(features.get("touches_cap", 0.0) > 0.5),
                "distance_to_cap_mm": features.get("distance_to_cap_mm"),
                "confidence": daughter.get("confidence"),
            }
            detections.append(record)
            all_detections.append(record)

        per_case.append({
            "case_id": case_id,
            "n_daughters": len(daughters),
            "aortic_length_mm": aortic_length_mm,
            "detections_per_10cm": len(daughters) / (aortic_length_mm / CM_MM) if aortic_length_mm > 0 else None,
            "detections": detections,
        })
        if verbose:
            print(f"{case_id:12s} n={len(daughters):2d} aortic_length={aortic_length_mm:6.1f}mm", file=sys.stderr)

    return per_case, all_detections


def analyze_plausibility(per_case, all_detections):
    densities = [c["detections_per_10cm"] for c in per_case if c["detections_per_10cm"] is not None]
    density_z = dict(zip(
        [c["case_id"] for c in per_case if c["detections_per_10cm"] is not None],
        _modified_z_scores(densities),
    ))

    case_flags = []
    for case in per_case:
        z = density_z.get(case["case_id"])
        if z is not None and abs(z) > OUTLIER_Z_THRESHOLD:
            case_flags.append({
                "case_id": case["case_id"], "reason": "branch_density_outlier",
                "detections_per_10cm": case["detections_per_10cm"], "modified_z": float(z),
            })
        if case["n_daughters"] == 0:
            case_flags.append({"case_id": case["case_id"], "reason": "zero_detections", "detail": None})

    radii = [d["radius_mm"] for d in all_detections]
    angles = [d["angle_to_centerline_deg"] for d in all_detections if d["angle_to_centerline_deg"] is not None]
    radius_z = dict(zip(range(len(radii)), _modified_z_scores(radii))) if radii else {}

    detection_flags = []
    for i, d in enumerate(all_detections):
        r = d["radius_mm"]
        if not (IMPLAUSIBLE_RADIUS_MM[0] <= r <= IMPLAUSIBLE_RADIUS_MM[1]):
            detection_flags.append({
                "case_id": d["case_id"], "instance_id": d["instance_id"], "reason": "implausible_radius",
                "radius_mm": r,
            })
        elif i in radius_z and abs(radius_z[i]) > OUTLIER_Z_THRESHOLD:
            detection_flags.append({
                "case_id": d["case_id"], "instance_id": d["instance_id"], "reason": "radius_outlier_in_cohort",
                "radius_mm": r, "modified_z": float(radius_z[i]),
            })

    n_total = len(all_detections)
    n_near_cap = sum(1 for d in all_detections if d["touches_cap"])

    summary = {
        "n_cases": len(per_case),
        "n_total_detections": n_total,
        "detections_per_10cm": {
            "mean": float(np.mean(densities)) if densities else None,
            "median": float(np.median(densities)) if densities else None,
            "min": float(np.min(densities)) if densities else None,
            "max": float(np.max(densities)) if densities else None,
        },
        "radius_mm": {
            "mean": float(np.mean(radii)) if radii else None,
            "median": float(np.median(radii)) if radii else None,
            "p05": float(np.percentile(radii, 5)) if radii else None,
            "p95": float(np.percentile(radii, 95)) if radii else None,
        },
        "angle_to_centerline_deg": {
            "mean": float(np.mean(angles)) if angles else None,
            "median": float(np.median(angles)) if angles else None,
            "p05": float(np.percentile(angles, 5)) if angles else None,
            "p95": float(np.percentile(angles, 95)) if angles else None,
        },
        "fraction_touching_cap": n_near_cap / n_total if n_total else None,
    }

    return summary, case_flags, detection_flags


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=None, help="Path to the TORALIS CHALLENGE dev-set folder.")
    parser.add_argument("--cases", nargs="*", default=None, help="Restrict to these case ids (e.g. subject001).")
    parser.add_argument("--report", default="plausibility_report.json")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    data_dir = find_real_data_dir(args.data_dir)
    if data_dir is None:
        print("No TORALIS CHALLENGE data folder found; nothing to do.", file=sys.stderr)
        return 1

    per_case, all_detections = collect_case_metrics(data_dir, case_names=args.cases)
    summary, case_flags, detection_flags = analyze_plausibility(per_case, all_detections)

    report = {
        "summary": summary, "case_flags": case_flags, "detection_flags": detection_flags, "cases": per_case,
    }
    with open(args.report, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print()
    print("=== cohort summary ===")
    print(json.dumps(summary, indent=2))

    print()
    print(f"=== case flags ({len(case_flags)}) ===")
    for flag in case_flags:
        print(f"  {flag}")

    print()
    print(f"=== detection flags ({len(detection_flags)}) ===")
    for flag in detection_flags:
        print(f"  {flag}")

    print()
    print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
