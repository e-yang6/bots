"""Robustness of the pipeline's own detections under parameter and noise
perturbation, on the real dev-set cases -- with no reference labels needed.

There is no ground truth for these cases, so this cannot say whether a
detection is CORRECT. What it can say is whether a detection is STABLE: does
it survive a reasonable +/-10% / +/-20% nudge to a tunable parameter, or a
plausible amount of added image noise, or does it appear, disappear, or move
by more than 2mm under that nudge. A detection that only exists at one exact
parameter setting is not trustworthy even without knowing the "right" answer;
one that survives every perturbation is trustworthy in the sense that matters
for a system with no labels to check against. See README.md for how this
sits alongside scripts/validate_phantom.py (which DOES have ground truth, but
only on synthetic geometry).

Four tunables are perturbed, all exposed by run.py/src.rules without any
change to pipeline logic: flood_budget_mm and flood_fraction_range (both
src.floodfill.robust_flood), min_length_mm (src.candidates.find_candidates /
src.parentage.split_into_instances), and confidence_threshold
(run.build_daughters, backed by src.rules.CONFIDENCE_THRESHOLD). Each is
perturbed independently (one variable changes at a time) by -20%, -10%,
+10%, +20% of its own default. Gaussian noise is added at a few HU sigma
levels on top of that, as a separate axis, on the already-cropped/resampled
volume so every run within one case shares the same geometry.

A baseline run's daughters are matched against each perturbed run's daughters
by nearest ostium (src.evaluate.match_and_score, reused here as a same-case
before/after matcher rather than a predictions-vs-labels one) at
MATCH_RADIUS_MM. A match distance over MOVE_THRESHOLD_MM counts as "moved";
no match at all counts as "disappeared" (baseline daughter) or "appeared"
(perturbed daughter with no baseline counterpart).

Usage:
    python -m scripts.stability [--data-dir PATH] [--cases subject001 ...]
                                [--report stability_report.json]
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run import analyze_volumes, build_daughters  # noqa: E402
from scripts.inspect_pipeline import find_real_data_dir, list_real_cases  # noqa: E402
from src.candidates import MIN_ELIGIBLE_LENGTH_MM  # noqa: E402
from src.evaluate import match_and_score  # noqa: E402
from src.floodfill import DEFAULT_BUDGET_MM, THRESHOLD_FRACTION_RANGE  # noqa: E402
from src.geometry import crop_to_mask_bbox, resample_isotropic  # noqa: E402
from src.io_utils import load_case  # noqa: E402
from src.rules import CONFIDENCE_THRESHOLD  # noqa: E402

PERTURBATION_FRACTIONS = (-0.20, -0.10, 0.10, 0.20)

# A same-case before/after match is a much stronger claim than a phantom
# prediction-vs-label match (evaluate.py's default 10mm): only one variable
# moved, so the same real ostium should barely shift. Looser than
# MOVE_THRESHOLD_MM so a "moved but still basically the same vessel" case
# still counts as matched (and gets flagged moved) rather than counted as a
# disappearance-plus-appearance pair.
MATCH_RADIUS_MM = 5.0
MOVE_THRESHOLD_MM = 2.0

# A few plausible HU noise levels: this cohort's lumen HU spans roughly
# 90-580 (src.floodfill's module docstring), so 15/30/50 HU sigma covers
# "mild scanner noise" through "noticeably grainy" without ever being large
# enough to swamp the lowest-HU cases entirely.
NOISE_SIGMA_HU_LEVELS = (15.0, 30.0, 50.0)

NOISE_RNG_SEED = 0


def _param_perturbation_runs():
    """(name, flood_kwargs, build_kwargs) for each parameter x fraction."""
    runs = []
    for frac in PERTURBATION_FRACTIONS:
        runs.append((
            f"flood_budget_mm{frac:+.0%}",
            {"flood_budget_mm": DEFAULT_BUDGET_MM * (1.0 + frac)}, {}, {},
        ))
    lo, hi = THRESHOLD_FRACTION_RANGE
    for frac in PERTURBATION_FRACTIONS:
        new_range = (
            float(np.clip(lo * (1.0 + frac), 0.05, 0.95)),
            float(np.clip(hi * (1.0 + frac), 0.05, 0.95)),
        )
        runs.append((f"flood_fraction_range{frac:+.0%}", {"flood_fraction_range": new_range}, {}, {}))
    for frac in PERTURBATION_FRACTIONS:
        runs.append((
            f"min_length_mm{frac:+.0%}",
            {}, {"min_length_mm": MIN_ELIGIBLE_LENGTH_MM * (1.0 + frac)}, {},
        ))
    for frac in PERTURBATION_FRACTIONS:
        runs.append((
            f"confidence_threshold{frac:+.0%}",
            {}, {}, {"threshold": CONFIDENCE_THRESHOLD * (1.0 + frac)},
        ))
    return runs


def _add_gaussian_noise(image, sigma_hu, rng):
    arr = sitk.GetArrayFromImage(image).astype(np.float32)
    noisy = arr + rng.normal(0.0, sigma_hu, size=arr.shape).astype(np.float32)
    out = sitk.GetImageFromArray(noisy)
    out.CopyInformation(image)
    return out


def _run_variant(resampled_image, resampled_mask, original_grid, flood_kwargs, candidate_kwargs, build_kwargs):
    context = analyze_volumes(
        resampled_image, resampled_mask, original_grid=original_grid,
        flood_budget_mm=flood_kwargs.get("flood_budget_mm"),
        flood_fraction_range=flood_kwargs.get("flood_fraction_range"),
        min_length_mm=candidate_kwargs.get("min_length_mm"),
    )
    daughters, _scored = build_daughters(context, **build_kwargs)
    return daughters


def _compare(baseline_daughters, perturbed_daughters):
    """One perturbation's effect on one case: which baseline detections
    survived (and whether they moved), and how many new ones showed up.
    """
    score = match_and_score(perturbed_daughters, baseline_daughters, distance_threshold_mm=MATCH_RADIUS_MM)
    matched_baseline = {j: d for _, j, d in score["matches"]}
    per_baseline = []
    for j in range(len(baseline_daughters)):
        if j in matched_baseline:
            distance = matched_baseline[j]
            per_baseline.append({"present": True, "moved_mm": distance, "moved": distance > MOVE_THRESHOLD_MM})
        else:
            per_baseline.append({"present": False, "moved_mm": None, "moved": False})
    n_appeared = score["false_positives"]  # perturbed daughters with no baseline match
    return per_baseline, n_appeared


def run_stability(data_dir, case_names=None, verbose=True):
    cases = list_real_cases(data_dir)
    if case_names:
        cases = [c for c in cases if c[0] in case_names]

    param_runs = _param_perturbation_runs()
    rng = np.random.default_rng(NOISE_RNG_SEED)

    report_cases = []
    for case_id, image_path, mask_path in cases:
        t0 = time.time()
        image, mask = load_case(image_path, mask_path)
        cropped_image, cropped_mask = crop_to_mask_bbox(image, mask, margin_mm=15)
        resampled_image, resampled_mask, original_grid = resample_isotropic(cropped_image, cropped_mask)

        baseline_context = analyze_volumes(resampled_image, resampled_mask, original_grid=original_grid)
        baseline_daughters, _ = build_daughters(baseline_context)

        n_baseline = len(baseline_daughters)
        per_detection_events = [[] for _ in range(n_baseline)]
        appearance_events = []

        for name, flood_kwargs, candidate_kwargs, build_kwargs in param_runs:
            perturbed = _run_variant(resampled_image, resampled_mask, original_grid,
                                     flood_kwargs, candidate_kwargs, build_kwargs)
            per_baseline, n_appeared = _compare(baseline_daughters, perturbed)
            for j, event in enumerate(per_baseline):
                per_detection_events[j].append({"perturbation": name, **event})
            if n_appeared:
                appearance_events.append({"perturbation": name, "n_appeared": n_appeared})

        for sigma in NOISE_SIGMA_HU_LEVELS:
            noisy_image = _add_gaussian_noise(resampled_image, sigma, rng)
            noisy_context = analyze_volumes(noisy_image, resampled_mask, original_grid=original_grid)
            perturbed, _ = build_daughters(noisy_context)
            name = f"gaussian_noise_sigma{sigma:.0f}hu"
            per_baseline, n_appeared = _compare(baseline_daughters, perturbed)
            for j, event in enumerate(per_baseline):
                per_detection_events[j].append({"perturbation": name, **event})
            if n_appeared:
                appearance_events.append({"perturbation": name, "n_appeared": n_appeared})

        n_perturbations = len(param_runs) + len(NOISE_SIGMA_HU_LEVELS)
        detections = []
        for j, daughter in enumerate(baseline_daughters):
            events = per_detection_events[j]
            n_stable = sum(1 for e in events if e["present"] and not e["moved"])
            stability = n_stable / n_perturbations if n_perturbations else 1.0
            disappeared_in = [e["perturbation"] for e in events if not e["present"]]
            moved_in = [e["perturbation"] for e in events if e["present"] and e["moved"]]
            detections.append({
                "instance_id": daughter["instance_id"],
                "ostium_xyz_mm": daughter["ostium_xyz_mm"],
                "stability": round(float(stability), 3),
                "disappeared_in": disappeared_in,
                "moved_in": moved_in,
            })

        case_stability = float(np.mean([d["stability"] for d in detections])) if detections else None
        spurious_rate = (
            len(appearance_events) / n_perturbations if n_perturbations and n_baseline == 0 else None
        )

        elapsed = time.time() - t0
        report_cases.append({
            "case_id": case_id,
            "n_baseline_daughters": n_baseline,
            "case_stability_score": case_stability,
            "spurious_appearance_rate_when_empty": spurious_rate,
            "detections": detections,
            "appearance_events": appearance_events,
            "elapsed_s": elapsed,
        })
        if verbose:
            score_str = f"{case_stability:.2f}" if case_stability is not None else "n/a (0 baseline)"
            print(f"{case_id:12s} n_daughters={n_baseline:2d} case_stability={score_str} "
                  f"appearances={len(appearance_events):2d} [{elapsed:.1f}s]", file=sys.stderr)

    scored_cases = [c for c in report_cases if c["case_stability_score"] is not None]
    overall = {
        "n_cases": len(report_cases),
        "n_cases_with_detections": len(scored_cases),
        "mean_case_stability_score": (
            float(np.mean([c["case_stability_score"] for c in scored_cases])) if scored_cases else None
        ),
        "n_perturbations_per_case": len(param_runs) + len(NOISE_SIGMA_HU_LEVELS),
    }
    return {"overall": overall, "cases": report_cases}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=None, help="Path to the TORALIS CHALLENGE dev-set folder.")
    parser.add_argument("--cases", nargs="*", default=None, help="Restrict to these case ids (e.g. subject001).")
    parser.add_argument("--report", default="stability_report.json")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    data_dir = find_real_data_dir(args.data_dir)
    if data_dir is None:
        print("No TORALIS CHALLENGE data folder found; nothing to do.", file=sys.stderr)
        return 1

    report = run_stability(data_dir, case_names=args.cases)
    with open(args.report, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print()
    print("=== overall ===")
    for key, value in report["overall"].items():
        print(f"  {key}: {value}")
    print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
