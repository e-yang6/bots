"""CLI entry point for the Branchseed daughter-artery detector.

Usage:
    python run.py --image image.nii.gz --aorta-mask aorta_mask.nii.gz --output prediction.json

Detection is wired up but deliberately not finalized: analyze_case() runs the
full candidate -> trace -> parentage chain and reports what it found, while
detect_branches() still returns an empty daughters list. Turning candidates
into reported daughters needs the classifier, dedup and eligibility filtering
that the next stage adds. The whole pipeline stays wrapped so any failure
still emits valid, schema-conformant JSON rather than crashing or producing
no output at all.
"""

import argparse
import json
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

import schema
from src.candidates import build_evidence, find_candidate_ostia
from src.geometry import (
    compute_centerline,
    compute_surface_normals,
    crop_to_mask_bbox,
    flag_end_caps,
    resample_isotropic,
)
from src.intensity import is_contrast_enhanced, lumen_stats
from src.io_utils import load_case
from src.parentage import check_branch_of_branch
from src.tracing import (
    detect_bifurcation,
    estimate_ostium_candidates,
    extract_seed_direction_radius,
    trace_branch,
    truncate_trace,
)

# Tracing is the expensive step, so only the strongest candidates are traced.
# Candidate scoring is deliberately recall-heavy at this stage; ranking and
# filtering properly is the next stage's job.
MAX_CANDIDATES_TO_TRACE = 40


def analyze_case(image_path, aorta_mask_path, target_spacing=0.8, max_candidates=MAX_CANDIDATES_TO_TRACE,
                 verbose=False):
    """Run the full detection chain and return everything it found.

    Returns a context dict (also used by scripts/visualize_candidates.py).
    No filtering or output formatting happens here.
    """
    timings = {}

    started = time.time()
    image, mask = load_case(image_path, aorta_mask_path)
    cropped_image, cropped_mask = crop_to_mask_bbox(image, mask, margin_mm=15)
    resampled_image, resampled_mask, original_grid = resample_isotropic(
        cropped_image, cropped_mask, target_spacing=target_spacing
    )
    timings["load_and_prepare_s"] = time.time() - started

    started = time.time()
    stats = lumen_stats(resampled_image, resampled_mask)
    contrast_enhanced = is_contrast_enhanced(stats)
    centerline = compute_centerline(resampled_mask)
    surface = compute_surface_normals(resampled_mask)
    caps = flag_end_caps(resampled_mask, surface, centerline)
    timings["geometry_s"] = time.time() - started

    context = {
        "image": resampled_image,
        "mask": resampled_mask,
        "original_grid": original_grid,
        "lumen_stats": stats,
        "contrast_enhanced": contrast_enhanced,
        "centerline": centerline,
        "surface": surface,
        "caps": caps,
        "candidates": [],
        "traces": [],
        "ostium_estimates": [],
        "seed_estimates": [],
        "bifurcations": [],
        "parentage": [],
        "timings": timings,
    }

    if not contrast_enhanced:
        # Nothing downstream is meaningful without a contrast-filled lumen.
        context["short_circuit_reason"] = "not_contrast_enhanced"
        return context

    started = time.time()
    evidence = build_evidence(resampled_image, resampled_mask, stats)
    candidates, evidence = find_candidate_ostia(
        resampled_image, resampled_mask, stats, surface, caps, evidence=evidence
    )
    timings["candidates_s"] = time.time() - started
    context["evidence"] = evidence
    context["candidates"] = candidates

    sampler = evidence["sampler"]
    lumen_threshold = evidence["lumen_threshold"]
    surface_points_mm = surface[0]
    surface_tree = cKDTree(surface_points_mm) if surface_points_mm.shape[0] else None

    started = time.time()
    to_trace = candidates[:max_candidates]
    for candidate in to_trace:
        trace = trace_branch(
            sampler,
            candidate["ostium_patch_centroid_mm"],
            candidate["outward_direction"],
            lumen_threshold,
        )
        bifurcation = detect_bifurcation(trace)
        if bifurcation["bifurcation_index"] is not None:
            trace = truncate_trace(trace, bifurcation["bifurcation_index"])

        ostium_estimate = estimate_ostium_candidates(
            sampler, candidate, trace, lumen_threshold, surface_tree, surface_points_mm
        )
        seed_estimate = extract_seed_direction_radius(
            sampler, trace, ostium_estimate["consensus_mm"], lumen_threshold
        )

        context["traces"].append(trace)
        context["bifurcations"].append(bifurcation)
        context["ostium_estimates"].append(ostium_estimate)
        context["seed_estimates"].append(seed_estimate)
    timings["tracing_s"] = time.time() - started

    started = time.time()
    context["parentage"] = check_branch_of_branch(
        to_trace,
        context["traces"],
        surface_tree,
        ostium_points_mm=[e["consensus_mm"] for e in context["ostium_estimates"]],
    )
    timings["parentage_s"] = time.time() - started

    if verbose:
        log_analysis(context, file=sys.stderr)

    return context


def log_analysis(context, file=sys.stderr):
    """Print the intermediate results -- this stage's actual product."""
    stats = context["lumen_stats"]
    print(
        f"lumen HU median={stats['median']:.0f} p25={stats['p25']:.0f} p75={stats['p75']:.0f} "
        f"contrast_enhanced={context['contrast_enhanced']}",
        file=file,
    )
    print(
        f"components kept={len(context['centerline']['kept'])} "
        f"dropped={len(context['centerline']['dropped'])} caps={len(context['caps'])} "
        f"surface_points={context['surface'][0].shape[0]}",
        file=file,
    )
    if "short_circuit_reason" in context:
        print(f"short-circuited: {context['short_circuit_reason']}", file=file)
        return

    evidence = context.get("evidence")
    if evidence is not None:
        print(
            f"contrast: lumen_ref={evidence['lumen_reference_hu']:.0f}HU "
            f"background={evidence['background_reference_hu']:.0f}HU "
            f"noise={evidence['background_noise_hu']:.0f}HU "
            f"noise_ratio={evidence['noise_ratio']:.2f} "
            f"band_tightening={evidence['band_tightening']:.2f} "
            f"band_low={evidence['lumen_band'][0]:.2f}/{evidence['lumen_band'][1]:.2f}",
            file=file,
        )

    independent = sum(1 for p in context["parentage"] if p["is_independent_origin"])
    shared = sum(1 for p in context["parentage"] if p["shares_vessel_with"] is not None)
    print(
        f"candidates={len(context['candidates'])} traced={len(context['traces'])} "
        f"independent_origins={independent} sharing_another_vessel={shared}",
        file=file,
    )
    for index, trace in enumerate(context["traces"]):
        candidate = context["candidates"][index]
        seed = context["seed_estimates"][index]
        ostium = context["ostium_estimates"][index]
        parent = context["parentage"][index]
        consensus = ostium["consensus_mm"]
        print(
            f"  #{index:2d} score={candidate['peak_score']:.2f} "
            f"traced={trace['traced_length_mm']:.1f}mm ({trace['status']}) "
            f"ostium=({consensus[0]:.1f},{consensus[1]:.1f},{consensus[2]:.1f}) "
            f"disagree={ostium['max_disagreement_mm']:.1f}mm "
            f"r={seed['radius_mm']:.2f}mm (dt={seed['radius_from_distance_transform_mm']:.2f} "
            f"ellipse={seed['radius_from_ellipse_mm']:.2f}"
            f"{' MISMATCH' if seed['radius_disagreement_flag'] else ''}) "
            f"d_cap={candidate['distance_to_cap_mm']:.1f}mm "
            f"shell={candidate['shell_volume_mm3']:.0f}mm3"
            f"{' BRANCH_OF_BRANCH#%d' % parent['parent_candidate_index'] if parent['is_branch_of_branch'] else ''}"
            f"{' SHARES_VESSEL#%d(%.2f)' % (parent['shares_vessel_with'], parent['containment_fraction']) if parent['shares_vessel_with'] is not None else ''}",
            file=file,
        )
    timings = context["timings"]
    print("timings: " + " ".join(f"{k}={v:.1f}s" for k, v in timings.items()), file=file)


def detect_branches(image_path, aorta_mask_path, verbose=False):
    """Returns the daughter dicts to report.

    Still an empty list: candidate detection and tracing run (and are logged),
    but turning them into reported daughters requires the classification,
    deduplication and eligibility filtering of the next stage.
    """
    analyze_case(image_path, aorta_mask_path, verbose=verbose)
    return []


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Detect arteries branching off the aorta from a CT scan and aorta mask."
    )
    parser.add_argument("--image", required=True, help="Path to the CT scan (NIfTI).")
    parser.add_argument("--aorta-mask", required=True, help="Path to the aorta-only mask (NIfTI).")
    parser.add_argument("--output", required=True, help="Path to write the output prediction JSON.")
    parser.add_argument(
        "--verbose", action="store_true", help="Log intermediate detection results to stderr."
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    case_id = "unknown"
    daughters = []
    try:
        case_id = schema.case_id_from_path(args.image)
        daughters = detect_branches(args.image, args.aorta_mask, verbose=args.verbose)
    except Exception as exc:
        print(f"warning: pipeline failed, emitting empty daughters list ({exc})", file=sys.stderr)

    prediction = schema.make_prediction(case_id, daughters)

    try:
        schema.write_prediction(prediction, args.output)
    except Exception as exc:
        print(f"error: failed to write output via schema.write_prediction ({exc})", file=sys.stderr)
        fallback = {
            "case_id": case_id,
            "parent": {"instance_id": schema.PARENT_INSTANCE_ID},
            "daughters": [],
        }
        with open(args.output, "w") as f:
            json.dump(fallback, f, indent=2)


if __name__ == "__main__":
    main()
