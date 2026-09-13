"""CLI entry point for the Branchseed daughter-artery detector.

Usage:
    python run.py --image image.nii.gz --aorta-mask aorta_mask.nii.gz --output prediction.json

analyze_case() runs the full flood -> candidates -> origins -> trace chain.
detect_branches() finishes it: every traced instance gets a feature vector
(src/features.py) and a transparent, rule-based confidence (src/rules.py --
no trained model, no labels to train one on); instances at or above
rules.CONFIDENCE_THRESHOLD are reported, with one further check that the
flood/split machinery cannot make on its own (build_daughters' docstring
explains why the usual eligibility rules are otherwise already enforced
upstream, by construction of this pipeline). The whole pipeline stays
wrapped so any failure still emits valid, schema-conformant JSON rather
than crashing or producing no output at all.
"""

import argparse
import json
import sys
import time

import numpy as np
import SimpleITK as sitk
from scipy.spatial import cKDTree

import schema
from src.candidates import find_candidates, log_candidates
from src.features import build_feature_matrix
from src.floodfill import log_flood, robust_flood
from src.geometry import (
    compute_centerline,
    compute_surface_normals,
    crop_to_mask_bbox,
    flag_end_caps,
    resample_isotropic,
)
from src.intensity import is_contrast_enhanced, lumen_stats
from src.io_utils import load_case
from src.lumen_evidence import VolumeSampler
from src.parentage import log_instances, split_into_instances
from src.rules import CONFIDENCE_THRESHOLD, explain
from src.tracing import (
    detect_bifurcation_by_frontier,
    estimate_ostium_candidates,
    extract_seed_direction_radius,
    trace_branch,
)

# A guard against a leaking flood shattering into hundreds of eligible
# fragments; on a clean case there are a few dozen instances at most.
MAX_INSTANCES_TO_TRACE = 40

# Two instances from DIFFERENT flood components (so src.parentage's own
# watershed/absorption logic, which only ever operates within one
# component, never had a chance to weigh in) whose final ostia land closer
# than this are worth a second look, per the brief's "merge ostia within
# ~3mm unless contact patches are genuinely disjoint".
OSTIUM_MERGE_DISTANCE_MM = 3.0

# How close two instances' contact patches (not just their ostium points)
# have to come before they count as touching rather than genuinely
# disjoint. Comfortably under a voxel diagonal at the 0.8mm working
# resolution, so this never fires on patches that only look close because
# their centroids happen to line up.
PATCH_DISJOINT_TOLERANCE_MM = 1.5


def analyze_case(image_path, aorta_mask_path, target_spacing=0.8, max_instances=MAX_INSTANCES_TO_TRACE,
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

    return analyze_volumes(
        resampled_image, resampled_mask, original_grid=original_grid,
        max_instances=max_instances, verbose=verbose, timings=timings,
    )


def analyze_volumes(resampled_image, resampled_mask, original_grid=None, max_instances=MAX_INSTANCES_TO_TRACE,
                    verbose=False, timings=None):
    """The detection chain on an already cropped and resampled image/mask pair.

    Split out of analyze_case so the synthetic phantoms in tests/ run exactly
    the chain real cases do.
    """
    timings = {} if timings is None else timings
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
        "flood": None,
        "detection": None,
        "candidates": [],
        "instances": [],
        "instance_summary": None,
        "bifurcations": [],
        "traces": [],
        "ostium_estimates": [],
        "seed_estimates": [],
        "timings": timings,
    }

    if not contrast_enhanced:
        # Nothing downstream is meaningful without a contrast-filled lumen.
        context["short_circuit_reason"] = "not_contrast_enhanced"
        if verbose:
            log_analysis(context, file=sys.stderr)
        return context

    started = time.time()
    flood = robust_flood(resampled_image, resampled_mask, stats)
    timings["flood_s"] = time.time() - started
    context["flood"] = flood

    started = time.time()
    detection = find_candidates(resampled_image, resampled_mask, stats, centerline, caps, flood=flood)
    timings["candidates_s"] = time.time() - started
    context["detection"] = detection
    context["candidates"] = detection["candidates"]

    started = time.time()
    instances, instance_summary = split_into_instances(detection["candidates"], detection)
    timings["parentage_s"] = time.time() - started
    context["instances"] = instances
    context["instance_summary"] = instance_summary

    hu_arr = sitk.GetArrayFromImage(resampled_image)
    mask_arr = sitk.GetArrayFromImage(resampled_mask).astype(bool)
    sampler = VolumeSampler(resampled_image)
    sampler.add("hu", hu_arr)
    sampler.add("mask", mask_arr)
    sampler.add("traversal", flood["traversal"])
    context["sampler"] = sampler
    surface_points_mm = surface[0]
    surface_tree = cKDTree(surface_points_mm)
    spacing = resampled_image.GetSpacing()

    started = time.time()
    tracing_log = []
    for instance in instances[:max_instances]:
        bifurcation = detect_bifurcation_by_frontier(instance, flood["dist"], spacing)
        trace = trace_branch(instance, flood["dist"], flood["parent"], resampled_image, bifurcation)
        ostium = estimate_ostium_candidates(instance, trace, resampled_image, surface_points_mm, surface_tree)
        seed = extract_seed_direction_radius(
            instance, trace, ostium["ostium_mm"], resampled_image, sampler, flood, mask_arr,
            verbose=verbose,
        )
        context["bifurcations"].append(bifurcation)
        context["traces"].append(trace)
        context["ostium_estimates"].append(ostium)
        context["seed_estimates"].append(seed)
    timings["tracing_s"] = time.time() - started

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

    log_flood(context["flood"], file=file)
    log_candidates(context["detection"], file=file)
    log_instances(context["instances"], context["instance_summary"], file=file)

    for index, trace in enumerate(context["traces"]):
        instance = context["instances"][index]
        bifurcation = context["bifurcations"][index]
        ostium = context["ostium_estimates"][index]
        seed = context["seed_estimates"][index]
        point = ostium["ostium_mm"]
        vesselness = instance.get("vesselness") or {}
        r_vessel = seed["radius_from_vesselness_mm"]
        shared = ",".join(
            f"#{other}@{instance['vessel_join_mm'][other]:.1f}mm" for other in instance["shares_vessel_with"]
        )
        split = f" split={instance['split']}" if instance["split"] != "none" else ""
        print(
            f"  #{index:2d} comp={instance['component_label']} len={instance['max_distance_mm']:.1f}mm "
            f"traced={trace['traced_length_mm']:.1f}mm ({trace['truncated_by']}"
            f"{'@%.1fmm' % bifurcation['distance_mm'] if bifurcation['distance_mm'] is not None else ''}) "
            f"ostium=({point[0]:.1f},{point[1]:.1f},{point[2]:.1f}) patch={instance['patch_area_mm2']:.0f}mm2 "
            f"disagree={ostium['max_disagreement_mm']:.1f}mm "
            f"r={seed['radius_mm']:.2f}mm (dt={seed['radius_from_distance_transform_mm']:.2f} "
            f"frontier={seed['radius_from_frontier_mm']:.2f}"
            f"{' vesselness=%.2f' % r_vessel if r_vessel is not None else ''}"
            f"{' MISMATCH' if seed['radius_disagreement_flag'] else ''}) "
            f"vness={vesselness.get('median_response', 0.0):.3g}"
            f"{' seed<5mm' if not seed['reached_seed_distance'] else ''}"
            f"{split}{' shares=' + shared if shared else ''}",
            file=file,
        )
    timings = context["timings"]
    print("timings: " + " ".join(f"{k}={v:.1f}s" for k, v in timings.items()), file=file)


def _patch_gap_mm(patch_a_mm, patch_b_mm):
    """Closest distance between two contact patches' own point sets -- not
    between their centroids or their ostium estimates, either of which can
    be close while the patches themselves never actually touch.
    """
    if patch_a_mm.shape[0] == 0 or patch_b_mm.shape[0] == 0:
        return np.inf
    tree = cKDTree(patch_b_mm)
    distances, _ = tree.query(patch_a_mm, k=1)
    return float(distances.min())


def build_daughters(context, threshold=CONFIDENCE_THRESHOLD,
                    merge_distance_mm=OSTIUM_MERGE_DISTANCE_MM,
                    patch_disjoint_mm=PATCH_DISJOINT_TOLERANCE_MM,
                    verbose=False, file=sys.stderr):
    """Score every traced instance and decide what to actually report.

    Eligibility beyond the confidence threshold is, by construction of this
    pipeline, already enforced upstream rather than needing a separate pass
    here:
      - under 5mm: src.candidates._reject's too_short rule, then re-applied
        on the SPLIT territory by src.parentage.split_into_instances's own
        min_length_mm (a component can pass length before splitting and an
        individual origin still come up short after). src.rules also vetoes
        it directly, redundantly, as its own explicit gate.
      - a daughter-of-a-daughter: never gets a contact patch of its own in
        this design (see src/parentage.py's module docstring) -- it stays
        part of its parent's traced component and is never an instance to
        begin with.
      - two watershed-split or narrow-piece-absorbed origins on ONE
        component: already resolved by split_into_instances, which is a
        real geometric decision (disjoint patches / a watershed saddle),
        not a distance heuristic. Nothing here re-merges what it already
        kept apart, and the merge check below explicitly skips same-
        component pairs for that reason.

    What split_into_instances cannot see is two SEPARATE flood components
    (different component_label) whose ostia land within merge_distance_mm
    of each other purely by anatomical coincidence. Only there, and only if
    their contact patches actually touch (not just their ostium points), is
    the lower-confidence one folded away.

    Returns (daughters, scored) -- scored is every instance's feature/score
    record, kept for log_scoring / scripts/visualize_candidates.py.
    """
    if not context["traces"]:
        return [], []

    feature_vectors = build_feature_matrix(context)
    scored = [
        {
            "instance": context["instances"][i],
            "ostium": context["ostium_estimates"][i],
            "seed": context["seed_estimates"][i],
            "features": features,
            "score": explain(features),
        }
        for i, features in enumerate(feature_vectors)
    ]

    accepted = sorted(
        (s for s in scored if s["score"]["confidence"] >= threshold),
        key=lambda s: -s["score"]["confidence"],
    )

    kept, merges = [], []
    for candidate in accepted:
        folded_into = None
        for other in kept:
            if candidate["instance"]["component_label"] == other["instance"]["component_label"]:
                continue  # same-component splits already resolved by parentage.py
            ostium_distance = float(np.linalg.norm(
                np.asarray(candidate["ostium"]["ostium_mm"]) - np.asarray(other["ostium"]["ostium_mm"])
            ))
            if ostium_distance > merge_distance_mm:
                continue
            gap = _patch_gap_mm(candidate["instance"]["patch_mm"], other["instance"]["patch_mm"])
            if gap <= patch_disjoint_mm:
                folded_into = other
                merges.append((candidate, other, ostium_distance, gap))
                break
        if folded_into is None:
            kept.append(candidate)

    if verbose:
        log_scoring(scored, kept, merges, threshold, file=file)

    daughters = []
    for index, entry in enumerate(kept, start=1):
        daughter = schema.make_daughter(
            instance_id=f"branch_{index:03d}",
            ostium_xyz_mm=entry["ostium"]["ostium_mm"],
            seed_xyz_mm=entry["seed"]["seed_mm"],
            radius_mm=entry["seed"]["radius_mm"],
            direction_xyz=entry["seed"]["direction_xyz"],
        )
        # Carry confidence as an optional extra field -- harmless to the
        # schema (write_prediction is a plain json.dump, no strict
        # validation) and useful to a downstream planning system deciding
        # what to trust. Dropped if it ever fails to serialize.
        try:
            daughter["confidence"] = round(float(entry["score"]["confidence"]), 4)
        except (TypeError, ValueError):
            pass
        daughters.append(daughter)
    return daughters, scored


def log_scoring(scored, kept, merges, threshold, file=sys.stderr):
    """One line per scored instance, accounting for every drop: vetoed,
    below threshold, merged into a neighbour, or kept.
    """
    kept_ids = {id(entry) for entry in kept}
    merged_reason = {
        id(candidate): f"within {distance:.1f}mm of instance {other['instance']['instance_id']} "
                       f"(patch gap {gap:.1f}mm <= {PATCH_DISJOINT_TOLERANCE_MM:.1f}mm)"
        for candidate, other, distance, gap in merges
    }
    print(f"scoring: {len(scored)} instances, confidence threshold={threshold:.2f}", file=file)
    for entry in scored:
        instance, score = entry["instance"], entry["score"]
        if id(entry) in kept_ids:
            status, reason = "KEPT", ""
        elif id(entry) in merged_reason:
            status, reason = "MERGED", merged_reason[id(entry)]
        elif score["veto"] is not None:
            status, reason = "VETOED", score["veto"]
        else:
            status, reason = "BELOW_THRESHOLD", f"confidence {score['confidence']:.2f} < {threshold:.2f}"
        print(
            f"  instance={instance['instance_id']:3d} comp={instance['component_label']} "
            f"confidence={score['confidence']:.2f} status={status}"
            f"{' (' + reason + ')' if reason else ''}",
            file=file,
        )
    print(f"kept {len(kept)}/{len(scored)} as daughters", file=file)


def detect_branches(image_path, aorta_mask_path, threshold=CONFIDENCE_THRESHOLD, verbose=False):
    """Returns the daughter dicts to report: the full flood -> candidates ->
    origins -> trace -> features -> rules chain, thresholded and deduped.
    """
    context = analyze_case(image_path, aorta_mask_path, verbose=verbose)
    daughters, _scored = build_daughters(context, threshold=threshold, verbose=verbose)
    return daughters


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
