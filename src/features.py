"""Flatten one traced instance's pipeline output into a numeric feature vector.

This is the interface src/classify.py (Session 4, not yet built) trains and
predicts on, and it is also the format scripts/review_candidates.py writes
its labelled dataset in. It lives in its own module, imported by both,
rather than inline in either -- so a reviewer's feature vector and a
classifier's training vector are, by construction, never allowed to drift
apart.

Every value here comes from src/candidates.py, src/tracing.py or
src/parentage.py; nothing is computed fresh. Optional fields (e.g. a trace
that never bifurcated) are filled with an explicit neutral default rather
than left as None or NaN, since most classifiers cannot train on missing
values -- see the inline comments for what "neutral" means for each one.

Written against the flood-fill detector: the unit of review is one
*instance* from src.parentage.split_into_instances, which already carries
its own parentage (is_independent_origin / shares_vessel_with /
vessel_join_mm) rather than it arriving as a separate per-candidate record.
"""

import numpy as np

FEATURE_NAMES = [
    # --- the instance itself: how much vessel the flood actually found
    "n_patch_voxels",
    "patch_area_mm2",
    "volume_mm3",
    "max_distance_mm",
    "radius_estimate_mm",
    "neck_mm",
    "patch_peak_mm",
    "vesselness_median_response",
    "vesselness_median_radius_mm",
    # --- end-cap evidence: is this the aorta's own cut end rather than a branch
    "touches_cap",
    "cap_face_fraction",
    # --- how the instance was carved out of its component
    "was_split",
    "absorbed_narrow_pieces",
    # --- trace
    "traced_length_mm",
    "truncation_mm",
    "truncated_by_max_length",
    "truncated_by_bifurcation",
    "truncated_by_vessel_end",
    # --- bifurcation
    "bifurcation_present",
    "bifurcation_distance_mm",
    "bifurcation_n_blobs",
    "bifurcation_slab_mm",
    # --- how much the independent ostium estimators disagreed
    "n_ostium_estimates",
    "ostium_max_disagreement_mm",
    "ostium_mean_disagreement_mm",
    "ostium_opening_arc_mm",
    # --- radius, and how much its independent estimators disagreed
    "radius_mm",
    "radius_from_distance_transform_mm",
    "radius_from_frontier_mm",
    "radius_from_vesselness_mm",
    "radius_disagreement",
    "radius_disagreement_flag",
    "cross_section_area_mm2",
    # --- seed placement
    "seed_arc_mm",
    "reached_seed_distance",
    "seed_recentred",
    "seed_shift_from_recentring_mm",
    # --- parentage: is this its own vessel, or a piece of a neighbour's
    "is_independent_origin",
    "n_shares_vessel_with",
    "min_vessel_join_mm",
]

# A vessel that never joins another has no join distance at all. Zero would
# read as "joins immediately", the exact opposite, so an out-of-range large
# value stands in for "no such join".
NO_JOIN_MM = 999.0


def _num(value, default=0.0):
    """float(value), with a fallback for the fields that can be None."""
    if value is None:
        return float(default)
    return float(value)


def _flag(value):
    return float(bool(value))


def extract_features(instance, trace, bifurcation, ostium_estimate, seed_estimate):
    """One flat, JSON-safe dict of numeric features for one traced instance.

    Arguments are exactly the per-instance outputs run.py.analyze_volumes
    already aligns by index: a src.parentage.split_into_instances entry, its
    src.tracing.trace_branch trace, the src.tracing bifurcation result that
    trace was truncated against, a src.tracing.estimate_ostium_candidates
    result, and a src.tracing.extract_seed_direction_radius result.
    """
    vesselness = instance.get("vesselness") or {}
    truncated_by = trace.get("truncated_by")

    joins = instance.get("vessel_join_mm") or {}
    shares = instance.get("shares_vessel_with") or []

    seed_shift = 0.0
    if instance is not None and seed_estimate.get("seed_before_recentring_mm") is not None:
        seed_shift = float(np.linalg.norm(
            np.asarray(seed_estimate["seed_mm"], dtype=float)
            - np.asarray(seed_estimate["seed_before_recentring_mm"], dtype=float)))

    features = {
        "n_patch_voxels": _num(instance.get("n_patch_voxels")),
        "patch_area_mm2": _num(instance.get("patch_area_mm2")),
        "volume_mm3": _num(instance.get("volume_mm3")),
        "max_distance_mm": _num(instance.get("max_distance_mm")),
        "radius_estimate_mm": _num(instance.get("radius_estimate_mm")),
        "neck_mm": _num(instance.get("neck_mm")),
        "patch_peak_mm": _num(instance.get("patch_peak_mm")),
        "vesselness_median_response": _num(vesselness.get("median_response")),
        "vesselness_median_radius_mm": _num(vesselness.get("median_radius_mm")),

        "touches_cap": _flag(instance.get("touches_cap")),
        "cap_face_fraction": _num(instance.get("cap_face_fraction")),

        # "split" is a string describing how the component was divided;
        # "none" means this instance is the whole component.
        "was_split": _flag(instance.get("split") not in (None, "none")),
        "absorbed_narrow_pieces": _num(instance.get("absorbed_narrow_pieces")),

        "traced_length_mm": _num(trace.get("traced_length_mm")),
        "truncation_mm": _num(trace.get("truncation_mm")),
        # One-hot rather than an arbitrary integer code: these three reasons
        # are unordered, and a classifier must not read "vessel_end" as
        # being twice "max_length".
        "truncated_by_max_length": _flag(truncated_by == "max_length"),
        "truncated_by_bifurcation": _flag(truncated_by == "bifurcation"),
        "truncated_by_vessel_end": _flag(truncated_by == "vessel_end"),

        # n_blobs is 1 for a vessel that simply continued; a bifurcation is
        # two or more blobs in the slab.
        "bifurcation_present": _flag((bifurcation.get("n_blobs") or 0) > 1),
        # Distance is None when nothing bifurcated. Neutral is the trace's
        # own length: "no bifurcation found within the traced extent".
        "bifurcation_distance_mm": _num(bifurcation.get("distance_mm"),
                                        default=_num(trace.get("traced_length_mm"))),
        "bifurcation_n_blobs": _num(bifurcation.get("n_blobs")),
        "bifurcation_slab_mm": _num(bifurcation.get("slab_mm")),

        "n_ostium_estimates": _num(len(ostium_estimate.get("estimates") or ())),
        "ostium_max_disagreement_mm": _num(ostium_estimate.get("max_disagreement_mm")),
        "ostium_mean_disagreement_mm": _num(ostium_estimate.get("mean_disagreement_mm")),
        "ostium_opening_arc_mm": _num(ostium_estimate.get("opening_arc_mm")),

        "radius_mm": _num(seed_estimate.get("radius_mm")),
        "radius_from_distance_transform_mm": _num(
            seed_estimate.get("radius_from_distance_transform_mm")),
        "radius_from_frontier_mm": _num(seed_estimate.get("radius_from_frontier_mm")),
        "radius_from_vesselness_mm": _num(seed_estimate.get("radius_from_vesselness_mm")),
        "radius_disagreement": _num(seed_estimate.get("radius_disagreement")),
        "radius_disagreement_flag": _flag(seed_estimate.get("radius_disagreement_flag")),
        "cross_section_area_mm2": _num(seed_estimate.get("cross_section_area_mm2")),

        "seed_arc_mm": _num(seed_estimate.get("seed_arc_mm")),
        "reached_seed_distance": _flag(seed_estimate.get("reached_seed_distance")),
        "seed_recentred": _flag(seed_estimate.get("recentred")),
        "seed_shift_from_recentring_mm": seed_shift,

        "is_independent_origin": _flag(instance.get("is_independent_origin")),
        "n_shares_vessel_with": _num(len(shares)),
        "min_vessel_join_mm": _num(min(joins.values()) if joins else None,
                                   default=NO_JOIN_MM),
    }

    missing = set(FEATURE_NAMES) - set(features)
    extra = set(features) - set(FEATURE_NAMES)
    if missing or extra:
        raise AssertionError(
            f"feature vector does not match FEATURE_NAMES (missing={sorted(missing)}, "
            f"extra={sorted(extra)})")
    return features


def feature_vector(features):
    """The dict from extract_features as a plain list, in FEATURE_NAMES order."""
    return [float(features[name]) for name in FEATURE_NAMES]
