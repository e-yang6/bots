"""Transparent, deterministic scoring of one traced instance's feature
vector into a confidence in [0, 1].

No trained model, no pickled weights, no training entry point: every weight
and cutoff below is a named module constant with a one-line reason, and
explain(features) breaks a score down term by term so a rejected candidate
can always be inspected by hand against the visual checks. There are no
labels a fitted model could be trusted on, and a model fit to a handful of
cases would be less trustworthy than these rules, not more.
"""

import numpy as np

# ---------------------------------------------------------------- vetoes
# Hard rejects, checked before any scoring. MIN_GEODESIC_LENGTH_MM and the
# cap vetoes mirror src.candidates' own _reject rules (too_short,
# aortic_continuation) and are therefore normally redundant: an instance
# whose parent candidate failed them never reaches find_candidates' survivor
# list, let alone this function. They are kept here anyway as an explicit,
# inspectable gate at the scoring layer itself, and as a safety net if a
# future change ever lets features reach score_candidate without having
# passed through candidates.py first. MIN_RADIUS_MM has no such upstream
# mirror -- it is the only gate on the final radius, enforced here alone.
MIN_GEODESIC_LENGTH_MM = 5.0
CAP_RADIUS_VETO_FRACTION = 0.4
CAP_ANGLE_VETO_DEG = 25.0

# Competition brief's minimum branch size: diameter >= 2mm, i.e. radius >=
# 1.0mm. Checked against radius_at_seed_mm, which is exactly the radius_mm
# schema.make_daughter later writes to output (src.features.build_feature_vector
# sets it to seed["radius_mm"] unchanged) -- so this gates the same number
# that ends up in the final JSON, not some intermediate estimate of it.
MIN_RADIUS_MM = 1.0

# Absorbs known measurement noise near the minimum: radius_at_seed_mm is now
# primarily the flood-frontier estimate (src.tracing.extract_seed_direction_
# radius), not the distance transform -- see that function's docstring and
# validate_phantom's radius comparison for why the switch was made. Re-derived
# against the frontier estimate's own phantom-suite error in the true-radius
# 1.0-2.0mm band that brackets this cutoff (n=89, two known bad-trace outliers
# excluded -- both flagged independently by >20deg direction/ostium-disagreement
# errors, not radius noise): mean -0.14mm, sd 0.15mm, so mean + ~1sd lands back
# at this same 0.3mm figure -- kept, but now anchored to the frontier
# estimate's measured bias+noise rather than the distance transform's (which
# ran closer to 0.5mm low at this scale). No true radius below 1.0mm exists in
# the phantom suite, so behaviour exactly at MIN_RADIUS_MM is extrapolated from
# the nearest tested band, not directly measured. Vetoing at MIN_RADIUS_MM
# minus this margin, rather than at MIN_RADIUS_MM itself, trades a little of
# the other direction (a genuinely sub-1mm branch measured a bit high could
# still slip through) for recovering the branches this noise band was wrongly
# dropping. See README.md's Phantom validation section for the measured
# trade-off.
RADIUS_VETO_MEASUREMENT_TOLERANCE_MM = 0.3
RADIUS_VETO_MM = MIN_RADIUS_MM - RADIUS_VETO_MEASUREMENT_TOLERANCE_MM

# ------------------------------------------------------------- penalties
# Each term ramps from 0 penalty at `onset` to full weight at `full` (onset
# and full may be given in either order to make the ramp run backwards).

# Parallelism to the aorta: a branch is expected to leave the wall at a
# real angle; a traced path running alongside the aorta's own centerline
# for its length (small angle) reads as a vein pressed against it, not a
# branch leaving it. Nothing above the onset angle is penalised -- branches
# leave at all sorts of angles and that alone is not suspicious.
PARALLEL_ANGLE_ONSET_DEG = 45.0
PARALLEL_ANGLE_FULL_DEG = 15.0
PARALLEL_PENALTY_WEIGHT = 0.30

# Cross-section flattening: a round arterial lumen has a major/minor axis
# ratio near 1; a vein pressed flat against the aortic wall, or two lumens
# merged in one plane, reads flattened.
FLATNESS_ONSET_RATIO = 1.6
FLATNESS_FULL_RATIO = 3.0
FLATNESS_PENALTY_WEIGHT = 0.20

# HU well below the lumen reference: arteries and veins enhance differently
# on a single-phase scan, so a lumen reading much darker than the aorta's
# own peak -- while still bright enough to have passed the flood's
# threshold -- is the venous signature here, not a dimmer artery.
HU_DEFICIT_ONSET_HU = 60.0
HU_DEFICIT_FULL_HU = 200.0
HU_DEFICIT_PENALTY_WEIGHT = 0.20

# Ostium-estimate disagreement: for a genuine funnel-shaped mouth the patch
# centroid, the back-projected trace and the opening's gradient all land
# within a mm or two of each other (src.tracing.estimate_ostium_candidates).
# Large disagreement is scatter, not signal.
DISAGREEMENT_ONSET_MM = 2.0
DISAGREEMENT_FULL_MM = 6.0
DISAGREEMENT_PENALTY_WEIGHT = 0.15

# Implausible radius relative to the nearby aorta: a direct daughter is
# always markedly narrower than the vessel it leaves. Above this fraction
# of the nearby aortic radius (features.nearby_aortic_radius_mm), "daughter"
# starts looking like the aorta continuing through a segmentation gap that
# candidates.py's own aortic_continuation rule didn't catch because this
# instance never touched a flagged cap.
RADIUS_RATIO_ONSET = 0.35
RADIUS_RATIO_FULL = 0.70
RADIUS_RATIO_PENALTY_WEIGHT = 0.20

# Ostium flare: a real ostium is a funnel -- the contact patch is wider
# than the vessel's own cross-section a couple of mm downstream, because the
# wall's partial-volume rim broadens the mouth. A patch no wider than the
# vessel it feeds (ratio <= onset) is a graze along the wall, not a mouth;
# nothing above onset is penalised, since a wide funnel is exactly what is
# wanted.
FLARE_RATIO_ONSET = 1.0
FLARE_RATIO_FULL = 0.6
FLARE_PENALTY_WEIGHT = 0.15

# Radius consistency: a genuine vessel holds a roughly constant calibre
# over its first few mm; a coefficient of variation above this is either a
# leak bulging into open tissue or a segmentation seam.
RADIUS_CV_ONSET = 0.35
RADIUS_CV_FULL = 0.90
RADIUS_CV_PENALTY_WEIGHT = 0.10

# Leak margin: see features._leak_margin. Below this fraction of the gap up
# to the lumen reference, the component was only reachable because the
# threshold search had to loosen the cutoff to include it -- the same
# thing src.floodfill.detect_leak's growth/explosion rules watch for at the
# whole-flood level, applied per component here.
LEAK_MARGIN_ONSET = 0.5
LEAK_MARGIN_FULL = 0.0
LEAK_MARGIN_PENALTY_WEIGHT = 0.15

# Flat penalty applied to every instance in a case whose flood is flagged
# leaking (src.floodfill.detect_leak): the threshold that produced every
# component in that case is then known to have let something get away, so
# nothing from it is scored quite as confidently.
CASE_LEAK_PENALTY = 0.15

# Sharing a vessel with another instance is not itself disqualifying -- the
# brief keeps both as independent origins -- but it is the split's own
# uncertainty signal: an instance whose boundary with a neighbour needed
# disambiguating (src.parentage.split_into_instances) is marked down a
# little relative to an otherwise-identical, cleanly separate instance.
SHARED_VESSEL_PENALTY = 0.05

# Prompt 5's threshold sweep starts here.
CONFIDENCE_THRESHOLD = 0.5


def _ramp(value, onset, full):
    """0 at onset, 1 at full (full below onset runs the ramp backwards)."""
    span = full - onset
    if span == 0:
        return 0.0
    return float(np.clip((value - onset) / span, 0.0, 1.0))


def _check_veto(features):
    if features["traced_length_mm"] < MIN_GEODESIC_LENGTH_MM:
        return f"traced length {features['traced_length_mm']:.1f}mm < {MIN_GEODESIC_LENGTH_MM:.0f}mm"
    if features["radius_at_seed_mm"] < RADIUS_VETO_MM:
        return (f"radius {features['radius_at_seed_mm']:.2f}mm < {RADIUS_VETO_MM:.2f}mm "
                f"({MIN_RADIUS_MM:.1f}mm minimum less {RADIUS_VETO_MEASUREMENT_TOLERANCE_MM:.1f}mm "
                f"measurement-noise margin)")
    if features["touches_cap"] > 0.5:
        if features["cap_radius_ratio"] > CAP_RADIUS_VETO_FRACTION:
            return (f"cap-touching with radius {features['cap_radius_ratio']:.2f}x local aorta "
                    f"> {CAP_RADIUS_VETO_FRACTION:.2f}")
        if features["angle_to_centerline_deg"] <= CAP_ANGLE_VETO_DEG:
            return (f"cap-touching within {features['angle_to_centerline_deg']:.0f}deg of "
                    f"centerline <= {CAP_ANGLE_VETO_DEG:.0f}deg")
    return None


def explain(features):
    """Every term's raw value, ramp, weight and contribution, plus the
    final veto/confidence -- what to print when a candidate needs
    debugging against the visual checks.
    """
    veto = _check_veto(features)
    terms = []

    def term(name, value, onset, full, weight, note):
        ramp = _ramp(value, onset, full)
        terms.append({
            "name": name, "value": float(value), "onset": onset, "full": full,
            "weight": weight, "ramp": ramp, "penalty": ramp * weight, "note": note,
        })

    term("parallel_to_aorta", features["angle_to_centerline_deg"],
        PARALLEL_ANGLE_ONSET_DEG, PARALLEL_ANGLE_FULL_DEG, PARALLEL_PENALTY_WEIGHT,
        "small angle to the centerline tangent = running alongside the aorta, vein-like")
    term("flattened_cross_section", features["cross_section_flatness"],
        FLATNESS_ONSET_RATIO, FLATNESS_FULL_RATIO, FLATNESS_PENALTY_WEIGHT,
        "major/minor axis ratio of the lumen blob at the seed")
    term("hu_below_lumen", -features["mean_hu_relative_to_lumen"],
        HU_DEFICIT_ONSET_HU, HU_DEFICIT_FULL_HU, HU_DEFICIT_PENALTY_WEIGHT,
        "mean trace HU well under the aortic lumen reference")
    term("ostium_disagreement", features["ostium_max_disagreement_mm"],
        DISAGREEMENT_ONSET_MM, DISAGREEMENT_FULL_MM, DISAGREEMENT_PENALTY_WEIGHT,
        "the three ostium estimates scatter rather than agree")

    radius_ratio = (
        features["radius_at_seed_mm"] / features["nearby_aortic_radius_mm"]
        if features["nearby_aortic_radius_mm"] > 0 else 0.0
    )
    term("radius_vs_aorta", radius_ratio,
        RADIUS_RATIO_ONSET, RADIUS_RATIO_FULL, RADIUS_RATIO_PENALTY_WEIGHT,
        "seed radius as a fraction of the nearby aortic radius")
    term("thin_ostium", features["ostium_flare_ratio"],
        FLARE_RATIO_ONSET, FLARE_RATIO_FULL, FLARE_PENALTY_WEIGHT,
        "contact patch area vs. the vessel's own cross-section 2mm out: a real "
        "ostium is a wider funnel, not equal to or narrower than the vessel")
    term("radius_inconsistent", features["radius_consistency_cv"],
        RADIUS_CV_ONSET, RADIUS_CV_FULL, RADIUS_CV_PENALTY_WEIGHT,
        "coefficient of variation of the frontier radius over the first few mm")
    term("leak_margin", features["leak_margin"],
        LEAK_MARGIN_ONSET, LEAK_MARGIN_FULL, LEAK_MARGIN_PENALTY_WEIGHT,
        "how far this component's own HU sits above the flood's chosen threshold")

    case_leak_penalty = CASE_LEAK_PENALTY if features["case_flood_leaking"] > 0.5 else 0.0
    shared_penalty = SHARED_VESSEL_PENALTY if features["shares_vessel_with_count"] > 0 else 0.0

    total_penalty = sum(t["penalty"] for t in terms) + case_leak_penalty + shared_penalty
    confidence = 0.0 if veto is not None else float(np.clip(1.0 - total_penalty, 0.0, 1.0))

    return {
        "veto": veto,
        "terms": terms,
        "case_flood_leaking_penalty": case_leak_penalty,
        "shares_vessel_penalty": shared_penalty,
        "total_penalty": float(total_penalty),
        "confidence": confidence,
    }


def score_candidate(features):
    """Confidence in [0, 1] that this traced instance is a real, direct
    daughter artery. 0 whenever a hard veto fires; otherwise 1 minus the
    sum of the graded penalties, floored at 0. See explain() for the
    per-term breakdown.
    """
    return explain(features)["confidence"]
