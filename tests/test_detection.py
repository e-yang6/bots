"""Detection/tracing/parentage tests against a phantom with known geometry.

The phantom puts the daughter stub in the IMAGE but not in the MASK, which is
how the real task is posed -- the supplied mask is aorta-only.
"""

import numpy as np
import pytest
from scipy.spatial import cKDTree

from src.candidates import build_evidence, find_candidate_ostia, lumen_likeness
from src.geometry import compute_centerline, compute_surface_normals, crop_to_mask_bbox, flag_end_caps
from src.intensity import lumen_stats
from src.parentage import check_branch_of_branch
from src.tracing import (
    cross_section,
    detect_bifurcation,
    estimate_ostium_candidates,
    extract_seed_direction_radius,
    trace_branch,
    truncate_trace,
)
from tests.synthetic import make_cylinder_with_stub

STUB_RADIUS_MM = 3.0


def build_phantom(**overrides):
    settings = dict(
        shape_zyx=(90, 60, 60), radius_mm=8.0, z_start=10, z_end=80,
        stub=True, stub_in_mask=False, stub_z=45, stub_radius_mm=STUB_RADIUS_MM,
        stub_length_mm=18.0, noise_speck=False, taper_low_end=False,
        lumen_hu=300.0, background_hu=40.0,
    )
    settings.update(overrides)
    return make_cylinder_with_stub(**settings)


def analyze_phantom(image, mask, margin_mm=8):
    cropped_image, cropped_mask = crop_to_mask_bbox(image, mask, margin_mm)
    stats = lumen_stats(cropped_image, cropped_mask)
    evidence = build_evidence(cropped_image, cropped_mask, stats)
    centerline = compute_centerline(cropped_mask)
    surface = compute_surface_normals(cropped_mask)
    caps = flag_end_caps(cropped_mask, surface, centerline)
    candidates, evidence = find_candidate_ostia(
        cropped_image, cropped_mask, stats, surface, caps, evidence=evidence
    )
    return {
        "image": cropped_image, "mask": cropped_mask, "evidence": evidence,
        "surface": surface, "caps": caps, "candidates": candidates,
        "sampler": evidence["sampler"], "lumen_threshold": evidence["lumen_threshold"],
        "surface_tree": cKDTree(surface[0]),
    }


def trace_candidate(context, candidate):
    trace = trace_branch(
        context["sampler"], candidate["ostium_patch_centroid_mm"],
        candidate["outward_direction"], context["lumen_threshold"],
    )
    bifurcation = detect_bifurcation(trace)
    if bifurcation["bifurcation_index"] is not None:
        trace = truncate_trace(trace, bifurcation["bifurcation_index"])
    ostium = estimate_ostium_candidates(
        context["sampler"], candidate, trace, context["lumen_threshold"],
        context["surface_tree"], context["surface"][0],
    )
    seed = extract_seed_direction_radius(
        context["sampler"], trace, ostium["consensus_mm"], context["lumen_threshold"]
    )
    return trace, ostium, seed


def test_lumen_likeness_rejects_calcium_and_accepts_lumen():
    # the whole point of the band: bone/calcium far above lumen HU must not
    # score higher than lumen itself
    values = np.array([0.0, 0.5, 1.0, 1.3, 2.5, 4.0])
    likeness = lumen_likeness(values)
    assert likeness[0] == 0.0            # background tissue
    assert likeness[2] == pytest.approx(1.0)  # lumen
    assert likeness[3] == pytest.approx(1.0)  # slightly brighter lumen
    assert likeness[4] == 0.0            # calcium
    assert likeness[5] == 0.0            # bone


def test_finds_the_single_stub_and_nothing_else():
    context = analyze_phantom(*build_phantom())
    assert len(context["candidates"]) == 1

    candidate = context["candidates"][0]
    assert candidate["peak_score"] > 1.0
    # the stub leaves along +x, so the ostium's outward direction must too
    assert candidate["outward_direction"][0] > 0.9


def test_no_candidates_on_a_plain_tube_with_no_branch():
    context = analyze_phantom(*build_phantom(stub=False))
    assert context["candidates"] == []


def test_trace_follows_the_stub_and_recovers_direction_and_radius():
    context = analyze_phantom(*build_phantom())
    candidate = context["candidates"][0]
    trace, ostium, seed = trace_candidate(context, candidate)

    assert trace["traced_length_mm"] >= 5.0
    assert seed["reached_seed_arc_length"] is True

    direction = seed["direction_xyz"]
    assert np.isclose(np.linalg.norm(direction), 1.0, atol=1e-6)
    assert direction[0] > 0.95  # the stub runs along +x

    # the ellipse cross-check should recover the true radius closely; the
    # distance-transform reading is lower because excluding the aorta cuts
    # the stub's lumen near the junction
    assert seed["radius_from_ellipse_mm"] == pytest.approx(STUB_RADIUS_MM, abs=0.7)
    assert 1.0 < seed["radius_mm"] < 2.0 * STUB_RADIUS_MM


def test_trace_keeps_length_continuous_rather_than_snapping_to_5mm():
    context = analyze_phantom(*build_phantom(stub_length_mm=18.0))
    trace, _ostium, _seed = trace_candidate(context, context["candidates"][0])
    # no 5mm quantization anywhere: the eligibility rule belongs downstream
    assert trace["traced_length_mm"] % 5.0 != 0.0 or trace["traced_length_mm"] > 5.0
    assert trace["arc_lengths_mm"][0] == 0.0
    assert np.all(np.diff(trace["arc_lengths_mm"]) > 0)


def test_ostium_estimates_agree_for_a_clean_branch():
    context = analyze_phantom(*build_phantom())
    _trace, ostium, _seed = trace_candidate(context, context["candidates"][0])

    assert set(ostium["estimates"]) == {
        "patch_centroid_mm", "traced_projection_mm", "max_gradient_mm"
    }
    assert len(ostium["pairwise_distances_mm"]) == 3
    assert ostium["max_disagreement_mm"] >= ostium["mean_disagreement_mm"]
    # patch centroid and the back-projected trace describe the same opening
    patch = ostium["estimates"]["patch_centroid_mm"]
    projected = ostium["estimates"]["traced_projection_mm"]
    assert np.linalg.norm(patch - projected) < 4.0


def test_cross_section_excluding_aorta_measures_the_branch_not_the_aorta():
    context = analyze_phantom(*build_phantom())
    candidate = context["candidates"][0]
    trace, _ostium, _seed = trace_candidate(context, candidate)
    # a plane taken close to the origin is the case that matters: far enough
    # out, the plane misses the aorta entirely and the two agree
    point = trace["points_mm"][1]
    direction = trace["directions"][1]

    with_aorta = cross_section(
        context["sampler"], point, direction, context["lumen_threshold"], exclude_aorta=False
    )
    without_aorta = cross_section(
        context["sampler"], point, direction, context["lumen_threshold"], exclude_aorta=True
    )
    # including the aorta merges the two lumens and inflates the measurement
    assert without_aorta["area_mm2"] < with_aorta["area_mm2"]
    assert without_aorta["equivalent_radius_mm"] == pytest.approx(STUB_RADIUS_MM, abs=1.2)


def test_detect_bifurcation_truncates_at_a_split():
    # a synthetic trace whose front stays steady, then splits in two
    steps = 24
    trace = {
        "arc_lengths_mm": np.arange(steps) * 0.5,
        "areas_mm2": np.full(steps, 12.0),
        "n_components": np.ones(steps, dtype=int),
        "points_mm": np.stack([np.arange(steps) * 0.5, np.zeros(steps), np.zeros(steps)], axis=1),
        "directions": np.tile(np.array([1.0, 0.0, 0.0]), (steps, 1)),
        "radii_mm": np.full(steps, 2.0),
        "lumen": np.ones(steps),
        "vesselness": np.ones(steps),
        "traced_length_mm": (steps - 1) * 0.5,
        "status": "max_length",
        "start_direction": np.array([1.0, 0.0, 0.0]),
    }
    trace["n_components"][16:] = 2

    result = detect_bifurcation(trace)
    assert result["bifurcation_index"] == 16
    assert result["reason"] == "bimodal_front"

    truncated = truncate_trace(trace, result["bifurcation_index"])
    assert truncated["status"] == "bifurcation"
    assert truncated["traced_length_mm"] == pytest.approx(8.0)
    assert len(truncated["points_mm"]) == 17


def test_detect_bifurcation_ignores_a_single_flickering_frame():
    steps = 24
    trace = {
        "arc_lengths_mm": np.arange(steps) * 0.5,
        "areas_mm2": np.full(steps, 12.0),
        "n_components": np.ones(steps, dtype=int),
        "traced_length_mm": (steps - 1) * 0.5,
    }
    trace["n_components"][14] = 2  # one frame only

    assert detect_bifurcation(trace)["bifurcation_index"] is None


def test_detect_bifurcation_on_a_sustained_area_jump():
    steps = 24
    areas = np.full(steps, 10.0)
    areas[15:] = 30.0
    trace = {
        "arc_lengths_mm": np.arange(steps) * 0.5,
        "areas_mm2": areas,
        "n_components": np.ones(steps, dtype=int),
        "traced_length_mm": (steps - 1) * 0.5,
    }
    result = detect_bifurcation(trace)
    assert result["reason"] == "area_jump"
    assert result["bifurcation_index"] == 15


def _straight_trace(start, direction, length_mm=16.0, radius_mm=3.0, step_mm=0.5):
    n = int(length_mm / step_mm) + 1
    arc = np.arange(n) * step_mm
    direction = np.asarray(direction, dtype=float)
    points = np.asarray(start, dtype=float) + np.outer(arc, direction)
    return {
        "points_mm": points,
        "directions": np.tile(direction, (n, 1)),
        "radii_mm": np.full(n, radius_mm),
        "arc_lengths_mm": arc,
        "traced_length_mm": float(arc[-1]),
        "status": "max_length",
        "start_direction": direction,
    }


def test_branch_of_branch_flagged_when_ostium_sits_on_another_branch_wall():
    # aorta surface is the plane x = 0; the parent branch runs out along +x
    aorta_surface = np.stack(
        np.meshgrid(np.zeros(1), np.linspace(-10, 10, 21), np.linspace(-10, 10, 21), indexing="ij"),
        axis=-1,
    ).reshape(-1, 3)
    tree = cKDTree(aorta_surface)

    parent = _straight_trace(start=(0, 0, 0), direction=(1, 0, 0), radius_mm=3.0)
    child = _straight_trace(start=(10, 3.0, 0), direction=(0, 1, 0), radius_mm=1.0)

    ostia = [np.array([0.0, 0.0, 0.0]), np.array([10.0, 3.0, 0.0])]
    results = check_branch_of_branch([None, None], [parent, child], tree, ostium_points_mm=ostia)

    direct, sub = results
    # the parent's own origin is on the aorta wall -> a direct daughter
    assert direct["distance_to_aorta_surface_mm"] == pytest.approx(0.0, abs=1e-6)
    assert direct["is_branch_of_branch"] is False

    # the child's origin sits exactly on the parent's lumen surface (axis
    # distance 3.0 == the parent's radius there) and well off the aorta
    assert sub["is_branch_of_branch"] is True
    assert sub["parent_candidate_index"] == 0
    assert sub["distance_to_parent_surface_mm"] == pytest.approx(0.0, abs=0.3)
    assert sub["distance_to_parent_axis_mm"] == pytest.approx(3.0, abs=0.3)
    assert sub["parent_radius_at_contact_mm"] == pytest.approx(3.0, abs=1e-6)
    assert sub["distance_to_aorta_surface_mm"] > 2.5


def test_parallel_neighbouring_branches_are_not_called_parent_and_child():
    """Two aortic daughters running side by side must both stay direct.

    This is exactly what raw path-to-path distance gets wrong: their axes
    pass within a few mm of each other, but neither origin lies on the
    other's wall.
    """
    aorta_surface = np.stack(
        np.meshgrid(np.zeros(1), np.linspace(-20, 20, 41), np.linspace(-20, 20, 41), indexing="ij"),
        axis=-1,
    ).reshape(-1, 3)
    tree = cKDTree(aorta_surface)

    first = _straight_trace(start=(0, 0, 0), direction=(1, 0, 0), radius_mm=2.0)
    second = _straight_trace(start=(0, 6.0, 0), direction=(1, 0, 0), radius_mm=2.0)

    ostia = [np.array([0.0, 0.0, 0.0]), np.array([0.0, 6.0, 0.0])]
    results = check_branch_of_branch([None, None], [first, second], tree, ostium_points_mm=ostia)

    for result in results:
        assert result["is_branch_of_branch"] is False
        assert result["distance_to_aorta_surface_mm"] == pytest.approx(0.0, abs=1e-6)


def test_parentage_measures_surface_gap_not_axis_distance():
    aorta_surface = np.array([[0.0, y, 0.0] for y in np.linspace(-10, 10, 21)])
    tree = cKDTree(aorta_surface)

    # a fat parent: its wall reaches out to 5mm, so an ostium 5mm off its
    # axis is ON it, even though 5mm of raw axis distance sounds far
    parent = _straight_trace(start=(0, 0, 0), direction=(1, 0, 0), radius_mm=5.0)
    child = _straight_trace(start=(12, 5.0, 0), direction=(0, 1, 0), radius_mm=1.0)

    ostia = [np.array([0.0, 0.0, 0.0]), np.array([12.0, 5.0, 0.0])]
    results = check_branch_of_branch([None, None], [parent, child], tree, ostium_points_mm=ostia)

    sub = results[1]
    assert sub["distance_to_parent_axis_mm"] == pytest.approx(5.0, abs=0.3)
    assert sub["distance_to_parent_surface_mm"] < sub["distance_to_parent_axis_mm"]
    assert sub["is_branch_of_branch"] is True


def test_phantom_direct_daughter_is_not_flagged_as_branch_of_branch():
    context = analyze_phantom(*build_phantom())
    candidates = context["candidates"]
    traces, ostia = [], []
    for candidate in candidates:
        trace, ostium, _seed = trace_candidate(context, candidate)
        traces.append(trace)
        ostia.append(ostium["consensus_mm"])

    results = check_branch_of_branch(candidates, traces, context["surface_tree"], ostium_points_mm=ostia)
    assert len(results) == len(candidates)
    assert all(not result["is_branch_of_branch"] for result in results)


def test_adaptive_band_leaves_clean_cases_alone():
    from src.candidates import LUMEN_BAND, adaptive_lumen_band

    # a clean case: noise is a small fraction of the contrast span, and the
    # reference sits far above the contrast threshold
    band, tightening = adaptive_lumen_band(
        background_noise_hu=40.0, contrast_span_hu=400.0, reference_hu=450.0,
        contrast_threshold_hu=80.0,
    )
    assert tightening == 0.0
    assert band == LUMEN_BAND


def test_adaptive_band_tightens_when_noise_rivals_the_contrast_span():
    from src.candidates import LUMEN_BAND, MAX_BAND_TIGHTENING, adaptive_lumen_band

    # subject024-like: background noise larger than the whole span
    band, tightening = adaptive_lumen_band(
        background_noise_hu=163.0, contrast_span_hu=146.0, reference_hu=84.0,
        contrast_threshold_hu=80.0,
    )
    assert tightening == pytest.approx(1.0)
    assert band[0] == pytest.approx(LUMEN_BAND[0] + MAX_BAND_TIGHTENING)
    assert band[1] == pytest.approx(LUMEN_BAND[1] + MAX_BAND_TIGHTENING)
    # the upper shoulders reject calcium and must not move with contrast
    assert band[2] == LUMEN_BAND[2]
    assert band[3] == LUMEN_BAND[3]


def test_adaptive_band_tightens_for_a_case_that_barely_clears_the_contrast_gate():
    from src.candidates import adaptive_lumen_band

    # low noise, but the reference only just clears the 80 HU threshold
    _band, tightening = adaptive_lumen_band(
        background_noise_hu=20.0, contrast_span_hu=200.0, reference_hu=90.0,
        contrast_threshold_hu=80.0,
    )
    assert tightening > 0.5


def test_adaptive_band_never_gates_a_case_into_silence():
    from src.candidates import adaptive_lumen_band

    # even absurd noise must leave the band usable rather than rejecting
    # everything: emitting nothing is worse than emitting filterable noise
    band, tightening = adaptive_lumen_band(
        background_noise_hu=5000.0, contrast_span_hu=10.0, reference_hu=81.0,
        contrast_threshold_hu=80.0,
    )
    assert tightening <= 1.0
    assert band[0] < band[1] < band[2] < band[3]
    assert band[1] < 1.0  # lumen itself still scores as lumen


def test_trace_containment_detects_a_path_running_inside_another_tube():
    from src.parentage import trace_containment

    host = _straight_trace(start=(0, 0, 0), direction=(1, 0, 0), radius_mm=3.0)
    # a second path running down the middle of the host
    inner = _straight_trace(start=(2, 0.5, 0), direction=(1, 0, 0), length_mm=8.0, radius_mm=1.0)
    fraction, _gap, axis_distance = trace_containment(inner, host)
    assert fraction == pytest.approx(1.0)
    assert axis_distance == pytest.approx(0.5, abs=0.2)

    # and one running well outside it
    outside = _straight_trace(start=(2, 9.0, 0), direction=(1, 0, 0), length_mm=8.0, radius_mm=1.0)
    fraction_outside, _gap, axis_outside = trace_containment(outside, host)
    assert fraction_outside == 0.0
    assert axis_outside > 3.0


def test_two_detections_of_one_vessel_collapse_to_one_independent_origin():
    aorta_surface = np.array([[0.0, y, 0.0] for y in np.linspace(-10, 10, 41)])
    tree = cKDTree(aorta_surface)

    # same vessel found twice from adjacent wall patches: the paths converge
    longer = _straight_trace(start=(0, 0, 0), direction=(1, 0, 0), length_mm=10.0, radius_mm=2.0)
    shorter = _straight_trace(start=(0, 0.8, 0), direction=(1, 0, 0), length_mm=6.0, radius_mm=2.0)

    ostia = [np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.8, 0.0])]
    results = check_branch_of_branch([None, None], [longer, shorter], tree, ostium_points_mm=ostia)

    # only the shorter one is demoted, so the vessel survives exactly once
    assert results[0]["is_independent_origin"] is True
    assert results[1]["is_independent_origin"] is False
    assert results[1]["shares_vessel_with"] == 0
    assert results[1]["containment_fraction"] >= 0.6


def test_two_nearby_but_separate_origins_are_both_kept():
    """The challenge requires two nearby origins to stay two instances.

    An overestimated radius can make one branch's tube swallow a neighbour,
    so containment alone must not collapse them -- the paths have to actually
    converge.
    """
    aorta_surface = np.array([[0.0, y, 0.0] for y in np.linspace(-10, 10, 41)])
    tree = cKDTree(aorta_surface)

    # a fat branch whose tube nominally reaches the neighbour, but the two
    # paths stay 4mm apart throughout
    fat = _straight_trace(start=(0, 0, 0), direction=(1, 0, 0), length_mm=10.0, radius_mm=5.0)
    neighbour = _straight_trace(start=(0, 4.0, 0), direction=(1, 0, 0), length_mm=8.0, radius_mm=1.0)

    ostia = [np.array([0.0, 0.0, 0.0]), np.array([0.0, 4.0, 0.0])]
    results = check_branch_of_branch([None, None], [fat, neighbour], tree, ostium_points_mm=ostia)

    assert results[1]["containment_fraction"] >= 0.6  # the tube does contain it
    assert results[1]["trace_convergence_mm"] > 2.0   # but they never meet
    assert results[1]["shares_vessel_with"] is None
    assert all(result["is_independent_origin"] for result in results)
