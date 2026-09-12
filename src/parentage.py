"""Decide whether a candidate is really an independent direct aortic daughter,
or whether it belongs to another candidate's vessel.

Two distinct relationships are tested, because on real data they fire in
very different circumstances.

Ostium-on-parent-wall: the candidate's origin sits on another branch's lumen
SURFACE. The test is surface distance (axis distance minus that branch's
local radius), not raw path-to-path distance -- two branches running a few mm
apart in parallel are both aortic daughters, whereas a vessel whose origin
sits right on another branch's wall is that branch's daughter.

Trace containment: the candidate's traced path runs INSIDE another
candidate's tube. This is the relationship that actually occurs here.
Candidate ostia are, by construction, points on the aorta surface -- measured
across the dev set they sit a median of 0.8mm and at most 1.4mm from the
wall -- so a genuine daughter-of-a-daughter arising well downstream never
produces its own aorta-surface candidate at all, and one arising close to the
aorta is legitimately near the wall. No amount of ostium refinement separates
those, which is why an "is it off the aortic wall" gate can never fire.
What does happen, and what must be caught before final output, is two
candidates whose traces run down the same vessel: either two detections of
one branch, or two sub-branches of one short common trunk. The challenge is
explicit that a common trunk has a single aortic origin even when it divides
shortly afterwards, so these have to collapse to one.
"""

import numpy as np


def _closest_point_on_polyline(polyline_mm, query_mm):
    """Closest point on a polyline to a query point.

    Returns (distance_mm, arc_length_mm_at_closest, segment_index, t) where t
    is the fractional position along that segment.
    """
    if polyline_mm.shape[0] == 0:
        return np.inf, 0.0, -1, 0.0
    if polyline_mm.shape[0] == 1:
        return float(np.linalg.norm(polyline_mm[0] - query_mm)), 0.0, 0, 0.0

    starts = polyline_mm[:-1]
    ends = polyline_mm[1:]
    segments = ends - starts
    lengths_squared = np.einsum("ij,ij->i", segments, segments)
    lengths_squared[lengths_squared == 0] = 1e-12

    t = np.einsum("ij,ij->i", query_mm - starts, segments) / lengths_squared
    t = np.clip(t, 0.0, 1.0)
    projections = starts + t[:, None] * segments
    distances = np.linalg.norm(projections - query_mm, axis=1)

    index = int(np.argmin(distances))
    segment_lengths = np.linalg.norm(segments, axis=1)
    arc_length = float(segment_lengths[:index].sum() + t[index] * segment_lengths[index])
    return float(distances[index]), arc_length, index, float(t[index])


def _radius_at_arc_length(trace, arc_length_mm):
    arc = trace["arc_lengths_mm"]
    radii = trace["radii_mm"]
    if arc.size == 0:
        return 0.0
    return float(np.interp(arc_length_mm, arc, radii))


def trace_containment(trace, other_trace, tolerance_mm=0.5):
    """How much of `trace` runs inside `other_trace`'s tube.

    Returns (fraction_inside, min_surface_gap_mm, min_axis_distance_mm). A
    point counts as inside when its distance to the other branch's axis is
    within that branch's local radius (plus a tolerance for the fact that
    both the axis and the radius are estimates).
    """
    points = trace["points_mm"]
    if points.shape[0] == 0 or other_trace["points_mm"].shape[0] == 0:
        return 0.0, np.inf, np.inf

    inside = 0
    min_gap = np.inf
    min_axis = np.inf
    for point in points:
        axis_distance, arc_length, _segment, _t = _closest_point_on_polyline(
            other_trace["points_mm"], point
        )
        radius = _radius_at_arc_length(other_trace, arc_length)
        min_axis = min(min_axis, axis_distance)
        min_gap = min(min_gap, abs(axis_distance - radius))
        if axis_distance <= radius + tolerance_mm:
            inside += 1

    return inside / points.shape[0], float(min_gap), float(min_axis)


def check_branch_of_branch(
    candidates,
    traces,
    aorta_surface_tree,
    ostium_points_mm=None,
    surface_tolerance_mm=1.5,
    aorta_margin_mm=2.5,
    min_parent_arc_mm=1.5,
    min_parent_length_mm=2.0,
    containment_fraction_threshold=0.6,
    containment_min_length_mm=2.0,
    containment_max_convergence_mm=2.0,
):
    """Flag candidates that arise from another candidate's branch.

    For each candidate's ostium, find the traced branch whose lumen surface
    it sits closest to. `distance_to_parent_surface_mm` is the gap between
    the ostium and that branch's wall (|distance to its axis| - its local
    radius), so it is near zero exactly when the ostium lies on the other
    vessel -- which raw axis-to-axis distance could never distinguish from
    a fatter parent passing nearby.

    A candidate is flagged only if it also stands off the aorta wall by more
    than aorta_margin_mm: a true aortic daughter sits ON the aorta, and where
    both parents are plausible the aorta wins (the challenge asks for direct
    aortic daughters).

    Returns one dict per candidate, in the same order.
    """
    if ostium_points_mm is None:
        ostium_points_mm = [np.asarray(c["ostium_patch_centroid_mm"], dtype=np.float64) for c in candidates]

    results = []
    for index, ostium in enumerate(ostium_points_mm):
        ostium = np.asarray(ostium, dtype=np.float64)

        distance_to_aorta, _nearest = aorta_surface_tree.query(ostium[None, :], k=1)
        distance_to_aorta_mm = float(distance_to_aorta[0])

        best = None
        for other_index, other_trace in enumerate(traces):
            if other_index == index or other_trace is None:
                continue
            if other_trace["traced_length_mm"] < min_parent_length_mm:
                continue

            axis_distance, arc_length, _segment, _t = _closest_point_on_polyline(
                other_trace["points_mm"], ostium
            )
            if arc_length < min_parent_arc_mm:
                # too close to the other branch's own origin to tell the two
                # apart -- they are neighbouring aortic daughters, not parent
                # and child
                continue

            parent_radius = _radius_at_arc_length(other_trace, arc_length)
            surface_gap = abs(axis_distance - parent_radius)

            if best is None or surface_gap < best["distance_to_parent_surface_mm"]:
                best = {
                    "parent_candidate_index": other_index,
                    "distance_to_parent_surface_mm": float(surface_gap),
                    "distance_to_parent_axis_mm": float(axis_distance),
                    "parent_radius_at_contact_mm": float(parent_radius),
                    "parent_arc_length_mm": float(arc_length),
                }

        result = {
            "candidate_index": index,
            "distance_to_aorta_surface_mm": distance_to_aorta_mm,
            "parent_candidate_index": None,
            "distance_to_parent_surface_mm": None,
            "distance_to_parent_axis_mm": None,
            "parent_radius_at_contact_mm": None,
            "parent_arc_length_mm": None,
            "is_branch_of_branch": False,
            "shares_vessel_with": None,
            "containment_fraction": 0.0,
            "trace_convergence_mm": None,
            "is_independent_origin": True,
        }
        if best is not None:
            result.update(best)
            result["is_branch_of_branch"] = bool(
                best["distance_to_parent_surface_mm"] <= surface_tolerance_mm
                and distance_to_aorta_mm > aorta_margin_mm
            )

        own_trace = traces[index] if index < len(traces) else None
        if own_trace is not None and own_trace["traced_length_mm"] >= containment_min_length_mm:
            best_containment = None
            for other_index, other_trace in enumerate(traces):
                if other_index == index or other_trace is None:
                    continue
                if other_trace["traced_length_mm"] < containment_min_length_mm:
                    continue
                # The longer trace is taken to represent the shared vessel, so
                # only the shorter of a pair is demoted. Without this both
                # members of a duplicate pair flag each other and neither
                # survives.
                if other_trace["traced_length_mm"] < own_trace["traced_length_mm"]:
                    continue

                fraction, surface_gap, axis_distance = trace_containment(own_trace, other_trace)
                if best_containment is None or fraction > best_containment[1]:
                    best_containment = (other_index, fraction, surface_gap, axis_distance)

            if best_containment is not None:
                other_index, fraction, _surface_gap, axis_distance = best_containment
                result["containment_fraction"] = float(fraction)
                result["trace_convergence_mm"] = float(axis_distance)
                # Containment alone is not enough: an overestimated radius on
                # the other branch inflates its tube until it swallows a
                # genuinely separate neighbour (observed at containment 1.00
                # with the two paths still ~4mm apart). The paths must also
                # actually meet, or two nearby origins -- which the challenge
                # requires be reported separately -- would be merged into one.
                if (
                    fraction >= containment_fraction_threshold
                    and axis_distance <= containment_max_convergence_mm
                ):
                    result["shares_vessel_with"] = int(other_index)

        result["is_independent_origin"] = not (
            result["is_branch_of_branch"] or result["shares_vessel_with"] is not None
        )
        results.append(result)

    return results
