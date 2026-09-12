"""Decide whether a candidate actually arises from another candidate's
branch rather than from the aorta itself.

The test is distance to the putative parent's lumen SURFACE, not raw
path-to-path distance: two branches running a few mm apart in parallel are
both aortic daughters, whereas a vessel whose origin sits right on another
branch's wall is that branch's daughter and must not be reported as a direct
aortic daughter.
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


def check_branch_of_branch(
    candidates,
    traces,
    aorta_surface_tree,
    ostium_points_mm=None,
    surface_tolerance_mm=1.5,
    aorta_margin_mm=2.5,
    min_parent_arc_mm=1.5,
    min_parent_length_mm=2.0,
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
        }
        if best is not None:
            result.update(best)
            result["is_branch_of_branch"] = bool(
                best["distance_to_parent_surface_mm"] <= surface_tolerance_mm
                and distance_to_aorta_mm > aorta_margin_mm
            )
        results.append(result)

    return results
