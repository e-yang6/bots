"""Flatten one traced instance's pipeline output into a numeric feature dict.

Every value here is either read directly off run.py's own pipeline objects
(instance / trace / bifurcation / ostium / seed / candidate / centerline /
flood) or computed fresh from the same primitives those already sample (the
HU array, the frontier radius, the case's own thresholds). Nothing is a
hand-tuned score -- that combination happens in src/rules.py, which is the
only place weights live.

"vein-likeness" and "leak-proximity" in the challenge brief are not single
numbers computed here. Vein-likeness is a *combination* of angle-to-
centerline, cross-section flattening, HU-relative-to-lumen and the ostium
flare ratio -- each already a feature below -- combined transparently in
src/rules.py so the weight on each stays visible rather than folded into one
opaque scalar here. Leak-proximity is leak_margin plus case_flood_leaking.

A "candidate" here, matching src/parentage.py's own vocabulary, is the
pre-split component an instance's contact patch came from (look it up as
context["candidates"][i] where candidate["label"] == instance["component_label"]
-- see src/parentage.split_into_instances's docstring for why the cap
radius-ratio / angle-to-centerline discriminators live on the candidate,
not on the split instance).
"""

import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from scipy.spatial import cKDTree

from src.candidates import RADIUS_WINDOW_MM, frontier_radius_mm, voxel_volume_mm3
from src.floodfill import aorta_cross_section_voxels
from src.geometry import _indices_to_physical
from src.lumen_evidence import perpendicular_basis

# Arc-length span probed for "radius consistency along the trace": far
# enough to see whether the tube holds its calibre, short enough to stay
# inside a single vessel's first segment before any real taper.
RADIUS_CONSISTENCY_SPAN_MM = (2.0, 8.0)
RADIUS_CONSISTENCY_STEP_MM = 1.0

# Where the ostium-flare cross-section is measured: 2mm out from the wall is
# past the partial-volume sleeve (src.floodfill.SLEEVE_MM) but still at the
# mouth, so a genuine funnel (patch wider than the vessel it feeds) is still
# visible; further out a real branch has already narrowed to travelling
# calibre and the comparison stops being about the ostium at all.
FLARE_DISTANCE_MM = 2.0
FLARE_WINDOW_MM = 1.0

# How far above/below the chosen flood threshold "leak margin" looks,
# expressed as a fraction of the (threshold, lumen_reference) gap so it
# scales with the case's own contrast the same way the threshold search does.
LEAK_MARGIN_REFERENCE_FRACTION = 1.0


def _num(value, default=0.0):
    return float(default) if value is None else float(value)


def _cap_points_mm(cap, reference_image):
    idx_zyx = np.argwhere(cap["cap_region_mask"])
    if idx_zyx.shape[0] == 0:
        return np.zeros((0, 3))
    return _indices_to_physical(idx_zyx[:, ::-1].astype(np.float64), reference_image)


def build_cap_trees(caps, reference_image):
    """One cKDTree per flagged cap, built once per case and reused across
    every instance's feature vector -- a cap's own voxels never move."""
    trees = []
    for cap in caps:
        points = _cap_points_mm(cap, reference_image)
        trees.append({"cap": cap, "tree": cKDTree(points) if points.shape[0] else None})
    return trees


def _distance_to_nearest_cap(point_mm, cap_trees):
    best_distance = None
    for entry in cap_trees:
        if entry["tree"] is None:
            continue
        distance, _ = entry["tree"].query(point_mm[None, :], k=1)
        distance = float(distance[0])
        if best_distance is None or distance < best_distance:
            best_distance = distance
    return best_distance


def build_centerline_lookup(centerline):
    """Nearest-centerline-point lookup: one tree over every kept component's
    points, each tagged with its own normalized (0..1) arc-length position.
    Built once per case, like build_cap_trees.
    """
    all_points, normalized = [], []
    for component in centerline.get("kept", []):
        points = component["points_mm"]
        if points.shape[0] == 0:
            continue
        segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
        total = cumulative[-1] if cumulative[-1] > 0 else 1.0
        all_points.append(points)
        normalized.append(cumulative / total)
    if not all_points:
        return None
    return {"tree": cKDTree(np.vstack(all_points)), "normalized": np.concatenate(normalized)}


def _normalized_position(point_mm, centerline_lookup):
    if centerline_lookup is None:
        return None
    _distance, index = centerline_lookup["tree"].query(point_mm[None, :], k=1)
    return float(centerline_lookup["normalized"][int(index[0])])


def _radius_consistency(instance, voxel_volume, span_mm=RADIUS_CONSISTENCY_SPAN_MM,
                        step_mm=RADIUS_CONSISTENCY_STEP_MM):
    """Coefficient of variation of the frontier radius over span_mm.

    A real tube holds roughly one calibre over a few mm; a component whose
    frontier balloons or pinches -- an aortic-continuation stub too subtle
    for candidates.py's own rule, or a leak into open tissue -- swings
    widely instead.
    """
    low, high = span_mm
    stop = min(high, instance["max_distance_mm"])
    positions = np.arange(low, stop + 1e-9, step_mm)
    if positions.size < 2:
        return 0.0
    radii = np.array([
        frontier_radius_mm(instance["voxels_dist"], voxel_volume, p, RADIUS_WINDOW_MM)
        for p in positions
    ])
    mean = radii.mean()
    return float(radii.std() / mean) if mean > 0 else 0.0


def _cross_section_flatness(sampler, point_mm, direction, threshold_hu, extent_mm=5.0, step_mm=0.4):
    """Major/minor semi-axis ratio of the lumen blob at point_mm, in the
    plane normal to direction. 1.0 is round; a flattened, elongated
    cross-section (well above 1) is the signature of a vein pressed flat
    against the aorta, not a round arterial lumen.

    Self-contained rather than reusing tracing.py's _recentre_on_lumen:
    that returns an area, not the shape needed to tell round from flat, and
    this file must not modify tracing.py.
    """
    u, v = perpendicular_basis(direction[None, :])
    u, v = u[0], v[0]
    axis = np.arange(-extent_mm, extent_mm + 1e-9, step_mm)
    grid_u, grid_v = np.meshgrid(axis, axis, indexing="ij")
    plane = point_mm + grid_u[..., None] * u + grid_v[..., None] * v
    flat_plane = plane.reshape(-1, 3)

    hu = sampler.sample("hu", flat_plane, cval=-1e4).reshape(grid_u.shape)
    in_aorta = sampler.sample("mask", flat_plane, cval=0.0, order=0).reshape(grid_u.shape) > 0.5
    lumen = (hu >= threshold_hu) & ~in_aorta

    labels, count = ndimage.label(lumen)
    if count == 0:
        return 1.0
    centre = (grid_u.shape[0] // 2, grid_u.shape[1] // 2)
    label = labels[centre]
    if label == 0:
        filled = np.argwhere(labels > 0)
        nearest = filled[np.argmin(np.sum((filled - np.array(centre)) ** 2, axis=1))]
        label = labels[nearest[0], nearest[1]]
    blob = labels == label
    if blob.sum() < 3:
        return 1.0

    coords_u, coords_v = grid_u[blob], grid_v[blob]
    covariance = np.cov(np.vstack([coords_u, coords_v]))
    eigenvalues = np.clip(np.linalg.eigvalsh(covariance), 1e-9, None)
    major, minor = float(np.sqrt(eigenvalues.max())), float(np.sqrt(eigenvalues.min()))
    return major / minor if minor > 1e-9 else float(major / 1e-9)


def _leak_margin(instance, hu_arr, threshold_hu, lumen_reference_hu,
                 reference_fraction=LEAK_MARGIN_REFERENCE_FRACTION):
    """How far this component's own voxels sit above the threshold that let
    the flood reach them, as a fraction of the gap up to the lumen itself.

    0 means the component is only bright at the exact cutoff the threshold
    search landed on -- push the threshold up at all and it would vanish,
    which is exactly what src.floodfill.robust_flood's own bisection search
    is testing for at the whole-flood level, applied per component here.
    1 means it is as bright as the aortic lumen itself.
    """
    core_hu = hu_arr.ravel()[instance["core_flat"]]
    if core_hu.size == 0:
        return 0.0
    span = max(lumen_reference_hu - threshold_hu, 1e-6) * reference_fraction
    margin = (float(np.median(core_hu)) - threshold_hu) / span
    return float(np.clip(margin, 0.0, 2.0))


def build_feature_vector(instance, trace, bifurcation, ostium, seed, candidate, flood,
                         sampler, hu_arr, centerline_lookup, cap_trees, case_aorta_radius_mm,
                         reference_image):
    """One flat, JSON-safe dict of numeric features for one traced instance.

    Arguments are exactly the per-instance objects run.py.analyze_volumes
    already produces, aligned by index, plus the per-case, build-once
    lookups (build_cap_trees, build_centerline_lookup, case_aorta_radius_mm)
    and the lumen_evidence.VolumeSampler used for the one thing the flood
    pipeline doesn't already compute: cross-section shape.
    """
    voxel_volume = voxel_volume_mm3(reference_image)
    threshold_hu = flood["threshold_hu"]
    lumen_reference = flood["lumen_reference_hu"]

    hu_along_trace = sampler.sample("hu", trace["points_mm"]) if trace["points_mm"].shape[0] else np.zeros(0)
    hu_relative = hu_along_trace - lumen_reference if hu_along_trace.size else np.zeros(0)

    flare_area = np.pi * frontier_radius_mm(
        instance["voxels_dist"], voxel_volume, FLARE_DISTANCE_MM, FLARE_WINDOW_MM
    ) ** 2
    flare_ratio = instance["patch_area_mm2"] / flare_area if flare_area > 0 else 0.0

    seed_direction = np.asarray(seed["direction_xyz"], dtype=np.float64)
    flatness = _cross_section_flatness(sampler, seed["seed_mm"], seed_direction, threshold_hu)

    patch_centroid = np.asarray(instance["patch_centroid_mm"], dtype=np.float64)
    normalized_position = _normalized_position(patch_centroid, centerline_lookup)
    cap_distance = _distance_to_nearest_cap(patch_centroid, cap_trees)

    local_radius = candidate.get("local_aortic_radius_mm")
    nearby_radius_is_local = local_radius is not None and local_radius > 0
    nearby_radius = float(local_radius) if nearby_radius_is_local else case_aorta_radius_mm

    vesselness = instance.get("vesselness") or {}

    return {
        # --- geometry of the trace itself ---
        "traced_length_mm": float(trace["traced_length_mm"]),
        "truncated_by_bifurcation": float(trace["truncated_by"] == "bifurcation"),
        "truncated_by_vessel_end": float(trace["truncated_by"] == "vessel_end"),
        "reached_seed_distance": float(bool(seed["reached_seed_distance"])),

        # --- intensity along the trace, relative to the lumen ---
        "mean_hu_relative_to_lumen": float(hu_relative.mean()) if hu_relative.size else 0.0,
        "peak_hu_relative_to_lumen": float(hu_relative.max()) if hu_relative.size else 0.0,

        # --- radius ---
        "radius_at_seed_mm": float(seed["radius_mm"]),
        "radius_consistency_cv": _radius_consistency(instance, voxel_volume),
        "radius_disagreement": float(seed["radius_disagreement"]),
        "radius_disagreement_flag": float(bool(seed["radius_disagreement_flag"])),
        "nearby_aortic_radius_mm": nearby_radius,
        "nearby_aortic_radius_is_local": float(nearby_radius_is_local),

        # --- contact patch / ostium shape ---
        "patch_area_mm2": float(instance["patch_area_mm2"]),
        "ostium_flare_ratio": float(flare_ratio),
        "cross_section_flatness": float(flatness),

        # --- orientation relative to the aorta ---
        "angle_to_centerline_deg": _num(candidate.get("angle_to_centerline_deg"), default=90.0),
        "normalized_position_along_aorta": _num(normalized_position, default=0.5),

        # --- caps ---
        "distance_to_cap_mm": _num(cap_distance, default=999.0),
        "touches_cap": float(bool(instance.get("touches_cap"))),
        "cap_face_fraction": float(instance.get("cap_face_fraction", 0.0)),
        "cap_radius_ratio": _num(candidate.get("radius_ratio"), default=0.0),

        # --- ostium-estimate agreement ---
        "ostium_max_disagreement_mm": float(ostium["max_disagreement_mm"]),
        "ostium_mean_disagreement_mm": float(ostium["mean_disagreement_mm"]),

        # --- vesselness along the trace ---
        "vesselness_median_response": float(vesselness.get("median_response", 0.0)),
        "vesselness_median_radius_mm": float(vesselness.get("median_radius_mm", 0.0)),

        # --- leak-proximity ---
        "leak_margin": _leak_margin(instance, hu_arr, threshold_hu, lumen_reference),
        "case_flood_leaking": float(bool(flood["leak"]["leak"])),

        # --- instance-splitting context (informational, not scored: see
        # src/rules.py -- absorbing a graze is not itself a bad sign) ---
        "shares_vessel_with_count": float(len(instance.get("shares_vessel_with", []))),
        "n_absorbed_narrow_pieces": float(instance.get("absorbed_narrow_pieces", 0)),
    }


def build_feature_matrix(context):
    """Feature dicts for every traced instance in a run.py analyze_volumes
    context, aligned with context["traces"] / context["instances"][:n].

    Builds the per-case lookups (cap trees, centerline arc-length lookup,
    case-wide aortic radius) once, since every instance queries the same
    caps and the same centerline.
    """
    if not context["traces"]:
        return []

    reference_image = context["image"]
    hu_arr = sitk.GetArrayFromImage(reference_image)
    mask_arr = sitk.GetArrayFromImage(context["mask"]).astype(bool)
    voxel_area = voxel_volume_mm3(reference_image) ** (2.0 / 3.0)
    case_aorta_radius_mm = float(
        np.sqrt(aorta_cross_section_voxels(mask_arr) * voxel_area / np.pi)
    )

    sampler = context["sampler"]
    flood = context["flood"]
    candidates_by_label = {c["label"]: c for c in context["candidates"]}
    cap_trees = build_cap_trees(context["caps"], reference_image)
    centerline_lookup = build_centerline_lookup(context["centerline"])

    vectors = []
    for i, trace in enumerate(context["traces"]):
        instance = context["instances"][i]
        candidate = candidates_by_label.get(instance["component_label"], {})
        vectors.append(build_feature_vector(
            instance, trace, context["bifurcations"][i], context["ostium_estimates"][i],
            context["seed_estimates"][i], candidate, flood, sampler, hu_arr,
            centerline_lookup, cap_trees, case_aorta_radius_mm, reference_image,
        ))
    return vectors
