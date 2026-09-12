"""Trace each instance through the flood's own shortest-path tree, and derive
the reported geometry -- ostium, seed, direction, radius -- from it.

There is no step-by-step tracer any more. The flood already holds, for every
voxel of a vessel, its geodesic distance from the aortic lumen and the
predecessor on its shortest path there. A branch centerline is one such path
followed backwards, and a bifurcation is where the flood front splits.

Geodesic distance is arc length from the aortic lumen by construction, so it
is used as the trace's arc-length parameter directly: the seed "5mm along the
vessel" is the path point at distance 5.0, never a re-measured Euclidean
distance from the ostium.
"""

import sys

import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from scipy.ndimage import gaussian_filter1d

from src.candidates import (
    RADIUS_WINDOW_MM,
    VESSELNESS_RADIUS_PER_SIGMA,
    fit_line_direction,
    flat_to_mm,
    flat_to_zyx,
    frontier_radius_mm,
    voxel_volume_mm3,
)
from src.geometry import _direction_matrix
from src.lumen_evidence import perpendicular_basis

MAX_TRACE_MM = 10.0
SEED_DISTANCE_MM = 5.0
TRACE_STEP_MM = 0.5
BAND_MM = 0.5

# A slab blob only counts toward a split if it is at least this fraction of
# the slab's largest blob (and a few voxels). The common-trunk rule is about
# a vessel dividing into comparable daughters; without this, a 1-voxel twig or
# a partial-volume nub on the wall would "bifurcate" nearly every trunk within
# a couple of mm and leave nothing to put the 5mm seed on.
BIFURCATION_MIN_BLOB_FRACTION = 0.25
BIFURCATION_MIN_BLOB_VOXELS = 4
BIFURCATION_PERSISTENCE = 2

# SimpleITK's signed Maurer map measures inside distances to the centres of
# the object's own boundary voxels, so at a tube's centre it reads about one
# voxel short of the true radius: 1.13mm for r=2mm and 2.26mm for r=3mm on a
# 0.8mm grid. One voxel (of the grid the map is computed on) is added back.
MAURER_CORRECTION_VOXELS = 1.0
MAURER_UPSAMPLE = 2

RADIUS_DISAGREEMENT_LOG_FRACTION = 0.5
OSTIUM_OPENING_WINDOW_MM = 1.0
OSTIUM_OPENING_EXTENT_MM = 6.0


def _crop_of(component, volume=None, margin=1):
    """Bounding-box crop around an instance's voxels.

    Returns (low_zyx, member_crop, volume_crop).
    """
    shape = component["shape"]
    zyx = flat_to_zyx(component["voxels_flat"], shape)
    low = np.maximum(zyx.min(axis=0) - margin, 0)
    high = np.minimum(zyx.max(axis=0) + margin + 1, shape)
    member = np.zeros(tuple(high - low), dtype=bool)
    local = zyx - low
    member[local[:, 0], local[:, 1], local[:, 2]] = True
    cropped = None
    if volume is not None:
        cropped = volume[low[0]:high[0], low[1]:high[1], low[2]:high[2]]
    return low, member, cropped


def _longest_edge_mm(spacing):
    return float(np.linalg.norm(np.asarray(spacing, dtype=np.float64)))


def detect_bifurcation_by_frontier(component, dist, spacing=(1.0, 1.0, 1.0), band_mm=BAND_MM,
                                   start_mm=None, stop_mm=None,
                                   min_blob_fraction=BIFURCATION_MIN_BLOB_FRACTION,
                                   min_blob_voxels=BIFURCATION_MIN_BLOB_VOXELS,
                                   persistence=BIFURCATION_PERSISTENCE):
    """First distance at which the flood front through this vessel splits.

    Walks outward in band_mm steps. At each step the vessel's voxels within a
    slab of distances are labelled (26-connectivity); the first slab holding
    two or more comparable, disconnected blobs is the bifurcation. That is an
    exact statement of "the vessel divided here", and it gives the brief's
    common-trunk rule directly: a coeliac trunk that divides 15mm out has one
    aortic origin and is truncated at its division.

    The slab is band_mm wide or one longest voxel edge, whichever is wider.
    Adjacent voxels differ in geodesic distance by up to one full edge
    (sqrt(3) * 0.8 = 1.39mm on the resampled grid), so a thinner slab can miss
    a whole layer of a straight tube on one side and cut a single vessel into
    rings -- a split that is pure quantization.

    The walk starts one slab beyond the instance's neck unless start_mm says
    otherwise: a slab overlapping the neck's inner face picks up the ragged
    edge of the partial-volume sleeve, whose nubs read as extra blobs. A split
    must also hold for `persistence` consecutive steps -- daughters that have
    genuinely divided stay divided, while quantization flickers.

    Returns {"distance_mm", "n_blobs", "blob_sizes", "slab_mm"}; distance_mm
    is None when the front never splits.
    """
    low, member, dist_crop = _crop_of(component, dist)
    values = np.where(member, dist_crop, np.inf)
    slab_mm = max(band_mm, _longest_edge_mm(spacing))
    start = component.get("neck_mm", 0.0) + slab_mm if start_mm is None else start_mm
    stop = float(np.max(component["voxels_dist"])) if stop_mm is None else stop_mm

    result = {"distance_mm": None, "n_blobs": 1, "blob_sizes": [], "slab_mm": slab_mm}
    run_start, run_length = None, 0
    for position in np.arange(start, stop, band_mm):
        slab = (values >= position) & (values < position + slab_mm)
        labels, count = ndimage.label(slab, structure=np.ones((3, 3, 3), dtype=bool))
        significant = np.zeros(0, dtype=np.int64)
        if count >= 2:
            sizes = np.bincount(labels.ravel())[1:]
            significant = sizes[(sizes >= min_blob_voxels) & (sizes >= min_blob_fraction * sizes.max())]

        if significant.size < 2:
            run_start, run_length = None, 0
            continue
        if run_start is None:
            run_start = float(position)
            result.update(n_blobs=int(significant.size), blob_sizes=sorted(significant.tolist(), reverse=True))
        run_length += 1
        if run_length >= persistence:
            result["distance_mm"] = run_start
            return result

    result.update(n_blobs=1, blob_sizes=[])
    return result


def trace_branch(component, dist, parent, reference_image, bifurcation=None,
                 max_length_mm=MAX_TRACE_MM, step_mm=TRACE_STEP_MM, smoothing_mm=1.0):
    """Centerline of an instance from the aortic lumen out to its truncation.

    Truncates at the bifurcation or max_length_mm, whichever comes first.
    The end point is a voxel of maximum geodesic distance within the truncated
    region; among the voxels within one band of that maximum, the one deepest
    inside the lumen is taken, since the single most distant voxel usually
    sits against the vessel wall and its shortest path hugs that wall. The
    centerline follows `parent` back from there to the aortic lumen,
    Gaussian-smoothed, then resampled at step_mm of geodesic distance.

    Returns a dict with points_mm / arc_lengths_mm (resampled; arc length is
    geodesic distance), raw_path_flat / raw_path_mm / raw_dist_mm,
    traced_length_mm and truncated_by ("bifurcation", "max_length" or
    "vessel_end").
    """
    shape = component["shape"]
    spacing = reference_image.GetSpacing()
    vessel_end = float(np.max(component["voxels_dist"]))

    limit, truncated_by = max_length_mm, "max_length"
    if bifurcation is not None and bifurcation.get("distance_mm") is not None \
            and bifurcation["distance_mm"] < limit:
        limit, truncated_by = bifurcation["distance_mm"], "bifurcation"
    if vessel_end <= limit:
        limit, truncated_by = vessel_end, "vessel_end"

    low, member, dist_crop = _crop_of(component, dist)
    within = member & (dist_crop <= limit)
    depth = ndimage.distance_transform_edt(member, sampling=np.asarray(spacing)[::-1])

    reached = dist_crop[within].max()
    near_end = within & (dist_crop >= reached - BAND_MM)
    candidates_zyx = np.argwhere(near_end)
    best = candidates_zyx[np.argmax(depth[near_end])]
    end_flat = int(np.ravel_multi_index(tuple(best + low), shape))

    path = [end_flat]
    parent_flat = parent.ravel()
    while parent_flat[path[-1]] >= 0:
        path.append(int(parent_flat[path[-1]]))
    path = np.array(path[::-1], dtype=np.int64)

    path_dist = dist.ravel()[path].astype(np.float64)
    path_mm = flat_to_mm(path, reference_image, shape)
    mean_step = max(float(np.mean(np.diff(path_dist))) if path.size > 1 else step_mm, 1e-6)
    smoothed = gaussian_filter1d(path_mm, sigma=smoothing_mm / mean_step, axis=0, mode="nearest") \
        if path.size > 2 else path_mm

    arc = np.arange(0.0, path_dist[-1] + 1e-9, step_mm)
    points = np.stack([np.interp(arc, path_dist, smoothed[:, axis]) for axis in range(3)], axis=1)

    return {
        "points_mm": points,
        "arc_lengths_mm": arc,
        "raw_path_flat": path,
        "raw_path_mm": path_mm,
        "raw_dist_mm": path_dist,
        "traced_length_mm": float(path_dist[-1]),
        "truncation_mm": float(limit),
        "truncated_by": truncated_by,
    }


def _point_at_arc(trace, arc_mm):
    arc = trace["arc_lengths_mm"]
    points = trace["points_mm"]
    reached = arc_mm <= arc[-1] + 1e-9
    target = min(arc_mm, arc[-1])
    point = np.array([np.interp(target, arc, points[:, axis]) for axis in range(3)])
    return point, bool(reached)


def _tangent_at_arc(trace, arc_mm, half_window_mm=1.0):
    arc = trace["arc_lengths_mm"]
    if arc.size < 2:
        return None
    before, _ = _point_at_arc(trace, max(arc_mm - half_window_mm, 0.0))
    after, _ = _point_at_arc(trace, min(arc_mm + half_window_mm, arc[-1]))
    tangent = after - before
    norm = np.linalg.norm(tangent)
    return tangent / norm if norm > 0 else None


def estimate_ostium_candidates(component, trace, reference_image, surface_points_mm, surface_tree,
                               opening_window_mm=OSTIUM_OPENING_WINDOW_MM,
                               opening_extent_mm=OSTIUM_OPENING_EXTENT_MM):
    """Three independent ostium estimates and their pairwise disagreement.

    patch_centroid_mm (primary)
        The contact patch centroid, projected onto the aortic surface. The
        patch is exactly the set of wall voxels the vessel is fed through, so
        this is the mouth itself. The component centroid would be biased
        several mm outward, and ostium localisation is 25% of the score.
    traced_projection_mm
        The traced centerline extended back along its own first 3mm until it
        meets the aortic surface.
    max_gradient_mm
        The lumen opening: walking out along the trace, where the vessel's
        frontier cross-section narrows fastest -- the funnel of the mouth
        closing down to the vessel. Unprojected, so a mouth that opens well
        off the wall shows up as disagreement.

    For a real branch leaving a wall these land within a mm or two of each
    other; disagreement is itself a false-positive signal.
    """
    patch_centroid = np.asarray(component["patch_centroid_mm"], dtype=np.float64)
    _d, nearest = surface_tree.query(patch_centroid, k=1)
    patch_projection = surface_points_mm[int(nearest)]

    arc = trace["arc_lengths_mm"]
    points = trace["points_mm"]
    early = points[arc <= min(3.0, arc[-1])]
    outward = points[-1] - points[0]
    direction = fit_line_direction(early, outward) if early.shape[0] >= 2 else None
    if direction is None:
        traced_projection = surface_points_mm[int(surface_tree.query(points[0], k=1)[1])]
    else:
        walk = early[-1][None, :] - np.outer(np.arange(0.0, 6.0 + 1e-9, 0.25), direction)
        gaps, indices = surface_tree.query(walk, k=1)
        traced_projection = surface_points_mm[int(indices[int(np.argmin(gaps))])]

    voxel_volume = voxel_volume_mm3(reference_image)
    stop = min(opening_extent_mm, float(arc[-1]))
    positions = np.arange(0.5 * opening_window_mm, stop + 1e-9, TRACE_STEP_MM)
    if positions.size >= 3:
        areas = np.array([
            np.pi * frontier_radius_mm(component["voxels_dist"], voxel_volume, p, opening_window_mm) ** 2
            for p in positions
        ])
        opening_arc = float(positions[int(np.argmin(np.gradient(areas, TRACE_STEP_MM)))])
    else:
        opening_arc = 0.0
    max_gradient_point, _ = _point_at_arc(trace, opening_arc)

    estimates = {
        "patch_centroid_mm": patch_projection,
        "traced_projection_mm": traced_projection,
        "max_gradient_mm": max_gradient_point,
    }
    keys = list(estimates)
    pairwise = {
        f"{keys[i]}__{keys[j]}": float(np.linalg.norm(estimates[keys[i]] - estimates[keys[j]]))
        for i in range(len(keys)) for j in range(i + 1, len(keys))
    }
    values = np.array(list(pairwise.values()))
    return {
        "ostium_mm": patch_projection,
        "primary": "patch_centroid_mm",
        "estimates": estimates,
        "opening_arc_mm": opening_arc,
        "pairwise_distances_mm": pairwise,
        "max_disagreement_mm": float(values.max()),
        "mean_disagreement_mm": float(values.mean()),
    }


def _recentre_on_lumen(sampler, point_mm, direction, threshold_hu, ceiling_hu, extent_mm=5.0, step_mm=0.25):
    """Intensity-weighted centroid of the lumen cross-section through point_mm
    in the plane normal to direction.

    The cross-section is the flood's own traversal volume (the opened binary
    the vessel was found in) with the aorta removed, so neither a plane near
    the wall nor bright tissue touching the vessel in-plane can pull the
    blob outward. Weights are HU above threshold clipped at the lumen
    reference: unclipped, one calcified voxel in the plane outweighs the whole
    lumen and drags the centroid onto the vessel wall.
    """
    u, v = perpendicular_basis(direction[None, :])
    u, v = u[0], v[0]
    axis = np.arange(-extent_mm, extent_mm + 1e-9, step_mm)
    grid_u, grid_v = np.meshgrid(axis, axis, indexing="ij")
    plane = point_mm + grid_u[..., None] * u + grid_v[..., None] * v
    flat_plane = plane.reshape(-1, 3)

    hu = sampler.sample("hu", flat_plane, cval=-1e4).reshape(grid_u.shape)
    in_aorta = sampler.sample("mask", flat_plane, cval=0.0, order=0).reshape(grid_u.shape) > 0.5
    if sampler.has("traversal"):
        bright = sampler.sample("traversal", flat_plane, cval=0.0, order=0).reshape(grid_u.shape) > 0.5
    else:
        bright = hu >= threshold_hu
    lumen = bright & ~in_aorta

    labels, count = ndimage.label(lumen)
    if count == 0:
        return point_mm.copy(), 0.0, False
    centre = (grid_u.shape[0] // 2, grid_u.shape[1] // 2)
    label = labels[centre]
    if label == 0:
        filled = np.argwhere(labels > 0)
        nearest = filled[np.argmin(np.sum((filled - np.array(centre)) ** 2, axis=1))]
        label = labels[nearest[0], nearest[1]]
    blob = labels == label

    weights = np.clip(hu - threshold_hu, 0.0, max(ceiling_hu - threshold_hu, 1.0)) * blob
    total = weights.sum()
    if total <= 0:
        weights, total = blob.astype(np.float64), float(blob.sum())
    offset_u = float((weights * grid_u).sum() / total)
    offset_v = float((weights * grid_v).sum() / total)
    area_mm2 = float(blob.sum()) * step_mm * step_mm
    return point_mm + offset_u * u + offset_v * v, area_mm2, True


def _maurer_radius_at(point_mm, reference_image, hu_array, mask_arr, threshold_hu,
                      half_width_mm=8.0, upsample=MAURER_UPSAMPLE):
    """Signed Maurer distance (useImageSpacing=True) at a point, inside the
    lumen with the aorta removed, plus the boundary correction.

    Computed on a crop around the point, upsampled `upsample` times with the
    lumen re-thresholded from linearly interpolated HU. On the 0.8mm working
    grid a 1mm-radius daughter is two or three voxels across, so every voxel
    of it is a boundary voxel and the map reads 0 throughout -- which is what
    most small instances on the dev cases returned before this.

    Returns (raw_mm, corrected_mm); corrected is 0 when the point is outside
    the lumen.
    """
    spacing = np.asarray(reference_image.GetSpacing(), dtype=np.float64)
    origin = np.asarray(reference_image.GetOrigin(), dtype=np.float64)
    direction = _direction_matrix(reference_image)
    index_zyx = (((np.asarray(point_mm) - origin) @ direction) / spacing)[::-1]
    shape = np.array(hu_array.shape)
    half = np.ceil(half_width_mm / spacing[::-1]).astype(int)
    centre = np.floor(index_zyx).astype(int)
    low = np.maximum(centre - half, 0)
    high = np.minimum(centre + half + 1, shape)
    if np.any(high - low < 2):
        return 0.0, 0.0

    region = tuple(slice(l, h) for l, h in zip(low, high))
    fine = upsample * (high - low - 1) + 1
    grid = [np.linspace(l, h - 1, n) for l, h, n in zip(low, high, fine)]
    coords = np.stack(np.meshgrid(*[g - l for g, l in zip(grid, low)], indexing="ij"))
    hu_fine = ndimage.map_coordinates(hu_array[region].astype(np.float32), coords, order=1)
    mask_fine = ndimage.map_coordinates(mask_arr[region].astype(np.float32), coords, order=0) > 0.5
    lumen_fine = (hu_fine >= threshold_hu) & ~mask_fine

    fine_spacing = spacing / upsample
    image = sitk.GetImageFromArray(lumen_fine.astype(np.uint8))
    image.SetSpacing(tuple(fine_spacing.tolist()))
    signed = sitk.GetArrayFromImage(
        sitk.SignedMaurerDistanceMap(image, insideIsPositive=True, squaredDistance=False,
                                     useImageSpacing=True)
    )
    point_fine = (index_zyx - low) * upsample
    raw = float(ndimage.map_coordinates(signed, point_fine[:, None], order=1, mode="nearest")[0])
    fine_voxel = float(fine_spacing.mean())
    if raw <= -0.5 * fine_voxel:
        return raw, 0.0
    return raw, max(raw, 0.0) + MAURER_CORRECTION_VOXELS * fine_voxel


def extract_seed_direction_radius(component, trace, ostium_mm, reference_image, sampler, flood,
                                  mask_arr, seed_distance_mm=SEED_DISTANCE_MM,
                                  fit_length_mm=SEED_DISTANCE_MM, verbose=False, file=sys.stderr):
    """Seed point, outward direction and radius for one traced instance.

    Seed: the trace point at geodesic distance seed_distance_mm, re-centred to
    the intensity-weighted centroid of the lumen cross-section in the plane
    normal to the local direction.
    Direction: PCA line fit over trace points from 0 to fit_length_mm,
    oriented outward (away from the ostium), unit length.
    Radius: signed Maurer distance transform at the re-centred seed, cross-
    checked against sqrt(frontier_area / pi) over the same distance band.
    Disagreements over 50% are logged. The vesselness scale at the seed is
    reported as a third reading when the candidate carries vesselness.

    sampler: a lumen_evidence.VolumeSampler holding "hu" and "mask" (and
    ideally "traversal"); flood: the floodfill.robust_flood result.
    """
    threshold_hu = flood["threshold_hu"]
    seed_point, reached = _point_at_arc(trace, seed_distance_mm)
    seed_arc = min(seed_distance_mm, float(trace["arc_lengths_mm"][-1]))

    fit_points = trace["points_mm"][trace["arc_lengths_mm"] <= fit_length_mm + 1e-9]
    outward = seed_point - np.asarray(ostium_mm, dtype=np.float64)
    direction = fit_line_direction(fit_points, outward)
    if direction is None:
        norm = np.linalg.norm(outward)
        direction = outward / norm if norm > 0 else np.array([0.0, 0.0, 1.0])

    local_direction = _tangent_at_arc(trace, seed_arc)
    if local_direction is None:
        local_direction = direction
    seed, section_area_mm2, recentred = _recentre_on_lumen(
        sampler, seed_point, local_direction, threshold_hu, flood["lumen_reference_hu"]
    )

    # re-orient on the final seed: the fit is over the path, the sign is not
    if np.dot(direction, seed - np.asarray(ostium_mm)) < 0:
        direction = -direction

    raw_dt, radius_dt = _maurer_radius_at(
        seed, reference_image, sampler.array("hu"), mask_arr, threshold_hu
    )
    radius_frontier = frontier_radius_mm(
        component["voxels_dist"], voxel_volume_mm3(reference_image), seed_arc, RADIUS_WINDOW_MM
    )
    largest = max(radius_dt, radius_frontier)
    disagreement = abs(radius_dt - radius_frontier) / largest if largest > 0 else 0.0
    flagged = disagreement > RADIUS_DISAGREEMENT_LOG_FRACTION

    radius_vesselness = None
    vesselness = component.get("vesselness")
    if vesselness is not None:
        index = np.rint(sampler.to_index(seed)[0][::-1]).astype(int) - vesselness["offset_zyx"]
        if np.all(index >= 0) and np.all(index < vesselness["sigma_mm"].shape):
            sigma = float(vesselness["sigma_mm"][tuple(index)])
            radius_vesselness = sigma * VESSELNESS_RADIUS_PER_SIGMA if sigma > 0 else None

    if flagged and verbose:
        print(
            f"  radius disagreement {disagreement:.0%} on instance {component.get('instance_id')}: "
            f"distance transform {radius_dt:.2f}mm vs frontier {radius_frontier:.2f}mm",
            file=file,
        )

    return {
        "seed_mm": seed,
        "seed_before_recentring_mm": seed_point,
        "recentred": recentred,
        "reached_seed_distance": reached,
        "seed_arc_mm": seed_arc,
        "direction_xyz": direction,
        "radius_mm": radius_dt if radius_dt > 0 else radius_frontier,
        "radius_from_distance_transform_mm": radius_dt,
        "radius_distance_transform_raw_mm": raw_dt,
        "radius_from_frontier_mm": radius_frontier,
        "radius_from_vesselness_mm": radius_vesselness,
        "radius_disagreement": float(disagreement),
        "radius_disagreement_flag": bool(flagged),
        "cross_section_area_mm2": section_area_mm2,
    }
