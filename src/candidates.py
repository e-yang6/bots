"""Candidate daughter vessels: connected components of the geodesic flood.

Every direct daughter artery is continuous with the aortic lumen through
contrast, so the set the flood reaches outside the aorta,
    reachable = (dist <= budget) & ~aorta_mask,
already contains every daughter and very little else. Candidates are its
connected components (26-connectivity), each with the contact patch through
which it leaves the aorta.

Components are formed only from reachable voxels further than NECK_MM beyond
the mask, not from all of `reachable`. The supplied masks sit ~2 voxels inside
the bright lumen (floodfill's module docstring, point 1), so the first ~2mm of
reachable is a sleeve of real lumen wrapping the whole aorta, and it joins
every branch into one component: labelling `reachable` directly gives a
single component holding 97-99% of all reachable voxels on subject001 and
subject010 (1 eligible component), against 5-8 eligible components once the
sleeve is left out -- a count that holds steady for any cut from 2.0 to
3.0mm.

Each component is then given back its stem through the neck: the neck voxels
lying on a shortest path between the aortic lumen and that component (see
component_stems). Its contact patch is, as it would have been without the
neck, the component voxels adjacent to the aorta mask surface -- but now only
those of its own stem, not the whole sleeve.

Rejections, applied in order and counted separately:
  a. end_cap              the component leaves through a mask end face
  b. aortic_continuation  it touches an end cap and is aorta-sized or runs
                          along the aorta's own axis
  c. too_short            its maximum geodesic distance is under 5mm
(plus no_contact_patch, a guard for a component whose stem never reaches the
wall; it has not fired on any of the 20 dev cases).

Nothing here decides a candidate is real; that is a later stage.
"""

import sys

import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from scipy.spatial import cKDTree

from src.floodfill import SLEEVE_MM, _flood, lumen_reference_hu, robust_flood
from src.geometry import _connected_components_filtered, _indices_to_physical

NECK_MM = SLEEVE_MM
MIN_ELIGIBLE_LENGTH_MM = 5.0

# Aortic-continuation discriminators (rule b). The same numbers
# geometry.flag_end_caps uses for its own cap features.
CONTINUATION_RADIUS_FRACTION = 0.4
CONTINUATION_ANGLE_DEG = 25.0

# Fraction of a contact patch lying beyond a mask component's extreme slice
# at which the component counts as leaving through the end face (rule a).
CAP_FACE_FRACTION = 0.5

# Width of the distance window a frontier cross-section is measured over.
# Geodesic distances on a voxel grid are quantized (on a 0.8mm grid the only
# values below 2mm are 0.8, 1.13, 1.39, 1.6 and 1.93), so a single 0.5mm
# band can hold one voxel layer or two and its count jumps by ~2x from band
# to band. Averaging over 2mm smooths that out.
RADIUS_WINDOW_MM = 2.0
DIRECTION_FIT_MM = 10.0

# Detour allowed, in longest voxel edges, for a neck voxel to count as on a
# shortest path into a component (see component_stems). One edge absorbs the
# grid's tie-breaking without reaching sideways along the sleeve.
STEM_SLACK_EDGES = 1.0

VESSELNESS_SIGMAS_MM = (0.8, 1.2, 1.8, 2.6, 3.6)
# Calibrated on ideal bright cylinders at 0.8mm spacing: the sigma^2-normalized
# ObjectnessMeasure response peaks at sigma = r / 1.67-2.0 for r = 1-5mm.
VESSELNESS_RADIUS_PER_SIGMA = 1.85

REJECTION_RULES = ("no_contact_patch", "end_cap", "aortic_continuation", "too_short")

_FULL = np.ones((3, 3, 3), dtype=bool)


def flat_to_zyx(flat_indices, shape):
    return np.stack(np.unravel_index(np.asarray(flat_indices, dtype=np.int64), shape), axis=1)


def flat_to_mm(flat_indices, reference_image, shape):
    """(N,) flat indices into a (z, y, x) array -> (N, 3) physical mm."""
    zyx = flat_to_zyx(flat_indices, shape)
    return _indices_to_physical(zyx[:, ::-1].astype(np.float64), reference_image)


def voxel_volume_mm3(reference_image):
    return float(np.prod(reference_image.GetSpacing()))


def frontier_radius_mm(distances_mm, voxel_volume, at_mm, window_mm=RADIUS_WINDOW_MM):
    """Equivalent radius sqrt(area / pi) of the flood frontier at at_mm.

    The voxels whose geodesic distance falls in a window of width w form a
    slab across the vessel of volume area * w, so area = count * V / w.
    """
    distances_mm = np.asarray(distances_mm)
    low, high = at_mm - 0.5 * window_mm, at_mm + 0.5 * window_mm
    count = int(np.count_nonzero((distances_mm >= low) & (distances_mm < high)))
    area_mm2 = count * voxel_volume / window_mm
    return float(np.sqrt(area_mm2 / np.pi))


def fit_line_direction(points_mm, outward_reference=None):
    """Unit principal direction of a point set, oriented along outward_reference."""
    points_mm = np.asarray(points_mm, dtype=np.float64)
    if points_mm.shape[0] < 2:
        return None
    centred = points_mm - points_mm.mean(axis=0)
    _u, _s, vh = np.linalg.svd(centred, full_matrices=False)
    direction = vh[0]
    if outward_reference is not None and np.dot(direction, outward_reference) < 0:
        direction = -direction
    return direction / np.linalg.norm(direction)


def compute_exit_voxels(parent, reachable, aorta_mask):
    """For each reachable voxel, the flat index of the voxel through which its
    shortest path left the aorta: the first voxel on its parent chain whose
    own parent lies inside the aorta mask. -1 outside `reachable`.

    Vectorized by pointer jumping: every voxel points at its parent, exits
    point at themselves, and repeatedly replacing each pointer with its
    pointer's pointer converges in log2(longest path) rounds.
    """
    exit_volume = np.full(parent.shape, -1, dtype=np.int32)
    flat = np.flatnonzero(reachable)
    if flat.size == 0:
        return exit_volume

    parents = parent.ravel()[flat]
    is_exit = (parents < 0) | aorta_mask.ravel()[np.maximum(parents, 0)]
    position = np.clip(np.searchsorted(flat, parents), 0, flat.size - 1)
    links_within = ~is_exit & (flat[position] == parents)

    pointer = np.arange(flat.size)
    pointer[links_within] = position[links_within]
    while True:
        jumped = pointer[pointer]
        if np.array_equal(jumped, pointer):
            break
        pointer = jumped

    exit_volume.ravel()[flat] = flat[pointer]
    return exit_volume


def aortic_shell(mask_arr):
    """Voxels just outside the aorta mask (26-adjacent): where contact patches live."""
    return ndimage.binary_dilation(mask_arr, structure=_FULL) & ~mask_arr


def component_stems(neck, core_labels, dist, spacing, slack_edges=STEM_SLACK_EDGES):
    """Assign neck voxels to the component whose vessel they lead into.

    A neck voxel belongs to component k's stem when it lies on a near-shortest
    path between the aortic lumen and k's core:
        dist(v) + d_core(v) - dist(f) <= slack,
    where d_core(v) is v's geodesic distance (through the neck) to the nearest
    core face voxel f, and f belongs to k. Voxels off to the side in the
    partial-volume sleeve need a detour to reach any core and are left out.

    Why not simply the voxels on the flood's own parent chains: shortest paths
    tie constantly on a voxel grid, and Dijkstra breaks every tie the same
    way, so a 12mm2 mouth is fed through as few as one or two exit voxels. The
    geodesic-tube test keeps every voxel that is on *a* shortest path, not just
    the one Dijkstra happened to record.

    Returns (stem_flat, stem_label).
    """
    core = core_labels > 0
    face = core & ndimage.binary_dilation(neck, structure=_FULL)
    if not face.any() or not neck.any():
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int32)

    edge_mm = float(np.linalg.norm(np.asarray(spacing, dtype=np.float64)))
    slack_mm = slack_edges * edge_mm
    face_depth = float(dist[face].max())
    d_core, _parent, _frontier, nearest = _flood(
        neck | face, face, np.zeros_like(neck), spacing, face_depth + slack_mm + edge_mm,
        return_nearest_source=True,
    )

    candidates = np.flatnonzero(neck & np.isfinite(d_core))
    source = nearest.ravel()[candidates]
    detour = dist.ravel()[candidates] + d_core.ravel()[candidates] - dist.ravel()[source]
    keep = detour <= slack_mm
    return candidates[keep].astype(np.int64), core_labels.ravel()[source[keep]]


def _centerline_tangent_lookup(centerline):
    points, tangents = [], []
    for component in centerline.get("kept", []):
        if len(component["points_mm"]):
            points.append(component["points_mm"])
            tangents.append(component["tangents_mm"])
    if not points:
        return None, None
    return cKDTree(np.vstack(points)), np.vstack(tangents)


def _mask_component_z_ranges(mask_image):
    cc_arr, _volumes, kept, _dropped = _connected_components_filtered(mask_image)
    ranges = {}
    for label in kept:
        z_indices = np.where((cc_arr == label).any(axis=(1, 2)))[0]
        ranges[int(label)] = (int(z_indices.min()), int(z_indices.max()))
    return ranges


def _cap_features(component, caps, z_ranges, shape, tangent_tree, tangents):
    """Which end cap (if any) the contact patch touches, how much of the patch
    lies on its end face, and the local aortic radius / centerline tangent the
    continuation rule compares against.
    """
    patch = component["patch_flat"]
    features = {
        "touches_cap": False, "cap_component_label": None, "cap_end": None,
        "cap_face_fraction": 0.0, "local_aortic_radius_mm": None,
        "radius_ratio": None, "angle_to_centerline_deg": None,
    }

    if tangent_tree is not None and component["direction_estimate"] is not None:
        _d, nearest = tangent_tree.query(component["patch_centroid_mm"], k=1)
        tangent = tangents[int(nearest)]
        norm = np.linalg.norm(tangent)
        if norm > 0:
            cos_angle = abs(float(np.dot(component["direction_estimate"], tangent / norm)))
            features["angle_to_centerline_deg"] = float(np.degrees(np.arccos(min(cos_angle, 1.0))))

    best_cap, best_count = None, 0
    for cap in caps:
        count = int(np.count_nonzero(cap["cap_region_mask"].ravel()[patch]))
        if count > best_count:
            best_cap, best_count = cap, count
    if best_cap is None:
        return features

    patch_z = flat_to_zyx(patch, shape)[:, 0]
    z_min, z_max = z_ranges.get(best_cap["component_label"], (None, None))
    if z_min is not None:
        beyond = patch_z > z_max if best_cap["end"] == "high" else patch_z < z_min
        features["cap_face_fraction"] = float(np.count_nonzero(beyond)) / patch.size

    local_radius = best_cap["local_aortic_radius_mm"]
    features.update(
        touches_cap=True,
        cap_component_label=best_cap["component_label"],
        cap_end=best_cap["end"],
        local_aortic_radius_mm=local_radius,
    )
    if local_radius and component["radius_estimate_mm"] is not None:
        features["radius_ratio"] = component["radius_estimate_mm"] / local_radius
    return features


def _radius_estimate(core_dist, voxel_volume, neck_mm, max_distance_mm):
    """Median frontier radius over the component's first few mm past the neck."""
    half = 0.5 * RADIUS_WINDOW_MM
    stop = min(MIN_ELIGIBLE_LENGTH_MM, max_distance_mm - half)
    positions = np.arange(neck_mm + half, stop + 1e-9, 0.5)
    if positions.size == 0:
        span = max(max_distance_mm - neck_mm, 1e-6)
        return frontier_radius_mm(core_dist, voxel_volume, neck_mm + 0.5 * span, window_mm=span)
    return float(np.median([frontier_radius_mm(core_dist, voxel_volume, p) for p in positions]))


def _reject(component, min_length_mm):
    if component["n_patch_voxels"] == 0:
        return "no_contact_patch", "empty contact patch"

    if component["touches_cap"] and component["cap_face_fraction"] >= CAP_FACE_FRACTION:
        return "end_cap", (
            f"{component['cap_face_fraction']:.0%} of patch on the {component['cap_end']} end face"
        )

    if component["touches_cap"]:
        ratio = component["radius_ratio"]
        angle = component["angle_to_centerline_deg"]
        too_wide = ratio is not None and ratio > CONTINUATION_RADIUS_FRACTION
        aligned = angle is not None and angle <= CONTINUATION_ANGLE_DEG
        if too_wide or aligned:
            reasons = []
            if too_wide:
                reasons.append(f"radius {ratio:.2f}x local aorta")
            if aligned:
                reasons.append(f"{angle:.0f}deg from centerline")
            return "aortic_continuation", ", ".join(reasons)

    if component["max_distance_mm"] < min_length_mm:
        return "too_short", f"max distance {component['max_distance_mm']:.1f}mm"

    return None, None


def compute_vesselness(hu_array, component, reference_image, lumen_reference, sigmas_mm=VESSELNESS_SIGMAS_MM):
    """Multi-scale Frangi objectness in a crop around one component only.

    Run over the whole volume this dominates the runtime budget; restricted to
    each surviving component's bounding box (plus enough margin for the
    largest Gaussian) it is a few hundred thousand voxels at most. It is a
    reported feature and a radius cross-check, never a gate.

    Intensity is divided by the lumen reference and clipped to [-0.2, 1], so
    bone and calcium read as a flat plateau at lumen level instead of as the
    brightest tubes in the volume.

    Returns {"offset_zyx", "response", "sigma_mm"} crops (response is
    sigma^2-normalized so scales compare) plus summary features.
    """
    shape = hu_array.shape
    spacing = np.asarray(reference_image.GetSpacing(), dtype=np.float64)
    margin = int(np.ceil(2.0 * max(sigmas_mm) / spacing.min()))
    zyx = flat_to_zyx(component["voxels_flat"], shape)
    low = np.maximum(zyx.min(axis=0) - margin, 0)
    high = np.minimum(zyx.max(axis=0) + margin + 1, shape)
    crop = hu_array[low[0]:high[0], low[1]:high[1], low[2]:high[2]].astype(np.float32)

    normalized = np.clip(crop / max(float(lumen_reference), 1.0), -0.2, 1.0)
    image = sitk.GetImageFromArray(normalized)
    image.SetSpacing(tuple(spacing.tolist()))

    best = np.zeros(crop.shape, dtype=np.float32)
    best_sigma = np.zeros(crop.shape, dtype=np.float32)
    for sigma in sigmas_mm:
        smoothed = sitk.SmoothingRecursiveGaussian(image, float(sigma))
        response = sitk.GetArrayFromImage(
            sitk.ObjectnessMeasure(
                smoothed, alpha=0.5, beta=0.5, gamma=5.0, scaleObjectnessMeasure=True,
                objectDimension=1, brightObject=True,
            )
        ) * float(sigma) ** 2
        better = response > best
        best[better] = response[better]
        best_sigma[better] = sigma

    local = flat_to_zyx(component["core_flat"], shape) - low
    within = component["core_dist"] <= DIRECTION_FIT_MM
    values = best[local[within, 0], local[within, 1], local[within, 2]]
    sigmas = best_sigma[local[within, 0], local[within, 1], local[within, 2]]
    return {
        "offset_zyx": low,
        "response": best,
        "sigma_mm": best_sigma,
        "median_response": float(np.median(values)) if values.size else 0.0,
        "median_radius_mm": float(np.median(sigmas)) * VESSELNESS_RADIUS_PER_SIGMA if sigmas.size else 0.0,
    }


def find_candidates(
    image,
    mask,
    lumen_stats,
    centerline,
    caps,
    flood=None,
    neck_mm=NECK_MM,
    min_length_mm=MIN_ELIGIBLE_LENGTH_MM,
    compute_vesselness_features=True,
    verbose=False,
    file=sys.stderr,
):
    """Flood, split into components, apply rejections a-c.

    Arguments:
      image, mask: SimpleITK images on the same grid (the resampled pair).
      centerline:  geometry.compute_centerline(mask), for local tangents.
      caps:        geometry.flag_end_caps(...), for rules a and b.
      flood:       floodfill.robust_flood result; computed if None.

    Returns a detection dict:
      flood, neck_mm, shape, reference_image, mask_image,
      aortic_shell  bool volume of voxels 26-adjacent to the mask,
      exit_voxels   int32 volume (see compute_exit_voxels),
      core_labels   int32 volume of component labels (voxels beyond the neck),
      components    every component, each with "rejected_by" (None or a rule),
      candidates    the components that survived,
      rejections    {rule: count}.
    """
    hu = sitk.GetArrayFromImage(image)
    mask_arr = sitk.GetArrayFromImage(mask).astype(bool)
    shape = mask_arr.shape
    if flood is None:
        flood = robust_flood(image, mask, lumen_stats, verbose=verbose, file=file)

    dist = flood["dist"]
    budget = flood["budget_mm"]
    voxel_volume = voxel_volume_mm3(image)
    voxel_area = voxel_volume ** (2.0 / 3.0)

    reachable = np.isfinite(dist) & (dist <= budget) & ~mask_arr
    core = reachable & (dist > neck_mm)
    exit_voxels = compute_exit_voxels(flood["parent"], reachable, mask_arr)
    core_labels, _count = ndimage.label(core, structure=_FULL)
    core_labels = core_labels.astype(np.int32)

    dist_flat = dist.ravel()
    shell = aortic_shell(mask_arr)
    shell_flat = shell.ravel()

    objects = ndimage.find_objects(core_labels)
    core_by_label = {}
    for label_id, bbox in enumerate(objects, start=1):
        if bbox is None:
            continue
        local = np.argwhere(core_labels[bbox] == label_id) + np.array([s.start for s in bbox])
        core_by_label[label_id] = np.ravel_multi_index(local.T, shape)

    stem_flat, stem_label = component_stems(reachable & ~core, core_labels, dist, image.GetSpacing())
    order = np.argsort(stem_label, kind="stable")
    stem_flat, stem_label = stem_flat[order], stem_label[order]
    boundaries = np.searchsorted(stem_label, np.arange(len(objects) + 2))

    tangent_tree, tangents = _centerline_tangent_lookup(centerline)
    z_ranges = _mask_component_z_ranges(mask)

    components = []
    for label_id, core_flat in core_by_label.items():
        core_dist = dist_flat[core_flat]
        stem = stem_flat[boundaries[label_id]:boundaries[label_id + 1]]
        voxels = np.concatenate([stem, core_flat])
        patch = np.sort(stem[shell_flat[stem]])

        max_distance = float(core_dist.max())
        patch_mm = flat_to_mm(patch, image, shape) if patch.size else np.zeros((0, 3))
        patch_centroid = patch_mm.mean(axis=0) if patch.size else None

        radius = _radius_estimate(core_dist, voxel_volume, neck_mm, max_distance)
        near = core_dist <= min(DIRECTION_FIT_MM, max_distance)
        near_mm = flat_to_mm(core_flat[near], image, shape)
        outward = near_mm.mean(axis=0) - patch_centroid if patch_centroid is not None else None
        direction = fit_line_direction(near_mm, outward)
        if direction is None and outward is not None and np.linalg.norm(outward) > 0:
            direction = outward / np.linalg.norm(outward)

        component = {
            "label": int(label_id),
            "shape": shape,
            "neck_mm": float(neck_mm),
            "core_flat": core_flat,
            "core_dist": core_dist,
            "voxels_flat": voxels,
            "voxels_dist": dist_flat[voxels],
            "patch_flat": patch,
            "n_patch_voxels": int(patch.size),
            "patch_mm": patch_mm,
            "patch_centroid_mm": patch_centroid,
            "patch_area_mm2": float(patch.size) * voxel_area,
            "n_core_voxels": int(core_flat.size),
            "volume_mm3": float(voxels.size) * voxel_volume,
            "max_distance_mm": max_distance,
            "radius_estimate_mm": radius,
            "direction_estimate": direction,
        }
        if patch.size:
            component.update(_cap_features(component, caps, z_ranges, shape, tangent_tree, tangents))
        else:
            component.update({"touches_cap": False, "cap_face_fraction": 0.0})

        rule, detail = _reject(component, min_length_mm)
        component["rejected_by"] = rule
        component["rejection_detail"] = detail
        components.append(component)

    candidates = [c for c in components if c["rejected_by"] is None]
    candidates.sort(key=lambda c: -c["volume_mm3"])

    if compute_vesselness_features:
        reference = lumen_reference_hu(lumen_stats)
        for candidate in candidates:
            candidate["vesselness"] = compute_vesselness(hu, candidate, image, reference)

    rejections = {rule: 0 for rule in REJECTION_RULES}
    for component in components:
        if component["rejected_by"] is not None:
            rejections[component["rejected_by"]] += 1

    detection = {
        "flood": flood,
        "neck_mm": float(neck_mm),
        "shape": shape,
        "reference_image": image,
        "mask_image": mask,
        "aortic_shell": shell,
        "exit_voxels": exit_voxels,
        "core_labels": core_labels,
        "components": components,
        "candidates": candidates,
        "rejections": rejections,
    }
    if verbose:
        log_candidates(detection, file=file)
    return detection


def log_candidates(detection, file=sys.stderr):
    rejections = detection["rejections"]
    print(
        f"components={len(detection['components'])} candidates={len(detection['candidates'])} "
        f"neck={detection['neck_mm']:.1f}mm rejected: "
        + " ".join(f"{rule}={count}" for rule, count in rejections.items()),
        file=file,
    )
    # Short fragments are the bulk of the rejections and say nothing; the cap
    # rules are the ones worth seeing one by one.
    for component in detection["components"]:
        if component["rejected_by"] in ("end_cap", "aortic_continuation"):
            print(
                f"  rejected {component['rejected_by']}: label={component['label']} "
                f"len={component['max_distance_mm']:.1f}mm r~{component['radius_estimate_mm']:.1f}mm "
                f"({component['rejection_detail']})",
                file=file,
            )
