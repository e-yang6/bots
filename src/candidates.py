"""Candidate ostium detection: score every eligible aorta-surface point on
"does a vessel leave in this direction", then cluster the local maxima into
candidate branch origins.

Nothing here decides whether a candidate is real -- that is a later stage.
The job here is recall plus rich per-candidate evidence.
"""

import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from scipy.spatial import cKDTree

from src.geometry import _direction_matrix, _mm_to_voxel_radius

# Multi-scale vesselness sigmas in mm, covering small daughters (~1mm radius,
# e.g. a lumbar/inferior mesenteric) up to large ones (~5mm, e.g. the SMA).
VESSELNESS_SIGMAS_MM = (0.8, 1.2, 1.8, 2.6, 3.6)

# Lumen-likeness band, in units of normalized intensity where 0 is
# perivascular background and 1 is the aortic lumen reference. Contrast-filled
# daughter lumen sits near 1 (partial volume drags small vessels down, hence
# the low shoulder), while anything far ABOVE the aortic lumen is calcified
# plaque, bone or a metal artefact -- not a vessel. Without the upper
# shoulder those dominate every measure here: raw normalized intensity makes
# a 900 HU calcification look 3.7x more "lumen" than lumen, and Frangi run on
# that field reports calcified plaque as the brightest tube in the volume.
LUMEN_BAND = (0.35, 0.60, 1.35, 2.00)

# Lumen-likeness at or above which a voxel counts as vessel for binarization
# (distance transform, shell components, cross-sections, ray-cast radius).
LUMEN_THRESHOLD = 0.5


def lumen_likeness(relative_intensity, band=LUMEN_BAND):
    """Map normalized intensity to [0, 1] with a trapezoid over LUMEN_BAND."""
    zero_low, full_low, full_high, zero_high = band
    rising = np.clip((relative_intensity - zero_low) / max(full_low - zero_low, 1e-6), 0.0, 1.0)
    falling = np.clip((zero_high - relative_intensity) / max(zero_high - full_high, 1e-6), 0.0, 1.0)
    return np.minimum(rising, falling)


class VolumeSampler:
    """Trilinear sampling of named volumes at physical (mm) points.

    Holds one image's geometry and any number of co-registered arrays in
    (z, y, x) order. Physical -> continuous index uses the image's own
    direction matrix, so this stays correct for oblique cases.
    """

    def __init__(self, reference_image):
        self.origin = np.array(reference_image.GetOrigin(), dtype=np.float64)
        self.spacing = np.array(reference_image.GetSpacing(), dtype=np.float64)
        self.direction = _direction_matrix(reference_image)
        self.size_xyz = np.array(reference_image.GetSize(), dtype=np.float64)
        self._arrays = {}

    def add(self, name, array_zyx):
        self._arrays[name] = np.ascontiguousarray(np.asarray(array_zyx, dtype=np.float32))

    def has(self, name):
        return name in self._arrays

    def array(self, name):
        return self._arrays[name]

    def to_index(self, points_mm):
        points_mm = np.atleast_2d(np.asarray(points_mm, dtype=np.float64))
        return ((points_mm - self.origin) @ self.direction) / self.spacing

    def to_physical(self, index_xyz):
        index_xyz = np.atleast_2d(np.asarray(index_xyz, dtype=np.float64))
        return (index_xyz * self.spacing) @ self.direction.T + self.origin

    def sample(self, name, points_mm, cval=0.0, order=1):
        index_xyz = self.to_index(points_mm)
        coords_zyx = index_xyz[:, ::-1].T
        return ndimage.map_coordinates(
            self._arrays[name], coords_zyx, order=order, mode="constant", cval=float(cval)
        )

    def in_bounds(self, points_mm):
        index_xyz = self.to_index(points_mm)
        return np.all((index_xyz >= 0) & (index_xyz <= self.size_xyz - 1), axis=1)


def perpendicular_basis(directions):
    """Two unit vectors spanning the plane perpendicular to each direction."""
    directions = np.atleast_2d(directions)
    helper = np.tile(np.array([1.0, 0.0, 0.0]), (directions.shape[0], 1))
    nearly_parallel = np.abs(directions[:, 0]) > 0.9
    helper[nearly_parallel] = np.array([0.0, 1.0, 0.0])

    u = np.cross(directions, helper)
    u /= np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-12)
    v = np.cross(directions, u)
    v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
    return u, v


def _multiscale_vesselness(normalized_image, sigmas_mm=VESSELNESS_SIGMAS_MM):
    """Frangi-style tubular objectness, maxed over scales.

    Run on the per-case normalized intensity field (background ~0, lumen ~1)
    rather than raw HU, so the response is comparable across cases with
    different contrast timing.
    """
    best = None
    for sigma in sigmas_mm:
        smoothed = sitk.SmoothingRecursiveGaussian(normalized_image, float(sigma))
        response = sitk.ObjectnessMeasure(
            smoothed,
            alpha=0.5,
            beta=0.5,
            gamma=5.0,
            scaleObjectnessMeasure=True,
            objectDimension=1,
            brightObject=True,
        )
        best = response if best is None else sitk.Maximum(best, response)
    return best


def build_evidence(image, mask, lumen_stats, shell_mm=4.0):
    """Precompute the volumes candidate scoring and tracing both sample from.

    Returns a dict with a VolumeSampler ("sampler") carrying:
      hu          raw intensity
      rel         (HU - background) / (lumen - background); ~0 tissue, ~1 lumen
      vesselness  multi-scale tubular objectness, normalized to its own p99
      mask        aorta mask as float
      vessel_dt   signed distance (mm) inside the thresholded vessel binary
    plus the per-case reference HU values and the dilated-shell components.
    """
    image_f = sitk.Cast(image, sitk.sitkFloat32)
    mask_u8 = sitk.Cast(mask, sitk.sitkUInt8)
    mask_arr = sitk.GetArrayFromImage(mask_u8).astype(bool)
    hu_arr = sitk.GetArrayFromImage(image_f)

    # Skewed lumens (thrombus + contrast) read low at the median, so reuse the
    # same p75-vs-median choice intensity.is_contrast_enhanced makes.
    iqr = lumen_stats["p75"] - lumen_stats["p25"]
    lumen_reference_hu = lumen_stats["p75"] if iqr > 150.0 else lumen_stats["median"]

    # Perivascular tissue reference: a shell standing off the aorta wall, so
    # it samples surrounding fat/muscle rather than the lumen itself.
    spacing = mask.GetSpacing()
    near = sitk.BinaryDilate(mask_u8, _mm_to_voxel_radius(3.0, spacing), sitk.sitkBall)
    far = sitk.BinaryDilate(mask_u8, _mm_to_voxel_radius(12.0, spacing), sitk.sitkBall)
    background_region = sitk.GetArrayFromImage(far).astype(bool) & ~sitk.GetArrayFromImage(near).astype(bool)
    if background_region.any():
        background_reference_hu = float(np.median(hu_arr[background_region]))
    else:
        background_reference_hu = float(np.percentile(hu_arr, 40))

    contrast_span = max(lumen_reference_hu - background_reference_hu, 1.0)
    rel_arr = (hu_arr - background_reference_hu) / contrast_span
    lumen_arr = lumen_likeness(rel_arr)

    # Kill the partial-volume rim around bone and calcium. A voxel on the edge
    # of a vertebra ramps from soft tissue to ~1500 HU, so it necessarily
    # passes through the lumen band on the way and reads as perfect lumen.
    # Only the immediate rim is suppressed (one voxel), so a genuine branch
    # running past a calcified plaque survives.
    hyperdense = rel_arr > LUMEN_BAND[3]
    if hyperdense.any():
        rim = ndimage.binary_dilation(hyperdense, iterations=1) & ~hyperdense
        lumen_arr[rim] = 0.0

    # Vesselness is computed on the intensity field clipped to the top of the
    # lumen band, so calcium and bone read as a flat plateau rather than as
    # the brightest tubes in the volume.
    clipped_arr = np.clip(rel_arr, -0.2, LUMEN_BAND[2])
    clipped_image = sitk.GetImageFromArray(clipped_arr)
    clipped_image.CopyInformation(image_f)
    vesselness_arr = sitk.GetArrayFromImage(_multiscale_vesselness(clipped_image))

    roi = sitk.GetArrayFromImage(far).astype(bool)
    scale = float(np.percentile(vesselness_arr[roi], 99)) if roi.any() else 0.0
    if scale > 0:
        vesselness_arr = vesselness_arr / scale

    # The aorta itself is excluded from the vessel binary the radius distance
    # transform is built on. A daughter lumen is continuous with the aortic
    # lumen, so without this the distance transform near the junction reports
    # the aorta's own half-width (~5mm) as the daughter's radius.
    vessel_binary = ((lumen_arr >= LUMEN_THRESHOLD) & ~mask_arr).astype(np.uint8)
    vessel_binary_image = sitk.GetImageFromArray(vessel_binary)
    vessel_binary_image.CopyInformation(mask_u8)
    vessel_dt = sitk.SignedMaurerDistanceMap(
        vessel_binary_image, insideIsPositive=True, squaredDistance=False, useImageSpacing=True
    )

    sampler = VolumeSampler(image_f)
    sampler.add("hu", hu_arr)
    sampler.add("rel", rel_arr)
    sampler.add("lumen", lumen_arr)
    sampler.add("vesselness", vesselness_arr)
    sampler.add("mask", mask_arr.astype(np.float32))
    sampler.add("vessel_dt", sitk.GetArrayFromImage(vessel_dt))

    shell = _shell_components(mask_u8, vessel_binary.astype(bool), shell_mm, spacing)

    return {
        "sampler": sampler,
        "lumen_reference_hu": lumen_reference_hu,
        "background_reference_hu": background_reference_hu,
        "vessel_hu_threshold": background_reference_hu + LUMEN_BAND[0] * contrast_span,
        "lumen_threshold": LUMEN_THRESHOLD,
        "shell": shell,
        "reference_image": image_f,
    }


def _shell_components(mask_u8, vessel_binary, shell_mm, spacing, inner_offset_mm=1.6):
    """Bright connected components in a thin shell standing off the aorta.

    The shell starts inner_offset_mm outside the mask, not at the wall: the
    supplied mask sits slightly inside the true bright lumen, so a shell
    flush with it just picks up the partial-volume sleeve, which wraps the
    whole aorta and fuses every branch stub into one component (observed:
    a single 6885mm3 blob covering everything).

    This is the crude "something sticks out here" signal. It is kept as
    supporting evidence attached to candidates rather than used as a detector
    on its own: the shell also lights up for adjacent unrelated vessels (e.g.
    a vein running alongside) and for calcified wall, so it cannot decide
    what is a branch, only corroborate a surface-scored candidate.
    """
    outer = sitk.GetArrayFromImage(
        sitk.BinaryDilate(mask_u8, _mm_to_voxel_radius(shell_mm, spacing), sitk.sitkBall)
    ).astype(bool)
    inner = sitk.GetArrayFromImage(
        sitk.BinaryDilate(mask_u8, _mm_to_voxel_radius(inner_offset_mm, spacing), sitk.sitkBall)
    ).astype(bool)
    shell_region = outer & ~inner & vessel_binary

    labels, n_labels = ndimage.label(shell_region)
    voxel_volume = float(np.prod(spacing))
    components = []
    if n_labels:
        sizes = ndimage.sum(np.ones_like(labels), labels, index=range(1, n_labels + 1))
        for label_id, size in zip(range(1, n_labels + 1), sizes):
            components.append({"label": int(label_id), "volume_mm3": float(size) * voxel_volume})
    return {"labels": labels, "components": components, "voxel_volume_mm3": voxel_volume}


def _surface_points_outside_caps(surface_points_mm, surface_normals, caps, mask, sampler):
    """Drop surface points lying in a flagged end-cap region.

    Cap faces are where the aorta was cut by the mask/FOV, not anatomy; a
    vessel "leaving" through a cap face is the parent aorta continuing, so
    those points must not be scored as branch origins.
    """
    if surface_points_mm.shape[0] == 0:
        return surface_points_mm, surface_normals, np.zeros(0, dtype=bool)

    cap_union = None
    for cap in caps:
        region = cap["cap_region_mask"]
        cap_union = region.copy() if cap_union is None else (cap_union | region)

    if cap_union is None:
        keep = np.ones(surface_points_mm.shape[0], dtype=bool)
        return surface_points_mm, surface_normals, keep

    sampler.add("_cap", cap_union.astype(np.float32))
    in_cap = sampler.sample("_cap", surface_points_mm, cval=0.0) > 0.5
    keep = ~in_cap
    return surface_points_mm[keep], surface_normals[keep], keep


def score_surface_points(
    sampler,
    points_mm,
    normals,
    probe_start_mm=0.8,
    probe_end_mm=5.0,
    probe_step_mm=0.6,
    ring_radius_mm=3.5,
    ring_samples=8,
    lumen_threshold=LUMEN_THRESHOLD,
):
    """Score each surface point on whether a vessel leaves along its normal.

    Probes outward along the normal and combines three independent pieces of
    evidence, so no single one can carry a candidate on its own:
      continuity  - sustained (not one-off) bright lumen along the probe,
      vesselness  - Frangi tubular response along the probe,
      contrast    - probe brighter than a ring of surrounding tissue,
    minus a penalty for probes that just re-enter the aorta (concavities).
    """
    n_points = points_mm.shape[0]
    if n_points == 0:
        empty = np.zeros(0)
        return {"score": empty, "continuity": empty, "vesselness": empty,
                "contrast": empty, "inside_fraction": empty}

    distances = np.arange(probe_start_mm, probe_end_mm + 1e-9, probe_step_mm)
    u, v = perpendicular_basis(normals)
    angles = np.linspace(0.0, 2.0 * np.pi, ring_samples, endpoint=False)

    lumen_probe = np.zeros((n_points, distances.size))
    vesselness_probe = np.zeros((n_points, distances.size))
    inside_probe = np.zeros((n_points, distances.size))
    ring_lumen = np.zeros((n_points, distances.size))

    for di, distance in enumerate(distances):
        probe = points_mm + normals * distance
        lumen_probe[:, di] = sampler.sample("lumen", probe, cval=0.0)
        vesselness_probe[:, di] = sampler.sample("vesselness", probe, cval=0.0)
        inside_probe[:, di] = sampler.sample("mask", probe, cval=0.0)

        ring_accumulator = np.zeros(n_points)
        for angle in angles:
            offset = ring_radius_mm * (np.cos(angle) * u + np.sin(angle) * v)
            ring_accumulator += sampler.sample("lumen", probe + offset, cval=0.0)
        ring_lumen[:, di] = ring_accumulator / ring_samples

    continuity = np.mean(lumen_probe >= lumen_threshold, axis=1)
    vesselness = np.mean(vesselness_probe, axis=1)
    contrast = np.mean(np.clip(lumen_probe - ring_lumen, -1.0, 2.0), axis=1)
    inside_fraction = np.mean(inside_probe > 0.5, axis=1)

    score = (
        1.0 * continuity
        + 0.8 * np.clip(vesselness, 0.0, 1.5)
        + 0.6 * np.clip(contrast, 0.0, 1.5)
        - 1.5 * inside_fraction
    )

    return {
        "score": score,
        "continuity": continuity,
        "vesselness": vesselness,
        "contrast": contrast,
        "inside_fraction": inside_fraction,
        "lumen_probe": lumen_probe,
        "distances": distances,
    }


def _local_maxima(points_mm, scores, radius_mm, score_threshold):
    tree = cKDTree(points_mm)
    neighbours = tree.query_ball_point(points_mm, r=radius_mm)
    maxima = []
    for i, neighbour_idx in enumerate(neighbours):
        if scores[i] < score_threshold:
            continue
        if scores[i] >= scores[neighbour_idx].max() - 1e-12:
            maxima.append(i)
    return np.array(maxima, dtype=int)


def _single_linkage_groups(coordinates, cutoff):
    """Single-linkage grouping without pulling in scipy.cluster: union-find
    over pairs closer than cutoff. Fine at these point counts (tens).
    """
    n = coordinates.shape[0]
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    tree = cKDTree(coordinates)
    for a, b in tree.query_pairs(r=cutoff):
        union(a, b)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def cluster_maxima(points_mm, normals, scores, maxima_idx, link_mm=3.5, split_angle_deg=45.0):
    """Group local maxima into candidate ostia.

    Two stages, because position alone is too blunt: maxima are first linked
    by proximity, then any group whose outward directions fall into
    separated bundles (> split_angle_deg apart) is split again. Two genuinely
    separate origins a few mm apart on the wall -- an accessory renal beside
    the main renal, say -- stay separate instead of merging into one.
    """
    if maxima_idx.size == 0:
        return []

    maxima_points = points_mm[maxima_idx]
    maxima_normals = normals[maxima_idx]

    clusters = []
    for group in _single_linkage_groups(maxima_points, link_mm):
        group = np.asarray(group, dtype=int)
        chord_cutoff = 2.0 * np.sin(np.radians(split_angle_deg) / 2.0)
        for direction_group in _single_linkage_groups(maxima_normals[group], chord_cutoff):
            clusters.append(maxima_idx[group[np.asarray(direction_group, dtype=int)]])

    clusters.sort(key=lambda idx: -scores[idx].max())
    return clusters


def _nearest_cap_features(point_mm, caps, sampler):
    """Distance from a point to the nearest flagged cap, plus that cap's
    radius/angle discriminators from geometry.flag_end_caps.
    """
    if not caps:
        return {
            "distance_to_cap_mm": None,
            "cap_component_label": None,
            "cap_end": None,
            "cap_radius_ratio": None,
            "cap_tangent_alignment_deg": None,
            "cap_likely_partial_coverage_edge": None,
        }

    best = None
    for cap in caps:
        if "_cap_points_mm" not in cap:
            idx_zyx = np.argwhere(cap["cap_region_mask"])
            cap["_cap_points_mm"] = sampler.to_physical(idx_zyx[:, ::-1].astype(np.float64))
        cap_points = cap["_cap_points_mm"]
        if cap_points.shape[0] == 0:
            continue
        distance = float(np.min(np.linalg.norm(cap_points - point_mm, axis=1)))
        if best is None or distance < best[0]:
            best = (distance, cap)

    if best is None:
        return _nearest_cap_features(point_mm, [], sampler)

    distance, cap = best
    return {
        "distance_to_cap_mm": distance,
        "cap_component_label": cap["component_label"],
        "cap_end": cap["end"],
        "cap_radius_ratio": cap["radius_ratio"],
        "cap_tangent_alignment_deg": cap["tangent_alignment_deg"],
        "cap_likely_partial_coverage_edge": cap["likely_partial_coverage_edge"],
    }


def _shell_support(candidate_point_mm, patch_points_mm, patch_normals, shell, sampler,
                   probe_distances_mm=(1.0, 2.0, 3.0, 4.0)):
    """Largest bright shell component this candidate's patch points into.

    The shell lives strictly outside the mask, so probing has to step outward
    along the patch normals -- sampling at the patch points themselves would
    only ever land back inside the aorta.
    """
    labels = shell["labels"]
    if labels.max() == 0:
        return {"shell_component_label": None, "shell_volume_mm3": 0.0}

    probes = [candidate_point_mm[None, :]]
    for distance in probe_distances_mm:
        probes.append(patch_points_mm + patch_normals * distance)
    probe_points = np.vstack(probes)
    index_xyz = np.rint(sampler.to_index(probe_points)).astype(int)
    shape_zyx = labels.shape
    index_zyx = index_xyz[:, ::-1]
    valid = np.all((index_zyx >= 0) & (index_zyx < np.array(shape_zyx)), axis=1)
    index_zyx = index_zyx[valid]
    if index_zyx.shape[0] == 0:
        return {"shell_component_label": None, "shell_volume_mm3": 0.0}

    found = labels[index_zyx[:, 0], index_zyx[:, 1], index_zyx[:, 2]]
    found = found[found > 0]
    if found.size == 0:
        return {"shell_component_label": None, "shell_volume_mm3": 0.0}

    volumes = {c["label"]: c["volume_mm3"] for c in shell["components"]}
    best_label = max(set(found.tolist()), key=lambda l: volumes.get(l, 0.0))
    return {"shell_component_label": int(best_label), "shell_volume_mm3": float(volumes.get(best_label, 0.0))}


def find_candidate_ostia(
    image,
    mask,
    lumen_stats,
    surface,
    caps,
    evidence=None,
    score_threshold=0.55,
    maxima_radius_mm=2.5,
    patch_radius_mm=3.0,
    cluster_link_mm=3.5,
    split_angle_deg=45.0,
):
    """Find candidate branch origins on the aorta surface.

    Arguments:
      surface: the (points_mm, normals) tuple from geometry.compute_surface_normals.
      caps:    the list from geometry.flag_end_caps (cap regions are excluded
               from scoring, and cap proximity is reported per candidate).

    Returns (candidates, evidence). Each candidate carries its contact-patch
    centroid and point set, the evidence values behind it, the shell
    supporting signal, and the nearest cap's radius/angle features.
    """
    if evidence is None:
        evidence = build_evidence(image, mask, lumen_stats)
    sampler = evidence["sampler"]

    surface_points_mm, surface_normals = surface
    eligible_points, eligible_normals, _keep = _surface_points_outside_caps(
        surface_points_mm, surface_normals, caps, mask, sampler
    )

    scored = score_surface_points(
        sampler, eligible_points, eligible_normals,
        lumen_threshold=evidence["lumen_threshold"],
    )
    scores = scored["score"]

    maxima_idx = _local_maxima(eligible_points, scores, maxima_radius_mm, score_threshold)
    clusters = cluster_maxima(
        eligible_points, eligible_normals, scores, maxima_idx,
        link_mm=cluster_link_mm, split_angle_deg=split_angle_deg,
    )

    patch_tree = cKDTree(eligible_points) if eligible_points.shape[0] else None

    candidates = []
    for cluster_idx in clusters:
        peak_score = float(scores[cluster_idx].max())
        patch_members = set()
        for i in cluster_idx:
            for j in patch_tree.query_ball_point(eligible_points[i], r=patch_radius_mm):
                if scores[j] >= 0.5 * peak_score:
                    patch_members.add(j)
        patch_members = np.array(sorted(patch_members), dtype=int)
        if patch_members.size == 0:
            patch_members = cluster_idx

        patch_points = eligible_points[patch_members]
        patch_normals = eligible_normals[patch_members]
        weights = np.clip(scores[patch_members], 1e-6, None)
        centroid = np.average(patch_points, axis=0, weights=weights)

        mean_normal = np.average(patch_normals, axis=0, weights=weights)
        norm = np.linalg.norm(mean_normal)
        mean_normal = mean_normal / norm if norm > 0 else patch_normals[0]

        candidate = {
            "ostium_patch_centroid_mm": centroid,
            "patch_points_mm": patch_points,
            "patch_normals": patch_normals,
            "patch_scores": scores[patch_members],
            "outward_direction": mean_normal,
            "peak_score": peak_score,
            "n_patch_points": int(patch_members.size),
            "patch_area_mm2": float(patch_members.size) * float(np.prod(mask.GetSpacing()) ** (2.0 / 3.0)),
            "evidence": {
                "continuity": float(scored["continuity"][patch_members].mean()),
                "vesselness": float(scored["vesselness"][patch_members].mean()),
                "contrast": float(scored["contrast"][patch_members].mean()),
                "inside_fraction": float(scored["inside_fraction"][patch_members].mean()),
            },
        }
        candidate.update(_nearest_cap_features(centroid, caps, sampler))
        candidate.update(
            _shell_support(centroid, patch_points, patch_normals, evidence["shell"], sampler)
        )
        candidates.append(candidate)

    return candidates, evidence
