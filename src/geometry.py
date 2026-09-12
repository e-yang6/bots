"""Geometric preprocessing shared by later branch-detection stages.

Everything here operates on SimpleITK images. Internally, numpy views of
those images are always (z, y, x) index order (sitk.GetArrayFromImage's
layout); physical millimetre coordinates are only produced via each image's
own spacing/origin/direction (see _indices_to_physical), matching what
SimpleITK.TransformIndexToPhysicalPoint would return.
"""

import numpy as np
import SimpleITK as sitk
from scipy.interpolate import splev, splprep
from scipy.spatial import cKDTree

# Connected-component volume, as a fraction of the largest component's
# volume, below which a component is treated as segmentation noise rather
# than a genuine (possibly interrupted) piece of the aorta. See
# _connected_components_filtered for how this was chosen.
MIN_COMPONENT_VOLUME_FRACTION = 0.01


def _direction_matrix(image):
    return np.array(image.GetDirection()).reshape(3, 3)


def _indices_to_physical(indices_xyz, image):
    """Vectorized equivalent of image.TransformIndexToPhysicalPoint for an
    (N, 3) array of (x, y, z) index coordinates -> (N, 3) physical mm.
    """
    spacing = np.array(image.GetSpacing())
    origin = np.array(image.GetOrigin())
    direction = _direction_matrix(image)
    return (indices_xyz * spacing) @ direction.T + origin


def _mm_to_voxel_radius(margin_mm, spacing_xyz):
    return [max(1, int(round(margin_mm / s))) for s in spacing_xyz]


def crop_to_mask_bbox(image, mask, margin_mm=15):
    """Dilate the mask by margin_mm and crop both volumes to its bounding box.

    Must run BEFORE resample_isotropic: at least one case in this dataset is
    large enough in-plane (e.g. 512x512x207) that resampling the full image
    to isotropic spacing first -- before cropping away everything outside
    the aorta's neighbourhood -- would multiply that volume several times
    over and risk exceeding an 8GB memory budget.
    """
    mask_u8 = sitk.Cast(mask, sitk.sitkUInt8)
    radius_voxels = _mm_to_voxel_radius(margin_mm, mask.GetSpacing())
    dilated = sitk.BinaryDilate(mask_u8, radius_voxels, sitk.sitkBall)

    dilated_arr = sitk.GetArrayFromImage(dilated)  # z, y, x
    nz = np.argwhere(dilated_arr > 0)
    if nz.size == 0:
        raise ValueError("mask is empty; cannot compute a bounding box")

    zmin, ymin, xmin = nz.min(axis=0)
    zmax, ymax, xmax = nz.max(axis=0)

    start = (int(xmin), int(ymin), int(zmin))
    size = (int(xmax - xmin + 1), int(ymax - ymin + 1), int(zmax - zmin + 1))

    cropped_image = sitk.RegionOfInterest(image, size, start)
    cropped_mask = sitk.RegionOfInterest(mask, size, start)
    return cropped_image, cropped_mask


def resample_isotropic(image, mask, target_spacing=0.8):
    """Resample image (linear) and mask (nearest-neighbor) to isotropic
    spacing. Must run AFTER crop_to_mask_bbox, for the memory reason noted
    there.

    Returns (resampled_image, resampled_mask, original_grid), where
    original_grid is the pre-resampling `image` (carrying its own spacing/
    origin/direction and TransformPhysicalPointToIndex/
    TransformIndexToPhysicalPoint). Resampling changes only the voxel
    discretization, not the shared physical coordinate frame -- a physical
    point computed from the resampled grid is already correct -- but
    downstream code that wants to sample the original, native-resolution
    array at a point found on the resampled grid (e.g. for a more accurate
    intensity read near a candidate ostium) needs original_grid to convert
    that physical point back into an index in the un-resampled array.
    """
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    new_spacing = [float(target_spacing)] * 3
    new_size = [
        max(1, int(round(osz * osp / target_spacing)))
        for osz, osp in zip(original_size, original_spacing)
    ]

    image_arr = sitk.GetArrayViewFromImage(image)
    background_hu = float(image_arr.min())  # per-case, never a hardcoded HU floor

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetTransform(sitk.Transform())

    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(background_hu)
    resampled_image = resampler.Execute(image)

    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    resampled_mask = resampler.Execute(mask)

    return resampled_image, resampled_mask, image


def _connected_components_filtered(mask, min_component_fraction=MIN_COMPONENT_VOLUME_FRACTION):
    """Label the mask's connected components and split them into "kept"
    (part of the aorta) vs "dropped" (segmentation noise).

    Why a relative-volume threshold, and why 1%: inspecting the real dev
    cases (see scripts/inspect_pipeline.py), most multi-component masks
    (e.g. subject010, which has 13 components) consist of one large main
    tube plus a handful of 1-80 voxel specks sitting well off to the side of
    it -- at most ~0.1-0.5% of the main component's volume, and visibly
    disconnected noise when plotted. One case (subject018) is different: its
    second component has ~4200 voxels (~14 mL), sits directly above and
    axially overlapping the main tube's upper end, and is ~10% of the main
    component's volume -- consistent with a genuinely interrupted aorta
    (e.g. a stenosis/low-contrast gap breaking the segmentation for a few
    slices) rather than noise. A 1% relative-volume cutoff cleanly separates
    the two groups across every real multi-component case inspected, with
    roughly two orders of magnitude of headroom on each side, so it is not
    finely tuned to any single case.
    """
    mask_u8 = sitk.Cast(mask, sitk.sitkUInt8)
    cc = sitk.ConnectedComponent(mask_u8)
    cc_arr = sitk.GetArrayFromImage(cc)  # z, y, x

    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(cc)
    labels = stats.GetLabels()

    if not labels:
        return cc_arr, {}, [], []

    volumes_mm3 = {label: stats.GetPhysicalSize(label) for label in labels}
    max_volume = max(volumes_mm3.values())
    threshold = max_volume * min_component_fraction

    kept = sorted((l for l in labels if volumes_mm3[l] >= threshold), key=lambda l: -volumes_mm3[l])
    dropped = sorted((l for l in labels if volumes_mm3[l] < threshold), key=lambda l: -volumes_mm3[l])

    return cc_arr, volumes_mm3, kept, dropped


def _principal_axis(points_mm):
    center = points_mm.mean(axis=0)
    centered = points_mm - center
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, np.argmax(eigvals)]
    if axis[2] < 0:  # keep a consistent "increasing" orientation along physical z
        axis = -axis
    return center, axis


def _component_centerline(component_mask_zyx, ref_image, bin_width_mm):
    """Per-slice (along the component's own principal axis) centroid
    tracking, spline-smoothed. "Slice" here means a bin perpendicular to the
    principal axis, not necessarily the acquisition (array z) axis -- this
    keeps the centerline correct even for the oblique case, where the
    acquisition axis is a few degrees off the vessel's true axis.
    """
    idx_zyx = np.argwhere(component_mask_zyx)
    if idx_zyx.shape[0] == 0:
        return np.zeros((0, 3)), np.zeros((0, 3))

    idx_xyz = idx_zyx[:, ::-1].astype(np.float64)
    points_mm = _indices_to_physical(idx_xyz, ref_image)

    center, axis = _principal_axis(points_mm)
    t = (points_mm - center) @ axis
    t_min, t_max = t.min(), t.max()

    n_bins = max(2, int(round((t_max - t_min) / bin_width_mm)) + 1)
    bin_edges = np.linspace(t_min, t_max, n_bins + 1)
    bin_idx = np.clip(np.digitize(t, bin_edges) - 1, 0, n_bins - 1)

    raw_points, raw_t = [], []
    for b in range(n_bins):
        sel = bin_idx == b
        if not np.any(sel):
            continue
        raw_points.append(points_mm[sel].mean(axis=0))
        raw_t.append(t[sel].mean())

    raw_points = np.array(raw_points)
    order = np.argsort(raw_t)
    raw_points = raw_points[order]

    if raw_points.shape[0] < 4:
        smoothed = raw_points
    else:
        k = min(3, raw_points.shape[0] - 1)
        tck, _u = splprep(raw_points.T, s=raw_points.shape[0], k=k)
        u_fine = np.linspace(0, 1, raw_points.shape[0])
        smoothed = np.array(splev(u_fine, tck)).T

    if smoothed.shape[0] < 2:
        tangents = np.zeros_like(smoothed)
    else:
        tangents = np.gradient(smoothed, axis=0)
        norms = np.linalg.norm(tangents, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        tangents = tangents / norms

    return smoothed, tangents


def compute_centerline(mask, bin_width_mm=None, min_component_fraction=MIN_COMPONENT_VOLUME_FRACTION):
    """Per-component, per-slice centroid tracking along the mask's principal
    axis, spline-smoothed. See _connected_components_filtered for how
    noise components are told apart from a genuinely interrupted aorta.

    Returns a dict:
      {
        "kept": [
          {"label": int, "volume_mm3": float,
           "points_mm": (N,3) array, "tangents_mm": (N,3) unit vectors},
          ...
        ],
        "dropped": [{"label": int, "volume_mm3": float}, ...],
      }
    """
    cc_arr, volumes_mm3, kept_labels, dropped_labels = _connected_components_filtered(
        mask, min_component_fraction
    )

    if bin_width_mm is None:
        bin_width_mm = min(mask.GetSpacing())

    kept = []
    for label in kept_labels:
        points_mm, tangents_mm = _component_centerline(cc_arr == label, mask, bin_width_mm)
        kept.append(
            {
                "label": int(label),
                "volume_mm3": float(volumes_mm3[label]),
                "points_mm": points_mm,
                "tangents_mm": tangents_mm,
            }
        )

    dropped = [{"label": int(l), "volume_mm3": float(volumes_mm3[l])} for l in dropped_labels]

    return {"kept": kept, "dropped": dropped}


def compute_surface_normals(mask):
    """Outward unit normal at each surface voxel of the mask.

    Computed from the gradient of a signed distance map (positive outside,
    negative inside), which is a standard, robust way to get an outward
    direction field without depending on local voxel-neighbourhood
    heuristics. The distance map's own gradient is taken in index-axis units
    via np.gradient (with physical spacing so magnitudes are already in
    mm), then rotated into physical (x, y, z) space through the image's
    direction matrix -- the same convention used everywhere else in this
    module, so this remains correct for the oblique case.

    Returns (points_mm, normals): both (N, 3) arrays, one row per surface
    voxel. At a singular point of the mask's shape (e.g. the exact apex of a
    tapering, cone-like closed end), the distance map's gradient is
    symmetric and legitimately zero -- there is no single well-defined
    outward direction there. Such rows are returned as the zero vector
    rather than an arbitrary unit vector; callers averaging normals over a
    region should be aware a small fraction can be exactly zero.
    """
    mask_u8 = sitk.Cast(mask, sitk.sitkUInt8)

    # SimpleITK's erosion/distance-map filters treat the array's own edge as
    # if the volume extended forever in that direction, not as background --
    # so a mask that is flat-cropped exactly at the array boundary (as
    # opposed to tapering off inside it) would never be recognized as having
    # a surface there. At least one real case has its mask spanning the
    # entire volume in z, so this isn't a hypothetical: pad with a voxel of
    # background on every side first so the true edge is always visible to
    # both filters. sitk.ConstantPad shifts the output's origin to match,
    # so physical coordinates computed from the padded image stay correct.
    pad = [1, 1, 1]
    padded = sitk.ConstantPad(mask_u8, pad, pad, 0)

    distance = sitk.SignedMaurerDistanceMap(
        padded, insideIsPositive=False, squaredDistance=False, useImageSpacing=True
    )
    distance_arr = sitk.GetArrayFromImage(distance)  # z, y, x; positive outside

    spacing = np.array(padded.GetSpacing())  # (x, y, z)
    direction = _direction_matrix(padded)

    grad_z, grad_y, grad_x = np.gradient(distance_arr, spacing[2], spacing[1], spacing[0])
    # distance increases outward, so its gradient already points outward.
    grad_index = np.stack([grad_x, grad_y, grad_z], axis=-1)  # components along index axes

    padded_arr = sitk.GetArrayFromImage(padded).astype(bool)
    eroded_arr = sitk.GetArrayFromImage(sitk.BinaryErode(padded, [1, 1, 1])).astype(bool)
    surface = padded_arr & ~eroded_arr

    idx_zyx = np.argwhere(surface)
    if idx_zyx.shape[0] == 0:
        return np.zeros((0, 3)), np.zeros((0, 3))

    grad_at_surface = grad_index[idx_zyx[:, 0], idx_zyx[:, 1], idx_zyx[:, 2]]
    normals = grad_at_surface @ direction.T
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normals = normals / norms

    idx_xyz = idx_zyx[:, ::-1].astype(np.float64)
    points_mm = _indices_to_physical(idx_xyz, padded)
    return points_mm, normals


def _cross_section_radius_mm(component_mask_zyx, z_slice, spacing_xy):
    """Approximate the vessel's cross-sectional radius at one array z-slice
    as sqrt(area / pi), area = voxel count * in-plane pixel area. This
    assumes the slice is roughly perpendicular to the vessel, which is a
    reasonable approximation for near-axial acquisitions (including the one
    mildly oblique case in this dataset, tilted only a few degrees) and is
    only used as a coarse feature here, not a final radius estimate.
    """
    voxel_count = int(component_mask_zyx[z_slice].sum())
    if voxel_count == 0:
        return 0.0
    area_mm2 = voxel_count * spacing_xy[0] * spacing_xy[1]
    return float(np.sqrt(area_mm2 / np.pi))


def flag_end_caps(
    mask,
    normals,
    tangents,
    cap_slice_count=2,
    dilate_mm=2.5,
    inward_reference_mm=8.0,
    radius_ratio_threshold=0.4,
    tangent_angle_threshold_deg=25.0,
    min_component_fraction=MIN_COMPONENT_VOLUME_FRACTION,
):
    """Flag the end-cap region of each kept mask component, and compute
    features that later stages use to tell a real branch stub apart from
    the aorta simply continuing past a cropped mask edge.

    Caps are identified as the extreme ARRAY z-slices of each mask
    CONNECTED COMPONENT (never of the full volume): several real cases have
    a mask that covers only part of the CT's z-extent (the real aorta
    continues in the image past the mask edge), so using the full volume's
    extreme slices would mislabel an ordinary cropped edge as if it were
    somewhere out in open air, and would miss it entirely as a cap when the
    component doesn't reach the volume boundary.

    For each end, this also computes (as features, not hard gates):
      - end_radius_mm: the aorta mask's own cross-sectional radius at the
        cap face itself.
      - local_aortic_radius_mm: its radius `inward_reference_mm` further
        inside, away from the cut edge.
      - radius_ratio = end_radius_mm / local_aortic_radius_mm, and whether
        it exceeds radius_ratio_threshold.
      - tangent_alignment_deg: angle between the cap face's mean outward
        normal and the local centerline tangent at that end, and whether
        it's within tangent_angle_threshold_deg.
    A cap where the vessel is still near full caliber (radius ratio above
    threshold) AND cut squarely across (small tangent angle) looks like an
    artificial crop the real aorta continues past, rather than a genuine
    anatomical terminus -- candidates found near such a cap should be
    treated with suspicion of being the continuing parent vessel, not a
    true daughter branch.

    Arguments:
      normals: the (points_mm, normal_vectors) tuple returned by
        compute_surface_normals(mask).
      tangents: the dict returned by compute_centerline(mask).

    Returns a list of cap dicts, up to two per kept component (low + high
    end; a very short component may have its single cap counted once).
    """
    cc_arr, volumes_mm3, kept_labels, _dropped = _connected_components_filtered(
        mask, min_component_fraction
    )

    spacing_xyz = mask.GetSpacing()
    spacing_xy = (spacing_xyz[0], spacing_xyz[1])
    radius_voxels = _mm_to_voxel_radius(dilate_mm, spacing_xyz)

    normal_points_mm, normal_vectors = normals
    normal_tree = cKDTree(normal_points_mm) if normal_points_mm.shape[0] else None
    centerline_by_label = {c["label"]: c for c in tangents["kept"]}

    caps = []
    for label in kept_labels:
        component_mask = cc_arr == label
        z_indices = np.where(component_mask.any(axis=(1, 2)))[0]
        if z_indices.size == 0:
            continue
        z_min, z_max = int(z_indices.min()), int(z_indices.max())

        centerline = centerline_by_label.get(label)
        points_mm = centerline["points_mm"] if centerline else np.zeros((0, 3))
        tangents_mm = centerline["tangents_mm"] if centerline else np.zeros((0, 3))

        for end, z_start, z_stop, tangent_at_end in (
            ("low", z_min, min(z_min + cap_slice_count, z_max + 1), tangents_mm[0] if len(tangents_mm) else None),
            ("high", max(z_max - cap_slice_count + 1, z_min), z_max + 1, tangents_mm[-1] if len(tangents_mm) else None),
        ):
            cap_slices = slice(z_start, z_stop)
            cap_region = np.zeros_like(component_mask)
            cap_region[cap_slices] = component_mask[cap_slices]
            if not cap_region.any():
                continue

            end_radius_mm = float(
                np.mean(
                    [
                        _cross_section_radius_mm(component_mask, z, spacing_xy)
                        for z in range(z_start, z_stop)
                    ]
                )
            )

            inward_bins = max(1, int(round(inward_reference_mm / spacing_xyz[2])))
            if end == "low":
                ref_start = min(z_stop, z_max)
                ref_stop = min(ref_start + inward_bins, z_max + 1)
            else:
                ref_stop = max(z_start, z_min + 1)
                ref_start = max(ref_stop - inward_bins, z_min)
            ref_stop = max(ref_stop, ref_start + 1)
            local_aortic_radius_mm = float(
                np.mean(
                    [
                        _cross_section_radius_mm(component_mask, z, spacing_xy)
                        for z in range(ref_start, ref_stop)
                    ]
                )
            )

            radius_ratio = end_radius_mm / local_aortic_radius_mm if local_aortic_radius_mm > 0 else 0.0

            cap_dilated = sitk.GetArrayFromImage(
                sitk.BinaryDilate(
                    sitk.GetImageFromArray(cap_region.astype(np.uint8)), radius_voxels, sitk.sitkBall
                )
            ).astype(bool)

            if normal_tree is not None and tangent_at_end is not None and np.linalg.norm(tangent_at_end) > 0:
                cap_point_idx_zyx = np.argwhere(cap_region)
                cap_point_idx_xyz = cap_point_idx_zyx[:, ::-1].astype(np.float64)
                cap_points_mm = _indices_to_physical(cap_point_idx_xyz, mask)
                # match each cap voxel to its nearest surface point's normal
                _dist, nearest = normal_tree.query(cap_points_mm, k=1)
                mean_normal = normal_vectors[nearest].mean(axis=0)
                mean_normal_norm = np.linalg.norm(mean_normal)
                if mean_normal_norm > 0:
                    mean_normal = mean_normal / mean_normal_norm
                    cos_angle = np.clip(np.dot(mean_normal, tangent_at_end), -1.0, 1.0)
                    tangent_alignment_deg = float(np.degrees(np.arccos(np.abs(cos_angle))))
                else:
                    tangent_alignment_deg = None
            else:
                tangent_alignment_deg = None

            caps.append(
                {
                    "component_label": int(label),
                    "end": end,
                    "cap_region_mask": cap_dilated,
                    "end_radius_mm": end_radius_mm,
                    "local_aortic_radius_mm": local_aortic_radius_mm,
                    "radius_ratio": radius_ratio,
                    "radius_exceeds_threshold": radius_ratio > radius_ratio_threshold,
                    "tangent_alignment_deg": tangent_alignment_deg,
                    "direction_within_threshold": (
                        tangent_alignment_deg is not None
                        and tangent_alignment_deg <= tangent_angle_threshold_deg
                    ),
                }
            )

    for cap in caps:
        cap["likely_partial_coverage_edge"] = bool(
            cap["radius_exceeds_threshold"] and cap["direction_within_threshold"]
        )

    return caps
