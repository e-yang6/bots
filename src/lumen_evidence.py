"""Per-case lumen-likeness evidence fields, and trilinear volume sampling.

Moved verbatim out of src/candidates.py when candidate detection switched from
scoring the aorta surface to flood-fill connectivity (src/floodfill.py). The
pipeline no longer builds these fields -- build_evidence runs multi-scale
vesselness over the whole cropped volume, which the connectivity detector
avoids -- but scripts/label_case.py still shades its 3D view with the "lumen"
field, and VolumeSampler / perpendicular_basis are shared with tracing.
"""

import numpy as np
import SimpleITK as sitk
from scipy import ndimage

from src.geometry import _direction_matrix, _mm_to_voxel_radius
from src.intensity import DEFAULT_CONTRAST_HU_THRESHOLD

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

# How far the band's low shoulders may be pushed up when a case's own
# statistics say its contrast is poor. See adaptive_lumen_band.
MAX_BAND_TIGHTENING = 0.25

# Perivascular noise (as a fraction of the contrast span) at which tightening
# starts and at which it saturates. Measured across the 25 dev cases: clean
# cases sit at 0.10-0.25, while the two that blow up sit at 0.84 and 1.12 --
# i.e. their background noise is comparable to or larger than the entire
# lumen-to-tissue contrast span.
NOISE_RATIO_CLEAN = 0.20
NOISE_RATIO_SATURATED = 0.60


def adaptive_lumen_band(background_noise_hu, contrast_span_hu, reference_hu,
                        contrast_threshold_hu, band=LUMEN_BAND):
    """Raise the band's low shoulders for cases with poor contrast.

    A fixed lower shoulder admits noise in exactly the cases that can least
    afford it: where perivascular noise is a large fraction of the whole
    lumen-to-tissue span, ordinary tissue fluctuation reaches into the band
    and the surface fills with spurious local maxima.

    Two per-case statistics drive it, and the stronger one wins:
      - noise_ratio: background MAD over contrast span. This is the sharper
        discriminator on the dev set (0.10-0.25 for clean cases; 0.84 and
        1.12 for the two that produce hundreds of candidates).
      - contrast margin: how far the same reference intensity that
        is_contrast_enhanced tests sits above its threshold. A case that
        only just clears that bar gets tightened even if its noise looks
        unremarkable.

    Only the low shoulders move; the upper ones reject calcium and have
    nothing to do with contrast quality. Tightening is capped so a case can
    never be gated into producing no candidates at all -- silently emitting
    nothing is worse than emitting noise a later stage can filter.

    Returns (band, tightening) with tightening in [0, 1] for logging.
    """
    noise_ratio = background_noise_hu / max(contrast_span_hu, 1e-6)
    span = max(NOISE_RATIO_SATURATED - NOISE_RATIO_CLEAN, 1e-6)
    from_noise = np.clip((noise_ratio - NOISE_RATIO_CLEAN) / span, 0.0, 1.0)

    margin_ratio = (reference_hu - contrast_threshold_hu) / max(abs(reference_hu), 1e-6)
    from_margin = np.clip(1.0 - margin_ratio / 0.5, 0.0, 1.0)

    tightening = float(max(from_noise, from_margin))
    shift = tightening * MAX_BAND_TIGHTENING
    zero_low, full_low, full_high, zero_high = band
    return (zero_low + shift, full_low + shift, full_high, zero_high), tightening


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
        background_values = hu_arr[background_region]
        background_reference_hu = float(np.median(background_values))
        background_noise_hu = float(np.median(np.abs(background_values - background_reference_hu)))
    else:
        background_reference_hu = float(np.percentile(hu_arr, 40))
        background_noise_hu = float(np.percentile(hu_arr, 60) - background_reference_hu)

    contrast_span = max(lumen_reference_hu - background_reference_hu, 1.0)
    band, band_tightening = adaptive_lumen_band(
        background_noise_hu, contrast_span, lumen_reference_hu, DEFAULT_CONTRAST_HU_THRESHOLD
    )

    rel_arr = (hu_arr - background_reference_hu) / contrast_span
    lumen_arr = lumen_likeness(rel_arr, band=band)

    # Kill the partial-volume rim around bone and calcium. A voxel on the edge
    # of a vertebra ramps from soft tissue to ~1500 HU, so it necessarily
    # passes through the lumen band on the way and reads as perfect lumen.
    # Only the immediate rim is suppressed (one voxel), so a genuine branch
    # running past a calcified plaque survives.
    hyperdense = rel_arr > band[3]
    if hyperdense.any():
        rim = ndimage.binary_dilation(hyperdense, iterations=1) & ~hyperdense
        lumen_arr[rim] = 0.0

    # Vesselness is computed on the intensity field clipped to the top of the
    # lumen band, so calcium and bone read as a flat plateau rather than as
    # the brightest tubes in the volume.
    clipped_arr = np.clip(rel_arr, -0.2, band[2])
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
        "vessel_hu_threshold": background_reference_hu + band[0] * contrast_span,
        "background_noise_hu": background_noise_hu,
        "noise_ratio": background_noise_hu / contrast_span,
        "lumen_band": band,
        "band_tightening": band_tightening,
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
