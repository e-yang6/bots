"""Lumen HU statistics and the contrast-enhancement gate.

Any code in this project that needs an intensity offset or histogram bin
edges must derive it from the actual per-case min/max (e.g. image_arr.min()),
never a hardcoded constant like -1024: this dataset's HU floor is not
standard -- one real case bottoms out around -9388.
"""

import numpy as np
import SimpleITK as sitk

DEFAULT_CONTRAST_HU_THRESHOLD = 80.0
SKEW_IQR_THRESHOLD_HU = 150.0


def lumen_stats(image, mask):
    """Erode the mask slightly and sample HU inside it.

    Returns {"median", "mad", "p25", "p75", "n_voxels"}. Falls back to the
    unmodified mask if the erosion leaves nothing (a very thin/small lumen),
    rather than sampling zero voxels.
    """
    mask_u8 = sitk.Cast(mask, sitk.sitkUInt8)
    eroded = sitk.BinaryErode(mask_u8, [1, 1, 1])
    eroded_arr = sitk.GetArrayFromImage(eroded).astype(bool)
    if not eroded_arr.any():
        eroded_arr = sitk.GetArrayFromImage(mask_u8).astype(bool)

    image_arr = sitk.GetArrayFromImage(image)
    values = image_arr[eroded_arr].astype(np.float64)

    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    p25 = float(np.percentile(values, 25))
    p75 = float(np.percentile(values, 75))

    return {"median": median, "mad": mad, "p25": p25, "p75": p75, "n_voxels": int(values.size)}


def is_contrast_enhanced(stats, hu_threshold=DEFAULT_CONTRAST_HU_THRESHOLD):
    """Whether the lumen looks contrast-filled.

    Returns False only if the reference intensity is below ~hu_threshold
    (never treats a HIGHER intensity as a reason to reject -- a stricter
    upper cutoff would incorrectly discard real aneurysmal/partially-
    thrombosed cases that still have a genuinely contrast-filled channel,
    just alongside unusually high-HU calcification or thrombus).

    When the lumen HU distribution is skewed (a large gap between p25 and
    p75 -- mixed low-density thrombus and high-density contrast in the same
    sampled lumen), p75 is used as the reference value instead of the
    median, since a long low-HU tail can drag the median below threshold
    even when a real contrast-filled channel is present.
    """
    iqr = stats["p75"] - stats["p25"]
    reference = stats["p75"] if iqr > SKEW_IQR_THRESHOLD_HU else stats["median"]
    return reference >= hu_threshold
