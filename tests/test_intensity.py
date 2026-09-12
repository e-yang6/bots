import numpy as np
import SimpleITK as sitk

from src.intensity import is_contrast_enhanced, lumen_stats
from tests.synthetic import make_cylinder_with_stub


def test_lumen_stats_on_uniform_contrast_filled_lumen():
    image, mask = make_cylinder_with_stub(lumen_hu=300.0, background_hu=-1000.0, stub=False, noise_speck=False)
    stats = lumen_stats(image, mask)

    assert stats["median"] == 300.0
    assert stats["p25"] == 300.0
    assert stats["p75"] == 300.0
    assert stats["mad"] == 0.0
    assert stats["n_voxels"] > 0
    assert is_contrast_enhanced(stats) is True


def test_is_contrast_enhanced_false_below_threshold():
    image, mask = make_cylinder_with_stub(lumen_hu=40.0, background_hu=-1000.0, stub=False, noise_speck=False)
    stats = lumen_stats(image, mask)
    assert is_contrast_enhanced(stats) is False


def test_is_contrast_enhanced_true_well_above_threshold_even_if_high():
    # a much-higher-than-threshold reference (e.g. dense calcification mixed
    # into the sample) must never itself be treated as a reason to reject
    image, mask = make_cylinder_with_stub(lumen_hu=1500.0, background_hu=-1000.0, stub=False, noise_speck=False)
    stats = lumen_stats(image, mask)
    assert is_contrast_enhanced(stats) is True


def test_is_contrast_enhanced_uses_p75_when_skewed_and_median_would_reject():
    image, mask = make_cylinder_with_stub(lumen_hu=300.0, background_hu=-1000.0, stub=False, noise_speck=False)
    mask_arr = sitk.GetArrayFromImage(mask).astype(bool)
    image_arr = sitk.GetArrayFromImage(image).copy()

    # simulate a lumen sample that is mostly low-density (partial thrombus,
    # ~20 HU) with a genuinely contrast-filled minority (~300 HU): the
    # median alone would read as non-enhanced, but a skew-aware reference
    # (p75) should still recognize the contrast-filled channel
    lumen_idx = np.argwhere(mask_arr)
    rng = np.random.default_rng(0)
    low_density_share = rng.random(len(lumen_idx)) < 0.7
    for (z, y, x), is_low in zip(lumen_idx, low_density_share):
        if is_low:
            image_arr[z, y, x] = 20

    skewed_image = sitk.GetImageFromArray(image_arr)
    skewed_image.CopyInformation(image)

    stats = lumen_stats(skewed_image, mask)
    assert stats["p75"] - stats["p25"] > 150  # confirms the skew condition triggers
    assert stats["median"] < 80  # median alone would say "not enhanced"
    assert is_contrast_enhanced(stats) is True


def test_lumen_stats_falls_back_when_erosion_removes_entire_thin_mask():
    image, mask = make_cylinder_with_stub(
        radius_mm=1.0, lumen_hu=300.0, background_hu=-1000.0, stub=False, noise_speck=False
    )
    stats = lumen_stats(image, mask)
    assert stats["n_voxels"] > 0
