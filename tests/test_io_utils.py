import gzip

import numpy as np
import pytest
import SimpleITK as sitk

from src.io_utils import CaseLoadError, load_case, read_volume


def _write_sitk_nii(image, path, gzip_disguised_as_plain=False):
    """Write a sitk image as a real .nii.gz then, if requested, decompress
    it and save the bytes under a plain .nii-looking name -- or the other
    way around -- so tests can exercise the gzip-vs-extension mismatch this
    dataset actually has.
    """
    tmp_gz = str(path) + ".real.nii.gz"
    sitk.WriteImage(image, tmp_gz)
    with open(tmp_gz, "rb") as f:
        raw = f.read()
    if gzip_disguised_as_plain:
        with open(path, "wb") as f:
            f.write(raw)  # gzip bytes, saved under a plain ".nii" name
    else:
        with open(path, "wb") as f:
            f.write(gzip.decompress(raw))  # plain bytes, saved as-is


def _make_small_image(value=42, size=(4, 5, 6), spacing=(1.0, 1.0, 1.0), direction=None):
    arr = np.full(size[::-1], value, dtype=np.int16)
    image = sitk.GetImageFromArray(arr)
    image.SetSpacing(spacing)
    if direction is not None:
        image.SetDirection(direction)
    return image


def test_read_volume_loads_gzip_content_disguised_with_plain_nii_extension(tmp_path):
    image = _make_small_image(value=7)
    path = tmp_path / "orig1.nii"
    _write_sitk_nii(image, path, gzip_disguised_as_plain=True)

    with open(path, "rb") as f:
        assert f.read(2) == b"\x1f\x8b"  # confirm the file really is gzip bytes

    loaded = read_volume(str(path))
    assert sitk.GetArrayFromImage(loaded).max() == 7
    assert loaded.GetSize() == image.GetSize()


def test_read_volume_loads_plain_content_with_nii_extension(tmp_path):
    image = _make_small_image(value=13)
    path = tmp_path / "orig2.nii"
    _write_sitk_nii(image, path, gzip_disguised_as_plain=False)

    with open(path, "rb") as f:
        assert f.read(2) != b"\x1f\x8b"

    loaded = read_volume(str(path))
    assert sitk.GetArrayFromImage(loaded).max() == 13


def test_load_case_succeeds_when_image_and_mask_share_grid(tmp_path):
    image = _make_small_image(value=100)
    mask = _make_small_image(value=1)

    image_path = tmp_path / "orig.nii"
    mask_path = tmp_path / "mask.nii"
    _write_sitk_nii(image, image_path)
    _write_sitk_nii(mask, mask_path)

    loaded_image, loaded_mask = load_case(str(image_path), str(mask_path))
    assert loaded_image.GetSize() == loaded_mask.GetSize()


def test_load_case_raises_on_mismatched_grid(tmp_path):
    image = _make_small_image(value=100, size=(4, 5, 6))
    mask = _make_small_image(value=1, size=(4, 5, 7))  # different size

    image_path = tmp_path / "orig.nii"
    mask_path = tmp_path / "mask.nii"
    _write_sitk_nii(image, image_path)
    _write_sitk_nii(mask, mask_path)

    with pytest.raises(CaseLoadError):
        load_case(str(image_path), str(mask_path))


def test_read_volume_falls_back_for_non_orthonormal_oblique_direction(tmp_path):
    nib = pytest.importorskip("nibabel")

    rng = np.random.default_rng(0)
    data = rng.integers(-1000, 1000, size=(6, 5, 4), dtype=np.int16)

    # a real rotation plus enough floating-point slack to trip ITK's
    # orthonormality check, mirroring the one real case that needs this path
    theta = np.radians(4.0)
    c, s = np.cos(theta), np.sin(theta)
    rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    perturbation = np.array([[0.0, 1e-3, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    affine = np.eye(4)
    affine[:3, :3] = (rotation + perturbation) * 1.5  # 1.5mm spacing
    affine[:3, 3] = [10.0, -20.0, 30.0]

    nib_img = nib.Nifti1Image(data.transpose(2, 1, 0), affine)
    path = tmp_path / "oblique.nii"
    nib.save(nib_img, str(path))

    with pytest.raises(RuntimeError):
        sitk.ReadImage(str(path))  # confirm this really does trip ITK's own reader

    loaded = read_volume(str(path))
    assert loaded.GetSize() == (4, 5, 6)

    direction = np.array(loaded.GetDirection()).reshape(3, 3)
    gram = direction.T @ direction
    assert np.allclose(gram, np.eye(3), atol=1e-6)  # exactly orthonormal now

    # RAS -> LPS: x, y should flip sign relative to the raw nibabel affine
    assert loaded.GetOrigin()[0] == pytest.approx(-10.0, abs=1e-3)
    assert loaded.GetOrigin()[1] == pytest.approx(20.0, abs=1e-3)
    assert loaded.GetOrigin()[2] == pytest.approx(30.0, abs=1e-3)
