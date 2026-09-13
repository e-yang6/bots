"""Load CT image / aorta-mask NIfTI pairs into SimpleITK images.

Internal convention for this project: once loaded, all volumes are handled
as SimpleITK images (numpy views are (z, y, x) index order, matching
sitk.GetArrayFromImage). Geometry -- spacing, origin, direction -- stays on
the SimpleITK image itself; conversion to physical millimetres happens only
via SimpleITK.TransformIndexToPhysicalPoint (or an equivalent vectorized
form using the image's own spacing/origin/direction), and only when
producing final output. nibabel is never used to compute physical
coordinates for this project: NIfTI's sform/qform is RAS+, while
SimpleITK/ITK's convention is LPS, so a raw nibabel affine disagrees with
SimpleITK by a sign flip on x and y. nibabel is used here only as a rescue
path for one real file ITK's own reader refuses to load (see
_read_sitk_oblique_fallback), and even then its affine is explicitly
converted to LPS before anything downstream ever sees it.
"""

import gzip
import os
import tempfile

import numpy as np
import SimpleITK as sitk

_GZIP_MAGIC = b"\x1f\x8b"


class CaseLoadError(RuntimeError):
    """Raised when an image/mask pair cannot be loaded or is geometrically inconsistent."""


def _read_bytes_decompressed(path):
    """Return the file's raw NIfTI bytes, sniffing the gzip magic number
    from the first two bytes rather than trusting the extension: some files
    in this dataset are named .nii but are actually gzip streams (and vice
    versa is possible in principle), so extension alone is not reliable.
    """
    with open(path, "rb") as f:
        raw = f.read()
    return gzip.decompress(raw) if raw[:2] == _GZIP_MAGIC else raw


def _read_sitk(path):
    """Read a NIfTI file with SimpleITK regardless of whether its content is
    gzip-compressed or plain, independent of its extension. SimpleITK picks
    its ImageIO from the file extension, so we always hand it a temp file
    with a matching, correct extension for the sniffed content.
    """
    data = _read_bytes_decompressed(path)
    with tempfile.NamedTemporaryFile(suffix=".nii", delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        return sitk.ReadImage(tmp_path)
    finally:
        os.unlink(tmp_path)


def _read_sitk_oblique_fallback(path):
    """Rescue path for a file whose sform/qform encodes a genuine oblique
    rotation with just enough floating-point slack that ITK's NiftiImageIO
    rejects it outright ("No orthonormal definition found"). Observed on one
    real case in this dataset (direction matrix off-orthonormality ~3e-4,
    consistent with a real ~3-4 degree gantry tilt plus ordinary float
    rounding, not a data error).

    nibabel imposes no orthonormality check, so it can still read the file.
    But its affine is in NIfTI's RAS+ convention, while SimpleITK expects
    LPS -- handing SimpleITK a raw nibabel affine silently flips the sign of
    x and y. So here we:
      1. read the raw sform-derived affine via nibabel,
      2. convert it from RAS+ to LPS by negating the x and y rows,
      3. replace its (near-orthonormal but not exactly so) rotation part
         with the nearest exactly-orthonormal matrix via polar decomposition
         (SVD), since SimpleITK requires an exactly orthonormal direction.
    The resulting sitk.Image is built directly from the array plus this
    corrected spacing/origin/direction; nibabel's affine values are never
    used past this conversion.
    """
    import nibabel as nib  # imported lazily: only needed on this rescue path

    data = _read_bytes_decompressed(path)
    with tempfile.NamedTemporaryFile(suffix=".nii", delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        nib_img = nib.load(tmp_path)
        voxels_xyz = np.array(nib_img.dataobj)
        affine_ras = nib_img.affine.astype(np.float64)
        # Close the file handle before unlinking (Windows requires this)
        if hasattr(nib_img, 'uncache'):
            nib_img.uncache()
        del nib_img
    finally:
        try:
            os.unlink(tmp_path)
        except PermissionError:
            pass  # Windows file lock; temp dir will clean up

    flip_xy = np.diag([-1.0, -1.0, 1.0, 1.0])
    affine_lps = flip_xy @ affine_ras

    rot_scale = affine_lps[:3, :3]
    spacing = np.linalg.norm(rot_scale, axis=0)
    direction_raw = rot_scale / spacing
    u, _s, vt = np.linalg.svd(direction_raw)
    direction = u @ vt
    origin = affine_lps[:3, 3]

    array_zyx = np.transpose(voxels_xyz, (2, 1, 0))
    # match the source dtype family (int16 CT / uint8 mask) rather than
    # silently upcasting everything to nibabel's default float64
    if np.issubdtype(voxels_xyz.dtype, np.floating):
        array_zyx = array_zyx.astype(np.float32)
    else:
        array_zyx = np.ascontiguousarray(array_zyx)

    image = sitk.GetImageFromArray(array_zyx)
    image.SetSpacing(tuple(spacing.tolist()))
    image.SetOrigin(tuple(origin.tolist()))
    image.SetDirection(tuple(direction.flatten().tolist()))
    return image


def read_volume(path):
    """Read one NIfTI volume (image or mask), handling gzip-vs-extension
    mismatches and falling back to a corrected nibabel-based reconstruction
    for oblique sforms that ITK's own reader refuses.
    """
    try:
        return _read_sitk(path)
    except RuntimeError:
        return _read_sitk_oblique_fallback(path)


def _assert_shared_grid(image, mask, image_path, mask_path):
    mismatches = []
    if image.GetSize() != mask.GetSize():
        mismatches.append(f"size: image={image.GetSize()} mask={mask.GetSize()}")
    if not np.allclose(image.GetSpacing(), mask.GetSpacing(), atol=1e-4):
        mismatches.append(f"spacing: image={image.GetSpacing()} mask={mask.GetSpacing()}")
    if not np.allclose(image.GetOrigin(), mask.GetOrigin(), atol=1e-3):
        mismatches.append(f"origin: image={image.GetOrigin()} mask={mask.GetOrigin()}")
    if not np.allclose(image.GetDirection(), mask.GetDirection(), atol=1e-4):
        mismatches.append(f"direction: image={image.GetDirection()} mask={mask.GetDirection()}")

    if mismatches:
        raise CaseLoadError(
            f"Image ({image_path}) and mask ({mask_path}) do not share the same "
            "grid/physical space:\n  " + "\n  ".join(mismatches)
        )


def load_case(image_path, mask_path):
    """Load a CT image and its aorta-only mask as SimpleITK images.

    Returns (image, mask). Raises CaseLoadError if they don't share the same
    origin, spacing, direction and size.
    """
    image = read_volume(image_path)
    mask = read_volume(mask_path)
    _assert_shared_grid(image, mask, image_path, mask_path)
    return image, mask
