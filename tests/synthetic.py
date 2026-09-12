"""Synthetic CT/mask volumes for testing the geometry/intensity/io pipeline
without real scan data: a vertical cylinder ("aorta") with a side stub
("daughter" branch), optionally with a tapered natural end vs a flat
cropped end, extra noise specks, and a genuinely separate second segment.
"""

import numpy as np
import SimpleITK as sitk


def _rotation_matrix(tilt_deg=0.0):
    if tilt_deg == 0.0:
        return np.eye(3)
    theta = np.radians(tilt_deg)
    c, s = np.cos(theta), np.sin(theta)
    # small tilt of the z axis towards x, keeping approx orthonormal
    return np.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ]
    )


def make_cylinder_with_stub(
    shape_zyx=(90, 60, 60),
    spacing=(1.0, 1.0, 1.0),
    tilt_deg=0.0,
    radius_mm=8.0,
    z_start=10,
    z_end=80,
    taper_low_end=True,
    stub=True,
    stub_z=45,
    stub_radius_mm=3.0,
    stub_length_mm=18.0,
    noise_speck=True,
    second_segment=False,
    lumen_hu=300.0,
    background_hu=-1000.0,
):
    """Build a synthetic (image, mask) pair as SimpleITK images.

    - Main tube: a vertical cylinder from z_start to z_end. If
      taper_low_end, the low end narrows to a point (a natural closure);
      the high end is always a flat cut (a "cropped" end).
    - stub: a smaller side branch attached partway up the tube (not used
      for detection this session -- just gives the mask a non-trivial
      principal axis / cross-section to exercise the geometry code).
    - noise_speck: a handful of isolated foreground voxels far from the
      tube, to exercise connected-component noise filtering.
    - second_segment: an extra, substantial, separate tube segment placed
      a few slices above z_end (with a gap), to exercise the "genuinely
      interrupted aorta" keep-path.

    Returns (image, mask) as SimpleITK images (image dtype int16, mask uint8).
    """
    nz, ny, nx = shape_zyx
    cz, cy, cx = ny // 2, ny // 2, nx // 2  # placeholder, fixed below
    cy, cx = ny // 2, nx // 2

    zz, yy, xx = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij")

    mask = np.zeros(shape_zyx, dtype=np.uint8)

    r_vox = radius_mm / spacing[0]
    dist2 = (yy - cy) ** 2 + (xx - cx) ** 2

    if taper_low_end:
        taper_zone = 8
        local_r = np.clip((zz - z_start) / taper_zone, 0.0, 1.0) * r_vox
    else:
        local_r = np.full(shape_zyx, r_vox)

    in_z = (zz >= z_start) & (zz <= z_end)
    tube = in_z & (dist2 <= local_r**2)
    mask[tube] = 1

    if stub:
        stub_r_vox = stub_radius_mm / spacing[0]
        stub_len_vox = int(round(stub_length_mm / spacing[0]))
        stub_z_lo = stub_z - 2
        stub_z_hi = stub_z + 2
        stub_x_lo = cx + int(round(r_vox * 0.3))
        stub_x_hi = stub_x_lo + stub_len_vox
        in_stub_z = (zz >= stub_z_lo) & (zz <= stub_z_hi)
        in_stub_x = (xx >= stub_x_lo) & (xx <= stub_x_hi)
        stub_dist2 = (yy - cy) ** 2
        stub_region = in_stub_z & in_stub_x & (stub_dist2 <= stub_r_vox**2)
        mask[stub_region] = 1

    if noise_speck:
        speck_z, speck_y, speck_x = min(nz - 2, z_start + 3), 4, 4
        mask[speck_z : speck_z + 1, speck_y : speck_y + 2, speck_x : speck_x + 2] = 1

    if second_segment:
        gap = 3
        seg_z_start = z_end + gap
        seg_z_end = min(nz - 1, seg_z_start + 15)
        seg_dist2 = (yy - cy) ** 2 + (xx - cx) ** 2
        seg = (zz >= seg_z_start) & (zz <= seg_z_end) & (seg_dist2 <= r_vox**2)
        mask[seg] = 1

    image_arr = np.full(shape_zyx, background_hu, dtype=np.float32)
    image_arr[mask.astype(bool)] = lumen_hu

    image = sitk.GetImageFromArray(image_arr.astype(np.int16))
    mask_img = sitk.GetImageFromArray(mask)

    direction = tuple(_rotation_matrix(tilt_deg).flatten().tolist())
    for img in (image, mask_img):
        img.SetSpacing(tuple(spacing))
        img.SetOrigin((0.0, 0.0, 0.0))
        img.SetDirection(direction)

    return image, mask_img
