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
    stub_in_mask=True,
    sub_stub=False,
    sub_stub_radius_mm=1.5,
    sub_stub_length_mm=10.0,
    noise_speck=True,
    second_segment=False,
    lumen_hu=300.0,
    background_hu=-1000.0,
):
    """Build a synthetic (image, mask) pair as SimpleITK images.

    - Main tube: a vertical cylinder from z_start to z_end. If
      taper_low_end, the low end narrows to a point (a natural closure);
      the high end is always a flat cut (a "cropped" end).
    - stub: a smaller side branch attached partway up the tube. With
      stub_in_mask=False it appears in the image but NOT in the mask, which
      is how the real task is posed (the supplied mask is aorta-only and the
      daughters have to be found in the CT); with stub_in_mask=True it is in
      both, which is only useful for exercising geometry code.
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

    stub_region = np.zeros(shape_zyx, dtype=bool)
    if stub:
        # a true cylinder along +x, axis through (z=stub_z, y=cy), starting
        # inside the tube so its lumen is continuous with the parent's
        stub_r_vox = stub_radius_mm / spacing[0]
        stub_len_vox = int(round(stub_length_mm / spacing[0]))
        stub_x_lo = cx + int(round(r_vox * 0.3))
        stub_x_hi = stub_x_lo + stub_len_vox
        radial = (yy - cy) ** 2 + (zz - stub_z) ** 2
        stub_region = (xx >= stub_x_lo) & (xx <= stub_x_hi) & (radial <= stub_r_vox**2)
        if stub_in_mask:
            mask[stub_region] = 1

        if sub_stub:
            # a smaller vessel leaving the STUB (not the aorta) partway along
            # it, running in +y: a daughter-of-a-daughter, which must not be
            # reported as a direct aortic daughter
            sub_r_vox = sub_stub_radius_mm / spacing[0]
            sub_len_vox = int(round(sub_stub_length_mm / spacing[0]))
            sub_x = stub_x_lo + int(round(0.6 * stub_len_vox))
            sub_y_lo = cy + int(round(stub_r_vox * 0.3))
            sub_y_hi = sub_y_lo + sub_len_vox
            sub_radial = (xx - sub_x) ** 2 + (zz - stub_z) ** 2
            sub_region = (yy >= sub_y_lo) & (yy <= sub_y_hi) & (sub_radial <= sub_r_vox**2)
            stub_region = stub_region | sub_region

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

    # The image is bright wherever there is lumen, which includes the stub
    # even when the stub is deliberately absent from the mask.
    image_arr = np.full(shape_zyx, background_hu, dtype=np.float32)
    image_arr[mask.astype(bool) | stub_region] = lumen_hu

    image = sitk.GetImageFromArray(image_arr.astype(np.int16))
    mask_img = sitk.GetImageFromArray(mask)

    direction = tuple(_rotation_matrix(tilt_deg).flatten().tolist())
    for img in (image, mask_img):
        img.SetSpacing(tuple(spacing))
        img.SetOrigin((0.0, 0.0, 0.0))
        img.SetDirection(direction)

    return image, mask_img


def _distance_to_segment(points_mm, start, end):
    start = np.asarray(start, dtype=np.float64)
    segment = np.asarray(end, dtype=np.float64) - start
    length_squared = max(float(segment @ segment), 1e-12)
    t = np.clip(((points_mm - start) @ segment) / length_squared, 0.0, 1.0)
    return np.linalg.norm(points_mm - (start + t[:, None] * segment), axis=1)


def make_capsule_phantom(
    shape_zyx=(90, 64, 64),
    spacing=(1.0, 1.0, 1.0),
    aorta_radius_mm=8.0,
    aorta_centre_xy_mm=None,
    aorta_z_mm=(10.0, 80.0),
    mask_z_mm=None,
    mask_taper_mm=0.0,
    branches=(),
    blocks=(),
    lumen_hu=300.0,
    background_hu=40.0,
):
    """Aorta plus arbitrary capsule-shaped vessels, for the flood-fill detector.

    Physical coordinates are index * spacing (origin 0, identity direction),
    in (x, y, z) mm.

    - The aorta is a vertical cylinder of aorta_radius_mm about
      aorta_centre_xy_mm (default: the volume's centre), present in the image
      over aorta_z_mm.
    - The mask is the same cylinder over mask_z_mm (default: aorta_z_mm), so a
      shorter mask leaves the real aorta continuing past the mask's end face.
      mask_taper_mm narrows the mask's high end linearly to 2mm over that
      length while the image aorta stays full width.
    - branches: dicts {"start": (x, y, z), "end": (x, y, z), "radius": r} --
      capsules (a segment dilated by r) drawn in the image only, as daughters
      are in the real task.
    - blocks: dicts {"low": (x, y, z), "high": (x, y, z), "hu": value} --
      axis-aligned boxes of the given intensity (e.g. enhancing tissue).

    Returns (image, mask) as SimpleITK images (int16 / uint8).
    """
    nz, ny, nx = shape_zyx
    sx, sy, sz = spacing
    if aorta_centre_xy_mm is None:
        aorta_centre_xy_mm = ((nx // 2) * sx, (ny // 2) * sy)
    if mask_z_mm is None:
        mask_z_mm = aorta_z_mm

    zz, yy, xx = np.meshgrid(np.arange(nz) * sz, np.arange(ny) * sy, np.arange(nx) * sx, indexing="ij")
    radial = np.sqrt((xx - aorta_centre_xy_mm[0]) ** 2 + (yy - aorta_centre_xy_mm[1]) ** 2)

    aorta = (radial <= aorta_radius_mm) & (zz >= aorta_z_mm[0]) & (zz <= aorta_z_mm[1])

    mask_radius = np.full(zz.shape, aorta_radius_mm)
    if mask_taper_mm > 0:
        taper_start = mask_z_mm[1] - mask_taper_mm
        fraction = np.clip((zz - taper_start) / mask_taper_mm, 0.0, 1.0)
        mask_radius = aorta_radius_mm - fraction * (aorta_radius_mm - 2.0)
    mask = (radial <= mask_radius) & (zz >= mask_z_mm[0]) & (zz <= mask_z_mm[1])

    image = np.full(shape_zyx, background_hu, dtype=np.float32)
    for block in blocks:
        low, high = block["low"], block["high"]
        inside = (
            (xx >= low[0]) & (xx <= high[0]) & (yy >= low[1]) & (yy <= high[1])
            & (zz >= low[2]) & (zz <= high[2])
        )
        image[inside] = block["hu"]

    points = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)
    lumen = aorta.copy()
    for branch in branches:
        inside = _distance_to_segment(points, branch["start"], branch["end"]) <= branch["radius"]
        lumen |= inside.reshape(shape_zyx)
    image[lumen] = lumen_hu

    image_sitk = sitk.GetImageFromArray(image.astype(np.int16))
    mask_sitk = sitk.GetImageFromArray(mask.astype(np.uint8))
    for volume in (image_sitk, mask_sitk):
        volume.SetSpacing(tuple(float(s) for s in spacing))
        volume.SetOrigin((0.0, 0.0, 0.0))
    return image_sitk, mask_sitk
