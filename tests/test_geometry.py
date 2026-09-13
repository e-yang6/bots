import numpy as np
import SimpleITK as sitk

from src.geometry import (
    compute_centerline,
    compute_surface_normals,
    crop_to_mask_bbox,
    flag_end_caps,
    resample_isotropic,
)
from tests.synthetic import make_cylinder_with_stub


def test_vectorized_coordinates_match_simpleitk_on_an_oblique_grid():
    from src.geometry import _indices_to_physical

    image = sitk.Image([7, 8, 9], sitk.sitkFloat32)
    image.SetSpacing((0.7, 1.5, 0.8))
    image.SetOrigin((12., -37., 91.))
    rotation = sitk.Euler3DTransform()
    rotation.SetRotation(0.2, -0.3, 0.4)
    image.SetDirection(rotation.GetMatrix())
    indices = np.array([[0, 0, 0], [2, 1, 3], [3, 2, 4]])
    expected = [image.TransformIndexToPhysicalPoint(tuple(map(int, point))) for point in indices]
    np.testing.assert_allclose(_indices_to_physical(indices, image), expected, atol=1e-9)


def test_resampling_rejects_an_oversized_working_grid(monkeypatch):
    import pytest
    from src import geometry

    monkeypatch.setattr(geometry, "MAX_WORKING_VOXELS", 1)
    image = sitk.Image([4, 4, 4], sitk.sitkFloat32)
    mask = sitk.Image([4, 4, 4], sitk.sitkUInt8)
    with pytest.raises(MemoryError, match="working grid"):
        geometry.resample_isotropic(image, mask)


def test_crop_to_mask_bbox_shrinks_volume_and_keeps_mask_intact():
    image, mask = make_cylinder_with_stub(shape_zyx=(90, 60, 60), stub=False, noise_speck=False)
    cropped_image, cropped_mask = crop_to_mask_bbox(image, mask, margin_mm=5)

    assert cropped_image.GetSize() == cropped_mask.GetSize()
    assert np.prod(cropped_image.GetSize()) < np.prod(image.GetSize())
    assert sitk.GetArrayFromImage(cropped_mask).sum() == sitk.GetArrayFromImage(mask).sum()


def test_crop_to_mask_bbox_preserves_physical_coordinates():
    image, mask = make_cylinder_with_stub(shape_zyx=(90, 60, 60), stub=False, noise_speck=False)
    cropped_image, _cropped_mask = crop_to_mask_bbox(image, mask, margin_mm=5)

    # a physical point derived from the cropped grid must land on the same
    # physical location as in the original (uncropped) grid
    p_cropped = cropped_image.TransformIndexToPhysicalPoint((0, 0, 0))
    start_index_in_original = image.TransformPhysicalPointToIndex(p_cropped)
    p_original = image.TransformIndexToPhysicalPoint(start_index_in_original)
    assert np.allclose(p_cropped, p_original, atol=1e-6)


def test_resample_isotropic_produces_isotropic_spacing_and_returns_original_grid():
    image, mask = make_cylinder_with_stub(
        shape_zyx=(90, 60, 60), spacing=(0.7, 0.7, 1.2), stub=False, noise_speck=False
    )
    cropped_image, cropped_mask = crop_to_mask_bbox(image, mask, margin_mm=5)
    resampled_image, resampled_mask, original_grid = resample_isotropic(
        cropped_image, cropped_mask, target_spacing=0.5
    )

    assert np.allclose(resampled_image.GetSpacing(), (0.5, 0.5, 0.5))
    assert resampled_image.GetSize() == resampled_mask.GetSize()
    assert original_grid is cropped_image
    # physical extent should be roughly preserved despite the different discretization
    orig_extent = np.array(cropped_image.GetSize()) * np.array(cropped_image.GetSpacing())
    new_extent = np.array(resampled_image.GetSize()) * np.array(resampled_image.GetSpacing())
    assert np.allclose(orig_extent, new_extent, atol=1.0)


def test_compute_centerline_drops_noise_keeps_genuine_second_segment():
    image, mask = make_cylinder_with_stub(
        shape_zyx=(110, 60, 60), z_start=10, z_end=80, stub=False, noise_speck=True, second_segment=True
    )
    cropped_image, cropped_mask = crop_to_mask_bbox(image, mask, margin_mm=5)
    result = compute_centerline(cropped_mask)

    kept_labels = {c["label"] for c in result["kept"]}
    dropped_labels = {d["label"] for d in result["dropped"]}

    assert len(kept_labels) == 2  # main tube + genuine second segment
    assert len(dropped_labels) == 1  # the noise speck
    dropped_volume = result["dropped"][0]["volume_mm3"]
    smallest_kept_volume = min(c["volume_mm3"] for c in result["kept"])
    assert dropped_volume < smallest_kept_volume * 0.05

    main = max(result["kept"], key=lambda c: c["volume_mm3"])
    assert main["points_mm"].shape[0] >= 4
    assert main["tangents_mm"].shape == main["points_mm"].shape
    # tangent should be predominantly along z for this vertical tube
    assert abs(main["tangents_mm"][len(main["tangents_mm"]) // 2][2]) > 0.9


def test_compute_centerline_single_component_no_drops():
    image, mask = make_cylinder_with_stub(stub=False, noise_speck=False, second_segment=False)
    cropped_image, cropped_mask = crop_to_mask_bbox(image, mask, margin_mm=5)
    result = compute_centerline(cropped_mask)

    assert len(result["kept"]) == 1
    assert result["dropped"] == []


def test_compute_surface_normals_point_outward():
    image, mask = make_cylinder_with_stub(stub=False, noise_speck=False)
    cropped_image, cropped_mask = crop_to_mask_bbox(image, mask, margin_mm=5)
    points_mm, normals = compute_surface_normals(cropped_mask)

    assert points_mm.shape == normals.shape
    assert points_mm.shape[0] > 0
    norms = np.linalg.norm(normals, axis=1)
    # nearly every normal should be a unit vector; the only exception is a
    # handful of singular points (e.g. the exact apex of the tapered end,
    # where the distance-map gradient is symmetric and legitimately zero)
    assert np.mean(np.isclose(norms, 1.0, atol=1e-6)) > 0.95

    # for points on the cylindrical wall (away from the flat/tapered ends),
    # the outward normal should point away from the tube's central axis
    mid_mask = np.abs(points_mm[:, 2] - points_mm[:, 2].mean()) < 5
    wall_points = points_mm[mid_mask]
    wall_normals = normals[mid_mask]
    axis_xy = wall_points[:, :2].mean(axis=0)
    radial_dir = wall_points[:, :2] - axis_xy
    radial_dir /= np.linalg.norm(radial_dir, axis=1, keepdims=True)
    dot = np.einsum("ij,ij->i", wall_normals[:, :2], radial_dir)
    assert np.mean(dot > 0) > 0.9


def test_flag_end_caps_only_fires_at_component_extremes_not_partway_down_vessel():
    image, mask = make_cylinder_with_stub(
        shape_zyx=(90, 60, 60), z_start=10, z_end=80, taper_low_end=True, stub=False, noise_speck=False
    )
    cropped_image, cropped_mask = crop_to_mask_bbox(image, mask, margin_mm=5)
    centerline = compute_centerline(cropped_mask)
    normals = compute_surface_normals(cropped_mask)
    caps = flag_end_caps(cropped_mask, normals, centerline)

    assert len(caps) == 2
    ends = {c["end"] for c in caps}
    assert ends == {"low", "high"}

    mask_arr = sitk.GetArrayFromImage(cropped_mask).astype(bool)
    z_indices = np.where(mask_arr.any(axis=(1, 2)))[0]
    z_min, z_max = z_indices.min(), z_indices.max()

    for cap in caps:
        cap_z = np.where(cap["cap_region_mask"].any(axis=(1, 2)))[0]
        # the (dilated) cap region must sit at the component's own z extremes,
        # never partway down the vessel
        if cap["end"] == "low":
            assert cap_z.min() <= z_min + 4
        else:
            assert cap_z.max() >= z_max - 4


def test_flag_end_caps_distinguishes_tapered_natural_end_from_flat_cropped_end():
    image, mask = make_cylinder_with_stub(
        shape_zyx=(90, 60, 60), z_start=10, z_end=80, taper_low_end=True, stub=False, noise_speck=False
    )
    cropped_image, cropped_mask = crop_to_mask_bbox(image, mask, margin_mm=5)
    centerline = compute_centerline(cropped_mask)
    normals = compute_surface_normals(cropped_mask)
    caps = flag_end_caps(cropped_mask, normals, centerline)

    by_end = {c["end"]: c for c in caps}
    # low end tapers to a point (natural closure): radius shrinks well below threshold
    assert by_end["low"]["radius_ratio"] < 0.4
    assert by_end["low"]["likely_partial_coverage_edge"] is False
    # high end is a flat cut at full caliber (artificial crop): both features fire
    assert by_end["high"]["radius_ratio"] > 0.9
    assert by_end["high"]["likely_partial_coverage_edge"] is True


def test_geometry_pipeline_handles_oblique_direction_without_crashing():
    image, mask = make_cylinder_with_stub(
        shape_zyx=(90, 60, 60), tilt_deg=4.0, stub=False, noise_speck=False
    )
    cropped_image, cropped_mask = crop_to_mask_bbox(image, mask, margin_mm=5)
    resampled_image, resampled_mask, _grid = resample_isotropic(cropped_image, cropped_mask, target_spacing=1.0)
    centerline = compute_centerline(cropped_mask)
    normals = compute_surface_normals(cropped_mask)
    caps = flag_end_caps(cropped_mask, normals, centerline)

    assert len(centerline["kept"]) == 1
    assert len(caps) == 2
