"""Test the viz export pipeline using synthetic data (no real scans needed)."""

import json
import os
import sys
import tempfile

import numpy as np
import SimpleITK as sitk
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schema import make_prediction, make_daughter, write_prediction


def _make_synthetic_cylinder(radius=12, height=80, spacing=(0.8, 0.8, 0.8)):
    """Create a synthetic cylinder mask and matching CT image.

    The cylinder runs along the z-axis with a side stub (fake branch)
    poking out at mid-height.
    """
    size = (80, 80, int(height / spacing[2]) + 20)  # x, y, z in sitk terms
    center_x, center_y = size[0] // 2, size[1] // 2

    mask_arr = np.zeros((size[2], size[1], size[0]), dtype=np.uint8)
    image_arr = np.full((size[2], size[1], size[0]), -1000, dtype=np.int16)

    for z in range(size[2]):
        for y in range(size[1]):
            for x in range(size[0]):
                dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                if dist <= radius:
                    mask_arr[z, y, x] = 1
                    image_arr[z, y, x] = 300  # contrast-enhanced HU

    # Add a side stub at mid-height
    mid_z = size[2] // 2
    stub_radius = 4
    for dz in range(-3, 4):
        for dy in range(-stub_radius, stub_radius + 1):
            for dx in range(center_x + radius, min(center_x + radius + 10, size[0])):
                z = mid_z + dz
                y = center_y + dy
                if 0 <= z < size[2] and 0 <= y < size[1] and 0 <= dx < size[0]:
                    dist = np.sqrt(dy**2 + dz**2)
                    if dist <= stub_radius:
                        image_arr[z, y, dx] = 300

    mask_sitk = sitk.GetImageFromArray(mask_arr)
    mask_sitk.SetSpacing(spacing)
    mask_sitk.SetOrigin((0.0, 0.0, 0.0))

    image_sitk = sitk.GetImageFromArray(image_arr)
    image_sitk.SetSpacing(spacing)
    image_sitk.SetOrigin((0.0, 0.0, 0.0))

    return image_sitk, mask_sitk


@pytest.fixture
def synthetic_case(tmp_path):
    """Create synthetic NIfTI files and a prediction JSON."""
    image_sitk, mask_sitk = _make_synthetic_cylinder()

    image_path = str(tmp_path / "orig1.nii")
    mask_path = str(tmp_path / "mask1.nii")
    sitk.WriteImage(image_sitk, image_path)
    sitk.WriteImage(mask_sitk, mask_path)

    pred = make_prediction("test_subject", [
        make_daughter(
            "branch_001",
            ostium_xyz_mm=[40.0, 32.0, 40.0],
            seed_xyz_mm=[44.0, 32.0, 40.0],
            radius_mm=3.2,
            direction_xyz=[1.0, 0.0, 0.0],
        ),
    ])
    pred_path = str(tmp_path / "prediction.json")
    write_prediction(pred, pred_path)

    return image_path, mask_path, pred_path, tmp_path


def test_export_produces_glb_and_scene_json(synthetic_case):
    """Export pipeline should produce a valid .glb and scene.json."""
    from viz.export_scene import export

    image_path, mask_path, pred_path, tmp_path = synthetic_case
    out_dir = str(tmp_path / "output")

    glb_path, scene_path = export(image_path, mask_path, pred_path, out_dir)

    assert os.path.exists(glb_path)
    assert os.path.exists(scene_path)
    assert glb_path.endswith(".glb")

    # GLB should have the glTF magic bytes
    with open(glb_path, "rb") as f:
        magic = f.read(4)
    assert magic == b"glTF", f"Expected glTF magic, got {magic}"

    # scene.json should be valid and contain expected fields
    with open(scene_path) as f:
        scene = json.load(f)

    assert scene["case_id"] == "test_subject"
    assert scene["mesh_file"] == "aorta.glb"
    assert len(scene["branches"]) == 1
    assert scene["branches"][0]["id"] == "branch_001"
    assert abs(scene["branches"][0]["radius_mm"] - 3.2) < 0.01
    assert len(scene["centroid_mm"]) == 3


def test_export_empty_predictions(synthetic_case):
    """Export with no branches should still produce valid outputs."""
    from viz.export_scene import export

    image_path, mask_path, _, tmp_path = synthetic_case

    # Write an empty prediction
    pred = make_prediction("test_empty", [])
    pred_path = str(tmp_path / "empty_pred.json")
    write_prediction(pred, pred_path)

    out_dir = str(tmp_path / "output_empty")
    glb_path, scene_path = export(image_path, mask_path, pred_path, out_dir)

    with open(scene_path) as f:
        scene = json.load(f)

    assert scene["case_id"] == "test_empty"
    assert len(scene["branches"]) == 0


def test_scene_json_coordinates_are_centered(synthetic_case):
    """Branch coordinates in scene.json should be offset by the mesh centroid."""
    from viz.export_scene import export

    image_path, mask_path, pred_path, tmp_path = synthetic_case
    out_dir = str(tmp_path / "output_centered")

    export(image_path, mask_path, pred_path, out_dir)

    with open(os.path.join(out_dir, "scene.json")) as f:
        scene = json.load(f)

    centroid = np.array(scene["centroid_mm"])
    branch = scene["branches"][0]

    # The original ostium was [40, 32, 40]. After centering, it should
    # be shifted by the centroid.
    original_ostium = np.array([40.0, 32.0, 40.0])
    expected_centered = original_ostium - centroid
    actual = np.array(branch["ostium"])

    np.testing.assert_allclose(actual, expected_centered, atol=0.1)
