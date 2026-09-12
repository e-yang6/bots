"""Export an aorta mask + prediction JSON into WebXR-ready assets.

Produces:
  - A .glb mesh of the aorta surface (decimated, smoothed, mobile-friendly)
  - A scene.json bundling the mesh filename, branch markers, and metadata

Usage:
  python -m viz.export_scene \
    --image orig1.nii --aorta-mask mask1.nii \
    --prediction prediction.json --output-dir viz/output/subject001
"""

import argparse
import json
import os
import sys

import numpy as np

# Add project root to path so we can import src/schema
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.io_utils import read_volume
from src.geometry import crop_to_mask_bbox
from schema import read_prediction


def mask_to_mesh(mask_sitk, target_faces=15000):
    """Convert a binary SimpleITK mask to a trimesh via marching cubes.

    Returns a trimesh.Trimesh in physical mm coordinates, decimated to
    roughly target_faces for mobile-friendly rendering.
    """
    import SimpleITK as sitk
    from skimage.measure import marching_cubes
    import trimesh

    arr = sitk.GetArrayFromImage(mask_sitk).astype(np.float32)  # z,y,x
    spacing = np.array(mask_sitk.GetSpacing())  # x,y,z
    origin = np.array(mask_sitk.GetOrigin())    # x,y,z
    direction = np.array(mask_sitk.GetDirection()).reshape(3, 3)

    # Marching cubes on the binary volume
    verts_zyx, faces, normals_zyx, _ = marching_cubes(
        arr, level=0.5, spacing=(spacing[2], spacing[1], spacing[0])
    )

    # Convert from (z,y,x) index-space-with-spacing to physical mm
    # verts_zyx already has spacing baked in from marching_cubes, but is
    # relative to array origin. Convert to (x,y,z) then apply direction+origin.
    verts_xyz = verts_zyx[:, ::-1]  # now x,y,z in mm from array corner
    verts_physical = verts_xyz @ direction.T + origin

    # Center the mesh at its own centroid for AR placement
    centroid = verts_physical.mean(axis=0)
    verts_centered = verts_physical - centroid

    mesh = trimesh.Trimesh(vertices=verts_centered, faces=faces, process=True)

    # Smooth aggressively first to remove voxel staircase artifacts,
    # then decimate. Smoothing before decimation gives much better results
    # than the reverse.
    trimesh.smoothing.filter_laplacian(mesh, iterations=30, lamb=0.5)

    # Decimate if too many faces.
    if len(mesh.faces) > target_faces:
        try:
            mesh = mesh.simplify_quadric_decimation(face_count=target_faces)
        except (ImportError, ModuleNotFoundError):
            # Fallback: voxel-based remeshing via trimesh
            pitch = mesh.extents.max() / (target_faces ** 0.5) * 2
            try:
                mesh = mesh.voxelized(pitch).marching_cubes
            except Exception:
                pass  # keep the smoothed mesh as-is

    # Final light smooth after decimation
    trimesh.smoothing.filter_laplacian(mesh, iterations=5, lamb=0.3)

    return mesh, centroid


def build_scene_json(prediction, centroid, mesh_filename):
    """Build the scene.json that the WebXR viewer loads.

    Branch coordinates are shifted by the same centroid offset applied to
    the mesh, so everything is in the same local coordinate system.
    """
    branches = []
    for d in prediction.get("daughters", []):
        ostium = np.array(d["ostium_xyz_mm"]) - centroid
        seed = np.array(d["seed_xyz_mm"]) - centroid
        direction = np.array(d["direction_xyz"])
        # Ensure unit vector
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction = direction / norm

        branches.append({
            "id": d["instance_id"],
            "ostium": ostium.tolist(),
            "seed": seed.tolist(),
            "direction": direction.tolist(),
            "radius_mm": d["radius_mm"],
        })

    return {
        "case_id": prediction.get("case_id", "unknown"),
        "mesh_file": mesh_filename,
        "centroid_mm": centroid.tolist(),
        "branches": branches,
        "scale_note": "Coordinates in mm, centered at aorta centroid",
    }


def export(image_path, mask_path, prediction_path, output_dir, target_faces=15000):
    """Full export pipeline: mask -> mesh + scene.json."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading mask: {mask_path}")
    mask_sitk = read_volume(mask_path)

    # Crop to mask region to reduce marching-cubes work
    # We only need the mask for the mesh, but crop_to_mask_bbox needs an image.
    # Load a minimal image just for cropping, or crop mask standalone.
    print("Cropping to mask region...")
    image_sitk = read_volume(image_path)
    _, cropped_mask = crop_to_mask_bbox(image_sitk, mask_sitk, margin_mm=5)
    del image_sitk  # free memory

    print("Generating mesh via marching cubes...")
    mesh, centroid = mask_to_mesh(cropped_mask, target_faces=target_faces)

    glb_filename = "aorta.glb"
    glb_path = os.path.join(output_dir, glb_filename)
    mesh.export(glb_path, file_type="glb")
    mesh_size_kb = os.path.getsize(glb_path) / 1024
    print(f"Mesh exported: {glb_path} ({mesh_size_kb:.0f} KB, {len(mesh.faces)} faces)")

    print(f"Loading prediction: {prediction_path}")
    prediction = read_prediction(prediction_path)

    scene = build_scene_json(prediction, centroid, glb_filename)
    scene_path = os.path.join(output_dir, "scene.json")
    with open(scene_path, "w") as f:
        json.dump(scene, f, indent=2)
    print(f"Scene metadata: {scene_path}")

    return glb_path, scene_path


def main():
    parser = argparse.ArgumentParser(
        description="Export aorta mask + predictions to WebXR-ready assets"
    )
    parser.add_argument("--image", required=True, help="CT image NIfTI path")
    parser.add_argument("--aorta-mask", required=True, help="Aorta mask NIfTI path")
    parser.add_argument("--prediction", required=True, help="Prediction JSON path")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--target-faces", type=int, default=15000,
                        help="Target face count for mesh decimation (default: 15000)")
    args = parser.parse_args()

    export(args.image, args.aorta_mask, args.prediction, args.output_dir, args.target_faces)


if __name__ == "__main__":
    main()
