"""Export aorta + branch vessel meshes from CT data into WebXR-ready assets.

Two modes:
  1. Pipeline mode (--image + --aorta-mask, no --prediction):
     Runs the detection pipeline, builds marching-cubes meshes from ALL
     candidates (accepted AND rejected), exports each as a separate .glb.

  2. Legacy mode (--image + --aorta-mask + --prediction):
     Loads a prediction JSON and exports aorta mesh + marker-only scene.json.

Usage:
  python -m viz.export_scene \
    --image orig1.nii --aorta-mask mask1.nii --output-dir viz/output/subject001

  python -m viz.export_scene \
    --image orig1.nii --aorta-mask mask1.nii \
    --prediction prediction.json --output-dir viz/output/subject001

  --full-res   Skip decimation (keep Laplacian smooth only)
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


def _volume_center_mm(sitk_image):
    """Physical center of the entire volume (not just the mask voxels)."""
    size = np.array(sitk_image.GetSize(), dtype=float)     # x,y,z
    spacing = np.array(sitk_image.GetSpacing())             # x,y,z
    origin = np.array(sitk_image.GetOrigin())               # x,y,z
    direction = np.array(sitk_image.GetDirection()).reshape(3, 3)
    center_index_xyz = (size - 1.0) / 2.0 * spacing
    return center_index_xyz @ direction.T + origin


def mask_to_mesh(mask_sitk, target_faces=15000, full_res=False, centroid=None):
    """Convert a binary SimpleITK mask to a trimesh via marching cubes.

    Returns a trimesh.Trimesh in physical mm coordinates.  When full_res
    is False (default), the mesh is decimated to roughly target_faces for
    mobile-friendly rendering.  When True, decimation is skipped but
    Laplacian smoothing is still applied.

    If centroid is provided, the mesh is centered at that point (so all
    meshes sharing a centroid land in the same local frame).  Otherwise
    the volume center is used.
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
    verts_xyz = verts_zyx[:, ::-1]  # now x,y,z in mm from array corner
    verts_physical = verts_xyz @ direction.T + origin

    if centroid is None:
        centroid = _volume_center_mm(mask_sitk)
    verts_centered = verts_physical - centroid

    mesh = trimesh.Trimesh(vertices=verts_centered, faces=faces, process=True)

    # Smooth to remove voxel staircase artifacts
    trimesh.smoothing.filter_laplacian(mesh, iterations=30, lamb=0.5)

    # Decimate if not full-res
    if not full_res and len(mesh.faces) > target_faces:
        try:
            mesh = mesh.simplify_quadric_decimation(face_count=target_faces)
        except Exception:
            pass  # keep the smoothed mesh as-is; voxel fallback destroys geometry

    # Final light smooth
    trimesh.smoothing.filter_laplacian(mesh, iterations=5, lamb=0.3)

    return mesh, centroid


def voxels_to_mesh(voxels_flat, shape, image_sitk, centroid, target_faces=3000):
    """Build a marching-cubes mesh from flat voxel indices.

    voxels_flat: 1-D array of raveled indices into shape (z, y, x).
    centroid: the aorta mesh centroid (x, y, z) to subtract so branch
              meshes share the same coordinate origin.
    """
    import SimpleITK as sitk
    from skimage.measure import marching_cubes
    import trimesh

    # Reconstruct binary volume from flat indices
    volume = np.zeros(shape, dtype=np.float32)
    volume.flat[voxels_flat] = 1.0

    spacing = np.array(image_sitk.GetSpacing())    # x,y,z
    origin = np.array(image_sitk.GetOrigin())      # x,y,z
    direction = np.array(image_sitk.GetDirection()).reshape(3, 3)

    try:
        verts_zyx, faces, _, _ = marching_cubes(
            volume, level=0.5, spacing=(spacing[2], spacing[1], spacing[0])
        )
    except (ValueError, RuntimeError):
        # Too few voxels for marching cubes
        return None

    if len(faces) < 4:
        return None

    verts_xyz = verts_zyx[:, ::-1]
    verts_physical = verts_xyz @ direction.T + origin
    verts_centered = verts_physical - centroid

    mesh = trimesh.Trimesh(vertices=verts_centered, faces=faces, process=True)
    trimesh.smoothing.filter_laplacian(mesh, iterations=20, lamb=0.5)

    if len(mesh.faces) > target_faces:
        try:
            mesh = mesh.simplify_quadric_decimation(face_count=target_faces)
        except (ImportError, ModuleNotFoundError):
            pass

    trimesh.smoothing.filter_laplacian(mesh, iterations=5, lamb=0.3)
    return mesh


def build_scene_json(prediction, centroid, mesh_filename):
    """Build the scene.json that the viewer loads (legacy mode).

    Branch coordinates are shifted by the same centroid offset applied to
    the mesh, so everything is in the same local coordinate system.
    """
    branches = []
    for d in prediction.get("daughters", []):
        ostium = np.array(d["ostium_xyz_mm"]) - centroid
        seed = np.array(d["seed_xyz_mm"]) - centroid
        direction = np.array(d["direction_xyz"])
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction = direction / norm

        entry = {
            "id": d["instance_id"],
            "ostium": ostium.tolist(),
            "seed": seed.tolist(),
            "direction": direction.tolist(),
            "radius_mm": d["radius_mm"],
        }

        if "confidence" in d:
            entry["confidence"] = d["confidence"]
        if "_features" in d:
            entry["features"] = d["_features"]
        if "_score" in d:
            entry["score"] = d["_score"]

        branches.append(entry)

    return {
        "case_id": prediction.get("case_id", "unknown"),
        "mesh_file": mesh_filename,
        "centroid_mm": centroid.tolist(),
        "branches": branches,
        "scale_note": "Coordinates in mm, centered at aorta centroid",
    }


def build_pipeline_scene_json(case_id, centroid, mesh_filename, scored, kept_set):
    """Build scene.json from the pipeline's scored candidate list.

    Every scored candidate gets an entry; accepted vs rejected is indicated
    by the 'accepted' flag. Each branch that produced a mesh gets a
    'mesh_file' key. The full rule breakdown from src.rules.explain() is
    included so the viewer can show per-term penalty details.
    """
    from src.rules import CONFIDENCE_THRESHOLD

    branches = []
    for i, entry in enumerate(scored):
        instance = entry["instance"]
        ostium_mm = np.array(entry["ostium"]["ostium_mm"])
        seed_mm = np.array(entry["seed"]["seed_mm"])
        direction = np.array(entry["seed"]["direction_xyz"])
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction = direction / norm

        accepted = id(entry) in kept_set
        confidence = float(entry["score"]["confidence"])

        branch_id = f"branch_{i + 1:03d}" if accepted else f"candidate_{i + 1:03d}"
        mesh_file = f"{branch_id}.glb"

        branch = {
            "id": branch_id,
            "mesh_file": mesh_file,
            "accepted": accepted,
            "confidence": round(confidence, 4),
            "ostium": (ostium_mm - centroid).tolist(),
            "seed": (seed_mm - centroid).tolist(),
            "direction": direction.tolist(),
            "radius_mm": float(entry["seed"]["radius_mm"]),
        }

        if not accepted:
            veto = entry["score"].get("veto")
            if veto:
                branch["reject_reason"] = veto
            elif confidence < CONFIDENCE_THRESHOLD:
                branch["reject_reason"] = f"below threshold ({confidence:.2f})"

        # Full rule breakdown for the viewer's penalty detail panel
        score = entry["score"]
        branch["rule_breakdown"] = {
            "veto": score.get("veto"),
            "terms": score["terms"],
            "case_flood_leaking_penalty": score["case_flood_leaking_penalty"],
            "shares_vessel_penalty": score["shares_vessel_penalty"],
            "total_penalty": score["total_penalty"],
        }

        branches.append(branch)

    return {
        "case_id": case_id,
        "mesh_file": mesh_filename,
        "centroid_mm": centroid.tolist(),
        "branches": branches,
        "scale_note": "Coordinates in mm, centered at aorta centroid",
    }


def export(image_path, mask_path, prediction_path, output_dir,
           target_faces=15000, full_res=False):
    """Legacy export: mask -> aorta mesh + prediction-based scene.json."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading mask: {mask_path}")
    mask_sitk = read_volume(mask_path)

    print("Cropping to mask region...")
    image_sitk = read_volume(image_path)
    _, cropped_mask = crop_to_mask_bbox(image_sitk, mask_sitk, margin_mm=5)
    del image_sitk

    label = "full-res" if full_res else f"target {target_faces} faces"
    print(f"Generating mesh via marching cubes ({label})...")
    mesh, centroid = mask_to_mesh(cropped_mask, target_faces=target_faces,
                                  full_res=full_res)

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


def export_pipeline(image_path, mask_path, output_dir,
                    target_faces=15000, branch_target_faces=3000,
                    full_res=False, verbose=False):
    """Pipeline export: run detection, build meshes from all candidates."""
    import SimpleITK as sitk
    import schema
    from run import analyze_case, build_daughters

    os.makedirs(output_dir, exist_ok=True)

    # --- Run detection pipeline FIRST so we can use its resampled volumes ---
    print(f"Running detection pipeline on: {image_path}")
    context = analyze_case(image_path, mask_path, verbose=verbose)
    daughters, scored = build_daughters(context, verbose=verbose)

    # --- Aorta mesh from the pipeline's resampled mask ---
    # Must use the same volume the branches come from so coordinates align.
    resampled_mask = context["mask"]
    resampled_image = context["image"]

    # Compute a single centroid from the volume center so aorta + all
    # branches share the exact same coordinate origin.
    centroid = _volume_center_mm(resampled_image)

    label = "full-res" if full_res else f"target {target_faces} faces"
    print(f"Generating aorta mesh ({label})...")
    aorta_mesh, centroid = mask_to_mesh(resampled_mask, target_faces=target_faces,
                                        full_res=full_res, centroid=centroid)

    glb_filename = "aorta.glb"
    glb_path = os.path.join(output_dir, glb_filename)
    aorta_mesh.export(glb_path, file_type="glb")
    aorta_kb = os.path.getsize(glb_path) / 1024
    print(f"Aorta mesh: {glb_path} ({aorta_kb:.0f} KB, {len(aorta_mesh.faces)} faces)")

    if not scored:
        print("No candidates found.")
        case_id = schema.case_id_from_path(image_path)
        scene = {
            "case_id": case_id, "mesh_file": glb_filename,
            "centroid_mm": centroid.tolist(), "branches": [],
            "scale_note": "Coordinates in mm, centered at aorta centroid",
        }
        scene_path = os.path.join(output_dir, "scene.json")
        with open(scene_path, "w") as f:
            json.dump(scene, f, indent=2)
        return glb_path, scene_path

    # Build set of kept entry ids for accepted/rejected tagging
    kept_set = set()
    for d in daughters:
        # Match kept daughters back to scored entries by ostium proximity
        for entry in scored:
            ostium_mm = np.array(entry["ostium"]["ostium_mm"])
            d_ostium = np.array(d["ostium_xyz_mm"])
            if np.linalg.norm(ostium_mm - d_ostium) < 0.1:
                kept_set.add(id(entry))
                break

    # --- Build mesh per candidate ---
    case_id = schema.case_id_from_path(image_path)
    branch_entries = []

    for i, entry in enumerate(scored):
        instance = entry["instance"]
        accepted = id(entry) in kept_set
        confidence = float(entry["score"]["confidence"])

        branch_id = f"branch_{i + 1:03d}" if accepted else f"candidate_{i + 1:03d}"
        mesh_file = f"{branch_id}.glb"

        voxels_flat = instance["voxels_flat"]
        shape = instance["shape"]

        print(f"  Building mesh for {branch_id} ({len(voxels_flat)} voxels, "
              f"{'accepted' if accepted else 'rejected'}, conf={confidence:.2f})...")

        branch_mesh = voxels_to_mesh(
            voxels_flat, shape, resampled_image, centroid,
            target_faces=branch_target_faces,
        )

        if branch_mesh is not None:
            mesh_path = os.path.join(output_dir, mesh_file)
            branch_mesh.export(mesh_path, file_type="glb")
            kb = os.path.getsize(mesh_path) / 1024
            print(f"    -> {mesh_file} ({kb:.0f} KB, {len(branch_mesh.faces)} faces)")
        else:
            print(f"    -> skipped (too few voxels for mesh)")
            mesh_file = None

        branch_entries.append((i, entry, accepted, mesh_file))

    # --- Build scene.json ---
    scene = build_pipeline_scene_json(case_id, centroid, glb_filename, scored, kept_set)

    # Remove mesh_file from branches where mesh generation failed
    for i, entry, accepted, mesh_file in branch_entries:
        if mesh_file is None:
            scene["branches"][i].pop("mesh_file", None)

    scene_path = os.path.join(output_dir, "scene.json")
    with open(scene_path, "w") as f:
        json.dump(scene, f, indent=2)

    n_accepted = sum(1 for b in scene["branches"] if b["accepted"])
    n_rejected = sum(1 for b in scene["branches"] if not b["accepted"])
    print(f"\nScene: {n_accepted} accepted + {n_rejected} rejected branches")
    print(f"Scene metadata: {scene_path}")

    return glb_path, scene_path


def main():
    parser = argparse.ArgumentParser(
        description="Export aorta mask + vessel meshes to WebXR-ready assets"
    )
    parser.add_argument("--image", required=True, help="CT image NIfTI path")
    parser.add_argument("--aorta-mask", required=True, help="Aorta mask NIfTI path")
    parser.add_argument("--prediction", default=None,
                        help="Prediction JSON path (legacy mode; omit to run pipeline)")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: viz/output/<case_id>)")
    parser.add_argument("--target-faces", type=int, default=15000,
                        help="Target face count for aorta mesh (default: 15000)")
    parser.add_argument("--branch-faces", type=int, default=3000,
                        help="Target face count per branch mesh (default: 3000)")
    parser.add_argument("--full-res", action="store_true",
                        help="Skip decimation (keep Laplacian smooth only)")
    parser.add_argument("--verbose", action="store_true",
                        help="Log pipeline details to stderr")
    parser.add_argument("--serve", action="store_true",
                        help="Start the viewer server after exporting")
    parser.add_argument("--port", type=int, default=8080,
                        help="Server port when using --serve (default: 8080)")
    parser.add_argument("--no-ssl", action="store_true",
                        help="Use plain HTTP instead of HTTPS (with --serve)")
    args = parser.parse_args()

    # Auto-derive output dir from case_id if not specified
    if args.output_dir is None:
        import schema
        case_id = schema.case_id_from_path(args.image)
        args.output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "viz", "output", case_id,
        )

    if args.prediction:
        export(args.image, args.aorta_mask, args.prediction, args.output_dir,
               args.target_faces, args.full_res)
    else:
        export_pipeline(args.image, args.aorta_mask, args.output_dir,
                        args.target_faces, args.branch_faces,
                        args.full_res, args.verbose)

    if args.serve:
        print("\nStarting viewer server...")
        from viz.serve import main as serve_main
        sys.argv = [
            "viz.serve",
            "--port", str(args.port),
            "--scene-dir", args.output_dir,
        ]
        if args.no_ssl:
            sys.argv.append("--no-ssl")
        serve_main()


if __name__ == "__main__":
    main()
