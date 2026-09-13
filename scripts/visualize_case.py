"""Render one case's final prediction for visual QC -- the actual demo
artifact, built from prediction.json alone (not the internal pipeline
context scripts/visualize_candidates.py inspects).

Two images per case:
  <case>_3d.png             aorta surface (from the mask) with every
                             reported daughter's ostium (sphere), direction
                             (arrow) and seed point (small marker) overlaid.
  <case>_cross_sections.png one panel per daughter: the CT sampled on the
                             plane perpendicular to its direction at its
                             seed point, with the fitted radius drawn as a
                             circle -- a direct picture of what "radius_mm"
                             claims to measure.

This is what "real-case evidence is visual checks" (see README.md) means in
practice: there is no reference to score against, only whether the picture
looks like a real ostium sitting on a real vessel of about the claimed size.

Usage:
    python -m scripts.visualize_case --image image.nii --aorta-mask mask.nii
                                     --prediction prediction.json
                                     [--out-dir viz_output]
"""

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
try:
    import pyvista as pv
except ImportError:
    pv = None
import SimpleITK as sitk

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import schema  # noqa: E402
from src.io_utils import load_case  # noqa: E402
from src.lumen_evidence import VolumeSampler, perpendicular_basis  # noqa: E402

if pv is not None:
    pv.OFF_SCREEN = True

CROSS_SECTION_HALF_EXTENT_MM = 8.0
CROSS_SECTION_STEP_MM = 0.25
ARROW_LENGTH_MM = 12.0
OSTIUM_MARKER_RADIUS_MM = 1.5
SEED_MARKER_RADIUS_MM = 0.8


def _mesh_from_mask(mask_image):
    """Aorta surface, in real physical mm, from the binary mask alone.

    pyvista's ImageData only supports axis-aligned spacing/origin, so the
    contour is generated in a local (spacing * index) frame with origin 0
    and no direction, then mapped to physical mm by the same formula
    SimpleITK itself uses (origin + direction @ (spacing * index)) -- correct
    for this cohort's axis-flip direction matrices, and general enough for
    an oblique one too.
    """
    arr = sitk.GetArrayFromImage(mask_image).astype(np.float32)  # z, y, x
    spacing = np.array(mask_image.GetSpacing())
    origin = np.array(mask_image.GetOrigin())
    direction = np.array(mask_image.GetDirection()).reshape(3, 3)

    grid = pv.ImageData(dimensions=arr.shape[::-1], spacing=spacing, origin=(0.0, 0.0, 0.0))
    grid.point_data["mask"] = arr.transpose(2, 1, 0).ravel(order="F")
    contour = grid.contour(isosurfaces=[0.5], scalars="mask")
    if contour.n_points == 0:
        return contour
    contour.points = contour.points @ direction.T + origin
    return contour


def render_3d_cpu(mask_image, daughters, case_id, out_path):
    from skimage.measure import marching_cubes
    from src.geometry import _indices_to_physical

    array = sitk.GetArrayFromImage(mask_image)
    vertices, faces, _normals, _values = marching_cubes(array, level=0.5, step_size=2)
    points = _indices_to_physical(vertices[:, ::-1], mask_image)
    figure = plt.figure(figsize=(10, 9))
    axis = figure.add_subplot(111, projection="3d")
    axis.plot_trisurf(*points.T, triangles=faces, color="tan", alpha=0.25, linewidth=0)
    for index, daughter in enumerate(daughters):
        colour = plt.cm.tab10.colors[index % 10]
        ostium = np.asarray(daughter["ostium_xyz_mm"])
        direction = np.asarray(daughter["direction_xyz"])
        axis.scatter(*ostium, color=colour, s=30)
        axis.scatter(*daughter["seed_xyz_mm"], color=colour, s=12)
        axis.quiver(*ostium, *direction, length=ARROW_LENGTH_MM, color=colour)
        axis.text(*ostium, daughter["instance_id"], color=colour, fontsize=7)
    axis.set_box_aspect(np.maximum(np.ptp(points, axis=0), 1.0))
    axis.set(xlabel="LPS X (mm)", ylabel="LPS Y (mm)", zlabel="LPS Z (mm)",
             title=f"{case_id}: {len(daughters)} daughters")
    figure.savefig(out_path, dpi=120)
    plt.close(figure)


def render_3d(mask_image, daughters, case_id, out_path):
    if pv is None:
        return render_3d_cpu(mask_image, daughters, case_id, out_path)
    mesh = _mesh_from_mask(mask_image)
    plotter = pv.Plotter(off_screen=True, window_size=(1400, 1100))
    plotter.set_background("white")
    if mesh.n_points:
        plotter.add_mesh(mesh, color="tan", opacity=0.35, smooth_shading=True)

    colours = plt.cm.tab10.colors
    for index, daughter in enumerate(daughters):
        colour = colours[index % len(colours)]
        ostium = np.asarray(daughter["ostium_xyz_mm"], dtype=float)
        seed = np.asarray(daughter["seed_xyz_mm"], dtype=float)
        direction = np.asarray(daughter["direction_xyz"], dtype=float)
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction = direction / norm

        plotter.add_mesh(pv.Sphere(radius=OSTIUM_MARKER_RADIUS_MM, center=ostium), color=colour)
        plotter.add_mesh(pv.Sphere(radius=SEED_MARKER_RADIUS_MM, center=seed), color=colour, opacity=0.6)
        arrow = pv.Arrow(start=ostium, direction=direction, scale=ARROW_LENGTH_MM)
        plotter.add_mesh(arrow, color=colour)
        label = f"{daughter['instance_id']}"
        if "confidence" in daughter:
            label += f" ({daughter['confidence']:.2f})"
        plotter.add_point_labels([ostium], [label], font_size=12, text_color=colour,
                                 shape_opacity=0.0, always_visible=True)

    plotter.add_text(f"{case_id}: {len(daughters)} daughters", font_size=14, color="black")
    plotter.camera_position = "iso"
    plotter.screenshot(out_path)
    plotter.close()


def _cross_section(sampler, seed_mm, direction_xyz, radius_mm,
                   half_extent_mm=CROSS_SECTION_HALF_EXTENT_MM, step_mm=CROSS_SECTION_STEP_MM):
    direction = np.asarray(direction_xyz, dtype=float)
    norm = np.linalg.norm(direction)
    direction = direction / norm if norm > 0 else np.array([0.0, 0.0, 1.0])
    u, v = perpendicular_basis(direction)
    u, v = u[0], v[0]

    axis = np.arange(-half_extent_mm, half_extent_mm + step_mm, step_mm)
    grid_u, grid_v = np.meshgrid(axis, axis, indexing="xy")
    points_mm = (
        np.asarray(seed_mm)[None, None, :]
        + grid_u[..., None] * u[None, None, :]
        + grid_v[..., None] * v[None, None, :]
    )
    flat_points = points_mm.reshape(-1, 3)
    values = sampler.sample("hu", flat_points)
    return axis, values.reshape(grid_u.shape)


def render_cross_sections(image, daughters, case_id, out_path):
    if not daughters:
        figure = plt.figure(figsize=(4, 2))
        plt.text(0.5, 0.5, f"{case_id}: no daughters reported", ha="center", va="center")
        plt.axis("off")
        figure.savefig(out_path, dpi=110)
        plt.close(figure)
        return

    hu_arr = sitk.GetArrayFromImage(image).astype(np.float32)
    sampler = VolumeSampler(image)
    sampler.add("hu", hu_arr)

    n = len(daughters)
    n_cols = min(4, n)
    n_rows = int(np.ceil(n / n_cols))
    figure, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 3.2 * n_rows), squeeze=False)

    for index, daughter in enumerate(daughters):
        axis = axes[index // n_cols][index % n_cols]
        radius_mm = daughter["radius_mm"]
        extent = max(CROSS_SECTION_HALF_EXTENT_MM, radius_mm * 2.0)
        axis_range, panel = _cross_section(
            sampler, daughter["seed_xyz_mm"], daughter["direction_xyz"], radius_mm, half_extent_mm=extent
        )
        axis.imshow(panel, extent=(axis_range[0], axis_range[-1], axis_range[0], axis_range[-1]),
                   origin="lower", cmap="gray")
        circle = plt.Circle((0, 0), radius_mm, fill=False, color="tab:red", linewidth=1.5)
        axis.add_patch(circle)
        axis.plot(0, 0, "+", color="tab:red", markersize=8)
        label = daughter["instance_id"]
        if "confidence" in daughter:
            label += f" conf={daughter['confidence']:.2f}"
        axis.set_title(f"{label}\nr={radius_mm:.2f}mm", fontsize=9)
        axis.set_xticks([])
        axis.set_yticks([])

    for index in range(n, n_rows * n_cols):
        axes[index // n_cols][index % n_cols].axis("off")

    figure.suptitle(f"{case_id}: cross-section at each seed, red circle = fitted radius")
    figure.tight_layout()
    figure.savefig(out_path, dpi=110)
    plt.close(figure)


def visualize_case(image_path, mask_path, prediction_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    image, mask = load_case(image_path, mask_path)
    prediction = schema.read_prediction(prediction_path)
    case_id = prediction.get("case_id", schema.case_id_from_path(image_path))
    daughters = prediction.get("daughters", [])

    path_3d = os.path.join(out_dir, f"{case_id}_3d.png")
    path_cross = os.path.join(out_dir, f"{case_id}_cross_sections.png")
    render_3d(mask, daughters, case_id, path_3d)
    render_cross_sections(image, daughters, case_id, path_cross)
    return path_3d, path_cross


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", required=True)
    parser.add_argument("--aorta-mask", required=True)
    parser.add_argument("--prediction", required=True)
    parser.add_argument("--out-dir", default="viz_output")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    path_3d, path_cross = visualize_case(args.image, args.aorta_mask, args.prediction, args.out_dir)
    print(f"wrote {path_3d}")
    print(f"wrote {path_cross}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
