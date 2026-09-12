"""Draw the flood-fill detector's intermediate results on the aorta, for eyeballing.

One PNG per case:
  top row     3D view of the aorta surface with every flood component
              (survivors coloured per instance, rejections tinted by rule),
              contact patches, traces, ostia and seeds; coronal and sagittal
              projections of the same; and the flood's frontier profile with
              the threshold search that produced it.
  bottom row  axial slices through the first instances' ostia: the CT, the
              aorta mask outline, the instance's territory and contact patch
              in that slice, and its trace.

Survivor colours match the #index printed by `run.py --verbose`.

Usage:
    python -m scripts.visualize_candidates [--cases subject001 subject018]
                                           [--data-dir PATH] [--out-dir PATH]
"""

import argparse
import glob
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from run import analyze_case  # noqa: E402
from src.candidates import flat_to_zyx  # noqa: E402
from src.floodfill import EARLY_WINDOW_MM, LATE_WINDOW_MM  # noqa: E402

DEFAULT_CASES = ("subject001", "subject010", "subject018", "subject007")
REJECTION_COLOURS = {
    "end_cap": "tab:red",
    "aortic_continuation": "tab:orange",
    "too_short": "0.55",
    "no_contact_patch": "black",
}


def find_data_dir(explicit=None):
    if explicit:
        return explicit if os.path.isdir(explicit) else None
    for root in (REPO_ROOT, os.path.dirname(REPO_ROOT)):
        for path in glob.glob(os.path.join(root, "TORALIS CHALLENGE*")):
            if os.path.isdir(path):
                return path
    return None


def case_paths(data_dir, name):
    subdir = os.path.join(data_dir, name)
    files = os.listdir(subdir)
    image = next(f for f in files if f.startswith("orig"))
    mask = next(f for f in files if f.startswith("mask"))
    return os.path.join(subdir, image), os.path.join(subdir, mask)


def physical_to_index_xyz(points_mm, image):
    points_mm = np.atleast_2d(np.asarray(points_mm, dtype=float))
    spacing = np.array(image.GetSpacing())
    origin = np.array(image.GetOrigin())
    direction = np.array(image.GetDirection()).reshape(3, 3)
    return ((points_mm - origin) @ direction) / spacing


def instance_colour(index):
    return plt.cm.tab10(index % 10)


def _paint(canvas, pixels_rc, colour, alpha):
    rows, cols = pixels_rc
    rgb = np.asarray(matplotlib.colors.to_rgb(colour))
    canvas[rows, cols, :3] = (1 - alpha) * canvas[rows, cols, :3] + alpha * rgb


def projection_panel(axis, context, project_axis, horizontal_axis, label):
    """Projection of everything the flood reached, with territories on top.

    Not an HU MIP: projected through the whole crop, vertebrae and ribs
    saturate it. What the flood reached is exactly the set these candidates
    were cut from, so it is the more useful backdrop.
    """
    image = context["image"]
    mask_arr = sitk.GetArrayFromImage(context["mask"]).astype(bool)
    flood = context["flood"]
    reached = np.isfinite(flood["dist"]) & ~mask_arr

    grey = 0.15 + 0.25 * mask_arr.any(axis=project_axis) + 0.45 * reached.any(axis=project_axis)
    canvas = np.dstack([grey, grey, grey])
    shape = mask_arr.shape
    keep_axes = [a for a in (0, 1, 2) if a != project_axis]

    def pixels(flat):
        zyx = flat_to_zyx(flat, shape)
        return zyx[:, keep_axes[0]], zyx[:, keep_axes[1]]

    for component in context["detection"]["components"]:
        if component["rejected_by"] is not None:
            _paint(canvas, pixels(component["voxels_flat"]), REJECTION_COLOURS[component["rejected_by"]], 0.45)
    for index, instance in enumerate(context["instances"][: len(context["traces"])]):
        _paint(canvas, pixels(instance["voxels_flat"]), instance_colour(index), 0.6)
        _paint(canvas, pixels(instance["patch_flat"]), "white", 0.8)

    axis.imshow(canvas, origin="lower", aspect="auto")
    axis.contour(mask_arr.max(axis=project_axis).astype(float), levels=[0.5], colors="tab:cyan", linewidths=0.7)

    # array axes kept are (z, horizontal); index xyz column for horizontal
    column = {1: 1, 2: 0}[horizontal_axis]
    for index, (trace, ostium, seed) in enumerate(
        zip(context["traces"], context["ostium_estimates"], context["seed_estimates"])
    ):
        path = physical_to_index_xyz(trace["points_mm"], image)
        axis.plot(path[:, column], path[:, 2], "-", color="black", linewidth=1.2)
        point = physical_to_index_xyz(ostium["ostium_mm"], image)[0]
        axis.plot(point[column], point[2], "x", color="black", markersize=5)
        axis.annotate(str(index), (point[column], point[2]), fontsize=7, color="black",
                      xytext=(3, 3), textcoords="offset points")
        seed_point = physical_to_index_xyz(seed["seed_mm"], image)[0]
        axis.plot(seed_point[column], seed_point[2], ".", color="magenta", markersize=5)
    axis.set_title(f"{label}\nwhite=contact patch x=ostium .=seed; red/orange=cap rejections", fontsize=8)
    axis.set_xticks([])
    axis.set_yticks([])


def surface_panel(axis, context, title, surface_stride=10, component_stride=6):
    surface_points = context["surface"][0][::surface_stride]
    axis.scatter(surface_points[:, 0], surface_points[:, 1], surface_points[:, 2], s=1, c="0.8", alpha=0.2)
    reference = context["image"]
    shape = context["detection"]["shape"]

    def to_mm(flat):
        zyx = flat_to_zyx(flat, shape)
        return (zyx[:, ::-1] * np.array(reference.GetSpacing())) @ np.array(
            reference.GetDirection()).reshape(3, 3).T + np.array(reference.GetOrigin())

    for component in context["detection"]["components"]:
        if component["rejected_by"] in ("end_cap", "aortic_continuation"):
            points = to_mm(component["voxels_flat"][::component_stride])
            axis.scatter(points[:, 0], points[:, 1], points[:, 2], s=2,
                         c=REJECTION_COLOURS[component["rejected_by"]], alpha=0.4)
    for index, instance in enumerate(context["instances"][: len(context["traces"])]):
        colour = instance_colour(index)
        points = to_mm(instance["voxels_flat"][::component_stride])
        axis.scatter(points[:, 0], points[:, 1], points[:, 2], s=2, color=colour, alpha=0.35)
        patch = instance["patch_mm"]
        axis.scatter(patch[:, 0], patch[:, 1], patch[:, 2], s=4, color=colour, alpha=0.9)
    for trace, ostium in zip(context["traces"], context["ostium_estimates"]):
        path = trace["points_mm"]
        axis.plot(path[:, 0], path[:, 1], path[:, 2], "-", color="black", linewidth=1.5)
        axis.scatter(*ostium["ostium_mm"], s=25, c="black", marker="x", depthshade=False)
    axis.set_title(f"{title}\n3D: instances, patches, traces", fontsize=9)
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_zlabel("z")


def frontier_panel(axis, context):
    flood = context["flood"]
    frontier = flood["frontier_sizes"]
    band = flood["band_mm"]
    distances = (np.arange(frontier.size) + 0.5) * band
    axis.bar(distances, frontier, width=band * 0.9, color="tab:blue")
    axis.axvspan(0, context["detection"]["neck_mm"], color="0.85", label="sleeve / neck")
    axis.axvspan(*EARLY_WINDOW_MM, color="tab:green", alpha=0.15, label="early window")
    axis.axvspan(flood["budget_mm"] - LATE_WINDOW_MM, flood["budget_mm"], color="tab:red", alpha=0.15,
                 label="late window")
    axis.axhline(flood["leak"]["cross_section_floor"], color="tab:red", linestyle="--", linewidth=0.8,
                 label="0.5 x aortic cross-section")
    visible = frontier[int(np.ceil(context["detection"]["neck_mm"] / band)):]
    if visible.size:
        axis.set_ylim(0, max(visible.max(), flood["leak"]["cross_section_floor"]) * 1.3)
    growth = flood["leak"]["growth"]
    attempts = " ".join(f"{a['fraction']:.2f}{'L' if a['leak'] else ''}" for a in flood["attempts"])
    axis.set_title(
        f"frontier: T={flood['threshold_hu']:.0f}HU ({flood['threshold_fraction']:.2f}x) "
        f"leak={flood['leak']['reason'] or 'none'} growth={'n/a' if growth is None else f'{growth:.2f}'}\n"
        f"search: {attempts}",
        fontsize=8,
    )
    axis.set_xlabel("geodesic distance beyond aorta (mm)")
    axis.set_ylabel("voxels per band")
    axis.legend(fontsize=6, loc="upper right")


def axial_panels(figure, context, first_subplot=5, n_panels=4):
    image = context["image"]
    hu = sitk.GetArrayFromImage(image)
    mask_arr = sitk.GetArrayFromImage(context["mask"]).astype(bool)
    shape = mask_arr.shape
    reference = context["flood"]["lumen_reference_hu"]

    for panel, (instance, trace, ostium) in enumerate(
        zip(context["instances"], context["traces"], context["ostium_estimates"])
    ):
        if panel >= n_panels:
            break
        point = physical_to_index_xyz(ostium["ostium_mm"], image)[0]
        z = int(np.clip(round(point[2]), 0, shape[0] - 1))

        grey = np.clip((hu[z] - (reference - 500.0)) / 700.0, 0, 1)
        canvas = np.dstack([grey, grey, grey])
        for flat, colour, alpha in ((instance["voxels_flat"], instance_colour(panel), 0.5),
                                    (instance["patch_flat"], "white", 0.8)):
            zyx = flat_to_zyx(flat, shape)
            in_slice = zyx[:, 0] == z
            _paint(canvas, (zyx[in_slice, 1], zyx[in_slice, 2]), colour, alpha)

        axis = figure.add_subplot(2, 4, first_subplot + panel)
        axis.imshow(canvas, origin="lower", aspect="equal")
        axis.contour(mask_arr[z].astype(float), levels=[0.5], colors="tab:cyan", linewidths=0.8)
        path = physical_to_index_xyz(trace["points_mm"], image)
        axis.plot(path[:, 0], path[:, 1], "-", color="yellow", linewidth=1.2)
        axis.plot(point[0], point[1], "x", color="red", markersize=7)

        half = 30
        cx, cy = int(round(point[0])), int(round(point[1]))
        axis.set_xlim(cx - half, cx + half)
        axis.set_ylim(cy - half, cy + half)
        axis.set_title(
            f"#{panel} z={z} len={instance['max_distance_mm']:.0f}mm traced={trace['traced_length_mm']:.1f}mm"
            f"\n{trace['truncated_by']} split={instance['split']}",
            fontsize=8,
        )
        axis.set_xticks([])
        axis.set_yticks([])


def plot_case(context, title, out_path):
    figure = plt.figure(figsize=(18, 9.5))
    if context["flood"] is None:
        axis = figure.add_subplot(1, 1, 1)
        axis.text(0.5, 0.5, f"{title}: {context.get('short_circuit_reason')}", ha="center")
        figure.savefig(out_path, dpi=110)
        plt.close(figure)
        return

    surface_panel(figure.add_subplot(2, 4, 1, projection="3d"), context, title)
    projection_panel(figure.add_subplot(2, 4, 2), context, project_axis=1, horizontal_axis=2,
                     label="coronal (z vs x)")
    projection_panel(figure.add_subplot(2, 4, 3), context, project_axis=2, horizontal_axis=1,
                     label="sagittal (z vs y)")
    frontier_panel(figure.add_subplot(2, 4, 4), context)
    axial_panels(figure, context)
    figure.tight_layout()
    figure.savefig(out_path, dpi=110)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="*", default=list(DEFAULT_CASES))
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--out-dir", default=os.path.join(REPO_ROOT, "scripts", "inspection_output"))
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    data_dir = find_data_dir(args.data_dir)
    if data_dir is None:
        print("No TORALIS CHALLENGE data folder found; nothing to visualize.", file=sys.stderr)
        return 1

    for name in args.cases:
        image_path, mask_path = case_paths(data_dir, name)
        context = analyze_case(image_path, mask_path, verbose=False)
        out_path = os.path.join(args.out_dir, f"{name}_flood.png")
        plot_case(context, name, out_path)
        print(f"{name}: {len(context['candidates'])} candidates, {len(context['instances'])} instances, "
              f"{len(context['traces'])} traced -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
