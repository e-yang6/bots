"""Overlay candidate ostia and their traces on the aorta surface, for eyeballing.

Produces, per case:
  - a 3D view of the aorta surface (subsampled) with candidate ostia and the
    traced paths, coloured by traced length,
  - two projection views (coronal / sagittal) with the same overlay, which
    are usually easier to read than the 3D one.

Usage:
    python -m scripts.visualize_candidates [--cases subject001 subject018]
                                           [--data-dir PATH] [--out-dir PATH]
                                           [--top N]
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run import analyze_case  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CASES = ("subject001", "subject010", "subject018", "subject024")


def find_data_dir(explicit=None):
    if explicit:
        return explicit if os.path.isdir(explicit) else None
    for path in glob.glob(os.path.join(REPO_ROOT, "TORALIS CHALLENGE*")):
        if os.path.isdir(path):
            return path
    return None


def case_paths(data_dir, name):
    subdir = os.path.join(data_dir, name)
    files = os.listdir(subdir)
    image = next(f for f in files if f.startswith("orig"))
    mask = next(f for f in files if f.startswith("mask"))
    return os.path.join(subdir, image), os.path.join(subdir, mask)


def _trace_colour(trace):
    """Green for traces that got somewhere, red for ones that died early."""
    length = trace["traced_length_mm"]
    return plt.cm.RdYlGn(np.clip(length / 10.0, 0.0, 1.0))


def plot_axial_panels(figure, context, traces, ostia, n_panels=4, start_index=5):
    """Axial slices through the top candidates' ostia.

    The most interpretable check available without reference annotations: an
    axial slice at a real origin shows the aorta with a vessel visibly
    leaving it, and the marker should sit on that vessel's mouth.
    """
    image = context["image"]
    hu_arr = sitk.GetArrayFromImage(image)
    mask_arr = sitk.GetArrayFromImage(context["mask"]).astype(bool)

    for panel, (trace, ostium) in enumerate(zip(traces[:n_panels], ostia[:n_panels])):
        index = np.array(
            image.TransformPhysicalPointToContinuousIndex(tuple(ostium["consensus_mm"]))
        )
        z = int(round(index[2]))
        z = int(np.clip(z, 0, hu_arr.shape[0] - 1))

        axis = figure.add_subplot(2, 4, start_index + panel)
        axis.imshow(hu_arr[z], origin="lower", cmap="gray", vmin=-100, vmax=500, aspect="equal")
        axis.contour(mask_arr[z].astype(float), levels=[0.5], colors="tab:cyan", linewidths=0.8)
        axis.plot(index[0], index[1], "o", color="red", markersize=7, fillstyle="none", markeredgewidth=1.5)

        path_index = np.array(
            [image.TransformPhysicalPointToContinuousIndex(tuple(p)) for p in trace["points_mm"]]
        )
        axis.plot(path_index[:, 0], path_index[:, 1], "-", color="yellow", linewidth=1.4)
        axis.set_title(
            f"#{panel} z={z} traced={trace['traced_length_mm']:.1f}mm", fontsize=8
        )
        axis.set_xticks([])
        axis.set_yticks([])


def plot_case(context, title, out_path, top=15, surface_stride=12):
    surface_points = context["surface"][0]
    mask_arr = sitk.GetArrayFromImage(context["mask"]).astype(bool)
    mask_image = context["mask"]
    # Project the lumen-likeness volume, not just the mask: the whole point is
    # to see whether a candidate sits on a vessel that is actually there.
    # Restricted to the aorta's neighbourhood, because a max-projection of the
    # whole volume is saturated by rib and vertebra edges, whose partial-volume
    # voxels pass straight through the lumen HU band on their way to bone.
    lumen_arr = context["evidence"]["sampler"].array("lumen")
    neighbourhood = sitk.GetArrayFromImage(
        sitk.BinaryDilate(sitk.Cast(mask_image, sitk.sitkUInt8), [18, 18, 18], sitk.sitkBall)
    ).astype(bool)
    lumen_arr = lumen_arr * neighbourhood

    traces = context["traces"][:top]
    ostia = context["ostium_estimates"][:top]
    seeds = context["seed_estimates"][:top]
    parentage = context["parentage"][:top]

    figure = plt.figure(figsize=(16, 9))

    axis3d = figure.add_subplot(2, 4, 1, projection="3d")
    sampled = surface_points[::surface_stride]
    axis3d.scatter(sampled[:, 0], sampled[:, 1], sampled[:, 2], s=1, c="0.75", alpha=0.25)
    for index, (trace, ostium) in enumerate(zip(traces, ostia)):
        path = trace["points_mm"]
        axis3d.plot(path[:, 0], path[:, 1], path[:, 2], "-", color=_trace_colour(trace), linewidth=2)
        point = ostium["consensus_mm"]
        axis3d.scatter(*point, s=28, c="blue", marker="o", depthshade=False)
    axis3d.set_title(f"{title}\n3D: surface + traces")
    axis3d.set_xlabel("x (mm)")
    axis3d.set_ylabel("y (mm)")
    axis3d.set_zlabel("z (mm)")

    # index-space projections, so the aorta mask can be shown as a silhouette
    for panel, (projection_axis, horizontal, label) in enumerate(
        [(1, 0, "coronal (z vs x)"), (2, 1, "sagittal (z vs y)")], start=2
    ):
        axis = figure.add_subplot(2, 4, panel)
        axis.imshow(lumen_arr.max(axis=projection_axis), origin="lower", cmap="bone", aspect="auto")
        axis.contour(mask_arr.max(axis=projection_axis).astype(float), levels=[0.5],
                     colors="tab:cyan", linewidths=0.7)

        for index, (trace, ostium, seed, parent) in enumerate(zip(traces, ostia, seeds, parentage)):
            path_index = np.array(
                [mask_image.TransformPhysicalPointToContinuousIndex(tuple(p)) for p in trace["points_mm"]]
            )
            axis.plot(path_index[:, horizontal], path_index[:, 2], "-",
                      color=_trace_colour(trace), linewidth=1.6)

            ostium_index = np.array(
                mask_image.TransformPhysicalPointToContinuousIndex(tuple(ostium["consensus_mm"]))
            )
            marker = "x" if parent["is_branch_of_branch"] else "o"
            axis.plot(ostium_index[horizontal], ostium_index[2], marker, color="blue", markersize=5)
            axis.annotate(str(index), (ostium_index[horizontal], ostium_index[2]),
                          fontsize=6, color="blue", xytext=(2, 2), textcoords="offset points")

            seed_index = np.array(
                mask_image.TransformPhysicalPointToContinuousIndex(tuple(seed["seed_mm"]))
            )
            axis.plot(seed_index[horizontal], seed_index[2], ".", color="magenta", markersize=4)

        axis.set_title(f"{label}\nblue=ostium (x = branch-of-branch), magenta=seed")

    plot_axial_panels(figure, context, traces, ostia)

    figure.tight_layout()
    figure.savefig(out_path, dpi=120)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="*", default=list(DEFAULT_CASES))
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--out-dir", default=os.path.join(REPO_ROOT, "scripts", "inspection_output"))
    parser.add_argument("--top", type=int, default=15, help="How many candidates to draw.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    data_dir = find_data_dir(args.data_dir)
    if data_dir is None:
        print("No TORALIS CHALLENGE data folder found; nothing to visualize.", file=sys.stderr)
        return 1

    for name in args.cases:
        image_path, mask_path = case_paths(data_dir, name)
        context = analyze_case(image_path, mask_path, verbose=False)
        traced = len(context["traces"])
        out_path = os.path.join(args.out_dir, f"{name}_candidates.png")
        plot_case(context, name, out_path, top=args.top)
        print(f"{name}: {len(context['candidates'])} candidates, {traced} traced -> {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
