"""Manual/visual verification of the geometric foundation (io_utils.py,
geometry.py, intensity.py): no automated assertions, just diagnostics and
saved PNGs to confirm by eye that:

  - gzip-disguised files load correctly (subject016+ in the dev set),
  - extreme-HU / oblique-direction cases don't crash (subject024),
  - cap flags only fire at true mask-component extremes -- never partway
    down a real vessel, and never at the middle of a partial-coverage mask,
  - centerline and HU stats look sane.

Uses the real "TORALIS CHALLENGE" dev-set folder if present next to the
repo root; otherwise falls back to a synthetic cylinder-with-side-stub
volume so this still runs somewhere without the real data.

Usage:
    python -m scripts.inspect_pipeline [--data-dir PATH] [--out-dir PATH]
"""

import argparse
import glob
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.geometry import (
    compute_centerline,
    compute_surface_normals,
    crop_to_mask_bbox,
    flag_end_caps,
    resample_isotropic,
)
from src.intensity import is_contrast_enhanced, lumen_stats
from src.io_utils import load_case
from tests.synthetic import make_cylinder_with_stub

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def find_real_data_dir(explicit_path=None):
    if explicit_path:
        return explicit_path if os.path.isdir(explicit_path) else None
    candidates = glob.glob(os.path.join(REPO_ROOT, "TORALIS CHALLENGE*"))
    for c in candidates:
        if os.path.isdir(c):
            return c
    return None


def list_real_cases(data_dir):
    cases = []
    for entry in sorted(os.listdir(data_dir)):
        subdir = os.path.join(data_dir, entry)
        if not os.path.isdir(subdir) or not entry.startswith("subject"):
            continue
        files = os.listdir(subdir)
        origs = [f for f in files if f.startswith("orig")]
        masks = [f for f in files if f.startswith("mask")]
        if origs and masks:
            cases.append((entry, os.path.join(subdir, origs[0]), os.path.join(subdir, masks[0])))
    return cases


def run_pipeline(image, mask, label):
    result = {"label": label}
    t0 = time.time()

    image_arr = sitk.GetArrayFromImage(image)
    result["hu_min"] = float(image_arr.min())
    result["hu_max"] = float(image_arr.max())

    cropped_image, cropped_mask = crop_to_mask_bbox(image, mask, margin_mm=15)
    result["cropped_size"] = cropped_image.GetSize()

    resampled_image, resampled_mask, _grid = resample_isotropic(cropped_image, cropped_mask, target_spacing=0.8)
    result["resampled_size"] = resampled_image.GetSize()

    centerline = compute_centerline(cropped_mask)
    result["kept_components"] = [(c["label"], round(c["volume_mm3"], 1)) for c in centerline["kept"]]
    result["dropped_components"] = [(d["label"], round(d["volume_mm3"], 1)) for d in centerline["dropped"]]

    normals = compute_surface_normals(cropped_mask)
    result["n_surface_points"] = normals[0].shape[0]

    caps = flag_end_caps(cropped_mask, normals, centerline)
    result["caps"] = caps

    stats = lumen_stats(image, mask)
    result["lumen_stats"] = stats
    result["contrast_enhanced"] = is_contrast_enhanced(stats)

    result["elapsed_s"] = time.time() - t0
    result["cropped_mask"] = cropped_mask
    result["centerline"] = centerline
    return result


def print_summary(result):
    print(f"--- {result['label']} ---")
    print(f"  HU range: [{result['hu_min']:.0f}, {result['hu_max']:.0f}]")
    print(f"  cropped size: {result['cropped_size']}  resampled size: {result['resampled_size']}")
    print(f"  kept components (label, mm3): {result['kept_components']}")
    print(f"  dropped components (label, mm3): {result['dropped_components']}")
    print(f"  surface points: {result['n_surface_points']}")
    stats = result["lumen_stats"]
    print(
        f"  lumen HU: median={stats['median']:.0f} p25={stats['p25']:.0f} p75={stats['p75']:.0f} "
        f"mad={stats['mad']:.0f} n={stats['n_voxels']}  contrast_enhanced={result['contrast_enhanced']}"
    )
    for cap in result["caps"]:
        print(
            f"  cap label={cap['component_label']} end={cap['end']:>4} "
            f"radius_ratio={cap['radius_ratio']:.2f} (>{'0.4' if cap['radius_exceeds_threshold'] else '-'}) "
            f"tangent_deg={cap['tangent_alignment_deg']:.1f} "
            f"(<25={cap['direction_within_threshold']})  "
            f"likely_partial_coverage_edge={cap['likely_partial_coverage_edge']}"
        )
    print(f"  elapsed: {result['elapsed_s']:.2f}s")


def save_visualization(result, out_path):
    cropped_mask = result["cropped_mask"]
    mask_arr = sitk.GetArrayFromImage(cropped_mask).astype(bool)  # z, y, x

    cap_overlay = np.zeros(mask_arr.shape, dtype=np.uint8)
    for cap in result["caps"]:
        code = 2 if cap["likely_partial_coverage_edge"] else 1
        cap_overlay[cap["cap_region_mask"]] = code

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, axis_name, proj_axis in [(axes[0], "coronal (z vs x)", 1), (axes[1], "sagittal (z vs y)", 2)]:
        mask_proj = mask_arr.max(axis=proj_axis)
        cap_proj = cap_overlay.max(axis=proj_axis)

        rgb = np.zeros(mask_proj.shape + (3,))
        rgb[..., 1] = mask_proj * 0.5  # dim green = mask
        rgb[cap_proj == 1] = (1.0, 0.6, 0.0)  # orange = flagged cap, not partial-coverage
        rgb[cap_proj == 2] = (1.0, 0.0, 0.0)  # red = flagged AND likely partial-coverage edge
        ax.imshow(rgb, origin="lower", aspect="auto")
        ax.set_title(f"{result['label']}\n{axis_name}")

        for comp in result["centerline"]["kept"]:
            points_mm = comp["points_mm"]
            if points_mm.shape[0] < 2:
                continue
            idx_points = np.array(
                [cropped_mask.TransformPhysicalPointToIndex(tuple(p)) for p in points_mm]
            )  # (x, y, z)
            if proj_axis == 1:
                ax.plot(idx_points[:, 0], idx_points[:, 2], "c.-", markersize=2, linewidth=1)
            else:
                ax.plot(idx_points[:, 1], idx_points[:, 2], "c.-", markersize=2, linewidth=1)

    plt.tight_layout()
    plt.savefig(out_path, dpi=110)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=None, help="Path to the TORALIS CHALLENGE dev-set folder.")
    parser.add_argument(
        "--out-dir",
        default=os.path.join(REPO_ROOT, "scripts", "inspection_output"),
        help="Where to write PNG visualizations.",
    )
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    data_dir = find_real_data_dir(args.data_dir)

    if data_dir is not None:
        print(f"Using real data from: {data_dir}\n")
        cases = list_real_cases(data_dir)
        print(f"Found {len(cases)} real cases. Running full pipeline on all of them (no visualization) "
              "to confirm nothing crashes...\n")

        failures = []
        for name, image_path, mask_path in cases:
            try:
                image, mask = load_case(image_path, mask_path)
                run_pipeline(image, mask, name)
                print(f"  {name}: OK")
            except Exception as e:  # noqa: BLE001 -- this script's whole point is to surface failures
                failures.append((name, e))
                print(f"  {name}: FAILED -- {type(e).__name__}: {e}")

        print()
        if failures:
            print(f"{len(failures)} case(s) FAILED: {[f[0] for f in failures]}")
        else:
            print("All real cases ran the full pipeline without crashing.")

        # curated subset with saved visualizations: gzip-disguised, oblique +
        # extreme-HU, multi-component-noise, genuine-second-segment,
        # partial-coverage, and one plain baseline case
        curated_names = {
            "subject001": "plain file, single component, full mask coverage",
            "subject010": "13 mask components -- all but the main one are noise specks",
            "subject016": "gzip content disguised with a .nii extension",
            "subject017": "partial coverage: mask starts well after the volume's first slice",
            "subject018": "one substantial second component: a genuinely interrupted aorta",
            "subject024": "oblique/non-orthonormal direction; HU as low as -9388",
        }
        by_name = {name: (p1, p2) for name, p1, p2 in cases}
        print("\nSaving visualizations for the curated cases that illustrate each finding:")
        for name, why in curated_names.items():
            if name not in by_name:
                continue
            image_path, mask_path = by_name[name]
            image, mask = load_case(image_path, mask_path)
            result = run_pipeline(image, mask, f"{name} ({why})")
            print_summary(result)
            out_path = os.path.join(args.out_dir, f"{name}.png")
            save_visualization(result, out_path)
            print(f"  saved {out_path}\n")

    else:
        print("No real data folder found; using a synthetic cylinder-with-side-stub volume.\n")

        synthetic_cases = [
            (
                "synthetic_plain",
                make_cylinder_with_stub(shape_zyx=(90, 60, 60), stub=True, noise_speck=False),
            ),
            (
                "synthetic_noise_and_second_segment",
                make_cylinder_with_stub(
                    shape_zyx=(110, 60, 60), z_start=10, z_end=80, stub=False,
                    noise_speck=True, second_segment=True,
                ),
            ),
            (
                "synthetic_oblique",
                make_cylinder_with_stub(shape_zyx=(90, 60, 60), tilt_deg=4.0, stub=False, noise_speck=False),
            ),
        ]
        for name, (image, mask) in synthetic_cases:
            result = run_pipeline(image, mask, name)
            print_summary(result)
            out_path = os.path.join(args.out_dir, f"{name}.png")
            save_visualization(result, out_path)
            print(f"  saved {out_path}\n")


if __name__ == "__main__":
    main()
