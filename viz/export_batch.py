"""Batch-export all cases that have predictions to WebXR-ready assets.

Usage:
  python -m viz.export_batch \
    --data-dir "TORALIS CHALLENGE -20260912T162830Z-1-001/TORALIS CHALLENGE" \
    --predictions-dir predictions/ \
    --output-dir viz/output

For each subjectNNN with a matching prediction JSON, exports aorta.glb + scene.json.
"""

import argparse
import glob
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from viz.export_scene import export


def find_cases(data_dir, predictions_dir):
    """Find subject folders that have both data files and a prediction JSON."""
    cases = []
    for subject_dir in sorted(glob.glob(os.path.join(data_dir, "subject*"))):
        subject_name = os.path.basename(subject_dir)

        # Find image and mask files
        nii_files = glob.glob(os.path.join(subject_dir, "*.nii")) + \
                    glob.glob(os.path.join(subject_dir, "*.nii.gz"))

        image_file = None
        mask_file = None
        for f in nii_files:
            base = os.path.basename(f).lower()
            if "mask" in base:
                mask_file = f
            elif "orig" in base:
                image_file = f

        if not image_file or not mask_file:
            continue

        # Look for prediction JSON
        pred_file = os.path.join(predictions_dir, f"{subject_name}.json")
        if not os.path.exists(pred_file):
            pred_file = os.path.join(predictions_dir, f"prediction_{subject_name}.json")
        if not os.path.exists(pred_file):
            continue

        cases.append({
            "subject": subject_name,
            "image": image_file,
            "mask": mask_file,
            "prediction": pred_file,
        })

    return cases


def main():
    parser = argparse.ArgumentParser(description="Batch export cases to WebXR assets")
    parser.add_argument("--data-dir", required=True, help="Root data directory with subject folders")
    parser.add_argument("--predictions-dir", required=True, help="Directory with prediction JSONs")
    parser.add_argument("--output-dir", default="viz/output", help="Output directory")
    parser.add_argument("--target-faces", type=int, default=15000)
    args = parser.parse_args()

    cases = find_cases(args.data_dir, args.predictions_dir)
    if not cases:
        print("No cases found with both data and predictions.")
        print(f"  Data dir: {args.data_dir}")
        print(f"  Predictions dir: {args.predictions_dir}")
        return

    print(f"Found {len(cases)} case(s) to export.\n")

    for i, case in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] {case['subject']}")
        out_dir = os.path.join(args.output_dir, case["subject"])
        t0 = time.time()
        try:
            export(case["image"], case["mask"], case["prediction"], out_dir, args.target_faces)
            print(f"  Done in {time.time()-t0:.1f}s\n")
        except Exception as e:
            print(f"  FAILED: {e}\n")


if __name__ == "__main__":
    main()
