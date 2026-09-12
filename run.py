"""CLI entry point for the Branchseed daughter-artery detector.

Usage:
    python run.py --image image.nii.gz --aorta-mask aorta_mask.nii.gz --output prediction.json

This session builds no detection logic. detect_branches() is a stub that
returns an empty daughters list; later sessions replace its internals.
The whole pipeline is wrapped so any failure still emits valid, schema-
conformant JSON with an empty daughters list instead of crashing or
producing no output at all.
"""

import argparse
import json
import sys

import schema


def detect_branches(image_path, aorta_mask_path):
    """Stub detection. Returns a list of daughter dicts (schema.make_daughter).

    No detection logic yet -- always returns an empty list.
    """
    return []


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Detect arteries branching off the aorta from a CT scan and aorta mask."
    )
    parser.add_argument("--image", required=True, help="Path to the CT scan (NIfTI).")
    parser.add_argument("--aorta-mask", required=True, help="Path to the aorta-only mask (NIfTI).")
    parser.add_argument("--output", required=True, help="Path to write the output prediction JSON.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    case_id = "unknown"
    daughters = []
    try:
        case_id = schema.case_id_from_path(args.image)
        daughters = detect_branches(args.image, args.aorta_mask)
    except Exception as exc:
        print(f"warning: pipeline failed, emitting empty daughters list ({exc})", file=sys.stderr)

    prediction = schema.make_prediction(case_id, daughters)

    try:
        schema.write_prediction(prediction, args.output)
    except Exception as exc:
        print(f"error: failed to write output via schema.write_prediction ({exc})", file=sys.stderr)
        fallback = {
            "case_id": case_id,
            "parent": {"instance_id": schema.PARENT_INSTANCE_ID},
            "daughters": [],
        }
        with open(args.output, "w") as f:
            json.dump(fallback, f, indent=2)


if __name__ == "__main__":
    main()
