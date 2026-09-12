"""Read/write helpers for the branch-detection prediction JSON schema.

Schema:
{
  "case_id": "subject001",
  "parent": {"instance_id": "aorta"},
  "daughters": [
    {
      "instance_id": "branch_001",
      "parent_instance_id": "aorta",
      "ostium_xyz_mm": [x, y, z],
      "seed_xyz_mm": [x, y, z],
      "radius_mm": r,
      "direction_xyz": [dx, dy, dz]
    }
  ]
}
"""

import json
import os

PARENT_INSTANCE_ID = "aorta"


def case_id_from_path(path):
    """Derive a case_id from an input path's parent folder name.

    Real data is laid out as subject001/orig1.nii, so the case_id is the
    directory containing the image, e.g. '.../subject001/orig1.nii' -> 'subject001'.
    Falls back to the filename (stripped of .nii/.nii.gz) if the path has no
    parent directory component.
    """
    parent = os.path.basename(os.path.dirname(os.path.normpath(path)))
    if parent:
        return parent

    base = os.path.basename(path)
    for ext in (".nii.gz", ".nii"):
        if base.endswith(ext):
            base = base[: -len(ext)]
            break
    else:
        base = os.path.splitext(base)[0]
    return base


def make_daughter(instance_id, ostium_xyz_mm, seed_xyz_mm, radius_mm, direction_xyz,
                   parent_instance_id=PARENT_INSTANCE_ID):
    return {
        "instance_id": instance_id,
        "parent_instance_id": parent_instance_id,
        "ostium_xyz_mm": [float(v) for v in ostium_xyz_mm],
        "seed_xyz_mm": [float(v) for v in seed_xyz_mm],
        "radius_mm": float(radius_mm),
        "direction_xyz": [float(v) for v in direction_xyz],
    }


def make_prediction(case_id, daughters=None):
    """Build a prediction dict matching the required schema."""
    return {
        "case_id": case_id,
        "parent": {"instance_id": PARENT_INSTANCE_ID},
        "daughters": list(daughters) if daughters else [],
    }


def write_prediction(prediction, output_path):
    """Write a prediction dict to output_path as JSON."""
    with open(output_path, "w") as f:
        json.dump(prediction, f, indent=2)


def read_prediction(path):
    """Read a prediction JSON file and return the dict."""
    with open(path) as f:
        return json.load(f)
