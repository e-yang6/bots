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
import math
import os
from numbers import Real
from pathlib import Path
import tempfile

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


def submission_prediction(prediction):
    if not isinstance(prediction, dict):
        raise ValueError("prediction must be an object")
    if not isinstance(prediction.get("case_id"), str) or not prediction["case_id"]:
        raise ValueError("case_id must be a nonempty string")
    if prediction.get("parent") != {"instance_id": PARENT_INSTANCE_ID}:
        raise ValueError("parent must identify the aorta")
    if not isinstance(prediction.get("daughters"), list):
        raise ValueError("daughters must be a list")

    def finite_number(value):
        return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)

    daughters = []
    for index, daughter in enumerate(prediction["daughters"], start=1):
        if not isinstance(daughter, dict):
            raise ValueError("every daughter must be an object")
        if daughter.get("instance_id") != f"branch_{index:03d}":
            raise ValueError("daughter IDs must be unique and sequential: branch_001, branch_002, ...")
        if daughter.get("parent_instance_id") != PARENT_INSTANCE_ID:
            raise ValueError("every daughter must link to aorta")
        for field in ("ostium_xyz_mm", "seed_xyz_mm", "direction_xyz"):
            value = daughter.get(field)
            if not isinstance(value, (list, tuple)) or len(value) != 3 or not all(map(finite_number, value)):
                raise ValueError(f"{field} must contain three finite numbers")
        radius = daughter.get("radius_mm")
        if not finite_number(radius) or radius <= 0:
            raise ValueError("radius_mm must be finite and positive")
        direction = daughter["direction_xyz"]
        if not math.isclose(math.hypot(*direction), 1.0, abs_tol=1e-6):
            raise ValueError("direction_xyz must be a unit vector")
        outward = [s - o for s, o in zip(daughter["seed_xyz_mm"], daughter["ostium_xyz_mm"])]
        if math.hypot(*outward) == 0 or sum(d * v for d, v in zip(direction, outward)) < -1e-6:
            raise ValueError("direction_xyz must point from the ostium into the daughter")
        daughters.append(make_daughter(
            daughter["instance_id"], daughter["ostium_xyz_mm"], daughter["seed_xyz_mm"], radius, direction
        ))
    return make_prediction(prediction["case_id"], daughters)


def write_prediction(prediction, output_path):
    """Write a prediction dict to output_path as JSON."""
    payload = json.dumps(submission_prediction(prediction), indent=2, allow_nan=False)
    destination = Path(output_path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=destination.parent,
                                         prefix=".branchseed-", suffix=".tmp", delete=False) as f:
            temporary = f.name
            f.write(payload)
        os.replace(temporary, destination)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)


def read_prediction(path):
    """Read a prediction JSON file and return the dict."""
    with open(path) as f:
        return json.load(f)
