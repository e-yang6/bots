"""Synthetic (image, mask, reference) triples with EXACT ground truth.

Every daughter's ostium/seed/radius/direction in reference.json is computed
directly from the construction parameters used to draw it -- never measured
back off the rendered voxels. That is the entire point: it lets
scripts/validate_phantom.py separate "is the geometry/measurement chain
correct" (this phantom suite can answer, exactly) from "does detection
succeed on real patients" (it cannot -- see scripts/stability.py,
scripts/sweep.py and scripts/plausibility.py for what actually stands in
for that on the real 25 cases).

Geometry
--------
The parent (aorta) centerline is a circular arc of radius CURVATURE_MM in
the x-z plane -- "gently curved" is a curvature radius much larger than the
vessel radius, not a straight line. At arc length s the point, unit
tangent, and an orthonormal (u, v) frame perpendicular to the tangent are
all closed-form (see _parent_frame): tangent always lies in the x-z plane,
so v = (0, 1, 0) exactly, and u = tangent x v.

A branch is specified by where it leaves (arc length s, azimuth in the
(u, v) frame) and how (elevation tilting it toward the tangent direction,
radius, length). Its ostium is exactly parent_radius_mm out along the
frame's radial direction; its direction leans between that radial direction
and the tangent by the elevation angle; its seed is exactly
SEED_DISTANCE_MM further along that same straight direction -- geodesic
arc length and Euclidean distance coincide by construction for a straight
capsule, so this is exact, not measured.

Two structural tests are built into every phantom, per the brief:
  - a common trunk that bifurcates BIFURCATION_ARC_MM out into two daughter
    capsules: ONE reference entry (the trunk's own ostium/seed, both well
    inside the bifurcation point), because the challenge counts a trunk
    that divides shortly after as a single origin.
  - a close pair CLOSE_PAIR_GAP_MM apart on the wall: TWO reference
    entries, because two genuine origins that happen to be near each other
    must stay two.

Five distractors that must never appear in the reference:
  - an IVC analogue: a lower-HU capsule running parallel to the parent for
    its whole length, touching it -- vein-likeness bait (angle-to-
    centerline, HU-relative-to-lumen) for src/rules.py.
  - a bone-like slab, bright but not touching anything -- a plausibility
    check that nothing about clipping/vesselness reaches across a real gap.
  - a spherical blob connected to the parent through a single VOXEL bridge
    (a true gap, not merely a box corner grazing the parent -- see
    LEAK_BLOB_RADIUS_MM) -- the exact partial-volume bridge
    src.floodfill.build_traversal_volume's binary opening exists to break;
    this is what "leak control works" means in scripts/validate_phantom.py.
  - a thin branch below the 2mm-diameter minimum (radius < 1.0mm) -- must
    never be reported; see THIN_BRANCH_RADIUS_MM's own comment for what this
    can and can't actually test about src.rules.RADIUS_VETO_MM.
  - an organ bed: a large lobulated blob at the DISTAL end of a dim but
    fully eligible daughter, at or above that daughter's own HU. Unlike
    every distractor above, this one is not bait for a rule -- it
    reproduces a detection failure found on the real eval set, where no
    single intensity threshold can separate the branch from the organ it
    feeds. See ORGAN_BED_HU_FRACTIONS for the mechanism and the three
    brightness variants; opt-in via organ_bed_variant, so the 27 grid
    cases and their recorded results are unaffected.

And one truncation feature: with truncate=True the MASK stops
TRUNCATION_MARGIN_MM before the parent capsule in the image does, so the
aorta visibly continues past the mask's cut face -- the aortic-continuation
trap src.candidates._reject's rule (and, as a second check, src.rules'
cap vetoes) is supposed to catch.
"""

import argparse
import json
import os

import numpy as np
import SimpleITK as sitk
from scipy.spatial import cKDTree

import schema
from src.tracing import SEED_DISTANCE_MM

# --- parent geometry -------------------------------------------------------
PARENT_LENGTH_MM = 150.0
PARENT_RADIUS_MM = 9.0
CURVATURE_RADIUS_MM = 500.0     # >> PARENT_RADIUS_MM: "gently curved"
LATERAL_MARGIN_MM = 45.0        # half-width of grid around the parent

# --- branches ----------------------------------------------------------
BRANCH_LENGTH_MM = 18.0
BRANCH_RADIUS_RANGE_MM = (1.0, 3.5)
BRANCH_ARC_FRACTIONS = (0.20, 0.35, 0.50, 0.62, 0.74, 0.86, 0.44, 0.55, 0.80)
BRANCH_AZIMUTH_DEG = (10.0, -20.0, 40.0, -45.0, 15.0, -10.0, 60.0, -55.0, 25.0)
BRANCH_ELEVATION_DEG = (5.0, -5.0, 10.0, 0.0, -8.0, 12.0, -3.0, 6.0, -10.0)

BIFURCATION_ARC_MM = 8.0
BIFURCATION_S_FRACTION = 0.30
BIFURCATION_AZIMUTH_DEG = 100.0
BIFURCATION_TRUNK_RADIUS_MM = 3.5
BIFURCATION_DAUGHTER_SPLIT_DEG = 22.0
BIFURCATION_DAUGHTER_EXTRA_MM = 8.0

CLOSE_PAIR_GAP_MM = 4.0
CLOSE_PAIR_S_FRACTION = 0.62
CLOSE_PAIR_AZIMUTH_DEG = -100.0
CLOSE_PAIR_RADII_MM = (2.4, 1.4)

# --- distractors -------------------------------------------------------
IVC_AZIMUTH_DEG = 180.0
IVC_RADIUS_MM = 7.0
IVC_HU_FRACTION = 0.55          # of (lumen_hu - background_hu), above background
IVC_TOUCH_OVERLAP_MM = 0.5      # overlaps the parent slightly so the flood can reach it

BONE_HU = 1600.0
BONE_OFFSET_MM = (35.0, -30.0, 0.0)   # (x, y, z) offset from grid centre; well clear of everything
BONE_HALF_SIZE_MM = (8.0, 8.0, 18.0)

LEAK_BLOB_AZIMUTH_DEG = 135.0
LEAK_BLOB_S_FRACTION = 0.45
LEAK_BLOB_STANDOFF_MM = 5.0     # gap between parent surface and the blob, bridged by one voxel
LEAK_BLOB_RADIUS_MM = 3.0        # spherical, so "reach toward the centerline" is this radius at
                                 # every azimuth -- an axis-aligned box's corner reaches further
                                 # than its half-size at a diagonal azimuth and silently overlaps
                                 # the parent, which is exactly what LEAK_BLOB_STANDOFF_MM must not do

# A genuine, connected, vessel-shaped branch below the competition's 2mm-
# diameter minimum (radius < 1.0mm) -- NOT added to the reference, and must
# never appear in the daughter list. Placed well clear of every other
# structure (own arc fraction and azimuth, checked empirically) so it forms
# its own flood component rather than being silently absorbed into a
# neighbour's.
#
# It does NOT reach src.rules' radius veto, though -- a binary search over
# radius (0.85/1.0/1.1/1.2/1.3/1.4mm, same isolated position) found that
# candidates.py's own stem/core split never gives anything below ~1.15mm
# true radius its own component at this pipeline's fixed 0.8mm working
# resolution: too little cross-sectional area survives build_traversal_volume's
# binary opening plus the neck-sleeve geometry for it to separate from the
# aorta's own reachable core. So this branch is "not reported" whether or
# not RADIUS_VETO_MM would have caught it -- it tests the pipeline's overall
# floor on detectable calibre, a real and useful thing to know, but not a
# dedicated test of the veto's tolerance margin specifically. See README.md's
# Phantom validation section for what that implies about testing the margin.
THIN_BRANCH_RADIUS_MM = 0.85
THIN_BRANCH_S_FRACTION = 0.09
THIN_BRANCH_AZIMUTH_DEG = 45.0
THIN_BRANCH_ELEVATION_DEG = 0.0

# The organ-bed distractor: a dim daughter that runs into the organ it feeds.
#
# Found on the real eval set (EVAL_SET case_20's four missed ostia and
# case_22's branch_001/branch_006): the branch's own lumen is dimmer than the
# aorta, and the bed it terminates in is at or above the branch's HU, so the
# two are ONE connected component at every threshold that reaches the branch
# at all. Lowering the threshold to reach the branch therefore also admits the
# whole bed, whose frontier growth src.floodfill.detect_leak correctly reports
# as a leak, and the threshold search settles back above the branch. Measured
# there: 96.8-99.8% of the territory a lower threshold newly reached was a
# single connected component, and 41/41 of the missed branches' own centerline
# points lay INSIDE that component -- so there is nothing for a leak-
# sensitivity or threshold-range change to separate.
#
# The branch itself is deliberately eligible on every other axis -- radius
# well above src.rules.RADIUS_VETO_MM and above the ~1.15mm calibre floor
# noted at THIN_BRANCH_RADIUS_MM, length well above the 5mm minimum -- so a
# miss here is attributable to connectivity alone.
# Both vessels leave the wall at the same arc length, a little apart in
# azimuth, and run into the same bed -- two arteries feeding one organ, one
# of them poorly opacified. The FEEDER is at full lumen HU, so the bed stays
# reachable through it no matter what the dim branch does: that is what
# forces the threshold search up above the dim branch's own level and keeps
# the trap closed. Without it the bed is reachable only via the dim branch,
# a clean threshold window opens between them, and the case does not
# reproduce -- measured, not assumed.
# Placed in the stretch past arc fraction 0.70, which the organ-bed cases
# keep clear by running n_branches=3 with no close pair, and far enough apart
# along the parent that the two ostia cannot cross-match inside
# validate_phantom's 10mm radius (0.095 * 150mm = 14.2mm of wall between
# them). Far enough from the parent's own end (arc fraction 1.0) that the
# bed never reaches the end cap.
ORGAN_FEEDER_S_FRACTION = 0.720
ORGAN_BRANCH_S_FRACTION = 0.815
ORGAN_BED_AZIMUTH_DEG = 70.0      # clear of every other structure's azimuth
ORGAN_BRANCH_RADIUS_MM = 1.8      # top of the 1.2-1.8mm dim-branch range: comfortably past the
                                  # ~1.15mm calibre floor, so the binary opening is not the
                                  # binding constraint and a miss means connectivity
ORGAN_FEEDER_RADIUS_MM = 2.6
ORGAN_BRANCH_LENGTH_MM = 10.0     # the 10mm the brief traces, and > SEED_DISTANCE_MM + margin
ORGAN_BED_STANDOFF_MM = 2.0       # past the vessel ends to the bed centre
# A union of jittered spheres: lobulated and non-tubular, so the bed's shape
# signature (low aspect ratio, large cross-section) differs from either
# vessel's own. Sized so that admitting it ALWAYS trips detect_leak's growth
# rule: at a smaller bed the threshold the search settled on was set by the
# rest of the phantom instead, the bed's own HU made no difference to the
# outcome, and whether the dim branch was missed came down to the seed.
ORGAN_BED_LOBES = 12
ORGAN_BED_LOBE_RADIUS_RANGE_MM = (6.0, 9.0)
ORGAN_BED_LOBE_SCATTER_MM = 6.0

# HU as fractions of (lumen_hu - background_hu) above background, never an
# absolute HU -- this cohort's HU floor is not standard (see CLAUDE.md).
# 0.45 puts the branch at roughly half the lumen reference, which is where
# the real missed branches sat (case_20's guides ran 0.54-0.56 of their
# case's lumen reference).
ORGAN_BRANCH_HU_FRACTION = 0.45
# The three brightness variants. The first two reproduce the failure; the
# third is its NEGATIVE CONTROL and is expected to be detected normally.
#
# Measured, at this bed size, over six seeds each: the failure needs
# bed >= branch, and it is then deterministic -- both reproducing variants
# miss the dim branch on every seed, and the threshold the search settles on
# tracks the BED's own level (bed at 0.60 of the span settles at 0.634 of the
# lumen reference; bed level with the branch settles at 0.494-0.509), which
# is what says the bed, and not the rest of the phantom, is what forces it.
#
# The boundary is sharp and sits just below equality. A bed 0.05 of the span
# under the branch -- "a bit brighter", within one GAUSSIAN_NOISE_SIGMA_HU --
# is already seed-dependent: found on 5 of 6 seeds, missed on the sixth. The
# control therefore sits a clear 0.15 below the branch, so it is a control
# and not a coin flip. Keeping it guards against a "fix" that merely lowers
# thresholds everywhere.
ORGAN_BED_HU_FRACTIONS = {
    "bed_brighter": 0.60,
    "equal_hu": ORGAN_BRANCH_HU_FRACTION,
    "branch_brighter_control": 0.30,
}
# The variants that must reproduce the miss; the rest are controls that must
# still be detected. scripts/validate_phantom.py asserts both directions.
ORGAN_BED_REPRODUCING_VARIANTS = ("bed_brighter", "equal_hu")

TRUNCATION_MARGIN_MM = 27.0      # the mask stops this far short of the parent's own end -- far
                                 # enough past the last real branch (arc fraction 0.74) that its
                                 # legitimate ostium isn't mistaken for a spurious cut-face branch

# --- intensity / noise ---------------------------------------------------
DEFAULT_BACKGROUND_HU = 50.0
GAUSSIAN_NOISE_SIGMA_HU = 15.0
POISSON_NOISE_SCALE = 0.03       # relative brightness-dependent shot noise

NON_CONTRAST_LUMEN_HU = 60.0
NON_CONTRAST_BACKGROUND_HU = 40.0


def _parent_frame(s, curvature_mm=CURVATURE_RADIUS_MM):
    """Point, unit tangent, and (u, v) perpendicular frame at arc length s
    along the parent's circular-arc centerline (see module docstring).
    """
    angle = np.asarray(s, dtype=np.float64) / curvature_mm
    point = np.stack([
        curvature_mm * (1.0 - np.cos(angle)),
        np.zeros_like(angle),
        curvature_mm * np.sin(angle),
    ], axis=-1)
    tangent = np.stack([np.sin(angle), np.zeros_like(angle), np.cos(angle)], axis=-1)
    v = np.array([0.0, 1.0, 0.0])
    u = np.cross(tangent, v)
    return point, tangent, u, v


def _branch_geometry(s, azimuth_deg, elevation_deg, radius_mm, length_mm,
                     parent_radius_mm=PARENT_RADIUS_MM, curvature_mm=CURVATURE_RADIUS_MM):
    """ostium / direction / seed / end-point for one straight capsule branch,
    all exact functions of the construction parameters -- see module docstring.
    """
    point, tangent, u, v = _parent_frame(s, curvature_mm)
    az, el = np.radians(azimuth_deg), np.radians(elevation_deg)
    radial = np.cos(az) * u + np.sin(az) * v
    ostium = point + parent_radius_mm * radial
    direction = np.cos(el) * radial + np.sin(el) * tangent
    direction = direction / np.linalg.norm(direction)
    seed = ostium + SEED_DISTANCE_MM * direction
    end = ostium + length_mm * direction
    return {
        "ostium": ostium, "direction": direction, "seed": seed, "end": end,
        "radius_mm": float(radius_mm), "arc_s_mm": float(s), "azimuth_deg": float(azimuth_deg),
    }


def _rotate_around_axis(vector, axis, angle_deg):
    """Rodrigues' rotation of `vector` around unit `axis` by angle_deg."""
    angle = np.radians(angle_deg)
    axis = axis / np.linalg.norm(axis)
    return (
        vector * np.cos(angle)
        + np.cross(axis, vector) * np.sin(angle)
        + axis * np.dot(axis, vector) * (1.0 - np.cos(angle))
    )


def _capsule_mask(points_flat, start, end, radius_mm):
    start, end = np.asarray(start, dtype=np.float64), np.asarray(end, dtype=np.float64)
    segment = end - start
    length_sq = max(float(segment @ segment), 1e-12)
    t = np.clip(((points_flat - start) @ segment) / length_sq, 0.0, 1.0)
    closest = start + t[:, None] * segment
    return np.linalg.norm(points_flat - closest, axis=1) <= radius_mm


def generate_phantom(
    seed=0,
    spacing=(0.8, 0.8, 0.8),
    lumen_hu=400.0,
    background_hu=DEFAULT_BACKGROUND_HU,
    n_branches=5,
    include_bifurcation=True,
    include_close_pair=True,
    include_ivc=True,
    include_bone_slab=True,
    include_leak_bridge=True,
    include_thin_branch=True,
    organ_bed_variant=None,
    truncate=False,
    non_contrast=False,
):
    """Build one (image, mask, reference) phantom triple.

    n_branches picks how many of the plain independent branches (from the
    BRANCH_* placement tables) are included, in addition to the bifurcating
    trunk / close pair / distractors, each independently toggleable.
    non_contrast overrides lumen_hu/background_hu to a barely-enhanced,
    branch-free case that must come back empty (is_contrast_enhanced's own
    gate, not a geometry test).
    organ_bed_variant, one of ORGAN_BED_HU_FRACTIONS, adds the dim-branch-
    into-organ-bed pair; the branch IS a reference daughter, the bed is not.

    Returns (image, mask, reference, meta): SimpleITK images (image int16,
    mask uint8), reference a schema.make_prediction dict, and meta a plain
    dict of the parameters used (for validate_phantom.py's breakdowns).
    """
    rng = np.random.default_rng(seed)
    n_branches = int(np.clip(n_branches, 0, len(BRANCH_ARC_FRACTIONS)))

    if non_contrast:
        lumen_hu = NON_CONTRAST_LUMEN_HU
        background_hu = NON_CONTRAST_BACKGROUND_HU
        n_branches, include_bifurcation, include_close_pair = 0, False, False
        include_ivc = include_bone_slab = include_leak_bridge = include_thin_branch = False
        organ_bed_variant = None

    if organ_bed_variant is not None and organ_bed_variant not in ORGAN_BED_HU_FRACTIONS:
        raise ValueError(
            f"unknown organ_bed_variant {organ_bed_variant!r}; "
            f"expected one of {sorted(ORGAN_BED_HU_FRACTIONS)}"
        )

    sx, sy, sz = (float(s) for s in spacing)
    nz = int(round((PARENT_LENGTH_MM + 20.0) / sz))
    ny = int(round((2 * LATERAL_MARGIN_MM) / sy))
    nx = int(round((2 * LATERAL_MARGIN_MM) / sx))
    shape_zyx = (nz, ny, nx)

    # Grid centred so the parent's un-curved start sits at (0, 0, 10mm) and
    # curves away in +x -- LATERAL_MARGIN_MM of room on every side covers the
    # branches, the IVC analogue, and the bone slab without clipping any of
    # them at the grid edge.
    origin_mm = np.array([-LATERAL_MARGIN_MM, -LATERAL_MARGIN_MM, -10.0])
    zz, yy, xx = np.meshgrid(
        np.arange(nz) * sz + origin_mm[2],
        np.arange(ny) * sy + origin_mm[1],
        np.arange(nx) * sx + origin_mm[0],
        indexing="ij",
    )
    grid_points = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)

    # Parent centerline, densely sampled for a nearest-point distance field.
    # The tube's curvature is gentle relative to its radius (CURVATURE_RADIUS_MM
    # >> PARENT_RADIUS_MM), so a fine polyline sample reproduces the circular
    # cross-section the analytic branch formulas assume to well under a voxel.
    s_samples = np.linspace(0.0, PARENT_LENGTH_MM, 400)
    centerline_points, _tangents, _u, _v = _parent_frame(s_samples)
    centerline_tree = cKDTree(centerline_points)
    distance_to_axis, nearest_index = centerline_tree.query(grid_points, k=1)
    s_of_point = s_samples[nearest_index]

    parent_full = (distance_to_axis <= PARENT_RADIUS_MM) & (s_of_point >= 0.0) & (s_of_point <= PARENT_LENGTH_MM)

    mask_flat = parent_full.copy()
    if truncate:
        mask_flat &= s_of_point <= (PARENT_LENGTH_MM - TRUNCATION_MARGIN_MM)

    lumen_flat = parent_full.copy()
    image_flat = np.full(grid_points.shape[0], background_hu, dtype=np.float64)

    daughters, meta_branches = [], []

    def place_branch(s_frac, azimuth_deg, elevation_deg, radius_mm, length_mm, label, reportable=True):
        s = s_frac * PARENT_LENGTH_MM
        geometry = _branch_geometry(s, azimuth_deg, elevation_deg, radius_mm, length_mm)
        inside = _capsule_mask(grid_points, geometry["ostium"], geometry["end"], radius_mm)
        lumen_flat[inside] = True
        instance_id = f"branch_{len(daughters) + 1:03d}" if reportable else None
        if reportable:
            daughters.append(schema.make_daughter(
                instance_id=instance_id, ostium_xyz_mm=geometry["ostium"], seed_xyz_mm=geometry["seed"],
                radius_mm=geometry["radius_mm"], direction_xyz=geometry["direction"],
            ))
        meta_branches.append({"label": label, "instance_id": instance_id, **geometry})
        return geometry

    for i in range(n_branches):
        radius = float(rng.uniform(*BRANCH_RADIUS_RANGE_MM))
        place_branch(
            BRANCH_ARC_FRACTIONS[i], BRANCH_AZIMUTH_DEG[i], BRANCH_ELEVATION_DEG[i],
            radius, BRANCH_LENGTH_MM, label=f"independent_{i}",
        )

    if include_thin_branch:
        place_branch(
            THIN_BRANCH_S_FRACTION, THIN_BRANCH_AZIMUTH_DEG, THIN_BRANCH_ELEVATION_DEG,
            THIN_BRANCH_RADIUS_MM, BRANCH_LENGTH_MM, label="thin_branch", reportable=False,
        )

    if include_bifurcation:
        s = BIFURCATION_S_FRACTION * PARENT_LENGTH_MM
        trunk = _branch_geometry(s, BIFURCATION_AZIMUTH_DEG, 0.0, BIFURCATION_TRUNK_RADIUS_MM, BIFURCATION_ARC_MM)
        lumen_flat[_capsule_mask(grid_points, trunk["ostium"], trunk["end"], BIFURCATION_TRUNK_RADIUS_MM)] = True
        # Two daughter capsules diverge from the bifurcation point -- NOT
        # added to the reference: the challenge counts the trunk once.
        _point, _tangent, u, _v = _parent_frame(s)
        for sign in (+1.0, -1.0):
            daughter_direction = _rotate_around_axis(trunk["direction"], u, sign * BIFURCATION_DAUGHTER_SPLIT_DEG)
            daughter_end = trunk["end"] + BIFURCATION_DAUGHTER_EXTRA_MM * daughter_direction
            lumen_flat[_capsule_mask(
                grid_points, trunk["end"], daughter_end, BIFURCATION_TRUNK_RADIUS_MM * 0.7
            )] = True
        instance_id = f"branch_{len(daughters) + 1:03d}"
        daughters.append(schema.make_daughter(
            instance_id=instance_id, ostium_xyz_mm=trunk["ostium"], seed_xyz_mm=trunk["seed"],
            radius_mm=trunk["radius_mm"], direction_xyz=trunk["direction"],
        ))
        meta_branches.append({"label": "bifurcation_trunk", "instance_id": instance_id, **trunk})

    if include_close_pair:
        azimuth_gap_deg = np.degrees(CLOSE_PAIR_GAP_MM / PARENT_RADIUS_MM)
        for i, radius in enumerate(CLOSE_PAIR_RADII_MM):
            place_branch(
                CLOSE_PAIR_S_FRACTION, CLOSE_PAIR_AZIMUTH_DEG + i * azimuth_gap_deg, 0.0,
                radius, BRANCH_LENGTH_MM, label=f"close_pair_{i}",
            )

    distractors = {}

    if include_ivc:
        s0, tangent0, u0, _v0 = _parent_frame(0.0)
        s1, _t1, u1, _v1 = _parent_frame(PARENT_LENGTH_MM)
        az = np.radians(IVC_AZIMUTH_DEG)
        offset = (PARENT_RADIUS_MM + IVC_RADIUS_MM - IVC_TOUCH_OVERLAP_MM)
        ivc_start = s0 + offset * (np.cos(az) * u0 + np.sin(az) * np.array([0.0, 1.0, 0.0]))
        ivc_end = s1 + offset * (np.cos(az) * u1 + np.sin(az) * np.array([0.0, 1.0, 0.0]))
        inside = _capsule_mask(grid_points, ivc_start, ivc_end, IVC_RADIUS_MM)
        ivc_hu = background_hu + IVC_HU_FRACTION * (lumen_hu - background_hu)
        image_flat[inside & ~lumen_flat] = ivc_hu
        distractors["ivc"] = {"start": ivc_start.tolist(), "end": ivc_end.tolist(), "radius_mm": IVC_RADIUS_MM}

    if include_bone_slab:
        centre = np.array([origin_mm[0] + LATERAL_MARGIN_MM, origin_mm[1] + LATERAL_MARGIN_MM, PARENT_LENGTH_MM / 2])
        centre = centre + np.asarray(BONE_OFFSET_MM)
        half = np.asarray(BONE_HALF_SIZE_MM)
        inside = np.all(np.abs(grid_points - centre) <= half, axis=1)
        image_flat[inside] = BONE_HU
        distractors["bone_slab"] = {"centre": centre.tolist(), "half_size_mm": list(BONE_HALF_SIZE_MM)}

    if include_leak_bridge:
        s = LEAK_BLOB_S_FRACTION * PARENT_LENGTH_MM
        point, _tangent, u, v = _parent_frame(s)
        az = np.radians(LEAK_BLOB_AZIMUTH_DEG)
        radial = np.cos(az) * u + np.sin(az) * v
        blob_centre = point + (PARENT_RADIUS_MM + LEAK_BLOB_STANDOFF_MM) * radial
        blob_inside = np.sum((grid_points - blob_centre) ** 2, axis=1) <= LEAK_BLOB_RADIUS_MM ** 2
        image_flat[blob_inside] = lumen_hu
        # a single voxel-wide bright line from the parent's surface to the
        # blob -- broken by build_traversal_volume's binary opening, which is
        # exactly the point of this distractor (see module docstring). The
        # blob is spherical, not an axis-aligned box, so this gap is real at
        # every azimuth: a box's corner can reach past its own half-size and
        # silently overlap the parent depending on orientation.
        bridge_start = point + PARENT_RADIUS_MM * radial
        n_steps = max(2, int(round(LEAK_BLOB_STANDOFF_MM / min(sx, sy, sz))) + 1)
        for t in np.linspace(0.0, 1.0, n_steps):
            bridge_point = bridge_start + t * (blob_centre - bridge_start)
            nearest = np.argmin(np.sum((grid_points - bridge_point) ** 2, axis=1))
            image_flat[nearest] = lumen_hu
        distractors["leak_blob"] = {"centre": blob_centre.tolist(), "radius_mm": LEAK_BLOB_RADIUS_MM}

    if organ_bed_variant is not None:
        branch_hu = background_hu + ORGAN_BRANCH_HU_FRACTION * (lumen_hu - background_hu)
        bed_hu = background_hu + ORGAN_BED_HU_FRACTIONS[organ_bed_variant] * (lumen_hu - background_hu)

        dim = _branch_geometry(ORGAN_BRANCH_S_FRACTION * PARENT_LENGTH_MM, ORGAN_BED_AZIMUTH_DEG,
                               0.0, ORGAN_BRANCH_RADIUS_MM, ORGAN_BRANCH_LENGTH_MM)
        feeder = _branch_geometry(ORGAN_FEEDER_S_FRACTION * PARENT_LENGTH_MM, ORGAN_BED_AZIMUTH_DEG,
                                  0.0, ORGAN_FEEDER_RADIUS_MM, ORGAN_BRANCH_LENGTH_MM)

        # Lobes scattered about a centre past both vessels' ends -- a union of
        # spheres, so the bed is lobulated rather than a sphere or a box. The
        # scatter is only ever outward (never back down the vessels), which is
        # both how an organ sits relative to its artery and what keeps the bed
        # off the parent wall.
        outward = dim["direction"] + feeder["direction"]
        outward = outward / np.linalg.norm(outward)
        bed_centre = 0.5 * (dim["end"] + feeder["end"]) + ORGAN_BED_STANDOFF_MM * outward
        bed_inside = np.zeros(grid_points.shape[0], dtype=bool)
        lobes = []
        for _ in range(ORGAN_BED_LOBES):
            offset = rng.normal(0.0, ORGAN_BED_LOBE_SCATTER_MM, size=3)
            offset -= min(0.0, float(offset @ outward)) * outward
            lobe_centre = bed_centre + offset
            lobe_radius = float(rng.uniform(*ORGAN_BED_LOBE_RADIUS_RANGE_MM))
            bed_inside |= np.sum((grid_points - lobe_centre) ** 2, axis=1) <= lobe_radius ** 2
            lobes.append({"centre": lobe_centre.tolist(), "radius_mm": lobe_radius})

        # Bed first, then the vessels over it: wherever they overlap at the
        # junction the voxels carry the VESSEL's own HU, so "the dim branch is
        # at branch_hu" is true of the rendered voxels, not just of the intent.
        image_flat[bed_inside & ~lumen_flat] = bed_hu
        image_flat[_capsule_mask(grid_points, dim["ostium"], dim["end"],
                                 ORGAN_BRANCH_RADIUS_MM) & ~lumen_flat] = branch_hu
        # The feeder is ordinary lumen, so it joins lumen_flat and is painted
        # at lumen_hu with everything else.
        lumen_flat[_capsule_mask(grid_points, feeder["ostium"], feeder["end"],
                                 ORGAN_FEEDER_RADIUS_MM)] = True

        # The bed must reach the parent only THROUGH a vessel; a lobe touching
        # the parent directly would be an unintended extra origin and the case
        # would stop testing what it claims to.
        gap_to_parent_mm = float(distance_to_axis[bed_inside].min() - PARENT_RADIUS_MM)
        if gap_to_parent_mm <= 0.0:
            raise ValueError(
                f"organ bed touches the parent (gap {gap_to_parent_mm:.1f}mm); "
                "raise ORGAN_BRANCH_LENGTH_MM or lower ORGAN_BED_LOBE_SCATTER_MM"
            )

        for label, geometry in (("organ_feeder", feeder), ("organ_bed_branch", dim)):
            instance_id = f"branch_{len(daughters) + 1:03d}"
            daughters.append(schema.make_daughter(
                instance_id=instance_id, ostium_xyz_mm=geometry["ostium"],
                seed_xyz_mm=geometry["seed"], radius_mm=geometry["radius_mm"],
                direction_xyz=geometry["direction"],
            ))
            meta_branches.append({"label": label, "instance_id": instance_id, **geometry})

        distractors["organ_bed"] = {
            "variant": organ_bed_variant, "centre": bed_centre.tolist(), "lobes": lobes,
            "bed_hu": float(bed_hu), "branch_hu": float(branch_hu),
            "lumen_hu": float(lumen_hu), "gap_to_parent_mm": gap_to_parent_mm,
            "n_bed_voxels": int(bed_inside.sum()),
        }

    image_flat[lumen_flat] = lumen_hu

    noise = rng.normal(0.0, GAUSSIAN_NOISE_SIGMA_HU, size=image_flat.shape)
    shot = rng.normal(0.0, 1.0, size=image_flat.shape) * np.abs(image_flat) * POISSON_NOISE_SCALE
    image_flat = image_flat + noise + shot

    image_arr = image_flat.reshape(shape_zyx).astype(np.float32)
    mask_arr = mask_flat.reshape(shape_zyx).astype(np.uint8)

    image = sitk.GetImageFromArray(image_arr.astype(np.int16))
    mask = sitk.GetImageFromArray(mask_arr)
    for volume in (image, mask):
        volume.SetSpacing((sx, sy, sz))
        volume.SetOrigin(tuple(origin_mm.tolist()))

    case_id = f"phantom_seed{seed}_sp{sx:.2g}x{sy:.2g}x{sz:.2g}_hu{int(lumen_hu)}_n{n_branches}"
    if organ_bed_variant is not None:
        case_id += f"_organbed_{organ_bed_variant}"
    if truncate:
        case_id += "_trunc"
    if non_contrast:
        case_id = f"phantom_seed{seed}_noncontrast"
    reference = schema.make_prediction(case_id, daughters)

    truncation_point = None
    if truncate:
        cut_s = PARENT_LENGTH_MM - TRUNCATION_MARGIN_MM
        truncation_point = _parent_frame(cut_s)[0].tolist()

    meta = {
        "case_id": case_id, "seed": seed, "spacing": (sx, sy, sz), "lumen_hu": lumen_hu,
        "background_hu": background_hu, "n_branches": n_branches, "truncate": truncate,
        "non_contrast": non_contrast, "organ_bed_variant": organ_bed_variant,
        "branches": meta_branches,
        "distractors": distractors, "truncation_point_mm": truncation_point,
    }
    return image, mask, reference, meta


def write_phantom_case(image, mask, reference, out_dir):
    """Write one case as out_dir/<case_id>/{image.nii.gz, mask.nii.gz,
    reference.json} -- the same subjectNNN/orig.nii layout the real data
    uses, so schema.case_id_from_path and run.py's CLI need no special-casing.
    """
    case_dir = os.path.join(out_dir, reference["case_id"])
    os.makedirs(case_dir, exist_ok=True)
    image_path = os.path.join(case_dir, "image.nii.gz")
    mask_path = os.path.join(case_dir, "mask.nii.gz")
    reference_path = os.path.join(case_dir, "reference.json")
    sitk.WriteImage(image, image_path)
    sitk.WriteImage(mask, mask_path)
    schema.write_prediction(reference, reference_path)
    return {"image": image_path, "mask": mask_path, "reference": reference_path, "case_dir": case_dir}


# The main accuracy-and-robustness suite: every combination of spacing x
# lumen HU x branch count, spanning the real cohort's own range (see the
# real-case lumen medians logged across sessions: 86-581 HU; spacings from
# isotropic 0.8mm up to clearly anisotropic), plus the two structural
# special cases (bifurcation truncation -> one instance is already inside
# every phantom by default; only the truncated-mask and non-contrast cases
# need a dedicated, separate phantom).
SUITE_SPACINGS = {
    "iso_0.8": (0.8, 0.8, 0.8),
    "iso_1.5": (1.5, 1.5, 1.5),
    "aniso_0.7x0.7x1.5": (0.7, 0.7, 1.5),
}
SUITE_LUMEN_HU = (250.0, 400.0, 550.0)
SUITE_BRANCH_COUNTS = (2, 5, 9)


def build_suite(base_seed=0):
    """Every (image, mask, reference, meta) phantom in the accuracy suite,
    plus the truncation and non-contrast special cases. Deterministic in
    base_seed: same seed always reproduces the same suite.
    """
    cases = []
    index = 0
    for spacing_name, spacing in SUITE_SPACINGS.items():
        for lumen_hu in SUITE_LUMEN_HU:
            for n_branches in SUITE_BRANCH_COUNTS:
                cases.append(generate_phantom(
                    seed=base_seed + index, spacing=spacing, lumen_hu=lumen_hu, n_branches=n_branches,
                ))
                index += 1
    cases.append(generate_phantom(seed=base_seed + index, spacing=(0.8, 0.8, 0.8), truncate=True))
    index += 1
    cases.append(generate_phantom(seed=base_seed + index, spacing=(0.8, 0.8, 0.8), non_contrast=True))
    index += 1

    # The organ-bed variants, appended last and with their own seeds so every
    # case above keeps the exact geometry and recorded result it had before
    # this distractor existed. One bright branch from the ordinary tables
    # rides along as a control: the same case must still find it, so a miss
    # on the dim branch is attributable to the bed and not to the phantom.
    for variant in ORGAN_BED_HU_FRACTIONS:
        # The IVC analogue is off here and only here: at IVC_HU_FRACTION it
        # sits near the bed's own level, and measurement showed it -- not the
        # bed -- was what the threshold search was settling against, which
        # would have made a miss unattributable. Every other distractor stays.
        cases.append(generate_phantom(
            seed=base_seed + index, spacing=(0.8, 0.8, 0.8), n_branches=3,
            include_ivc=False, include_close_pair=False, organ_bed_variant=variant,
        ))
        index += 1
    return cases


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", default="phantom_suite", help="Where to write the phantom cases")
    parser.add_argument("--seed", type=int, default=0, help="Base RNG seed for the whole suite")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cases = build_suite(base_seed=args.seed)
    manifest = []
    for image, mask, reference, meta in cases:
        paths = write_phantom_case(image, mask, reference, args.out_dir)
        manifest.append({**paths, "meta": {k: v for k, v in meta.items() if k != "branches"}})
        print(f"wrote {reference['case_id']} ({len(reference['daughters'])} reference daughters)")
    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"{len(cases)} phantom cases written to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
