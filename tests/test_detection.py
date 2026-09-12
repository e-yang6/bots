"""Candidates, rejections, origin splitting and tracing on phantoms with known
geometry.

Phantoms put daughters in the IMAGE but not in the MASK, which is how the real
task is posed -- the supplied mask is aorta-only. Unless stated otherwise the
aorta is a vertical cylinder of radius 8mm about (x=32, y=32), so its wall on
the +x side is at x=40.
"""

import numpy as np
import pytest

from run import analyze_volumes
from src.candidates import compute_exit_voxels
from src.parentage import watershed_basins
from tests.synthetic import make_capsule_phantom

WALL_X = 40.0


def stub(z=45.0, length_beyond_wall=18.0, radius=3.0, y=32.0):
    return {"start": (36.0, y, z), "end": (WALL_X + length_beyond_wall, y, z), "radius": radius}


def analyze(**phantom):
    image, mask = make_capsule_phantom(**phantom)
    return analyze_volumes(image, mask)


def rejected(context, rule):
    return [c for c in context["detection"]["components"] if c["rejected_by"] == rule]


def test_single_stub_is_one_candidate_and_one_instance():
    context = analyze(branches=[stub()])
    assert len(context["candidates"]) == 1
    assert len(context["instances"]) == 1

    candidate = context["candidates"][0]
    assert candidate["direction_estimate"][0] > 0.9
    assert candidate["max_distance_mm"] >= 15.0
    assert context["instances"][0]["split"] == "none"
    assert rejected(context, "end_cap") == [] and rejected(context, "aortic_continuation") == []


def test_plain_tube_has_no_candidates():
    context = analyze()
    assert context["detection"]["components"] == []
    assert context["instances"] == []


def test_contact_patch_is_the_stub_mouth_on_the_aorta_wall():
    context = analyze(branches=[stub()])
    instance = context["instances"][0]
    centroid = instance["patch_centroid_mm"]
    assert centroid == pytest.approx([WALL_X + 1.0, 32.0, 45.0], abs=1.2)
    # the mouth of a 3mm-radius stub is ~28mm2; the patch must be that mouth,
    # not the whole wall and not a stray voxel or two
    assert 15.0 < instance["patch_area_mm2"] < 90.0


def test_trace_seed_direction_and_radius_of_a_straight_stub():
    context = analyze(branches=[stub()])
    trace = context["traces"][0]
    seed = context["seed_estimates"][0]

    assert trace["truncated_by"] == "max_length"
    assert trace["traced_length_mm"] == pytest.approx(10.0, abs=0.6)
    assert trace["arc_lengths_mm"][0] == 0.0
    assert np.all(np.diff(trace["arc_lengths_mm"]) == pytest.approx(0.5))

    # 5mm of geodesic distance beyond the lumen voxel at x=40
    assert seed["reached_seed_distance"] is True
    assert seed["seed_mm"] == pytest.approx([45.0, 32.0, 45.0], abs=1.0)

    direction = seed["direction_xyz"]
    assert np.linalg.norm(direction) == pytest.approx(1.0)
    assert direction[0] > 0.97

    assert seed["radius_from_distance_transform_mm"] == pytest.approx(3.0, abs=0.6)
    assert seed["radius_from_frontier_mm"] == pytest.approx(3.0, abs=1.0)
    assert seed["radius_disagreement_flag"] is False


def test_ostium_estimates_land_on_the_mouth_and_agree():
    context = analyze(branches=[stub()])
    ostium = context["ostium_estimates"][0]
    assert ostium["primary"] == "patch_centroid_mm"
    assert set(ostium["estimates"]) == {"patch_centroid_mm", "traced_projection_mm", "max_gradient_mm"}
    assert ostium["ostium_mm"] == pytest.approx([WALL_X, 32.0, 45.0], abs=1.5)
    assert ostium["max_disagreement_mm"] < 4.0
    assert len(ostium["pairwise_distances_mm"]) == 3


def test_aorta_continuing_past_a_cut_mask_face_is_rejected_as_end_cap():
    # the image aorta runs to z=85 but the mask stops flat at z=60
    context = analyze(aorta_z_mm=(10.0, 85.0), mask_z_mm=(10.0, 60.0), branches=[stub(z=35.0)])
    caps = rejected(context, "end_cap")
    assert len(caps) == 1
    assert caps[0]["cap_face_fraction"] >= 0.5
    # and the real branch lower down still survives
    assert len(context["candidates"]) == 1
    assert context["candidates"][0]["patch_centroid_mm"][2] == pytest.approx(35.0, abs=2.0)


def test_aorta_continuing_past_a_tapered_mask_end_is_rejected_as_continuation():
    # the mask narrows over its last 8mm while the real aorta stays full width
    # and continues: the flood leaves through the tapering side wall, not the
    # end face, so it is the continuation rule that must catch it
    context = analyze(aorta_z_mm=(10.0, 85.0), mask_z_mm=(10.0, 60.0), mask_taper_mm=8.0)
    continuation = rejected(context, "aortic_continuation")
    assert len(continuation) == 1
    assert continuation[0]["cap_face_fraction"] < 0.5
    assert continuation[0]["radius_ratio"] > 0.4 or continuation[0]["angle_to_centerline_deg"] <= 25.0
    assert context["candidates"] == []


def test_small_perpendicular_branch_beside_a_cap_is_not_a_continuation():
    context = analyze(mask_z_mm=(10.0, 80.0), branches=[stub(z=77.0, radius=2.0)])
    assert len(context["candidates"]) == 1
    candidate = context["candidates"][0]
    assert candidate["touches_cap"] is True
    assert candidate["angle_to_centerline_deg"] > 60.0


def test_stub_shorter_than_5mm_is_rejected_as_too_short():
    # capsule ends are hemispherical: an axis ending at the wall with a 3mm
    # radius puts the vessel tip 3mm beyond it
    context = analyze(branches=[stub(length_beyond_wall=0.0)])
    assert len(rejected(context, "too_short")) == 1
    assert context["candidates"] == []


def test_common_trunk_is_one_origin_truncated_at_its_bifurcation():
    # a 6mm trunk dividing into two diverging daughters
    trunk_end = (WALL_X + 6.0, 32.0, 45.0)
    context = analyze(branches=[
        {"start": (36.0, 32.0, 45.0), "end": trunk_end, "radius": 2.5},
        {"start": trunk_end, "end": (WALL_X + 18.0, 32.0, 57.0), "radius": 2.0},
        {"start": trunk_end, "end": (WALL_X + 18.0, 32.0, 33.0), "radius": 2.0},
    ])
    assert len(context["instances"]) == 1
    bifurcation = context["bifurcations"][0]
    assert bifurcation["distance_mm"] == pytest.approx(7.5, abs=2.5)
    trace = context["traces"][0]
    assert trace["truncated_by"] == "bifurcation"
    assert trace["traced_length_mm"] <= bifurcation["distance_mm"] + 1e-6
    assert context["seed_estimates"][0]["reached_seed_distance"] is True


def test_division_beyond_10mm_is_truncated_at_10mm_not_at_the_division():
    trunk_end = (WALL_X + 10.5, 32.0, 45.0)
    context = analyze(branches=[
        {"start": (36.0, 32.0, 45.0), "end": trunk_end, "radius": 2.5},
        {"start": trunk_end, "end": (WALL_X + 20.0, 32.0, 53.0), "radius": 2.0},
        {"start": trunk_end, "end": (WALL_X + 20.0, 32.0, 37.0), "radius": 2.0},
    ])
    assert context["bifurcations"][0]["distance_mm"] > 10.0
    assert context["traces"][0]["truncated_by"] == "max_length"


def test_straight_stub_has_no_bifurcation():
    context = analyze(branches=[stub()])
    assert context["bifurcations"][0]["distance_mm"] is None


def test_two_separate_origins_are_two_instances():
    context = analyze(branches=[stub(z=30.0), stub(z=60.0)])
    assert len(context["instances"]) == 2
    assert all(instance["shares_vessel_with"] == [] for instance in context["instances"])
    heights = sorted(instance["patch_centroid_mm"][2] for instance in context["instances"])
    assert heights == pytest.approx([30.0, 60.0], abs=2.0)


def test_mouths_fused_at_the_wall_are_split_into_two_origins_by_watershed():
    # two parallel 4mm vessels 7mm apart: their lumens overlap by 1mm all
    # along, so the flood sees one component with one dumbbell-shaped patch
    # (at 1mm voxels, smaller mouths than this have no resolvable waist)
    context = analyze(branches=[stub(z=41.0, radius=4.0), stub(z=48.0, radius=4.0)])
    assert len(context["candidates"]) == 1
    instances = context["instances"]
    assert len(instances) == 2
    assert {instance["split"] for instance in instances} == {"watershed"}
    heights = sorted(instance["patch_centroid_mm"][2] for instance in instances)
    assert heights == pytest.approx([41.0, 48.0], abs=1.5)
    assert instances[0]["shares_vessel_with"] == [instances[1]["instance_id"]]


def test_disjoint_origins_whose_vessels_join_are_kept_but_share_a_vessel():
    # a U-shaped vessel leaving the aorta at z=36 and z=52, joined 5mm out:
    # the loop's far point is 5 + 8 = 13mm of vessel from either mouth, inside
    # the 15mm flood budget
    loop_x = WALL_X + 5.0
    context = analyze(branches=[
        {"start": (36.0, 32.0, 36.0), "end": (loop_x, 32.0, 36.0), "radius": 2.5},
        {"start": (36.0, 32.0, 52.0), "end": (loop_x, 32.0, 52.0), "radius": 2.5},
        {"start": (loop_x, 32.0, 36.0), "end": (loop_x, 32.0, 52.0), "radius": 2.5},
    ])
    assert len(context["candidates"]) == 1
    instances = context["instances"]
    assert len(instances) == 2
    assert all("disjoint_patches" in instance["split"] for instance in instances)
    first, second = instances
    assert first["shares_vessel_with"] == [second["instance_id"]]
    # they meet only out at the loop, never at the wall
    assert first["vessel_join_mm"][second["instance_id"]] > 8.0


def test_vessel_grazing_the_wall_downstream_is_still_one_origin():
    # leaves the aorta at z=60, then turns back down along it and brushes the
    # wall around z=40: the graze is a separate, narrow contact spot
    bend = (WALL_X + 7.0, 32.0, 57.0)
    context = analyze(branches=[
        {"start": (36.0, 32.0, 60.0), "end": bend, "radius": 3.0},
        {"start": bend, "end": (WALL_X + 4.0, 32.0, 40.0), "radius": 3.0},
    ])
    assert len(context["candidates"]) == 1
    instances = context["instances"]
    assert len(instances) == 1
    assert instances[0]["absorbed_narrow_pieces"] >= 1
    assert instances[0]["patch_centroid_mm"][2] > 52.0


def test_watershed_keeps_a_deep_saddle_and_merges_a_shallow_one():
    region = np.ones((1, 1, 13), dtype=bool)
    deep = np.array([[[1, 2, 3, 2.5, 2, 1.5, 1, 1.5, 2, 2.5, 3, 2, 1]]], dtype=float)
    labels, basins = watershed_basins(region, deep)
    assert len(basins) == 2
    assert labels[0, 0, 2] != labels[0, 0, 10]

    shallow = np.array([[[1, 2, 3, 2.8, 2.6, 2.5, 2.4, 2.5, 2.6, 2.8, 3, 2, 1]]], dtype=float)
    _labels, basins = watershed_basins(region, shallow)
    assert len(basins) == 1


def _flat_sheet_patch(inside):
    """A one-voxel-thick flat 'aortic surface' sheet with a patch on it."""
    size = 40
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    shell = np.ones((1, size, size), dtype=bool)
    patch = inside(yy, xx)[None, :, :]
    patch_flat = np.flatnonzero(patch)
    return patch_flat, shell


def _basins_of(patch_flat, shell):
    from src.parentage import patch_distance_transform

    low, closed, distance = patch_distance_transform(patch_flat, shell, shell.shape, (1.0, 1.0, 1.0))
    return watershed_basins(closed, distance)[1]


def test_dumbbell_patch_on_the_surface_splits_into_two_basins():
    def two_discs(yy, xx):
        return ((yy - 20) ** 2 + (xx - 15) ** 2 <= 16) | ((yy - 20) ** 2 + (xx - 22) ** 2 <= 16)

    assert len(_basins_of(*_flat_sheet_patch(two_discs))) == 2


def test_elongated_single_mouth_is_not_split():
    def ellipse(yy, xx):
        return ((yy - 20) / 3.5) ** 2 + ((xx - 20) / 7.5) ** 2 <= 1.0

    assert len(_basins_of(*_flat_sheet_patch(ellipse))) == 1


def test_exit_voxels_follow_parents_back_to_the_aorta():
    shape = (1, 1, 6)
    aorta = np.zeros(shape, dtype=bool)
    aorta[0, 0, 0] = True
    reachable = ~aorta
    parent = np.array([[[-1, 0, 1, 2, 3, 4]]], dtype=np.int32)
    exits = compute_exit_voxels(parent, reachable, aorta)
    assert exits[0, 0, 0] == -1
    assert np.all(exits[0, 0, 1:] == 1)


def test_analyze_case_runs_end_to_end_from_nifti_files(tmp_path):
    import SimpleITK as sitk

    from run import analyze_case

    image, mask = make_capsule_phantom(branches=[stub()])
    case_dir = tmp_path / "subject900"
    case_dir.mkdir()
    sitk.WriteImage(image, str(case_dir / "orig.nii.gz"))
    sitk.WriteImage(mask, str(case_dir / "mask.nii.gz"))

    # crop + 0.8mm resampling, exactly as real cases
    context = analyze_case(str(case_dir / "orig.nii.gz"), str(case_dir / "mask.nii.gz"))
    assert len(context["instances"]) == 1
    assert context["seed_estimates"][0]["direction_xyz"][0] > 0.95
    assert context["ostium_estimates"][0]["ostium_mm"] == pytest.approx([WALL_X, 32.0, 45.0], abs=1.5)
