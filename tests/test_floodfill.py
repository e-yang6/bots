"""Geodesic flood, traversal threshold and leak detection."""

import numpy as np
import pytest

from src.floodfill import (
    BAND_MM,
    build_traversal_volume,
    detect_leak,
    geodesic_bfs,
    lumen_reference_hu,
    robust_flood,
)
from tests.synthetic import make_capsule_phantom


def _stats(median, p25=None, p75=None):
    return {
        "median": median, "mad": 10.0,
        "p25": median - 10.0 if p25 is None else p25,
        "p75": median + 10.0 if p75 is None else p75,
        "n_voxels": 1000,
    }


def test_geodesic_distance_is_in_millimetres_on_an_anisotropic_grid():
    B = np.zeros((1, 3, 12), dtype=bool)
    B[0, 0, :] = True          # a row along x
    B[0, 1, 1] = True          # one voxel diagonal (x+1, y+1) from the seed
    seed = np.zeros_like(B)
    seed[0, 0, 0] = True

    spacing = (2.0, 1.0, 0.5)  # (x, y, z): x steps cost 2mm, y steps 1mm
    dist, parent, _frontier = geodesic_bfs(B, seed, spacing, budget_mm=100.0)

    assert dist[0, 0, 0] == 0.0
    np.testing.assert_allclose(dist[0, 0, :], 2.0 * np.arange(12), atol=1e-5)
    # the diagonal step is sqrt(2^2 + 1^2), not one hop and not 2 or 3mm
    assert dist[0, 1, 1] == pytest.approx(np.sqrt(5.0), abs=1e-5)
    assert np.isinf(dist[0, 2, 5])


def test_parent_chain_leads_back_to_the_seed():
    B = np.zeros((1, 1, 10), dtype=bool)
    B[0, 0, :] = True
    seed = np.zeros_like(B)
    seed[0, 0, 0] = True
    dist, parent, _frontier = geodesic_bfs(B, seed, (1.0, 1.0, 1.0), budget_mm=100.0)

    assert parent[0, 0, 0] == -1
    node, steps = int(np.ravel_multi_index((0, 0, 9), B.shape)), 0
    while parent.ravel()[node] >= 0:
        node = int(parent.ravel()[node])
        steps += 1
    assert node == 0
    assert steps == 9


def test_flood_stops_at_the_budget_and_counts_its_frontier():
    B = np.zeros((1, 1, 40), dtype=bool)
    B[0, 0, :] = True
    seed = np.zeros_like(B)
    seed[0, 0, 0] = True
    dist, _parent, frontier = geodesic_bfs(B, seed, (1.0, 1.0, 1.0), budget_mm=9.5)

    assert np.isfinite(dist[0, 0, 9]) and np.isinf(dist[0, 0, 10])
    # one voxel per mm beyond the seed, 9 of them, binned in 0.5mm bands
    assert frontier.sum() == 9
    assert frontier.size == int(np.ceil(9.5 / BAND_MM))


def test_distance_is_measured_beyond_the_aortic_surface_not_from_the_seed():
    B = np.zeros((1, 1, 30), dtype=bool)
    B[0, 0, :] = True
    aorta = np.zeros_like(B)
    aorta[0, 0, :20] = True
    seed = np.zeros_like(B)
    seed[0, 0, 2] = True  # deep inside the aorta, 17 voxels from its surface

    dist, _parent, frontier = geodesic_bfs(B, seed, (1.0, 1.0, 1.0), budget_mm=15.0, aorta_mask=aorta)
    assert dist[0, 0, 19] == 0.0             # all connected lumen is a source
    assert dist[0, 0, 20] == pytest.approx(1.0)
    assert dist[0, 0, 25] == pytest.approx(6.0)
    assert frontier.sum() == 10              # only voxels outside the aorta


def test_flood_does_not_cross_dark_voxels():
    B = np.zeros((1, 1, 20), dtype=bool)
    B[0, 0, :8] = True
    B[0, 0, 9:] = True  # a one-voxel dark gap at x=8
    seed = np.zeros_like(B)
    seed[0, 0, 0] = True
    dist, _parent, _frontier = geodesic_bfs(B, seed, (1.0, 1.0, 1.0), budget_mm=50.0)
    assert np.isfinite(dist[0, 0, 7])
    assert np.all(np.isinf(dist[0, 0, 9:]))


def test_threshold_scales_with_each_cases_own_lumen():
    image = np.zeros((5, 5, 5))
    _B, low_contrast = build_traversal_volume(image, _stats(90.0), fraction=0.5)
    _B, high_contrast = build_traversal_volume(image, _stats(580.0), fraction=0.5)
    assert low_contrast == pytest.approx(45.0)
    assert high_contrast == pytest.approx(290.0)

    # a skewed (thrombus + contrast) lumen is referenced to p75, like the
    # contrast gate in src.intensity
    skewed = _stats(120.0, p25=40.0, p75=400.0)
    assert lumen_reference_hu(skewed) == 400.0


def test_opening_breaks_a_thin_partial_volume_bridge():
    image = np.zeros((20, 20, 40))
    image[5:15, 5:15, 2:15] = 300.0
    image[5:15, 5:15, 25:38] = 300.0
    image[10, 10, 15:25] = 300.0  # a one-voxel bridge between the two blobs
    B, _threshold = build_traversal_volume(image, _stats(300.0), fraction=0.5)
    assert B[10, 10, 8] and B[10, 10, 30]
    assert not B[10, 10, 20]


def _sleeve_then(frontier_after_sleeve, sleeve=(4000, 3000, 2000, 1500)):
    return np.array(list(sleeve) + list(frontier_after_sleeve), dtype=float)


def test_flat_frontier_after_the_sleeve_is_not_a_leak():
    frontier = _sleeve_then([40] * 26)
    result = detect_leak(frontier, aorta_cross_section_voxels=200)
    assert result["leak"] is False
    assert result["growth"] == pytest.approx(1.0)


def test_huge_sleeve_bands_alone_do_not_flag_a_leak():
    # every real case starts with thousands of voxels per band here
    frontier = _sleeve_then([60] * 26, sleeve=(9000, 9000, 9000, 9000))
    assert detect_leak(frontier, aorta_cross_section_voxels=150)["leak"] is False


def test_sudden_frontier_explosion_is_a_leak():
    frontier = _sleeve_then([40] * 12 + [900] * 14)
    result = detect_leak(frontier, aorta_cross_section_voxels=200)
    assert result["leak"] is True
    assert result["reason"] == "explosion"
    assert result["band_index"] == 4 + 12


def test_steady_growth_into_tissue_is_a_leak():
    # subject001-like: never more than ~1.2x band to band, 3.5x overall
    frontier = _sleeve_then(np.linspace(300, 1100, 26))
    result = detect_leak(frontier, aorta_cross_section_voxels=170)
    assert result["leak"] is True
    assert result["reason"] == "growth"


def test_weak_growth_of_a_tiny_frontier_is_not_a_leak():
    # 1.5x growth, but on a few dozen voxels -- far below half an aortic
    # cross-section, where band counts are too noisy to read ratios into
    frontier = _sleeve_then(np.linspace(30, 50, 26))
    assert detect_leak(frontier, aorta_cross_section_voxels=400)["leak"] is False
    # the same ratio on a frontier bigger than that floor is a leak
    frontier = _sleeve_then(np.linspace(300, 500, 26))
    assert detect_leak(frontier, aorta_cross_section_voxels=400)["leak"] is True


def test_robust_flood_uses_the_lowest_threshold_that_does_not_leak():
    # A stub whose tip touches a big slab of enhancing tissue at 160 HU. Below
    # T=160 (fraction 0.533 of a 300 HU lumen) the flood pours into the slab.
    image, mask = make_capsule_phantom(
        shape_zyx=(60, 64, 96), aorta_centre_xy_mm=(32.0, 32.0), aorta_z_mm=(5.0, 55.0),
        branches=[{"start": (36.0, 32.0, 30.0), "end": (47.0, 32.0, 30.0), "radius": 3.0}],
        blocks=[{"low": (48.0, 8.0, 10.0), "high": (95.0, 56.0, 50.0), "hu": 160.0}],
    )
    from src.intensity import lumen_stats

    flood = robust_flood(image, mask, lumen_stats(image, mask))

    assert flood["attempts"][0]["leak"] is True
    assert flood["leak"]["leak"] is False
    assert 160.0 / 300.0 < flood["threshold_fraction"] < 0.60
    assert flood["retries"] == len(flood["attempts"]) - 1 <= 6
    # at the chosen threshold the stub is reached and the slab is not
    assert np.isfinite(flood["dist"][30, 32, 44])
    assert np.isinf(flood["dist"][30, 32, 70])


def _bone_contact_phantom(with_cortex):
    # The stub's tip meets a block through a 160 HU contact layer, so below
    # fraction 0.533 of the 300 HU lumen the flood gets into the block and
    # leaks. With cortex, the block is bone: a 900 HU shell (brighter than any
    # lumen voxel) around 160 HU marrow. Without, it is plain 160 HU tissue.
    # A dim 130 HU box vessel (fraction 0.433) leaves the opposite wall and is
    # only reachable once the search gets past the block's leak.
    # (the stub's rounded tip reaches x=50, so the contact runs past it)
    contact = {"low": (47.5, 26.0, 24.0), "high": (52.0, 38.0, 36.0), "hu": 160.0}
    dim_branch = {"low": (14.0, 30.0, 28.0), "high": (24.5, 34.0, 32.0), "hu": 130.0}
    if with_cortex:
        block = [{"low": (52.0, 18.0, 16.0), "high": (74.0, 46.0, 44.0), "hu": 900.0},
                 {"low": (55.0, 21.0, 19.0), "high": (71.0, 43.0, 41.0), "hu": 160.0}]
    else:
        block = [{"low": (52.0, 18.0, 16.0), "high": (74.0, 46.0, 44.0), "hu": 160.0}]
    return make_capsule_phantom(
        shape_zyx=(60, 64, 96), aorta_centre_xy_mm=(32.0, 32.0), aorta_z_mm=(5.0, 55.0),
        branches=[{"start": (36.0, 32.0, 30.0), "end": (47.0, 32.0, 30.0), "radius": 3.0}],
        blocks=block + [contact, dim_branch],
    )


def test_bone_leak_is_excised_and_the_search_continues_below_it():
    from src.intensity import lumen_stats

    image, mask = _bone_contact_phantom(with_cortex=True)
    flood = robust_flood(image, mask, lumen_stats(image, mask))

    bisection = flood["attempts"][:flood["bisection_attempts"]]
    settled = min(a["fraction"] for a in bisection if not a["leak"])
    assert settled > 160.0 / 300.0
    assert flood["leak"]["leak"] is False
    assert flood["threshold_fraction"] < 130.0 / 300.0
    assert flood["bone_excised_voxels"] > 0
    assert all(c["bright_share"] >= 0.10 for c in flood["bone_excised_components"])
    assert np.isfinite(flood["dist"][30, 32, 18])     # the dim branch is reached
    assert np.isfinite(flood["dist"][30, 32, 44])     # the bright stub still is
    assert np.isinf(flood["dist"][30, 32, 53])        # the cortex is not
    assert np.isinf(flood["dist"][30, 32, 57])        # nor the marrow behind it


def test_tissue_leak_still_stops_the_search_where_bisection_left_it():
    from src.intensity import lumen_stats

    image, mask = _bone_contact_phantom(with_cortex=False)
    flood = robust_flood(image, mask, lumen_stats(image, mask))

    assert flood["attempts"][flood["bisection_attempts"]:] != []   # the descent was tried
    assert flood["bone_excised_voxels"] == 0
    assert flood["leak"]["leak"] is False
    assert flood["threshold_fraction"] > 160.0 / 300.0
    assert np.isinf(flood["dist"][30, 32, 18])        # the dim branch stays out of reach
    assert np.isinf(flood["dist"][30, 32, 57])        # and so does the tissue


def test_robust_flood_needs_no_retries_when_nothing_leaks():
    image, mask = make_capsule_phantom(
        branches=[{"start": (36.0, 32.0, 45.0), "end": (58.0, 32.0, 45.0), "radius": 3.0}],
    )
    from src.intensity import lumen_stats

    flood = robust_flood(image, mask, lumen_stats(image, mask))
    assert flood["retries"] == 0
    assert flood["threshold_fraction"] == pytest.approx(0.40)
