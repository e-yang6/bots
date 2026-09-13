"""Seeded geodesic flood fill from the aortic lumen through bright voxels.

Why connectivity rather than brightness: bright does not mean artery. Bone,
bowel contrast and enhancing kidney parenchyma are all as bright as a
contrast-filled vessel, but none of them connect to the aortic lumen through
continuous contrast, while every direct daughter artery does. A flood that
starts inside the aorta and may only step through bright voxels therefore
reaches the aorta, its branches, and very little else.

Arrays are (z, y, x) index order throughout (sitk.GetArrayFromImage layout);
`spacing` arguments are SimpleITK (x, y, z) order, as GetSpacing() returns.


What the real data added to that picture
----------------------------------------
Two things, both measured over all 20 dev cases (frontier voxel counts per
distance band at thresholds from 0.45 to 0.85 of the lumen reference):

1. The supplied masks sit ~2 voxels inside the bright lumen. The first ~2mm
   beyond the mask surface is therefore a sleeve of genuine lumen wrapping the
   entire aorta: 2,000-10,000 frontier voxels per band in every case, against
   a few hundred once past it. Any leak test has to start beyond that sleeve
   or it flags every case at the first band (see SLEEVE_MM).

2. Leaks here are rarely an explosion. The "frontier suddenly jumps 8x"
   signature assumes the flood escapes through one thin bridge into open
   territory; what actually happens at too low a threshold is that it oozes
   into enhancing tissue next to the aorta and the frontier grows steadily --
   e.g. subject001 at 0.45 of its lumen HU goes 1,490 -> 5,546 voxels per mm
   between 3 and 15mm, with no single band more than ~1.2x its predecessor.
   A flood confined to branches instead keeps a flat or shrinking frontier
   (growth 0.5-1.6x over the same span, on every case where the threshold
   was high enough), while leaking floods grow 1.8-4x. So a growth rule is
   checked alongside the explosion rule (see detect_leak).

3. On the 1.5mm eval-set cases, the leak that pins the threshold search is
   often not tissue at all but vertebral bone pressed against the aortic
   wall: its partial-volume ramp crosses the lumen's HU on the way to cortex,
   so the flood walks straight in, the frontier grows, and every threshold
   below that point is rejected -- taking every dimmer daughter elsewhere on
   the aorta down with it (case_22 settled at 0.84 of its lumen for exactly
   this reason; case_20 at 0.60). Bone is separable from contrast-filled
   vessel where tissue is not: nothing the aorta feeds can be brighter than
   the aorta itself, so a newly reached component with a substantial share
   of voxels above the lumen's own upper tail is not a vessel. After the
   ordinary search settles, robust_flood therefore keeps stepping down,
   excising such components and re-testing, and stops at the first leak
   that excision does not explain (see _bone_like and robust_flood). Tissue
   leaks -- an organ bed a branch runs into -- are untouched by this and
   still stop the search, exactly as before.
"""

import sys

import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from src.intensity import SKEW_IQR_THRESHOLD_HU

DEFAULT_BUDGET_MM = 15.0
BAND_MM = 0.5

# Threshold search range, as fractions of the per-case lumen reference HU. The
# cohort spans lumen medians from 86 to 581 HU, so no fixed HU value can work;
# a fraction of the case's own lumen carries across it. 0.40 is low enough
# that 5 of 20 dev cases flood without leaking even there; 0.90 is the most
# that still keeps most of the lumen itself above threshold.
THRESHOLD_FRACTION_RANGE = (0.40, 0.90)
MAX_SEARCH_ITERATIONS = 5

# Depth of the partial-volume sleeve between the mask surface and the true
# lumen edge (see module docstring, point 1). Across the dev cases the
# frontier falls to its plateau between 1.5 and 2.0mm.
SLEEVE_MM = 2.0

# Leak rules; see detect_leak for where each number comes from.
EXPLOSION_RATIO = 8.0
CROSS_SECTION_FRACTION = 0.5
STRONG_GROWTH_RATIO = 2.0
WEAK_GROWTH_RATIO = 1.25
EARLY_WINDOW_MM = (3.0, 6.0)
LATE_WINDOW_MM = 3.0
MAX_VISITED_RATIO = 40.0

# Bone excision; see module docstring point 3 and _bone_like.
# The ceiling is this percentile of the eroded lumen's own HU: contrast blood
# downstream of the aorta cannot be brighter than the aorta, so a genuine
# vessel has about (100 - percentile)% of its voxels above it at most, and
# fewer once partial volume dims it.
BONE_CEILING_PERCENTILE = 99.0
# 10x that 1% noise rate. On the draft eval set the large components rendered
# and confirmed as vertebral bone ran 0.16-0.47, and the tissue, organ and
# branch components beside them 0.00-0.04.
BONE_MIN_BRIGHT_SHARE = 0.10
# Ignore specks: a few noisy bright voxels are not a structure.
BONE_MIN_COMPONENT_MM3 = 10.0
# Descent step below the settled fraction, and excise/re-flood rounds per step.
BONE_DESCENT_STEP = 0.025
MAX_EXCISION_ROUNDS = 3

_CROSS = ndimage.generate_binary_structure(3, 1)
_FULL = np.ones((3, 3, 3), dtype=bool)


def lumen_reference_hu(lumen_stats):
    """The lumen intensity thresholds are scaled from.

    Same choice intensity.is_contrast_enhanced makes: p75 for skewed
    (thrombus + contrast) lumens, whose median reads low, else the median.
    """
    iqr = lumen_stats["p75"] - lumen_stats["p25"]
    return lumen_stats["p75"] if iqr > SKEW_IQR_THRESHOLD_HU else lumen_stats["median"]


def _as_array(image):
    if hasattr(image, "GetSize"):
        import SimpleITK as sitk

        return sitk.GetArrayFromImage(image)
    return np.asarray(image)


def build_traversal_volume(image, lumen_stats, fraction=THRESHOLD_FRACTION_RANGE[0]):
    """Binary volume of voxels the flood may step through.

    T = fraction * lumen reference HU. B = image >= T, then a binary opening
    with a 1-voxel ball to break the thin partial-volume bridges that let a
    flood escape into neighbouring bright structures.

    Returns (B, T).
    """
    threshold = float(fraction) * float(lumen_reference_hu(lumen_stats))
    traversal = _as_array(image) >= threshold
    traversal = ndimage.binary_opening(traversal, structure=_CROSS)
    return traversal, threshold


def aorta_cross_section_voxels(aorta_mask):
    """Typical aortic cross-section, in voxels: the median per-slice count
    over array z-slices containing the mask. Slightly overestimates where the
    aorta runs obliquely to z, which only makes the leak floor more lenient.
    """
    mask = np.asarray(aorta_mask, dtype=bool)
    per_slice = mask.sum(axis=(1, 2))
    per_slice = per_slice[per_slice > 0]
    return float(np.median(per_slice)) if per_slice.size else 0.0


def _neighbour_offsets():
    """The 13 'forward' offsets of the 26-neighbourhood (each edge once)."""
    return [
        (dz, dy, dx)
        for dz in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if (dz, dy, dx) > (0, 0, 0)
    ]


def _pair_slices(offset, shape):
    """Slices a, b such that array[a] and array[b] are neighbours at offset."""
    slices_a, slices_b = [], []
    for delta, size in zip(offset, shape):
        slices_a.append(slice(max(0, -delta), size - max(0, delta)))
        slices_b.append(slice(max(0, delta), size - max(0, -delta)))
    return tuple(slices_a), tuple(slices_b)


def _source_region(traversal, seed_mask, aorta_mask):
    """In-aorta lumen voxels reachable from the seeds at zero cost.

    Movement inside the aorta is free: distances are measured from the aortic
    lumen outward, so a branch's geodesic distance is its distance beyond the
    aortic surface, wherever along the aorta the flood happened to enter it.
    """
    lumen = aorta_mask & traversal
    labels, _count = ndimage.label(lumen, structure=_FULL)
    seeded = np.unique(labels[seed_mask & lumen])
    seeded = seeded[seeded > 0]
    if seeded.size == 0:
        return np.zeros_like(lumen)
    return np.isin(labels, seeded)


def _flood(traversal, source, aorta_mask, spacing, budget_mm, candidate_region=None, band_mm=BAND_MM,
           return_nearest_source=False):
    """Dijkstra from `source` through `traversal`, outside the aorta.

    The graph holds only the voxels that can matter: bright voxels outside the
    aorta (optionally restricted to `candidate_region`, which must contain
    everything within budget) plus the in-aorta source voxels. Aorta voxels
    that are not sources -- dark thrombus, or lumen not connected to a seed --
    are excluded, so the flood can neither pass through nor restart from them.

    Dijkstra itself is scipy.sparse.csgraph's C implementation with a distance
    limit, which is exactly a heap-based Dijkstra that stops at budget_mm. The
    threshold search re-runs this up to seven times per case, so it is worth
    not doing the relaxations in Python.

    With return_nearest_source, also returns an int32 volume holding the flat
    index of the source each voxel was reached from (-1 where unreached).
    """
    shape = traversal.shape
    if np.prod(shape, dtype=np.int64) >= np.iinfo(np.int32).max:
        raise ValueError("volume too large for int32 parent indices")

    spacing_zyx = np.asarray(spacing, dtype=np.float64)[::-1]
    dist = np.full(shape, np.inf, dtype=np.float32)
    parent = np.full(shape, -1, dtype=np.int32)
    nearest_source = np.full(shape, -1, dtype=np.int32) if return_nearest_source else None
    n_bands = int(np.ceil(budget_mm / band_mm))
    frontier = np.zeros(n_bands, dtype=np.int64)

    outside = traversal & ~aorta_mask
    if candidate_region is not None:
        outside &= candidate_region
    nodes = outside | source
    node_flat = np.flatnonzero(nodes).astype(np.int32)
    if node_flat.size == 0 or not source.any():
        if return_nearest_source:
            return dist, parent, frontier, nearest_source
        return dist, parent, frontier

    index = np.full(shape, -1, dtype=np.int32)
    index.ravel()[node_flat] = np.arange(node_flat.size, dtype=np.int32)

    rows, cols, weights = [], [], []
    for offset in _neighbour_offsets():
        slices_a, slices_b = _pair_slices(offset, shape)
        a = index[slices_a]
        b = index[slices_b]
        # both ends in the graph, and not both sources (source-source moves are
        # free and never needed: every source already sits at distance 0)
        linked = (a >= 0) & (b >= 0) & ~(source[slices_a] & source[slices_b])
        rows.append(a[linked])
        cols.append(b[linked])
        step_mm = float(np.sqrt(np.sum((np.asarray(offset) * spacing_zyx) ** 2)))
        weights.append(np.full(rows[-1].size, step_mm, dtype=np.float64))

    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    weights = np.concatenate(weights)
    graph = coo_matrix((weights, (rows, cols)), shape=(node_flat.size, node_flat.size)).tocsr()

    source_nodes = index.ravel()[np.flatnonzero(source)]
    node_dist, predecessors, sources = dijkstra(
        graph, directed=False, indices=source_nodes, limit=budget_mm,
        min_only=True, return_predecessors=True,
    )

    dist.ravel()[node_flat] = node_dist.astype(np.float32)
    dist[source] = 0.0
    has_parent = predecessors >= 0
    parent.ravel()[node_flat[has_parent]] = node_flat[predecessors[has_parent]]

    reached_outside = np.isfinite(node_dist) & ~source.ravel()[node_flat]
    bands = np.minimum((node_dist[reached_outside] / band_mm).astype(np.int64), n_bands - 1)
    frontier = np.bincount(bands, minlength=n_bands)[:n_bands]
    if return_nearest_source:
        has_source = sources >= 0
        nearest_source.ravel()[node_flat[has_source]] = node_flat[sources[has_source]]
        return dist, parent, frontier, nearest_source
    return dist, parent, frontier


def geodesic_bfs(B, seed_mask, spacing, budget_mm=DEFAULT_BUDGET_MM, aorta_mask=None, band_mm=BAND_MM):
    """Geodesic (millimetre-weighted) flood from seed_mask through B.

    26-connectivity; each edge costs the true inter-voxel distance from
    `spacing` ((x, y, z), SimpleITK order). An unweighted hop count would be
    wrong even on an isotropic grid -- a diagonal step is sqrt(3) times
    longer than a face step -- and badly wrong on native CT spacings.

    If aorta_mask is given, the whole lumen connected to the seeds inside it
    is a zero-distance source, so distances are measured beyond the aortic
    surface and the flood stops budget_mm past it. Without it, seed_mask
    alone is the source.

    Returns (dist, parent, frontier_sizes):
      dist            float32, mm; 0 at sources, inf where unreached
      parent          int32 flat index into the volume of each voxel's
                      predecessor on its shortest path; -1 at sources and
                      unreached voxels
      frontier_sizes  voxels outside the aorta per band_mm distance band
    """
    B = np.asarray(B, dtype=bool)
    seed_mask = np.asarray(seed_mask, dtype=bool)
    aorta = seed_mask if aorta_mask is None else np.asarray(aorta_mask, dtype=bool)
    source = _source_region(B, seed_mask, aorta)
    return _flood(B, source, aorta, spacing, budget_mm, band_mm=band_mm)


def detect_leak(
    frontier_sizes,
    aorta_cross_section_voxels,
    band_mm=BAND_MM,
    sleeve_mm=SLEEVE_MM,
    total_visited=None,
    aorta_voxels=None,
    explosion_ratio=EXPLOSION_RATIO,
    cross_section_fraction=CROSS_SECTION_FRACTION,
    strong_growth_ratio=STRONG_GROWTH_RATIO,
    weak_growth_ratio=WEAK_GROWTH_RATIO,
    early_window_mm=EARLY_WINDOW_MM,
    late_window_mm=LATE_WINDOW_MM,
    max_visited_ratio=MAX_VISITED_RATIO,
):
    """Decide whether a flood escaped the vessel tree.

    A real vessel's frontier stays near its own cross-section, a few dozen
    voxels per band; a flood that has escaped into tissue grows. Rules, in
    order (the first that fires is reported):

      explosion  a band beyond the sleeve exceeds
                 max(explosion_ratio * median of the previous post-sleeve
                 bands, cross_section_fraction * aortic cross-section).
                 The sudden escape through a partial-volume bridge.
      growth     the median band in the last late_window_mm of the budget is
                 at least strong_growth_ratio times the median band in
                 early_window_mm; or at least weak_growth_ratio times it
                 while also above cross_section_fraction of an aortic
                 cross-section. The gradual ooze into enhancing tissue that
                 is what leaks actually look like on this data. On the dev
                 set, flat branch-only floods sit at 0.5-1.6x; the weak ratio
                 only counts when the frontier is also big, which is what
                 keeps the noisy ratios of tiny frontiers (~40 voxels) from
                 firing.
      visited    total_visited exceeds max_visited_ratio times the aorta's own
                 voxel count. A runaway guard: under a 15mm budget the
                 reachable region is bounded to a few aortic volumes, so on
                 real anatomy the growth rule always fires first.

    Bands inside sleeve_mm are ignored by every rule (see module docstring).

    Returns a dict with "leak", "reason", "band_index" and the numbers the
    rules were evaluated on, for logging.
    """
    frontier = np.asarray(frontier_sizes, dtype=np.float64)
    floor = cross_section_fraction * float(aorta_cross_section_voxels)
    first_band = int(np.ceil(sleeve_mm / band_mm))

    result = {
        "leak": False, "reason": None, "band_index": None,
        "growth": None, "early_frontier": None, "late_frontier": None,
        "cross_section_floor": floor,
    }

    for band in range(first_band + 2, frontier.size):
        previous = frontier[first_band:band]
        limit = max(explosion_ratio * float(np.median(previous)), floor)
        if frontier[band] > limit:
            result.update(leak=True, reason="explosion", band_index=band)
            break

    early_lo = int(round(early_window_mm[0] / band_mm))
    early_hi = int(round(early_window_mm[1] / band_mm))
    late_lo = frontier.size - int(round(late_window_mm / band_mm))
    if early_lo >= first_band and early_hi <= late_lo and early_hi > early_lo:
        early = float(np.median(frontier[early_lo:early_hi]))
        late = float(np.median(frontier[late_lo:]))
        growth = late / early if early > 0 else (np.inf if late > 0 else 0.0)
        result.update(growth=float(growth), early_frontier=early, late_frontier=late)
        grows = growth >= strong_growth_ratio or (growth >= weak_growth_ratio and late > floor)
        if grows and not result["leak"]:
            result.update(leak=True, reason="growth", band_index=late_lo)

    if (
        not result["leak"]
        and total_visited is not None
        and aorta_voxels
        and total_visited > max_visited_ratio * aorta_voxels
    ):
        result.update(leak=True, reason="visited")

    return result


def _bone_like(new_territory, hu, ceiling_hu, min_voxels,
               min_bright_share=BONE_MIN_BRIGHT_SHARE):
    """Union of the 26-connected components of new_territory that are bone.

    A component is bone-like when at least min_bright_share of its voxels
    are brighter than ceiling_hu (the lumen's own upper tail) and it has at
    least min_voxels voxels. Returns (mask, one summary dict per excised
    component).
    """
    labels, count = ndimage.label(new_territory, structure=_FULL)
    if count == 0:
        return np.zeros_like(new_territory), []
    flat = labels.ravel()
    sizes = np.bincount(flat, minlength=count + 1)
    bright = np.bincount(flat, weights=(hu > ceiling_hu).ravel().astype(np.float64), minlength=count + 1)
    share = bright / np.maximum(sizes, 1)
    keep = (sizes >= min_voxels) & (share >= min_bright_share)
    keep[0] = False
    excised = [{"voxels": int(sizes[k]), "bright_share": float(share[k])} for k in np.flatnonzero(keep)]
    return keep[labels], excised


def robust_flood(
    image,
    aorta_mask,
    lumen_stats,
    budget_mm=DEFAULT_BUDGET_MM,
    fraction_range=THRESHOLD_FRACTION_RANGE,
    max_iterations=MAX_SEARCH_ITERATIONS,
    band_mm=BAND_MM,
    verbose=False,
    file=sys.stderr,
):
    """Flood at the lowest threshold that does not leak.

    Tries the bottom of fraction_range first; if that leaks, binary-searches
    upward for at most max_iterations and keeps the LOWEST non-leaking
    threshold found -- too high a threshold and small branches thin out and
    vanish before reaching 5mm. If nothing in range stops leaking, the top of
    the range is used and the result stays flagged as leaking.

    If the search settled on a non-leaking threshold with a leak below it,
    it then steps down by BONE_DESCENT_STEP. At each step, if the flood
    leaks, the bone-like components (see _bone_like) of the territory it
    newly reached beyond the previous accepted flood are excised from the
    traversal volume and the step is re-flooded; the step is accepted only
    once it stops leaking. The descent ends at the first step whose leak
    excision does not remove, or at the bottom of fraction_range. Excised
    voxels stay excised for every lower step. A case that never leaks, or
    whose leak is not bone, ends exactly where the bisection left it.

    image / aorta_mask: SimpleITK images (or arrays, with spacing taken as
    isotropic 1mm -- tests only).

    Returns a dict: dist, parent, frontier_sizes, traversal (B), threshold_hu,
    threshold_fraction, lumen_reference_hu, leak (detect_leak's dict),
    retries, attempts (one summary per flood), budget_mm, band_mm, spacing.
    """
    spacing = tuple(image.GetSpacing()) if hasattr(image, "GetSpacing") else (1.0, 1.0, 1.0)
    hu = _as_array(image).astype(np.float32, copy=False)
    mask = _as_array(aorta_mask).astype(bool)

    seeds = ndimage.binary_erosion(mask, structure=_CROSS)
    if not seeds.any():
        seeds = mask
    cross_section = aorta_cross_section_voxels(mask)
    aorta_voxels = int(mask.sum())

    # Geodesic distance is never shorter than straight-line distance, so
    # nothing further than budget (plus one diagonal step of slack) from the
    # mask can be reached; leaving it out keeps every graph small.
    spacing_zyx = np.asarray(spacing, dtype=np.float64)[::-1]
    outside_distance = ndimage.distance_transform_edt(~mask, sampling=spacing_zyx)
    candidate_region = outside_distance <= budget_mm + float(np.linalg.norm(spacing_zyx))
    del outside_distance

    reference = float(lumen_reference_hu(lumen_stats))
    attempts = []

    def attempt(fraction, excised=None):
        traversal, threshold = build_traversal_volume(hu, lumen_stats, fraction)
        if excised is not None:
            traversal &= ~excised
        source = _source_region(traversal, seeds, mask)
        dist, parent, frontier = _flood(
            traversal, source, mask, spacing, budget_mm, candidate_region, band_mm
        )
        visited = int(frontier.sum())
        leak = detect_leak(
            frontier, cross_section, band_mm=band_mm,
            total_visited=visited, aorta_voxels=aorta_voxels,
        )
        attempts.append({
            "fraction": float(fraction), "threshold_hu": threshold, "leak": leak["leak"],
            "reason": leak["reason"], "growth": leak["growth"], "visited": visited,
            "excised_voxels": 0 if excised is None else int(excised.sum()),
        })
        return {
            "dist": dist, "parent": parent, "frontier_sizes": frontier, "traversal": traversal,
            "threshold_hu": threshold, "threshold_fraction": float(fraction), "leak": leak,
        }

    low, high = fraction_range
    chosen = attempt(low)
    if chosen["leak"]["leak"]:
        chosen = None
        for _ in range(max_iterations):
            middle = 0.5 * (low + high)
            result = attempt(middle)
            if result["leak"]["leak"]:
                low = middle
            else:
                high = middle
                chosen = result
        if chosen is None:
            chosen = attempt(fraction_range[1])

    bisection_attempts = len(attempts)
    leaked_below = any(a["leak"] and a["fraction"] < chosen["threshold_fraction"] for a in attempts)
    ceiling = float(np.percentile(hu[seeds], BONE_CEILING_PERCENTILE))
    excised = np.zeros_like(mask)
    excised_components = []
    if not chosen["leak"]["leak"] and leaked_below:
        min_voxels = BONE_MIN_COMPONENT_MM3 / float(np.prod(spacing))
        # Nothing the bisection's own flood reached is ever excised, so the
        # descent adds territory rather than trading some away.
        protected = np.isfinite(chosen["dist"]) | mask
        fraction = chosen["threshold_fraction"] - BONE_DESCENT_STEP
        while fraction >= fraction_range[0] - 1e-9:
            accepted_reach = np.isfinite(chosen["dist"]) & ~mask
            trial = excised.copy()
            trial_components = []
            result = attempt(fraction, trial)
            for _ in range(MAX_EXCISION_ROUNDS):
                if not result["leak"]["leak"]:
                    break
                reached = np.isfinite(result["dist"]) & ~mask
                bone, components = _bone_like(reached & ~accepted_reach, hu, ceiling, min_voxels)
                if not components:
                    break
                trial |= bone & ~protected
                trial_components.extend(components)
                result = attempt(fraction, trial)
            if result["leak"]["leak"]:
                break
            chosen, excised = result, trial
            excised_components.extend(trial_components)
            fraction -= BONE_DESCENT_STEP

    chosen.update({
        "lumen_reference_hu": reference,
        "retries": len(attempts) - 1,
        "attempts": attempts,
        "budget_mm": float(budget_mm),
        "band_mm": float(band_mm),
        "spacing": spacing,
        "aorta_cross_section_voxels": cross_section,
        "bisection_attempts": bisection_attempts,
        "bone_ceiling_hu": ceiling,
        "bone_excised_voxels": int(excised.sum()),
        "bone_excised_components": excised_components,
    })

    if verbose:
        log_flood(chosen, file=file)
    return chosen


def log_flood(flood, file=sys.stderr):
    leak = flood["leak"]
    growth = "n/a" if leak["growth"] is None else f"{leak['growth']:.2f}"
    print(
        f"flood: T={flood['threshold_hu']:.0f}HU ({flood['threshold_fraction']:.3f} x lumen ref "
        f"{flood['lumen_reference_hu']:.0f}HU) retries={flood['retries']} "
        f"leak={leak['reason'] if leak['leak'] else 'none'} growth={growth} "
        f"reached={int(flood['frontier_sizes'].sum())}",
        file=file,
    )
    split = flood.get("bisection_attempts", len(flood["attempts"]))

    def step(a):
        cut = f"(-{a['excised_voxels']}bone)" if a.get("excised_voxels") else ""
        return f"{a['fraction']:.3f}{cut}->{'LEAK:' + a['reason'] if a['leak'] else 'ok'}"

    print("  search: " + " ".join(step(a) for a in flood["attempts"][:split]), file=file)
    if len(flood["attempts"]) > split:
        print(
            f"  bone descent (ceiling {flood['bone_ceiling_hu']:.0f}HU, excised "
            f"{flood['bone_excised_voxels']} voxels in {len(flood['bone_excised_components'])} components): "
            + " ".join(step(a) for a in flood["attempts"][split:]),
            file=file,
        )
