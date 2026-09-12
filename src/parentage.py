"""Split candidate components into independent aortic origins, and record
which origins lie on a shared vessel.

Connectivity makes this cheap, because the flood already answers the two
questions the surface-scoring detector had to approximate with path-to-path
proximity:

  Do two contact patches lie on one vessel?  Then a path between them runs
  entirely through that vessel's lumen and never re-enters the aorta mask.
  Every flood voxel's shortest path leaves the aorta through exactly one
  contact-patch voxel, so each patch owns a territory -- the voxels fed
  through it -- and those territories touch only where the vessels do. The
  lowest geodesic distance at which two territories touch (vessel_join_mm) is
  the bottleneck of the best path between the patches that stays outside the
  aorta: every voxel's parent chain lies inside its own territory with
  strictly decreasing distance, so no path can join them lower. It is
  reported per pair.

  Are two nearby origins really two?  Yes when their contact patches are
  spatially disjoint on the aortic surface. And a single patch can still hold
  two origins: two ostia a few mm apart (a main and an accessory renal, the
  coeliac beside the SMA) fuse through partial volume at the wall into one
  bimodal patch. Those are split by watershed on the patch's own distance
  transform, measured along the aortic surface.

  Where the two rules collide -- a disjoint piece of patch that lies on the
  same vessel as another -- width decides. A piece too narrow to be the mouth
  of an eligible vessel is a vessel grazing the wall downstream of its real
  origin, and is folded into that origin (see _absorb_narrow_pieces). A piece
  wide enough to be a mouth stays its own origin, with the shared vessel
  recorded.

A daughter-of-a-daughter never touches the aorta, so it never gets a contact
patch of its own: it is simply part of its parent's component, and tracing
truncates at the bifurcation (see tracing.detect_bifurcation_by_frontier).

The split is the independence decision: every instance returned here is its
own origin. shares_vessel_with is reported alongside for the classifier
stage, which will decide what a disjoint origin that joins another vessel
3mm out actually is.
"""

import heapq
import sys

import numpy as np
from scipy import ndimage

from src.candidates import MIN_ELIGIBLE_LENGTH_MM, flat_to_mm, flat_to_zyx, voxel_volume_mm3
from src.floodfill import _neighbour_offsets, _pair_slices, geodesic_bfs

# A second watershed basin survives as its own origin only if its peak (the
# half-width of its patch, in mm along the surface) stands at least this far
# above the saddle joining it to a higher basin -- and by at least this
# fraction of its own peak. The absolute floor is one voxel: along a voxel
# surface, geodesic distance itself wobbles by ~0.3mm between adjacent voxels,
# so shallower dips are quantization, not a waist between two mouths. The
# fraction is small because real waists are shallow: two equal mouths of
# radius R overlapping by R/2 dip by only ~0.3R (half-width sqrt(R * overlap)),
# and a 1mm voxel graph measures less than that again, while a single
# elliptical mouth has no dip at all beyond the quantization the floor covers.
WATERSHED_MIN_DEPTH_MM = 0.8
WATERSHED_DEPTH_FRACTION = 0.2
# Basins narrower than this cannot be the mouth of an eligible vessel.
MIN_OSTIUM_HALF_WIDTH_MM = 1.0

_FULL = np.ones((3, 3, 3), dtype=bool)


def _crop_bounds(zyx, shape, margin):
    low = np.maximum(zyx.min(axis=0) - margin, 0)
    high = np.minimum(zyx.max(axis=0) + margin + 1, shape)
    return low, high


def patch_distance_transform(patch_flat, shell, shape, spacing):
    """Geodesic distance from each patch voxel to the patch rim, along the
    aortic surface.

    A plain 3D distance transform is useless on a patch: it is a curved sheet
    one or two voxels thick, so every voxel is ~1 voxel from background. The
    distance that says how wide the patch is runs along the surface, so this
    floods the shell from the shell voxels that are NOT in the patch.

    Holes are filled first -- pockets of shell entirely enclosed by the patch
    -- since each would otherwise read as rim in the middle of the mouth. Not
    a morphological closing: closing also fills the notch between two fused
    mouths, which is precisely the waist the watershed needs to see.

    Returns (low_zyx, filled_patch_crop, distance_crop), or None if the patch
    has no rim in its neighbourhood (it wraps the aorta).
    """
    zyx = flat_to_zyx(patch_flat, shape)
    low, high = _crop_bounds(zyx, shape, margin=3)
    crop = tuple(slice(l, h) for l, h in zip(low, high))
    shell_crop = shell[crop]

    patch_crop = np.zeros(shell_crop.shape, dtype=bool)
    local = zyx - low
    patch_crop[local[:, 0], local[:, 1], local[:, 2]] = True

    outside_patch, _count = ndimage.label(shell_crop & ~patch_crop, structure=_FULL)
    border = np.zeros(shell_crop.shape, dtype=bool)
    for axis in range(3):
        border[(slice(None),) * axis + (0,)] = True
        border[(slice(None),) * axis + (-1,)] = True
    open_labels = np.unique(outside_patch[border & (outside_patch > 0)])
    holes = (outside_patch > 0) & ~np.isin(outside_patch, open_labels)
    closed = patch_crop | holes

    rim = shell_crop & ~closed
    if not rim.any():
        return None
    # budget: comfortably wider than any ostium's half-width
    distance, _parent, _frontier = geodesic_bfs(shell_crop | closed, rim, spacing, budget_mm=50.0)
    distance[~np.isfinite(distance)] = 50.0
    distance[~closed] = 0.0
    return low, closed, distance


def watershed_basins(region, height, min_depth_mm=WATERSHED_MIN_DEPTH_MM,
                     depth_fraction=WATERSHED_DEPTH_FRACTION,
                     min_peak_mm=MIN_OSTIUM_HALF_WIDTH_MM):
    """Watershed of `height` over `region` (26-connected), from the top down.

    Two passes. First, which basins exist: voxels are added in decreasing
    height; a voxel touching no processed voxel starts a basin, and one
    touching several is a saddle, where every basin except the highest either
    merges into it (too shallow or too narrow) or stays separate. Depth only
    grows as the flood descends, so a basin kept separate at its first saddle
    stays separate.

    Second, which voxels belong to each: a priority flood from the surviving
    peaks, highest voxels first and first-come-first-served within a height.
    The first pass alone would settle every tie on a plateau by array order,
    handing most of a flat-topped mouth to whichever basin was scanned first.

    Returns (labels, basins): labels is an int array over region (0 outside,
    1..k basins, ordered by decreasing peak); basins is a list of
    {"label", "peak_mm", "saddle_mm"} (saddle None for the highest basin of
    each connected piece).
    """
    coords = np.argwhere(region)
    labels = np.zeros(region.shape, dtype=np.int32)
    if coords.shape[0] == 0:
        return labels, []

    values = height[region].astype(np.float64)
    index = np.full(region.shape, -1, dtype=np.int64)
    index[coords[:, 0], coords[:, 1], coords[:, 2]] = np.arange(coords.shape[0])

    union = np.arange(coords.shape[0])
    peak = values.copy()
    saddle = {}
    processed = np.zeros(coords.shape[0], dtype=bool)
    offsets = [(dz, dy, dx) for dz in (-1, 0, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
               if (dz, dy, dx) != (0, 0, 0)]
    shape = region.shape

    def find(node):
        while union[node] != node:
            union[node] = union[union[node]]
            node = union[node]
        return node

    for node in np.argsort(-values, kind="stable"):
        z, y, x = coords[node]
        roots = set()
        for dz, dy, dx in offsets:
            nz, ny, nx = z + dz, y + dy, x + dx
            if 0 <= nz < shape[0] and 0 <= ny < shape[1] and 0 <= nx < shape[2]:
                neighbour = index[nz, ny, nx]
                if neighbour >= 0 and processed[neighbour]:
                    roots.add(find(neighbour))
        processed[node] = True
        if not roots:
            continue

        ranked = sorted(roots, key=lambda root: -peak[root])
        highest = ranked[0]
        union[node] = highest
        for root in ranked[1:]:
            depth = peak[root] - values[node]
            if peak[root] < min_peak_mm or depth < max(min_depth_mm, depth_fraction * peak[root]):
                union[root] = highest
            else:
                saddle[root] = max(saddle.get(root, -np.inf), float(values[node]))

    unique_roots = sorted({find(node) for node in range(coords.shape[0])}, key=lambda root: -peak[root])
    relabel = {root: i + 1 for i, root in enumerate(unique_roots)}

    node_label = np.zeros(coords.shape[0], dtype=np.int32)
    queue, counter = [], 0
    for root in unique_roots:
        node_label[root] = relabel[root]
        heapq.heappush(queue, (-values[root], counter, root))
        counter += 1
    while queue:
        _height, _order, node = heapq.heappop(queue)
        z, y, x = coords[node]
        for dz, dy, dx in offsets:
            nz, ny, nx = z + dz, y + dy, x + dx
            if 0 <= nz < shape[0] and 0 <= ny < shape[1] and 0 <= nx < shape[2]:
                neighbour = index[nz, ny, nx]
                if neighbour >= 0 and node_label[neighbour] == 0:
                    node_label[neighbour] = node_label[node]
                    heapq.heappush(queue, (-values[neighbour], counter, neighbour))
                    counter += 1

    labels[coords[:, 0], coords[:, 1], coords[:, 2]] = node_label
    basins = [
        {"label": relabel[root], "peak_mm": float(peak[root]), "saddle_mm": saddle.get(root)}
        for root in unique_roots
    ]
    return labels, basins


def _min_peak_mm(spacing):
    """Smallest patch peak that can be a mouth. Rim distances are measured from
    the centres of the rim voxels, half a voxel outside the patch's true edge,
    so the half-width floor is raised by that much.
    """
    return MIN_OSTIUM_HALF_WIDTH_MM + 0.5 * float(np.mean(spacing))


def _split_patch(component, shell, spacing):
    """Label every contact-patch voxel with the origin it belongs to: one per
    spatially disjoint piece of the patch, and one per watershed basin within
    a piece.

    Returns (origin_of_patch_voxel, origins): an int array aligned with
    component["patch_flat"] (origins numbered from 1), and one dict per origin.
    """
    patch = component["patch_flat"]
    shape = component["shape"]
    transformed = patch_distance_transform(patch, shell, shape, spacing)
    if transformed is None:
        return np.ones(patch.size, dtype=np.int32), [
            {"piece": 1, "basin": 1, "peak_mm": None, "saddle_mm": None}
        ]

    low, closed, distance = transformed
    pieces, n_pieces = ndimage.label(closed, structure=_FULL)
    origin_crop = np.zeros(closed.shape, dtype=np.int32)
    origins = []
    for piece in range(1, n_pieces + 1):
        basin_labels, basins = watershed_basins(pieces == piece, distance, min_peak_mm=_min_peak_mm(spacing))
        for basin in basins:
            origins.append({
                "piece": piece,
                "basin": basin["label"],
                "peak_mm": basin["peak_mm"],
                "saddle_mm": basin["saddle_mm"],
            })
            origin_crop[basin_labels == basin["label"]] = len(origins)

    local = flat_to_zyx(patch, shape) - low
    return origin_crop[local[:, 0], local[:, 1], local[:, 2]], origins


def _absorb_narrow_pieces(origins, joins, min_peak_mm):
    """Fold contact spots too narrow to be a mouth into the vessel they belong to.

    A daughter that runs alongside the aorta (the SMA, typically) can graze the
    wall's partial-volume rim at several points past its real origin. Each
    graze is a separate, spatially disjoint piece of contact patch, a voxel or
    a few across, and each would otherwise claim part of the vessel as its own
    "origin" -- on subject015 one vessel came out as four. The watershed already
    refuses basins narrower than a mouth inside one piece; this applies the same
    floor across pieces. A narrow origin is merged into the origin whose
    territory it joins at the lowest geodesic distance, taking the lowest joins
    first; a narrow origin that joins nothing stays as it is.

    Returns {origin number: surviving origin number}.
    """
    parent = {number: number for number in range(1, len(origins) + 1)}
    peak = {number: (origin["peak_mm"] if origin["peak_mm"] is not None else np.inf)
            for number, origin in enumerate(origins, start=1)}

    def find(number):
        while parent[number] != number:
            number = parent[number]
        return number

    for (a, b), _join in sorted(joins.items(), key=lambda item: item[1]):
        root_a, root_b = find(a), find(b)
        if root_a == root_b:
            continue
        narrow_a, narrow_b = peak[root_a] < min_peak_mm, peak[root_b] < min_peak_mm
        if not (narrow_a or narrow_b):
            continue
        keep, fold = (root_a, root_b) if peak[root_a] >= peak[root_b] else (root_b, root_a)
        parent[fold] = keep
    return {number: find(number) for number in parent}


def _origin_of_exits(exits, patch_sorted, origin_of_patch):
    """Origin number for each exit voxel (0 if the exit is not in the patch)."""
    position = np.clip(np.searchsorted(patch_sorted, exits), 0, max(patch_sorted.size - 1, 0))
    found = patch_sorted.size > 0
    matches = (patch_sorted[position] == exits) if found else np.zeros(exits.size, dtype=bool)
    return np.where(matches, origin_of_patch[position] if found else 0, 0).astype(np.int32)


def vessel_join_distances(territory_labels, dist_crop):
    """Lowest geodesic distance at which each pair of territories touch.

    territory_labels: int crop (0 = none, 1..k); dist_crop: flood distance.
    Returns {(a, b): join_mm} for a < b, only for pairs that touch.
    """
    joins = {}
    shape = territory_labels.shape
    for offset in _neighbour_offsets():
        slices_a, slices_b = _pair_slices(offset, shape)
        a = territory_labels[slices_a]
        b = territory_labels[slices_b]
        touching = (a > 0) & (b > 0) & (a != b)
        if not touching.any():
            continue
        la, lb = a[touching], b[touching]
        height = np.maximum(dist_crop[slices_a][touching], dist_crop[slices_b][touching])
        low_label, high_label = np.minimum(la, lb), np.maximum(la, lb)
        for pair_a, pair_b, value in zip(low_label, high_label, height):
            key = (int(pair_a), int(pair_b))
            if value < joins.get(key, np.inf):
                joins[key] = float(value)
    return joins


def split_into_instances(candidates, detection, min_length_mm=MIN_ELIGIBLE_LENGTH_MM,
                         verbose=False, file=sys.stderr):
    """One instance per independent aortic origin.

    Each instance carries the same voxel/patch keys as a candidate component,
    restricted to its own territory, plus:
      component_label, piece, basin, split ("none", "disjoint_patches",
      "watershed" or both), patch_peak_mm, watershed_saddle_mm,
      absorbed_narrow_pieces, shares_vessel_with (other instance ids whose
      territory touches this one's), vessel_join_mm ({other id: mm}),
      is_independent_origin.

    Instances whose territory never reaches min_length_mm are dropped -- the
    eligibility rule, applied again after the split -- and counted.

    Returns (instances, summary).
    """
    reference_image = detection["reference_image"]
    spacing = reference_image.GetSpacing()
    shape = detection["shape"]
    dist_flat = detection["flood"]["dist"].ravel()
    exit_flat = detection["exit_voxels"].ravel()
    voxel_volume = voxel_volume_mm3(reference_image)
    voxel_area = voxel_volume ** (2.0 / 3.0)

    shell = detection["aortic_shell"]

    instances = []
    summary = {"components": len(candidates), "split_components": 0, "absorbed_narrow_pieces": 0,
               "instance_too_short": 0}

    for component in candidates:
        patch = component["patch_flat"]
        origin_of_patch, origins = _split_patch(component, shell, spacing)

        # territory: each voxel belongs to the origin its exit voxel belongs to
        voxel_origin = _origin_of_exits(exit_flat[component["voxels_flat"]], patch, origin_of_patch)
        core_origin = _origin_of_exits(exit_flat[component["core_flat"]], patch, origin_of_patch)

        zyx = flat_to_zyx(component["voxels_flat"], shape)
        low, high = _crop_bounds(zyx, shape, margin=1)
        territory_crop = np.zeros(tuple(high - low), dtype=np.int32)
        local = zyx - low
        dist_crop = detection["flood"]["dist"][low[0]:high[0], low[1]:high[1], low[2]:high[2]]

        joins = {}
        absorbed = {number: 0 for number in range(1, len(origins) + 1)}
        if len(origins) > 1:
            territory_crop[local[:, 0], local[:, 1], local[:, 2]] = voxel_origin
            survivor = _absorb_narrow_pieces(
                origins, vessel_join_distances(territory_crop, dist_crop), _min_peak_mm(spacing)
            )
            summary["absorbed_narrow_pieces"] += sum(1 for n, s in survivor.items() if n != s)
            for number, target in survivor.items():
                if number != target:
                    absorbed[target] += 1
            lookup = np.array([0] + [survivor[n] for n in range(1, len(origins) + 1)], dtype=np.int32)
            origin_of_patch = lookup[origin_of_patch]
            voxel_origin = lookup[voxel_origin]
            core_origin = lookup[core_origin]
            territory_crop[local[:, 0], local[:, 1], local[:, 2]] = voxel_origin
            joins = vessel_join_distances(territory_crop, dist_crop)

        surviving = sorted(set(origin_of_patch.tolist()))
        pieces_used = {origins[n - 1]["piece"] for n in surviving}
        if len(surviving) > 1:
            summary["split_components"] += 1

        local_ids = {}
        for number in surviving:
            origin = origins[number - 1]
            kinds = []
            if len(pieces_used) > 1:
                kinds.append("disjoint_patches")
            if sum(1 for n in surviving if origins[n - 1]["piece"] == origin["piece"]) > 1:
                kinds.append("watershed")
            origin["split"] = "+".join(kinds) if kinds else "none"

            in_origin = voxel_origin == number
            core = component["core_flat"][core_origin == number]
            if core.size == 0:
                summary["instance_too_short"] += 1
                continue
            core_dist = dist_flat[core]
            if float(core_dist.max()) < min_length_mm:
                summary["instance_too_short"] += 1
                continue

            own_patch = patch[origin_of_patch == number]
            patch_mm = flat_to_mm(own_patch, reference_image, shape)
            instance = {
                key: component[key]
                for key in ("shape", "neck_mm", "vesselness", "touches_cap", "cap_face_fraction",
                            "radius_estimate_mm", "direction_estimate")
                if key in component
            }
            instance.update({
                "instance_id": len(instances),
                "component_label": component["label"],
                "core_flat": core,
                "core_dist": core_dist,
                "voxels_flat": component["voxels_flat"][in_origin],
                "voxels_dist": component["voxels_dist"][in_origin],
                "patch_flat": own_patch,
                "n_patch_voxels": int(own_patch.size),
                "patch_mm": patch_mm,
                "patch_centroid_mm": patch_mm.mean(axis=0),
                "patch_area_mm2": float(own_patch.size) * voxel_area,
                "max_distance_mm": float(core_dist.max()),
                "volume_mm3": float(np.count_nonzero(in_origin)) * voxel_volume,
                "piece": origin["piece"],
                "basin": origin["basin"],
                "split": origin["split"],
                "patch_peak_mm": origin["peak_mm"],
                "watershed_saddle_mm": origin["saddle_mm"],
                "absorbed_narrow_pieces": absorbed[number],
                "shares_vessel_with": [],
                "vessel_join_mm": {},
                "is_independent_origin": True,
            })
            local_ids[number] = instance["instance_id"]
            instances.append(instance)

        for (a, b), join_mm in joins.items():
            if a in local_ids and b in local_ids:
                first, second = instances[local_ids[a]], instances[local_ids[b]]
                first["shares_vessel_with"].append(second["instance_id"])
                second["shares_vessel_with"].append(first["instance_id"])
                first["vessel_join_mm"][second["instance_id"]] = join_mm
                second["vessel_join_mm"][first["instance_id"]] = join_mm

    summary["instances"] = len(instances)
    if verbose:
        log_instances(instances, summary, file=file)
    return instances, summary


def log_instances(instances, summary, file=sys.stderr):
    print(
        f"instances={summary['instances']} from {summary['components']} candidates "
        f"(split_components={summary['split_components']} "
        f"absorbed_narrow_pieces={summary['absorbed_narrow_pieces']} "
        f"instance_too_short={summary['instance_too_short']})",
        file=file,
    )
