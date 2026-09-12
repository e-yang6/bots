"""Trace a candidate branch outward from the aorta wall, and derive the
reported geometry (ostium, seed, direction, radius) from that trace.

Traced length is kept continuous here -- the 5mm eligibility rule is a
scoring/filtering decision for a later stage, not something to bake into
the tracer, which would throw away the evidence needed to make that call.
"""

import numpy as np

from src.candidates import perpendicular_basis as _perpendicular_basis

DEFAULT_MAX_LENGTH_MM = 10.0
SEED_ARC_LENGTH_MM = 5.0


def _unit(vector):
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector


def _cone_directions(direction, max_turn_deg, n_rings=2, n_per_ring=8):
    """Candidate step directions inside a cone around `direction`."""
    direction = _unit(np.asarray(direction, dtype=np.float64))
    u, v = _perpendicular_basis(direction[None, :])
    u, v = u[0], v[0]

    directions = [direction]
    for ring in range(1, n_rings + 1):
        tilt = np.radians(max_turn_deg) * ring / n_rings
        for angle in np.linspace(0.0, 2.0 * np.pi, n_per_ring, endpoint=False):
            offset = np.tan(tilt) * (np.cos(angle) * u + np.sin(angle) * v)
            directions.append(_unit(direction + offset))
    return np.array(directions)


def cross_section(sampler, point_mm, direction, lumen_threshold, extent_mm=6.0, step_mm=0.4,
                  exclude_aorta=True, neighbourhood_mm=4.0):
    """Sample the lumen cross-section in the plane perpendicular to `direction`.

    Returns the area of the vessel component containing the centre, how many
    other comparable components share the plane (bimodality, i.e. a split),
    the intensity-weighted centroid of the central component, and its
    second-moment ellipse semi-axes.

    exclude_aorta drops voxels inside the aorta mask before labelling. A
    daughter's lumen is continuous with the aortic lumen, so a plane taken
    anywhere near the origin otherwise merges the two into one component and
    reports the aorta's cross-section (~4-5mm equivalent radius) as the
    daughter's.
    """
    from scipy import ndimage

    direction = _unit(np.asarray(direction, dtype=np.float64))
    u, v = _perpendicular_basis(direction[None, :])
    u, v = u[0], v[0]

    axis = np.arange(-extent_mm, extent_mm + 1e-9, step_mm)
    grid_u, grid_v = np.meshgrid(axis, axis, indexing="ij")
    offsets = grid_u[..., None] * u + grid_v[..., None] * v
    plane_points = point_mm + offsets.reshape(-1, 3)

    lumen = sampler.sample("lumen", plane_points, cval=0.0).reshape(grid_u.shape)
    binary = lumen >= lumen_threshold
    if exclude_aorta:
        in_aorta = sampler.sample("mask", plane_points, cval=0.0).reshape(grid_u.shape) > 0.5
        binary = binary & ~in_aorta

    labels, n_labels = ndimage.label(binary)
    centre = (labels.shape[0] // 2, labels.shape[1] // 2)
    centre_label = labels[centre]

    if centre_label == 0:
        # centre fell outside the lumen; fall back to the component whose
        # voxels come closest to the centre
        if n_labels == 0:
            return {
                "area_mm2": 0.0, "n_components": 0, "centroid_mm": point_mm.copy(),
                "semi_axes_mm": (0.0, 0.0), "equivalent_radius_mm": 0.0, "valid": False,
            }
        filled = np.argwhere(labels > 0)
        offsets = filled - np.array(centre)
        nearest = filled[np.argmin(np.einsum("ij,ij->i", offsets, offsets))]
        centre_label = int(labels[nearest[0], nearest[1]])

    component = labels == centre_label
    pixel_area = step_mm * step_mm
    area_mm2 = float(component.sum()) * pixel_area

    # Count only components that are both comparable in size to the central
    # one AND close to it. Excluding the aorta leaves unrelated lumen
    # elsewhere in the plane (the far aortic wall, the IVC, a neighbouring
    # branch), which must not be mistaken for this vessel splitting.
    comparable = 0
    for label_id in range(1, n_labels + 1):
        member = labels == label_id
        member_area = float(member.sum()) * pixel_area
        if member_area < 0.3 * area_mm2:
            continue
        offsets_to_centre = np.argwhere(member) - np.array(centre)
        nearest_mm = float(np.sqrt(np.min(np.einsum("ij,ij->i", offsets_to_centre, offsets_to_centre)))) * step_mm
        if nearest_mm <= neighbourhood_mm:
            comparable += 1

    weights = np.clip(lumen, 0.0, None) * component
    total_weight = weights.sum()
    if total_weight > 0:
        centroid_u = float((weights * grid_u).sum() / total_weight)
        centroid_v = float((weights * grid_v).sum() / total_weight)
    else:
        centroid_u = centroid_v = 0.0
    centroid_mm = point_mm + centroid_u * u + centroid_v * v

    coords_u = grid_u[component] - centroid_u
    coords_v = grid_v[component] - centroid_v
    if coords_u.size >= 3:
        covariance = np.cov(np.vstack([coords_u, coords_v]))
        eigenvalues = np.clip(np.linalg.eigvalsh(covariance), 0.0, None)
        semi_axes = tuple(float(2.0 * np.sqrt(value)) for value in np.sort(eigenvalues)[::-1])
    else:
        semi_axes = (0.0, 0.0)
    equivalent_radius = float(np.sqrt(max(semi_axes[0] * semi_axes[1], 0.0)))

    return {
        "area_mm2": area_mm2,
        "n_components": comparable,
        "centroid_mm": centroid_mm,
        "semi_axes_mm": semi_axes,
        "equivalent_radius_mm": equivalent_radius,
        "valid": True,
    }


def _radius_by_ray_casting(sampler, point_mm, direction, lumen_threshold, max_radius_mm=6.0, step_mm=0.25, n_rays=12):
    """Local lumen radius: cast rays perpendicular to `direction` and take the
    median distance at which intensity drops out of the lumen.
    """
    direction = _unit(np.asarray(direction, dtype=np.float64))
    u, v = _perpendicular_basis(direction[None, :])
    u, v = u[0], v[0]

    distances = np.arange(step_mm, max_radius_mm + 1e-9, step_mm)
    hits = []
    for angle in np.linspace(0.0, 2.0 * np.pi, n_rays, endpoint=False):
        ray = np.cos(angle) * u + np.sin(angle) * v
        ray_points = point_mm + np.outer(distances, ray)
        samples = sampler.sample("lumen", ray_points, cval=0.0)
        # a ray that runs into the aorta has left this vessel, same as one
        # that runs out of lumen -- otherwise rays fired near the ostium
        # traverse the whole aortic lumen and report its radius
        in_aorta = sampler.sample("mask", ray_points, cval=0.0) > 0.5
        outside = np.where((samples < lumen_threshold) | in_aorta)[0]
        hits.append(distances[outside[0]] if outside.size else max_radius_mm)
    return float(np.median(hits))


def trace_branch(
    sampler,
    start_point_mm,
    start_direction,
    lumen_threshold,
    max_length_mm=DEFAULT_MAX_LENGTH_MM,
    step_mm=0.5,
    max_turn_deg=35.0,
    min_lumen=None,
    min_vesselness=0.05,
    radius_jump_tolerance=0.6,
    wall_grace_mm=1.5,
    patience=2,
):
    """Cone-search outward from a candidate's contact patch.

    At each step the next direction is chosen from a cone around the current
    one, scored by lumen intensity and vesselness and penalised for turning
    sharply, drifting backward relative to the original outward direction, or
    jumping implausibly in radius. Stops on running out of evidence, on
    re-entering the aorta, on leaving the volume, or at max_length_mm.

    Two allowances keep real branches from being abandoned on the doorstep.
    Within wall_grace_mm the evidence test is suspended: crossing the aortic
    wall genuinely dips dark from partial volume, and without this almost
    every trace died after one or two steps. Beyond that, `patience`
    consecutive weak steps are tolerated before giving up, since single dark
    voxels from noise are common inside small vessels.
    """
    if min_lumen is None:
        min_lumen = 0.75 * lumen_threshold

    start_direction = _unit(np.asarray(start_direction, dtype=np.float64))
    point = np.asarray(start_point_mm, dtype=np.float64).copy()
    direction = start_direction.copy()

    points = [point.copy()]
    directions = [direction.copy()]
    radii = [_radius_by_ray_casting(sampler, point, direction, lumen_threshold)]
    lumen_values = [float(sampler.sample("lumen", point[None, :], cval=0.0)[0])]
    vesselness_values = [float(sampler.sample("vesselness", point[None, :], cval=0.0)[0])]
    areas = [cross_section(sampler, point, direction, lumen_threshold)["area_mm2"]]
    arc_lengths = [0.0]
    n_components = [1]

    status = "max_length"
    left_aorta = False
    weak_steps = 0

    while arc_lengths[-1] < max_length_mm:
        options = _cone_directions(direction, max_turn_deg)
        forward = options @ start_direction > 0.0  # no doubling back on the origin
        options = options[forward] if forward.any() else options[:1]

        next_points = point + step_mm * options
        lumen_next = sampler.sample("lumen", next_points, cval=0.0)
        vesselness_next = sampler.sample("vesselness", next_points, cval=0.0)
        turn_cos = np.clip(options @ direction, -1.0, 1.0)

        scores = (
            1.0 * np.clip(lumen_next, 0.0, 1.5)
            + 0.8 * np.clip(vesselness_next, 0.0, 1.5)
            - 1.0 * (1.0 - turn_cos)
        )

        best = int(np.argmax(scores))
        candidate_point = next_points[best]
        candidate_direction = _unit(options[best])

        if not sampler.in_bounds(candidate_point[None, :])[0]:
            status = "out_of_bounds"
            break

        inside_aorta = sampler.sample("mask", candidate_point[None, :], cval=0.0)[0] > 0.5
        if not inside_aorta:
            left_aorta = True
        elif left_aorta:
            status = "reentered_aorta"
            break

        weak = lumen_next[best] < min_lumen and vesselness_next[best] < min_vesselness
        if weak and arc_lengths[-1] >= wall_grace_mm:
            weak_steps += 1
            if weak_steps > patience:
                status = "low_evidence"
                break
        elif not weak:
            weak_steps = 0

        candidate_radius = _radius_by_ray_casting(
            sampler, candidate_point, candidate_direction, lumen_threshold
        )
        # Radius is meaningless while still crossing the wall -- rays fired
        # from a point on the aorta surface immediately run into the aorta and
        # measure ~0 -- so the jump test only applies past the grace distance,
        # and its denominator is floored to keep a near-zero previous radius
        # from making every subsequent step look like a huge jump.
        previous_radius = radii[-1]
        if arc_lengths[-1] >= wall_grace_mm:
            relative_jump = abs(candidate_radius - previous_radius) / max(previous_radius, 0.5)
            if relative_jump > radius_jump_tolerance and candidate_radius > previous_radius:
                status = "radius_jump"
                break

        section = cross_section(sampler, candidate_point, candidate_direction, lumen_threshold)

        point = candidate_point
        direction = candidate_direction
        points.append(point.copy())
        directions.append(direction.copy())
        radii.append(candidate_radius)
        lumen_values.append(float(lumen_next[best]))
        vesselness_values.append(float(vesselness_next[best]))
        areas.append(section["area_mm2"])
        n_components.append(section["n_components"])
        arc_lengths.append(arc_lengths[-1] + step_mm)

    return {
        "points_mm": np.array(points),
        "directions": np.array(directions),
        "radii_mm": np.array(radii),
        "lumen": np.array(lumen_values),
        "vesselness": np.array(vesselness_values),
        "areas_mm2": np.array(areas),
        "n_components": np.array(n_components),
        "arc_lengths_mm": np.array(arc_lengths),
        "traced_length_mm": float(arc_lengths[-1]),
        "status": status,
        "start_direction": start_direction,
    }


def detect_bifurcation(trace, area_jump_ratio=1.8, min_arc_mm=3.0, min_area_mm2=0.5,
                       persistence=2):
    """Find where the growth front splits, so a common trunk counts once.

    Two signals on the cross-sectional area of the front: a sudden jump
    relative to the running median (the two daughters still merged into one
    fat section), or the front resolving into two comparable components
    (already separated). The trace is truncated there -- a celiac trunk then
    contributes one origin, not two.

    Both signals must hold for `persistence` consecutive steps, and only
    past min_arc_mm. A real split stays split, whereas a single frame of
    two components is usually the near-origin cross-section fragmenting
    against the aortic wall.
    """
    areas = trace["areas_mm2"]
    components = trace["n_components"]
    arcs = trace["arc_lengths_mm"]

    split_run = 0
    jump_run = 0
    for index in range(len(areas)):
        if arcs[index] < min_arc_mm:
            continue

        previous = areas[max(0, index - 5):index]
        running_median = float(np.median(previous)) if previous.size else 0.0

        split_run = split_run + 1 if components[index] >= 2 else 0
        if split_run >= persistence:
            start = index - persistence + 1
            return {"bifurcation_index": start, "reason": "bimodal_front",
                    "arc_length_mm": float(arcs[start])}

        is_jump = running_median >= min_area_mm2 and areas[index] > area_jump_ratio * running_median
        jump_run = jump_run + 1 if is_jump else 0
        if jump_run >= persistence:
            start = index - persistence + 1
            return {"bifurcation_index": start, "reason": "area_jump",
                    "arc_length_mm": float(arcs[start])}

    return {"bifurcation_index": None, "reason": None, "arc_length_mm": None}


def truncate_trace(trace, index):
    """Cut a trace at `index` (inclusive), keeping all per-step arrays aligned."""
    if index is None or index >= len(trace["arc_lengths_mm"]):
        return trace

    truncated = dict(trace)
    for key in ("points_mm", "directions", "radii_mm", "lumen", "vesselness",
                "areas_mm2", "n_components", "arc_lengths_mm"):
        truncated[key] = trace[key][: index + 1]
    truncated["traced_length_mm"] = float(truncated["arc_lengths_mm"][-1])
    truncated["status"] = "bifurcation"
    return truncated


def estimate_ostium_candidates(sampler, candidate, trace, lumen_threshold, aorta_surface_tree,
                               aorta_surface_points_mm, search_mm=4.0, step_mm=0.25):
    """Three independent estimates of the ostium, plus their disagreement.

    Large disagreement between them is itself a false-positive signal: for a
    real branch leaving a wall they should land within a millimetre or two of
    each other, whereas for noise they scatter.
    """
    patch_centroid = np.asarray(candidate["ostium_patch_centroid_mm"], dtype=np.float64)
    outward = _unit(np.asarray(candidate["outward_direction"], dtype=np.float64))

    # 2. the traced path walked back onto the aorta surface
    if trace["points_mm"].shape[0] >= 2:
        back_direction = _unit(trace["points_mm"][0] - trace["points_mm"][min(4, len(trace["points_mm"]) - 1)])
    else:
        back_direction = -outward
    walk = trace["points_mm"][0] + np.outer(np.arange(0.0, search_mm + 1e-9, step_mm), back_direction)
    inside = sampler.sample("mask", walk, cval=0.0) > 0.5
    entry = walk[np.argmax(inside)] if inside.any() else trace["points_mm"][0]
    _distance, nearest = aorta_surface_tree.query(entry[None, :], k=1)
    traced_projection = aorta_surface_points_mm[int(nearest[0])]

    # 3. the lumen opening: strongest intensity gradient along the outward ray
    ray_offsets = np.arange(-search_mm, search_mm + 1e-9, step_mm)
    ray = patch_centroid + np.outer(ray_offsets, outward)
    lumen_along_ray = sampler.sample("lumen", ray, cval=0.0)
    gradient = np.abs(np.gradient(lumen_along_ray, step_mm))
    max_gradient_point = ray[int(np.argmax(gradient))]

    estimates = {
        "patch_centroid_mm": patch_centroid,
        "traced_projection_mm": traced_projection,
        "max_gradient_mm": max_gradient_point,
    }
    keys = list(estimates)
    pairwise = {}
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            distance = float(np.linalg.norm(estimates[keys[i]] - estimates[keys[j]]))
            pairwise[f"{keys[i]}__{keys[j]}"] = distance

    values = np.array(list(pairwise.values()))
    return {
        "estimates": estimates,
        "pairwise_distances_mm": pairwise,
        "max_disagreement_mm": float(values.max()) if values.size else 0.0,
        "mean_disagreement_mm": float(values.mean()) if values.size else 0.0,
        "consensus_mm": np.mean(np.array(list(estimates.values())), axis=0),
    }


def _interpolate_along_trace(trace, target_arc_mm):
    arc = trace["arc_lengths_mm"]
    points = trace["points_mm"]
    if arc[-1] <= 0:
        return points[0].copy(), trace["start_direction"].copy(), False

    reached = target_arc_mm <= arc[-1]
    target = min(target_arc_mm, arc[-1])
    index = int(np.searchsorted(arc, target))
    index = min(max(index, 1), len(arc) - 1)

    span = arc[index] - arc[index - 1]
    weight = 0.0 if span <= 0 else (target - arc[index - 1]) / span
    point = points[index - 1] + weight * (points[index] - points[index - 1])
    direction = trace["directions"][index]
    return point, direction, reached


def extract_seed_direction_radius(sampler, trace, ostium_mm, lumen_threshold,
                                  seed_arc_mm=SEED_ARC_LENGTH_MM, fit_length_mm=5.0,
                                  radius_disagreement_threshold=0.4):
    """Seed point, outward direction and radius for one traced branch.

    Seed sits at seed_arc_mm along the trace, re-centred to the
    intensity-weighted centroid of the cross-section normal to the local
    direction. Direction comes from a PCA line fit over the first
    fit_length_mm (robust to the tracer's step-to-step wobble). Radius is
    read from the signed distance transform at the re-centred seed and
    cross-checked against the cross-section's equivalent-area ellipse.
    """
    seed_point, local_direction, reached_full_length = _interpolate_along_trace(trace, seed_arc_mm)

    section = cross_section(sampler, seed_point, local_direction, lumen_threshold)
    recentred_seed = section["centroid_mm"] if section["valid"] else seed_point

    within_fit = trace["arc_lengths_mm"] <= fit_length_mm
    fit_points = trace["points_mm"][within_fit]
    if fit_points.shape[0] >= 3:
        centred = fit_points - fit_points.mean(axis=0)
        _u, _s, vh = np.linalg.svd(centred, full_matrices=False)
        direction = _unit(vh[0])
    else:
        direction = _unit(trace["start_direction"])

    outward_reference = recentred_seed - np.asarray(ostium_mm, dtype=np.float64)
    if np.dot(direction, outward_reference) < 0:
        direction = -direction

    radius_from_distance = float(sampler.sample("vessel_dt", recentred_seed[None, :], cval=0.0)[0])
    radius_from_distance = max(radius_from_distance, 0.0)
    radius_from_ellipse = float(section["equivalent_radius_mm"]) if section["valid"] else 0.0

    largest = max(radius_from_distance, radius_from_ellipse)
    disagreement = abs(radius_from_distance - radius_from_ellipse) / largest if largest > 0 else 0.0

    return {
        "seed_mm": recentred_seed,
        "seed_before_recentring_mm": seed_point,
        "reached_seed_arc_length": bool(reached_full_length),
        "direction_xyz": direction,
        "radius_mm": radius_from_distance if radius_from_distance > 0 else radius_from_ellipse,
        "radius_from_distance_transform_mm": radius_from_distance,
        "radius_from_ellipse_mm": radius_from_ellipse,
        "radius_disagreement": float(disagreement),
        "radius_disagreement_flag": bool(disagreement > radius_disagreement_threshold),
        "cross_section_area_mm2": float(section["area_mm2"]),
        "cross_section_semi_axes_mm": section["semi_axes_mm"],
    }
