"""Interactive manual labelling of ground-truth branch ostia on a real case.

This is a human-in-the-loop annotation tool, deliberately kept separate from
the detection pipeline in src/. The organizers ship no reference labels, so
the only way to score src/evaluate.py is for a person to look at a case and
mark, by eye, where each daughter vessel leaves the aorta. This script writes
exactly the schema.py prediction format, so its output drops straight into
`python -m src.evaluate --references <this file>` with no reformatting.

Usage:
    python scripts/label_case.py --image  "TORALIS CHALLENGE/subject001/orig1.nii" \
                                 --aorta-mask "TORALIS CHALLENGE/subject001/mask1.nii" \
                                 --output ground_truth.json

Layout, left to right: the 3D aorta surface (drag to rotate, scroll to zoom);
two thick-slab maximum-intensity projections; and the axial CT for whatever
point you last clicked -- a full slice on top, a zoomed crop underneath.

Work from the middle outwards. A single axial slice shows each vessel as an
isolated circle, so spotting a branch in one means scrubbing and remembering
whether a bright blob ever fuses into the aorta. A MIP collapses a whole slab
by taking the brightest voxel along each ray, so the same vessels appear as
continuous branching lines and the coeliac / SMA / renal pattern is visible
at a glance. Click the branch you want on a MIP, then refine and confirm the
exact position on the axial panels, which are still the arbiter.

The two projections are complementary, which is why both are shown: the
coronal one collapses front-to-back and so shows the laterally-directed
renal arteries, while the sagittal one collapses left-to-right and shows the
anteriorly-directed coeliac trunk and SMA.


What a real branch looks like on this kind of scan
--------------------------------------------------
The aorta mask is aorta-only: it is a smooth tube that does NOT include the
branches, so the bare mask surface shows you almost nothing about where
branches are. What gives them away is the CT intensity just outside the
mask, which is why the surface here is coloured by it (see `probe_surface`).

  - A real branch is a BRIGHT, TUBE-LIKE structure in direct continuity with
    the aorta's bright contrast-filled lumen -- same rough HU as the lumen
    itself, clearly brighter than the surrounding darker tissue and fat.
  - Anatomically, expect them along the FRONT and SIDES of the upper
    abdominal aorta: the coeliac trunk and the superior mesenteric artery
    leave anteriorly, the renal arteries leave laterally a little below them.
  - On the 3D view they show up as hot (yellow/white) patches. Treat those as
    a place to look, not as an answer -- confirm every one on the axial
    panels before accepting it. At a true ostium the slice shows the aorta
    with a vessel visibly leaving its wall, and the marker should sit on that
    vessel's mouth.

THE INFERIOR VENA CAVA IS THE MAIN TRAP. It is a large, contrast-filled vein
that runs directly alongside the aorta for the whole scan, so it scores just
as "vessel-like" as any artery and lights up much of the 3D view. It is not a
daughter. The cheapest way to tell them apart is to scrub slices with [ and ]
at the point you are considering:

  - the IVC is a big round tube that persists, essentially unchanged, over a
    hundred-odd slices and never actually joins the aortic lumen;
  - a real daughter appears over roughly ten slices, is narrower, and is
    continuous with the lumen at its ostium before heading away from it.

Anything bright that runs alongside the aorta but never touches the lumen is
vein or bowel, not a daughter. If the bright thing vanishes one or two slices
up or down, it was noise.


Controls
--------
    click (MIP)  place the pending ostium on the branch under the cursor;
                 the in-plane position is exact, the depth is inferred from
                 the most lumen-like voxel along that ray
    click (3D)   place / move the pending ostium at the frontmost surface
                 point under the cursor
    click (axial or zoom panel)
                 nudge the pending ostium within that slice -- use this, the
                 slice is where a branch is actually visible
    [ / ]        step the displayed slice down / up one
    m            show / hide the MIP panels
    - / =        thin / thicken the MIP slab by 5 mm
    y            confirm the pending point as a branch
    n / escape   discard the pending point
    d            after confirming: the next 3D click sets the direction
    k            skip the direction pick (falls back to the outward normal)
    r            edit the radius (type digits, Enter to accept)
    u            undo the last confirmed branch
    s            save to --output
    q            save and quit
"""

import argparse
import os
import sys

import matplotlib
import numpy as np
import SimpleITK as sitk

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# NB: scripts/visualize_candidates.py is deliberately NOT imported here even
# though this tool follows its rendering conventions (origin="lower" axial
# slices, cyan mask contour, hollow red ostium marker). That module forces
# matplotlib's non-interactive "Agg" backend at import time, which would make
# this window unclickable. The drawing conventions are mirrored instead.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401,E402  (registers "3d")

import matplotlib.pyplot as plt  # noqa: E402

from schema import (  # noqa: E402
    case_id_from_path,
    make_daughter,
    make_prediction,
    read_prediction,
    write_prediction,
)
from src.candidates import build_evidence  # noqa: E402
from src.geometry import compute_surface_normals  # noqa: E402
from src.intensity import lumen_stats  # noqa: E402
from src.io_utils import load_case  # noqa: E402

# How far outside the mask surface to look for a vessel when shading the 3D
# view. The lower bound is the load-bearing one: the aorta's own partial-
# volume rim ramps from lumen to soft tissue over the first ~2mm and reads as
# near-perfect lumen the whole way, so probing from 1mm lights up about 80%
# of the surface and hides the branches entirely. From 2.5mm out that rim is
# cleared and only ~7% of the surface scores as vessel-like. The upper bound
# keeps the probe from reaching unrelated bowel and vein.
PROBE_DISTANCES_MM = (2.5, 3.0, 3.5, 4.0, 5.0, 6.0)

# Screen radius, in pixels, within which a 3D click is considered to be "on"
# a surface point.
PICK_RADIUS_PX = 18.0

# Depth window, in mm, used to separate the wall facing the camera from the
# one behind it when resolving a click. Comfortably under an aortic diameter,
# so the far wall is always excluded, but wide enough to keep the whole of
# the near wall's curvature inside the pick disc.
NEAR_WALL_DEPTH_MM = 8.0

DEFAULT_RADIUS_MM = 3.0
SEED_OFFSET_MM = 5.0
ZOOM_HALF_WIDTH_MM = 25.0

# Thick-slab MIP settings. 35mm is about the depth that holds the whole
# coeliac/SMA trunk or a renal artery in one projection without also folding
# in unrelated bowel and vein.
DEFAULT_SLAB_MM = 35.0
SLAB_STEP_MM = 5.0
SLAB_LIMITS_MM = (5.0, 120.0)

# Lateral half-width of the MIP panels around the aorta. The full volume is
# ~400mm wide but only ~140mm tall, which renders as an unreadable letterbox;
# cropping to the aorta's neighbourhood gives the branches usable pixels.
MIP_HALF_WIDTH_MM = 80.0

# How far above the lumen HU to clip before projecting. A MIP takes the
# brightest voxel along each ray, and vertebral bone is several times
# brighter than contrast, so an unclipped projection is a picture of the
# spine with the vessels lost inside it. Clipping flattens bone to the same
# ceiling as the lumen, which leaves the vessel tree legible against it.
MIP_CLIP_ABOVE_LUMEN_HU = 150.0

# Recovering depth under a MIP click means finding the vessel along that ray,
# and the trap is bone. A vertebra ramps from soft tissue to ~1500 HU, so its
# partial-volume rim necessarily passes through the lumen HU on the way and
# ties with a real vessel. Anything this far above the lumen is treated as
# bone or calcium, and the rim either side of it is excluded too. This is the
# same guard src.candidates applies to its lumen field, for the same reason.
MIP_BONE_MARGIN_HU = 250.0
MIP_BONE_RIM_VOXELS = 2

# Depths whose HU is within this much of the best match along the ray count
# as tied, and the tie is broken towards the centre of the slab -- which is
# where the aorta is, and so where a daughter's ostium must also be.
MIP_DEPTH_TIE_HU = 25.0


def physical_to_continuous_index(points_mm, image):
    """Vectorized image.TransformPhysicalPointToContinuousIndex for (N, 3) mm.

    The inverse of src.geometry._indices_to_physical, and it stays in this
    project's SimpleITK/LPS frame -- never a nibabel RAS+ affine.
    """
    points_mm = np.atleast_2d(np.asarray(points_mm, dtype=float))
    spacing = np.array(image.GetSpacing())
    origin = np.array(image.GetOrigin())
    direction = np.array(image.GetDirection()).reshape(3, 3)
    return ((points_mm - origin) @ direction) / spacing


def index_to_physical(index_xyz, image):
    """Vectorized image.TransformContinuousIndexToPhysicalPoint for (N, 3)
    (x, y, z) index coordinates. The inverse of physical_to_continuous_index.
    """
    index_xyz = np.atleast_2d(np.asarray(index_xyz, dtype=float))
    spacing = np.array(image.GetSpacing())
    origin = np.array(image.GetOrigin())
    direction = np.array(image.GetDirection()).reshape(3, 3)
    return (index_xyz * spacing) @ direction.T + origin


def sample_hu(points_mm, image, image_arr):
    """Nearest-neighbour HU at (N, 3) physical points; out-of-bounds -> -1000."""
    index = physical_to_continuous_index(points_mm, image)
    ijk = np.rint(index).astype(int)
    size = np.array(image.GetSize())
    inside = np.all((ijk >= 0) & (ijk < size), axis=1)

    values = np.full(len(ijk), -1000.0)
    valid = ijk[inside]
    values[inside] = image_arr[valid[:, 2], valid[:, 1], valid[:, 0]]
    return values


def probe_surface(surface_mm, normals, sampler):
    """Strongest lumen-likeness found just outside the mask along each normal.

    This is what turns the 3D view from a featureless grey sausage into
    something worth clicking on: branches are excluded from the aorta-only
    mask, so their only trace on the surface is the contrast-filled lumen
    sitting immediately outside it.

    It samples src.candidates' own "lumen" evidence field rather than raw HU,
    for the reason documented there: raw HU cannot tell a contrast-filled
    vessel from bone or calcification, and ribs and vertebrae run close
    enough to the aorta that a raw-HU probe lights up most of the surface.
    The lumen field is already normalized per case and flattened above the
    lumen band, so only genuinely vessel-like tissue scores near 1.
    """
    best = np.zeros(len(surface_mm))
    for distance in PROBE_DISTANCES_MM:
        best = np.maximum(best, sampler.sample("lumen", surface_mm + normals * distance))
    return best


def probe_surface_raw_hu(surface_mm, normals, image, image_arr, lumen_median):
    """Cheap stand-in for probe_surface that skips building the evidence
    volume. Scores each surface point by how close the brightest nearby voxel
    is to this case's lumen HU, so bone (far above the lumen) is penalised
    rather than rewarded -- but it has no vesselness term and no rim
    handling, so it is noticeably noisier. Used only for --raw-shading.
    """
    best = np.full(len(surface_mm), -1000.0)
    for distance in PROBE_DISTANCES_MM:
        best = np.maximum(best, sample_hu(surface_mm + normals * distance, image, image_arr))
    return np.clip(1.0 - np.abs(best - lumen_median) / max(lumen_median, 1.0), 0.0, 1.0)


class Labeller:
    """Click-to-confirm labelling session over one case."""

    def __init__(self, image, mask, case_id, output_path, surface_stride=1,
                 raw_shading=False):
        self.image = image
        self.mask = mask
        self.case_id = case_id
        self.output_path = output_path

        self.image_arr = sitk.GetArrayFromImage(image)
        self.mask_arr = sitk.GetArrayFromImage(mask).astype(bool)

        # Display window centred on this case's own lumen, never hardcoded:
        # one case in this dataset bottoms out near -9000 HU, so a fixed
        # window is not safe (see src/intensity's module docstring).
        stats = lumen_stats(image, mask)
        self.lumen_median = stats["median"]

        surface_mm, normals = compute_surface_normals(mask)
        self.surface_mm = surface_mm[::surface_stride]
        normals = normals[::surface_stride]
        # compute_surface_normals returns raw image-gradient normals, which
        # are only approximately unit length; the probe steps below are in
        # millimetres, so they have to be normalized first.
        lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        self.normals = np.divide(normals, lengths, out=np.zeros_like(normals),
                                 where=lengths > 1e-6)

        if raw_shading:
            self.brightness = probe_surface_raw_hu(
                self.surface_mm, self.normals, image, self.image_arr, self.lumen_median)
        else:
            evidence = build_evidence(image, mask, stats)
            self.brightness = probe_surface(self.surface_mm, self.normals, evidence["sampler"])
        self.slice_vmin = self.lumen_median - 400.0
        self.slice_vmax = self.lumen_median + 200.0

        self.daughters = []           # confirmed branches, schema dicts
        self.state = "IDLE"           # IDLE | PENDING | DIRECTION
        self.pending_mm = None        # ostium being considered
        self.pending_normal = None
        self.radius_mm = DEFAULT_RADIUS_MM
        self.radius_buffer = None     # non-None while typing a radius
        self.slice_z = None
        self.message = "Find a branch on the MIP panels, then click to place a point."

        # Thick-slab MIP state. The projection is built from HU clipped just
        # above the lumen (see MIP_CLIP_ABOVE_LUMEN_HU) rather than from the
        # lumen evidence field: that field is a near-binary band-pass built
        # for thresholding, and projecting it saturates kidney, marrow and
        # vessel alike into one white blob. Clipped HU keeps the tonal
        # gradation that makes a vessel tree readable.
        self.mip_volume = np.clip(
            self.image_arr, self.slice_vmin, self.lumen_median + MIP_CLIP_ABOVE_LUMEN_HU)
        self.mip_vmax = self.lumen_median + MIP_CLIP_ABOVE_LUMEN_HU
        self.show_mip = True
        self.slab_mm = DEFAULT_SLAB_MM
        # Centre the slabs on the aorta until the user places a point, so the
        # MIP is useful immediately -- finding a branch is what it is for.
        centre_zyx = np.argwhere(self.mask_arr).mean(axis=0)
        self.aorta_centre_mm = index_to_physical(centre_zyx[::-1], image)[0]

        self._press_xy = None
        self._build_figure()

    @property
    def mip_centre_mm(self):
        """What the MIP slabs are centred on: the pending point if there is
        one, otherwise the middle of the aorta."""
        return self.aorta_centre_mm if self.pending_mm is None else self.pending_mm

    # ---------------------------------------------------------------- figure

    def _build_figure(self):
        self.figure = plt.figure(figsize=(15, 8))
        if self.figure.canvas.manager is not None:
            self.figure.canvas.manager.set_window_title(f"label_case: {self.case_id}")

        # Two layouts, switched by "m". Rather than rebuilding the figure on
        # every toggle -- which would re-scatter 10k points and throw away the
        # camera angle -- both gridspecs are kept and the axes are just moved.
        self._grid_mip_off = self.figure.add_gridspec(
            2, 3, width_ratios=[1.30, 1.30, 1.05], left=0.03, right=0.99,
            top=0.93, bottom=0.11, wspace=0.12, hspace=0.18)
        self._grid_mip_on = self.figure.add_gridspec(
            2, 3, width_ratios=[0.95, 1.35, 1.05], left=0.03, right=0.99,
            top=0.93, bottom=0.11, wspace=0.12, hspace=0.18)

        self.ax3d = self.figure.add_subplot(self._grid_mip_on[:, 0], projection="3d")
        self.ax_cor = self.figure.add_subplot(self._grid_mip_on[0, 1])
        self.ax_sag = self.figure.add_subplot(self._grid_mip_on[1, 1])
        self.ax_slice = self.figure.add_subplot(self._grid_mip_on[0, 2])
        self.ax_zoom = self.figure.add_subplot(self._grid_mip_on[1, 2])

        self.ax3d.scatter(
            self.surface_mm[:, 0], self.surface_mm[:, 1], self.surface_mm[:, 2],
            c=self.brightness, cmap="inferno", vmin=0.0, vmax=1.0,
            s=6, alpha=0.85, depthshade=False,
        )
        # No colourbar: the shading is a qualitative "worth a look" cue rather
        # than a measurement, and a colourbar would not follow ax3d when the
        # MIP toggle moves it, leaving a stranded scale behind.
        self.ax3d.set_xlabel("x (mm)", fontsize=8)
        self.ax3d.set_ylabel("y (mm)", fontsize=8)
        self.ax3d.set_zlabel("z (mm)", fontsize=8)
        self.ax3d.tick_params(labelsize=7)
        self.ax3d.locator_params(nbins=4)
        self.ax3d.set_title("aorta surface\nhot = vessel-like (the IVC scores high too)",
                            fontsize=8)
        self._equalize_3d_aspect()

        self.status = self.figure.text(0.01, 0.015, "", fontsize=10, family="monospace")

        self.figure.canvas.mpl_connect("button_press_event", self._on_press)
        self.figure.canvas.mpl_connect("button_release_event", self._on_release)
        self.figure.canvas.mpl_connect("key_press_event", self._on_key)

        self._apply_layout()
        self._redraw_slices()
        self._refresh()

    def _apply_layout(self):
        """Move the axes between the MIP-on and MIP-off layouts."""
        grid = self._grid_mip_on if self.show_mip else self._grid_mip_off
        self.ax3d.set_position(
            (grid[:, 0] if self.show_mip else grid[:, 0:2]).get_position(self.figure))
        self.ax_slice.set_position(grid[0, 2].get_position(self.figure))
        self.ax_zoom.set_position(grid[1, 2].get_position(self.figure))
        for axis, cell in ((self.ax_cor, grid[0, 1]), (self.ax_sag, grid[1, 1])):
            axis.set_visible(self.show_mip)
            if self.show_mip:
                axis.set_position(cell.get_position(self.figure))

    def _equalize_3d_aspect(self):
        """Equal mm-per-unit on all three axes, so the aorta is not stretched
        into an unrecognisable shape by its own tall, narrow bounding box."""
        low = self.surface_mm.min(axis=0)
        high = self.surface_mm.max(axis=0)
        centre = (low + high) / 2.0
        half = (high - low).max() / 2.0
        self.ax3d.set_xlim(centre[0] - half, centre[0] + half)
        self.ax3d.set_ylim(centre[1] - half, centre[1] + half)
        self.ax3d.set_zlim(centre[2] - half, centre[2] + half)

    # ----------------------------------------------------------------- picks

    def _pick_surface_point(self, event):
        """The surface point under the cursor, on the wall facing the camera.

        Matplotlib's 3D axes has no depth buffer, so a click near the aorta's
        silhouette is ambiguous between the near and far wall -- and the far
        wall is a whole aortic diameter away, far more than evaluate.py's
        10 mm matching tolerance. Depth is therefore resolved explicitly, in
        two stages:

          1. Keep only points within NEAR_WALL_DEPTH_MM of the frontmost
             candidate. That discards the far wall.
          2. Among those, take the one closest to the cursor *on screen*.

        Doing it in that order matters. Taking the frontmost candidate
        outright -- the obvious one-stage version -- silently drags every
        pick toward the tube's visual centre ridge, because on a convex
        surface the most camera-facing point inside the pick disc is the one
        nearest that ridge, not the one under the cursor.
        """
        projection = self.ax3d.get_proj()
        homogeneous = np.column_stack([self.surface_mm, np.ones(len(self.surface_mm))])
        projected = homogeneous @ projection.T
        projected = projected[:, :3] / projected[:, 3:4]

        display = self.ax3d.transData.transform(projected[:, :2])
        offsets = display - np.array([event.x, event.y])
        screen_distance_sq = (offsets ** 2).sum(axis=1)
        near = np.flatnonzero(screen_distance_sq <= PICK_RADIUS_PX ** 2)
        if near.size == 0:
            return None

        elevation = np.radians(self.ax3d.elev)
        azimuth = np.radians(self.ax3d.azim)
        eye = np.array([
            np.cos(elevation) * np.cos(azimuth),
            np.cos(elevation) * np.sin(azimuth),
            np.sin(elevation),
        ])
        # The eye direction above is defined in the axes' normalised cube, so
        # the points have to be divided by each axis' span before projecting
        # onto it; the resulting depth is then in units of that span.
        spans = np.array([
            np.ptp(self.ax3d.get_xlim()), np.ptp(self.ax3d.get_ylim()), np.ptp(self.ax3d.get_zlim()),
        ])
        depth = (self.surface_mm[near] / spans) @ eye
        tolerance = NEAR_WALL_DEPTH_MM / max(float(np.mean(spans)), 1e-6)
        facing = depth >= depth.max() - tolerance

        candidates = near[facing]
        return int(candidates[np.argmin(screen_distance_sq[candidates])])

    def _slice_click_to_mm(self, event):
        """Map a click in an axial panel back to physical mm, keeping z."""
        if self.pending_mm is None or event.xdata is None:
            return None
        index = physical_to_continuous_index(self.pending_mm, self.image)[0]
        index[0] = event.xdata
        index[1] = event.ydata
        index[2] = self.slice_z

        spacing = np.array(self.image.GetSpacing())
        origin = np.array(self.image.GetOrigin())
        direction = np.array(self.image.GetDirection()).reshape(3, 3)
        return (index * spacing) @ direction.T + origin

    def _on_press(self, event):
        self._press_xy = (event.x, event.y)

    def _on_release(self, event):
        if self._press_xy is None:
            return
        dragged = abs(event.x - self._press_xy[0]) > 3 or abs(event.y - self._press_xy[1]) > 3
        self._press_xy = None
        if dragged or event.button != 1:
            return  # a rotate/pan gesture, not a click

        if event.inaxes is self.ax3d:
            self._handle_3d_click(event)
        elif event.inaxes is self.ax_cor:
            self._handle_mip_click(event, "coronal")
        elif event.inaxes is self.ax_sag:
            self._handle_mip_click(event, "sagittal")
        elif event.inaxes in (self.ax_slice, self.ax_zoom):
            moved = self._slice_click_to_mm(event)
            if moved is not None and self.state == "PENDING":
                self.pending_mm = moved
                self.message = "Point nudged within the slice."
                self._redraw_slices()
                self._refresh()

    def _handle_3d_click(self, event):
        index = self._pick_surface_point(event)
        if index is None:
            self.message = "No surface point near that click."
            self._refresh()
            return
        self._place_point(self.surface_mm[index], self.normals[index],
                          "Check the axial panels, then y to confirm or n to discard.")

    def _handle_mip_click(self, event, view):
        point = self._mip_click_to_mm(event, view)
        if point is None:
            self.message = "That click was outside the volume."
            self._refresh()
            return
        # A MIP click is a coarse localisation: the in-plane position is
        # exact but the depth was inferred. The point is deliberately left
        # where the user put it rather than snapped to the mask surface, so
        # the axial panels stay the thing that decides the final position.
        nearest = int(np.argmin(np.linalg.norm(self.surface_mm - point, axis=1)))
        self._place_point(point, self.normals[nearest],
                          f"Placed from the {view} MIP (depth inferred). "
                          "Refine it on the axial panels, then y to confirm.")

    def _place_point(self, point_mm, normal, message):
        """Shared endpoint for every way of putting a point down."""
        point_mm = np.asarray(point_mm, dtype=float)
        if self.state == "DIRECTION":
            self._finish_direction(point_mm)
            self._refresh()
            return

        self.pending_mm = point_mm
        self.pending_normal = normal
        self.state = "PENDING"
        self.slice_z = int(np.clip(
            round(physical_to_continuous_index(point_mm, self.image)[0][2]),
            0, self.image_arr.shape[0] - 1))
        self.message = message
        self._redraw_slices()
        self._refresh()

    # ------------------------------------------------------------ key events

    def _on_key(self, event):
        key = event.key
        if self.radius_buffer is not None:
            self._handle_radius_key(key)
            return

        if key == "r":
            self.radius_buffer = ""
            self.message = "Type a radius in mm, Enter to accept, escape to cancel."
        elif key in ("y", "enter") and self.state == "PENDING":
            self._confirm_pending()
        elif key in ("n", "escape") and self.state in ("PENDING", "DIRECTION"):
            self._cancel_pending()
        elif key == "d" and self.state == "DIRECTION":
            self.message = "Click the 3D surface further along the branch."
        elif key == "k" and self.state == "DIRECTION":
            self._finish_direction(None)
        elif key in ("[", "]") and self.pending_mm is not None:
            self._step_slice(-1 if key == "[" else 1)
        elif key == "m":
            self.show_mip = not self.show_mip
            self._apply_layout()
            self._redraw_mips()
            self.message = f"MIP panels {'shown' if self.show_mip else 'hidden'}."
        elif key in ("-", "=", "+") and self.show_mip:
            self._resize_slab(-SLAB_STEP_MM if key == "-" else SLAB_STEP_MM)
        elif key == "u":
            self._undo()
        elif key == "s":
            self._save()
        elif key == "q":
            self._save()
            plt.close(self.figure)
            return
        else:
            return
        self._refresh()

    def _handle_radius_key(self, key):
        if key == "escape":
            self.radius_buffer = None
            self.message = "Radius unchanged."
        elif key in ("enter", "return"):
            try:
                value = float(self.radius_buffer)
                if value <= 0:
                    raise ValueError
                self.radius_mm = value
                self.message = f"Radius set to {value:.1f} mm."
            except ValueError:
                self.message = f"'{self.radius_buffer}' is not a positive number; radius unchanged."
            self.radius_buffer = None
        elif key == "backspace":
            self.radius_buffer = self.radius_buffer[:-1]
        elif key in "0123456789.":
            self.radius_buffer += key
        self._refresh()

    def _step_slice(self, delta):
        self.slice_z = int(np.clip(self.slice_z + delta, 0, self.image_arr.shape[0] - 1))
        self._redraw_slices()

    def _resize_slab(self, delta):
        low, high = SLAB_LIMITS_MM
        self.slab_mm = float(np.clip(self.slab_mm + delta, low, high))
        self._redraw_mips()
        self.message = f"MIP slab {self.slab_mm:.0f} mm."

    # -------------------------------------------------------------- workflow

    def _confirm_pending(self):
        self.state = "DIRECTION"
        self.message = ("Confirmed. Click a second point further along the branch "
                        "to set its direction, or k to use the outward normal.")

    def _cancel_pending(self):
        self.state = "IDLE"
        self.pending_mm = None
        self.pending_normal = None
        self.message = "Point discarded."
        self._redraw_slices()

    def _finish_direction(self, second_point):
        if second_point is None:
            direction = self.pending_normal
        else:
            delta = np.asarray(second_point, dtype=float) - self.pending_mm
            norm = np.linalg.norm(delta)
            if norm < 1e-6:
                self.message = "Second point coincides with the first; using the outward normal."
                direction = self.pending_normal
            else:
                direction = delta / norm

        norm = np.linalg.norm(direction)
        direction = direction / norm if norm > 1e-6 else np.array([0.0, 0.0, 1.0])

        instance_id = f"branch_{len(self.daughters) + 1:03d}"
        self.daughters.append(make_daughter(
            instance_id=instance_id,
            ostium_xyz_mm=self.pending_mm,
            seed_xyz_mm=self.pending_mm + direction * SEED_OFFSET_MM,
            radius_mm=self.radius_mm,
            direction_xyz=direction,
        ))
        self.message = f"Saved {instance_id} (r={self.radius_mm:.1f} mm). {len(self.daughters)} total."
        self.state = "IDLE"
        self.pending_mm = None
        self.pending_normal = None
        self._redraw_slices()

    def _undo(self):
        if not self.daughters:
            self.message = "Nothing to undo."
            return
        removed = self.daughters.pop()
        # Keep instance_ids contiguous, so branch_001..branch_00N always holds.
        for position, daughter in enumerate(self.daughters, start=1):
            daughter["instance_id"] = f"branch_{position:03d}"
        self.message = f"Removed {removed['instance_id']}. {len(self.daughters)} left."
        self._redraw_slices()

    def _save(self):
        prediction = make_prediction(self.case_id, self.daughters)
        write_prediction(prediction, self.output_path)
        self.message = f"Wrote {len(self.daughters)} branches to {self.output_path}"

    # ------------------------------------------------------------------ MIP

    # Which array axis each MIP collapses, and what the two surviving axes
    # mean. The volume is (z, y, x), so collapsing axis 1 leaves (z, x) --
    # a coronal view -- and collapsing axis 2 leaves (z, y), a sagittal one.
    # `index_axis` is the position of the collapsed axis in (x, y, z) index
    # order, and `plot_axis` the position of the one drawn horizontally.
    MIP_VIEWS = {
        "coronal": {"array_axis": 1, "index_axis": 1, "plot_axis": 0,
                    "label": "coronal MIP - renal arteries leave sideways here"},
        "sagittal": {"array_axis": 2, "index_axis": 0, "plot_axis": 1,
                     "label": "sagittal MIP - coeliac / SMA leave forwards here"},
    }

    def _slab_bounds(self, view):
        """Index range of the slab along the axis this view collapses."""
        spec = self.MIP_VIEWS[view]
        centre_index = physical_to_continuous_index(self.mip_centre_mm, self.image)[0]
        centre = int(round(centre_index[spec["index_axis"]]))
        spacing = self.image.GetSpacing()[spec["index_axis"]]
        half = max(1, int(round((self.slab_mm / 2.0) / spacing)))
        extent = self.image_arr.shape[spec["array_axis"]]
        return max(0, centre - half), min(extent, centre + half + 1)

    def _project(self, view):
        """Thick-slab maximum-intensity projection plus the mask silhouette."""
        spec = self.MIP_VIEWS[view]
        low, high = self._slab_bounds(view)
        axis = spec["array_axis"]
        slab = self.mip_volume.take(range(low, high), axis=axis)
        mask_slab = self.mask_arr.take(range(low, high), axis=axis)
        return slab.max(axis=axis), mask_slab.any(axis=axis), (low, high)

    def _mip_click_to_mm(self, event, view):
        """Turn a click on a MIP panel back into a 3D physical point.

        A projection has thrown the depth away, so it has to be recovered.
        Rather than defaulting to the middle of the slab, the depth is taken
        from the voxel in that ray that looks most like vessel: on a vessel
        pixel that lands on the vessel itself, which is the whole reason the
        user clicked there. Taking argmax of the projection instead would not
        work, since the clipping needed to tame bone leaves long ties at the
        ceiling. See _depth_along_ray for the bone guard that this needs.
        """
        if event.xdata is None or event.ydata is None:
            return None
        spec = self.MIP_VIEWS[view]
        low, high = self._slab_bounds(view)

        z = int(round(event.ydata))
        across = int(round(event.xdata))
        if not (0 <= z < self.image_arr.shape[0]):
            return None
        limit = self.image_arr.shape[2 if spec["plot_axis"] == 0 else 1]
        if not (0 <= across < limit):
            return None

        ray = (self.image_arr[z, low:high, across] if view == "coronal"
               else self.image_arr[z, across, low:high])
        depth = low + self._depth_along_ray(ray)

        index = np.zeros(3)
        index[2] = z
        index[spec["plot_axis"]] = across
        index[spec["index_axis"]] = depth
        return index_to_physical(index, self.image)[0]

    def _depth_along_ray(self, ray):
        """Offset of the most vessel-like voxel along one ray through a slab.

        Scored by closeness to this case's lumen HU, with bone and its
        partial-volume rim excluded outright -- without that guard a coronal
        ray near the midline resolves onto the vertebra, whose rim ties with
        a real vessel at exactly the lumen HU. Remaining ties break towards
        the centre of the slab, where the aorta is.
        """
        ray = np.asarray(ray, dtype=float)
        if ray.size == 0:
            return 0

        bone = ray > self.lumen_median + MIP_BONE_MARGIN_HU
        if bone.any():
            blocked = bone.copy()
            for shift in range(1, MIP_BONE_RIM_VOXELS + 1):
                blocked[shift:] |= bone[:-shift]
                blocked[:-shift] |= bone[shift:]
        else:
            blocked = np.zeros_like(bone)

        distance = np.abs(ray - self.lumen_median)
        # If bone fills the whole ray there is nothing vessel-like to find;
        # fall back to the unguarded score rather than returning nothing.
        if not blocked.all():
            distance = np.where(blocked, np.inf, distance)

        tied = np.flatnonzero(distance <= distance.min() + MIP_DEPTH_TIE_HU)
        centre = (ray.size - 1) / 2.0
        return int(tied[np.argmin(np.abs(tied - centre))])

    def _redraw_mips(self):
        if not self.show_mip:
            return
        spacing = self.image.GetSpacing()
        centre_index = physical_to_continuous_index(self.mip_centre_mm, self.image)[0]

        for view, axis in (("coronal", self.ax_cor), ("sagittal", self.ax_sag)):
            spec = self.MIP_VIEWS[view]
            axis.clear()
            axis.set_xticks([])
            axis.set_yticks([])

            projection, silhouette, (low, high) = self._project(view)
            across_spacing = spacing[spec["plot_axis"]]
            axis.imshow(projection, origin="lower", cmap="gray",
                        vmin=self.slice_vmin, vmax=self.mip_vmax,
                        aspect=spacing[2] / across_spacing)
            axis.contour(silhouette.astype(float), levels=[0.5],
                         colors="tab:cyan", linewidths=0.7, alpha=0.8)

            half = MIP_HALF_WIDTH_MM / across_spacing
            axis.set_xlim(centre_index[spec["plot_axis"]] - half,
                          centre_index[spec["plot_axis"]] + half)

            for daughter in self.daughters:
                marker = physical_to_continuous_index(daughter["ostium_xyz_mm"], self.image)[0]
                axis.plot(marker[spec["plot_axis"]], marker[2], "o", color="lime",
                          markersize=7, fillstyle="none", markeredgewidth=1.6)
            if self.pending_mm is not None:
                marker = physical_to_continuous_index(self.pending_mm, self.image)[0]
                axis.plot(marker[spec["plot_axis"]], marker[2], "X", color="red", markersize=9)

            thickness = (high - low) * spacing[spec["index_axis"]]
            axis.set_title(f"{spec['label']}  |  slab {thickness:.0f} mm (- / = to resize)",
                           fontsize=8)

    # -------------------------------------------------------------- drawing

    def _redraw_slices(self):
        self._redraw_mips()
        for axis in (self.ax_slice, self.ax_zoom):
            axis.clear()
            axis.set_xticks([])
            axis.set_yticks([])

        if self.pending_mm is None:
            self.ax_slice.set_title("no point selected", fontsize=9)
            self.ax_zoom.set_title("click a branch on a MIP panel to inspect it here", fontsize=9)
            self._draw_3d_markers()
            return

        z = self.slice_z
        index = physical_to_continuous_index(self.pending_mm, self.image)[0]

        for axis in (self.ax_slice, self.ax_zoom):
            axis.imshow(self.image_arr[z], origin="lower", cmap="gray",
                        vmin=self.slice_vmin, vmax=self.slice_vmax, aspect="equal")
            if self.mask_arr[z].any():
                axis.contour(self.mask_arr[z].astype(float), levels=[0.5],
                             colors="tab:cyan", linewidths=0.8)
            axis.plot(index[0], index[1], "o", color="red", markersize=9,
                      fillstyle="none", markeredgewidth=1.5)

        offset = index[2] - z
        self.ax_slice.set_title(
            f"axial z={z} ({offset:+.1f} slices from the point) - [ / ] to step", fontsize=9)

        spacing = self.image.GetSpacing()
        half_x = ZOOM_HALF_WIDTH_MM / spacing[0]
        half_y = ZOOM_HALF_WIDTH_MM / spacing[1]
        self.ax_zoom.set_xlim(index[0] - half_x, index[0] + half_x)
        self.ax_zoom.set_ylim(index[1] - half_y, index[1] + half_y)
        self.ax_zoom.set_title(
            f"zoom +/-{ZOOM_HALF_WIDTH_MM:.0f} mm - a branch is a bright tube "
            "touching the lumen", fontsize=9)

        self._draw_3d_markers()

    def _draw_3d_markers(self):
        for artist in getattr(self, "_markers", []):
            artist.remove()
        self._markers = []

        for daughter in self.daughters:
            ostium = daughter["ostium_xyz_mm"]
            seed = daughter["seed_xyz_mm"]
            self._markers.append(self.ax3d.scatter(
                *ostium, s=60, c="lime", marker="o", depthshade=False, zorder=5))
            line = np.array([ostium, seed])
            self._markers.extend(self.ax3d.plot(
                line[:, 0], line[:, 1], line[:, 2], "-", color="lime", linewidth=2))

        if self.pending_mm is not None:
            self._markers.append(self.ax3d.scatter(
                *self.pending_mm, s=90, c="red", marker="X", depthshade=False, zorder=6))

    def _refresh(self):
        if self.radius_buffer is not None:
            radius_text = f"radius: {self.radius_buffer}_"
        else:
            radius_text = f"radius: {self.radius_mm:.1f} mm (r to edit)"

        if self.pending_mm is None:
            point_text = "point: -"
        else:
            x, y, z = self.pending_mm
            point_text = f"point: ({x:7.1f}, {y:7.1f}, {z:7.1f}) mm"

        self.status.set_text(
            f"[{self.state}] {point_text}   {radius_text}   "
            f"branches: {len(self.daughters)}\n{self.message}\n"
            "click a MIP or the 3D surface to place a point   |   "
            "y confirm  n discard  d/k direction  [ ] slice  m MIP  - = slab  "
            "r radius  u undo  s save  q save+quit"
        )
        self.figure.canvas.draw_idle()

    def run(self):
        plt.show()
        self._save()
        return self.daughters


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Manually label ground-truth branch ostia on one case.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--image", required=True, help="CT volume (.nii/.nii.gz)")
    parser.add_argument("--aorta-mask", required=True, help="Aorta-only mask (.nii/.nii.gz)")
    parser.add_argument("--output", required=True, help="Where to write the ground-truth JSON")
    parser.add_argument("--case-id", default=None,
                        help="Override the case_id (default: the image's parent folder name)")
    parser.add_argument("--surface-stride", type=int, default=1,
                        help="Subsample the surface points, if rotation feels sluggish")
    parser.add_argument("--raw-shading", action="store_true",
                        help="Shade the surface from raw HU instead of building the "
                             "pipeline's lumen evidence volume: starts in about a second "
                             "rather than ~20, at the cost of a noisier view")
    parser.add_argument("--resume", action="store_true",
                        help="Load branches already in --output and keep adding to them")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    image, mask = load_case(args.image, args.aorta_mask)
    case_id = args.case_id or case_id_from_path(args.image)

    labeller = Labeller(image, mask, case_id, args.output,
                        surface_stride=max(1, args.surface_stride),
                        raw_shading=args.raw_shading)

    if args.resume and os.path.exists(args.output):
        existing = read_prediction(args.output)
        labeller.daughters = existing.get("daughters", [])
        labeller.message = f"Resumed with {len(labeller.daughters)} existing branches."
        labeller._redraw_slices()
        labeller._refresh()

    print(f"{case_id}: {len(labeller.surface_mm)} surface points, "
          f"lumen median {labeller.lumen_median:.0f} HU")
    print("Close the window or press q to save.")
    labeller.run()
    print(f"Wrote {len(labeller.daughters)} branches to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
