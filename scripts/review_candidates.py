"""Review-based labelling: confirm or reject the pipeline's own candidates,
rather than finding branches from scratch.

scripts/label_case.py works but is slow -- a human has to spot every branch
unassisted. This script runs the real detection pipeline first (the same
src/floodfill.py + src/candidates.py + src/tracing.py + src/parentage.py
chain run.py wires up), then presents the instances it found one at a time,
longest-traced first, with the ostium/radius/direction it already computed. The reviewer
only has to answer "is this a real branch" -- y, n, or u for a borderline
one to skip rather than force a guess.

That produces two outputs from one pass:
  --output-labels       every reviewed candidate's full feature vector
                         (src/features.py) plus its y/n label -- training
                         data for src/classify.py's classifier.
  --output-groundtruth  the confirmed candidates, in the standard schema,
                         since a confirmed candidate already has its
                         ostium/seed/radius/direction computed -- usable
                         directly with `python -m src.evaluate`.

After the ranked list is exhausted, control falls through to
label_case.py's own free-click flow (its Labeller class is subclassed
wholesale, not reimplemented) so the reviewer can add anything visually
obvious that never produced a candidate at all -- a recall check the
ranked pass can't do by itself. Those free-click additions land in the
ground-truth output only: with no candidate behind them they have no
feature vector to label.

Usage:
    python scripts/review_candidates.py \\
        --image "TORALIS CHALLENGE/subject001/orig1.nii" \\
        --aorta-mask "TORALIS CHALLENGE/subject001/mask1.nii" \\
        --output-labels labels_for_classifier.json \\
        --output-groundtruth ground_truth.json

Controls, ranked-review pass:
    y            confirm: a real branch, using the pipeline's own ostium/
                 seed/radius/direction
    n            reject: not a real branch
    u            unsure / borderline: skip, logged as neither
    [ / ]        step the displayed axial slice
    m            show / hide the MIP panels
    - / =        thin / thicken the MIP slab
    s / q        save (/ save and quit)

Once every candidate has been reviewed, control switches to label_case.py's
free-click flow (click to place, y/n to confirm/discard, d/k for direction,
r for radius, u for undo -- see its own docstring) for the "did we miss
anything" pass over the full MIP/3D view.
"""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from schema import case_id_from_path, make_daughter  # noqa: E402
from src.features import extract_features  # noqa: E402
from src.io_utils import load_case  # noqa: E402

from label_case import (  # noqa: E402
    Labeller,
    SLAB_STEP_MM,
    physical_to_continuous_index,
)

from run import MAX_INSTANCES_TO_TRACE, analyze_case  # noqa: E402


class ReviewLabeller(Labeller):
    """label_case.Labeller, preceded by a ranked pass over real candidates.

    Everything about rendering (3D surface, MIP panels, axial + zoom,
    schema-JSON saving) and the eventual free-click pass is inherited
    unchanged. Only the review pass itself -- picking the next candidate to
    show, and what y/n/u do while one is pending -- is new.
    """

    def __init__(self, image, mask, case_id, context, output_groundtruth, output_labels,
                surface_stride=1, raw_shading=False):
        self.context = context
        self.output_labels = output_labels
        self.labeled_examples = []

        # Only instances run.py actually traced carry the trace/ostium/seed
        # evidence a review needs. Instances beyond MAX_INSTANCES_TO_TRACE
        # (run.py's own recall-vs-cost cutoff) have none of that and are left
        # to the free-click pass instead.
        #
        # The flood-fill detector has no single "peak_score" to rank by, so
        # review order is longest-traced first, breaking ties on flood volume:
        # both are proxies for "there is really a vessel here", which is what
        # puts the easy calls at the front of the queue.
        n_reviewable = len(context["traces"])
        self.review_order = sorted(
            range(n_reviewable),
            key=lambda i: (-context["traces"][i]["traced_length_mm"],
                           -context["instances"][i]["volume_mm3"]),
        )
        self.review_position = 0
        self.review_mode = len(self.review_order) > 0

        super().__init__(image, mask, case_id, output_groundtruth,
                         surface_stride=surface_stride, raw_shading=raw_shading)

        if self.review_mode:
            self._show_review_candidate()
        else:
            self.message = "No traced candidates to review -- go straight to the free-click pass."
            self._refresh()

    # ------------------------------------------------------------- review

    def _current_index(self):
        return self.review_order[self.review_position]

    def _show_review_candidate(self):
        idx = self._current_index()
        instance = self.context["instances"][idx]
        trace = self.context["traces"][idx]
        ostium_estimate = self.context["ostium_estimates"][idx]
        seed_estimate = self.context["seed_estimates"][idx]

        self.pending_mm = np.asarray(ostium_estimate["ostium_mm"], dtype=float)
        self.pending_normal = np.asarray(instance["direction_estimate"], dtype=float)
        self.radius_mm = float(seed_estimate["radius_mm"])
        self.state = "REVIEW"
        self.slice_z = int(np.clip(
            round(physical_to_continuous_index(self.pending_mm, self.image)[0][2]),
            0, self.image_arr.shape[0] - 1))

        total = len(self.review_order)
        shares = instance.get("shares_vessel_with") or []
        note = f"  (shares a vessel with #{', #'.join(str(s) for s in shares)})" if shares else ""
        if instance.get("touches_cap"):
            note += "  (touches an end cap -- often the aorta's own cut end)"
        self.message = (
            f"Instance {self.review_position + 1}/{total}  "
            f"vol={instance['volume_mm3']:.0f}mm3  "
            f"traced={trace['traced_length_mm']:.1f}mm ({trace['truncated_by']})  "
            f"r={self.radius_mm:.1f}mm{note}\n"
            "y = confirm real branch   n = reject   u = unsure, skip"
        )
        self._redraw_slices()
        self._refresh()

    def _confirm_review_candidate(self, label):
        idx = self._current_index()
        instance = self.context["instances"][idx]
        trace = self.context["traces"][idx]
        bifurcation = self.context["bifurcations"][idx]
        ostium_estimate = self.context["ostium_estimates"][idx]
        seed_estimate = self.context["seed_estimates"][idx]

        if label is not None:
            features = extract_features(
                instance, trace, bifurcation, ostium_estimate, seed_estimate
            )
            self.labeled_examples.append({
                "case_id": self.case_id,
                "instance_index": int(idx),
                "label": int(label),
                "features": features,
            })

        if label == 1:
            instance_id = f"branch_{len(self.daughters) + 1:03d}"
            self.daughters.append(make_daughter(
                instance_id=instance_id,
                ostium_xyz_mm=ostium_estimate["ostium_mm"],
                seed_xyz_mm=seed_estimate["seed_mm"],
                radius_mm=seed_estimate["radius_mm"],
                direction_xyz=seed_estimate["direction_xyz"],
            ))

        self.review_position += 1
        if self.review_position < len(self.review_order):
            self._show_review_candidate()
        else:
            self._enter_freeform_pass()

    def _enter_freeform_pass(self):
        self.review_mode = False
        self.state = "IDLE"
        self.pending_mm = None
        self.pending_normal = None

        confirmed = sum(1 for e in self.labeled_examples if e["label"] == 1)
        rejected = sum(1 for e in self.labeled_examples if e["label"] == 0)
        skipped = len(self.review_order) - confirmed - rejected
        self.message = (
            f"Reviewed all {len(self.review_order)} candidates: "
            f"{confirmed} confirmed, {rejected} rejected, {skipped} skipped.\n"
            "Now check the MIP / 3D views for anything obvious the candidate list "
            "missed. Click to place a point, y to confirm, n to discard "
            "(label_case.py controls apply from here)."
        )
        self._redraw_slices()
        self._refresh()

    # --------------------------------------------------------- key events

    def _on_key(self, event):
        if not (self.review_mode and self.state == "REVIEW"):
            super()._on_key(event)
            return

        key = event.key
        if key == "y":
            self._confirm_review_candidate(1)
        elif key == "n":
            self._confirm_review_candidate(0)
        elif key == "u":
            self._confirm_review_candidate(None)
        elif key in ("[", "]"):
            self._step_slice(-1 if key == "[" else 1)
        elif key == "m":
            self.show_mip = not self.show_mip
            self._apply_layout()
            self._redraw_mips()
        elif key in ("-", "=", "+") and self.show_mip:
            self._resize_slab(-SLAB_STEP_MM if key == "-" else SLAB_STEP_MM)
        elif key == "s":
            self._save()
        elif key == "q":
            self._save()
            plt.close(self.figure)
            return
        else:
            return
        self._refresh()

    # ------------------------------------------------------------ drawing

    def _draw_3d_markers(self):
        super()._draw_3d_markers()
        if self.review_mode and self.state == "REVIEW":
            trace_points = self.context["traces"][self._current_index()]["points_mm"]
            if trace_points.shape[0] >= 2:
                self._markers.extend(self.ax3d.plot(
                    trace_points[:, 0], trace_points[:, 1], trace_points[:, 2],
                    "-", color="orange", linewidth=2.5, zorder=7))

    def _redraw_mips(self):
        super()._redraw_mips()
        if not (self.show_mip and self.review_mode and self.state == "REVIEW"):
            return
        trace_points = self.context["traces"][self._current_index()]["points_mm"]
        if trace_points.shape[0] < 2:
            return
        index_points = physical_to_continuous_index(trace_points, self.image)
        for view, axis in (("coronal", self.ax_cor), ("sagittal", self.ax_sag)):
            spec = self.MIP_VIEWS[view]
            axis.plot(index_points[:, spec["plot_axis"]], index_points[:, 2],
                      "-", color="orange", linewidth=1.8, zorder=6)

    def _refresh(self):
        if self.review_mode:
            help_line = ("y confirm   n reject   u unsure/skip   [ ] slice   m MIP   "
                        "- = slab   s save   q save+quit")
        else:
            help_line = ("click a MIP or the 3D surface to place a point   |   "
                        "y confirm  n discard  d/k direction  [ ] slice  m MIP  - = slab  "
                        "r radius  u undo  s save  q save+quit")

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
            f"[{self.state}] {point_text}   {radius_text}   branches: {len(self.daughters)}\n"
            f"{self.message}\n{help_line}"
        )
        self.figure.canvas.draw_idle()

    # ---------------------------------------------------------------- save

    def _save(self):
        super()._save()
        with open(self.output_labels, "w") as f:
            json.dump(self.labeled_examples, f, indent=2)
        self.message += f"\nWrote {len(self.labeled_examples)} labelled examples to {self.output_labels}"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Review the detection pipeline's own candidates and label them.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--image", required=True, help="CT volume (.nii/.nii.gz)")
    parser.add_argument("--aorta-mask", required=True, help="Aorta-only mask (.nii/.nii.gz)")
    parser.add_argument("--output-labels", required=True,
                        help="Where to write the feature-vector + label dataset (JSON)")
    parser.add_argument("--output-groundtruth", required=True,
                        help="Where to write confirmed branches, in schema format")
    parser.add_argument("--case-id", default=None,
                        help="Override the case_id (default: the image's parent folder name)")
    parser.add_argument("--target-spacing", type=float, default=0.8,
                        help="Isotropic resampling spacing (mm) the pipeline runs at")
    parser.add_argument("--max-candidates", type=int, default=MAX_INSTANCES_TO_TRACE,
                        help="How many instances to trace and offer for review")
    parser.add_argument("--surface-stride", type=int, default=1,
                        help="Subsample the display surface, if rotation feels sluggish")
    parser.add_argument("--raw-shading", action="store_true",
                        help="Shade the 3D surface from raw HU instead of rebuilding the "
                             "evidence volume for display -- faster, noisier; the pipeline "
                             "itself always uses the full evidence regardless of this flag")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    case_id = args.case_id or case_id_from_path(args.image)

    print(f"Running the flood-fill / trace / parentage pipeline on {case_id} ...")
    context = analyze_case(
        args.image, args.aorta_mask,
        target_spacing=args.target_spacing, max_instances=args.max_candidates, verbose=True,
    )
    print(f"{len(context['candidates'])} candidate components, "
          f"{len(context['instances'])} instances, "
          f"{len(context['traces'])} traced and ready to review.")

    # Loaded again on the original grid: analyze_case works on a cropped,
    # resampled copy for the pipeline's own sake, but physical mm coordinates
    # are grid-independent, so display can use the native volume directly --
    # exactly what label_case.py itself shows.
    image, mask = load_case(args.image, args.aorta_mask)

    reviewer = ReviewLabeller(
        image, mask, case_id, context,
        output_groundtruth=args.output_groundtruth, output_labels=args.output_labels,
        surface_stride=max(1, args.surface_stride), raw_shading=args.raw_shading,
    )

    print("Close the window or press q to save.")
    reviewer.run()
    print(f"Wrote {len(reviewer.daughters)} confirmed branches to {args.output_groundtruth}")
    print(f"Wrote {len(reviewer.labeled_examples)} labelled examples to {args.output_labels}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
