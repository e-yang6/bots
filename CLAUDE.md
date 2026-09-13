# CLAUDE.md

Aortic daughter-artery detection: given a CT volume and an aorta-only binary
mask, report every artery branching off the aorta as JSON.

**`README.md` is the source of truth for architecture, validation strategy and
results.** Read it before changing pipeline behaviour. This file holds only the
things that are easy to get wrong and are not obvious from the code.

## Commands

```bash
python3 -m pytest -q                     # full suite; keep it green
python3 -m pytest tests/test_detection.py -q

python3 run.py --image IMG.nii --aorta-mask MASK.nii --output pred.json [--verbose]
python3 -m src.evaluate --predictions pred.json --references ref.json

python3 -m scripts.validate_phantom      # synthetic ground truth (accuracy)
python3 -m scripts.stability             # real cases, no labels
python3 -m scripts.sweep                 # threshold / flood-budget defensibility
python3 -m scripts.plausibility          # cohort sanity checks
python3 -m scripts.benchmark             # wall-clock + peak memory
python3 scripts/label_case.py --image IMG.nii --aorta-mask MASK.nii --output gt.json
```

`python` does not exist on this machine — always `python3`.

Each script overwrites its own file in `reports/`. Those are generated output:
**never hand-edit a number in `reports/` or in the README's result tables.**
Re-run the script that produced it.

## Non-negotiable conventions

**Coordinates.** Physical millimetres come from SimpleITK only, in ITK's LPS
frame. NIfTI's sform/qform is RAS+, so a raw nibabel affine disagrees by a sign
flip on x and y. nibabel appears exactly once, as a rescue reader for files ITK
rejects, and its affine is converted to LPS before anything downstream sees it
(`src/io_utils._read_sitk_oblique_fallback`). Do not add a second path that
derives mm from nibabel.

**Index order.** numpy views of a volume are `(z, y, x)`; SimpleITK index tuples
and every `*_index_xyz` array are `(x, y, z)`. Most real bugs here are a silent
transposition between the two. When converting, use the existing helpers
(`src/geometry._indices_to_physical`, `src/lumen_evidence.VolumeSampler.to_index`,
`scripts/label_case.physical_to_continuous_index`) rather than writing the
arithmetic again.

**Never hardcode an HU constant.** This cohort's HU floor is not standard — one
case bottoms out near -9000. Any intensity offset, threshold, histogram edge or
display window must be derived from the case's own statistics (`lumen_stats`,
`image_arr.min()`). Existing code that looks like a magic number is a named
constant expressed *relative* to the case's lumen; keep it that way.

**Crop before resampling.** `crop_to_mask_bbox` must run before
`resample_isotropic`. At least one case is large enough in-plane that resampling
first would multiply the volume several times over and blow the memory budget.

**No trained model, anywhere.** Detection is entirely rule-based: every
accept/reject is a named constant with a one-line reason, and
`src.rules.explain(features)` reproduces any decision term by term. Do not
introduce a classifier, a pickle, or a fitted weight without being asked — the
transparency is a deliberate design property, not an accident.

**Fail soft.** `run.py` wraps the chain so any failure still emits valid,
schema-conformant JSON with an empty daughter list. Don't add a path that
crashes out instead.

## Output schema

`schema.py` owns it; build daughters with `make_daughter` rather than dict
literals.

```json
{"case_id": "subject001",
 "parent": {"instance_id": "aorta"},
 "daughters": [{"instance_id": "branch_001", "parent_instance_id": "aorta",
                "ostium_xyz_mm": [x,y,z], "seed_xyz_mm": [x,y,z],
                "radius_mm": r, "direction_xyz": [dx,dy,dz]}]}
```

`case_id` comes from the image's **parent folder**, not its filename
(`subject001/orig1.nii` -> `subject001`). `direction_xyz` is unit length.
`src/evaluate.py` matches ostia one-to-one with the Hungarian algorithm at a
10mm tolerance — any hand-made coordinate must land well inside that.

**The seed is not 5mm from the ostium in a straight line.** The pipeline walks
`SEED_DISTANCE_MM = 5.0` of *arc length along the traced path*
(`src/tracing.py`), so on a curved branch the Euclidean gap is larger — 5.6mm
to 13.2mm across subject001's own daughters. `scripts/label_case.py` instead
places its seed a straight 5mm along the marked direction, because a hand-
marked point has no traced path to walk. Both are defensible; they are not the
same rule, so `mean_seed_error_mm` between a manual label and a pipeline
prediction carries that definitional difference on top of any real error. Do
not "fix" either to match the other without deciding which definition wins.

## The data

Lives in `TORALIS CHALLENGE /` — **the folder name ends in a space**, and it is
gitignored. Quote it, or glob it, but don't retype it from memory.
20 subjects, each `subjectNNN/origN.nii` + `subjectNNN/maskN.nii`. Extensions
lie: some `.nii` files are actually gzip streams, which is why `io_utils` sniffs
the magic number instead of trusting the suffix.

Two tiers, and they are not interchangeable:

| cases | spacing | notes |
|---|---|---|
| subject001–015 | 0.6–1.0mm in-plane, 0.8mm slices | the usable cohort |
| subject016–020 | 1.5mm isotropic | a 2mm-radius artery is ~1 voxel wide |

- **subject017** covers only 48mm of aorta (~2k mask voxels vs ~60k typical).
- **subject017 / subject020** return zero detections *by design* — both confirmed
  correct rejections, not failures. Don't "fix" them.
- **subject018** is aneurysmal (diameter to 51.7mm vs ~15mm median) and is the
  measured outlier in every unlabelled check. Expected, disclosed.

If you need cases for manual labelling, take them from 001–015.

## Testing

Build pipeline context in tests by calling `run.analyze_volumes` on a phantom
from `tests/synthetic.py` (`make_capsule_phantom` / `make_cylinder_with_stub`).
Do **not** hand-wire the detector's stages to assemble a context: such a test
keeps passing after the pipeline changes shape underneath it, which has already
happened once in this repo.

Accuracy claims come only from the phantom suite, and only about geometry. Real
cases have no ground truth — `stability`/`sweep`/`plausibility` speak to
stability and plausibility, never accuracy. Keep that distinction in any wording
you add.

## scripts/label_case.py (manual ground-truth tool)

Interactive: 3D aorta surface + coronal/sagittal thick-slab MIPs + axial panels.
Needs a real GUI backend, so it must not import `scripts/visualize_candidates.py`,
which forces matplotlib's `Agg` at import time. Drive it headlessly in tests by
stubbing `plt.show` and synthesising events.

Four findings behind constants that look arbitrary — don't "simplify" them away:

- **Probe floor 2.5mm** (`PROBE_DISTANCES_MM`). The aorta's own partial-volume
  rim reads as near-perfect lumen for the first ~2mm; probing from 1mm lights up
  ~80% of the surface and hides every branch.
- **Two-stage picking** (`_pick_surface_point`). Taking the frontmost point in
  the click disc drags picks toward the tube's visual centre ridge: measured 9.7mm
  median error, only 59% inside the 10mm tolerance. Gate on depth first, then take
  nearest-on-screen: 0.00mm median.
- **Bone guard on MIP depth** (`_depth_along_ray`). Bone ramps through the lumen HU
  on its way to ~1500, so it ties with a real vessel. Without the guard, 13.9% of
  sagittal clicks resolved onto the vertebra.
- **MIP from clipped raw HU, not the lumen evidence field.** That field is a
  near-binary band-pass; projecting it saturates kidney, marrow and vessel into
  one white blob.

The IVC is the main confuser and cannot be suppressed automatically — it is a
large contrast-filled vein hugging the aorta for the whole scan. Distinguish it
by scrubbing slices: it persists essentially unchanged over ~100 slices and never
joins the lumen.

## Local Windows environment

On this Windows checkout, use `py -3.11` for the commands above: Python 3.11.9
has the required pipeline dependencies, while `python3` resolves to the Microsoft
Store stub. `rtk` is not available in Bash or PowerShell. The local data folder is
`TORALIS CHALLENGE` without a trailing space and currently contains subjects
001–025. The draft references are in `TORALIS CHALLENGE/EVAL_SET/case_19` through
`case_23`, with `origN.nii.gz`, `aortaN.nii.gz`, and `annotations.json` per case.
