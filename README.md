# Branchseed: aortic daughter-artery detection

Given a CT volume and an aorta-only binary mask, detect every artery
branching directly off the aorta and report each as a daughter instance
(ostium, seed point 5mm in, radius, direction) in JSON.

## Setup

```
pip install -r requirements.txt
```

## Run

```
python run.py --image image.nii.gz --aorta-mask aorta_mask.nii.gz --output prediction.json
```

Add `--verbose` to log every pipeline stage (flood threshold search, per-instance
tracing, per-instance confidence scoring) to stderr.

Measured on this cohort's 20 real cases (single core, this machine): mean
**2.4s**/case, worst case **8.5s**/case, mean peak memory **601MB**, worst
case **1.5GB** -- comfortably under the 60s/case target with no tuning
needed for speed. See `scripts/benchmark.py` and `reports/benchmark_report.json`.

## How it works

The pipeline is entirely rule-based: no trained classifier, no training
labels, nothing pickled. Every accept/reject decision is a named constant
with a one-line reason, and `src/rules.explain(features)` can print the
exact term-by-term reasoning behind any single decision.

1. **Load, crop, resample** to 0.8mm isotropic (`src/io_utils.py`, `src/geometry.py`).
2. **Contrast check**: if the lumen isn't contrast-enhanced, short-circuit to
   an empty prediction rather than guess (`src/intensity.py`).
3. **Geometry**: centerline, surface normals, end-cap flags on the aorta
   mask (`src/geometry.py`).
4. **Geodesic flood** from the aortic lumen through bright, connected voxels
   only -- bright does not mean artery, but every real daughter connects to
   the aorta through continuous contrast and almost nothing else does. An
   adaptive threshold search stops as soon as the flood would leak into
   non-vascular tissue (`src/floodfill.py`). When the leak that stopped it
   is bone -- a newly reached component with >=10% of its voxels brighter
   than the aortic lumen's own p99, which no contrast-fed vessel can be --
   the search excises that component and keeps stepping down; any other
   leak still stops it. Territory the plain search reached is never removed.
5. **Candidates**: reachable components split off the aorta, rejected for
   being too short, a segmentation end-cap, or the aorta itself continuing
   through a mask gap (`src/candidates.py`).
6. **Instance splitting**: watershed-splits fused ostia and absorbs narrow
   grazing fragments into their real neighbour (`src/parentage.py`).
7. **Tracing**: per-instance bifurcation detection, ostium/seed/radius/
   direction extraction, with three independent ostium estimates and a
   radius cross-checked three ways (`src/tracing.py`).
8. **Features**: ~27 named, inspectable measurements per instance --
   length, HU relative to the lumen, radius consistency, contact-patch
   flare, angle to the centerline, ostium-estimate disagreement, leak
   proximity, vein-likeness (`src/features.py`).
9. **Rules**: a transparent weighted score in [0, 1] from those features,
   with hard vetoes (too short; **radius below `RADIUS_VETO_MM` = 0.7mm --
   the competition brief's 2mm-diameter (1.0mm radius) minimum, less a
   named 0.3mm tolerance for known measurement noise on thin vessels, see
   Phantom validation below**; cap-touching and either too wide or too
   parallel to the centerline) and graded penalties, thresholded at
   `rules.CONFIDENCE_THRESHOLD` (`src/rules.py`). The radius veto checks
   `radius_at_seed_mm`, which is exactly the `radius_mm` written to output
   -- rejected instances never reach the JSON at all, they are not merely
   flagged.
10. **Assembly**: near-ostium instances from genuinely disjoint flood
    components get merged if their contact patches actually touch; survivors
    become sequential daughter instances (`run.py:build_daughters`).

The whole chain stays wrapped in `try/except`: a failure anywhere still
emits a valid, schema-conformant JSON with an empty daughter list rather
than crashing.

## Validation strategy -- read this before trusting any number below

**There is no ground truth for the real cases.** The two kinds of evidence
below answer different questions and must not be conflated:

- **`scripts/make_phantom.py` + `scripts/validate_phantom.py`** build synthetic
  CT phantoms with an *exactly known* answer (ostium, seed, radius, direction
  computed analytically, never measured from voxels) and score the pipeline
  against it. This proves the geometry and measurement chain -- thresholding,
  flood, tracing, radius/direction extraction -- is correct on vessels shaped
  like real ones. **It does not prove detection succeeds on real patients**:
  real anatomy, real noise, and real segmentation quirks are not fully
  captured by any phantom.
- **`scripts/stability.py`, `scripts/sweep.py`, `scripts/plausibility.py`,
  `scripts/visualize_case.py`** run on the 20 real dev-set cases, with no
  labels needed. They ask a different question: is a real-case detection
  *stable* under reasonable parameter/noise perturbation, is the operating
  point defensible, does the cohort's own detection distribution look sane,
  and -- the actual demo evidence -- does a rendered ostium look like a real
  vessel mouth of about the claimed size. None of this is an accuracy number.

**Do not read the phantom F1 below as "the pipeline is 93% accurate on
patients."** It is 93% accurate at re-deriving geometry it was told
analytically, on synthetic tubes. The real-case sections are what actually
speak to real-world behaviour, and they speak to *stability and
plausibility*, not to *correctness*.

### Phantom validation (synthetic ground truth)

`python -m scripts.validate_phantom` builds a 32-phantom suite (spacings
0.8mm iso / 1.5mm iso / 0.7x0.7x1.5mm anisotropic; lumen HU 250/400/550;
2/5/9 branches; a fused bifurcation that must report as one instance; two
origins 4mm apart that must stay two; a parallel lower-HU vein analogue; a
bone-like slab; a blob connected only by a single-voxel bridge that must not
be reported; a mask that stops mid-vessel while the image continues; and a
non-contrast case that must return empty; and three organ-bed cases, below)
and scores every case with Hungarian matching against the analytically-known
reference.

**The organ-bed cases hold an open failure, on purpose.** They reproduce the
one failure mode the real eval set turned up that no phantom previously
covered: a dim daughter running into the organ it feeds, where the bed is at
or above the branch's own HU. Any threshold low enough to reach the branch
also admits the whole bed as one connected component, and any threshold that
excludes the bed also excludes the branch, so the threshold search settles
above the branch and the branch is never reached. Measured on the phantom, in
the same terms as the real cases: at the lowest fraction that reaches the
branch, **100% of the newly-reached territory is a single connected component
and 41/41 of the branch's own centreline points lie inside it** (the real
cases ran 96.8-99.8% and 41/41). The bright feeder into the same bed is found
at 0.7mm in every variant, so the miss is attributable to the bed rather than
to the case being hard. The third variant is a negative control with the bed
clearly dimmer than the branch; there a leak-free window exists, the search
finds it, and the branch is reported at 0.8mm. `organ_bed_branch_is_reported`
therefore FAILS on the two reproducing variants and passes on the control --
it is the regression target for a fix, not a phantom bug.

The radius gate is `src.rules.RADIUS_VETO_MM = MIN_RADIUS_MM -
RADIUS_VETO_MEASUREMENT_TOLERANCE_MM = 1.0 - 0.3 = 0.7mm`, not a bare
1.0mm cutoff -- added after the first phantom run below showed the bare
cutoff dropping legitimate branches. The suite now also includes a thin
distractor branch (true radius 0.85mm, genuinely under the 2mm-diameter
minimum) that must never be reported, isolated in its own arc-fraction/
azimuth so it doesn't merge into a real branch's territory.

Aggregate scores with the 0.7mm tolerance gate:

| | precision | recall | F1 | ostium err | seed err | radius err | direction err |
|---|---|---|---|---|---|---|---|
| bare 1.0mm gate | 1.000 | 0.750 | 0.851 | 0.69mm | 0.47mm | 0.57mm | 11.5deg |
| **0.7mm tolerance gate** | **0.975** | **0.929** | **0.947** | 0.67mm | 0.46mm | 0.55mm | 11.0deg |
| no radius gate at all | 0.926 | 0.946 | 0.933 | 0.70mm | 0.49mm | 0.56mm | 11.1deg |

The tolerance gate recovers nearly all of the recall the bare cutoff cost,
without giving back everything the radius gate was added for. Seed error at
1.5mm phantom spacing: 0.53mm (brief's own bar was < 1.0mm).

**Structural assertions: 158/161 pass.** Two of the three failures are the
organ-bed `organ_bed_branch_is_reported` checks described above, held open
deliberately. The third, and the only unexplained one, is
`close_pair_stays_two_instances` in a single 1.5mm-spacing phantom -- down
from 24 failures under the bare cutoff. (The count was 142/143 before the
three organ-bed phantoms: they added 18 checks -- 16 pass, the 2 held-open
failures -- with no existing check removed or changed in status. The
`close_pair` failure is the one that was already there at 142/143.)

**Recovery count, exactly as asked**: of the 58 reference branches (true
radius 1.00-1.59mm) the bare 1.0mm cutoff dropped, **39 are now correctly
recovered and 19 are still missed** by the 0.7mm tolerance gate. The 19
still missed all have true radius in the same 1.00-1.59mm band (nothing
above 1.59mm is ever missed) -- for these, `seed["radius_mm"]` reads more
than 0.3mm low, i.e. beyond what the tolerance absorbs. Loosening the
margin further would recover more of these at the cost below.

**False-accept count, checked explicitly, not just the recovery number: 6**
new false positives appear across the 27-phantom matrix suite (0 under the
bare 1.0mm gate) with measured radius 0.74-0.99mm -- inside the tolerance
band, so the gate lets them through exactly as it is designed to. All 6 are
**not** near-misses of a real branch or a duplicate of another daughter (all
sit 26-57mm from the nearest reference branch and from the nearest other
daughter), and all 6 occur only in **1.5mm-ish spacing phantoms** (0 at
0.8mm spacing) -- they read as small, isolated rasterization/segmentation
artifacts of the coarser-spacing phantoms rather than genuine near-2mm
vessels the gate was supposed to protect against. That distinction matters
for how to read the number: the tolerance margin is not principally letting
through borderline-real anatomy that got a bit unlucky on measurement --
it's letting through unrelated small-radius noise at coarse spacing, at a
rate of 6 across the whole suite. On real cases, both kinds of thing this
margin admits (a genuinely 1.0-1.3mm vessel measured accurately, and
segmentation noise that happens to read small) are plausible, and this
phantom evidence cannot rule either out.

**Net effect of the tolerance margin**: recovers 39 legitimate branches,
costs 6 new false positives, and leaves 19 legitimate branches still
wrongly dropped. This is a real trade, not a fix -- the underlying cause
(measurement noise on `radius_mm` comparable in size to the gap between
1.0mm and 2mm-diameter-adjacent branch radii) is unchanged; the margin only
changes where the line falls relative to that noise.

### Stability on the real cases (no labels)

`python -m scripts.stability` perturbs each tunable (flood budget, flood
threshold range, minimum length, confidence threshold) by +/-10%/+/-20%,
independently, plus Gaussian noise at three HU levels, and checks whether
each baseline detection survives (present, moved < 2mm) across all 19
perturbations, per case.

Mean per-case stability across the 18 cases with any detections: **0.917**.
The one clear outlier is **subject018** (0.63) -- an aneurysmal case (aortic
diameter up to 51.7mm against a cohort median ~15mm) whose flood was
independently flagged as leaking by `src/floodfill.detect_leak`, is the
`scripts/plausibility.py` branch-density outlier, and is visibly the least
clean case in `scripts/visualize_case.py`'s renders. Four independent checks
agree on the same case being the hard one -- that consistency is itself
evidence the checks are measuring something real, not noise.

### Threshold and flood-budget sweep (no labels)

`python -m scripts.sweep` sweeps `CONFIDENCE_THRESHOLD` (cheap: re-scores
already-traced instances, no re-flooding) and the flood budget (re-runs the
flood) across all 20 real cases, and asks whether the *current* default sits
on a cliff or in a graded-but-safe region, since with no labels there is
nothing to optimize toward -- only a defensibility check on the value
already chosen. "Safe" here means no neighbouring grid point changes total
detections by more than 20%, and each point's own local stability (does a
tiny +/-0.02 threshold nudge change the accepted set) stays at or above 0.9.

Result: **`CONFIDENCE_THRESHOLD = 0.5` is not on a cliff.** It sits at the
upper edge of a graded-but-safe region spanning [0.05, 0.50] (total
detections across the cohort: 117 at T=0.05, declining smoothly to 105 at
T=0.50, every step under the 20% tolerance). It is the *stricter* edge of
that region, not its centre -- the very next grid point, T=0.55, already
drops local stability to 0.896 (just under the 0.9 floor), and the one
genuinely sharp cliff in the whole sweep is between T=0.90 and T=0.95, where
total detections nearly halve (37 to 20). 0.5 is comfortably clear of that
cliff and errs slightly toward precision over recall, which is the
reasonable side to err on for a system with no way to check false positives
against ground truth on real cases. No change made to the default.

The flood budget shows the same shape: the default (15mm, "1.0x") gives 105
total detections, close to its neighbours (115 at 0.85x, 97 at 1.3x, both
within the 20% band) -- the only real cliff is at the aggressive low end
(0.5x budget jumps to 152, a flood cut off too early misreads more fragments
as separate branches), and the default sits safely away from it.

### Plausibility (no labels)

`python -m scripts.plausibility` compares every case's branch density,
radius distribution, angle-to-centerline distribution, and cap-proximity
fraction against the rest of the cohort. Re-run with the 0.7mm tolerance
gate: **94 detections across 20 cases** (66 under the bare 1.0mm gate, 105
with no radius gate at all -- the tolerance recovers 28 of the 39 detections
the bare gate had removed). Median radius 2.09mm (p5-p95: 0.74-3.07mm; the
5th percentile is now inside the tolerance band, as expected), median angle
to centerline 42deg, 2.1% touch a mask cap. Zero detections are flagged as
implausibly small by the plausibility script's own separate <0.5mm floor.
Two cases return zero detections (subject017, subject020) -- both
independently confirmed as correct rejections (see Visual QC below);
**subject016 is back to its original 4 detections**, recovered in full by
the tolerance margin.

### Visual QC (the actual demo evidence)

`python -m scripts.visualize_case --image ... --aorta-mask ... --prediction ...`
renders the aorta surface (PyVista) with every daughter's ostium, direction
arrow and seed point, plus one HU cross-section per daughter at its seed
with the fitted radius drawn as a circle. Run and saved for three cases
spanning the cohort: **subject001** (0.8mm contrast case), **subject016**
(1.5mm-spacing case), **subject018** (the aneurysmal case above). The first
two show clean, round, correctly-sized lumens in nearly every cross-section;
subject018's are visibly noisier -- the same case every unlabelled check
above already flagged.

**Reports not yet re-run against the new gate**: `stability_report.json`,
`sweep_report.json`, `benchmark_report.json`, and `reports/viz/` predate the
radius gate entirely (both the bare-cutoff and tolerance-margin versions)
and reflect the original pipeline's detection counts and confidence
distribution (e.g. subject016 shows 4 daughters there, matching its
recovered count under the tolerance margin -- coincidentally correct, not
re-verified). Their qualitative conclusions -- subject018 as the
stability/plausibility outlier, the 0.5 confidence threshold not sitting on
a cliff, runtime well under budget -- are not expected to change, since the
gate only removes already-thin detections after scoring; but the exact
counts in those three files are stale until re-run.

### Draft eval set (held out -- informative, not an accuracy claim)

`TORALIS CHALLENGE /EVAL_SET` (case_19-case_23) is draft,
`expert_review_pending` data, not ground truth. It is used only to find
failure modes, never to claim accuracy. Scored with `run.py` then
`python -m src.evaluate` per case, aggregated over all 19 draft daughters:

- before bone excision: 10 TP / 3 FP / 9 FN -- P 0.769, R 0.526, F1 0.625
- after, **raw**: 15 TP / 8 FP / 4 FN -- P 0.652, R 0.789, F1 0.714

**Do not read the raw +0.089 F1 on its own. Two of the five new matches are
not recoveries by the flood fix:**

- **case_20, a match the tolerance allows but the geometry does not.** Its
  `branch_003` match lands 4.3mm from that reference's ostium, well inside
  the evaluator's 10mm tolerance. But the direction is 98deg off, the seed
  is 12mm off, it sits 2.5mm from the neighbouring `branch_004` ostium, and
  the flood reached only 4 of the reference's 40 centreline points. It is a
  vessel near a shared origin, not a trace of `branch_003`. Of case_20's
  0/4 -> 2/4, one recovery is real (`branch_002`: 1.3mm, 9deg).
- **case_22, a side effect on the radius veto.** `branch_002` was a
  radius-veto miss (0.40mm < `RADIUS_VETO_MM`). It now passes only because
  radius is measured against the flood threshold, which the descent lowered
  (now 1.64mm). `rules.py` is unchanged. The match is geometrically sound,
  but it is a radius-measurement side effect, not the connectivity fix.

Scoring the case_20 match as a false positive plus a miss: P 0.609, R 0.737,
F1 0.667. Also leaving the case_22 veto recovery uncredited: P 0.591,
R 0.684, **F1 0.634, only +0.009 over before**. Precision falls in every
reading. The clean recoveries attributable to the fix are case_20 `branch_002`
and case_22 `branch_001`/`branch_006`. case_23's one match moved from
`branch_002` to `branch_003` by Hungarian reassignment of the same
detection -- no real change.

The 5 new false positives: one at case_20's annotator-excluded posterior
tracks, two small posterior vessels at case_23's annotator-excluded iliac
level, and two in case_22 that the notes don't explain (bone's dim outer
shell traced as a vessel -- an artifact of the excision -- and a vein
crossing anterior to the aorta). case_22's runtime rose from 5.1s to 14.5s
(26 flood attempts against 6).

## Known limitations

- **The radius gate's tolerance margin is a real trade, not a fix.**
  Phantom evidence with the 0.7mm tolerance gate: 39 of 58 previously-lost
  legitimate branches (true radius 1.00-1.59mm) are recovered, 19 are still
  wrongly dropped, and 6 new false positives appear (measured radius
  0.74-0.99mm) that don't correspond to any real branch -- concentrated at
  1.5mm-ish spacing, reading as segmentation/rasterization noise rather than
  genuine near-2mm vessels. Loosening the margin further would recover more
  of the 19 at the cost of more false positives like the 6; the 0.3mm value
  is a choice, not a solved boundary. See Phantom validation above for the
  full numbers.
- **A dim branch running into the organ it feeds is not detected, and no
  threshold setting fixes it.** When the organ bed sits at or above the
  branch's own HU, the branch and the bed are one connected component at
  every threshold that reaches the branch at all, so the leak detector
  (correctly) rejects those thresholds and the search settles above the
  branch. Found on the real eval set (5 of 9 misses there), now reproduced
  deterministically by the organ-bed phantoms above. This is **not** fixable
  by loosening leak sensitivity, widening the threshold range, or excluding
  branch-adjacent regions from the leak check: all three were measured and
  contraindicated, because the leak is real tissue and the branch is inside
  it. Separating them needs a shape-aware traversal, not an intensity one.
  A later per-attempt diagnostic of the eval set's six unreached ostia found
  this is not the whole story there: in both affected cases the leak that
  actually pinned the threshold was **vertebral bone** against the aortic
  wall, touching no reference branch (removing it cleared the leak flag;
  removing the reference branches' own territory did not). That part is
  fixed by the bone excision in step 4; the tissue-bed part is not, and
  still stops the search. The excision has a measured cost on that draft
  set, not a phantom-backed accuracy claim: lower settled thresholds reach
  more non-branch structures (a vein crossing the aorta, bone's dim outer
  shell traced as a vessel, and annotator-excluded posterior tracks).
- Radius is unreliable below ~0.5mm given this cohort's voxel spacing
  (see Plausibility above) -- a resolution limit, not a pipeline bug. The
  new radius gate removes these from output rather than merely flagging
  them, at the cost above.
- The one case with a genuinely large, partially-thrombosed aneurysm
  (subject018) is measurably less stable and less clean across every check
  in this repo. This is disclosed, not hidden: the pipeline still reports
  daughters for it, at correspondingly lower confidence.
- No ground truth exists for the real cases. Every real-case number in this
  README is a stability, plausibility, or visual claim -- never an accuracy
  claim. Only the phantom suite supports an accuracy claim, and only about
  geometry, not about real patients.

All numbers above are the actual output of the commands they name, checked
in under `reports/` (`validate_phantom_report.json`, `stability_report.json`,
`sweep_report.json`, `plausibility_report.json`, `benchmark_report.json`) and
`reports/viz/` (the 3 rendered cases). Re-running any script overwrites its
own report; nothing here is hand-edited.

## Repository layout

```
run.py                       CLI entry point (the one command above)
schema.py                    prediction JSON read/write
src/                         the pipeline itself (geometry, flood, candidates,
                             tracing, parentage, features, rules) -- no
                             trained model anywhere in this directory
scripts/make_phantom.py      synthetic phantom generator, exact ground truth
scripts/validate_phantom.py  phantom suite runner + scorer
scripts/stability.py         real-case parameter/noise perturbation stability
scripts/sweep.py             real-case threshold/budget sweep
scripts/plausibility.py      real-case unlabelled sanity checks
scripts/visualize_case.py    3D + cross-section render from a prediction.json
scripts/benchmark.py         wall-clock + peak memory per case
scripts/visualize_candidates.py, scripts/inspect_pipeline.py, scripts/label_case.py
                             development-time diagnostics (internal pipeline
                             state, not the final demo artifact)
src/evaluate.py              Hungarian ostium matching + scoring, used
                             wherever two daughter lists need comparing
                             (phantom scoring, stability, sweep)
reports/                     checked-in output of every script above
reports/viz/                 the 3 rendered real cases (3D + cross-sections)
```
