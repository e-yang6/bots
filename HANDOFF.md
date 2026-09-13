# Handoff: Branchseed Challenge — vesselness-gated flood fix

Repo: `e-yang6/tung-tung-saharteries`. This document exists so you can paste
one prompt into a fresh Claude Code session and continue with full context,
without re-deriving anything already established.

## What this project is

A rule-based (no ML/no trained model) pipeline that takes a CT volume + a
binary aorta-only mask and reports every artery branching directly off the
aorta as JSON (`branch_001`, `branch_002`, ... with ostium/seed/radius/
direction). Entry point: `run.py --image ... --aorta-mask ... --output ...`.
`README.md` is the source of truth for architecture and validation strategy —
read it before touching pipeline behaviour. `CLAUDE.md` holds the "easy to
get wrong" conventions (coordinate frames, index order, no hardcoded HU,
crop-before-resample, etc.) — also read it first.

**Validation discipline, load-bearing, do not violate it:**
`scripts/validate_phantom.py` is the ONLY place with exact ground truth
(synthetic phantoms, analytically-known answers) and the only place that can
honestly report precision/recall/F1/error-in-mm. `scripts/stability.py`,
`sweep.py`, `plausibility.py` speak to real-case stability/plausibility, never
accuracy — there is no ground truth on real cases. The eval set at
`TORALIS CHALLENGE  2/EVAL_SET/` (note: **folder name has a trailing space
after "CHALLENGE", quote it or glob it**) is draft, `expert_review_pending`,
non-adjudicated data — useful for finding failure modes, never for tuning or
for accuracy claims. **Any fix must be developed and validated against the
phantom suite. EVAL_SET is read-only / held-out, checked only as an
afterthought, and reported as "informative, not an accuracy claim."**

## What's already been done (chronological, this all actually happened and is verified)

1. **Radius eligibility gate.** `src/rules.py` has `MIN_RADIUS_MM = 1.0`
   (competition's 2mm-diameter minimum) and `RADIUS_VETO_MM = 0.7` (a
   measured 0.3mm tolerance for DT-based radius-measurement noise on thin
   vessels — bare 1.0mm cutoff cost 58 legitimate branches in the phantom
   suite; the 0.7mm gate recovers 39 of them at the cost of 6 new false
   positives). This is settled and documented in README's "Phantom
   validation" section. **Out of scope for this task — do not touch.**

2. **EVAL_SET located and scored** (5 draft cases, case_19–case_23).
   Aggregate on the current, unmodified pipeline: **P=0.769, R=0.526,
   F1=0.625** (10 TP, 3 FP, 9 FN across 19 draft daughters). Per-case
   numbers, ostium/seed/direction error, and full FN/FP attribution are
   written up in `EVAL_SCORECARD.md` at the repo root (also published as an
   artifact if that link still resolves:
   https://claude.ai/code/artifact/d4a29513-d351-4a5e-9c0b-caf7e785ec34).
   **This is the baseline the new work must be compared against.**

3. **Root-caused the FN/FP pattern.** Of 9 FN: 1 is the known radius veto,
   1 is the known 5mm length rule, and **6 are a new, previously undocumented
   failure**: the flood's adaptive HU-threshold search settles too strict
   because a genuine daughter runs into a brighter (or equally bright),
   non-tubular organ bed (kidney/bowel-type tissue) it feeds — any threshold
   low enough to reach the branch also admits the whole bed as one connected
   component (measured: 96.8–99.8% of newly-reached territory in one
   component, 41/41 of the missed branches' own centerline points inside it).
   Of 3 FP in case_21, 2 correspond almost exactly to locations the eval
   set's own annotator flagged as "unresolved"/"excluded" in
   `review_notes.md` — not confirmed pipeline errors.

4. **Three candidate fixes were tested and are CONTRAINDICATED — do not
   redo this work:**
   - Loosening leak-detection sensitivity — the leaks are real, contiguous
     tissue, not misclassified vessels.
   - Letting the bisection search try lower thresholds — 2 of the 6 misses
     aren't even threshold-limited (one branch's dimmest voxels sit below the
     search range floor entirely).
   - Excluding branch-adjacent regions from the leak check — impossible; the
     branch's own centerline sits *inside* the leak component, so there is
     nothing to carve out.
   - Also measured: disabling/loosening `binary_opening` in
     `build_traversal_volume` — reaches more branches in isolation but is
     **net-negative end-to-end** (P/R/F1 on EVAL_SET dropped in every variant
     tried), because reaching further elsewhere raises the settled threshold
     and drops other cases. This is the "recovers 3 but costs 5" trap —
     measure end-to-end, not just reach.

5. **A phantom that reproduces the failure was built and is now in the
   repo**, in `scripts/make_phantom.py` (`organ_bed_variant` param — three
   variants: `bed_brighter`, `equal_hu`, `branch_brighter_control`) with
   matching structural checks in `scripts/validate_phantom.py`
   (`organ_bed_not_reported`, `organ_feeder_is_reported`,
   `organ_bed_branch_is_reported`) and construction-invariant unit tests in
   `tests/test_detection.py::test_organ_bed_phantom_construction`. Verified
   deterministic across 6 seeds. Full mechanism, tuning history (why the
   first two attempts didn't reproduce, and why the third variant is a
   *negative control* not a bug), and exact numbers are in `README.md`'s
   "Phantom validation" and "Known limitations" sections — **read those
   before changing this phantom.**
   - **Current phantom suite: 158/161 structural checks pass.** The 3
     failures are: the 2 organ-bed `organ_bed_branch_is_reported` checks
     (deliberately open — this is the fix's regression target) and 1
     pre-existing unrelated `close_pair_stays_two_instances` failure at
     1.5mm spacing (not in scope here).
   - **71/71 unit tests pass** (`python3 -m pytest -q`).

6. **Vesselness in this codebase, exactly where it lives (read before
   writing code):**
   - `src/candidates.py:301` `compute_vesselness()` is the ACTIVE one — runs
     multi-scale Frangi objectness, but **only in a crop around each
     surviving component, after candidates already exist** (called from
     inside candidate feature computation, ~line 473). Its own docstring:
     *"Run over the whole volume this dominates the runtime budget; restricted
     to each surviving component's bounding box... it is a few hundred
     thousand voxels at most. It is a reported feature and a radius
     cross-check, never a gate."* — i.e. today it's a diagnostic, never a
     hard gate, and it deliberately avoids running over the whole volume for
     cost reasons.
   - `src/lumen_evidence.py` has an OLDER, now-unused
     `_multiscale_vesselness()` that DOES run over the whole cropped volume —
     its own module docstring says *"The pipeline no longer builds these
     fields... the connectivity detector avoids [it]"*. This is legacy code
     kept for `scripts/label_case.py`'s visualization only.
   - **Implication for this task:** gating the FLOOD on vesselness means
     computing it BEFORE/DURING the flood, over the whole traversal volume
     (or at least the full budget region), every threshold-search iteration
     (up to `MAX_SEARCH_ITERATIONS` + 1 times per case, per
     `src/floodfill.py`'s `robust_flood`). That is exactly the cost the
     existing code chose to avoid. Budget and measure this explicitly.

## The task (paste this into a fresh session)

```
Repo: e-yang6/tung-tung-saharteries

Now implement the vesselness-gated flood fix, developed against the new
dim-branch-into-blob phantom (and the full existing phantom suite) — NOT
against EVAL_SET, which stays held out per this project's own rule that
accuracy claims only come from the phantom suite.

Approach: modify the flood traversal so that, at low thresholds where a
branch and a non-tubular blob would otherwise merge into one connected
component, the flood only steps through voxels with vesselness (already
computed in src/lumen_evidence.py) above some gate value — i.e. the flood
can travel through tube-like structure even at low intensity, but is blocked
from spreading into blob-like structure regardless of brightness. This
should let the flood follow the dim branch without also engulfing the blob
it feeds into.

Tune the vesselness gate threshold against the phantom suite (including the
new distractor) only. Report:
- Whether the new distractor phantom is now correctly detected (dim branch
  found, blob not reported as a branch).
- Full existing phantom suite pass rate (currently 158/161) — confirm no
  regression.
- Runtime impact of adding vesselness computation earlier/more broadly in
  the flood step.

Only after phantom-suite validation is solid: re-run the EVAL_SET scoring
(src/evaluate.py) as a held-out check, and report the new aggregate P/R/F1
against the current baseline (P=0.769, R=0.526, F1=0.625), plus whether
case_20 and case_22's specific missed branches are now recovered. Report
this as informative, not as an accuracy claim, consistent with the eval
set's draft/unreviewed status.

Also re-check: does the vesselness gate introduce any new false positives on
EVAL_SET (a vesselness-gated flood being too permissive elsewhere)? Report
explicitly, don't just report recoveries.
```

**One correction to make to the prompt above before pasting, if you want it
accurate:** it says "developed against ... 142/143" in earlier drafts — the
current true baseline is **158/161** (32-phantom suite, post organ-bed
addition). The copy above already has the corrected number.

## Known risks / things to watch for (so budget isn't wasted)

- **Scale problem.** The dim branch is 1.8mm radius at this pipeline's
  0.8mm working resolution — about 2.25 voxels across. Frangi/vesselness
  filters want several voxels across a structure for a confident tubular
  response; right at the branch/bed junction (exactly where the gate has to
  work) the response may be ambiguous by construction. Test this on ONE
  organ-bed phantom before wiring anything into `floodfill.py` — compute
  vesselness on it, look at the response at the junction, and confirm
  branch-vs-bed actually separates before spending time on integration.
- **Cost of tuning.** The full suite is 32 cases, run repeatedly during
  gate-value tuning. Prefer iterating on a small subset (the 3 organ-bed
  cases + ~6 representative grid cases) and only run the full 32-case suite
  once at the end, to keep iteration cheap.
- **Do not re-litigate the three ruled-out fixes** (leak sensitivity,
  search range, region exclusion) — they're contraindicated by measured
  evidence already in README.md's "Known limitations." Re-testing them
  would just re-burn budget for the same negative result.
- **`RADIUS_VETO_MM` / `MIN_RADIUS_MM` (rules.py) and the 5mm minimum-length
  rule are explicitly out of scope** — separate, already-tuned, don't touch
  in this task.
- **Uncommitted state right now:** `README.md`, `scripts/make_phantom.py`,
  `tests/test_detection.py`, and `validate_phantom_report.json` are modified
  but not committed. There's also a commit `d88f061 "phantom suite building"`
  already on the branch that captured an earlier version of
  `scripts/validate_phantom.py` — check `git log`/`git diff` before assuming
  a clean baseline, and probably commit the current working tree before
  starting the vesselness work so it's a clean diff from here.
- **Stale generated reports not related to this task, but worth knowing
  about:** `reports/stability_report.json`, `sweep_report.json`,
  `benchmark_report.json`, and `reports/viz/` (the 3 required visual-check
  renders) all predate the radius-gate tuning and the organ-bed phantom.
  Not blocking, but don't cite numbers from them as current.

## Budget reality check (why this is a fresh-session task)

The prior session that produced all of the above (radius tuning, EVAL_SET
diagnosis, ostium-error decomposition, flood-connectivity diagnostic,
organ-bed phantom construction) cost **~$25.60** and used **35 min of API
time** in one long-running conversation, and left the account at ~8% of
session budget and ~17% of weekly budget remaining. This task (implementing
+ tuning a vesselness gate, with iterative full-suite reruns) is at least as
large as that. Recommendations if picking this up:
- Start genuinely fresh — don't paste this into a session that's already
  deep in context; >150k-token context was 86% of that session's cost.
- Tune against a subset per iteration (see above), full suite once at the
  end.
- If the weekly limit is tight, it's fine to wait for the reset rather than
  push through on fumes — nothing here is time-critical within a day or two.

## Quick reference: commands

```bash
cd "/Users/jeffreywongbusiness/bots"
python3 -m pytest -q                                    # unit tests, keep green
python3 -m scripts.validate_phantom                      # the only accuracy-claim suite
python3 run.py --image IMG --aorta-mask MASK --output pred.json [--verbose]
python3 -m src.evaluate --predictions pred.json --references ref.json
```

`python` does not exist on this machine — always `python3`.
