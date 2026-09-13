# Branchseed eval-set scorecard

Toralis Labs · Branchseed Challenge

Five cases (`case_19`–`case_23`), scored against the challenge's five weighted
criteria: branch discovery, ostium localisation, daughter-instance quality,
compute efficiency and reproducibility.

> **Draft data.** These references are not adjudicated ground truth. Every
> branch in this eval set carries `review_status: expert_review_pending`, and
> the reviewer sign-off register lists all 19 candidate daughters as
> **Pending** with no reviewer or date recorded. 16 of 19 reference branches
> have a `null` radius (search-envelope-limited), so radius error cannot be
> scored here at all.
>
> The numbers below are a real, reproducible run against this specific draft
> package — useful for finding failure modes, not a claim about final
> accuracy.

## Aggregate, by rubric category

| Category | Weight | Result | Detail |
|---|---|---|---|
| Branch discovery | 45% | P 0.769 · R 0.526 · F1 0.625 | 10 TP, 3 FP, 9 FN across 19 draft daughters, 13 predictions |
| Ostium localisation | 25% | 1.44 mm | Mean physical distance, matched pairs only (n=10) |
| Instance quality | 15% | 1.28 mm · 14.0° | Mean seed error, mean direction error. Radius error: 0/10 pairs comparable (null references) |
| Compute efficiency | 10% | 1.58 s avg | 215–531 MB peak RSS per case. Single CPU core, no GPU |
| Reproducibility | 5% | 5 / 5 | Every case ran via the documented CLI unmodified and emitted valid schema JSON, including the one zero-daughter case |

## Miss attribution key

- **Known limitation** — disclosed radius veto or length rule
- **Annotator-flagged** — the annotator's own notes flag this exact location as unresolved/excluded
- **New finding** — not previously disclosed
- **Ambiguous** — touches a known limitation but doesn't cleanly fit it

## Per case

### case_19

250×250×169 · 1.5mm spacing · 3 predicted / 3 reference

| P | R | F1 |
|---|---|---|
| 1.000 | 1.000 | 1.000 |

| Metric (weight) | Value |
|---|---|
| Discovery (45%) | 3/3 matched |
| Ostium err (25%) | 1.69 mm |
| Seed / dir err (15%) | 1.67 mm · 22.4° |
| Runtime / mem (10%) | 1.51 s · 311 MB |
| Valid output (5%) | Pass |

| pred ↔ ref | ostium | seed | direction | pred radius |
|---|---|---|---|---|
| branch_001 ↔ branch_002 | 1.05mm | 1.38mm | 13.2° | 0.92mm |
| branch_002 ↔ branch_001 | 1.06mm | 1.26mm | 18.1° | 1.13mm |
| branch_003 ↔ branch_003 | 2.96mm | 2.36mm | 35.8° | 2.22mm |

No misses. Every draft daughter matched, both directions.

### case_20

299×299×201 · 1.5mm spacing · 0 predicted / 4 reference

| P | R | F1 |
|---|---|---|
| — | 0.000 | — |

| Metric (weight) | Value |
|---|---|
| Discovery (45%) | 0/4 matched |
| Ostium err (25%) | N/A |
| Seed / dir err (15%) | N/A |
| Runtime / mem (10%) | 1.29 s · 430 MB |
| Valid output (5%) | Pass (empty list) |

**FN — `branch_001`–`branch_004`, all four** · **new finding**
The flood's adaptive threshold settles at 297.9 HU (0.603× lumen reference),
above the annotator's own manual thresholds (225–252 HU) for these branches
and above 2 of 4 branches' median centerline HU. Every reference ostium reads
`flood_dist = inf` — the geodesic search never connects to any of them,
upstream of both the radius veto and the absorption ceiling.

### case_21

217×217×202 · 1.5mm spacing · 6 predicted / 3 reference

| P | R | F1 |
|---|---|---|
| 0.500 | 1.000 | 0.667 |

| Metric (weight) | Value |
|---|---|
| Discovery (45%) | 3/3 matched, 3 extra |
| Ostium err (25%) | 1.94 mm |
| Seed / dir err (15%) | 1.62 mm · 7.9° |
| Runtime / mem (10%) | 1.81 s · 368 MB |
| Valid output (5%) | Pass |

| pred ↔ ref | ostium | seed | direction | pred radius |
|---|---|---|---|---|
| branch_001 ↔ branch_001 | 1.38mm | 1.58mm | 7.7° | 2.15mm |
| branch_003 ↔ branch_002 | 1.87mm | 1.30mm | 7.3° | 1.52mm |
| branch_006 ↔ branch_003 | 2.58mm | 1.97mm | 8.6° | 2.24mm |

**FP — `branch_002`** · **annotator-flagged**
0.96mm from voxel `[98,123,154]`, which this case's `review_notes.md` names
directly: "unresolved diameter and separate-origin versus common-trunk
anatomy." Not confirmed absent — the annotator declined to rule either way.

**FP — `branch_004`** · **annotator-flagged**
4.2mm from voxel `[92,130,143]`, the "looping vessel... excluded: tracking did
not establish an independent aortic origin."

**FP — `branch_005`** · **new finding**
No correspondence to anything in this case's excluded-candidate list or review
notes — unexplained by the available documentation.

### case_22

247×247×280 · 1.5mm spacing · 3 predicted / 6 reference

| P | R | F1 |
|---|---|---|
| 1.000 | 0.500 | 0.667 |

| Metric (weight) | Value |
|---|---|
| Discovery (45%) | 3/6 matched |
| Ostium err (25%) | 0.92 mm |
| Seed / dir err (15%) | 0.81 mm · 13.1° |
| Runtime / mem (10%) | 2.28 s · 531 MB |
| Valid output (5%) | Pass |

| pred ↔ ref | ostium | seed | direction | pred radius |
|---|---|---|---|---|
| branch_001 ↔ branch_003 | 0.93mm | 0.54mm | 4.5° | 1.15mm |
| branch_002 ↔ branch_005 | 1.02mm | 0.75mm | 11.3° | 2.63mm |
| branch_003 ↔ branch_004 | 0.82mm | 1.13mm | 23.5° | 2.65mm |

**FN — `branch_002`** · **known: radius veto**
Nearest scored candidate sits 1.12mm from this ostium, confidence 0.0, veto
reason "radius 0.40mm < 0.70mm." Clean match to the disclosed
`RADIUS_VETO_MM` limit.

**FN — `branch_001`** · **new finding**
`flood_dist = inf` at the ostium; no raw candidate component within 46mm.
Same connectivity gap as case_20.

**FN — `branch_006`** · **new finding**
`flood_dist = inf` at the ostium; no raw candidate component within 53mm.
Same connectivity gap as case_20.

### case_23

244×244×179 · 1.5mm spacing · 1 predicted / 3 reference

| P | R | F1 |
|---|---|---|
| 1.000 | 0.333 | 0.500 |

| Metric (weight) | Value |
|---|---|
| Discovery (45%) | 1/3 matched |
| Ostium err (25%) | 0.74 mm |
| Seed / dir err (15%) | 0.58 mm · 9.8° |
| Runtime / mem (10%) | 0.99 s · 215 MB |
| Valid output (5%) | Pass |

| pred ↔ ref | ostium | seed | direction | pred radius |
|---|---|---|---|---|
| branch_001 ↔ branch_002 | 0.74mm | 0.58mm | 9.8° | 0.87mm |

**FN — `branch_003`** · **known: length rule**
Nearest raw component (2.1mm away, one of only 4 in this case) rejected by
the pre-existing 5mm minimum traced-length rule — disclosed separately, not a
radius-specific limitation.

**FN — `branch_001`** · **ambiguous**
`flood_dist = 0.0` — the flood does reach this ostium — but no discrete
candidate component ever formed there (nearest is 16.4mm away, in a case with
only 4 components total). Closer to the absorption-ceiling pattern than a
clean miss, but it isn't the textbook "merged into a named neighbour"
signature either.

---

Predictions generated by `run.py` unmodified, one case at a time, on the
documented CLI. Matching is Hungarian assignment on ostium distance at a
10mm threshold, mirroring `src/evaluate.py`; radius comparison additionally
requires a non-null reference `radius_mm`, true for 3 of 19 draft daughters
and none of the 10 matched pairs. Runtime measured wall-clock per case on this
machine; peak memory measured separately via `/usr/bin/time -l` maximum
resident set size.
