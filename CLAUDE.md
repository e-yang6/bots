# CLAUDE.md

## What This Project Is
Hackathon submission for the TORALIS Challenge: given a CT scan and a binary aorta mask (NIfTI), detect every artery branching directly off the aorta and return each as a separate instance with its origin point (ostium), direction, and radius.

## Branches
- `main` — base.
- `ar` — working branch: detection pipeline work plus `viz/` (brought over from `origin/webxr-viz`) and gesture mode. Do all work here.
- `origin/webxr-viz` — original visualization branch (export pipeline + WebXR viewer).

## Git
- Never commit. The user commits themselves.

## Repository Structure

### Detection pipeline
- `run.py` — CLI entry point. Always emits valid JSON, even on failure.
  - Contract: `python run.py --image image.nii.gz --aorta-mask aorta_mask.nii.gz --output prediction.json`
- `schema.py` — JSON read/write helpers for the prediction format.
- `src/evaluate.py` — Hungarian-matching scorer (precision, recall, F1, ostium/seed/direction/radius errors).
- `src/io_utils.py` — Robust NIfTI loader. Handles gzip-disguised `.nii` files, oblique sforms, extreme HU values.
- `src/geometry.py` — Crop-then-resample, per-component centerline, surface normals, end-cap detection with radius/angle discriminators.
- `src/intensity.py` — Lumen HU stats, contrast-enhancement gating (p75-based skew handling).
- `src/candidates.py`, `src/tracing.py`, `src/parentage.py` — candidate detection, branch tracing, parentage checking (sessions 3+).
- `scripts/visualize_candidates.py` — debug visualization of candidates.
- `tests/` — pytest tests for the above, plus fixtures (`tests/synthetic.py`).

### Visualization (`webxr-viz`)
- `viz/export_scene.py` — aorta mask → `.glb` mesh (marching cubes + 30 iters Laplacian smoothing, fallback decimation) and prediction JSON → `scene.json`. Everything centered at the mesh centroid.
  - `python -m viz.export_scene --image orig1.nii --aorta-mask mask1.nii --prediction prediction.json --output-dir viz/output/subject001`
- `viz/export_batch.py` — batch export all cases with predictions.
- `viz/serve.py` — local HTTPS dev server; auto-generates a self-signed cert via openssl (falls back to HTTP; `--no-ssl` forces HTTP).
  - `python -m viz.serve --port 8080 --scene-dir viz/output/subject001`
- `viz/viewer/` — Three.js viewer.
  - `app.js` — scene, mesh loading, mode switching (desktop / WebXR AR / gesture), AR touch gestures.
  - `gesture-mode.js` — camera stream (`getUserMedia`), lazy-loaded MediaPipe Pose Landmarker, detection per video frame, skeleton overlay.
  - `pose-gestures.js` — pure gesture logic (landmarks → scale/yaw/tilt), no DOM. Tunables in `DEFAULT_OPTIONS`.
  - `index.html`, `style.css`.
- `tests/test_viz_export.py` — export pipeline tests with synthetic data.
- `tests/test_pose_gestures.mjs` — gesture logic tests: `node --test tests/test_pose_gestures.mjs`.

## Viewer State

### Works
- Desktop: orbit controls (drag rotate, scroll zoom).
- AR mode (Android Chrome over HTTPS): WebXR hit-test surface placement, tap to place, then pinch-to-scale, two-finger rotate, two-finger drag.
- The mesh is the patient's real anatomy from the mask.
- Branch markers (ostium spheres, seed points, direction arrows, radius rings) were stripped from the viewer; the export pipeline still writes them to `scene.json`. Old marker/info-panel/raycasting/legend code is in `webxr-viz` git history.

### Gesture mode (implemented, needs on-device testing)
- Separate from WebXR AR, toggled by the "Gesture mode" button. Camera video behind a transparent Three.js canvas; the model floats in front of the camera. Not WebXR `camera-access` (experimental, flagged, janky with MediaPipe). Pose, not Hands — full body at a distance.
- MediaPipe `@mediapipe/tasks-vision@0.10.14` (import map), `pose_landmarker_lite`, GPU delegate with CPU fallback, `VIDEO` mode, 1 pose.
- Gestures are clutched: only active while both wrists are above the hips; the pose at engage time is the baseline.
  - Wrist spread / shoulder width (world landmarks) → scale (clamped 0.2–5×).
  - Shoulder-line heading in x–z (world) → yaw, gain 2.
  - Shoulder-line angle on screen → tilt, gain 2, ±90°.
  - Both hands above the nose for 1.5 s → reset.
  - Time-based EMA smoothing, dead zones, 300 ms grace for dropouts.
- Signals are computed in display coordinates (x flipped when the video is mirrored) so the model follows what the viewer sees.

### Next
1. **Tune gesture mode on a phone** (gains, dead zones, thresholds in `DEFAULT_OPTIONS`).
2. **Re-add branch markers** once the detection pipeline produces real predictions.
3. **Visual QC for submission** — at least 3 cases showing aorta mask, detected ostia, and daughter-direction arrows.

## Key Technical Decisions
- **Coordinates**: internal = (z, y, x) numpy index space. Physical mm ONLY via SimpleITK `TransformIndexToPhysicalPoint`. Never use nibabel's affine (x/y sign flip vs SimpleITK).
- **Mesh**: centered at centroid; branch coordinates in `scene.json` offset by the same centroid. Viewer scales mm → m with 0.001.
- **HTTPS for AR/camera**: WebXR and `getUserMedia` require a secure context except on localhost. Android users must accept the self-signed cert warning.
- **No dataset in git**: `TORALIS CHALLENGE*/` is gitignored. It was committed once and purged via filter-branch — never add it back.

## Dependencies
```
numpy, scipy, pytest, SimpleITK, nibabel, matplotlib  # detection pipeline
trimesh, scikit-image                                   # viz export
```
Three.js v0.164.1 from CDN (import map in `viz/viewer/index.html`).

## Data Location
`TORALIS CHALLENGE -20260912T162830Z-1-001/TORALIS CHALLENGE/subjectNNN/` — 25 subjects, each with `origN.nii` (CT) and `maskN.nii` (aorta mask).

## UI Style
Minimal and clean. No glowing borders, no neon, no gaming-UI look. Muted colors, simple text — should look like a medical tool.

## Environment Notes
- Windows 11, PowerShell + Git Bash. `rtk` is not installed here — use plain commands.
