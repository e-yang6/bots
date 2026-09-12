# Project Context

## What This Project Is
A hackathon submission for the TORALIS Challenge: given a CT scan and binary aorta mask (NIfTI), detect every artery branching directly off the aorta and return each as a separate instance with its origin point, direction, and radius.

## Repository Structure

### Detection Pipeline (teammate's work, in progress)
Built following a 5-session plan. Sessions 1-2 are done, 3-5 are pending.

- `run.py` — CLI entry point. Always emits valid JSON even on failure. Currently returns empty daughters list (detection logic not wired yet).
  - Contract: `python run.py --image image.nii.gz --aorta-mask aorta_mask.nii.gz --output prediction.json`
- `schema.py` — JSON read/write helpers for the prediction format.
- `src/evaluate.py` — Hungarian-matching scorer (precision, recall, F1, ostium/seed/direction/radius errors).
- `src/io_utils.py` — Robust NIfTI loader. Handles gzip-disguised .nii files, oblique sforms, extreme HU values.
- `src/geometry.py` — Crop-then-resample, per-component centerline, surface normals, end-cap detection with radius/angle discriminators.
- `src/intensity.py` — Lumen HU stats, contrast-enhancement gating (p75-based skew handling).
- `tests/` — Tests for all above modules plus fixtures.

### Visualization (WebXR branch: `webxr-viz`)
Interactive 3D/AR viewer for the aorta mesh and detected branches.

- `viz/export_scene.py` — Converts aorta mask → .glb mesh (marching cubes + Laplacian smoothing) and prediction JSON → scene.json. Centers everything at mesh centroid.
  - Usage: `python -m viz.export_scene --image orig1.nii --aorta-mask mask1.nii --prediction prediction.json --output-dir viz/output/subject001`
- `viz/export_batch.py` — Batch export all cases with predictions.
- `viz/serve.py` — Local HTTPS dev server. Auto-generates self-signed cert via openssl on first run. Falls back to HTTP if openssl unavailable. Use `--no-ssl` to force HTTP.
  - Usage: `python -m viz.serve --port 8080 --scene-dir viz/output/subject001`
- `viz/viewer/` — Three.js WebXR viewer (index.html, app.js, style.css).
- `viz/README.md` — Setup and usage instructions.
- `tests/test_viz_export.py` — Export pipeline tests with synthetic data.

## Current State of the Viewer

### What works
- Desktop: 3D orbit controls (drag to rotate, scroll to zoom).
- AR mode (Android Chrome over HTTPS): WebXR hit-test places model on a flat surface. Tap to place.
- After placement: pinch-to-scale, two-finger rotate, two-finger drag (touch gestures on screen).
- The aorta mesh is generated directly from the patient's mask — it's the real anatomy, not a generic model.
- Branch markers (ostium spheres, seed points, direction arrows, radius rings) are coded but currently stripped from the viewer. They'll be re-added when the detection pipeline produces real predictions.

### What needs to happen next
1. **Hand/body gesture control**: We want a person to be able to point the camera at another person, and that person's body gestures control the model (arms apart = scale up, arms together = scale down, lean/twist = rotate, etc.).
   - The plan is to build a **custom AR-like view** (not WebXR) that uses `getUserMedia` for camera feed as background, Three.js model on top, and **MediaPipe Pose Landmarker** (full body, 33 landmarks) for gesture detection.
   - WebXR's `camera-access` feature was considered but rejected — it's experimental, behind Chrome flags, and running MediaPipe + WebXR simultaneously causes jank.
   - This should be a separate mode from the WebXR AR mode, toggled by a button. Both modes should exist:
     - **AR mode**: WebXR, surface placement, touch gestures (current implementation).
     - **Gesture mode**: Custom camera view, MediaPipe Pose, body gesture control.
   - MediaPipe Pose is the right model (not Hands) because we need full body detection of a person at a distance, not close-up hand tracking.
   - This has NOT been built yet. It's the next task.

2. **Re-add branch markers**: Once the teammate's detection pipeline (sessions 3-5) produces real predictions, re-add the branch marker code to app.js. The export pipeline already handles this — just need the viewer to render them. The old code for markers, info panel, raycasting, and legend was removed (check git history on webxr-viz branch for reference).

3. **Visual QC for submission**: The challenge requires visual checks for at least 3 cases showing the aorta mask, detected ostia, and daughter-direction arrows. The export + viewer pipeline can produce these once real predictions exist.

## Key Technical Decisions

- **Coordinates**: Everything internal is (z,y,x) numpy index space. Physical mm coordinates ONLY via SimpleITK's TransformIndexToPhysicalPoint. Never use nibabel's affine (sign flip on x/y vs SimpleITK).
- **Mesh export**: Marching cubes on the binary mask, 30 iterations of Laplacian smoothing to remove voxel staircase, fallback decimation if fast-simplification isn't installed. Mesh is centered at its centroid; branch coordinates in scene.json are offset by the same centroid.
- **HTTPS for AR**: WebXR requires HTTPS except on localhost. The serve.py auto-generates a self-signed cert. On Android, user must accept the certificate warning.
- **No dataset in git**: The TORALIS CHALLENGE folder is gitignored (`TORALIS CHALLENGE*/`). It was accidentally committed once and purged via filter-branch.

## Dependencies
```
numpy, scipy, pytest, SimpleITK, nibabel, matplotlib  # detection pipeline
trimesh, scikit-image                                   # viz export
```
Three.js loaded from CDN (v0.164.1) in the viewer HTML.

## Data Location
`TORALIS CHALLENGE -20260912T162830Z-1-001/TORALIS CHALLENGE/subjectNNN/` — 25 subjects, each with `origN.nii` (CT) and `maskN.nii` (aorta mask).

## UI Style
Keep it minimal and clean. No glowing borders, no neon colors, no vibeslop. Muted colors, simple text, looks like a medical tool not a gaming UI.
