# Aorta Branch Viewer (WebXR)

Interactive 3D/AR viewer for aorta branch detection results. View on desktop or place the model on a table using AR on Android.

## Setup

```bash
pip install trimesh scikit-image SimpleITK
```

## Export a case

```bash
python -m viz.export_scene \
  --image path/to/orig1.nii \
  --aorta-mask path/to/mask1.nii \
  --prediction path/to/prediction.json \
  --output-dir viz/output/subject001
```

## View on desktop

```bash
python -m viz.serve --port 8080 --scene-dir viz/output/subject001
```

Open http://localhost:8080

Click and drag to rotate, scroll to zoom, click a marker for details.

## View on Android (AR)

1. Make sure your phone and laptop are on the **same Wi-Fi network**
2. Start the server (same command above) — it prints your local IP
3. On your Android phone, open **Chrome** and go to `http://<your-ip>:8080`
4. Tap **Start AR**
5. Point your camera at a flat surface (table, desk, floor)
6. Tap to place the model
7. Walk around it, tap branch markers for info

If it doesn't connect, allow Python through Windows Firewall when prompted.

## Gesture mode (body gestures)

Point the camera at a person; their body movements control the model. Works on Android Chrome and on a desktop webcam. No WebXR needed.

1. Open the viewer (phone: `https://<your-ip>:8080`, accept the certificate warning; desktop: `http://localhost:8080`)
2. Tap **Gesture mode** and allow camera access
3. The person stands in view with their upper body visible (about 2–3 m away)

| Gesture | Effect |
|---|---|
| Raise both hands above the hips | Start controlling (the current pose becomes the baseline) |
| Move arms apart / together | Scale up / down |
| Twist the torso | Spin the model |
| Lean left / right | Tilt the model |
| Lower the hands | Stop controlling; the model stays as it is |
| Both hands above the head for 1.5 s | Reset size and rotation |

**Flip camera** switches between rear and front cameras (the front view is mirrored). **Skeleton** draws the detected pose over the video, which helps when tuning.

Notes:
- The camera needs a secure context: HTTPS, or `localhost` on desktop.
- The pose model (MediaPipe Pose Landmarker lite) downloads from Google's CDN the first time gesture mode starts, so the phone needs internet access.
- Gesture thresholds and gains are in `DEFAULT_OPTIONS` in `viewer/pose-gestures.js`. Unit tests: `node --test tests/test_pose_gestures.mjs`.

## What the markers mean

- **Red translucent** — aorta surface
- **Blue spheres** — ostium (where a branch leaves the aorta)
- **Dark gold spheres** — seed point (5mm along the branch)
- **Green arrows** — branch direction
- **Gold rings** — estimated vessel radius at the seed
