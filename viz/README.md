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

## Gesture mode (person-anchored AR)

Point the camera at a person. Their aorta is drawn in their chest at life size and stays attached to them like a real object: pan the phone, walk closer, or walk around them to see it from the side or back. Their hands control the scale. Works on Android Chrome and on a desktop webcam. No WebXR needed.

1. Open the viewer (phone: `https://<your-ip>:8080`, accept the certificate warning; desktop: `http://localhost:8080`)
2. Tap **Gesture mode** and allow camera access
3. The person stands in view, about 2–3 m away, with shoulders and ideally hips visible

| Movement | Effect |
|---|---|
| Walk around the person, or they turn | See the aorta from that side |
| Person leans, or the phone tilts | Model tilts with the torso |
| Move closer / further | Model grows / shrinks with the person |
| Raise both hands above the hips, move arms apart / together | Scale up / down |
| Lower the hands | Keep the current scale |
| Both hands above the head for 1.5 s | Reset to life size |
| Two-finger rotate / pinch on screen | Extra spin / scale |

**Flip camera** switches between rear and front cameras (the front view is mirrored, and so is the anatomy). **Skeleton** draws the detected pose over the video, which helps when tuning.

Notes:
- The camera needs a secure context: HTTPS, or `localhost` on desktop.
- The pose model (MediaPipe Pose Landmarker lite) downloads from Google's CDN the first time gesture mode starts, so the phone needs internet access.
- Life size assumes average adult proportions (shoulder-to-hip 0.50 m). Looking from well above or below the person is not tracked, and pose detection gets less reliable in pure side and back views.
- Tunables are in `DEFAULT_OPTIONS` in `viewer/pose-gestures.js` (body proportions, anchor height, smoothing, gesture thresholds). If the model turns the opposite way to the person, set `invertHeading: true`. Unit tests: `node --test tests/test_pose_gestures.mjs`.

## What the markers mean

- **Red translucent** — aorta surface
- **Blue spheres** — ostium (where a branch leaves the aorta)
- **Dark gold spheres** — seed point (5mm along the branch)
- **Green arrows** — branch direction
- **Gold rings** — estimated vessel radius at the seed
