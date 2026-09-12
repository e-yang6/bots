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

Open http://localhost:8080/?scene=scenes/scene.json

Click and drag to rotate, scroll to zoom, click a marker for details.

## View on Android (AR)

1. Make sure your phone and laptop are on the **same Wi-Fi network**
2. Start the server (same command above) — it prints your local IP
3. On your Android phone, open **Chrome** and go to `http://<your-ip>:8080/?scene=scenes/scene.json`
4. Tap **Start AR**
5. Point your camera at a flat surface (table, desk, floor)
6. Tap to place the model
7. Walk around it, tap branch markers for info

If it doesn't connect, allow Python through Windows Firewall when prompted.

## What the markers mean

- **Red translucent** — aorta surface
- **Blue spheres** — ostium (where a branch leaves the aorta)
- **Dark gold spheres** — seed point (5mm along the branch)
- **Green arrows** — branch direction
- **Gold rings** — estimated vessel radius at the seed
