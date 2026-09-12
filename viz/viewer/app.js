/**
 * Viewer for aorta visualization. Three modes:
 *   desktop  - orbit controls
 *   AR       - AR.js Hiro marker tracking, camera background
 *   gesture  - camera feed + MediaPipe Pose body anchor
 * AR and gesture can run simultaneously: marker positions, body gestures scale.
 */

import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GestureMode } from './gesture-mode.js';

// AR.js classic build needs a global THREE
window.THREE = THREE;

let scene, camera, renderer, controls;
let modelGroup, anatomyGroup;
let aortaMaterial = null;

// AR.js state — we manage the camera ourselves (not ArToolkitSource)
// so the #camera-feed video works as the background, same as gesture mode.
let arLoaded = false;
let arActive = false;
let arStream = null;          // MediaStream for AR camera
let arToolkitContext = null;
let arCamera = null;
let markerGroup = null;
let arVideoEl = null;         // reference to #camera-feed when in AR

// Touch state
let touches = {};
let prevPinchDist = 0;
let prevTouchAngle = 0;
let prevTouchCenter = { x: 0, y: 0 };
let modelScale = 0.001; // mm → m

// Gesture mode state
let gestureMode;
let isGestureActive = false;
let lastGesture = null;
let touchYaw = 0;
const MODEL_OPACITY = 0.35;
const GESTURE_MODEL_OPACITY = 0.6;
const GESTURE_DEPTH_M = 1.5;

const ARJS_URL = 'https://raw.githack.com/AR-js-org/AR.js/master/three.js/build/ar-threex.js';
const CAMERA_PARAM_URL = 'https://raw.githack.com/AR-js-org/AR.js/master/data/data/camera_para.dat';
const HIRO_PATTERN_URL = 'https://raw.githack.com/AR-js-org/AR.js/master/data/data/patt.hiro';

function loadScript(url) {
    return new Promise((resolve, reject) => {
        const s = document.createElement('script');
        s.src = url;
        s.onload = resolve;
        s.onerror = reject;
        document.head.appendChild(s);
    });
}

async function init() {
    scene = new THREE.Scene();

    camera = new THREE.PerspectiveCamera(70, window.innerWidth / window.innerHeight, 0.01, 100);
    camera.position.set(0, 0.15, 0.3);

    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setClearColor(0x000000, 0);
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    document.getElementById('viewport').appendChild(renderer.domElement);

    scene.add(new THREE.AmbientLight(0xffffff, 0.6));
    const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
    dirLight.position.set(0.5, 1, 0.5);
    scene.add(dirLight);

    modelGroup = new THREE.Group();
    modelGroup.scale.setScalar(modelScale);
    scene.add(modelGroup);
    anatomyGroup = new THREE.Group();
    modelGroup.add(anatomyGroup);

    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.target.set(0, 0, 0);

    // Load AR.js — button shown only on success
    try {
        await loadScript(ARJS_URL);
        arLoaded = true;
        document.getElementById('ar-button').hidden = false;
    } catch (err) {
        console.warn('AR.js failed to load:', err);
    }

    setupGestureMode();
    setupButtons();

    renderer.domElement.addEventListener('touchstart', onTouchStart, { passive: false });
    renderer.domElement.addEventListener('touchmove', onTouchMove, { passive: false });
    renderer.domElement.addEventListener('touchend', onTouchEnd);
    window.addEventListener('resize', onWindowResize);

    loadScene('scene.json');
    requestAnimationFrame(animate);
}

// ── Scene loading ────────────────────────────────────────────────────────

async function loadScene(url) {
    const statusEl = document.getElementById('status');
    statusEl.textContent = 'Loading...';
    try {
        const resp = await fetch(url);
        const data = await resp.json();
        const gltf = await new GLTFLoader().loadAsync(data.mesh_file);

        aortaMaterial = new THREE.MeshStandardMaterial({
            color: 0xb85555,
            transparent: true,
            opacity: isGestureActive ? GESTURE_MODEL_OPACITY : MODEL_OPACITY,
            roughness: 0.7, metalness: 0.0,
            side: THREE.DoubleSide, depthWrite: false,
        });
        gltf.scene.traverse(c => { if (c.isMesh) c.material = aortaMaterial; });
        anatomyGroup.add(gltf.scene);

        document.getElementById('case-id').textContent = data.case_id;
        if (statusEl.textContent === 'Loading...') statusEl.textContent = '';

        if (!isGestureActive && !arActive) {
            modelGroup.visible = true;
            fitCameraToModel();
        }
    } catch (err) {
        statusEl.textContent = 'Error: ' + err.message;
        console.error(err);
    }
}

function fitCameraToModel() {
    const box = new THREE.Box3().setFromObject(modelGroup);
    const size = box.getSize(new THREE.Vector3());
    const center = box.getCenter(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z);
    const dist = maxDim * 1.5;
    camera.position.set(center.x + dist * 0.5, center.y + dist * 0.3, center.z + dist);
    controls.target.copy(center);
    controls.update();
}

// ── AR.js (Hiro marker) ─────────────────────────────────────────────────
// We open the camera ourselves onto #camera-feed (which already has the
// right CSS from gesture mode) and feed it to ArToolkitContext for marker
// detection. This avoids ArToolkitSource's DOM/z-index issues.

async function enterARMode() {
    if (!arLoaded || arActive) return;
    const statusEl = document.getElementById('status');
    statusEl.textContent = 'Starting AR...';

    try {
        // Open camera onto #camera-feed (same element gesture mode uses)
        arVideoEl = document.getElementById('camera-feed');
        arStream = await navigator.mediaDevices.getUserMedia({
            audio: false,
            video: { facingMode: 'environment', width: { ideal: 640 }, height: { ideal: 480 } },
        });
        arVideoEl.srcObject = arStream;
        arVideoEl.hidden = false;
        await arVideoEl.play();

        // AR camera — projection set by AR.js calibration data
        arCamera = new THREE.Camera();
        scene.add(arCamera);

        // Marker root — AR.js toggles .visible on detection/loss
        markerGroup = new THREE.Group();
        markerGroup.visible = false;
        scene.add(markerGroup);

        // Re-parent model under marker
        scene.remove(modelGroup);
        markerGroup.add(modelGroup);
        modelGroup.visible = true;

        // Rotate anatomy so superior (LPS z) points up (scene y)
        anatomyGroup.rotation.set(-Math.PI / 2, 0, 0);

        // ArToolkitContext — marker detection engine
        arToolkitContext = new THREEx.ArToolkitContext({
            cameraParametersUrl: CAMERA_PARAM_URL,
            detectionMode: 'mono',
        });
        await new Promise((resolve) => arToolkitContext.init(() => resolve()));
        arCamera.projectionMatrix.copy(arToolkitContext.getProjectionMatrix());

        // Hiro marker controls
        new THREEx.ArMarkerControls(arToolkitContext, markerGroup, {
            type: 'pattern',
            patternUrl: HIRO_PATTERN_URL,
        });

        arActive = true;
        controls.enabled = false;
        statusEl.textContent = 'Point at Hiro marker';

        // UI
        document.getElementById('mode-buttons').hidden = true;
        document.getElementById('ar-controls').hidden = false;
    } catch (err) {
        console.error('AR init failed:', err);
        statusEl.textContent = 'AR failed: ' + err.message;
        cleanupAR();
    }
}

function exitARMode() {
    if (isGestureActive) exitGestureInAR();
    arActive = false;
    cleanupAR();

    // Restore desktop state
    anatomyGroup.rotation.set(0, 0, 0);
    modelGroup.position.set(0, 0, 0);
    modelGroup.quaternion.identity();
    modelGroup.scale.setScalar(modelScale);
    modelGroup.visible = true;
    controls.enabled = true;
    fitCameraToModel();

    document.getElementById('ar-controls').hidden = true;
    document.getElementById('mode-buttons').hidden = false;
    document.getElementById('status').textContent = '';
}

/** Tear down AR.js objects and re-parent model to scene root. */
function cleanupAR() {
    if (markerGroup) {
        markerGroup.remove(modelGroup);
        scene.remove(markerGroup);
        markerGroup = null;
    }
    scene.add(modelGroup);
    if (arCamera) { scene.remove(arCamera); arCamera = null; }

    // Stop our camera stream and hide the video
    if (arStream) {
        arStream.getTracks().forEach(t => t.stop());
        arStream = null;
    }
    if (arVideoEl) {
        arVideoEl.pause();
        arVideoEl.srcObject = null;
        arVideoEl.hidden = true;
        arVideoEl = null;
    }
    arToolkitContext = null;
}

// ── Gesture mode ─────────────────────────────────────────────────────────

function setupGestureMode() {
    gestureMode = new GestureMode({
        video: document.getElementById('camera-feed'),
        skeleton: document.getElementById('skeleton'),
        onStatus: (t) => { document.getElementById('status').textContent = t; },
    });
    gestureMode.resizeSkeleton();
}

function setupButtons() {
    const modeButtons = document.getElementById('mode-buttons');
    const gestureControls = document.getElementById('gesture-controls');
    const flipBtn = document.getElementById('flip-camera-button');
    const skelBtn = document.getElementById('skeleton-button');

    // Desktop → standalone gesture
    document.getElementById('gesture-button').addEventListener('click', async () => {
        if (arActive || isGestureActive) return;
        enterGestureMode();
        modeButtons.hidden = true;
        gestureControls.hidden = false;
        flipBtn.hidden = false;
        await gestureMode.start();
    });

    // Exit gesture (works in both standalone and AR+gesture)
    document.getElementById('exit-gesture-button').addEventListener('click', () => {
        if (arActive) {
            exitGestureInAR();
        } else {
            exitGestureMode();
            gestureControls.hidden = true;
            modeButtons.hidden = false;
        }
    });

    // Flip camera (standalone gesture only)
    flipBtn.addEventListener('click', async () => {
        if (gestureMode._useExternalVideo) return;
        flipBtn.disabled = true;
        lastGesture = null;
        modelGroup.visible = false;
        await gestureMode.flipCamera();
        flipBtn.disabled = false;
    });

    // Skeleton toggle (shared by standalone and AR+gesture)
    skelBtn.addEventListener('click', () => {
        const show = !gestureMode.showSkeleton;
        gestureMode.setSkeletonVisible(show);
        skelBtn.setAttribute('aria-pressed', String(show));
    });

    // Desktop → AR
    document.getElementById('ar-button').addEventListener('click', () => enterARMode());

    // AR → desktop
    document.getElementById('exit-ar-button').addEventListener('click', () => exitARMode());

    // AR → AR+gesture
    document.getElementById('ar-gesture-button').addEventListener('click', () => enterGestureInAR());
}

function enterGestureMode() {
    isGestureActive = true;
    lastGesture = null;
    touchYaw = 0;
    controls.enabled = false;
    if (aortaMaterial) aortaMaterial.opacity = GESTURE_MODEL_OPACITY;
    camera.position.set(0, 0, 0);
    camera.quaternion.identity();
    camera.updateMatrixWorld();
    anatomyGroup.rotation.set(-Math.PI / 2, 0, 0);
    modelGroup.rotation.set(0, 0, 0, 'ZYX');
    modelGroup.visible = false;
}

function exitGestureMode() {
    isGestureActive = false;
    lastGesture = null;
    gestureMode.stop();
    if (aortaMaterial) aortaMaterial.opacity = MODEL_OPACITY;
    anatomyGroup.rotation.set(0, 0, 0);
    modelGroup.rotation.set(0, 0, 0, 'XYZ');
    modelGroup.position.set(0, 0, 0);
    modelGroup.scale.setScalar(modelScale);
    modelGroup.visible = true;
    controls.enabled = true;
    fitCameraToModel();
}

async function enterGestureInAR() {
    if (!arActive || isGestureActive) return;
    isGestureActive = true;
    lastGesture = null;
    touchYaw = 0;
    if (aortaMaterial) aortaMaterial.opacity = GESTURE_MODEL_OPACITY;

    document.getElementById('ar-controls').hidden = true;
    document.getElementById('gesture-controls').hidden = false;
    document.getElementById('flip-camera-button').hidden = true;

    // Share the AR camera video with gesture mode (no new getUserMedia)
    await gestureMode.start({ externalVideo: arVideoEl });
}

function exitGestureInAR() {
    isGestureActive = false;
    lastGesture = null;
    gestureMode.stop();
    if (aortaMaterial) aortaMaterial.opacity = MODEL_OPACITY;
    modelGroup.scale.setScalar(modelScale);

    document.getElementById('gesture-controls').hidden = true;
    document.getElementById('ar-controls').hidden = false;
}

/**
 * Standalone gesture: place the model in the detected person's chest at
 * life size, turned and tilted with their torso.
 */
function applyGesture({ anchor, transform }) {
    const screen = anchor.visible ? gestureMode.videoToScreen(anchor.x, anchor.y) : null;
    if (!screen) { modelGroup.visible = false; return; }

    const ndc = new THREE.Vector3(
        (screen.x / window.innerWidth) * 2 - 1,
        -(screen.y / window.innerHeight) * 2 + 1, 0.5,
    ).unproject(camera);
    const ray = ndc.sub(camera.position).normalize();
    modelGroup.position.copy(ray.multiplyScalar(GESTURE_DEPTH_M / -ray.z));

    const screenPxPerMeter = anchor.heightPerMeter * screen.videoHeightPx;
    const scenePerPx = (2 * GESTURE_DEPTH_M * Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2)) / window.innerHeight;
    const s = modelScale * screenPxPerMeter * scenePerPx * transform.scale;
    modelGroup.scale.set(gestureMode.mirrored ? -s : s, s, s);
    modelGroup.rotation.set(0, anchor.heading + touchYaw, anchor.roll, 'ZYX');
    modelGroup.visible = true;
}

// ── Touch gestures (pinch to scale, two-finger rotate) ───────────────────

function getTouchDist(a, b) { return Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY); }
function getTouchAngle(a, b) { return Math.atan2(a.clientY - b.clientY, a.clientX - b.clientX); }
function getTouchCenter(a, b) { return { x: (a.clientX + b.clientX) / 2, y: (a.clientY + b.clientY) / 2 }; }

function onTouchStart(e) {
    for (const t of e.changedTouches) touches[t.identifier] = t;
    const ids = Object.keys(touches);
    if (ids.length === 2) {
        e.preventDefault();
        const [a, b] = [touches[ids[0]], touches[ids[1]]];
        prevPinchDist = getTouchDist(a, b);
        prevTouchAngle = getTouchAngle(a, b);
        prevTouchCenter = getTouchCenter(a, b);
    }
}

function onTouchMove(e) {
    for (const t of e.changedTouches) touches[t.identifier] = t;
    const ids = Object.keys(touches);
    if (ids.length !== 2) return;
    e.preventDefault();
    const [a, b] = [touches[ids[0]], touches[ids[1]]];

    // Pinch to scale
    const dist = getTouchDist(a, b);
    if (prevPinchDist > 0) {
        modelScale *= dist / prevPinchDist;
        modelScale = Math.max(0.0002, Math.min(0.01, modelScale));
        if (!isGestureActive) modelGroup.scale.setScalar(modelScale);
    }
    prevPinchDist = dist;

    // Two-finger rotate
    const angle = getTouchAngle(a, b);
    const da = angle - prevTouchAngle;
    if (isGestureActive) touchYaw += da;
    else modelGroup.rotation.y += da;
    prevTouchAngle = angle;

    // Two-finger drag (fine-tune position in AR)
    const center = getTouchCenter(a, b);
    if (arActive && !isGestureActive) {
        modelGroup.position.x += (center.x - prevTouchCenter.x) * 0.0005;
        modelGroup.position.z += (center.y - prevTouchCenter.y) * -0.0005;
    }
    prevTouchCenter = center;
}

function onTouchEnd(e) {
    for (const t of e.changedTouches) delete touches[t.identifier];
    if (Object.keys(touches).length < 2) prevPinchDist = 0;
}

// ── Render loop ──────────────────────────────────────────────────────────

function onWindowResize() {
    if (!arActive) {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
    }
    renderer.setSize(window.innerWidth, window.innerHeight);
    if (gestureMode) gestureMode.resizeSkeleton();
}

function animate() {
    requestAnimationFrame(animate);

    // AR.js marker detection — feed our #camera-feed video to the context
    if (arActive && arVideoEl && arToolkitContext) {
        arToolkitContext.update(arVideoEl);
    }

    if (isGestureActive) {
        lastGesture = gestureMode.tick() ?? lastGesture;
        if (arActive) {
            // AR + gesture: marker positions, gesture controls scale
            if (lastGesture) {
                const s = modelScale * lastGesture.transform.scale;
                modelGroup.scale.setScalar(s);
            }
        } else {
            // Standalone gesture: full body anchor
            if (lastGesture) applyGesture(lastGesture);
        }
    } else if (!arActive) {
        controls.update();
    }

    renderer.render(scene, arActive ? arCamera : camera);
}

init();
