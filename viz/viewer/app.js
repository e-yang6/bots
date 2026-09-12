/**
 * Viewer for aorta visualization. Three modes:
 *   desktop  - orbit controls
 *   AR       - WebXR hit-test placement, touch gestures
 *   gesture  - camera feed background, body gestures via MediaPipe Pose
 *
 * Loads a .glb aorta mesh. Branch markers will be added later
 * when the detection pipeline produces real predictions.
 */

import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { ARButton } from 'three/addons/webxr/ARButton.js';
import { GestureMode } from './gesture-mode.js';

let scene, camera, renderer, controls;
let modelGroup;
let reticle;
let hitTestSource = null;
let hitTestSourceRequested = false;
let modelPlaced = false;
let isARActive = false;

// Pinch-to-scale and drag-to-rotate state
let touches = {};
let prevPinchDist = 0;
let prevTouchAngle = 0;
let prevTouchCenter = { x: 0, y: 0 };
let modelScale = 0.001; // base scale (mm to meters)

// Gesture mode state
let gestureMode;
let isGestureActive = false;
let aortaMaterial = null;
const MODEL_OPACITY = 0.35;
const GESTURE_MODEL_OPACITY = 0.6; // more solid over a busy camera image

function init() {
    scene = new THREE.Scene();

    camera = new THREE.PerspectiveCamera(70, window.innerWidth / window.innerHeight, 0.01, 100);
    camera.position.set(0, 0.15, 0.3);

    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.xr.enabled = true;
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    document.getElementById('viewport').appendChild(renderer.domElement);

    scene.add(new THREE.AmbientLight(0xffffff, 0.6));
    const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
    dirLight.position.set(0.5, 1, 0.5);
    scene.add(dirLight);

    modelGroup = new THREE.Group();
    modelGroup.scale.setScalar(0.001); // mm to meters
    scene.add(modelGroup);

    // AR reticle
    const reticleGeo = new THREE.RingGeometry(0.03, 0.04, 32);
    reticleGeo.rotateX(-Math.PI / 2);
    reticle = new THREE.Mesh(reticleGeo, new THREE.MeshBasicMaterial({ color: 0x999999 }));
    reticle.matrixAutoUpdate = false;
    reticle.visible = false;
    scene.add(reticle);

    // Orbit controls for desktop
    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.target.set(0, 0, 0);

    // AR button
    if ('xr' in navigator) {
        const arButton = ARButton.createButton(renderer, {
            requiredFeatures: ['hit-test'],
            optionalFeatures: ['dom-overlay'],
            domOverlay: { root: document.getElementById('overlay') },
        });
        document.getElementById('ar-button-container').appendChild(arButton);

        renderer.xr.addEventListener('sessionstart', () => {
            isARActive = true;
            modelPlaced = false;
            modelGroup.visible = false;
            controls.enabled = false;
            document.getElementById('gesture-button').hidden = true;
        });
        renderer.xr.addEventListener('sessionend', () => {
            isARActive = false;
            // Request a fresh hit-test source if AR is started again
            hitTestSource = null;
            hitTestSourceRequested = false;
            modelGroup.visible = true;
            modelGroup.position.set(0, 0, 0);
            modelGroup.quaternion.identity();
            controls.enabled = true;
            document.getElementById('gesture-button').hidden = false;
        });
    }

    setupGestureMode();

    // AR taps come through as 'select' on the controller, not DOM pointerdown
    renderer.xr.addEventListener('sessionstart', () => {
        const session = renderer.xr.getSession();
        session.addEventListener('select', onARSelect);
    });
    renderer.domElement.addEventListener('pointerdown', onPointerDown);
    renderer.domElement.addEventListener('touchstart', onTouchStart, { passive: false });
    renderer.domElement.addEventListener('touchmove', onTouchMove, { passive: false });
    renderer.domElement.addEventListener('touchend', onTouchEnd);
    window.addEventListener('resize', onWindowResize);

    loadScene('scene.json');
    renderer.setAnimationLoop(animate);
}

async function loadScene(sceneJsonUrl) {
    const statusEl = document.getElementById('status');
    statusEl.textContent = 'Loading...';

    try {
        const resp = await fetch(sceneJsonUrl);
        const sceneData = await resp.json();

        const loader = new GLTFLoader();
        const gltf = await loader.loadAsync(sceneData.mesh_file);

        const aortaMesh = gltf.scene;
        aortaMaterial = new THREE.MeshStandardMaterial({
            color: 0xb85555,
            transparent: true,
            opacity: isGestureActive ? GESTURE_MODEL_OPACITY : MODEL_OPACITY,
            roughness: 0.7,
            metalness: 0.0,
            side: THREE.DoubleSide,
            depthWrite: false,
        });
        aortaMesh.traverse((child) => {
            if (child.isMesh) child.material = aortaMaterial;
        });
        modelGroup.add(aortaMesh);

        document.getElementById('case-id').textContent = sceneData.case_id;
        if (statusEl.textContent === 'Loading...') statusEl.textContent = '';

        if (isGestureActive) {
            placeModelForGesture();
        } else if (!isARActive) {
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
    const distance = maxDim * 1.5;

    camera.position.set(center.x + distance * 0.5, center.y + distance * 0.3, center.z + distance);
    controls.target.copy(center);
    controls.update();
}

// ─── Gesture mode (camera feed + MediaPipe Pose) ────────────────────────

function setupGestureMode() {
    const statusEl = document.getElementById('status');
    const modeButtons = document.getElementById('mode-buttons');
    const gestureControls = document.getElementById('gesture-controls');
    const skeletonButton = document.getElementById('skeleton-button');
    const flipButton = document.getElementById('flip-camera-button');

    gestureMode = new GestureMode({
        video: document.getElementById('camera-feed'),
        skeleton: document.getElementById('skeleton'),
        onStatus: (text) => { statusEl.textContent = text; },
    });
    gestureMode.resizeSkeleton();

    document.getElementById('gesture-button').addEventListener('click', async () => {
        if (isARActive || isGestureActive) return;
        enterGestureMode();
        modeButtons.hidden = true;
        gestureControls.hidden = false;
        await gestureMode.start();
    });

    document.getElementById('exit-gesture-button').addEventListener('click', () => {
        exitGestureMode();
        gestureControls.hidden = true;
        modeButtons.hidden = false;
    });

    flipButton.addEventListener('click', async () => {
        flipButton.disabled = true;
        await gestureMode.flipCamera();
        flipButton.disabled = false;
    });

    skeletonButton.addEventListener('click', () => {
        const show = !gestureMode.showSkeleton;
        gestureMode.setSkeletonVisible(show);
        skeletonButton.setAttribute('aria-pressed', String(show));
    });
}

function enterGestureMode() {
    isGestureActive = true;
    controls.enabled = false;
    if (aortaMaterial) aortaMaterial.opacity = GESTURE_MODEL_OPACITY;
    // Camera at the origin looking down -z; the model floats in front of it
    camera.position.set(0, 0, 0);
    camera.quaternion.identity();
    // Yaw about the model's vertical axis first, then tilt about the view axis
    modelGroup.rotation.order = 'ZYX';
    placeModelForGesture();
}

function exitGestureMode() {
    isGestureActive = false;
    gestureMode.stop();
    if (aortaMaterial) aortaMaterial.opacity = MODEL_OPACITY;
    modelGroup.rotation.set(0, 0, 0, 'XYZ');
    modelGroup.position.set(0, 0, 0);
    modelGroup.scale.setScalar(modelScale);
    modelGroup.visible = true;
    controls.enabled = true;
    fitCameraToModel();
}

/** Put the model in front of the camera, at a distance where it fits the view. */
function placeModelForGesture() {
    modelGroup.position.set(0, 0, 0);
    modelGroup.rotation.set(0, 0, 0);
    modelGroup.scale.setScalar(modelScale);
    modelGroup.visible = true;
    modelGroup.updateMatrixWorld(true);

    const sphere = new THREE.Box3().setFromObject(modelGroup).getBoundingSphere(new THREE.Sphere());
    const radius = sphere.radius > 0 ? sphere.radius : 0.15;
    const halfV = THREE.MathUtils.degToRad(camera.fov) / 2;
    const halfH = Math.atan(Math.tan(halfV) * camera.aspect);
    const distance = (radius / Math.sin(Math.min(halfV, halfH))) * 1.15;

    modelGroup.position.set(-sphere.center.x, -sphere.center.y, -distance - sphere.center.z);
    applyGestureTransform(gestureMode.controller.transform);
}

function applyGestureTransform(t) {
    modelGroup.scale.setScalar(modelScale * t.scale);
    modelGroup.rotation.set(0, t.yaw, t.tilt);
}

// ─── AR placement ───────────────────────────────────────────────────────

function placeModel() {
    if (!modelPlaced && reticle.visible) {
        modelGroup.position.setFromMatrixPosition(reticle.matrix);
        modelGroup.visible = true;
        modelPlaced = true;
        reticle.visible = false;
        document.getElementById('status').textContent = '';
    }
}

function onARSelect() {
    if (isARActive) placeModel();
}

function onPointerDown() {
    if (isARActive) placeModel();
}

// ─── Touch gestures (pinch to scale, two-finger rotate) ─────────────────

function getTouchDist(t1, t2) {
    const dx = t1.clientX - t2.clientX;
    const dy = t1.clientY - t2.clientY;
    return Math.sqrt(dx * dx + dy * dy);
}

function getTouchAngle(t1, t2) {
    return Math.atan2(t1.clientY - t2.clientY, t1.clientX - t2.clientX);
}

function getTouchCenter(t1, t2) {
    return { x: (t1.clientX + t2.clientX) / 2, y: (t1.clientY + t2.clientY) / 2 };
}

function onTouchStart(e) {
    if (!modelPlaced) return;
    for (const t of e.changedTouches) {
        touches[t.identifier] = t;
    }
    const ids = Object.keys(touches);
    if (ids.length === 2) {
        e.preventDefault();
        const t1 = touches[ids[0]], t2 = touches[ids[1]];
        prevPinchDist = getTouchDist(t1, t2);
        prevTouchAngle = getTouchAngle(t1, t2);
        prevTouchCenter = getTouchCenter(t1, t2);
    }
}

function onTouchMove(e) {
    if (!modelPlaced) return;
    for (const t of e.changedTouches) {
        touches[t.identifier] = t;
    }
    const ids = Object.keys(touches);
    if (ids.length === 2) {
        e.preventDefault();
        const t1 = touches[ids[0]], t2 = touches[ids[1]];

        // Pinch to scale
        const dist = getTouchDist(t1, t2);
        if (prevPinchDist > 0) {
            const scaleFactor = dist / prevPinchDist;
            modelScale *= scaleFactor;
            modelScale = Math.max(0.0002, Math.min(0.01, modelScale));
            modelGroup.scale.setScalar(modelScale);
        }
        prevPinchDist = dist;

        // Two-finger rotate (around Y axis)
        const angle = getTouchAngle(t1, t2);
        const deltaAngle = angle - prevTouchAngle;
        modelGroup.rotation.y += deltaAngle;
        prevTouchAngle = angle;

        // Two-finger drag to move
        const center = getTouchCenter(t1, t2);
        if (isARActive) {
            const dx = (center.x - prevTouchCenter.x) * 0.0005;
            const dy = (center.y - prevTouchCenter.y) * -0.0005;
            modelGroup.position.x += dx;
            modelGroup.position.z += dy;
        }
        prevTouchCenter = center;
    }
}

function onTouchEnd(e) {
    for (const t of e.changedTouches) {
        delete touches[t.identifier];
    }
    if (Object.keys(touches).length < 2) {
        prevPinchDist = 0;
    }
}

function onXRFrame(timestamp, frame) {
    if (!isARActive || modelPlaced) return;

    const session = renderer.xr.getSession();
    const refSpace = renderer.xr.getReferenceSpace();

    if (!hitTestSourceRequested) {
        session.requestReferenceSpace('viewer').then((viewerSpace) => {
            session.requestHitTestSource({ space: viewerSpace }).then((source) => {
                hitTestSource = source;
            });
        });
        hitTestSourceRequested = true;
    }

    if (hitTestSource) {
        const results = frame.getHitTestResults(hitTestSource);
        if (results.length > 0) {
            const hit = results[0];
            const pose = hit.getPose(refSpace);
            reticle.visible = true;
            reticle.matrix.fromArray(pose.transform.matrix);
        } else {
            reticle.visible = false;
        }
    }
}

function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
    if (gestureMode) gestureMode.resizeSkeleton();
    if (isGestureActive) placeModelForGesture();
}

function animate(timestamp, frame) {
    if (frame) onXRFrame(timestamp, frame);
    if (isGestureActive) {
        const t = gestureMode.tick();
        if (t) applyGestureTransform(t);
    } else if (!isARActive) {
        controls.update();
    }
    renderer.render(scene, camera);
}

init();
