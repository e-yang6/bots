/**
 * WebXR AR viewer for aorta visualization with gesture control.
 *
 * AR mode: place model on surface via hit-test, then control with
 * body gestures (MediaPipe Pose) or touch fallback.
 */

import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { ARButton } from 'three/addons/webxr/ARButton.js';
import { initPoseDetection, processXRFrame, processVideoFrame, applyGestures, getGestureState } from './gesture.js';

let scene, camera, renderer, controls;
let modelGroup;
let reticle;
let hitTestSource = null;
let hitTestSourceRequested = false;
let modelPlaced = false;
let isARActive = false;

// Gesture control
let gestureReady = false;
let cameraAccessAvailable = false;
let fallbackVideo = null;
let baseScale = 0.001;

// Touch fallback state
let touches = {};
let prevPinchDist = 0;
let prevTouchAngle = 0;
let prevTouchCenter = { x: 0, y: 0 };
let modelScale = 0.001;

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
    modelGroup.scale.setScalar(baseScale);
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

    // AR button — request camera-access as optional so it works even if unsupported
    if ('xr' in navigator) {
        const arButton = ARButton.createButton(renderer, {
            requiredFeatures: ['hit-test'],
            optionalFeatures: ['dom-overlay', 'camera-access'],
            domOverlay: { root: document.getElementById('overlay') },
        });
        document.getElementById('ar-button-container').appendChild(arButton);

        renderer.xr.addEventListener('sessionstart', () => {
            isARActive = true;
            modelPlaced = false;
            modelGroup.visible = false;
            controls.enabled = false;
            startGestureDetection();
        });
        renderer.xr.addEventListener('sessionend', () => {
            isARActive = false;
            modelGroup.visible = true;
            modelGroup.position.set(0, 0, 0);
            modelGroup.quaternion.identity();
            modelGroup.scale.setScalar(baseScale);
            controls.enabled = true;
            stopFallbackVideo();
        });
    }

    // AR tap to place
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

// ─── Gesture detection setup ────────────────────────────────────────────────

async function startGestureDetection() {
    if (gestureReady) return;

    const statusEl = document.getElementById('status');
    statusEl.textContent = 'Loading gesture detection...';

    try {
        await initPoseDetection();
        gestureReady = true;
        statusEl.textContent = '';
    } catch (err) {
        console.warn('Pose detection init failed:', err);
        statusEl.textContent = '';
    }
}

function startFallbackVideo() {
    // If XR camera-access isn't available, open a separate camera stream
    if (fallbackVideo) return;

    navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment', width: 320, height: 240 }
    }).then(stream => {
        fallbackVideo = document.createElement('video');
        fallbackVideo.srcObject = stream;
        fallbackVideo.setAttribute('playsinline', '');
        fallbackVideo.play();
        console.log('Fallback camera started for gesture detection');
    }).catch(err => {
        console.warn('Could not open fallback camera:', err);
    });
}

function stopFallbackVideo() {
    if (fallbackVideo && fallbackVideo.srcObject) {
        fallbackVideo.srcObject.getTracks().forEach(t => t.stop());
        fallbackVideo = null;
    }
}

// ─── Scene loading ──────────────────────────────────────────────────────────

async function loadScene(sceneJsonUrl) {
    const statusEl = document.getElementById('status');
    statusEl.textContent = 'Loading...';

    try {
        const resp = await fetch(sceneJsonUrl);
        const sceneData = await resp.json();

        const loader = new GLTFLoader();
        const gltf = await loader.loadAsync(sceneData.mesh_file);

        const aortaMesh = gltf.scene;
        aortaMesh.traverse((child) => {
            if (child.isMesh) {
                child.material = new THREE.MeshStandardMaterial({
                    color: 0xb85555,
                    transparent: true,
                    opacity: 0.35,
                    roughness: 0.7,
                    metalness: 0.0,
                    side: THREE.DoubleSide,
                    depthWrite: false,
                });
            }
        });
        modelGroup.add(aortaMesh);

        document.getElementById('case-id').textContent = sceneData.case_id;
        statusEl.textContent = '';

        if (!isARActive) {
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

// ─── Placement ──────────────────────────────────────────────────────────────

function placeModel() {
    if (!modelPlaced && reticle.visible) {
        modelGroup.position.setFromMatrixPosition(reticle.matrix);
        modelGroup.visible = true;
        modelPlaced = true;
        reticle.visible = false;

        const state = getGestureState();
        if (gestureReady) {
            document.getElementById('status').textContent = state.personDetected
                ? 'Gesture control active'
                : 'Point camera at a person for gesture control';
        } else {
            document.getElementById('status').textContent = '';
        }
    }
}

function onARSelect() {
    if (isARActive) placeModel();
}

function onPointerDown() {
    if (isARActive) placeModel();
}

// ─── Touch gestures (fallback) ──────────────────────────────────────────────

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
    for (const t of e.changedTouches) touches[t.identifier] = t;
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
    for (const t of e.changedTouches) touches[t.identifier] = t;
    const ids = Object.keys(touches);
    if (ids.length === 2) {
        e.preventDefault();
        const t1 = touches[ids[0]], t2 = touches[ids[1]];

        const dist = getTouchDist(t1, t2);
        if (prevPinchDist > 0) {
            const scaleFactor = dist / prevPinchDist;
            modelScale *= scaleFactor;
            modelScale = Math.max(0.0002, Math.min(0.01, modelScale));
            modelGroup.scale.setScalar(modelScale);
        }
        prevPinchDist = dist;

        const angle = getTouchAngle(t1, t2);
        modelGroup.rotation.y += angle - prevTouchAngle;
        prevTouchAngle = angle;

        const center = getTouchCenter(t1, t2);
        if (isARActive) {
            modelGroup.position.x += (center.x - prevTouchCenter.x) * 0.0005;
            modelGroup.position.z += (center.y - prevTouchCenter.y) * -0.0005;
        }
        prevTouchCenter = center;
    }
}

function onTouchEnd(e) {
    for (const t of e.changedTouches) delete touches[t.identifier];
    if (Object.keys(touches).length < 2) prevPinchDist = 0;
}

// ─── XR frame handling ─────────────────────────────────────────────────────

function onXRFrame(timestamp, frame) {
    if (!isARActive) return;

    // Hit-test for placement
    if (!modelPlaced) {
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
        return;
    }

    // Gesture detection on placed model
    if (gestureReady) {
        // Try XR camera-access first
        let usedXRCamera = false;
        try {
            processXRFrame(renderer, frame);
            const state = getGestureState();
            if (state.personDetected) usedXRCamera = true;
        } catch (e) {
            // camera-access not available
        }

        // If XR camera didn't work, try fallback video
        if (!usedXRCamera && !cameraAccessAvailable) {
            if (!fallbackVideo) startFallbackVideo();
            if (fallbackVideo && fallbackVideo.readyState >= 2) {
                processVideoFrame(fallbackVideo);
            }
        } else {
            cameraAccessAvailable = true;
        }

        // Apply detected gestures to the model
        const state = getGestureState();
        if (state.personDetected) {
            applyGestures(modelGroup, baseScale);
            document.getElementById('status').textContent = 'Gesture control active';
        }
    }
}

// ─── Resize + render ────────────────────────────────────────────────────────

function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
}

function animate(timestamp, frame) {
    if (frame) onXRFrame(timestamp, frame);
    if (!isARActive) controls.update();
    renderer.render(scene, camera);
}

init();
