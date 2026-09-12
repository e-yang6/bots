/**
 * WebXR AR viewer for aorta visualization.
 *
 * AR mode: place model on a surface via hit-test, tap to place.
 *   After placement: pinch to scale, two-finger twist to rotate.
 * Desktop: orbit controls.
 */

import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { ARButton } from 'three/addons/webxr/ARButton.js';

let scene, camera, renderer, controls;
let modelGroup;
let reticle;
let hitTestSource = null;
let hitTestSourceRequested = false;
let modelPlaced = false;
let isARActive = false;

// Pinch/rotate gesture state (pointer-event based, works in WebXR dom-overlay)
const activePointers = new Map();
let prevPinchDist = null;
let prevPinchAngle = null;
const MIN_SCALE = 0.0002;
const MAX_SCALE = 0.5;
const DESKTOP_SCALE = 0.001;

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
    modelGroup.scale.setScalar(DESKTOP_SCALE);
    scene.add(modelGroup);

    const reticleGeo = new THREE.RingGeometry(0.03, 0.04, 32);
    reticleGeo.rotateX(-Math.PI / 2);
    reticle = new THREE.Mesh(reticleGeo, new THREE.MeshBasicMaterial({ color: 0x999999 }));
    reticle.matrixAutoUpdate = false;
    reticle.visible = false;
    scene.add(reticle);

    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.target.set(0, 0, 0);

    const overlay = document.getElementById('overlay');

    if ('xr' in navigator) {
        const arButton = ARButton.createButton(renderer, {
            requiredFeatures: ['hit-test'],
            optionalFeatures: ['dom-overlay'],
            domOverlay: { root: overlay },
        });
        document.getElementById('ar-button-container').appendChild(arButton);

        renderer.xr.addEventListener('sessionstart', () => {
            isARActive = true;
            modelPlaced = false;
            modelGroup.visible = false;
            modelGroup.scale.setScalar(DESKTOP_SCALE);
            modelGroup.rotation.set(0, 0, 0);
            controls.enabled = false;
            // Make overlay receive pointer events during XR
            overlay.style.pointerEvents = 'auto';
            overlay.style.touchAction = 'none';
        });
        renderer.xr.addEventListener('sessionend', () => {
            isARActive = false;
            modelGroup.visible = true;
            modelGroup.position.set(0, 0, 0);
            modelGroup.quaternion.identity();
            modelGroup.scale.setScalar(DESKTOP_SCALE);
            controls.enabled = true;
            // Restore overlay passthrough for desktop
            overlay.style.pointerEvents = '';
            overlay.style.touchAction = '';
            resetGesture();
        });
    }

    renderer.xr.addEventListener('sessionstart', () => {
        const session = renderer.xr.getSession();
        session.addEventListener('select', onARSelect);
    });

    // Pointer events on the overlay — these fire during WebXR dom-overlay
    overlay.addEventListener('pointerdown', onGesturePointerDown);
    overlay.addEventListener('pointermove', onGesturePointerMove);
    overlay.addEventListener('pointerup', onGesturePointerUp);
    overlay.addEventListener('pointercancel', onGesturePointerUp);

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
            const pose = results[0].getPose(refSpace);
            reticle.visible = true;
            reticle.matrix.fromArray(pose.transform.matrix);
        } else {
            reticle.visible = false;
        }
    }
}

// --- Pointer-event gesture handlers (work in WebXR dom-overlay) ---

function getPointerDist(a, b) {
    const dx = a.clientX - b.clientX;
    const dy = a.clientY - b.clientY;
    return Math.sqrt(dx * dx + dy * dy);
}

function getPointerAngle(a, b) {
    return Math.atan2(b.clientY - a.clientY, b.clientX - a.clientX);
}

function resetGesture() {
    activePointers.clear();
    prevPinchDist = null;
    prevPinchAngle = null;
}

function onGesturePointerDown(event) {
    if (!isARActive) return;

    // Single tap before placement → place model
    if (!modelPlaced) {
        placeModel();
        return;
    }

    activePointers.set(event.pointerId, { clientX: event.clientX, clientY: event.clientY });

    if (activePointers.size === 2) {
        const [a, b] = [...activePointers.values()];
        prevPinchDist = getPointerDist(a, b);
        prevPinchAngle = getPointerAngle(a, b);
    }
}

function onGesturePointerMove(event) {
    if (!isARActive || !modelPlaced) return;
    if (!activePointers.has(event.pointerId)) return;

    activePointers.set(event.pointerId, { clientX: event.clientX, clientY: event.clientY });

    if (activePointers.size === 2 && prevPinchDist !== null) {
        const [a, b] = [...activePointers.values()];

        // Pinch to scale
        const dist = getPointerDist(a, b);
        const scaleRatio = dist / prevPinchDist;
        const newScale = THREE.MathUtils.clamp(
            modelGroup.scale.x * scaleRatio,
            MIN_SCALE,
            MAX_SCALE
        );
        modelGroup.scale.setScalar(newScale);
        prevPinchDist = dist;

        // Two-finger rotate
        const angle = getPointerAngle(a, b);
        const angleDelta = angle - prevPinchAngle;
        modelGroup.rotation.y += angleDelta;
        prevPinchAngle = angle;
    }
}

function onGesturePointerUp(event) {
    activePointers.delete(event.pointerId);
    if (activePointers.size < 2) {
        prevPinchDist = null;
        prevPinchAngle = null;
    }
}

// ---

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
