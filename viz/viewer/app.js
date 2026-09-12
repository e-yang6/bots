/**
 * WebXR AR viewer for aorta visualization with gesture control.
 *
 * AR mode: place model on surface, then control scale with body gestures.
 * Raise both arms above shoulders → bigger.
 * Lower both arms below hips → smaller.
 * Arms at rest → hold current size.
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

let gestureReady = false;
let cameraAccessAvailable = false;
let fallbackVideo = null;
let currentScale = 0.001; // mm to meters

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
    modelGroup.scale.setScalar(currentScale);
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
            currentScale = 0.001;
            modelGroup.scale.setScalar(currentScale);
            controls.enabled = true;
            stopFallbackVideo();
        });
    }

    renderer.xr.addEventListener('sessionstart', () => {
        const session = renderer.xr.getSession();
        session.addEventListener('select', onARSelect);
    });
    renderer.domElement.addEventListener('pointerdown', onPointerDown);
    window.addEventListener('resize', onWindowResize);

    loadScene('scene.json');
    renderer.setAnimationLoop(animate);
}

async function startGestureDetection() {
    if (gestureReady) return;
    document.getElementById('status').textContent = 'Loading gesture detection...';
    try {
        await initPoseDetection();
        gestureReady = true;
        document.getElementById('status').textContent = '';
    } catch (err) {
        console.warn('Pose detection init failed:', err);
        document.getElementById('status').textContent = '';
    }
}

function startFallbackVideo() {
    if (fallbackVideo) return;
    navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment', width: 320, height: 240 }
    }).then(stream => {
        fallbackVideo = document.createElement('video');
        fallbackVideo.srcObject = stream;
        fallbackVideo.setAttribute('playsinline', '');
        fallbackVideo.play();
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
        document.getElementById('status').textContent = gestureReady
            ? 'Raise arms to grow, lower to shrink'
            : '';
    }
}

function onARSelect() {
    if (isARActive) placeModel();
}

function onPointerDown() {
    if (isARActive) placeModel();
}

function onXRFrame(timestamp, frame) {
    if (!isARActive) return;

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
                const pose = results[0].getPose(refSpace);
                reticle.visible = true;
                reticle.matrix.fromArray(pose.transform.matrix);
            } else {
                reticle.visible = false;
            }
        }
        return;
    }

    // Gesture detection
    if (gestureReady) {
        let usedXRCamera = false;
        try {
            processXRFrame(renderer, frame);
            if (getGestureState().personDetected) usedXRCamera = true;
        } catch (e) {}

        if (!usedXRCamera && !cameraAccessAvailable) {
            if (!fallbackVideo) startFallbackVideo();
            if (fallbackVideo && fallbackVideo.readyState >= 2) {
                processVideoFrame(fallbackVideo);
            }
        } else {
            cameraAccessAvailable = true;
        }

        const state = getGestureState();
        if (state.personDetected) {
            currentScale = applyGestures(modelGroup, currentScale);
            if (state.action === 'up') {
                document.getElementById('status').textContent = 'Growing...';
            } else if (state.action === 'down') {
                document.getElementById('status').textContent = 'Shrinking...';
            } else {
                document.getElementById('status').textContent = 'Raise arms to grow, lower to shrink';
            }
        }
    }
}

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
