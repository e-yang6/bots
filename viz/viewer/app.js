/**
 * WebXR AR viewer for aorta branch visualization.
 *
 * Loads a .glb aorta mesh and scene.json with branch markers.
 * Supports:
 *   - AR mode (Android Chrome): place model on a surface, tap branches for info
 *   - Fallback 3D mode (desktop/non-AR): orbit controls, same interaction
 */

import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { ARButton } from 'three/addons/webxr/ARButton.js';

// ─── State ───────────────────────────────────────────────────────────────────

let scene, camera, renderer, controls;
let modelGroup;          // holds mesh + markers, moved as a unit in AR
let branchMarkers = [];  // { mesh, data } for raycasting
let sceneData = null;
let reticle;             // AR hit-test reticle
let hitTestSource = null;
let hitTestSourceRequested = false;
let modelPlaced = false;
let isARActive = false;

const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();

// ─── Initialization ─────────────────────────────────────────────────────────

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

    // Lighting
    scene.add(new THREE.AmbientLight(0xffffff, 0.6));
    const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
    dirLight.position.set(0.5, 1, 0.5);
    scene.add(dirLight);

    // Group that holds the aorta model + branch markers
    modelGroup = new THREE.Group();
    // Scale mm → meters for AR (1mm = 0.001m), then scale up for visibility
    // A typical aorta segment is ~100-200mm tall → 0.1-0.2m, good AR size
    modelGroup.scale.setScalar(0.001);
    scene.add(modelGroup);

    // AR reticle (ring shown on detected surfaces before placement)
    const reticleGeo = new THREE.RingGeometry(0.03, 0.04, 32);
    reticleGeo.rotateX(-Math.PI / 2);
    reticle = new THREE.Mesh(reticleGeo, new THREE.MeshBasicMaterial({ color: 0x00ff88 }));
    reticle.matrixAutoUpdate = false;
    reticle.visible = false;
    scene.add(reticle);

    // Orbit controls for non-AR fallback
    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.target.set(0, 0, 0);

    // AR button — only shown if WebXR is available
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
        });
        renderer.xr.addEventListener('sessionend', () => {
            isARActive = false;
            modelGroup.visible = true;
            modelGroup.position.set(0, 0, 0);
            modelGroup.quaternion.identity();
            controls.enabled = true;
        });
    }

    // Interaction
    renderer.domElement.addEventListener('pointerdown', onPointerDown);
    window.addEventListener('resize', onWindowResize);

    // Load scene data from URL params or default path
    const params = new URLSearchParams(window.location.search);
    const sceneUrl = params.get('scene') || 'scene.json';
    loadScene(sceneUrl);

    renderer.setAnimationLoop(animate);
}

// ─── Scene loading ──────────────────────────────────────────────────────────

async function loadScene(sceneJsonUrl) {
    const statusEl = document.getElementById('status');
    statusEl.textContent = 'Loading scene...';

    try {
        const baseUrl = sceneJsonUrl.substring(0, sceneJsonUrl.lastIndexOf('/') + 1);
        const resp = await fetch(sceneJsonUrl);
        sceneData = await resp.json();

        // Load aorta mesh
        const meshUrl = baseUrl + sceneData.mesh_file;
        const loader = new GLTFLoader();
        const gltf = await loader.loadAsync(meshUrl);

        const aortaMesh = gltf.scene;
        // Semi-transparent red aorta
        aortaMesh.traverse((child) => {
            if (child.isMesh) {
                child.material = new THREE.MeshPhysicalMaterial({
                    color: 0xcc3333,
                    transparent: true,
                    opacity: 0.45,
                    roughness: 0.4,
                    metalness: 0.1,
                    side: THREE.DoubleSide,
                    depthWrite: false,
                });
            }
        });
        modelGroup.add(aortaMesh);

        // Add branch markers
        addBranchMarkers(sceneData.branches);

        // Update info panel
        document.getElementById('case-id').textContent = sceneData.case_id;
        document.getElementById('branch-count').textContent = sceneData.branches.length;

        statusEl.textContent = sceneData.branches.length > 0
            ? 'Tap a branch marker for details'
            : 'No branches detected';

        // In non-AR mode, make sure model is visible and centered
        if (!isARActive) {
            modelGroup.visible = true;
            fitCameraToModel();
        }
    } catch (err) {
        statusEl.textContent = 'Error loading scene: ' + err.message;
        console.error(err);
    }
}

function addBranchMarkers(branches) {
    const markerGeo = new THREE.SphereGeometry(2.5, 16, 16);  // 2.5mm radius sphere
    const markerMat = new THREE.MeshPhysicalMaterial({
        color: 0x00ccff,
        emissive: 0x004466,
        emissiveIntensity: 0.5,
        roughness: 0.3,
    });

    const arrowLength = 12;  // mm
    const arrowHeadLength = 3;
    const arrowHeadWidth = 2;

    branches.forEach((branch) => {
        // Ostium marker sphere
        const marker = new THREE.Mesh(markerGeo, markerMat.clone());
        marker.position.set(branch.ostium[0], branch.ostium[1], branch.ostium[2]);
        modelGroup.add(marker);

        // Direction arrow from ostium along branch direction
        const dir = new THREE.Vector3(...branch.direction).normalize();
        const origin = new THREE.Vector3(...branch.ostium);
        const arrow = new THREE.ArrowHelper(
            dir, origin, arrowLength, 0x00ff88, arrowHeadLength, arrowHeadWidth
        );
        modelGroup.add(arrow);

        // Seed point (smaller, different color)
        const seedGeo = new THREE.SphereGeometry(1.5, 12, 12);
        const seedMat = new THREE.MeshPhysicalMaterial({
            color: 0xffaa00,
            emissive: 0x553300,
            emissiveIntensity: 0.4,
            roughness: 0.3,
        });
        const seedMarker = new THREE.Mesh(seedGeo, seedMat);
        seedMarker.position.set(branch.seed[0], branch.seed[1], branch.seed[2]);
        modelGroup.add(seedMarker);

        // Radius ring at seed point
        const ringGeo = new THREE.TorusGeometry(branch.radius_mm, 0.3, 8, 32);
        const ringMat = new THREE.MeshBasicMaterial({ color: 0xffaa00, transparent: true, opacity: 0.6 });
        const ring = new THREE.Mesh(ringGeo, ringMat);
        ring.position.set(branch.seed[0], branch.seed[1], branch.seed[2]);
        // Orient ring perpendicular to branch direction
        ring.lookAt(
            branch.seed[0] + branch.direction[0],
            branch.seed[1] + branch.direction[1],
            branch.seed[2] + branch.direction[2]
        );
        modelGroup.add(ring);

        branchMarkers.push({ mesh: marker, data: branch });
    });
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

// ─── Interaction ────────────────────────────────────────────────────────────

function onPointerDown(event) {
    // In AR mode, first tap places the model
    if (isARActive && !modelPlaced && reticle.visible) {
        modelGroup.position.setFromMatrixPosition(reticle.matrix);
        modelGroup.visible = true;
        modelPlaced = true;
        document.getElementById('status').textContent = 'Model placed. Tap a branch marker for details.';
        return;
    }

    // Branch marker hit test
    const rect = renderer.domElement.getBoundingClientRect();
    pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

    raycaster.setFromCamera(pointer, camera);
    const markerMeshes = branchMarkers.map(b => b.mesh);
    const intersects = raycaster.intersectObjects(markerMeshes, false);

    if (intersects.length > 0) {
        const hit = branchMarkers.find(b => b.mesh === intersects[0].object);
        if (hit) showBranchInfo(hit);
    } else {
        hideBranchInfo();
    }
}

function showBranchInfo(branch) {
    const panel = document.getElementById('info-panel');
    const d = branch.data;

    // Highlight selected marker
    branchMarkers.forEach(b => {
        b.mesh.material.emissive.setHex(b === branch ? 0x00ffff : 0x004466);
        b.mesh.material.emissiveIntensity = b === branch ? 1.0 : 0.5;
    });

    document.getElementById('info-id').textContent = d.id;
    document.getElementById('info-radius').textContent = d.radius_mm.toFixed(1) + ' mm';
    document.getElementById('info-ostium').textContent =
        `(${d.ostium[0].toFixed(1)}, ${d.ostium[1].toFixed(1)}, ${d.ostium[2].toFixed(1)})`;
    document.getElementById('info-direction').textContent =
        `(${d.direction[0].toFixed(2)}, ${d.direction[1].toFixed(2)}, ${d.direction[2].toFixed(2)})`;

    panel.classList.add('visible');
}

function hideBranchInfo() {
    document.getElementById('info-panel').classList.remove('visible');
    branchMarkers.forEach(b => {
        b.mesh.material.emissive.setHex(0x004466);
        b.mesh.material.emissiveIntensity = 0.5;
    });
}

// ─── AR hit-test ────────────────────────────────────────────────────────────

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

// ─── Resize ─────────────────────────────────────────────────────────────────

function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
}

// ─── Render loop ────────────────────────────────────────────────────────────

function animate(timestamp, frame) {
    if (frame) onXRFrame(timestamp, frame);
    if (!isARActive) controls.update();
    renderer.render(scene, camera);
}

// ─── Start ──────────────────────────────────────────────────────────────────

init();
