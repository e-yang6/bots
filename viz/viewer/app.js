/**
 * WebXR AR viewer for aorta branch visualization.
 *
 * Loads a .glb aorta mesh and per-branch vessel meshes from scene.json.
 * Desktop always shows metrics panel on the left + main view on the right.
 * Main view = live camera feed from phone (via WebRTC) with 3D model as
 * fallback when no feed is connected. Clicking branches on the 3D model
 * (meshes or ostium spheres) opens the info/summary panel.
 *
 * Supports:
 *   - AR mode (phone): place model on surface, pinch-to-scale, rotate
 *   - Desktop: orbit controls, click branches for info
 *   - Color modes: Distinct / Confidence
 *   - Toggle rejected branches visibility
 */

import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { ARButton } from 'three/addons/webxr/ARButton.js';

// ─── State ───────────────────────────────────────────────────────────────────

let scene, camera, renderer, controls;
let modelGroup;
let branchMarkers = [];   // { ostiumMesh, branchMesh, meshChildren[], arrow, ring, data, index }
let sceneData = null;
let reticle;
let hitTestSource = null;
let hitTestSourceRequested = false;
let hitTestSourcePending = false;   // guard against stale promise resolution
let modelPlaced = false;
let isARActive = false;
let arSessionId = 0;                // incremented each session to detect stale callbacks
let selectedBranchIndex = -1;
let showRejected = true;
let hasVideoFeed = false;

const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();

// Gesture state
const activePointers = new Map();
let prevPinchDist = null;
let prevPinchAngle = null;
const MIN_SCALE = 0.0002;
const MAX_SCALE = 0.5;
const DESKTOP_SCALE = 0.001;

// Color mode: 'distinct' or 'confidence'
let colorMode = 'distinct';

// WebSocket / WebRTC
const params = new URLSearchParams(window.location.search);
const isPhone = /Mobi|Android/i.test(navigator.userAgent);
let ws = null;
let wsSendInterval = null;
let peerConnection = null;

// ─── Initialization ─────────────────────────────────────────────────────────

function init() {
    // Desktop always gets the companion layout (metrics panel)
    if (!isPhone) {
        document.body.classList.add('companion');
    }

    scene = new THREE.Scene();

    camera = new THREE.PerspectiveCamera(70, 1, 0.01, 100);
    camera.position.set(0, 0.15, 0.3);

    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.xr.enabled = true;
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    document.getElementById('viewport').appendChild(renderer.domElement);
    onWindowResize();

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
            arSessionId++;
            isARActive = true;
            modelPlaced = false;
            modelGroup.visible = false;
            modelGroup.scale.setScalar(DESKTOP_SCALE);
            modelGroup.rotation.set(0, 0, 0);
            controls.enabled = false;
            overlay.style.pointerEvents = 'auto';
            overlay.style.touchAction = 'none';
            reticle.visible = false;
            // Reset hit-test so a fresh source is requested for this new session
            if (hitTestSource) {
                hitTestSource.cancel();
                hitTestSource = null;
            }
            hitTestSourceRequested = false;
            hitTestSourcePending = false;
            startSyncSend();
        });
        renderer.xr.addEventListener('sessionend', () => {
            arSessionId++;
            isARActive = false;
            modelGroup.visible = true;
            modelGroup.position.set(0, 0, 0);
            modelGroup.quaternion.identity();
            modelGroup.scale.setScalar(DESKTOP_SCALE);
            controls.enabled = true;
            overlay.style.pointerEvents = '';
            overlay.style.touchAction = '';
            reticle.visible = false;
            resetGesture();
            stopSyncSend();
            // Cancel and reset hit-test state
            if (hitTestSource) {
                hitTestSource.cancel();
                hitTestSource = null;
            }
            hitTestSourceRequested = false;
            hitTestSourcePending = false;
        });
    }

    renderer.xr.addEventListener('sessionstart', () => {
        const session = renderer.xr.getSession();
        session.addEventListener('select', onARSelect);
    });

    // Pointer events for gestures (AR) and raycasting (desktop)
    overlay.addEventListener('pointerdown', onGesturePointerDown);
    overlay.addEventListener('pointermove', onGesturePointerMove);
    overlay.addEventListener('pointerup', onGesturePointerUp);
    overlay.addEventListener('pointercancel', onGesturePointerUp);
    renderer.domElement.addEventListener('pointerdown', onPointerDown);

    // Toggle buttons
    document.getElementById('btn-distinct').addEventListener('click', () => setColorMode('distinct'));
    document.getElementById('btn-confidence').addEventListener('click', () => setColorMode('confidence'));
    document.getElementById('btn-rejected').addEventListener('click', toggleRejected);

    window.addEventListener('resize', onWindowResize);

    const sceneUrl = params.get('scene') || 'scene.json';
    loadScene(sceneUrl);

    // Connect WebSocket for sync
    connectWebSocket();
    if (isPhone) {
        startSyncSend();
        startCameraStream();
    }

    renderer.setAnimationLoop(animate);
}

// ─── Video feed layout ──────────────────────────────────────────────────────

function activateVideoFeed() {
    if (hasVideoFeed) return;
    hasVideoFeed = true;
    document.body.classList.add('has-video');
}

// ─── Scene loading ──────────────────────────────────────────────────────────

async function loadScene(sceneJsonUrl) {
    const statusEl = document.getElementById('status');
    statusEl.textContent = 'Loading...';

    try {
        const baseUrl = sceneJsonUrl.substring(0, sceneJsonUrl.lastIndexOf('/') + 1);
        const resp = await fetch(sceneJsonUrl);
        sceneData = await resp.json();

        // Load aorta mesh
        const meshUrl = baseUrl + sceneData.mesh_file;
        const loader = new GLTFLoader();
        const gltf = await loader.loadAsync(meshUrl);

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

        // Load branch meshes and add markers
        await addBranchMeshes(sceneData.branches, baseUrl, loader);

        // Update UI
        const acceptedCount = sceneData.branches.filter(b => b.accepted !== false).length;
        const rejectedCount = sceneData.branches.filter(b => b.accepted === false).length;
        document.getElementById('case-id').textContent = sceneData.case_id;
        document.getElementById('branch-count').textContent =
            acceptedCount + (rejectedCount > 0 ? ` (+${rejectedCount} rejected)` : '');

        // Always build metrics panel (desktop is always companion now)
        const metricsCase = document.getElementById('metrics-case');
        const metricsCount = document.getElementById('metrics-count');
        if (metricsCase) metricsCase.textContent = sceneData.case_id;
        if (metricsCount) metricsCount.textContent =
            acceptedCount + ' accepted, ' + rejectedCount + ' rejected';
        buildMetricsPanel(sceneData.branches);

        statusEl.textContent = sceneData.branches.length > 0
            ? 'Tap a branch for details'
            : 'No branches detected';

        if (!isARActive) {
            modelGroup.visible = true;
            fitCameraToModel();
        }
    } catch (err) {
        statusEl.textContent = 'Error: ' + err.message;
        console.error(err);
    }
}

// ─── Branch meshes + markers ────────────────────────────────────────────────

function branchColor(index, total, mode, branch) {
    if (mode === 'confidence') {
        const conf = branch && branch.confidence != null ? branch.confidence : 0.5;
        const hue = conf * 0.33;
        return new THREE.Color().setHSL(hue, 0.8, 0.5);
    }
    const hue = index / Math.max(total, 1);
    return new THREE.Color().setHSL(hue, 0.75, 0.55);
}

const REJECTED_COLOR = new THREE.Color(0.35, 0.35, 0.38);

async function addBranchMeshes(branches, baseUrl, loader) {
    const arrowLength = 12;
    const arrowHeadLength = 3;
    const arrowHeadWidth = 2;

    for (let i = 0; i < branches.length; i++) {
        const branch = branches[i];
        const accepted = branch.accepted !== false;
        const color = accepted
            ? branchColor(i, branches.length, colorMode, branch)
            : REJECTED_COLOR.clone();

        let branchMesh = null;
        let meshChildren = [];  // collect child meshes for raycasting

        // Load branch mesh if available
        if (branch.mesh_file) {
            try {
                const meshUrl = baseUrl + branch.mesh_file;
                const gltf = await loader.loadAsync(meshUrl);
                branchMesh = gltf.scene;
                branchMesh.traverse((child) => {
                    if (child.isMesh) {
                        child.material = new THREE.MeshPhysicalMaterial({
                            color: color,
                            transparent: !accepted,
                            opacity: accepted ? 1.0 : 0.4,
                            roughness: 0.25,
                            metalness: 0.05,
                            clearcoat: 0.6,
                            clearcoatRoughness: 0.2,
                            side: THREE.DoubleSide,
                            depthWrite: accepted,
                        });
                        meshChildren.push(child);
                    }
                });
                branchMesh.visible = accepted || showRejected;
                modelGroup.add(branchMesh);
            } catch (e) {
                console.warn(`Failed to load mesh for ${branch.id}:`, e);
            }
        }

        // Ostium sphere (raycast target)
        const ostiumGeo = new THREE.SphereGeometry(2.0, 16, 16);
        const ostiumMat = new THREE.MeshPhysicalMaterial({
            color: color,
            emissive: color.clone().multiplyScalar(0.3),
            emissiveIntensity: 0.5,
            roughness: 0.3,
            transparent: true,
            opacity: branchMesh ? 0.6 : 1.0,
        });
        const ostiumMesh = new THREE.Mesh(ostiumGeo, ostiumMat);
        ostiumMesh.position.set(branch.ostium[0], branch.ostium[1], branch.ostium[2]);
        ostiumMesh.visible = accepted || showRejected;
        modelGroup.add(ostiumMesh);

        // Direction arrow
        const dir = new THREE.Vector3(...branch.direction).normalize();
        const origin = new THREE.Vector3(...branch.ostium);
        const arrow = new THREE.ArrowHelper(
            dir, origin, arrowLength, color.getHex(), arrowHeadLength, arrowHeadWidth
        );
        arrow.visible = accepted || showRejected;
        modelGroup.add(arrow);

        // Radius ring at seed
        const ringGeo = new THREE.TorusGeometry(branch.radius_mm, 0.3, 8, 32);
        const ringMat = new THREE.MeshBasicMaterial({
            color: color, transparent: true, opacity: 0.5,
        });
        const ring = new THREE.Mesh(ringGeo, ringMat);
        ring.position.set(branch.seed[0], branch.seed[1], branch.seed[2]);
        ring.lookAt(
            branch.seed[0] + branch.direction[0],
            branch.seed[1] + branch.direction[1],
            branch.seed[2] + branch.direction[2]
        );
        ring.visible = accepted || showRejected;
        modelGroup.add(ring);

        branchMarkers.push({
            ostiumMesh, branchMesh, meshChildren, arrow, ring,
            data: branch, index: i, accepted,
        });
    }
}

function recolorAllMarkers() {
    const total = branchMarkers.length;
    branchMarkers.forEach((m, i) => {
        const accepted = m.accepted;
        const color = accepted
            ? branchColor(i, total, colorMode, m.data)
            : REJECTED_COLOR.clone();

        m.ostiumMesh.material.color.copy(color);
        m.ostiumMesh.material.emissive.copy(color).multiplyScalar(0.3);

        m.arrow.setColor(color.getHex());
        m.ring.material.color.copy(color);

        if (m.branchMesh) {
            m.branchMesh.traverse((child) => {
                if (child.isMesh) {
                    child.material.color.copy(color);
                }
            });
        }

        // Update companion panel border color
        const item = document.querySelector(`.branch-item[data-index="${i}"]`);
        if (item) {
            item.style.borderLeftColor = '#' + color.getHexString();
        }
    });
}

function toggleRejected() {
    showRejected = !showRejected;
    const btn = document.getElementById('btn-rejected');
    btn.classList.toggle('active', showRejected);
    btn.textContent = showRejected ? 'Rejected: On' : 'Rejected: Off';

    branchMarkers.forEach(m => {
        if (!m.accepted) {
            const vis = showRejected;
            m.ostiumMesh.visible = vis;
            m.arrow.visible = vis;
            m.ring.visible = vis;
            if (m.branchMesh) m.branchMesh.visible = vis;
        }
    });
}

function setColorMode(mode) {
    colorMode = mode;
    document.getElementById('btn-distinct').classList.toggle('active', mode === 'distinct');
    document.getElementById('btn-confidence').classList.toggle('active', mode === 'confidence');
    recolorAllMarkers();
}

// ─── Metrics panel ──────────────────────────────────────────────────────────

function buildMetricsPanel(branches) {
    const list = document.getElementById('branch-list');
    if (!list) return;
    list.innerHTML = '';

    branches.forEach((branch, i) => {
        const accepted = branch.accepted !== false;
        const color = accepted
            ? branchColor(i, branches.length, colorMode, branch)
            : REJECTED_COLOR;
        const item = document.createElement('div');
        item.className = 'branch-item' + (accepted ? '' : ' rejected');
        item.dataset.index = i;
        item.style.borderLeftColor = '#' + color.getHexString();

        const conf = branch.confidence != null ? branch.confidence.toFixed(2) : '---';
        const statusBadge = accepted
            ? '<span class="branch-badge accepted">accepted</span>'
            : '<span class="branch-badge rejected-badge">rejected</span>';
        const reason = branch.reject_reason ? `<div class="branch-item-reason">${branch.reject_reason}</div>` : '';

        item.innerHTML = `
            <div class="branch-item-header">
                <span class="branch-item-id">${branch.id}</span>
                ${statusBadge}
                <span class="branch-item-conf">${conf}</span>
            </div>
            <div class="branch-item-details">
                r=${branch.radius_mm.toFixed(1)}mm
            </div>
            ${reason}
        `;

        item.addEventListener('click', () => selectBranch(i));
        list.appendChild(item);
    });
}

function selectBranch(index) {
    selectedBranchIndex = index;

    document.querySelectorAll('.branch-item').forEach((el, i) => {
        el.classList.toggle('selected', i === index);
    });

    branchMarkers.forEach((m, i) => {
        const isSel = i === index;
        m.ostiumMesh.material.emissiveIntensity = isSel ? 1.0 : 0.5;
        if (m.branchMesh) {
            m.branchMesh.traverse((child) => {
                if (child.isMesh) {
                    child.material.emissive = isSel
                        ? child.material.color.clone().multiplyScalar(0.3)
                        : new THREE.Color(0, 0, 0);
                }
            });
        }
    });

    if (index >= 0 && index < branchMarkers.length) {
        showBranchInfo(branchMarkers[index]);
    }
}

// ─── Camera ─────────────────────────────────────────────────────────────────

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
    if (isARActive && !modelPlaced && reticle.visible) {
        modelGroup.position.setFromMatrixPosition(reticle.matrix);
        modelGroup.visible = true;
        modelPlaced = true;
        document.getElementById('status').textContent = '';
        return;
    }

    const rect = renderer.domElement.getBoundingClientRect();
    pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

    raycaster.setFromCamera(pointer, camera);

    // Collect all clickable objects: ostium spheres + branch mesh children
    const hitTargets = [];
    const targetToMarker = new Map();

    for (const m of branchMarkers) {
        if (!m.ostiumMesh.visible) continue;
        hitTargets.push(m.ostiumMesh);
        targetToMarker.set(m.ostiumMesh, m);
        for (const child of m.meshChildren) {
            hitTargets.push(child);
            targetToMarker.set(child, m);
        }
    }

    const intersects = raycaster.intersectObjects(hitTargets, false);

    if (intersects.length > 0) {
        const hit = targetToMarker.get(intersects[0].object);
        if (hit) selectBranch(hit.index);
    } else {
        hideBranchInfo();
    }
}

function showBranchInfo(marker) {
    const panel = document.getElementById('info-panel');
    const d = marker.data;
    const color = marker.accepted
        ? branchColor(marker.index, branchMarkers.length, colorMode, d)
        : REJECTED_COLOR;

    document.querySelector('#info-panel .dot').style.background = '#' + color.getHexString();
    document.getElementById('info-id').textContent = d.id;
    document.getElementById('info-radius').textContent = d.radius_mm.toFixed(1) + ' mm';
    document.getElementById('info-confidence').textContent =
        d.confidence != null ? d.confidence.toFixed(2) : '---';
    const physicalOstium = d.ostium.map((v, axis) => v + (sceneData.centroid_mm?.[axis] ?? 0));
    document.getElementById('info-ostium').textContent =
        `(${physicalOstium.map(v => v.toFixed(1)).join(', ')}) mm LPS`;
    document.getElementById('info-direction').textContent =
        `(${d.direction[0].toFixed(2)}, ${d.direction[1].toFixed(2)}, ${d.direction[2].toFixed(2)})`;

    const statusEl = document.getElementById('info-status');
    if (statusEl) {
        if (d.accepted === false) {
            statusEl.textContent = 'Rejected' + (d.reject_reason ? ': ' + d.reject_reason : '');
            statusEl.className = 'value reject-text';
        } else {
            statusEl.textContent = 'Accepted';
            statusEl.className = 'value accept-text';
        }
    }

    // Rule breakdown
    const rulesSection = document.getElementById('info-rules');
    const vetoEl = document.getElementById('rule-veto');
    const termsEl = document.getElementById('rule-terms');
    const flatEl = document.getElementById('rule-flat-penalties');
    const totalEl = document.getElementById('rule-total');

    if (d.rule_breakdown) {
        rulesSection.style.display = '';
        const rb = d.rule_breakdown;

        // Veto
        if (rb.veto) {
            vetoEl.innerHTML = `<span class="veto-badge">VETOED</span> <span class="veto-reason">${rb.veto}</span>`;
            vetoEl.style.display = '';
        } else {
            vetoEl.style.display = 'none';
        }

        // Terms
        termsEl.innerHTML = '';
        for (const t of rb.terms) {
            const pct = Math.round(t.ramp * 100);
            // Color: green (low ramp) -> amber -> red (high ramp)
            const hue = Math.round((1 - t.ramp) * 120); // 120=green, 0=red
            const barColor = `hsl(${hue}, 70%, 45%)`;
            const row = document.createElement('div');
            row.className = 'rule-term';
            row.innerHTML =
                `<span class="rule-term-name">${t.name}</span>` +
                `<div class="rule-term-bar-track">` +
                    `<div class="rule-term-bar" style="width:${pct}%;background:${barColor}"></div>` +
                `</div>` +
                `<span class="rule-term-penalty">${t.penalty > 0 ? '-' : ''}${t.penalty.toFixed(2)}</span>` +
                `<div class="rule-term-note">${t.note}</div>`;
            termsEl.appendChild(row);
        }

        // Flat penalties
        const flatParts = [];
        if (rb.case_flood_leaking_penalty > 0) {
            flatParts.push(`Flood leaking: -${rb.case_flood_leaking_penalty.toFixed(2)}`);
        }
        if (rb.shares_vessel_penalty > 0) {
            flatParts.push(`Shared vessel: -${rb.shares_vessel_penalty.toFixed(2)}`);
        }
        flatEl.innerHTML = flatParts.length
            ? flatParts.map(p => `<span class="flat-penalty">${p}</span>`).join('')
            : '';

        // Total
        totalEl.innerHTML =
            `<span class="rule-total-label">Total penalty</span>` +
            `<span class="rule-total-value">-${rb.total_penalty.toFixed(2)}</span>` +
            `<span class="rule-total-label">Confidence</span>` +
            `<span class="rule-total-value">${d.confidence != null ? d.confidence.toFixed(2) : '---'}</span>`;
    } else {
        rulesSection.style.display = 'none';
    }

    panel.classList.add('visible');
}

function hideBranchInfo() {
    document.getElementById('info-panel').classList.remove('visible');
    selectedBranchIndex = -1;
    branchMarkers.forEach(m => {
        m.ostiumMesh.material.emissiveIntensity = 0.5;
        if (m.branchMesh) {
            m.branchMesh.traverse((child) => {
                if (child.isMesh) {
                    child.material.emissive = new THREE.Color(0, 0, 0);
                }
            });
        }
    });
    document.querySelectorAll('.branch-item').forEach(el => el.classList.remove('selected'));
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

// ─── AR gestures (pointer-event based) ──────────────────────────────────────

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
    if (!modelPlaced) { placeModel(); return; }

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
        const dist = getPointerDist(a, b);
        const scaleRatio = dist / prevPinchDist;
        const newScale = THREE.MathUtils.clamp(
            modelGroup.scale.x * scaleRatio, MIN_SCALE, MAX_SCALE
        );
        modelGroup.scale.setScalar(newScale);
        prevPinchDist = dist;

        const angle = getPointerAngle(a, b);
        modelGroup.rotation.y += angle - prevPinchAngle;
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

// ─── WebSocket sync ─────────────────────────────────────────────────────────

function connectWebSocket() {
    if (params.get('sync') !== '1') return;
    const loc = window.location;
    const wsProto = loc.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsPort = parseInt(loc.port || (loc.protocol === 'https:' ? '443' : '80')) + 1;
    const wsUrl = `${wsProto}//${loc.hostname}:${wsPort}`;

    try {
        ws = new WebSocket(wsUrl);
        ws.onopen = () => {
            // Laptop: request the phone to send its WebRTC offer
            if (!isPhone) {
                ws.send(JSON.stringify({ type: 'webrtc-request' }));
            }
        };
        ws.onmessage = (evt) => {
            try {
                const msg = JSON.parse(evt.data);
                if (msg.type === 'sync' && !isPhone) {
                    modelGroup.position.fromArray(msg.position);
                    modelGroup.quaternion.fromArray(msg.quaternion);
                    modelGroup.scale.setScalar(msg.scale);
                }
                // WebRTC signaling
                if (msg.type === 'webrtc-offer' && !isPhone) {
                    handleWebRTCOffer(msg);
                }
                if (msg.type === 'webrtc-answer' && isPhone) {
                    handleWebRTCAnswer(msg);
                }
                if (msg.type === 'webrtc-ice') {
                    if (peerConnection) {
                        peerConnection.addIceCandidate(new RTCIceCandidate(msg.candidate));
                    }
                }
                // Phone: laptop is asking us to (re-)send the offer
                if (msg.type === 'webrtc-request' && isPhone) {
                    resendWebRTCOffer();
                }
            } catch (e) { /* ignore bad messages */ }
        };
        ws.onclose = () => {
            ws = null;
            setTimeout(connectWebSocket, 2000);
        };
        ws.onerror = () => {
            ws = null;
        };
    } catch (e) {
        // WebSocket not available
    }
}

function sendSyncState() {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    const msg = {
        type: 'sync',
        position: modelGroup.position.toArray(),
        quaternion: modelGroup.quaternion.toArray(),
        scale: modelGroup.scale.x,
    };
    ws.send(JSON.stringify(msg));
}

function startSyncSend() {
    if (!isPhone) return;
    stopSyncSend();
    wsSendInterval = setInterval(sendSyncState, 100);
}

function stopSyncSend() {
    if (wsSendInterval) {
        clearInterval(wsSendInterval);
        wsSendInterval = null;
    }
}

// ─── WebRTC camera feed ─────────────────────────────────────────────────────

let localStream = null;

async function startCameraStream() {
    try {
        localStream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } },
        });
        // Create and send the initial offer
        await createAndSendOffer();
    } catch (e) {
        console.warn('Camera access not available:', e);
    }
}

async function createAndSendOffer() {
    if (!localStream) return;

    // Close previous peer connection if any
    if (peerConnection) {
        peerConnection.close();
        peerConnection = null;
    }

    peerConnection = new RTCPeerConnection({
        iceServers: [],
    });

    localStream.getTracks().forEach(track => peerConnection.addTrack(track, localStream));

    peerConnection.onicecandidate = (event) => {
        if (event.candidate && ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'webrtc-ice', candidate: event.candidate }));
        }
    };

    const offer = await peerConnection.createOffer();
    await peerConnection.setLocalDescription(offer);

    const sendOffer = () => {
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'webrtc-offer', sdp: offer.sdp }));
        } else {
            setTimeout(sendOffer, 500);
        }
    };
    sendOffer();
}

async function resendWebRTCOffer() {
    // Laptop just connected and is asking for the feed — create a fresh offer
    await createAndSendOffer();
}

async function handleWebRTCOffer(msg) {
    peerConnection = new RTCPeerConnection({
        iceServers: [],
    });

    peerConnection.onicecandidate = (event) => {
        if (event.candidate && ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'webrtc-ice', candidate: event.candidate }));
        }
    };

    peerConnection.ontrack = (event) => {
        const videoEl = document.getElementById('video-feed');
        if (videoEl && event.streams[0]) {
            videoEl.srcObject = event.streams[0];
            activateVideoFeed();
        }
    };

    await peerConnection.setRemoteDescription(
        new RTCSessionDescription({ type: 'offer', sdp: msg.sdp })
    );
    const answer = await peerConnection.createAnswer();
    await peerConnection.setLocalDescription(answer);

    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'webrtc-answer', sdp: answer.sdp }));
    }
}

async function handleWebRTCAnswer(msg) {
    if (peerConnection) {
        await peerConnection.setRemoteDescription(
            new RTCSessionDescription({ type: 'answer', sdp: msg.sdp })
        );
    }
}

// ─── AR hit-test ────────────────────────────────────────────────────────────

function onXRFrame(timestamp, frame) {
    if (!isARActive || modelPlaced) return;

    const session = renderer.xr.getSession();
    const refSpace = renderer.xr.getReferenceSpace();

    if (!hitTestSourceRequested) {
        hitTestSourceRequested = true;
        const mySessionId = arSessionId;
        session.requestReferenceSpace('viewer').then((viewerSpace) => {
            if (arSessionId !== mySessionId) return;  // session changed, discard
            return session.requestHitTestSource({ space: viewerSpace });
        }).then((source) => {
            if (!source) return;
            if (arSessionId !== mySessionId) { source.cancel(); return; }
            hitTestSource = source;
        }).catch((e) => {
            console.warn('Hit test source request failed:', e);
            // Allow retry on next frame
            if (arSessionId === mySessionId) hitTestSourceRequested = false;
        });
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

// ─── Resize ─────────────────────────────────────────────────────────────────

function onWindowResize() {
    const vp = document.getElementById('viewport');
    const w = vp.clientWidth;
    const h = vp.clientHeight;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
}

// ─── Render loop ────────────────────────────────────────────────────────────

function animate(timestamp, frame) {
    if (frame) onXRFrame(timestamp, frame);
    if (!isARActive) controls.update();
    renderer.render(scene, camera);
}

// ─── Start ──────────────────────────────────────────────────────────────────

init();
