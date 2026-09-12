/**
 * MediaPipe Pose gesture detection — scale only.
 *
 * Detects a person's arm height to control model scale:
 *   - Arms raised above shoulders → model gets bigger
 *   - Arms lowered below hips → model gets smaller
 *   - Arms at rest (between shoulders and hips) → no change
 *
 * This avoids the problem of needing to bring hands together
 * before spreading them. Raising/lowering arms is unambiguous.
 */

const POSE_PROCESS_INTERVAL = 200; // ~5fps

// Landmark indices
const LEFT_SHOULDER = 11;
const RIGHT_SHOULDER = 12;
const LEFT_WRIST = 15;
const RIGHT_WRIST = 16;
const LEFT_HIP = 23;
const RIGHT_HIP = 24;

let poseLandmarker = null;
let lastPoseTime = 0;

let gestureState = {
    personDetected: false,
    // 'up' = both arms raised above shoulders → scale up
    // 'down' = both arms below hips → scale down
    // 'neutral' = anything else → hold current scale
    action: 'neutral',
};

let offscreenCanvas = null;
let offscreenCtx = null;

export function getGestureState() {
    return gestureState;
}

export async function initPoseDetection() {
    const { PoseLandmarker, FilesetResolver } = await import(
        'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs'
    );

    const filesetResolver = await FilesetResolver.forVisionTasks(
        'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm'
    );

    poseLandmarker = await PoseLandmarker.createFromOptions(filesetResolver, {
        baseOptions: {
            modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
            delegate: 'GPU',
        },
        runningMode: 'VIDEO',
        numPoses: 1,
    });

    offscreenCanvas = document.createElement('canvas');
    offscreenCtx = offscreenCanvas.getContext('2d');

    console.log('Pose detection initialized');
    return true;
}

export function processXRFrame(renderer, frame) {
    if (!poseLandmarker || !frame) return;

    const now = performance.now();
    if (now - lastPoseTime < POSE_PROCESS_INTERVAL) return;

    const session = renderer.xr.getSession();
    const refSpace = renderer.xr.getReferenceSpace();
    const pose = frame.getViewerPose(refSpace);
    if (!pose || !pose.views || pose.views.length === 0) return;

    const view = pose.views[0];
    if (!view.camera) return;

    const glBinding = new XRWebGLBinding(session, renderer.getContext());
    let cameraTexture;
    try {
        cameraTexture = glBinding.getCameraImage(view.camera);
    } catch (e) {
        return;
    }
    if (!cameraTexture) return;

    const gl = renderer.getContext();
    const width = view.camera.width;
    const height = view.camera.height;

    const fb = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, cameraTexture, 0);

    if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) {
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        gl.deleteFramebuffer(fb);
        return;
    }

    const scale = Math.min(1, 320 / Math.max(width, height));
    const w = Math.round(width * scale);
    const h = Math.round(height * scale);

    const pixels = new Uint8Array(w * h * 4);
    gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.deleteFramebuffer(fb);

    offscreenCanvas.width = w;
    offscreenCanvas.height = h;
    const imageData = new ImageData(new Uint8ClampedArray(pixels.buffer), w, h);
    offscreenCtx.putImageData(imageData, 0, 0);

    offscreenCtx.save();
    offscreenCtx.scale(1, -1);
    offscreenCtx.drawImage(offscreenCanvas, 0, -h);
    offscreenCtx.restore();

    try {
        const results = poseLandmarker.detectForVideo(offscreenCanvas, now);
        if (results.landmarks && results.landmarks.length > 0) {
            updateGestureState(results.landmarks[0]);
        } else {
            gestureState.personDetected = false;
            gestureState.action = 'neutral';
        }
    } catch (e) {}

    lastPoseTime = now;
}

export function processVideoFrame(videoElement) {
    if (!poseLandmarker || !videoElement.videoWidth) return;

    const now = performance.now();
    if (now - lastPoseTime < POSE_PROCESS_INTERVAL) return;

    try {
        const results = poseLandmarker.detectForVideo(videoElement, now);
        if (results.landmarks && results.landmarks.length > 0) {
            updateGestureState(results.landmarks[0]);
        } else {
            gestureState.personDetected = false;
            gestureState.action = 'neutral';
        }
    } catch (e) {}

    lastPoseTime = now;
}

function updateGestureState(landmarks) {
    gestureState.personDetected = true;

    const lw = landmarks[LEFT_WRIST];
    const rw = landmarks[RIGHT_WRIST];
    const ls = landmarks[LEFT_SHOULDER];
    const rs = landmarks[RIGHT_SHOULDER];
    const lh = landmarks[LEFT_HIP];
    const rh = landmarks[RIGHT_HIP];

    // Average Y positions (in MediaPipe, Y increases downward)
    const shoulderY = (ls.y + rs.y) / 2;
    const hipY = (lh.y + rh.y) / 2;
    const leftWristY = lw.y;
    const rightWristY = rw.y;

    // Both wrists above shoulders → scale up
    if (leftWristY < shoulderY && rightWristY < shoulderY) {
        gestureState.action = 'up';
    }
    // Both wrists below hips → scale down
    else if (leftWristY > hipY && rightWristY > hipY) {
        gestureState.action = 'down';
    }
    // Anything else → neutral (hold)
    else {
        gestureState.action = 'neutral';
    }
}

/**
 * Apply gesture to model scale.
 * Call each frame. Returns the current scale.
 *
 * When arms are raised: scale grows continuously.
 * When arms are lowered: scale shrinks continuously.
 * When neutral: scale stays put.
 */
const SCALE_SPEED = 0.008; // per frame
const MIN_SCALE = 0.0003;
const MAX_SCALE = 0.008;

export function applyGestures(modelGroup, currentScale) {
    if (!gestureState.personDetected) return currentScale;

    let newScale = currentScale;

    if (gestureState.action === 'up') {
        newScale = currentScale * (1 + SCALE_SPEED);
    } else if (gestureState.action === 'down') {
        newScale = currentScale * (1 - SCALE_SPEED);
    }

    newScale = Math.max(MIN_SCALE, Math.min(MAX_SCALE, newScale));
    modelGroup.scale.setScalar(newScale);
    return newScale;
}
