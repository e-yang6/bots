/**
 * MediaPipe Pose gesture detection for controlling the 3D model.
 *
 * Reads the XR camera frame (via camera-access feature), feeds it to
 * MediaPipe Pose Landmarker at ~5fps, and maps body gestures to
 * scale/rotate/move commands.
 *
 * Gestures (detected on a person standing in frame):
 *   - Arms spread apart / together → scale up / down
 *   - Body twist (shoulders rotate) → rotate model
 *   - Both hands raised above head → reset scale and rotation
 */

const POSE_PROCESS_INTERVAL = 200; // ms between pose detections (~5fps)
const SMOOTHING = 0.3; // lerp factor for gesture values (0 = ignore new, 1 = instant)

// Landmark indices (MediaPipe Pose 33-point model)
const LEFT_SHOULDER = 11;
const RIGHT_SHOULDER = 12;
const LEFT_WRIST = 15;
const RIGHT_WRIST = 16;
const LEFT_ELBOW = 13;
const RIGHT_ELBOW = 14;
const NOSE = 0;
const LEFT_HIP = 23;
const RIGHT_HIP = 24;

let poseLandmarker = null;
let lastPoseTime = 0;
let gestureState = {
    armSpan: null,       // normalized distance between wrists
    shoulderAngle: null, // rotation of shoulder line
    handsAboveHead: false,
    personDetected: false,
};

// Offscreen canvas for reading XR camera frames
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

/**
 * Process an XR camera frame for pose detection.
 * Call this from the render loop with the XR frame's camera view.
 */
export function processXRFrame(renderer, frame) {
    if (!poseLandmarker || !frame) return;

    const now = performance.now();
    if (now - lastPoseTime < POSE_PROCESS_INTERVAL) return;

    const session = renderer.xr.getSession();
    const refSpace = renderer.xr.getReferenceSpace();
    const pose = frame.getViewerPose(refSpace);
    if (!pose || !pose.views || pose.views.length === 0) return;

    const view = pose.views[0];
    if (!view.camera) return; // camera-access not available

    const glBinding = new XRWebGLBinding(session, renderer.getContext());
    let cameraTexture;
    try {
        cameraTexture = glBinding.getCameraImage(view.camera);
    } catch (e) {
        return; // camera-access not supported on this device
    }
    if (!cameraTexture) return;

    // Read the WebGL texture into our offscreen canvas
    const gl = renderer.getContext();
    const width = view.camera.width;
    const height = view.camera.height;

    // Create framebuffer to read from the camera texture
    const fb = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, cameraTexture, 0);

    if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) {
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        gl.deleteFramebuffer(fb);
        return;
    }

    // Read pixels — use a smaller resolution for performance
    const scale = Math.min(1, 320 / Math.max(width, height));
    const w = Math.round(width * scale);
    const h = Math.round(height * scale);

    const pixels = new Uint8Array(w * h * 4);
    gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.deleteFramebuffer(fb);

    // Put into offscreen canvas for MediaPipe
    offscreenCanvas.width = w;
    offscreenCanvas.height = h;
    const imageData = new ImageData(new Uint8ClampedArray(pixels.buffer), w, h);
    offscreenCtx.putImageData(imageData, 0, 0);

    // Flip vertically (WebGL reads bottom-up)
    offscreenCtx.save();
    offscreenCtx.scale(1, -1);
    offscreenCtx.drawImage(offscreenCanvas, 0, -h);
    offscreenCtx.restore();

    // Run pose detection
    try {
        const results = poseLandmarker.detectForVideo(offscreenCanvas, now);
        if (results.landmarks && results.landmarks.length > 0) {
            updateGestureState(results.landmarks[0]);
        } else {
            gestureState.personDetected = false;
        }
    } catch (e) {
        // Silently skip detection errors
    }

    lastPoseTime = now;
}

/**
 * Fallback: process a regular video element (for non-XR camera-access).
 */
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
        }
    } catch (e) {
        // skip
    }

    lastPoseTime = now;
}

function updateGestureState(landmarks) {
    gestureState.personDetected = true;

    const lw = landmarks[LEFT_WRIST];
    const rw = landmarks[RIGHT_WRIST];
    const ls = landmarks[LEFT_SHOULDER];
    const rs = landmarks[RIGHT_SHOULDER];
    const nose = landmarks[NOSE];

    // Arm span: distance between wrists, normalized by shoulder width
    const shoulderWidth = Math.sqrt(
        (ls.x - rs.x) ** 2 + (ls.y - rs.y) ** 2
    );
    const wristDist = Math.sqrt(
        (lw.x - rw.x) ** 2 + (lw.y - rw.y) ** 2
    );
    const armSpan = shoulderWidth > 0.01 ? wristDist / shoulderWidth : 1;

    // Shoulder angle: rotation of the line between shoulders
    const shoulderAngle = Math.atan2(ls.y - rs.y, ls.x - rs.x);

    // Hands above head: both wrists above nose
    const handsAboveHead = lw.y < nose.y - 0.1 && rw.y < nose.y - 0.1;

    // Smooth values
    if (gestureState.armSpan === null) {
        gestureState.armSpan = armSpan;
        gestureState.shoulderAngle = shoulderAngle;
    } else {
        gestureState.armSpan += (armSpan - gestureState.armSpan) * SMOOTHING;
        gestureState.shoulderAngle += angleDiff(shoulderAngle, gestureState.shoulderAngle) * SMOOTHING;
    }

    gestureState.handsAboveHead = handsAboveHead;
}

function angleDiff(a, b) {
    let d = a - b;
    while (d > Math.PI) d -= 2 * Math.PI;
    while (d < -Math.PI) d += 2 * Math.PI;
    return d;
}

/**
 * Apply gesture state to a model group.
 * Call each frame after processXRFrame/processVideoFrame.
 *
 * baseScale: the initial scale (e.g., 0.001 for mm→m)
 * Returns the new scale value.
 */
export function applyGestures(modelGroup, baseScale) {
    if (!gestureState.personDetected) return baseScale;

    // Reset on hands above head
    if (gestureState.handsAboveHead) {
        modelGroup.rotation.y = 0;
        modelGroup.scale.setScalar(baseScale);
        return baseScale;
    }

    // Scale: arm span of ~1 = neutral (arms at shoulder width).
    // Spread arms = bigger, bring together = smaller.
    // Map armSpan [0.3, 4.0] → scale multiplier [0.3, 3.0]
    const span = gestureState.armSpan || 1;
    const scaleMultiplier = Math.max(0.3, Math.min(3.0, span / 1.2));
    const newScale = baseScale * scaleMultiplier;
    modelGroup.scale.setScalar(newScale);

    // Rotate: shoulder twist maps to Y rotation
    // Neutral shoulder angle is ~0 (horizontal). Twist maps to rotation.
    if (gestureState.shoulderAngle !== null) {
        modelGroup.rotation.y = gestureState.shoulderAngle * 3;
    }

    return newScale;
}
