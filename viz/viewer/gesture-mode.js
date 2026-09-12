/**
 * Gesture mode: camera feed as background, MediaPipe Pose Landmarker on
 * each new video frame, body gestures mapped to a model transform.
 *
 * MediaPipe is imported lazily so desktop and AR users never download it.
 */

import { GestureController, measurePose, POSE_CONNECTIONS } from './pose-gestures.js';

const WASM_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm';
const MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task';

const STATUS_TEXT = {
    'no-person': 'No person in view',
    'idle': 'Raise both hands to control',
    'tracking': 'Tracking',
};

export class GestureMode {
    /**
     * @param video     <video> element used as the background
     * @param skeleton  <canvas> drawn over the video for the optional skeleton
     * @param onStatus  called with a short status string
     */
    constructor({ video, skeleton, onStatus }) {
        this.video = video;
        this.skeleton = skeleton;
        this.onStatus = onStatus;
        this.controller = new GestureController();
        this.active = false;
        this.showSkeleton = false;
        this.mirrored = false;
        this.facingMode = 'environment';
        this._stream = null;
        this._landmarker = null;
        this._landmarkerPromise = null;
        this._lastVideoTime = -1;
        this._lastTimestamp = 0;
        this._lastStatus = '';
        this._lastLandmarks = null;
    }

    async start() {
        this.active = true;
        this.controller.reset();
        this._setStatus('Starting camera...');

        if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
            this._setStatus('Camera needs HTTPS (or localhost)');
            return false;
        }
        try {
            await this._openCamera();
        } catch (err) {
            console.error(err);
            this._setStatus(err.name === 'NotAllowedError' ? 'Camera access denied' : 'Camera error: ' + err.message);
            return false;
        }
        if (!this.active) {
            // Exited while the permission prompt was open.
            this._closeCamera();
            return false;
        }

        this._setStatus('Loading pose model...');
        try {
            await this._loadLandmarker();
        } catch (err) {
            console.error(err);
            this._setStatus('Could not load pose model');
            return false;
        }
        return this.active;
    }

    stop() {
        this.active = false;
        this._closeCamera();
        this._lastLandmarks = null;
        this._clearSkeleton();
        this._setStatus('');
        // The landmarker is kept so re-entering gesture mode is instant.
    }

    async flipCamera() {
        this.facingMode = this.facingMode === 'environment' ? 'user' : 'environment';
        this._closeCamera();
        // Mirroring may change, which flips the signals: re-baseline.
        this.controller.release();
        try {
            await this._openCamera();
            if (!this.active) this._closeCamera();
        } catch (err) {
            console.error(err);
            this._setStatus('Camera error: ' + err.message);
        }
    }

    setSkeletonVisible(visible) {
        this.showSkeleton = visible;
        if (!visible) this._clearSkeleton();
    }

    resizeSkeleton() {
        const dpr = window.devicePixelRatio || 1;
        this.skeleton.width = Math.round(window.innerWidth * dpr);
        this.skeleton.height = Math.round(window.innerHeight * dpr);
        if (this.showSkeleton && this._lastLandmarks) this._drawSkeleton(this._lastLandmarks);
    }

    /**
     * Run detection if the video has a new frame. Call once per render frame.
     * @returns the current model transform {scale, yaw, tilt}, or null if
     *          there is nothing new to apply
     */
    tick() {
        if (!this.active || !this._landmarker || !this._stream) return null;
        const video = this.video;
        if (video.readyState < 2 || video.currentTime === this._lastVideoTime) return null;
        this._lastVideoTime = video.currentTime;

        // detectForVideo needs strictly increasing timestamps.
        const now = Math.max(performance.now(), this._lastTimestamp + 1);
        this._lastTimestamp = now;

        let result;
        try {
            result = this._landmarker.detectForVideo(video, now);
        } catch (err) {
            console.error(err);
            return null;
        }

        const landmarks = result.landmarks?.[0] ?? null;
        const world = result.worldLandmarks?.[0] ?? null;
        const aspect = video.videoWidth / video.videoHeight;
        const measurement = measurePose(landmarks, world, aspect, this.mirrored, this.controller.options);
        const out = this.controller.update(measurement, now);

        if (out.status === 'resetting') {
            this._setStatus(out.resetProgress >= 1 ? 'Reset' : `Hold to reset ${Math.round(out.resetProgress * 100)}%`);
        } else {
            this._setStatus(STATUS_TEXT[out.status]);
        }

        this._lastLandmarks = landmarks;
        if (this.showSkeleton) this._drawSkeleton(landmarks);

        return out.transform;
    }

    async _openCamera() {
        this._stream = await navigator.mediaDevices.getUserMedia({
            audio: false,
            video: {
                facingMode: { ideal: this.facingMode },
                width: { ideal: 1280 },
                height: { ideal: 720 },
            },
        });
        // Laptop webcams usually report no facingMode; they face the user.
        const settings = this._stream.getVideoTracks()[0]?.getSettings() ?? {};
        this.mirrored = settings.facingMode !== 'environment';
        this.video.classList.toggle('mirrored', this.mirrored);

        this.video.srcObject = this._stream;
        this.video.hidden = false;
        this._lastVideoTime = -1;
        await this.video.play();
    }

    _closeCamera() {
        if (this._stream) {
            for (const track of this._stream.getTracks()) track.stop();
            this._stream = null;
        }
        this.video.pause();
        this.video.srcObject = null;
        this.video.hidden = true;
    }

    _loadLandmarker() {
        if (!this._landmarkerPromise) {
            this._landmarkerPromise = (async () => {
                const { FilesetResolver, PoseLandmarker } = await import('@mediapipe/tasks-vision');
                const fileset = await FilesetResolver.forVisionTasks(WASM_URL);
                const create = (delegate) => PoseLandmarker.createFromOptions(fileset, {
                    baseOptions: { modelAssetPath: MODEL_URL, delegate },
                    runningMode: 'VIDEO',
                    numPoses: 1,
                });
                try {
                    return await create('GPU');
                } catch (err) {
                    console.warn('GPU delegate unavailable, using CPU', err);
                    return await create('CPU');
                }
            })();
            // Allow a retry if loading failed (e.g. offline).
            this._landmarkerPromise.catch(() => { this._landmarkerPromise = null; });
        }
        return this._landmarkerPromise.then((landmarker) => { this._landmarker = landmarker; });
    }

    _setStatus(text) {
        if (text === this._lastStatus) return;
        this._lastStatus = text;
        this.onStatus(text);
    }

    _clearSkeleton() {
        const ctx = this.skeleton.getContext('2d');
        ctx.clearRect(0, 0, this.skeleton.width, this.skeleton.height);
    }

    _drawSkeleton(landmarks) {
        this._clearSkeleton();
        const video = this.video;
        if (!landmarks || !video.videoWidth) return;

        // Match the video's object-fit: cover crop.
        const cw = this.skeleton.width, ch = this.skeleton.height;
        const s = Math.max(cw / video.videoWidth, ch / video.videoHeight);
        const dw = video.videoWidth * s, dh = video.videoHeight * s;
        const ox = (cw - dw) / 2, oy = (ch - dh) / 2;
        const px = (lm) => ox + (this.mirrored ? 1 - lm.x : lm.x) * dw;
        const py = (lm) => oy + lm.y * dh;
        const minVis = this.controller.options.minVisibility;
        const ok = (lm) => lm && (lm.visibility ?? 1) >= minVis;

        const ctx = this.skeleton.getContext('2d');
        const dpr = window.devicePixelRatio || 1;
        ctx.lineWidth = 2 * dpr;
        ctx.strokeStyle = 'rgba(210, 210, 210, 0.7)';
        ctx.fillStyle = 'rgba(210, 210, 210, 0.9)';

        ctx.beginPath();
        for (const [a, b] of POSE_CONNECTIONS) {
            if (!ok(landmarks[a]) || !ok(landmarks[b])) continue;
            ctx.moveTo(px(landmarks[a]), py(landmarks[a]));
            ctx.lineTo(px(landmarks[b]), py(landmarks[b]));
        }
        ctx.stroke();

        const used = new Set(POSE_CONNECTIONS.flat());
        for (const i of used) {
            if (!ok(landmarks[i])) continue;
            ctx.beginPath();
            ctx.arc(px(landmarks[i]), py(landmarks[i]), 3 * dpr, 0, 2 * Math.PI);
            ctx.fill();
        }
    }
}
