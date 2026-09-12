/**
 * Body-gesture interpretation for gesture mode.
 *
 * Pure logic, no DOM or Three.js, so it can be unit tested with Node
 * (see tests/test_pose_gestures.mjs).
 *
 * Input is one MediaPipe Pose Landmarker result (33 image landmarks and
 * 33 world landmarks). Output is a model transform: a scale multiplier,
 * a yaw (rotation about the vertical axis) and a tilt (rotation about
 * the camera's view axis), all relative to the model's default pose.
 *
 * Gestures (only while both hands are raised above the hips):
 *   arms apart / together  -> scale up / down
 *   twist torso            -> yaw
 *   lean left / right      -> tilt
 *   both hands above head, held  -> reset
 *
 * Control is clutched like a touch pinch: when the hands come up, the
 * current body pose becomes the baseline, and only changes from that
 * baseline move the model. Dropping the hands leaves the model as is.
 */

export const LM = {
    NOSE: 0,
    L_SHOULDER: 11,
    R_SHOULDER: 12,
    L_WRIST: 15,
    R_WRIST: 16,
    L_HIP: 23,
    R_HIP: 24,
};

export const POSE_CONNECTIONS = [
    [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
    [11, 23], [12, 24], [23, 24],
    [23, 25], [25, 27], [24, 26], [26, 28],
];

export const DEFAULT_OPTIONS = {
    minVisibility: 0.5,
    // Torso length as a multiple of shoulder width, used to estimate hip
    // height when the hips are out of frame.
    torsoToShoulderRatio: 1.4,
    scaleMin: 0.2,
    scaleMax: 5.0,
    yawGain: 2.0,
    tiltGain: 2.0,
    tiltMax: Math.PI / 2,
    // Dead zones on the change from baseline.
    spreadDeadzone: 0.04,    // fraction of the baseline ratio
    angleDeadzone: 0.03,     // radians
    smoothingTauMs: 80,
    engageGraceMs: 300,
    resetHoldMs: 1500,
};

export const DEFAULT_TRANSFORM = { scale: 1, yaw: 0, tilt: 0 };

function wrapAngle(a) {
    while (a > Math.PI) a -= 2 * Math.PI;
    while (a < -Math.PI) a += 2 * Math.PI;
    return a;
}

function applyDeadzone(value, zone) {
    if (Math.abs(value) <= zone) return 0;
    return value - Math.sign(value) * zone;
}

function clamp(v, lo, hi) {
    return Math.max(lo, Math.min(hi, v));
}

function visible(lm, minVisibility) {
    return lm !== undefined && (lm.visibility ?? 1) >= minVisibility;
}

/**
 * Reduce one pose to the few signals the gestures need.
 *
 * @param landmarks       33 normalized image landmarks {x, y, z, visibility}
 * @param worldLandmarks  33 world landmarks in meters {x, y, z}
 * @param aspect          video width / height
 * @param mirrored        true if the video is displayed mirrored (front camera);
 *                        signals are then computed in display coordinates so
 *                        the model follows what the viewer sees on screen
 * @returns null if the upper body is not visible enough, else
 *          { handsRaised, handsOverHead, spread, twist, lean }
 */
export function measurePose(landmarks, worldLandmarks, aspect, mirrored = false, options = DEFAULT_OPTIONS) {
    if (!landmarks || !worldLandmarks) return null;
    const minVis = options.minVisibility;
    const required = [LM.L_SHOULDER, LM.R_SHOULDER, LM.L_WRIST, LM.R_WRIST];
    if (!required.every((i) => visible(landmarks[i], minVis))) return null;

    const sx = mirrored ? -1 : 1;
    // Image points in display space, in units of image height (isotropic).
    const img = (i) => ({
        x: (mirrored ? 1 - landmarks[i].x : landmarks[i].x) * aspect,
        y: landmarks[i].y,
    });
    const world = (i) => ({
        x: worldLandmarks[i].x * sx,
        y: worldLandmarks[i].y,
        z: worldLandmarks[i].z,
    });

    const ls = img(LM.L_SHOULDER), rs = img(LM.R_SHOULDER);
    const lw = img(LM.L_WRIST), rw = img(LM.R_WRIST);
    const shoulderY = (ls.y + rs.y) / 2;
    const shoulderWidthImg = Math.hypot(ls.x - rs.x, ls.y - rs.y);

    let hipY;
    if (visible(landmarks[LM.L_HIP], minVis) && visible(landmarks[LM.R_HIP], minVis)) {
        hipY = (landmarks[LM.L_HIP].y + landmarks[LM.R_HIP].y) / 2;
    } else {
        hipY = shoulderY + options.torsoToShoulderRatio * shoulderWidthImg;
    }
    // Image y grows downward: "above" means smaller y.
    const handsRaised = lw.y < hipY && rw.y < hipY;

    let handsOverHead = false;
    if (visible(landmarks[LM.NOSE], minVis)) {
        const noseY = landmarks[LM.NOSE].y;
        handsOverHead = lw.y < noseY && rw.y < noseY;
    }

    // Scale signal: wrist spread in shoulder widths. World landmarks are
    // metric and centered on the hips, so this is independent of how far
    // the person stands from the camera.
    const wls = world(LM.L_SHOULDER), wrs = world(LM.R_SHOULDER);
    const wlw = world(LM.L_WRIST), wrw = world(LM.R_WRIST);
    const shoulderWidth = Math.hypot(wls.x - wrs.x, wls.y - wrs.y, wls.z - wrs.z);
    if (shoulderWidth < 1e-6) return null;
    const spread = Math.hypot(wlw.x - wrw.x, wlw.y - wrw.y, wlw.z - wrw.z) / shoulderWidth;

    // Twist: heading of the shoulder line in the horizontal plane. MediaPipe
    // z decreases toward the camera, Three.js z increases toward it, so
    // atan2(z_mp, x) grows with a positive Three.js rotation.y.
    const twist = Math.atan2(wls.z - wrs.z, wls.x - wrs.x);

    // Lean: angle of the shoulder line on screen. Image y points down, so
    // negate to make a counter-clockwise lean on screen positive, matching
    // a positive Three.js rotation.z.
    const lean = -Math.atan2(ls.y - rs.y, ls.x - rs.x);

    return { handsRaised, handsOverHead, spread, twist, lean };
}

/**
 * Turns a stream of pose measurements into a smoothed model transform.
 */
export class GestureController {
    constructor(options = {}) {
        this.options = { ...DEFAULT_OPTIONS, ...options };
        this.transform = { ...DEFAULT_TRANSFORM };
        this._engaged = false;
        this._lastRaisedMs = -Infinity;
        this._lastSeenMs = -Infinity;
        this._lastUpdateMs = null;
        this._smoothed = null;
        this._baseline = null;
        this._overHeadSinceMs = null;
    }

    /** Drop the clutch but keep the current transform. */
    release() {
        this._engaged = false;
        this._baseline = null;
        this._smoothed = null;
        this._overHeadSinceMs = null;
    }

    /** Drop the clutch and return the model to its default transform. */
    reset() {
        this.release();
        this.transform = { ...DEFAULT_TRANSFORM };
    }

    /**
     * @param m      result of measurePose(), or null when nobody is detected
     * @param nowMs  monotonic time in milliseconds
     * @returns { status, transform, resetProgress }
     *   status: 'no-person' | 'idle' | 'tracking' | 'resetting'
     *   resetProgress: 0..1 while hands are held over the head
     */
    update(m, nowMs) {
        const o = this.options;
        const dt = this._lastUpdateMs === null ? 0 : Math.max(0, nowMs - this._lastUpdateMs);
        this._lastUpdateMs = nowMs;

        if (m) this._lastSeenMs = nowMs;
        if (m && m.handsRaised) this._lastRaisedMs = nowMs;

        // Short dropouts (a missed detection, a hand dipping for a moment)
        // hold the current transform instead of dropping the clutch.
        if (!m || !m.handsRaised) {
            this._overHeadSinceMs = null;
            const lost = nowMs - this._lastSeenMs > o.engageGraceMs;
            const lowered = nowMs - this._lastRaisedMs > o.engageGraceMs;
            if (lost || lowered) {
                this._engaged = false;
                this._baseline = null;
                this._smoothed = null;
            }
            let status = 'idle';
            if (lost) status = 'no-person';
            else if (this._engaged) status = 'tracking';
            return { status, transform: { ...this.transform }, resetProgress: 0 };
        }

        // Reset: both hands above the head, held.
        if (m.handsOverHead) {
            if (this._overHeadSinceMs === null) this._overHeadSinceMs = nowMs;
            const progress = clamp((nowMs - this._overHeadSinceMs) / o.resetHoldMs, 0, 1);
            if (progress >= 1) {
                this.transform = { ...DEFAULT_TRANSFORM };
                this._overHeadSinceMs = null;
                // Re-baseline once the hands come back down.
                this._baseline = null;
                this._smoothed = null;
                this._engaged = false;
                return { status: 'resetting', transform: { ...this.transform }, resetProgress: 1 };
            }
            // Freeze while the arms travel up, so raising them does not
            // spin or scale the model on the way to a reset.
            return { status: 'resetting', transform: { ...this.transform }, resetProgress: progress };
        }
        if (this._overHeadSinceMs !== null) {
            // Hands came down before the reset finished: re-baseline so the
            // pose after lowering them does not cause a jump.
            this._overHeadSinceMs = null;
            this._baseline = null;
            this._smoothed = null;
        }

        this._smooth(m, dt);

        if (!this._engaged || this._baseline === null) {
            this._engaged = true;
            this._baseline = {
                spread: this._smoothed.spread,
                twist: this._smoothed.twist,
                lean: this._smoothed.lean,
                transform: { ...this.transform },
            };
            return { status: 'tracking', transform: { ...this.transform }, resetProgress: 0 };
        }

        const b = this._baseline;
        const s = this._smoothed;

        let spreadRatio = 1;
        if (b.spread > 1e-6) {
            spreadRatio = 1 + applyDeadzone(s.spread / b.spread - 1, o.spreadDeadzone);
        }
        const dTwist = applyDeadzone(wrapAngle(s.twist - b.twist), o.angleDeadzone);
        const dLean = applyDeadzone(wrapAngle(s.lean - b.lean), o.angleDeadzone);

        this.transform = {
            scale: clamp(b.transform.scale * Math.max(spreadRatio, 0), o.scaleMin, o.scaleMax),
            yaw: wrapAngle(b.transform.yaw + o.yawGain * dTwist),
            tilt: clamp(b.transform.tilt + o.tiltGain * dLean, -o.tiltMax, o.tiltMax),
        };
        return { status: 'tracking', transform: { ...this.transform }, resetProgress: 0 };
    }

    _smooth(m, dt) {
        if (this._smoothed === null) {
            this._smoothed = { spread: m.spread, twist: m.twist, lean: m.lean };
            return;
        }
        const a = 1 - Math.exp(-dt / this.options.smoothingTauMs);
        const s = this._smoothed;
        s.spread += a * (m.spread - s.spread);
        // Smooth angles along the shortest arc.
        s.twist = wrapAngle(s.twist + a * wrapAngle(m.twist - s.twist));
        s.lean = wrapAngle(s.lean + a * wrapAngle(m.lean - s.lean));
    }
}
