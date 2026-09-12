/**
 * Body tracking and gesture interpretation for gesture mode.
 *
 * Pure logic, no DOM or Three.js, so it can be unit tested with Node
 * (see tests/test_pose_gestures.mjs).
 *
 * Input is one MediaPipe Pose Landmarker result (33 image landmarks and
 * 33 world landmarks). Two things come out of it:
 *
 * 1. A body anchor (measureBody + BodyAnchor). The person in front of the
 *    camera is the AR anchor: the aorta is drawn in their chest, at life
 *    size, turned and tilted with their torso. Panning the phone, walking
 *    closer, or walking around the person keeps the model attached to
 *    them, so it behaves like a real object and can be viewed from any side.
 *
 * 2. A scale gesture (measureHands + GestureController), active only while
 *    both hands are raised above the hips:
 *      arms apart / together          -> scale up / down
 *      both hands above head, held    -> reset scale
 *    Control is clutched like a touch pinch: when the hands come up, the
 *    current spread becomes the baseline. Dropping the hands keeps the scale.
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

    // Adult body proportions used to turn image size into life size.
    shoulderWidthM: 0.38,     // between shoulder landmarks
    torsoLengthM: 0.50,       // shoulder midpoint to hip midpoint
    // Where the aorta centroid sits: this fraction of the way from the
    // shoulder midpoint to the hip midpoint.
    aortaTorsoFraction: 0.45,
    // Floor on |cos(heading)| when correcting shoulder width for turning,
    // so a side-on view does not blow up the scale.
    minHeadingCos: 0.35,
    // Set true if the model turns the opposite way to the person.
    invertHeading: false,

    // Anchor filtering (One Euro filter: min cutoff in Hz, beta per unit/s).
    positionMinCutoff: 1.0,
    positionBeta: 8.0,
    angleMinCutoff: 0.8,
    angleBeta: 0.6,
    sizeMinCutoff: 0.6,
    sizeBeta: 1.0,
    anchorLostMs: 500,

    // Scale gesture.
    scaleMin: 0.2,
    scaleMax: 5.0,
    spreadDeadzone: 0.04,     // fraction of the baseline spread
    smoothingTauMs: 80,
    engageGraceMs: 300,
    resetHoldMs: 1500,
};

export const DEFAULT_TRANSFORM = { scale: 1 };

export function wrapAngle(a) {
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

/** Image point in display space, in units of image height (isotropic). */
function displayPoint(landmarks, i, aspect, mirrored) {
    return {
        x: (mirrored ? 1 - landmarks[i].x : landmarks[i].x) * aspect,
        y: landmarks[i].y,
    };
}

/**
 * Where the person is, how big they appear, and which way their torso faces.
 *
 * @param landmarks       33 normalized image landmarks {x, y, z, visibility}
 * @param worldLandmarks  33 world landmarks in meters {x, y, z}
 * @param aspect          video width / height
 * @param mirrored        true if the video is displayed mirrored (front camera)
 * @returns null if the shoulders are not visible, else
 *   x, y             aorta anchor in normalized display coordinates (0..1)
 *   heightPerMeter   image heights per real-world meter at the person
 *   heading          torso rotation about vertical, radians; 0 = facing the
 *                    camera, positive = Three.js rotation.y direction, as seen
 *                    on screen
 *   roll             shoulder-line tilt on screen, radians, counter-clockwise
 *                    positive (Three.js rotation.z)
 */
export function measureBody(landmarks, worldLandmarks, aspect, mirrored = false, options = DEFAULT_OPTIONS) {
    if (!landmarks || !worldLandmarks) return null;
    const o = options;
    if (!visible(landmarks[LM.L_SHOULDER], o.minVisibility) || !visible(landmarks[LM.R_SHOULDER], o.minVisibility)) {
        return null;
    }

    // Heading from world landmarks (metric, camera-aligned axes). MediaPipe
    // z decreases toward the camera, Three.js z increases toward it, so
    // atan2(z_mp, x) grows with a positive Three.js rotation.y. A mirrored
    // display shows the mirror image, which turns the other way.
    const wl = worldLandmarks[LM.L_SHOULDER], wr = worldLandmarks[LM.R_SHOULDER];
    let heading = Math.atan2(wl.z - wr.z, wl.x - wr.x);
    if (mirrored !== o.invertHeading) heading = wrapAngle(-heading);

    const ls = displayPoint(landmarks, LM.L_SHOULDER, aspect, mirrored);
    const rs = displayPoint(landmarks, LM.R_SHOULDER, aspect, mirrored);
    const shoulderMid = { x: (ls.x + rs.x) / 2, y: (ls.y + rs.y) / 2 };
    const shoulderPx = Math.hypot(ls.x - rs.x, ls.y - rs.y);
    if (shoulderPx < 1e-6) return null;

    // Shoulder line as seen on screen, oriented left to right, so the roll
    // is the same whether the person faces toward or away from the camera.
    let ux = (ls.x - rs.x) / shoulderPx, uy = (ls.y - rs.y) / shoulderPx;
    if (ux < 0) { ux = -ux; uy = -uy; }
    // Image y points down: negate so counter-clockwise on screen is positive.
    const roll = -Math.atan2(uy, ux);

    const hipsVisible = visible(landmarks[LM.L_HIP], o.minVisibility) && visible(landmarks[LM.R_HIP], o.minVisibility);
    let heightPerMeter, anchor;
    if (hipsVisible) {
        const lh = displayPoint(landmarks, LM.L_HIP, aspect, mirrored);
        const rh = displayPoint(landmarks, LM.R_HIP, aspect, mirrored);
        const hipMid = { x: (lh.x + rh.x) / 2, y: (lh.y + rh.y) / 2 };
        // Torso length does not shrink when the person turns.
        heightPerMeter = Math.hypot(hipMid.x - shoulderMid.x, hipMid.y - shoulderMid.y) / o.torsoLengthM;
        anchor = {
            x: shoulderMid.x + o.aortaTorsoFraction * (hipMid.x - shoulderMid.x),
            y: shoulderMid.y + o.aortaTorsoFraction * (hipMid.y - shoulderMid.y),
        };
    } else {
        // Hips out of frame: use shoulder width, corrected for turning, and
        // go down the torso perpendicular to the shoulder line.
        const cos = Math.max(Math.abs(Math.cos(heading)), o.minHeadingCos);
        heightPerMeter = shoulderPx / (o.shoulderWidthM * cos);
        const down = o.aortaTorsoFraction * o.torsoLengthM * heightPerMeter;
        anchor = { x: shoulderMid.x - uy * down, y: shoulderMid.y + ux * down };
    }
    if (!(heightPerMeter > 0)) return null;

    return { x: anchor.x / aspect, y: anchor.y, heightPerMeter, heading, roll };
}

/**
 * Hand signals for the scale gesture.
 *
 * @returns null if shoulders or wrists are not visible, else
 *          { handsRaised, handsOverHead, spread }
 */
export function measureHands(landmarks, worldLandmarks, aspect, mirrored = false, options = DEFAULT_OPTIONS) {
    if (!landmarks || !worldLandmarks) return null;
    const minVis = options.minVisibility;
    const required = [LM.L_SHOULDER, LM.R_SHOULDER, LM.L_WRIST, LM.R_WRIST];
    if (!required.every((i) => visible(landmarks[i], minVis))) return null;

    const ls = displayPoint(landmarks, LM.L_SHOULDER, aspect, mirrored);
    const rs = displayPoint(landmarks, LM.R_SHOULDER, aspect, mirrored);
    const lw = displayPoint(landmarks, LM.L_WRIST, aspect, mirrored);
    const rw = displayPoint(landmarks, LM.R_WRIST, aspect, mirrored);
    const shoulderY = (ls.y + rs.y) / 2;
    const shoulderWidthImg = Math.hypot(ls.x - rs.x, ls.y - rs.y);

    let hipY;
    if (visible(landmarks[LM.L_HIP], minVis) && visible(landmarks[LM.R_HIP], minVis)) {
        hipY = (landmarks[LM.L_HIP].y + landmarks[LM.R_HIP].y) / 2;
    } else {
        hipY = shoulderY + (options.torsoLengthM / options.shoulderWidthM) * shoulderWidthImg;
    }
    // Image y grows downward: "above" means smaller y.
    const handsRaised = lw.y < hipY && rw.y < hipY;

    let handsOverHead = false;
    if (visible(landmarks[LM.NOSE], minVis)) {
        const noseY = landmarks[LM.NOSE].y;
        handsOverHead = lw.y < noseY && rw.y < noseY;
    }

    // Wrist spread in shoulder widths, from metric world landmarks, so it
    // does not depend on distance to the camera.
    const w = worldLandmarks;
    const d = (a, b) => Math.hypot(w[a].x - w[b].x, w[a].y - w[b].y, w[a].z - w[b].z);
    const shoulderWidth = d(LM.L_SHOULDER, LM.R_SHOULDER);
    if (shoulderWidth < 1e-6) return null;
    const spread = d(LM.L_WRIST, LM.R_WRIST) / shoulderWidth;

    return { handsRaised, handsOverHead, spread };
}

/** One Euro filter (Casiez et al. 2012): smooth at rest, responsive in motion. */
export class OneEuroFilter {
    constructor(minCutoff, beta, dCutoff = 1.0) {
        this.minCutoff = minCutoff;
        this.beta = beta;
        this.dCutoff = dCutoff;
        this.reset();
    }

    reset() {
        this._x = null;
        this._dx = 0;
        this._t = null;
    }

    filter(x, tSec) {
        if (this._x === null) {
            this._x = x;
            this._t = tSec;
            return x;
        }
        const dt = tSec - this._t;
        if (dt <= 0) return this._x;
        this._t = tSec;
        const alpha = (cutoff) => 1 / (1 + 1 / (2 * Math.PI * cutoff * dt));
        const dx = (x - this._x) / dt;
        this._dx += alpha(this.dCutoff) * (dx - this._dx);
        const cutoff = this.minCutoff + this.beta * Math.abs(this._dx);
        this._x += alpha(cutoff) * (x - this._x);
        return this._x;
    }
}

/**
 * Filters measureBody() output over time and holds the last pose through
 * short detection dropouts.
 */
export class BodyAnchor {
    constructor(options = {}) {
        this.options = { ...DEFAULT_OPTIONS, ...options };
        const o = this.options;
        this._fx = new OneEuroFilter(o.positionMinCutoff, o.positionBeta);
        this._fy = new OneEuroFilter(o.positionMinCutoff, o.positionBeta);
        this._fSize = new OneEuroFilter(o.sizeMinCutoff, o.sizeBeta);
        this._fHeading = new OneEuroFilter(o.angleMinCutoff, o.angleBeta);
        this._fRoll = new OneEuroFilter(o.angleMinCutoff, o.angleBeta);
        this.reset();
    }

    reset() {
        for (const f of [this._fx, this._fy, this._fSize, this._fHeading, this._fRoll]) f.reset();
        this._lastSeenMs = -Infinity;
        this._heading = null;   // unwrapped, so filtering never jumps at ±π
        this._state = null;
    }

    /**
     * @param body   result of measureBody(), or null
     * @param nowMs  monotonic time in milliseconds
     * @returns { visible, x, y, heightPerMeter, heading, roll }
     */
    update(body, nowMs) {
        const o = this.options;
        if (!body) {
            if (this._state && nowMs - this._lastSeenMs <= o.anchorLostMs) {
                return { visible: true, ...this._state };
            }
            this.reset();
            return { visible: false };
        }

        this._lastSeenMs = nowMs;
        const t = nowMs / 1000;
        this._heading = this._heading === null
            ? body.heading
            : this._heading + wrapAngle(body.heading - this._heading);

        this._state = {
            x: this._fx.filter(body.x, t),
            y: this._fy.filter(body.y, t),
            // Filter size in log space so growing and shrinking behave alike.
            heightPerMeter: Math.exp(this._fSize.filter(Math.log(body.heightPerMeter), t)),
            heading: wrapAngle(this._fHeading.filter(this._heading, t)),
            roll: this._fRoll.filter(body.roll, t),
        };
        return { visible: true, ...this._state };
    }
}

/**
 * Turns a stream of hand measurements into a smoothed scale multiplier.
 */
export class GestureController {
    constructor(options = {}) {
        this.options = { ...DEFAULT_OPTIONS, ...options };
        this.transform = { ...DEFAULT_TRANSFORM };
        this._engaged = false;
        this._lastRaisedMs = -Infinity;
        this._lastSeenMs = -Infinity;
        this._lastUpdateMs = null;
        this._smoothedSpread = null;
        this._baseline = null;
        this._overHeadSinceMs = null;
    }

    /** Drop the clutch but keep the current transform. */
    release() {
        this._engaged = false;
        this._baseline = null;
        this._smoothedSpread = null;
        this._overHeadSinceMs = null;
    }

    /** Drop the clutch and return to the default transform. */
    reset() {
        this.release();
        this.transform = { ...DEFAULT_TRANSFORM };
    }

    /**
     * @param m      result of measureHands(), or null when hands are not visible
     * @param nowMs  monotonic time in milliseconds
     * @returns { status, transform, resetProgress }
     *   status: 'no-hands' | 'idle' | 'tracking' | 'resetting'
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
            if (lost || lowered) this.release();
            let status = 'idle';
            if (lost) status = 'no-hands';
            else if (this._engaged) status = 'tracking';
            return { status, transform: { ...this.transform }, resetProgress: 0 };
        }

        // Reset: both hands above the head, held.
        if (m.handsOverHead) {
            if (this._overHeadSinceMs === null) this._overHeadSinceMs = nowMs;
            const progress = clamp((nowMs - this._overHeadSinceMs) / o.resetHoldMs, 0, 1);
            if (progress >= 1) {
                this.reset();
                return { status: 'resetting', transform: { ...this.transform }, resetProgress: 1 };
            }
            // Freeze while the arms travel up, so raising them does not
            // scale the model on the way to a reset.
            return { status: 'resetting', transform: { ...this.transform }, resetProgress: progress };
        }
        if (this._overHeadSinceMs !== null) {
            // Hands came down before the reset finished: re-baseline so the
            // pose after lowering them does not cause a jump.
            this.release();
        }

        if (this._smoothedSpread === null) {
            this._smoothedSpread = m.spread;
        } else {
            const a = 1 - Math.exp(-dt / o.smoothingTauMs);
            this._smoothedSpread += a * (m.spread - this._smoothedSpread);
        }

        if (!this._engaged || this._baseline === null) {
            this._engaged = true;
            this._baseline = { spread: this._smoothedSpread, scale: this.transform.scale };
            return { status: 'tracking', transform: { ...this.transform }, resetProgress: 0 };
        }

        const b = this._baseline;
        let ratio = 1;
        if (b.spread > 1e-6) {
            ratio = 1 + applyDeadzone(this._smoothedSpread / b.spread - 1, o.spreadDeadzone);
        }
        this.transform = { scale: clamp(b.scale * Math.max(ratio, 0), o.scaleMin, o.scaleMax) };
        return { status: 'tracking', transform: { ...this.transform }, resetProgress: 0 };
    }
}
