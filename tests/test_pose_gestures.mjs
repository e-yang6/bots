// Unit tests for viz/viewer/pose-gestures.js.
// Run with: node --test tests/test_pose_gestures.mjs

import { test } from 'node:test';
import assert from 'node:assert/strict';

import {
    GestureController,
    LM,
    measurePose,
} from '../viz/viewer/pose-gestures.js';

const ASPECT = 16 / 9;
const deg = (d) => (d * Math.PI) / 180;

/**
 * Build a synthetic pose for a person facing the camera.
 *
 * spread: wrist distance in shoulder widths
 * twist:  torso rotation about vertical (radians)
 * lean:   shoulder-line tilt on screen (radians, counter-clockwise positive)
 * hands:  'down' | 'up' | 'overhead'
 * hipsVisible: whether hip landmarks are in frame
 */
function makePose({ spread = 1.5, twist = 0, lean = 0, hands = 'up', hipsVisible = true } = {}) {
    const image = Array.from({ length: 33 }, () => ({ x: 0.5, y: 0.5, z: 0, visibility: 0.0 }));
    const world = Array.from({ length: 33 }, () => ({ x: 0, y: 0, z: 0 }));
    const set = (i, img, w) => {
        image[i] = { ...img, z: 0, visibility: 0.99 };
        world[i] = w;
    };

    const halfShoulderM = 0.18;
    const shoulderYImg = 0.35;
    const cx = 0.5;
    // Half shoulder width in image-x units (image height = 1 in isotropic units).
    const halfShoulderImgIso = 0.1;

    // Person faces camera, rear camera (not mirrored): their left shoulder
    // appears on the image right. Image y points down.
    const shoulderImg = (side) => {
        const s = side === 'L' ? 1 : -1;
        const dxIso = s * halfShoulderImgIso * Math.cos(lean);
        const dy = -s * halfShoulderImgIso * Math.sin(lean);
        return { x: cx + dxIso / ASPECT, y: shoulderYImg + dy };
    };
    // World: x right in image, z = depth (smaller is closer). Twist rotates
    // the shoulder line in the x-z plane.
    const shoulderWorld = (side) => {
        const s = side === 'L' ? 1 : -1;
        return {
            x: s * halfShoulderM * Math.cos(twist),
            y: -0.45,
            z: s * halfShoulderM * Math.sin(twist),
        };
    };

    set(LM.L_SHOULDER, shoulderImg('L'), shoulderWorld('L'));
    set(LM.R_SHOULDER, shoulderImg('R'), shoulderWorld('R'));
    set(LM.NOSE, { x: cx, y: 0.2 }, { x: 0, y: -0.6, z: -0.1 });

    const wristY = { down: 0.8, up: 0.4, overhead: 0.05 }[hands];
    const halfSpreadM = (spread * 2 * halfShoulderM) / 2;
    const halfSpreadImgIso = (spread * 2 * halfShoulderImgIso) / 2;
    set(LM.L_WRIST, { x: cx + halfSpreadImgIso / ASPECT, y: wristY }, { x: halfSpreadM, y: -0.3, z: -0.2 });
    set(LM.R_WRIST, { x: cx - halfSpreadImgIso / ASPECT, y: wristY }, { x: -halfSpreadM, y: -0.3, z: -0.2 });

    set(LM.L_HIP, { x: cx + 0.05, y: 0.65 }, { x: 0.1, y: 0, z: 0 });
    set(LM.R_HIP, { x: cx - 0.05, y: 0.65 }, { x: -0.1, y: 0, z: 0 });
    if (!hipsVisible) {
        image[LM.L_HIP].visibility = 0.1;
        image[LM.R_HIP].visibility = 0.1;
    }
    return { image, world };
}

const measure = (opts, mirrored = false) => {
    const p = makePose(opts);
    return measurePose(p.image, p.world, ASPECT, mirrored);
};

/** Feed the same pose for `ms` milliseconds at 30 fps; return the last output. */
function hold(ctrl, opts, startMs, ms, mirrored = false) {
    let out;
    let t = startMs;
    for (; t <= startMs + ms; t += 33) {
        out = ctrl.update(measure(opts, mirrored), t);
    }
    return { out, t };
}

// ─── measurePose ────────────────────────────────────────────────────────

test('measurePose returns null when upper body is not visible', () => {
    const p = makePose();
    p.image[LM.L_WRIST].visibility = 0.1;
    assert.equal(measurePose(p.image, p.world, ASPECT), null);
    assert.equal(measurePose(null, null, ASPECT), null);
});

test('measurePose detects hands raised vs down', () => {
    assert.equal(measure({ hands: 'up' }).handsRaised, true);
    assert.equal(measure({ hands: 'down' }).handsRaised, false);
    assert.equal(measure({ hands: 'up' }).handsOverHead, false);
    assert.equal(measure({ hands: 'overhead' }).handsOverHead, true);
});

test('measurePose estimates hip height when hips are out of frame', () => {
    assert.equal(measure({ hands: 'up', hipsVisible: false }).handsRaised, true);
    assert.equal(measure({ hands: 'down', hipsVisible: false }).handsRaised, false);
});

test('measurePose spread is in shoulder widths', () => {
    assert.ok(Math.abs(measure({ spread: 1 }).spread - 1) < 1e-9);
    assert.ok(Math.abs(measure({ spread: 3 }).spread - 3) < 1e-9);
});

test('measurePose twist and lean track the body', () => {
    const base = measure({});
    const twisted = measure({ twist: deg(20) });
    const leaned = measure({ lean: deg(15) });
    assert.ok(Math.abs(twisted.twist - base.twist - deg(20)) < 1e-9);
    assert.ok(Math.abs(leaned.lean - base.lean - deg(15)) < 1e-9);
});

test('mirroring flips twist and lean direction', () => {
    // Mirrored, the shoulder line points the other way (angle near ±π), so
    // compare wrapped differences, as the controller does.
    const wrap = (a) => Math.atan2(Math.sin(a), Math.cos(a));
    const d = (opts, mirrored) => {
        const b = measure({}, mirrored), m = measure(opts, mirrored);
        return { twist: wrap(m.twist - b.twist), lean: wrap(m.lean - b.lean) };
    };
    const normal = d({ twist: deg(20), lean: deg(15) }, false);
    const mirrored = d({ twist: deg(20), lean: deg(15) }, true);
    assert.ok(Math.abs(normal.twist + mirrored.twist) < 1e-9);
    assert.ok(Math.abs(normal.lean + mirrored.lean) < 1e-9);
});

// ─── GestureController ──────────────────────────────────────────────────

test('no person gives no-person status and default transform', () => {
    const ctrl = new GestureController();
    const out = ctrl.update(null, 1000);
    assert.equal(out.status, 'no-person');
    assert.deepEqual(out.transform, { scale: 1, yaw: 0, tilt: 0 });
});

test('hands down is idle and does not move the model', () => {
    const ctrl = new GestureController();
    const { out } = hold(ctrl, { hands: 'down', spread: 4, twist: deg(40) }, 0, 1000);
    assert.equal(out.status, 'idle');
    assert.deepEqual(out.transform, { scale: 1, yaw: 0, tilt: 0 });
});

test('raising hands engages without jumping', () => {
    const ctrl = new GestureController();
    const { out } = hold(ctrl, { hands: 'up', spread: 3, twist: deg(30), lean: deg(10) }, 0, 1000);
    assert.equal(out.status, 'tracking');
    assert.ok(Math.abs(out.transform.scale - 1) < 1e-6);
    assert.ok(Math.abs(out.transform.yaw) < 1e-6);
    assert.ok(Math.abs(out.transform.tilt) < 1e-6);
});

test('arms apart scales up, arms together scales down', () => {
    const ctrl = new GestureController();
    let { t } = hold(ctrl, { spread: 2 }, 0, 500);
    let r = hold(ctrl, { spread: 4 }, t, 1000);
    assert.ok(r.out.transform.scale > 1.8, `scale ${r.out.transform.scale}`);

    r = hold(ctrl, { spread: 1 }, r.t, 1000);
    assert.ok(r.out.transform.scale < 0.6, `scale ${r.out.transform.scale}`);
});

test('scale is clamped', () => {
    const ctrl = new GestureController();
    let { t } = hold(ctrl, { spread: 0.5 }, 0, 500);
    const r = hold(ctrl, { spread: 20 }, t, 1000);
    assert.equal(r.out.transform.scale, ctrl.options.scaleMax);
});

test('twist rotates yaw with gain, lean tilts', () => {
    const ctrl = new GestureController();
    let { t } = hold(ctrl, {}, 0, 500);
    const r = hold(ctrl, { twist: deg(20), lean: deg(10) }, t, 1500);
    const { yawGain, tiltGain, angleDeadzone } = ctrl.options;
    assert.ok(Math.abs(r.out.transform.yaw - yawGain * (deg(20) - angleDeadzone)) < 1e-3);
    assert.ok(Math.abs(r.out.transform.tilt - tiltGain * (deg(10) - angleDeadzone)) < 1e-3);
});

test('small jitter inside the dead zone does not move the model', () => {
    const ctrl = new GestureController();
    let { t } = hold(ctrl, {}, 0, 500);
    const r = hold(ctrl, { twist: deg(1), lean: deg(-1), spread: 1.52 }, t, 1000);
    assert.deepEqual(r.out.transform, { scale: 1, yaw: 0, tilt: 0 });
});

test('dropping hands holds the transform, re-raising re-baselines', () => {
    const ctrl = new GestureController();
    let r = hold(ctrl, { spread: 2 }, 0, 500);
    r = hold(ctrl, { spread: 4, twist: deg(20) }, r.t, 1000);
    const held = r.out.transform;

    // Drop hands in a different pose: nothing changes.
    r = hold(ctrl, { hands: 'down', spread: 1, twist: 0 }, r.t, 1000);
    assert.equal(r.out.status, 'idle');
    assert.deepEqual(r.out.transform, held);

    // Raise again in a new pose: no jump, continues from held transform.
    r = hold(ctrl, { spread: 1, twist: deg(-30) }, r.t, 1000);
    assert.ok(Math.abs(r.out.transform.scale - held.scale) < 1e-6);
    assert.ok(Math.abs(r.out.transform.yaw - held.yaw) < 1e-6);
});

test('brief detection dropout keeps tracking', () => {
    const ctrl = new GestureController();
    let r = hold(ctrl, { spread: 2 }, 0, 500);
    r = hold(ctrl, { spread: 4 }, r.t, 1000);
    const before = r.out.transform;
    const out = ctrl.update(null, r.t + 100);
    assert.equal(out.status, 'tracking');
    assert.deepEqual(out.transform, before);
    // Continuing with the same pose does not jump.
    const after = ctrl.update(measure({ spread: 4 }), r.t + 150);
    assert.ok(Math.abs(after.transform.scale - before.scale) < 0.05);
});

test('hands over head held resets, released early does not', () => {
    const ctrl = new GestureController();
    let r = hold(ctrl, { spread: 2 }, 0, 500);
    r = hold(ctrl, { spread: 4, twist: deg(30) }, r.t, 1000);
    const moved = r.out.transform;
    assert.ok(moved.scale > 1.5);

    // Short hold: progress but no reset.
    r = hold(ctrl, { hands: 'overhead' }, r.t, 700);
    assert.equal(r.out.status, 'resetting');
    assert.ok(r.out.resetProgress > 0 && r.out.resetProgress < 1);
    assert.deepEqual(r.out.transform, moved);

    r = hold(ctrl, { spread: 3 }, r.t, 500);
    assert.ok(Math.abs(r.out.transform.scale - moved.scale) < 1e-6);

    // Full hold: reset.
    r = hold(ctrl, { hands: 'overhead' }, r.t, 1600);
    assert.deepEqual(r.out.transform, { scale: 1, yaw: 0, tilt: 0 });
});
