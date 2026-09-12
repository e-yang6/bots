// Unit tests for viz/viewer/pose-gestures.js.
// Run with: node --test tests/test_pose_gestures.mjs

import { test } from 'node:test';
import assert from 'node:assert/strict';

import {
    BodyAnchor,
    DEFAULT_OPTIONS,
    GestureController,
    LM,
    measureBody,
    measureHands,
    OneEuroFilter,
    wrapAngle,
} from '../viz/viewer/pose-gestures.js';

const ASPECT = 16 / 9;
const deg = (d) => (d * Math.PI) / 180;
const near = (a, b, eps = 1e-9) => Math.abs(a - b) < eps;

/**
 * Synthetic pose for a person seen by a rear (unmirrored) camera.
 *
 * cx, shoulderY  shoulder midpoint in normalized image coordinates
 * torso          shoulder-to-hip distance in image heights
 * spread         wrist distance in shoulder widths
 * heading        torso rotation about vertical (radians), 0 = facing camera
 * roll           shoulder-line tilt on screen (radians, counter-clockwise)
 * hands          'down' | 'up' | 'overhead'
 */
function makePose({
    cx = 0.5, shoulderY = 0.35, torso = 0.3, spread = 1.5,
    heading = 0, roll = 0, hands = 'up', hipsVisible = true,
} = {}) {
    const image = Array.from({ length: 33 }, () => ({ x: 0.5, y: 0.5, z: 0, visibility: 0.0 }));
    const world = Array.from({ length: 33 }, () => ({ x: 0, y: 0, z: 0 }));
    const set = (i, img, w) => {
        image[i] = { ...img, z: 0, visibility: 0.99 };
        world[i] = w;
    };

    const halfShoulderM = 0.19;
    // Half shoulder width on screen, in image heights, foreshortened by heading.
    const halfShoulderIso = 0.1 * Math.cos(heading);

    // Facing the camera, the left shoulder appears on the image right.
    // Image y points down; positive roll lifts the left (screen-right) shoulder.
    const shoulderImg = (s) => ({
        x: cx + (s * halfShoulderIso * Math.cos(roll)) / ASPECT,
        y: shoulderY - s * halfShoulderIso * Math.sin(roll),
    });
    const shoulderWorld = (s) => ({
        x: s * halfShoulderM * Math.cos(heading),
        y: -0.45,
        z: s * halfShoulderM * Math.sin(heading),
    });
    set(LM.L_SHOULDER, shoulderImg(1), shoulderWorld(1));
    set(LM.R_SHOULDER, shoulderImg(-1), shoulderWorld(-1));
    set(LM.NOSE, { x: cx, y: shoulderY - 0.15 }, { x: 0, y: -0.6, z: -0.1 });

    const wristY = { down: shoulderY + torso + 0.15, up: shoulderY + 0.05, overhead: shoulderY - 0.3 }[hands];
    const halfSpreadM = spread * halfShoulderM;
    const halfSpreadIso = spread * 0.1;
    set(LM.L_WRIST, { x: cx + halfSpreadIso / ASPECT, y: wristY }, { x: halfSpreadM, y: -0.3, z: -0.2 });
    set(LM.R_WRIST, { x: cx - halfSpreadIso / ASPECT, y: wristY }, { x: -halfSpreadM, y: -0.3, z: -0.2 });

    // Hips straight below the shoulders along the tilted torso axis.
    const hipX = cx + (torso * Math.sin(roll)) / ASPECT;
    const hipY = shoulderY + torso * Math.cos(roll);
    set(LM.L_HIP, { x: hipX + 0.03, y: hipY }, { x: 0.1, y: 0, z: 0 });
    set(LM.R_HIP, { x: hipX - 0.03, y: hipY }, { x: -0.1, y: 0, z: 0 });
    if (!hipsVisible) {
        image[LM.L_HIP].visibility = 0.1;
        image[LM.R_HIP].visibility = 0.1;
    }
    return { image, world };
}

const body = (opts, mirrored = false) => {
    const p = makePose(opts);
    return measureBody(p.image, p.world, ASPECT, mirrored);
};
const hands = (opts, mirrored = false) => {
    const p = makePose(opts);
    return measureHands(p.image, p.world, ASPECT, mirrored);
};

/** Feed the same hand pose for `ms` milliseconds at 30 fps; return the last output. */
function hold(ctrl, opts, startMs, ms) {
    let out;
    let t = startMs;
    for (; t <= startMs + ms; t += 33) {
        out = ctrl.update(hands(opts), t);
    }
    return { out, t };
}

// ─── measureBody ────────────────────────────────────────────────────────

test('measureBody returns null without visible shoulders', () => {
    const p = makePose();
    p.image[LM.R_SHOULDER].visibility = 0.1;
    assert.equal(measureBody(p.image, p.world, ASPECT), null);
    assert.equal(measureBody(null, null, ASPECT), null);
});

test('measureBody anchors the aorta partway down the torso', () => {
    const b = body({ cx: 0.3, shoulderY: 0.2, torso: 0.4 });
    const f = DEFAULT_OPTIONS.aortaTorsoFraction;
    assert.ok(near(b.x, 0.3));
    assert.ok(near(b.y, 0.2 + f * 0.4));
});

test('measureBody size comes from torso length', () => {
    const b = body({ torso: 0.3 });
    assert.ok(near(b.heightPerMeter, 0.3 / DEFAULT_OPTIONS.torsoLengthM));
    // Twice as close looks twice as big.
    assert.ok(near(body({ torso: 0.6 }).heightPerMeter, 2 * b.heightPerMeter));
});

test('measureBody size does not change when the person turns', () => {
    const front = body({ heading: 0 });
    const turned = body({ heading: deg(50) });
    assert.ok(near(front.heightPerMeter, turned.heightPerMeter));
});

test('measureBody falls back to shoulder width when hips are out of frame', () => {
    const b = body({ hipsVisible: false });
    // Shoulder width in the synthetic pose is 0.2 image heights.
    const hpm = 0.2 / DEFAULT_OPTIONS.shoulderWidthM;
    assert.ok(near(b.heightPerMeter, hpm));
    assert.ok(near(b.x, 0.5));
    assert.ok(near(b.y, 0.35 + DEFAULT_OPTIONS.aortaTorsoFraction * DEFAULT_OPTIONS.torsoLengthM * hpm));
    // Turning is corrected for.
    assert.ok(near(body({ hipsVisible: false, heading: deg(40) }).heightPerMeter, hpm, 1e-6));
});

test('measureBody heading follows the torso', () => {
    assert.ok(near(body({ heading: 0 }).heading, 0));
    assert.ok(near(body({ heading: deg(30) }).heading, deg(30)));
    assert.ok(near(body({ heading: deg(-60) }).heading, deg(-60)));
});

test('measureBody roll follows the shoulder line on screen', () => {
    assert.ok(near(body({ roll: 0 }).roll, 0));
    assert.ok(near(body({ roll: deg(12) }).roll, deg(12)));
    assert.ok(near(body({ roll: deg(-8) }).roll, deg(-8)));
});

test('mirrored view flips heading, roll and x, but not y or size', () => {
    const opts = { cx: 0.3, heading: deg(25), roll: deg(10) };
    const n = body(opts), m = body(opts, true);
    assert.ok(near(m.heading, -n.heading));
    assert.ok(near(m.roll, -n.roll));
    assert.ok(near(m.x, 1 - n.x));
    assert.ok(near(m.y, n.y));
    assert.ok(near(m.heightPerMeter, n.heightPerMeter));
});

test('roll stays small when the person faces away', () => {
    const b = body({ heading: deg(180), roll: deg(5) });
    assert.ok(Math.abs(b.roll) < deg(30), `roll ${b.roll}`);
    assert.ok(near(Math.abs(wrapAngle(b.heading)), Math.PI, 1e-6));
});

// ─── OneEuroFilter / BodyAnchor ─────────────────────────────────────────

test('OneEuroFilter passes the first value and converges to a constant', () => {
    const f = new OneEuroFilter(1.0, 0.5);
    assert.equal(f.filter(0, 0), 0);
    let y = 0;
    for (let i = 1; i <= 90; i++) y = f.filter(1, i / 30);
    assert.ok(near(y, 1, 1e-3), `y ${y}`);
});

test('OneEuroFilter damps jitter around a constant', () => {
    const f = new OneEuroFilter(1.0, 0.5);
    let maxDev = 0;
    for (let i = 0; i < 300; i++) {
        const y = f.filter(0.5 + (i % 2 ? 0.01 : -0.01), i / 30);
        if (i > 30) maxDev = Math.max(maxDev, Math.abs(y - 0.5));
    }
    assert.ok(maxDev < 0.005, `maxDev ${maxDev}`);
});

test('BodyAnchor holds through short dropouts, then hides', () => {
    const a = new BodyAnchor();
    const first = a.update(body({}), 0);
    assert.equal(first.visible, true);
    const held = a.update(null, DEFAULT_OPTIONS.anchorLostMs - 50);
    assert.equal(held.visible, true);
    assert.ok(near(held.x, first.x));
    assert.equal(a.update(null, DEFAULT_OPTIONS.anchorLostMs + 50).visible, false);
});

test('BodyAnchor heading does not spin the long way across ±π', () => {
    const a = new BodyAnchor();
    let t = 0;
    let out;
    for (const h of [170, 175, 179, -179, -175, -170]) {
        for (let i = 0; i < 10; i++, t += 33) out = a.update(body({ heading: deg(h) }), t);
        // Always close to ±180°, never swinging through 0.
        assert.ok(Math.abs(out.heading) > deg(150), `heading ${out.heading} at ${h}`);
    }
});

// ─── measureHands ───────────────────────────────────────────────────────

test('measureHands returns null when wrists are not visible', () => {
    const p = makePose();
    p.image[LM.L_WRIST].visibility = 0.1;
    assert.equal(measureHands(p.image, p.world, ASPECT), null);
});

test('measureHands detects raised, lowered and overhead hands', () => {
    assert.equal(hands({ hands: 'up' }).handsRaised, true);
    assert.equal(hands({ hands: 'down' }).handsRaised, false);
    assert.equal(hands({ hands: 'up' }).handsOverHead, false);
    assert.equal(hands({ hands: 'overhead' }).handsOverHead, true);
    assert.equal(hands({ hands: 'up', hipsVisible: false }).handsRaised, true);
    assert.equal(hands({ hands: 'down', hipsVisible: false }).handsRaised, false);
});

test('measureHands spread is in shoulder widths', () => {
    assert.ok(near(hands({ spread: 1 }).spread, 1));
    assert.ok(near(hands({ spread: 3 }).spread, 3));
});

// ─── GestureController (scale) ──────────────────────────────────────────

test('hands down does not scale', () => {
    const ctrl = new GestureController();
    const { out } = hold(ctrl, { hands: 'down', spread: 4 }, 0, 1000);
    assert.equal(out.status, 'idle');
    assert.deepEqual(out.transform, { scale: 1 });
});

test('raising hands engages without jumping', () => {
    const ctrl = new GestureController();
    const { out } = hold(ctrl, { spread: 3 }, 0, 1000);
    assert.equal(out.status, 'tracking');
    assert.ok(near(out.transform.scale, 1, 1e-6));
});

test('arms apart scales up, arms together scales down', () => {
    const ctrl = new GestureController();
    let r = hold(ctrl, { spread: 2 }, 0, 500);
    r = hold(ctrl, { spread: 4 }, r.t, 1000);
    assert.ok(r.out.transform.scale > 1.8, `scale ${r.out.transform.scale}`);
    r = hold(ctrl, { spread: 1 }, r.t, 1000);
    assert.ok(r.out.transform.scale < 0.6, `scale ${r.out.transform.scale}`);
});

test('scale is clamped', () => {
    const ctrl = new GestureController();
    const r = hold(ctrl, { spread: 0.5 }, 0, 500);
    assert.equal(hold(ctrl, { spread: 20 }, r.t, 1000).out.transform.scale, ctrl.options.scaleMax);
});

test('small spread jitter inside the dead zone does not scale', () => {
    const ctrl = new GestureController();
    const r = hold(ctrl, { spread: 1.5 }, 0, 500);
    assert.deepEqual(hold(ctrl, { spread: 1.53 }, r.t, 1000).out.transform, { scale: 1 });
});

test('dropping hands keeps the scale, re-raising continues from it', () => {
    const ctrl = new GestureController();
    let r = hold(ctrl, { spread: 2 }, 0, 500);
    r = hold(ctrl, { spread: 4 }, r.t, 1000);
    const held = r.out.transform.scale;

    r = hold(ctrl, { hands: 'down', spread: 1 }, r.t, 1000);
    assert.equal(r.out.transform.scale, held);

    r = hold(ctrl, { spread: 1 }, r.t, 1000);
    assert.ok(near(r.out.transform.scale, held, 1e-6));
});

test('brief hand dropout keeps tracking', () => {
    const ctrl = new GestureController();
    let r = hold(ctrl, { spread: 2 }, 0, 500);
    r = hold(ctrl, { spread: 4 }, r.t, 1000);
    const before = r.out.transform.scale;
    const out = ctrl.update(null, r.t + 100);
    assert.equal(out.status, 'tracking');
    const after = ctrl.update(hands({ spread: 4 }), r.t + 150);
    assert.ok(Math.abs(after.transform.scale - before) < 0.05);
});

test('hands over head held resets scale, released early does not', () => {
    const ctrl = new GestureController();
    let r = hold(ctrl, { spread: 2 }, 0, 500);
    r = hold(ctrl, { spread: 4 }, r.t, 1000);
    const moved = r.out.transform.scale;
    assert.ok(moved > 1.5);

    r = hold(ctrl, { hands: 'overhead' }, r.t, 700);
    assert.equal(r.out.status, 'resetting');
    assert.ok(r.out.resetProgress > 0 && r.out.resetProgress < 1);
    assert.equal(r.out.transform.scale, moved);

    r = hold(ctrl, { spread: 3 }, r.t, 500);
    assert.ok(near(r.out.transform.scale, moved, 1e-6));

    r = hold(ctrl, { hands: 'overhead' }, r.t, 1600);
    assert.deepEqual(r.out.transform, { scale: 1 });
});
