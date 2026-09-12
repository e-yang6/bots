"""Headless smoke test for scripts/review_candidates.py against the
synthetic phantom -- drives the review programmatically (no GUI event
loop) so it runs in CI/pytest rather than requiring a human at a window.
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")

import pytest
from scipy.spatial import cKDTree

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

from run import analyze_volumes  # noqa: E402
from src.features import FEATURE_NAMES  # noqa: E402
from tests.synthetic import make_capsule_phantom  # noqa: E402

import review_candidates  # noqa: E402  (scripts/review_candidates.py)


WALL_X = 40.0


def _stub(z=45.0, length_beyond_wall=18.0, radius=3.0, y=32.0):
    """One clearly-real branch leaving the phantom aorta's wall."""
    return {"start": (36.0, y, z), "end": (WALL_X + length_beyond_wall, y, z), "radius": radius}


def _context(**phantom):
    """A real run.py context for the phantom.

    Built by calling analyze_volumes rather than re-wiring the detector's
    stages here: the review UI consumes whatever run.py produces, so a test
    that assembled its own context by hand would keep passing after the
    pipeline changed shape underneath it -- which is exactly what happened
    to the previous version of this file.
    """
    image, mask = make_capsule_phantom(**phantom)
    return image, mask, analyze_volumes(image, mask)


@pytest.fixture
def phantom_context():
    return _context(branches=[_stub()])


def test_context_has_one_reviewable_instance(phantom_context):
    _image, _mask, context = phantom_context
    assert len(context["instances"]) == 1
    assert len(context["traces"]) == 1


def test_confirm_writes_a_daughter_and_a_labelled_example(phantom_context, tmp_path):
    image, mask, context = phantom_context
    output_gt = tmp_path / "ground_truth.json"
    output_labels = tmp_path / "labels.json"

    reviewer = review_candidates.ReviewLabeller(
        image, mask, "phantom001", context,
        output_groundtruth=str(output_gt), output_labels=str(output_labels),
        raw_shading=True,
    )
    try:
        assert reviewer.review_mode is True
        assert reviewer.state == "REVIEW"
        assert reviewer.pending_mm is not None

        reviewer._confirm_review_candidate(1)  # 'y'

        assert reviewer.review_mode is False  # one instance: falls straight through
        assert len(reviewer.daughters) == 1
        assert len(reviewer.labeled_examples) == 1
        assert reviewer.labeled_examples[0]["label"] == 1
        assert set(reviewer.labeled_examples[0]["features"]) == set(FEATURE_NAMES)

        reviewer._save()

        with open(output_gt) as f:
            ground_truth = json.load(f)
        assert ground_truth["case_id"] == "phantom001"
        assert len(ground_truth["daughters"]) == 1

        with open(output_labels) as f:
            labels = json.load(f)
        assert len(labels) == 1
        assert labels[0]["label"] == 1
    finally:
        import matplotlib.pyplot as plt
        plt.close(reviewer.figure)


def test_reject_labels_but_writes_no_daughter(phantom_context, tmp_path):
    image, mask, context = phantom_context
    reviewer = review_candidates.ReviewLabeller(
        image, mask, "phantom001", context,
        output_groundtruth=str(tmp_path / "gt.json"), output_labels=str(tmp_path / "labels.json"),
        raw_shading=True,
    )
    try:
        reviewer._confirm_review_candidate(0)  # 'n'
        assert reviewer.daughters == []
        assert len(reviewer.labeled_examples) == 1
        assert reviewer.labeled_examples[0]["label"] == 0
    finally:
        import matplotlib.pyplot as plt
        plt.close(reviewer.figure)


def test_unsure_skips_without_labelling_or_saving_a_daughter(phantom_context, tmp_path):
    image, mask, context = phantom_context
    reviewer = review_candidates.ReviewLabeller(
        image, mask, "phantom001", context,
        output_groundtruth=str(tmp_path / "gt.json"), output_labels=str(tmp_path / "labels.json"),
        raw_shading=True,
    )
    try:
        reviewer._confirm_review_candidate(None)  # 'u'
        assert reviewer.daughters == []
        assert reviewer.labeled_examples == []
        assert reviewer.review_mode is False  # exhausted the one instance either way
    finally:
        import matplotlib.pyplot as plt
        plt.close(reviewer.figure)


def test_no_instances_falls_straight_to_freeform(tmp_path):
    image, mask, context = _context(branches=[])
    assert context["instances"] == []

    reviewer = review_candidates.ReviewLabeller(
        image, mask, "phantom002", context,
        output_groundtruth=str(tmp_path / "gt.json"), output_labels=str(tmp_path / "labels.json"),
        raw_shading=True,
    )
    try:
        assert reviewer.review_mode is False
        assert reviewer.state == "IDLE"
    finally:
        import matplotlib.pyplot as plt
        plt.close(reviewer.figure)


def test_labelled_example_carries_the_full_feature_vector(phantom_context, tmp_path):
    """The reviewer's vector and src/classify.py's training vector must not
    drift apart, so the key set is pinned to FEATURE_NAMES exactly."""
    image, mask, context = phantom_context
    reviewer = review_candidates.ReviewLabeller(
        image, mask, "phantom001", context,
        output_groundtruth=str(tmp_path / "gt.json"), output_labels=str(tmp_path / "labels.json"),
        raw_shading=True,
    )
    try:
        reviewer._confirm_review_candidate(1)
        features = reviewer.labeled_examples[0]["features"]
        assert list(features) != []
        assert set(features) == set(FEATURE_NAMES)
        assert all(isinstance(v, float) for v in features.values())
        json.dumps(features)  # must stay JSON-safe for the labels file
    finally:
        import matplotlib.pyplot as plt
        plt.close(reviewer.figure)
