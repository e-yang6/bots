import json
import subprocess
import sys
from pathlib import Path

import run

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_run_end_to_end_stub_pipeline_emits_valid_empty_json(tmp_path):
    image = tmp_path / "subject001_orig.nii.gz"
    aorta_mask = tmp_path / "subject001_mask.nii.gz"
    output = tmp_path / "prediction.json"
    image.write_bytes(b"")
    aorta_mask.write_bytes(b"")

    run.main([
        "--image", str(image),
        "--aorta-mask", str(aorta_mask),
        "--output", str(output),
    ])

    assert output.exists()
    data = json.loads(output.read_text())
    assert data == {
        "case_id": "subject001_orig",
        "parent": {"instance_id": "aorta"},
        "daughters": [],
    }


def test_run_cli_subprocess_matches_exact_contract(tmp_path):
    image = tmp_path / "subject002_orig.nii.gz"
    aorta_mask = tmp_path / "subject002_mask.nii.gz"
    output = tmp_path / "prediction.json"
    image.write_bytes(b"")
    aorta_mask.write_bytes(b"")

    result = subprocess.run(
        [
            sys.executable, str(REPO_ROOT / "run.py"),
            "--image", str(image),
            "--aorta-mask", str(aorta_mask),
            "--output", str(output),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    data = json.loads(output.read_text())
    assert data["case_id"] == "subject002_orig"
    assert data["parent"] == {"instance_id": "aorta"}
    assert data["daughters"] == []


def test_run_never_crashes_and_still_writes_valid_json_on_pipeline_failure(tmp_path, monkeypatch):
    output = tmp_path / "prediction.json"

    def boom(image_path, aorta_mask_path):
        raise RuntimeError("simulated detection failure")

    monkeypatch.setattr(run, "detect_branches", boom)

    run.main([
        "--image", "does/not/exist.nii.gz",
        "--aorta-mask", "does/not/exist_mask.nii.gz",
        "--output", str(output),
    ])

    assert output.exists()
    data = json.loads(output.read_text())
    assert data["daughters"] == []
    assert data["parent"] == {"instance_id": "aorta"}
