import json
import subprocess
import sys
from pathlib import Path

import run

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_supervisor_timeout_does_not_reuse_stale_output(tmp_path):
    output = tmp_path / "prediction.json"
    output.write_text('{"case_id": "stale", "daughters": []}')
    status = run.supervise_cli([sys.executable, "-c", "import time; time.sleep(5)"], output, "subject001", timeout_s=0.15)
    assert status["stop_reason"] == "deadline"
    assert json.loads(output.read_text()) == {"case_id": "subject001", "parent": {"instance_id": "aorta"}, "daughters": []}


def test_supervisor_preserves_a_valid_checkpoint(tmp_path):
    import schema

    output = tmp_path / "prediction.json"
    expected = schema.make_prediction("subject001", [schema.make_daughter("branch_001", [0, 0, 0], [5, 0, 0], 1., [1, 0, 0])])
    code = f"import time; from pathlib import Path; Path({str(output)!r}).write_text({json.dumps(expected)!r}); time.sleep(5)"
    status = run.supervise_cli([sys.executable, "-c", code], output, "subject001", timeout_s=1.0)
    assert status["stop_reason"] == "deadline"
    assert json.loads(output.read_text()) == expected


def test_supervisor_enforces_memory_budget(tmp_path):
    status = run.supervise_cli([sys.executable, "-c", "import time; time.sleep(5)"], tmp_path / "prediction.json",
                               "subject001", memory_limit_mib=0.01)
    assert status["stop_reason"] == "memory"


def test_run_end_to_end_stub_pipeline_emits_valid_empty_json(tmp_path):
    case_dir = tmp_path / "subject001"
    case_dir.mkdir()
    image = case_dir / "orig1.nii.gz"
    aorta_mask = case_dir / "mask1.nii.gz"
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
        "case_id": "subject001",
        "parent": {"instance_id": "aorta"},
        "daughters": [],
    }


def test_run_cli_subprocess_matches_exact_contract(tmp_path):
    case_dir = tmp_path / "subject002"
    case_dir.mkdir()
    image = case_dir / "orig2.nii.gz"
    aorta_mask = case_dir / "mask2.nii.gz"
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
    assert data["case_id"] == "subject002"
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
