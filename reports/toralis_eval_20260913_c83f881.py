import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from toralis_cli_20260913_c83f881 import validate_prediction

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluate import _direction_angle_deg, match_and_score

DATA = ROOT / "TORALIS CHALLENGE" / "EVAL_SET"
OUTPUT = ROOT / "reports" / "toralis_eval_20260913_c83f881"
COUNTS = {19: 3, 20: 4, 21: 3, 22: 6, 23: 3}
FIELDS = ("instance_id", "parent_instance_id", "ostium_xyz_mm", "seed_xyz_mm", "radius_mm", "direction_xyz", "review_status")


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def main():
    inputs = []
    for number, expected_count in COUNTS.items():
        case_id = f"case_{number}"
        folder = DATA / case_id
        image, mask, reference = folder / f"orig{number}.nii.gz", folder / f"aorta{number}.nii.gz", folder / "annotations.json"
        assert image.is_file() and mask.is_file() and reference.is_file(), case_id
        annotations = json.loads(reference.read_text(encoding="utf-8"))
        assert annotations["case_id"] == case_id
        assert annotations["coordinate_system"] == "SimpleITK physical LPS millimetres"
        assert len(annotations["daughters"]) == expected_count
        assert len({d["instance_id"] for d in annotations["daughters"]}) == expected_count
        inputs.append((case_id, image, mask, reference, annotations))
    OUTPUT.mkdir(exist_ok=False)
    report = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "data_dir": str(DATA),
        "interpretation": "Informative comparison to draft expert-review-pending labels, not an adjudicated accuracy claim.",
        "matching": "Unmodified src.evaluate: Hungarian assignment on physical LPS ostium distance, then accept at <=10 mm; no direction-based reassignment or exclusions.",
        "runtime_method": "One sequential run.py subprocess per case, default pipeline/thread settings and --verbose. Includes startup, imports and I/O; excludes evaluation.",
        "cases": [],
    }
    report_path = OUTPUT / "report.json"
    for case_id, image, mask, reference, annotations in inputs:
        prediction_path = OUTPUT / f"{case_id}_prediction.json"
        command = [sys.executable, str(ROOT / "run.py"), "--image", str(image), "--aorta-mask", str(mask), "--output", str(prediction_path), "--verbose"]
        print(f"Running {case_id}", flush=True)
        started = time.perf_counter()
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
        elapsed = time.perf_counter() - started
        (OUTPUT / f"{case_id}_stdout.log").write_text(result.stdout, encoding="utf-8")
        (OUTPUT / f"{case_id}_stderr.log").write_text(result.stderr, encoding="utf-8")
        assert result.returncode == 0, (case_id, result.returncode, result.stderr)
        assert not any(marker in result.stderr.lower() for marker in ("pipeline failed", "error: failed to write", "traceback (most recent call last)")), (case_id, result.stderr)
        prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
        errors = validate_prediction(prediction, case_id)
        assert not errors, (case_id, errors)
        score_command = [sys.executable, "-m", "src.evaluate", "--predictions", str(prediction_path), "--references", str(reference), "--distance-threshold-mm", "10"]
        evaluation = subprocess.run(score_command, cwd=ROOT, capture_output=True, text=True, check=False)
        (OUTPUT / f"{case_id}_scores.json").write_text(evaluation.stdout, encoding="utf-8")
        (OUTPUT / f"{case_id}_evaluation_stderr.log").write_text(evaluation.stderr, encoding="utf-8")
        assert evaluation.returncode == 0, (case_id, evaluation.stderr)
        scores = json.loads(evaluation.stdout)
        detailed = match_and_score(prediction, annotations, distance_threshold_mm=10)
        assert scores == {key: value for key, value in detailed.items() if key != "matches"}
        preds, refs = prediction["daughters"], annotations["daughters"]
        matches = []
        for i, j, distance in detailed["matches"]:
            p, r = preds[i], refs[j]
            matches.append({
                "prediction_id": p["instance_id"], "reference_id": r["instance_id"],
                "ostium_error_mm": distance,
                "seed_error_mm": math.dist(p["seed_xyz_mm"], r["seed_xyz_mm"]),
                "direction_error_deg": _direction_angle_deg(p["direction_xyz"], r["direction_xyz"]),
                "radius_error_mm": abs(p["radius_mm"] - r["radius_mm"]) if r["radius_mm"] is not None else None,
            })
        matched_p = {i for i, _, _ in detailed["matches"]}
        matched_r = {j for _, j, _ in detailed["matches"]}
        row = {
            "case_id": case_id, "command": command, "score_command": score_command,
            "input_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (image, mask, reference)},
            "elapsed_s": elapsed, "return_code": result.returncode, "schema_valid": True,
            "n_predictions": len(preds), "n_references": len(refs), "scores": scores,
            "matches": matches,
            "unmatched_prediction_ids": [p["instance_id"] for i, p in enumerate(preds) if i not in matched_p],
            "unmatched_reference_ids": [r["instance_id"] for j, r in enumerate(refs) if j not in matched_r],
            "reference_snapshot": [{key: r[key] for key in FIELDS} for r in refs],
        }
        report["cases"].append(row)
        save(report_path, report)
        print(json.dumps({"case_id": case_id, "elapsed_s": elapsed, **scores}), flush=True)
    rows = report["cases"]
    tp, fp, fn = (sum(row["scores"][key] for row in rows) for key in ("true_positives", "false_positives", "false_negatives"))
    matches = [match for row in rows for match in row["matches"]]
    assert tp == len(matches) and tp + fn == 19 and tp + fp == sum(row["n_predictions"] for row in rows)
    summary = {
        "n_cases": len(rows), "n_schema_valid": sum(row["schema_valid"] for row in rows),
        "n_predictions": tp + fp, "n_references": tp + fn,
        "true_positives": tp, "false_positives": fp, "false_negatives": fn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "radius_scored_matches": sum(match["radius_error_mm"] is not None for match in matches),
        "mean_elapsed_s": sum(row["elapsed_s"] for row in rows) / len(rows),
        "max_elapsed_s": max(row["elapsed_s"] for row in rows),
    }
    for metric in ("ostium_error_mm", "seed_error_mm", "direction_error_deg", "radius_error_mm"):
        values = [match[metric] for match in matches if match[metric] is not None]
        summary[f"mean_{metric}"] = sum(values) / len(values) if values else None
    report["summary"] = summary
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    save(report_path, report)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Report: {report_path}", flush=True)


if __name__ == "__main__":
    main()
