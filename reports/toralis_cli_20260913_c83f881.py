import json
import math
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "TORALIS CHALLENGE"
OUTPUT = ROOT / "reports" / "toralis_cli_20260913_c83f881"


def validate_prediction(prediction, case_id):
    errors = []
    if not isinstance(prediction, dict):
        return ["prediction is not an object"]
    if prediction.get("case_id") != case_id:
        errors.append("case_id mismatch")
    if prediction.get("parent") != {"instance_id": "aorta"}:
        errors.append("invalid parent")
    daughters = prediction.get("daughters")
    if not isinstance(daughters, list):
        return errors + ["daughters is not a list"]
    ids = set()
    for index, daughter in enumerate(daughters, 1):
        prefix = f"daughter {index}: "
        if not isinstance(daughter, dict):
            errors.append(prefix + "not an object")
            continue
        instance_id = daughter.get("instance_id")
        if not isinstance(instance_id, str) or instance_id in ids:
            errors.append(prefix + "invalid or duplicate instance_id")
        else:
            ids.add(instance_id)
        if daughter.get("parent_instance_id") != "aorta":
            errors.append(prefix + "invalid parent_instance_id")
        for key in ("ostium_xyz_mm", "seed_xyz_mm", "direction_xyz"):
            value = daughter.get(key)
            valid = isinstance(value, list) and len(value) == 3 and all(
                isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
                for x in value
            )
            if not valid:
                errors.append(prefix + f"invalid {key}")
            elif key == "direction_xyz" and not math.isclose(math.hypot(*value), 1.0, abs_tol=1e-6):
                errors.append(prefix + "direction is not unit length")
        radius = daughter.get("radius_mm")
        if not isinstance(radius, (int, float)) or isinstance(radius, bool) or not math.isfinite(radius) or radius <= 0:
            errors.append(prefix + "invalid radius_mm")
    return errors


def main():
    cases = []
    for number in range(1, 26):
        case_id = f"subject{number:03d}"
        image = DATA / case_id / f"orig{number}.nii"
        mask = DATA / case_id / f"mask{number}.nii"
        if not image.is_file() or not mask.is_file():
            raise FileNotFoundError(f"Missing input pair: {case_id}")
        cases.append((case_id, image, mask))
    OUTPUT.mkdir(exist_ok=False)
    report = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_status_before": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "packages": {name: version(name) for name in ("numpy", "scipy", "SimpleITK", "nibabel")},
        "data_dir": str(DATA),
        "method": "One sequential CLI subprocess per case; default pipeline settings; --verbose; inherited thread settings; wall time includes interpreter startup, imports, I/O and log writing.",
        "reference_scoring": "Not performed: no EVAL_SET or daughter-reference JSON files are present in the local TORALIS CHALLENGE folder.",
        "cases": [],
    }
    report_path = OUTPUT / "report.json"
    for case_id, image, mask in cases:
        prediction_path = OUTPUT / f"{case_id}_prediction.json"
        stdout_path = OUTPUT / f"{case_id}_stdout.log"
        stderr_path = OUTPUT / f"{case_id}_stderr.log"
        command = [sys.executable, str(ROOT / "run.py"), "--image", str(image), "--aorta-mask", str(mask), "--output", str(prediction_path), "--verbose"]
        print(f"Starting {case_id}", flush=True)
        started = time.perf_counter()
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
            process = subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr, check=False)
        elapsed = time.perf_counter() - started
        stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace")
        failure_messages = [line for line in stderr_text.splitlines() if any(marker in line.lower() for marker in ("pipeline failed", "error: failed to write", "traceback (most recent call last)"))]
        warning_lines = [line for line in stderr_text.splitlines() if "warning" in line.lower()]
        prediction = None
        try:
            prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
            schema_errors = validate_prediction(prediction, case_id)
        except (OSError, ValueError) as exc:
            schema_errors = [str(exc)]
        daughters = prediction.get("daughters") if isinstance(prediction, dict) else None
        row = {
            "case_id": case_id,
            "command": command,
            "return_code": process.returncode,
            "elapsed_s": elapsed,
            "n_daughters": len(daughters) if isinstance(daughters, list) else None,
            "schema_valid": not schema_errors,
            "schema_errors": schema_errors,
            "pipeline_failure_messages": failure_messages,
            "warning_lines": warning_lines,
            "success": process.returncode == 0 and not schema_errors and not failure_messages,
            "prediction_file": str(prediction_path),
            "stdout_file": str(stdout_path),
            "stderr_file": str(stderr_path),
        }
        report["cases"].append(row)
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        print(f"{case_id}: daughters={row['n_daughters']} elapsed={elapsed:.3f}s success={row['success']}", flush=True)
    rows = report["cases"]
    times = sorted(row["elapsed_s"] for row in rows)
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    report["summary"] = {
        "n_cases": len(rows),
        "n_success": sum(row["success"] for row in rows),
        "n_schema_valid": sum(row["schema_valid"] for row in rows),
        "n_pipeline_failures": sum(bool(row["pipeline_failure_messages"]) for row in rows),
        "n_cases_with_warnings": sum(bool(row["warning_lines"]) for row in rows),
        "total_daughters": sum(row["n_daughters"] or 0 for row in rows),
        "zero_detection_cases": [row["case_id"] for row in rows if row["n_daughters"] == 0],
        "total_elapsed_s": sum(times),
        "mean_elapsed_s": sum(times) / len(times),
        "median_elapsed_s": times[len(times) // 2],
        "max_elapsed_s": max(times),
        "slowest_case": max(rows, key=lambda row: row["elapsed_s"])["case_id"],
        "n_over_60s": sum(value > 60.0 for value in times),
    }
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2), flush=True)
    print(f"Report: {report_path}", flush=True)
    return 0 if all(row["success"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
