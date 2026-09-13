"""Wall-clock time and peak memory of the actual CLI, per real case.

Runs `python run.py --image ... --aorta-mask ... --output ...` as a real
subprocess (not an in-process call) so process startup, imports, and I/O are
all included -- the same cost a grader invoking the CLI once per case would
see. Peak RSS is sampled by polling the child process (psutil), since
platform-specific rusage units (ru_maxrss is bytes on macOS, KB on Linux)
are an easy source of a silently-wrong number.

The brief's target is well under 60s/case on 4 cores. This machine's actual
core count may differ; the pipeline's own hot path (flood-fill's Dijkstra
call, per-candidate tracing) is single-threaded, so wall-clock here is not
sensitive to available cores beyond whatever BLAS uses internally --
running MORE cases in parallel is where extra cores would help, not any one
case finishing faster. If any case is not comfortably under budget, check
first whether crop_to_mask_bbox runs before resample_isotropic (cropping
after resampling redoes the expensive part of resampling on full-body
volume) and whether compute_vesselness only runs inside the flood's
reachable set rather than the whole crop.

Usage:
    python -m scripts.benchmark [--data-dir PATH] [--cases subject001 ...]
                                [--report benchmark_report.json]
"""

import argparse
from contextlib import nullcontext
import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.inspect_pipeline import find_real_data_dir, list_real_cases  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET_SECONDS_PER_CASE = 60.0
POLL_INTERVAL_S = 0.05


def _run_and_measure(image_path, mask_path, output_path):
    import psutil

    cmd = [sys.executable, "run.py", "--image", image_path, "--aorta-mask", mask_path,
           "--output", output_path]
    started = time.time()
    log = open(output_path + ".stderr.log", "w", encoding="utf-8")
    process = subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=subprocess.DEVNULL, stderr=log)
    watcher = psutil.Process(process.pid)
    if hasattr(watcher, "cpu_affinity"):
        watcher.cpu_affinity(watcher.cpu_affinity()[:4])

    peak_rss = 0
    stop = threading.Event()

    def poll():
        nonlocal peak_rss
        while not stop.is_set():
            try:
                rss = watcher.memory_info().rss
                for child in watcher.children(recursive=True):
                    rss += child.memory_info().rss
                peak_rss = max(peak_rss, rss)
            except psutil.NoSuchProcess:
                break
            time.sleep(POLL_INTERVAL_S)

    thread = threading.Thread(target=poll)
    thread.start()
    return_code = process.wait()
    elapsed = time.time() - started
    stop.set()
    thread.join()
    log.close()

    return elapsed, peak_rss, return_code


def run_benchmark(data_dir, case_names=None, verbose=True, output_dir=None):
    cases = list_real_cases(data_dir)
    if case_names:
        cases = [c for c in cases if c[0] in case_names]

    rows = []
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=False)
    with (nullcontext(output_dir) if output_dir is not None else tempfile.TemporaryDirectory()) as tmp_dir:
        for case_id, image_path, mask_path in cases:
            output_path = os.path.join(tmp_dir, f"{case_id}_prediction.json")
            elapsed, peak_rss, return_code = _run_and_measure(image_path, mask_path, output_path)

            import schema

            n_daughters, schema_valid = None, False
            if return_code == 0 and os.path.exists(output_path):
                try:
                    prediction = schema.read_prediction(output_path)
                    schema.submission_prediction(prediction)
                    schema_valid = prediction["case_id"] == case_id
                    n_daughters = len(prediction["daughters"])
                except (ValueError, KeyError, TypeError):
                    pass
            with open(output_path + ".stderr.log", encoding="utf-8") as log:
                pipeline_failed = "pipeline failed" in log.read()

            row = {
                "case_id": case_id, "elapsed_s": elapsed, "peak_rss_mb": peak_rss / (1024.0 * 1024.0),
                "return_code": return_code, "n_daughters": n_daughters,
                "schema_valid": schema_valid, "pipeline_failed": pipeline_failed,
                "under_target": elapsed <= TARGET_SECONDS_PER_CASE,
            }
            rows.append(row)
            if verbose:
                print(f"{case_id:12s} elapsed={elapsed:6.1f}s peak_rss={row['peak_rss_mb']:7.1f}MB "
                      f"n_daughters={n_daughters} {'FAILED' if pipeline_failed or not schema_valid else 'OK' if row['under_target'] else 'OVER BUDGET'}",
                      file=sys.stderr)

    elapsed_values = [r["elapsed_s"] for r in rows]
    rss_values = [r["peak_rss_mb"] for r in rows]
    summary = {
        "n_cases": len(rows),
        "n_schema_valid": sum(r["schema_valid"] for r in rows),
        "n_failures": sum(r["pipeline_failed"] or not r["schema_valid"] or r["return_code"] != 0 for r in rows),
        "total_daughters": sum(r["n_daughters"] or 0 for r in rows),
        "target_seconds_per_case": TARGET_SECONDS_PER_CASE,
        "n_over_budget": sum(1 for r in rows if not r["under_target"]),
        "mean_elapsed_s": float(np.mean(elapsed_values)) if elapsed_values else None,
        "median_elapsed_s": float(np.median(elapsed_values)) if elapsed_values else None,
        "max_elapsed_s": float(np.max(elapsed_values)) if elapsed_values else None,
        "mean_peak_rss_mb": float(np.mean(rss_values)) if rss_values else None,
        "max_peak_rss_mb": float(np.max(rss_values)) if rss_values else None,
    }
    root = Path(REPO_ROOT)
    sources = [root / "run.py", root / "schema.py", *sorted((root / "src").glob("*.py"))]
    return {"summary": summary, "cases": rows, "python": sys.version,
            "method": "Sequential CLI runs; four-core affinity; sampled process-tree RSS including supervisor; no reference scoring.",
            "source_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=None, help="Path to the TORALIS CHALLENGE dev-set folder.")
    parser.add_argument("--cases", nargs="*", default=None, help="Restrict to these case ids (e.g. subject001).")
    parser.add_argument("--report", default="benchmark_report.json")
    parser.add_argument("--output-dir", default=None, help="Keep predictions and logs in a new directory.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    data_dir = find_real_data_dir(args.data_dir)
    if data_dir is None:
        print("No TORALIS CHALLENGE data folder found; nothing to do.", file=sys.stderr)
        return 1

    report = run_benchmark(data_dir, case_names=args.cases, output_dir=args.output_dir)
    with open(args.report, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print()
    print("=== summary ===")
    for key, value in report["summary"].items():
        print(f"  {key}: {value}")
    print(f"wrote {args.report}")
    return 0 if report["summary"]["n_over_budget"] == 0 and report["summary"]["n_failures"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
