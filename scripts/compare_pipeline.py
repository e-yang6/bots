import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import psutil

from src.evaluate import _direction_angle_deg, match_and_score

ROOT = Path(__file__).resolve().parents[1]
DAUGHTER_FIELDS = {"instance_id", "parent_instance_id", "ostium_xyz_mm", "seed_xyz_mm", "radius_mm", "direction_xyz"}


def measure(command, stdout_path, stderr_path, cores=4, timeout=60.0):
    env = dict(os.environ, ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=str(cores), OMP_NUM_THREADS=str(cores),
               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    started = time.perf_counter()
    peak = 0
    timed_out = False
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr)
        watcher = psutil.Process(process.pid)
        affinity = None
        if hasattr(watcher, "cpu_affinity"):
            affinity = watcher.cpu_affinity()[:cores]
            watcher.cpu_affinity(affinity)
        while process.poll() is None:
            try:
                peak = max(peak, watcher.memory_info().rss + sum(p.memory_info().rss for p in watcher.children(recursive=True)))
            except psutil.NoSuchProcess:
                pass
            if time.perf_counter() - started > timeout:
                for child in watcher.children(recursive=True):
                    try:
                        child.kill()
                    except psutil.NoSuchProcess:
                        pass
                process.kill()
                timed_out = True
                break
            time.sleep(0.02)
        return_code = process.wait()
    return {"elapsed_s": time.perf_counter() - started, "peak_rss_mib": peak / 1024 ** 2,
            "return_code": return_code, "timed_out": timed_out, "cpu_affinity": affinity}


def confidence_sweep(args, inputs, report):
    import SimpleITK as sitk
    import schema
    from run import analyze_case, build_daughters

    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(args.cores)
    process = psutil.Process()
    if hasattr(process, "cpu_affinity"):
        process.cpu_affinity(process.cpu_affinity()[:args.cores])
    thresholds = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75)
    grouped = {str(t): [] for t in thresholds}
    report["flood_budget_mm"] = args.flood_budget_mm
    report["measurement"] = "In-process confidence sweep, not a CLI resource benchmark."
    for case_id, image, mask, reference in inputs:
        context = analyze_case(str(image), str(mask), flood_budget_mm=args.flood_budget_mm)
        refs = json.loads(reference.read_text(encoding="utf-8"))
        row = {"case_id": case_id, "n_references": len(refs["daughters"]), "thresholds": {}}
        for threshold in thresholds:
            daughters, scored = build_daughters(context, threshold=threshold)
            prediction = schema.make_prediction(case_id, daughters)
            score = {k: v for k, v in match_and_score(prediction, refs).items() if k != "matches"}
            score["n_references"] = len(refs["daughters"])
            row["thresholds"][str(threshold)] = score
            grouped[str(threshold)].append(score)
            schema.write_prediction(prediction, args.output_dir / f"{case_id}_threshold_{threshold:.2f}.json")
        row["candidates"] = [{"instance": e["instance"]["instance_id"], "features": e["features"],
                              "score": e["score"], "ostium_mm": e["ostium"]["ostium_mm"].tolist()} for e in scored]
        report["cases"].append(row)
        print(f"{case_id}: " + " ".join(f"{t:.2f}:{row['thresholds'][str(t)]['f1']:.3f}" for t in thresholds), flush=True)
        (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    report["summary"] = {}
    for threshold, rows in grouped.items():
        tp, fp, fn = (sum(r[k] for r in rows) for k in ("true_positives", "false_positives", "false_negatives"))
        scored_rows = [r for r in rows if r["n_references"] > 0]
        report["summary"][threshold] = {"tp": tp, "fp": fp, "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else 0., "recall": tp / (tp + fn) if tp + fn else 0.,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.,
            "mean_case_f1": sum(r["f1"] for r in scored_rows) / len(scored_rows) if scored_rows else None}
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2), flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "TORALIS CHALLENGE" / "EVAL_SET")
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--cases", nargs="*", type=int, default=[19, 20, 21, 22, 23])
    parser.add_argument("--phantoms", action="store_true")
    parser.add_argument("--confidence-sweep", action="store_true")
    parser.add_argument("--flood-budget-mm", type=float)
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    source_files = [ROOT / "run.py", ROOT / "schema.py", *sorted((ROOT / "src").glob("*.py"))]
    report = {"interpretation": "Development evaluation against draft references; not independent validation.",
              "python": sys.version, "platform": platform.platform(), "cores": args.cores,
              "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
              "cases": []}
    (args.output_dir / "source.diff").write_text(subprocess.check_output(["git", "diff"], cwd=ROOT, text=True), encoding="utf-8")
    if args.phantoms:
        inputs = [(p.parent.name, p.parent / "image.nii.gz", p.parent / "mask.nii.gz", p)
                  for p in sorted(args.data_dir.glob("*/reference.json")) if "_n5_organbed_" not in p.parent.name]
        report["interpretation"] = "Regression on cached synthetic phantoms; not patient accuracy."
    else:
        inputs = [(f"case_{n}", args.data_dir / f"case_{n}" / f"orig{n}.nii.gz",
                   args.data_dir / f"case_{n}" / f"aorta{n}.nii.gz", args.data_dir / f"case_{n}" / "annotations.json")
                  for n in args.cases]
    if args.confidence_sweep:
        return confidence_sweep(args, inputs, report)
    matches = []
    for case_id, image_path, mask_path, reference_path in inputs:
        references = json.loads(reference_path.read_text(encoding="utf-8"))
        prediction_path = args.output_dir / f"{case_id}_prediction.json"
        stderr_path = args.output_dir / f"{case_id}_stderr.log"
        command = [sys.executable, str(ROOT / "run.py"), "--image", str(image_path),
                   "--aorta-mask", str(mask_path), "--output", str(prediction_path), "--verbose"]
        row = {"case_id": case_id, **measure(command, args.output_dir / f"{case_id}_stdout.log", stderr_path, args.cores, args.timeout)}
        log = stderr_path.read_text(encoding="utf-8")
        row["pipeline_failed"] = "pipeline failed" in log or "error: failed to write" in log
        prediction = json.loads(prediction_path.read_text(encoding="utf-8")) if prediction_path.exists() else {"daughters": []}
        daughters = prediction["daughters"]
        row.update(n_predictions=len(daughters), n_references=len(references["daughters"]))
        row["exact_keys"] = set(prediction) == {"case_id", "parent", "daughters"} and all(set(d) == DAUGHTER_FIELDS for d in daughters)
        row["seed_chords_over_5mm"] = sum(math.dist(d["ostium_xyz_mm"], d["seed_xyz_mm"]) > 5.01 for d in daughters)
        score = match_and_score(prediction, references)
        row["scores"] = {k: v for k, v in score.items() if k != "matches"}
        row["matches"] = []
        for i, j, distance in score["matches"]:
            p, r = daughters[i], references["daughters"][j]
            pair = {"prediction_id": p["instance_id"], "reference_id": r["instance_id"], "ostium_error_mm": distance,
                    "seed_error_mm": math.dist(p["seed_xyz_mm"], r["seed_xyz_mm"]),
                    "direction_error_deg": _direction_angle_deg(p["direction_xyz"], r["direction_xyz"]),
                    "radius_error_mm": abs(p["radius_mm"] - r["radius_mm"]) if r.get("radius_mm") is not None else None}
            row["matches"].append(pair)
            matches.append(pair)
        row["scores_at_3mm"] = {k: v for k, v in match_and_score(prediction, references, 3.0).items() if k != "matches"}
        report["cases"].append(row)
        print(f"{case_id}: {len(daughters)}/{row['n_references']} predictions/references; "
              f"TP/FP/FN={score['true_positives']}/{score['false_positives']}/{score['false_negatives']} "
              f"F1={score['f1']:.3f} time={row['elapsed_s']:.2f}s peak={row['peak_rss_mib']:.0f}MiB", flush=True)
        (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    rows = report["cases"]
    tp, fp, fn = (sum(r["scores"][key] for r in rows) for key in ("true_positives", "false_positives", "false_negatives"))
    summary = {"tp": tp, "fp": fp, "fn": fn, "precision": tp / (tp + fp) if tp + fp else 0,
               "recall": tp / (tp + fn) if tp + fn else 0, "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0,
               "mean_elapsed_s": sum(r["elapsed_s"] for r in rows) / len(rows),
               "max_elapsed_s": max(r["elapsed_s"] for r in rows), "max_peak_rss_mib": max(r["peak_rss_mib"] for r in rows),
               "seed_chords_over_5mm": sum(r["seed_chords_over_5mm"] for r in rows),
               "failures": sum(r["pipeline_failed"] or r["timed_out"] or r["return_code"] != 0 for r in rows)}
    for key in ("ostium_error_mm", "seed_error_mm", "direction_error_deg", "radius_error_mm"):
        values = [p[key] for p in matches if p[key] is not None]
        summary[f"mean_{key}"] = sum(values) / len(values) if values else None
    scoring_rows = [r for r in rows if r["n_references"] > 0]
    summary["mean_case_f1"] = sum(r["scores"]["f1"] for r in scoring_rows) / len(scoring_rows) if scoring_rows else None
    report["summary"] = summary
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return int(summary["failures"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
