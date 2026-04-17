from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import re
from typing import Any

from eeg_pipeline.adm_events import extract_annotation_events, generate_adm_events
from eeg_pipeline.config import PipelineConfig, load_dataset_metadata, resolve_edf_data_root
from eeg_pipeline.export_and_viz import (
    ensure_output_dirs,
    export_csv,
    export_json,
    plot_channel_heatmap,
    plot_event_rate,
    plot_raw_with_events,
)
from eeg_pipeline.preprocessing import build_default_file, load_raw_edf, make_qc_summary, preprocess_raw


SUBJECT_PATTERN = re.compile(r"^([A-Za-z]+)(\d+)$")


def _split_tokens(value: str) -> list[str]:
    return [token.strip() for token in value.split(",") if token.strip()]


def parse_subjects_arg(subjects_arg: str) -> list[str]:
    tokens = _split_tokens(subjects_arg)
    if not tokens:
        raise ValueError("--subjects must contain at least one subject token")

    expanded: list[str] = []
    seen: set[str] = set()

    for token in tokens:
        if "-" not in token:
            match = SUBJECT_PATTERN.fullmatch(token)
            if match is None:
                raise ValueError(f"Invalid subject token '{token}'. Expected format like S001")
            if token not in seen:
                seen.add(token)
                expanded.append(token)
            continue

        left, right = token.split("-", 1)
        left_match = SUBJECT_PATTERN.fullmatch(left)
        right_match = SUBJECT_PATTERN.fullmatch(right)
        if left_match is None or right_match is None:
            raise ValueError(f"Invalid subject range '{token}'. Expected format like S001-S005")

        left_prefix, left_num = left_match.groups()
        right_prefix, right_num = right_match.groups()

        if left_prefix != right_prefix:
            raise ValueError(f"Subject range '{token}' has mismatched prefixes")
        if len(left_num) != len(right_num):
            raise ValueError(f"Subject range '{token}' has mismatched numeric widths")

        start = int(left_num)
        stop = int(right_num)
        if stop < start:
            raise ValueError(f"Subject range '{token}' is descending")

        width = len(left_num)
        for subject_num in range(start, stop + 1):
            subject = f"{left_prefix}{subject_num:0{width}d}"
            if subject not in seen:
                seen.add(subject)
                expanded.append(subject)

    return expanded


def parse_runs_arg(runs_arg: str, min_run: int = 1, max_run: int = 14) -> list[int]:
    tokens = _split_tokens(runs_arg)
    if not tokens:
        raise ValueError("--runs must contain at least one run token")

    expanded: list[int] = []
    seen: set[int] = set()

    for token in tokens:
        if "-" not in token:
            if not token.isdigit():
                raise ValueError(f"Invalid run token '{token}'. Expected an integer in {min_run}-{max_run}")
            run = int(token)
            if not (min_run <= run <= max_run):
                raise ValueError(f"Run '{run}' is out of bounds. Expected {min_run}-{max_run}")
            if run not in seen:
                seen.add(run)
                expanded.append(run)
            continue

        left, right = token.split("-", 1)
        if not left.isdigit() or not right.isdigit():
            raise ValueError(f"Invalid run range '{token}'. Expected format like 1-5")

        start = int(left)
        stop = int(right)
        if stop < start:
            raise ValueError(f"Run range '{token}' is descending")

        for run in range(start, stop + 1):
            if not (min_run <= run <= max_run):
                raise ValueError(f"Run '{run}' is out of bounds. Expected {min_run}-{max_run}")
            if run not in seen:
                seen.add(run)
                expanded.append(run)

    return expanded


def build_job_pairs(subjects: list[str], runs: list[int]) -> list[tuple[str, int]]:
    return [(subject, run) for subject in subjects for run in runs]


def parse_args() -> argparse.Namespace:
    default_cfg = PipelineConfig()

    parser = argparse.ArgumentParser(description="EEG ADM pipeline: preprocess -> events -> export -> plots")
    parser.add_argument("--data-root", type=Path, default=default_cfg.data_root, help="Root EEG data folder")
    parser.add_argument("--output-root", type=Path, default=default_cfg.output_root, help="Output folder")
    parser.add_argument(
        "--subjects",
        required=True,
        help="Subject tokens (comma list and ranges), e.g. S001,S003-S005",
    )
    parser.add_argument(
        "--runs",
        required=True,
        help="Run tokens (comma list and ranges), e.g. 1,3-5",
    )
    parser.add_argument("--parallel", action="store_true", help="Run jobs in parallel")
    parser.add_argument("--max-workers", type=int, default=1, help="Number of workers when parallel mode is enabled")
    parser.add_argument("--threshold-uv", type=float, default=default_cfg.adm.threshold_uv, help="ADM threshold in microvolts")
    parser.add_argument("--refractory-ms", type=float, default=default_cfg.adm.refractory_ms, help="ADM refractory in milliseconds")
    parser.add_argument("--suppression-ms", type=float, default=default_cfg.adm.suppression_ms, help="Post-event suppression window in milliseconds")
    parser.add_argument("--recovery-tc-ms", type=float, default=default_cfg.adm.recovery_tc_ms, help="Suppression recovery time constant in milliseconds")
    parser.add_argument("--post-event-gain", type=float, default=default_cfg.adm.post_event_gain, help="Immediate gain after an event (0 < gain <= 1)")
    parser.add_argument("--event-method", type=str, default=default_cfg.adm.event_method, help="ADM event method")
    parser.add_argument("--l-freq", type=float, default=default_cfg.preprocess.bandpass_low_hz, help="Band-pass low cutoff")
    parser.add_argument("--h-freq", type=float, default=default_cfg.preprocess.bandpass_high_hz, help="Band-pass high cutoff")
    parser.add_argument("--no-notch", action="store_true", help="Disable 60 Hz notch filter")
    args = parser.parse_args()
    if args.max_workers < 1:
        parser.error("--max-workers must be >= 1")
    return args


def run_pipeline(args: argparse.Namespace, subject: str, run: int) -> dict[str, Any]:
    '''
    Main pipline function to run the EEG processing steps:
    1. Load raw EDF data
    2. Preprocess the data (filtering, re-referencing)
    3. Generate ADM events
    4. Extract annotation events
    5. Create QC summary
    6. Export events and summary to CSV and JSON
    7. Generate plots for raw data with events, event rate, and channel heatmap
    '''
    dataset_meta = load_dataset_metadata(args.data_root)
    edf_data_root = resolve_edf_data_root(args.data_root, dataset_meta)

    cfg = PipelineConfig(data_root=edf_data_root, output_root=args.output_root)
    cfg.preprocess.bandpass_low_hz = args.l_freq
    cfg.preprocess.bandpass_high_hz = args.h_freq
    cfg.preprocess.apply_notch = not args.no_notch
    cfg.adm.threshold_uv = args.threshold_uv
    cfg.adm.refractory_ms = args.refractory_ms
    cfg.adm.suppression_ms = args.suppression_ms
    cfg.adm.recovery_tc_ms = args.recovery_tc_ms
    cfg.adm.post_event_gain = args.post_event_gain
    cfg.adm.event_method = args.event_method

    edf_path = build_default_file(cfg.data_root, subject, run)
    raw = load_raw_edf(edf_path)
    raw_pp = preprocess_raw(raw, cfg.preprocess)

    adm_events, adm_summary = generate_adm_events(raw_pp, cfg.adm)
    annotation_events = extract_annotation_events(raw, annotation_label_map=dataset_meta.annotation_label_map)
    qc = make_qc_summary(
        raw,
        raw_pp,
        subject,
        run,
        run_description_map=dataset_meta.run_description,
    )

    run_tag = f"{subject}_R{run:02d}"
    out_dirs = ensure_output_dirs(cfg.output_root / subject / run_tag)

    csv_path = out_dirs["csv"] / f"{run_tag}_events.csv"
    json_path = out_dirs["json"] / f"{run_tag}_summary.json"

    export_csv(adm_events, annotation_events, csv_path)

    payload = {
        "subject": subject,
        "run": run,
        "run_description": dataset_meta.run_description.get(run, "Unknown"),
        "input_file": str(edf_path),
        "dataset": {
            "dataset_id": dataset_meta.dataset_id,
            "dataset_name": dataset_meta.dataset_name,
            "requested_data_root": str(args.data_root),
            "resolved_edf_root": str(edf_data_root),
        },
        "qc": qc,
        "adm_summary": adm_summary,
        "n_annotation_events": len(annotation_events),
        "config": {
            "bandpass_low_hz": cfg.preprocess.bandpass_low_hz,
            "bandpass_high_hz": cfg.preprocess.bandpass_high_hz,
            "notch_hz": cfg.preprocess.notch_hz,
            "apply_notch": cfg.preprocess.apply_notch,
            "rereference": cfg.preprocess.rereference,
            "threshold_uv": cfg.adm.threshold_uv,
            "refractory_ms": cfg.adm.refractory_ms,
            "suppression_ms": cfg.adm.suppression_ms,
            "recovery_tc_ms": cfg.adm.recovery_tc_ms,
            "post_event_gain": cfg.adm.post_event_gain,
            "event_method": cfg.adm.event_method,
        },
    }
    export_json(payload, json_path)

    plot_raw_with_events(
        raw_pp,
        adm_events,
        out_dirs["plots"] / f"{run_tag}_raw_with_events.png",
    )
    plot_event_rate(
        adm_events,
        duration_s=float(raw_pp.times[-1]),
        out_path=out_dirs["plots"] / f"{run_tag}_event_rate.png",
    )
    plot_channel_heatmap(
        adm_events,
        ch_names=list(raw_pp.ch_names),
        out_path=out_dirs["plots"] / f"{run_tag}_channel_heatmap.png",
    )

    return {
        "run_tag": run_tag,
        "subject": subject,
        "run": run,
        "csv_path": str(csv_path),
        "json_path": str(json_path),
        "plot_dir": str(out_dirs["plots"]),
        "adm_summary": adm_summary,
    }


def _run_job_safe(args: argparse.Namespace, subject: str, run: int) -> dict[str, Any]:
    try:
        result = run_pipeline(args, subject, run)
        return {
            "success": True,
            "subject": subject,
            "run": run,
            "result": result,
            "error": None,
        }
    except Exception as exc:  # pragma: no cover
        return {
            "success": False,
            "subject": subject,
            "run": run,
            "result": None,
            "error": str(exc),
        }


def _execute_sequential(args: argparse.Namespace, jobs: list[tuple[str, int]]) -> list[dict[str, Any]]:
    return [_run_job_safe(args, subject, run) for subject, run in jobs]


def _execute_parallel(args: argparse.Namespace, jobs: list[tuple[str, int]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(_run_job_safe, args, subject, run) for subject, run in jobs]
        for future in as_completed(futures):
            results.append(future.result())
    return results


def main() -> None:
    args = parse_args()
    try:
        subjects = parse_subjects_arg(args.subjects)
        runs = parse_runs_arg(args.runs)
    except ValueError as exc:
        raise SystemExit(f"Invalid batch arguments: {exc}") from exc

    jobs = build_job_pairs(subjects, runs)
    if not jobs:
        raise SystemExit("No jobs to execute after parsing --subjects and --runs")

    use_parallel = args.parallel and args.max_workers > 1 and len(jobs) > 1
    results = _execute_parallel(args, jobs) if use_parallel else _execute_sequential(args, jobs)

    results_by_job = {(r["subject"], r["run"]): r for r in results}
    ordered_results = [results_by_job[(subject, run)] for subject, run in jobs]

    success_count = 0
    failures: list[dict[str, Any]] = []

    print("\n=== Batch Pipeline Completed ===")
    for item in ordered_results:
        run_tag = f"{item['subject']}_R{item['run']:02d}"
        if item["success"]:
            success_count += 1
            run_result = item["result"]
            assert run_result is not None
            print(f"[OK]  {run_tag} | CSV: {run_result['csv_path']}")
        else:
            failures.append(item)
            print(f"[ERR] {run_tag} | {item['error']}")

    print(f"\nTotal jobs      : {len(jobs)}")
    print(f"Successful jobs : {success_count}")
    print(f"Failed jobs     : {len(failures)}")

    if failures:
        print("\nFailed pairs:")
        for item in failures:
            print(f"- {item['subject']}_R{item['run']:02d}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
