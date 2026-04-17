from __future__ import annotations

import argparse
from pathlib import Path
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


def parse_args() -> argparse.Namespace:
    default_cfg = PipelineConfig()

    parser = argparse.ArgumentParser(description="EEG ADM pipeline: preprocess -> events -> export -> plots")
    parser.add_argument("--data-root", type=Path, default=default_cfg.data_root, help="Root EEG data folder")
    parser.add_argument("--output-root", type=Path, default=default_cfg.output_root, help="Output folder")
    parser.add_argument("--subject", default="S001", help="Subject folder (e.g., S001)")
    parser.add_argument("--run", type=int, default=2, choices=range(1, 15), metavar="[1-14]", help="Run number")
    parser.add_argument("--threshold-uv", type=float, default=default_cfg.adm.threshold_uv, help="ADM threshold in microvolts")
    parser.add_argument("--refractory-ms", type=float, default=default_cfg.adm.refractory_ms, help="ADM refractory in milliseconds")
    parser.add_argument("--suppression-ms", type=float, default=default_cfg.adm.suppression_ms, help="Post-event suppression window in milliseconds")
    parser.add_argument("--recovery-tc-ms", type=float, default=default_cfg.adm.recovery_tc_ms, help="Suppression recovery time constant in milliseconds")
    parser.add_argument("--post-event-gain", type=float, default=default_cfg.adm.post_event_gain, help="Immediate gain after an event (0 < gain <= 1)")
    parser.add_argument("--event-method", type=str, default=default_cfg.adm.event_method, help="ADM event method")
    parser.add_argument("--l-freq", type=float, default=default_cfg.preprocess.bandpass_low_hz, help="Band-pass low cutoff")
    parser.add_argument("--h-freq", type=float, default=default_cfg.preprocess.bandpass_high_hz, help="Band-pass high cutoff")
    parser.add_argument("--no-notch", action="store_true", help="Disable 60 Hz notch filter")
    return parser.parse_args()


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
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

    edf_path = build_default_file(cfg.data_root, args.subject, args.run)
    raw = load_raw_edf(edf_path)
    raw_pp = preprocess_raw(raw, cfg.preprocess)

    adm_events, adm_summary = generate_adm_events(raw_pp, cfg.adm)
    annotation_events = extract_annotation_events(raw, annotation_label_map=dataset_meta.annotation_label_map)
    qc = make_qc_summary(
        raw,
        raw_pp,
        args.subject,
        args.run,
        run_description_map=dataset_meta.run_description,
    )

    run_tag = f"{args.subject}_R{args.run:02d}"
    out_dirs = ensure_output_dirs(cfg.output_root / args.subject / run_tag)

    csv_path = out_dirs["csv"] / f"{run_tag}_events.csv"
    json_path = out_dirs["json"] / f"{run_tag}_summary.json"

    export_csv(adm_events, annotation_events, csv_path)

    payload = {
        "subject": args.subject,
        "run": args.run,
        "run_description": dataset_meta.run_description.get(args.run, "Unknown"),
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
        "csv_path": str(csv_path),
        "json_path": str(json_path),
        "plot_dir": str(out_dirs["plots"]),
        "adm_summary": adm_summary,
    }


def main() -> None:
    args = parse_args()
    result = run_pipeline(args)

    print("\n=== Pipeline Completed ===")
    print(f"Run            : {result['run_tag']}")
    print(f"CSV            : {result['csv_path']}")
    print(f"JSON           : {result['json_path']}")
    print(f"Plots          : {result['plot_dir']}")
    print(f"Total ADM evts : {result['adm_summary']['n_events_total']}")
    print(f"UP / DN        : {result['adm_summary']['n_up']} / {result['adm_summary']['n_dn']}")


if __name__ == "__main__":
    main()
