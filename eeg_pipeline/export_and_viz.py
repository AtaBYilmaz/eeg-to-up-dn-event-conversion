from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


def ensure_output_dirs(base_dir: Path) -> dict[str, Path]:
    paths = {
        "base": base_dir,
        "csv": base_dir / "csv",
        "json": base_dir / "json",
        "plots": base_dir / "plots",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def export_csv(event_rows: list[dict[str, Any]], annotation_rows: list[dict[str, Any]], out_csv: Path) -> None:
    event_df = pd.DataFrame(event_rows)
    if not event_df.empty:
        event_df.insert(0, "event_type", "ADM")

    annotation_df = pd.DataFrame(annotation_rows)

    merged = pd.concat([event_df, annotation_df], ignore_index=True, sort=False)
    if "timestamp_s" in merged.columns:
        merged = merged.sort_values(by=["timestamp_s"], na_position="last")

    merged.to_csv(out_csv, index=False)


def export_json(payload: dict[str, Any], out_json: Path) -> None:
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def plot_raw_with_events(raw_pp: Any, event_rows: list[dict[str, Any]], out_path: Path) -> None:
    sfreq = float(raw_pp.info["sfreq"])
    ch_names = raw_pp.ch_names
    data = raw_pp.get_data(picks=ch_names) * 1e6
    total_duration = float(data.shape[1] / sfreq) if sfreq > 0 else 0.0

    window_s = 20.0
    windows_per_figure = 6
    n_windows = max(1, int(np.ceil(total_duration / window_s)))

    by_channel_dir = out_path.parent / f"{out_path.stem}_by_channel"
    by_channel_dir.mkdir(parents=True, exist_ok=True)

    empty_ts = np.array([], dtype=float)
    up_ts_by_channel: dict[str, np.ndarray] = {}
    dn_ts_by_channel: dict[str, np.ndarray] = {}
    for ch_name in ch_names:
        up_ts_by_channel[ch_name] = np.array(
            [
                row["timestamp_s"]
                for row in event_rows
                if row.get("channel") == ch_name and row.get("polarity") == "UP" and "timestamp_s" in row
            ],
            dtype=float,
        )
        dn_ts_by_channel[ch_name] = np.array(
            [
                row["timestamp_s"]
                for row in event_rows
                if row.get("channel") == ch_name and row.get("polarity") == "DN" and "timestamp_s" in row
            ],
            dtype=float,
        )

    for ch_idx, ch_name in enumerate(ch_names):
        ch_data = data[ch_idx]
        ch_amp = float(np.nanmax(np.abs(ch_data))) if ch_data.size else 0.0
        y_limit = max(20.0, ch_amp * 1.2)
        up_ts = up_ts_by_channel.get(ch_name, empty_ts)
        dn_ts = dn_ts_by_channel.get(ch_name, empty_ts)

        for fig_start in range(0, n_windows, windows_per_figure):
            fig_end = min(fig_start + windows_per_figure, n_windows)
            n_rows = fig_end - fig_start

            fig, axes = plt.subplots(n_rows, 1, figsize=(14, 2.2 * n_rows), sharey=True)
            if n_rows == 1:
                axes = [axes]

            for row_idx, win_idx in enumerate(range(fig_start, fig_end)):
                ax = axes[row_idx]
                t0 = win_idx * window_s
                t1 = min(total_duration, (win_idx + 1) * window_s)
                s0 = int(max(0, np.floor(t0 * sfreq)))
                s1 = int(min(ch_data.size, np.ceil(t1 * sfreq)))

                if s1 <= s0:
                    continue

                t_seg = np.arange(s0, s1, dtype=float) / sfreq
                y_seg = ch_data[s0:s1]

                ax.plot(t_seg, y_seg, color="tab:blue", linewidth=0.7)

                up_seg = up_ts[(up_ts >= t0) & (up_ts < t1)] if up_ts.size else empty_ts
                dn_seg = dn_ts[(dn_ts >= t0) & (dn_ts < t1)] if dn_ts.size else empty_ts

                for ts in up_seg:
                    ax.axvline(float(ts), color="tab:green", alpha=0.25, linewidth=0.8)
                for ts in dn_seg:
                    ax.axvline(float(ts), color="tab:red", alpha=0.25, linewidth=0.8)

                ax.set_xlim(t0, t1)
                ax.set_ylim(-y_limit, y_limit)
                ax.grid(alpha=0.2)
                ax.set_ylabel("uV")
                ax.set_title(f"{ch_name} | {t0:.1f}s to {t1:.1f}s", fontsize=9)

            axes[-1].set_xlabel("Time (s)")
            fig.suptitle(f"Preprocessed EEG with ADM Events | Channel {ch_name}", fontsize=11)
            fig.tight_layout(rect=(0.0, 0.01, 1.0, 0.98))

            part_idx = fig_start // windows_per_figure + 1
            out_file = by_channel_dir / f"{_safe_name(ch_name)}_part{part_idx:02d}.svg"
            fig.savefig(out_file, format="svg")
            plt.close(fig)


def plot_event_rate(event_rows: list[dict[str, Any]], duration_s: float, out_path: Path, bin_s: float = 1.0) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))

    if not event_rows:
        ax.text(0.5, 0.5, "No ADM events generated", ha="center", va="center")
        ax.set_axis_off()
    else:
        ts = np.array([row["timestamp_s"] for row in event_rows], dtype=float)
        bins = np.arange(0.0, max(duration_s + bin_s, bin_s), bin_s)
        counts, edges = np.histogram(ts, bins=bins)
        ax.step(edges[:-1], counts, where="post", linewidth=1.5)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(f"Events per {bin_s:.1f}s")
        ax.set_title("ADM Event Rate Over Time")
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_channel_heatmap(event_rows: list[dict[str, Any]], ch_names: list[str], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))

    if not event_rows or not ch_names:
        ax.text(0.5, 0.5, "No channel events available", ha="center", va="center")
        ax.set_axis_off()
    else:
        counts = {ch: 0 for ch in ch_names}
        for row in event_rows:
            channel = row.get("channel")
            if channel in counts:
                counts[channel] += 1

        values = np.array([counts[ch] for ch in ch_names], dtype=float)
        heat = values.reshape(1, -1)
        im = ax.imshow(heat, aspect="auto", cmap="viridis")
        ax.set_yticks([0])
        ax.set_yticklabels(["events"])
        ax.set_xticks(np.arange(len(ch_names)))
        ax.set_xticklabels(ch_names, rotation=90, fontsize=7)
        ax.set_title("Event Count per Channel")
        fig.colorbar(im, ax=ax, orientation="vertical", label="count")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
