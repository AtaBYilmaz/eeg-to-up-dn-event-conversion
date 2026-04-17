from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

try:
    import mne
except ImportError as exc:  # pragma: no cover
    raise SystemExit("mne is required. Install it with: pip install mne edfio") from exc


DEFAULT_ADHD_CHANNELS: list[str] = [
    "Fz",
    "Cz",
    "Pz",
    "C3",
    "T3",
    "C4",
    "T4",
    "Fp1",
    "Fp2",
    "F3",
    "F4",
    "F7",
    "F8",
    "P3",
    "P4",
    "T5",
    "T6",
    "O1",
    "O2",
]

CHANNEL_ALIASES: dict[str, list[str]] = {
    "T3": ["T3", "T7"],
    "T4": ["T4", "T8"],
    "T5": ["T5", "P7"],
    "T6": ["T6", "P8"],
}


def _subject_code(index_1_based: int) -> str:
    return f"S{index_1_based:03d}"


def _resolve_source_column(channel_name: str, header: Sequence[str]) -> str:
    candidates = CHANNEL_ALIASES.get(channel_name, [channel_name])
    for candidate in candidates:
        if candidate in header:
            return candidate
    raise ValueError(f"EEG column not found in CSV: {channel_name}")


def csv_to_edf_by_id(
    csv_path: Path,
    output_dir: Path,
    sfreq: float = 128.0,
    id_col: str = "ID",
    class_col: str = "Class",
    eeg_columns: Iterable[str] | None = DEFAULT_ADHD_CHANNELS,
) -> list[Path]:
    """
    Convert an EEG CSV into EDF files using a PhysioNet-like layout.

    Expected CSV format (matching `adhdata.csv`):
    - 19 EEG channel columns
    - `Class` column (optional annotation label)
    - `ID` column used to split subjects/records

    Parameters
    ----------
    csv_path:
        Input CSV path.
    output_dir:
        Folder where EDF files are written.
    sfreq:
        Sampling frequency in Hz. Must match the source dataset.
    id_col:
        Column used to split rows into separate subjects.
    class_col:
        Label column used to create a full-recording annotation.
    eeg_columns:
        Explicit EEG columns. If None, all columns except `id_col` and
        `class_col` are treated as EEG channels.

    Returns
    -------
    list[Path]
        Written EDF paths.

    Side effects
    ------------
    Writes these companion files in `output_dir`:
    - `RECORDS`: one relative EDF path per line (PhysioNet-like index)
    - `id_map.csv`: maps original participant ID to generated subject code
    """
    if sfreq <= 0:
        raise ValueError("sfreq must be > 0")

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("CSV has no header")

        header = reader.fieldnames
        if id_col not in header:
            raise ValueError(f"Missing ID column: {id_col}")

        if eeg_columns is None:
            eeg_cols = [c for c in header if c not in {id_col, class_col}]
        else:
            eeg_cols = [c for c in eeg_columns]

        if not eeg_cols:
            raise ValueError("No EEG channel columns found")

        source_col_for_channel = {ch: _resolve_source_column(ch, header) for ch in eeg_cols}

        grouped: dict[str, dict[str, object]] = {}

        for row in reader:
            rec_id = str(row[id_col]).strip()
            if not rec_id:
                continue

            if rec_id not in grouped:
                grouped[rec_id] = {
                    "class_label": str(row.get(class_col, "")).strip(),
                    "channels": {ch: [] for ch in eeg_cols},
                }

            channel_store = grouped[rec_id]["channels"]
            assert isinstance(channel_store, dict)

            for ch in eeg_cols:
                raw_val = row.get(source_col_for_channel[ch], "")
                if raw_val is None or raw_val == "":
                    channel_store[ch].append(np.nan)
                else:
                    channel_store[ch].append(float(raw_val))

    sorted_ids = sorted(grouped.keys())
    written: list[Path] = []
    records_lines: list[str] = []
    id_map_rows: list[tuple[str, str, str, int]] = []

    for idx, rec_id in enumerate(sorted_ids, start=1):
        payload = grouped[rec_id]
        channel_store = payload["channels"]
        class_label = str(payload.get("class_label", ""))
        assert isinstance(channel_store, dict)

        data_uv = np.array([channel_store[ch] for ch in eeg_cols], dtype=np.float64)

        # Fill missing values per-channel with channel mean (or zero if all-NaN).
        for i in range(data_uv.shape[0]):
            row = data_uv[i]
            if np.isnan(row).all():
                data_uv[i] = 0.0
            elif np.isnan(row).any():
                mean_val = np.nanmean(row)
                data_uv[i] = np.where(np.isnan(row), mean_val, row)

        # MNE expects Volts for EEG channels.
        data_v = data_uv * 1e-6

        info = mne.create_info(ch_names=eeg_cols, sfreq=sfreq, ch_types="eeg")
        raw = mne.io.RawArray(data_v, info, verbose=False)

        if class_label:
            duration = float(raw.n_times / sfreq)
            raw.set_annotations(
                mne.Annotations(
                    onset=[0.0],
                    duration=[duration],
                    description=[class_label],
                )
            )

        subject = _subject_code(idx)
        run_name = f"{subject}R01"
        subject_dir = output_dir / subject
        subject_dir.mkdir(parents=True, exist_ok=True)
        out_path = subject_dir / f"{run_name}.edf"
        try:
            mne.export.export_raw(out_path, raw, fmt="edf", physical_range="auto", overwrite=True)
        except RuntimeError as exc:
            if "edfio" in str(exc).lower():
                raise RuntimeError("EDF export requires 'edfio'. Install with: pip install edfio") from exc
            raise

        written.append(out_path)
        records_lines.append(f"{subject}/{run_name}.edf")
        id_map_rows.append((subject, rec_id, class_label, int(raw.n_times)))

    if not written:
        raise ValueError("No EDF files were written. Check that ID values are present.")

    records_path = output_dir / "RECORDS"
    records_path.write_text("\n".join(records_lines) + "\n", encoding="utf-8")

    id_map_path = output_dir / "id_map.csv"
    with id_map_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["subject", "source_id", "class", "n_samples", "sfreq_hz"])
        for subject, source_id, class_label, n_samples in id_map_rows:
            writer.writerow([subject, source_id, class_label, n_samples, sfreq])

    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert ADHD CSV format to a PhysioNet-like EDF layout")
    parser.add_argument("--csv", type=Path, required=True, help="Input CSV path")
    parser.add_argument("--out", type=Path, required=True, help="Output root directory")
    parser.add_argument("--sfreq", type=float, default=128.0, help="Sampling rate in Hz (default: 128)")
    parser.add_argument("--id-col", default="ID", help="ID column name")
    parser.add_argument("--class-col", default="Class", help="Class column name")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = csv_to_edf_by_id(
        csv_path=args.csv,
        output_dir=args.out,
        sfreq=args.sfreq,
        id_col=args.id_col,
        class_col=args.class_col,
    )
    print(f"Wrote {len(paths)} EDF files to: {args.out}")
    print(f"Created index: {args.out / 'RECORDS'}")
    print(f"Created ID map: {args.out / 'id_map.csv'}")


if __name__ == "__main__":
    main()
