from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .config import DEFAULT_RUN_DESCRIPTION, PreprocessConfig

try:
    import mne
except ImportError as exc:  # pragma: no cover
    raise SystemExit("mne is required. Install it with: pip install mne") from exc


def build_default_file(data_root: Path, subject: str, run: int) -> Path:
    return data_root / subject / f"{subject}R{run:02d}.edf"


def load_raw_edf(edf_path: Path) -> "mne.io.BaseRaw":
    if not edf_path.exists():
        raise FileNotFoundError(f"EDF file not found: {edf_path}")
    return mne.io.read_raw_edf(edf_path, preload=True, verbose=False)


def preprocess_raw(raw: "mne.io.BaseRaw", cfg: PreprocessConfig) -> "mne.io.BaseRaw":
    '''
    Apply bandpass and notch filtering, and re-reference the data.
    raw: The raw MNE object containing the EEG data.
    cfg: PreprocessConfig object with filter settings.
    '''
    raw_pp = raw.copy()
    raw_pp.filter(
        l_freq=cfg.bandpass_low_hz,
        h_freq=cfg.bandpass_high_hz,
        verbose=False,
    )

    if cfg.apply_notch and cfg.notch_hz > 0:
        raw_pp.notch_filter(freqs=[cfg.notch_hz], verbose=False)

    if cfg.rereference == "average":
        raw_pp.set_eeg_reference("average", projection=False, verbose=False)

    return raw_pp


def make_qc_summary(
    raw: "mne.io.BaseRaw",
    raw_pp: "mne.io.BaseRaw",
    subject: str,
    run: int,
    run_description_map: dict[int, str] | None = None,
) -> dict[str, Any]:
    '''
    Generate a quality control summary for the raw and preprocessed data.
    raw: The original raw MNE object.
    raw_pp: The preprocessed raw MNE object.
    subject: Subject identifier (e.g., "S001").
    run: Run number (e.g., 1-14).
    '''
    data = raw_pp.get_data()
    duration_s = float(raw_pp.times[-1]) if len(raw_pp.times) else 0.0

    return {
        "subject": subject,
        "run": run,
        "run_description": (run_description_map or DEFAULT_RUN_DESCRIPTION).get(run, "Unknown"),
        "n_channels": int(raw_pp.info["nchan"]),
        "sampling_rate_hz": float(raw_pp.info["sfreq"]),
        "duration_s": duration_s,
        "has_nan": bool(np.isnan(data).any()),
        "signal_min_uv": float(np.min(data) * 1e6),
        "signal_max_uv": float(np.max(data) * 1e6),
        "signal_std_uv": float(np.std(data) * 1e6),
        "n_annotations": int(len(raw.annotations)),
    }
