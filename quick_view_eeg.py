"""Quick EEG EDF viewer for the EEG_data folder.

Usage examples:
  python quick_view_eeg.py
  python quick_view_eeg.py --subject S001 --run 4
  python quick_view_eeg.py --file EEG_data/S001/S001R06.edf --seconds 30
    python quick_view_eeg.py --subject S001 --run 4 --channel C3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from eeg_pipeline.config import DEFAULT_RUN_DESCRIPTION, load_dataset_metadata, resolve_edf_data_root
from eeg_pipeline.preprocessing import build_default_file

try:
    import mne
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "mne is required. Install it with: pip install mne matplotlib"
    ) from exc

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick EDF visualization for EEG data")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("EEG_datasets"),
        help="Root EEG data folder (default: EEG_datasets)",
    )
    parser.add_argument(
        "--subject",
        default="S001",
        help="Subject folder name like S001 (ignored if --file is set)",
    )
    parser.add_argument(
        "--run",
        type=int,
        default=1,
        choices=range(1, 15),
        metavar="[1-14]",
        help="Run number (default: 1)",
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=None,
        help="Optional explicit EDF path",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=20.0,
        help="Initial seconds to display in the raw viewer (default: 20)",
    )
    parser.add_argument(
        "--start",
        type=float,
        default=0.0,
        help="Start time in seconds for initial view (default: 0)",
    )
    parser.add_argument(
        "--channel",
        type=str,
        default=None,
        help="Optional single channel name to display (example: C3)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_meta = load_dataset_metadata(args.data_root)
    edf_data_root = resolve_edf_data_root(args.data_root, dataset_meta)
    run_description_map = dataset_meta.run_description or DEFAULT_RUN_DESCRIPTION

    edf_path = args.file or build_default_file(edf_data_root, args.subject, args.run)
    if not edf_path.exists():
        raise SystemExit(f"EDF file not found: {edf_path}")

    print(f"Loading: {edf_path}")
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

    sfreq = raw.info["sfreq"]
    n_ch = raw.info["nchan"]
    duration = raw.times[-1]

    print("\n=== Recording Summary ===")
    print(f"Run label      : R{args.run:02d} - {run_description_map.get(args.run, 'Unknown')}")
    print(f"Channels       : {n_ch}")
    print(f"Sampling rate  : {sfreq:.1f} Hz")
    print(f"Duration       : {duration:.1f} s")
    print(f"Channel names  : {', '.join(raw.ch_names[:10])} ...")

    picks = None
    n_channels_plot = min(20, n_ch)
    if args.channel is not None:
        if args.channel not in raw.ch_names:
            available = ", ".join(raw.ch_names)
            raise SystemExit(
                f"Channel '{args.channel}' not found. Available channels: {available}"
            )
        picks = [args.channel]
        n_channels_plot = 1
        print(f"Selected channel: {args.channel}")

    if len(raw.annotations) > 0:
        print(f"Annotations    : {len(raw.annotations)} found")
        print("First annotations:")
        for onset, desc in zip(raw.annotations.onset[:5], raw.annotations.description[:5]):
            print(f"  - t={onset:7.2f}s -> {desc}")
    else:
        print("Annotations    : none")

    # Interactive browser for quick inspection of channels and artifacts.
    raw.plot(
        duration=args.seconds,
        start=args.start,
        n_channels=n_channels_plot,
        picks=picks,
        scalings="auto",
        title=f"{edf_path.name} | R{args.run:02d}",
        show=True,
        block=False,
    )

    # Power spectral density gives a quick frequency-domain overview.
    raw.compute_psd(fmax=60, picks=picks).plot(average=True)

    plt.show()


if __name__ == "__main__":
    main()
