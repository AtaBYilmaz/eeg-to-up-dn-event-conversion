# EEG Processing Pipeline

This repository provides an end-to-end EEG processing workflow for EDF recordings:

1. Load raw EDF for a subject/run
2. Preprocess EEG (band-pass, optional notch, average re-reference)
3. Detect ADM-style events (multiple detector modes)
4. Parse dataset annotations
5. Export merged event tables and run summary
6. Generate visualizations for review

The main entrypoint is `main.py`. Core implementation lives in the `eeg_pipeline/` package.

## Repository Layout

- `main.py`: Primary CLI entrypoint for the full EEG pipeline
- `eeg_pipeline/config.py`: Pipeline and dataset metadata configuration
- `eeg_pipeline/preprocessing.py`: EDF loading, preprocessing, QC summary
- `eeg_pipeline/adm_events.py`: Event detectors and annotation extraction
- `eeg_pipeline/export_and_viz.py`: CSV/JSON export and plotting
- `quick_view_eeg.py`: Optional quick viewer for EDF sanity checks
- `csv_to_edf.py`: Utility for converting ADHD CSV format into EDF files
- `tests/`: Lightweight tests for deterministic logic

## Dataset Handling

This repository does not track raw datasets in git.

Expected default data root is `EEG_datasets/`, for example:

- `EEG_datasets/eeg-motor-movementimagery-dataset-1.0.0/files/S001/S001R01.edf`

You can always override paths with CLI flags.

## Setup

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run The Main Pipeline

```bash
python main.py --subject S001 --run 2
```

Common options:

- `--data-root`: Root dataset path
- `--output-root`: Output folder (default `outputs`)
- `--threshold-uv`: ADM threshold
- `--event-method`: `threshold_refractory`, `sample_to_sample`, or `suppression_recovery`
- `--l-freq` / `--h-freq`: Band-pass limits
- `--no-notch`: Disable notch filtering

Use built-in help for all options:

```bash
python main.py --help
```

## Quick EDF Viewer

```bash
python quick_view_eeg.py --subject S001 --run 4
```

This opens an interactive MNE browser and PSD figure for rapid inspection.

## Outputs

Per run, outputs are written under:

- `outputs/<SUBJECT>/<SUBJECT>_R<NN>/csv/`
- `outputs/<SUBJECT>/<SUBJECT>_R<NN>/json/`
- `outputs/<SUBJECT>/<SUBJECT>_R<NN>/plots/`

Artifacts include:

- merged event CSV (`ADM` + annotation rows)
- run summary JSON (QC + ADM summary + config)
- event rate and channel heatmap plots
- per-channel SVG review plots

## Tests

```bash
pytest
```

Tests focus on deterministic behavior (config defaults, detector selection, annotation mapping).

## Publishing Checklist

- Confirm `.gitignore` excludes datasets and generated artifacts
- Run one smoke pipeline command and verify outputs
- Run `pytest`
- Update README examples if CLI flags change
