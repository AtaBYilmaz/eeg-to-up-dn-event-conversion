from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_RUN_DESCRIPTION = {
    1: "Baseline, eyes open",
    2: "Baseline, eyes closed",
    3: "Task 1: move left/right fist",
    4: "Task 2: imagine left/right fist movement",
    5: "Task 3: move both fists or both feet",
    6: "Task 4: imagine both fists or both feet",
    7: "Task 1: move left/right fist",
    8: "Task 2: imagine left/right fist movement",
    9: "Task 3: move both fists or both feet",
    10: "Task 4: imagine both fists or both feet",
    11: "Task 1: move left/right fist",
    12: "Task 2: imagine left/right fist movement",
    13: "Task 3: move both fists or both feet",
    14: "Task 4: imagine both fists or both feet",
}


DEFAULT_ANNOTATION_LABEL_MAP = {
    "T0": "REST",
    "T1": "CLASS_1",
    "T2": "CLASS_2",
}


# Backward-compatible aliases for existing imports.
RUN_DESCRIPTION = DEFAULT_RUN_DESCRIPTION
ANNOTATION_LABEL_MAP = DEFAULT_ANNOTATION_LABEL_MAP


@dataclass
class DatasetMetadata:
    dataset_id: str = "default"
    dataset_name: str = "Default metadata"
    dataset_root: Path | None = None
    edf_root_relative: str = ""
    run_description: dict[int, str] = field(default_factory=lambda: dict(DEFAULT_RUN_DESCRIPTION))
    annotation_label_map: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_ANNOTATION_LABEL_MAP))

    @property
    def edf_root(self) -> Path | None:
        if self.dataset_root is None:
            return None
        if not self.edf_root_relative:
            return self.dataset_root
        return self.dataset_root / self.edf_root_relative


def _coerce_run_description(value: Any) -> dict[int, str]:
    if not isinstance(value, dict):
        return dict(DEFAULT_RUN_DESCRIPTION)

    parsed: dict[int, str] = {}
    for key, label in value.items():
        try:
            run = int(key)
        except (TypeError, ValueError):
            continue
        parsed[run] = str(label)

    return parsed or dict(DEFAULT_RUN_DESCRIPTION)


def _coerce_annotation_label_map(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return dict(DEFAULT_ANNOTATION_LABEL_MAP)

    parsed = {str(key): str(label) for key, label in value.items()}
    return parsed


def load_dataset_metadata(data_root: Path) -> DatasetMetadata:
    candidates = [data_root / "dataset_meta.json", data_root.parent / "dataset_meta.json"]

    metadata_path = next((path for path in candidates if path.exists()), None)
    if metadata_path is None:
        return DatasetMetadata(dataset_root=data_root)

    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return DatasetMetadata(dataset_root=data_root)

    dataset_root = metadata_path.parent
    return DatasetMetadata(
        dataset_id=str(payload.get("dataset_id", "default")),
        dataset_name=str(payload.get("dataset_name", "Default metadata")),
        dataset_root=dataset_root,
        edf_root_relative=str(payload.get("edf_root_relative", "")),
        run_description=_coerce_run_description(payload.get("run_description")),
        annotation_label_map=_coerce_annotation_label_map(payload.get("annotation_label_map")),
    )


def resolve_edf_data_root(requested_data_root: Path, metadata: DatasetMetadata) -> Path:
    edf_root = metadata.edf_root
    if edf_root is not None and edf_root.exists():
        return edf_root
    return requested_data_root


@dataclass
class PreprocessConfig:
    bandpass_low_hz: float = 1.0
    bandpass_high_hz: float = 40.0
    notch_hz: float = 60.0
    apply_notch: bool = True
    rereference: str = "average"


@dataclass
class AdmConfig:
    threshold_uv: float = 120.0
    refractory_ms: float = 25.0
    suppression_ms: float = 25.0
    recovery_tc_ms: float = 150.0
    post_event_gain: float = 0.25
    event_method: str = "suppression_recovery"


@dataclass
class PipelineConfig:
    data_root: Path = Path("EEG_datasets")
    output_root: Path = Path("outputs")
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    adm: AdmConfig = field(default_factory=AdmConfig)
