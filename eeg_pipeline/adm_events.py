from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

from .config import DEFAULT_ANNOTATION_LABEL_MAP, AdmConfig


@dataclass
class EventRow:
    timestamp_s: float
    sample_index: int
    channel: str
    polarity: str
    amplitude_uv: float


DetectorFn = Callable[[Any, float, list[str], AdmConfig], list[EventRow]]


def detect_events_threshold_refractory(
    data: Any,
    sfreq: float,
    ch_names: list[str],
    cfg: AdmConfig,
) -> list[EventRow]:
    '''
    Uses an adaptive baseline per channel (initialized at first sample and updated at each event).
    An UP event is emitted when current sample - baseline >= threshold.
    A DN event is emitted when current sample - baseline <= -threshold.
    After any event, baseline is reset to the event sample, and refractory time blocks nearby re-triggers.
    '''

    # Convert user threshold from microvolts to volts because raw data are in volts.
    threshold_v = cfg.threshold_uv * 1e-6
    # Convert refractory window from milliseconds to samples, with a minimum of 1 sample.
    refractory_samples = max(1, int((cfg.refractory_ms / 1000.0) * sfreq))
    events: list[EventRow] = []

    for ch_idx, channel in enumerate(ch_names):
        x = data[ch_idx]
        # Baseline starts at the first sample and is updated after each event.
        baseline = x[0]
        # Negative start allows event detection immediately at beginning of the trace.
        last_event_sample = -refractory_samples

        for sample_idx in range(1, len(x)):
            # Skip samples inside the post-event lockout window.
            if sample_idx - last_event_sample < refractory_samples:
                continue

            delta = x[sample_idx] - baseline
            if delta >= threshold_v:
                events.append(
                    EventRow(
                        timestamp_s=sample_idx / sfreq,
                        sample_index=sample_idx,
                        channel=channel,
                        polarity="UP",
                        amplitude_uv=float(delta * 1e6),
                    )
                )
                # Anchor future deltas to the point where this event occurred.
                baseline = x[sample_idx]
                last_event_sample = sample_idx
            elif delta <= -threshold_v:
                events.append(
                    EventRow(
                        timestamp_s=sample_idx / sfreq,
                        sample_index=sample_idx,
                        channel=channel,
                        polarity="DN",
                        amplitude_uv=float(delta * 1e6),
                    )
                )
                # Same baseline reset rule for downward events.
                baseline = x[sample_idx]
                last_event_sample = sample_idx

    return events


def detect_events_sample_to_sample(
    data: Any,
    sfreq: float,
    ch_names: list[str],
    cfg: AdmConfig,
) -> list[EventRow]:
    '''
    Uses the local slope between consecutive samples.
    An UP event is emitted when x[t] - x[t-1] >= threshold.
    A DN event is emitted when x[t] - x[t-1] <= -threshold.
    Refractory time is still enforced to avoid counting one sharp transient multiple times.
    '''

    # Convert user threshold from microvolts to volts because raw data are in volts.
    threshold_v = cfg.threshold_uv * 1e-6
    # Convert refractory window from milliseconds to samples, with a minimum of 1 sample.
    refractory_samples = max(1, int((cfg.refractory_ms / 1000.0) * sfreq))
    events: list[EventRow] = []

    for ch_idx, channel in enumerate(ch_names):
        x = data[ch_idx]
        # Negative start allows event detection immediately at beginning of the trace.
        last_event_sample = -refractory_samples

        for sample_idx in range(1, len(x)):
            # Skip samples inside the post-event lockout window.
            if sample_idx - last_event_sample < refractory_samples:
                continue

            # Local first difference: captures abrupt point-to-point changes.
            delta = x[sample_idx] - x[sample_idx - 1]
            if delta >= threshold_v:
                events.append(
                    EventRow(
                        timestamp_s=sample_idx / sfreq,
                        sample_index=sample_idx,
                        channel=channel,
                        polarity="UP",
                        amplitude_uv=float(delta * 1e6),
                    )
                )
                last_event_sample = sample_idx
            elif delta <= -threshold_v:
                events.append(
                    EventRow(
                        timestamp_s=sample_idx / sfreq,
                        sample_index=sample_idx,
                        channel=channel,
                        polarity="DN",
                        amplitude_uv=float(delta * 1e6),
                    )
                )
                last_event_sample = sample_idx

    return events


def detect_events_suppression_recovery(
    data: Any,
    sfreq: float,
    ch_names: list[str],
    cfg: AdmConfig,
) -> list[EventRow]:
    '''
    Uses adaptive baseline thresholding with post-event gain suppression.
    Detection operates on the suppressed signal, where gain drops after each event
    and exponentially recovers toward unity with a time constant.
    '''

    # Convert user threshold from microvolts to volts because raw data are in volts.
    threshold_v = cfg.threshold_uv * 1e-6
    suppression_samples = max(1, int((cfg.suppression_ms / 1000.0) * sfreq))

    recovery_tc_s = cfg.recovery_tc_ms / 1000.0
    if recovery_tc_s <= 0:
        raise ValueError("recovery_tc_ms must be > 0 for suppression_recovery method")

    post_event_gain = float(cfg.post_event_gain)
    if not (0.0 < post_event_gain <= 1.0):
        raise ValueError("post_event_gain must satisfy 0 < post_event_gain <= 1")

    # Per-sample exponential recovery: gain <- 1 - (1 - gain) * alpha
    alpha = math.exp(-1.0 / (sfreq * recovery_tc_s))
    events: list[EventRow] = []

    for ch_idx, channel in enumerate(ch_names):
        x = data[ch_idx]
        baseline = x[0]
        gain = 1.0
        last_event_sample = -suppression_samples

        for sample_idx in range(1, len(x)):
            gain = 1.0 - (1.0 - gain) * alpha

            # Suppression window replaces refractory behavior for this method.
            if sample_idx - last_event_sample < suppression_samples:
                continue

            suppressed_sample = baseline + gain * (x[sample_idx] - baseline)
            delta_eff = suppressed_sample - baseline

            if delta_eff >= threshold_v:
                events.append(
                    EventRow(
                        timestamp_s=sample_idx / sfreq,
                        sample_index=sample_idx,
                        channel=channel,
                        polarity="UP",
                        amplitude_uv=float(delta_eff * 1e6),
                    )
                )
                baseline = x[sample_idx]
                gain = post_event_gain
                last_event_sample = sample_idx
            elif delta_eff <= -threshold_v:
                events.append(
                    EventRow(
                        timestamp_s=sample_idx / sfreq,
                        sample_index=sample_idx,
                        channel=channel,
                        polarity="DN",
                        amplitude_uv=float(delta_eff * 1e6),
                    )
                )
                baseline = x[sample_idx]
                gain = post_event_gain
                last_event_sample = sample_idx

    return events


EVENT_DETECTORS: dict[str, DetectorFn] = {
    "threshold_refractory": detect_events_threshold_refractory,
    "sample_to_sample": detect_events_sample_to_sample,
    "suppression_recovery": detect_events_suppression_recovery,
}


def generate_adm_events(raw_pp: Any, cfg: AdmConfig) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    data = raw_pp.get_data()
    sfreq = float(raw_pp.info["sfreq"])
    ch_names = list(raw_pp.ch_names)

    # Identify the event detection method to use
    method_name = str(getattr(cfg, "event_method", "threshold_refractory")).strip()
    detector = EVENT_DETECTORS.get(method_name)
    if detector is None:
        valid = ", ".join(sorted(EVENT_DETECTORS))
        raise ValueError(f"Unknown ADM event method '{method_name}'. Valid methods: {valid}")

    # Detect events using the selected method
    events = detector(data, sfreq, ch_names, cfg)

    # The events are sorted by sample index, then by channel name, and finally by polarity (UP before DN)
    events.sort(key=lambda e: (e.sample_index, e.channel, e.polarity))

    n_up = sum(1 for e in events if e.polarity == "UP")
    n_dn = sum(1 for e in events if e.polarity == "DN")

    event_rows = [
        {
            "timestamp_s": e.timestamp_s,
            "sample_index": e.sample_index,
            "channel": e.channel,
            "polarity": e.polarity,
            "amplitude_uv": e.amplitude_uv,
        }
        for e in events
    ]

    summary = {
        "n_events_total": len(event_rows),
        "n_up": n_up,
        "n_dn": n_dn,
        "up_dn_ratio": float(n_up / n_dn) if n_dn else None,
        "threshold_uv": cfg.threshold_uv,
        "refractory_ms": cfg.refractory_ms,
        "suppression_ms": float(getattr(cfg, "suppression_ms", 0.0)),
        "recovery_tc_ms": float(getattr(cfg, "recovery_tc_ms", 0.0)),
        "post_event_gain": float(getattr(cfg, "post_event_gain", 1.0)),
        "event_method": method_name,
    }
    return event_rows, summary


def extract_annotation_events(
    raw: Any,
    annotation_label_map: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    label_map = annotation_label_map or DEFAULT_ANNOTATION_LABEL_MAP
    rows: list[dict[str, Any]] = []
    for onset, duration, desc in zip(raw.annotations.onset, raw.annotations.duration, raw.annotations.description):
        desc_clean = str(desc).strip()
        desc_key = desc_clean.replace(" ", "")
        label = label_map.get(desc_key, desc_clean)
        rows.append(
            {
                "event_type": "ANNOTATION",
                "label": label,
                "raw_label": desc_clean,
                "onset_s": float(onset),
                "duration_s": float(duration),
            }
        )
    return rows
