from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import librosa
import numpy as np

NOTE_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]

MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


@dataclass(frozen=True)
class Template:
    suffix: str
    intervals: tuple[int, ...]
    weight: float = 1.0


TEMPLATES = [
    Template("", (0, 4, 7), 1.00),
    Template("m", (0, 3, 7), 1.00),
    Template("7", (0, 4, 7, 10), 1.06),
    Template("maj7", (0, 4, 7, 11), 1.06),
    Template("m7", (0, 3, 7, 10), 1.06),
    Template("6", (0, 4, 7, 9), 1.02),
    Template("m6", (0, 3, 7, 9), 1.02),
    Template("sus2", (0, 2, 7), 0.97),
    Template("sus4", (0, 5, 7), 0.97),
    Template("dim", (0, 3, 6), 0.99),
    Template("dim7", (0, 3, 6, 9), 1.02),
    Template("m7b5", (0, 3, 6, 10), 1.04),
    Template("aug", (0, 4, 8), 0.97),
    Template("9", (0, 2, 4, 7, 10), 1.04),
    Template("maj9", (0, 2, 4, 7, 11), 1.04),
    Template("m9", (0, 2, 3, 7, 10), 1.04),
]


def _normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-9 else v


def estimate_key(global_chroma: np.ndarray) -> tuple[str, float]:
    x = _normalize(global_chroma.astype(float))
    candidates: list[tuple[float, str]] = []
    for root in range(12):
        maj = _normalize(np.roll(MAJOR_PROFILE, root))
        min_ = _normalize(np.roll(MINOR_PROFILE, root))
        candidates.append((float(np.dot(x, maj)), f"{NOTE_NAMES[root]} major"))
        candidates.append((float(np.dot(x, min_)), f"{NOTE_NAMES[root]} minor"))
    candidates.sort(reverse=True)
    best, name = candidates[0]
    second = candidates[1][0]
    confidence = max(0.0, min(1.0, (best - second) * 3.5 + 0.55))
    return name, round(confidence, 3)


def build_chord_templates() -> list[tuple[str, np.ndarray, float]]:
    result = []
    for root in range(12):
        for t in TEMPLATES:
            vec = np.full(12, 0.04, dtype=float)
            for interval in t.intervals:
                vec[(root + interval) % 12] = 1.0
            # Root is slightly more important than upper tones.
            vec[root] = 1.18
            result.append((f"{NOTE_NAMES[root]}{t.suffix}", _normalize(vec), t.weight))
    return result


CHORD_TEMPLATES = build_chord_templates()


def classify_chroma(chroma: np.ndarray) -> tuple[str, float, list[dict]]:
    if float(np.sum(chroma)) < 1e-6:
        return "N.C.", 0.0, []

    x = _normalize(chroma.astype(float))
    scored: list[tuple[float, str]] = []
    for name, template, complexity_weight in CHORD_TEMPLATES:
        raw = float(np.dot(x, template))
        # Small penalty prevents extended chords winning on incidental overtones.
        score = raw / complexity_weight
        scored.append((score, name))
    scored.sort(reverse=True)

    best_score, best_name = scored[0]
    second_score = scored[1][0]
    margin = max(0.0, best_score - second_score)
    confidence = max(0.0, min(0.99, 0.42 + margin * 3.2 + (best_score - 0.65) * 0.85))
    alternatives = [
        {"chord": name, "score": round(max(0.0, min(1.0, score)), 3)}
        for score, name in scored[:3]
    ]
    return best_name, round(confidence, 3), alternatives


def _smooth_labels(events: list[dict]) -> list[dict]:
    if len(events) < 3:
        return events
    result = [dict(e) for e in events]
    for i in range(1, len(events) - 1):
        prev_c = events[i - 1]["chord"]
        this_c = events[i]["chord"]
        next_c = events[i + 1]["chord"]
        if prev_c == next_c != this_c and events[i]["confidence"] < 0.68:
            result[i]["chord"] = prev_c
            result[i]["confidence"] = round((events[i - 1]["confidence"] + events[i + 1]["confidence"]) / 2, 3)
    return result


def _group_repeated(events: Iterable[dict]) -> list[dict]:
    grouped: list[dict] = []
    for event in events:
        if grouped and grouped[-1]["chord"] == event["chord"]:
            grouped[-1]["end"] = event["end"]
            grouped[-1]["beats"] += 1
            grouped[-1]["confidence"] = round(
                (grouped[-1]["confidence"] * (grouped[-1]["beats"] - 1) + event["confidence"])
                / grouped[-1]["beats"],
                3,
            )
        else:
            grouped.append({**event, "beats": 1})
    return grouped


def analyze_audio(path: str, filename: str = "audio") -> dict:
    y, sr = librosa.load(path, sr=22050, mono=True)
    duration = float(librosa.get_duration(y=y, sr=sr))
    if duration < 1.0:
        raise ValueError("audio trop court")

    # Harmonic source is more stable for chord recognition than the full mix.
    y_harmonic = librosa.effects.harmonic(y, margin=4.0)

    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units="frames")
    tempo = float(np.atleast_1d(tempo)[0])
    beat_frames = np.asarray(beat_frames, dtype=int)

    if beat_frames.size < 2:
        # Fallback grid at estimated quarter notes.
        hop_length = 512
        approx_period = max(0.25, 60.0 / max(tempo, 90.0))
        beat_times = np.arange(0.0, duration + approx_period, approx_period)
        beat_frames = librosa.time_to_frames(beat_times, sr=sr, hop_length=hop_length)

    hop_length = 512
    chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr, hop_length=hop_length, n_chroma=12)
    global_chroma = np.mean(chroma, axis=1)
    key, key_confidence = estimate_key(global_chroma)

    # Align chroma to beat boundaries.
    beat_frames = beat_frames[beat_frames < chroma.shape[1]]
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)
    if len(beat_times) < 2:
        raise ValueError("impossible de détecter une pulsation stable")

    events: list[dict] = []
    for i, start_frame in enumerate(beat_frames):
        end_frame = beat_frames[i + 1] if i + 1 < len(beat_frames) else chroma.shape[1]
        if end_frame <= start_frame:
            continue
        segment = np.median(chroma[:, start_frame:end_frame], axis=1)
        chord, confidence, alternatives = classify_chroma(segment)
        start = float(librosa.frames_to_time(start_frame, sr=sr, hop_length=hop_length))
        end = float(librosa.frames_to_time(end_frame, sr=sr, hop_length=hop_length))
        beat_number = (i % 4) + 1
        measure = (i // 4) + 1
        events.append(
            {
                "id": i + 1,
                "start": round(start, 3),
                "end": round(min(end, duration), 3),
                "beat": beat_number,
                "measure": measure,
                "chord": chord,
                "confidence": confidence,
                "alternatives": alternatives,
            }
        )

    events = _smooth_labels(events)

    return {
        "filename": filename,
        "duration": round(duration, 2),
        "tempo": round(tempo, 1),
        "key": key,
        "keyConfidence": key_confidence,
        "timeSignature": "4/4",
        "beatsPerBar": 4,
        "chords": events,
        "sections": [],
        "engine": "ChordGrid DSP v0.1",
        "warning": "V1: mesure 4/4 supposée; les accords enrichis restent des estimations à valider à l'oreille.",
    }
