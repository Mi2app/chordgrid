from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
import logging

import librosa
import numpy as np

logger = logging.getLogger("chordgrid.analyzer")

NOTE_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


@dataclass(frozen=True)
class Template:
    suffix: str
    intervals: tuple[int, ...]
    complexity: float
    modes: tuple[str, ...]


TEMPLATES = [
    Template("", (0, 4, 7), 0.00, ("simple", "standard", "jazz")),
    Template("m", (0, 3, 7), 0.00, ("simple", "standard", "jazz")),
    Template("7", (0, 4, 7, 10), 0.025, ("simple", "standard", "jazz")),
    Template("maj7", (0, 4, 7, 11), 0.030, ("standard", "jazz")),
    Template("m7", (0, 3, 7, 10), 0.030, ("standard", "jazz")),
    Template("6", (0, 4, 7, 9), 0.050, ("standard", "jazz")),
    Template("m6", (0, 3, 7, 9), 0.050, ("standard", "jazz")),
    Template("sus2", (0, 2, 7), 0.095, ("standard", "jazz")),
    Template("sus4", (0, 5, 7), 0.090, ("standard", "jazz")),
    Template("dim", (0, 3, 6), 0.085, ("standard", "jazz")),
    Template("dim7", (0, 3, 6, 9), 0.100, ("jazz",)),
    Template("m7b5", (0, 3, 6, 10), 0.060, ("jazz",)),
    Template("aug", (0, 4, 8), 0.125, ("jazz",)),
    Template("9", (0, 2, 4, 7, 10), 0.080, ("jazz",)),
    Template("maj9", (0, 2, 4, 7, 11), 0.085, ("jazz",)),
    Template("m9", (0, 2, 3, 7, 10), 0.085, ("jazz",)),
    Template("m11", (0, 2, 3, 5, 7, 10), 0.110, ("jazz",)),
]


def _normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-9 else v


def estimate_key(global_chroma: np.ndarray) -> tuple[str, float, int, bool]:
    x = _normalize(global_chroma.astype(float))
    candidates: list[tuple[float, str, int, bool]] = []
    for root in range(12):
        candidates.append((float(np.dot(x, _normalize(np.roll(MAJOR_PROFILE, root)))), f"{NOTE_NAMES[root]} major", root, True))
        candidates.append((float(np.dot(x, _normalize(np.roll(MINOR_PROFILE, root)))), f"{NOTE_NAMES[root]} minor", root, False))
    candidates.sort(reverse=True)
    best, name, root, is_major = candidates[0]
    second = candidates[1][0]
    confidence = max(0.0, min(1.0, (best - second) * 3.5 + 0.55))
    return name, round(confidence, 3), root, is_major


def _diatonic_roots(key_root: int, is_major: bool) -> set[int]:
    intervals = (0, 2, 4, 5, 7, 9, 11) if is_major else (0, 2, 3, 5, 7, 8, 10)
    return {(key_root + i) % 12 for i in intervals}


def build_chord_templates(mode: str) -> list[tuple[str, int, np.ndarray, float]]:
    result = []
    for root in range(12):
        for t in TEMPLATES:
            if mode not in t.modes:
                continue
            vec = np.full(12, 0.025, dtype=float)
            for interval in t.intervals:
                vec[(root + interval) % 12] = 1.0
            vec[root] = 1.22
            result.append((f"{NOTE_NAMES[root]}{t.suffix}", root, _normalize(vec), t.complexity))
    return result


def _bass_chroma(y: np.ndarray, sr: int, hop_length: int) -> np.ndarray:
    # 3 octaves from C1: enough to identify likely bass/root while avoiding upper voicing clutter.
    cqt = np.abs(librosa.cqt(y=y, sr=sr, hop_length=hop_length, fmin=librosa.note_to_hz("C1"), n_bins=36, bins_per_octave=12))
    folded = np.zeros((12, cqt.shape[1]), dtype=float)
    for i in range(cqt.shape[0]):
        folded[i % 12] += cqt[i]
    denom = np.max(folded, axis=0, keepdims=True) + 1e-9
    return folded / denom


def _score_segment(chroma: np.ndarray, bass: np.ndarray, templates, diatonic_roots: set[int]) -> list[tuple[float, str, int]]:
    x = _normalize(chroma.astype(float))
    bass_pc = int(np.argmax(bass)) if float(np.sum(bass)) > 1e-8 else -1
    scored: list[tuple[float, str, int]] = []
    for name, root, template, complexity in templates:
        score = float(np.dot(x, template)) - complexity
        # v0.3: the bass is only a weak prior. A stronger bonus here tends to
        # collapse real slash chords (G/C -> C...) into a root-position chord.
        if root == bass_pc:
            score += 0.018
        if root in diatonic_roots:
            score += 0.018
        scored.append((score, name, root))
    scored.sort(reverse=True)
    return scored


def _bass_note(bass_segment: np.ndarray, mode: str) -> tuple[int, float]:
    """Return the most likely bass pitch class and a conservative confidence.

    The folded low-CQT contains harmonics as well as the fundamental, so we
    require both absolute dominance and separation from the runner-up before
    emitting a slash chord.
    """
    b = np.maximum(bass_segment.astype(float), 0.0)
    total = float(np.sum(b))
    if total < 1e-8:
        return -1, 0.0
    order = np.argsort(b)[::-1]
    first = int(order[0])
    second = int(order[1]) if len(order) > 1 else first
    share = float(b[first] / (total + 1e-9))
    margin = float((b[first] - b[second]) / (float(np.max(b)) + 1e-9))
    confidence = max(0.0, min(1.0, 0.55 * min(1.0, share / 0.28) + 0.45 * max(0.0, margin)))
    threshold = {"simple": 0.64, "standard": 0.58, "jazz": 0.52}.get(mode, 0.58)
    return (first, confidence) if confidence >= threshold else (-1, confidence)


def _viterbi(score_rows: list[list[tuple[float, str, int]]]) -> list[int]:
    if not score_rows:
        return []
    names = [row[0][1] for row in score_rows[:1]]  # dummy to keep type checkers quiet
    all_names = [x[1] for x in score_rows[0]]
    name_to_idx = {n: i for i, n in enumerate(all_names)}
    n_states = len(all_names)
    T = len(score_rows)
    dp = np.full((T, n_states), -1e9, dtype=float)
    back = np.zeros((T, n_states), dtype=int)
    first_map = {name: score for score, name, _ in score_rows[0]}
    for s, name in enumerate(all_names):
        dp[0, s] = first_map[name]

    for t in range(1, T):
        row_map = {name: (score, root) for score, name, root in score_rows[t]}
        prev_roots = {name: root for _, name, root in score_rows[t - 1]}
        for s, name in enumerate(all_names):
            obs, root = row_map[name]
            best_val = -1e9
            best_prev = 0
            for p, pname in enumerate(all_names):
                transition = 0.035 if pname == name else -0.018
                # Mild extra penalty for implausibly frantic root jumps, not enough to block real changes.
                proot = prev_roots.get(pname, root)
                distance = min((root - proot) % 12, (proot - root) % 12)
                if pname != name and distance >= 5:
                    transition -= 0.008
                val = dp[t - 1, p] + transition + obs
                if val > best_val:
                    best_val, best_prev = val, p
            dp[t, s] = best_val
            back[t, s] = best_prev

    state = int(np.argmax(dp[-1]))
    path = [state]
    for t in range(T - 1, 0, -1):
        state = int(back[t, state])
        path.append(state)
    return path[::-1]


def analyze_audio(path: str, filename: str = "audio", mode: str = "standard") -> dict:
    started = perf_counter()
    if mode not in {"simple", "standard", "jazz"}:
        mode = "standard"
    logger.info("analysis_start file=%s mode=%s", filename, mode)

    # v0.2: 16 kHz is a useful speed/quality compromise for harmonic analysis.
    y, sr = librosa.load(path, sr=16000, mono=True)
    duration = float(librosa.get_duration(y=y, sr=sr))
    if duration < 1.0:
        raise ValueError("audio trop court")

    y_harmonic = librosa.effects.harmonic(y, margin=4.0)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units="frames", hop_length=512)
    tempo = float(np.atleast_1d(tempo)[0])
    beat_frames = np.asarray(beat_frames, dtype=int)

    hop_length = 512
    chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr, hop_length=hop_length, n_chroma=12)
    bass = _bass_chroma(y_harmonic, sr, hop_length)
    global_chroma = np.mean(chroma, axis=1)
    key, key_confidence, key_root, is_major = estimate_key(global_chroma)
    diatonic = _diatonic_roots(key_root, is_major)

    if beat_frames.size < 2:
        approx_period = max(0.25, 60.0 / max(tempo, 90.0))
        beat_times = np.arange(0.0, duration + approx_period, approx_period)
        beat_frames = librosa.time_to_frames(beat_times, sr=sr, hop_length=hop_length)

    max_frames = min(chroma.shape[1], bass.shape[1])
    beat_frames = beat_frames[beat_frames < max_frames]
    if len(beat_frames) < 2:
        raise ValueError("impossible de détecter une pulsation stable")

    templates = build_chord_templates(mode)
    score_rows: list[list[tuple[float, str, int]]] = []
    frame_ranges: list[tuple[int, int]] = []
    for i, start_frame in enumerate(beat_frames):
        end_frame = beat_frames[i + 1] if i + 1 < len(beat_frames) else max_frames
        if end_frame <= start_frame:
            continue
        segment = np.median(chroma[:, start_frame:end_frame], axis=1)
        bass_segment = np.median(bass[:, start_frame:end_frame], axis=1)
        score_rows.append(_score_segment(segment, bass_segment, templates, diatonic))
        frame_ranges.append((int(start_frame), int(end_frame)))

    if not score_rows:
        raise ValueError("aucun accord détectable")

    path = _viterbi(score_rows)
    all_names = [x[1] for x in score_rows[0]]
    events: list[dict] = []
    for i, ((start_frame, end_frame), row, state) in enumerate(zip(frame_ranges, score_rows, path)):
        chosen_name = all_names[state]
        row_by_name = {name: score for score, name, _ in row}
        root_by_name = {name: root for _, name, root in row}
        chosen_root = root_by_name[chosen_name]
        best = row_by_name[chosen_name]
        ordered = sorted(row_by_name.items(), key=lambda x: x[1], reverse=True)
        second = ordered[1][1] if len(ordered) > 1 else best - 0.1
        margin = max(0.0, best - second)
        confidence = max(0.35, min(0.98, 0.48 + margin * 3.4 + (best - 0.72) * 0.7))

        bass_segment = np.median(bass[:, start_frame:end_frame], axis=1)
        bass_pc, bass_confidence = _bass_note(bass_segment, mode)
        display_name = chosen_name
        # Standard slash notation: chord/bass, e.g. G/C = G chord over C bass.
        if bass_pc >= 0 and bass_pc != chosen_root:
            display_name = f"{chosen_name}/{NOTE_NAMES[bass_pc]}"

        alternatives = [{"chord": n, "score": round(max(0.0, min(1.0, s)), 3)} for n, s in ordered[:3]]
        start = float(librosa.frames_to_time(start_frame, sr=sr, hop_length=hop_length))
        end = float(librosa.frames_to_time(end_frame, sr=sr, hop_length=hop_length))
        events.append({
            "id": i + 1,
            "start": round(start, 3),
            "end": round(min(end, duration), 3),
            "beat": (i % 4) + 1,
            "measure": (i // 4) + 1,
            "chord": display_name,
            "baseChord": chosen_name,
            "bass": NOTE_NAMES[bass_pc] if bass_pc >= 0 else None,
            "bassConfidence": round(bass_confidence, 3),
            "confidence": round(confidence, 3),
            "alternatives": alternatives,
        })

    elapsed = perf_counter() - started
    logger.info("analysis_done file=%s mode=%s duration=%.1fs tempo=%.1f key=%s elapsed=%.2fs beats=%d", filename, mode, duration, tempo, key, elapsed, len(events))

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
        "engine": "ChordGrid DSP v0.3",
        "mode": mode,
        "processingSeconds": round(elapsed, 2),
        "warning": "V0.3 : mesure 4/4 supposée. Les slash chords sont estimés à partir d’une analyse séparée du registre grave.",
    }
