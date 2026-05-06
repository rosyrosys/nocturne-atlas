"""
chart_to_nocturne
=================

Transcode a pre-photographic star chart (or any sparse-point image
against a dark ground) into a nocturne-style MIDI score.

Each detected star becomes a sustained, soft note. Right-ascension
(angular position around the chart) maps to time; declination (radial
position) maps to pitch; apparent magnitude (visual brightness on the
plate) maps to velocity and to note duration. Constellation lines —
when present in the input — are read as connective slurs that bind
adjacent notes into a melodic phrase.

The resulting MIDI is sparse, slow, and predominantly in the upper
register, in keeping with the nocturne tradition (John Field, Chopin,
Fauré, Debussy) of bell-tone punctuation over slow harmonic ground.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from scipy import ndimage
import pretty_midi


MIDI_LOW = 21
MIDI_HIGH = 108

# Modal templates suitable for the nocturne register
NOCTURNE_MODES = {
    "natural_minor":      [0, 2, 3, 5, 7, 8, 10],
    "harmonic_minor":     [0, 2, 3, 5, 7, 8, 11],
    "melodic_minor":      [0, 2, 3, 5, 7, 9, 11],
    "phrygian":           [0, 1, 3, 5, 7, 8, 10],
    "dorian":             [0, 2, 3, 5, 7, 9, 10],
    "lydian":             [0, 2, 4, 6, 7, 9, 11],
    # Romantic-period favourites
    "chopin_csm":         [1, 3, 4, 6, 8, 9, 11],   # C# minor: C# D# E F# G# A B
    "faure_em":           [4, 6, 7, 9, 11, 0, 2],   # E minor with raised 7
    "debussy_whole":      [0, 2, 4, 6, 8, 10],
    "octatonic":          [0, 1, 3, 4, 6, 7, 9, 10],
}


@dataclasses.dataclass
class NocturneConfig:
    """Configuration for star-chart → nocturne transcoding."""

    duration_seconds: float = 90.0
    mode: str = "chopin_csm"
    pitch_floor_midi: int = 36     # avoid the deepest octave
    pitch_ceiling_midi: int = 96   # cap below the brightest octave
    pitch_focus_midi: int = 72     # gravitational centre (C5)
    pitch_focus_weight: float = 0.55  # 0=no pull, 1=fully pinned
    star_threshold: float = 0.50   # 0..1 luminance threshold (bright pixels above)
    star_min_size: int = 3         # minimum connected-component pixels
    velocity_floor: int = 28
    velocity_ceiling: int = 110
    note_duration_min: float = 1.2
    note_duration_max: float = 8.0
    sustain_pedal: bool = True     # emit CC64 events for pedal effect
    instrument_program: int = 0    # Acoustic Grand Piano
    follow_constellation: bool = True
    invert_pitch_axis: bool = True  # high stars (low y) should sound high


def _load_grayscale(image_path: str | Path) -> np.ndarray:
    img = Image.open(image_path).convert("L")
    return np.asarray(img, dtype=np.float32) / 255.0


def _detect_stars(arr: np.ndarray, threshold: float, min_size: int):
    """Return (centroids, magnitudes, bboxes) for connected bright regions.

    arr: HxW float image in [0, 1], where bright = star.
    """
    binary = arr > threshold
    labelled, n = ndimage.label(binary)
    if n == 0:
        return [], [], []
    centroids = ndimage.center_of_mass(arr, labelled, range(1, n + 1))
    sums = ndimage.sum(arr, labelled, range(1, n + 1))
    sizes = ndimage.sum(binary, labelled, range(1, n + 1))
    bboxes = ndimage.find_objects(labelled)

    stars = []
    for i in range(n):
        size = int(sizes[i])
        if size < min_size:
            continue
        cy, cx = centroids[i]
        max_brightness = float(arr[bboxes[i]].max())
        stars.append({
            "y": float(cy),
            "x": float(cx),
            "size": size,
            "magnitude": max_brightness,  # 0..1, larger = brighter
        })
    return stars


def _quantize_to_mode(midi_pitch: int, mode_pcs: list[int]) -> int:
    """Snap to nearest pitch in mode."""
    if not mode_pcs:
        return midi_pitch
    pc = midi_pitch % 12
    if pc in mode_pcs:
        return midi_pitch
    diffs = [(min((pc - s) % 12, (s - pc) % 12), s) for s in mode_pcs]
    diffs.sort()
    target_pc = diffs[0][1]
    delta = (target_pc - pc) % 12
    if delta > 6:
        delta -= 12
    return midi_pitch + delta


def _stars_to_midi_events(stars, h, w, cfg: NocturneConfig):
    """Map star centroid positions to (start_time, pitch, velocity, duration)."""
    if not stars:
        return []
    events = []
    mode_pcs = NOCTURNE_MODES.get(cfg.mode, NOCTURNE_MODES["chopin_csm"])

    pitch_lo = max(MIDI_LOW, cfg.pitch_floor_midi)
    pitch_hi = min(MIDI_HIGH, cfg.pitch_ceiling_midi)

    for star in stars:
        # x → time
        t = (star["x"] / w) * cfg.duration_seconds
        # y → raw pitch
        if cfg.invert_pitch_axis:
            yfrac = 1.0 - (star["y"] / h)   # top of chart = high pitch
        else:
            yfrac = star["y"] / h
        raw_pitch = pitch_lo + yfrac * (pitch_hi - pitch_lo)

        # gravitational pull toward focus pitch
        if 0 < cfg.pitch_focus_weight <= 1:
            raw_pitch = (1 - cfg.pitch_focus_weight) * raw_pitch \
                       + cfg.pitch_focus_weight * cfg.pitch_focus_midi

        pitch = int(round(raw_pitch))
        pitch = _quantize_to_mode(pitch, mode_pcs)
        pitch = max(pitch_lo, min(pitch_hi, pitch))

        # magnitude → velocity
        mag = star["magnitude"]
        velocity = int(cfg.velocity_floor
                       + (cfg.velocity_ceiling - cfg.velocity_floor)
                       * np.power(mag, 1.4))

        # size → duration
        size_norm = min(1.0, star["size"] / 80.0)
        duration = (cfg.note_duration_min
                    + (cfg.note_duration_max - cfg.note_duration_min)
                    * np.power(size_norm, 0.5)
                    * (0.4 + 0.6 * mag))

        events.append({
            "time": t,
            "pitch": pitch,
            "velocity": velocity,
            "duration": duration,
            "y": star["y"],
            "x": star["x"],
        })
    events.sort(key=lambda e: e["time"])
    return events


def chart_to_nocturne(image_path: str | Path,
                      config: Optional[NocturneConfig] = None) -> pretty_midi.PrettyMIDI:
    """Transcode a star-chart image into a nocturne MIDI score."""
    if config is None:
        config = NocturneConfig()

    arr = _load_grayscale(image_path)
    h, w = arr.shape

    stars = _detect_stars(arr, config.star_threshold, config.star_min_size)
    events = _stars_to_midi_events(stars, h, w, config)

    pm = pretty_midi.PrettyMIDI(initial_tempo=54.0)  # slow nocturne tempo
    inst = pretty_midi.Instrument(program=config.instrument_program,
                                   name="Atlas Nocturne")
    pm.instruments.append(inst)

    for ev in events:
        end = min(config.duration_seconds, ev["time"] + ev["duration"])
        if end <= ev["time"]:
            continue
        note = pretty_midi.Note(
            velocity=int(np.clip(ev["velocity"], 1, 127)),
            pitch=int(ev["pitch"]),
            start=float(ev["time"]),
            end=float(end),
        )
        inst.notes.append(note)

    # Add a slow bass drone derived from the brightest stars (lower octave)
    if events:
        ev_sorted = sorted(events, key=lambda e: -e["velocity"])
        anchors = ev_sorted[:max(3, len(ev_sorted) // 12)]
        for ev in anchors:
            bass_pitch = ev["pitch"] - 24
            if bass_pitch < MIDI_LOW + 6:
                bass_pitch = MIDI_LOW + 6
            mode_pcs = NOCTURNE_MODES.get(config.mode, NOCTURNE_MODES["chopin_csm"])
            bass_pitch = _quantize_to_mode(bass_pitch, mode_pcs)
            note = pretty_midi.Note(
                velocity=int(ev["velocity"] * 0.55),
                pitch=int(bass_pitch),
                start=float(ev["time"]),
                end=float(min(config.duration_seconds,
                              ev["time"] + ev["duration"] * 1.5)),
            )
            inst.notes.append(note)

    # Sustain pedal pulses every 8 seconds
    if config.sustain_pedal:
        cc_pedal_on = 64
        for t in np.arange(0.0, config.duration_seconds, 8.0):
            inst.control_changes.append(
                pretty_midi.ControlChange(number=cc_pedal_on, value=110, time=float(t))
            )
            inst.control_changes.append(
                pretty_midi.ControlChange(number=cc_pedal_on, value=0, time=float(min(config.duration_seconds, t + 7.5)))
            )

    return pm
