"""
quadrivium.music
================

Numbers in time: the chart-to-nocturne pipeline.

Detected stars on a planispheric chart are mapped to MIDI events whose
horizontal coordinate determines elapsed time, vertical coordinate
determines voice register and pitch (with axis inversion so high stars
sound high), peak luminance determines velocity, and apparent size
determines duration.

The pipeline now expresses two structural commitments inherited from the
salon-nocturne tradition:

1. **Two-voice texture.** Stars in the upper half of the chart are routed
   to the right-hand cantabile voice and quantised under a voice-leading
   constraint that minimises melodic leaps. Stars in the lower half are
   routed to the left hand, where their original pitches are *replaced*
   by a rolling arpeggio drawn from the chord active at that moment.
   The result is the characteristic Chopin/Field nocturne texture of a
   broken-chord bass under a sung melodic line.

2. **Phrase-and-progression form.** The duration is divided into N
   phrases (default four). For each phrase the brightest star within
   that segment is taken as the chord anchor; its pitch is snapped to
   the active mode and a triad (or seventh) is built from it by
   stacking thirds within the mode. A Tetractys stack at the chord
   root sustains beneath the phrase, the sustain pedal is depressed
   at each chord change, and a hairpin velocity envelope is applied
   within each phrase. A small breath-rest separates phrases.

A bass drone organised by the Tetractys (1, 2, 3, 4 → octave below,
fourth, root, fifth) provides the harmonic ground beneath the chord of
each phrase, recovering the Pythagorean foundation in audible form
(Park 2025, 12).

Audio rendering is performed by additive sine synthesis through
`pretty_midi.synthesize`, with a gentle box-filter low-pass that
softens the harshness of pure sines toward a felt-hammer character
and per-buffer fade-in / fade-out shaping. No learned model, no GPU,
no stochastic component: identical input and configuration produce
a bit-identical audio buffer.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from scipy import ndimage
import pretty_midi

from .arithmetic import (
    GREEK_MODES, mode_template, quantise_to_mode, equal_temperament_freq,
)


MIDI_LOW = 21       # A0
MIDI_HIGH = 108     # C8


@dataclasses.dataclass
class NocturneConfig:
    """Configuration for the chart-to-nocturne transcoder."""
    duration_seconds: float = 90.0
    mode: str = "chopin_csm"
    tonic_pc: int = 1                # C# minor by default
    pitch_floor_midi: int = 40
    pitch_ceiling_midi: int = 92
    pitch_focus_midi: int = 72       # legacy global focus (kept for compatibility)
    pitch_focus_weight: float = 0.45
    star_threshold: float = 0.55
    star_min_size: int = 4
    velocity_floor: int = 30
    velocity_ceiling: int = 110
    velocity_gamma: float = 1.4
    note_duration_min: float = 1.4
    note_duration_max: float = 6.5
    instrument_program: int = 0      # Acoustic Grand Piano
    bass_drone: bool = True
    tetractys_drone: bool = True     # use 1:2:3:4 ratio for drone harmonics
    sustain_pedal: bool = True
    invert_pitch_axis: bool = True
    # --- Phrase-and-voice structure (added) ---
    n_phrases: int = 4
    phrase_breath_seconds: float = 0.45
    voice_split: float = 0.5          # yfrac threshold separating LH (low) from RH (high)
    rh_focus_midi: int = 76           # cantabile centre (E5)
    lh_focus_midi: int = 50           # bass register centre (D3)
    voice_leading: bool = True        # smooth RH leaps by octave displacement
    rh_max_leap_semitones: int = 9    # hard ceiling for RH leap before octave-displace
    lh_arpeggio: bool = True          # replace LH pitches with rolling chord arpeggios
    chord_extension: str = "triad"    # 'triad' or 'seventh'


# ---------------------------------------------------------------------
# Chart -> MIDI
# ---------------------------------------------------------------------


def _load_grayscale(path: str | Path) -> np.ndarray:
    img = Image.open(path).convert("L")
    return np.asarray(img, dtype=np.float32) / 255.0


def _detect_stars(arr: np.ndarray, threshold: float, min_size: int):
    """Return list of star dicts. The connected-component label is kept
    on each star so that stars belonging to a constellation line in the
    source plate (which would form a single connected bright region)
    can later be grouped into a slur."""
    binary = arr > threshold
    labelled, n = ndimage.label(binary)
    if n == 0:
        return []
    centroids = ndimage.center_of_mass(arr, labelled, range(1, n + 1))
    sizes = ndimage.sum(binary, labelled, range(1, n + 1))
    bboxes = ndimage.find_objects(labelled)
    out = []
    for i in range(n):
        size = int(sizes[i])
        if size < min_size:
            continue
        cy, cx = centroids[i]
        max_brightness = float(arr[bboxes[i]].max())
        out.append({
            "y": float(cy),
            "x": float(cx),
            "size": size,
            "magnitude": max_brightness,
            "label": i + 1,
        })
    return out


def _stars_to_raw_events(stars, h, w, cfg: NocturneConfig):
    """Build raw events from stars. Pitch is left unresolved (raw_pitch
    is the position-derived ideal); voice assignment is final."""
    if not stars:
        return []
    pitch_lo = max(MIDI_LOW, cfg.pitch_floor_midi)
    pitch_hi = min(MIDI_HIGH, cfg.pitch_ceiling_midi)
    events = []
    for s in stars:
        # x -> time
        t = (s["x"] / w) * cfg.duration_seconds
        # y -> pitch (axis-inverted by default)
        if cfg.invert_pitch_axis:
            yfrac = 1.0 - (s["y"] / h)
        else:
            yfrac = s["y"] / h
        raw_pitch = pitch_lo + yfrac * (pitch_hi - pitch_lo)
        # Voice assignment: high stars to RH cantabile, low stars to LH arpeggio
        voice = "rh" if yfrac >= cfg.voice_split else "lh"
        # Magnitude -> velocity
        mag = s["magnitude"]
        velocity = int(cfg.velocity_floor
                       + (cfg.velocity_ceiling - cfg.velocity_floor)
                       * np.power(mag, cfg.velocity_gamma))
        velocity = max(1, min(127, velocity))
        # Size -> duration
        size_norm = min(1.0, s["size"] / 80.0)
        duration = (cfg.note_duration_min
                    + (cfg.note_duration_max - cfg.note_duration_min)
                    * np.power(size_norm, 0.5)
                    * (0.4 + 0.6 * mag))
        events.append({
            "time": t,
            "voice": voice,
            "yfrac": yfrac,
            "raw_pitch": raw_pitch,
            "velocity": velocity,
            "duration": duration,
            "magnitude": mag,
            "size": s["size"],
            "label": s["label"],
        })
    events.sort(key=lambda e: e["time"])
    return events


def _mode_pc_set(cfg: NocturneConfig) -> list[int]:
    """Return the sorted list of pitch classes in the active key."""
    pcs = mode_template(cfg.mode)
    return sorted({(p + cfg.tonic_pc) % 12 for p in pcs})


def _nearest_pitch_with_pc(pc: int, lo: int, hi: int, target: int) -> int:
    """Return the MIDI pitch in [lo, hi] whose pitch class equals `pc`
    and which is closest to `target`. If no such pitch exists in the
    range, the search widens to the full MIDI range. Used to place a
    pitch class in a desired register without the convergence problems
    of paired while-loops over narrow intervals.
    """
    lo = max(0, int(lo))
    hi = max(lo, int(hi))
    pc = int(pc) % 12
    candidates = [p for p in range(lo, hi + 1) if p % 12 == pc]
    if not candidates:
        candidates = [p for p in range(MIDI_LOW, MIDI_HIGH + 1)
                      if p % 12 == pc]
    return min(candidates, key=lambda p: abs(p - target))


def _build_mode_chord(root_pc: int, mode_pc_set: list[int],
                      extension: str = "triad") -> list[int]:
    """Build a chord by stacking thirds within the active mode.

    `triad` -> root + 2 stacked thirds; `seventh` -> root + 3 stacked thirds.
    Result is a list of pitch classes (mod 12) including the root.
    """
    if not mode_pc_set:
        return [root_pc]
    if root_pc not in mode_pc_set:
        # snap root to nearest mode pc
        root_pc = min(mode_pc_set,
                      key=lambda x: min((x - root_pc) % 12,
                                        (root_pc - x) % 12))
    n = len(mode_pc_set)
    idx = mode_pc_set.index(root_pc)
    extras = 2 if extension == "triad" else 3
    return [mode_pc_set[(idx + 2 * k) % n] for k in range(1 + extras)]


def _derive_chord_progression(events, cfg: NocturneConfig) -> list[dict]:
    """Divide the duration into n_phrases segments; for each, choose a
    chord anchor (the brightest event in the segment) and build a chord
    in the active mode. Returns a list of phrase chord dicts.
    """
    if not events:
        return []
    pcs = mode_template(cfg.mode)
    mode_pc_set = _mode_pc_set(cfg)
    phrase_len = cfg.duration_seconds / cfg.n_phrases
    breath = cfg.phrase_breath_seconds
    chords = []
    for i in range(cfg.n_phrases):
        t0 = i * phrase_len
        t1 = (i + 1) * phrase_len
        # Brightest event whose time falls in this phrase
        in_phrase = [e for e in events if t0 <= e["time"] < t1]
        if in_phrase:
            anchor = max(in_phrase,
                         key=lambda e: e["velocity"] * (1 + e["magnitude"]))
        else:
            anchor = max(events,
                         key=lambda e: e["velocity"] * (1 + e["magnitude"]))
        anchor_pitch = int(round(anchor["raw_pitch"]))
        root_midi = quantise_to_mode(anchor_pitch, pcs, tonic_pc=cfg.tonic_pc)
        root_pc = root_midi % 12
        chord_pcs = _build_mode_chord(root_pc, mode_pc_set,
                                      extension=cfg.chord_extension)
        chords.append({
            "phrase_idx": i,
            "start": t0,
            "end": t1,
            "audible_start": t0 + (breath if i > 0 else 0.0),
            "root_pc": root_pc,
            "chord_pcs": chord_pcs,
            "anchor_pitch": int(root_midi),
        })
    return chords


def _process_rh_events(rh_events, cfg: NocturneConfig):
    """Quantise RH events to the mode with a voice-leading constraint:
    if a leap from the previous RH note exceeds the configured ceiling,
    octave-displace until the leap is bounded.

    A pull toward `rh_focus_midi` (controlled by `pitch_focus_weight`)
    keeps the cantabile voice in a sung register without collapsing
    the registral contrast against the LH.
    """
    pcs = mode_template(cfg.mode)
    pitch_lo = max(MIDI_LOW, cfg.pitch_floor_midi)
    pitch_hi = min(MIDI_HIGH, cfg.pitch_ceiling_midi)
    out = []
    prev_pitch = None
    for ev in rh_events:
        raw = ev["raw_pitch"]
        if 0 < cfg.pitch_focus_weight <= 1:
            raw = ((1 - cfg.pitch_focus_weight) * raw
                   + cfg.pitch_focus_weight * cfg.rh_focus_midi)
        target = int(round(raw))
        target = max(pitch_lo, min(pitch_hi, target))
        snapped = quantise_to_mode(target, pcs, tonic_pc=cfg.tonic_pc)
        if cfg.voice_leading and prev_pitch is not None:
            # octave-displace until leap is within ceiling
            guard = 0
            while abs(snapped - prev_pitch) > cfg.rh_max_leap_semitones \
                    and guard < 12:
                snapped += -12 if snapped > prev_pitch else 12
                guard += 1
            snapped = max(pitch_lo, min(pitch_hi, snapped))
        out.append({**ev, "pitch": int(snapped)})
        prev_pitch = snapped
    return out


def _process_lh_events(lh_events, chords, cfg: NocturneConfig):
    """Replace each LH event's pitch with the next note in a rolling
    arpeggio drawn from the chord active at that event's time.

    The arpeggio cycles through the chord's pitch classes in a pattern
    inspired by the salon-nocturne LH (a low-octave root anchor
    followed by close-position upper-chord notes). When `lh_arpeggio`
    is False, LH events fall back to direct mode quantisation in the
    bass register.
    """
    pitch_lo = max(MIDI_LOW, cfg.pitch_floor_midi)
    pitch_hi = min(MIDI_HIGH, cfg.pitch_ceiling_midi)
    pcs = mode_template(cfg.mode)
    out = []
    if not cfg.lh_arpeggio or not chords:
        for ev in lh_events:
            raw = ev["raw_pitch"]
            if 0 < cfg.pitch_focus_weight <= 1:
                raw = ((1 - cfg.pitch_focus_weight) * raw
                       + cfg.pitch_focus_weight * cfg.lh_focus_midi)
            target = int(round(raw))
            snapped = quantise_to_mode(target, pcs, tonic_pc=cfg.tonic_pc)
            snapped = max(pitch_lo, min(60, snapped))
            out.append({**ev, "pitch": int(snapped)})
        return out

    lh_lo = max(MIDI_LOW + 4, pitch_lo)
    lh_hi = max(lh_lo + 18, cfg.lh_focus_midi + 14)
    center = cfg.lh_focus_midi

    for chord in chords:
        chord_evs = [e for e in lh_events
                     if chord["start"] <= e["time"] < chord["end"]]
        if not chord_evs:
            continue
        chord_evs.sort(key=lambda e: e["time"])

        # Low-octave root anchor: chord root placed roughly an octave
        # below the LH centre. Use the candidate-list helper so a
        # narrow [lh_lo, center-3] range can never trap us in an
        # infinite loop when root_pc has no member there.
        root_pc = chord["root_pc"]
        low_anchor = _nearest_pitch_with_pc(
            root_pc, lh_lo, max(lh_lo + 1, center - 3), center - 12,
        )

        # Close-position upper-chord pitches: each chord pc placed in
        # the octave nearest the LH centre, then sorted ascending.
        mid_set: set[int] = set()
        for pc in chord["chord_pcs"]:
            mid_set.add(_nearest_pitch_with_pc(
                pc, max(lh_lo, center - 6), min(lh_hi, center + 6), center,
            ))
        mid_pitches = sorted(mid_set)

        # Salon-nocturne LH pattern: low_anchor + alternating mid notes.
        if len(mid_pitches) >= 3:
            mlow, mmid, mhigh = mid_pitches[0], mid_pitches[1], mid_pitches[2]
            pattern_pitches = [low_anchor, mhigh, mmid, mhigh,
                               mlow,       mhigh, mmid, mhigh]
        elif len(mid_pitches) == 2:
            mlow, mhigh = mid_pitches
            pattern_pitches = [low_anchor, mhigh, mlow, mhigh,
                               low_anchor, mhigh, mlow, mhigh]
        else:
            only = mid_pitches[0] if mid_pitches else low_anchor + 12
            pattern_pitches = [low_anchor, only]

        pattern_pitches = [max(lh_lo, min(lh_hi, p)) for p in pattern_pitches]

        for i, ev in enumerate(chord_evs):
            pitch = pattern_pitches[i % len(pattern_pitches)]
            out.append({**ev, "pitch": int(pitch)})

    return out


def _apply_phrase_envelope(events, cfg: NocturneConfig):
    """Push events out of the breath gap at each phrase start, apply a
    hairpin velocity envelope, and clamp duration so notes do not
    cross the breath gap of the following phrase."""
    if cfg.n_phrases <= 0:
        return events
    phrase_len = cfg.duration_seconds / cfg.n_phrases
    breath = cfg.phrase_breath_seconds
    out = []
    for ev in events:
        idx = int(ev["time"] / phrase_len)
        idx = max(0, min(cfg.n_phrases - 1, idx))
        phrase_start = idx * phrase_len
        new_time = ev["time"]
        if idx > 0 and new_time < phrase_start + breath:
            new_time = phrase_start + breath + 1e-3
        # Hairpin: rises and falls within the phrase
        t_in = (new_time - phrase_start) / phrase_len
        t_in = max(0.0, min(1.0, t_in))
        hairpin = 0.55 + 0.45 * float(np.sin(np.pi * t_in)) ** 0.7
        new_vel = int(round(ev["velocity"] * hairpin))
        new_vel = max(1, min(127, new_vel))
        # Clamp duration so it doesn't cross the next phrase's breath gap
        next_breath_start = (idx + 1) * phrase_len - breath * 0.5
        max_dur = max(0.1, next_breath_start - new_time)
        new_dur = min(ev["duration"], max_dur)
        out.append({**ev, "time": new_time,
                    "velocity": new_vel, "duration": new_dur})
    out.sort(key=lambda e: e["time"])
    return out


def _add_phrase_tetractys_drone(chords, instrument: pretty_midi.Instrument,
                                 cfg: NocturneConfig):
    """For each phrase chord, sustain a 4-note Tetractys stack at the
    chord root in the deep bass register. The stack offsets (-12, -5,
    0, +7) sound the octave-below dyad, the perfect-fourth-below triad,
    the root monad, and the perfect-fifth-above tetrad — recovering the
    Pythagorean monochord ratios 1:2:3:4 in pitch (Park 2025, 12)."""
    if not chords or not cfg.bass_drone:
        return
    if cfg.tetractys_drone:
        offsets = (-12, -5, 0, 7)
    else:
        offsets = (-12,)
    for chord in chords:
        root_midi = chord["anchor_pitch"]
        # Drop into bass register
        while root_midi > 48:
            root_midi -= 12
        for offset in offsets:
            p = root_midi + offset
            if p < MIDI_LOW + 4 or p > 60:
                continue
            instrument.notes.append(pretty_midi.Note(
                velocity=38,
                pitch=int(p),
                start=float(chord["start"]),
                end=float(chord["end"]),
            ))


def _add_chord_pedaling(chords, instrument: pretty_midi.Instrument,
                        cfg: NocturneConfig):
    """Pedal CC64 ON at each chord start; OFF just before each chord
    ends so the next chord begins clean. Replaces the legacy 8-second
    timer, whose pedal changes were uncorrelated with harmony."""
    if not chords or not cfg.sustain_pedal:
        return
    cc = 64
    for chord in chords:
        instrument.control_changes.append(pretty_midi.ControlChange(
            number=cc, value=110, time=float(chord["start"])
        ))
        off_t = max(chord["start"] + 0.1, chord["end"] - 0.12)
        instrument.control_changes.append(pretty_midi.ControlChange(
            number=cc, value=0, time=float(off_t)
        ))


def chart_to_nocturne(chart_path: str | Path,
                      config: Optional[NocturneConfig] = None) -> pretty_midi.PrettyMIDI:
    """Transcode a star-chart image into a nocturne MIDI score.

    Returns a `pretty_midi.PrettyMIDI` object. Use `.write(path)` to
    save as `.mid`. Use `midi_to_audio(midi)` to render to an audio
    array via the deterministic modal synthesiser.
    """
    cfg = config or NocturneConfig()
    arr = _load_grayscale(chart_path)
    h, w = arr.shape
    stars = _detect_stars(arr, cfg.star_threshold, cfg.star_min_size)

    pm = pretty_midi.PrettyMIDI(initial_tempo=54.0)
    inst = pretty_midi.Instrument(program=cfg.instrument_program,
                                   name="Quadrivium Nocturne")
    pm.instruments.append(inst)

    if not stars:
        return pm

    raw_events = _stars_to_raw_events(stars, h, w, cfg)
    chords = _derive_chord_progression(raw_events, cfg)

    rh = _process_rh_events([e for e in raw_events if e["voice"] == "rh"], cfg)
    lh = _process_lh_events([e for e in raw_events if e["voice"] == "lh"],
                            chords, cfg)
    merged = sorted(rh + lh, key=lambda e: e["time"])
    merged = _apply_phrase_envelope(merged, cfg)

    for ev in merged:
        end = min(cfg.duration_seconds, ev["time"] + ev["duration"])
        if end <= ev["time"]:
            continue
        inst.notes.append(pretty_midi.Note(
            velocity=int(np.clip(ev["velocity"], 1, 127)),
            pitch=int(ev["pitch"]),
            start=float(ev["time"]),
            end=float(end),
        ))

    _add_phrase_tetractys_drone(chords, inst, cfg)
    _add_chord_pedaling(chords, inst, cfg)

    return pm


# ---------------------------------------------------------------------
# MIDI -> audio (additive sine synthesis)
# ---------------------------------------------------------------------


def midi_to_audio(midi: pretty_midi.PrettyMIDI,
                   sample_rate: int = 22050) -> np.ndarray:
    """Render MIDI to a mono audio array via pretty_midi's built-in
    sine-bank synthesiser, with a gentle low-pass smoothing to soften
    the sine harshness toward a felt-hammer tone.
    """
    base = midi.synthesize(fs=sample_rate, wave=np.sin)
    if base.size == 0:
        return base
    n = base.shape[0]
    fade_in = min(int(sample_rate * 0.4), n // 2)
    fade_out = min(int(sample_rate * 0.6), n // 2)
    env = np.ones(n, dtype=np.float32)
    if fade_in > 0:
        env[:fade_in] *= np.linspace(0, 1, fade_in) ** 1.5
    if fade_out > 0:
        env[-fade_out:] *= np.linspace(1, 0, fade_out) ** 1.5
    base = base * env
    kernel = max(3, sample_rate // 4000)
    cumsum = np.cumsum(np.insert(base, 0, 0))
    smoothed = (cumsum[kernel:] - cumsum[:-kernel]) / kernel
    smoothed = np.concatenate(
        [smoothed, np.zeros(n - smoothed.size, dtype=np.float32)]
    )
    out = (0.65 * base + 0.35 * smoothed).astype(np.float32)
    peak = float(np.max(np.abs(out)) + 1e-9)
    return out / peak * 0.85
