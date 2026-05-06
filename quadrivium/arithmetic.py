"""
quadrivium.arithmetic
=====================

The Pythagorean foundation: numerical ratios, the Tetractys, and the
just-intonation tuning system.

After Pythagoras's discovery that musical consonance is governed by
ratios of small whole numbers, the entire Quadrivium tradition treated
these ratios as the formal substance of the audible scale. This module
provides them as code constants and as scale-construction utilities,
together with the alternative twelve-tone equal-temperament approximation
that became canonical with J. S. Bach and that the salon nocturne of the
nineteenth century presupposes.

Reference: Park (2025), Sound of Music, ch. 1, esp. pp. 5–12.
"""

from __future__ import annotations

import math
from typing import Iterable


# ----------------------------------------------------------------------
# Pythagorean ratios (Park 2025, 6–11)
# ----------------------------------------------------------------------

# Frequency ratios between two pitches separated by named intervals.
# These are the original Pythagorean discoveries from the hammer and
# monochord experiments. They are exact rational numbers, not equal-
# tempered approximations.
PYTHAGOREAN_RATIOS = {
    "unison":          1,                     # 1:1
    "octave":          2,                     # 2:1   (Park 2025, 6)
    "perfect_fifth":   3 / 2,                 # 3:2   (Park 2025, 6)
    "perfect_fourth":  4 / 3,                 # 4:3   (Park 2025, 6)
    "major_second":    9 / 8,                 # 9:8   (Park 2025, 6)
    "major_third":     81 / 64,               # 81:64 (derived: 4 stacked fifths down 2 octaves)
    "minor_third":     32 / 27,               # 32:27
    "major_sixth":     27 / 16,               # 27:16
    "minor_sixth":     128 / 81,              # 128:81
    "major_seventh":   243 / 128,             # 243:128
    "minor_seventh":   16 / 9,                # 16:9
    "tritone":         729 / 512,             # 729:512  (Pythagorean diminished fifth)
}


# ----------------------------------------------------------------------
# The Tetractys (Park 2025, 12)
# ----------------------------------------------------------------------

# 1 + 2 + 3 + 4 = 10
# The four numbers symbolised, respectively, Monad (unity), Dyad (power),
# Triad (harmony), Tetrad (stability); their sum was Dekad (cosmos).
TETRACTYS = (1, 2, 3, 4)
TETRACTYS_SUM = sum(TETRACTYS)  # = 10
TETRACTYS_NAMES = {
    1: "Monad",
    2: "Dyad",
    3: "Triad",
    4: "Tetrad",
    10: "Dekad",
}


def tetractys_intervals():
    """Return the three Tetractys-pair intervals as Pythagorean ratios.

    The pairs (1,2), (2,3), (3,4) generate the octave, perfect fifth,
    and perfect fourth respectively — the three principal consonances
    Pythagoras identified through the monochord experiment.
    """
    return [
        ("1:2", PYTHAGOREAN_RATIOS["octave"]),
        ("2:3", 1 / PYTHAGOREAN_RATIOS["perfect_fifth"]),  # 2/3 = string-length ratio
        ("3:4", 1 / PYTHAGOREAN_RATIOS["perfect_fourth"]),
    ]


# ----------------------------------------------------------------------
# The Greek modes / Pythagorean diatonic
# ----------------------------------------------------------------------

# A diatonic mode is a sequence of seven pitches reached from a
# tonic by a chain of just-intonation perfect fifths and octave
# reductions. The seven Greek modes differ by which scale degree
# is treated as the tonic.

# Pythagorean diatonic intervals from the tonic, expressed as ratios.
# The major scale (Lydian, in older Greek nomenclature):
PYTHAGOREAN_MAJOR = [
    1,            # tonic
    9 / 8,        # major 2nd
    81 / 64,      # major 3rd (Pythagorean — slightly sharper than 5/4)
    4 / 3,        # perfect 4th
    3 / 2,        # perfect 5th
    27 / 16,      # major 6th
    243 / 128,    # major 7th
    2,            # octave
]

# Pythagorean natural minor (Aeolian):
PYTHAGOREAN_MINOR = [
    1,
    9 / 8,
    32 / 27,      # minor 3rd
    4 / 3,
    3 / 2,
    128 / 81,     # minor 6th
    16 / 9,       # minor 7th
    2,
]


# Greek modes as scale-degree templates relative to the tonic, expressed
# as offsets in the chromatic 12-step octave. These match the historical
# Greek-medieval mode names used in modal counterpoint.
GREEK_MODES = {
    "ionian":        (0, 2, 4, 5, 7, 9, 11),    # major
    "dorian":        (0, 2, 3, 5, 7, 9, 10),
    "phrygian":      (0, 1, 3, 5, 7, 8, 10),
    "lydian":        (0, 2, 4, 6, 7, 9, 11),
    "mixolydian":    (0, 2, 4, 5, 7, 9, 10),
    "aeolian":       (0, 2, 3, 5, 7, 8, 10),    # natural minor
    "locrian":       (0, 1, 3, 5, 6, 8, 10),
    # Romantic-period & post-Pythagorean modes the nocturne tradition mobilises:
    "harmonic_minor":     (0, 2, 3, 5, 7, 8, 11),
    "melodic_minor":      (0, 2, 3, 5, 7, 9, 11),
    "phrygian_dominant":  (0, 1, 4, 5, 7, 8, 10),
    "octatonic":          (0, 1, 3, 4, 6, 7, 9, 10),
    "whole_tone":         (0, 2, 4, 6, 8, 10),
    "chopin_csm":         (1, 3, 4, 6, 8, 9, 11),  # C-sharp minor (Chopin Op. 27 No. 1)
    "chromatic":          tuple(range(12)),
}


def mode_template(name: str) -> tuple:
    """Return the chromatic-step pitch-class template for a named mode."""
    if name not in GREEK_MODES:
        raise ValueError(f"Unknown mode '{name}'. "
                         f"Available: {sorted(GREEK_MODES.keys())}")
    return GREEK_MODES[name]


# ----------------------------------------------------------------------
# Tuning systems
# ----------------------------------------------------------------------


def just_intonation_freq(pitch_class_index: int,
                          tonic_freq: float = 261.6256) -> float:
    """Return the frequency of a scale-degree-by-index in just intonation.

    pitch_class_index 0..7 indexes the eight degrees of the Pythagorean
    major scale (tonic, 2nd, 3rd, 4th, 5th, 6th, 7th, octave).
    tonic_freq defaults to middle C (261.6256 Hz, A4=440 reference).
    """
    if not (0 <= pitch_class_index < len(PYTHAGOREAN_MAJOR)):
        raise IndexError(pitch_class_index)
    return tonic_freq * PYTHAGOREAN_MAJOR[pitch_class_index]


def equal_temperament_freq(midi_note: int) -> float:
    """Return the frequency of a MIDI note in twelve-tone equal temperament.

    Reference: A4 = MIDI 69 = 440 Hz. f(n) = 440 * 2^((n-69)/12).
    """
    return 440.0 * 2 ** ((midi_note - 69) / 12)


def pythagorean_pitch_classes(tonic_pc: int = 0,
                               n_classes: int = 12) -> list[int]:
    """Generate `n_classes` pitch classes by stacking perfect fifths from
    `tonic_pc` and reducing to one octave.

    This is the original Pythagorean construction of the chromatic
    scale: starting from the tonic, one ascends by perfect fifths
    (3:2 ratio) twelve times and reduces each result to within one
    octave. After 12 fifths one returns to the starting pitch class
    (with a small Pythagorean comma; we ignore it for chromatic
    quantisation).
    """
    pcs = []
    pc = tonic_pc
    for _ in range(n_classes):
        pcs.append(pc % 12)
        pc = (pc + 7) % 12  # perfect fifth = +7 semitones in 12TET
    return pcs


# ----------------------------------------------------------------------
# Quantisation
# ----------------------------------------------------------------------


def quantise_to_mode(midi_pitch: int,
                      mode_pcs: Iterable[int],
                      tonic_pc: int = 0) -> int:
    """Snap a MIDI pitch to the nearest pitch class in a mode template.

    Parameters
    ----------
    midi_pitch : int
        Input MIDI pitch.
    mode_pcs : iterable of int (0..11)
        Pitch-class set of the mode, expressed relative to the tonic.
    tonic_pc : int
        Pitch class (0=C, 1=C#, ..., 11=B) of the mode's tonic.

    Returns the MIDI pitch closest to the input that belongs to the
    mode. The choice minimises absolute chromatic distance.
    """
    mode_set = {(p + tonic_pc) % 12 for p in mode_pcs}
    if not mode_set:
        return midi_pitch
    pc = midi_pitch % 12
    if pc in mode_set:
        return midi_pitch
    # Search for nearest member
    best_pc = min(mode_set, key=lambda s: min((pc - s) % 12, (s - pc) % 12))
    delta = (best_pc - pc) % 12
    if delta > 6:
        delta -= 12
    return midi_pitch + delta
