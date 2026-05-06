"""
run_quadrivium.py
=================

End-to-end runner for the Nocturne Atlas in the Pythagorean Quadrivium
framing. Generates two paired plates in the parchment register and
reports the Music-of-the-Spheres ring statistics for the audio-derived
charts.

Outputs are written to `outputs/<label>_quadrivium/`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from quadrivium import (
    chart_to_nocturne, NocturneConfig,
    nocturne_to_chart, ChartConfig,
    midi_to_audio,
    parchment_score, parchment_waveform,
    parchment_chart_overlay, parchment_plate,
    detect_concentric_rings,
)
from quadrivium.astronomy import _detect_spectral_peaks, _peaks_to_polar
import librosa

INPUTS = ROOT / "inputs"
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)


def run_pair(chart_path: Path, audio_path: Path, label: str,
             plate_roman: str,
             palette: str = "midnight",
             mode: str = "chopin_csm",
             tonic_pc: int = 1,
             duration: float = 90.0):
    out_dir = OUTPUTS / f"{label}_quadrivium"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== {label} (quadrivium) ===")
    metadata = {"label": label, "plate": plate_roman,
                 "palette": palette, "mode": mode, "tonic_pc": tonic_pc}

    # 1) chart -> MIDI
    pm = chart_to_nocturne(chart_path, config=NocturneConfig(
        duration_seconds=duration, mode=mode, tonic_pc=tonic_pc,
        pitch_floor_midi=40, pitch_ceiling_midi=92,
        pitch_focus_midi=72, pitch_focus_weight=0.45,
        star_threshold=0.55, star_min_size=4,
        velocity_floor=30, velocity_ceiling=110,
        bass_drone=True, tetractys_drone=True,
    ))
    midi_path = out_dir / f"{label}_chart_to_nocturne.mid"
    pm.write(str(midi_path))
    n_notes = sum(len(i.notes) for i in pm.instruments)
    metadata["n_midi_events"] = n_notes
    print(f"  midi: {midi_path.name} ({n_notes} events)")

    # 2) MIDI -> audio
    rendered = midi_to_audio(pm, sample_rate=22050)
    sf.write(str(out_dir / f"{label}_chart_to_nocturne.wav"),
              rendered.astype(np.float32), 22050, subtype="PCM_16")

    # 3) Score in parchment register
    score = out_dir / f"{label}_score_parchment.png"
    parchment_score(pm, score,
                     title="CHART  TRANSCODED  TO  NOCTURNE",
                     subtitle=f"Tabula {plate_roman}  ·  {label.replace('_', ' ')}",
                     width=2400, height=900, seed=37)

    # 4) Source chart -> parchment
    src_chart = Image.open(chart_path).convert("RGBA")
    src_p = parchment_chart_overlay(src_chart, seed=11)
    src_out = out_dir / f"{label}_source_chart_parchment.png"
    src_p.convert("RGB").save(src_out)

    # 5) Audio -> chart
    chart_cfg = ChartConfig(
        width=2200, height=2200, sr=22050, n_fft=4096, hop_length=2048,
        palette=palette, n_peaks_per_frame=4, min_peak_db=-50.0,
        cartouche_text_top="NOCTVRNVS  IN  STELLAS",
        cartouche_text_subtitle=f"Tabula {plate_roman} :: {label.replace('_', ' ')}",
        cartouche_text_latin="DETERMINISTICE  ·  TRANSCODATA",
    )
    chart_img = nocturne_to_chart(audio_path, config=chart_cfg)
    chart_p = parchment_chart_overlay(chart_img, seed=53)
    chart_out = out_dir / f"{label}_nocturne_to_chart.png"
    chart_p.convert("RGB").save(chart_out)

    # 5b) Music-of-the-Spheres diagnostic: ring detection
    y_audio, sr = librosa.load(str(audio_path), sr=22050,
                                 duration=chart_cfg.duration, mono=True)
    peaks, n_frames = _detect_spectral_peaks(y_audio, sr, chart_cfg)
    points = _peaks_to_polar(peaks, n_frames)
    rings = detect_concentric_rings(points, n_radial_bins=64, min_density=0.08)
    metadata["concentric_rings"] = rings
    print(f"  rings detected: {len(rings)}")
    for ring in rings:
        print(f"    ring at radius={ring['radius']:.3f} "
              f"density={ring['density']:.3f} "
              f"mag_mean={ring['mag_mean']:.3f}")

    # 6) Waveform
    wave_path = out_dir / f"{label}_waveform_parchment.png"
    parchment_waveform(audio_path, wave_path, width=2400, height=900, seed=29)

    # 7) Atlas plate
    plate_path = out_dir / f"{label}_plate_parchment.png"
    parchment_plate(
        panels=[src_out, score, wave_path, chart_out],
        labels=[
            "Source chart  (numbers in time and space)",
            "Chart  →  nocturne  (numbers in time)",
            "Source nocturne  (audio waveform)",
            "Nocturne  →  chart  (Music of the Spheres)",
        ],
        output_path=plate_path,
        title="NOCTURNE  ATLAS",
        plate_number=plate_roman,
        panel_width=1100, panel_height=1100,
        gap=56, margin=80, seed=23,
    )
    print(f"  plate: {plate_path.name}")

    # Save metadata
    with (out_dir / f"{label}_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    return plate_path


def main():
    audio_path = INPUTS / "nocturne_csm_standin.wav"
    if not audio_path.exists():
        sys.exit("Missing demo audio.")
    pairs = [
        ("plate_i_borealis", "tabula_stellarum_standin.png", "I",
         "midnight", "chopin_csm", 1),
        ("plate_ii_australis", "tabula_stellarum_alt.png", "II",
         "indigo", "octatonic", 0),
    ]
    for label, name, roman, palette, mode, tonic_pc in pairs:
        chart_path = INPUTS / name
        if not chart_path.exists():
            continue
        run_pair(chart_path, audio_path, label, roman,
                 palette=palette, mode=mode, tonic_pc=tonic_pc)
    print(f"\nAll outputs in {OUTPUTS}")


if __name__ == "__main__":
    main()
