"""
run_paper_real.py
=================

End-to-end runner for Plate III, which transcodes a real public-domain
scan of Bode's *Uranographia* (1801) — the engraver's plate that
Plates I and II's procedural stand-ins were designed to anticipate.

The Bode JPG is preprocessed by `quadrivium.preprocess.engraving_to_chart`
into a stand-in-compatible PNG (bright star markers on a dark celestial
ground), then fed through the same paper-register pipeline used for
the procedural plates. The audio side reuses the procedural nocturne
input so that the audio-to-chart panel remains byte-identical to the
corresponding panels of Plates I and II — preserving the deterministic-
equivalence demonstration.

Output goes to `outputs/plate_iii_aries_paper/`.
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
    paper_score, paper_waveform,
    paper_chart_overlay, paper_plate,
    detect_concentric_rings,
)
from quadrivium.astronomy import _detect_spectral_peaks, _peaks_to_polar
from quadrivium.preprocess import engraving_to_chart, EngravingConfig
import librosa

INPUTS = ROOT / "inputs"
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)


def main():
    audio_path = INPUTS / "nocturne_csm_standin.wav"
    raw_chart = INPUTS / "bode_1801_aries_planisphere.jpg"
    if not audio_path.exists():
        sys.exit("Missing audio.")
    if not raw_chart.exists():
        sys.exit(f"Missing Bode plate at {raw_chart}.")

    label = "plate_iii_aries"
    plate_roman = "III"
    palette = "iron"
    mode = "harmonic_minor"
    tonic_pc = 1
    duration = 90.0

    out_dir = OUTPUTS / f"{label}_paper"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== {label} (paper register, real Bode plate) ===")
    metadata = {"label": label, "plate": plate_roman,
                "palette": palette, "mode": mode, "tonic_pc": tonic_pc,
                "source_chart": "Bode, Uranographia (1801), Tabula 1: Coelum Stellatum Hemisphaerii Arietis"}

    # 0) Preprocess the Bode plate into a stand-in-compatible PNG
    processed_chart = INPUTS / "bode_1801_planisphere_processed.png"
    info = engraving_to_chart(raw_chart, processed_chart, EngravingConfig())
    metadata["preprocessing"] = info
    print(f"  preprocessed: {info['n_stars_detected']} stars detected at "
          f"src disc r={info['source_disc_radius_px']:.0f}px")

    # 1) Chart -> MIDI
    pm = chart_to_nocturne(processed_chart, config=NocturneConfig(
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

    # 3) Score in paper register
    score = out_dir / f"{label}_score_paper.png"
    paper_score(pm, score,
                title="CHART  TRANSCODED  TO  NOCTURNE",
                subtitle=f"Tabula {plate_roman}  ·  Bode 1801  ·  Hemisphaerium Arietis",
                width=2400, height=900, seed=41)

    # 4) Source chart (already a midnight-style PNG from preprocessor) -> paper
    src_chart_img = Image.open(processed_chart).convert("RGBA")
    src_p = paper_chart_overlay(src_chart_img, seed=13, with_atlas=True)
    src_out = out_dir / f"{label}_source_chart_paper.png"
    src_p.convert("RGB").save(src_out)

    # 5) Audio -> chart (byte-identical to Plates I and II since audio is the same)
    chart_cfg = ChartConfig(
        width=2200, height=2200, sr=22050, n_fft=4096, hop_length=2048,
        palette=palette, n_peaks_per_frame=4, min_peak_db=-50.0,
        cartouche_text_top="NOCTVRNVS  IN  STELLAS",
        cartouche_text_subtitle=f"Tabula {plate_roman} :: {label.replace('_', ' ')}",
        cartouche_text_latin="DETERMINISTICE  ·  TRANSCODATA",
    )
    chart_img = nocturne_to_chart(audio_path, config=chart_cfg)
    chart_p = paper_chart_overlay(chart_img, seed=59)
    chart_out = out_dir / f"{label}_nocturne_to_chart.png"
    chart_p.convert("RGB").save(chart_out)

    # 5b) Music-of-the-Spheres diagnostic
    y_audio, sr = librosa.load(str(audio_path), sr=22050,
                                duration=chart_cfg.duration, mono=True)
    peaks, n_frames = _detect_spectral_peaks(y_audio, sr, chart_cfg)
    points = _peaks_to_polar(peaks, n_frames)
    rings = detect_concentric_rings(points, n_radial_bins=64, min_density=0.08)
    metadata["concentric_rings"] = rings
    print(f"  rings detected: {len(rings)}")

    # 6) Waveform
    wave_path = out_dir / f"{label}_waveform_paper.png"
    paper_waveform(audio_path, wave_path, width=2400, height=900, seed=37)

    # 7) Atlas plate
    plate_path = out_dir / f"{label}_plate_paper.png"
    paper_plate(
        panels=[src_out, score, wave_path, chart_out],
        labels=[
            "Source chart  (Bode 1801, Hemisphaerium Arietis, preprocessed)",
            "Chart  →  nocturne  (numbers in time)",
            "Source nocturne  (audio waveform)",
            "Nocturne  →  chart  (Music of the Spheres)",
        ],
        output_path=plate_path,
        title="NOCTURNE  ATLAS",
        plate_number=plate_roman,
        panel_width=1100, panel_height=1100,
        gap=56, margin=80, seed=29,
    )
    print(f"  plate: {plate_path.name}")

    with (out_dir / f"{label}_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"\nOutputs in {out_dir}")


if __name__ == "__main__":
    main()
