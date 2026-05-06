"""
run_atlas_parchment.py
======================

Generate Plate I and Plate II in the parchment register: dark sepia
ink on warm aged paper. Outputs go to `outputs/<label>_parchment/`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from transcoder import (   # noqa: E402
    chart_to_nocturne, NocturneConfig,
    nocturne_to_chart, ChartConfig,
)
from transcoder.parchment_rendering import (   # noqa: E402
    parchment_score, parchment_waveform,
    parchment_chart_overlay, parchment_plate,
)
from transcoder.rendering import midi_to_audio   # noqa: E402

INPUTS = ROOT / "inputs"
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)


def run_pair(chart_path: Path, audio_path: Path, label: str,
             plate_roman: str,
             palette: str = "midnight",
             mode: str = "chopin_csm",
             duration: float = 90.0):
    out_dir = OUTPUTS / f"{label}_parchment"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== {label} (parchment) ===")

    # 1) chart → MIDI
    pm = chart_to_nocturne(chart_path, config=NocturneConfig(
        duration_seconds=duration, mode=mode,
        pitch_floor_midi=40, pitch_ceiling_midi=92,
        pitch_focus_midi=72, pitch_focus_weight=0.45,
        star_threshold=0.55, star_min_size=4,
        velocity_floor=30, velocity_ceiling=110,
    ))
    midi_path = out_dir / f"{label}_chart_to_nocturne.mid"
    pm.write(str(midi_path))

    # 2) MIDI → audio
    rendered = midi_to_audio(pm, sample_rate=22050)
    sf.write(str(out_dir / f"{label}_chart_to_nocturne.wav"),
              rendered.astype(np.float32), 22050, subtype="PCM_16")

    # 3) Score in parchment register
    score = out_dir / f"{label}_score_parchment.png"
    parchment_score(pm, score,
                     title="CHART  TRANSCODED  TO  NOCTURNE",
                     subtitle=f"Tabula {plate_roman}  ·  {label.replace('_', ' ')}",
                     width=2400, height=900, seed=37)
    print(f"  score: {score.name}")

    # 4) Source chart → parchment
    from PIL import Image
    src_chart = Image.open(chart_path).convert("RGBA")
    src_p = parchment_chart_overlay(src_chart, seed=11)
    src_out = out_dir / f"{label}_source_chart_parchment.png"
    src_p.convert("RGB").save(src_out)
    print(f"  source: {src_out.name}")

    # 5) audio → planispheric chart, then parchment overlay
    chart_cfg = ChartConfig(
        width=2200, height=2200,
        sr=22050, n_fft=4096, hop_length=2048,
        palette=palette,
        n_peaks_per_frame=4, min_peak_db=-50.0,
        cartouche_text_top="NOCTVRNVS  IN  STELLAS",
        cartouche_text_subtitle=f"Tabula {plate_roman} :: {label.replace('_', ' ')}",
        cartouche_text_latin="DETERMINISTICE  ·  TRANSCODATA",
    )
    chart_img = nocturne_to_chart(audio_path, config=chart_cfg)
    chart_p = parchment_chart_overlay(chart_img, seed=53)
    chart_out = out_dir / f"{label}_nocturne_to_chart.png"
    chart_p.convert("RGB").save(chart_out)
    print(f"  chart: {chart_out.name}")

    # 6) Waveform on parchment
    wave_path = out_dir / f"{label}_waveform_parchment.png"
    parchment_waveform(audio_path, wave_path, width=2400, height=900, seed=29)

    # 7) Composite plate on parchment
    plate_path = out_dir / f"{label}_plate_parchment.png"
    parchment_plate(
        panels=[src_out, score, wave_path, chart_out],
        labels=[
            "Source chart  (engraved planisphere)",
            "Chart  →  nocturne  (luminous score)",
            "Source nocturne  (audio waveform)",
            "Nocturne  →  chart  (planisphere)",
        ],
        output_path=plate_path,
        title="NOCTURNE  ATLAS",
        plate_number=plate_roman,
        panel_width=1100, panel_height=1100,
        gap=56, margin=80, seed=23,
    )
    print(f"  plate: {plate_path.name}")
    return plate_path


def main():
    audio_path = INPUTS / "nocturne_csm_standin.wav"
    if not audio_path.exists():
        sys.exit("Missing demo audio.")
    pairs = [
        ("plate_i_borealis", "tabula_stellarum_standin.png", "I",
         "midnight", "chopin_csm"),
        ("plate_ii_australis", "tabula_stellarum_alt.png", "II",
         "indigo", "octatonic"),
    ]
    for label, name, roman, palette, mode in pairs:
        chart_path = INPUTS / name
        if not chart_path.exists():
            continue
        run_pair(chart_path, audio_path, label, roman,
                 palette=palette, mode=mode)
    print(f"\nAll outputs in {OUTPUTS}")


if __name__ == "__main__":
    main()
