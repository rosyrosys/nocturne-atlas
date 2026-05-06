"""
run_atlas_artistic.py
=====================

Generate Plate I and Plate II using the artistic renderers in
`transcoder.artistic_rendering`. Outputs go to `outputs/<label>_artistic/`.

Each plate is a 4-panel composite with engraved corner ornaments,
decorative cartouche with scrollwork, Roman-numeral panel indices, and
paper-aging vignette. The score plot is rendered in the same midnight
register as the planisphere (no matplotlib).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from transcoder import (   # noqa: E402
    chart_to_nocturne, NocturneConfig,
    nocturne_to_chart, ChartConfig,
)
from transcoder.artistic_rendering import (   # noqa: E402
    render_artistic_score, render_artistic_plate, _apply_paper_aging,
    _midnight_gradient_fill, SCORE_BG_INNER, SCORE_BG_OUTER, SCORE_FRAME,
)
from transcoder.rendering import midi_to_audio   # noqa: E402

INPUTS = ROOT / "inputs"
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)


def waveform_artistic(audio_path: Path,
                      width: int = 2400,
                      height: int = 900) -> Image.Image:
    """Waveform rendered against the midnight ground with engraved frame."""
    audio, sr = sf.read(str(audio_path))
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    img = Image.new("RGBA", (width, height), (0, 0, 0, 255))
    img = _midnight_gradient_fill(img, SCORE_BG_INNER, SCORE_BG_OUTER)

    n = audio.shape[0]
    bins = width - 80
    if n < bins:
        return img.convert("RGB")
    step = n / bins
    peaks = np.zeros((bins, 2), dtype=np.float32)
    for i in range(bins):
        seg = audio[int(i * step):int((i + 1) * step)]
        if seg.size > 0:
            peaks[i, 0] = seg.min()
            peaks[i, 1] = seg.max()
    peaks_norm = peaks / (np.max(np.abs(peaks)) + 1e-9)
    mid = height // 2
    half = height // 2 - 80

    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    # axis
    draw.line([(40, mid), (width - 40, mid)], fill=(38, 50, 84, 200), width=1)
    for i in range(bins):
        x = i + 40
        y_top = int(mid - peaks_norm[i, 1] * half)
        y_bot = int(mid - peaks_norm[i, 0] * half)
        amp = max(abs(peaks_norm[i, 1]), abs(peaks_norm[i, 0]))
        intensity = int(140 + 110 * amp)
        col = (intensity, intensity - 8, intensity - 32, 230)
        draw.line([(x, y_top), (x, y_bot)], fill=col, width=1)
    img.alpha_composite(layer)

    # frame
    frame_layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    fd = ImageDraw.Draw(frame_layer)
    pad = 24
    for w in (3, 1, 1):
        fd.rectangle([(pad, pad), (width - pad, height - pad)],
                     outline=(*SCORE_FRAME, 230), width=w)
        pad += 8
    img.alpha_composite(frame_layer)
    return _apply_paper_aging(img).convert("RGB")


def run_pair(chart_path: Path, audio_path: Path, label: str,
             plate_roman: str,
             palette: str = "midnight",
             mode: str = "chopin_csm",
             duration: float = 90.0):
    out_dir = OUTPUTS / f"{label}_artistic"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== {label} (artistic) ===")

    # 1) chart -> MIDI
    midi_cfg = NocturneConfig(
        duration_seconds=duration,
        mode=mode,
        pitch_floor_midi=40,
        pitch_ceiling_midi=92,
        pitch_focus_midi=72,
        pitch_focus_weight=0.45,
        star_threshold=0.55,
        star_min_size=4,
        velocity_floor=30,
        velocity_ceiling=110,
    )
    pm = chart_to_nocturne(chart_path, config=midi_cfg)
    midi_path = out_dir / f"{label}_chart_to_nocturne.mid"
    pm.write(str(midi_path))
    print(f"  midi: {midi_path.name} ({sum(len(i.notes) for i in pm.instruments)} notes)")

    # 2) MIDI -> audio
    rendered = midi_to_audio(pm, sample_rate=22050)
    midi_wav = out_dir / f"{label}_chart_to_nocturne.wav"
    sf.write(str(midi_wav), rendered.astype(np.float32), 22050, subtype="PCM_16")

    # 3) Artistic score plot
    score_path = out_dir / f"{label}_score_artistic.png"
    render_artistic_score(
        pm, score_path,
        title="CHART  TRANSCODED  TO  NOCTURNE",
        subtitle=f"Tabula {plate_roman}  ·  {label.replace('_', ' ')}",
        width=2400, height=900,
    )
    print(f"  score: {score_path.name}")

    # 4) audio -> chart
    chart_cfg = ChartConfig(
        width=2200, height=2200,
        sr=22050, n_fft=4096, hop_length=2048,
        palette=palette,
        n_peaks_per_frame=4, min_peak_db=-50.0,
        cartouche_text_top=f"NOCTVRNVS  IN  STELLAS",
        cartouche_text_subtitle=f"Tabula {plate_roman} :: {label.replace('_', ' ')}",
        cartouche_text_latin="DETERMINISTICE  ·  TRANSCODATA",
    )
    chart_img = nocturne_to_chart(audio_path, config=chart_cfg)
    chart_aged = _apply_paper_aging(chart_img.convert("RGBA")).convert("RGB")
    chart_out = out_dir / f"{label}_nocturne_to_chart.png"
    chart_aged.save(chart_out)
    print(f"  derived chart: {chart_out.name}")

    # 5) waveform (artistic)
    wave_path = out_dir / f"{label}_waveform_artistic.png"
    waveform_artistic(audio_path).save(wave_path)

    # 6) Apply paper aging to the input chart for visual coherence
    src_chart = Image.open(chart_path).convert("RGBA")
    src_aged_path = out_dir / f"{label}_source_chart_aged.png"
    _apply_paper_aging(src_chart).convert("RGB").save(src_aged_path)

    # 7) Final atlas plate (4-panel artistic composite)
    plate_path = out_dir / f"{label}_plate_artistic.png"
    render_artistic_plate(
        panels=[src_aged_path, score_path, wave_path, chart_out],
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
        gap=56, margin=80,
    )
    print(f"  plate: {plate_path.name}")
    return plate_path


def main():
    audio_path = INPUTS / "nocturne_csm_standin.wav"
    if not audio_path.exists():
        print(f"Missing demo audio at {audio_path}.")
        sys.exit(1)

    pairs = [
        ("plate_i_borealis", "tabula_stellarum_standin.png", "I",
         "midnight", "chopin_csm"),
        ("plate_ii_australis", "tabula_stellarum_alt.png", "II",
         "indigo", "octatonic"),
    ]
    for label, name, roman, palette, mode in pairs:
        chart_path = INPUTS / name
        if not chart_path.exists():
            print(f"Skipping {label}: {chart_path} missing.")
            continue
        run_pair(chart_path, audio_path, label, roman,
                 palette=palette, mode=mode)
    print(f"\nAll outputs in {OUTPUTS}")


if __name__ == "__main__":
    main()
