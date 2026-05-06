"""
rendering
=========

Score-plot and composite renderer for the Nocturne Atlas. The aesthetic
register is engraved-atlas: ivory linework on a midnight ground, with
serif labels and an ornate frame.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import pretty_midi


# Palette used in the score plot — matches the chart's midnight/copper palette
SCORE_BG = "#0a1226"          # midnight ground
SCORE_BG_OUTER = "#040712"    # vignette outer
SCORE_LINE = "#604528"        # frame
SCORE_NOTE = "#f4ecd8"        # bone ivory
SCORE_NOTE_DIM = "#aa9c80"
SCORE_GRID = "#1a2238"
SCORE_TEXT = "#dccfa8"


def render_score_plot(midi: pretty_midi.PrettyMIDI,
                      output_path,
                      title: str = "",
                      width: int = 2400,
                      height: int = 700,
                      dpi: int = 100):
    """Render a piano-roll-style score in the engraved-atlas register."""
    output_path = Path(output_path)
    fig_w = width / dpi
    fig_h = height / dpi
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    ax.set_facecolor(SCORE_BG)
    fig.patch.set_facecolor(SCORE_BG_OUTER)

    if not midi.instruments:
        ax.text(0.5, 0.5, "(empty)", color=SCORE_TEXT,
                transform=ax.transAxes, ha="center", va="center")
    else:
        end_time = max(1.0, midi.get_end_time())
        # Faint horizontal staff guides
        for p in (36, 48, 60, 72, 84, 96):
            ax.axhline(p, color=SCORE_GRID, linewidth=0.5, alpha=0.7)

        for inst in midi.instruments:
            for note in inst.notes:
                vel_norm = note.velocity / 127.0
                # bone-ivory color, dimmed by velocity
                color = (
                    0.957 * (0.55 + 0.45 * vel_norm),
                    0.925 * (0.55 + 0.45 * vel_norm),
                    0.846 * (0.55 + 0.45 * vel_norm),
                    0.45 + 0.5 * vel_norm,
                )
                rect = plt.Rectangle(
                    (note.start, note.pitch - 0.45),
                    note.end - note.start, 0.9,
                    facecolor=color, edgecolor="none",
                )
                ax.add_patch(rect)
        ax.set_xlim(0, end_time)
        ax.set_ylim(28, 100)

    for spine in ax.spines.values():
        spine.set_color(SCORE_LINE)
        spine.set_linewidth(1.4)
    ax.tick_params(colors=SCORE_TEXT, labelsize=8)
    ax.set_xlabel("time (s)", color=SCORE_TEXT, fontsize=10)
    ax.set_ylabel("MIDI pitch (A0=21, C8=108)", color=SCORE_TEXT, fontsize=10)
    if title:
        ax.set_title(title, color=SCORE_TEXT, fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, facecolor=SCORE_BG_OUTER, bbox_inches="tight")
    plt.close(fig)
    return output_path


def render_atlas_plate(images,
                       labels,
                       output_path,
                       panel_height: int = 1100,
                       panel_width: int = 1100,
                       gap: int = 32,
                       margin: int = 40,
                       background=(8, 12, 24),
                       title: str = "",
                       layout: str = "grid"):
    """Compose a 2x2 plate from four artifacts in the atlas register.

    layout="grid":
        [source chart]    [chart -> nocturne piano roll]
        [source nocturne] [nocturne -> chart planisphere]

    layout="stacked":
        single column, panels stacked vertically.
    """
    output_path = Path(output_path)
    pil = []
    for src in images:
        if isinstance(src, (str, Path)):
            img = Image.open(src).convert("RGB")
        else:
            img = src.convert("RGB")
        frame = Image.new("RGB", (panel_width, panel_height), background)
        ratio = min(panel_width / img.width, panel_height / img.height)
        new_w = max(1, int(img.width * ratio))
        new_h = max(1, int(img.height * ratio))
        scaled = img.resize((new_w, new_h), Image.LANCZOS)
        frame.paste(scaled, ((panel_width - new_w) // 2,
                             (panel_height - new_h) // 2))
        pil.append(frame)

    if layout == "grid" and len(pil) == 4:
        title_pad = 64 if title else 0
        label_pad = 38
        cell_h = panel_height + label_pad
        cell_w = panel_width
        total_w = margin * 2 + cell_w * 2 + gap
        total_h = margin * 2 + title_pad + cell_h * 2 + gap

        canvas = Image.new("RGB", (total_w, total_h), background)
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

        if title and font:
            bbox = draw.textbbox((0, 0), title, font=font)
            tw = bbox[2] - bbox[0]
            draw.text(((total_w - tw) // 2, margin),
                      title, fill=(220, 200, 160), font=font)

        positions = [
            (margin, margin + title_pad),
            (margin + cell_w + gap, margin + title_pad),
            (margin, margin + title_pad + cell_h + gap),
            (margin + cell_w + gap, margin + title_pad + cell_h + gap),
        ]
        for img, label, (x, y) in zip(pil, labels, positions):
            canvas.paste(img, (x, y))
            # Engraved frame around panel
            draw.rectangle([(x - 2, y - 2),
                            (x + panel_width + 1, y + panel_height + 1)],
                           outline=(140, 110, 60), width=1)
            if font:
                bbox = draw.textbbox((0, 0), label, font=font)
                tw = bbox[2] - bbox[0]
                draw.text((x + (panel_width - tw) // 2, y + panel_height + 8),
                          label, fill=(200, 184, 144), font=font)

        canvas.save(output_path)
        return output_path

    # stacked layout
    n = len(pil)
    title_pad = 60 if title else 0
    label_pad = 36
    total_w = margin * 2 + panel_width
    total_h = margin * 2 + title_pad + n * (panel_height + label_pad) + max(0, n - 1) * gap

    canvas = Image.new("RGB", (total_w, total_h), background)
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    if title and font:
        bbox = draw.textbbox((0, 0), title, font=font)
        tw = bbox[2] - bbox[0]
        draw.text(((total_w - tw) // 2, margin),
                  title, fill=(220, 200, 160), font=font)

    y_cursor = margin + title_pad
    for i, (img, label) in enumerate(zip(pil, labels)):
        x = (total_w - panel_width) // 2
        canvas.paste(img, (x, y_cursor))
        draw.rectangle([(x - 2, y_cursor - 2),
                        (x + panel_width + 1, y_cursor + panel_height + 1)],
                       outline=(140, 110, 60), width=1)
        y_cursor += panel_height
        if font:
            bbox = draw.textbbox((0, 0), label, font=font)
            tw = bbox[2] - bbox[0]
            draw.text(((total_w - tw) // 2, y_cursor + 8),
                      label, fill=(200, 184, 144), font=font)
        y_cursor += label_pad
        if i < n - 1:
            y_cursor += gap

    canvas.save(output_path)
    return output_path


def midi_to_audio(midi: pretty_midi.PrettyMIDI, sample_rate: int = 22050):
    """Render MIDI to a mono audio buffer using a soft-attack envelope.

    pretty_midi's built-in synthesize() uses sine tones. We post-process
    with a slow attack/release envelope and add a touch of ringing
    sustain to evoke the felt-hammer character of a salon piano.
    """
    base = midi.synthesize(fs=sample_rate, wave=np.sin)
    if base.size == 0:
        return base
    # Soft-knee global envelope: 0.4s fade in, 0.6s fade out
    n = base.shape[0]
    fade_in = min(int(sample_rate * 0.4), n // 2)
    fade_out = min(int(sample_rate * 0.6), n // 2)
    env = np.ones(n, dtype=np.float32)
    if fade_in > 0:
        env[:fade_in] *= np.linspace(0, 1, fade_in) ** 1.5
    if fade_out > 0:
        env[-fade_out:] *= np.linspace(1, 0, fade_out) ** 1.5
    base = base * env
    # Gentle low-pass via running mean to soften the sine harshness
    kernel = max(3, sample_rate // 4000)
    cumsum = np.cumsum(np.insert(base, 0, 0))
    smoothed = (cumsum[kernel:] - cumsum[:-kernel]) / kernel
    smoothed = np.concatenate([smoothed, np.zeros(n - smoothed.size, dtype=np.float32)])
    out = (0.65 * base + 0.35 * smoothed).astype(np.float32)
    peak = float(np.max(np.abs(out)) + 1e-9)
    out = out / peak * 0.85
    return out
