"""
parchment_rendering
===================

Render the Nocturne Atlas in the parchment register: dark sepia ink on
warm aged paper, in the manner of an antiquarian astronomical chart.

This module provides parchment-styled equivalents of the chart, score,
waveform, and four-panel atlas plate, replacing the midnight palette
of the artistic register with a warm cream paper ground.

  - make_parchment_ground(w, h, seed)
  - parchment_score(midi, output_path, ...)
  - parchment_waveform(audio_path, output_path, ...)
  - parchment_chart_overlay(planisphere_image)   (post-process: invert + tint)
  - parchment_plate(panels, labels, output_path, ...)
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import librosa
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import pretty_midi


# Parchment palette
PAPER_INNER = (236, 220, 188)   # warm cream centre
PAPER_OUTER = (210, 188, 152)   # aged darker rim
PAPER_SHADOW = (158, 132, 96)   # very dark edge wear
INK_DARK = (52, 32, 16)         # main ink (notes, frame strokes)
INK_WARM = (108, 72, 32)        # warm halo/wash ink
INK_DIM = (150, 116, 76)        # very faint ink
FRAME_INK = (96, 60, 24)        # frame strokes
FIBER = (200, 178, 142)         # paper fibre highlights
FOXING = (124, 78, 36)          # foxing stains


# -------------------------------------------------------------------
# Parchment ground
# -------------------------------------------------------------------


def make_parchment_ground(w: int, h: int, seed: int = 42) -> Image.Image:
    """Create a warm parchment ground with subtle texture."""
    rng = np.random.RandomState(seed)

    # Base cream
    base = np.tile(np.array(PAPER_INNER, dtype=np.float32), (h, w, 1))

    # Multi-octave color noise (low-freq variations)
    for step, weight in zip((48, 96, 192), (0.55, 0.32, 0.13)):
        sw = max(2, w // step)
        sh = max(2, h // step)
        small = rng.normal(0, 1.0, (sh, sw, 3)).astype(np.float32)
        small = np.clip(small * 14 + 128, 0, 255).astype(np.uint8)
        scaled = Image.fromarray(small).resize((w, h), Image.BICUBIC)
        var = (np.asarray(scaled, dtype=np.float32) - 128) * weight * 0.7
        base = base + var

    # Edge darkening toward warmer shadow tone
    yy, xx = np.mgrid[:h, :w]
    cx, cy = w / 2.0, h / 2.0
    R = math.hypot(cx, cy)
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) / R
    rr = np.clip(rr, 0.0, 1.0)
    shadow_strength = (rr ** 2.4) * 0.42
    outer = np.array(PAPER_OUTER, dtype=np.float32)
    base = base * (1.0 - shadow_strength[..., None]) + outer * shadow_strength[..., None]

    # Fine grain
    grain = rng.normal(0, 5.5, (h, w, 3))
    base = base + grain

    # Faint horizontal fibre striations
    fibre_layer = np.zeros((h, w, 3), dtype=np.float32)
    n_fibres = max(40, h // 25)
    fibre_color = np.array(FIBER, dtype=np.float32)
    for _ in range(n_fibres):
        y = rng.randint(0, h)
        thickness = rng.choice([1, 1, 1, 2])
        alpha = rng.uniform(0.03, 0.10)
        x_start = rng.randint(0, w // 4)
        x_end = rng.randint(3 * w // 4, w)
        fibre_layer[y:y + thickness, x_start:x_end, :] = fibre_color * alpha
    base = base + fibre_layer

    # Foxing spots
    n_spots = rng.randint(10, 16)
    for _ in range(n_spots):
        cx_s = rng.randint(0, w)
        cy_s = rng.randint(0, h)
        size = rng.uniform(10, 28)
        intensity = rng.uniform(0.08, 0.22)
        # render disc with falling alpha
        y_min = max(0, int(cy_s - size))
        y_max = min(h, int(cy_s + size))
        x_min = max(0, int(cx_s - size))
        x_max = min(w, int(cx_s + size))
        ys, xs = np.mgrid[y_min:y_max, x_min:x_max]
        d = np.sqrt((xs - cx_s) ** 2 + (ys - cy_s) ** 2) / max(size, 1.0)
        d = np.clip(d, 0.0, 1.0)
        spot_alpha = (1.0 - d) * intensity
        fox = np.array(FOXING, dtype=np.float32)
        region = base[y_min:y_max, x_min:x_max, :]
        region = region * (1 - spot_alpha[..., None]) + fox * spot_alpha[..., None]
        base[y_min:y_max, x_min:x_max, :] = region

    base = np.clip(base, 0, 255).astype(np.uint8)
    return Image.fromarray(base, mode="RGB").convert("RGBA")


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------


def _serif_font(size: int):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSerif.ttf",
        "C:\\Windows\\Fonts\\Times.ttf",
    ]
    for p in candidates:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return ImageFont.load_default()


def _draw_engraved_frame(img: Image.Image, pad: int):
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    w, h = img.size
    p = pad
    for stroke_w in (3, 1, 1):
        draw.rectangle([(p, p), (w - p, h - p)],
                       outline=(*FRAME_INK, 230), width=stroke_w)
        p += 8
    img.alpha_composite(layer)


# -------------------------------------------------------------------
# Score plot in parchment register
# -------------------------------------------------------------------


def parchment_score(midi: pretty_midi.PrettyMIDI,
                     output_path,
                     title: str = "",
                     subtitle: str = "",
                     width: int = 2400, height: int = 900,
                     seed: int = 42):
    output_path = Path(output_path)
    img = make_parchment_ground(width, height, seed=seed)

    margin = 56
    inner = (margin, margin, width - margin, height - margin)
    inner_w = inner[2] - inner[0]
    inner_h = inner[3] - inner[1]

    _draw_engraved_frame(img, margin)

    if not midi.instruments:
        return _save(img, output_path)

    end_time = max(1.0, midi.get_end_time())
    pitch_lo, pitch_hi = 28, 100
    pitch_range = pitch_hi - pitch_lo

    # Faint horizontal staff guides at octave Cs (in dim ink)
    guide_layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    g_draw = ImageDraw.Draw(guide_layer)
    for octave_c in range(36, 109, 12):
        if octave_c < pitch_lo or octave_c > pitch_hi:
            continue
        y = inner[3] - (octave_c - pitch_lo) / pitch_range * inner_h
        g_draw.line([(inner[0] + 8, y), (inner[2] - 8, y)],
                    fill=(*INK_DIM, 90), width=1)
    img.alpha_composite(guide_layer)

    # Notes as dark ink dots with thin trailing line
    halo_layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    halo_draw = ImageDraw.Draw(halo_layer)
    core_layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    core_draw = ImageDraw.Draw(core_layer)

    for inst in midi.instruments:
        for note in inst.notes:
            if note.pitch < pitch_lo or note.pitch > pitch_hi:
                continue
            x_start = inner[0] + (note.start / end_time) * inner_w
            x_end = inner[0] + (note.end / end_time) * inner_w
            y = inner[3] - (note.pitch - pitch_lo) / pitch_range * inner_h
            mag = note.velocity / 127.0

            attack_radius = 1.6 + mag * 4.5
            line_alpha = int(70 + 130 * mag)
            halo_alpha = int(40 + 110 * mag)

            # Trailing line in warm ink
            halo_draw.line([(x_start, y), (x_end, y)],
                            fill=(*INK_WARM, line_alpha),
                            width=max(1, int(1 + mag * 2)))
            # Soft warm halo around the attack
            halo_r = attack_radius * 2.4
            halo_draw.ellipse(
                [(x_start - halo_r, y - halo_r),
                 (x_start + halo_r, y + halo_r)],
                fill=(*INK_WARM, halo_alpha),
            )
            # Dark ink core
            core_draw.ellipse(
                [(x_start - attack_radius, y - attack_radius),
                 (x_start + attack_radius, y + attack_radius)],
                fill=(*INK_DARK, int(220 + 30 * mag)),
            )

            # 4-point ink spike for the loudest notes
            if mag > 0.55:
                spike = halo_r * 1.4
                core_draw.line(
                    [(x_start - spike, y), (x_start + spike, y)],
                    fill=(*INK_DARK, int(140 + 100 * (mag - 0.55) / 0.45)),
                    width=1,
                )
                core_draw.line(
                    [(x_start, y - spike * 0.6), (x_start, y + spike * 0.6)],
                    fill=(*INK_DARK, int(140 + 100 * (mag - 0.55) / 0.45)),
                    width=1,
                )

    halo_layer = halo_layer.filter(ImageFilter.GaussianBlur(1.6))
    img.alpha_composite(halo_layer)
    img.alpha_composite(core_layer)

    # Latin axis labels in dark ink
    font = _serif_font(20)
    _draw_text(img, "TEMPVS  →", (margin + 10, height - margin + 10),
               font=font, color=(*INK_DARK, 230))
    _draw_text(img, "↑  ALTITVDO", (margin + 10, margin - 32),
               font=font, color=(*INK_DARK, 230))

    if title:
        title_font = _serif_font(28)
        _draw_text(img, title.upper(), (width // 2, 22), font=title_font,
                   color=(*INK_DARK, 240), centered=True)
    if subtitle:
        sub_font = _serif_font(16)
        _draw_text(img, subtitle, (width // 2, height - 30), font=sub_font,
                   color=(*INK_WARM, 220), centered=True, italic=True)

    return _save(img, output_path)


# -------------------------------------------------------------------
# Waveform on parchment
# -------------------------------------------------------------------


def parchment_waveform(audio_path,
                        output_path,
                        width: int = 2400, height: int = 900,
                        seed: int = 11):
    import soundfile as sf
    output_path = Path(output_path)
    audio, sr = sf.read(str(audio_path))
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    img = make_parchment_ground(width, height, seed=seed)

    n = audio.shape[0]
    bins = width - 80
    if n < bins:
        return _save(img, output_path)
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
    # axis line
    draw.line([(40, mid), (width - 40, mid)], fill=(*INK_DIM, 180), width=1)
    for i in range(bins):
        x = i + 40
        y_top = int(mid - peaks_norm[i, 1] * half)
        y_bot = int(mid - peaks_norm[i, 0] * half)
        amp = max(abs(peaks_norm[i, 1]), abs(peaks_norm[i, 0]))
        intensity_alpha = int(160 + 90 * amp)
        col = (*INK_DARK, intensity_alpha)
        draw.line([(x, y_top), (x, y_bot)], fill=col, width=1)
    img.alpha_composite(layer)

    _draw_engraved_frame(img, 24)
    return _save(img, output_path)


# -------------------------------------------------------------------
# Chart conversion: invert the midnight chart into parchment register
# -------------------------------------------------------------------


def parchment_chart_overlay(midnight_chart: Image.Image,
                              seed: int = 31) -> Image.Image:
    """Re-render a midnight planispheric chart on parchment ground.

    The original midnight chart has bright stars on dark ground. We
    extract the brightness map and use it to draw dark sepia stars on
    a parchment ground. A boost curve raises mid-luminance pixels so
    that faint background stars and constellation lines remain visible
    in the parchment-ink rendering.
    """
    src = midnight_chart.convert("RGB")
    arr = np.asarray(src, dtype=np.float32)
    # luminance
    lum = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    lum = lum / 255.0
    # boost curve: brighten faint pixels, preserve bright cores
    boosted = np.power(np.clip(lum, 0.0, 1.0), 0.62)

    h, w = lum.shape
    paper = make_parchment_ground(w, h, seed=seed)
    paper_arr = np.asarray(paper.convert("RGB"), dtype=np.float32)

    # Tint with dark ink: two layers, warm halo (broad faint) + dark core (sharp)
    ink = np.array(INK_DARK, dtype=np.float32)
    ink_warm = np.array(INK_WARM, dtype=np.float32)

    # Warm halo: any visible pixel adds some warm ink (faint)
    warm_halo = boosted * 0.46
    # Dark core: stronger ink where the original was bright
    dark_core_mask = np.clip((boosted - 0.35) / 0.65, 0.0, 1.0)
    dark_core = np.power(dark_core_mask, 0.85) * 0.92

    out = paper_arr * (1 - warm_halo[..., None]) + ink_warm * warm_halo[..., None]
    out = out * (1 - dark_core[..., None]) + ink * dark_core[..., None]

    out = np.clip(out, 0, 255).astype(np.uint8)
    return Image.fromarray(out).convert("RGBA")


# -------------------------------------------------------------------
# Composite plate on parchment
# -------------------------------------------------------------------


def parchment_plate(panels: Sequence,
                     labels: Sequence[str],
                     output_path,
                     title: str = "",
                     plate_number: str = "I",
                     panel_width: int = 1100,
                     panel_height: int = 1100,
                     gap: int = 56,
                     margin: int = 80,
                     seed: int = 23):
    output_path = Path(output_path)
    cell_w = panel_width
    cell_h = panel_height
    title_pad = 90 if title else 0
    label_pad = 56
    total_w = margin * 2 + cell_w * 2 + gap
    total_h = margin * 2 + title_pad + (cell_h + label_pad) * 2 + gap

    canvas = make_parchment_ground(total_w, total_h, seed=seed)
    _draw_engraved_frame(canvas, margin // 2 + 8)

    # Title cartouche
    if title:
        cy = margin + title_pad // 2
        cw = int(total_w * 0.62)
        ch = 70
        cx = total_w // 2
        cart_layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        cd = ImageDraw.Draw(cart_layer)
        box = [(cx - cw // 2, cy - ch // 2),
               (cx + cw // 2, cy + ch // 2)]
        # paper-toned cartouche with dark ink frame
        cd.rounded_rectangle(box, radius=12,
                              fill=(255, 248, 222, 200),
                              outline=(*FRAME_INK, 230), width=2)
        cd.rounded_rectangle(
            [(box[0][0] + 6, box[0][1] + 6),
             (box[1][0] - 6, box[1][1] - 6)],
            radius=8, outline=(*FRAME_INK, 150), width=1,
        )
        canvas.alpha_composite(cart_layer)
        title_font = _serif_font(28)
        plate_font = _serif_font(16)
        _draw_text(canvas, title.upper(), (cx, cy - 8), font=title_font,
                   color=(*INK_DARK, 240), centered=True)
        if plate_number:
            _draw_text(canvas, f"·  TABVLA  {plate_number}  ·",
                        (cx, cy + 18), font=plate_font,
                        color=(*INK_WARM, 220), centered=True, italic=True)

    # Place panels
    positions = [
        (margin, margin + title_pad),
        (margin + cell_w + gap, margin + title_pad),
        (margin, margin + title_pad + cell_h + label_pad + gap),
        (margin + cell_w + gap, margin + title_pad + cell_h + label_pad + gap),
    ]
    label_font = _serif_font(20)

    for src, label, (x, y) in zip(panels, labels, positions):
        if isinstance(src, (str, Path)):
            panel = Image.open(src).convert("RGBA")
        else:
            panel = src.convert("RGBA")
        ratio = min(cell_w / panel.width, cell_h / panel.height)
        new_w = max(1, int(panel.width * ratio))
        new_h = max(1, int(panel.height * ratio))
        scaled = panel.resize((new_w, new_h), Image.LANCZOS)
        # composite onto a small parchment cell so cell background matches paper
        cell_canvas = make_parchment_ground(cell_w, cell_h, seed=seed + (x + y))
        cell_canvas.paste(scaled,
                          ((cell_w - new_w) // 2, (cell_h - new_h) // 2),
                          mask=scaled if scaled.mode == "RGBA" else None)
        canvas.alpha_composite(cell_canvas, dest=(x, y))
        # cell frame in dark ink
        cell_layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        cd = ImageDraw.Draw(cell_layer)
        cd.rectangle([(x - 2, y - 2), (x + cell_w + 1, y + cell_h + 1)],
                     outline=(*FRAME_INK, 230), width=2)
        cd.rectangle([(x - 6, y - 6), (x + cell_w + 5, y + cell_h + 5)],
                     outline=(*FRAME_INK, 110), width=1)
        canvas.alpha_composite(cell_layer)

        # Roman numeral
        cell_idx = positions.index((x, y)) + 1
        roman = ["I", "II", "III", "IV"][cell_idx - 1]
        _draw_text(canvas, roman, (x + 14, y + 10),
                   font=_serif_font(22),
                   color=(*INK_WARM, 200), italic=True)
        # caption
        _draw_text(canvas, label,
                   (x + cell_w // 2, y + cell_h + 10),
                   font=label_font,
                   color=(*INK_DARK, 230), centered=True)

    # Bottom motto cartouche
    cy_b = total_h - margin // 2 - 10
    w_box = int(total_w * 0.46)
    h_box = 56
    cx_b = total_w // 2
    cart_b = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    cdb = ImageDraw.Draw(cart_b)
    box_b = [(cx_b - w_box // 2, cy_b - h_box // 2),
             (cx_b + w_box // 2, cy_b + h_box // 2)]
    cdb.rounded_rectangle(box_b, radius=10,
                            fill=(255, 248, 222, 200),
                            outline=(*FRAME_INK, 230), width=2)
    cdb.rounded_rectangle(
        [(box_b[0][0] + 5, box_b[0][1] + 5),
         (box_b[1][0] - 5, box_b[1][1] - 5)],
        radius=7, outline=(*FRAME_INK, 130), width=1,
    )
    canvas.alpha_composite(cart_b)
    _draw_text(canvas,
                "AVDITVS  IN  STELLAS  ·  STELLAE  IN  AVDITVM",
                (cx_b, cy_b - 4),
                font=_serif_font(15),
                color=(*INK_DARK, 220), centered=True, italic=True)

    return _save(canvas, output_path)


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------


def _draw_text(img: Image.Image, text: str, xy, font=None, color=None,
                centered: bool = False, italic: bool = False):
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    if font is None:
        font = _serif_font(18)
    if color is None:
        color = (*INK_DARK, 230)
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
    except Exception:
        tw = len(text) * 8
        th = 14
    if centered:
        x = xy[0] - tw // 2
        y = xy[1] - th // 2
    else:
        x, y = xy
    draw.text((x, y), text, fill=color, font=font)
    img.alpha_composite(layer)


def _save(img: Image.Image, path: Path):
    img.convert("RGB").save(path)
    return path
