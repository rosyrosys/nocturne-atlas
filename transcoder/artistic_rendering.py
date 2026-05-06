"""
artistic_rendering
==================

Engraved-atlas-style renderers for the Nocturne Atlas:

  - render_artistic_score : a custom piano-roll-style score plot rendered
    in the same midnight/copper register as the planispheric chart, with
    note glyphs as luminous dots (like stars in a horizontal sky strip),
    period-style frame, and Latin axis labels.

  - render_artistic_plate : a four-panel composite plate with corner
    ornaments, decorative cartouche with scrollwork, plate-numbered title
    in serif caps, and ornamental dividers between panels.

  - apply_paper_aging : a deterministic post-process that adds subtle
    paper grain, sepia tint at the rim, and a soft vignette to evoke the
    look of a centuries-old engraved plate.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageEnhance
import pretty_midi


# Palette aligned with the planisphere's midnight register
SCORE_BG_INNER = (10, 16, 38)
SCORE_BG_OUTER = (4, 7, 18)
SCORE_FRAME = (140, 110, 60)
SCORE_NOTE_CORE = (248, 240, 218)
SCORE_NOTE_HALO = (208, 196, 168)
SCORE_GUIDELINE = (38, 50, 84)
SCORE_TEXT = (220, 200, 160)


# -----------------------------------------------------------
# Custom artistic piano-roll renderer
# -----------------------------------------------------------


def render_artistic_score(midi: pretty_midi.PrettyMIDI,
                          output_path,
                          title: str = "",
                          subtitle: str = "",
                          width: int = 2400,
                          height: int = 900) -> Path:
    """Render a MIDI object as a luminous-dot score on midnight ground."""
    output_path = Path(output_path)
    img = _midnight_ground(width, height)

    # Inner content rect with margin
    margin = 56
    inner = (margin, margin, width - margin, height - margin)
    inner_w = inner[2] - inner[0]
    inner_h = inner[3] - inner[1]

    # Frame with corner ornaments
    _draw_engraved_frame(img, margin)

    if not midi.instruments:
        return _save_with_aging(img, output_path)

    end_time = max(1.0, midi.get_end_time())
    pitch_lo, pitch_hi = 28, 100
    pitch_range = pitch_hi - pitch_lo

    # Faint horizontal staff guides at octave Cs
    guide_layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    g_draw = ImageDraw.Draw(guide_layer)
    for octave_c in range(36, 109, 12):
        if octave_c < pitch_lo or octave_c > pitch_hi:
            continue
        y = inner[3] - (octave_c - pitch_lo) / pitch_range * inner_h
        g_draw.line([(inner[0] + 8, y), (inner[2] - 8, y)],
                    fill=(*SCORE_GUIDELINE, 110), width=1)
    img.alpha_composite(guide_layer)

    # Note glyphs as luminous dots
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

            # Sustained note: a bright dot at attack + faint trailing line
            attack_radius = 2.4 + mag * 6.0
            halo_radius = 5 + mag * 18
            line_alpha = int(60 + 130 * mag)

            # Trailing line
            halo_draw.line([(x_start, y), (x_end, y)],
                           fill=(*SCORE_NOTE_HALO, line_alpha),
                           width=max(1, int(1 + mag * 2)))

            # Halo (large, soft)
            halo_draw.ellipse(
                [(x_start - halo_radius, y - halo_radius),
                 (x_start + halo_radius, y + halo_radius)],
                fill=(*SCORE_NOTE_HALO, int(80 + 140 * mag)),
            )

            # Bright core
            core_draw.ellipse(
                [(x_start - attack_radius, y - attack_radius),
                 (x_start + attack_radius, y + attack_radius)],
                fill=(*SCORE_NOTE_CORE, int(220 + 30 * mag)),
            )

            # Diffraction spike for the loudest notes
            if mag > 0.55:
                spike = halo_radius * 1.8
                spike_alpha = int(140 + 100 * (mag - 0.55) / 0.45)
                core_draw.line(
                    [(x_start - spike, y), (x_start + spike, y)],
                    fill=(*SCORE_NOTE_CORE, spike_alpha), width=1)
                core_draw.line(
                    [(x_start, y - spike * 0.6), (x_start, y + spike * 0.6)],
                    fill=(*SCORE_NOTE_CORE, spike_alpha), width=1)

    halo_layer = halo_layer.filter(ImageFilter.GaussianBlur(2.4))
    img.alpha_composite(halo_layer)
    img.alpha_composite(core_layer)

    # Latin axis labels in serif at the corners
    font = _serif_font(20)
    _draw_text(img, "TEMPVS  →", (margin + 10, height - margin + 10),
               font=font, color=(*SCORE_TEXT, 230))
    _draw_text(img, "↑  ALTITVDO", (margin + 10, margin - 32),
               font=font, color=(*SCORE_TEXT, 230))

    # Title in serif caps
    if title:
        title_font = _serif_font(28)
        _draw_text(img, title.upper(),
                   (width // 2, 22), font=title_font,
                   color=(*SCORE_TEXT, 240), centered=True)
    if subtitle:
        sub_font = _serif_font(16)
        _draw_text(img, subtitle,
                   (width // 2, height - 30), font=sub_font,
                   color=(*SCORE_TEXT, 200), centered=True, italic=True)

    return _save_with_aging(img, output_path)


# -----------------------------------------------------------
# Decorative atlas plate composite
# -----------------------------------------------------------


def render_artistic_plate(panels,
                          labels,
                          output_path,
                          title: str = "",
                          plate_number: str = "I",
                          panel_width: int = 1100,
                          panel_height: int = 1100,
                          gap: int = 56,
                          margin: int = 80) -> Path:
    """Compose a four-panel atlas plate with engraved-style decorations.

    panels   : sequence of 4 image paths or PIL Images
    labels   : sequence of 4 caption strings
    """
    output_path = Path(output_path)

    cell_w = panel_width
    cell_h = panel_height
    title_pad = 90 if title else 0
    label_pad = 56
    total_w = margin * 2 + cell_w * 2 + gap
    total_h = margin * 2 + title_pad + (cell_h + label_pad) * 2 + gap

    canvas = Image.new("RGBA", (total_w, total_h), (*SCORE_BG_OUTER, 255))
    canvas = _midnight_gradient_fill(canvas, SCORE_BG_INNER, SCORE_BG_OUTER)

    # Outer frame with corner ornaments
    _draw_engraved_frame(canvas, margin // 2 + 8)
    _draw_corner_ornaments(canvas, margin // 2 + 8)

    # Title cartouche
    if title:
        _draw_title_cartouche(canvas, title, plate_number,
                               cy=margin + title_pad // 2,
                               width=int(total_w * 0.62))

    # Place the four panels in 2x2 grid
    positions = [
        (margin, margin + title_pad),                     # top-left
        (margin + cell_w + gap, margin + title_pad),      # top-right
        (margin, margin + title_pad + cell_h + label_pad + gap),
        (margin + cell_w + gap, margin + title_pad + cell_h + label_pad + gap),
    ]
    label_font = _serif_font(20)

    for src, label, (x, y) in zip(panels, labels, positions):
        if isinstance(src, (str, Path)):
            panel = Image.open(src).convert("RGBA")
        else:
            panel = src.convert("RGBA")
        # letterbox into the cell
        ratio = min(cell_w / panel.width, cell_h / panel.height)
        new_w = max(1, int(panel.width * ratio))
        new_h = max(1, int(panel.height * ratio))
        scaled = panel.resize((new_w, new_h), Image.LANCZOS)
        cell_canvas = Image.new("RGBA", (cell_w, cell_h), (*SCORE_BG_INNER, 0))
        cell_canvas.paste(scaled,
                          ((cell_w - new_w) // 2, (cell_h - new_h) // 2))
        canvas.alpha_composite(cell_canvas, dest=(x, y))
        # decorative cell frame
        cell_layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        cd = ImageDraw.Draw(cell_layer)
        cd.rectangle(
            [(x - 2, y - 2), (x + cell_w + 1, y + cell_h + 1)],
            outline=(*SCORE_FRAME, 230), width=2,
        )
        cd.rectangle(
            [(x - 6, y - 6), (x + cell_w + 5, y + cell_h + 5)],
            outline=(*SCORE_FRAME, 110), width=1,
        )
        canvas.alpha_composite(cell_layer)
        # Roman-numeral panel index in the upper-left corner of the cell
        cell_idx = positions.index((x, y)) + 1
        roman = ["I", "II", "III", "IV"][cell_idx - 1]
        _draw_text(canvas, roman, (x + 14, y + 10),
                   font=_serif_font(22),
                   color=(*SCORE_TEXT, 210), italic=True)
        # caption under the cell
        _draw_text(canvas, label,
                   (x + cell_w // 2, y + cell_h + 10),
                   font=label_font,
                   color=(*SCORE_TEXT, 220),
                   centered=True)

    # Bottom plate-number cartouche
    _draw_bottom_cartouche(canvas, plate_number,
                           total_w, total_h, margin)

    # Apply paper aging
    final = _apply_paper_aging(canvas)
    final.convert("RGB").save(output_path)
    return output_path


# -----------------------------------------------------------
# Helpers
# -----------------------------------------------------------


def _midnight_ground(w: int, h: int) -> Image.Image:
    """Create a subtle radial-gradient midnight ground."""
    img = _midnight_gradient_fill(
        Image.new("RGBA", (w, h), (0, 0, 0, 255)),
        SCORE_BG_INNER, SCORE_BG_OUTER)
    return img


def _midnight_gradient_fill(img: Image.Image, inner_rgb, outer_rgb) -> Image.Image:
    w, h = img.size
    cy, cx = h / 2.0, w / 2.0
    R = max(cy, cx) * 1.2
    yy, xx = np.mgrid[:h, :w]
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) / R
    rr = np.clip(rr, 0.0, 1.0)
    inner = np.array(inner_rgb, dtype=np.float32)
    outer = np.array(outer_rgb, dtype=np.float32)
    arr = inner * (1 - rr[..., None]) + outer * rr[..., None]
    rng = np.random.RandomState(13)
    grain = rng.normal(0, 4.0, (h, w, 3))
    arr = np.clip(arr + grain, 0, 255).astype(np.uint8)
    rgba = np.dstack([arr, np.full((h, w), 255, dtype=np.uint8)])
    return Image.fromarray(rgba, mode="RGBA")


def _draw_engraved_frame(img: Image.Image, pad: int):
    """Draw a triple-rule engraved border."""
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    w, h = img.size
    p = pad
    for stroke_w in (3, 1, 1):
        draw.rectangle([(p, p), (w - p, h - p)],
                       outline=(*SCORE_FRAME, 230), width=stroke_w)
        p += 8
    img.alpha_composite(layer)


def _draw_corner_ornaments(img: Image.Image, pad: int):
    """Decorative quarter-circle ornaments at each frame corner."""
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    w, h = img.size
    size = 64
    # corners: (cx, cy, start, end)
    corners = [
        (pad, pad, 0, 90),               # top-left
        (w - pad, pad, 90, 180),         # top-right
        (pad, h - pad, 270, 360),        # bottom-left
        (w - pad, h - pad, 180, 270),    # bottom-right
    ]
    for cx, cy, a0, a1 in corners:
        # nested arcs
        for r in (size, size - 12, size - 22):
            draw.arc([(cx - r, cy - r), (cx + r, cy + r)],
                     start=a0, end=a1,
                     fill=(*SCORE_FRAME, 200), width=1)
        # small filled diamond at the inner corner
        d = 6
        draw.polygon([(cx, cy - d), (cx + d, cy),
                      (cx, cy + d), (cx - d, cy)],
                     fill=(*SCORE_FRAME, 220))
        # small radiating sun-rays
        for k in range(5):
            ang = math.radians(a0 + (a1 - a0) * (k + 1) / 6)
            x_in = cx + (size - 30) * math.cos(ang)
            y_in = cy + (size - 30) * math.sin(ang)
            x_out = cx + (size - 6) * math.cos(ang)
            y_out = cy + (size - 6) * math.sin(ang)
            draw.line([(x_in, y_in), (x_out, y_out)],
                      fill=(*SCORE_FRAME, 160), width=1)
    img.alpha_composite(layer)


def _draw_title_cartouche(img: Image.Image, title: str, plate_number: str,
                           cy: int, width: int):
    """Decorative cartouche with scrollwork at top of plate."""
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    iw = img.size[0]
    cx = iw // 2
    h_box = 70
    box = [(cx - width // 2, cy - h_box // 2),
           (cx + width // 2, cy + h_box // 2)]

    # filled cartouche (translucent dark)
    draw.rounded_rectangle(box, radius=12,
                           fill=(*SCORE_BG_OUTER, 220),
                           outline=(*SCORE_FRAME, 230), width=2)
    # inner rule
    draw.rounded_rectangle(
        [(box[0][0] + 6, box[0][1] + 6),
         (box[1][0] - 6, box[1][1] - 6)],
        radius=8,
        outline=(*SCORE_FRAME, 140), width=1)

    # scrollwork at left and right
    sl, st = box[0]
    sr, sb = box[1]
    for sign, base_x in ((-1, sl), (1, sr)):
        for k in range(3):
            r = 18 - k * 4
            cx_s = base_x + sign * (12 + k * 6)
            cy_s = (st + sb) // 2
            draw.arc([(cx_s - r, cy_s - r), (cx_s + r, cy_s + r)],
                     start=0 if sign > 0 else 180,
                     end=180 if sign > 0 else 360,
                     fill=(*SCORE_FRAME, 200), width=1)

    img.alpha_composite(layer)

    # Title text
    title_font = _serif_font(28)
    plate_font = _serif_font(16)
    _draw_text(img, title.upper(), (cx, cy - 8), font=title_font,
               color=(*SCORE_TEXT, 240), centered=True)
    if plate_number:
        _draw_text(img, f"·  TABVLA  {plate_number}  ·",
                   (cx, cy + 18), font=plate_font,
                   color=(*SCORE_TEXT, 200), centered=True, italic=True)


def _draw_bottom_cartouche(img: Image.Image, plate_number: str,
                            total_w: int, total_h: int, margin: int):
    """Decorative bottom cartouche with Latin motto."""
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    w_box = int(total_w * 0.46)
    h_box = 56
    cx = total_w // 2
    cy = total_h - margin // 2 - 10
    box = [(cx - w_box // 2, cy - h_box // 2),
           (cx + w_box // 2, cy + h_box // 2)]
    draw.rounded_rectangle(box, radius=10,
                           fill=(*SCORE_BG_OUTER, 220),
                           outline=(*SCORE_FRAME, 220), width=2)
    draw.rounded_rectangle(
        [(box[0][0] + 5, box[0][1] + 5),
         (box[1][0] - 5, box[1][1] - 5)],
        radius=7,
        outline=(*SCORE_FRAME, 130), width=1)
    img.alpha_composite(layer)

    motto = "AVDITVS  IN  STELLAS  ·  STELLAE  IN  AVDITVM"
    font = _serif_font(15)
    _draw_text(img, motto, (cx, cy - 4), font=font,
               color=(*SCORE_TEXT, 220), centered=True, italic=True)


def _serif_font(size: int):
    """Try to load a serif TrueType; fall back to PIL default."""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSerif.ttf",
        "C:\\Windows\\Fonts\\Times.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def _draw_text(img: Image.Image, text: str, xy, font=None,
               color=(220, 200, 160, 220), centered: bool = False,
               italic: bool = False):
    """Robust text drawer that centres if requested."""
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    if font is None:
        font = _serif_font(18)
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


def _apply_paper_aging(img: Image.Image) -> Image.Image:
    """Subtle vignette + warm tint at the rim to evoke aged engraving."""
    w, h = img.size
    yy, xx = np.mgrid[:h, :w]
    cy, cx = h / 2.0, w / 2.0
    R = math.hypot(cx, cy)
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) / R
    rr = np.clip(rr, 0.0, 1.0)

    # darken vignette
    shadow_strength = (rr ** 2.4) * 0.30
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    arr = arr * (1.0 - shadow_strength[..., None])

    # warm tint at rim
    tint_strength = (rr ** 1.6) * 0.06
    warm = np.array([12, 4, -6], dtype=np.float32)  # +R, +G a tiny, -B
    arr = arr + warm * tint_strength[..., None] * 255 * 0.18

    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr).convert("RGBA")


def _save_with_aging(img: Image.Image, output_path: Path) -> Path:
    aged = _apply_paper_aging(img)
    aged.convert("RGB").save(output_path)
    return output_path
