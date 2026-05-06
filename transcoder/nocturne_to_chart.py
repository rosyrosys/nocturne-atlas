"""
nocturne_to_chart
=================

Transcode a nocturne audio recording into a planispheric star chart in
the engraved-atlas style of Bayer (1603), Hevelius (1690), and Bode
(1801).

The audio is decomposed into spectral peaks over time. Each peak is
placed on a circular planisphere, with:

  - elapsed time → angular position around the celestial pole
  - frequency → radial position (low = outer rim, high = the pole)
  - spectral magnitude → star apparent magnitude (visual size + halo)

Onsets and sustained peaks are rendered as bright stars; persistent
spectral lines connecting nearby peaks are rendered as constellation
linework in faint copper. An ornate engraved-style border, an ecliptic
arc, and a Latin-style cartouche are added to evoke the
pre-photographic atlas tradition.

Output: a PIL.Image rendered against a deep midnight ground.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from scipy import ndimage


# Palette: deep night ground, bone-white stars, dim copper lines
PALETTES = {
    "midnight":  {
        "ground":        (10, 16, 38),
        "ground_outer":  (4, 7, 18),
        "star_core":     (248, 240, 218),
        "star_halo":     (208, 196, 168),
        "constellation": (172, 132, 78),
        "ecliptic":      (140, 104, 56),
        "frame":         (96, 70, 36),
        "cartouche_ink": (228, 212, 178),
    },
    "indigo": {
        "ground":        (16, 22, 56),
        "ground_outer":  (6, 10, 28),
        "star_core":     (240, 232, 220),
        "star_halo":     (180, 170, 200),
        "constellation": (140, 124, 188),
        "ecliptic":      (110, 96, 156),
        "frame":         (80, 64, 124),
        "cartouche_ink": (220, 210, 200),
    },
    "iron": {
        "ground":        (22, 22, 24),
        "ground_outer":  (8, 8, 10),
        "star_core":     (240, 234, 220),
        "star_halo":     (200, 188, 168),
        "constellation": (160, 132, 88),
        "ecliptic":      (130, 100, 64),
        "frame":         (90, 72, 44),
        "cartouche_ink": (224, 212, 180),
    },
}


@dataclasses.dataclass
class ChartConfig:
    width: int = 2200
    height: int = 2200
    sr: int = 22050
    n_fft: int = 4096
    hop_length: int = 1024
    duration: Optional[float] = None
    palette: str = "midnight"
    n_peaks_per_frame: int = 6
    min_peak_db: float = -55.0       # peaks below this magnitude (dBFS) ignored
    star_size_max: int = 14
    halo_size_max: int = 64
    constellation_threshold: float = 0.66   # connect peaks above this magnitude
    constellation_max_distance: float = 0.10  # in normalised radius units
    show_ecliptic: bool = True
    cartouche_text_top: str = "Nocturne Atlas"
    cartouche_text_subtitle: str = "Plate I :: Audio Transcoded to Planisphere"
    cartouche_text_latin: str = "AVDITVS IN STELLAS · CONVERSVS"


def _radial_gradient_ground(cfg: ChartConfig, palette) -> Image.Image:
    """Create the dark planisphere ground with a soft radial gradient."""
    W, H = cfg.width, cfg.height
    cy, cx = H / 2.0, W / 2.0
    R = min(cy, cx)
    yy, xx = np.mgrid[:H, :W]
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) / R
    rr = np.clip(rr, 0.0, 1.4)
    inner = np.array(palette["ground"], dtype=np.float32)
    outer = np.array(palette["ground_outer"], dtype=np.float32)
    t = np.clip(rr, 0.0, 1.0)[..., None]
    arr = inner * (1 - t) + outer * t
    # Star-field micro-noise (faint background dust)
    rng = np.random.RandomState(11)
    dust = rng.normal(0, 4, (H, W, 3))
    arr = np.clip(arr + dust, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _polar_to_xy(angle_rad: float, radius_norm: float,
                 cx: float, cy: float, R: float) -> tuple[float, float]:
    """Convert (angle, normalised radius) to image (x, y).

    angle: 0 at top (12 o'clock), increasing clockwise
    radius_norm: 0 = centre, 1 = rim
    """
    theta = -np.pi / 2 + angle_rad
    x = cx + radius_norm * R * np.cos(theta)
    y = cy + radius_norm * R * np.sin(theta)
    return float(x), float(y)


def _detect_spectral_peaks(y: np.ndarray, sr: int, cfg: ChartConfig):
    """Return per-frame spectral peaks: list of (frame, freq_hz, mag_db, mag_norm).

    We use the magnitude spectrogram in dB scale; for each frame we keep
    the top-K peaks above min_peak_db.
    """
    S = np.abs(librosa.stft(y, n_fft=cfg.n_fft, hop_length=cfg.hop_length))
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    n_freqs, n_frames = S_db.shape
    freqs = librosa.fft_frequencies(sr=sr, n_fft=cfg.n_fft)

    peaks = []
    for f in range(n_frames):
        col = S_db[:, f]
        # Local maxima above threshold
        is_peak = (col[1:-1] > col[:-2]) & (col[1:-1] > col[2:]) & (col[1:-1] > cfg.min_peak_db)
        idx = np.where(is_peak)[0] + 1
        if idx.size == 0:
            continue
        # Keep top K by magnitude
        order = np.argsort(col[idx])[::-1][:cfg.n_peaks_per_frame]
        for k in idx[order]:
            mag_db = float(col[k])
            mag_norm = float((mag_db - cfg.min_peak_db) /
                             (col.max() - cfg.min_peak_db + 1e-9))
            mag_norm = max(0.0, min(1.0, mag_norm))
            peaks.append({
                "frame": int(f),
                "freq_hz": float(freqs[k]),
                "mag_db": mag_db,
                "mag_norm": mag_norm,
                "n_frames": n_frames,
            })
    return peaks, n_frames


def _peaks_to_polar(peaks, cfg: ChartConfig):
    """Map each peak to (angle_rad, radius_norm, mag_norm, freq)."""
    if not peaks:
        return []
    n_frames = peaks[0]["n_frames"]
    f_min, f_max = librosa.note_to_hz('A0'), librosa.note_to_hz('C8')

    points = []
    for p in peaks:
        # angle: time → 0..2pi (clockwise)
        angle = 2.0 * np.pi * (p["frame"] / max(1, n_frames - 1))

        # radius: log-frequency → 0..1, with low freq = outer rim
        f = max(f_min, min(f_max, p["freq_hz"]))
        f_log = (np.log(f) - np.log(f_min)) / (np.log(f_max) - np.log(f_min) + 1e-9)
        # f_log: 0 = low freq (we want outer), 1 = high freq (we want inner)
        radius = (1.0 - f_log) * 0.92 + 0.04  # leave a small margin near the pole

        points.append({
            "angle": float(angle),
            "radius": float(radius),
            "mag": float(p["mag_norm"]),
            "freq": float(p["freq_hz"]),
        })
    return points


def _draw_stars_batch(canvas: Image.Image, stars_xy_mag, palette, cfg: ChartConfig):
    """Batch-render all stars onto two layers (halo + core) with a single
    blur each. This is dramatically faster than per-star compositing.

    stars_xy_mag: iterable of (x, y, magnitude) tuples.
    """
    halo_layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    halo_draw = ImageDraw.Draw(halo_layer)
    core_layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    core_draw = ImageDraw.Draw(core_layer)
    spike_layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    spike_draw = ImageDraw.Draw(spike_layer)
    halo_color = palette["star_halo"]
    core_color = palette["star_core"]
    has_spikes = False

    for x, y, magnitude in stars_xy_mag:
        halo_radius = max(2.0, magnitude * cfg.halo_size_max)
        core_radius = max(0.6, magnitude * cfg.star_size_max * 0.32 + 0.6)
        halo_alpha = int(min(220, 70 + 150 * magnitude))
        halo_draw.ellipse([(x - halo_radius, y - halo_radius),
                            (x + halo_radius, y + halo_radius)],
                           fill=(*halo_color, halo_alpha))
        core_draw.ellipse([(x - core_radius, y - core_radius),
                            (x + core_radius, y + core_radius)],
                           fill=(*core_color, 255))
        if magnitude > 0.55:
            has_spikes = True
            spike_len = halo_radius * 1.4
            spike_alpha = int(min(220, 80 + 140 * (magnitude - 0.55) / 0.45))
            spike_draw.line([(x - spike_len, y), (x + spike_len, y)],
                            fill=(*core_color, spike_alpha), width=1)
            spike_draw.line([(x, y - spike_len), (x, y + spike_len)],
                            fill=(*core_color, spike_alpha), width=1)

    halo_layer = halo_layer.filter(ImageFilter.GaussianBlur(radius=8))
    canvas.alpha_composite(halo_layer)
    if has_spikes:
        spike_layer = spike_layer.filter(ImageFilter.GaussianBlur(radius=0.6))
        canvas.alpha_composite(spike_layer)
    canvas.alpha_composite(core_layer)


def _draw_star(canvas, x, y, magnitude, palette, cfg):
    """Single-star wrapper kept for API compatibility; batches via the
    fast path."""
    _draw_stars_batch(canvas, [(x, y, magnitude)], palette, cfg)


def _draw_constellations(canvas: Image.Image, points, palette, cfg: ChartConfig):
    """Draw faint copper lines connecting strong, nearby stars."""
    bright = [p for p in points if p["mag"] >= cfg.constellation_threshold]
    if len(bright) < 2:
        return
    # Sort by angle to encourage time-adjacent connections
    bright.sort(key=lambda p: p["angle"])
    cy, cx = canvas.size[1] / 2.0, canvas.size[0] / 2.0
    R = min(cx, cy) * 0.95

    layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    line_color = palette["constellation"]

    for i in range(len(bright) - 1):
        a, b = bright[i], bright[i + 1]
        ax, ay = _polar_to_xy(a["angle"], a["radius"], cx, cy, R)
        bx, by = _polar_to_xy(b["angle"], b["radius"], cx, cy, R)
        # angular distance in normalised units
        dtheta = abs(a["angle"] - b["angle"]) / (2 * np.pi)
        dr = abs(a["radius"] - b["radius"])
        dist = (dtheta ** 2 + dr ** 2) ** 0.5
        if dist > cfg.constellation_max_distance:
            continue
        alpha = int(120 + 100 * min(a["mag"], b["mag"]))
        draw.line([(ax, ay), (bx, by)], fill=(*line_color, alpha), width=1)

    layer = layer.filter(ImageFilter.GaussianBlur(0.5))
    canvas.alpha_composite(layer)


def _draw_frame_and_ecliptic(canvas: Image.Image, palette, cfg: ChartConfig):
    """Draw the planisphere rim, ecliptic arc, and engraved frame."""
    W, H = cfg.width, cfg.height
    cy, cx = H / 2.0, W / 2.0
    R = min(cx, cy) * 0.95
    layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    frame_color = palette["frame"]

    # Outer engraved frame (concentric rectangles)
    pad = 24
    for w in (3, 1, 1):
        draw.rectangle([(pad, pad), (W - pad, H - pad)],
                       outline=(*frame_color, 220), width=w)
        pad += 8

    # Planisphere outer circle
    draw.ellipse([(cx - R, cy - R), (cx + R, cy + R)],
                 outline=(*frame_color, 240), width=3)
    # inner concentric circles (latitude rings)
    for r_ratio in (0.25, 0.5, 0.75):
        rr = R * r_ratio
        draw.ellipse([(cx - rr, cy - rr), (cx + rr, cy + rr)],
                     outline=(*frame_color, 110), width=1)

    # Hour-angle radial ticks (24 divisions)
    for k in range(24):
        ang = 2 * np.pi * k / 24 - np.pi / 2
        x_outer = cx + R * np.cos(ang)
        y_outer = cy + R * np.sin(ang)
        x_inner = cx + R * 0.96 * np.cos(ang)
        y_inner = cy + R * 0.96 * np.sin(ang)
        draw.line([(x_inner, y_inner), (x_outer, y_outer)],
                  fill=(*frame_color, 200), width=2 if k % 6 == 0 else 1)

    # Ecliptic: an off-centre tilted circle
    if cfg.show_ecliptic:
        ecl_color = palette["ecliptic"]
        steps = 240
        prev = None
        tilt = np.deg2rad(23.4)
        for s in range(steps + 1):
            phi = 2 * np.pi * s / steps
            # parametric ellipse in chart coordinates
            x = cx + R * 0.78 * np.cos(phi)
            y = cy + R * 0.78 * np.sin(phi) * np.cos(tilt) - R * 0.18 * np.sin(tilt) * np.sin(phi)
            if prev is not None:
                draw.line([prev, (x, y)], fill=(*ecl_color, 170), width=1)
            prev = (x, y)

    canvas.alpha_composite(layer)


def _draw_cartouche(canvas: Image.Image, palette, cfg: ChartConfig):
    """Draw a small engraved-style cartouche with title text in the lower
    margin. Uses PIL's default font; for production substitute a serif
    font face via ImageFont.truetype()."""
    W, H = cfg.width, cfg.height
    cy, cx = H / 2.0, W / 2.0
    R = min(cx, cy) * 0.95

    layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    text_color = palette["cartouche_ink"]
    frame_color = palette["frame"]

    cw = int(W * 0.36)
    ch = int(H * 0.06)
    cx0 = (W - cw) // 2
    cy0 = int(H - 96 - ch)

    # Soft-edge cartouche with engraved border
    draw.rectangle([(cx0, cy0), (cx0 + cw, cy0 + ch)],
                   fill=(*palette["ground_outer"], 220),
                   outline=(*frame_color, 230), width=2)
    draw.rectangle([(cx0 + 6, cy0 + 6), (cx0 + cw - 6, cy0 + ch - 6)],
                   outline=(*frame_color, 160), width=1)

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    if font:
        for i, line in enumerate((cfg.cartouche_text_top,
                                   cfg.cartouche_text_subtitle,
                                   cfg.cartouche_text_latin)):
            bbox = draw.textbbox((0, 0), line, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            ty = cy0 + 8 + i * (th + 4)
            draw.text((cx0 + (cw - tw) // 2, ty), line,
                      fill=(*text_color, 235), font=font)

    canvas.alpha_composite(layer)


def nocturne_to_chart(audio_path: str | Path,
                      config: Optional[ChartConfig] = None) -> Image.Image:
    """Transcode an audio recording to a planispheric star chart."""
    if config is None:
        config = ChartConfig()
    palette = PALETTES.get(config.palette, PALETTES["midnight"])

    y, sr = librosa.load(audio_path, sr=config.sr,
                          duration=config.duration, mono=True)
    if y.size == 0:
        raise ValueError(f"Empty audio: {audio_path}")

    peaks, n_frames = _detect_spectral_peaks(y, sr, config)
    points = _peaks_to_polar(peaks, config)

    canvas = _radial_gradient_ground(config, palette).convert("RGBA")

    # Frame + ecliptic first so stars overlay them
    _draw_frame_and_ecliptic(canvas, palette, config)

    # Constellation lines underneath the stars
    _draw_constellations(canvas, points, palette, config)

    cy, cx = config.height / 2.0, config.width / 2.0
    R = min(cx, cy) * 0.95
    stars_to_draw = []
    for p in points:
        if p["mag"] < 0.18:
            continue
        x, y_ = _polar_to_xy(p["angle"], p["radius"], cx, cy, R)
        stars_to_draw.append((x, y_, p["mag"]))
    _draw_stars_batch(canvas, stars_to_draw, palette, config)

    _draw_cartouche(canvas, palette, config)

    return canvas.convert("RGB")