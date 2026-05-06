"""
quadrivium.astronomy
====================

Numbers in time and space: the nocturne-to-chart pipeline.

The audio recording is decomposed into a magnitude spectrogram. For
each frame we retain the top-K spectral peaks above a magnitude
floor; each peak is placed on a planispheric chart whose angular
coordinate maps to elapsed time and whose radial coordinate maps to
log-frequency, with the literal Pythagorean 1:2 ratio holding at every
audible octave.

A diagnostic function `detect_concentric_rings` identifies the
characteristic Music-of-the-Spheres artifact: harmonically periodic
input audio produces concentric rings on the chart at radii that
correspond to the harmonic period. The function returns a list of
detected ring radii together with their angular peak counts, providing
a quantitative measure of cosmic-harmonic structure latent in the
audio.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path
from typing import List, Optional, Tuple

import librosa
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .geometry import (
    polar_to_xy, time_to_angle, log_frequency_to_radius,
    planisphere_extent,
)


# ----------------------------------------------------------------------
# Palettes (engraved-atlas register; the parchment register is in
# quadrivium.parchment)
# ----------------------------------------------------------------------

PALETTES = {
    "midnight": {
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


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------


@dataclasses.dataclass
class ChartConfig:
    """Configuration for the nocturne-to-chart transcoder."""
    width: int = 2200
    height: int = 2200
    sr: int = 22050
    n_fft: int = 4096
    hop_length: int = 1024
    duration: Optional[float] = None
    palette: str = "midnight"
    n_peaks_per_frame: int = 6
    min_peak_db: float = -55.0
    star_size_max: int = 14
    halo_size_max: int = 64
    constellation_threshold: float = 0.66
    constellation_max_distance: float = 0.10
    show_ecliptic: bool = True
    cartouche_text_top: str = "Nocturne Atlas"
    cartouche_text_subtitle: str = "Audio Transcoded to Planisphere"
    cartouche_text_latin: str = "AVDITVS IN STELLAS · CONVERSVS"


# ----------------------------------------------------------------------
# Spectral analysis
# ----------------------------------------------------------------------


def _detect_spectral_peaks(y: np.ndarray, sr: int, cfg: ChartConfig):
    S = np.abs(librosa.stft(y, n_fft=cfg.n_fft, hop_length=cfg.hop_length))
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    n_freqs, n_frames = S_db.shape
    freqs = librosa.fft_frequencies(sr=sr, n_fft=cfg.n_fft)

    peaks = []
    for f in range(n_frames):
        col = S_db[:, f]
        is_peak = (
            (col[1:-1] > col[:-2])
            & (col[1:-1] > col[2:])
            & (col[1:-1] > cfg.min_peak_db)
        )
        idx = np.where(is_peak)[0] + 1
        if idx.size == 0:
            continue
        order = np.argsort(col[idx])[::-1][:cfg.n_peaks_per_frame]
        for k in idx[order]:
            mag_db = float(col[k])
            mag_norm = float(
                (mag_db - cfg.min_peak_db) /
                (col.max() - cfg.min_peak_db + 1e-9)
            )
            mag_norm = max(0.0, min(1.0, mag_norm))
            peaks.append({
                "frame": int(f),
                "freq_hz": float(freqs[k]),
                "mag_db": mag_db,
                "mag_norm": mag_norm,
                "n_frames": n_frames,
            })
    return peaks, n_frames


def _peaks_to_polar(peaks, n_frames: int):
    """Map each spectral peak to (angle, radius, magnitude, freq).

    The radial mapping uses the Pythagorean log-frequency rule from
    quadrivium.geometry: every doubling of frequency halves the
    radius, recapitulating the 1:2 octave ratio across the disc.
    """
    points = []
    f_min = librosa.note_to_hz("A0")
    f_max = librosa.note_to_hz("C8")
    for p in peaks:
        angle = time_to_angle(p["frame"], max(1, n_frames - 1))
        radius = log_frequency_to_radius(p["freq_hz"],
                                          f_min=f_min, f_max=f_max,
                                          inner_margin=0.04, outer_margin=0.04)
        points.append({
            "angle": float(angle),
            "radius": float(radius),
            "mag": float(p["mag_norm"]),
            "freq": float(p["freq_hz"]),
        })
    return points


# ----------------------------------------------------------------------
# Concentric-ring detection: Music-of-the-Spheres diagnostic
# ----------------------------------------------------------------------


def detect_concentric_rings(points,
                              n_radial_bins: int = 64,
                              min_density: float = 0.12) -> List[dict]:
    """Diagnose the concentric-ring artifact in audio-derived points.

    Bins points by radius; rings are radial bins whose density of high-
    magnitude peaks substantially exceeds the global mean. Returns a
    list of detected rings, each as a dict with keys:

        radius   : centre radius of the ring (normalised, 0..1)
        density  : fractional count of peaks at that radius
        mag_mean : mean magnitude of peaks at that radius

    The function is intended as a quantitative measure of the
    Music-of-the-Spheres pattern: harmonic periodicity in the input
    audio appears as a small number of high-density rings, where
    pure noise input would produce a flat radial distribution.
    """
    if not points:
        return []
    radii = np.array([p["radius"] for p in points])
    mags = np.array([p["mag"] for p in points])
    bins = np.linspace(0.04, 0.96, n_radial_bins + 1)
    density = np.zeros(n_radial_bins)
    mag_mean = np.zeros(n_radial_bins)
    for i in range(n_radial_bins):
        mask = (radii >= bins[i]) & (radii < bins[i + 1])
        if mask.sum() == 0:
            continue
        density[i] = mask.sum() / len(points)
        mag_mean[i] = mags[mask].mean()
    peaks_idx = []
    for i in range(1, n_radial_bins - 1):
        if density[i] > density[i - 1] and density[i] > density[i + 1]:
            if density[i] > min_density:
                peaks_idx.append(i)
    return [
        {
            "radius": float((bins[i] + bins[i + 1]) / 2),
            "density": float(density[i]),
            "mag_mean": float(mag_mean[i]),
        }
        for i in peaks_idx
    ]


# ----------------------------------------------------------------------
# Rendering primitives
# ----------------------------------------------------------------------


def _radial_gradient_ground(cfg: ChartConfig, palette) -> Image.Image:
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
    rng = np.random.RandomState(11)
    dust = rng.normal(0, 4, (H, W, 3))
    arr = np.clip(arr + dust, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _draw_stars_batch(canvas: Image.Image, stars_xy_mag, palette, cfg: ChartConfig):
    """Render stars in the engraved-atlas style: a small dense core,
    a tight halo, and four-point diffraction spikes whose length and
    opacity scale with the star's magnitude.

    The previous version made the halo radius up to `halo_size_max=64`
    and softened it with `GaussianBlur(8)`, which was so soft that
    overlapping stars merged into cloud-like blobs. The current
    settings give each star a clearly individuated point of light
    while preserving the warm halo. Diffraction spikes are now drawn
    for every star (not only the brightest), so the engraved-plate
    register is consistent across the field.
    """
    halo_layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    halo_draw = ImageDraw.Draw(halo_layer)
    core_layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    core_draw = ImageDraw.Draw(core_layer)
    spike_layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    spike_draw = ImageDraw.Draw(spike_layer)
    halo_color = palette["star_halo"]
    core_color = palette["star_core"]
    for x, y, magnitude in stars_xy_mag:
        # Tighter halo than before: scale the cap by ~0.55 so even the
        # brightest stars do not bleed into their neighbours.
        halo_radius = max(2.0, magnitude * cfg.halo_size_max * 0.55)
        core_radius = max(0.7, magnitude * cfg.star_size_max * 0.34 + 0.7)
        halo_alpha = int(min(190, 50 + 130 * magnitude))
        halo_draw.ellipse([(x - halo_radius, y - halo_radius),
                            (x + halo_radius, y + halo_radius)],
                           fill=(*halo_color, halo_alpha))
        core_draw.ellipse([(x - core_radius, y - core_radius),
                            (x + core_radius, y + core_radius)],
                           fill=(*core_color, 255))
        # Four-point diffraction spikes for every star, length and alpha
        # scaled by magnitude so dim stars get short whisper spikes and
        # bright stars get long luminous ones.
        spike_len = max(2.0, halo_radius * 1.6 + 1.2)
        spike_alpha = int(min(220, 60 + 180 * magnitude))
        spike_draw.line([(x - spike_len, y), (x + spike_len, y)],
                         fill=(*core_color, spike_alpha), width=1)
        spike_draw.line([(x, y - spike_len), (x, y + spike_len)],
                         fill=(*core_color, spike_alpha), width=1)
    halo_layer = halo_layer.filter(ImageFilter.GaussianBlur(radius=3.5))
    canvas.alpha_composite(halo_layer)
    spike_layer = spike_layer.filter(ImageFilter.GaussianBlur(radius=0.5))
    canvas.alpha_composite(spike_layer)
    canvas.alpha_composite(core_layer)


def _draw_constellations(canvas, points, palette, cfg):
    bright = [p for p in points if p["mag"] >= cfg.constellation_threshold]
    if len(bright) < 2:
        return
    bright.sort(key=lambda p: p["angle"])
    cx, cy, R = planisphere_extent(canvas.size[0], canvas.size[1])
    layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    line_color = palette["constellation"]
    for i in range(len(bright) - 1):
        a, b = bright[i], bright[i + 1]
        ax, ay = polar_to_xy(a["angle"], a["radius"], cx, cy, R)
        bx, by = polar_to_xy(b["angle"], b["radius"], cx, cy, R)
        dtheta = abs(a["angle"] - b["angle"]) / (2 * math.pi)
        dr = abs(a["radius"] - b["radius"])
        dist = (dtheta ** 2 + dr ** 2) ** 0.5
        if dist > cfg.constellation_max_distance:
            continue
        alpha = int(120 + 100 * min(a["mag"], b["mag"]))
        draw.line([(ax, ay), (bx, by)], fill=(*line_color, alpha), width=1)
    layer = layer.filter(ImageFilter.GaussianBlur(0.5))
    canvas.alpha_composite(layer)


def _draw_frame_and_ecliptic(canvas, palette, cfg):
    W, H = cfg.width, cfg.height
    cx, cy, R = planisphere_extent(W, H)
    layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    frame_color = palette["frame"]
    pad = 24
    for w in (3, 1, 1):
        draw.rectangle([(pad, pad), (W - pad, H - pad)],
                       outline=(*frame_color, 220), width=w)
        pad += 8
    draw.ellipse([(cx - R, cy - R), (cx + R, cy + R)],
                 outline=(*frame_color, 240), width=3)
    for r_ratio in (0.25, 0.5, 0.75):
        rr = R * r_ratio
        draw.ellipse([(cx - rr, cy - rr), (cx + rr, cy + rr)],
                     outline=(*frame_color, 110), width=1)
    for k in range(24):
        ang = 2 * math.pi * k / 24 - math.pi / 2
        x_outer = cx + R * math.cos(ang)
        y_outer = cy + R * math.sin(ang)
        x_inner = cx + R * 0.96 * math.cos(ang)
        y_inner = cy + R * 0.96 * math.sin(ang)
        draw.line([(x_inner, y_inner), (x_outer, y_outer)],
                  fill=(*frame_color, 200),
                  width=2 if k % 6 == 0 else 1)
    if cfg.show_ecliptic:
        ecl_color = palette["ecliptic"]
        steps = 240
        prev = None
        tilt = math.radians(23.4)
        for s in range(steps + 1):
            phi = 2 * math.pi * s / steps
            x = cx + R * 0.78 * math.cos(phi)
            y = cy + R * 0.78 * math.sin(phi) * math.cos(tilt) \
                - R * 0.18 * math.sin(tilt) * math.sin(phi)
            if prev is not None:
                draw.line([prev, (x, y)], fill=(*ecl_color, 170), width=1)
            prev = (x, y)
    canvas.alpha_composite(layer)


_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSerif.ttf",
    "/usr/share/fonts/dejavu/DejaVuSerif.ttf",
    "C:\\Windows\\Fonts\\Times.ttf",
    "C:\\Windows\\Fonts\\times.ttf",
    "C:\\Windows\\Fonts\\Georgia.ttf",
    "/System/Library/Fonts/Times.ttc",
)


def _serif_font(size: int):
    for p in _FONT_CANDIDATES:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return ImageFont.load_default()


def _draw_cartouche(canvas, palette, cfg):
    """Render the engraved cartouche at the foot of the planisphere.

    Three-line layout (title / subtitle / Latin motto) on a recessed
    plate framed in the chart's frame ink. Previously fell back to
    PIL's pixel default font, which read as amateur next to the
    serif typography of the surrounding plate. Now uses the same
    serif fallback chain as the paper-register renderer, so the
    type matches both the title cartouche and the panel labels.
    """
    W, H = cfg.width, cfg.height
    layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    text_color = palette["cartouche_ink"]
    frame_color = palette["frame"]
    cw = int(W * 0.42)
    ch = int(H * 0.085)
    cx0 = (W - cw) // 2
    cy0 = int(H - 96 - ch)
    draw.rectangle([(cx0, cy0), (cx0 + cw, cy0 + ch)],
                   fill=(*palette["ground_outer"], 230),
                   outline=(*frame_color, 240), width=2)
    draw.rectangle([(cx0 + 6, cy0 + 6), (cx0 + cw - 6, cy0 + ch - 6)],
                   outline=(*frame_color, 170), width=1)

    title_font = _serif_font(max(18, int(ch * 0.34)))
    sub_font = _serif_font(max(13, int(ch * 0.22)))
    latin_font = _serif_font(max(13, int(ch * 0.20)))

    fonts = (title_font, sub_font, latin_font)
    lines = (cfg.cartouche_text_top,
             cfg.cartouche_text_subtitle,
             cfg.cartouche_text_latin)
    # Measure to centre vertically inside the plate
    heights = []
    for line, font in zip(lines, fonts):
        bbox = draw.textbbox((0, 0), line, font=font)
        heights.append(bbox[3] - bbox[1])
    gap = max(2, int(ch * 0.06))
    total_h = sum(heights) + gap * (len(lines) - 1)
    y_cursor = cy0 + (ch - total_h) // 2 - 2
    for line, font, hh in zip(lines, fonts, heights):
        bbox = draw.textbbox((0, 0), line, font=font)
        tw = bbox[2] - bbox[0]
        draw.text((cx0 + (cw - tw) // 2, y_cursor), line,
                  fill=(*text_color, 240), font=font)
        y_cursor += hh + gap
    canvas.alpha_composite(layer)


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------


def nocturne_to_chart(audio_path: str | Path,
                       config: Optional[ChartConfig] = None) -> Image.Image:
    """Transcode an audio recording to a planispheric star chart in
    the engraved-atlas register (midnight ground).

    Returns a PIL.Image. For the parchment register, post-process
    the result through quadrivium.parchment.parchment_chart_overlay.
    """
    cfg = config or ChartConfig()
    palette = PALETTES.get(cfg.palette, PALETTES["midnight"])

    y, sr = librosa.load(audio_path, sr=cfg.sr,
                          duration=cfg.duration, mono=True)
    if y.size == 0:
        raise ValueError(f"Empty audio: {audio_path}")

    peaks, n_frames = _detect_spectral_peaks(y, sr, cfg)
    points = _peaks_to_polar(peaks, n_frames)

    canvas = _radial_gradient_ground(cfg, palette).convert("RGBA")
    _draw_frame_and_ecliptic(canvas, palette, cfg)
    _draw_constellations(canvas, points, palette, cfg)

    cx, cy, R = planisphere_extent(cfg.width, cfg.height)
    stars_to_draw = []
    for p in points:
        if p["mag"] < 0.18:
            continue
        x, y_ = polar_to_xy(p["angle"], p["radius"], cx, cy, R)
        stars_to_draw.append((x, y_, p["mag"]))
    _draw_stars_batch(canvas, stars_to_draw, palette, cfg)

    # The cartouche routine `_draw_cartouche` is intentionally NOT
    # called here: the chart is meant to read as a purely graphic
    # engraved plate, with no alphabet or descriptive captions inside
    # the disc. The function is left defined for any consumer that
    # might want it.

    return canvas.convert("RGB")
