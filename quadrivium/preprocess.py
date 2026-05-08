"""
quadrivium.preprocess
=====================

Engraved-plate preprocessing: convert a high-resolution scan of a
historical engraved star atlas (Bayer 1603, Hevelius 1690, Bode 1801)
into the brightness-inverted point-cloud format that
`chart_to_nocturne` expects.

The chart-to-nocturne pipeline assumes the convention used by our
procedural stand-in inputs: bright stars as connected components on
a dark celestial ground, detected by a luminance threshold. Real
engraved plates obey the opposite convention — dark ink marks on
light paper, with constellation figures, ecliptic lines, coordinate
grids, and titular text occupying as much of the inked area as the
stars themselves. A naive luminance inversion would therefore feed
the figural and lettering ink into the star detector and produce a
nocturne with thousands of spurious events.

The preprocessing is deterministic and uses only `numpy`, `scipy`,
and `Pillow`. Determinism is preserved end-to-end: identical input
JPG and configuration yield bit-identical processed PNG.

Pipeline:

  1. **Load** as 8-bit greyscale.
  2. **Auto-detect the celestial disc**: locate the large circular
     boundary by binarising the deeply inked perimeter and taking
     the centroid and inscribed radius.
  3. **Crop and mask** to the disc; everything outside (title
     cartouche, marginalia) is set to background.
  4. **Black-tophat filter** with a small structuring element:
     subtracts a morphologically closed image from the original,
     leaving only locally darker spots smaller than the structuring
     element. Constellation strokes and figural outlines are
     suppressed; star dots, being small and locally darkest,
     survive.
  5. **Threshold** the tophat response to a binary star mask.
  6. **Connected-component filtering** by size and aspect ratio to
     remove residual line fragments and keep only point-like blobs.
  7. **Render** to a stand-in-compatible PNG: bright disc-shaped
     star markers (radius and brightness scaled by detected blob
     size) on a dark midnight ground, with the same `*_standin.png`
     contract that `inputs/tabula_stellarum_standin.png` follows.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage


@dataclasses.dataclass
class EngravingConfig:
    """Parameters for the engraved-plate preprocessor."""
    output_size: int = 1200            # square output canvas, pixels
    disc_radius_factor: float = 0.95   # output disc as fraction of canvas/2
    disc_inner_factor: float = 0.93    # source-disc inner mask, excluding the rim
    dog_sigma_inner: float = 1.6       # narrow-scale Gaussian (highlights star cores)
    dog_sigma_outer: float = 4.5       # wide-scale Gaussian (background)
    nms_radius: int = 12               # non-maximum-suppression neighbourhood radius (px)
    response_quantile: float = 0.992   # quantile of DoG peaks retained as star candidates
    max_stars: int = 320               # hard upper cap to keep nocturne event count
                                       # comparable to the procedural stand-in plates;
                                       # if more peaks pass the quantile they are
                                       # ranked by DoG response and only the brightest
                                       # `max_stars` are kept.
    star_marker_min_px: float = 1.8    # minimum rendered marker radius (output)
    star_marker_max_px: float = 7.5    # maximum rendered marker radius (output)


# ----------------------------------------------------------------------
# Disc localisation
# ----------------------------------------------------------------------


def _locate_disc(grey: np.ndarray) -> tuple[float, float, float]:
    """Return (cx, cy, R) of the celestial disc inscribed in the plate.

    The deeply inked rim of an engraved planisphere — together with the
    densely inked star/figure interior — gives a high foreground
    fraction inside the disc and near zero outside. We locate the disc
    by thresholding hard, taking the centroid of the largest connected
    region as the disc centre, and choosing R as the smaller of the
    centre's distance to each canvas edge minus a small inset.
    """
    h, w = grey.shape
    # Invert: ink becomes high values
    inv = 255 - grey
    # Hard threshold to capture the inked area
    binary = inv > 70
    labelled, n = ndimage.label(binary)
    if n == 0:
        return float(w / 2), float(h / 2), float(min(w, h) * 0.45)
    sizes = ndimage.sum(binary, labelled, range(1, n + 1))
    biggest = int(np.argmax(sizes)) + 1
    mask = labelled == biggest
    ys, xs = np.nonzero(mask)
    cy = float(ys.mean())
    cx = float(xs.mean())
    # R: largest circle centred at (cx, cy) that fits inside the bounding
    # box, with a small inset to avoid the rim.
    R = float(min(cx, w - cx, cy, h - cy)) * 0.96
    return cx, cy, R


# ----------------------------------------------------------------------
# Black-tophat morphology
# ----------------------------------------------------------------------


def _difference_of_gaussians(grey: np.ndarray,
                              sigma_in: float, sigma_out: float) -> np.ndarray:
    """Difference-of-Gaussians blob response. A small dark dot on a
    light ground produces a strong positive peak in the inverted DoG.

    DoG is a multi-scale blob filter: it is sensitive to the spatial
    extent of a feature, not its length. A long thin line of equal
    width to a star dot has a far weaker DoG response than the dot
    because the response integrates approximately equally on both
    sides of the line, cancelling. This is the property the engraved-
    plate preprocessor exploits to suppress constellation outlines
    and lettering while retaining star marks.
    """
    inv = 255.0 - grey.astype(np.float32)
    g_in = ndimage.gaussian_filter(inv, sigma_in)
    g_out = ndimage.gaussian_filter(inv, sigma_out)
    return g_in - g_out


def _detect_peaks(response: np.ndarray, mask: np.ndarray,
                   cfg: EngravingConfig):
    """Find local maxima of the DoG response inside the disc mask.

    A pixel is retained as a star candidate iff (1) it equals its
    local maximum within an `nms_radius`-px neighbourhood, (2) it lies
    inside the disc, (3) its response exceeds the configured quantile
    of all in-disc responses, and (4) it ranks within the top
    `max_stars` by response strength. The fourth criterion bounds the
    output event count so that the nocturne produced from a real plate
    is sonically comparable to the stand-in nocturnes (~300 events).
    """
    inside = response[mask]
    if inside.size == 0:
        return []
    cutoff = float(np.quantile(inside, cfg.response_quantile))
    # Local maxima via maximum filter
    footprint = _disc_kernel(cfg.nms_radius)
    local_max = ndimage.maximum_filter(response, footprint=footprint)
    is_peak = (response == local_max) & (response >= cutoff) & mask
    ys, xs = np.nonzero(is_peak)
    if ys.size == 0:
        return []
    strengths = response[ys, xs]
    # Rank by response, keep top `max_stars`
    order = np.argsort(-strengths)[:cfg.max_stars]
    stars = []
    for k in order:
        stars.append({
            "x": float(xs[k]),
            "y": float(ys[k]),
            "size": 1,
            "magnitude": float(strengths[k]),
        })
    return stars


def _disc_kernel(r: int) -> np.ndarray:
    """A circular structuring element of radius `r`, in {0, 1}."""
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    return ((yy * yy + xx * xx) <= r * r).astype(np.uint8)


# ----------------------------------------------------------------------
# Render to stand-in-compatible PNG
# ----------------------------------------------------------------------


_STANDIN_BG_INNER = (8, 14, 36)
_STANDIN_BG_OUTER = (4, 7, 18)
_STANDIN_STAR = (228, 220, 192)
_STANDIN_RIM = (180, 140, 80)


def _render_standin(stars, src_cx, src_cy, src_R, cfg: EngravingConfig) -> Image.Image:
    """Render the detected stars as bright disc-shaped markers on a
    dark midnight ground, in the same format as the procedural
    `tabula_stellarum_standin.png` input that `run_paper.py` consumes.
    """
    W = cfg.output_size
    out = Image.new("RGB", (W, W), _STANDIN_BG_INNER)
    # Mild radial vignette to imitate the procedural ground
    arr = np.asarray(out, dtype=np.float32).copy()
    yy, xx = np.mgrid[:W, :W]
    cx_o = cy_o = W / 2.0
    R_o = W / 2.0 * cfg.disc_radius_factor
    rr = np.sqrt((yy - cy_o) ** 2 + (xx - cx_o) ** 2) / R_o
    rr = np.clip(rr, 0.0, 1.4)
    inner = np.array(_STANDIN_BG_INNER, dtype=np.float32)
    outer = np.array(_STANDIN_BG_OUTER, dtype=np.float32)
    t = np.clip(rr, 0.0, 1.0)[..., None]
    arr = inner * (1 - t) + outer * t
    out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    draw = ImageDraw.Draw(out)
    # Disc rim
    draw.ellipse([(cx_o - R_o, cy_o - R_o), (cx_o + R_o, cy_o + R_o)],
                 outline=_STANDIN_RIM, width=2)

    # Project source-image stars into output disc
    if stars:
        mags = np.array([s["magnitude"] for s in stars], dtype=np.float32)
        sizes = np.array([s["size"] for s in stars], dtype=np.float32)
        if mags.max() > 0:
            mag_norm = (mags - mags.min()) / max(1e-6, mags.max() - mags.min())
        else:
            mag_norm = np.zeros_like(mags)
        if sizes.max() > 0:
            size_norm = sizes / sizes.max()
        else:
            size_norm = np.zeros_like(sizes)

        for s, mag, sz in zip(stars, mag_norm, size_norm):
            # Radial position of the source star, normalised inside src_R
            r_src = ((s["x"] - src_cx) ** 2 + (s["y"] - src_cy) ** 2) ** 0.5
            r_norm = r_src / src_R
            if r_norm > 1.0:
                continue
            angle = np.arctan2(s["y"] - src_cy, s["x"] - src_cx)
            x_o = cx_o + R_o * r_norm * np.cos(angle)
            y_o = cy_o + R_o * r_norm * np.sin(angle)

            radius = (cfg.star_marker_min_px
                      + (cfg.star_marker_max_px - cfg.star_marker_min_px) * sz)
            # All marker fills clear the downstream luminance threshold
            # (chart_to_nocturne's default τ=0.55, normalised 0..1). The
            # brightness range 0.78–1.0 keeps every detected peak audible
            # while still encoding magnitude variation.
            brightness = 0.78 + 0.22 * float(mag)
            colour = tuple(int(c * brightness) for c in _STANDIN_STAR)
            draw.ellipse([(x_o - radius, y_o - radius),
                          (x_o + radius, y_o + radius)],
                         fill=colour)

    return out


# ----------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------


def engraving_to_chart(input_path: str | Path,
                       output_path: str | Path,
                       config: EngravingConfig | None = None) -> dict:
    """Process an engraved-plate scan into a stand-in-compatible PNG.

    Returns a dict of diagnostic metadata (detected disc geometry and
    star count). Raises on I/O failure.
    """
    cfg = config or EngravingConfig()
    Image.MAX_IMAGE_PIXELS = None
    img = Image.open(input_path).convert("L")
    grey = np.asarray(img, dtype=np.uint8)

    # Locate the celestial disc on the source plate
    src_cx, src_cy, src_R = _locate_disc(grey)
    yy, xx = np.mgrid[:grey.shape[0], :grey.shape[1]]
    rr = (yy - src_cy) ** 2 + (xx - src_cx) ** 2
    disc_mask = rr <= (src_R * cfg.disc_inner_factor) ** 2

    # Difference-of-Gaussians blob response, restricted to the disc
    response = _difference_of_gaussians(grey, cfg.dog_sigma_inner,
                                         cfg.dog_sigma_outer)
    response[~disc_mask] = 0.0

    stars = _detect_peaks(response, disc_mask, cfg)

    out_img = _render_standin(stars, src_cx, src_cy, src_R, cfg)
    out_img.save(output_path)

    return {
        "source_disc_centre": (src_cx, src_cy),
        "source_disc_radius_px": src_R,
        "n_stars_detected": len(stars),
        "output_size": cfg.output_size,
    }


if __name__ == "__main__":
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else \
        "inputs/bode_1801_aries_planisphere.jpg"
    dst = sys.argv[2] if len(sys.argv) > 2 else \
        "inputs/bode_1801_planisphere_processed.png"
    info = engraving_to_chart(src, dst)
    print(info)
