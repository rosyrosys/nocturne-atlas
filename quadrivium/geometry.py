"""
quadrivium.geometry
===================

Numbers in space: planispheric projection and the polar-coordinate
system used by the historical-atlas tradition.

The planispheric chart projects the celestial sphere onto a flat
disc through stereographic projection from one celestial pole. The
hour-angle around the pole becomes a circular angular coordinate;
declination from the pole becomes a radial coordinate. This module
provides utilities for the analogous mapping on which our nocturne-to-
chart pipeline operates: time becomes angle (a single revolution
spans the duration of the recording), and log-frequency becomes
radius from outer rim (low frequencies) to centre (high frequencies).

The log-frequency-to-radius mapping is itself a Pythagorean transposition.
At every doubling of frequency — every audible octave, ratio 2:1 — the
radius halves, recapitulating the 1:2 octave ratio across the visible
disc (Park 2025, 6, 11). This is not a metaphor: the visible distance
from rim to pole is the literal numerical mediation of pitch.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np


# ----------------------------------------------------------------------
# Polar / cartesian conversion
# ----------------------------------------------------------------------


def polar_to_xy(angle_rad: float, radius_norm: float,
                 cx: float, cy: float, R: float) -> Tuple[float, float]:
    """Convert (angle, normalised radius) to image (x, y).

    Convention: angle = 0 at the top (12 o'clock position), increasing
    clockwise. radius_norm = 0 at the centre (celestial pole),
    radius_norm = 1 at the outer rim. R is the effective radius of
    the planisphere in image pixels.

    Returns (x, y) image coordinates.
    """
    theta = -math.pi / 2 + angle_rad
    x = cx + radius_norm * R * math.cos(theta)
    y = cy + radius_norm * R * math.sin(theta)
    return float(x), float(y)


def time_to_angle(elapsed_seconds: float, total_seconds: float) -> float:
    """Map elapsed time within a recording to angular position.

    Returns an angle in radians from 0 to 2π for elapsed_seconds in
    [0, total_seconds].
    """
    if total_seconds <= 0:
        return 0.0
    return 2 * math.pi * (elapsed_seconds / total_seconds)


def log_frequency_to_radius(freq_hz: float,
                              f_min: float = 27.5,    # A0
                              f_max: float = 4186.0,  # C8
                              inner_margin: float = 0.04,
                              outer_margin: float = 0.04) -> float:
    """Map a frequency to a radial position on the planisphere.

    Implements the Pythagorean ratio mapping: at every doubling of
    frequency (octave, ratio 2:1), the radial position halves. The
    output is normalised to [outer_margin, 1 - inner_margin] so a
    small ring at the rim and at the pole remain free of marks (a
    cartographic convention preserved from the engraved-atlas
    tradition).

    Parameters
    ----------
    freq_hz : float
        Frequency in Hz.
    f_min, f_max : float
        Frequency range covered by the radial axis. Defaults span
        the chromatic gamut of the modern piano.
    inner_margin : float
        Fraction of the radius left empty near the celestial pole.
    outer_margin : float
        Fraction of the radius left empty near the outer rim.

    Returns
    -------
    radius_norm : float in [outer_margin, 1 - inner_margin]
        Normalised radial position, where 0 = pole and 1 = rim.
    """
    f = max(f_min, min(f_max, freq_hz))
    log_f = math.log(f)
    log_min = math.log(f_min)
    log_max = math.log(f_max)
    f_log = (log_f - log_min) / (log_max - log_min + 1e-9)
    # f_log: 0 = low freq (we want outer), 1 = high freq (we want inner)
    # An octave doubling moves log_f by log(2); accordingly the radius
    # decreases by a constant amount (log(2) / (log_max - log_min)) of the
    # full radial range. This is the literal Pythagorean 1:2 transposition.
    available = 1.0 - inner_margin - outer_margin
    radius = (1.0 - f_log) * available + outer_margin
    return float(radius)


def radius_to_log_frequency(radius_norm: float,
                              f_min: float = 27.5,
                              f_max: float = 4186.0,
                              inner_margin: float = 0.04,
                              outer_margin: float = 0.04) -> float:
    """Inverse of `log_frequency_to_radius`: recover frequency from radius."""
    available = 1.0 - inner_margin - outer_margin
    f_log = 1.0 - (radius_norm - outer_margin) / max(available, 1e-9)
    f_log = max(0.0, min(1.0, f_log))
    log_min = math.log(f_min)
    log_max = math.log(f_max)
    return float(math.exp(log_min + f_log * (log_max - log_min)))


def planisphere_extent(width: int, height: int,
                        rim_factor: float = 0.95) -> Tuple[float, float, float]:
    """Return (centre_x, centre_y, effective_radius_pixels) for a chart
    rendered into an image of given width x height. The effective radius
    is shrunk by `rim_factor` to leave a cartouche margin."""
    cx = width / 2.0
    cy = height / 2.0
    R = min(cx, cy) * rim_factor
    return float(cx), float(cy), float(R)


# ----------------------------------------------------------------------
# Anamorphic disc-to-linear projection (for chart-to-nocturne)
# ----------------------------------------------------------------------


def disc_to_linear(image_array: np.ndarray,
                    n_radial: int,
                    n_angular: int,
                    r_inner: float = 0.0,
                    r_outer: float = 1.0,
                    centre_xy: Tuple[float, float] | None = None,
                    radius: float | None = None) -> np.ndarray:
    """Anamorphic projection: convert a circular disc image to a
    rectangular array indexed by (radial_index, angular_index).

    This operation is the inverse of polar_to_xy. It is included
    because some chart-to-nocturne approaches scan the chart radially
    rather than column-by-column; both modes are supported by the
    music module.
    """
    h, w = image_array.shape[:2]
    if centre_xy is None:
        cx, cy = w / 2.0, h / 2.0
    else:
        cx, cy = centre_xy
    if radius is None:
        radius = min(cx, cy)

    radii = np.linspace(r_inner * radius, r_outer * radius, n_radial)
    angles = np.linspace(0.0, 2 * np.pi, n_angular, endpoint=False)

    out_shape = (n_radial, n_angular)
    if image_array.ndim == 3:
        out_shape = (n_radial, n_angular, image_array.shape[2])
    out = np.zeros(out_shape, dtype=image_array.dtype)

    for j, theta in enumerate(angles):
        ys = cy + radii * np.sin(theta)
        xs = cx + radii * np.cos(theta)
        ys = np.clip(ys.astype(np.int32), 0, h - 1)
        xs = np.clip(xs.astype(np.int32), 0, w - 1)
        out[:, j] = image_array[ys, xs]
    return out
