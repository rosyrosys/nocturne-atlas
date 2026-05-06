"""
synthesize_demo_inputs.py
=========================

Generate two stand-in input materials for the Nocturne Atlas:

  1. A procedural pre-photographic-style star chart (engraved appearance,
     midnight ground, scattered bright stars with a rough constellation
     suggestion).
  2. A short procedural nocturne in C# minor (slow harmonic ground,
     sparse upper-register melody, sustain pedal effect).

These are stand-ins; replace with high-resolution scans of historical
plates (Bayer's Uranometria, Bode's Uranographia, Hevelius's Firmamentum)
and a public-domain recording of, e.g., Chopin's Nocturne Op. 27 No. 1.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw, ImageFilter


ROOT = Path(__file__).resolve().parent
INPUTS = ROOT / "inputs"
INPUTS.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# 1. Procedural star chart (Bayer-Uranometria stand-in)
# ---------------------------------------------------------------------


def make_star_chart(path: Path, size=(2200, 2200), seed=23) -> Path:
    """Procedural planispheric chart with a believable star distribution.

    Bright pixels = stars. Dark midnight ground. Rough constellation
    linework in dim copper. Frame and concentric rings for cartographic
    feel."""
    W, H = size
    cy, cx = H / 2.0, W / 2.0
    R = min(cy, cx) * 0.95

    rng = np.random.RandomState(seed)

    # Midnight ground with vignette
    yy, xx = np.mgrid[:H, :W]
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) / R
    inner = np.array([10, 16, 38], dtype=np.float32)
    outer = np.array([4, 7, 18], dtype=np.float32)
    t = np.clip(rr, 0.0, 1.0)[..., None]
    arr = inner * (1 - t) + outer * t

    # Background stardust (faint)
    dust = rng.normal(0, 5, (H, W, 3))
    arr = np.clip(arr + dust, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr).convert("RGBA")
    draw = ImageDraw.Draw(img)

    # Generate stars: random angle, biased radial distribution toward rim
    n_stars = 240
    star_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    sl_draw = ImageDraw.Draw(star_layer)
    bright_cluster_centres = []
    for _ in range(n_stars):
        # planar uniform-density requires sqrt(u) for radius
        r_norm = np.sqrt(rng.uniform(0.02, 0.95))
        ang = rng.uniform(0, 2 * np.pi)
        x = cx + r_norm * R * np.cos(ang)
        y = cy + r_norm * R * np.sin(ang)
        # magnitude distribution: most faint, few very bright
        m = rng.beta(1.6, 5.5)
        halo_radius = 2 + m * 28
        core_radius = 0.6 + m * 4
        halo_alpha = int(60 + 160 * m)
        sl_draw.ellipse([(x - halo_radius, y - halo_radius),
                         (x + halo_radius, y + halo_radius)],
                        fill=(208, 196, 168, halo_alpha))
        if m > 0.7:
            bright_cluster_centres.append((x, y, m))
    star_layer = star_layer.filter(ImageFilter.GaussianBlur(2.5))
    img.alpha_composite(star_layer)

    # Add bright cores on top of halos
    core_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    cl_draw = ImageDraw.Draw(core_layer)
    rng2 = np.random.RandomState(seed + 1)
    for _ in range(n_stars):
        r_norm = np.sqrt(rng2.uniform(0.02, 0.95))
        ang = rng2.uniform(0, 2 * np.pi)
        x = cx + r_norm * R * np.cos(ang)
        y = cy + r_norm * R * np.sin(ang)
        m = rng2.beta(1.6, 5.5)
        cr = 0.6 + m * 3.2
        cl_draw.ellipse([(x - cr, y - cr), (x + cr, y + cr)],
                        fill=(248, 240, 218, int(220 + 35 * m)))
        if m > 0.7:
            spike = cr * 5
            cl_draw.line([(x - spike, y), (x + spike, y)],
                         fill=(248, 240, 218, 200), width=1)
            cl_draw.line([(x, y - spike), (x, y + spike)],
                         fill=(248, 240, 218, 200), width=1)
    img.alpha_composite(core_layer)

    # Constellation lines: connect 12 random pairs of bright stars that are nearby
    if bright_cluster_centres:
        line_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        ll_draw = ImageDraw.Draw(line_layer)
        # connect nearest neighbours
        coords = np.array([(x, y) for x, y, _ in bright_cluster_centres])
        used = set()
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                d = np.hypot(*(coords[i] - coords[j]))
                if d < R * 0.18 and len(used) < 36 and (i, j) not in used:
                    ll_draw.line([tuple(coords[i]), tuple(coords[j])],
                                 fill=(172, 132, 78, 140), width=1)
                    used.add((i, j))
        line_layer = line_layer.filter(ImageFilter.GaussianBlur(0.5))
        img.alpha_composite(line_layer)

    # Frame: outer rectangle + planisphere circle + radial ticks
    frame_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    fl_draw = ImageDraw.Draw(frame_layer)
    pad = 24
    for w in (3, 1, 1):
        fl_draw.rectangle([(pad, pad), (W - pad, H - pad)],
                          outline=(140, 110, 60, 220), width=w)
        pad += 8
    fl_draw.ellipse([(cx - R, cy - R), (cx + R, cy + R)],
                    outline=(140, 110, 60, 240), width=3)
    for ratio in (0.25, 0.5, 0.75):
        rr_in = R * ratio
        fl_draw.ellipse([(cx - rr_in, cy - rr_in), (cx + rr_in, cy + rr_in)],
                        outline=(140, 110, 60, 110), width=1)
    for k in range(24):
        ang = 2 * np.pi * k / 24 - np.pi / 2
        x_outer = cx + R * np.cos(ang)
        y_outer = cy + R * np.sin(ang)
        x_inner = cx + R * 0.96 * np.cos(ang)
        y_inner = cy + R * 0.96 * np.sin(ang)
        fl_draw.line([(x_inner, y_inner), (x_outer, y_outer)],
                     fill=(140, 110, 60, 200), width=2 if k % 6 == 0 else 1)
    img.alpha_composite(frame_layer)

    # Cartouche
    cart_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    cd = ImageDraw.Draw(cart_layer)
    cw = int(W * 0.32)
    ch = int(H * 0.05)
    cx0 = (W - cw) // 2
    cy0 = int(H - 96 - ch)
    cd.rectangle([(cx0, cy0), (cx0 + cw, cy0 + ch)],
                 fill=(4, 7, 18, 220), outline=(140, 110, 60, 230), width=2)
    cd.rectangle([(cx0 + 6, cy0 + 6), (cx0 + cw - 6, cy0 + ch - 6)],
                 outline=(140, 110, 60, 160), width=1)
    try:
        from PIL import ImageFont
        font = ImageFont.load_default()
        for i, line in enumerate(("Tabula Stellarum :: Plate I",
                                   "Hemisphaerium Borealis (placeholder)",
                                   "AD VSVM ATLANTIS NOCTVRNI")):
            bbox = cd.textbbox((0, 0), line, font=font)
            tw = bbox[2] - bbox[0]
            th_ = bbox[3] - bbox[1]
            cd.text((cx0 + (cw - tw) // 2, cy0 + 8 + i * (th_ + 4)),
                    line, fill=(228, 212, 178, 235), font=font)
    except Exception:
        pass
    img.alpha_composite(cart_layer)

    img.convert("RGB").save(path)
    return path


# ---------------------------------------------------------------------
# 2. Procedural nocturne audio (C# minor)
# ---------------------------------------------------------------------


def make_nocturne_audio(path: Path,
                        duration: float = 60.0,
                        sr: int = 22050,
                        seed: int = 42) -> Path:
    """A short procedural nocturne in C# minor (Chopin Op. 27 No. 1 mood).

    Slow arpeggiated bass under a sparse upper-register melody, sustain
    pedal effect via long decay envelopes."""
    rng = np.random.RandomState(seed)
    N = int(duration * sr)
    audio = np.zeros(N, dtype=np.float32)

    # Pitch class names → MIDI helper
    pc = {"C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4,
          "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9,
          "A#": 10, "Bb": 10, "B": 11}

    def midi(name, octv): return 12 * (octv + 1) + pc[name]

    def freq(m): return 440.0 * 2 ** ((m - 69) / 12)

    # Harmonic plan: i - VI - III - VII - i over 8 bars, repeating 2x ≈ 60s
    chords = [
        ("C#", "minor"), ("A", "major"), ("E", "major"), ("B", "major"),
        ("C#", "minor"), ("F#", "minor"), ("G#", "major"), ("C#", "minor"),
    ]
    bar_seconds = duration / (len(chords) * 2)  # two passes through

    def chord_tones(root_name, quality):
        root = midi(root_name, 3)
        if quality == "minor":
            return [root, root + 3, root + 7, root + 12]
        else:
            return [root, root + 4, root + 7, root + 12]

    def piano_envelope(n_samples, sr, attack=0.008, decay=0.25,
                       sustain=0.55, release=0.7):
        env = np.ones(n_samples, dtype=np.float32)
        a = max(1, int(attack * sr))
        d = max(1, int(decay * sr))
        r = max(1, int(release * sr))
        env[:a] = np.linspace(0, 1, a)
        if a + d < n_samples:
            env[a:a + d] = np.linspace(1, sustain, d)
            env[a + d:] = sustain
        rs = max(0, n_samples - r)
        env[rs:] = env[rs:] * np.linspace(1, 0, n_samples - rs) ** 1.4
        return env

    def piano_tone(f, dur, vel):
        n = max(1, int(dur * sr))
        tt = np.arange(n) / sr
        sig = (np.sin(2 * np.pi * f * tt)
               + 0.42 * np.sin(2 * np.pi * 2 * f * tt)
               + 0.18 * np.sin(2 * np.pi * 3.01 * f * tt)
               + 0.08 * np.sin(2 * np.pi * 4.0 * f * tt))
        env = piano_envelope(n, sr)
        return (sig * env * vel * 0.18).astype(np.float32)

    def add_at(start_sec, sig):
        idx = int(start_sec * sr)
        if idx >= N:
            return
        end = min(N, idx + sig.size)
        audio[idx:end] += sig[:end - idx]

    # Arpeggiated bass: 4 notes per bar, low register
    t = 0.0
    pass_count = 2
    for _ in range(pass_count):
        for chord_root, quality in chords:
            tones = chord_tones(chord_root, quality)
            # bass arpeggio: tonic, fifth, octave, fifth (12-tone style)
            ord_idx = [0, 2, 3, 2]
            for i in range(4):
                m = tones[ord_idx[i] % len(tones)] - 12  # an octave lower
                f = freq(m)
                dur = bar_seconds / 4 * 1.6  # slight overlap for sustain feel
                vel = 0.36 + 0.04 * (i % 2)
                add_at(t + i * (bar_seconds / 4),
                       piano_tone(f, dur, vel))
            t += bar_seconds

    # Upper-register melody: 1-3 notes per bar, slow rubato, focused on bell tones
    melodic_pool_offsets = [12, 14, 15, 19, 22, 24, 27]  # minor mode upper octave
    t = 0.4  # slight delay so the melody enters after the bass
    for pass_idx in range(pass_count):
        for chord_idx, (chord_root, quality) in enumerate(chords):
            tones = chord_tones(chord_root, quality)
            # 1, 2, or 3 melody notes per bar (denser in second pass)
            n_notes = rng.choice([1, 2, 2, 3]) if pass_idx == 0 else rng.choice([2, 3, 3])
            for k in range(n_notes):
                offset = rng.choice(melodic_pool_offsets)
                m = tones[0] + offset
                # add chromatic ornament occasionally
                if rng.random() < 0.18:
                    m += rng.choice([-1, 1])
                f = freq(m)
                # melody durations are long (nocturne sustain)
                dur = rng.uniform(1.2, 3.4)
                # Position within the bar
                bar_offset = (k + rng.uniform(0.05, 0.85)) * bar_seconds / max(1, n_notes)
                start = t + chord_idx * bar_seconds + bar_offset
                vel = 0.46 + 0.18 * rng.random()
                add_at(start, piano_tone(f, dur, vel))
        t += len(chords) * bar_seconds

    # Final tonic pedal: long C# bell over the closing seconds
    add_at(duration - 6.0,
           piano_tone(freq(midi("C#", 5)), 5.5, 0.55))
    add_at(duration - 5.8,
           piano_tone(freq(midi("E", 5)), 4.5, 0.45))
    add_at(duration - 5.6,
           piano_tone(freq(midi("G#", 5)), 4.0, 0.42))

    # Normalise
    peak = float(np.max(np.abs(audio)) + 1e-9)
    audio = audio / peak * 0.85
    sf.write(path, audio, sr, subtype="PCM_16")
    return path


def main():
    p1 = make_star_chart(INPUTS / "tabula_stellarum_standin.png", seed=23)
    print("wrote", p1)
    p2 = make_star_chart(INPUTS / "tabula_stellarum_alt.png", seed=71)
    print("wrote", p2)
    p3 = make_nocturne_audio(INPUTS / "nocturne_csm_standin.wav", duration=60.0)
    print("wrote", p3)


if __name__ == "__main__":
    main()
