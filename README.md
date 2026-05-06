# Nocturne Atlas

A deterministic bidirectional transcoder between pre-photographic Western star charts and the nineteenth-century European nocturne. Implemented in pure Python (no learned model, no GPU); outputs are bit-identical given fixed input and configuration.

This repository accompanies the article *"Nocturne Atlas: a deterministic bidirectional transcoder between pre-photographic star charts and the nineteenth-century nocturne"* (manuscript under review).

## Overview

The transcoder runs in two directions.

**Chart → nocturne.** Detected stars on a planispheric plate are mapped to MIDI events. Star declination determines pitch (snapped to a Romantic-period mode), apparent magnitude determines velocity and duration, and right-ascension determines onset time. A slow bass drone and sustain-pedal pulses are added to produce a Chopin-style nocturne register.

**Nocturne → chart.** Spectral peaks of an audio recording are placed on a planispheric chart. Elapsed time becomes angular position around the celestial pole; log-frequency becomes radial position from outer rim (low) to centre (high); spectral magnitude becomes apparent stellar brightness. An engraved-style frame, declination circles, ecliptic arc, and four-point diffraction-spike stars complete the plate.

Two paired plates — *Hemisphaerium Borealis* (midnight palette) and *Hemisphaerium Australis* (indigo palette) — demonstrate the system. A concentric-ring artifact appearing in audio-derived charts is identified as a diagnostic formal residue of harmonically periodic input.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Generate procedural stand-in inputs (replace with historical scans for production)
python synthesize_demo_inputs.py

# Generate the two paired plates in the paper register (default, canonical)
python run_paper.py

# Alternate visual register: heavy parchment instead of paper
python run_quadrivium.py
```

Outputs are written to `outputs/<label>_paper/`:

```
outputs/plate_i_borealis_paper/
├── plate_i_borealis_chart_to_nocturne.mid     # MIDI from chart
├── plate_i_borealis_chart_to_nocturne.wav     # synthesised audio
├── plate_i_borealis_score_paper.png           # score plot on ivory paper
├── plate_i_borealis_nocturne_to_chart.png     # planisphere from audio
├── plate_i_borealis_source_chart_paper.png    # source chart on ivory paper
├── plate_i_borealis_waveform_paper.png        # waveform on ivory paper
├── plate_i_borealis_plate_paper.png           # 4-panel atlas plate
└── plate_i_borealis_metadata.json             # event count + ring statistics
```

## Architecture

The codebase is roughly 2,500 lines of deterministic Python over six modules.

```
quadrivium/
├── __init__.py
├── arithmetic.py            # Pythagorean ratios, Greek modes, Tetractys
├── geometry.py              # planispheric projection, log-frequency-to-radius mapping
├── music.py                 # chart → MIDI (RH cantabile + LH rolling arpeggio,
│                            #   four-phrase chord progression, Tetractys drone)
├── astronomy.py             # audio → planisphere (numbers in time and space) + ring detection
├── paper.py                 # paper register: clean ivory engraving paper (default)
└── parchment.py             # parchment register: dark sepia ink on aged paper
```

The four submodules of `quadrivium/` correspond to the four sciences of the Pythagorean Quadrivium as Park (2025) describes them: arithmetic (pure numbers), geometry (numbers in space), music (numbers in time), astronomy (numbers in time and space). The bidirectional transcoder is the operative numerical mediation between music and astronomy that the historical Quadrivium tradition treated as their underlying kinship.

Dependencies: `numpy`, `scipy`, `librosa`, `pretty_midi`, `Pillow`, `matplotlib`. CPU-only; no GPU and no learned model required.

## Reproducibility

Every output is determined by its input and configuration. Random seeds are fixed throughout for any pseudo-random operations (paper grain, chart generation noise). Identical input and configuration yield bit-identical output.

Runtime on a single CPU thread:

| Direction | Typical time |
|-----------|--------------|
| Chart → nocturne MIDI | ≈ 0.3 s |
| MIDI → audio (additive synth) | ≈ 0.5 s |
| Nocturne → planispheric chart (60 s audio, 2200×2200) | ≈ 1.8 s |
| Atlas plate composite (4 panels, 2400×2600) | ≈ 0.6 s |

## Source materials

The repository ships with **procedural stand-in inputs**, not historical reproductions. For publication-grade artifacts, replace them with:

- **Charts**: high-resolution scans of Bayer's *Uranometria* (1603), Hevelius's *Firmamentum Sobiescianum* (1690), or Bode's *Uranographia* (1801). Public-domain plates are available from the Linda Hall Library, the British Library, the Bibliothèque nationale de France, and the Royal Astronomical Society.
- **Audio**: public-domain recordings of Chopin's nocturnes (Op. 9 No. 2, Op. 27 No. 1, Op. 48 No. 1, and Op. 62 No. 1 are particularly well-suited), Field's nocturnes, Fauré's late nocturnes, or Debussy's *Trois Nocturnes* of 1899.

Place inputs in `inputs/` with the naming convention used by `run_paper.py` (`tabula_stellarum_standin.png` and `tabula_stellarum_alt.png` for the two charts; `nocturne_csm_standin.wav` for the audio).

## Custom prompts and modes

Modal templates available for `chart_to_nocturne`:

| Mode | Pitch classes (C-relative) | Suggested for |
|------|----------------------------|---------------|
| `chopin_csm` | {1, 3, 4, 6, 8, 9, 11} | C-sharp minor (Chopin Op. 27 No. 1) |
| `natural_minor` | {0, 2, 3, 5, 7, 8, 10} | classical minor mode |
| `harmonic_minor` | {0, 2, 3, 5, 7, 8, 11} | Romantic-style minor |
| `phrygian` | {0, 1, 3, 5, 7, 8, 10} | dark, archaic |
| `dorian` | {0, 2, 3, 5, 7, 9, 10} | modal-jazz hint |
| `lydian` | {0, 2, 4, 6, 7, 9, 11} | bright, suspended |
| `octatonic` | {0, 1, 3, 4, 6, 7, 9, 10} | symmetric, dissonant |
| `debussy_whole` | {0, 2, 4, 6, 8, 10} | whole-tone (Debussy) |

Palettes for the planisphere: `midnight`, `indigo`, `iron`.

## Citation

If you reference this work, please cite the article:

> [Author redacted]. (2026). Nocturne Atlas: a deterministic bidirectional transcoder between pre-photographic star charts and the nineteenth-century nocturne. *[Journal title to be confirmed upon acceptance]*. DOI: [to be inserted].

A formal `CITATION.cff` will be added upon acceptance.

## License

The source code is released under the MIT License (see `LICENSE`).

Generated artifacts (MIDI scores, audio renderings, planispheric charts, atlas plates) are released for academic and exhibition use; please credit the work.

## Acknowledgements

Procedural input generators are pseudonymous stand-ins; replace with historical sources where appropriate. No external services or human study participants were involved.

---

*The chart we have produced is not a sky; the score we have produced is not a nocturne. They are both, instead, a small registration of what an instrument has the capacity to witness when its witnessing is mediated by another instrument from a parallel tradition.*
