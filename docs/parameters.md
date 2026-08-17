# Parameter Reference

All parameters are module-level constants in `grain.py`. The most commonly tuned ones (`FILLING`, `CRYSTAL_DIAMETERS_PX`, `NUM_LAYERS`, `CRYSTAL_SIZE_STD`, `EMULSION_TYPE`) are also exposed as CLI flags — see `python grain.py --help`. Everything else is edited directly in the file.

## Emulsion

| Parameter | Default | Description |
|---|---|---|
| `FORMAT_SIZE_MM` | `36` | Physical long-axis dimension of the simulated film format (mm). Drives the micron-to-pixel scaling in Section 2.4 of the README. |
| `FILLING` | `1.0` | Target maximum silver coverage in highlights (0–1). Higher values produce denser, more visible grain. |
| `CRYSTAL_DIAMETERS_PX` | `[7, 7, 7]` | Baseline crystal diameter in pixels, per population `[fast, medium, slow]`. Valid range roughly 3–11px before kernels become disproportionately expensive. |
| `POPULATION_PROPORTIONS` | `[0.25, 0.35, 0.40]` | Fraction of total layers assigned to each population `[fast, medium, slow]`. Must sum to 1.0. |
| `NUM_LAYERS` | `40` | Total number of crystal layers simulated. Higher values give smoother statistics at proportionally higher runtime cost. |
| `CRYSTAL_SIZE_STD` | `0.25` | Log-normal sigma for the crystal size distribution (Section 2.1). Larger values widen the size spread. |
| `RANDOM_SEED` | `42` | Seed for the shared RNG. Fixing this makes output reproducible for a given input and parameter set. |
| `POROSITY` | `0.50` | Fraction of each crystal's polygon interior that's filled with filament texture (Section 2.1). `1.0` gives a solid disc. |

## Optical scattering

| Parameter | Default | Description |
|---|---|---|
| `SCATTER_MICRONS` | `[0.0, 3.0, 7.0]` | Gaussian scattering radius in microns, per population `[fast, medium, slow]`. The fast population is left unblurred by default. |

## Shadow roll-off

| Parameter | Default | Description |
|---|---|---|
| `SHADOW_ROLLOFF_CUTOFF` | `"auto"` | Fixed float (e.g. `0.01`) for a hard cutoff, or `"auto"` to anchor dynamically to the image's own black point. |
| `SHADOW_ROLLOFF_ATTENUATION` | `1.0` | Strength of the shadow roll-off (0–1). `0` disables it entirely. |
| `TOE_RATIO` | `1.5` | Multiplier applied to the image's black point when `SHADOW_ROLLOFF_CUTOFF = "auto"`. |

## Adjacency (Nelson model)

| Parameter | Default | Description |
|---|---|---|
| `ADJACENCY_COEFF` | `2.0` | Strength of the edge-density effect. `0` disables adjacency. |
| `ADJACENCY_RANGE_MICRONS` | `30.0` | Physical range of lateral chemical diffusion during development (typically 15–60 μm). |

## Substrate mottle

| Parameter | Default | Description |
|---|---|---|
| `SUBSTRATE_MOTTLE_AMPLITUDE` | `0.002` | Standard deviation of base thickness variation, relative to mean grain signal. `0` disables mottle. |
| `MOTTLE_ALPHA` | `3.80` | Spectral slope of the 1/f mottle noise. Higher values bias toward lower spatial frequencies (smoother mottle). |
| `ANISOTROPY_RATIO` | `1.70` | Stretch ratio of the mottle along the image's long axis, simulating scan/casting direction. |

## Morphology

| Parameter | Default | Description |
|---|---|---|
| `EMULSION_TYPE` | `"traditional"` | `"traditional"` for branched pebble grain, or `"tabular"` for flat T-grain-style plates with a twin-line defect. |
