"""
Photographic Grain Synthesis — Physical Emulsion Simulator
============================================================

Synthesizes photographic grain texture by modeling film emulsion as a stack
of discrete silver-halide (AgX) crystal layers. Each layer is seeded by a
spatial Poisson process and rendered via FFT convolution with a
procedurally generated crystal kernel; layers are partitioned into three
exposure-weighted populations (fast/medium/slow) to approximate a real
multi-layer emulsion. See README.md for the full method description and
references.

Accepts a single-channel or RGB float image. Color inputs are synthesized
one channel at a time, since scene-referred grain amplitude varies with
the intensity of each channel. B&W inputs (R == G == B) are collapsed to
luma before synthesis and expanded back to three channels on output, since
B&W exports from raw processors are typically still stored as RGB.

Reference emulsion layer depths (typical color negative stock):
  Blue-sensitive layer (top):     9.6 μm
  Green-sensitive layer (middle): 21.7 μm
  Red-sensitive layer (bottom):   35.5 μm
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve
import scipy.fft as sp_fft

# ---------------------------------------------------------------------------
# Emulsion Parameters
# Baseline values calibrated against 6144px-long-axis reference scans
# (Fuji Frontier SP3000 benchmark).
# ---------------------------------------------------------------------------

FORMAT_SIZE_MM = 36  # Physical long-axis dimension of a 135/35mm frame

FILLING              = 1.0                  # Target maximum coverage (opaque silver) in highlights
CRYSTAL_DIAMETERS_PX = [7, 7, 7]             # Baseline crystal diameters in pixels [FAST, MEDIUM, SLOW], 3-11 px
POPULATION_PROPORTIONS = [0.25, 0.35, 0.40]  # Layer distribution ratios [FAST, MEDIUM, SLOW]
NUM_LAYERS           = 40                    # Total number of silver halide crystal layers
CRYSTAL_SIZE_STD     = 0.25                  # Log-normal sigma of the crystal size distribution
RANDOM_SEED          = 42
POROSITY             = 0.50                  # Filament porosity factor (phi) inside the crystal

# --- Physical light scattering (microns) ---
SCATTER_MICRONS = [0.0, 3.0, 7.0]  # Standard LSF scattering [FAST, MED, SLOW]

# --- Shadow roll-off parameters ---
# Fixed float (e.g. 0.01) for a classic hard cutoff, or "auto" to align the
# roll-off dynamically with the image's actual black point.
SHADOW_ROLLOFF_CUTOFF      = "auto"
SHADOW_ROLLOFF_ATTENUATION = 1.0     # Strength of shadow attenuation (0.0-1.0)
TOE_RATIO                  = 1.5     # Paper saturation-toe ratio; only used when cutoff is "auto"

# ===========================================================================
# Photographic Adjacency (Nelson Model, 1971)
# ===========================================================================

ADJACENCY_COEFF         = 2.0   # Chemical intensity of the development edge effect
ADJACENCY_RANGE_MICRONS = 30.0  # Physical range of lateral chemical diffusion (15-60 um typical)

# --- Cellulose Triacetate (CTA) substrate parameters ---
# Calibrated against real-world 6048x4011 scans of Rollei RPX 400 and Kodak Gold 200.
SUBSTRATE_MOTTLE_AMPLITUDE = 0.002  # Std dev of CTA thickness variation (0.0 for none)   Gold: 0.001
MOTTLE_ALPHA                = 3.80   # Spectral slope (alpha) for CTA fluid mottle         Gold: 4.35
ANISOTROPY_RATIO            = 1.70   # Horizontal-to-vertical stretch (SP3000 carrier signature)  Gold: 1.5

# Crystal morphology:
#   "traditional" -> branched silver filaments (classic pebble grain)
#   "tabular"     -> flat, high-aspect-ratio hexagonal plates (modern fine grain)
EMULSION_TYPE = "traditional"


# ---------------------------------------------------------------------------
# Crystal geometry
# ---------------------------------------------------------------------------

def create_anisotropic_noise(width, angle_rad, rng: np.random.Generator):
    """
    Generate a 2D noise field smoothed with a highly directional, rotated
    Gaussian filter to emulate elongated metallic silver fibrils.
    """
    noise = rng.normal(size=(width, width))

    # Major axis scales with crystal size (long fibers); minor axis controls
    # fiber thickness (kept cohesive, never subpixel).
    sigma_major = max(width / 3.0, 1.5)
    sigma_minor = 1.0  # FWHM ~2.35 px ensures no single-pixel dust

    ax = np.arange(-width // 2 + 1, width // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)

    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    x_rot = xx * cos_a + yy * sin_a
    y_rot = -xx * sin_a + yy * cos_a

    kernel = np.exp(-0.5 * ((x_rot ** 2 / sigma_major ** 2) + (y_rot ** 2 / sigma_minor ** 2)))
    kernel /= np.sum(kernel)

    padded_noise = np.pad(noise, width, mode="edge")
    smoothed_padded = fftconvolve(padded_noise, kernel, mode="same")

    return smoothed_padded[width:-width, width:-width]


def synthesize_crystal_kernel(width, n_vertices, rotation, rng: np.random.Generator,
                               porosity=POROSITY, stretch_factor=1.0):
    """
    Build a binary kernel representing a single AgX crystal clump.

    Supports both traditional porous pebbles (an isotropic polygon filled
    with crossing anisotropic noise) and solid, high-acutance tabular
    plates (a stretched hexagonal envelope with a twin-line defect).

    The polygon envelope is defined by a polar boundary function
    envelope_r(theta, n_vertices) evaluated against each pixel's query
    radius rho; a pixel is inside the crystal when
    envelope_r >= rho - eps. Fully vectorized (no per-pixel Python loop).
    """
    eps    = 1.0 / width
    radius = max(int((width - 1) / 2.0), 1)

    ii, jj = np.indices((width, width))
    x = jj / radius - 1.0
    y = (ii / radius - 1.0) * stretch_factor

    cos_r, sin_r = np.cos(rotation), np.sin(rotation)
    x_rot = x * cos_r + y * sin_r
    y_rot = -x * sin_r + y * cos_r

    rho = np.hypot(x_rot, y_rot)
    rho = np.where(rho == 0, 1e-5, rho)

    envelope_r = np.cos(np.pi / n_vertices) / np.cos(
        (2.0 * np.arcsin(np.cos(n_vertices * np.arctan2(y_rot, x_rot))) + np.pi) / (2.0 * n_vertices)
    )

    inside_mask = envelope_r >= rho - eps
    is_twin_line = inside_mask & (stretch_factor > 1.0) & (np.abs(y_rot) <= 0.25)

    num_inside = np.sum(inside_mask)
    if num_inside == 0:
        return np.zeros((width, width))

    # Tabular grains bypass the porosity thresholding below: the noise-based
    # fill fragments thin, flat plates into disconnected sub-pixel islands.
    # Instead, return the solid plate with its central twin-line defect.
    if stretch_factor > 1.0:
        kernel = inside_mask.astype(float)
        kernel[is_twin_line] = 0.70
        return kernel

    if porosity >= 1.0:
        return inside_mask.astype(float)

    angle1 = rng.random() * np.pi
    angle2 = angle1 + rng.uniform(0.25 * np.pi, 0.75 * np.pi)

    filament_field1 = create_anisotropic_noise(width, angle1, rng)
    filament_field2 = create_anisotropic_noise(width, angle2, rng)
    combined_noise = filament_field1 + filament_field2

    inside_values = combined_noise[inside_mask]
    target_count = max(int(np.round(porosity * num_inside)), 1)
    # k-th order statistic via partial selection (O(m)) rather than a full
    # sort (O(m log m)) — only the threshold value is needed, not the order.
    threshold_val = np.partition(inside_values, -target_count)[-target_count]

    kernel = np.zeros((width, width))
    kernel[inside_mask] = (inside_values >= threshold_val).astype(float)

    return kernel


def sample_crystal_kernel(mean_diameter_px, size_std, rng: np.random.Generator):
    """
    Sample a random crystal size, shape, and orientation, then synthesize
    its kernel.

    Crystal size follows a log-normal distribution: this matches empirical
    measurements of AgX grain-size distributions, where most grains cluster
    near the mean but a long tail of larger grains exists. Shape (n-gon
    vertex count) is drawn from a normal distribution centered on 6
    (hexagon), clipped to [3, 10] — real AgX crystals are predominantly
    hexagonal but exhibit triangular and higher-order polyhedral forms.

    Args:
        mean_diameter_px (int):     Target mean crystal diameter in pixels.
        size_std (float):            Log-normal sigma for the crystal size distribution.
        rng (np.random.Generator):   Seeded RNG for reproducibility.

    Returns:
        np.ndarray: The synthesized crystal kernel.
    """
    log_normal_diameter = rng.lognormal(mean=np.log(mean_diameter_px), sigma=size_std)
    diameter_px = int(min(max(log_normal_diameter, 1), 3 * mean_diameter_px))
    if diameter_px % 2 == 0:
        diameter_px += 1  # kernel width must be odd for an exact center pixel

    if EMULSION_TYPE == "traditional":
        n_vertices = int(np.clip(rng.normal(6, 1.5), 3, 10))
        rotation = rng.uniform(0, 2 * np.pi)
        return synthesize_crystal_kernel(diameter_px, n_vertices, rotation, rng, POROSITY, stretch_factor=1.0)
    else:
        rotation = rng.uniform(-np.radians(15), np.radians(15))
        stretch = rng.uniform(1.8, 2.4)
        return synthesize_crystal_kernel(diameter_px, 6, rotation, rng, POROSITY, stretch_factor=stretch)


# ---------------------------------------------------------------------------
# Spatial & tonal effects
# ---------------------------------------------------------------------------

def apply_nelson_adjacency(target_density, coeff, sigma, exposure):
    """
    Apply a physically-constrained Nelson adjacency model.

    Includes a developer saturation envelope that shuts off the effect in
    extreme highlights (reproducing the shoulder limit of the H&D curve),
    and a soft-clipping limiter that caps the maximum chemical diffusion
    gradient to prevent runaway values at specular highlights.
    """
    if coeff <= 1e-4 or sigma <= 1e-4:
        return target_density

    neighborhood_density = gaussian_filter(target_density, sigma=sigma)
    diff = target_density - neighborhood_density

    limit_threshold = 1.0
    compressed_diff = diff / (1.0 + np.abs(diff) / limit_threshold)

    saturation_envelope = 1.0 - exposure
    edge_effect = coeff * target_density * compressed_diff * saturation_envelope

    return target_density + edge_effect


def compute_shadow_rolloff_mask(exposure, cutoff, attenuation, toe_ratio=1.5):
    """
    Compute the shadow roll-off mask.

    Supports both a fixed numerical cutoff and a dynamic "auto" cutoff. In
    "auto" mode, the cutoff scales relative to the physical black floor of
    the exposure map using a sensitometric paper toe ratio.

    Args:
        exposure (np.ndarray):   Scene-linear exposure map.
        cutoff (float or str):    Fixed numeric cutoff, or "auto" for dynamic tuning.
        attenuation (float):      Strength of the shadow roll-off (0.0-1.0).
        toe_ratio (float):        Ratio used to find the paper's saturation toe.

    Returns:
        np.ndarray: A 2D mask used to scale down grain seeds in the shadows.
    """
    if isinstance(cutoff, str) and cutoff.lower() == "auto":
        black_floor = np.min(exposure)
        # Cap at 0.05 so high-key images don't roll off midtone grain.
        resolved_cutoff = np.minimum(0.05, np.maximum(0.01, black_floor * toe_ratio))
    else:
        resolved_cutoff = float(cutoff)

    resolved_cutoff = np.maximum(resolved_cutoff, 1e-5)  # guard against divide-by-zero on empty/black frames

    logging.debug(f"shadow mask — minimum exposure: {np.min(exposure):.5f}, cutoff: {resolved_cutoff:.5f}")

    t = np.clip(exposure / resolved_cutoff, 0.0, 1.0)
    smooth_t = 3 * (t ** 2) - 2 * (t ** 3)  # cubic Hermite (smoothstep) roll-off

    return 1.0 - attenuation * (1.0 - smooth_t)


def synthesize_substrate_mottle(shape, alpha=3.80, amplitude=0.057, anisotropy_ratio=1.70, seed=42):
    """
    Generate a 2D cellulose triacetate (CTA) base mottle map.

    Dynamically inspects the aspect ratio of the target canvas so the
    anisotropic casting/scanning mottle aligns with the longest side of the
    physical negative (machine direction).
    """
    H, W = shape[:2]
    rng = np.random.default_rng(seed)

    white_noise = rng.normal(size=(H, W))
    f_transform = np.fft.fft2(white_noise)

    u = np.fft.fftfreq(W)
    v = np.fft.fftfreq(H)
    uu, vv = np.meshgrid(u, v)

    if H > W:
        r = np.sqrt(uu ** 2 + (anisotropy_ratio * vv) ** 2)
    else:
        r = np.sqrt((anisotropy_ratio * uu) ** 2 + vv ** 2)
    r[0, 0] = 1e-5  # avoid division by zero at the DC component

    spectral_filter = 1.0 / (r ** (alpha / 2.0))
    spectral_filter[0, 0] = 0.0  # zero out DC to keep the mean at 0.0

    mottle = np.fft.ifft2(f_transform * spectral_filter).real

    mottle_std = np.std(mottle)
    if mottle_std > 0:
        mottle = (mottle / mottle_std) * amplitude

    return mottle


# ---------------------------------------------------------------------------
# Core grain synthesis
# ---------------------------------------------------------------------------

def synthesize_crystal_field(image, filling=FILLING, crystal_diameters=CRYSTAL_DIAMETERS_PX,
                              num_layers=NUM_LAYERS, size_std=CRYSTAL_SIZE_STD):
    """
    Synthesize photographic grain for a single-channel exposure map.

    Models the emulsion as `num_layers` discrete crystal layers, partitioned
    into fast/medium/slow populations by local exposure. Each layer's
    crystal is sampled independently (size, shape, rotation — see
    sample_crystal_kernel) and rendered via FFT convolution. Adjacency
    (edge) effects, per-layer optical scattering, shadow roll-off, and
    substrate mottle are applied before compositing all layers into the
    final image.

    The output is reconstructed entirely from the synthesized crystal
    field — the input `image` is used only to compute local seeding
    density; original pixel values are not blended back into the result.

    Args:
        image (np.ndarray):             2D exposure map, float, values in (0, 1).
        filling (float):                 Target maximum silver coverage in highlights.
        crystal_diameters (list[int]):   Baseline crystal diameters in pixels, [fast, medium, slow].
        num_layers (int):                 Total number of crystal layers.
        size_std (float):                  Log-normal sigma for the crystal size distribution.

    Returns:
        np.ndarray: The synthesized grain image, float32, clipped to [0, 1].
    """
    t_start = time.perf_counter()

    rng    = np.random.default_rng(RANDOM_SEED)
    result = np.zeros_like(image)

    exposure = np.clip(image, 1e-4, 0.999)

    # --- Monotonic target seeding map ---
    filling_floor = 0.01
    local_filling_compensated = filling_floor + (filling - filling_floor) * 3.85 * exposure

    # --- Layer split determination ---
    n_fast = int(POPULATION_PROPORTIONS[0] * num_layers)
    n_med  = int(POPULATION_PROPORTIONS[1] * num_layers)
    n_slow = num_layers - n_fast - n_med

    is_portrait = image.shape[0] > image.shape[1]

    # --- Crystal kernel sampling (each layer's crystal is drawn independently) ---
    kernels, kernel_areas = [], []

    for i in range(num_layers):
        if i < n_fast:
            mean_diameter_px = crystal_diameters[0]
        elif i < n_fast + n_med:
            mean_diameter_px = crystal_diameters[1]
        else:
            mean_diameter_px = crystal_diameters[2]

        crystal = sample_crystal_kernel(mean_diameter_px, size_std, rng)

        # Rotate horizontally-elongated tabular grain 90 degrees in portrait orientation.
        if is_portrait and EMULSION_TYPE == "tabular":
            crystal = np.rot90(crystal)

        kernels.append(crystal)
        kernel_areas.append(float(crystal.sum()))

    # --- Metric-to-pixel physical conversion ---
    film_width_microns = FORMAT_SIZE_MM * 1000.0
    long_axis_px = max(image.shape[0], image.shape[1])
    pixel_pitch_microns = film_width_microns / long_axis_px

    layer_scatter_sigma = [microns / pixel_pitch_microns for microns in SCATTER_MICRONS]
    adjacency_sigma = ADJACENCY_RANGE_MICRONS / pixel_pitch_microns

    shadow_mask = compute_shadow_rolloff_mask(
        exposure=exposure,
        cutoff=SHADOW_ROLLOFF_CUTOFF,
        attenuation=SHADOW_ROLLOFF_ATTENUATION,
        toe_ratio=TOE_RATIO,
    )
    # Loop-invariant: fold the 1/num_layers seed-count normalization into the
    # shadow mask once, rather than re-multiplying by a constant every layer.
    scaled_shadow_mask = (shadow_mask / num_layers).astype(np.float32)

    # --- Exposure-weighted population partition ---
    w_fast = np.exp(-4.0 * exposure)
    w_med  = np.exp(-10.0 * (exposure - 0.35) ** 2)
    w_slow = np.exp(-6.0 * (1.0 - exposure) ** 2)

    total_w = w_fast + w_med + w_slow
    W_fast, W_med, W_slow = w_fast / total_w, w_med / total_w, w_slow / total_w

    target_fast = local_filling_compensated * W_fast
    target_med  = local_filling_compensated * W_med
    target_slow = local_filling_compensated * W_slow

    # Adjacency and scattering are applied once per population (not per
    # layer), since all layers within a population share the same target.
    target_fast_processed = apply_nelson_adjacency(target_fast, ADJACENCY_COEFF, adjacency_sigma, exposure)

    target_med_blurred = (
        gaussian_filter(target_med, sigma=layer_scatter_sigma[1])
        if layer_scatter_sigma[1] > 1e-4 else target_med
    )
    target_med_processed = apply_nelson_adjacency(target_med_blurred, ADJACENCY_COEFF, adjacency_sigma, exposure)

    target_slow_blurred = (
        gaussian_filter(target_slow, sigma=layer_scatter_sigma[2])
        if layer_scatter_sigma[2] > 1e-4 else target_slow
    )
    target_slow_processed = apply_nelson_adjacency(target_slow_blurred, ADJACENCY_COEFF, adjacency_sigma, exposure)

    # --- Precompute each kernel's FFT once, reused across the layer loop ---
    s1 = image.shape[:2]
    kernel_ffts, fshapes, output_slices = [], [], []

    for crystal in kernels:
        s2 = crystal.shape
        shape = [s1[0] + s2[0] - 1, s1[1] + s2[1] - 1]
        fshape = [sp_fft.next_fast_len(shape[0], True), sp_fft.next_fast_len(shape[1], True)]

        kernel_ffts.append(sp_fft.rfft2(crystal, fshape, workers=-1))
        fshapes.append(fshape)

        startind = [(s2[j] - 1) // 2 for j in range(2)]
        endind = [startind[j] + s1[j] for j in range(2)]
        output_slices.append(tuple(slice(startind[j], endind[j]) for j in range(2)))

    target_fast_scaled = np.maximum(target_fast_processed, 0.0).astype(np.float64) * num_layers
    target_med_scaled  = np.maximum(target_med_processed, 0.0).astype(np.float64) * num_layers
    target_slow_scaled = np.maximum(target_slow_processed, 0.0).astype(np.float64) * num_layers

    # --- Layer loop: seed, mask, convolve, accumulate ---
    for i in range(num_layers):
        crystal_area = kernel_areas[i]
        fshape = fshapes[i]
        kernel_fft = kernel_ffts[i]
        output_slice = output_slices[i]

        if i < n_fast:
            target_scaled, n_group = target_fast_scaled, n_fast
        elif i < n_fast + n_med:
            target_scaled, n_group = target_med_scaled, n_med
        else:
            target_scaled, n_group = target_slow_scaled, n_slow

        local_rate = target_scaled / (n_group * crystal_area)
        seeds = rng.poisson(lam=local_rate).astype(np.float32, copy=False)
        seeds *= scaled_shadow_mask

        seed_fft = sp_fft.rfft2(seeds, fshape, workers=-1)
        seed_fft *= kernel_fft
        grains = sp_fft.irfft2(seed_fft, fshape, workers=-1)[output_slice]

        result += grains

    # --- Acetate substrate mottle ---
    if SUBSTRATE_MOTTLE_AMPLITUDE > 1e-5:
        required_mottle_amplitude = np.mean(result) * SUBSTRATE_MOTTLE_AMPLITUDE
        mottle = synthesize_substrate_mottle(
            shape=image.shape[:2],
            alpha=MOTTLE_ALPHA,
            amplitude=required_mottle_amplitude,
            anisotropy_ratio=ANISOTROPY_RATIO,
            seed=RANDOM_SEED,
        )
        result += mottle

    # --- Global calibration ---
    coef = np.mean(exposure) / np.mean(result)
    grainy_image = np.clip(result * coef, 0.0, 1.0)

    logging.debug(f"synthesize_crystal_field(): coef={coef:.4f}, completed in {time.perf_counter() - t_start:.2f}s")

    return grainy_image


# ---------------------------------------------------------------------------
# Image I/O
# ---------------------------------------------------------------------------

def load_image(path: Path) -> np.ndarray:
    """Load a TIFF as a float32 array in [0, 1], shape (H, W, 3)."""
    raw = tifffile.imread(path)

    if np.issubdtype(raw.dtype, np.integer):
        arr = raw.astype(np.float32) / np.iinfo(raw.dtype).max
    else:
        arr = raw.astype(np.float32)

    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.shape[-1] == 4:
        arr = arr[..., :3]

    return arr


def save_image(arr: np.ndarray, path: Path) -> None:
    """Save a float32 [0, 1] array as a 16-bit TIFF."""
    clipped = np.clip(arr, 0.0, 1.0)
    out16 = (clipped * 65535.0).round().astype(np.uint16)
    tifffile.imwrite(path, out16)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Physically-grounded photographic grain synthesis.")
    parser.add_argument("--input", "-i", type=Path, required=True, help="Path to the input TIFF.")
    parser.add_argument("--output", "-o", type=Path, default=None,
                         help="Output path. Defaults to '<input>_grain<ext>'.")
    parser.add_argument("--filling", type=float, default=FILLING,
                         help="Target maximum silver coverage in highlights.")
    parser.add_argument("--size", type=int, nargs=3, default=CRYSTAL_DIAMETERS_PX, metavar=("FAST", "MEDIUM", "SLOW"),
                         help="Baseline crystal diameters in pixels, per population.")
    parser.add_argument("--layers", type=int, default=NUM_LAYERS, help="Total number of crystal layers.")
    parser.add_argument("--std", type=float, default=CRYSTAL_SIZE_STD,
                         help="Log-normal sigma for the crystal size distribution.")
    parser.add_argument("--emulsion", choices=["traditional", "tabular"], default=EMULSION_TYPE,
                         help="Crystal morphology.")
    parser.add_argument("--verbose", "-v", action="store_true",
                         help="Enable debug-level logging (includes shadow-mask diagnostics).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(message)s")

    global EMULSION_TYPE
    EMULSION_TYPE = args.emulsion

    output_path = args.output or args.input.with_stem(args.input.stem + "_grain")

    logging.info(f"Loading {args.input}...")
    image = load_image(args.input)

    # Robust automatic grayscale detection: color-space conversions and
    # floating-point rounding can cause R, G, and B of a B&W export to drift
    # apart slightly, so equality is tested with a relative tolerance on a
    # coarse subsample rather than requiring an exact match.
    sub_r, sub_g, sub_b = image[::100, ::100, 0], image[::100, ::100, 1], image[::100, ::100, 2]
    luma_sample = (sub_r + sub_g + sub_b) / 3.0
    valid = luma_sample > 1e-4

    if np.any(valid):
        max_diff = np.maximum(np.abs(sub_r[valid] - sub_g[valid]), np.abs(sub_g[valid] - sub_b[valid]))
        is_bw = np.mean(max_diff / luma_sample[valid]) < 0.005
    else:
        is_bw = True

    if is_bw:
        logging.info("B&W input detected — synthesizing grain (single-channel).")
        # AP1 luminance weights (sum to 1.0); since R == G == B for a true
        # B&W export, this collapse is lossless and reversible.
        luma = 0.2722287 * image[:, :, 0] + 0.6740818 * image[:, :, 1] + 0.0536895 * image[:, :, 2]
        grainy = synthesize_crystal_field(luma, filling=args.filling, crystal_diameters=args.size,
                                           num_layers=args.layers, size_std=args.std)
        grainy_rgb = np.stack([grainy, grainy, grainy], axis=-1)
    else:
        logging.info("Color input detected — synthesizing grain per-channel.")
        # To model per-channel physical differences (e.g. a coarser blue
        # layer), replace the shared filling/size/std below with per-channel values.
        channels = []
        for ch in range(3):
            logging.info(f"  channel {ch}...")
            channels.append(
                synthesize_crystal_field(image[:, :, ch], filling=args.filling, crystal_diameters=args.size,
                                          num_layers=args.layers, size_std=args.std)
            )
        grainy_rgb = np.stack(channels, axis=-1)

    save_image(grainy_rgb, output_path)
    logging.info(f"Saved {output_path}")


if __name__ == "__main__":
    main()
