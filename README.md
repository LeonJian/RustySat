# Rusty Sat

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

A Rust-native satellite data processing library designed for Satpy compatibility.

## What is Rusty Sat?

Rusty Sat is a Rust workspace that implements satellite data reading, calibration, resampling, compositing, and image writing — mirroring the Python Satpy library's functionality with Rust's performance and memory safety guarantees.

**Key Goals:**
- **Satpy-compatible behavior** — For supported inputs, produce the same datasets, calibration, geometry, and imagery as Satpy
- **Rust-native APIs** — Idiomatic ownership/borrowing, zero-copy where possible, type-safe generics
- **Performance-first** — Cache-friendly layouts, in-place operations, parallel processing via Rayon
- **Incremental adoption** — Small, tested vertical slices that build on each other

## Quick Start

### Prerequisites

- **Rust 1.70+** — [rustup.rs](https://rustup.rs)
- **Git** — for version control
- **AHI HSD test data** (optional) — for running reader integration tests

### Installation

```bash
git clone https://github.com/pytroll/rusty-sat.git
cd rusty-sat
cargo build --release
```

### Example: Load and Calibrate AHI Data

```rust
use rusty_sat_core::{DataQuery, Scene};
use rusty_sat_readers::{AhiCalibration, AhiHsdFileHandler, AhiHsdReader};
use rusty_sat_writers::{SimpleImageWriter, Writer};
use std::path::Path;

fn main() -> rusty_sat_core::Result<()> {
    // Load AHI HSD segment files
    let handlers: Vec<AhiHsdFileHandler> = vec![
        AhiHsdFileHandler::from_path("HS_H09_20250923_0720_B03_FLDK_R05_S0110.DAT.bz2", "hsd_b03", seg_info)?,
        // ... additional segments
    ];

    // Create reader with calibration mode and attach it to a Scene
    let reader = AhiHsdReader::with_calibration(AhiCalibration::Reflectance, handlers)?;
    let mut scene = Scene::with_loader(reader);

    // Load dataset by name (Satpy-style `scene.load(["B03"])`)
    scene.load([DataQuery::named("B03")?])?;

    // Save as PNG
    let id = &scene.available_dataset_ids()[0];
    scene.save_dataset(id, &SimpleImageWriter::default(), "output.png")?;

    Ok(())
}
```

### Example: RGB Composite

```rust
use rusty_sat_composites::RgbCompositor;
use rusty_sat_core::{Dataset, DataId};

// Assume band1, band2, band3 are loaded single-band datasets
let compositor = RgbCompositor::new("true_color");
let rgb_dataset = compositor.compose(&[band1, band2, band3])?;

// rgb_dataset is now a [3, y, x] array ready for display
```

### Example: Resampling

```rust
use rusty_sat_resample::{resample_dataset, NearestAreaResampler, AreaDefinition};

let target_area = AreaDefinition::from_extent(
    "target", projection, extent, resolution
)?;

let resampled = resample_dataset(&dataset, &target_area, Default::default())?;
```

## Architecture

Rusty Sat is organized as a Cargo workspace with 8 crates:

```
┌─────────────────────────────────────────────────────────────┐
│                        rusty_sat_cli                        │
│                         (entry point)                       │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────┼─────────────────────────────┐
│                             ▼                             │
│                      rusty_sat_core                       │
│              (DataId, Dataset, Scene, Error)               │
└─────────────────────────────┬─────────────────────────────┘
                              │
       ┌──────────────────────┼──────────────────────┐
       ▼                      ▼                      ▼
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│   readers    │      │   resample   │      │   config     │
│ (file I/O)   │      │ (geometry)   │      │ (YAML cfg)   │
└──────────────┘      └──────┬───────┘      └──────────────┘
                             │
       ┌─────────────────────┼─────────────────────┐
       ▼                     ▼                     ▼
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│    image     │      │  composites  │      │   writers    │
│ (enhance)    │      │ (combine)    │      │ (output)     │
└──────────────┘      └──────────────┘      └──────────────┘
```

**Data Flow:**
```
Files (.DAT.bz2, .nc) → Readers → Scene → Composites/Resample → Image → Writers (PNG, GeoTIFF)
```

### Crate Desstractions

| Crate | Purpose | Key Types |
|-------|---------|-----------|
| `rusty_sat_core` | Foundation types, dataset model, scene orchestration | `DataId`, `Dataset`, `Scene`, `RustySatError` |
| `rusty_sat_readers` | File format parsers and data loaders | `Reader` trait, `AhiHsdReader`, `NetCdfFileHandler` |
| `rusty_sat_resample` | Spatial resampling algorithms | `AreaDefinition`, `NearestAreaResampler`, `BilinearAreaResampler` |
| `rusty_sat_composites` | Image compositing and enhancement | `RgbCompositor`, `SelfSharpenedRgb`, `DayNightCompositor`, `EnhancementExecutor` |
| `rusty_sat_image` | Image buffer management | `FloatImage`, `ValidityMask` |
| `rusty_sat_writers` | File output serializers | `SimpleImageWriter`, `FloatTiffWriter`, `PgmWriter` |
| `rusty_sat_modifiers` | Atmospheric and geometric corrections | `RayleighCorrector`, `SunZenithCorrector`, angle computation |
| `rusty_sat_config` | YAML configuration loading | `AppConfig`, area definitions |
| `rusty_sat_cli` | Command-line interface (skeleton) | `main()` |

## Current Capabilities

### Scene Orchestration

- **Scene lifecycle** — `Scene::with_loader(reader)` / `with_loaders([...])` mirrors Satpy's `Scene(filenames, reader)`; `Scene::load([DataQuery])` resolves queries against attached readers, batches loads per reader, and tracks the wishlist
- **Discovery** — `available_dataset_ids()`, `available_dataset_names()`, `missing_datasets()`
- **Metadata** — `start_time()`, `end_time()`, `sensor_names()` derived from loaded dataset attrs with reader-level fallback
- **SceneLoader contract** — readers bridge to `Scene` via explicit `SceneLoader` impls; `Scene::save_dataset()` writes through the `DatasetWriter` contract

### Readers

- **AHI HSD** — Himawari-8/9 Advanced Himawari Imager (full production reader)
  - Header parsing, multi-segment assembly, bzip2 decompression
  - Visible/IR calibration with f32 display and f64 scientific output
  - Geostationary navigation and area metadata

- **AHI NetCDF** — Himawari NetCDF format
  - Groups, variables, global attrs, scale/offset handling
  - Valid-range/fill masking

- **FCI L1C** — Meteosat Third Generation (fixture-backed)
  - Measured-channel counts loading
  - Scene load integration

- **Text Grid** — Simple text-based data
- **YAML Reader** — Metadata-driven configuration

### Resampling

| Method | Source → Target | Use Case |
|--------|-----------------|----------|
| Nearest | Area/Swath → Area | Fast interpolation, categorical data |
| Bilinear | Area → Area | Smooth interpolation |
| EWA | Swath → Area | High-quality swath projection |
| Native | Area → Area | Repeat/aggregate at native resolution |
| Bucket | Swath → Area | Average/sum/count/fraction aggregation |

### Composites

- **RGB Compositor** — 3 bands → true/false color
- **Arithmetic Compositor** — NDVI, difference, ratio, sum
- **Spectral Blender** — weighted band combination
- **Band Replacement** — patch corrected channels into composites
- **Self-Sharpened RGB** — high-resolution band sharpening (Satpy `resolution.py`)
- **Day/Night Blend** — corrected/uncorrected RGB blending by solar-zenith weights (Satpy `fill.py`), used by the JMA `true_color_reproduction` composite

### Image Processing

- **FloatImage<f32/f64>** — precision-aware image buffers
- **Auto-stretch** — percentile-based normalization
- **Gamma/Invert** — tone curve adjustments
- **ValidityMask** — bit-packed missing pixel tracking

### Modifiers

- **Sun-Zenith Correction** — TOA reflectance normalization with Satpy-style 88°→95° angle-domain gradient falloff
- **Day/Night Blend Weights** — cos-zenith weights for the `DayNightCompositor` (Satpy `DayNightCompositor._get_coszen_blending_weights`)
- **Rayleigh Correction** — atmospheric scattering removal
  - Pyspectral-compatible LUT loading
  - Parallel pixel interpolation with LUT-boundary secant clamping
  - Cloud relaxation via the red band (Satpy `rayleigh_corrected` semantics) and high-zenith handling
- **Exact geos inverse** — curved-Earth ray–ellipsoid geolocation for full-disk limb-correct angles

### Writers

| Format | Bit Depth | Features |
|--------|-----------|----------|
| PNG | 8-bit, 16-bit grayscale | Auto-stretch from float |
| JPEG | 8-bit | Lossy compression |
| GeoTIFF | f32, f64, u16 | GeoKey georeferencing, Deflate, tiling |
| PGM | 8-bit, 16-bit | Portable GrayMap |

## What's Next

Rusty Sat is under active development. Current focus areas:

- **Broader reader support** — Expanding beyond AHI to ABI, AMI, SEVIRI, VIIRS, MODIS
- **Advanced compositing** — Mask-aware composites, resolution blending, lookup tables
- **CLI tool** — Full command-line interface for batch processing
- **Lazy evaluation** — Chunked arrays for memory-efficient large dataset processing

See [AGENTS.md](AGENTS.md) for the complete roadmap with 337 tracked features.

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for:

- Development environment setup
- Coding standards and lints
- Pull request workflow
- Commit message conventions

## License

This project is licensed under the **GNU General Public License v3.0** — see [LICENSE](LICENSE) for details.

## Acknowledgments

Rusty Sat is a Rust rewrite of the Python [Satpy](https://github.com/pytroll/satpy) library, part of the [PyTroll](https://pytroll.github.io/) ecosystem. We gratefully acknowledge the Satpy maintainers for their excellent reference implementation.

---

**Note:** Rusty Sat is under active development. APIs may change before v1.0. Contributions and feedback are welcome!
