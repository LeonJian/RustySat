# Rusty Sat

**Rust-native, Satpy-compatible satellite data processing library.**

Rusty Sat is a high-performance reimplementation of the Python [Satpy](https://github.com/pytroll/satpy) library in Rust. It reads, calibrates, composites, enhances, resamples, and writes meteorological satellite imagery — with zero Python runtime dependency.

**This project is currently in very early Alpha version, and not use in any professional occasion.**

---

## Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [Architecture](#architecture)
- [Crates](#crates)
- [Supported Sensors](#supported-sensors)
- [Supported Output Formats](#supported-output-formats)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)

---

## Features

| Area | Status | Description |
|------|--------|-------------|
| **AHI HSD Reader** | ✅ | Himawari-8/9 full-disk binary segment files (`.DAT` / `.DAT.bz2`) |
| **AHI L2 NetCDF Reader** | ✅ | Cloud mask, type, height products from NOAA enterprise data |
| **Calibration** | ✅ | Counts → Radiance → Reflectance (visible) and BT (infrared), dual f32/f64 paths |
| **Segment Assembly** | ✅ | Multi-segment concatenation into full-disk with validation |
| **User Calibration** | ✅ | Radiance correction and digital number modes, per-band |
| **Resampling** | ✅ | Nearest, bilinear, EWA, native, bucket (avg/sum/count/fraction) |
| **RGB Compositing** | ✅ | Three-band RGB with common-channel masking |
| **Spectral Blending** | ✅ | Weighted band blending for corrected-green products |
| **Arithmetic Compositing** | ✅ | Difference, ratio, sum, normalized-difference |
| **Image Enhancement** | ✅ | Crude stretch, gamma correction, inversion with history |
| **Output** | ✅ | PNG (8/16-bit), GeoTIFF (float32/64, uint16 scaled), PGM, JPEG |
| **YAML Config** | ⚠️ | Reader config parsing done; composites/enhancements config not yet wired |
| **CLI** | 🚧 | Binary scaffold, not yet functional |

---

## Quick Start

```bash
# Build
cargo build --release

# Generate first AHI image (B03, 0.64 μm visible, full-disk)
cargo run --release --example ahi_first_figure -- \
  local_data/ahi_input/data/20250923/07 B03
# → local_data/ahi_output/B03_reflectance.png (22000×22000, 8-bit grayscale)

# Run all integration tests
cargo test --package rusty_sat_readers \
  --test ahi_first_figure_integration --release -- --nocapture
```

---

## Installation

**Prerequisites:** Rust 1.70+ ([rustup](https://rustup.rs))

```bash
git clone https://github.com/pytroll/satpy.git
cd satpy
cargo build --release
```

Binary at `target/release/rusty-sat`.

**As a library** (path dependency, not yet published to crates.io):

```toml
[dependencies]
rusty_sat_core = { path = "path/to/satpy/crates/rusty_sat_core" }
rusty_sat_readers = { path = "path/to/satpy/crates/rusty_sat_readers" }
rusty_sat_writers = { path = "path/to/satpy/crates/rusty_sat_writers" }
```

Testing requires AHI HSD data files. Set `AHI_DATA_DIR` if not at the default location.

---

## Usage

### Load and Save a Single Band

```rust
use rusty_sat_readers::{
    AhiHsdFileHandler, AhiHsdReader, AhiCalibration, AhiSegmentInfo, Reader,
};
use rusty_sat_writers::{SimpleImageWriter, Writer};

fn main() -> rusty_sat_core::Result<()> {
    // Open all 10 segments of B03 (0.64 μm, 0.5 km resolution)
    let handlers: Vec<_> = (1..=10)
        .map(|seg| {
            AhiHsdFileHandler::from_path(
                format!("data/HS_H09_20250923_0720_B03_FLDK_R05_S{seg:02}10.DAT.bz2"),
                "hsd_b03",
                AhiSegmentInfo::new(seg, 10)?,
            )
        })
        .collect::<Result<Vec<_>>>()?;

    // Reader with reflectance calibration (Counts → Radiance → Reflectance %)
    let reader = AhiHsdReader::with_calibration(AhiCalibration::Reflectance, handlers)?;
    let id = reader.available_dataset_ids().first().cloned().unwrap();
    let dataset = reader.load(&id)?; // Assembles 10 segments → [22000, 22000] f32

    // Save as auto-stretched 8-bit PNG
    SimpleImageWriter::default().save_dataset(&dataset, "B03_reflectance.png")?;
    Ok(())
}
```

### Infrared Calibration

```rust
let reader = AhiHsdReader::with_calibration(
    AhiCalibration::BrightnessTemperature,  // Counts → Radiance → BT (Kelvin)
    handlers,
)?;
let bt = reader.load(&id)?; // values in ~200–320 K range
```

### User Calibration

```rust
use rusty_sat_readers::{AhiUserCalibration, AhiUserCalibrationCoefficients};

// Radiance correction: (radiance - offset) / slope
let corrected = handler.with_user_calibration(
    AhiUserCalibration::radiance_correction([(
        "B04",
        AhiUserCalibrationCoefficients { slope: 0.95, offset: -0.1 },
    )])?,
);

// Digital Number mode: count × slope + offset
let dn = handler.with_user_calibration(
    AhiUserCalibration::digital_number([(
        "B04",
        AhiUserCalibrationCoefficients { slope: -0.0032, offset: 15.20 },
    )])?,
);
```

### RGB Compositing

```rust
use rusty_sat_composites::RgbCompositor;

let rgb = RgbCompositor::new("true_color")?
    .compose_rgb_owned(vec![red_dataset, green_dataset, blue_dataset])?;
// rgb has shape [3, height, width], mode="RGB"
```

### Resample to Target Grid

```rust
use rusty_sat_resample::{
    resample_dataset_from_attrs, ResampleOptions, AreaDefinition,
};

let target = AreaDefinition::new("euro_1km", 2000, 3000)?;
let resampled = resample_dataset_from_attrs(
    &dataset,
    &target,
    ResampleOptions::nearest_area().with_radius_of_influence(5000.0)?,
)?;
```

---

## Architecture

### Compile-Time Dependency Graph

All crates depend on `rusty_sat_core` as the foundation. `core` has zero internal dependencies.

```
                            ┌─────────┐
                            │   CLI   │
                            └────┬────┘
                                 │ (core only)
                            ┌────▼────┐
                            │  core   │  ← DataId, Dataset, Scene, DataArray, error types
                            └────┬────┘
         ┌──────────────────────┼──────────────────────┐
         │                      │                      │
    ┌────▼────┐           ┌────▼────┐           ┌────▼────┐
    │ config  │           │resample │           │ readers │
    │─────────│           │─────────│           │─────────│
    │core     │           │core     │           │core     │
    └─────────┘           └───┬─────┘           └─────────┘
                              │
                        ┌─────▼─────┐
                        │   image   │
                        │───────────│
                        │core       │
                        └─────┬─────┘
                   ┌──────────┴──────────┐
                   │                     │
              ┌────▼──────┐        ┌─────▼──────┐
              │composites │        │  writers   │
              │───────────│        │────────────│
              │core+image │        │core+image  │
              └───────────┘        │+resample   │
                                   └────────────┘
```

### Runtime Data Flow

```
  ┌─────────┐      ┌─────────┐      ┌──────────┐      ┌───────────┐      ┌─────────┐
  │  Files  │ ───→ │ Readers │ ───→ │  Scene   │ ───→ │Composites │ ───→ │ Writers │
  │(.DAT,   │      │(ahi_hsd,│      │(core)    │      │/Resample  │      │(PNG,    │
  │ .nc,    │      │ ahi_l2) │      │Dataset集 │      │           │      │GeoTIFF) │
  │ .bz2)   │      └─────────┘      │合+依賴圖  │      └─────┬─────┘      └─────────┘
  └─────────┘                       └──────────┘            │
                                                      ┌────▼────┐
                                                      │  Image  │
                                                      │增强/拉伸 │
                                                      └─────────┘
```

1. **Readers** parse satellite data files into typed `Dataset` objects (with `DataArray<T>` + coordinates + masks)
2. **Scene** (in core) holds datasets, manages a dependency graph, and plans what to load/compute
3. **Composites/Resample** transform datasets (RGB combination, spatial regridding)
4. **Image** applies enhancement operations (contrast stretch, gamma correction)
5. **Writers** serialize to disk (PNG, GeoTIFF, PGM, JPEG)

For detailed crate-level documentation, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## Crates

| Crate | LOC | Description |
|-------|-----|-------------|
| [`rusty_sat_core`](crates/rusty_sat_core/) | ~2,400 | Foundation: `DataId`, `Dataset`, `Scene`, `DataArray<T>`, `AnyDataArray`, `ValidityMask`, dependency graph |
| [`rusty_sat_config`](crates/rusty_sat_config/) | ~50 | YAML config search path, loading, and deep-merge |
| [`rusty_sat_readers`](crates/rusty_sat_readers/) | ~5,500 | Satellite data readers: AHI HSD, AHI L2 NC, NetCDF, YAML reader config, filename patterns |
| [`rusty_sat_resample`](crates/rusty_sat_resample/) | ~3,500 | Spatial resampling: nearest, bilinear, EWA, native, bucket, KD-tree spatial index |
| [`rusty_sat_composites`](crates/rusty_sat_composites/) | ~2,500 | RGB compositing, spectral blending, arithmetic compositing, YAML config parsing |
| [`rusty_sat_image`](crates/rusty_sat_image/) | ~1,700 | Image types (`Image`, `Image16`, `FloatImage<T>`), enhancement operations |
| [`rusty_sat_writers`](crates/rusty_sat_writers/) | ~1,100 | Output: PNG, GeoTIFF (with GeoKeys), PGM, JPEG |
| [`rusty_sat_cli`](crates/rusty_sat_cli/) | ~10 | CLI binary scaffold |

---

## Supported Sensors

| Sensor | Platform | Reader | Status |
|--------|----------|--------|--------|
| AHI (Advanced Himawari Imager) | Himawari-8/9 | `ahi_hsd` | ✅ All 16 bands, 0.5/1/2 km, segment assembly |
| AHI L2 Cloud Products | Himawari-8/9 | `ahi_l2_nc` | ✅ Cloud mask/type/height, 31 products |
| FCI (Flexible Combined Imager) | MTG-I1 | `fci_l1c_nc` | 🚧 Measured-channel loading |

---

## Supported Output Formats

| Format | Bit Depth | Channels | Georeferencing |
|--------|-----------|----------|----------------|
| PNG | 8-bit | Grayscale | — |
| PNG | 16-bit | Grayscale | — |
| GeoTIFF | float32 | Grayscale | ✅ Full GeoKeys |
| GeoTIFF | float64 | Grayscale | ✅ Full GeoKeys |
| GeoTIFF | uint16 (scaled) | Grayscale | ✅ Full GeoKeys |
| PGM | 8/16-bit | Grayscale | — |
| JPEG | 8-bit | Grayscale | — |

---

## Roadmap

Detailed milestones in [AGENTS.md](AGENTS.md). Highlights:

- [x] **M3-reader-a**: AHI HSD binary header parsing + calibration
- [x] **M7-AHI-prod-e1**: AHI L2 NetCDF metadata, Scene integration, PNG/GeoTIFF output
- [ ] **Composites wiring**: Connect `RgbCompositor` to YAML config + Scene + CLI
- [ ] **True color**: B01+B02+B03 → RGB with Rayleigh correction
- [ ] **CLI v0.1**: Functional command-line interface
- [ ] **More readers**: VIIRS, MODIS, SEVIRI, FCI, OLCI
- [ ] **COG writer**: Cloud-Optimized GeoTIFF with overview pyramids

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

Quick checklist before PR:

```bash
cargo fmt --all -- --check
cargo clippy --workspace -- -D warnings
cargo test --workspace
```

Workspace lint rules:
- `forbid(unsafe_code)` — no unsafe Rust anywhere
- `deny(unwrap_used)` — use `?` with proper error types
- `deny(dbg_macro)` — no debug prints committed
- `deny(todo)` — no half-finished implementations

---

## License

GPL-3.0-only.

This is a ground-up Rust reimplementation inspired by [Satpy](https://github.com/pytroll/satpy) (also GPL). 
