# Rusty Sat Architecture

## Overview

Rusty Sat is a **Cargo workspace** of 8 crates that together implement a satellite data processing pipeline. The architecture follows a layered design:

- **Foundation layer**: `rusty_sat_core` — all other crates depend on it
- **Domain layer**: `rusty_sat_readers`, `rusty_sat_resample`, `rusty_sat_composites`, `rusty_sat_image` — each provides one processing domain
- **Infrastructure layer**: `rusty_sat_config` — YAML config loading; `rusty_sat_writers` — file output
- **Application layer**: `rusty_sat_cli` — entry point

---

## Crate Dependency Graph

```
                              ┌─────────┐
                              │   CLI   │
                              │ (core)  │
                              └────┬────┘
                                   │
                              ┌────▼────┐
                              │  core   │  ← zero dependencies
                              └────┬────┘
           ┌──────────────────────┼──────────────────────┐
           │                      │                      │
      ┌────▼────┐           ┌────▼────┐           ┌────▼────┐
      │ config  │           │resample │           │ readers │
      │ (core)  │           │ (core)  │           │ (core)  │
      └─────────┘           └───┬─────┘           └─────────┘
                                │
                          ┌─────▼─────┐
                          │   image   │
                          │  (core)   │
                          └─────┬─────┘
                     ┌──────────┴──────────┐
                     │                     │
                ┌────▼──────┐        ┌─────▼──────┐
                │composites │        │  writers   │
                │(core+img) │        │(core+img   │
                └───────────┘        │+resample)  │
                                     └────────────┘
```

**Key**: `(core)` = depends on `rusty_sat_core`. `(core+img)` = depends on `rusty_sat_core` and `rusty_sat_image`.

---

## Runtime Data Flow

The processing pipeline transforms raw satellite files into calibrated, enhanced image files:

```
┌──────────┐    ┌──────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────┐
│  Files   │───→│ Readers  │───→│    Scene     │───→│ Composites  │───→│ Writers  │
│ .DAT.bz2 │    │ ahi_hsd  │    │  (in core)   │    │ / Resample  │    │ PNG,     │
│ .nc      │    │ ahi_l2   │    │              │    │             │    │ GeoTIFF  │
│ .yaml    │    │ netcdf   │    │ Dataset集合   │    │             │    │ PGM      │
└──────────┘    └──────────┘    │ + 依赖图      │    └──────┬──────┘    └──────────┘
                                └──────────────┘           │
                                                     ┌─────▼─────┐
                                                     │   Image   │
                                                     │ (增强/拉伸) │
                                                     └───────────┘
```

### Step-by-Step

1. **Readers** parse binary/NetCDF/YAML files into typed `Dataset` objects. Each `Dataset` contains:
   - `DataId` (name + qualifiers like wavelength, calibration)
   - `AnyDataArray` (typed n-dimensional array: F32/F64/U8/U16/I16)
   - `ValidityMask` (bit-packed mask for invalid pixels)
   - `Coordinate` axes (x, y projection coordinates)
   - `MetadataValue` attributes (platform, sensor, area, etc.)

2. **Scene** (in `rusty_sat_core`) acts as a container and orchestrator:
   - Holds all loaded `Dataset` objects indexed by `DataId`
   - Owns one or more `SceneLoader` sources (readers bridge to this contract) and loads datasets by `DataQuery` through `Scene::load()`, mirroring Satpy's `Scene(filenames, reader)` + `Scene.load(wishlist)` lifecycle
   - `available_dataset_ids()` / `available_dataset_names()` / `missing_datasets()` for discovery; `start_time()` / `end_time()` / `sensor_names()` derived from dataset attrs with loader fallback
   - Manages a `DependencyGraph` tracking what each dataset depends on
   - `plan_reader_loads()` matches `DataQuery` patterns against `ReaderInventory` from each reader
   - `save_dataset()` delegates to `DatasetWriter` implementations

3. **Composites** combine multiple single-band datasets:
   - `RgbCompositor`: 3 bands → [3, y, x] RGB dataset
   - `SpectralBlender`: weighted sum of multiple bands
   - `ArithmeticCompositor`: difference, ratio, sum, normalized difference (NDVI)
   - `BandReplacementCompositor`: patch corrected channel into a composite

4. **Modifiers** apply atmospheric/geometric corrections:
   - `RayleighCorrector`: removes Rayleigh scattering from visible bands
   - `AngleSet`: computes solar/satellite zenith/azimuth for each pixel
   - Uses LUT interpolation with pyspectral-compatible data

5. **Resampling** transforms datasets between spatial grids:
   - Multiple methods: Nearest, Bilinear, EWA, Native, Bucket
   - `AreaDefinition` (regular grid) or `SwathDefinition` (irregular lon/lat)
   - `ResampleOptions` controls radius, fill values, mask policy

6. **Image** enhancement layer:
   - `FloatImage<f32/f64>`: intermediate float representation
   - Operations: auto-stretch, gamma, invert
   - Tracks history for reproducibility

7. **Writers** serialize to disk:
   - `SimpleImageWriter`: PNG (8/16-bit), JPEG (8-bit)
   - `FloatTiffWriter`: GeoTIFF with GeoKeys, float32/64/u16
   - `PgmWriter`: Portable GrayMap (8/16-bit)
   - `BuiltinWriterFactory`: selects writer by file extension

---

## Crate Reference

### `rusty_sat_core` — Foundation

**Purpose**: Zero-dependency foundation with shared types used by all other crates.

**Key Abstractions**:
- `DataId` — named dataset identifier with qualifier map (wavelength, calibration, modifiers)
- `DataQuery` — pattern for matching/ranking datasets with `Any`, `One`, `AnyOf` matching
- `DataValue` — typed qualifier value: `Text`, `Number`, `Wavelength`, `Modifiers`
- `Dataset` — complete satellite dataset: identity + array + metadata + coordinates + aux vars
- `AnyDataArray` — runtime-typed numeric array (`F32`, `F64`, `U8`, `U16`, `I16`)
- `DataArray<T>` — owned n-dimensional array with shape, dimensions, values, mask, coordinates, chunks
- `DataGrid` — type alias for `DataArray<f64>` (2D legacy compatibility)
- `Scene` — main container: datasets, wishlist, dependency graph, reader load planning, and query-based loading through attached `SceneLoader` sources
- `SceneLoader` — dataset discovery/loading contract (`name`, `available_dataset_ids`, `load`, `load_batch`, time/sensor metadata) that readers bridge to
- `DependencyGraph` — directed graph tracking dataset dependencies
- `ValidityMask` — bit-packed mask where set bit = invalid pixel
- `Coordinate` — named coordinate axis with f64 values
- `MetadataValue` — Satpy-style attribute (`Null`, `String`, `Bool`, `Integer`, `Float`, `List`, `Map`)
- `RustySatError` — unified error type: `Unsupported`, `InvalidInput`, `NotFound`, `Ambiguous`
- `ReaderInventory` — reader name + available DataIds

**Design Patterns**:
- `DataArray<T>` is generic over `NumericElement` trait (f32/f64/u8/u16/i16)
- `AnyDataArray` provides runtime dtype dispatch via `DataType` enum
- `Scene` owns datasets and orchestrates reader/compositor/modifier execution; readers attach as `Box<dyn SceneLoader>` so the core crate stays reader-agnostic
- Single error type eliminates per-crate error conversions

---

### `rusty_sat_readers` — File Format Parsers

**Purpose**: Parse satellite files into typed `Dataset` objects.

**Supported Formats**:
- **AHI HSD**: Himawari-8/9 Advanced Himawari Imager (production ready)
  - Header parsing, multi-segment assembly, bzip2 decompression
  - Visible/IR calibration with f32 display and f64 scientific output
  - Geostationary navigation and area metadata
- **AHI NetCDF**: Himawari NetCDF format
  - Groups, variables, global attrs, scale/offset handling
  - Valid-range/fill masking
- **FCI L1C**: Meteosat Third Generation (fixture-backed)
- **Text Grid**: Simple text-based data
- **YAML Reader**: Metadata-driven configuration

**Key Abstractions**:
- `Reader` trait — `name()`, `available_dataset_ids()`, `load(&DataId) -> Dataset`, plus time/sensor metadata hooks
- Every concrete reader also implements the core `SceneLoader` contract (explicit forwarding impls) so it can be attached to a `Scene`
- `AhiHsdReader` — full AHI HSD implementation with calibration modes
- `AhiHsdFileHandler` — single HSD segment file parser
- `NetCdfFileHandler` — generic NetCDF file handler
- `filename_pattern` — trollsift-compatible filename parsing/matching

**Design Patterns**:
- Readers produce `Dataset` objects with populated `DataId`, `AnyDataArray`, `ValidityMask`
- Calibration modes control output dtype: `Reflectance` (f32 display) vs `ScientificF64`
- Multi-segment assembly merges partial images into full-disk products
- File handlers are independent of `Scene`; readers orchestrate handlers

---

### `rusty_sat_resample` — Spatial Resampling

**Purpose**: Transform datasets between spatial grids using various algorithms.

**Resampling Methods**:

| Method | Source → Target | Key Type |
|--------|-----------------|----------|
| Nearest | Area/Swath → Area | `NearestAreaResampler` |
| Bilinear | Area → Area | `BilinearAreaResampler` |
| EWA | Swath → Area | `EwaResampler` |
| Native | Area → Area | `NativeResampler` |
| Bucket | Swath → Area | `BucketResampler` |

**Geometry Types**:
- `AreaDefinition` — regular grid in a map projection
- `SwathDefinition` — irregular swath with optional lon/lat arrays
- `CoordinateDefinition` — point coordinates
- `GridDefinition` — lon/lat grid
- `ProjectionDefinition` — pixel-to-projection coordinate math

**Pipeline Components**:
- `Resampler` trait — `name()`, `resample()`, `resample_owned()`
- `ResampleOptions` — radius, fill values, mask policy, data reduction
- `SourceGeometry` — inferred from dataset attributes or explicit
- `ResamplerCache` — caches prepared resamplers by source/destination key
- `ImageContainer` — binds dataset to source geometry for sampling
- `NeighbourInfo` — nearest/bilinear neighbour lookup data
- `Slicer` — area slicing for data reduction
- `DataReduction` — coarse lon/lat boundary filtering

**Design Patterns**:
- Borrowed (`&Dataset`) and owned (`Dataset`) resampling APIs
- Pipeline preparation separates resampler creation from execution
- Geometry inference from dataset attributes when possible
- Fill-vs-mask missing value policy with default NaN fill

---

### `rusty_sat_composites` — Image Compositing

**Purpose**: Combine multiple bands into composite products.

**Compositors**:
- `RgbCompositor` — 3 single-band datasets → [3, y, x] RGB dataset
- `ArithmeticCompositor` — binary ops: difference, ratio, sum, normalized difference (NDVI)
- `SpectralBlender` — weighted sum of N bands
- `BandReplacementCompositor` — in-place band replacement
- `SelfSharpenedRgb` — high-resolution band sharpening (Satpy `resolution.py`)
- `DayNightCompositor` — blend corrected (day) and uncorrected (night) band-major RGB composites with per-pixel solar-zenith weights (Satpy `fill.py`; weights computed externally, e.g. `rusty_sat_modifiers::daynight_blend_weights`)

**Enhancement**:
- `EnhancementExecutor` — safely executes YAML-defined operations (stretch, gamma, invert)
- `CompositeRegistryConfig` — parses Satpy-style YAML composite/enhancement configs

**Design Patterns**:
- All compositors implement `Compositor` trait
- Common-channel mask propagation (invalid if any input invalid)
- Shape validation before composition
- Metadata extraction from input datasets

---

### `rusty_sat_image` — Image Buffer Management

**Purpose**: Bridge scientific float arrays and display-ready pixel buffers.

**Image Types**:
- `FloatImage<T>` — intermediate float representation (f32 or f64)
  - Enhancement operations (stretch, gamma, invert) applied here
  - Tracks history for reproducibility
- `Image` — 8-bit output (u8 pixels) via `to_u8_image()`
- `Image16` — 16-bit output (u16 pixels) for HDR/scientific display

**Enhancement Finalizers**:
- `finalize_rgb_cira_u8` — fused CIRA log stretch (Satpy `true_color_default`)
- `finalize_rgb_jma_u8` — fused JMA True Color Reproduction enhancement: per-pixel color conversion matrix (Satpy `enhancements/ahi.py`, Himawari-8/9) + log stretch min 3/max 150 (Satpy `true_color_reproduction_color_stretch`)

**Image Modes**:
- `ImageMode::Luma` — 1 channel (grayscale)
- `ImageMode::Rgb` — 3 channels
- `ImageMode::Rgba` — 4 channels with alpha

**Design Patterns**:
- `ImageFloat` trait abstracts f32/f64 precision
- Auto-stretch computes percentile range from valid pixels
- Mask-aware operations exclude invalid pixels from statistics
- History tracking enables reproducible enhancement chains

---

### `rusty_sat_writers` — File Output

**Purpose**: Serialize `Dataset` and `Image` objects to disk.

**Writers**:
- `SimpleImageWriter` — PNG (8/16-bit grayscale), JPEG (8-bit)
- `FloatTiffWriter` — GeoTIFF with GeoKeys
  - Supports float32, float64, uint16-scaled output
  - Optional Deflate compression and tiling
- `PgmWriter` — Portable GrayMap (8/16-bit)
- `BuiltinWriterFactory` — selects writer by file extension

**Design Patterns**:
- `Writer` trait — `name()`, `save_image()`, `save_dataset()`
- `DatasetWriter` trait — core integration point
- Factory pattern for writer selection
- `TiffSamplePolicy` controls GeoTIFF pixel format
- Fill values for masked/invalid pixels

---

### `rusty_sat_modifiers` — Atmospheric/Geometric Corrections

**Purpose**: Apply corrections to satellite imagery.

**Modules**:
- `astronomy` — solar position math (GMST, sun RA/DEC, cos_zen, alt/az) ported from pyorbital
- `geos` — geostationary projection inverse (x/y meters → lon/lat degrees) via exact ray–ellipsoid intersection (PROJ `geos` / pyresample `get_lonlats` parity; replaces the old flat-plane approximation that was off by ~57° at the disk limb)
- `orbital` — satellite look angles (azimuth, elevation, zenith) ported from pyorbital
- `angles` — combined angle computation for dataset grids (exact geos inverse + pyorbital solar/satellite angles; strip-parallel per-pixel)
- `sun_zenith` — solar-zenith correction with Satpy-style 88°–max_sza angle-domain gradient falloff, plus `daynight_blend_weights` (cos-zenith blend weights for the `DayNightCompositor`, Satpy `DayNightCompositor._get_coszen_blending_weights` parity)
- `rayleigh` — Rayleigh scattering correction modifier (delegates LUT I/O and interpolation to `rustyspectral` crate), pyspectral-parity red-band cloud relaxation via `RedBandSource` (`None` / `Dataset` / `SunZenithCorrectedVis`), LUT-boundary angle clipping, correction clip to [0,100]

**Design Patterns**:
- `UtcInstant` for time representation (dependency-free)
- Parallel pixel interpolation via rayon
- Memory-efficient consuming APIs
- Pyspectral-compatible LUT data format

---

### `rusty_sat_config` — Configuration Loading

**Purpose**: YAML-driven configuration for readers, composites, enhancements.

**Key Abstractions**:
- `ConfigSearchPath` — ordered search paths for YAML files
- `AppConfig` — merged configuration from multiple sources
- `readers/`, `composites/`, `enhancements/` directory layout

**Design Patterns**:
- Deterministic search path ordering
- Recursive YAML merging with depth limits
- Environment variable injection (`RUSTY_SAT_CONFIG_PATH`)
- Satpy-compatible config path fallback

---

### `rusty_sat_cli` — Command-Line Interface

**Purpose**: Entry point for Rusty Sat (currently minimal skeleton).

**Current State**:
- Prints version and dataset count
- Will expand to full CLI for batch processing

---

## Cross-cutting Concerns

### Error Handling

Single error type across workspace:

```rust
pub enum RustySatError {
    Unsupported { feature: String },
    InvalidInput { message: String },
    NotFound { item: String },
    Ambiguous { message: String },
}
```

Factory methods: `RustySatError::unsupported(...)`, `::invalid_input(...)`, `::not_found(...)`, `::ambiguous(...)`.

Type alias: `pub type Result<T> = std::result::Result<T, RustySatError>`.

---

### Safety Guarantees

- `forbid(unsafe_code)` — no `unsafe` blocks in workspace
- `deny(unwrap_used)` — all fallible operations use `?` or explicit error handling
- `deny(dbg_macro)` — no debug prints in committed code
- `deny(todo)` — incomplete features return `Err(Unsupported{...})` instead of panicking

---

### Performance Strategy

- **Zero-copy where possible** — consuming APIs (`self`) avoid cloning
- **Dual precision paths** — f32 for display, f64 for scientific accuracy
- **Parallel processing** — rayon for per-pixel operations (Rayleigh correction, resampling)
- **Cache-friendly layouts** — row-major arrays, chunk-aware loading
- **Builder pattern** — fluent API for complex configuration (`ResampleOptions`)

---

### Memory Management

- **Consuming APIs** — methods take `self` to enable zero-copy transformations
- **Lazy evaluation foundations** — `LazyDataArray` for deferred file-backed loading
- **Bit-packed masks** — `ValidityMask` uses 1 bit per pixel instead of full byte
- **Explicit ownership** — Rust's borrow checker enforces safe mutation patterns

---

### Testing Strategy

- **Unit tests** — per-module, covering success/error/edge paths
- **Parity tests** — validate against Python Satpy outputs for same inputs
- **Integration tests** — end-to-end reader → compositing → writer pipelines
- **Property-based testing** — for geometric calculations (future)

---

## Architecture Decisions

### Why Rust?

- **Performance** — AOT compilation, zero-cost abstractions, SIMD support
- **Safety** — ownership/borrowing prevents data races and use-after-free
- **Memory efficiency** — no GC pauses, deterministic deallocation
- **Ecosystem** — growing scientific Rust libraries (ndarray, hdf5, proj)

### Why Satpy-compatible?

- **Correctness guarantee** — same inputs → same outputs as reference implementation
- **Incremental validation** — can compare Rust vs Python for any supported reader
- **User migration** — existing Satpy workflows work unchanged
- **Documentation** — Satpy's docs serve as behavior specification

### Design Trade-offs

- **Single error type** — simpler propagation but less granular error context
- **Generic DataArray<T>** — type safety at cost of monomorphization bloat
- **Runtime dtype dispatch** — `AnyDataArray` adds indirection for multi-dtype flexibility
- **Trait-based readers** — standardized interface limits format-specific optimizations

---

## Bounds and Limits

| Limit | Value | Where |
|-------|-------|-------|
| Max HSD file size | 2 GB | `MAX_HSD_FILE_BYTES` |
| Initial header read | 4096 bytes | `INITIAL_HEADER_PREFIX_LEN` |
| Max YAML file size | 8 MB | readers, composites, config |
| Max YAML nesting depth | 96 | readers, composites, config |
| Supported dtypes | f32, f64, u8, u16, i16 | `DataType` enum |
| Max DataArray dimensions | 4D (time, bands, y, x) | `DataArray<T>` |

---

## Future Architecture Considerations

- **Lazy evaluation** — full Dask-like chunked loading with lazy IO
- **Distributed processing** — potential for data-parallel workflows
- **GPU acceleration** — CUDA/Metal for resampling and compositing
- **Streaming I/O** — incremental dataset loading for large files

---

*Last updated: 2026-07-09*
