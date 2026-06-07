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
   - Manages a `DependencyGraph` tracking what each dataset depends on
   - `plan_reader_loads()` matches `DataQuery` patterns against `ReaderInventory` from each reader
   - `save_dataset()` delegates to `DatasetWriter` implementations

3. **Composites** combine multiple single-band datasets:
   - `RgbCompositor`: 3 bands → [3, y, x] RGB dataset
   - `SpectralBlender`: weighted sum of multiple bands
   - `ArithmeticCompositor`: difference, ratio, sum, normalized-difference

4. **Resample** transforms datasets between spatial grids:
   - Nearest-neighbour (area-to-area, swath-to-area)
   - Bilinear interpolation
   - Elliptical Weighted Averaging (EWA) for swath data
   - Bucket resampling (average, sum, count, fraction)
   - Native resolution (aggregation or repetition)

5. **Image** converts Dataset arrays to display-ready pixels:
   - `crude_stretch`: linear min→max contrast stretch
   - `gamma_correction`: power-law adjustment
   - `invert`: channel inversion
   - Output: `Image` (u8) or `Image16` (u16)

6. **Writers** serialize to disk:
   - `SimpleImageWriter`: PNG (8/16-bit grayscale) via the `image` crate
   - `FloatTiffWriter`: GeoTIFF with full GeoKey georeferencing
   - `PgmWriter`: Portable GrayMap

---

## Crate Details

### rusty_sat_core

Foundation crate — zero internal dependencies. Provides all shared types used across the workspace.

**Key Types:**

| Type | Purpose |
|------|---------|
| `DataId` | Named dataset identifier with qualifier map (wavelength, calibration, modifiers) |
| `DataQuery` | Pattern for matching/finding datasets; supports `Any`, `One`, `AnyOf` matching |
| `DataValue` | Qualifier value: `Text`, `Number(FloatValue)`, `Wavelength(WavelengthRange)`, `Modifiers(ModifierTuple)` |
| `WavelengthRange` | (min, central, max, unit) with distance/containment math for band matching |
| `Dataset` | Complete dataset: DataId + AnyDataArray + attributes + coordinates + ancillary variables |
| `AnyDataArray` | Runtime-typed array: `F32(DataArray<f32>)`, `F64`, `U8`, `U16`, `I16` |
| `DataArray<T>` | Owned n-dimensional numeric array with shape, dims, values, mask, coordinates, chunks |
| `DataGrid` | Type alias: `DataArray<f64>` (2D f64 grid for legacy compatibility) |
| `ValidityMask` | Bit-packed mask where set bit = invalid data value |
| `Coordinate` | Named coordinate axis with f64 values |
| `ChunkShape` | Per-dimension chunk sizes for lazy/deferred arrays |
| `LazyDataArray<T>` | Deferred array with `ChunkSource<T>` trait for on-demand chunk loading |
| `Scene` | Main container: datasets map, wishlist, dependency graph, load planning |
| `DependencyGraph` | Directed graph of `DependencyNode` (DataId + source + dependencies) |
| `MetadataValue` | Satpy-style attribute value: `Null`, `String`, `Bool`, `Integer`, `Float`, `List`, `Map` |
| `RustySatError` | Unified error type: `Unsupported`, `InvalidInput`, `NotFound`, `Ambiguous` |
| `ReaderInventory` | Reader name + set of available DataIds |
| `SceneLoadPlan` | Result of matching queries against reader inventories |

**Data Flow Through Core:**

```
DataQuery ──matches──→ DataId ──identifies──→ Dataset
    │                       │                     │
    │  .best_match()        │  .qualifiers()      │  .array() → AnyDataArray
    │  .filter_data_ids()   │  .name()            │  .attr()  → MetadataValue
    │  .sort_data_ids()     │  .modifiers()       │  .id()    → DataId
    │                       │                     │
    ▼                       ▼                     ▼
 Scene.plan_reader_loads()  Reader.load()    Writer.save_dataset()
```

### rusty_sat_config

YAML configuration loading and merging. Single file, ~50 lines.

**Key Types:**
- `ConfigSearchPath` — ordered search path from env vars (`RUSTY_SAT_CONFIG_PATH`, `SATPY_CONFIG_PATH`)
- `ConfigComponent` — enum: `Readers`, `Writers`, `Composites`, `Enhancements`
- `load_and_merge_yaml_files()` — deep-merge multiple YAML files (later files override)
- `load_yaml_file()` — single-file YAML loader with safety limits (8 MB, depth 96)

### rusty_sat_readers

Satellite data file readers. Largest crate (~5,500 lines).

**Modules:**

| Module | File | Purpose |
|--------|------|---------|
| `ahi_hsd` | `ahi_hsd.rs` (~3,100 lines) | Himawari AHI HSD binary reader |
| `ahi_l2_nc` | `ahi_l2_nc.rs` (~1,100 lines) | Himawari AHI L2 NetCDF reader |
| `fci_l1c_nc` | `fci_l1c_nc.rs` | MTG FCI L1C NetCDF reader (foundations) |
| `netcdf` | `netcdf.rs` | Generic NetCDF metadata and data access layer |
| `yaml_reader` | `yaml_reader.rs` | Satpy-style YAML reader config parsing |
| `filename_pattern` | `filename_pattern.rs` | Trollsift-compatible filename pattern matching |
| `text_grid` | `text_grid.rs` | Simple text-grid reader |

**Core Trait:**

```rust
pub trait Reader {
    fn name(&self) -> &str;
    fn available_dataset_ids(&self) -> Vec<DataId>;
    fn load(&self, id: &DataId) -> Result<Dataset>;
}
```

**AHI HSD Reader Architecture (`ahi_hsd.rs`):**

```
AhiHsdFileHandler (per-file)
  ├── from_path() / from_header_bytes()
  ├── header() → AhiHsdHeader (blocks 1-7 parsed)
  ├── load_counts_dataset() → u16 raw counts + mask
  ├── load_calibrated_dataset(calib) → f32 calibrated + mask
  ├── load_calibrated_dataset_f64(calib) → f64 calibrated + mask
  ├── calibrate_counts_to_f32/f64(counts, calib) → Vec<f32/f64>
  ├── area_metadata_value() → MetadataValue (geostationary area)
  └── band_name(), dataset_id(), segment()

AhiHsdReader (multi-file aggregator)
  ├── with_calibration(calib, handlers[])
  ├── with_output(DisplayF32 | ScientificF64)
  ├── load(id) → single segment or assembled full-disk
  │   ├── load_handler_dataset() — single segment
  │   └── load_assembled_dataset() — concatenate yx arrays
  └── implements Reader trait
```

**AHI Calibration Chain:**

```
Raw counts (u16)
  │  counts_to_radiance_f32/f64:
  │    counts × gain + offset
  │    (with optional user calibration: radiance correction or DN mode)
  ▼
Radiance (f32/f64)
  │  Visible: radiance_to_reflectance:
  │    radiance × coeff_rad_to_albedo × 100 → %
  │  Infrared: radiance_to_brightness_temperature:
  │    Planck function inversion → Kelvin
  ▼
Reflectance % (visible) or Brightness Temperature K (infrared)
```

**AhiCalibration variants:**
- `Counts` — raw 16-bit digital numbers
- `Radiance` — calibrated spectral radiance
- `Reflectance` — visible band reflectance (0–100+ %)
- `BrightnessTemperature` — infrared band brightness temperature (Kelvin)

**AhiCalibrationMode:**
- `Nominal` — use block-5 base gain/offset
- `Update` — use calibrated gain/offset from block-5 extension if available (default)

**AhiCalibrationOutput:**
- `DisplayF32` — f32 values for memory-efficient display (default)
- `ScientificF64` — f64 values for scientific precision

**AhiUserCalibration:**
- `RadianceCorrection` — (radiance - offset) / slope per band
- `DigitalNumber` — count × slope + offset per band (bypasses file calibration)

### rusty_sat_resample

Spatial resampling algorithms (~3,500 lines, 20 files).

**Geometry Types:**

| Type | Trait | Description |
|------|-------|-------------|
| `AreaDefinition` | `ProjectionDefinition` | Regular grid in a map projection |
| `SwathDefinition` | `GeometryDefinition` | Irregular swath with lon/lat arrays |
| `CoordinateDefinition` | `GeometryDefinition` | Point coordinates |
| `GridDefinition` | `GeometryDefinition` | Lon/lat grid |

**Resampler Trait:**

```rust
pub trait Resampler {
    fn name(&self) -> &str;
    fn resample(&self, dataset: &Dataset, destination: &AreaDefinition) -> Result<Dataset>;
    fn resample_owned(&self, dataset: Dataset, destination: &AreaDefinition) -> Result<Dataset>;
}
```

**Resampling Methods:**

| Method | Struct | Source Type | Algorithm |
|--------|--------|-------------|-----------|
| Nearest Area | `NearestAreaResampler` | Area | KD-tree accelerated nearest-neighbour |
| Bilinear | `BilinearAreaResampler` | Area | 4-point bilinear interpolation |
| Native | `NativeResampler` | Area | Aggregation (shrink) or repetition (expand) |
| EWA | `EwaResampler` | Swath | Elliptical Weighted Averaging |
| Bucket Avg | `BucketResampler` | Swath | Drop-in-a-bucket averaging |
| Bucket Sum | `BucketResampler` | Swath | Drop-in-a-bucket summation |
| Bucket Count | `BucketResampler` | Swath | Drop-in-a-bucket counting |
| Bucket Fraction | `BucketFractionResampler` | Swath | Category fraction per bucket |

**Pipeline Convenience:**

```rust
// One-liner: infer source geometry from dataset attrs, pick method, resample
resample_dataset_from_attrs(&dataset, &target_area, ResampleOptions::nearest_area())?;
```

**Key supporting types:**
- `KdPointIndex2D` — 2D KD-tree for nearest-neighbour queries
- `NeighbourInfo` — precomputed neighbour mapping for KD-tree output
- `AreaSlice` / `AreaCrop` / `AreaDataReduction` — data reduction helpers
- `ProjCrs` — CRS metadata wrapper with PROJ string parsing
- `GeoKeyDef` / `GeoTiffGeoKeyFinal` — GeoTIFF georeferencing tag generation

### rusty_sat_composites

Compositing, spectral blending, arithmetic operations, and enhancement (~2,500 lines, 6 files).

**Compositor Trait:**

```rust
pub trait Compositor {
    fn name(&self) -> &str;
    fn compose(&self, inputs: &[Dataset]) -> Result<Dataset>;
}
```

**Compositors:**

| Type | Input | Output | Use Case |
|------|-------|--------|----------|
| `RgbCompositor` | 3 single-band Datasets | [3, y, x] f64, mode="RGB" | True color, false color |
| `SpectralBlender` | N single-band Datasets | [y, x] f64 weighted blend | Corrected green band |
| `ArithmeticCompositor` | 2 single-band Datasets | [y, x] f64 | Difference, ratio, NDVI, etc. |
| `BandReplacementCompositor` | Band-major + replacement | Band-major with one band replaced | Channel patching |

**ArithmeticOperation variants:** `Difference`, `Ratio`, `Sum`, `NormalizedDifference`

**Enhancement:**
- `EnhancementExecutor` — safely executes enhancement operations (allowlist: `stretch`, `gamma`, `invert`)
- `CompositeRegistryConfig` — parses Satpy-style YAML composite and enhancement definitions
- `EnhancementDefinition` / `EnhancementOperation` — structured representation of YAML configs

### rusty_sat_image

Image types and enhancement operations (~1,700 lines).

**Image Types:**

| Type | Pixel Depth | Use |
|------|------------|-----|
| `FloatImage<T>` (T: f32 or f64) | float | Intermediate: enhancement operations track history |
| `Image` | u8 | Final: 8-bit grayscale/RGB/RGBA output |
| `Image16` | u16 | Final: 16-bit grayscale/RGB/RGBA output |

**Key Operations:**

```
FloatImage (from Dataset array)
  │  .crude_stretched(min, max) — linear stretch, auto-range if None
  │  .gamma_corrected(gamma)     — power-law adjustment
  │  .inverted(invert)           — channel inversion
  ▼
  .to_u8_image(fill_value)  → Image (u8)
  .to_u16_image(fill_value) → Image16 (u16)
```

**StretchRecord** tracks `(scale, offset)` per channel for reproducible enhancement.

**ImageMode:** `Luma` (1 channel), `Rgb` (3 channels), `Rgba` (4 channels).

### rusty_sat_writers

File output (~1,100 lines, 4 files).

**Writer Trait:**

```rust
pub trait Writer {
    fn name(&self) -> &str;
    fn save_image(&self, image: &Image, path: &Path) -> Result<()>;
    fn save_dataset(&self, dataset: &Dataset, path: &Path) -> Result<()>;
}
```

**Writers:**

| Writer | Formats | Features |
|--------|---------|----------|
| `SimpleImageWriter` | PNG (8/16-bit), JPEG (8-bit) | Auto-detects format from extension, 8/16-bit dataset output |
| `FloatTiffWriter` | GeoTIFF (float32/64, uint16) | Full GeoKeys, Deflate compression, tiled output, scaled uint16 |
| `PgmWriter` | PGM (8/16-bit) | Linear scaling with fill value support |

**BuiltinWriter enum + BuiltinWriterFactory** — automatic writer selection by file extension.

**TiffSamplePolicy:** `Float32`, `Float64`, `UInt16Scaled` — controls the pixel data format in TIFF files.

### rusty_sat_cli

Minimal CLI skeleton (~10 lines). Prints version and exits. Will be expanded to a full command-line tool.

---

## Key Design Patterns

### 1. Reader Trait

All readers implement a single `Reader` trait. The `Scene` orchestrates readers through this trait without knowing their internal format details:

```rust
// Scene calls:
let plan = scene.plan_reader_loads(wishlist, &[reader.inventory()?])?;
// Scene calls reader.load(id) for each planned dataset
```

### 2. Consuming APIs

Methods that take ownership (`self`) enable zero-copy transformations:

```rust
// Owned path: dataset is consumed, array is moved
let resampled = resample_dataset_owned(dataset, &target, options)?;

// Borrowed path: dataset is cloned internally
let resampled = resample_dataset(&dataset, &target, options)?;
```

### 3. Dual Precision

F32 for display memory efficiency, F64 for scientific precision:

```rust
// Display path (f32, ~half memory)
let ds = handler.load_calibrated_dataset(AhiCalibration::Reflectance)?;

// Scientific path (f64, full double precision)
let ds = handler.load_calibrated_dataset_f64(AhiCalibration::Reflectance)?;

// Reader-level control
let reader = AhiHsdReader::with_calibration(calib, handlers)?
    .with_output(AhiCalibrationOutput::ScientificF64);
```

### 4. Builder Pattern

Complex configuration uses the builder pattern:

```rust
let options = ResampleOptions::nearest_area()
    .with_radius_of_influence(5000.0)?
    .with_fill_value(f64::NAN);

let writer = FloatTiffWriter::default()
    .with_compression(TiffCompression::Deflate)
    .with_tiles(TiffTileOptions { width: 256, height: 256 });
```

### 5. Mask Propagation

`ValidityMask` is preserved through the entire pipeline: calibration → assembly → resampling → compositing → output. Masked pixels are excluded from auto-stretch range computation and rendered as fill values in final images.

---

## Error Handling

Single error type across the workspace:

```rust
pub enum RustySatError {
    Unsupported { feature: String },   // Not yet implemented
    InvalidInput { message: String },  // Bad user input or malformed data
    NotFound { item: String },         // Dataset/variable not found
    Ambiguous { message: String },     // Multiple matches when one expected
}
```

Factory methods: `RustySatError::unsupported(...)`, `::invalid_input(...)`, `::not_found(...)`, `::ambiguous(...)`.

Type alias: `pub type Result<T> = std::result::Result<T, RustySatError>`.

---

## Safety Guarantees

- `forbid(unsafe_code)` — no `unsafe` blocks anywhere in the workspace
- `deny(unwrap_used)` — all fallible operations use `?` or explicit error handling
- `deny(dbg_macro)` — no debug prints in committed code
- `deny(todo)` — incomplete features return `Err(Unsupported{...})` instead of panicking

---

## Bounds and Limits

| Limit | Value | Where |
|-------|-------|-------|
| Max HSD file size | 2 GB | `MAX_HSD_FILE_BYTES` |
| Initial header read | 4096 bytes | `INITIAL_HEADER_PREFIX_LEN` |
| Max YAML file size | 8 MB | `MAX_COMPOSITE_YAML_BYTES` |
| Max YAML nesting depth | 96 | `MAX_COMPOSITE_YAML_DEPTH` |
| Max config YAML file size | 8 MB | config module |
| Max config YAML nesting | 96 | config module |
