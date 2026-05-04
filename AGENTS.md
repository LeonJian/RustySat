# Rusty Sat Agent Guide

## Mission

Rusty Sat is a Rust-native, Satpy-compatible rewrite. The long-term goal is full Satpy capability: YAML-driven readers, composites, modifiers, resampling, corrections/enhancements, and writers that can produce corrected and resampled imagery efficiently.

Satpy compatibility has two layers:

1. **Behavior contract**: For the same supported input, Rusty Sat must produce the same dataset choices, metadata interpretation, dependency decisions, geometry, imagery, and writer output as Satpy. This is the correctness guarantee.
2. **API contract**: Public APIs must be idiomatic Rust — ownership/borrowing, zero-copy where possible, explicit mutation, and type-safe generics. Do not clone Python method signatures or xarray semantics literally. Translate Satpy concepts into Rust-native patterns.

Performance and memory efficiency are first-class requirements, not afterthoughts. Every implementation step should evaluate whether it can operate in-place, avoid unnecessary allocations, and use cache-friendly data layouts.

This project must move slowly and deliberately. Do not attempt to rewrite all of Satpy in one change. Every implementation step should be small, tested, and easy for the next agent to continue.

## Reference Code & High-Risk Areas

- `satpy/` is the Python Satpy reference implementation.
- `deps/` contains local reference dependencies: `pyorbital`, `pyresample`, `trollimage`, and `trollsift`.
- Treat these folders as read-only references unless the user explicitly asks otherwise.
- Before implementing each roadmap item, check the relevant Satpy and dependency source/docs first. Use `satpy/doc`, `deps/*/doc` or `deps/*/docs`, and the implementation modules as the behavior reference.

The following reference areas require especially careful parity work — do not rewrite blindly:

- Satpy `Scene`, `DataId`, `DataQuery`, dependency tree, YAML readers, composites, and resampling flow.
- Pyresample geometry and resampling algorithms.
- Trollsift filename parser and formatter.
- Trollimage `XRImage`, colormaps, alpha handling, and writer paths.
- Pyorbital TLE, orbital, astronomy, and scan geolocation math.

**Build risks**: HDF4 and HDF-EOS may require native libraries. Prefer a tiny, well-tested common file-handler abstraction before porting format-specific behavior. Add dependencies only when the corresponding substep starts, and record build assumptions. Python YAML tags must be parsed safely — never execute Python-like tags.

## Incremental Workflow

1. Read this file first.
2. Check the current implementation state below.
3. Pick the next unchecked roadmap item only.
4. Read the matching Python reference docs and code before designing Rust behavior.
5. Implement the smallest useful Rust capability.
6. Add focused tests.
7. Run `cargo check --workspace` and `cargo test --workspace`.
8. Update this file with completed work and known gaps.
9. Commit the completed step before moving to the next step.

Do not bundle unrelated roadmap items together. If a Satpy update introduces new behavior, track it as a separate task and implement it separately.

When a step is too large, split it into smaller lettered substeps **in this file first**. Implement the current substep to a quality stopping point, test it, update this file, and commit before continuing. Do not leave a lettered substep half-done unless it is explicitly marked blocked with the reason.

Every rewrite step must aim for Satpy-compatible results, not only similar-looking APIs. For behavior copied from Satpy or a dependency, cite the inspected reference paths in the implementation notes or tests when practical, and add parity tests that use representative inputs from the Python docs or tests.

## Git Rules

- The repository root should be a Git repository for Rusty Sat work.
- Commit every completed roadmap step or substep.
- Keep commits small and named after the completed step, for example `step 3a data query matching`.
- Do not commit generated build artifacts, `.DS_Store`, or `target/`.
- Do not commit the nested Python reference checkouts under `satpy/` or `deps/`; they are local reference material unless the user explicitly requests otherwise.
- Before each commit, run `cargo fmt --all -- --check`, `cargo check --workspace`, and `cargo test --workspace`.

## Status Markers

- `[ ]` not started
- `[~]` in progress
- `[x]` done
- `[!]` blocked
- `[@]` selected roadmap item for the current milestone

## Current Milestone Roadmap Marker

Use `[@]` in the roadmap itself to mark a not-yet-complete roadmap item that must be implemented for the active milestone. It is a status marker like `[ ]`, `[~]`, `[x]`, and `[!]`, not a suffix tag.

Example: if the active milestone is `M2-image` and it needs the PNG writer, mark the roadmap item as:

```text
- `[@]` W2: Simple image writer...
```

Milestone check rule:

1. Before starting a milestone, mark every needed roadmap item with `[@]`.
2. When implementing one selected roadmap item, change it from `[@]` to `[~]`.
3. When that roadmap slice is complete, change it to `[x]`.
4. Do not mark a milestone substep complete unless its roadmap slice is `[x]`.
5. Do not mark a milestone complete unless all roadmap items selected by `[@]` for that milestone have become `[x]`.

## Precision And Output Depth Policy

`f32` is acceptable for narrow first vertical slices and for cases where the inspected Trollimage/Satpy behavior explicitly converts integer stretch operations to `float32`. It is not the project-wide precision limit.

Rusty Sat must support:

- `f64` internal image/enhancement paths when source data or scientific correction precision requires it.
- In-place or consuming APIs for both `f32` and `f64` image buffers to avoid avoidable copies.
- 8-bit quicklook output for common PNG/JPEG workflows.
- 16-bit output for high-dynamic-range PNG/TIFF/GeoTIFF-style imagery.
- Float output for scientific GeoTIFF/CF workflows where preserving calibrated values matters more than display-ready scaling.

Do not force all image data through u8. `Image<u8>`/display output, `FloatImage<f32/f64>` enhancement buffers, and future writer-specific output dtypes should remain separate concepts.

## Completed Early Roadmap

- `[x]` Step 0: Repository orientation and rules.
- `[x]` Step 1: Rust workspace skeleton.
- `[x]` Step 2: Config system.
- `[x]` Step 3: `DataId` and `DataQuery` parity.
  - `[x]` Step 3a: typed `DataId` values, wavelength ranges, modifiers, and basic `DataQuery` matching.
  - `[x]` Step 3b: Satpy-like best-match preference sorting and ambiguity errors.
  - `[x]` Step 3c: compatibility tests from representative Satpy dataset IDs.
- `[x]` Step 4: `Scene` core and dependency graph.
  - `[x]` Step 4a: dependency graph data model, user-provided dataset leaves, and Scene removal semantics.
  - `[x]` Step 4b: Scene load request planning against available reader dataset IDs.
  - `[x]` Step 4c: dependency graph population for composites and modifiers.
- `[x]` Step 5: fake reader vertical slice.
  - `[x]` Step 5a: in-memory fake reader inventory and dataset loading.
- `[x]` Step 6: filename pattern parser compatible with `trollsift`.
  - `[x]` Step 6a: basic parser keys, parse, validate, compose, and globify for common Satpy filename patterns.
  - `[x]` Step 6b: trollsift custom compose conversions plus richer integer and fixed-point parsing.
  - `[x]` Step 6c: typed datetime values for common numeric strftime filename fields.
  - `[x]` Step 6d: remaining initial trollsift edge-case parity for partial compose and conversion errors.
- `[x]` Step 7: area definitions and YAML area loading.
  - `[x]` Step 7a: `AreaDefinition` metadata, shape/extent validation, and common Satpy/Pyresample YAML area loading.
  - `[x]` Step 7b: projection-unit resolution-derived shapes/extents for common Pyresample `create_area_def` inputs.
  - `[x]` Step 7c: swath definition coordinate storage/loading foundations.
- `[x]` Step 8: first real reader.
  - `[x]` Step 8a: Satpy-style reader YAML metadata parsing and dataset inventory generation.
  - `[x]` Step 8b: YAML-backed filename matching and file grouping using the existing filename pattern parser.
  - `[x]` Step 8c: first small file handler that loads real array values from an openly documented simple format.
- `[x]` Step 9: nearest resampling.
  - `[x]` Step 9a: projection-coordinate nearest resampling from one `AreaDefinition` grid to another.
  - `[x]` Step 9b: radius/fill behavior parity tests against representative Pyresample cases.
  - `[x]` Step 9c: swath-to-area nearest foundations.
- `[x]` Step 10: first writer.
  - `[x]` Step 10a: first grayscale image writer using binary PGM output from `DataGrid`.

## Long-Term Roadmap

### P0: Foundation Completion

Highest priority. Complete these before major reader/composite/writer expansion.

- `[x]` P0-1: DataArray/DataGrid foundation.
  - `[~]` P0.1.1: Replace f64-only 2D `DataGrid` with a Rust-native `DataArray`/`ArrayD<T>` style model supporting numeric dtypes such as `f32`, `f64`, `u8`, `u16`, and `i16`.
    - `[x]` P0.1.1a: Add owned generic `DataArray<T>`, runtime `AnyDataArray`, dtype markers, and keep `DataGrid = DataArray<f64>` compatibility for existing vertical slices.
    - `[x]` P0.1.1b: Migrate `Dataset` storage from f64-only `DataGrid` to runtime typed arrays while preserving f64 grid helpers for resampling/writer code.
    - `[x]` P0.1.1c: Audit reader/resampler/writer APIs and extend the first image writer to accept runtime-typed numeric arrays directly.
    - `[x]` P0.1.1d: Keep nearest resampling f64-only until mask/chunk foundations clarify generic numeric output behavior.
  - `[x]` P0.1.2: Support 1D/2D/3D/4D shapes with named dimensions compatible with xarray-style `DataArray` concepts.
    - `[x]` P0.1.2a: Add validated dimension names to `DataArray`; default 1D to `y`, 2D to `y,x`, 3D to `bands,y,x`, and 4D to `time,bands,y,x`.
    - `[x]` P0.1.2b: Add dimension-aware shape helpers and enforce image writer dimensional expectations through `y,x` names.
    - `[x]` P0.1.2c: Keep nearest resampling shape-based until mask/chunk semantics are implemented.
  - `[x]` P0.1.3: Add an independent mask model; represent fill/missing values separately from `NaN`, with efficient bitmask storage where practical.
    - `[x]` P0.1.3a: Add packed `ValidityMask` storage to `DataArray` and make PGM output fill masked pixels while ignoring them for autoscale.
    - `[x]` P0.1.3b: Propagate source masks through current nearest area/swath resampling; outside-radius pixels still use fill values until a broader policy is added.
    - `[x]` P0.1.3c: Define current nearest fill-vs-mask policy with default fill-value behavior and opt-in masked-missing output.
  - `[x]` P0.1.4: Add lazy chunk foundations for Dask-like chunked loading without copying whole products into memory.
    - `[x]` P0.1.4a: Add validated chunk-shape metadata to `DataArray` and runtime typed arrays without introducing lazy IO yet.
    - `[x]` P0.1.4b: Add a lazy chunk source abstraction for deferred file-backed array loading.
    - `[x]` P0.1.4c: Teach readers/resamplers/writers to preserve or consume chunked arrays incrementally.
      - `[x]` P0.1.4c1: Teach the first PGM writer to consume 2D lazy arrays by reading chunks into one y-stripe at a time.
      - `[x]` P0.1.4c2: Add reader-side lazy chunk source fixtures before production NetCDF/HDF handlers.
      - `[x]` P0.1.4c3: Add resampler-side lazy input consumption for current f64 nearest area resampling. Chunk-preserving lazy output remains future work once generic numeric resampling is ready.
  - `[x]` P0.1.5: Replace flat metadata-only assumptions with nested metadata values compatible with xarray attrs dictionaries.
    - `[x]` P0.1.5a: Add nested `MetadataValue` attrs to `Dataset` while preserving legacy flat string metadata helpers.
    - `[x]` P0.1.5b: Migrate current reader/resampler/writer vertical slices to preserve nested attrs where they currently copy flat metadata.
    - `[x]` P0.1.5c: Add YAML/NetCDF-style metadata value parsing for lists, maps, booleans, numbers, and fill values.
  - `[x]` P0.1.6: Attach named coordinates and coordinate axes, initially x/y and later lon/lat/time/band coordinates.
    - `[x]` P0.1.6a: Add numeric coordinate storage to `DataArray`/`AnyDataArray` and attach destination x/y projection axes from current area resampling.
    - `[x]` P0.1.6b: Preserve non-x/y coordinates through current resampling paths following Satpy's `resample.base._update_resampled_coords` behavior.
    - `[x]` P0.1.6c: Add swath longitude/latitude coordinate attachment and reader-driven coordinate dataset linking.
- `[x]` P0-2: CRS and projection system.
  - `[x]` P0.2.1: Add `ProjCrs` wrapper and choose projection dependency strategy after inspecting Pyresample/pyproj behavior and Rust crate build requirements.
  - `[x]` P0.2.2: Add forward/inverse coordinate transformation APIs.
  - `[x]` P0.2.3: Parse, validate, and normalize proj4 strings beyond the current string-map metadata.
- `[x]` P0-3: `DataId`/`DataQuery` completion.
  - `[x]` P0.3.1: Complete modifier-chain matching with shortest-path/preference behavior after inspecting Satpy modifier dependency logic.
  - `[x]` P0.3.2: Add `ancillary_variables` query/filter support based on Satpy `anc_vars.py` behavior.

### R: Readers

- `[ ]` R0: Reader core framework.
  - `[ ]` R0.1: `BaseFileHandler` trait, lifecycle fields, registration, and file-handler errors.
  - `[ ]` R0.2: NetCDF common base: metadata collection, groups/variables/global attrs, scale/offset, and xarray-like variable access.
  - `[ ]` R0.3: HDF5 common base including object/region reference handling.
  - `[ ]` R0.4: HDF4 common base for MODIS and other legacy products.
  - `[ ]` R0.5: HDF-EOS base.
  - `[ ]` R0.6: HRIT headers, image navigation, prologue/epilogue parsing.
  - `[ ]` R0.7: Instrument/product base modules for EUMETSAT, VII/VIIRS, SEVIRI, ABI, FCI, FY-4, LI, Landsat, HRIT JMA, and VIIRS/ATMS SDR.
  - `[ ]` R0.8: YAML reader completion: safe Python-tag representation, FileHandler instantiation, composite IDs, groups/bound groups, and delayed loading.
  - `[ ]` R0.9: Filename matching and grouping: `group_files`, multi-time grouping, and advanced matching.
- `[ ]` R1: GEO readers, including ABI, AHI, AMI, SEVIRI, FCI, AGRI, HRIT, GOES Imager, GOCI-II, INSAT, and JMA HRIT.
- `[ ]` R2: LEO L1B readers, including VIIRS, MODIS, AVHRR, EPS, AAPP, OLCI, SLSTR, FY-3, ATMS, MetOp-SG, EarthCARE, PACE, MAIA, SCMI, and Satpy CF re-read.
- `[ ]` R3: L2 product readers, including VIIRS L2/EDR, CLAVR-x, CMSAF, MIRS, NUCAPS, ACSPO, AMSR2, CALIOP, NWC SAF, IASI, TROPOMI, SeaDAS, Sentinel SAFE/SAR/MSI, GeoCAT, MERIS, OLCI, and SLSTR.
- `[ ]` R4: Microwave, radio, lightning, and auxiliary readers.
- `[ ]` R5: Special format readers, including GRIB, BUFR, generic image, ISCCP-NG, GMS VISSR, XML, and microwave channel definitions.

### S: Resampling

- `[ ]` S1: Pyresample core geometry completion: `BaseDefinition`, coordinate/grid/projection definitions, full `AreaDefinition`, stacked/dynamic areas, full `SwathDefinition`, spherical coordinates, polygons/arcs, overlap utilities, grid filters, and spherical area math.
- `[ ]` S2: KD-tree nearest resampling: tree creation, neighbour info, sampled output, radius calculation, Gaussian weights, great-circle distances, chunked parallel processing, `BaseResampler`, and full swath-to-grid support.
- `[ ]` S3: Bilinear, cubic, and spline interpolation.
- `[ ]` S4: Bucket resampling: average, sum, count, fraction, and multi-dimensional buckets.
- `[ ]` S5: EWA/Fornavy/LLS2 resampling wrappers.
- `[ ]` S6: Native resampler: repeat, aggregate, and native-resolution pipelines.
- `[ ]` S7: Resampling pipeline helpers: `resample_dataset`, resampler preparation, data reduction, slicers, image containers, cropping, and CRS cross-projection resampling.

### I: Image And Enhancement

- `[~]` I1: Trollimage-like `XRImage` core: construction, dimension correction, stretch modes, gamma, invert, finalize, alpha, mode conversion, colorize, stack/merge, scaling history, and save helpers.
  - `[x]` I1-m2a: Owned image buffer construction and mask-aware luma finalization foundation for M2-image.
  - `[x]` I1-m2b: Crude stretch foundation with in-place float normalization and scale/offset history for M2-image.
  - `[x]` I1-m2prec: Add generic or parallel `f64` float image/enhancement path; keep `f32` only as an optimization/compatibility dtype, not the only enhancement dtype.
  - `[ ]` I1-next: Broader XRImage parity: band-aware dimensions, gamma, invert, alpha/finalize policy, mode conversion, and save helpers.
- `[ ]` I2: Colormap system: validation, colorize/palettize, RGB/RGBA conversion, merging, reversing/ranging, export, and YAML loading.
- `[ ]` I3: Legacy `Image` compatibility where needed.
- `[ ]` I4: Color-space conversion and utility ramps.
- `[ ]` I5: Satpy enhancer framework and YAML enhancement chains.
- `[ ]` I6: Instrument enhancements for ABI, AHI, VIIRS, MIMIC, and enhancement YAML data.
- `[ ]` I7: Convolution filters and overlays.

### C: Composites

- `[~]` C0: Composite core: `CompositeBase`, generic/single-band/RGB compositors, enhance-to-dataset helpers, band handling, and mode checks.
  - `[x]` C0-m2c: RGB compositor vertical slice for three matching single-band datasets, with Satpy-like common-channel mask behavior.
  - `[ ]` C0-next: Broader composite parity: `CompositeBase`, generic/single-band compositors, enhance-to-dataset helpers, band/mode checks, metadata combination, optional prerequisites, and YAML integration.
- `[ ]` C1: Arithmetic composites such as NDVI/EVI/diff/ratio/sum and channel-operation compositors.
- `[ ]` C2: Spectral composites, weighted blends, and band replacement/mapping.
- `[ ]` C3: SEVIRI composites and YAML parity.
- `[ ]` C4: ABI, AHI, AMI, AGRI, and VIIRS composites plus YAML data.
- `[ ]` C5: Advanced composites: masks, resolution-aware composites, lookup tables, fill, auxiliary data, cloud products, lightning overlays, SAR, and config loading.

### M: Modifiers

- `[ ]` M1: Modifier base and spectral modifiers.
- `[ ]` M2: Atmospheric modifiers: Rayleigh reflectance, atmospheric correction, and CO2 correction.
- `[ ]` M3: Geometry modifiers: sun-zenith correction/reduction, solar path length, angles, and parallax.
- `[ ]` M4: CREFL algorithms and helpers.
- `[ ]` M5: Spatial filters: Gaussian, median, sharpen/blur/edge, and morphology.

### W: Writers

- `[ ]` W1: Writer framework completion: writer trait, image-writer base, extension-based factory, and writer YAML config.
- `[@]` W2: Simple image writer: PNG/JPEG output, format detection, transparency/fill/mode handling, PNG metadata, and 8-bit/16-bit PNG output paths.
- `[ ]` W3: GeoTIFF writer: CRS tags, Cloud Optimized GeoTIFF behavior, GDAL metadata, pixel scale/tie point, and float32 support.
  - `[ ]` W3-hdr: Add 16-bit integer and float HDR/scientific GeoTIFF output policy, including scale/fill handling.
- `[ ]` W4: NINJO, MI, and AWIPS writers.
- `[ ]` W5: CF NetCDF writer: dataset saving, CF dimensions/variables/global attrs, geolocation coordinates, `da2cf`, encoding, compression, and chunks.

### O: Orbit And Astronomy

- `[ ]` O1: TLE parsing, SGP4 binding strategy, TLE fetching, and cache management.
- `[ ]` O2: Orbital propagation and observer look geometry.
- `[ ]` O3: Astronomy functions: Julian days, GMST/LMST, solar position, alt/az/SZA, sun-earth distance correction, and observer position.
- `[ ]` O4: Scan geometry geolocation, pixel computation, quaternion rotation, and Earth intersection.
- `[ ]` O5: AVHRR GCP geolocation.
- `[ ]` O6: Instrument scan geometry definitions and pyorbital config helpers.

### SC: Scene Integration

- `[ ]` SC1: Scene construction/lifecycle: from readers/files, load, available datasets, start/end time, sensors, and missing datasets.
- `[ ]` SC2: Scene spatial operations: finest/coarsest area, crop, aggregate, slice, copy, same-area/proj checks, and area iteration.
- `[ ]` SC3: Scene resampling pipeline and integration with all resamplers.
- `[@]` SC4: Scene save/show/to-xarray APIs.
- `[ ]` SC5: Composite/modifier Scene integration and dependency execution.
- `[ ]` SC6: Multi-scene support and optional animation output.

### CLI, YAML, And Tests

- `[ ]` CLI: `info`, list commands, image generation, batch processing, optional serve mode, help/logging, and error messages.
- `[ ]` Y: Full YAML system: reader/composite/enhancement config conversion, safe Python-tag handling, search-path merging, and complete `areas.yaml`.
- `[ ]` T: Test infrastructure: Python Satpy golden-output generation, minimal fixtures, end-to-end tests, Criterion benchmarks, property tests, and CI.

## Milestones

### Milestone Roadmap Dependencies

Before starting or closing a milestone, check this table and update both the milestone and the referenced roadmap status. Milestone substeps are only execution slices; the source of truth for project completeness remains the roadmap above.

| Milestone | Must Finish Roadmap Items | Current Roadmap Gaps |
|-----------|---------------------------|----------------------|
| M1 | Step0-10a early vertical slice | Done |
| M2-foundation | P0-1, P0-2, P0-3 | Done |
| M2-image | I1 partial (`XRImage` construction, crude stretch, f32/f64 enhancement buffers, finalize-to-u8 basics), C0 partial (generic/RGB compositor), W2 partial (PNG/simple image writer with 8-bit and planned 16-bit path), SC4 partial (`Scene.save_dataset`) | Roadmap currently selected with `[@]`: W2, SC4 |
| M3-reader | R0.1, R0.2, R0.8 partial, R1.1 ABI L1B partial, SC1 load path, W2 output path | Blocked until M2-image produces PNG output |
| M4-resample | S1 partial, S2 partial, R1.13 or another NetCDF reader slice, SC3 resampling integration | Needs real CRS transform/KD-tree foundations beyond current nearest metadata-only path |
| M5-enhance-composite | I1 broader stretch/finalize, I5 enhancer framework, C1 arithmetic, C2 spectral | Needs M2-image primitives first |
| M6-resampling-full | S1-S7 | Needs M4 resampling architecture first |
| M7-writers-composites-full | W1-W5 and C0-C5 | Needs M2/M5 output and composite foundations |
| M8-readers-modifiers-orbit | R0-R5, M1-M5, O1-O6 | Needs reader framework and test infrastructure |
| M9-production | SC1-SC6, CLI, Y, T | Needs all prior functional milestones |

- `[x]` M1: Early vertical slice: text grid data can become a grayscale PGM image.
- `[x]` M2-foundation: Complete P0 DataArray/DataGrid, mask, metadata, coordinates, and CRS foundations. This supersedes the earlier idea of jumping directly to PNG/RGB composite work.
- `[~]` M2-image: After P0 foundations, implement partial image model, crude stretch, RGB composite, PNG writer, and `Scene.save_dataset` so self-made data can produce color PNG.
  - `[x]` M2-image-a: Review Trollimage `XRImage`, Satpy `PillowWriter`, and `GenericCompositor`; add an owned u8 image buffer with mask-aware luma conversion from datasets. Roadmap: I1.
  - `[x]` M2-image-b: Add crude stretch foundations for float image data following Trollimage per-band min/max behavior. Roadmap: I1.
  - `[x]` M2-image-b2: Add `f64` image/enhancement buffer path or generic float buffer abstraction before relying on this image foundation for scientific corrections. Roadmap: I1-m2prec.
  - `[x]` M2-image-c: Add RGB compositor vertical slice for three matching single-band datasets. Roadmap: C0.
  - `[ ]` M2-image-d: Add PNG writer using the Rust `image` crate, with format detection and luma/RGB/RGBA support. Roadmap: W2.
  - `[ ]` M2-image-d2: Add 16-bit PNG/HDR output path or a clearly typed writer interface that can preserve 16-bit display output. Roadmap: W2.
  - `[ ]` M2-image-e: Add `Scene.save_dataset` wrapper for self-made datasets and generated images. Roadmap: SC4.
- `[ ]` M3-reader: NetCDF base plus ABI L1B reader and Scene load path sufficient for a GOES ABI sample to output a basic image.
- `[ ]` M4-resample: FCI/another NetCDF reader plus CRS/KD-tree foundations sufficient for real projection-aware resampling.
- `[ ]` M5-enhance-composite: Broaden image enhancement and arithmetic/spectral composites.
- `[ ]` M6-resampling-full: Complete major resampler families and performance work.
- `[ ]` M7-writers-composites-full: GeoTIFF, CF, and broader composite parity.
- `[ ]` M8-readers-modifiers-orbit: Expand real readers, modifiers, and orbit/geolocation support.
- `[ ]` M9-production: Complete Scene API, CLI, QA, benchmarks, and CI hardening.

## Workspace Architecture

- `rusty_sat_core`: public API foundations, shared errors, `Scene`, `Dataset`, `DataId`, `DataQuery`, and later dependency graph types.
- `rusty_sat_config`: YAML config loading and future Satpy-compatible search path behavior.
- `rusty_sat_readers`: reader traits, fake/test readers, and later YAML-backed readers.
- `rusty_sat_resample`: area/swath types and future resampling algorithms.
- `rusty_sat_composites`: compositor/modifier traits and dependency integration.
- `rusty_sat_image`: image model, color maps, and enhancement pipeline.
- `rusty_sat_writers`: writer traits and later PNG, GeoTIFF, and CF-style output.
- `rusty_sat_cli`: thin command-line wrapper around the library crates.

## Coding Rules

- Use explicit `Result<T, RustySatError>` returns for fallible operations.
- Keep incomplete behavior explicit with placeholder errors, not silent defaults.
- Keep public types documented enough for future agents to understand intent.
- Avoid adding heavy dependencies until the relevant roadmap step needs them.
- Split growing modules into focused files before they become hard to review.
- Keep tests close to the module they validate; move larger fixtures or broad parity suites into separate test files.
- Prefer APIs that mutate in place or consume owned inputs when the caller no longer needs the source data. Avoid cloning large arrays, masks, metadata, or coordinate maps unless the borrowed API contract requires it; document unavoidable allocations in tests or implementation notes.

## Upstream Satpy Tracking

When Satpy changes upstream:

1. Identify whether the change is core, config, reader, writer, composite, modifier, resampler, image, or CLI behavior.
2. Add a separate roadmap/task entry before implementation.
3. Add a compatibility test or fixture whenever possible.
4. Implement the new behavior separately from unrelated work.
5. Record the supported Satpy behavior in this file or a future `UPSTREAM_COMPAT.md`.

## Testing Expectations

Every step should leave the workspace passing:

```sh
cargo check --workspace
cargo test --workspace
```

Early tests should focus on construction and API shape. Later tests should compare Rusty Sat output against Python Satpy on small fixtures.

## Current Implementation State

> **UPDATE RULE: After completing any roadmap item, update this section immediately.**
> Only list each crate's **capability boundaries** — what it CAN and CANNOT do right now.
> Do NOT re-describe completed roadmap steps (those are already tracked in the roadmap above).
> Keep each entry to 2-4 lines max. Focus on: data types handled, operations supported, and the hard limit that blocks the next step.

### rusty_sat_core

| Can | Cannot |
|-----|--------|
| Store owned nD arrays (`f32`/`f64`/`u8`/`u16`/`i16`) via `DataArray<T>` | Zero-copy from file memory maps; all data is owned `Vec<T>` |
| Runtime-typed `AnyDataArray` with method dispatch across 5 variants | In-place mutation of array values (no `&mut self` transform API) |
| Named dimensions (1D–4D defaults), coordinates (1D/2D/scalar), packed `ValidityMask`, `ChunkShape` | Real lazy loading; `LazyDataArray` is contract-only, no file-backed source |
| `DataId`/`DataQuery` matching, scoring, best-match, ambiguity detection, ordered modifier-chain prefix matching, less-modified query creation, and ancillary-variable dataset filtering | Composite/modifier execution from these queries |
| `Dataset` dual metadata (flat `BTreeMap<String,String>` + nested `MetadataValue` attrs) | Single-source metadata; `insert_metadata()` still writes to BOTH maps (transitional) |
| `Dataset` typed ancillary variable links by `DataId`, with find/replace helpers modeled after Satpy `anc_vars.py` | Embedding full ancillary arrays inside attrs; only typed links are stored |
| `Scene` insert/remove datasets, plan reader loads, register composites/modifiers | Actual composite/modifier execution, resampling delegation, save/show |

### rusty_sat_config

| Can | Cannot |
|-----|--------|
| Load Satpy default config path (`satpy/satpy/etc`) | Format-specific config (reader YAML, composite YAML, enhancement YAML) |
| Support `RUSTY_SAT_CONFIG_PATH` / `SATPY_CONFIG_PATH` env vars | Config search path merging, YAML recursive merge beyond basics |
| Component config lookup (readers, writers, composites, enhancements) | |

### rusty_sat_readers

| Can | Cannot |
|-----|--------|
| Parse filename patterns with `FilenamePattern` (trollsift-compatible: keys, parse, compose, globify, typed datetimes) | Full trollsift parity (every custom compose/conversion edge case) |
| Parse Satpy reader YAML metadata via `YamlMetadataReader` (`reader`/`file_types`/`datasets` sections, Python tags as `MetadataValue`) | Safe YAML tag deserialization into typed structs; tags are currently stored as metadata values |
| `FakeReader` in-memory inventory + dataset loading for Scene planning tests | Real satellite file I/O (NetCDF/HDF/GeoTIFF/etc.) |
| `TextGridReader` reads plain text numeric grids + YAML metadata; provides `TextGridChunkSource` lazy fixture | Production file handlers; `yaml_reader` can inventory datasets but cannot load array data |

### rusty_sat_resample

| Can | Cannot |
|-----|--------|
| `AreaDefinition`: id, projection, shape, extent, pixel-size helpers, YAML loading, projection-unit resolution derivation | Stacked/dynamic areas, spherical polygon math, overlap utilities |
| `SwathDefinition`: dimension-only or lon/lat coordinate-backed swaths, WGS84 CRS convention, YAML loading | Real geocentric resolution, aggregation, boundary extraction |
| `ProjCrs`: WGS84/PROJ/EPSG parsing, normalization (numeric canonicalization, `latlong`→`longlat`, `+init=EPSG:`→`epsg`), identity-only transforms | Real cross-CRS transforms; backend is `MetadataOnly` |
| `Coordinate2D` finite-validated coordinate + transform API (identity for geographic/same-CRS) | Projected or cross-CRS forward/inverse transforms (returns `Unsupported`) |
| `NearestAreaResampler`: area-to-area and swath-to-area nearest, radius of influence, fill value, mask propagation, coordinate preservation, lazy input consumption | KD-tree acceleration, CRS transforms, anti-meridian handling, geocentric distances, multi-band, chunk-preserving lazy output |

### rusty_sat_composites

| Can | Cannot |
|-----|--------|
| Execute `RgbCompositor` for three matching 2D single-band runtime-typed datasets into a band-major `bands,y,x` f64 dataset with Satpy-like common-channel mask behavior | Full `CompositeBase`/`GenericCompositor` parity, metadata combination, optional prerequisites, YAML composite loading, or Scene dependency execution |
| Define `CompositeRecipe` and `ModifierRecipe` in `rusty_sat_core` | Execute registered composite/modifier recipes through `Scene` |

### rusty_sat_image

| Can | Cannot |
|-----|--------|
| Store owned u8 `Image` pixels and generic owned `FloatImage<f32/f64>` pixels for Luma/RGB/RGBA; construct luma images from 2D runtime-typed datasets; apply in-place crude stretch with scale/offset history | 16-bit/HDR finalization, broader XRImage parity: gamma/invert, alpha/finalize policy, colorize, mode conversion, or save helpers |

### rusty_sat_writers

| Can | Cannot |
|-----|--------|
| `PgmWriter` write binary PGM (P5) from `DataGrid`, `AnyDataArray`, or `LazyDataArray<T>`; autoscale, fill value, mask-aware | PNG/JPEG output, GeoTIFF, CF NetCDF, color image output |
| Lazy PGM writes read chunks into one y-stripe at a time (incremental) | Single-pass autoscale+write (currently reads chunks twice: autoscale then write) |

### rusty_sat_cli

| Can | Cannot |
|-----|--------|
| (Skeleton only) | Any CLI commands; not yet implemented |

### Known Inefficiencies (to address in future roadmap steps)

- `Dataset::insert_metadata()` writes to both `self.metadata` and `self.attrs` — transitional dual-write; plan to remove flat `metadata` map entirely.
- Area nearest resampling now has a consuming `resample_area_nearest_owned` path that moves source values, mask, and preserved coordinates. It still allocates a new output grid because the destination shape and sampling order may differ; future same-shape or in-place kernels should reuse capacity where mathematically safe.
- `AnyDataArray` dispatch uses exhaustive 5-arm match on every method — correct but verbose; a macro could reduce boilerplate.
- `SourceChunkCache` in nearest resampler is bounded with a small FIFO cache. Future work should make cache sizing configurable and consider LRU/tile traversal for better hit rates on large scenes.
- `autoscale_lazy` in PGM writer reads all chunks twice (once for min/max, once for write) — should be single-pass with stripe-level caching.
