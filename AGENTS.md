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
- Roadmap-only planning edits stay on the current branch and do not require a new feature branch.
- Start each implementation feature or lettered implementation substep on a fresh branch named after the feature itself, for example `w2-m2d2-16-bit-png-hdr`.
- Do not merge a completed feature branch immediately. First start the next implementation feature on its own new feature-named branch, then merge the previously completed branch after the user has had a chance to review that completed branch.
- Before starting a new feature branch, confirm the previous branch is committed and the current roadmap status in this file is updated.
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

- `[~]` R0: Reader core framework.
  - `[ ]` R0.1: `BaseFileHandler` trait, lifecycle fields, registration, and file-handler errors.
  - `[~]` R0.2: NetCDF common base: metadata collection, groups/variables/global attrs, scale/offset, and xarray-like variable access.
    - `[x]` R0.2-m4d1: Add Satpy-style NetCDF metadata index foundations for groups, variables, dimensions, global attrs, object attrs, and `required_netcdf_variables` replacement expansion without selecting a native NetCDF backend yet.
    - `[x]` R0.2-m4d2a: Add a Satpy-like NetCDF file-handler facade over the metadata index, including filename/filetype info, full-vs-required metadata collection, typed lookup helpers, and backend-neutral metadata source wiring.
    - `[x]` R0.2-m4d2b: Add backend-neutral NetCDF variable data loading from a `NetCdfDataSource`, array shape/dimension validation against metadata, and a first FCI L1C measured-channel counts loader with valid-range/fill masking.
    - `[x]` R0.2-m4d2c: Add a documented YAML fixture-backed `NetCdfFixtureSource` for real file-backed tests without native NetCDF/HDF build requirements; this is the adapter contract for the later native backend.
    - `[ ]` R0.2-next: Add native NetCDF/HDF adapter, real auto mask/scale handling, xarray-like variable access, and broader FCI/NetCDF data loading.
  - `[ ]` R0.3: HDF5 common base including object/region reference handling.
  - `[ ]` R0.4: HDF4 common base for MODIS and other legacy products.
  - `[ ]` R0.5: HDF-EOS base.
  - `[ ]` R0.6: HRIT headers, image navigation, prologue/epilogue parsing.
  - `[ ]` R0.7: Instrument/product base modules for EUMETSAT, VII/VIIRS, SEVIRI, ABI, FCI, FY-4, LI, Landsat, HRIT JMA, and VIIRS/ATMS SDR.
  - `[ ]` R0.8: YAML reader completion: safe Python-tag representation, FileHandler instantiation, composite IDs, groups/bound groups, and delayed loading.
  - `[ ]` R0.9: Filename matching and grouping: `group_files`, multi-time grouping, and advanced matching.
- `[~]` R1: GEO readers, including ABI, AHI, AMI, SEVIRI, FCI, AGRI, HRIT, GOES Imager, GOCI-II, INSAT, and JMA HRIT.
  - `[x]` R1-m4d3: Add first FCI L1C fixture-backed reader integration that exposes counts dataset IDs, loads measured-channel counts through the NetCDF handler, and participates in Scene load planning.
- `[ ]` R2: LEO L1B readers, including VIIRS, MODIS, AVHRR, EPS, AAPP, OLCI, SLSTR, FY-3, ATMS, MetOp-SG, EarthCARE, PACE, MAIA, SCMI, and Satpy CF re-read.
- `[ ]` R3: L2 product readers, including VIIRS L2/EDR, CLAVR-x, CMSAF, MIRS, NUCAPS, ACSPO, AMSR2, CALIOP, NWC SAF, IASI, TROPOMI, SeaDAS, Sentinel SAFE/SAR/MSI, GeoCAT, MERIS, OLCI, and SLSTR.
- `[ ]` R4: Microwave, radio, lightning, and auxiliary readers.
- `[ ]` R5: Special format readers, including GRIB, BUFR, generic image, ISCCP-NG, GMS VISSR, XML, and microwave channel definitions.

### S: Resampling

- `[~]` S1: Pyresample core geometry completion: `BaseDefinition`, coordinate/grid/projection definitions, full `AreaDefinition`, stacked/dynamic areas, full `SwathDefinition`, spherical coordinates, polygons/arcs, overlap utilities, grid filters, and spherical area math.
  - `[x]` S1-m4a: Add shared geometry definition foundations (`GeometryDefinition`, `CoordinateDefinition`, and `GridDefinition`) after inspecting Pyresample `geometry.py`.
  - `[x]` S1-m4b: Add projection-definition/area helpers (`ProjectionDefinition`, pixel size, upper-left pixel center, pixel offsets, and projection-coordinate iterators) needed by KD-tree setup.
  - `[ ]` S1-next: Add stacked/dynamic areas and geocentric/spherical geometry pieces needed by KD-tree resampling.
- `[~]` S2: KD-tree nearest resampling: tree creation, neighbour info, sampled output, radius calculation, Gaussian weights, great-circle distances, chunked parallel processing, `BaseResampler`, and full swath-to-grid support.
  - `[x]` S2-m4c: Add Pyresample-style neighbour-info data model and projection-coordinate area-to-area nearest neighbour query foundation.
  - `[x]` S2-m4c2: Add nearest sampled-output helper from `NeighbourInfo`, including fill-vs-mask missing handling and source-mask propagation.
  - `[x]` S2-m4f1: Add a dependency-free exact 2D KD point index and route lon/lat swath nearest queries through it, preserving radius-of-influence and source-mask behavior.
  - `[x]` S2-m6g: Add area-to-area top-k neighbour-info generation and weighted multi-neighbour sampling foundations after inspecting Pyresample `get_sample_from_neighbour_info`.
  - `[x]` S2-m6j: Add single-neighbour swath-to-area neighbour-info generation using the existing exact 2D KD index, Pyresample-like lon/lat validity masks, and existing nearest sampling helpers.
  - `[ ]` S2-next: Add geocentric/great-circle KD distances, chunked parallel processing, multi-neighbour KD-tree acceleration, uncertainty outputs, and multi-neighbour swath neighbour-info generation.
- `[~]` S3: Bilinear, cubic, and spline interpolation.
  - `[x]` S3-m6c: Add first same-projection 2D area-to-area bilinear resampler with fill/mask missing policy and pipeline selection after inspecting Satpy/Pyresample bilinear paths.
  - `[ ]` S3-next: Add irregular swath bilinear coefficients, multi-neighbour lookup, cubic interpolation, spline interpolation, and higher-dimensional/band-aware sampling.
- `[~]` S4: Bucket resampling: average, sum, count, fraction, and multi-dimensional buckets.
  - `[x]` S4-m6d: Add first lon/lat swath-to-area bucket average, sum, and count foundations after inspecting Satpy/Pyresample bucket resamplers.
  - `[x]` S4-m6e: Add manual-category bucket fraction output as a `categories,y,x` `DataArray<f64>` plus skipna=false sum guardrail tests.
  - `[x]` S4-m6h: Add bucket average/sum/count preparation through the typed resampling pipeline using explicit swath source geometry.
  - `[x]` S4-m6k: Add explicit-category bucket fraction preparation through the typed resampling pipeline.
  - `[x]` S4-m6l: Add automatic finite/unmasked category discovery for bucket fraction direct and pipeline paths.
  - `[ ]` S4-next: Add projected target backends, multidimensional/band-aware buckets, chunked execution, and Scene-level preparation integration.
- `[~]` S5: EWA/Fornavy/LLS2 resampling wrappers.
  - `[x]` S5-m6f: Add a dependency-free lon/lat swath-to-geographic-area EWA-style weighted accumulation foundation after inspecting Satpy/Pyresample EWA and Fornav wrappers.
  - `[x]` S5-m6h: Add EWA preparation through the typed resampling pipeline using explicit swath source geometry and required radius-of-influence.
  - `[ ]` S5-next: Add real Fornav/LLS2 parity, scan geometry and `rows_per_scan`, maximum-weight mode, chunked execution, multi-band sampling, and production Pyresample-compatible weighting.
- `[~]` S6: Native resampler: repeat, aggregate, and native-resolution pipelines.
  - `[x]` S6-m6a: Add 2D native repeat/aggregate-mean foundations and a `NativeResampler` for integer y/x scale factors after inspecting Satpy `resample/native.py`.
  - `[x]` S6-m6i: Add higher-dimensional f64 native repeat/aggregate over named `y,x` axes while preserving non-spatial dimensions and coordinates.
  - `[x]` S6-m6m: Add runtime-typed native resampling: identity/repeat preserve numeric dtype, aggregate-mean promotes to `f64`, and `NativeResampler` accepts `AnyDataArray` datasets.
  - `[ ]` S6-next: Add chunked/lazy native execution and full Satpy native-resampler integration with Scene area-choice helpers.
- `[~]` S7: Resampling pipeline helpers: `resample_dataset`, resampler preparation, data reduction, slicers, image containers, cropping, and CRS cross-projection resampling.
  - `[x]` S7-m6b: Add typed `prepare_resampler`, `resample_dataset`, and `resample_dataset_owned` helpers for current nearest/native resamplers after inspecting Satpy `resample/base.py`.
  - `[x]` S7-m6h: Add explicit `SourceGeometry` pipeline preparation for current swath-based bucket and EWA resamplers while preserving area-only convenience APIs.
  - `[x]` S7-m6k: Add typed pipeline method/options for explicit-category bucket fraction output.
  - `[x]` S7-m6n: Add explicit `ResamplerCache` for prepared resamplers keyed by source geometry, destination area, and options, including cached borrowed/owned dataset helpers.
  - `[x]` S7-m6o: Add Pyresample-style lon/lat boundary and grid validity filtering foundations for coarse swath data reduction.
  - `[x]` S7-m6p: Add same-projection `AreaDefinition` slicing and crop-source-area helpers after inspecting Pyresample `get_area_slices`, `slicer.py`, and Satpy `_reduce_data`.
  - `[x]` S7-m6q: Add dtype-preserving `y`/`x` `DataArray`/`AnyDataArray` and `Dataset` slicing helpers for Satpy-style data reduction, including consuming paths.
  - `[x]` S7-m6r: Add combined area dataset reduction helpers that return the sliced dataset, reduced source area, and slice metadata with Satpy-like shape validation.
  - `[x]` S7-m6s: Add opt-in reduction-aware area resampling pipeline helpers that crop data before preparing/resampling, with cached and consuming variants.
  - `[x]` S7-m6t: Add Satpy-style data-reduction options (`reduce_data` and `shape_divisible_by`) to `ResampleOptions`, including cache-key separation and reduced-area helper integration after inspecting Satpy `Scene._reduce_data`.
  - `[ ]` S7-next: Add swath/cross-projection slicers, image containers, CRS cross-projection resampling, and automatic source-area/swath lookup from dataset attrs.

### I: Image And Enhancement

- `[~]` I1: Trollimage-like `XRImage` core: construction, dimension correction, stretch modes, gamma, invert, finalize, alpha, mode conversion, colorize, stack/merge, scaling history, and save helpers.
  - `[x]` I1-m2a: Owned image buffer construction and mask-aware luma finalization foundation for M2-image.
  - `[x]` I1-m2b: Crude stretch foundation with in-place float normalization and scale/offset history for M2-image.
  - `[x]` I1-m2prec: Add generic or parallel `f64` float image/enhancement path; keep `f32` only as an optimization/compatibility dtype, not the only enhancement dtype.
  - `[~]` I1-next: Broader XRImage parity: band-aware dimensions, gamma, invert, alpha/finalize policy, mode conversion, and save helpers.
    - `[x]` I1-m5a1: Add Trollimage-style gamma/invert operations for `FloatImage<f32/f64>` plus mask-aware RGBA finalization without forcing the working buffer to u8.
    - `[ ]` I1-next2: Add band-aware dimensions, broader alpha/finalize policy, mode conversion, and save helpers.
- `[ ]` I2: Colormap system: validation, colorize/palettize, RGB/RGBA conversion, merging, reversing/ranging, export, and YAML loading.
- `[ ]` I3: Legacy `Image` compatibility where needed.
- `[ ]` I4: Color-space conversion and utility ramps.
- `[~]` I5: Satpy enhancer framework and YAML enhancement chains.
  - `[x]` I5-m5d1: Add inert YAML enhancement registry parsing for Satpy-style enhancement match entries and ordered operations, storing Python method tags as plain strings.
  - `[ ]` I5-next: Execute supported enhancement operations against `FloatImage` and integrate default/sensor enhancement lookup.
- `[ ]` I6: Instrument enhancements for ABI, AHI, VIIRS, MIMIC, and enhancement YAML data.
- `[ ]` I7: Convolution filters and overlays.

### C: Composites

- `[~]` C0: Composite core: `CompositeBase`, generic/single-band/RGB compositors, enhance-to-dataset helpers, band handling, and mode checks.
  - `[x]` C0-m2c: RGB compositor vertical slice for three matching single-band datasets, with Satpy-like common-channel mask behavior.
  - `[~]` C0-next: Broader composite parity: `CompositeBase`, generic/single-band compositors, enhance-to-dataset helpers, band/mode checks, metadata combination, optional prerequisites, and YAML integration.
    - `[x]` C0-m5d1: Add inert YAML composite registry parsing for compositor names, prerequisites, optional prerequisites, inline composite dependencies, and attrs.
    - `[ ]` C0-next2: Add real YAML compositor instantiation, optional prerequisite handling, dependency graph execution, and full metadata combination parity.
- `[x]` C1: Arithmetic composites such as NDVI/EVI/diff/ratio/sum and channel-operation compositors.
  - `[x]` C1-m5b1: Add `ArithmeticCompositor` foundations for difference, ratio, sum, and normalized difference over matching runtime-typed arrays, with mask propagation and an owned path that mutates the consumed left-hand f64 buffer.
- `[x]` C2: Spectral composites, weighted blends, and band replacement/mapping.
  - `[x]` C2-m5c1: Add `SpectralBlender` weighted single-band blending and `BandReplacementCompositor` for patching a corrected channel into a band-major image, with mask propagation and consuming APIs.
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
- `[~]` W2: Simple image writer: PNG/JPEG output, format detection, transparency/fill/mode handling, PNG metadata, and 8-bit/16-bit PNG output paths.
  - `[x]` W2-m2d: PNG writer using the Rust `image` crate with format detection and u8 Luma/RGB/RGBA image support.
  - `[x]` W2-m2d2: Add 16-bit PNG/HDR output path or a clearly typed writer interface that can preserve 16-bit display output.
  - `[ ]` W2-next: JPEG output, transparency/fill/mode polish, PNG metadata parity, and broader Satpy `PillowWriter` behavior.
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
- `[~]` SC3: Scene resampling pipeline and integration with all resamplers.
  - `[x]` SC3-m4e: Add `SceneResampleExt` in `rusty_sat_resample` so callers can resample all currently loaded datasets through a Rust-native Scene-level workflow without making `rusty_sat_core` depend on resampling crates.
  - `[ ]` SC3-next: Add Satpy-like `Scene::resample` parity: dataset selection, area choice helpers, resampler preparation/cache, and integration with future KD-tree/native/bilinear resamplers.
- `[~]` SC4: Scene save/show/to-xarray APIs.
  - `[x]` SC4-m2e: Add `Scene::save_dataset` wrapper for self-made datasets through a Rust-native writer contract.
  - `[ ]` SC4-next: Add `show`, `to_xarray`/export model, writer selection helpers, filename templating, and broader Satpy save API parity.
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
| M2-image | I1 partial (`XRImage` construction, crude stretch, f32/f64 enhancement buffers, finalize-to-u8 basics), C0 partial (generic/RGB compositor), W2 partial (PNG/simple image writer with 8-bit and 16-bit output paths), SC4 partial (`Scene.save_dataset`) | Done |
| M3-reader | R0.1, R0.3 or format-specific HSD base as needed, R0.8 partial, R1.3 AHI HSD/L1B priority slice, SC1 load path, W2 output path | Done for synthetic/local uncompressed  HSD basic PNG output; production HSD gaps remain for follow-up reader roadmap slices |
| M4-resample | S1 partial, S2 partial, R1.13 or another NetCDF reader slice, SC3 resampling integration | Done for geometry, fixture-backed FCI/NetCDF reader slice, Scene resampling extension, and exact 2D KD point-index acceleration; native NetCDF/HDF and full Pyresample KD parity remain later roadmap work |
| M5-enhance-composite | I1 broader stretch/finalize, I5 enhancer framework, C1 arithmetic, C2 spectral | Selected next |
| M6-resampling-full | S1-S7 | Started with S2 top-k/weighted neighbour-info foundations and single-neighbour swath neighbour-info, S3 bilinear, S4 bucket avg/sum/count/fraction plus pipeline preparation including explicit and auto-discovered fractions, S5 EWA-style weighted accumulation plus pipeline preparation, S6 native repeat/aggregate including higher-dimensional f64 y/x axes and runtime-typed repeat/aggregate support, and S7 typed source-geometry/cache/data-reduction/area-crop/dataset-slice/reduced-dataset/reduced-resample/reduction-option foundations; remaining gaps are S1-next, S2-next, S3-next, S4-next, S5-next, S6-next, and S7-next |
| M7-writers-composites-full | W1-W5 and C0-C5 | Needs M2/M5 output and composite foundations |
| M8-readers-modifiers-orbit | R0-R5, M1-M5, O1-O6 | Needs reader framework and test infrastructure |
| M9-production | SC1-SC6, CLI, Y, T | Needs all prior functional milestones |

- `[x]` M1: Early vertical slice: text grid data can become a grayscale PGM image.
- `[x]` M2-foundation: Complete P0 DataArray/DataGrid, mask, metadata, coordinates, and CRS foundations. This supersedes the earlier idea of jumping directly to PNG/RGB composite work.
- `[x]` M2-image: After P0 foundations, implement partial image model, crude stretch, RGB composite, PNG writer, and `Scene.save_dataset` so self-made data can produce color PNG.
  - `[x]` M2-image-a: Review Trollimage `XRImage`, Satpy `PillowWriter`, and `GenericCompositor`; add an owned u8 image buffer with mask-aware luma conversion from datasets. Roadmap: I1.
  - `[x]` M2-image-b: Add crude stretch foundations for float image data following Trollimage per-band min/max behavior. Roadmap: I1.
  - `[x]` M2-image-b2: Add `f64` image/enhancement buffer path or generic float buffer abstraction before relying on this image foundation for scientific corrections. Roadmap: I1-m2prec.
  - `[x]` M2-image-c: Add RGB compositor vertical slice for three matching single-band datasets. Roadmap: C0.
  - `[x]` M2-image-d: Add PNG writer using the Rust `image` crate, with format detection and luma/RGB/RGBA support. Roadmap: W2.
  - `[x]` M2-image-d2: Add 16-bit PNG/HDR output path or a clearly typed writer interface that can preserve 16-bit display output. Roadmap: W2.
  - `[x]` M2-image-e: Add `Scene.save_dataset` wrapper for self-made datasets and generated images. Roadmap: SC4.
- `[x]` M3-reader: Prioritize Himawari AHI first. Implement the AHI reader path, minimal format/file-handler foundations needed for AHI HSD or an AHI L1B sample, and Scene load path sufficient for an AHI sample to output a basic image.
  - `[x]` M3-reader-a: Inspect root `HS_D_users_guide_en_v12.pdf` and Satpy `satpy/readers/ahi_hsd.py`; add AHI HSD binary header parsing foundations.
  - `[x]` M3-reader-b: Add AHI HSD file-handler skeleton with filename/YAML inventory integration and segment metadata.
  - `[x]` M3-reader-c: Load raw AHI count arrays for a tiny local/synthetic HSD fixture, preserving dtype and metadata.
  - `[x]` M3-reader-d: Apply first visible/IR calibration path needed for basic image output.
  - `[x]` M3-reader-e: Integrate AHI reader with `Scene` load path and write a basic PNG from an AHI sample.
- `[x]` M4-resample: FCI/another NetCDF reader plus CRS/KD-tree foundations sufficient for real projection-aware resampling.
  - `[x]` M4-resample-a: Add Pyresample-style shared geometry definition foundations. Roadmap: S1.
  - `[x]` M4-resample-b: Add projection-definition/area completeness needed by KD-tree setup. Roadmap: S1.
  - `[x]` M4-resample-c: Add KD-tree neighbour information foundation. Roadmap: S2.
  - `[x]` M4-resample-d: Add first NetCDF/FCI-or-equivalent reader slice for resampling-oriented real data. Roadmap: R1/R0.2.
    - `[x]` M4-resample-d1: Add NetCDF metadata/file-content foundation after inspecting Satpy `NetCDF4FileHandler` and FCI docs. Roadmap: R0.2.
    - `[x]` M4-resample-d2: Add a backend adapter strategy and FCI-or-equivalent data-loading slice. Roadmap: R0.2/R1.
      - `[x]` M4-resample-d2a: Add backend-neutral NetCDF file-handler facade and metadata source trait. Roadmap: R0.2.
      - `[x]` M4-resample-d2b: Add backend-neutral variable data source and first FCI L1C measured-channel counts dataset loader. Roadmap: R0.2/R1.
      - `[x]` M4-resample-d2c: Add documented YAML fixture-backed source for real file-backed tests while postponing native NetCDF/HDF dependency choice. Roadmap: R0.2/R1.
    - `[x]` M4-resample-d3: Add FCI fixture reader integration with Reader inventory/load and Scene planning. Roadmap: R1.
  - `[x]` M4-resample-e: Add Scene-level resampling integration for current nearest path. Roadmap: SC3.
  - `[x]` M4-resample-f: Add real KD-tree backend or accelerated nearest-neighbour structure for the current nearest path. Roadmap: S2.
    - `[x]` M4-resample-f1: Add exact 2D KD point index and route swath nearest through it. Roadmap: S2.
- `[x]` M5-enhance-composite: Broaden image enhancement and arithmetic/spectral composites.
  - `[x]` M5-enhance-composite-a: Inspect Trollimage `XRImage` and Satpy enhancement chain behavior; add gamma/invert/alpha-finalize foundations without collapsing f64 paths to u8. Roadmap: I1/I5.
  - `[x]` M5-enhance-composite-b: Add arithmetic composite foundations such as normalized difference, ratio, sum, and difference with mask propagation and consuming APIs. Roadmap: C1.
  - `[x]` M5-enhance-composite-c: Add spectral composite foundations for weighted blends and band replacement/mapping. Roadmap: C2.
  - `[x]` M5-enhance-composite-d: Add YAML-driven enhancer/composite registration slice. Roadmap: I5/C0/C2.
- `[~]` M6-resampling-full: Complete major resampler families and performance work.
  - `[x]` M6-resampling-full-a: Add native resampler repeat/aggregate foundation. Roadmap: S6.
  - `[x]` M6-resampling-full-b: Add resampler preparation and dataset-level pipeline helpers. Roadmap: S7.
  - `[x]` M6-resampling-full-c: Add first same-projection area bilinear resampler and pipeline method. Roadmap: S3.
  - `[x]` M6-resampling-full-d: Add first swath-to-area bucket avg/sum/count resampling foundation. Roadmap: S4.
  - `[x]` M6-resampling-full-e: Add bucket fraction category-axis output and sum skipna=false guardrail. Roadmap: S4.
  - `[x]` M6-resampling-full-f: Add first dependency-free EWA-style weighted accumulation foundation. Roadmap: S5.
  - `[x]` M6-resampling-full-g: Add area-to-area top-k neighbour-info and weighted sampling foundations. Roadmap: S2.
  - `[x]` M6-resampling-full-h: Add typed source-geometry pipeline preparation for bucket and EWA swath resamplers. Roadmap: S7/S4/S5.
  - `[x]` M6-resampling-full-i: Add higher-dimensional f64 native resampling over named `y,x` axes. Roadmap: S6.
  - `[x]` M6-resampling-full-j: Add single-neighbour swath-to-area neighbour-info generation for current KD-indexed nearest workflows. Roadmap: S2.
  - `[x]` M6-resampling-full-k: Add explicit-category bucket fraction through the typed resampling pipeline. Roadmap: S4/S7.
  - `[x]` M6-resampling-full-l: Add automatic bucket-fraction category discovery for direct and pipeline usage. Roadmap: S4.
  - `[x]` M6-resampling-full-m: Add runtime-typed native resampling with dtype-preserving repeat and `f64` aggregate means. Roadmap: S6.
  - `[x]` M6-resampling-full-n: Add explicit prepared-resampler cache and cached dataset resampling helpers. Roadmap: S7.
  - `[x]` M6-resampling-full-o: Add lon/lat grid and boundary validity filters for coarse swath data reduction. Roadmap: S7.
  - `[x]` M6-resampling-full-p: Add same-projection area slices and crop-source-area helpers for the first Satpy/Pyresample-style area data-reduction path. Roadmap: S7.
  - `[x]` M6-resampling-full-q: Add dtype-preserving dataset `y`/`x` slicing helpers for area data reduction, including consuming array paths. Roadmap: S7.
  - `[x]` M6-resampling-full-r: Add combined area dataset reduction helpers with Satpy-like reduced-area shape validation. Roadmap: S7.
  - `[x]` M6-resampling-full-s: Add opt-in reduction-aware area resampling pipeline helpers with cached and consuming variants. Roadmap: S7.
  - `[x]` M6-resampling-full-t: Add Satpy-style reduction options to the resampling pipeline, including `shape_divisible_by` handling for reduced same-projection area resampling and cache separation. Roadmap: S7.
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
- For large-array operations such as composites, resampling, enhancement, and writing, add a consuming or in-place API in the same step as the borrowed API whenever the caller can reasonably give up ownership. Borrowed convenience APIs are acceptable for tests and small data, but they must not be the only path for full-disk imagery.
- When you intentionally leave a known memory or performance inefficiency in place (because the fix belongs to a later roadmap step, or the current vertical slice doesn't justify the complexity), add it to the **Known Inefficiencies** section below with the reason and the planned fix step.

## Upstream Satpy Tracking

When Satpy changes upstream:

1. Identify whether the change is core, config, reader, writer, composite, modifier, resampler, image, or CLI behavior.
2. Add a separate roadmap/task entry before implementation.
3. Add a compatibility test or fixture whenever possible.
4. Implement the new behavior separately from unrelated work.
5. Record the supported Satpy behavior in this file or a future `UPSTREAM_COMPAT.md`.

### Tracked Upstream Changes (2026-05-09 audit, satpy main 4f1d7236b)

| Date | Satpy Commit | Category | Change | Affected Rust Roadmap | Status |
|------|-------------|----------|--------|----------------------|--------|
| 2026-05-06 | `ca8e6d09e` | **core** | `open_dataset()` now accepts `str \| FSFile` (fsspec remote file support) | R0.1 `BaseFileHandler` trait | ⚠️ design input needed |
| 2026-05-06 | `ea93a0d41` / `c47b792b1` | **composite** | VIIRS: new `day_cloud_type` / `day_cloud_type_distinction` composites; `night_microphysics` no longer uses DNB | C4 VIIRS composites | 📋 tracked |
| 2026-05-06 | `f5f6fdc5c`–`97d6314dd` | **reader** | OMPS EDR reader rewritten for official NOAA NetCDF4 format | R3 L2 product readers | 📋 tracked |

No tracked changes affect the currently completed M3-reader AHI HSD path.

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
| Runtime-typed `AnyDataArray` with method dispatch across 5 variants, consuming value/mask extraction helpers, and dtype-preserving `y`/`x` slicing | In-place mutation of array values (no `&mut self` transform API) |
| Named dimensions (1D–4D defaults), coordinates (1D/2D/scalar), packed `ValidityMask`, `ChunkShape`; eager `y`/`x` slices preserve matching coordinates and masks | Real lazy loading; `LazyDataArray` is contract-only, no file-backed source |
| `DataId`/`DataQuery` matching, scoring, best-match, ambiguity detection, ordered modifier-chain prefix matching, less-modified query creation, and ancillary-variable dataset filtering | Composite/modifier execution from these queries |
| `Dataset` dual metadata (flat `BTreeMap<String,String>` + nested `MetadataValue` attrs), borrowed/consuming/take array access, and array replacement | Single-source metadata; `insert_metadata()` still writes to BOTH maps (transitional) |
| `Dataset` typed ancillary variable links by `DataId`, with find/replace helpers modeled after Satpy `anc_vars.py` | Embedding full ancillary arrays inside attrs; only typed links are stored |
| `Scene` insert/remove datasets, borrowed dataset iteration, plan reader loads, register composites/modifiers, and save a dataset through the `DatasetWriter` contract | Actual composite/modifier execution, resampling delegation, show/to-xarray |

### rusty_sat_config

| Can | Cannot |
|-----|--------|
| Load Satpy default config path (`satpy/satpy/etc`) with `serde_norway`, byte-size limits, and YAML nesting-depth validation | Format-specific config (reader YAML, composite YAML, enhancement YAML) |
| Support `RUSTY_SAT_CONFIG_PATH` / `SATPY_CONFIG_PATH` env vars | Full untrusted-YAML sandboxing; parser is still backed by `unsafe-libyaml-norway` |
| Component config lookup (readers, writers, composites, enhancements) | |

### rusty_sat_readers

| Can | Cannot |
|-----|--------|
| Parse filename patterns with `FilenamePattern` (trollsift-compatible: keys, parse, compose, globify, typed datetimes) | Full trollsift parity (every custom compose/conversion edge case) |
| Parse Satpy reader YAML metadata via `YamlMetadataReader` (`reader`/`file_types`/`datasets` sections, Python tags as `MetadataValue`) with byte-size/depth guardrails | Safe YAML tag deserialization into typed structs; tags are currently stored as metadata values |
| `FakeReader` in-memory inventory + dataset loading for Scene planning tests | Real satellite file I/O (NetCDF/HDF/GeoTIFF/etc.) |
| `TextGridReader` reads plain text numeric grids + YAML metadata; provides `TextGridChunkSource` lazy fixture that caches the parsed fixture grid instead of rereading per chunk | Production file handlers; `yaml_reader` can inventory datasets but cannot load array data |
| AHI HSD initial header parser plus file-handler/reader skeleton can expose band/counts dataset IDs and segment metadata from YAML filename matches | Production AHI HSD segment assembly, navigation arrays, bzip2-compressed file handling, and full Satpy YAML instantiation |
| AHI HSD handler can load uncompressed local/synthetic block-12 raw counts into a `u16` `DataArray` and mask Satpy error/outside-scan count values with a whole-file byte safety limit | Streaming/chunked HSD reads; current raw-count path materializes the requested file/byte buffer |
| AHI HSD handler can parse visible/IR block-5 calibration extensions and produce `f32` radiance/reflectance/brightness-temperature datasets for Satpy-like display paths and `f64` calibrated datasets when precision is requested | User calibration overrides, updated visible calibration fallback modes, GSICS correction, and production fixture parity tests |
| `AhiHsdReader` can expose a configured calibration, load a local uncompressed HSD file through `Scene` planning, and write a basic PNG through `SimpleImageWriter` in tests | Real Satpy YAML reader instantiation, multi-file/segment grouping, and production sample output |
| Current reader inventory/load path still uses `f32` by default for memory-efficient display output | Final scientific/HDR HSD workflows need writer-preserving float/16-bit policies and public selection of the f64 calibrated path |
| NetCDF metadata/file-handler foundation can build Satpy-style `file_content` keys, validate loaded variable arrays against metadata, load an FCI L1C measured-channel counts dataset, read YAML fixture-backed NetCDF trees/arrays from disk, and expose a fixture-backed FCI reader for Scene planning | Native NetCDF/HDF backend opening, real auto mask/scale application, chunked variable data access, and calibrated FCI dataset loading |

### rusty_sat_resample

| Can | Cannot |
|-----|--------|
| Shared Pyresample-style geometry foundations: `GeometryDefinition`, `CoordinateDefinition`, and `GridDefinition` for shape/size/dimensionality and finite lon/lat-backed coordinate arrays | Full Pyresample equality/hash/cartesian-coordinate behavior |
| `AreaDefinition`: id, projection, shape, extent, pixel-size helpers, guarded YAML loading, projection-unit resolution derivation, shared geometry/projection trait implementations, allocation-free projection-coordinate iterators, same-projection area slices, and crop-source-area helpers | Stacked/dynamic areas, spherical polygon math, cross-projection/swath slicers, overlap utilities |
| `SwathDefinition`: dimension-only or lon/lat coordinate-backed swaths, WGS84 CRS convention, guarded YAML loading, and shared geometry trait implementation | Real geocentric resolution, aggregation, boundary extraction |
| `ProjCrs`: WGS84/PROJ/EPSG parsing, normalization (numeric canonicalization, `latlong`→`longlat`, `+init=EPSG:`→`epsg`), identity-only transforms | Real cross-CRS transforms; backend is `MetadataOnly` |
| `Coordinate2D` finite-validated coordinate + transform API (identity for geographic/same-CRS) | Projected or cross-CRS forward/inverse transforms (returns `Unsupported`) |
| `NeighbourInfo`: Pyresample-style valid input/output flags, nearest/top-k index arrays, distance arrays, missing-neighbour sentinel, cached valid-input/output counts, area-to-area nearest/top-k neighbour queries, single-neighbour swath-to-area neighbour queries, nearest sampled-output helpers, weighted multi-neighbour sampling helpers, and Gaussian weight helper for `DataGrid` with both borrowed and owned variants | Multi-neighbour swath neighbour-info arrays, geocentric/great-circle distances, chunked/parallel neighbour queries, uncertainty outputs, multi-band weighted sampling |
| `KdPointIndex2D`: dependency-free exact 2D KD-tree over finite source points for nearest-point lookup with optional radius-of-influence pruning | Pyresample-compatible geocentric KD-tree, multi-neighbour output, chunked/parallel queries |
| `SceneResampleExt`: Scene-level resampling extension in `rusty_sat_resample` for all currently loaded datasets through a supplied `Resampler` and destination area, with both borrowed (`&self`) and consuming (`self`) APIs | Full Satpy `Scene.resample` parity: dataset selection, area helpers, resampler cache/preparation, and multiple resampler families |
| `NearestAreaResampler`: area-to-area and KD-indexed swath-to-area nearest, radius of influence, fill value, mask propagation, coordinate preservation, lazy input consumption, and `resample_owned` through the `Resampler` trait | CRS transforms, anti-meridian handling, geocentric distances, multi-band, chunk-preserving lazy output |
| `BilinearAreaResampler`: same-projection 2D f64 area-to-area bilinear interpolation, fill or mask missing behavior, strict source-mask/non-finite handling, metadata preservation, destination x/y coordinates, and `resample_owned` support | Irregular swath bilinear coefficients, radius/neighbour based bilinear lookup, cubic/spline interpolation, higher-dimensional/band-aware sampling, and chunked/lazy bilinear execution |
| `BucketResampler` / `BucketFractionResampler`: lon/lat swath-to-geographic-area bucket average, sum, count, and explicit/auto-category fraction for 2D f64 grids; fractions return `categories,y,x` arrays; skipna/fill behavior, owned resampling, destination x/y coordinates, metadata preservation, Satpy-like count attrs, and typed pipeline preparation through `SourceGeometry::Swath` | Projected target backends, multidimensional/band-aware buckets, chunked/lazy bucket execution, and Scene-level preparation integration |
| `EwaResampler`: dependency-free lon/lat swath-to-geographic-area EWA-style weighted accumulation for 2D f64 grids, configurable radius/weight/fill/masked-missing policy, source-mask skipping, destination x/y coordinates, metadata preservation, owned resampling, and typed pipeline preparation through `SourceGeometry::Swath` | Full Pyresample Fornav/LLS2 parity, scan-aware `rows_per_scan`, maximum-weight mode, chunked execution, multi-band sampling, geocentric/cross-projection distances, and production EWA performance |
| `NativeResampler`: Satpy-style native repeat for integer upscaling, nanmean aggregation for integer downscaling, equal-shape pass-through, mixed-axis rejection, mask propagation, destination x/y coordinates, metadata preservation, higher-dimensional arrays by resampling named `y,x` axes, runtime-typed identity/repeat that preserves `f32`/`f64`/`u8`/`u16`/`i16`, and aggregate means promoted to `f64` | Lazy/chunked native execution and full Satpy area-choice integration |
| `prepare_resampler`, `prepare_resampler_for_geometry`, `ResamplerCache`, `resample_dataset`, `resample_dataset_owned`, `SourceGeometry`, `data_reduce`, and `slicer`: typed Satpy-style pipeline helpers for selecting current area-based nearest/bilinear/native resamplers and swath-based bucket/EWA resamplers, including explicit-category bucket fractions, explicit prepared-resampler caching, Pyresample-style lon/lat validity filtering, same-projection area crop helpers, borrowed/consuming dtype-preserving dataset slice paths, combined area dataset reduction helpers, and opt-in reduction-aware area resampling helpers with `reduce_data` and `shape_divisible_by` options | Automatic source-area/swath lookup from dataset attrs, swath/cross-projection slicers, or CRS cross-projection resampling |

### rusty_sat_composites

| Can | Cannot |
|-----|--------|
| Execute `RgbCompositor` for three matching 2D single-band runtime-typed datasets into a band-major `bands,y,x` f64 dataset with Satpy-like common-channel mask behavior; large callers should use the consuming `compose_rgb_owned` path | Full `CompositeBase`/`GenericCompositor` parity, metadata combination, optional prerequisites, YAML composite loading, or Scene dependency execution |
| Execute `ArithmeticCompositor` for matching runtime-typed arrays with difference, ratio, sum, and normalized-difference operations; masks are OR-propagated and large callers can use `compose_owned` to reuse the consumed left-hand f64 buffer for output | Full arithmetic YAML integration, metadata combination parity, multi-input/channel-operation compositors, or Scene dependency execution |
| Execute `SpectralBlender` weighted 2D channel blends and `BandReplacementCompositor` band-major channel replacement with mask propagation and owned variants for large buffers | NDVI hybrid green, natural enhancement, spectral YAML integration, metadata combination parity, or Scene dependency execution |
| Parse Satpy-style composite/enhancement YAML sections into inert typed registry definitions, including `!!python/name:` tags as non-executable strings and inline composite dependencies | Instantiate compositors/enhancers from YAML, execute enhancement chains, merge sensor/default configs, or run composites through `Scene` |
| Define `CompositeRecipe` and `ModifierRecipe` in `rusty_sat_core` | Execute registered composite/modifier recipes through `Scene` |

### rusty_sat_image

| Can | Cannot |
|-----|--------|
| Store owned u8 `Image`, owned u16 `Image16`, and generic owned `FloatImage<f32/f64>` pixels for Luma/RGB/RGBA; construct luma images from 2D runtime-typed datasets; apply in-place crude stretch, Trollimage-style gamma/invert, and mask-aware RGBA finalization | Broader XRImage parity: band-aware dimensions, broader alpha/finalize policy, colorize, mode conversion, or save helpers |

### rusty_sat_writers

| Can | Cannot |
|-----|--------|
| `PgmWriter` write binary PGM (P5) from `DataGrid`, `AnyDataArray`, or `LazyDataArray<T>`; autoscale, fill value, mask-aware | JPEG output, GeoTIFF, CF NetCDF |
| `SimpleImageWriter` write u8 PNG from finalized Luma/RGB/RGBA `Image` buffers, write u16 PNG from finalized `Image16` buffers, save 2D datasets through current luma finalization, and work through `Scene::save_dataset` | PNG metadata parity, alpha/fill polish beyond existing image buffers |
| Lazy PGM writes read chunks into one y-stripe at a time (incremental) | Single-pass autoscale+write (currently reads chunks twice: autoscale then write) |

### rusty_sat_cli

| Can | Cannot |
|-----|--------|
| (Skeleton only) | Any CLI commands; not yet implemented |

### Known Inefficiencies (to address in future roadmap steps)

- `Dataset::insert_metadata()` writes to both `self.metadata` and `self.attrs` — transitional dual-write; plan to remove flat `metadata` map entirely.
- `ResamplerCache` uses unbounded linear-scan lookup. For multi-channel scene processing the current O(n) scan and unlimited growth are acceptable, but production throughput would benefit from a `HashMap` key and a configurable capacity limit with LRU eviction.
- YAML parsing has byte-size and nesting-depth guardrails, but the current maintained `serde_norway` stack still uses `unsafe-libyaml-norway`. Revisit if a mature pure-Rust Serde YAML frontend becomes available without losing Satpy YAML compatibility.
- `SourceChunkCache` in nearest resampler is bounded with a small FIFO cache. Future work should make cache sizing configurable and consider LRU/tile traversal for better hit rates on large scenes.
- `autoscale_lazy` in PGM writer reads all chunks twice (once for min/max, once for write) — when the caller does not provide an explicit scale, the only way to avoid the double pass is to cache chunked f64 data during the first pass and encode from cache, trading memory for I/O.
- `encode_pgm_values` in PGM writer collects its `impl IntoIterator<Item = f64>` argument into an intermediate `Vec<f64>` before autoscaling and encoding. Callers that already have a `Vec<f64>` (e.g. `encode_pgm_array` non-f64 path via `values_as_f64()`) pay a second allocation of the same data. Accept `Vec<f64>` directly to let callers materialize once.
- `dataset_metadata_pairs` and `dataset_attr_pairs` in nearest resampling clone every metadata key/value and attrs key/value into intermediate `Vec`s before re-inserting them into the resampled dataset. The borrowed `resample` path still uses these; the owned `resample_owned` path iterates the source maps directly for `NearestAreaResampler`, `BilinearAreaResampler`, `BucketResampler`, `EwaResampler`, and `NativeResampler`. Unify the remaining borrowed paths to clone individual entries inline.
- `KdPointIndex2D` construction uses `sort_by` (O(n log² n)) and stores ~48 bytes-per-node `Vec<KdNode>`. For a production VIIRS swath (~20.5M valid points) this means ~984 MB tree memory and slower build times. Replace `sort_by` with `select_nth_unstable` for O(n log n) construction and consider `f32` coordinate storage or compressed node layouts when the first real swath reader lands.
- `native_repeat_yx` / `native_aggregate_mean_yx` allocate a `Vec<usize>` per output pixel via `unravel_index` for multi-dimensional index arithmetic. For a 3-band 1000×1000 image this is ~1M allocations/deallocations. Pre-compute row-major index maps or use a reusable stack-allocated buffer when multi-dimensional native resampling reaches production workloads.
- `FciL1cFixtureReader` stores the metadata tree (`NetCdfFixtureSource`) AND the flat metadata index (`NetCdfFileHandler`) in the same struct. For fixture data (<1 MB metadata) this is negligible. Production readers should discard the tree after building the flat index to avoid the dual storage.
- `AreaDefinition::projection_x_coords()` / `projection_y_coords()` still allocate `Vec<f64>` axes for compatibility. Prefer `iter_projection_x_coords()` / `iter_projection_y_coords()` / `iter_projection_coords()` in KD-tree and repeated large-area operations.
- `GeometryDefinition::shape()` returns an owned `Vec<usize>` so `AreaDefinition`, `SwathDefinition`, and `CoordinateDefinition` allocate when callers ask for trait-erased shapes. Their `ndim()` and `size()` implementations avoid this allocation; avoid calling `shape()` in KD-tree hot loops unless the trait is redesigned around borrowed/small-array shapes.
- `mask_fci_counts_array` has an optimized u16 fast path that iterates raw integer values directly. Non-u16 dtypes still fall back to `values_as_f64()` which allocates a full `Vec<f64>` for mask computation. Add dtype-specific fast paths for f32, u8, and i16 when readers for those dtypes land.
- `SwathDefinition::longitude_coordinate()` / `latitude_coordinate()` clone full lon/lat arrays because `Coordinate` currently owns vectors. Add borrowed or shared coordinate storage before large production swath workflows.
- `Image::from_luma_dataset` / `from_luma_array` forces a `FloatImage<f32>` intermediate allocation for every source dtype, including u8 and u16. For a 4K×4K u8 source (16 MB) this wastes 64 MB on the intermediate `Vec<f32>`, and the float round-trip (u8→f32→u8) adds CPU overhead. Add a direct integer→u8 fast path or a dtype-aware finalization path in the image-enhancement milestone.
- `write_png_image` accepts only `&Image` and copies pixel data into the PNG encoder's internal buffers via `save_buffer_with_format`. A consuming `write_png_image_owned(image: Image, …)` that hands `image.into_pixels()` directly to the PNG encoder would avoid the internal copy when the caller no longer needs the `Image`. Add during W2-next writer polish.
- `write_png16_image` converts u16 samples to PNG-required big-endian bytes before encoding. This preserves 16-bit values, but duplicates the image byte size during encode. Future writer work should stream rows into the PNG encoder if the crate API permits it.
- PNG compression level is not exposed to callers — `save_buffer_with_format` uses the `image`/`png` crate defaults. Expose a compression preset (Fast/Default/Best) on `SimpleImageWriter` when PNG metadata parity is addressed in W2-next.
