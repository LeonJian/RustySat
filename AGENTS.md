# Rusty Sat Agent Guide

## Mission

Rusty Sat is a Rust-native, Satpy-compatible rewrite. The long-term goal is full Satpy capability: YAML-driven readers, composites, modifiers, resampling, corrections/enhancements, and writers that can produce corrected and resampled imagery efficiently.

This project must move slowly and deliberately. Do not attempt to rewrite all of Satpy in one change. Every implementation step should be small, tested, and easy for the next agent to continue.

## Reference Trees

- `satpy/` is the Python Satpy reference implementation.
- `deps/` contains local reference dependencies: `pyorbital`, `pyresample`, `trollimage`, and `trollsift`.
- Treat these folders as read-only references unless the user explicitly asks otherwise.
- Before implementing each roadmap item, check the relevant Satpy and dependency source/docs first. Use `satpy/doc`, `deps/*/doc` or `deps/*/docs`, and the implementation modules as the behavior reference.

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

If a roadmap step is too large, split it into smaller lettered substeps before implementation. Complete one substep at a time, update this file after each substep, and leave the next substep clear for future work.

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

## Roadmap

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
- `[~]` Step 9: nearest resampling.
  - `[x]` Step 9a: projection-coordinate nearest resampling from one `AreaDefinition` grid to another.
  - `[ ]` Step 9b: radius/fill behavior parity tests against representative Pyresample cases.
  - `[ ]` Step 9c: swath-to-area nearest foundations.
- `[ ]` Step 10: first writer.
- `[ ]` Step 11: first composite.
- `[ ]` Step 12+: expand Satpy parity feature by feature.

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

- Prefer Rust-native APIs over Python-shaped clones.
- Preserve Satpy YAML compatibility as the main compatibility contract.
- Preserve Satpy behavior as the result contract: for the same supported input, Rusty Sat should produce the same dataset choices, metadata interpretation, dependency decisions, geometry, imagery, or writer output as Satpy.
- Use explicit `Result<T, RustySatError>` returns for fallible operations.
- Keep incomplete behavior explicit with placeholder errors, not silent defaults.
- Keep public types documented enough for future agents to understand intent.
- Avoid adding heavy dependencies until the relevant roadmap step needs them.
- Split growing modules into focused files before they become hard to review. Avoid letting one source file become the dumping ground for unrelated core, reader, parser, resampling, or writer behavior.
- Keep tests close to the module they validate, but move larger fixtures or broad parity suites into separate test files when they would make a source file hard to scan.

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

The initial Rust workspace exists. It contains compile-only crate skeletons and minimal core stubs for:

- `Scene`
- `Dataset`
- `DataId` with typed qualifier values
- `DataQuery` with exact, one-of, wildcard, wavelength containment matching, best-match sorting, and ambiguity errors
- `DependencyGraph` with node sources, dependency edges, leaves, dependents, and Scene integration for user-provided datasets.
- `ReaderInventory` and `SceneLoadPlan` for planning reader-backed dataset loads without reading data yet.
- `CompositeRecipe` and `ModifierRecipe` for populating dependency graph edges before real generation exists.
- shared `RustySatError`

The readers crate now has an in-memory `FakeReader` that can expose an inventory, load cloned datasets, and drive a `Scene` planning/insertion vertical slice in tests.

The readers crate also has `filename_pattern::FilenamePattern`, a focused trollsift-compatible starter parser. It supports keys, full-match parsing, non-greedy string fields, integer/float conversion, repeated-field equality checks, strict/partial compose, trollsift string conversions, typed datetime-like values for common numeric strftime fields, validation, and globify for common Satpy filename patterns. It is not a byte-for-byte trollsift clone, but it now covers the core filename behavior expected by early YAML reader work.

The readers crate now has a metadata-only `yaml_reader` module based on inspected Satpy reader docs and `satpy.readers.core.yaml_reader`:

- Parses Satpy-style `reader`, `file_types`, and `datasets` YAML sections, including YAML Python tags as metadata values.
- Builds `DataId` inventory entries from dataset names, scalar resolutions, wavelength triplets, polarization, modifiers, and calibration variants.
- Matches configured file type patterns against filename tails, parses filename metadata with `FilenamePattern`, filters selected filenames, and sorts file types after their `requires` dependencies.
- Exposes `YamlMetadataReader` through the common `Reader` trait, but dataset array loading is intentionally unsupported until a real file handler substep.

The readers crate now has the first real array-loading vertical slice:

- `rusty_sat_core::DataGrid` stores 2D f64 dataset values with shape validation.
- `text_grid::TextGridReader` combines Satpy-style YAML metadata, filename matching, and a tiny plain-text numeric grid file handler.
- This proves the Reader trait can return real `Dataset` values. It is intentionally not a production satellite reader; NetCDF/HDF/GeoTIFF product handlers remain future work.

The config crate now has the first real foundation:

- Satpy-reference default config path: `satpy/satpy/etc`.
- `RUSTY_SAT_CONFIG_PATH` and `SATPY_CONFIG_PATH` environment path support.
- Component config lookup for readers, writers, composites, and enhancements.
- YAML file loading with recursive merge where later files override earlier files.

The resample crate now has a focused `area` module based on inspected Pyresample/Satpy references:

- `AreaDefinition` with id, description, projection id, projection parameters, shape, area extent, and pixel-size helpers.
- YAML loading for common Satpy/Pyresample area definitions with mapping projections, PROJ strings, `shape.height`/`shape.width`, flat `area_extent`, and `lower_left_xy`/`upper_right_xy`.
- Projection-unit resolution helpers for deterministic Pyresample-style derivations: `area_extent + resolution` derives shape, and `center + radius + resolution` derives extent and shape. Unit conversion, pyproj CRS validation, and lon/lat-driven dynamic freezing are still future work.
- Validation for empty ids, zero-sized shapes, invalid extents, missing area ids, and malformed YAML.

The resample crate also has a focused `swath` module based on inspected Pyresample references:

- `SwathDefinition` can represent dimension-only swaths or validated longitude/latitude coordinate arrays.
- It preserves Pyresample's default lon/lat WGS84 CRS convention as explicit metadata.
- YAML loading supports small 1D and 2D longitude/latitude fixtures for tests and future reader work.
- Real geocentric resolution, aggregation, boundary extraction, CRS transforms, and resampling behavior are still future work.

The resample crate now has a first `nearest` module based on inspected Pyresample nearest-neighbor docs/code:

- `NearestAreaResampler` resamples 2D `DataGrid` datasets from a source `AreaDefinition` to a destination `AreaDefinition` using projection-coordinate pixel centers.
- It supports an optional radius of influence and configurable fill value.
- It requires matching projection metadata and does not yet implement Pyresample's kd-tree, swath nearest, CRS transforms, masks, or multi-band handling.

No production Satpy satellite reader, compositor, enhancement, or writer behavior has been ported yet. Current reader/resampler work is limited to early, testable vertical slices.

## High-Risk Reference Areas

Do not rewrite these blindly:

- Satpy `Scene`, `DataId`, `DataQuery`, dependency tree, YAML readers, composites, and resampling flow.
- Pyresample geometry and resampling algorithms.
- Trollsift filename parser and formatter.
- Trollimage `XRImage`, colormaps, alpha handling, and writer paths.
- Pyorbital TLE, orbital, astronomy, and scan geolocation math.
