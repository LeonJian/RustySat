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
4. Read the matching Python reference code before designing Rust behavior.
5. Implement the smallest useful Rust capability.
6. Add focused tests.
7. Run `cargo check --workspace` and `cargo test --workspace`.
8. Update this file with completed work and known gaps.
9. Commit the completed step before moving to the next step.

Do not bundle unrelated roadmap items together. If a Satpy update introduces new behavior, track it as a separate task and implement it separately.

If a roadmap step is too large, split it into smaller lettered substeps before implementation. Complete one substep at a time, update this file after each substep, and leave the next substep clear for future work.

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
- `[ ]` Step 5: fake reader vertical slice.
- `[ ]` Step 6: filename pattern parser compatible with `trollsift`.
- `[ ]` Step 7: area definitions and YAML area loading.
- `[ ]` Step 8: first real reader.
- `[ ]` Step 9: nearest resampling.
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
- Use explicit `Result<T, RustySatError>` returns for fallible operations.
- Keep incomplete behavior explicit with placeholder errors, not silent defaults.
- Keep public types documented enough for future agents to understand intent.
- Avoid adding heavy dependencies until the relevant roadmap step needs them.

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

The config crate now has the first real foundation:

- Satpy-reference default config path: `satpy/satpy/etc`.
- `RUSTY_SAT_CONFIG_PATH` and `SATPY_CONFIG_PATH` environment path support.
- Component config lookup for readers, writers, composites, and enhancements.
- YAML file loading with recursive merge where later files override earlier files.

No real Satpy reader, resampler, compositor, enhancement, or writer behavior has been ported yet.

## High-Risk Reference Areas

Do not rewrite these blindly:

- Satpy `Scene`, `DataId`, `DataQuery`, dependency tree, YAML readers, composites, and resampling flow.
- Pyresample geometry and resampling algorithms.
- Trollsift filename parser and formatter.
- Trollimage `XRImage`, colormaps, alpha handling, and writer paths.
- Pyorbital TLE, orbital, astronomy, and scan geolocation math.
