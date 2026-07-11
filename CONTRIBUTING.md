# Contributing to Rusty Sat

## Development Environment

### Prerequisites

- **Rust 1.70+** — [rustup.rs](https://rustup.rs)
- **Git** — for version control
- **AHI HSD test data** (optional) — for running reader integration tests

### Setup

```bash
git clone https://github.com/pytroll/rusty-sat.git
cd rusty-sat
cargo build
cargo nextest run --workspace
```

### IDE Configuration

The workspace uses standard Rust tooling. For VS Code, install `rust-analyzer`. There are no special IDE plugins required.

---

## Workflow

### Branch Strategy

1. Create a feature branch from `master`:
   ```bash
   git checkout master
   git pull
   git checkout -b feature/your-feature-name
   ```

2. Make focused, incremental commits. Each commit should:
   - Be a working vertical slice
   - Have a descriptive message explaining **why**, not just what
   - Follow the existing commit style (lowercase, imperative mood)

3. When ready, push and open a PR against `master`.

### Commit Message Style

```
ahi hsd: add bzip2 data block decompression

The HSD user guide §4.2 documents compression flag=2 as bzip2. Handle
decompressed data block validation against expected byte count from header.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
```

---

## Coding Standards

### Workspace Lints (Enforced)

All crates inherit these from the workspace `Cargo.toml`:

```toml
[workspace.lints.rust]
unsafe_code = "forbid"

[workspace.lints.clippy]
dbg_macro = "deny"
todo = "deny"
unwrap_used = "deny"
```

### Pre-Commit Checklist

```bash
# Formatting
cargo fmt --all -- --check

# Linting
cargo clippy --workspace -- -D warnings

# Tests
cargo nextest run --workspace

# If you have AHI test data:
cargo test --package rusty_sat_readers \
  --test ahi_first_figure_integration --release -- --nocapture
```

### Code Style

1. **Error handling**: Use `Result<T, RustySatError>` throughout. Never `unwrap()` or `expect()` in production code — use `?` and the error factory methods (`RustySatError::invalid_input(...)`, etc.).

2. **Comments**: Default to no comments. Only add a comment when the **why** is non-obvious — a hidden constraint, a subtle invariant, a workaround for a specific bug. Well-named identifiers should explain the **what**.

3. **No half-finished code**: The `deny(todo)` lint prevents committed `todo!()` calls. If a feature is incomplete, make that explicit through the API (e.g., return `Err(RustySatError::unsupported(...))`).

4. **Consuming APIs preferred**: Design methods to take ownership (`self`) when the caller would likely drop the value afterward. This reduces unnecessary clones.

5. **F32 for display, F64 for science**: The dual-precision calibration pattern is intentional. Keep both paths available — don't remove one.

6. **No feature flags or backwards-compatibility shims**: If something is unused, delete it. If the API needs to change, change it directly.

### Test Expectations

Every feature should have:

| Layer | What | Example |
|-------|------|---------|
| **Unit test** | Synthetic data, fast, deterministic | `parses_initial_ahi_hsd_header_blocks` |
| **Integration test** | Real data, validates actual values | `ahi_first_figure_integration.rs` |
| **Pipeline test** | End-to-end: read → calibrate → assemble → write | `end_to_end_pipeline_timing_and_statistics` |

---

## Project Structure

```
rusty-sat/
├── AGENTS.md              ← Agent guide + roadmap
├── README.md              ← Project overview
├── CONTRIBUTING.md        ← This file
├── Cargo.toml             ← Workspace root
├── docs/
│   └── ARCHITECTURE.md    ← Detailed architecture docs
├── crates/
│   ├── rusty_sat_core/    ← Foundation types (DataId, Dataset, Scene, DataArray)
│   ├── rusty_sat_config/  ← Config loading and merging
│   ├── rusty_sat_readers/ ← Satellite data readers
│   ├── rusty_sat_resample/← Spatial resampling algorithms
│   ├── rusty_sat_composites/← Compositing and enhancement
│   ├── rusty_sat_image/   ← Image types and operations
│   ├── rusty_sat_writers/ ← File output (PNG, GeoTIFF, PGM, JPEG)
│   ├── rusty_sat_modifiers/← Atmospheric and geometric corrections
│   └── rusty_sat_cli/     ← CLI entry point
├── satpy/                 ← Python Satpy reference (read-only, not compiled)
├── deps/                  ← Python dependency references (read-only)
└── local_data/            ← Test data (git-ignored)
```

---

## Adding a New Reader

1. Read the corresponding Satpy Python reader in `satpy/satpy/readers/` for reference behavior
2. Read the file format specification / user guide
3. Create a new module in `crates/rusty_sat_readers/src/`
4. Implement `Reader` trait (at minimum: `name()`, `available_dataset_ids()`, `load()`)
5. Add re-exports to `crates/rusty_sat_readers/src/lib.rs`
6. Write unit tests with synthetic data
7. Write integration tests with real data (if available)

### Reader Trait

```rust
pub trait Reader {
    /// Unique reader name (e.g., "ahi_hsd", "modis_l1b").
    fn name(&self) -> &str;

    /// All DataIds this reader can produce.
    fn available_dataset_ids(&self) -> Vec<DataId>;

    /// Load a single dataset by DataId.
    fn load(&self, id: &DataId) -> Result<Dataset>;
}
```

---

## Adding a New Output Format

1. Create a writer struct implementing `Writer` (and optionally `DatasetWriter`) in `crates/rusty_sat_writers/src/`
2. Implement `save_dataset()` — convert `Dataset` → bytes → file
3. Add to `BuiltinWriter` enum and `BuiltinWriterFactory` in `lib.rs`
4. Write tests that verify file format magic bytes and structure

---

## Adding a New Modifier

1. Read the corresponding Python reference in `deps/pyspectral/` or `deps/pyorbital/`
2. Create a new module in `crates/rusty_sat_modifiers/src/`
3. Implement the correction logic with consuming APIs where possible
4. Add parallel processing via `rayon` for grids >10k pixels
5. Write unit tests with synthetic data and integration tests with real data

---

## Reference Python Code

The `satpy/` directory contains the full Python Satpy source as **read-only design reference**. Key files:

| Python File | Rust Equivalent |
|-------------|-----------------|
| `satpy/readers/ahi_hsd.py` | `crates/rusty_sat_readers/src/ahi_hsd.rs` |
| `satpy/readers/ahi_l2_nc.py` | `crates/rusty_sat_readers/src/ahi_l2_nc.rs` |
| `satpy/etc/readers/ahi_hsd.yaml` | Config parsed by `YamlMetadataReader` |
| `pyspectral/rayleigh.py` | `crates/rusty_sat_modifiers/src/rayleigh.rs` (delegates to `rustyspectral` crate) |
| `pyorbital/astronomy.py` | `crates/rusty_sat_modifiers/src/astronomy.rs` |

The Python code documents expected behavior (dtype layouts, calibration formulas, segment numbering rules). It is **never** compiled or executed by the Rust build.

---

## Getting Help

- Check [AGENTS.md](AGENTS.md) for the full roadmap and implementation state
- Check [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for crate-level documentation
- Run `cargo doc --open` for rustdoc API documentation
