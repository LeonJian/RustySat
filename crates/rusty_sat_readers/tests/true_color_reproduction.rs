//! Integration test: True-color reproduction pipeline (0.5 km output).
//!
//! AHI B03 is 0.5 km, B01/B02/B04 are 1 km.  Pipeline:
//!   Scene.load → combined sun-zenith+Rayleigh per band → hybrid green →
//!   SelfSharpenedRgb up-sample to 0.5 km → cira stretch → 8-bit PNG.
//!
//! Memory: f32 calibration, consuming APIs, drop intermediates promptly.
//!
//! Requires HSD data at `local_data/ahi_input/data/20260728/02/`.
//! LUT auto-downloaded via rustyspectral (cache persists after first run).
//!
//! Run:
//!   cargo test --package rusty_sat_readers --test true_color_reproduction --release -- --nocapture

#![allow(clippy::unwrap_used)]

use rusty_sat_composites::{SelfSharpenedRgb, SpectralBlender};
use rusty_sat_core::{AnyDataArray, DataQuery, Dataset, MetadataValue, NumericElement, Scene};
use rusty_sat_image::finalize_rgb_cira_u8;
use rusty_sat_modifiers::{
    rayleigh_correct_with_sun_zenith, sun_zenith_correct, Atmosphere, RayleighConfig,
    RayleighCorrector, RedBandSource, UtcInstant,
};
use rusty_sat_readers::{AhiCalibration, AhiHsdFileHandler, AhiHsdReader, AhiSegmentInfo};
use rusty_sat_resample::{
    area_from_metadata_value, resample_dataset_from_attrs, with_area_attr, ResampleOptions,
};
use rusty_sat_writers::{SimpleImageWriter, Writer};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

// ── Data helpers ────────────────────────────────────────────────────────

fn data_dir() -> Option<PathBuf> {
    if let Ok(dir) = std::env::var("AHI_DATA_DIR") {
        let path = PathBuf::from(dir);
        if path.is_dir() {
            return Some(path);
        }
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("root");
    let d = root.join("local_data/ahi_input/data/20260728/02");
    if d.is_dir() {
        Some(d)
    } else {
        None
    }
}

fn output_dir() -> PathBuf {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("root")
        .join("local_data/ahi_output");
    std::fs::create_dir_all(&dir).ok();
    dir
}

fn scan_hsd_files(dir: &Path) -> BTreeMap<String, Vec<(PathBuf, AhiSegmentInfo)>> {
    let mut by_band: BTreeMap<String, Vec<(PathBuf, AhiSegmentInfo)>> = BTreeMap::new();
    for entry in std::fs::read_dir(dir).into_iter().flatten().flatten() {
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        let Some((band, seg)) = parse_hsd_filename(name) else {
            continue;
        };
        by_band.entry(band).or_default().push((path, seg));
    }
    for files in by_band.values_mut() {
        files.sort_by_key(|(_, seg)| seg.segment_number);
    }
    by_band
}

fn parse_hsd_filename(name: &str) -> Option<(String, AhiSegmentInfo)> {
    if !name.starts_with("HS_") {
        return None;
    }
    let ext = if name.ends_with(".DAT.bz2") {
        ".DAT.bz2"
    } else if name.ends_with(".DAT") {
        ".DAT"
    } else {
        return None;
    };
    let stem = name.strip_suffix(ext)?;
    let parts: Vec<&str> = stem.split('_').collect();
    let band = parts
        .iter()
        .find(|p| p.starts_with('B') && p.len() == 3)?
        .to_string();
    let seg_part = parts.iter().find(|p| p.starts_with('S') && p.len() == 5)?;
    let seg =
        AhiSegmentInfo::new(seg_part[1..3].parse().ok()?, seg_part[3..5].parse().ok()?).ok()?;
    Some((band, seg))
}

fn try_files(dir: &Path, band: &str) -> Option<Vec<(PathBuf, AhiSegmentInfo)>> {
    scan_hsd_files(dir)
        .get(band)
        .cloned()
        .filter(|f| !f.is_empty())
}

fn ahi_time_to_utc(days: f64) -> UtcInstant {
    UtcInstant::from_unix(((days - 40587.0) * 86400.0) as i64)
}

// ── Pipeline helpers ─────────────────────────────────────────────────────

fn build_corrector(nm: f64) -> Option<RayleighCorrector> {
    let cfg = RayleighConfig {
        platform_name: "Himawari-9".into(),
        sensor: "ahi".into(),
        atmosphere: Atmosphere::UsStandard,
        aerosol_type: rusty_sat_modifiers::AerosolType::RayleighOnly,
        reduce_lim_low: 70.0,
        reduce_lim_high: 105.0,
        reduce_strength: 0.0,
    };
    match RayleighCorrector::with_config_auto(cfg.clone(), nm) {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("  auto-download failed: {e}");
            let local = Path::new(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .and_then(Path::parent)
                .expect("root")
                .join("pyspectral_atm_correction_luts_marine_clean_aerosol")
                .join("rayleigh_lut_us-standard.h5");
            if local.is_file() {
                RayleighCorrector::with_config(&local, cfg, nm).ok()
            } else {
                eprintln!("SKIP: LUT not available");
                None
            }
        }
    }
}

macro_rules! lut {
    ($x:expr) => {
        match $x {
            Some(v) => v,
            None => panic!("LUT required but not available"),
        }
    };
}

/// Correct a band with combined sunz+rayleigh (no resampling).
///
/// `red` follows Satpy's `rayleigh_corrected` red prerequisite (the
/// sun-zenith-corrected B03, resampled to the band's grid when needed).
fn correct_band(ds: Dataset, red: RedBandSource, t: UtcInstant, nm: f64, max_sza: f64) -> Dataset {
    rayleigh_correct_with_sun_zenith(lut!(build_corrector(nm)), ds, red, t, max_sza)
        .expect("combined")
}

/// Convert a dataset's runtime-typed array to f64, preserving dims, mask,
/// coordinates, and attrs.
///
/// The area nearest resampler is f64-only (P0.1.1d), so the 0.5 km red band
/// must be promoted before the 0.5→1 km nearest resample. The temporary f64
/// 0.5 km copy is dropped right after resampling to bound peak memory.
fn to_f64_dataset(dataset: &Dataset) -> Dataset {
    let array = dataset.array().expect("array");
    let dims = array.dims().to_vec();
    let values: Vec<f64> = match array {
        AnyDataArray::F32(a) => a.values().iter().map(|v| f64::from(*v)).collect(),
        AnyDataArray::F64(a) => a.values().to_vec(),
        AnyDataArray::U8(a) => a.values().iter().map(|v| f64::from(*v)).collect(),
        AnyDataArray::U16(a) => a.values().iter().map(|v| f64::from(*v)).collect(),
        AnyDataArray::I16(a) => a.values().iter().map(|v| f64::from(*v)).collect(),
    };
    let mask = array.mask().cloned();
    let coords = array.coords().clone();
    let mut da =
        rusty_sat_core::DataArray::<f64>::from_vec_named(array.shape().to_vec(), dims, values)
            .expect("f64 array");
    if let Some(m) = mask {
        da = da.with_mask(m).expect("mask");
    }
    for (name, coord) in coords {
        da = da.with_coordinate(&name, coord).expect("coord");
    }
    let mut ds = Dataset::new(dataset.id().clone()).with_array(da);
    for (key, value) in dataset.attrs() {
        ds.insert_attr(key.clone(), value.clone()).expect("attr");
    }
    ds
}

/// Resolve a loaded band by name and move it out of the Scene (no array copy).
fn take_dataset(scene: &mut Scene, name: &str) -> Dataset {
    let id = DataQuery::named(name)
        .expect("valid query")
        .best_match(scene.available_dataset_ids().iter())
        .expect("band id")
        .clone();
    scene.remove_dataset(&id).expect("band present in scene")
}

/// Mask-aware finite mean over any runtime-typed array, iterating the native
/// dtype directly (no full f64 copy).
fn finite_mean(array: &AnyDataArray) -> f64 {
    let mask = array.mask();
    let mut sum = 0.0;
    let mut count = 0u64;
    match array {
        AnyDataArray::F32(a) => accumulate_numeric(a.values(), mask, &mut sum, &mut count),
        AnyDataArray::F64(a) => accumulate_numeric(a.values(), mask, &mut sum, &mut count),
        AnyDataArray::U8(a) => accumulate_numeric(a.values(), mask, &mut sum, &mut count),
        AnyDataArray::U16(a) => accumulate_numeric(a.values(), mask, &mut sum, &mut count),
        AnyDataArray::I16(a) => accumulate_numeric(a.values(), mask, &mut sum, &mut count),
    }
    if count == 0 {
        0.0
    } else {
        sum / count as f64
    }
}

fn accumulate_numeric<T: NumericElement>(
    values: &[T],
    mask: Option<&rusty_sat_core::ValidityMask>,
    sum: &mut f64,
    count: &mut u64,
) {
    for (i, v) in values.iter().enumerate() {
        let value = v.to_f64();
        if value.is_finite() && mask.is_none_or(|m| m.is_masked(i) != Some(true)) {
            *sum += value;
            *count += 1;
        }
    }
}

// ── Test ─────────────────────────────────────────────────────────────────

#[test]
fn true_color_reproduction() {
    let Some(dir) = data_dir() else {
        eprintln!("SKIP: no AHI data");
        return;
    };
    let out = output_dir();
    let max_sza = 95.0; // Satpy SunZenithCorrector default

    eprintln!("\n========== True-Color 0.5 km Pipeline (Scene) ==========\n");

    // ── Step 1: one AHI reader per band, loaded through a Scene ──────────
    let mut obs_time = None;
    let mut readers = Vec::new();
    for band in ["B01", "B02", "B03", "B04"] {
        let Some(files) = try_files(&dir, band) else {
            eprintln!("SKIP: {band} not in test data");
            return;
        };
        let file_type = format!("hsd_{}", band.to_lowercase());
        let handlers: Vec<_> = files
            .iter()
            .map(|(path, seg)| AhiHsdFileHandler::from_path(path, &file_type, *seg).expect("open"))
            .collect();
        if obs_time.is_none() {
            obs_time = Some(ahi_time_to_utc(
                handlers[0].header().basic.observation_start_time_days,
            ));
        }
        readers.push(
            AhiHsdReader::with_calibration(AhiCalibration::Reflectance, handlers)
                .expect("reader")
                // 8-way segment parallelism: bzip2 decompression is
                // single-threaded per segment, so more concurrent segments
                // use more cores during the load phase. Peak assembly memory
                // grows to assembled + 8 segment buffers, which fits after
                // the later pipeline stages were made leaner.
                .with_parallel_segments(8)
                .expect("valid segment parallelism"),
        );
    }
    let t = obs_time.expect("observation time");
    let mut scene = Scene::with_loaders(readers);
    scene
        .load(
            ["B01", "B02", "B03", "B04"]
                .into_iter()
                .map(|name| DataQuery::named(name).expect("valid query")),
        )
        .expect("load bands through Scene");

    assert_eq!(
        scene.available_dataset_names(),
        vec!["B01", "B02", "B03", "B04"]
    );
    assert_eq!(scene.len(), 4);
    assert!(scene.missing_datasets().is_empty());
    assert_eq!(scene.sensor_names(), vec!["ahi".to_string()]);
    eprintln!("  Scene loaded B01/B02/B03/B04");

    // ── Step 2: B03 sunz-only red band (Satpy rayleigh_corrected red prereq) ─
    // Satpy feeds the sun-zenith-corrected B03 as the red band for the
    // Rayleigh cloud relaxation of every band. For the 1 km bands (B01/B02)
    // it is nearest-resampled to 1 km (we resample the raw B03 first and
    // sunz-correct on the 1 km grid — equivalent within a pixel); for B03
    // itself the red is the in-place sunz-corrected value.
    let b03_raw = take_dataset(&mut scene, "B03"); // 0.5 km raw
    let b02_id = DataQuery::named("B02")
        .expect("query")
        .best_match(scene.available_dataset_ids().iter())
        .expect("B02 id")
        .clone();
    let b02_area = area_from_metadata_value(
        scene
            .get(&b02_id)
            .expect("B02 in scene")
            .attr("area")
            .expect("B02 area attr"),
    )
    .expect("B02 area");
    // The area nearest resampler is f64-only, so promote the 0.5 km raw band
    // for the borrowed 0.5→1 km resample; the temporary f64 copy is dropped
    // right after.
    let b03_f64 = to_f64_dataset(&b03_raw);
    let b03_1km = resample_dataset_from_attrs(&b03_f64, &b02_area, ResampleOptions::default())
        .expect("B03 -> 1 km nearest");
    drop(b03_f64);
    // The resampler leaves the nested `area` attr at the destination id
    // string; re-attach the destination area metadata for downstream
    // angle/geometry consumers.
    let b03_1km = with_area_attr(b03_1km, &b02_area).expect("attach 1 km area");
    let b03_red_1km = sun_zenith_correct(b03_1km, t).expect("sunz-corrected 1 km B03 red");

    // ── Step 3: B02 (green) correct at native 1 km + sanity check ────────
    eprintln!("--- B02: correct (1 km) ---");
    let orig_mean = finite_mean(scene.get(&b02_id).unwrap().array().expect("arr"));
    let d02_corr = correct_band(
        scene.remove_dataset(&b02_id).unwrap(),
        RedBandSource::Dataset(&b03_red_1km),
        t,
        510.0,
        max_sza,
    );
    let corr_mean = finite_mean(d02_corr.array().expect("arr"));
    assert_eq!(
        d02_corr.attr("modifier").and_then(MetadataValue::as_str),
        Some("combined_sun_zenith_rayleigh_correction")
    );
    // No monotonicity guarantee: the sun-zenith amplification (mean factor
    // ~2x on this disk) usually exceeds the Rayleigh subtraction, so the
    // combined mean rises. Assert a sane, finite range instead.
    assert!(
        corr_mean.is_finite() && corr_mean > 0.0 && corr_mean < orig_mean * 4.0,
        "combined correction in a sane range ({corr_mean:.2} vs raw {orig_mean:.2})"
    );
    eprintln!("  B02: {orig_mean:.2} → {corr_mean:.2}");

    // ── Step 4: B04 (NIR) correct, then hybrid green at 1 km ─────────────
    // B04 (0.86 µm) is outside the Rayleigh LUT range (400–800 nm), so its
    // correction is the sunz-only path; the red band is irrelevant there.
    eprintln!("--- B04 + hybrid green (1 km) ---");
    let d04_corr = correct_band(
        take_dataset(&mut scene, "B04"),
        RedBandSource::None,
        t,
        860.0,
        max_sza,
    );
    let hybrid = SpectralBlender::new("hybrid_green", vec![0.85, 0.15])
        .expect("blender")
        .compose_owned(vec![d02_corr, d04_corr])
        .expect("hybrid");
    assert!(hybrid.array().is_some());

    // ── Step 5: B03 (red, 0.5 km) and B01 (blue, 1 km) correct ───────────
    eprintln!("--- B03 (0.5 km) + B01: correct ---");
    // B03's own red is its sunz-corrected self (Satpy behavior); B01 uses the
    // 1 km resampled B03-sunz red.
    let d03_corr = correct_band(
        b03_raw,
        RedBandSource::SunZenithCorrectedVis,
        t,
        640.0,
        max_sza,
    );
    let d01_corr = correct_band(
        take_dataset(&mut scene, "B01"),
        RedBandSource::Dataset(&b03_red_1km),
        t,
        470.0,
        max_sza,
    );
    assert!(scene.is_empty(), "all bands taken out for processing");

    // ── Step 6: SelfSharpenedRgb(R_05, G_1km, B_1km) → 0.5 km RGB ────────
    eprintln!("--- SelfSharpenedRgb composite ---");
    let rgb = SelfSharpenedRgb::new("true_color")
        .expect("compositor")
        .compose_rgb_owned(vec![d03_corr, hybrid, d01_corr])
        .expect("compose");
    {
        let a = rgb.array().expect("arr");
        let s = a.shape().to_vec();
        assert_eq!(s.len(), 3);
        assert_eq!(s[0], 3, "band axis");
        assert_eq!(
            rgb.attr("mode").and_then(MetadataValue::as_str),
            Some("RGB")
        );
        eprintln!("  RGB shape: [{}, {}, {}]", s[0], s[1], s[2]);
    };

    // ── Step 7: fused cira stretch → 8-bit PNG (Satpy true_color_default) ──
    // Memory: band-major f32 [3,y,x] (~5.8 GB) is finalized straight to
    // interleaved u8 with the CIRA stretch applied per pixel, so the
    // interleaved f32 FloatImage intermediate (~5.8 GB) never exists.
    eprintln!("--- cira_stretch + save ---");
    let u8_img = finalize_rgb_cira_u8(rgb.array().expect("arr"), 0).expect("u8 image");
    drop(rgb); // free band-major [3,y,x] f32 (~5.8 GB)

    // ── Step 8: edge-contrast regression assertions ──────────────────────
    // The exact geos inverse (curved-Earth) must brighten the limb regions
    // through the sun-zenith/Rayleigh corrections instead of leaving them
    // flat-dark (pre-fix bottom limb mean ~50, left-limb std ~19). The disk
    // center must be unchanged.
    eprintln!("--- edge contrast verification ---");
    {
        let (h, w) = u8_img.shape();
        let pixels = u8_img.pixels();

        // Disk bounding box over pixels with any content (mean > 5).
        let mut y_min = usize::MAX;
        let mut y_max = 0usize;
        let mut x_min = usize::MAX;
        let mut x_max = 0usize;
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) * 3;
                if (pixels[i] + pixels[i + 1] + pixels[i + 2]) / 3 > 5 {
                    y_min = y_min.min(y);
                    y_max = y_max.max(y);
                    x_min = x_min.min(x);
                    x_max = x_max.max(x);
                }
            }
        }
        let cy = (y_max + y_min) / 2;
        let cx = (x_max + x_min) / 2;
        eprintln!("  disk bbox y {y_min}..{y_max}, x {x_min}..{x_max}, center ({cy},{cx})");

        let region_stats = |y0: usize, y1: usize, x0: usize, x1: usize| {
            // (mean per band, std per band) over pixels with content.
            let mut sums = [0.0f64; 3];
            let mut sumsq = [0.0f64; 3];
            let mut count = 0u64;
            for y in y0..y1 {
                for x in x0..x1 {
                    let i = (y * w + x) * 3;
                    if (pixels[i] + pixels[i + 1] + pixels[i + 2]) / 3 <= 5 {
                        continue;
                    }
                    count += 1;
                    for b in 0..3 {
                        let v = f64::from(pixels[i + b]);
                        sums[b] += v;
                        sumsq[b] += v * v;
                    }
                }
            }
            let n = count as f64;
            let mean: Vec<f64> = sums.iter().map(|s| s / n).collect();
            let std: Vec<f64> = (0..3)
                .map(|b| (sumsq[b] / n - mean[b] * mean[b]).max(0.0).sqrt())
                .collect();
            (mean, std)
        };

        let (c_mean, c_std) = region_stats(cy - 500, cy + 500, cx - 500, cx + 500);
        let (t_mean, t_std) = region_stats(y_min, y_min + 300, cx - 500, cx + 500);
        let (b_mean, b_std) = region_stats(y_max - 300, y_max, cx - 500, cx + 500);
        let (l_mean, l_std) = region_stats(cy - 500, cy + 500, x_min, x_min + 300);

        eprintln!("  center   mean={c_mean:?} std={c_std:?}");
        eprintln!("  top      mean={t_mean:?} std={t_std:?}");
        eprintln!("  bottom   mean={b_mean:?} std={b_std:?}");
        eprintln!("  left     mean={l_mean:?} std={l_std:?}");

        // Center contrast is unchanged by the geometry fix.
        assert!(
            c_mean.iter().all(|m| (90.0..140.0).contains(m)),
            "center means in the expected band: {c_mean:?}"
        );
        assert!(
            c_std.iter().all(|s| *s > 35.0),
            "center contrast preserved: {c_std:?}"
        );
        // The dark flat bottom limb (mean ~50 pre-fix) must be brightened.
        assert!(
            b_mean.iter().all(|m| *m > 120.0),
            "bottom limb brightened by the curved-Earth sunz correction: {b_mean:?}"
        );
        // The left limb must be bright AND retain structure (std > 25).
        assert!(
            l_mean.iter().all(|m| *m > 120.0),
            "left limb brightened: {l_mean:?}"
        );
        assert!(
            l_std[0] > 25.0,
            "left limb contrast recovered (was ~19 pre-fix): {l_std:?}"
        );
        assert!(
            t_std.iter().all(|s| *s > 20.0),
            "top limb contrast: {t_std:?}"
        );
    }

    let png = out.join("true_color_05km.png");
    SimpleImageWriter::default()
        .save_image(&u8_img, &png)
        .expect("save");
    assert!(png.is_file());
    let sz = std::fs::metadata(&png).expect("meta").len();
    assert!(sz > 1024, "size={sz}");
    eprintln!("  output: {} ({sz} bytes)", png.display());

    eprintln!("\nPASS: True-color 0.5 km pipeline");
}
