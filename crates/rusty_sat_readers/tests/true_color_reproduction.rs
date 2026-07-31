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
    rayleigh_correct_with_sun_zenith, Atmosphere, RayleighConfig, RayleighCorrector, UtcInstant,
};
use rusty_sat_readers::{AhiCalibration, AhiHsdFileHandler, AhiHsdReader, AhiSegmentInfo};
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
fn correct_band(ds: Dataset, t: UtcInstant, nm: f64, max_sza: f64) -> Dataset {
    rayleigh_correct_with_sun_zenith(lut!(build_corrector(nm)), ds, None, t, max_sza)
        .expect("combined")
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

    // ── Step 2: B02 (green) correct at native 1 km + sanity check ────────
    eprintln!("--- B02: correct (1 km) ---");
    let b02_id = DataQuery::named("B02")
        .expect("query")
        .best_match(scene.available_dataset_ids().iter())
        .expect("B02 id")
        .clone();
    let orig_mean = finite_mean(scene.get(&b02_id).unwrap().array().expect("arr"));
    let d02_corr = correct_band(scene.remove_dataset(&b02_id).unwrap(), t, 510.0, max_sza);
    let corr_mean = finite_mean(d02_corr.array().expect("arr"));
    assert_eq!(
        d02_corr.attr("modifier").and_then(MetadataValue::as_str),
        Some("combined_sun_zenith_rayleigh_correction")
    );
    assert!(
        corr_mean < orig_mean,
        "Rayleigh should reduce ({corr_mean:.2} < {orig_mean:.2})"
    );
    eprintln!("  B02: {orig_mean:.2} → {corr_mean:.2}");

    // ── Step 3: B04 (NIR) correct, then hybrid green at 1 km ─────────────
    eprintln!("--- B04 + hybrid green (1 km) ---");
    let d04_corr = correct_band(take_dataset(&mut scene, "B04"), t, 860.0, max_sza);
    let hybrid = SpectralBlender::new("hybrid_green", vec![0.85, 0.15])
        .expect("blender")
        .compose_owned(vec![d02_corr, d04_corr])
        .expect("hybrid");
    assert!(hybrid.array().is_some());

    // ── Step 4-5: B03 (red, 0.5 km) and B01 (blue, 1 km) correct ─────────
    eprintln!("--- B03 (0.5 km) + B01: correct ---");
    let d03_corr = correct_band(take_dataset(&mut scene, "B03"), t, 640.0, max_sza);
    let d01_corr = correct_band(take_dataset(&mut scene, "B01"), t, 470.0, max_sza);
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
    let u8 = finalize_rgb_cira_u8(rgb.array().expect("arr"), 0).expect("u8");
    drop(rgb); // free band-major [3,y,x] f32 (~5.8 GB)
    let png = out.join("true_color_05km.png");
    SimpleImageWriter::default()
        .save_image(&u8, &png)
        .expect("save");
    assert!(png.is_file());
    let sz = std::fs::metadata(&png).expect("meta").len();
    assert!(sz > 1024, "size={sz}");
    eprintln!("  output: {} ({sz} bytes)", png.display());

    eprintln!("\nPASS: True-color 0.5 km pipeline");
}
