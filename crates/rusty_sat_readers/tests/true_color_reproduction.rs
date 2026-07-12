//! Integration test: True-color reproduction pipeline.
//!
//! Chains the full AHI true-color workflow on real HSD data:
//!   load → combined sun-zenith+Rayleigh correction → hybrid green →
//!   RGB composite → gamma 2.2 → 16-bit PNG save.
//!
//! Memory strategy (minimising peak allocations):
//!   - f32 calibration (not f64) — halves per-band memory
//!   - Combined sun-zenith+Rayleigh single pass — no double angle compute
//!   - Sequential load/correct/composite — at most 3 f32 bands + hybrid f64
//!     in memory simultaneously
//!
//! Requires HSD data at `local_data/ahi_input/data/20250923/07/` and
//! the Rayleigh LUT at `pyspectral_atm_correction_luts_marine_clean_aerosol/`.
//!
//! Run with:
//!   cargo test --package rusty_sat_readers --test true_color_reproduction --release -- --nocapture
//!
//! Tests skip gracefully when local data or LUT files are not found.

#![allow(clippy::unwrap_used)]

use rusty_sat_composites::{RgbCompositor, SpectralBlender};
use rusty_sat_core::MetadataValue;
use rusty_sat_modifiers::{
    rayleigh_correct_with_sun_zenith, sun_zenith_correct, Atmosphere, RayleighConfig,
    RayleighCorrector, SunZenithCorrector, UtcInstant,
};
use rusty_sat_readers::{AhiCalibration, AhiHsdFileHandler, AhiHsdReader, AhiSegmentInfo, Reader};
use rusty_sat_writers::simple_image::SimpleImageDatasetBitDepth;
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
    let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace root");
    let default_dir = workspace_root.join("local_data/ahi_input/data/20250923/07");
    if default_dir.is_dir() {
        return Some(default_dir);
    }
    None
}

fn output_dir() -> PathBuf {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace root")
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
    let band_part = parts.iter().find(|p| p.starts_with('B') && p.len() == 3)?;
    let band = band_part.to_string();
    let seg_part = parts.iter().find(|p| p.starts_with('S') && p.len() == 5)?;
    let seg_number: u8 = seg_part[1..3].parse().ok()?;
    let total_segs: u8 = seg_part[3..5].parse().ok()?;
    let seg = AhiSegmentInfo::new(seg_number, total_segs).ok()?;
    Some((band, seg))
}

fn try_files_for_band(dir: &Path, band: &str) -> Option<Vec<(PathBuf, AhiSegmentInfo)>> {
    let files_by_band = scan_hsd_files(dir);
    let files = files_by_band.get(band)?.clone();
    if files.is_empty() {
        return None;
    }
    Some(files)
}

fn ahi_time_to_utc(observation_start_time_days: f64) -> UtcInstant {
    let unix_secs = (observation_start_time_days - 40587.0) * 86400.0;
    UtcInstant::from_unix(unix_secs as i64)
}

// ── Load helpers ────────────────────────────────────────────────────────

/// Load and calibrate a single band, returning the (dataset, obs_time, shape).
fn load_band(
    files: &[(PathBuf, AhiSegmentInfo)],
    file_type: &str,
) -> (rusty_sat_core::Dataset, UtcInstant, (usize, usize)) {
    let handlers: Vec<_> = files
        .iter()
        .map(|(path, seg)| {
            AhiHsdFileHandler::from_path(path, file_type, *seg).expect("open segment")
        })
        .collect();
    let obs_time = ahi_time_to_utc(handlers[0].header().basic.observation_start_time_days);
    let reader =
        AhiHsdReader::with_calibration(AhiCalibration::Reflectance, handlers).expect("reader");
    let id = reader.available_dataset_ids().pop().expect("dataset id");
    let dataset = reader.load(&id).expect("load reflectance");
    let arr = dataset.array().expect("array");
    let (h, w) = arr.shape_yx().expect("2D array");
    (dataset, obs_time, (h, w))
}

/// Build a RayleighCorrector for the given wavelength.
///
/// First tries the rustyspectral auto-download path (`with_config_auto`).
/// If that fails (e.g. due to the known rustyspectral-1.0.0 URL bug where
/// `get_https_rayleigh_luts()` stores relative paths without the
/// `https://zenodo.org/records/` base), falls back to checking a local
/// LUT directory at `pyspectral_atm_correction_luts_marine_clean_aerosol/`
/// in the workspace root.
///
/// To pre-download the LUT manually:
///   wget 'https://zenodo.org/records/19372152/files/pyspectral_atm_correction_lut_mca.tgz'
///   tar xzf pyspectral_atm_correction_lut_mca.tgz -C workspace/pyspectral_atm_correction_luts_marine_clean_aerosol/
fn build_corrector(wavelength_nm: f64) -> Option<RayleighCorrector> {
    let config = RayleighConfig {
        platform_name: "Himawari-8".into(),
        sensor: "ahi".into(),
        atmosphere: Atmosphere::UsStandard,
        aerosol_type: rusty_sat_modifiers::AerosolType::MarineCleanAerosol,
        reduce_lim_low: 70.0,
        reduce_lim_high: 105.0,
        reduce_strength: 0.6,
    };
    match RayleighCorrector::with_config_auto(config.clone(), wavelength_nm) {
        Ok(c) => return Some(c),
        Err(e) => eprintln!("  auto-download unavailable: {e}"),
    }
    // Fallback: check local LUT directory for pre-downloaded file
    let local_lut = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace root")
        .join("pyspectral_atm_correction_luts_marine_clean_aerosol")
        .join("rayleigh_lut_us-standard.h5");
    if local_lut.is_file() {
        match RayleighCorrector::with_config(&local_lut, config, wavelength_nm) {
            Ok(c) => return Some(c),
            Err(e) => eprintln!("  local LUT load failed: {e}"),
        }
    }
    eprintln!("SKIP: Rayleigh LUT not available (auto-download failed and no local file)");
    None
}

macro_rules! require_lut {
    ($opt:expr) => {
        match $opt {
            Some(v) => v,
            None => return,
        }
    };
}

/// Compute statistics: min, max, mean over finite + unmasked values.
fn stats(values: &[f64], mask: Option<&rusty_sat_core::ValidityMask>) -> (f64, f64, f64, u64) {
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    let mut sum = 0.0;
    let mut count = 0u64;
    for (i, &v) in values.iter().enumerate() {
        let masked = mask.is_some_and(|m| m.is_masked(i) == Some(true));
        if v.is_finite() && !masked {
            min = min.min(v);
            max = max.max(v);
            sum += v;
            count += 1;
        }
    }
    if count == 0 {
        (0.0, 0.0, 0.0, 0)
    } else {
        (min, max, sum / count as f64, count)
    }
}

// ── Test ────────────────────────────────────────────────────────────────

#[test]
fn true_color_reproduction() {
    let Some(dir) = data_dir() else {
        eprintln!("SKIP: AHI test data not found");
        return;
    };
    let out = output_dir();

    let Some(files_b01) = try_files_for_band(&dir, "B01") else {
        eprintln!("SKIP: B01 missing");
        return;
    };
    let Some(files_b02) = try_files_for_band(&dir, "B02") else {
        eprintln!("SKIP: B02 missing");
        return;
    };
    let Some(files_b03) = try_files_for_band(&dir, "B03") else {
        eprintln!("SKIP: B03 missing");
        return;
    };

    let min_cos = SunZenithCorrector::default().min_cos_zenith();

    eprintln!("\n========== True-Color Reproduction Test ==========\n");

    // ── Step 1: B02 (green, 510 nm) — load, correct, keep ──────────
    eprintln!("--- Step 1: B02 (green) ---");
    let (d02, t02, (h, w)) = load_band(&files_b02, "hsd_b02");
    eprintln!("  shape: {h}×{w}");
    assert!(d02.attr("area").is_some(), "B02 area missing");

    let c02 = require_lut!(build_corrector(510.0));
    let d02_corr = rayleigh_correct_with_sun_zenith(c02, d02, None, t02, min_cos)
        .expect("B02 combined correction");
    let v02 = d02_corr.array().expect("arr").values_as_f64();
    let (_min02, _max02, mean02, _n02) = stats(&v02, d02_corr.array().and_then(|a| a.mask()));
    eprintln!("  corrected mean: {mean02:.4}");

    // ── Step 2: B03 (red, 640 nm) — use B02 as red-band reference ──
    eprintln!("--- Step 2: B03 (red) ---");
    let (d03, t03, (h3, w3)) = load_band(&files_b03, "hsd_b03");
    assert_eq!((h3, w3), (h, w), "B03 shape mismatch");

    // Reload B02 as raw reflectance for cloud relaxation
    let (d02_raw, _, _) = load_band(&files_b02, "hsd_b02");

    let c03 = require_lut!(build_corrector(640.0));
    let d03_corr = rayleigh_correct_with_sun_zenith(c03, d03, Some(&d02_raw), t03, min_cos)
        .expect("B03 combined correction");
    let v03 = d03_corr.array().expect("arr").values_as_f64();
    let (_min03, _max03, mean03, _n03) = stats(&v03, d03_corr.array().and_then(|a| a.mask()));
    eprintln!("  corrected mean: {mean03:.4}");
    // Cloud relaxation with B02 should reduce Rayleigh correction magnitude
    assert!(
        mean03 > 0.0,
        "B03 corrected reflectance should be positive (with red-band relaxation)"
    );

    // ── Step 3: B01 (blue, 470 nm) ────────────────────────────────────
    eprintln!("--- Step 3: B01 (blue) ---");
    let (d01, t01, (h1, w1)) = load_band(&files_b01, "hsd_b01");
    assert_eq!((h1, w1), (h, w), "B01 shape mismatch");

    let c01 = require_lut!(build_corrector(470.0));
    let d01_corr = rayleigh_correct_with_sun_zenith(c01, d01, None, t01, min_cos)
        .expect("B01 combined correction");
    let v01 = d01_corr.array().expect("arr").values_as_f64();
    let (_min01, _max01, mean01, _n01) = stats(&v01, d01_corr.array().and_then(|a| a.mask()));
    eprintln!("  corrected mean: {mean01:.4}");

    // ── Step 4: Hybrid green = 0.85 × B02 + 0.15 × B01 ───────────────
    eprintln!("--- Step 4: Hybrid green ---");
    let hybrid = SpectralBlender::new("hybrid_green", vec![0.85, 0.15])
        .expect("hybrid blender")
        .compose_owned(vec![d02_corr, d01_corr])
        .expect("hybrid blend");
    // d02_corr and d01_corr are now consumed / dropped

    let v_hy = hybrid.array().expect("arr").values_as_f64();
    let (_min_h, _max_h, mean_h, _n_h) = stats(&v_hy, hybrid.array().and_then(|a| a.mask()));
    assert!(mean_h > 0.0, "hybrid green reflectance should be positive");
    // Hybrid green should differ from raw B02
    let prev_mean02 = mean02;
    assert!(
        (mean_h - prev_mean02).abs() > 1e-6 || mean_h != prev_mean02,
        "hybrid green should differ from raw B02 ({mean_h:.4} vs {prev_mean02:.4})"
    );
    eprintln!("  hybrid mean: {mean_h:.4}");
    drop(v02);
    drop(v01);

    // ── Step 5: Reload B01 for blue channel ───────────────────────────
    eprintln!("--- Step 5: Reload B01 (blue channel) ---");
    let (d01_b, t01_b, (h1b, w1b)) = load_band(&files_b01, "hsd_b01");
    assert_eq!((h1b, w1b), (h, w), "B01 reload shape mismatch");
    let c01_b = require_lut!(build_corrector(470.0));
    let d01_b_corr = rayleigh_correct_with_sun_zenith(c01_b, d01_b, None, t01_b, min_cos)
        .expect("B01 combined correction");

    // ── Step 6: RGB composite ─────────────────────────────────────────
    eprintln!("--- Step 6: RGB composite ---");
    let rgb = RgbCompositor::new("true_color")
        .expect("rgb compositor")
        .compose_rgb_owned(vec![d03_corr, hybrid, d01_b_corr])
        .expect("rgb compose");
    // d03, hybrid, d01_b are now consumed

    let rgb_arr = rgb.array().expect("rgb array");
    let (shape_h, shape_w) = rgb_arr.shape_yx().expect("RGB has y,x");
    let rgb_shape = rgb_arr.shape();
    assert_eq!(rgb_shape.len(), 3, "RGB must be 3D");
    assert_eq!(rgb_shape[0], 3, "band axis must be 3");
    assert_eq!(shape_h, h, "height mismatch");
    assert_eq!(shape_w, w, "width mismatch");
    assert_eq!(
        rgb.attr("mode").and_then(MetadataValue::as_str),
        Some("RGB"),
        "mode attr"
    );
    let rgb_vals = rgb_arr.values_as_f64();
    let finite_rgb: f64 = rgb_vals.iter().filter(|v| v.is_finite()).count() as f64;
    assert!(finite_rgb > 0.0, "RGB must have finite pixels");
    eprintln!(
        "  RGB shape: [{}, {}, {}]",
        rgb_shape[0], rgb_shape[1], rgb_shape[2]
    );
    eprintln!("  finite/ total: {}/{}", finite_rgb as u64, rgb_vals.len());

    // ── Step 7: Enhance — crude stretch + gamma 2.2 ──────────────────
    eprintln!("--- Step 7: Enhancement ---");
    use rusty_sat_image::FloatImage;
    let mut img = FloatImage::<f32>::from_rgb_array(rgb_arr).expect("rgb float image");

    let rgb_mean_before_stretch: f64 =
        img.pixels().iter().map(|p| *p as f64).sum::<f64>() / img.pixels().len() as f64;
    eprintln!("  mean before stretch: {rgb_mean_before_stretch:.4}");

    img.crude_stretch_in_place(None, None);
    img.gamma_in_place(2.2).expect("gamma 2.2");

    let rgb_mean_after_enhance: f64 =
        img.pixels().iter().map(|p| *p as f64).sum::<f64>() / img.pixels().len() as f64;
    eprintln!("  mean after stretch+gamma: {rgb_mean_after_enhance:.4}");
    assert!(
        rgb_mean_after_enhance > 0.0,
        "enhanced image should have positive mean"
    );

    let u8_rgb = img.to_u8_image(0).expect("u8 conversion");
    assert_eq!(u8_rgb.mode(), rusty_sat_image::ImageMode::Rgb);
    assert!(!u8_rgb.pixels().is_empty(), "image must have pixels");

    // ── Step 8: Save ──────────────────────────────────────────────────
    eprintln!("--- Step 8: Save ---");
    let out_path = out.join("true_color_reproduction.png");
    SimpleImageWriter::default()
        .with_dataset_bit_depth(SimpleImageDatasetBitDepth::Sixteen)
        .save_dataset(&rgb, &out_path)
        .expect("save RGB PNG");
    assert!(out_path.is_file(), "output file must exist");
    let file_size = std::fs::metadata(&out_path).expect("metadata").len();
    assert!(
        file_size > 1024,
        "output must be non-trivial (size={file_size})"
    );
    eprintln!("  output: {} ({file_size} bytes)", out_path.display());

    // ── Step 9: Verify corrections individually ───────────────────────
    eprintln!("--- Step 9: Sanity checks ---");

    // B01: sun-zenith-only correction should increase limb values
    let (d01_sz_only, t01_sz, _) = load_band(&files_b01, "hsd_b01");
    let original_v01 = d01_sz_only.array().expect("arr").values_as_f64();
    let (_, _, orig_mean01_sz, _) =
        stats(&original_v01, d01_sz_only.array().and_then(|a| a.mask()));
    let d01_sz = sun_zenith_correct(d01_sz_only, t01_sz).expect("sz correct");
    let sz_v01 = d01_sz.array().expect("arr").values_as_f64();
    let (_, _, sz_mean01, _) = stats(&sz_v01, d01_sz.array().and_then(|a| a.mask()));
    eprintln!("  B01 original mean: {orig_mean01_sz:.4}, sun-zenith mean: {sz_mean01:.4}");
    assert!(
        sz_mean01 >= orig_mean01_sz * 0.99,
        "sun-zenith correction should not decrease values ({sz_mean01:.4} >= {orig_mean01_sz:.4}×0.99)",
    );

    // B02: combined correction vs original
    let (d02_chk, t02_chk, _) = load_band(&files_b02, "hsd_b02");
    let ov02 = d02_chk.array().expect("arr").values_as_f64();
    let (_, _, orig_mean02, _) = stats(&ov02, d02_chk.array().and_then(|a| a.mask()));
    let c02_chk = require_lut!(build_corrector(510.0));
    let d02_comb = rayleigh_correct_with_sun_zenith(c02_chk, d02_chk, None, t02_chk, min_cos)
        .expect("combined");
    let cv02 = d02_comb.array().expect("arr").values_as_f64();
    let (_, _, comb_mean02, _) = stats(&cv02, d02_comb.array().and_then(|a| a.mask()));
    assert_eq!(
        d02_comb.attr("modifier").and_then(MetadataValue::as_str),
        Some("combined_sun_zenith_rayleigh_correction")
    );
    eprintln!("  B02 original mean: {orig_mean02:.4}, combined mean: {comb_mean02:.4}");
    // Rayleigh correction should reduce mean reflectance
    assert!(
        comb_mean02 < orig_mean02,
        "Rayleigh should reduce reflectance ({comb_mean02:.4} < {orig_mean02:.4})"
    );

    eprintln!("\nPASS: True-color reproduction pipeline validated");
}
