//! Integration test: Rayleigh scattering correction on real AHI HSD data.
//!
//! Validates the full Rayleigh correction pipeline using the `rustyspectral`
//! crate for LUT I/O, wavelength adjustment, and trilinear interpolation.
//!
//! Requires HSD data at `local_data/ahi_input/data/20250923/07/` and
//! the Rayleigh LUT at `pyspectral_atm_correction_luts_marine_clean_aerosol/`.
//!
//! Run with:
//!   cargo test --package rusty_sat_readers --test rayleigh_correction_integration --release -- --nocapture
//!
//! Tests skip gracefully when local data or LUT files are not found.

#![allow(clippy::unwrap_used)]

use rusty_sat_core::{AnyDataArray, MetadataValue};
use rusty_sat_modifiers::{
    rayleigh_correct, AerosolType, Atmosphere, RayleighConfig, RayleighCorrector, RedBandSource,
    UtcInstant,
};
use rusty_sat_readers::{AhiCalibration, AhiHsdFileHandler, AhiHsdReader, AhiSegmentInfo, Reader};
use rusty_sat_writers::{SimpleImageWriter, Writer};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

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

fn lut_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace root")
        .join("pyspectral_atm_correction_luts_marine_clean_aerosol")
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

/// Convert AHI HSD observation_start_time_days (days since 1858-11-17,
/// the Modified Julian Day epoch) to a UtcInstant.
fn ahi_time_to_utc(observation_start_time_days: f64) -> UtcInstant {
    let unix_secs = (observation_start_time_days - 40587.0) * 86400.0;
    UtcInstant::from_unix(unix_secs as i64)
}

#[test]
fn rayleigh_correction_on_ahi_b01() {
    let Some(dir) = data_dir() else {
        eprintln!("SKIP: AHI test data not found");
        return;
    };
    let out = output_dir();
    let lut_base = lut_dir();

    let Some(files) = try_files_for_band(&dir, "B01") else {
        eprintln!("SKIP: B01 not in test data");
        return;
    };

    eprintln!("\n=== Rayleigh Correction Integration Test: B01 ===\n");

    // Step 1: Load AHI HSD B01 reflectance
    let t_load = Instant::now();
    let handlers: Vec<_> = files
        .iter()
        .map(|(path, seg)| {
            AhiHsdFileHandler::from_path(path, "hsd_b01", *seg).expect("open segment")
        })
        .collect();

    let obs_time = ahi_time_to_utc(handlers[0].header().basic.observation_start_time_days);
    eprintln!(
        "  Observation time: {} days since J2000",
        obs_time.days_since_j2000()
    );

    let reader =
        AhiHsdReader::with_calibration(AhiCalibration::Reflectance, handlers).expect("reader");
    let id = reader.available_dataset_ids().pop().expect("dataset id");
    let dataset = reader.load(&id).expect("load reflectance");
    let load_time = t_load.elapsed();

    let AnyDataArray::F32(arr) = dataset.array().expect("array") else {
        panic!("expected F32 reflectance");
    };
    let (height, width) = (arr.shape_nd()[0], arr.shape_nd()[1]);
    eprintln!("  Loaded B01: {}x{}", height, width);
    eprintln!("  Load time: {:.2}s", load_time.as_secs_f64());

    assert!(
        dataset.attr("area").is_some(),
        "dataset must have area attr"
    );
    assert!(arr.coord("x").is_some(), "dataset must have x coordinate");
    assert!(arr.coord("y").is_some(), "dataset must have y coordinate");

    let original_vals = arr.values();
    let mask = arr.mask();
    let mut orig_min = f64::INFINITY;
    let mut orig_max = f64::NEG_INFINITY;
    let mut orig_sum = 0.0;
    let mut count = 0u64;
    for (i, &v) in original_vals.iter().enumerate() {
        if v.is_finite() && mask.is_none_or(|m| m.is_masked(i) != Some(true)) {
            let v = v as f64;
            orig_min = orig_min.min(v);
            orig_max = orig_max.max(v);
            orig_sum += v;
            count += 1;
        }
    }
    let orig_mean = if count > 0 {
        orig_sum / count as f64
    } else {
        0.0
    };
    eprintln!(
        "  Original reflectance: min={:.4}, max={:.4}, mean={:.4} ({} valid pixels)",
        orig_min, orig_max, orig_mean, count
    );

    // Step 2: Load the Rayleigh LUT (rustyspectral handles LUT I/O internally)
    let t_lut = Instant::now();
    let lut_path = lut_base.join("rayleigh_lut_us-standard.h5");
    if !lut_path.is_file() {
        eprintln!("SKIP: Rayleigh LUT not found at {}", lut_path.display());
        return;
    }
    let wavelength_nm = 470.0;
    let corrector = RayleighCorrector::with_config(
        &lut_path,
        RayleighConfig {
            platform_name: "Himawari-8".into(),
            sensor: "ahi".into(),
            atmosphere: Atmosphere::UsStandard,
            aerosol_type: AerosolType::MarineCleanAerosol,
            reduce_lim_low: 70.0,
            reduce_lim_high: 105.0,
            reduce_strength: 0.6,
        },
        wavelength_nm,
    )
    .expect("create corrector");
    eprintln!("  LUT load time: {:.2}s", t_lut.elapsed().as_secs_f64());

    // Step 3: Apply Rayleigh correction
    let t_corr = Instant::now();
    let corrected = rayleigh_correct(corrector, dataset, RedBandSource::None, obs_time)
        .expect("rayleigh correction");
    let corr_time = t_corr.elapsed();
    eprintln!("  Correction time: {:.2}s", corr_time.as_secs_f64());

    // Step 4: Verify the correction
    let corr_array = corrected.array().expect("corrected array");
    let corr_vals = corr_array.values_as_f64();
    let corr_mask = corr_array.mask();

    let mut corr_min = f64::INFINITY;
    let mut corr_max = f64::NEG_INFINITY;
    let mut corr_sum = 0.0;
    let mut corr_count = 0u64;
    for (i, &v) in corr_vals.iter().enumerate() {
        if v.is_finite() && corr_mask.is_none_or(|m| m.is_masked(i) != Some(true)) {
            corr_min = corr_min.min(v);
            corr_max = corr_max.max(v);
            corr_sum += v;
            corr_count += 1;
        }
    }
    let corr_mean = if corr_count > 0 {
        corr_sum / corr_count as f64
    } else {
        0.0
    };
    eprintln!(
        "  Corrected reflectance: min={:.4}, max={:.4}, mean={:.4} ({} valid pixels)",
        corr_min, corr_max, corr_mean, corr_count
    );

    assert!(
        corr_mean < orig_mean,
        "corrected mean ({corr_mean:.4}) should be < original mean ({orig_mean:.4})"
    );
    eprintln!(
        "  Mean reflectance reduction: {:.4} ({:.1}%)",
        orig_mean - corr_mean,
        (orig_mean - corr_mean) / orig_mean * 100.0
    );

    // Verify modifier metadata.
    assert_eq!(
        corrected.attr("modifier").and_then(MetadataValue::as_str),
        Some("rayleigh_correction")
    );
    assert_eq!(
        corrected.attr("atmosphere").and_then(MetadataValue::as_str),
        Some("us-standard")
    );

    // Step 5: Save before/after PNG comparison
    let t_save = Instant::now();

    let handlers2: Vec<_> = files
        .iter()
        .map(|(path, seg)| {
            AhiHsdFileHandler::from_path(path, "hsd_b01", *seg).expect("open segment")
        })
        .collect();
    let reader2 =
        AhiHsdReader::with_calibration(AhiCalibration::Reflectance, handlers2).expect("reader");
    let id2 = reader2.available_dataset_ids().pop().expect("dataset id");
    let original_ds = reader2.load(&id2).expect("reload original");

    let orig_png = out.join("B01_rayleigh_before.png");
    SimpleImageWriter::default()
        .save_dataset(&original_ds, &orig_png)
        .expect("save original PNG");

    let corr_png = out.join("B01_rayleigh_after.png");
    SimpleImageWriter::default()
        .save_dataset(&corrected, &corr_png)
        .expect("save corrected PNG");

    eprintln!("  Save time: {:.2}s", t_save.elapsed().as_secs_f64());
    eprintln!("  Original PNG:  {}", orig_png.display());
    eprintln!("  Corrected PNG: {}", corr_png.display());

    let total = t_load.elapsed() + t_lut.elapsed() + corr_time + t_save.elapsed();
    eprintln!("\n  Total pipeline time: {:.2}s", total.as_secs_f64());
    eprintln!("PASS: Rayleigh correction on B01 validated");
}

#[test]
fn rayleigh_correction_on_ahi_b03_with_red_band() {
    let Some(dir) = data_dir() else {
        eprintln!("SKIP: AHI test data not found");
        return;
    };
    let out = output_dir();
    let lut_base = lut_dir();

    let Some(files_b03) = try_files_for_band(&dir, "B03") else {
        eprintln!("SKIP: B03 not in test data");
        return;
    };
    let Some(files_b02) = try_files_for_band(&dir, "B02") else {
        eprintln!("SKIP: B02 not in test data");
        return;
    };

    eprintln!("\n=== Rayleigh Correction Integration Test: B03 with red band B02 ===\n");

    let handlers_b03: Vec<_> = files_b03
        .iter()
        .map(|(path, seg)| {
            AhiHsdFileHandler::from_path(path, "hsd_b03", *seg).expect("open b03 segment")
        })
        .collect();
    let obs_time = ahi_time_to_utc(handlers_b03[0].header().basic.observation_start_time_days);

    let reader_b03 = AhiHsdReader::with_calibration(AhiCalibration::Reflectance, handlers_b03)
        .expect("b03 reader");
    let id_b03 = reader_b03.available_dataset_ids().pop().expect("b03 id");
    let b03_dataset = reader_b03.load(&id_b03).expect("load b03");

    let AnyDataArray::F32(b03_arr) = b03_dataset.array().expect("b03 array") else {
        panic!("expected F32");
    };
    eprintln!("  B03: {}x{}", b03_arr.shape_nd()[0], b03_arr.shape_nd()[1]);

    let handlers_b02: Vec<_> = files_b02
        .iter()
        .map(|(path, seg)| {
            AhiHsdFileHandler::from_path(path, "hsd_b02", *seg).expect("open b02 segment")
        })
        .collect();
    let reader_b02 = AhiHsdReader::with_calibration(AhiCalibration::Reflectance, handlers_b02)
        .expect("b02 reader");
    let id_b02 = reader_b02.available_dataset_ids().pop().expect("b02 id");
    let b02_dataset = reader_b02.load(&id_b02).expect("load b02");

    let AnyDataArray::F32(b02_arr) = b02_dataset.array().expect("b02 array") else {
        panic!("expected F32");
    };
    eprintln!(
        "  B02 (red): {}x{}",
        b02_arr.shape_nd()[0],
        b02_arr.shape_nd()[1]
    );

    let red_source = if b03_arr.shape_nd() == b02_arr.shape_nd() {
        eprintln!("  B03 and B02 shapes match — using cloud relaxation");
        RedBandSource::Dataset(&b02_dataset)
    } else {
        eprintln!(
            "  B03 ({}) and B02 ({}) shapes differ — skipping cloud relaxation",
            b03_arr.shape_nd()[0],
            b02_arr.shape_nd()[0]
        );
        RedBandSource::None
    };

    let lut_path = lut_base.join("rayleigh_lut_us-standard.h5");
    if !lut_path.is_file() {
        eprintln!("SKIP: Rayleigh LUT not found at {}", lut_path.display());
        return;
    }
    let wavelength_nm = 640.0;
    let corrector =
        RayleighCorrector::with_config(&lut_path, RayleighConfig::default(), wavelength_nm)
            .expect("create corrector");

    let corrected = rayleigh_correct(corrector, b03_dataset, red_source, obs_time)
        .expect("rayleigh correction");

    let corr_arr = corrected.array().expect("corrected array");
    eprintln!(
        "  Corrected B03: shape={:?}, dtype={}",
        corr_arr.shape(),
        corr_arr.dtype().name()
    );

    let corr_png = out.join("B03_rayleigh_corrected.png");
    SimpleImageWriter::default()
        .save_dataset(&corrected, &corr_png)
        .expect("save corrected PNG");
    eprintln!("  Output: {}", corr_png.display());

    eprintln!("PASS: Rayleigh correction on B03 with red band validated");
}
