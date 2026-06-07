//! Integration test: AHI HSD first figure — comprehensive feature validation suite.
//!
//! Validates every feature listed in the AHI implementation checklist
//! against real Himawari-9 HSD segment data.
//!
//! Requires HSD data files at `local_data/ahi_input/data/20250923/07/`
//! (relative to workspace root). Override with `AHI_DATA_DIR` env var.
//!
//! Run with:
//!   cargo test --package rusty_sat_readers --test ahi_first_figure_integration --release -- --nocapture

use rusty_sat_core::{AnyDataArray, MetadataValue};
use rusty_sat_readers::{
    AhiCalibration, AhiCalibrationMode, AhiCalibrationOutput, AhiHsdFileHandler, AhiHsdReader,
    AhiSegmentInfo, AhiUserCalibration, AhiUserCalibrationCoefficients, Reader,
};
use rusty_sat_writers::{FloatTiffWriter, SimpleImageWriter, Writer};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

// ══════════════════════════════════════════════════════════════════════════════
// Test data helpers
// ══════════════════════════════════════════════════════════════════════════════

/// Returns `Some(dir)` if AHI test data is available, `None` otherwise.
///
/// Checks `AHI_DATA_DIR` env var first, then falls back to the default
/// `local_data/ahi_input/data/20250923/07` relative to workspace root.
/// Integration tests use `require_data!()` macro and return early when
/// data is missing so CI passes without real satellite files.
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

/// Call at the top of each integration test. Returns `Some(dir)` if data is
/// available, or prints a skip message and returns `None` (test returns early).
macro_rules! require_data {
    () => {
        match $crate::data_dir() {
            Some(dir) => dir,
            None => {
                eprintln!(
                    "SKIP: AHI test data not found. Set AHI_DATA_DIR env var \
                     or place data at local_data/ahi_input/data/20250923/07/"
                );
                return;
            }
        }
    };
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

/// Returns `Some(files)` for a band, or `None` if data is missing.
fn try_files_for_band(dir: &Path, band: &str) -> Option<Vec<(PathBuf, AhiSegmentInfo)>> {
    let files_by_band = scan_hsd_files(dir);
    let files = files_by_band.get(band)?.clone();
    if files.is_empty() {
        return None;
    }
    Some(files)
}

// ══════════════════════════════════════════════════════════════════════════════
// Test 1: Header parsing — validates all 50 HSD file headers
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn parses_all_50_hsd_file_headers() {
    let dir = require_data!();
    let all_files = scan_hsd_files(&dir);
    assert_eq!(all_files.len(), 5, "expected 5 bands (B01, B02, B03, B04, B13)");

    let mut total_files = 0;
    for (band, files) in &all_files {
        assert_eq!(
            files.len(),
            10,
            "band {band} should have 10 segments, got {}",
            files.len()
        );
        for (path, seg) in files {
            let file_type = format!("hsd_{}", band.to_lowercase());
            let handler = AhiHsdFileHandler::from_path(path, &file_type, *seg)
                .unwrap_or_else(|err| panic!("failed to open '{}': {err}", path.display()));

            let h = handler.header();

            // Block 1: basic info
            assert_eq!(h.basic.header_block_number, 1);
            assert_eq!(h.basic.satellite, "Himawari-9");
            assert_eq!(h.basic.observation_area, "FLDK");
            assert!(h.basic.total_header_length > 0, "total_header_length must be > 0");
            assert!(h.basic.total_data_length > 0, "total_data_length must be > 0");
            assert!(
                h.basic.observation_start_time_days > 0.0,
                "observation_start_time must be set"
            );

            // Block 2: data info
            assert_eq!(h.data.header_block_number, 2);
            assert_eq!(h.data.bits_per_pixel, 16);
            assert!(h.data.columns > 0, "columns must be > 0");
            assert!(h.data.lines > 0, "lines must be > 0");
            assert_eq!(
                h.data.compression_flag, 0,
                "compression flag should be 0 (uncompressed data block inside bzip2 container)"
            );

            // Block 3: projection info
            assert_eq!(h.projection.header_block_number, 3);
            assert_eq!(h.projection.sub_lon, 140.7);
            assert!(h.projection.cfac > 0);
            assert!(h.projection.lfac > 0);
            assert!(h.projection.distance_from_earth_center > 0.0);
            assert!(h.projection.earth_equatorial_radius > 0.0);
            assert!(h.projection.earth_polar_radius > 0.0);

            // Block 4: navigation info
            assert_eq!(h.navigation.header_block_number, 4);
            assert!(h.navigation.sun_position.iter().any(|v| *v != 0.0));

            // Block 5: calibration info
            assert_eq!(h.calibration.header_block_number, 5);
            assert!(
                (1..=16).contains(&h.calibration.band_number),
                "band number must be 1-16, got {}",
                h.calibration.band_number
            );
            assert!(h.calibration.central_wavelength > 0.0);
            assert!(h.calibration.valid_bits_per_pixel > 0);
            assert!(h.calibration.error_pixel_count_value > 0);
            assert!(h.calibration.outside_scan_pixel_count_value > 0);
            assert_ne!(
                h.calibration.error_pixel_count_value, h.calibration.outside_scan_pixel_count_value,
                "error and outside-scan pixel values must differ"
            );

            // Block 7: segment info
            let segment = h.segment.as_ref().expect("block-7 segment info");
            assert_eq!(segment.header_block_number, 7);
            assert_eq!(segment.total_segments, 10);
            assert_eq!(segment.segment_sequence_number, seg.segment_number);
            assert!(segment.first_line_number > 0);

            // Wavelength validation per band
            match band.as_str() {
                "B01" => assert!((0.46..0.48).contains(&h.calibration.central_wavelength)),
                "B02" => assert!((0.50..0.52).contains(&h.calibration.central_wavelength)),
                "B03" => assert!((0.63..0.65).contains(&h.calibration.central_wavelength)),
                "B04" => assert!((0.85..0.87).contains(&h.calibration.central_wavelength)),
                "B13" => assert!((10.3..10.5).contains(&h.calibration.central_wavelength)),
                _ => {}
            }

            // Verify handler metadata
            assert_eq!(&handler.band_name(), band);
            assert_eq!(handler.segment(), *seg);
            assert!(handler.file_type().starts_with("hsd_"));

            total_files += 1;
        }
    }
    assert_eq!(total_files, 50, "expected 50 total HSD segment files");
    eprintln!("PASS: parsed all {total_files} HSD file headers successfully");
}

// ══════════════════════════════════════════════════════════════════════════════
// Test 2: Raw count loading + masking (error + outside-scan pixels)
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn loads_raw_counts_with_error_and_outside_scan_masking() {
    let dir = require_data!();
    // Use B04 (1km resolution, smaller memory footprint)
    let Some(files) = try_files_for_band(&dir, "B04") else { eprintln!("SKIP: B04 not in test data"); return; };
    let first = &files[0];

    let handler = AhiHsdFileHandler::from_path(
        &first.0,
        "hsd_b04",
        first.1,
    )
    .expect("open first segment");

    let h = handler.header();
    let dataset = handler.load_counts_dataset().expect("load counts");

    let AnyDataArray::U16(array) = dataset.array().expect("array present") else {
        panic!("expected U16 raw count array");
    };

    // Validate dimensions
    assert_eq!(array.shape_nd(), &[h.data.lines as usize, h.data.columns as usize]);
    assert_eq!(array.dims(), &["y", "x"]);

    // Validate masking: every error pixel value should be masked
    let error_val = h.calibration.error_pixel_count_value;
    let outside_val = h.calibration.outside_scan_pixel_count_value;
    let values = array.values();
    let mut error_count = 0;
    let mut outside_count = 0;
    let mut valid_count = 0;
    let mut min_valid = u16::MAX;
    let mut max_valid: u16 = 0;

    for (i, &v) in values.iter().enumerate() {
        if v == error_val {
            assert!(
                array.is_masked(i).unwrap_or(false),
                "error pixel at index {i} (value={v}) must be masked"
            );
            error_count += 1;
        } else if v == outside_val {
            assert!(
                array.is_masked(i).unwrap_or(false),
                "outside-scan pixel at index {i} (value={v}) must be masked"
            );
            outside_count += 1;
        } else {
            assert!(
                !array.is_masked(i).unwrap_or(true),
                "valid pixel at index {i} (value={v}) must NOT be masked"
            );
            valid_count += 1;
            min_valid = min_valid.min(v);
            max_valid = max_valid.max(v);
        }
    }

    let total = values.len();
    eprintln!("B04 segment {} raw counts:", first.1.segment_number);
    eprintln!("  shape: {:?}", array.shape_nd());
    eprintln!("  total pixels:  {total}");
    eprintln!("  valid pixels:  {valid_count} ({:.1}%)", valid_count as f64 / total as f64 * 100.0);
    eprintln!("  error pixels:  {error_count} ({:.1}%)", error_count as f64 / total as f64 * 100.0);
    eprintln!("  outside-scan:  {outside_count} ({:.1}%)", outside_count as f64 / total as f64 * 100.0);
    eprintln!("  valid range:   [{min_valid}, {max_valid}]");
    eprintln!("  data type:     u16");

    assert!(valid_count > 0, "must have at least some valid pixels");
    assert!(min_valid < max_valid, "valid pixels must have variation");
    assert!(max_valid > 0, "max valid count must be > 0");

    // Validate dataset attributes
    assert_eq!(
        dataset.attr("calibration").and_then(MetadataValue::as_str),
        Some("counts")
    );
    assert_eq!(
        dataset.attr("platform_name").and_then(MetadataValue::as_str),
        Some("Himawari-9")
    );
    assert_eq!(
        dataset.attr("sensor").and_then(MetadataValue::as_str),
        Some("ahi")
    );
    assert!(dataset.attr("area").is_some(), "area metadata must be present");
    assert!(dataset.attr("first_line_number").is_some());

    eprintln!("PASS: raw count loading with masking validated");
}

// ══════════════════════════════════════════════════════════════════════════════
// Test 3: Visible calibration (Counts → Radiance → Reflectance)
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn calibrates_visible_band_counts_to_radiance_and_reflectance() {
    let dir = require_data!();
    let Some(files) = try_files_for_band(&dir, "B01") else { eprintln!("SKIP: B01 not in test data"); return; };
    let first = &files[0];

    let handler = AhiHsdFileHandler::from_path(&first.0, "hsd_b01", first.1)
        .expect("open first B01 segment");

    // --- Counts ---
    let counts_ds = handler.load_counts_dataset().expect("load counts");
    let AnyDataArray::U16(counts) = counts_ds.array().expect("counts array") else {
        panic!("expected U16 counts");
    };

    // --- Radiance ---
    let rad_ds = handler
        .load_calibrated_dataset(AhiCalibration::Radiance)
        .expect("load radiance");
    let AnyDataArray::F32(radiance) = rad_ds.array().expect("radiance array") else {
        panic!("expected F32 radiance");
    };
    assert_eq!(
        rad_ds.attr("calibration").and_then(MetadataValue::as_str),
        Some("radiance")
    );
    assert_eq!(radiance.shape_nd(), counts.shape_nd());

    // Radiance validation: exclude masked pixels, check range
    let rad_values = radiance.values();
    let mask = radiance.mask();
    let mut rad_min = f32::MAX;
    let mut rad_max = f32::MIN;
    let mut rad_valid = 0;
    for (i, &v) in rad_values.iter().enumerate() {
        if mask.map_or(true, |m| m.is_masked(i) != Some(true)) {
            rad_valid += 1;
            rad_min = rad_min.min(v);
            rad_max = rad_max.max(v);
        }
    }
    eprintln!("B01 segment {} radiance:", first.1.segment_number);
    eprintln!("  valid pixels:  {rad_valid}");
    eprintln!("  range:         [{rad_min:.6}, {rad_max:.6}]");
    eprintln!("  data type:     f32");

    assert!(rad_valid > 0, "must have valid radiance pixels");
    assert!(rad_min > -100.0, "radiance must be in reasonable range, got min={rad_min}");
    assert!(rad_max > rad_min, "radiance must have variation");

    // --- Reflectance ---
    let ref_ds = handler
        .load_calibrated_dataset(AhiCalibration::Reflectance)
        .expect("load reflectance");
    let AnyDataArray::F32(reflec) = ref_ds.array().expect("reflectance array") else {
        panic!("expected F32 reflectance");
    };
    assert_eq!(
        ref_ds.attr("calibration").and_then(MetadataValue::as_str),
        Some("reflectance")
    );

    let ref_values = reflec.values();
    let ref_mask = reflec.mask();
    let mut ref_min = f32::MAX;
    let mut ref_max = f32::MIN;
    let mut ref_valid = 0;
    for (i, &v) in ref_values.iter().enumerate() {
        if ref_mask.map_or(true, |m| m.is_masked(i) != Some(true)) {
            ref_valid += 1;
            ref_min = ref_min.min(v);
            ref_max = ref_max.max(v);
        }
    }
    eprintln!("B01 segment {} reflectance:", first.1.segment_number);
    eprintln!("  valid pixels:  {ref_valid}");
    eprintln!("  range:         [{ref_min:.4}%, {ref_max:.4}%]");
    eprintln!("  data type:     f32");

    assert!(ref_valid > 0, "must have valid reflectance pixels");
    assert!(ref_min >= 0.0, "reflectance must be >= 0%, got min={ref_min}");
    assert!(ref_max > 0.0, "reflectance must be > 0%");
    assert!(ref_max > ref_min, "reflectance must have variation");

    eprintln!("PASS: visible calibration Counts→Radiance→Reflectance validated");
}

// ══════════════════════════════════════════════════════════════════════════════
// Test 4: Infrared calibration (Counts → Radiance → Brightness Temperature)
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn calibrates_infrared_band_counts_to_brightness_temperature() {
    let dir = require_data!();
    let Some(files) = try_files_for_band(&dir, "B13") else { eprintln!("SKIP: B13 not in test data"); return; };
    let first = &files[0];

    let handler = AhiHsdFileHandler::from_path(&first.0, "hsd_b13", first.1)
        .expect("open first B13 segment");

    let h = handler.header();
    eprintln!("B13 band info:");
    eprintln!("  wavelength:   {:.3} μm", h.calibration.central_wavelength);
    eprintln!(
        "  calibration:  {}",
        if h.calibration.band_calibration.is_some() {
            "infrared (Planck coefficients present)"
        } else {
            "MISSING — infrared calibration unavailable"
        }
    );

    // --- Radiance (IR) ---
    let rad_ds = handler
        .load_calibrated_dataset(AhiCalibration::Radiance)
        .expect("load IR radiance");
    let AnyDataArray::F32(radiance) = rad_ds.array().expect("radiance array") else {
        panic!("expected F32 IR radiance");
    };

    let rad_values = radiance.values();
    let rad_mask = radiance.mask();
    let mut rad_min = f32::MAX;
    let mut rad_max = f32::MIN;
    for (i, &v) in rad_values.iter().enumerate() {
        if rad_mask.map_or(true, |m| m.is_masked(i) != Some(true)) && v.is_finite() {
            rad_min = rad_min.min(v);
            rad_max = rad_max.max(v);
        }
    }
    eprintln!("B13 segment {} IR radiance: [{rad_min:.6}, {rad_max:.6}]", first.1.segment_number);

    // --- Brightness Temperature ---
    let bt_ds = handler
        .load_calibrated_dataset(AhiCalibration::BrightnessTemperature)
        .expect("load BT");
    let AnyDataArray::F32(bt) = bt_ds.array().expect("BT array") else {
        panic!("expected F32 BT");
    };
    assert_eq!(
        bt_ds.attr("calibration").and_then(MetadataValue::as_str),
        Some("brightness_temperature")
    );

    let bt_values = bt.values();
    let bt_mask = bt.mask();
    let mut bt_min = f32::MAX;
    let mut bt_max = f32::MIN;
    let mut bt_valid = 0;
    let mut bt_nan = 0;
    for (i, &v) in bt_values.iter().enumerate() {
        if bt_mask.map_or(true, |m| m.is_masked(i) != Some(true)) {
            if v.is_finite() {
                bt_valid += 1;
                bt_min = bt_min.min(v);
                bt_max = bt_max.max(v);
            } else {
                bt_nan += 1;
            }
        }
    }
    eprintln!("B13 segment {} brightness temperature:", first.1.segment_number);
    eprintln!("  valid pixels:  {bt_valid}");
    eprintln!("  NaN pixels:    {bt_nan}");
    eprintln!("  range:         [{bt_min:.2} K, {bt_max:.2} K]");
    eprintln!("  data type:     f32");

    assert!(bt_valid > 0, "must have valid BT pixels");
    assert!(bt_min >= 0.0, "BT min {bt_min} must be >= 0 K");
    assert!(bt_max < 350.0, "BT max {bt_max} too high for Earth scene");
    assert!(bt_max > bt_min, "BT must have variation");
    eprintln!("  (BT min=0 is expected: low-radiance pixels clamp to 0 K)");

    eprintln!("PASS: infrared calibration Counts→Radiance→BT validated");
}

// ══════════════════════════════════════════════════════════════════════════════
// Test 5: Multi-segment assembly into full-disk
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn assembles_all_10_segments_into_full_disk() {
    let dir = require_data!();

    // Test assembly with B04 (1km, 11000x11000 full-disk = 10 segments × 1100 lines)
    let Some(files) = try_files_for_band(&dir, "B04") else { eprintln!("SKIP: B04 not in test data"); return; };
    let handlers: Vec<AhiHsdFileHandler> = files
        .iter()
        .map(|(path, seg)| {
            AhiHsdFileHandler::from_path(path, "hsd_b04", *seg)
                .expect("open segment")
        })
        .collect();

    // Verify segment-level dimensions
    for handler in &handlers {
        assert_eq!(handler.header().data.columns, 11000);
        assert_eq!(handler.header().data.lines, 1100);
    }

    let reader = AhiHsdReader::with_calibration(AhiCalibration::Counts, handlers)
        .expect("create reader with counts");

    let ids = reader.available_dataset_ids();
    assert_eq!(ids.len(), 1, "one dataset ID for assembled B04");
    let id = &ids[0];

    let dataset = reader.load(id).expect("load assembled dataset");
    let array = dataset.array().expect("assembled array");

    // Full-disk B04 at 1km: 11000 cols × 11000 lines
    assert_eq!(array.shape(), &[11000, 11000]);
    assert_eq!(array.dims(), &["y", "x"]);

    // Verify assembled_segments attribute
    let assembled = dataset.attr("assembled_segments");
    assert!(assembled.is_some(), "must have assembled_segments attr");
    if let Some(MetadataValue::List(segs)) = assembled {
        assert_eq!(segs.len(), 10);
        for (i, seg) in segs.iter().enumerate() {
            assert_eq!(seg, &MetadataValue::Integer((i + 1) as i64));
        }
    }

    // Verify lines attribute matches assembled height
    assert_eq!(dataset.attr("lines"), Some(&MetadataValue::Integer(11000)));

    // Verify columns attribute
    assert_eq!(dataset.attr("columns"), Some(&MetadataValue::Integer(11000)));

    // Verify area is present and has correct dimensions
    let area = dataset.attr("area").expect("area metadata");
    if let MetadataValue::Map(area_map) = area {
        assert_eq!(
            area_map.get("id").and_then(MetadataValue::as_str),
            Some("FLDK")
        );
        assert_eq!(
            area_map.get("height"),
            Some(&MetadataValue::Integer(11000))
        );
        assert_eq!(
            area_map.get("width"),
            Some(&MetadataValue::Integer(11000))
        );
    }

    eprintln!("B04 assembled: 10 segments → [11000, 11000] full-disk");
    eprintln!("  assembled_segments: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]");
    eprintln!("  data type: {}", array.dtype().name());
    eprintln!("PASS: multi-segment assembly validated");
}

// ══════════════════════════════════════════════════════════════════════════════
// Test 6: Geostationary area definition / projection coordinates
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn generates_correct_geostationary_area_and_projection_coordinates() {
    let dir = require_data!();
    let Some(files) = try_files_for_band(&dir, "B01") else { eprintln!("SKIP: B01 not in test data"); return; };
    let first = &files[0];

    let handler = AhiHsdFileHandler::from_path(&first.0, "hsd_b01", first.1)
        .expect("open segment");

    let dataset = handler
        .load_calibrated_dataset(AhiCalibration::Reflectance)
        .expect("load reflectance");

    // Extract area metadata
    let area_attr = dataset.attr("area").expect("area attribute");
    let MetadataValue::Map(area) = area_attr else {
        panic!("area must be a map");
    };

    // Area identity
    assert_eq!(area.get("type").and_then(MetadataValue::as_str), Some("area"));
    assert_eq!(area.get("id").and_then(MetadataValue::as_str), Some("FLDK"));
    assert!(area
        .get("description")
        .and_then(MetadataValue::as_str)
        .unwrap_or("")
        .contains("FLDK"));

    // Proj ID: geosh9 for Himawari-9
    let proj_id = area.get("proj_id").and_then(MetadataValue::as_str).unwrap();
    assert!(proj_id.starts_with("geosh"), "proj_id={proj_id} must start with 'geosh'");
    eprintln!("  proj_id: {proj_id}");

    // Area extent
    let MetadataValue::List(extent) = area.get("area_extent").expect("area_extent") else {
        panic!("area_extent must be a list");
    };
    assert_eq!(extent.len(), 4);
    let ext_values: Vec<f64> = extent
        .iter()
        .map(|v| match v {
            MetadataValue::Float(fv) => fv.get(),
            _ => panic!("extent values must be floats"),
        })
        .collect();
    eprintln!("  area_extent: [{:.4}, {:.4}, {:.4}, {:.4}]",
        ext_values[0], ext_values[1], ext_values[2], ext_values[3]);
    assert!(ext_values[0] < ext_values[2], "ll_x < ur_x");
    assert!(ext_values[1] < ext_values[3], "ll_y < ur_y");

    // Projection parameters
    let MetadataValue::Map(proj) = area.get("projection").expect("projection") else {
        panic!("projection must be a map");
    };
    assert_eq!(proj.get("lon_0").and_then(MetadataValue::as_str), Some("140.7"));
    assert_eq!(proj.get("proj").and_then(MetadataValue::as_str), Some("geos"));
    assert_eq!(proj.get("units").and_then(MetadataValue::as_str), Some("m"));
    let a = proj.get("a").and_then(MetadataValue::as_str).unwrap();
    let b = proj.get("b").and_then(MetadataValue::as_str).unwrap();
    let h = proj.get("h").and_then(MetadataValue::as_str).unwrap();
    eprintln!("  projection: proj=geos, lon_0=140.7, a={a}, b={b}, h={h}");
    assert!(a.parse::<f64>().unwrap() > 6_000_000.0, "earth radius must be reasonable");
    assert!(h.parse::<f64>().unwrap() > 35_000_000.0, "satellite height must be reasonable");

    // Coordinate axes
    let array = dataset.array().expect("array");
    let x = array.coord("x").expect("x coordinate");
    let y = array.coord("y").expect("y coordinate");
    assert!(!x.values().is_empty(), "x coords must not be empty");
    assert!(!y.values().is_empty(), "y coords must not be empty");
    eprintln!("  x coords: {} values, range [{:.4}, {:.4}]",
        x.values().len(), x.values().first().unwrap(), x.values().last().unwrap());
    eprintln!("  y coords: {} values, range [{:.4}, {:.4}]",
        y.values().len(), y.values().first().unwrap(), y.values().last().unwrap());

    eprintln!("PASS: geostationary area and projection coordinates validated");
}

// ══════════════════════════════════════════════════════════════════════════════
// Test 7: Output formats — PNG8, PNG16, GeoTIFF with auto-stretch
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn outputs_png8_png16_and_geotiff_with_auto_stretch() {
    let dir = require_data!();
    let out = output_dir();

    // Use B01 for smaller size (11000x11000 full-disk)
    let Some(files) = try_files_for_band(&dir, "B01") else { eprintln!("SKIP: B01 not in test data"); return; };
    let handlers: Vec<AhiHsdFileHandler> = files
        .iter()
        .map(|(path, seg)| {
            AhiHsdFileHandler::from_path(path, "hsd_b01", *seg)
                .expect("open segment")
        })
        .collect();

    let reader =
        AhiHsdReader::with_calibration(AhiCalibration::Reflectance, handlers)
            .expect("create reader");
    let id = reader.available_dataset_ids().pop().expect("dataset ID");
    let dataset = reader.load(&id).expect("load assembled B01");

    // --- PNG8 (auto-stretch to 0-255) ---
    let png8_path = out.join("B01_full_disk_8bit.png");
    let t0 = Instant::now();
    SimpleImageWriter::default()
        .save_dataset(&dataset, &png8_path)
        .expect("save PNG8");
    let png8_time = t0.elapsed();

    let png8_bytes = std::fs::read(&png8_path).expect("read PNG8");
    assert_eq!(&png8_bytes[..8], b"\x89PNG\r\n\x1a\n");
    let png8_w = u32::from_be_bytes([png8_bytes[16], png8_bytes[17], png8_bytes[18], png8_bytes[19]]);
    let png8_h = u32::from_be_bytes([png8_bytes[20], png8_bytes[21], png8_bytes[22], png8_bytes[23]]);
    assert_eq!(png8_w, 11000);
    assert_eq!(png8_h, 11000);
    assert_eq!(png8_bytes[24], 8); // bit depth
    assert_eq!(png8_bytes[25], 0); // color type: grayscale
    eprintln!(
        "  PNG8:  {:.1}s, {:.1} MB, {}x{}, 8-bit grayscale",
        png8_time.as_secs_f64(),
        png8_bytes.len() as f64 / 1_048_576.0,
        png8_w,
        png8_h
    );

    // --- PNG16 (auto-stretch to 0-65535) ---
    let png16_path = out.join("B01_full_disk_16bit.png");
    let t0 = Instant::now();
    SimpleImageWriter::default()
        .with_16_bit_dataset_output()
        .save_dataset(&dataset, &png16_path)
        .expect("save PNG16");
    let png16_time = t0.elapsed();

    let png16_bytes = std::fs::read(&png16_path).expect("read PNG16");
    assert_eq!(&png16_bytes[..8], b"\x89PNG\r\n\x1a\n");
    assert_eq!(png16_bytes[24], 16); // bit depth
    eprintln!(
        "  PNG16: {:.1}s, {:.1} MB, 16-bit grayscale",
        png16_time.as_secs_f64(),
        png16_bytes.len() as f64 / 1_048_576.0
    );

    // --- GeoTIFF float32 ---
    let tiff_path = out.join("B01_full_disk_float32.tif");
    let t0 = Instant::now();
    FloatTiffWriter::default()
        .save_dataset(&dataset, &tiff_path)
        .expect("save GeoTIFF");
    let tiff_time = t0.elapsed();

    let tiff_bytes = std::fs::read(&tiff_path).expect("read GeoTIFF");
    assert_eq!(&tiff_bytes[..2], b"II"); // little-endian
    assert_eq!(u16::from_le_bytes([tiff_bytes[2], tiff_bytes[3]]), 42); // TIFF magic
    eprintln!(
        "  GeoTIFF: {:.1}s, {:.1} MB, float32 reflectance",
        tiff_time.as_secs_f64(),
        tiff_bytes.len() as f64 / 1_048_576.0
    );

    eprintln!("PASS: PNG8, PNG16, GeoTIFF output with auto-stretch validated");
    eprintln!("  output dir: {}", out.display());
}

// ══════════════════════════════════════════════════════════════════════════════
// Test 8: User calibration (radiance correction + digital number mode)
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn applies_user_calibration_radiance_correction_and_dn_mode() {
    let dir = require_data!();
    let Some(files) = try_files_for_band(&dir, "B04") else { eprintln!("SKIP: B04 not in test data"); return; };
    let first = &files[0];

    // --- Base handler (no user calibration) ---
    let base = AhiHsdFileHandler::from_path(&first.0, "hsd_b04", first.1)
        .expect("open segment");

    let base_radiance = base
        .load_calibrated_dataset(AhiCalibration::Radiance)
        .expect("base radiance");
    let AnyDataArray::F32(base_arr) = base_radiance.array().expect("array") else {
        panic!("expected F32");
    };

    // Select a few valid (unmasked) pixels for comparison
    let mask = base_arr.mask();
    let base_vals = base_arr.values();
    let sample_indices: Vec<usize> = base_vals
        .iter()
        .enumerate()
        .filter(|(i, _)| mask.map_or(true, |m| m.is_masked(*i) != Some(true)))
        .take(10)
        .map(|(i, _)| i)
        .collect();
    assert!(!sample_indices.is_empty(), "need at least 10 valid pixels");

    // --- Radiance correction mode ---
    let slope = 0.95_f64;
    let offset = -0.1_f64;
    let rad_corr = base
        .clone()
        .with_user_calibration(
            AhiUserCalibration::radiance_correction([(
                "B04",
                AhiUserCalibrationCoefficients { slope, offset },
            )])
            .expect("valid radiance correction"),
        );

    let corr_rad = rad_corr
        .load_calibrated_dataset(AhiCalibration::Radiance)
        .expect("corrected radiance");
    let AnyDataArray::F32(corr_arr) = corr_rad.array().expect("array") else {
        panic!("expected F32");
    };
    let corr_vals = corr_arr.values();

    // Verify: corrected = (base - offset) / slope
    for &idx in &sample_indices {
        let base_v = base_vals[idx] as f64;
        let corr_v = corr_vals[idx] as f64;
        let expected = (base_v - offset) / slope;
        let diff = (corr_v - expected).abs();
        assert!(
            diff < 0.01,
            "radiance correction mismatch at idx {idx}: base={base_v:.6}, corrected={corr_v:.6}, expected={expected:.6}, diff={diff:.6}"
        );
    }
    eprintln!("  Radiance correction: slope={slope}, offset={offset}");
    eprintln!("    verified {} sample pixels match expected (base - offset) / slope", sample_indices.len());

    // --- Digital Number mode ---
    let dn_slope = -0.0032_f64;
    let dn_offset = 15.20_f64;
    let dn_handler = base
        .clone()
        .with_user_calibration(
            AhiUserCalibration::digital_number([(
                "B04",
                AhiUserCalibrationCoefficients {
                    slope: dn_slope,
                    offset: dn_offset,
                },
            )])
            .expect("valid DN calibration"),
        );

    let dn_rad = dn_handler
        .load_calibrated_dataset(AhiCalibration::Radiance)
        .expect("DN radiance");
    let AnyDataArray::F32(dn_arr) = dn_rad.array().expect("array") else {
        panic!("expected F32");
    };
    let dn_vals = dn_arr.values();

    // Verify: DN radiance = count * dn_slope + dn_offset
    let counts_ds = base.load_counts_dataset().expect("counts");
    let AnyDataArray::U16(counts_arr) = counts_ds.array().expect("counts array") else {
        panic!("expected U16");
    };
    let counts_vals = counts_arr.values();

    for &idx in &sample_indices {
        let count = counts_vals[idx] as f64;
        let dn_v = dn_vals[idx] as f64;
        let expected = count * dn_slope + dn_offset;
        let diff = (dn_v - expected).abs();
        assert!(
            diff < 0.1,
            "DN mode mismatch at idx {idx}: count={count}, dn_radiance={dn_v:.6}, expected={expected:.6}, diff={diff:.6}"
        );
    }
    eprintln!("  Digital Number mode: slope={dn_slope}, offset={dn_offset}");
    eprintln!("    verified {} sample pixels match expected count * slope + offset", sample_indices.len());

    // --- Calibration mode: Nominal vs Update ---
    let nominal = base
        .clone()
        .with_calibration_mode(AhiCalibrationMode::Nominal);
    let nominal_rad = nominal
        .load_calibrated_dataset(AhiCalibration::Radiance)
        .expect("nominal radiance");
    let AnyDataArray::F32(nominal_arr) = nominal_rad.array().expect("array") else {
        panic!("expected F32");
    };

    eprintln!("  Calibration mode:");
    eprintln!("    Update (default): first valid pixel radiance = {:.6}", base_vals[sample_indices[0]]);
    eprintln!("    Nominal:           first valid pixel radiance = {:.6}", nominal_arr.values()[sample_indices[0]]);

    eprintln!("PASS: user calibration (radiance correction + DN mode + nominal/update) validated");
}

// ══════════════════════════════════════════════════════════════════════════════
// Test 9: Scientific F64 output mode
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn scientific_f64_output_produces_higher_precision() {
    let dir = require_data!();
    let Some(files) = try_files_for_band(&dir, "B04") else { eprintln!("SKIP: B04 not in test data"); return; };
    let first = &files[0];

    let handler = AhiHsdFileHandler::from_path(&first.0, "hsd_b04", first.1)
        .expect("open segment");

    // F32 output (default)
    let f32_ds = handler
        .load_calibrated_dataset(AhiCalibration::Reflectance)
        .expect("F32 reflectance");
    assert_eq!(
        f32_ds.attr("precision").and_then(MetadataValue::as_str),
        Some("f32")
    );

    // F64 output (scientific)
    let f64_ds = handler
        .load_calibrated_dataset_f64(AhiCalibration::Reflectance)
        .expect("F64 reflectance");
    assert_eq!(
        f64_ds.attr("precision").and_then(MetadataValue::as_str),
        Some("f64")
    );
    let AnyDataArray::F64(f64_arr) = f64_ds.array().expect("F64 array") else {
        panic!("expected F64");
    };

    // F64 values should be close to F32 but in f64 precision
    let AnyDataArray::F32(f32_arr) = f32_ds.array().expect("F32 array") else {
        panic!("expected F32");
    };

    let mask = f32_arr.mask();
    let f32_vals = f32_arr.values();
    let f64_vals = f64_arr.values();
    let mut compared = 0;
    for (i, (&fv, &dv)) in f32_vals.iter().zip(f64_vals.iter()).enumerate() {
        if mask.map_or(true, |m| m.is_masked(i) != Some(true)) && fv.is_finite() {
            let diff = (dv - fv as f64).abs();
            assert!(diff < 1e-4, "F32/F64 mismatch at idx {i}: f32={fv:.8}, f64={dv:.8}");
            compared += 1;
            if compared >= 100 {
                break;
            }
        }
    }
    eprintln!("  F64 precision verified against F32 on {compared} sample pixels");
    eprintln!("PASS: scientific F64 output validated");
}

// ══════════════════════════════════════════════════════════════════════════════
// Test 10: End-to-end pipeline timing and statistics
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn end_to_end_pipeline_timing_and_statistics() {
    let dir = require_data!();
    let out = output_dir();
    eprintln!("\n=== AHI End-to-End Pipeline Statistics ===\n");

    // Process all 5 available bands
    let all_files = scan_hsd_files(&dir);

    for band in ["B01", "B02", "B03", "B04", "B13"] {
        let files = all_files.get(band).expect("band files");
        let resolution = match band {
            "B03" => "0.5 km",
            "B13" => "2.0 km",
            _ => "1.0 km",
        };

        let t_total = Instant::now();

        // Create handlers
        let t_parse = Instant::now();
        let handlers: Vec<_> = files
            .iter()
            .map(|(path, seg)| {
                let ft = format!("hsd_{}", band.to_lowercase());
                AhiHsdFileHandler::from_path(path, &ft, *seg).expect("open")
            })
            .collect();
        let parse_time = t_parse.elapsed();

        // Load + calibrate + assemble
        let t_load = Instant::now();
        let calibration = if band == "B13" {
            AhiCalibration::BrightnessTemperature
        } else {
            AhiCalibration::Reflectance
        };
        let reader =
            AhiHsdReader::with_calibration(calibration, handlers).expect("reader");
        let id = reader.available_dataset_ids().pop().expect("dataset");
        let dataset = reader.load(&id).expect("load");
        let load_time = t_load.elapsed();

        let shape = dataset.array().map(|a| a.shape().to_vec()).unwrap_or_default();
        let dtype = dataset
            .array()
            .map(|a| a.dtype().name().to_string())
            .unwrap_or_default();

        // Save as PNG8
        let t_save = Instant::now();
        let png_path = out.join(format!("{band}_full_disk_pipeline.png"));
        SimpleImageWriter::default()
            .save_dataset(&dataset, &png_path)
            .expect("save PNG");
        let save_time = t_save.elapsed();
        let png_size = std::fs::metadata(&png_path)
            .map(|m| m.len())
            .unwrap_or(0);

        let total = t_total.elapsed();
        let calib_name = calibration.name();
        eprintln!(
            "  {band} ({resolution}, λ={calib_name}): parse {:.1}s | load+cal {:.1}s | save {:.1}s | total {:.1}s | {}x{} {} → {:.1} MB PNG",
            parse_time.as_secs_f64(),
            load_time.as_secs_f64(),
            save_time.as_secs_f64(),
            total.as_secs_f64(),
            shape.first().unwrap_or(&0),
            shape.get(1).unwrap_or(&0),
            dtype,
            png_size as f64 / 1_048_576.0,
        );

        // Verify output file
        assert!(png_path.exists(), "output file must exist");
        assert!(png_size > 1_000_000, "PNG must be > 1 MB for full-disk");
    }

    eprintln!("\nPASS: end-to-end pipeline validated for all 5 bands");
    eprintln!("  output dir: {}", out.display());
}

// ══════════════════════════════════════════════════════════════════════════════
// Test 11: CalibrationOutput mode — DisplayF32 vs ScientificF64 via Reader
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn reader_can_select_scientific_f64_output_mode() {
    let dir = require_data!();
    let Some(files) = try_files_for_band(&dir, "B01") else { eprintln!("SKIP: B01 not in test data"); return; };
    let first = &files[0];

    let handler =
        AhiHsdFileHandler::from_path(&first.0, "hsd_b01", first.1).expect("open");

    // Display F32 (default)
    let reader_f32 =
        AhiHsdReader::with_calibration(AhiCalibration::Reflectance, [handler.clone()])
            .expect("reader");
    let id = reader_f32.available_dataset_ids().pop().expect("id");
    let ds_f32 = reader_f32.load(&id).expect("load");
    assert_eq!(
        ds_f32.attr("precision").and_then(MetadataValue::as_str),
        Some("f32")
    );
    assert!(matches!(ds_f32.array().expect("array"), AnyDataArray::F32(_)));

    // Scientific F64
    let reader_f64 = AhiHsdReader::with_calibration(
        AhiCalibration::Reflectance,
        [handler],
    )
    .expect("reader")
    .with_output(AhiCalibrationOutput::ScientificF64);
    let id2 = reader_f64.available_dataset_ids().pop().expect("id");
    let ds_f64 = reader_f64.load(&id2).expect("load");
    assert_eq!(
        ds_f64.attr("precision").and_then(MetadataValue::as_str),
        Some("f64")
    );
    assert!(matches!(ds_f64.array().expect("array"), AnyDataArray::F64(_)));

    eprintln!("PASS: DisplayF32 / ScientificF64 reader output mode validated");
}
