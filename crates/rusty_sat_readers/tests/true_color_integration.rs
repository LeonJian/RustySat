//! Integration test: Himawari AHI True Color RGB reproduction.
//!
//! Validates the full true-color pipeline:
//! 1. Load B01 (blue), B02 (green), B03 (red) from real HSD data
//! 2. Calibrate all three to reflectance
//! 3. Resample B01/B02 from 1km to B03's 0.5km resolution
//! 4. Compose RGB via `RgbCompositor`
//! 5. Save 8-bit and 16-bit color PNG
//!
//! Requires HSD data files at `local_data/ahi_input/data/20250923/07/`
//! (relative to workspace root). Override with `AHI_DATA_DIR` env var.
//!
//! Run with:
//!   cargo test -p rusty_sat_readers --test true_color_integration --release -- --nocapture

#![allow(clippy::unwrap_used)]

use rusty_sat_composites::RgbCompositor;
use rusty_sat_core::{AnyDataArray, MetadataValue};
use rusty_sat_readers::{AhiCalibration, AhiHsdFileHandler, AhiHsdReader, AhiSegmentInfo, Reader};
use rusty_sat_resample::{area_from_metadata_value, resample_dataset_from_attrs, ResampleOptions};
use rusty_sat_writers::{SimpleImageWriter, Writer};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

// ══════════════════════════════════════════════════════════════════════════════
// Test data helpers
// ══════════════════════════════════════════════════════════════════════════════

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

fn load_band(
    files: &[(PathBuf, AhiSegmentInfo)],
    file_type: &str,
) -> (rusty_sat_core::Dataset, f64) {
    let handlers: Vec<_> = files
        .iter()
        .map(|(path, seg)| {
            AhiHsdFileHandler::from_path(path, file_type, *seg).expect("open segment")
        })
        .collect();
    let obs_time = handlers[0].header().basic.observation_start_time_days;
    let reader =
        AhiHsdReader::with_calibration(AhiCalibration::Reflectance, handlers).expect("reader");
    let id = reader.available_dataset_ids()[0].clone();
    (reader.load(&id).expect("load dataset"), obs_time)
}

#[test]
fn himawari_true_color_reproduction() {
    let Some(dir) = data_dir() else {
        eprintln!("SKIP: AHI test data not found");
        return;
    };
    let out = output_dir();
    let files_by_band = scan_hsd_files(&dir);

    // Require B01, B02, B03
    let Some(b01_files) = files_by_band.get("B01").cloned() else {
        eprintln!("SKIP: B01 not in test data");
        return;
    };
    let Some(b02_files) = files_by_band.get("B02").cloned() else {
        eprintln!("SKIP: B02 not in test data");
        return;
    };
    let Some(b03_files) = files_by_band.get("B03").cloned() else {
        eprintln!("SKIP: B03 not in test data");
        return;
    };

    eprintln!("\n=== Himawari True Color Reproduction ===\n");

    // Step 1: Load bands
    let t_load = Instant::now();
    let (b01, _) = load_band(&b01_files, "hsd_b01");
    let (b02, _) = load_band(&b02_files, "hsd_b02");
    let (b03, _) = load_band(&b03_files, "hsd_b03");
    eprintln!("  Load time: {:.2}s", t_load.elapsed().as_secs_f64());

    // Step 2: Check shapes
    let AnyDataArray::F32(b03_arr) = b03.array().expect("b03 array") else {
        panic!("expected B03 F32");
    };
    let b03_h = b03_arr.shape_nd()[0];
    let b03_w = b03_arr.shape_nd()[1];
    eprintln!("  B03: {b03_h}×{b03_w} (0.5 km)");

    let AnyDataArray::F32(b01_arr) = b01.array().expect("b01 array") else {
        panic!("expected B01 F32");
    };
    eprintln!(
        "  B01: {}×{} (1.0 km)",
        b01_arr.shape_nd()[0],
        b01_arr.shape_nd()[1]
    );

    let AnyDataArray::F32(b02_arr) = b02.array().expect("b02 array") else {
        panic!("expected B02 F32");
    };
    eprintln!(
        "  B02: {}×{} (1.0 km)",
        b02_arr.shape_nd()[0],
        b02_arr.shape_nd()[1]
    );

    // Step 3: Get B03's area as the resample target
    let b03_area_attr = b03.attr("area").expect("B03 must have area attr");
    let b03_area = area_from_metadata_value(b03_area_attr).expect("decode area");

    // Step 4: Resample B01 and B02 to B03's 0.5 km area
    let t_resample = Instant::now();

    let b01_resampled = if b01_arr.shape_nd()[0] == b03_h {
        b01
    } else {
        resample_dataset_from_attrs(&b01, &b03_area, ResampleOptions::native())
            .expect("resample B01")
    };
    let b02_resampled = if b02_arr.shape_nd()[0] == b03_h {
        b02
    } else {
        resample_dataset_from_attrs(&b02, &b03_area, ResampleOptions::native())
            .expect("resample B02")
    };
    eprintln!(
        "  Resample time: {:.2}s",
        t_resample.elapsed().as_secs_f64()
    );

    // Verify resampled shapes match B03
    let r01_arr = b01_resampled.array().expect("b01 resampled array");
    let r02_arr = b02_resampled.array().expect("b02 resampled array");
    assert_eq!(
        r01_arr.shape(),
        &[b03_h, b03_w],
        "B01 resampled shape mismatch"
    );
    assert_eq!(
        r02_arr.shape(),
        &[b03_h, b03_w],
        "B02 resampled shape mismatch"
    );
    eprintln!(
        "  B01→{}×{}, B02→{}×{}",
        r01_arr.shape()[0],
        r01_arr.shape()[1],
        r02_arr.shape()[0],
        r02_arr.shape()[1]
    );

    // Step 5: Compose RGB (R=B03, G=B02, B=B01)
    let t_compose = Instant::now();
    let compositor = RgbCompositor::new("true_color").expect("create compositor");
    let rgb = compositor
        .compose_rgb_owned(vec![b03, b02_resampled, b01_resampled])
        .expect("compose RGB");
    eprintln!("  Compose time: {:.2}s", t_compose.elapsed().as_secs_f64());

    // Step 6: Assert RGB output structure
    let rgb_arr = rgb.array().expect("RGB array");
    assert_eq!(rgb_arr.shape(), &[3, b03_h, b03_w], "RGB shape mismatch");
    assert_eq!(
        rgb_arr.dims(),
        &["bands".to_string(), "y".to_string(), "x".to_string()],
        "RGB dims mismatch"
    );
    assert_eq!(
        rgb.attr("mode").and_then(MetadataValue::as_str),
        Some("RGB"),
        "mode attr must be RGB"
    );
    assert!(
        rgb_arr.coord("bands").is_some(),
        "RGB must have bands coord"
    );
    eprintln!(
        "  RGB: shape={:?}, dims={:?}, dtype={}",
        rgb_arr.shape(),
        rgb_arr.dims(),
        rgb_arr.dtype().name()
    );

    // Step 7: Save 8-bit PNG
    let t_save = Instant::now();
    let png8_path = out.join("true_color.png");
    SimpleImageWriter::default()
        .save_dataset(&rgb, &png8_path)
        .expect("save 8-bit PNG");
    assert!(png8_path.is_file(), "8-bit PNG file must exist");
    let size8 = std::fs::metadata(&png8_path)
        .expect("read PNG metadata")
        .len();
    assert!(size8 > 0, "8-bit PNG must be non-empty");
    eprintln!("  {} → {} bytes", png8_path.display(), size8);

    // Step 8: Save 16-bit PNG
    let png16_path = out.join("true_color_16.png");
    SimpleImageWriter::default()
        .with_16_bit_dataset_output()
        .save_dataset(&rgb, &png16_path)
        .expect("save 16-bit PNG");
    assert!(png16_path.is_file(), "16-bit PNG file must exist");
    let size16 = std::fs::metadata(&png16_path)
        .expect("read PNG metadata")
        .len();
    assert!(size16 > 0, "16-bit PNG must be non-empty");
    eprintln!("  {} → {} bytes", png16_path.display(), size16);

    eprintln!("  Save time: {:.2}s", t_save.elapsed().as_secs_f64());

    // Step 9: Quick sanity — spot-check a few RGB pixel values
    let vals = rgb_arr.values_as_f64();
    let mask = rgb_arr.mask();
    let mut finite_count = 0u64;
    let mut masked_count = 0u64;
    for (i, &v) in vals.iter().enumerate() {
        if mask.as_ref().is_some_and(|m| m.is_masked(i) == Some(true)) {
            masked_count += 1;
        } else if v.is_finite() {
            finite_count += 1;
        }
    }
    let total = vals.len();
    eprintln!("  RGB pixels: total={total}, finite={finite_count}, masked={masked_count}");
    assert!(finite_count > 0, "expected some finite RGB pixels, got 0");

    let total_elapsed =
        t_load.elapsed() + t_resample.elapsed() + t_compose.elapsed() + t_save.elapsed();
    eprintln!(
        "\n  Total pipeline time: {:.2}s",
        total_elapsed.as_secs_f64()
    );
    eprintln!("PASS: Himawari True Color reproduction");
}
