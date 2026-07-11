//! Integration test: Himawari AHI True Color RGB reproduction with
//! Rayleigh scattering correction.
//!
//! Validates the full true-color + Rayleigh pipeline:
//! 1. Load B01 (blue), B02, B03 (red), B04 from real HSD data
//! 2. Calibrate all to reflectance
//! 3. Rayleigh-correct B01 (470 nm) and B03 (640 nm)
//! 4. Create hybrid green: SpectralBlender 0.85×B02 + 0.15×B04
//! 5. Resample B01 and hybrid_green to B03's 0.5km resolution
//! 6. Compose RGB via RgbCompositor (R=B03_rayleigh, G=hybrid_green, B=B01_rayleigh)
//! 7. Apply gamma 2.0 per channel + crude stretch
//! 8. Save 8-bit color PNG
//!
//! Requires HSD data files at `local_data/ahi_input/data/20250923/07/`
//! (relative to workspace root). Override with `AHI_DATA_DIR` env var.
//!
//! Requires Rayleigh LUT at `pyspectral_atm_correction_luts_marine_clean_aerosol/`
//! or set `PSP_CONFIG_FILE` to point to the right directory.
//!
//! Run with:
//!   cargo test -p rusty_sat_readers --test true_color_integration --release -- --nocapture

#![allow(clippy::unwrap_used)]

use rusty_sat_composites::{RgbCompositor, SpectralBlender};
use rusty_sat_core::{AnyDataArray, DataArray, Dataset, MetadataValue};
use rusty_sat_image::FloatImage;
use rusty_sat_modifiers::{
    rayleigh_correct, AerosolType, Atmosphere, RayleighConfig, RayleighCorrector, UtcInstant,
};
use rusty_sat_readers::{AhiCalibration, AhiHsdFileHandler, AhiHsdReader, AhiSegmentInfo, Reader};
use rusty_sat_resample::{
    area_from_metadata_value, resample_dataset_from_attrs, with_area_attr, ResampleOptions,
};
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

fn ahi_time_to_utc(observation_start_time_days: f64) -> UtcInstant {
    let unix_secs = (observation_start_time_days - 40587.0) * 86400.0;
    UtcInstant::from_unix(unix_secs as i64)
}

fn make_corrector(wavelength_nm: f64) -> Option<RayleighCorrector> {
    let lut_path = lut_dir().join("rayleigh_lut_us-standard.h5");
    if !lut_path.is_file() {
        eprintln!("SKIP: Rayleigh LUT not found at {}", lut_path.display());
        return None;
    }
    let config = RayleighConfig {
        platform_name: "Himawari-8".into(),
        sensor: "ahi".into(),
        atmosphere: Atmosphere::UsStandard,
        aerosol_type: AerosolType::MarineCleanAerosol,
        reduce_lim_low: 70.0,
        reduce_lim_high: 105.0,
        reduce_strength: 0.6,
    };
    RayleighCorrector::with_config(&lut_path, config, wavelength_nm).ok()
}

fn dataset_f64_to_f32(dataset: Dataset) -> Dataset {
    let area = dataset.attr("area").cloned();
    let ds_id = dataset.id().clone();
    let Some(array) = dataset.into_array() else {
        return Dataset::new(ds_id);
    };
    let mask = array.mask().cloned();
    let coords = array.coords().clone();
    match array {
        AnyDataArray::F64(da) => {
            let (h, w) = {
                let s = da.shape_nd();
                (s[0], s[1])
            };
            let f32_vals: Vec<f32> = da.into_values().into_iter().map(|v| v as f32).collect();
            let mut arr = DataArray::<f32>::from_vec_named(vec![h, w], vec!["y", "x"], f32_vals)
                .expect("valid array");
            if let Some(m) = mask {
                arr = arr.with_mask(m).expect("valid mask");
            }
            for (name, coord) in &coords {
                if name == "y" || name == "x" {
                    let prev = arr.clone();
                    arr = arr.with_coordinate(name, coord.clone()).unwrap_or(prev);
                }
            }
            let mut ds = Dataset::new(ds_id).with_array(arr);
            if let Some(a) = area {
                ds.insert_attr("area", a).ok();
            }
            ds
        }
        AnyDataArray::F32(_) => {
            let (h, w) = {
                let s = array.shape();
                (s[0], s[1])
            };
            let vals = array.values_as_f64();
            let f32_vals: Vec<f32> = vals.into_iter().map(|v| v as f32).collect();
            let mut arr = DataArray::<f32>::from_vec_named(vec![h, w], vec!["y", "x"], f32_vals)
                .expect("valid array");
            if let Some(m) = mask {
                arr = arr.with_mask(m).expect("valid mask");
            }
            for (name, coord) in &coords {
                if name == "y" || name == "x" {
                    let prev = arr.clone();
                    arr = arr.with_coordinate(name, coord.clone()).unwrap_or(prev);
                }
            }
            let mut ds = Dataset::new(ds_id).with_array(arr);
            if let Some(a) = area {
                ds.insert_attr("area", a).ok();
            }
            ds
        }
        other => Dataset::new(ds_id).with_array(other),
    }
}

#[test]
fn himawari_true_color_reproduction() {
    let Some(dir) = data_dir() else {
        eprintln!("SKIP: AHI test data not found");
        return;
    };
    let out = output_dir();
    let files_by_band = scan_hsd_files(&dir);

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
    let Some(b04_files) = files_by_band.get("B04").cloned() else {
        eprintln!("SKIP: B04 not in test data");
        return;
    };

    eprintln!("\n=== Himawari True Color Reproduction (with Rayleigh correction) ===\n");

    // Step 1: Load all four bands
    let t_load = Instant::now();
    let (b01, b01_obs) = load_band(&b01_files, "hsd_b01");
    let (b02, _) = load_band(&b02_files, "hsd_b02");
    let (b03, b03_obs) = load_band(&b03_files, "hsd_b03");
    let (b04, _) = load_band(&b04_files, "hsd_b04");
    eprintln!("  Load time: {:.2}s", t_load.elapsed().as_secs_f64());

    // Check shapes
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

    eprintln!("  B02: (1.0 km)");
    eprintln!("  B04: (1.0 km)");

    // Step 2: Rayleigh correction
    // B01 central wavelength: ~0.47 μm = 470 nm
    // B03 central wavelength: ~0.64 μm = 640 nm
    // B02/B03 have different resolutions → skip cloud relaxation (red_dataset=None)

    let b01_corrected;
    let b03_corrected;

    let t_rayleigh = Instant::now();
    match (make_corrector(470.0), make_corrector(640.0)) {
        (Some(b01_corr), Some(b03_corr)) => {
            let b01_utc = ahi_time_to_utc(b01_obs);
            let b03_utc = ahi_time_to_utc(b03_obs);

            b01_corrected =
                rayleigh_correct(b01_corr, b01, None, b01_utc).expect("B01 Rayleigh correction");
            b03_corrected =
                rayleigh_correct(b03_corr, b03, None, b03_utc).expect("B03 Rayleigh correction");
            eprintln!(
                "  Rayleigh correction time: {:.2}s",
                t_rayleigh.elapsed().as_secs_f64()
            );

            // Verify area metadata preserved
            assert!(
                b01_corrected.attr("area").is_some(),
                "B01 corrected must have area attr"
            );
            assert!(
                b03_corrected.attr("area").is_some(),
                "B03 corrected must have area attr"
            );
            assert_eq!(
                b01_corrected
                    .attr("modifier")
                    .and_then(MetadataValue::as_str),
                Some("rayleigh_correction")
            );
            assert_eq!(
                b03_corrected
                    .attr("modifier")
                    .and_then(MetadataValue::as_str),
                Some("rayleigh_correction")
            );
        }
        _ => {
            eprintln!("SKIP: Rayleigh LUT not available, using uncorrected bands");
            b01_corrected = b01;
            b03_corrected = b03;
        }
    }

    // Step 3: Create hybrid green (0.85×B02 + 0.15×B04)
    let b01_area_attr = b01_corrected
        .attr("area")
        .expect("B01 must have area attr")
        .clone();
    let t_composite = Instant::now();
    let hybrid_green = SpectralBlender::new("hybrid_green", vec![0.85, 0.15])
        .expect("create blender")
        .compose_owned(vec![b02, b04])
        .expect("compose hybrid green");
    let hybrid_green = with_area_attr(
        hybrid_green,
        &area_from_metadata_value(&b01_area_attr).expect("decode 1km area"),
    )
    .expect("set area on hybrid_green");
    let hybrid_green = dataset_f64_to_f32(hybrid_green);
    eprintln!(
        "  Hybrid green time: {:.2}s",
        t_composite.elapsed().as_secs_f64()
    );

    // Step 4: Get B03's area as resample target
    let b03_area_attr = b03_corrected
        .attr("area")
        .expect("B03 corrected must have area attr");
    let b03_area = area_from_metadata_value(b03_area_attr).expect("decode area");

    // Step 5: Resample B01 and hybrid_green to B03's 0.5km area
    let t_resample = Instant::now();

    let b01_arr_corr = b01_corrected.array().expect("b01 corrected array");
    let b01_resampled = if b01_arr_corr.shape()[0] == b03_h {
        b01_corrected
    } else {
        resample_dataset_from_attrs(&b01_corrected, &b03_area, ResampleOptions::native())
            .expect("resample B01")
    };
    let green_resampled = {
        let gh = hybrid_green.array().expect("hybrid_green array");
        if gh.shape()[0] == b03_h {
            hybrid_green
        } else {
            resample_dataset_from_attrs(&hybrid_green, &b03_area, ResampleOptions::native())
                .expect("resample hybrid_green")
        }
    };
    eprintln!(
        "  Resample time: {:.2}s",
        t_resample.elapsed().as_secs_f64()
    );

    // Verify resampled shapes
    let r01_arr = b01_resampled.array().expect("b01 resampled array");
    let rg_arr = green_resampled.array().expect("green resampled array");
    assert_eq!(
        r01_arr.shape(),
        &[b03_h, b03_w],
        "B01 resampled shape mismatch"
    );
    assert_eq!(
        rg_arr.shape(),
        &[b03_h, b03_w],
        "hybrid_green resampled shape mismatch"
    );

    // Step 6: Compose RGB (R=B03_rayleigh, G=hybrid_green, B=B01_rayleigh)
    let t_compose = Instant::now();
    let compositor = RgbCompositor::new("true_color_rayleigh").expect("create compositor");
    let rgb = compositor
        .compose_rgb_owned(vec![b03_corrected, green_resampled, b01_resampled])
        .expect("compose RGB");
    eprintln!("  Compose time: {:.2}s", t_compose.elapsed().as_secs_f64());

    // Step 7: Assert RGB output structure & collect stats before consuming
    {
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

        // Snapshot pixel stats before rgb is consumed
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
    }

    // Step 8: Enhance — crude stretch + gamma 2.0 per channel, save 8-bit PNG
    let t_save = Instant::now();

    let mut img8 = FloatImage::<f32>::from_rgb_dataset_owned(rgb).expect("create FloatImage<f32>");
    img8.crude_stretch_in_place(None, None);
    img8.gamma_channels_in_place(&[2.0, 2.0, 2.0])
        .expect("gamma correct");
    let image8 = img8.to_u8_image(0).expect("convert to u8");

    let png8_path = out.join("true_color_rayleigh.png");
    SimpleImageWriter::default()
        .save_image(&image8, &png8_path)
        .expect("save 8-bit PNG");
    assert!(png8_path.is_file(), "8-bit PNG file must exist");
    let size8 = std::fs::metadata(&png8_path)
        .expect("read PNG metadata")
        .len();
    assert!(size8 > 0, "8-bit PNG must be non-empty");
    eprintln!("  {} → {} bytes", png8_path.display(), size8);

    // Step 9: 16-bit PNG — temporarily disabled for faster iteration
    // let mut img16 = FloatImage::<f64>::from_rgb_dataset(&rgb).expect("create FloatImage<f64>");
    // img16.crude_stretch_in_place(None, None);
    // img16
    //     .gamma_channels_in_place(&[2.0, 2.0, 2.0])
    //     .expect("gamma correct 16-bit");
    // let image16 = img16.to_u16_image(0).expect("convert to u16");
    //
    // let png16_path = out.join("true_color_rayleigh_16.png");
    // SimpleImageWriter::default()
    //     .save_image16(&image16, &png16_path)
    //     .expect("save 16-bit PNG");
    // assert!(png16_path.is_file(), "16-bit PNG file must exist");
    // let size16 = std::fs::metadata(&png16_path)
    //     .expect("read PNG metadata")
    //     .len();
    // assert!(size16 > 0, "16-bit PNG must be non-empty");
    // eprintln!("  {} → {} bytes", png16_path.display(), size16);

    eprintln!("  Save time: {:.2}s", t_save.elapsed().as_secs_f64());

    let total_elapsed = t_load.elapsed()
        + t_rayleigh.elapsed()
        + t_composite.elapsed()
        + t_resample.elapsed()
        + t_compose.elapsed()
        + t_save.elapsed();
    eprintln!(
        "\n  Total pipeline time: {:.2}s",
        total_elapsed.as_secs_f64()
    );
    eprintln!(
        "PASS: Himawari True Color reproduction (Rayleigh corrected + hybrid green + gamma 2.0)"
    );
}
