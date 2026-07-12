//! Integration test: True-color reproduction pipeline (0.5 km output).
//!
//! AHI B03 is 0.5 km, B01/B02 are 1 km.  Pipeline:
//!   load → combined sun-zenith+Rayleigh → up-sample B01/B02 to 0.5 km →
//!   hybrid green → RGB composite → 8-bit PNG (writer handles enhancement).
//!
//! Memory: f32 calibration, consuming APIs, drop intermediates promptly.
//!
//! Requires HSD data at `local_data/ahi_input/data/20250923/07/`.
//! LUT auto-downloaded via rustyspectral (cache persists after first run).
//!
//! Run:
//!   cargo test --package rusty_sat_readers --test true_color_reproduction --release -- --nocapture

#![allow(clippy::unwrap_used)]

use rusty_sat_composites::{RgbCompositor, SpectralBlender};
use rusty_sat_core::MetadataValue;
use rusty_sat_modifiers::{
    rayleigh_correct_with_sun_zenith, Atmosphere, RayleighConfig, RayleighCorrector,
    SunZenithCorrector, UtcInstant,
};
use rusty_sat_readers::{AhiCalibration, AhiHsdFileHandler, AhiHsdReader, AhiSegmentInfo, Reader};
use rusty_sat_resample::{area_from_metadata_value, AreaDefinition, NativeResampler, Resampler};
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
    let d = root.join("local_data/ahi_input/data/20250923/07");
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

fn load_band(
    files: &[(PathBuf, AhiSegmentInfo)],
    ft: &str,
) -> (rusty_sat_core::Dataset, UtcInstant, (usize, usize)) {
    let handlers: Vec<_> = files
        .iter()
        .map(|(p, s)| AhiHsdFileHandler::from_path(p, ft, *s).expect("open"))
        .collect();
    let t = ahi_time_to_utc(handlers[0].header().basic.observation_start_time_days);
    let r = AhiHsdReader::with_calibration(AhiCalibration::Reflectance, handlers).expect("reader");
    let id = r.available_dataset_ids().pop().expect("id");
    let ds = r.load(&id).expect("load");
    let (h, w) = ds.array().expect("arr").shape_yx().expect("2D");
    (ds, t, (h, w))
}

fn build_corrector(nm: f64) -> Option<RayleighCorrector> {
    let cfg = RayleighConfig {
        platform_name: "Himawari-8".into(),
        sensor: "ahi".into(),
        atmosphere: Atmosphere::UsStandard,
        aerosol_type: rusty_sat_modifiers::AerosolType::MarineCleanAerosol,
        reduce_lim_low: 70.0,
        reduce_lim_high: 105.0,
        reduce_strength: 0.6,
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

/// Correct a band and upsample to 0.5 km (repeat 2× if source is 1 km).
fn correct_and_upsample(
    ds: rusty_sat_core::Dataset,
    t: UtcInstant,
    nm: f64,
    red: Option<&rusty_sat_core::Dataset>,
    min_cos: f64,
) -> rusty_sat_core::Dataset {
    let c = lut!(build_corrector(nm));
    let corr = rayleigh_correct_with_sun_zenith(c, ds, red, t, min_cos).expect("combined");
    let area = area_from_metadata_value(corr.attr("area").expect("area")).expect("parse area");
    let (src_h, src_w) = (area.height(), area.width());
    let is_05km = src_h >= 20000; // B03 0.5 km is ~22000, B01/B02/B04 1 km is ~11000
    if is_05km {
        return corr;
    }
    let dest = AreaDefinition::from_parts(
        format!("{}_05", area.id()),
        area.description(),
        area.proj_id(),
        area.projection().clone(),
        src_h * 2,
        src_w * 2,
        area.area_extent(),
    )
    .expect("dest area");
    NativeResampler::new(area)
        .resample_owned(corr, &dest)
        .expect("upsample")
}

// ── Test ─────────────────────────────────────────────────────────────────

#[test]
fn true_color_reproduction() {
    let Some(dir) = data_dir() else {
        eprintln!("SKIP: no AHI data");
        return;
    };
    let out = output_dir();

    let Some(f01) = try_files(&dir, "B01") else {
        eprintln!("SKIP: B01");
        return;
    };
    let Some(f02) = try_files(&dir, "B02") else {
        eprintln!("SKIP: B02");
        return;
    };
    let Some(f03) = try_files(&dir, "B03") else {
        eprintln!("SKIP: B03");
        return;
    };

    let Some(f04) = try_files(&dir, "B04") else {
        eprintln!("SKIP: B04");
        return;
    };

    let min_cos = SunZenithCorrector::default().min_cos_zenith();
    eprintln!("\n========== True-Color 0.5 km Pipeline ==========\n");

    // Step 1: B02 (green) → correct → up-sample to 0.5 km
    eprintln!("--- B02: correct + up-sample to 0.5 km ---");
    let (d02, t02, (h02, w02)) = load_band(&f02, "hsd_b02");
    eprintln!("  1 km shape: {h02}×{w02}");
    let d02_05 = correct_and_upsample(d02, t02, 510.0, None, min_cos);
    let (h05, w05) = d02_05.array().expect("arr").shape_yx().expect("2D");
    eprintln!("  0.5 km shape: {h05}×{w05}");

    // Step 2: B03 (red, 0.5 km) — load first copy for hybrid green blend
    eprintln!("--- B03: correct (for hybrid blend) ---");
    let (d03_h, t03_h, (h03, w03)) = load_band(&f03, "hsd_b03");
    eprintln!("  0.5 km shape: {h03}×{w03}");
    assert_eq!((h03, w03), (h05, w05), "B03 shape mismatch");
    let d03_hybrid = correct_and_upsample(d03_h, t03_h, 640.0, None, min_cos);

    // Step 3: B04 (NIR, 860 nm, 1 km) → correct → up-sample
    eprintln!("--- B04: correct + up-sample to 0.5 km ---");
    let (d04, t04, _) = load_band(&f04, "hsd_b04");
    let d04_05 = correct_and_upsample(d04, t04, 860.0, None, min_cos);

    // Step 4: Hybrid green = 0.6321×G + 0.2928×R + 0.0751×N
    // G=B02(green), R=B03(red), N=B04(NIR) — Satpy AHI true_color recipe
    eprintln!("--- Hybrid green (3-band: B02+B03+B04) ---");
    let hybrid = SpectralBlender::new("hybrid_green", vec![0.6321, 0.2928, 0.0751])
        .expect("blender")
        .compose_owned(vec![d02_05, d03_hybrid, d04_05])
        .expect("hybrid");
    assert!(hybrid.array().is_some());

    // Step 5: B03 (red) — reload second copy for red channel
    eprintln!("--- B03: reload for red channel ---");
    let (d03_r, t03_r, _) = load_band(&f03, "hsd_b03");
    let d03_red = correct_and_upsample(d03_r, t03_r, 640.0, None, min_cos);

    // Step 6: B01 (blue) → correct → up-sample
    eprintln!("--- B01: correct + up-sample to 0.5 km ---");
    let (d01, t01, _) = load_band(&f01, "hsd_b01");
    let d01_05 = correct_and_upsample(d01, t01, 470.0, None, min_cos);

    // Step 7: RGB composite R=B03, G=hybrid, B=B01
    eprintln!("--- RGB composite ---");
    let rgb = RgbCompositor::new("true_color")
        .expect("rgb")
        .compose_rgb_owned(vec![d03_red, hybrid, d01_05])
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

    // Step 8: Gamma 2.2 → save 8-bit PNG
    // Memory: band-major f32[rgb] released when rgb drops after from_rgb_array
    eprintln!("--- Gamma 2.2 + save ---");
    use rusty_sat_image::FloatImage;
    let rgb_arr = rgb.array().expect("arr");
    let mut img = FloatImage::<f32>::from_rgb_array(rgb_arr).expect("float image");
    drop(rgb_arr);
    drop(rgb); // free band-major [3,y,x] f32 (~5.8 GB)
    img.crude_stretch_in_place(None, None);
    img.gamma_in_place(2.2).expect("gamma");
    let u8 = img.to_u8_image(0).expect("u8");
    drop(img); // free f32 interleaved (~5.8 GB)
    let png = out.join("true_color_05km.png");
    SimpleImageWriter::default()
        .save_image(&u8, &png)
        .expect("save");
    assert!(png.is_file());
    let sz = std::fs::metadata(&png).expect("meta").len();
    assert!(sz > 1024, "size={sz}");
    eprintln!("  output: {} ({sz} bytes)", png.display());

    // ── Sanity checks ──────────────────────────────────────────────────
    eprintln!("--- Sanity ---");

    // Combined correction reduces mean reflectance
    let (d02c, t02c, _) = load_band(&f02, "hsd_b02");
    let orig_v = d02c.array().expect("arr").values_as_f64();
    let orig_mean = orig_v.iter().filter(|v| v.is_finite()).sum::<f64>()
        / orig_v.iter().filter(|v| v.is_finite()).count().max(1) as f64;
    let d02comb =
        rayleigh_correct_with_sun_zenith(lut!(build_corrector(510.0)), d02c, None, t02c, min_cos)
            .expect("comb");
    let cv = d02comb.array().expect("arr").values_as_f64();
    let corr_mean = cv.iter().filter(|v| v.is_finite()).sum::<f64>()
        / cv.iter().filter(|v| v.is_finite()).count().max(1) as f64;
    assert_eq!(
        d02comb.attr("modifier").and_then(MetadataValue::as_str),
        Some("combined_sun_zenith_rayleigh_correction")
    );
    assert!(
        corr_mean < orig_mean,
        "Rayleigh should reduce ({corr_mean:.2} < {orig_mean:.2})"
    );
    eprintln!("  B02: {orig_mean:.2} → {corr_mean:.2}");

    eprintln!("\nPASS: True-color 0.5 km pipeline");
}
