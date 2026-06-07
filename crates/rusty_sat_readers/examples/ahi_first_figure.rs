//! Standalone test program that generates the first AHI figure from real HSD data.
//!
//! Loads Himawari-8/9 AHI HSD segment files (.DAT or .DAT.bz2), calibrates to
//! reflectance, assembles the full-disk image, and saves it as a PNG.
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example ahi_first_figure -- [DATA_DIR] [BAND]
//! cargo run --example ahi_first_figure -- local_data/ahi_input/data/20250923/07 B03
//! ```
//!
//! If DATA_DIR is omitted, defaults to `local_data/ahi_input/data/20250923/07`.
//! If BAND is omitted, defaults to `B03` (0.64 μm visible, 0.5 km resolution).
//!
//! Available bands in the default dataset: B01, B02, B03, B04, B13.

use rusty_sat_readers::{AhiCalibration, AhiHsdFileHandler, AhiHsdReader, AhiSegmentInfo, Reader};
use rusty_sat_writers::{SimpleImageWriter, Writer};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let data_dir = args
        .get(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("local_data/ahi_input/data/20250923/07"));
    let target_band = args.get(2).cloned().unwrap_or_else(|| "B03".to_string());

    println!("=== AHI First Figure Test ===");
    println!("Data directory: {}", data_dir.display());
    println!("Target band:   {target_band}");
    println!();

    let files_by_band = scan_hsd_files(&data_dir);
    if files_by_band.is_empty() {
        eprintln!("No .DAT.bz2 / .DAT files found in '{}'", data_dir.display());
        std::process::exit(1);
    }

    println!("Found {} band(s):", files_by_band.len());
    for (band, files) in &files_by_band {
        println!("  {band}: {} segment(s)", files.len());
    }
    println!();

    let band_files = files_by_band.get(&target_band).unwrap_or_else(|| {
        eprintln!(
            "Band '{target_band}' not found. Available: {:?}",
            files_by_band.keys().collect::<Vec<_>>()
        );
        std::process::exit(1);
    });

    println!(
        "--- Loading {target_band} ({}) segments ---",
        band_files.len()
    );
    let t0 = Instant::now();

    let handlers: Vec<AhiHsdFileHandler> = band_files
        .iter()
        .map(|(path, seg)| {
            let file_type = format!("hsd_{}", target_band.to_lowercase());
            print!(
                "  {} (S{:02}{:02}) ... ",
                path.file_name().unwrap().to_string_lossy(),
                seg.segment_number,
                seg.total_segments
            );
            let t = Instant::now();
            let handler =
                AhiHsdFileHandler::from_path(path, &file_type, *seg).unwrap_or_else(|err| {
                    panic!("failed to open '{}': {err}", path.display());
                });
            let h = handler.header();
            println!(
                "ok ({:.0}s) sat={} area={} cols={} lines={} bpp={} λ={:.3}μm compr={}",
                t.elapsed().as_secs_f64(),
                h.basic.satellite,
                h.basic.observation_area,
                h.data.columns,
                h.data.lines,
                h.data.bits_per_pixel,
                h.calibration.central_wavelength,
                if h.data.compression_flag == 2 {
                    "bzip2"
                } else {
                    "none"
                }
            );
            handler
        })
        .collect();

    let header_time = t0.elapsed();
    println!("Headers parsed in {:.1}s", header_time.as_secs_f64());
    println!();

    println!("--- Assembling and calibrating (reflectance) ---");
    let t1 = Instant::now();
    let reader = AhiHsdReader::with_calibration(AhiCalibration::Reflectance, handlers)
        .expect("create reader");

    let ids = reader.available_dataset_ids();
    let id = ids.first().expect("no dataset available");
    println!("Dataset ID: {id:?}");

    let dataset = reader.load(id).unwrap_or_else(|err| {
        panic!("failed to load assembled dataset: {err}");
    });

    let load_time = t1.elapsed();
    let shape = dataset
        .array()
        .map(|a: &rusty_sat_core::AnyDataArray| (a.shape().to_vec(), a.dtype().name().to_string()))
        .unwrap_or_default();
    println!(
        "Assembled + calibrated in {:.1}s | shape={:?} dtype={}",
        load_time.as_secs_f64(),
        shape.0,
        shape.1,
    );
    println!();

    let output_path = data_dir
        .parent() // 07
        .and_then(|p| p.parent()) // 20250923
        .and_then(|p| p.parent()) // data
        .and_then(|p| p.parent()) // ahi_input
        .map(|p| p.join("ahi_output"))
        .unwrap_or_else(|| PathBuf::from("local_data/ahi_output"));
    std::fs::create_dir_all(&output_path).ok();

    let png_path = output_path.join(format!("{target_band}_reflectance.png"));
    println!("--- Saving PNG to {} ---", png_path.display());
    let t2 = Instant::now();
    SimpleImageWriter::default()
        .save_dataset(&dataset, &png_path)
        .unwrap_or_else(|err| {
            panic!("failed to save PNG: {err}");
        });
    let save_time = t2.elapsed();
    let png_size = std::fs::metadata(&png_path).map(|m| m.len()).unwrap_or(0);

    println!(
        "Saved in {:.1}s | {:.1} MB",
        save_time.as_secs_f64(),
        png_size as f64 / 1_048_576.0
    );
    println!();

    let total = t0.elapsed();
    println!("=== Done in {:.1}s ===", total.as_secs_f64());
    println!("Output: {}", png_path.display());
}

/// Scan a directory for HSD segment files, grouped by band number.
fn scan_hsd_files(dir: &Path) -> BTreeMap<String, Vec<(PathBuf, AhiSegmentInfo)>> {
    let mut by_band: BTreeMap<String, Vec<(PathBuf, AhiSegmentInfo)>> = BTreeMap::new();

    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(_) => return by_band,
    };

    for entry in entries.flatten() {
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

/// Parse an HSD filename like `HS_H09_20250923_0720_B03_FLDK_R05_S0110.DAT.bz2`
/// returning the band label and segment info.
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
