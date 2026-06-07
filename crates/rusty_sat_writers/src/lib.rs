//! File output writers for satellite imagery.
//!
//! This crate serializes [`Dataset`] and [`Image`] objects to disk in standard
//! geospatial and image formats. All writers implement the [`Writer`] trait.
//!
//! # Writers
//!
//! - [`SimpleImageWriter`](simple_image::SimpleImageWriter) — PNG (8/16-bit
//!   grayscale) and JPEG (8-bit) via the `image` crate. Auto-stretches float
//!   datasets to the target bit depth.
//! - [`FloatTiffWriter`](float_tiff::FloatTiffWriter) — GeoTIFF with full
//!   GeoKey georeferencing. Supports float32, float64, and uint16-scaled
//!   output, with optional Deflate compression and tiling.
//! - [`PgmWriter`](pgm::PgmWriter) — Portable GrayMap (8/16-bit) with
//!   configurable linear scaling.
//!
//! # Factory
//!
//! [`BuiltinWriterFactory`] selects the correct writer by file extension
//! (`.png` → SimpleImageWriter, `.tif` → FloatTiffWriter, `.pgm` → PgmWriter).
//!
//! # Quick Start
//!
//! ```ignore
//! use rusty_sat_writers::{SimpleImageWriter, Writer};
//! SimpleImageWriter::default().save_dataset(&dataset, "output.png")?;
//! ```

use std::path::Path;

pub mod float_tiff;
pub mod pgm;
pub mod simple_image;

pub use float_tiff::{
    write_float64_tiff_dataset, write_float_tiff_dataset, write_u16_scaled_tiff_dataset,
    FloatTiffWriter, TiffSamplePolicy,
};
pub use pgm::{
    encode_pgm, encode_pgm_array, encode_pgm_from_f64, write_pgm, write_pgm_array, LinearScale,
    PgmWriter,
};
pub use simple_image::{
    write_image, write_jpeg_image, write_png16_image, write_png_image, SimpleImageDatasetBitDepth,
    SimpleImageWriter,
};

use rusty_sat_core::{Dataset, DatasetWriter, Result, RustySatError};
use rusty_sat_image::Image;

pub trait Writer {
    fn name(&self) -> &str;

    fn save_image(&self, _image: &Image, _path: &Path) -> Result<()> {
        Err(RustySatError::unsupported(format!(
            "{} writer",
            self.name()
        )))
    }

    fn save_dataset(&self, _dataset: &Dataset, _path: &Path) -> Result<()> {
        Err(RustySatError::unsupported(format!(
            "{} dataset writer",
            self.name()
        )))
    }
}

impl DatasetWriter for PgmWriter {
    fn save_dataset(&self, dataset: &Dataset, path: &Path) -> Result<()> {
        PgmWriter::save_dataset(self, dataset, path)
    }
}

impl DatasetWriter for FloatTiffWriter {
    fn save_dataset(&self, dataset: &Dataset, path: &Path) -> Result<()> {
        FloatTiffWriter::save_dataset(self, dataset, path)
    }
}

impl DatasetWriter for SimpleImageWriter {
    fn save_dataset(&self, dataset: &Dataset, path: &Path) -> Result<()> {
        Writer::save_dataset(self, dataset, path)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltinWriterKind {
    Pgm,
    SimpleImage,
    FloatTiff,
}

#[derive(Debug, Clone)]
pub enum BuiltinWriter {
    Pgm(PgmWriter),
    SimpleImage(SimpleImageWriter),
    FloatTiff(FloatTiffWriter),
}

impl BuiltinWriter {
    pub fn kind(&self) -> BuiltinWriterKind {
        match self {
            Self::Pgm(_) => BuiltinWriterKind::Pgm,
            Self::SimpleImage(_) => BuiltinWriterKind::SimpleImage,
            Self::FloatTiff(_) => BuiltinWriterKind::FloatTiff,
        }
    }

    pub fn name(&self) -> &str {
        match self {
            Self::Pgm(writer) => Writer::name(writer),
            Self::SimpleImage(writer) => Writer::name(writer),
            Self::FloatTiff(writer) => Writer::name(writer),
        }
    }

    pub fn with_16_bit_png_dataset_output(self) -> Self {
        match self {
            Self::SimpleImage(writer) => Self::SimpleImage(writer.with_16_bit_dataset_output()),
            other => other,
        }
    }

    pub fn with_float64_tiff_output(self, fill_value: f64) -> Self {
        match self {
            Self::FloatTiff(writer) => Self::FloatTiff(writer.with_float64_output(fill_value)),
            other => other,
        }
    }

    pub fn with_u16_scaled_tiff_output(self, min: f64, max: f64, fill_value: u16) -> Result<Self> {
        match self {
            Self::FloatTiff(writer) => Ok(Self::FloatTiff(
                writer.with_u16_scaled_output(min, max, fill_value)?,
            )),
            other => Ok(other),
        }
    }
}

impl DatasetWriter for BuiltinWriter {
    fn save_dataset(&self, dataset: &Dataset, path: &Path) -> Result<()> {
        match self {
            Self::Pgm(writer) => DatasetWriter::save_dataset(writer, dataset, path),
            Self::SimpleImage(writer) => DatasetWriter::save_dataset(writer, dataset, path),
            Self::FloatTiff(writer) => DatasetWriter::save_dataset(writer, dataset, path),
        }
    }
}

impl Writer for BuiltinWriter {
    fn name(&self) -> &str {
        BuiltinWriter::name(self)
    }

    fn save_image(&self, image: &Image, path: &Path) -> Result<()> {
        match self {
            Self::Pgm(_) => Err(RustySatError::unsupported(
                "PGM writer cannot save finalized RGB/RGBA image buffers",
            )),
            Self::SimpleImage(writer) => writer.save_image(image, path),
            Self::FloatTiff(writer) => writer.save_image(image, path),
        }
    }

    fn save_dataset(&self, dataset: &Dataset, path: &Path) -> Result<()> {
        DatasetWriter::save_dataset(self, dataset, path)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct BuiltinWriterFactory {
    png_bit_depth: SimpleImageDatasetBitDepth,
    tiff_sample_policy: TiffSamplePolicy,
}

impl BuiltinWriterFactory {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_png_bit_depth(mut self, bit_depth: SimpleImageDatasetBitDepth) -> Self {
        self.png_bit_depth = bit_depth;
        self
    }

    pub fn with_tiff_sample_policy(mut self, sample_policy: TiffSamplePolicy) -> Result<Self> {
        validate_tiff_sample_policy(&sample_policy)?;
        self.tiff_sample_policy = sample_policy;
        Ok(self)
    }

    pub fn writer_for_path(&self, path: impl AsRef<Path>) -> Result<BuiltinWriter> {
        let path = path.as_ref();
        let Some(extension) = path.extension().and_then(|extension| extension.to_str()) else {
            return Err(RustySatError::invalid_input(
                "writer selection requires a filename extension",
            ));
        };
        self.writer_for_extension(extension)
    }

    pub fn writer_for_extension(&self, extension: &str) -> Result<BuiltinWriter> {
        let extension = extension
            .trim()
            .trim_start_matches('.')
            .to_ascii_lowercase();
        match extension.as_str() {
            "pgm" => Ok(BuiltinWriter::Pgm(PgmWriter::default())),
            "png" | "jpg" | "jpeg" => Ok(BuiltinWriter::SimpleImage(
                SimpleImageWriter::default().with_dataset_bit_depth(self.png_bit_depth),
            )),
            "tif" | "tiff" => Ok(BuiltinWriter::FloatTiff(float_tiff_writer_from_policy(
                &self.tiff_sample_policy,
            )?)),
            "" => Err(RustySatError::invalid_input(
                "writer selection requires a non-empty extension",
            )),
            other => Err(RustySatError::unsupported(format!(
                "built-in writer for extension '{other}'"
            ))),
        }
    }
}

impl Default for BuiltinWriterFactory {
    fn default() -> Self {
        Self {
            png_bit_depth: SimpleImageDatasetBitDepth::Eight,
            tiff_sample_policy: TiffSamplePolicy::Float32 {
                fill_value: f32::NAN,
            },
        }
    }
}

pub fn writer_for_path(path: impl AsRef<Path>) -> Result<BuiltinWriter> {
    BuiltinWriterFactory::default().writer_for_path(path)
}

pub fn writer_for_extension(extension: &str) -> Result<BuiltinWriter> {
    BuiltinWriterFactory::default().writer_for_extension(extension)
}

fn float_tiff_writer_from_policy(sample_policy: &TiffSamplePolicy) -> Result<FloatTiffWriter> {
    validate_tiff_sample_policy(sample_policy)?;
    Ok(match sample_policy {
        TiffSamplePolicy::Float32 { fill_value } => {
            FloatTiffWriter::default().with_fill_value(*fill_value)
        }
        TiffSamplePolicy::Float64 { fill_value } => {
            FloatTiffWriter::default().with_float64_output(*fill_value)
        }
        TiffSamplePolicy::UInt16Scaled {
            min: Some(min),
            max: Some(max),
            fill_value,
        } => FloatTiffWriter::default().with_u16_scaled_output(*min, *max, *fill_value)?,
        TiffSamplePolicy::UInt16Scaled {
            min: None,
            max: None,
            fill_value,
        } => FloatTiffWriter::default().with_u16_auto_scaled_output(*fill_value),
        TiffSamplePolicy::UInt16Scaled { .. } => {
            return Err(RustySatError::invalid_input(
                "u16 scaled TIFF output requires both min and max or neither",
            ));
        }
    })
}

fn validate_tiff_sample_policy(sample_policy: &TiffSamplePolicy) -> Result<()> {
    match sample_policy {
        TiffSamplePolicy::Float32 { fill_value } => {
            finite_or_nan(f64::from(*fill_value), "float32 TIFF fill value")
        }
        TiffSamplePolicy::Float64 { fill_value } => {
            finite_or_nan(*fill_value, "float64 TIFF fill value")
        }
        TiffSamplePolicy::UInt16Scaled {
            min: Some(min),
            max: Some(max),
            ..
        } if min.is_finite() && max.is_finite() && max > min => Ok(()),
        TiffSamplePolicy::UInt16Scaled {
            min: None,
            max: None,
            ..
        } => Ok(()),
        TiffSamplePolicy::UInt16Scaled { .. } => Err(RustySatError::invalid_input(
            "u16 scaled TIFF output requires finite max greater than min, or autoscale",
        )),
    }
}

fn finite_or_nan(value: f64, name: &str) -> Result<()> {
    if value.is_finite() || value.is_nan() {
        Ok(())
    } else {
        Err(RustySatError::invalid_input(format!(
            "{name} must be finite or NaN"
        )))
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;
    use rusty_sat_core::{DataArray, DataId, Scene};
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    struct PlaceholderWriter;

    impl Writer for PlaceholderWriter {
        fn name(&self) -> &str {
            "placeholder"
        }
    }

    #[test]
    fn writer_trait_compiles() {
        let writer = PlaceholderWriter;
        assert_eq!(writer.name(), "placeholder");
    }

    #[test]
    fn factory_selects_builtin_writers_by_extension() -> Result<()> {
        assert_eq!(writer_for_extension("pgm")?.kind(), BuiltinWriterKind::Pgm);
        assert_eq!(
            writer_for_extension(".PNG")?.kind(),
            BuiltinWriterKind::SimpleImage
        );
        assert_eq!(
            writer_for_extension("jpeg")?.kind(),
            BuiltinWriterKind::SimpleImage
        );
        assert_eq!(
            writer_for_path("output.tiff")?.kind(),
            BuiltinWriterKind::FloatTiff
        );
        Ok(())
    }

    #[test]
    fn factory_reports_missing_or_unknown_extension() {
        assert!(writer_for_path("output")
            .unwrap_err()
            .to_string()
            .contains("extension"));
        assert!(writer_for_extension("gif")
            .unwrap_err()
            .to_string()
            .contains("unsupported feature"));
    }

    #[test]
    fn selected_writer_can_save_scene_dataset() -> Result<()> {
        let path = temp_path("selected_writer_can_save_scene_dataset", "pgm");
        let data_id = DataId::new("B03")?;
        let dataset = rusty_sat_core::Dataset::new(data_id.clone()).with_array(
            DataArray::<u16>::from_vec_named([1, 2], ["y", "x"], vec![0, 1024])?,
        );
        let mut scene = Scene::new();
        scene.insert_dataset(dataset);
        let writer = writer_for_path(&path)?;

        scene.save_dataset(&data_id, &writer, &path)?;

        let bytes = fs::read(&path)
            .map_err(|err| RustySatError::invalid_input(format!("failed to read PGM: {err}")))?;
        assert!(bytes.starts_with(b"P5\n2 1\n255\n"));
        fs::remove_file(path).ok();
        Ok(())
    }

    #[test]
    fn factory_options_select_16_bit_png_and_float64_tiff() -> Result<()> {
        let png_writer = BuiltinWriterFactory::default()
            .with_png_bit_depth(SimpleImageDatasetBitDepth::Sixteen)
            .writer_for_extension("png")?;
        let tiff_writer = BuiltinWriterFactory::default()
            .with_tiff_sample_policy(TiffSamplePolicy::Float64 {
                fill_value: f64::NAN,
            })?
            .writer_for_extension("tif")?;

        assert_eq!(png_writer.kind(), BuiltinWriterKind::SimpleImage);
        assert_eq!(tiff_writer.kind(), BuiltinWriterKind::FloatTiff);
        Ok(())
    }

    #[test]
    fn factory_rejects_invalid_tiff_sample_policy() {
        let err = BuiltinWriterFactory::default()
            .with_tiff_sample_policy(TiffSamplePolicy::UInt16Scaled {
                min: Some(1.0),
                max: Some(1.0),
                fill_value: 0,
            })
            .unwrap_err();
        assert!(err.to_string().contains("max greater than min"));
    }

    fn temp_path(name: &str, extension: &str) -> std::path::PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time")
            .as_nanos();
        std::env::temp_dir().join(format!("rusty_sat_{name}_{nanos}.{extension}"))
    }
}
