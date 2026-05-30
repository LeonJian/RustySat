//! Minimal scientific TIFF writer foundation.
//!
//! This is intentionally a baseline uncompressed TIFF writer, not full GeoTIFF
//! parity yet. It preserves calibrated values as 32-bit floating point samples
//! so AHI scientific output paths do not have to go through display scaling.

use crate::Writer;
use rusty_sat_core::{Dataset, Result, RustySatError};
use rusty_sat_image::Image;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

const TIFF_MAGIC: u16 = 42;
const IFD_OFFSET: u32 = 8;

const TAG_IMAGE_WIDTH: u16 = 256;
const TAG_IMAGE_LENGTH: u16 = 257;
const TAG_BITS_PER_SAMPLE: u16 = 258;
const TAG_COMPRESSION: u16 = 259;
const TAG_PHOTOMETRIC_INTERPRETATION: u16 = 262;
const TAG_STRIP_OFFSETS: u16 = 273;
const TAG_SAMPLES_PER_PIXEL: u16 = 277;
const TAG_ROWS_PER_STRIP: u16 = 278;
const TAG_STRIP_BYTE_COUNTS: u16 = 279;
const TAG_SAMPLE_FORMAT: u16 = 339;

const TYPE_SHORT: u16 = 3;
const TYPE_LONG: u16 = 4;

const COMPRESSION_NONE: u16 = 1;
const PHOTOMETRIC_BLACK_IS_ZERO: u16 = 1;
const SAMPLE_FORMAT_IEEE_FLOAT: u16 = 3;

#[derive(Debug, Clone, PartialEq)]
pub struct FloatTiffWriter {
    name: String,
    fill_value: f32,
}

impl FloatTiffWriter {
    pub fn new(name: impl Into<String>) -> Result<Self> {
        let name = name.into();
        if name.trim().is_empty() {
            return Err(RustySatError::invalid_input(
                "float TIFF writer name cannot be empty",
            ));
        }
        Ok(Self {
            name,
            fill_value: f32::NAN,
        })
    }

    pub fn with_fill_value(mut self, fill_value: f32) -> Self {
        self.fill_value = fill_value;
        self
    }

    pub fn save_dataset(&self, dataset: &Dataset, path: impl AsRef<Path>) -> Result<()> {
        write_float_tiff_dataset(dataset, path, self.fill_value)
    }
}

impl Default for FloatTiffWriter {
    fn default() -> Self {
        Self {
            name: "float_tiff".to_string(),
            fill_value: f32::NAN,
        }
    }
}

impl Writer for FloatTiffWriter {
    fn name(&self) -> &str {
        &self.name
    }

    fn save_image(&self, _image: &Image, _path: &Path) -> Result<()> {
        Err(RustySatError::unsupported(
            "float TIFF image writing from finalized display images",
        ))
    }

    fn save_dataset(&self, dataset: &Dataset, path: &Path) -> Result<()> {
        FloatTiffWriter::save_dataset(self, dataset, path)
    }
}

pub fn write_float_tiff_dataset(
    dataset: &Dataset,
    path: impl AsRef<Path>,
    fill_value: f32,
) -> Result<()> {
    let path = path.as_ref();
    validate_tiff_extension(path)?;
    let Some(array) = dataset.array() else {
        return Err(RustySatError::invalid_input(format!(
            "dataset '{}' has no array data",
            dataset.id().name()
        )));
    };
    if array.ndim() != 2 {
        return Err(RustySatError::invalid_input(format!(
            "float TIFF writer requires a 2D y/x array, got shape {:?}",
            array.shape()
        )));
    }
    let (height, width) = array.shape_yx()?;
    let values = array.values_as_f64();
    let mask = array.mask();
    let pixels = values
        .iter()
        .enumerate()
        .map(|(idx, value)| {
            if mask.is_some_and(|mask| mask.is_masked(idx) == Some(true)) || !value.is_finite() {
                fill_value
            } else {
                *value as f32
            }
        })
        .collect::<Vec<_>>();

    write_float_tiff_pixels(path, width, height, &pixels)
}

fn write_float_tiff_pixels(path: &Path, width: usize, height: usize, pixels: &[f32]) -> Result<()> {
    if width == 0 || height == 0 {
        return Err(RustySatError::invalid_input(
            "float TIFF dimensions must be non-zero",
        ));
    }
    let pixel_count = width
        .checked_mul(height)
        .ok_or_else(|| RustySatError::invalid_input("float TIFF pixel count overflow"))?;
    if pixel_count != pixels.len() {
        return Err(RustySatError::invalid_input(format!(
            "float TIFF has {} pixels but shape ({height}, {width}) requires {pixel_count}",
            pixels.len()
        )));
    }
    let width_u32 = u32::try_from(width)
        .map_err(|_| RustySatError::invalid_input("float TIFF width does not fit u32"))?;
    let height_u32 = u32::try_from(height)
        .map_err(|_| RustySatError::invalid_input("float TIFF height does not fit u32"))?;
    let byte_count = pixels
        .len()
        .checked_mul(4)
        .and_then(|count| u32::try_from(count).ok())
        .ok_or_else(|| RustySatError::invalid_input("float TIFF byte count overflow"))?;

    let entry_count = 10_u16;
    let ifd_bytes = 2_u32 + u32::from(entry_count) * 12 + 4;
    let pixel_offset = IFD_OFFSET
        .checked_add(ifd_bytes)
        .ok_or_else(|| RustySatError::invalid_input("float TIFF pixel offset overflow"))?;

    let file = File::create(path).map_err(|err| {
        RustySatError::invalid_input(format!("failed to create float TIFF file: {err}"))
    })?;
    let mut writer = BufWriter::new(file);
    writer.write_all(b"II").map_err(write_error)?;
    write_u16(&mut writer, TIFF_MAGIC)?;
    write_u32(&mut writer, IFD_OFFSET)?;
    write_u16(&mut writer, entry_count)?;
    write_ifd_long(&mut writer, TAG_IMAGE_WIDTH, width_u32)?;
    write_ifd_long(&mut writer, TAG_IMAGE_LENGTH, height_u32)?;
    write_ifd_short(&mut writer, TAG_BITS_PER_SAMPLE, 32)?;
    write_ifd_short(&mut writer, TAG_COMPRESSION, COMPRESSION_NONE)?;
    write_ifd_short(
        &mut writer,
        TAG_PHOTOMETRIC_INTERPRETATION,
        PHOTOMETRIC_BLACK_IS_ZERO,
    )?;
    write_ifd_long(&mut writer, TAG_STRIP_OFFSETS, pixel_offset)?;
    write_ifd_short(&mut writer, TAG_SAMPLES_PER_PIXEL, 1)?;
    write_ifd_long(&mut writer, TAG_ROWS_PER_STRIP, height_u32)?;
    write_ifd_long(&mut writer, TAG_STRIP_BYTE_COUNTS, byte_count)?;
    write_ifd_short(&mut writer, TAG_SAMPLE_FORMAT, SAMPLE_FORMAT_IEEE_FLOAT)?;
    write_u32(&mut writer, 0)?;
    for pixel in pixels {
        writer
            .write_all(&pixel.to_le_bytes())
            .map_err(write_error)?;
    }
    Ok(())
}

fn validate_tiff_extension(path: &Path) -> Result<()> {
    let Some(extension) = path.extension().and_then(|extension| extension.to_str()) else {
        return Err(RustySatError::invalid_input(
            "float TIFF filename must include an extension",
        ));
    };
    if extension.eq_ignore_ascii_case("tif") || extension.eq_ignore_ascii_case("tiff") {
        Ok(())
    } else {
        Err(RustySatError::unsupported(format!(
            "float TIFF format '{extension}'"
        )))
    }
}

fn write_ifd_short(writer: &mut impl Write, tag: u16, value: u16) -> Result<()> {
    write_u16(writer, tag)?;
    write_u16(writer, TYPE_SHORT)?;
    write_u32(writer, 1)?;
    write_u16(writer, value)?;
    write_u16(writer, 0)
}

fn write_ifd_long(writer: &mut impl Write, tag: u16, value: u32) -> Result<()> {
    write_u16(writer, tag)?;
    write_u16(writer, TYPE_LONG)?;
    write_u32(writer, 1)?;
    write_u32(writer, value)
}

fn write_u16(writer: &mut impl Write, value: u16) -> Result<()> {
    writer.write_all(&value.to_le_bytes()).map_err(write_error)
}

fn write_u32(writer: &mut impl Write, value: u32) -> Result<()> {
    writer.write_all(&value.to_le_bytes()).map_err(write_error)
}

fn write_error(err: std::io::Error) -> RustySatError {
    RustySatError::invalid_input(format!("failed to write float TIFF file: {err}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rusty_sat_core::{DataArray, DataId, Dataset, Scene, ValidityMask};
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn writes_float32_tiff_dataset_without_display_scaling() -> Result<()> {
        let path = temp_tiff_path("writes_float32_tiff_dataset_without_display_scaling");
        let dataset = Dataset::new(DataId::new("B03")?).with_array(
            DataArray::<f64>::from_vec_named([1, 3], ["y", "x"], vec![-1.25, 0.5, 1000.0])?,
        );

        FloatTiffWriter::default().save_dataset(&dataset, &path)?;

        let bytes = fs::read(&path)
            .map_err(|err| RustySatError::invalid_input(format!("failed to read TIFF: {err}")))?;
        assert_eq!(&bytes[..2], b"II");
        assert_eq!(read_u16(&bytes, 2), TIFF_MAGIC);
        assert_eq!(read_tag_u32(&bytes, TAG_IMAGE_WIDTH), 3);
        assert_eq!(read_tag_u32(&bytes, TAG_IMAGE_LENGTH), 1);
        assert_eq!(read_tag_u16(&bytes, TAG_BITS_PER_SAMPLE), 32);
        assert_eq!(
            read_tag_u16(&bytes, TAG_SAMPLE_FORMAT),
            SAMPLE_FORMAT_IEEE_FLOAT
        );
        let offset = read_tag_u32(&bytes, TAG_STRIP_OFFSETS) as usize;
        assert_eq!(read_f32(&bytes, offset), -1.25);
        assert_eq!(read_f32(&bytes, offset + 4), 0.5);
        assert_eq!(read_f32(&bytes, offset + 8), 1000.0);
        fs::remove_file(path).ok();
        Ok(())
    }

    #[test]
    fn writes_masked_pixels_as_configured_fill_value() -> Result<()> {
        let path = temp_tiff_path("writes_masked_pixels_as_configured_fill_value");
        let mask = ValidityMask::from_masked_flags([false, true, false]);
        let array = DataArray::<f32>::from_vec_named([1, 3], ["y", "x"], vec![1.0, 2.0, 3.0])?
            .with_mask(mask)?;
        let dataset = Dataset::new(DataId::new("B03")?).with_array(array);

        FloatTiffWriter::default()
            .with_fill_value(-9999.0)
            .save_dataset(&dataset, &path)?;

        let bytes = fs::read(&path)
            .map_err(|err| RustySatError::invalid_input(format!("failed to read TIFF: {err}")))?;
        let offset = read_tag_u32(&bytes, TAG_STRIP_OFFSETS) as usize;
        assert_eq!(read_f32(&bytes, offset), 1.0);
        assert_eq!(read_f32(&bytes, offset + 4), -9999.0);
        assert_eq!(read_f32(&bytes, offset + 8), 3.0);
        fs::remove_file(path).ok();
        Ok(())
    }

    #[test]
    fn scene_saves_dataset_as_float32_tiff() -> Result<()> {
        let path = temp_tiff_path("scene_saves_dataset_as_float32_tiff");
        let data_id = DataId::new("B03")?;
        let dataset = Dataset::new(data_id.clone()).with_array(DataArray::<f64>::from_vec_named(
            [1, 2],
            ["y", "x"],
            vec![10.0, 20.0],
        )?);
        let mut scene = Scene::new();
        scene.insert_dataset(dataset);

        scene.save_dataset(&data_id, &FloatTiffWriter::default(), &path)?;

        let bytes = fs::read(&path)
            .map_err(|err| RustySatError::invalid_input(format!("failed to read TIFF: {err}")))?;
        assert_eq!(read_tag_u32(&bytes, TAG_IMAGE_WIDTH), 2);
        assert_eq!(read_tag_u32(&bytes, TAG_IMAGE_LENGTH), 1);
        let offset = read_tag_u32(&bytes, TAG_STRIP_OFFSETS) as usize;
        assert_eq!(read_f32(&bytes, offset), 10.0);
        assert_eq!(read_f32(&bytes, offset + 4), 20.0);
        fs::remove_file(path).ok();
        Ok(())
    }

    #[test]
    fn rejects_non_tiff_extension() -> Result<()> {
        let path = temp_path("rejects_non_tiff_extension", "png");
        let dataset = Dataset::new(DataId::new("B03")?).with_array(
            DataArray::<f64>::from_vec_named([1, 1], ["y", "x"], vec![1.0])?,
        );

        let err = FloatTiffWriter::default()
            .save_dataset(&dataset, &path)
            .unwrap_err();

        assert!(err.to_string().contains("unsupported feature"));
        Ok(())
    }

    fn read_tag_u16(bytes: &[u8], tag: u16) -> u16 {
        let offset = find_ifd_entry(bytes, tag) + 8;
        read_u16(bytes, offset)
    }

    fn read_tag_u32(bytes: &[u8], tag: u16) -> u32 {
        let offset = find_ifd_entry(bytes, tag) + 8;
        read_u32(bytes, offset)
    }

    fn find_ifd_entry(bytes: &[u8], tag: u16) -> usize {
        let ifd_offset = read_u32(bytes, 4) as usize;
        let count = read_u16(bytes, ifd_offset) as usize;
        for index in 0..count {
            let offset = ifd_offset + 2 + index * 12;
            if read_u16(bytes, offset) == tag {
                return offset;
            }
        }
        panic!("missing TIFF tag {tag}");
    }

    fn read_u16(bytes: &[u8], offset: usize) -> u16 {
        u16::from_le_bytes([bytes[offset], bytes[offset + 1]])
    }

    fn read_u32(bytes: &[u8], offset: usize) -> u32 {
        u32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ])
    }

    fn read_f32(bytes: &[u8], offset: usize) -> f32 {
        f32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ])
    }

    fn temp_tiff_path(name: &str) -> std::path::PathBuf {
        temp_path(name, "tif")
    }

    fn temp_path(name: &str, extension: &str) -> std::path::PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time")
            .as_nanos();
        std::env::temp_dir().join(format!("rusty_sat_{name}_{nanos}.{extension}"))
    }
}
