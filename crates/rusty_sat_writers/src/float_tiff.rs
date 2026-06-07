//! Minimal scientific TIFF/GeoTIFF writer foundation.
//!
//! This is intentionally a baseline uncompressed TIFF writer, not full GeoTIFF
//! parity yet. It preserves calibrated values as floating point samples, and
//! provides an explicit scaled u16 path for HDR display-oriented products.

use crate::Writer;
use rusty_sat_core::{Dataset, MetadataValue, Result, RustySatError};
use rusty_sat_image::Image;
use rusty_sat_resample::geo_keys::{
    finalize_geo_key_defs, serialize_geo_key_directory, GeoKeyDef, GEO_USER_DEFINED,
    GT_MODEL_TYPE_GEO_KEY, GT_RASTER_TYPE_GEO_KEY, MODEL_TYPE_PROJECTED,
    PROJECTED_CITATION_GEO_KEY, PROJECTED_CS_TYPE_GEO_KEY, RASTER_PIXEL_IS_AREA,
    TIFFTAG_GEO_ASCII_PARAMS, TIFFTAG_GEO_DOUBLE_PARAMS,
};
use std::collections::BTreeMap;
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
const TAG_MODEL_PIXEL_SCALE: u16 = 33550;
const TAG_MODEL_TIEPOINT: u16 = 33922;
const TAG_SAMPLE_FORMAT: u16 = 339;
const TAG_GEO_KEY_DIRECTORY: u16 = 34735;

const TYPE_ASCII: u16 = 2;
const TYPE_SHORT: u16 = 3;
const TYPE_LONG: u16 = 4;
const TYPE_DOUBLE: u16 = 12;

const GEO_KEY_DIRECTORY_VERSION: u16 = 1;
const GEO_KEY_REVISION: u16 = 1;
const GEO_KEY_MINOR_REVISION: u16 = 0;

const COMPRESSION_NONE: u16 = 1;
const COMPRESSION_DEFLATE: u16 = 32946;

const TAG_TILE_WIDTH: u16 = 322;
const TAG_TILE_LENGTH: u16 = 323;
const TAG_TILE_OFFSETS: u16 = 324;
const TAG_TILE_BYTE_COUNTS: u16 = 325;
const PHOTOMETRIC_BLACK_IS_ZERO: u16 = 1;
const SAMPLE_FORMAT_UNSIGNED_INTEGER: u16 = 1;
const SAMPLE_FORMAT_IEEE_FLOAT: u16 = 3;

#[derive(Debug, Clone, PartialEq)]
pub struct FloatTiffWriter {
    name: String,
    sample_policy: TiffSamplePolicy,
    compression: TiffCompression,
    tile_options: Option<TiffTileOptions>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TiffSamplePolicy {
    Float32 {
        fill_value: f32,
    },
    Float64 {
        fill_value: f64,
    },
    UInt16Scaled {
        min: Option<f64>,
        max: Option<f64>,
        fill_value: u16,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TiffCompression {
    None,
    Deflate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TiffTileOptions {
    pub width: usize,
    pub height: usize,
}

impl Default for TiffTileOptions {
    fn default() -> Self {
        Self {
            width: 256,
            height: 256,
        }
    }
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
            sample_policy: TiffSamplePolicy::Float32 {
                fill_value: f32::NAN,
            },
            compression: TiffCompression::None,
            tile_options: None,
        })
    }

    pub fn with_fill_value(mut self, fill_value: f32) -> Self {
        self.sample_policy = TiffSamplePolicy::Float32 { fill_value };
        self
    }

    pub fn with_float64_output(mut self, fill_value: f64) -> Self {
        self.sample_policy = TiffSamplePolicy::Float64 { fill_value };
        self
    }

    pub fn with_u16_auto_scaled_output(mut self, fill_value: u16) -> Self {
        self.sample_policy = TiffSamplePolicy::UInt16Scaled {
            min: None,
            max: None,
            fill_value,
        };
        self
    }

    pub fn with_u16_scaled_output(mut self, min: f64, max: f64, fill_value: u16) -> Result<Self> {
        validate_u16_scale(min, max)?;
        self.sample_policy = TiffSamplePolicy::UInt16Scaled {
            min: Some(min),
            max: Some(max),
            fill_value,
        };
        self.sample_policy.validate().map(|()| self)
    }

    pub fn with_compression(mut self, compression: TiffCompression) -> Self {
        self.compression = compression;
        self
    }

    pub fn with_tiles(mut self, options: TiffTileOptions) -> Self {
        self.tile_options = Some(options);
        self
    }

    pub fn save_dataset(&self, dataset: &Dataset, path: impl AsRef<Path>) -> Result<()> {
        write_tiff_dataset_with_policy(
            dataset,
            path,
            &self.sample_policy,
            self.compression,
            self.tile_options,
        )
    }
}

impl Default for FloatTiffWriter {
    fn default() -> Self {
        Self {
            name: "float_tiff".to_string(),
            sample_policy: TiffSamplePolicy::Float32 {
                fill_value: f32::NAN,
            },
            compression: TiffCompression::None,
            tile_options: None,
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
    write_tiff_dataset_with_policy(
        dataset,
        path,
        &TiffSamplePolicy::Float32 { fill_value },
        TiffCompression::None,
        None,
    )
}

pub fn write_float64_tiff_dataset(
    dataset: &Dataset,
    path: impl AsRef<Path>,
    fill_value: f64,
) -> Result<()> {
    write_tiff_dataset_with_policy(
        dataset,
        path,
        &TiffSamplePolicy::Float64 { fill_value },
        TiffCompression::None,
        None,
    )
}

pub fn write_u16_scaled_tiff_dataset(
    dataset: &Dataset,
    path: impl AsRef<Path>,
    min: f64,
    max: f64,
    fill_value: u16,
) -> Result<()> {
    write_tiff_dataset_with_policy(
        dataset,
        path,
        &TiffSamplePolicy::UInt16Scaled {
            min: Some(min),
            max: Some(max),
            fill_value,
        },
        TiffCompression::None,
        None,
    )
}

fn write_tiff_dataset_with_policy(
    dataset: &Dataset,
    path: impl AsRef<Path>,
    sample_policy: &TiffSamplePolicy,
    compression: TiffCompression,
    tile_options: Option<TiffTileOptions>,
) -> Result<()> {
    sample_policy.validate()?;
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
    let sample_data = sample_policy.encode_values(&values, mask)?;
    let geo_key_defs = geo_key_defs_from_dataset(dataset);

    write_tiff_pixels(
        path,
        width,
        height,
        &sample_data,
        geo_info_from_dataset(dataset)?,
        geo_key_defs,
        compression,
        tile_options,
    )
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct TiffSampleData {
    bits_per_sample: u16,
    sample_format: u16,
    bytes: Vec<u8>,
}

impl TiffSamplePolicy {
    fn validate(&self) -> Result<()> {
        match self {
            Self::Float32 { fill_value } => finite_or_nan(*fill_value as f64, "float32 fill value"),
            Self::Float64 { fill_value } => finite_or_nan(*fill_value, "float64 fill value"),
            Self::UInt16Scaled { min, max, .. } => match (*min, *max) {
                (Some(min), Some(max)) => validate_u16_scale(min, max),
                (None, None) => Ok(()),
                _ => Err(RustySatError::invalid_input(
                    "u16 scaled TIFF output requires both min and max or neither",
                )),
            },
        }
    }

    fn encode_values(
        &self,
        values: &[f64],
        mask: Option<&rusty_sat_core::ValidityMask>,
    ) -> Result<TiffSampleData> {
        match self {
            Self::Float32 { fill_value } => Ok(TiffSampleData {
                bits_per_sample: 32,
                sample_format: SAMPLE_FORMAT_IEEE_FLOAT,
                bytes: encode_f32_samples(values, mask, *fill_value),
            }),
            Self::Float64 { fill_value } => Ok(TiffSampleData {
                bits_per_sample: 64,
                sample_format: SAMPLE_FORMAT_IEEE_FLOAT,
                bytes: encode_f64_samples(values, mask, *fill_value),
            }),
            Self::UInt16Scaled {
                min,
                max,
                fill_value,
            } => {
                let (scale_min, scale_max) = match (*min, *max) {
                    (Some(min), Some(max)) => (min, max),
                    (None, None) => finite_min_max(values, mask).ok_or_else(|| {
                        RustySatError::invalid_input(
                            "u16 scaled TIFF output has no finite unmasked values for autoscale",
                        )
                    })?,
                    _ => unreachable!("sample policy validation rejects partial u16 scale"),
                };
                validate_u16_scale(scale_min, scale_max)?;
                Ok(TiffSampleData {
                    bits_per_sample: 16,
                    sample_format: SAMPLE_FORMAT_UNSIGNED_INTEGER,
                    bytes: encode_u16_scaled_samples(
                        values,
                        mask,
                        scale_min,
                        scale_max,
                        *fill_value,
                    ),
                })
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn write_tiff_pixels(
    path: &Path,
    width: usize,
    height: usize,
    sample_data: &TiffSampleData,
    geo_info: Option<GeoTiffInfo>,
    geo_key_defs: Option<Vec<GeoKeyDef>>,
    compression: TiffCompression,
    tile_options: Option<TiffTileOptions>,
) -> Result<()> {
    if width == 0 || height == 0 {
        return Err(RustySatError::invalid_input(
            "float TIFF dimensions must be non-zero",
        ));
    }
    let pixel_count = width
        .checked_mul(height)
        .ok_or_else(|| RustySatError::invalid_input("float TIFF pixel count overflow"))?;
    let bytes_per_sample = usize::from(sample_data.bits_per_sample / 8);
    if !sample_data.bits_per_sample.is_multiple_of(8) || bytes_per_sample == 0 {
        return Err(RustySatError::invalid_input(
            "TIFF bits per sample must be a positive multiple of 8",
        ));
    }
    let expected_byte_count = pixel_count
        .checked_mul(bytes_per_sample)
        .ok_or_else(|| RustySatError::invalid_input("float TIFF byte count overflow"))?;
    if expected_byte_count != sample_data.bytes.len() {
        return Err(RustySatError::invalid_input(format!(
            "float TIFF has {} sample bytes but shape ({height}, {width}) requires {expected_byte_count}",
            sample_data.bytes.len()
        )));
    }
    let width_u32 = u32::try_from(width)
        .map_err(|_| RustySatError::invalid_input("float TIFF width does not fit u32"))?;
    let height_u32 = u32::try_from(height)
        .map_err(|_| RustySatError::invalid_input("float TIFF height does not fit u32"))?;
    let _byte_count = u32::try_from(sample_data.bytes.len())
        .map_err(|_| RustySatError::invalid_input("float TIFF byte count overflow"))?;

    let geotiff_data = geo_info
        .as_ref()
        .map(|info| GeoTiffExtraData::from_info(info, geo_key_defs.as_deref()))
        .transpose()?;
    let is_tiled = tile_options.is_some();
    let base_entry_count: u16 = if is_tiled { 11 } else { 10 };
    let entry_count = base_entry_count
        + geotiff_data
            .as_ref()
            .map(GeoTiffExtraData::geotiff_tag_count)
            .unwrap_or(0);
    let ifd_bytes = 2_u32 + u32::from(entry_count) * 12 + 4;
    let extra_bytes = geotiff_data
        .as_ref()
        .map(GeoTiffExtraData::byte_len)
        .unwrap_or(0);
    let pixel_offset = IFD_OFFSET
        .checked_add(ifd_bytes)
        .and_then(|offset| offset.checked_add(extra_bytes))
        .ok_or_else(|| RustySatError::invalid_input("float TIFF pixel offset overflow"))?;
    let extra_offset = IFD_OFFSET
        .checked_add(ifd_bytes)
        .ok_or_else(|| RustySatError::invalid_input("float TIFF GeoTIFF offset overflow"))?;

    let compression_code = match compression {
        TiffCompression::None => COMPRESSION_NONE,
        TiffCompression::Deflate => COMPRESSION_DEFLATE,
    };

    // Pre-compute pixel blocks (strips or tiles)
    let bytes_per_pixel = usize::from(sample_data.bits_per_sample / 8);
    let blocks: Vec<Vec<u8>> = if let Some(tiles) = tile_options {
        build_tile_blocks(
            &sample_data.bytes,
            width,
            height,
            tiles.width,
            tiles.height,
            bytes_per_pixel,
        )
    } else {
        vec![sample_data.bytes.clone()]
    };

    // Compress blocks if requested
    let compressed_blocks: Vec<Vec<u8>> = if compression == TiffCompression::Deflate {
        blocks
            .iter()
            .map(|b| compress_deflate(b))
            .collect::<Result<Vec<_>>>()?
    } else {
        blocks
    };

    // Compute block offsets
    let mut block_offsets = Vec::with_capacity(compressed_blocks.len());
    let mut current_offset = pixel_offset;
    for block in &compressed_blocks {
        block_offsets.push(current_offset);
        current_offset = current_offset
            .checked_add(
                u32::try_from(block.len())
                    .map_err(|_| RustySatError::invalid_input("TIFF block byte count overflow"))?,
            )
            .ok_or_else(|| RustySatError::invalid_input("float TIFF block offset overflow"))?;
    }

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
    write_ifd_short(
        &mut writer,
        TAG_BITS_PER_SAMPLE,
        sample_data.bits_per_sample,
    )?;
    write_ifd_short(&mut writer, TAG_COMPRESSION, compression_code)?;
    write_ifd_short(
        &mut writer,
        TAG_PHOTOMETRIC_INTERPRETATION,
        PHOTOMETRIC_BLACK_IS_ZERO,
    )?;
    if is_tiled {
        let tiles = tile_options.expect("tile_options present when is_tiled");
        write_ifd_long(&mut writer, TAG_TILE_WIDTH, tiles.width as u32)?;
        write_ifd_long(&mut writer, TAG_TILE_LENGTH, tiles.height as u32)?;
        // Write tile offsets as a LONG array
        let offsets_bytes: Vec<u8> = block_offsets.iter().flat_map(|o| o.to_le_bytes()).collect();
        let offsets_data_offset = current_offset;
        current_offset = current_offset
            .checked_add(
                u32::try_from(offsets_bytes.len())
                    .map_err(|_| RustySatError::invalid_input("TIFF tile offsets overflow"))?,
            )
            .ok_or_else(|| RustySatError::invalid_input("float TIFF tile offsets overflow"))?;
        write_ifd_offset(
            &mut writer,
            TAG_TILE_OFFSETS,
            TYPE_LONG,
            u32::try_from(block_offsets.len())
                .map_err(|_| RustySatError::invalid_input("TIFF tile count overflow"))?,
            offsets_data_offset,
        )?;
        // Write tile byte counts
        let counts_bytes: Vec<u8> = compressed_blocks
            .iter()
            .flat_map(|b| u32::try_from(b.len()).unwrap_or(0).to_le_bytes())
            .collect();
        let counts_data_offset = current_offset;
        current_offset = current_offset
            .checked_add(
                u32::try_from(counts_bytes.len())
                    .map_err(|_| RustySatError::invalid_input("TIFF tile byte counts overflow"))?,
            )
            .ok_or_else(|| RustySatError::invalid_input("float TIFF tile byte counts overflow"))?;
        let _ = current_offset; // offsets/counts are written last, after tile data
        write_ifd_offset(
            &mut writer,
            TAG_TILE_BYTE_COUNTS,
            TYPE_LONG,
            u32::try_from(compressed_blocks.len())
                .map_err(|_| RustySatError::invalid_input("TIFF tile count overflow"))?,
            counts_data_offset,
        )?;
        write_ifd_short(&mut writer, TAG_SAMPLES_PER_PIXEL, 1)?;
        write_ifd_short(&mut writer, TAG_SAMPLE_FORMAT, sample_data.sample_format)?;
        if let Some(geotiff_data) = geotiff_data.as_ref() {
            geotiff_data.write_ifd_entries(&mut writer, extra_offset)?;
        }
        write_u32(&mut writer, 0)?;
        if let Some(geotiff_data) = geotiff_data.as_ref() {
            geotiff_data.write_data(&mut writer)?;
        }
        // Write tile data
        for block in &compressed_blocks {
            writer.write_all(block).map_err(write_error)?;
        }
        // Write offsets and counts arrays
        writer.write_all(&offsets_bytes).map_err(write_error)?;
        writer.write_all(&counts_bytes).map_err(write_error)?;
    } else {
        write_ifd_long(&mut writer, TAG_STRIP_OFFSETS, pixel_offset)?;
        write_ifd_short(&mut writer, TAG_SAMPLES_PER_PIXEL, 1)?;
        write_ifd_long(&mut writer, TAG_ROWS_PER_STRIP, height_u32)?;
        let strip_byte_count = u32::try_from(compressed_blocks[0].len())
            .map_err(|_| RustySatError::invalid_input("compressed TIFF byte count overflow"))?;
        write_ifd_long(&mut writer, TAG_STRIP_BYTE_COUNTS, strip_byte_count)?;
        write_ifd_short(&mut writer, TAG_SAMPLE_FORMAT, sample_data.sample_format)?;
        if let Some(geotiff_data) = geotiff_data.as_ref() {
            geotiff_data.write_ifd_entries(&mut writer, extra_offset)?;
        }
        write_u32(&mut writer, 0)?;
        if let Some(geotiff_data) = geotiff_data.as_ref() {
            geotiff_data.write_data(&mut writer)?;
        }
        writer
            .write_all(&compressed_blocks[0])
            .map_err(write_error)?;
    }
    Ok(())
}

fn encode_f32_samples(
    values: &[f64],
    mask: Option<&rusty_sat_core::ValidityMask>,
    fill_value: f32,
) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for (idx, value) in values.iter().enumerate() {
        let sample = if is_missing(mask, idx, *value) {
            fill_value
        } else {
            *value as f32
        };
        bytes.extend_from_slice(&sample.to_le_bytes());
    }
    bytes
}

fn encode_f64_samples(
    values: &[f64],
    mask: Option<&rusty_sat_core::ValidityMask>,
    fill_value: f64,
) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 8);
    for (idx, value) in values.iter().enumerate() {
        let sample = if is_missing(mask, idx, *value) {
            fill_value
        } else {
            *value
        };
        bytes.extend_from_slice(&sample.to_le_bytes());
    }
    bytes
}

fn encode_u16_scaled_samples(
    values: &[f64],
    mask: Option<&rusty_sat_core::ValidityMask>,
    min: f64,
    max: f64,
    fill_value: u16,
) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 2);
    let scale = f64::from(u16::MAX) / (max - min);
    for (idx, value) in values.iter().enumerate() {
        let sample = if is_missing(mask, idx, *value) {
            fill_value
        } else {
            ((*value).clamp(min, max) - min).mul_add(scale, 0.0).round() as u16
        };
        bytes.extend_from_slice(&sample.to_le_bytes());
    }
    bytes
}

fn finite_min_max(
    values: &[f64],
    mask: Option<&rusty_sat_core::ValidityMask>,
) -> Option<(f64, f64)> {
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for (idx, value) in values.iter().enumerate() {
        if is_missing(mask, idx, *value) {
            continue;
        }
        min = min.min(*value);
        max = max.max(*value);
    }
    (min.is_finite() && max.is_finite()).then_some((min, max))
}

fn is_missing(mask: Option<&rusty_sat_core::ValidityMask>, idx: usize, value: f64) -> bool {
    mask.is_some_and(|mask| mask.is_masked(idx) == Some(true)) || !value.is_finite()
}

fn validate_u16_scale(min: f64, max: f64) -> Result<()> {
    if !min.is_finite() || !max.is_finite() || max <= min {
        return Err(RustySatError::invalid_input(
            "u16 scaled TIFF output requires finite max greater than min",
        ));
    }
    Ok(())
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

#[derive(Debug, Clone, PartialEq)]
struct GeoTiffInfo {
    pixel_scale: [f64; 3],
    tiepoint: [f64; 6],
    citation: String,
}

impl GeoTiffInfo {
    fn from_area_attr(area: &MetadataValue) -> Result<Option<Self>> {
        let MetadataValue::Map(map) = area else {
            return Ok(None);
        };
        let Some(extent) = map.get("area_extent").and_then(metadata_f64_list4) else {
            return Ok(None);
        };
        let Some(height) = map.get("height").and_then(metadata_usize) else {
            return Ok(None);
        };
        let Some(width) = map.get("width").and_then(metadata_usize) else {
            return Ok(None);
        };
        if height == 0 || width == 0 {
            return Ok(None);
        }
        let pixel_size_x = (extent[2] - extent[0]) / width as f64;
        let pixel_size_y = (extent[3] - extent[1]) / height as f64;
        if !pixel_size_x.is_finite() || !pixel_size_y.is_finite() {
            return Ok(None);
        }
        Ok(Some(Self {
            pixel_scale: [pixel_size_x.abs(), pixel_size_y.abs(), 0.0],
            tiepoint: [0.0, 0.0, 0.0, extent[0], extent[3], 0.0],
            citation: geotiff_citation(map),
        }))
    }

    fn from_xy_coords(array: &rusty_sat_core::AnyDataArray) -> Result<Option<Self>> {
        let Some(x_coord) = array.coord("x") else {
            return Ok(None);
        };
        let Some(y_coord) = array.coord("y") else {
            return Ok(None);
        };
        let x = x_coord.values();
        let y = y_coord.values();
        if x.is_empty() || y.is_empty() {
            return Ok(None);
        }
        let pixel_size_x = coord_spacing(x).unwrap_or(1.0).abs();
        let pixel_size_y = coord_spacing(y).unwrap_or(1.0).abs();
        if !pixel_size_x.is_finite() || !pixel_size_y.is_finite() {
            return Ok(None);
        }
        Ok(Some(Self {
            pixel_scale: [pixel_size_x, pixel_size_y, 0.0],
            tiepoint: [
                0.0,
                0.0,
                0.0,
                x[0] - 0.5 * pixel_size_x,
                y[0] + 0.5 * pixel_size_y,
                0.0,
            ],
            citation: "RustySat x/y coordinate axes".to_string(),
        }))
    }
}

#[derive(Debug, Clone, PartialEq)]
struct GeoTiffExtraData {
    pixel_scale: Vec<u8>,
    tiepoint: Vec<u8>,
    geo_key_directory: Vec<u8>,
    geo_ascii_params: Vec<u8>,
    geo_double_params: Vec<u8>,
}

impl GeoTiffExtraData {
    /// Build GeoTIFF extra data from geometry info and optional ProjCrs GeoKey defs.
    ///
    /// When `geo_key_defs` is `Some` and non-empty, the GeoKey directory is
    /// built from the CRS-aware definitions (including `GTRasterTypeGeoKey`).
    /// When `None` or empty, falls back to the legacy 4-key hardcoded directory.
    fn from_info(info: &GeoTiffInfo, geo_key_defs: Option<&[GeoKeyDef]>) -> Result<Self> {
        let (geo_key_directory, geo_ascii_params, geo_double_params) =
            if let Some(defs) = geo_key_defs.filter(|d| !d.is_empty()) {
                let mut defs = defs.to_vec();
                defs.push(GeoKeyDef::short(
                    GT_RASTER_TYPE_GEO_KEY,
                    RASTER_PIXEL_IS_AREA,
                ));
                let finalized = finalize_geo_key_defs(&defs);
                let dir = serialize_geo_key_directory(&finalized);
                (dir, finalized.ascii_params, finalized.double_params)
            } else {
                // Legacy fallback: 4 hardcoded keys
                let mut ascii = info.citation.clone();
                if !ascii.ends_with('|') {
                    ascii.push('|');
                }
                let ascii_bytes = ascii.into_bytes();
                let ascii_len = u16::try_from(ascii_bytes.len())
                    .map_err(|_| RustySatError::invalid_input("GeoTIFF citation is too long"))?;
                let keys = [
                    GT_MODEL_TYPE_GEO_KEY,
                    0,
                    1,
                    MODEL_TYPE_PROJECTED,
                    GT_RASTER_TYPE_GEO_KEY,
                    0,
                    1,
                    RASTER_PIXEL_IS_AREA,
                    PROJECTED_CS_TYPE_GEO_KEY,
                    0,
                    1,
                    GEO_USER_DEFINED,
                    PROJECTED_CITATION_GEO_KEY,
                    TIFFTAG_GEO_ASCII_PARAMS,
                    ascii_len,
                    0,
                ];
                let mut dir = Vec::with_capacity((4 + keys.len()) * 2);
                for value in [
                    GEO_KEY_DIRECTORY_VERSION,
                    GEO_KEY_REVISION,
                    GEO_KEY_MINOR_REVISION,
                    4u16,
                ] {
                    dir.extend_from_slice(&value.to_le_bytes());
                }
                for value in keys {
                    dir.extend_from_slice(&value.to_le_bytes());
                }
                (dir, ascii_bytes, Vec::new())
            };
        Ok(Self {
            pixel_scale: f64_values_to_bytes(&info.pixel_scale),
            tiepoint: f64_values_to_bytes(&info.tiepoint),
            geo_key_directory,
            geo_ascii_params,
            geo_double_params,
        })
    }

    fn byte_len(&self) -> u32 {
        (self.pixel_scale.len()
            + self.tiepoint.len()
            + self.geo_key_directory.len()
            + self.geo_ascii_params.len()
            + self.geo_double_params.len()) as u32
    }

    fn geotiff_tag_count(&self) -> u16 {
        let mut count = 4u16; // pixel_scale, tiepoint, key_directory, ascii_params
        if !self.geo_double_params.is_empty() {
            count += 1; // + TIFFTAG_GEO_DOUBLE_PARAMS
        }
        count
    }

    fn write_ifd_entries(&self, writer: &mut impl Write, offset: u32) -> Result<()> {
        let pixel_scale_offset = offset;
        let tiepoint_offset = pixel_scale_offset + self.pixel_scale.len() as u32;
        let geo_key_offset = tiepoint_offset + self.tiepoint.len() as u32;
        let geo_ascii_offset = geo_key_offset + self.geo_key_directory.len() as u32;
        let geo_double_offset = geo_ascii_offset + self.geo_ascii_params.len() as u32;
        write_ifd_offset(
            writer,
            TAG_MODEL_PIXEL_SCALE,
            TYPE_DOUBLE,
            3,
            pixel_scale_offset,
        )?;
        write_ifd_offset(writer, TAG_MODEL_TIEPOINT, TYPE_DOUBLE, 6, tiepoint_offset)?;
        write_ifd_offset(
            writer,
            TAG_GEO_KEY_DIRECTORY,
            TYPE_SHORT,
            u32::try_from(self.geo_key_directory.len() / 2).map_err(|_| {
                RustySatError::invalid_input("GeoTIFF key directory count overflow")
            })?,
            geo_key_offset,
        )?;
        write_ifd_offset(
            writer,
            TIFFTAG_GEO_ASCII_PARAMS,
            TYPE_ASCII,
            u32::try_from(self.geo_ascii_params.len())
                .map_err(|_| RustySatError::invalid_input("GeoTIFF ASCII count overflow"))?,
            geo_ascii_offset,
        )?;
        if !self.geo_double_params.is_empty() {
            write_ifd_offset(
                writer,
                TIFFTAG_GEO_DOUBLE_PARAMS,
                TYPE_DOUBLE,
                u32::try_from(self.geo_double_params.len() / 8)
                    .map_err(|_| RustySatError::invalid_input("GeoTIFF double count overflow"))?,
                geo_double_offset,
            )?;
        }
        Ok(())
    }

    fn write_data(&self, writer: &mut impl Write) -> Result<()> {
        writer.write_all(&self.pixel_scale).map_err(write_error)?;
        writer.write_all(&self.tiepoint).map_err(write_error)?;
        writer
            .write_all(&self.geo_key_directory)
            .map_err(write_error)?;
        writer
            .write_all(&self.geo_ascii_params)
            .map_err(write_error)?;
        if !self.geo_double_params.is_empty() {
            writer
                .write_all(&self.geo_double_params)
                .map_err(write_error)?;
        }
        Ok(())
    }
}

fn geo_info_from_dataset(dataset: &Dataset) -> Result<Option<GeoTiffInfo>> {
    if let Some(info) = dataset
        .attr("area")
        .map(GeoTiffInfo::from_area_attr)
        .transpose()
        .map(Option::flatten)?
    {
        return Ok(Some(info));
    }
    let Some(array) = dataset.array() else {
        return Ok(None);
    };
    GeoTiffInfo::from_xy_coords(array)
}

/// Build ProjCrs-based GeoKey definitions from dataset area attrs.
///
/// Reads the `"area"` attribute, extracts the `"projection"` sub-map,
/// and converts it to GeoTIFF GeoKey entries via `ProjCrs`.
/// Returns `None` when no area/projection metadata is available.
fn geo_key_defs_from_dataset(dataset: &Dataset) -> Option<Vec<GeoKeyDef>> {
    use rusty_sat_resample::ProjCrs;
    let area = dataset.attr("area")?;
    let area_map = match area {
        MetadataValue::Map(m) => m,
        _ => return None,
    };
    let projection = area_map.get("projection")?;
    let proj_map = match projection {
        MetadataValue::Map(m) => m,
        _ => return None,
    };
    let params: BTreeMap<String, String> = proj_map
        .iter()
        .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string())))
        .collect();
    let crs = ProjCrs::from_projection_map(&params).ok()?;
    let defs = crs.to_geotiff_geo_key_defs();
    if defs.is_empty() {
        None
    } else {
        Some(defs)
    }
}

fn geotiff_citation(area: &std::collections::BTreeMap<String, MetadataValue>) -> String {
    let proj_id = area
        .get("proj_id")
        .and_then(MetadataValue::as_str)
        .unwrap_or("unknown");
    let projection = area
        .get("projection")
        .and_then(|value| value.get_path(&["proj"]))
        .and_then(MetadataValue::as_str)
        .unwrap_or("unknown");
    format!("RustySat area {proj_id} proj={projection}")
}

fn metadata_usize(value: &MetadataValue) -> Option<usize> {
    let MetadataValue::Integer(value) = value else {
        return None;
    };
    usize::try_from(*value).ok()
}

fn metadata_f64_list4(value: &MetadataValue) -> Option<[f64; 4]> {
    let MetadataValue::List(values) = value else {
        return None;
    };
    if values.len() != 4 {
        return None;
    }
    Some([
        metadata_f64(&values[0])?,
        metadata_f64(&values[1])?,
        metadata_f64(&values[2])?,
        metadata_f64(&values[3])?,
    ])
}

fn metadata_f64(value: &MetadataValue) -> Option<f64> {
    match value {
        MetadataValue::Float(value) => Some(value.get()),
        MetadataValue::Integer(value) => Some(*value as f64),
        _ => None,
    }
}

fn coord_spacing(values: &[f64]) -> Option<f64> {
    if values.len() < 2 {
        return Some(1.0);
    }
    let spacing = values[1] - values[0];
    spacing.is_finite().then_some(spacing)
}

fn build_tile_blocks(
    pixel_bytes: &[u8],
    img_width: usize,
    img_height: usize,
    tile_width: usize,
    tile_height: usize,
    bytes_per_pixel: usize,
) -> Vec<Vec<u8>> {
    if tile_width == 0 || tile_height == 0 {
        return vec![pixel_bytes.to_vec()];
    }
    let tiles_x = img_width.div_ceil(tile_width);
    let tiles_y = img_height.div_ceil(tile_height);
    let tile_byte_size = tile_width * tile_height * bytes_per_pixel;
    let mut blocks = Vec::with_capacity(tiles_x * tiles_y);
    for ty in 0..tiles_y {
        for tx in 0..tiles_x {
            let mut tile = vec![0u8; tile_byte_size];
            let x_start = tx * tile_width;
            let y_start = ty * tile_height;
            for (local_y, img_y) in (y_start..(y_start + tile_height).min(img_height)).enumerate() {
                let src_row_begin = img_y * img_width * bytes_per_pixel;
                let _src_row_end =
                    src_row_begin + (img_width * bytes_per_pixel).min(pixel_bytes.len());
                let src_col_start = x_start * bytes_per_pixel;
                let col_bytes = ((x_start + tile_width).min(img_width) - x_start) * bytes_per_pixel;
                let dst_row_begin = local_y * tile_width * bytes_per_pixel;
                if src_row_begin < pixel_bytes.len() && src_col_start < pixel_bytes.len() {
                    let src_begin = (src_row_begin + src_col_start).min(pixel_bytes.len());
                    let src_end = (src_begin + col_bytes).min(pixel_bytes.len());
                    let bytes_to_copy = src_end.saturating_sub(src_begin);
                    let dst_begin = dst_row_begin.min(tile.len());
                    let dst_end = (dst_begin + bytes_to_copy).min(tile.len());
                    if bytes_to_copy > 0 && dst_begin < tile.len() {
                        tile[dst_begin..dst_end].copy_from_slice(&pixel_bytes[src_begin..src_end]);
                    }
                }
            }
            blocks.push(tile);
        }
    }
    blocks
}

fn compress_deflate(data: &[u8]) -> Result<Vec<u8>> {
    use flate2::write::DeflateEncoder;
    use flate2::Compression;
    use std::io::Write;
    let mut encoder = DeflateEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(data).map_err(|err| {
        RustySatError::invalid_input(format!("DEFLATE compression failed: {err}"))
    })?;
    encoder
        .finish()
        .map_err(|err| RustySatError::invalid_input(format!("DEFLATE compression failed: {err}")))
}

fn f64_values_to_bytes(values: &[f64]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 8);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
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

fn write_ifd_offset(
    writer: &mut impl Write,
    tag: u16,
    field_type: u16,
    count: u32,
    offset: u32,
) -> Result<()> {
    write_u16(writer, tag)?;
    write_u16(writer, field_type)?;
    write_u32(writer, count)?;
    write_u32(writer, offset)
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
    #![allow(clippy::unwrap_used)]
    use super::*;
    use rusty_sat_core::{DataArray, DataId, Dataset, MetadataValue, Scene, ValidityMask};
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
    fn writes_float64_tiff_dataset_without_precision_truncation() -> Result<()> {
        let path = temp_tiff_path("writes_float64_tiff_dataset_without_precision_truncation");
        let dataset =
            Dataset::new(DataId::new("B03")?).with_array(DataArray::<f64>::from_vec_named(
                [1, 2],
                ["y", "x"],
                vec![1.000_000_119_209_289_6, 42.25],
            )?);

        FloatTiffWriter::default()
            .with_float64_output(f64::NAN)
            .save_dataset(&dataset, &path)?;

        let bytes = fs::read(&path)
            .map_err(|err| RustySatError::invalid_input(format!("failed to read TIFF: {err}")))?;
        assert_eq!(read_tag_u16(&bytes, TAG_BITS_PER_SAMPLE), 64);
        assert_eq!(
            read_tag_u16(&bytes, TAG_SAMPLE_FORMAT),
            SAMPLE_FORMAT_IEEE_FLOAT
        );
        let offset = read_tag_u32(&bytes, TAG_STRIP_OFFSETS) as usize;
        assert_eq!(read_f64(&bytes, offset), 1.000_000_119_209_289_6);
        assert_eq!(read_f64(&bytes, offset + 8), 42.25);
        fs::remove_file(path).ok();
        Ok(())
    }

    #[test]
    fn writes_u16_scaled_tiff_dataset_with_fill_and_clamp_policy() -> Result<()> {
        let path = temp_tiff_path("writes_u16_scaled_tiff_dataset_with_fill_and_clamp_policy");
        let mask = ValidityMask::from_masked_flags([false, false, false, true, false]);
        let array = DataArray::<f64>::from_vec_named(
            [1, 5],
            ["y", "x"],
            vec![-10.0, 0.0, 50.0, 75.0, 120.0],
        )?
        .with_mask(mask)?;
        let dataset = Dataset::new(DataId::new("B03")?).with_array(array);

        FloatTiffWriter::default()
            .with_u16_scaled_output(0.0, 100.0, 17)?
            .save_dataset(&dataset, &path)?;

        let bytes = fs::read(&path)
            .map_err(|err| RustySatError::invalid_input(format!("failed to read TIFF: {err}")))?;
        assert_eq!(read_tag_u16(&bytes, TAG_BITS_PER_SAMPLE), 16);
        assert_eq!(
            read_tag_u16(&bytes, TAG_SAMPLE_FORMAT),
            SAMPLE_FORMAT_UNSIGNED_INTEGER
        );
        let offset = read_tag_u32(&bytes, TAG_STRIP_OFFSETS) as usize;
        assert_eq!(read_u16(&bytes, offset), 0);
        assert_eq!(read_u16(&bytes, offset + 2), 0);
        assert_eq!(read_u16(&bytes, offset + 4), 32768);
        assert_eq!(read_u16(&bytes, offset + 6), 17);
        assert_eq!(read_u16(&bytes, offset + 8), 65535);
        fs::remove_file(path).ok();
        Ok(())
    }

    #[test]
    fn writes_u16_auto_scaled_tiff_dataset_from_unmasked_finite_values() -> Result<()> {
        let path =
            temp_tiff_path("writes_u16_auto_scaled_tiff_dataset_from_unmasked_finite_values");
        let mask = ValidityMask::from_masked_flags([true, false, false]);
        let array = DataArray::<f32>::from_vec_named([1, 3], ["y", "x"], vec![1000.0, 10.0, 20.0])?
            .with_mask(mask)?;
        let dataset = Dataset::new(DataId::new("B03")?).with_array(array);

        FloatTiffWriter::default()
            .with_u16_auto_scaled_output(9)
            .save_dataset(&dataset, &path)?;

        let bytes = fs::read(&path)
            .map_err(|err| RustySatError::invalid_input(format!("failed to read TIFF: {err}")))?;
        let offset = read_tag_u32(&bytes, TAG_STRIP_OFFSETS) as usize;
        assert_eq!(read_u16(&bytes, offset), 9);
        assert_eq!(read_u16(&bytes, offset + 2), 0);
        assert_eq!(read_u16(&bytes, offset + 4), 65535);
        fs::remove_file(path).ok();
        Ok(())
    }

    #[test]
    fn rejects_invalid_u16_scale_policy() -> Result<()> {
        let err = FloatTiffWriter::default()
            .with_u16_scaled_output(1.0, 1.0, 0)
            .unwrap_err();
        assert!(err.to_string().contains("max greater than min"));
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
    fn writes_tiled_tiff() -> Result<()> {
        let path = temp_tiff_path("writes_tiled_tiff");
        let dataset =
            Dataset::new(DataId::new("B03")?).with_array(DataArray::<f64>::from_vec_named(
                [5, 7],
                ["y", "x"],
                (0..35).map(|i| i as f64).collect::<Vec<_>>(),
            )?);
        let tiles = TiffTileOptions {
            width: 3,
            height: 2,
        };
        FloatTiffWriter::new("tiled")?
            .with_tiles(tiles)
            .save_dataset(&dataset, &path)?;

        let bytes = fs::read(&path)
            .map_err(|err| RustySatError::invalid_input(format!("failed to read TIFF: {err}")))?;
        assert_eq!(read_tag_u32(&bytes, TAG_IMAGE_WIDTH), 7);
        assert_eq!(read_tag_u32(&bytes, TAG_IMAGE_LENGTH), 5);
        assert_eq!(read_tag_u32(&bytes, TAG_TILE_WIDTH), 3);
        assert_eq!(read_tag_u32(&bytes, TAG_TILE_LENGTH), 2);
        // 7×5 with 3×2 tiles = 3 tiles_x × 3 tiles_y = 9 tiles
        let _ = find_ifd_entry(&bytes, TAG_TILE_OFFSETS); // verify tag exists
        let tile_offsets_data = read_tag_u32(&bytes, TAG_TILE_OFFSETS) as usize;
        // Read all 9 tile offsets
        let mut offsets = Vec::new();
        for i in 0..9 {
            offsets.push(read_u32(&bytes, tile_offsets_data + i * 4));
        }
        assert_eq!(offsets.len(), 9);
        // offsets must be strictly increasing
        for w in offsets.windows(2) {
            assert!(w[0] < w[1], "tile offsets must be strictly increasing");
        }
        fs::remove_file(path).ok();
        Ok(())
    }

    #[test]
    fn writes_deflate_compressed_tiff() -> Result<()> {
        use flate2::read::DeflateDecoder;
        use std::io::Read;
        let path = temp_tiff_path("writes_deflate_compressed_tiff");
        let dataset = Dataset::new(DataId::new("B03")?).with_array(
            DataArray::<f64>::from_vec_named([2, 3], ["y", "x"], vec![1.0f64; 6])?,
        );
        let writer = FloatTiffWriter::new("deflate")?.with_compression(TiffCompression::Deflate);
        writer.save_dataset(&dataset, &path)?;

        let bytes = fs::read(&path)
            .map_err(|err| RustySatError::invalid_input(format!("failed to read TIFF: {err}")))?;
        assert_eq!(read_tag_u32(&bytes, TAG_IMAGE_WIDTH), 3);
        assert_eq!(read_tag_u32(&bytes, TAG_IMAGE_LENGTH), 2);
        assert_eq!(read_tag_u16(&bytes, TAG_COMPRESSION), COMPRESSION_DEFLATE);
        // compressed data should be smaller than raw (6 × 4 = 24 bytes)
        let strip_byte_count = read_tag_u32(&bytes, TAG_STRIP_BYTE_COUNTS) as usize;
        assert!(
            strip_byte_count < 24,
            "DEFLATE should reduce 24 bytes to < 24"
        );
        // decompress and verify pixel data
        let strip_offset = read_tag_u32(&bytes, TAG_STRIP_OFFSETS) as usize;
        let compressed = &bytes[strip_offset..strip_offset + strip_byte_count];
        let mut decoder = DeflateDecoder::new(compressed);
        let mut decompressed = Vec::new();
        decoder.read_to_end(&mut decompressed).map_err(|err| {
            RustySatError::invalid_input(format!("DEFLATE decompression failed: {err}"))
        })?;
        assert_eq!(decompressed.len(), 24);
        // verify first pixel = 1.0f32 (LE bytes)
        let pixel: f32 = f32::from_le_bytes(decompressed[0..4].try_into().unwrap());
        assert!((pixel - 1.0).abs() < 1e-10);
        fs::remove_file(path).ok();
        Ok(())
    }

    #[test]
    fn writes_geotiff_tags_from_area_metadata() -> Result<()> {
        let path = temp_tiff_path("writes_geotiff_tags_from_area_metadata");
        let mut dataset = Dataset::new(DataId::new("B03")?).with_array(
            DataArray::<f64>::from_vec_named([2, 3], ["y", "x"], vec![1.0; 6])?,
        );
        dataset.insert_attr("area", test_area_attr()?)?;

        FloatTiffWriter::default().save_dataset(&dataset, &path)?;

        let bytes = fs::read(&path)
            .map_err(|err| RustySatError::invalid_input(format!("failed to read TIFF: {err}")))?;
        // pixel scale / tiepoint unchanged
        let scale_offset = read_tag_u32(&bytes, TAG_MODEL_PIXEL_SCALE) as usize;
        assert_eq!(read_f64(&bytes, scale_offset), 10.0);
        let tiepoint_offset = read_tag_u32(&bytes, TAG_MODEL_TIEPOINT) as usize;
        assert_eq!(read_f64(&bytes, tiepoint_offset + 24), 100.0);
        assert_eq!(read_f64(&bytes, tiepoint_offset + 32), 220.0);
        // GeoKey directory now uses ProjCrs-based keys (>=7 keys for geos)
        let key_offset = read_tag_u32(&bytes, TAG_GEO_KEY_DIRECTORY) as usize;
        assert_eq!(read_u16(&bytes, key_offset), GEO_KEY_DIRECTORY_VERSION);
        let key_count = read_u16(&bytes, key_offset + 6);
        assert!(
            key_count >= 7,
            "expected >= 7 GeoKeys for geos projection, got {key_count}"
        );
        // Verify GeoAsciiParams contains ProjectedCitation
        let ascii_offset = read_tag_u32(&bytes, TIFFTAG_GEO_ASCII_PARAMS) as usize;
        let ascii_count = read_tag_count(&bytes, TIFFTAG_GEO_ASCII_PARAMS) as usize;
        let ascii = std::str::from_utf8(&bytes[ascii_offset..ascii_offset + ascii_count])
            .map_err(|err| RustySatError::invalid_input(err.to_string()))?;
        assert!(ascii.contains("GEOS"));
        // Verify GeoDoubleParams tag exists with correct count
        let double_offset = read_tag_u32(&bytes, TIFFTAG_GEO_DOUBLE_PARAMS) as usize;
        let double_count = read_tag_count(&bytes, TIFFTAG_GEO_DOUBLE_PARAMS) as usize;
        assert!(
            double_count >= 2,
            "expected >= 2 double params for geos, got {double_count}"
        );
        // Verify Double value at offset 0 is lon_0 = 140.7
        let lon_0_value = read_f64(&bytes, double_offset);
        assert!(
            (lon_0_value - 140.7).abs() < 1e-10,
            "expected lon_0=140.7 in double params, got {lon_0_value}"
        );
        fs::remove_file(path).ok();
        Ok(())
    }

    #[test]
    fn writes_geotiff_tags_from_xy_coordinate_axes() -> Result<()> {
        let path = temp_tiff_path("writes_geotiff_tags_from_xy_coordinate_axes");
        let array = DataArray::<f64>::from_vec_named([2, 3], ["y", "x"], vec![1.0; 6])?
            .with_coordinate(
                "x",
                rusty_sat_core::Coordinate::axis("x", vec![105.0, 115.0, 125.0])?,
            )?
            .with_coordinate(
                "y",
                rusty_sat_core::Coordinate::axis("y", vec![215.0, 205.0])?,
            )?;
        let dataset = Dataset::new(DataId::new("B03")?).with_array(array);

        FloatTiffWriter::default().save_dataset(&dataset, &path)?;

        let bytes = fs::read(&path)
            .map_err(|err| RustySatError::invalid_input(format!("failed to read TIFF: {err}")))?;
        let scale_offset = read_tag_u32(&bytes, TAG_MODEL_PIXEL_SCALE) as usize;
        assert_eq!(read_f64(&bytes, scale_offset), 10.0);
        assert_eq!(read_f64(&bytes, scale_offset + 8), 10.0);
        let tiepoint_offset = read_tag_u32(&bytes, TAG_MODEL_TIEPOINT) as usize;
        assert_eq!(read_f64(&bytes, tiepoint_offset + 24), 100.0);
        assert_eq!(read_f64(&bytes, tiepoint_offset + 32), 220.0);
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

    fn read_tag_count(bytes: &[u8], tag: u16) -> u32 {
        let offset = find_ifd_entry(bytes, tag) + 4;
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

    fn read_f64(bytes: &[u8], offset: usize) -> f64 {
        f64::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
            bytes[offset + 4],
            bytes[offset + 5],
            bytes[offset + 6],
            bytes[offset + 7],
        ])
    }

    fn test_area_attr() -> Result<MetadataValue> {
        Ok(MetadataValue::map([
            ("type", MetadataValue::string("area")),
            ("id", MetadataValue::string("test_area")),
            ("description", MetadataValue::string("test area")),
            ("proj_id", MetadataValue::string("test_area")),
            (
                "projection",
                MetadataValue::map([
                    ("proj", MetadataValue::string("geos")),
                    ("lon_0", MetadataValue::string("140.7")),
                ]),
            ),
            ("height", MetadataValue::Integer(2)),
            ("width", MetadataValue::Integer(3)),
            (
                "area_extent",
                MetadataValue::List(
                    [100.0, 200.0, 130.0, 220.0]
                        .into_iter()
                        .map(MetadataValue::float)
                        .collect::<Result<Vec<_>>>()?,
                ),
            ),
        ]))
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
