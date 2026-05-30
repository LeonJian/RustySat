//! Minimal scientific TIFF writer foundation.
//!
//! This is intentionally a baseline uncompressed TIFF writer, not full GeoTIFF
//! parity yet. It preserves calibrated values as 32-bit floating point samples
//! so AHI scientific output paths do not have to go through display scaling.

use crate::Writer;
use rusty_sat_core::{Dataset, MetadataValue, Result, RustySatError};
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
const TAG_MODEL_PIXEL_SCALE: u16 = 33550;
const TAG_MODEL_TIEPOINT: u16 = 33922;
const TAG_SAMPLE_FORMAT: u16 = 339;
const TAG_GEO_KEY_DIRECTORY: u16 = 34735;
const TAG_GEO_ASCII_PARAMS: u16 = 34737;

const TYPE_ASCII: u16 = 2;
const TYPE_SHORT: u16 = 3;
const TYPE_LONG: u16 = 4;
const TYPE_DOUBLE: u16 = 12;

const COMPRESSION_NONE: u16 = 1;
const PHOTOMETRIC_BLACK_IS_ZERO: u16 = 1;
const SAMPLE_FORMAT_IEEE_FLOAT: u16 = 3;
const GEO_KEY_DIRECTORY_VERSION: u16 = 1;
const GEO_KEY_REVISION: u16 = 1;
const GEO_KEY_MINOR_REVISION: u16 = 0;
const GEO_KEY_MODEL_TYPE: u16 = 1024;
const GEO_KEY_RASTER_TYPE: u16 = 1025;
const GEO_KEY_PROJECTED_CS_TYPE: u16 = 3072;
const GEO_KEY_PROJECTED_CITATION: u16 = 3073;
const GEO_MODEL_TYPE_PROJECTED: u16 = 1;
const GEO_RASTER_PIXEL_IS_AREA: u16 = 1;
const GEO_USER_DEFINED: u16 = 32767;

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

    write_float_tiff_pixels(
        path,
        width,
        height,
        &pixels,
        geo_info_from_dataset(dataset)?,
    )
}

fn write_float_tiff_pixels(
    path: &Path,
    width: usize,
    height: usize,
    pixels: &[f32],
    geo_info: Option<GeoTiffInfo>,
) -> Result<()> {
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

    let geotiff_data = geo_info
        .as_ref()
        .map(GeoTiffExtraData::from_info)
        .transpose()?;
    let entry_count = 10_u16 + if geotiff_data.is_some() { 4_u16 } else { 0_u16 };
    let ifd_bytes = 2_u32 + u32::from(entry_count) * 12 + 4;
    let extra_bytes = geotiff_data
        .as_ref()
        .map(GeoTiffExtraData::byte_len)
        .unwrap_or(0);
    let pixel_offset = IFD_OFFSET
        .checked_add(ifd_bytes)
        .and_then(|offset| offset.checked_add(u32::try_from(extra_bytes).ok()?))
        .ok_or_else(|| RustySatError::invalid_input("float TIFF pixel offset overflow"))?;
    let extra_offset = IFD_OFFSET
        .checked_add(ifd_bytes)
        .ok_or_else(|| RustySatError::invalid_input("float TIFF GeoTIFF offset overflow"))?;

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
    if let Some(geotiff_data) = geotiff_data.as_ref() {
        geotiff_data.write_ifd_entries(&mut writer, extra_offset)?;
    }
    write_u32(&mut writer, 0)?;
    if let Some(geotiff_data) = geotiff_data.as_ref() {
        geotiff_data.write_data(&mut writer)?;
    }
    for pixel in pixels {
        writer
            .write_all(&pixel.to_le_bytes())
            .map_err(write_error)?;
    }
    Ok(())
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
}

impl GeoTiffExtraData {
    fn from_info(info: &GeoTiffInfo) -> Result<Self> {
        let mut ascii = info.citation.clone();
        if !ascii.ends_with('|') {
            ascii.push('|');
        }
        let ascii_bytes = ascii.into_bytes();
        let ascii_len = u16::try_from(ascii_bytes.len())
            .map_err(|_| RustySatError::invalid_input("GeoTIFF citation is too long"))?;
        let keys = [
            GEO_KEY_MODEL_TYPE,
            0,
            1,
            GEO_MODEL_TYPE_PROJECTED,
            GEO_KEY_RASTER_TYPE,
            0,
            1,
            GEO_RASTER_PIXEL_IS_AREA,
            GEO_KEY_PROJECTED_CS_TYPE,
            0,
            1,
            GEO_USER_DEFINED,
            GEO_KEY_PROJECTED_CITATION,
            TAG_GEO_ASCII_PARAMS,
            ascii_len,
            0,
        ];
        let mut geo_key_directory = Vec::with_capacity((4 + keys.len()) * 2);
        for value in [
            GEO_KEY_DIRECTORY_VERSION,
            GEO_KEY_REVISION,
            GEO_KEY_MINOR_REVISION,
            4,
        ] {
            geo_key_directory.extend_from_slice(&value.to_le_bytes());
        }
        for value in keys {
            geo_key_directory.extend_from_slice(&value.to_le_bytes());
        }
        Ok(Self {
            pixel_scale: f64_values_to_bytes(&info.pixel_scale),
            tiepoint: f64_values_to_bytes(&info.tiepoint),
            geo_key_directory,
            geo_ascii_params: ascii_bytes,
        })
    }

    fn byte_len(&self) -> u32 {
        (self.pixel_scale.len()
            + self.tiepoint.len()
            + self.geo_key_directory.len()
            + self.geo_ascii_params.len()) as u32
    }

    fn write_ifd_entries(&self, writer: &mut impl Write, offset: u32) -> Result<()> {
        let pixel_scale_offset = offset;
        let tiepoint_offset = pixel_scale_offset + self.pixel_scale.len() as u32;
        let geo_key_offset = tiepoint_offset + self.tiepoint.len() as u32;
        let geo_ascii_offset = geo_key_offset + self.geo_key_directory.len() as u32;
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
            TAG_GEO_ASCII_PARAMS,
            TYPE_ASCII,
            u32::try_from(self.geo_ascii_params.len())
                .map_err(|_| RustySatError::invalid_input("GeoTIFF ASCII count overflow"))?,
            geo_ascii_offset,
        )
    }

    fn write_data(&self, writer: &mut impl Write) -> Result<()> {
        writer.write_all(&self.pixel_scale).map_err(write_error)?;
        writer.write_all(&self.tiepoint).map_err(write_error)?;
        writer
            .write_all(&self.geo_key_directory)
            .map_err(write_error)?;
        writer
            .write_all(&self.geo_ascii_params)
            .map_err(write_error)
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
    fn writes_geotiff_tags_from_area_metadata() -> Result<()> {
        let path = temp_tiff_path("writes_geotiff_tags_from_area_metadata");
        let mut dataset = Dataset::new(DataId::new("B03")?).with_array(
            DataArray::<f64>::from_vec_named([2, 3], ["y", "x"], vec![1.0; 6])?,
        );
        dataset.insert_attr("area", test_area_attr()?)?;

        FloatTiffWriter::default().save_dataset(&dataset, &path)?;

        let bytes = fs::read(&path)
            .map_err(|err| RustySatError::invalid_input(format!("failed to read TIFF: {err}")))?;
        let scale_offset = read_tag_u32(&bytes, TAG_MODEL_PIXEL_SCALE) as usize;
        assert_eq!(read_f64(&bytes, scale_offset), 10.0);
        assert_eq!(read_f64(&bytes, scale_offset + 8), 10.0);
        assert_eq!(read_f64(&bytes, scale_offset + 16), 0.0);
        let tiepoint_offset = read_tag_u32(&bytes, TAG_MODEL_TIEPOINT) as usize;
        assert_eq!(read_f64(&bytes, tiepoint_offset), 0.0);
        assert_eq!(read_f64(&bytes, tiepoint_offset + 8), 0.0);
        assert_eq!(read_f64(&bytes, tiepoint_offset + 24), 100.0);
        assert_eq!(read_f64(&bytes, tiepoint_offset + 32), 220.0);
        let key_offset = read_tag_u32(&bytes, TAG_GEO_KEY_DIRECTORY) as usize;
        assert_eq!(read_u16(&bytes, key_offset), GEO_KEY_DIRECTORY_VERSION);
        assert_eq!(read_u16(&bytes, key_offset + 6), 4);
        let ascii_offset = read_tag_u32(&bytes, TAG_GEO_ASCII_PARAMS) as usize;
        let ascii_count = read_tag_count(&bytes, TAG_GEO_ASCII_PARAMS) as usize;
        let ascii = std::str::from_utf8(&bytes[ascii_offset..ascii_offset + ascii_count])
            .map_err(|err| RustySatError::invalid_input(err.to_string()))?;
        assert_eq!(ascii, "RustySat area test_area proj=geos|");
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
