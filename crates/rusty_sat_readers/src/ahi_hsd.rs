//! Himawari AHI HSD binary header foundations.
//!
//! Reference behavior inspected before implementation:
//! - Root `HS_D_users_guide_en_v12.pdf` is the local HSD user guide reference.
//! - `satpy/satpy/readers/ahi_hsd.py` defines the NumPy dtypes for HSD header
//!   blocks 1-5 and reads them in sequence before dataset loading.
//!
//! This module is intentionally limited to fixed-size initial header parsing.
//! It does not read image data, calibration update blocks, or segment arrays
//! yet.

use rusty_sat_core::{Result, RustySatError};

const BASIC_INFO_LEN: usize = 282;
const DATA_INFO_LEN: usize = 50;
const PROJECTION_INFO_LEN: usize = 127;
const NAVIGATION_INFO_LEN: usize = 139;
const CALIBRATION_INFO_LEN: usize = 35;

#[derive(Debug, Clone, PartialEq)]
pub struct AhiHsdHeader {
    pub basic: AhiBasicInfo,
    pub data: AhiDataInfo,
    pub projection: AhiProjectionInfo,
    pub navigation: AhiNavigationInfo,
    pub calibration: AhiCalibrationInfo,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AhiBasicInfo {
    pub header_block_number: u8,
    pub block_length: u16,
    pub total_header_blocks: u16,
    pub byte_order: u8,
    pub satellite: String,
    pub processing_center_name: String,
    pub observation_area: String,
    pub observation_timeline: u16,
    pub observation_start_time_days: f64,
    pub observation_end_time_days: f64,
    pub file_creation_time_days: f64,
    pub total_header_length: u32,
    pub total_data_length: u32,
    pub file_format_version: String,
    pub file_name: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AhiDataInfo {
    pub header_block_number: u8,
    pub block_length: u16,
    pub bits_per_pixel: u16,
    pub columns: u16,
    pub lines: u16,
    pub compression_flag: u8,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AhiProjectionInfo {
    pub header_block_number: u8,
    pub block_length: u16,
    pub sub_lon: f64,
    pub cfac: u32,
    pub lfac: u32,
    pub coff: f32,
    pub loff: f32,
    pub distance_from_earth_center: f64,
    pub earth_equatorial_radius: f64,
    pub earth_polar_radius: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AhiNavigationInfo {
    pub header_block_number: u8,
    pub block_length: u16,
    pub navigation_info_time_days: f64,
    pub ssp_longitude: f64,
    pub ssp_latitude: f64,
    pub distance_earth_center_to_satellite: f64,
    pub nadir_longitude: f64,
    pub nadir_latitude: f64,
    pub sun_position: [f64; 3],
    pub moon_position: [f64; 3],
}

#[derive(Debug, Clone, PartialEq)]
pub struct AhiCalibrationInfo {
    pub header_block_number: u8,
    pub block_length: u16,
    pub band_number: u16,
    pub central_wavelength: f64,
    pub valid_bits_per_pixel: u16,
    pub error_pixel_count_value: u16,
    pub outside_scan_pixel_count_value: u16,
    pub gain_count_to_radiance: f64,
    pub offset_count_to_radiance: f64,
}

pub fn parse_initial_hsd_header(bytes: &[u8]) -> Result<AhiHsdHeader> {
    let basic = AhiBasicInfo::parse(take_block(bytes, 0, BASIC_INFO_LEN, "basic information")?)?;
    let data_offset = usize::from(basic.block_length);
    let data = AhiDataInfo::parse(take_block(
        bytes,
        data_offset,
        DATA_INFO_LEN,
        "data information",
    )?)?;
    let projection_offset = data_offset + usize::from(data.block_length);
    let projection = AhiProjectionInfo::parse(take_block(
        bytes,
        projection_offset,
        PROJECTION_INFO_LEN,
        "projection information",
    )?)?;
    let navigation_offset = projection_offset + usize::from(projection.block_length);
    let navigation = AhiNavigationInfo::parse(take_block(
        bytes,
        navigation_offset,
        NAVIGATION_INFO_LEN,
        "navigation information",
    )?)?;
    let calibration_offset = navigation_offset + usize::from(navigation.block_length);
    let calibration = AhiCalibrationInfo::parse(take_block(
        bytes,
        calibration_offset,
        CALIBRATION_INFO_LEN,
        "calibration information",
    )?)?;

    Ok(AhiHsdHeader {
        basic,
        data,
        projection,
        navigation,
        calibration,
    })
}

impl AhiBasicInfo {
    fn parse(bytes: &[u8]) -> Result<Self> {
        Ok(Self {
            header_block_number: read_u8(bytes, 0, "basic hblock_number")?,
            block_length: read_u16_le(bytes, 1, "basic blocklength")?,
            total_header_blocks: read_u16_le(bytes, 3, "basic total_number_of_hblocks")?,
            byte_order: read_u8(bytes, 5, "basic byte_order")?,
            satellite: read_fixed_string(bytes, 6, 16, "basic satellite")?,
            processing_center_name: read_fixed_string(bytes, 22, 16, "basic processing center")?,
            observation_area: read_fixed_string(bytes, 38, 4, "basic observation area")?,
            observation_timeline: read_u16_le(bytes, 44, "basic observation timeline")?,
            observation_start_time_days: read_f64_le(bytes, 46, "basic observation start time")?,
            observation_end_time_days: read_f64_le(bytes, 54, "basic observation end time")?,
            file_creation_time_days: read_f64_le(bytes, 62, "basic file creation time")?,
            total_header_length: read_u32_le(bytes, 70, "basic total header length")?,
            total_data_length: read_u32_le(bytes, 74, "basic total data length")?,
            file_format_version: read_fixed_string(bytes, 82, 32, "basic file format version")?,
            file_name: read_fixed_string(bytes, 114, 128, "basic file name")?,
        })
    }
}

impl AhiDataInfo {
    fn parse(bytes: &[u8]) -> Result<Self> {
        Ok(Self {
            header_block_number: read_u8(bytes, 0, "data hblock_number")?,
            block_length: read_u16_le(bytes, 1, "data blocklength")?,
            bits_per_pixel: read_u16_le(bytes, 3, "data bits per pixel")?,
            columns: read_u16_le(bytes, 5, "data columns")?,
            lines: read_u16_le(bytes, 7, "data lines")?,
            compression_flag: read_u8(bytes, 9, "data compression flag")?,
        })
    }
}

impl AhiProjectionInfo {
    fn parse(bytes: &[u8]) -> Result<Self> {
        Ok(Self {
            header_block_number: read_u8(bytes, 0, "projection hblock_number")?,
            block_length: read_u16_le(bytes, 1, "projection blocklength")?,
            sub_lon: read_f64_le(bytes, 3, "projection sub_lon")?,
            cfac: read_u32_le(bytes, 11, "projection CFAC")?,
            lfac: read_u32_le(bytes, 15, "projection LFAC")?,
            coff: read_f32_le(bytes, 19, "projection COFF")?,
            loff: read_f32_le(bytes, 23, "projection LOFF")?,
            distance_from_earth_center: read_f64_le(
                bytes,
                27,
                "projection distance from earth center",
            )?,
            earth_equatorial_radius: read_f64_le(bytes, 35, "projection equatorial radius")?,
            earth_polar_radius: read_f64_le(bytes, 43, "projection polar radius")?,
        })
    }
}

impl AhiNavigationInfo {
    fn parse(bytes: &[u8]) -> Result<Self> {
        Ok(Self {
            header_block_number: read_u8(bytes, 0, "navigation hblock_number")?,
            block_length: read_u16_le(bytes, 1, "navigation blocklength")?,
            navigation_info_time_days: read_f64_le(bytes, 3, "navigation info time")?,
            ssp_longitude: read_f64_le(bytes, 11, "navigation SSP longitude")?,
            ssp_latitude: read_f64_le(bytes, 19, "navigation SSP latitude")?,
            distance_earth_center_to_satellite: read_f64_le(bytes, 27, "navigation distance")?,
            nadir_longitude: read_f64_le(bytes, 35, "navigation nadir longitude")?,
            nadir_latitude: read_f64_le(bytes, 43, "navigation nadir latitude")?,
            sun_position: [
                read_f64_le(bytes, 51, "navigation sun position x")?,
                read_f64_le(bytes, 59, "navigation sun position y")?,
                read_f64_le(bytes, 67, "navigation sun position z")?,
            ],
            moon_position: [
                read_f64_le(bytes, 75, "navigation moon position x")?,
                read_f64_le(bytes, 83, "navigation moon position y")?,
                read_f64_le(bytes, 91, "navigation moon position z")?,
            ],
        })
    }
}

impl AhiCalibrationInfo {
    fn parse(bytes: &[u8]) -> Result<Self> {
        Ok(Self {
            header_block_number: read_u8(bytes, 0, "calibration hblock_number")?,
            block_length: read_u16_le(bytes, 1, "calibration blocklength")?,
            band_number: read_u16_le(bytes, 3, "calibration band number")?,
            central_wavelength: read_f64_le(bytes, 5, "calibration central wavelength")?,
            valid_bits_per_pixel: read_u16_le(bytes, 13, "calibration valid bits")?,
            error_pixel_count_value: read_u16_le(bytes, 15, "calibration error pixel value")?,
            outside_scan_pixel_count_value: read_u16_le(
                bytes,
                17,
                "calibration outside scan pixel value",
            )?,
            gain_count_to_radiance: read_f64_le(bytes, 19, "calibration gain")?,
            offset_count_to_radiance: read_f64_le(bytes, 27, "calibration offset")?,
        })
    }
}

fn take_block<'a>(bytes: &'a [u8], offset: usize, min_len: usize, name: &str) -> Result<&'a [u8]> {
    bytes
        .get(offset..offset + min_len)
        .ok_or_else(|| RustySatError::invalid_input(format!("AHI HSD {name} block is truncated")))
}

fn read_u8(bytes: &[u8], offset: usize, field: &str) -> Result<u8> {
    bytes
        .get(offset)
        .copied()
        .ok_or_else(|| truncated_field(field))
}

fn read_u16_le(bytes: &[u8], offset: usize, field: &str) -> Result<u16> {
    Ok(u16::from_le_bytes(read_array(bytes, offset, field)?))
}

fn read_u32_le(bytes: &[u8], offset: usize, field: &str) -> Result<u32> {
    Ok(u32::from_le_bytes(read_array(bytes, offset, field)?))
}

fn read_f32_le(bytes: &[u8], offset: usize, field: &str) -> Result<f32> {
    Ok(f32::from_le_bytes(read_array(bytes, offset, field)?))
}

fn read_f64_le(bytes: &[u8], offset: usize, field: &str) -> Result<f64> {
    Ok(f64::from_le_bytes(read_array(bytes, offset, field)?))
}

fn read_array<const N: usize>(bytes: &[u8], offset: usize, field: &str) -> Result<[u8; N]> {
    bytes
        .get(offset..offset + N)
        .and_then(|value| value.try_into().ok())
        .ok_or_else(|| truncated_field(field))
}

fn read_fixed_string(bytes: &[u8], offset: usize, len: usize, field: &str) -> Result<String> {
    let raw = bytes
        .get(offset..offset + len)
        .ok_or_else(|| truncated_field(field))?;
    let end = raw.iter().position(|byte| *byte == 0).unwrap_or(raw.len());
    Ok(String::from_utf8_lossy(&raw[..end]).trim().to_string())
}

fn truncated_field(field: &str) -> RustySatError {
    RustySatError::invalid_input(format!("AHI HSD field '{field}' is truncated"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_initial_ahi_hsd_header_blocks() {
        let bytes = synthetic_header();

        let header = parse_initial_hsd_header(&bytes).unwrap();

        assert_eq!(header.basic.header_block_number, 1);
        assert_eq!(header.basic.block_length, 282);
        assert_eq!(header.basic.satellite, "Himawari-8");
        assert_eq!(header.basic.observation_area, "FLDK");
        assert_eq!(header.basic.observation_timeline, 1234);
        assert_eq!(header.basic.total_header_length, 500);
        assert_eq!(header.data.bits_per_pixel, 16);
        assert_eq!(header.data.columns, 10);
        assert_eq!(header.data.lines, 20);
        assert_eq!(header.projection.sub_lon, 140.7);
        assert_eq!(header.projection.cfac, 40932549);
        assert_eq!(header.navigation.sun_position, [1.0, 2.0, 3.0]);
        assert_eq!(header.calibration.band_number, 3);
        assert_eq!(header.calibration.central_wavelength, 0.64);
        assert_eq!(header.calibration.gain_count_to_radiance, 0.01);
    }

    #[test]
    fn rejects_truncated_initial_header() {
        let bytes = synthetic_header();
        let err = parse_initial_hsd_header(&bytes[..100]).unwrap_err();

        assert!(err.to_string().contains("truncated"));
    }

    fn synthetic_header() -> Vec<u8> {
        let mut bytes = vec![
            0;
            BASIC_INFO_LEN
                + DATA_INFO_LEN
                + PROJECTION_INFO_LEN
                + NAVIGATION_INFO_LEN
                + CALIBRATION_INFO_LEN
        ];
        write_u8(&mut bytes, 0, 1);
        write_u16(&mut bytes, 1, BASIC_INFO_LEN as u16);
        write_u16(&mut bytes, 3, 11);
        write_u8(&mut bytes, 5, 1);
        write_string(&mut bytes, 6, 16, "Himawari-8");
        write_string(&mut bytes, 22, 16, "MSC");
        write_string(&mut bytes, 38, 4, "FLDK");
        write_u16(&mut bytes, 44, 1234);
        write_f64(&mut bytes, 46, 60000.25);
        write_f64(&mut bytes, 54, 60000.30);
        write_f64(&mut bytes, 62, 60000.35);
        write_u32(&mut bytes, 70, 500);
        write_u32(&mut bytes, 74, 200);
        write_string(&mut bytes, 82, 32, "HSD-v1");
        write_string(&mut bytes, 114, 128, "HS_H08_20200101.dat");

        let data = BASIC_INFO_LEN;
        write_u8(&mut bytes, data, 2);
        write_u16(&mut bytes, data + 1, DATA_INFO_LEN as u16);
        write_u16(&mut bytes, data + 3, 16);
        write_u16(&mut bytes, data + 5, 10);
        write_u16(&mut bytes, data + 7, 20);
        write_u8(&mut bytes, data + 9, 0);

        let proj = data + DATA_INFO_LEN;
        write_u8(&mut bytes, proj, 3);
        write_u16(&mut bytes, proj + 1, PROJECTION_INFO_LEN as u16);
        write_f64(&mut bytes, proj + 3, 140.7);
        write_u32(&mut bytes, proj + 11, 40932549);
        write_u32(&mut bytes, proj + 15, 40932549);
        write_f32(&mut bytes, proj + 19, 5500.5);
        write_f32(&mut bytes, proj + 23, 5500.5);
        write_f64(&mut bytes, proj + 27, 42164.0);
        write_f64(&mut bytes, proj + 35, 6378.137);
        write_f64(&mut bytes, proj + 43, 6356.7523);

        let nav = proj + PROJECTION_INFO_LEN;
        write_u8(&mut bytes, nav, 4);
        write_u16(&mut bytes, nav + 1, NAVIGATION_INFO_LEN as u16);
        write_f64(&mut bytes, nav + 3, 60000.26);
        write_f64(&mut bytes, nav + 11, 140.7);
        write_f64(&mut bytes, nav + 19, 0.0);
        write_f64(&mut bytes, nav + 27, 42164.0);
        write_f64(&mut bytes, nav + 35, 140.7);
        write_f64(&mut bytes, nav + 43, 0.0);
        for (idx, value) in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0].into_iter().enumerate() {
            write_f64(&mut bytes, nav + 51 + idx * 8, value);
        }

        let cal = nav + NAVIGATION_INFO_LEN;
        write_u8(&mut bytes, cal, 5);
        write_u16(&mut bytes, cal + 1, CALIBRATION_INFO_LEN as u16);
        write_u16(&mut bytes, cal + 3, 3);
        write_f64(&mut bytes, cal + 5, 0.64);
        write_u16(&mut bytes, cal + 13, 12);
        write_u16(&mut bytes, cal + 15, 65535);
        write_u16(&mut bytes, cal + 17, 65534);
        write_f64(&mut bytes, cal + 19, 0.01);
        write_f64(&mut bytes, cal + 27, -1.0);

        bytes
    }

    fn write_u8(bytes: &mut [u8], offset: usize, value: u8) {
        bytes[offset] = value;
    }

    fn write_u16(bytes: &mut [u8], offset: usize, value: u16) {
        bytes[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
    }

    fn write_u32(bytes: &mut [u8], offset: usize, value: u32) {
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }

    fn write_f32(bytes: &mut [u8], offset: usize, value: f32) {
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }

    fn write_f64(bytes: &mut [u8], offset: usize, value: f64) {
        bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
    }

    fn write_string(bytes: &mut [u8], offset: usize, len: usize, value: &str) {
        let raw = value.as_bytes();
        bytes[offset..offset + raw.len().min(len)].copy_from_slice(&raw[..raw.len().min(len)]);
    }
}
