//! Himawari AHI HSD binary header foundations.
//!
//! Reference behavior inspected before implementation:
//! - Root `HS_D_users_guide_en_v12.pdf` is the local HSD user guide reference.
//! - `satpy/satpy/readers/ahi_hsd.py` defines the NumPy dtypes for HSD header
//!   blocks 1-5 and reads them in sequence before dataset loading.
//!
//! This module is intentionally limited to fixed-size initial header parsing,
//! bounded bzip2/uncompressed raw-count loading, segment assembly, geostationary
//! area metadata, and first-pass calibration. Satpy's display calibration path
//! uses float32-like arithmetic for memory-efficient imagery; Rusty Sat also
//! exposes f64 calibration helpers for future scientific/HDR output paths where
//! precision preservation matters more than display memory.

use crate::filename_pattern::PatternValue;
use crate::yaml_reader::FileMatch;
use crate::Reader;
use bzip2::bufread::MultiBzDecoder;
use rayon::prelude::*;
use rusty_sat_core::{
    AnyDataArray, Coordinate, DataArray, DataId, Dataset, MetadataValue, NumericElement,
    ReaderInventory, Result, RustySatError, ValidityMask, WavelengthRange,
};
use std::borrow::Cow;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

const BASIC_INFO_LEN: usize = 282;
const DATA_INFO_LEN: usize = 50;
const PROJECTION_INFO_LEN: usize = 127;
const NAVIGATION_INFO_LEN: usize = 139;
const CALIBRATION_INFO_LEN: usize = 35;
const BAND_CALIBRATION_EXTENSION_LEN: usize = 112;
const VISIBLE_CALIBRATION_INFO_LEN: usize = CALIBRATION_INFO_LEN + BAND_CALIBRATION_EXTENSION_LEN;
const INFRARED_CALIBRATION_INFO_LEN: usize = CALIBRATION_INFO_LEN + BAND_CALIBRATION_EXTENSION_LEN;
const INITIAL_HEADER_PREFIX_LEN: u64 = 4096;
const MAX_HSD_FILE_BYTES: u64 = 2 * 1024 * 1024 * 1024;
const BZIP2_MAGIC: [u8; 3] = *b"BZh";

/// Bounds how many AHI HSD segments are loaded/calibrated at once.
///
/// Segment loads are independent and parallelized with rayon, but each
/// segment holds a full-width array in memory during assembly; a bounded
/// chunk keeps the peak memory of `assembled + concurrent segments` in check
/// while still parallelizing the dominant bzip2/calibration cost.
const CONCURRENT_SEGMENT_LOADS: usize = 4;

#[derive(Debug, Clone, PartialEq)]
pub struct AhiHsdHeader {
    pub basic: AhiBasicInfo,
    pub data: AhiDataInfo,
    pub projection: AhiProjectionInfo,
    pub navigation: AhiNavigationInfo,
    pub calibration: AhiCalibrationInfo,
    pub segment: Option<AhiSegmentBlockInfo>,
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
    pub band_calibration: Option<AhiBandCalibration>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum AhiBandCalibration {
    Visible {
        coeff_rad_to_albedo: f64,
        coeff_update_time_days: f64,
        calibrated_gain_count_to_radiance: f64,
        calibrated_offset_count_to_radiance: f64,
    },
    Infrared {
        c0_rad_to_tb: f64,
        c1_rad_to_tb: f64,
        c2_rad_to_tb: f64,
        c0_tb_to_rad: f64,
        c1_tb_to_rad: f64,
        c2_tb_to_rad: f64,
        speed_of_light: f64,
        planck_constant: f64,
        boltzmann_constant: f64,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AhiCalibration {
    Counts,
    Radiance,
    Reflectance,
    BrightnessTemperature,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AhiCalibrationMode {
    Nominal,
    Update,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AhiCalibrationOutput {
    DisplayF32,
    ScientificF64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AhiUserCalibrationType {
    RadianceCorrection,
    DigitalNumber,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AhiUserCalibrationCoefficients {
    pub slope: f64,
    pub offset: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AhiUserCalibration {
    correction_type: AhiUserCalibrationType,
    coefficients: BTreeMap<String, AhiUserCalibrationCoefficients>,
}

impl AhiUserCalibration {
    pub fn radiance_correction(
        coefficients: impl IntoIterator<Item = (impl Into<String>, AhiUserCalibrationCoefficients)>,
    ) -> Result<Self> {
        Self::new(AhiUserCalibrationType::RadianceCorrection, coefficients)
    }

    pub fn digital_number(
        coefficients: impl IntoIterator<Item = (impl Into<String>, AhiUserCalibrationCoefficients)>,
    ) -> Result<Self> {
        Self::new(AhiUserCalibrationType::DigitalNumber, coefficients)
    }

    fn new(
        correction_type: AhiUserCalibrationType,
        coefficients: impl IntoIterator<Item = (impl Into<String>, AhiUserCalibrationCoefficients)>,
    ) -> Result<Self> {
        let mut values = BTreeMap::new();
        for (band, coeffs) in coefficients {
            let band = band.into();
            if band.trim().is_empty() {
                return Err(RustySatError::invalid_input(
                    "AHI user calibration band name cannot be empty",
                ));
            }
            if !coeffs.slope.is_finite() || !coeffs.offset.is_finite() {
                return Err(RustySatError::invalid_input(
                    "AHI user calibration coefficients must be finite",
                ));
            }
            if correction_type == AhiUserCalibrationType::RadianceCorrection && coeffs.slope == 0.0
            {
                return Err(RustySatError::invalid_input(
                    "AHI radiance correction slope cannot be zero",
                ));
            }
            values.insert(band, coeffs);
        }
        Ok(Self {
            correction_type,
            coefficients: values,
        })
    }

    fn coefficients_for(&self, band_name: &str) -> AhiUserCalibrationCoefficients {
        self.coefficients
            .get(band_name)
            .copied()
            .unwrap_or(AhiUserCalibrationCoefficients {
                slope: 1.0,
                offset: 0.0,
            })
    }

    fn correction_type(&self) -> AhiUserCalibrationType {
        self.correction_type
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AhiSegmentBlockInfo {
    pub header_block_number: u8,
    pub block_length: u16,
    pub total_segments: u8,
    pub segment_sequence_number: u8,
    pub first_line_number: u16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AhiSegmentInfo {
    pub segment_number: u8,
    pub total_segments: u8,
}

impl AhiSegmentInfo {
    pub fn new(segment_number: u8, total_segments: u8) -> Result<Self> {
        if segment_number == 0 {
            return Err(RustySatError::invalid_input(
                "AHI HSD segment number must be greater than zero",
            ));
        }
        if total_segments == 0 {
            return Err(RustySatError::invalid_input(
                "AHI HSD total segments must be greater than zero",
            ));
        }
        if segment_number > total_segments {
            return Err(RustySatError::invalid_input(
                "AHI HSD segment number cannot exceed total segments",
            ));
        }
        Ok(Self {
            segment_number,
            total_segments,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct AhiHsdFileHandler {
    filename: PathBuf,
    file_type: String,
    header: AhiHsdHeader,
    segment: AhiSegmentInfo,
    calibration_mode: AhiCalibrationMode,
    user_calibration: Option<AhiUserCalibration>,
}

impl AhiHsdFileHandler {
    pub fn from_header_bytes(
        filename: impl Into<PathBuf>,
        file_type: impl Into<String>,
        segment: AhiSegmentInfo,
        bytes: &[u8],
    ) -> Result<Self> {
        let file_type = file_type.into();
        if file_type.trim().is_empty() {
            return Err(RustySatError::invalid_input(
                "AHI HSD file type cannot be empty",
            ));
        }
        Ok(Self {
            filename: filename.into(),
            file_type,
            header: parse_initial_hsd_header(bytes)?,
            segment,
            calibration_mode: AhiCalibrationMode::Update,
            user_calibration: None,
        })
    }

    pub fn from_file_match_and_header_bytes(file_match: &FileMatch, bytes: &[u8]) -> Result<Self> {
        let segment_number = required_filename_u8(file_match, "segment")?;
        let total_segments = required_filename_u8(file_match, "total_segments")?;
        Self::from_header_bytes(
            file_match.filename(),
            file_match.file_type(),
            AhiSegmentInfo::new(segment_number, total_segments)?,
            bytes,
        )
    }

    pub fn from_path(
        filename: impl Into<PathBuf>,
        file_type: impl Into<String>,
        segment: AhiSegmentInfo,
    ) -> Result<Self> {
        let filename = filename.into();
        let bytes = read_initial_hsd_header_prefix(&filename)?;
        Self::from_header_bytes(filename, file_type, segment, &bytes)
    }

    pub fn filename(&self) -> &Path {
        &self.filename
    }

    pub fn file_type(&self) -> &str {
        &self.file_type
    }

    pub fn header(&self) -> &AhiHsdHeader {
        &self.header
    }

    pub fn segment(&self) -> AhiSegmentInfo {
        self.segment
    }

    pub fn calibration_mode(&self) -> AhiCalibrationMode {
        self.calibration_mode
    }

    pub fn with_calibration_mode(mut self, mode: AhiCalibrationMode) -> Self {
        self.calibration_mode = mode;
        self
    }

    pub fn with_user_calibration(mut self, user_calibration: AhiUserCalibration) -> Self {
        self.user_calibration = Some(user_calibration);
        self
    }

    pub fn band_name(&self) -> String {
        format!("B{:02}", self.header.calibration.band_number)
    }

    pub fn dataset_id(&self) -> Result<DataId> {
        let wavelength = self.header.calibration.central_wavelength;
        DataId::new(self.band_name())?
            .with_qualifier(
                "wavelength",
                WavelengthRange::micrometers(wavelength, wavelength, wavelength)?,
            )?
            .with_qualifier("calibration", "counts")
    }

    pub fn dataset_stub(&self) -> Result<Dataset> {
        let mut dataset = Dataset::new(self.dataset_id()?);
        self.attach_common_attrs(&mut dataset)?;
        self.attach_area_attr(&mut dataset)?;
        Ok(dataset)
    }

    pub fn counts_dataset_from_bytes(&self, bytes: &[u8]) -> Result<Dataset> {
        let values = self.raw_count_values_from_bytes(bytes)?;
        self.counts_dataset_from_values(values)
    }

    fn counts_dataset_from_values(&self, values: Vec<u16>) -> Result<Dataset> {
        let mask = ValidityMask::from_masked_flags(values.iter().map(|value| {
            *value == self.header.calibration.error_pixel_count_value
                || *value == self.header.calibration.outside_scan_pixel_count_value
        }));
        let array = self
            .attach_projection_coordinates(DataArray::<u16>::from_vec_named(
                vec![
                    usize::from(self.header.data.lines),
                    usize::from(self.header.data.columns),
                ],
                ["y", "x"],
                values,
            )?)?
            .with_mask(mask)?;

        let mut dataset = Dataset::new(self.dataset_id()?);
        self.attach_common_attrs(&mut dataset)?;
        self.attach_area_attr(&mut dataset)?;
        dataset.insert_attr("calibration", "counts")?;
        dataset.insert_attr("precision", "native")?;
        dataset.set_array(array);
        Ok(dataset)
    }

    pub fn calibrated_dataset_from_bytes(
        &self,
        bytes: &[u8],
        calibration: AhiCalibration,
    ) -> Result<Dataset> {
        if calibration == AhiCalibration::Counts {
            return self.counts_dataset_from_bytes(bytes);
        }
        let values = self.raw_count_values_from_bytes(bytes)?;
        self.calibrated_dataset_from_values_f32(values, calibration)
    }

    fn calibrated_dataset_from_values_f32(
        &self,
        values: Vec<u16>,
        calibration: AhiCalibration,
    ) -> Result<Dataset> {
        let mask = ValidityMask::from_masked_flags(values.iter().map(|value| {
            *value == self.header.calibration.error_pixel_count_value
                || *value == self.header.calibration.outside_scan_pixel_count_value
        }));
        let calibrated_values = self.calibrate_counts_to_f32(&values, calibration)?;
        let array = self
            .attach_projection_coordinates(DataArray::<f32>::from_vec_named(
                vec![
                    usize::from(self.header.data.lines),
                    usize::from(self.header.data.columns),
                ],
                ["y", "x"],
                calibrated_values,
            )?)?
            .with_mask(mask)?;

        let mut dataset = Dataset::new(self.dataset_id_for_calibration(calibration)?);
        self.attach_common_attrs(&mut dataset)?;
        self.attach_area_attr(&mut dataset)?;
        dataset.insert_attr("calibration", calibration.name())?;
        dataset.insert_attr("precision", "f32")?;
        dataset.insert_attr("calibration_mode", self.calibration_mode.name())?;
        dataset.set_array(array);
        Ok(dataset)
    }

    pub fn calibrated_dataset_from_bytes_f64(
        &self,
        bytes: &[u8],
        calibration: AhiCalibration,
    ) -> Result<Dataset> {
        if calibration == AhiCalibration::Counts {
            return self.counts_dataset_from_bytes(bytes);
        }
        let values = self.raw_count_values_from_bytes(bytes)?;
        self.calibrated_dataset_from_values_f64(values, calibration)
    }

    fn calibrated_dataset_from_values_f64(
        &self,
        values: Vec<u16>,
        calibration: AhiCalibration,
    ) -> Result<Dataset> {
        let mask = ValidityMask::from_masked_flags(values.iter().map(|value| {
            *value == self.header.calibration.error_pixel_count_value
                || *value == self.header.calibration.outside_scan_pixel_count_value
        }));
        let calibrated_values = self.calibrate_counts_to_f64(&values, calibration)?;
        let array = self
            .attach_projection_coordinates(DataArray::<f64>::from_vec_named(
                vec![
                    usize::from(self.header.data.lines),
                    usize::from(self.header.data.columns),
                ],
                ["y", "x"],
                calibrated_values,
            )?)?
            .with_mask(mask)?;

        let mut dataset = Dataset::new(self.dataset_id_for_calibration(calibration)?);
        self.attach_common_attrs(&mut dataset)?;
        self.attach_area_attr(&mut dataset)?;
        dataset.insert_attr("calibration", calibration.name())?;
        dataset.insert_attr("precision", "f64")?;
        dataset.insert_attr("calibration_mode", self.calibration_mode.name())?;
        dataset.set_array(array);
        Ok(dataset)
    }

    pub fn calibrate_counts_to_f32(
        &self,
        counts: &[u16],
        calibration: AhiCalibration,
    ) -> Result<Vec<f32>> {
        if calibration == AhiCalibration::Counts {
            return Ok(counts.iter().map(|value| f32::from(*value)).collect());
        }
        let radiance = self.counts_to_radiance_f32(counts);
        match calibration {
            AhiCalibration::Counts => unreachable!("handled above"),
            AhiCalibration::Radiance => Ok(radiance),
            AhiCalibration::Reflectance => self.radiance_to_reflectance_f32(radiance),
            AhiCalibration::BrightnessTemperature => {
                self.radiance_to_brightness_temperature_f32(radiance)
            }
        }
    }

    pub fn calibrate_counts_to_f64(
        &self,
        counts: &[u16],
        calibration: AhiCalibration,
    ) -> Result<Vec<f64>> {
        if calibration == AhiCalibration::Counts {
            return Ok(counts.iter().map(|value| f64::from(*value)).collect());
        }
        let radiance = self.counts_to_radiance_f64(counts);
        match calibration {
            AhiCalibration::Counts => unreachable!("handled above"),
            AhiCalibration::Radiance => Ok(radiance),
            AhiCalibration::Reflectance => self.radiance_to_reflectance_f64(radiance),
            AhiCalibration::BrightnessTemperature => {
                self.radiance_to_brightness_temperature_f64(radiance)
            }
        }
    }

    pub fn raw_count_values_from_bytes(&self, bytes: &[u8]) -> Result<Vec<u16>> {
        if self.header.data.bits_per_pixel != 16 {
            return Err(RustySatError::unsupported(format!(
                "AHI HSD raw count loading for {} bits per pixel",
                self.header.data.bits_per_pixel
            )));
        }
        let rows = usize::from(self.header.data.lines);
        let cols = usize::from(self.header.data.columns);
        let pixel_count = rows
            .checked_mul(cols)
            .ok_or_else(|| RustySatError::invalid_input("AHI HSD pixel count overflow"))?;
        let data_offset = usize::try_from(self.header.basic.total_header_length).map_err(|_| {
            RustySatError::invalid_input("AHI HSD total header length does not fit in usize")
        })?;
        let byte_count = pixel_count
            .checked_mul(2)
            .ok_or_else(|| RustySatError::invalid_input("AHI HSD data byte count overflow"))?;
        let data = data_block_bytes(
            bytes,
            data_offset,
            byte_count,
            self.header.data.compression_flag,
        )?;
        Ok(data
            .chunks_exact(2)
            .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
            .collect())
    }

    pub fn load_counts_dataset(&self) -> Result<Dataset> {
        let values = self.raw_count_values_from_file()?;
        self.counts_dataset_from_values(values)
    }

    pub fn load_calibrated_dataset(&self, calibration: AhiCalibration) -> Result<Dataset> {
        if calibration == AhiCalibration::Counts {
            return self.load_counts_dataset();
        }
        let values = self.raw_count_values_from_file()?;
        self.calibrated_dataset_from_values_f32(values, calibration)
    }

    pub fn load_calibrated_dataset_f64(&self, calibration: AhiCalibration) -> Result<Dataset> {
        if calibration == AhiCalibration::Counts {
            return self.load_counts_dataset();
        }
        let values = self.raw_count_values_from_file()?;
        self.calibrated_dataset_from_values_f64(values, calibration)
    }

    /// Stream the raw count values straight from the HSD file, skipping the
    /// whole-file decompression buffer that `read_file_bytes` materializes.
    ///
    /// Handles three layouts: plain files (seek to the data block), whole-file
    /// bzip2 (skip the header inside the decoder), and bzip2-compressed data
    /// blocks. Only the `byte_count` data bytes are ever decompressed.
    fn raw_count_values_from_file(&self) -> Result<Vec<u16>> {
        if self.header.data.bits_per_pixel != 16 {
            return Err(RustySatError::unsupported(format!(
                "AHI HSD raw count loading for {} bits per pixel",
                self.header.data.bits_per_pixel
            )));
        }
        let rows = usize::from(self.header.data.lines);
        let cols = usize::from(self.header.data.columns);
        let pixel_count = rows
            .checked_mul(cols)
            .ok_or_else(|| RustySatError::invalid_input("AHI HSD pixel count overflow"))?;
        let data_offset = usize::try_from(self.header.basic.total_header_length).map_err(|_| {
            RustySatError::invalid_input("AHI HSD total header length does not fit in usize")
        })?;
        let byte_count = pixel_count
            .checked_mul(2)
            .ok_or_else(|| RustySatError::invalid_input("AHI HSD data byte count overflow"))?;

        if data_offset > MAX_HSD_FILE_BYTES as usize {
            return Err(RustySatError::invalid_input(format!(
                "AHI HSD header length {data_offset} exceeds the current safety limit of {MAX_HSD_FILE_BYTES} bytes"
            )));
        }
        let file = File::open(&self.filename).map_err(|err| {
            RustySatError::invalid_input(format!(
                "failed to open AHI HSD file '{}': {err}",
                self.filename.display()
            ))
        })?;
        if file_has_bzip2_magic(&self.filename)? {
            let mut decoder = MultiBzDecoder::new(BufReader::new(file));
            let mut sink = std::io::sink();
            std::io::copy(&mut decoder.by_ref().take(data_offset as u64), &mut sink).map_err(
                |err| {
                    RustySatError::invalid_input(format!(
                        "failed to skip AHI HSD header in '{}': {err}",
                        self.filename.display()
                    ))
                },
            )?;
            return self
                .raw_count_values_from_reader(&mut decoder.by_ref().take(byte_count as u64 + 1));
        }
        let mut file = BufReader::new(file);
        use std::io::Seek;
        file.seek(std::io::SeekFrom::Start(data_offset as u64))
            .map_err(|err| {
                RustySatError::invalid_input(format!(
                    "failed to seek AHI HSD data block in '{}': {err}",
                    self.filename.display()
                ))
            })?;
        match self.header.data.compression_flag {
            0 => self.raw_count_values_from_reader(&mut file.take(byte_count as u64 + 1)),
            2 => {
                let mut decoder = MultiBzDecoder::new(file);
                self.raw_count_values_from_reader(&mut decoder.by_ref().take(byte_count as u64 + 1))
            }
            other => Err(RustySatError::unsupported(format!(
                "AHI HSD data compression flag {other}"
            ))),
        }
    }

    /// Read exactly `byte_count` bytes from `reader` as little-endian u16
    /// samples, converting pairs directly (no intermediate decompressed
    /// `Vec<u8>`). Errors on truncation and on extra trailing data.
    fn raw_count_values_from_reader(&self, reader: &mut impl Read) -> Result<Vec<u16>> {
        let rows = usize::from(self.header.data.lines);
        let cols = usize::from(self.header.data.columns);
        let pixel_count = rows
            .checked_mul(cols)
            .ok_or_else(|| RustySatError::invalid_input("AHI HSD pixel count overflow"))?;
        let byte_count = pixel_count
            .checked_mul(2)
            .ok_or_else(|| RustySatError::invalid_input("AHI HSD data byte count overflow"))?;

        let mut values = Vec::with_capacity(pixel_count);
        let mut buffer = [0u8; 8192];
        let mut carry: Option<u8> = None;
        let mut remaining = byte_count;
        while remaining > 0 {
            let want = buffer.len().min(remaining);
            let read = reader.read(&mut buffer[..want]).map_err(|err| {
                RustySatError::invalid_input(format!("AHI HSD data read failed: {err}"))
            })?;
            if read == 0 {
                return Err(RustySatError::invalid_input(format!(
                    "AHI HSD data block is truncated: need {remaining} more bytes"
                )));
            }
            let mut idx = 0;
            if let Some(first) = carry.take() {
                if idx < read {
                    values.push(u16::from_le_bytes([first, buffer[idx]]));
                    idx += 1;
                } else {
                    carry = Some(first);
                }
            }
            while idx + 1 < read {
                values.push(u16::from_le_bytes([buffer[idx], buffer[idx + 1]]));
                idx += 2;
            }
            if idx < read {
                carry = Some(buffer[idx]);
            }
            remaining -= read;
        }
        // Probe for extra decompressed data beyond the expected block size.
        let mut probe = [0u8; 1];
        let extra = reader.read(&mut probe).map_err(|err| {
            RustySatError::invalid_input(format!("AHI HSD data read failed: {err}"))
        })?;
        if extra > 0 {
            return Err(RustySatError::invalid_input(format!(
                "AHI HSD data block decompressed to more than the expected {byte_count} bytes"
            )));
        }
        Ok(values)
    }

    pub fn dataset_id_for_calibration(&self, calibration: AhiCalibration) -> Result<DataId> {
        let wavelength = self.header.calibration.central_wavelength;
        DataId::new(self.band_name())?
            .with_qualifier(
                "wavelength",
                WavelengthRange::micrometers(wavelength, wavelength, wavelength)?,
            )?
            .with_qualifier("calibration", calibration.name())
    }

    fn counts_to_radiance_f32(&self, counts: &[u16]) -> Vec<f32> {
        let (gain, offset) = self.radiance_gain_offset();
        let mut radiance = counts
            .iter()
            .map(|value| f32::from(*value) * gain as f32 + offset as f32)
            .collect::<Vec<_>>();
        if let Some(user) = &self.user_calibration {
            if user.correction_type() == AhiUserCalibrationType::RadianceCorrection {
                let coeffs = user.coefficients_for(&self.band_name());
                for value in &mut radiance {
                    *value = (*value - coeffs.offset as f32) / coeffs.slope as f32;
                }
            }
        }
        radiance
    }

    fn counts_to_radiance_f64(&self, counts: &[u16]) -> Vec<f64> {
        let (gain, offset) = self.radiance_gain_offset();
        let mut radiance = counts
            .iter()
            .map(|value| f64::from(*value) * gain + offset)
            .collect::<Vec<_>>();
        if let Some(user) = &self.user_calibration {
            if user.correction_type() == AhiUserCalibrationType::RadianceCorrection {
                let coeffs = user.coefficients_for(&self.band_name());
                for value in &mut radiance {
                    *value = (*value - coeffs.offset) / coeffs.slope;
                }
            }
        }
        radiance
    }

    fn radiance_gain_offset(&self) -> (f64, f64) {
        if let Some(user) = &self.user_calibration {
            if user.correction_type() == AhiUserCalibrationType::DigitalNumber {
                let coeffs = user.coefficients_for(&self.band_name());
                return (coeffs.slope, coeffs.offset);
            }
        }
        if self.calibration_mode == AhiCalibrationMode::Update
            && self.header.calibration.band_number < 7
        {
            if let Some(AhiBandCalibration::Visible {
                calibrated_gain_count_to_radiance,
                calibrated_offset_count_to_radiance,
                ..
            }) = &self.header.calibration.band_calibration
            {
                if *calibrated_gain_count_to_radiance != 0.0
                    || *calibrated_offset_count_to_radiance != 0.0
                {
                    return (
                        *calibrated_gain_count_to_radiance,
                        *calibrated_offset_count_to_radiance,
                    );
                }
            }
        }
        (
            self.header.calibration.gain_count_to_radiance,
            self.header.calibration.offset_count_to_radiance,
        )
    }

    fn radiance_to_reflectance_f32(&self, mut radiance: Vec<f32>) -> Result<Vec<f32>> {
        let coeff = self.required_visible_albedo_coeff()? as f32;
        for value in radiance.iter_mut() {
            *value = (*value * coeff * 100.0).max(0.0);
        }
        Ok(radiance)
    }

    fn radiance_to_reflectance_f64(&self, mut radiance: Vec<f64>) -> Result<Vec<f64>> {
        let coeff = self.required_visible_albedo_coeff()?;
        for value in radiance.iter_mut() {
            *value = (*value * coeff * 100.0).max(0.0);
        }
        Ok(radiance)
    }

    fn radiance_to_brightness_temperature_f32(&self, mut radiance: Vec<f32>) -> Result<Vec<f32>> {
        // Compute in place on a single f32 buffer. Each value is promoted to
        // f64 only for the Planck inversion (matching the f64 path) and written
        // straight back as f32, avoiding the previous f32->f64->f64->f32 chain
        // of full-array allocations.
        let (
            c0_rad_to_tb,
            c1_rad_to_tb,
            c2_rad_to_tb,
            speed_of_light,
            planck_constant,
            boltzmann_constant,
        ) = self.required_infrared_coefficients()?;
        let cwl = self.header.calibration.central_wavelength * 1.0e-6;
        let a = (planck_constant * speed_of_light) / (boltzmann_constant * cwl);
        let b_const = (2.0 * planck_constant * speed_of_light.powi(2)) / (1.0e6 * cwl.powi(5));
        for value in radiance.iter_mut() {
            let rad = f64::from(*value);
            *value = if rad == 0.0 {
                f32::NAN
            } else {
                let b = b_const / rad + 1.0;
                let te = a / b.ln();
                ((c0_rad_to_tb + c1_rad_to_tb * te + c2_rad_to_tb * te.powi(2)).max(0.0)) as f32
            };
        }
        Ok(radiance)
    }

    fn radiance_to_brightness_temperature_f64(&self, radiance: Vec<f64>) -> Result<Vec<f64>> {
        let (
            c0_rad_to_tb,
            c1_rad_to_tb,
            c2_rad_to_tb,
            speed_of_light,
            planck_constant,
            boltzmann_constant,
        ) = self.required_infrared_coefficients()?;
        let cwl = self.header.calibration.central_wavelength * 1.0e-6;
        let a = (planck_constant * speed_of_light) / (boltzmann_constant * cwl);
        let b_const = (2.0 * planck_constant * speed_of_light.powi(2)) / (1.0e6 * cwl.powi(5));
        Ok(radiance
            .into_iter()
            .map(|value| {
                if value == 0.0 {
                    return f64::NAN;
                }
                let b = b_const / value + 1.0;
                let te = a / b.ln();
                (c0_rad_to_tb + c1_rad_to_tb * te + c2_rad_to_tb * te.powi(2)).max(0.0)
            })
            .collect())
    }

    fn required_visible_albedo_coeff(&self) -> Result<f64> {
        match &self.header.calibration.band_calibration {
            Some(AhiBandCalibration::Visible {
                coeff_rad_to_albedo,
                ..
            }) => Ok(*coeff_rad_to_albedo),
            _ => Err(RustySatError::unsupported(format!(
                "AHI HSD reflectance calibration for band {} without visible block-5 coefficients",
                self.header.calibration.band_number
            ))),
        }
    }

    fn required_infrared_coefficients(&self) -> Result<(f64, f64, f64, f64, f64, f64)> {
        match &self.header.calibration.band_calibration {
            Some(AhiBandCalibration::Infrared {
                c0_rad_to_tb,
                c1_rad_to_tb,
                c2_rad_to_tb,
                speed_of_light,
                planck_constant,
                boltzmann_constant,
                ..
            }) => Ok((
                *c0_rad_to_tb,
                *c1_rad_to_tb,
                *c2_rad_to_tb,
                *speed_of_light,
                *planck_constant,
                *boltzmann_constant,
            )),
            _ => Err(RustySatError::unsupported(format!(
                "AHI HSD brightness-temperature calibration for band {} without infrared block-5 coefficients",
                self.header.calibration.band_number
            ))),
        }
    }

    fn attach_common_attrs(&self, dataset: &mut Dataset) -> Result<()> {
        dataset.insert_attr("reader", "ahi_hsd")?;
        dataset.insert_attr("file_type", self.file_type.clone())?;
        dataset.insert_attr("filename", self.filename.to_string_lossy().to_string())?;
        dataset.insert_attr(
            "platform_name",
            MetadataValue::string(self.header.basic.satellite.clone()),
        )?;
        dataset.insert_attr("sensor", "ahi")?;
        dataset.insert_attr("band_name", self.band_name())?;
        dataset.insert_attr("segment_number", i64::from(self.segment.segment_number))?;
        dataset.insert_attr("total_segments", i64::from(self.segment.total_segments))?;
        if let Some(segment) = &self.header.segment {
            dataset.insert_attr("first_line_number", i64::from(segment.first_line_number))?;
        }
        dataset.insert_attr("columns", i64::from(self.header.data.columns))?;
        dataset.insert_attr("lines", i64::from(self.header.data.lines))?;
        dataset.insert_attr("bits_per_pixel", i64::from(self.header.data.bits_per_pixel))?;
        dataset.insert_attr(
            "central_wavelength",
            MetadataValue::float(self.header.calibration.central_wavelength)?,
        )?;
        Ok(())
    }

    pub fn area_metadata_value(&self) -> Result<MetadataValue> {
        ahi_area_metadata_value(
            self.area_id(),
            self.area_description(),
            self.proj_id(),
            self.geos_projection(),
            usize::from(self.header.data.lines),
            usize::from(self.header.data.columns),
            self.area_extent_for_segment(),
        )
    }

    fn attach_area_attr(&self, dataset: &mut Dataset) -> Result<()> {
        dataset.insert_attr("area", self.area_metadata_value()?)
    }

    fn attach_projection_coordinates<T: NumericElement>(
        &self,
        array: DataArray<T>,
    ) -> Result<DataArray<T>> {
        attach_projection_coordinates_to_array(array, self.area_extent_for_segment())
    }

    fn geos_projection(&self) -> BTreeMap<String, String> {
        let a = self.header.projection.earth_equatorial_radius * 1000.0;
        let b = self.header.projection.earth_polar_radius * 1000.0;
        let h = self.header.projection.distance_from_earth_center * 1000.0 - a;
        BTreeMap::from([
            ("a".to_string(), a.to_string()),
            ("b".to_string(), b.to_string()),
            ("h".to_string(), h.to_string()),
            (
                "lon_0".to_string(),
                self.header.projection.sub_lon.to_string(),
            ),
            ("proj".to_string(), "geos".to_string()),
            ("units".to_string(), "m".to_string()),
        ])
    }

    fn area_id(&self) -> String {
        self.header.basic.observation_area.clone()
    }

    fn area_description(&self) -> String {
        format!("AHI {} area", self.header.basic.observation_area)
    }

    fn proj_id(&self) -> String {
        let suffix = self
            .header
            .basic
            .satellite
            .chars()
            .rev()
            .find(|ch| ch.is_ascii_digit())
            .unwrap_or('x');
        format!("geosh{suffix}")
    }

    fn area_extent_for_segment(&self) -> [f64; 4] {
        let lines = usize::from(self.header.data.lines);
        let segment_offset = usize::from(self.segment.segment_number);
        self.area_extent_for(lines, segment_offset)
    }

    fn area_extent_for(&self, lines: usize, segment_offset: usize) -> [f64; 4] {
        let h = self.header.projection.distance_from_earth_center * 1000.0
            - self.header.projection.earth_equatorial_radius * 1000.0;
        // Satpy convention: loff = -LOFF + 1 + segment_number * nlines.
        // Reference: `satpy/satpy/readers/ahi_hsd.py` lines ~466-480.
        //
        // For single-segment calls, `segment_offset` is the 1-based segment
        // number and `lines` is the per-segment line count.
        // For full-disk assembly, `segment_offset` is 1 and `lines` is the
        // total height — we use `lines` as `nlines` so the offset positions
        // the full-disk grid correctly centered on the sub-satellite point.
        let nlines = if segment_offset > 0 && lines > usize::from(self.header.data.lines) {
            lines
        } else {
            usize::from(self.header.data.lines)
        };
        let loff =
            -f64::from(self.header.projection.loff) + 1.0 + segment_offset as f64 * nlines as f64;
        geos_area_extent(
            lines,
            usize::from(self.header.data.columns),
            self.header.projection.cfac,
            self.header.projection.lfac,
            f64::from(self.header.projection.coff),
            loff,
            h,
        )
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct AhiHsdReader {
    name: String,
    handlers: Vec<AhiHsdFileHandler>,
    calibration: AhiCalibration,
    output: AhiCalibrationOutput,
    parallel_segments: usize,
}

impl AhiHsdReader {
    pub fn new(handlers: impl IntoIterator<Item = AhiHsdFileHandler>) -> Result<Self> {
        Self::with_name_and_calibration("ahi_hsd", AhiCalibration::Counts, handlers)
    }

    pub fn with_name(
        name: impl Into<String>,
        handlers: impl IntoIterator<Item = AhiHsdFileHandler>,
    ) -> Result<Self> {
        Self::with_name_and_calibration(name, AhiCalibration::Counts, handlers)
    }

    pub fn with_calibration(
        calibration: AhiCalibration,
        handlers: impl IntoIterator<Item = AhiHsdFileHandler>,
    ) -> Result<Self> {
        Self::with_name_and_calibration("ahi_hsd", calibration, handlers)
    }

    pub fn with_name_and_calibration(
        name: impl Into<String>,
        calibration: AhiCalibration,
        handlers: impl IntoIterator<Item = AhiHsdFileHandler>,
    ) -> Result<Self> {
        let name = name.into();
        if name.trim().is_empty() {
            return Err(RustySatError::invalid_input(
                "AHI HSD reader name cannot be empty",
            ));
        }
        Ok(Self {
            name,
            handlers: handlers.into_iter().collect(),
            calibration,
            output: AhiCalibrationOutput::DisplayF32,
            parallel_segments: CONCURRENT_SEGMENT_LOADS,
        })
    }

    /// Set how many HSD segments are loaded/calibrated concurrently during
    /// full-disk assembly.
    ///
    /// Higher values use more CPU (bzip2 decompression is single-threaded per
    /// segment) at the cost of peak memory: `assembled buffer + N segment
    /// buffers` coexist. The default is [`CONCURRENT_SEGMENT_LOADS`].
    pub fn with_parallel_segments(mut self, parallel_segments: usize) -> Result<Self> {
        if parallel_segments == 0 {
            return Err(RustySatError::invalid_input(
                "AHI HSD parallel segment count must be at least 1",
            ));
        }
        self.parallel_segments = parallel_segments;
        Ok(self)
    }

    pub fn handlers(&self) -> &[AhiHsdFileHandler] {
        &self.handlers
    }

    pub fn calibration(&self) -> AhiCalibration {
        self.calibration
    }

    pub fn output(&self) -> AhiCalibrationOutput {
        self.output
    }

    pub fn with_output(mut self, output: AhiCalibrationOutput) -> Self {
        self.output = output;
        self
    }

    pub fn inventory(&self) -> Result<ReaderInventory> {
        ReaderInventory::new(self.name.clone(), self.available_dataset_ids())
    }
}

impl Reader for AhiHsdReader {
    fn name(&self) -> &str {
        &self.name
    }

    fn available_dataset_ids(&self) -> Vec<DataId> {
        let mut ids = Vec::new();
        for handler in &self.handlers {
            let Ok(id) = handler.dataset_id_for_calibration(self.calibration) else {
                continue;
            };
            if !ids.contains(&id) {
                ids.push(id);
            }
        }
        ids
    }

    fn load(&self, id: &DataId) -> Result<Dataset> {
        let mut matching_handlers = Vec::new();
        for handler in &self.handlers {
            if handler.dataset_id_for_calibration(self.calibration)? == *id {
                matching_handlers.push(handler);
            }
        }
        match matching_handlers.as_slice() {
            [] => Err(RustySatError::not_found(format!(
                "AHI HSD dataset '{}'",
                id.name()
            ))),
            [handler] => self.load_handler_dataset(handler),
            _ => self.load_assembled_dataset(matching_handlers),
        }
    }
}

impl AhiHsdReader {
    fn load_handler_dataset(&self, handler: &AhiHsdFileHandler) -> Result<Dataset> {
        match self.output {
            AhiCalibrationOutput::DisplayF32 => handler.load_calibrated_dataset(self.calibration),
            AhiCalibrationOutput::ScientificF64 => {
                handler.load_calibrated_dataset_f64(self.calibration)
            }
        }
    }

    fn load_assembled_dataset(&self, handlers: Vec<&AhiHsdFileHandler>) -> Result<Dataset> {
        let sorted_handlers = sorted_complete_segment_handlers(handlers)?;
        let Some(first_handler) = sorted_handlers.first() else {
            return Err(RustySatError::invalid_input(
                "AHI HSD segment assembly requires at least one segment",
            ));
        };

        let first_dataset = self.load_handler_dataset(first_handler)?;
        let mut output = Dataset::new(first_dataset.id().clone());

        let first_array = first_dataset.into_array().ok_or_else(|| {
            RustySatError::invalid_input("AHI HSD segment dataset is missing array data")
        })?;

        let width = first_array.shape()[1];
        let mut total_height = first_array.shape()[0];
        for handler in sorted_handlers.iter().skip(1) {
            total_height += usize::from(handler.header.data.lines);
        }
        let total_len = total_height * width;

        macro_rules! assemble_variant {
            ($first_typed:expr, $variant:ident, $ty:ty) => {{
                let (first_values, _, mut acc_mask) = $first_typed.into_parts();
                let mut values = Vec::with_capacity(total_len);
                values.extend(first_values);
                for chunk in sorted_handlers[1..].chunks(self.parallel_segments) {
                    let chunk_datasets: Vec<Dataset> = chunk
                        .par_iter()
                        .map(|handler| self.load_handler_dataset(handler))
                        .collect::<Result<Vec<_>>>()?;
                    for ds in chunk_datasets {
                        let arr = ds.into_array().ok_or_else(|| {
                            RustySatError::invalid_input(
                                "AHI HSD segment dataset is missing array data",
                            )
                        })?;
                        let AnyDataArray::$variant(typed_arr) = arr else {
                            return Err(RustySatError::invalid_input(
                                "AHI HSD segment dtype mismatch",
                            ));
                        };
                        let (segment_values, _, segment_mask) = typed_arr.into_parts();
                        let segment_len = segment_values.len();
                        match acc_mask.as_mut() {
                            Some(mask) => mask.extend(segment_mask.as_ref(), segment_len),
                            None if segment_mask.is_some() => {
                                let mut mask = ValidityMask::all_valid(values.len());
                                mask.extend(segment_mask.as_ref(), segment_len);
                                acc_mask = Some(mask);
                            }
                            None => {}
                        }
                        values.extend(segment_values);
                    }
                }
                let mut array =
                    DataArray::<$ty>::from_vec_named([total_height, width], ["y", "x"], values)?;
                if let Some(mask) = acc_mask {
                    array.set_mask(mask)?;
                }
                AnyDataArray::$variant(array)
            }};
        }

        let assembled_array = match first_array {
            AnyDataArray::F32(first_typed) => assemble_variant!(first_typed, F32, f32),
            AnyDataArray::F64(first_typed) => assemble_variant!(first_typed, F64, f64),
            AnyDataArray::U16(first_typed) => assemble_variant!(first_typed, U16, u16),
            AnyDataArray::U8(first_typed) => assemble_variant!(first_typed, U8, u8),
            AnyDataArray::I16(first_typed) => assemble_variant!(first_typed, I16, i16),
        };

        let array = attach_projection_coordinates_to_any_array(
            assembled_array,
            first_handler.area_extent_for(total_height, 1),
        )?;

        first_handler.attach_common_attrs(&mut output)?;
        output.insert_attr(
            "area",
            ahi_area_metadata_value(
                first_handler.area_id(),
                first_handler.area_description(),
                first_handler.proj_id(),
                first_handler.geos_projection(),
                array.shape()[0],
                array.shape()[1],
                first_handler.area_extent_for(array.shape()[0], 1),
            )?,
        )?;
        output.insert_attr("calibration", self.calibration.name())?;
        output.insert_attr(
            "lines",
            i64::try_from(array.shape()[0]).map_err(|_| {
                RustySatError::invalid_input("assembled AHI HSD line count does not fit in i64")
            })?,
        )?;
        output.insert_attr(
            "assembled_segments",
            MetadataValue::List(
                sorted_handlers
                    .iter()
                    .map(|h| MetadataValue::Integer(i64::from(h.segment.segment_number)))
                    .collect(),
            ),
        )?;
        output.set_array(array);
        Ok(output)
    }
}

fn sorted_complete_segment_handlers(
    handlers: Vec<&AhiHsdFileHandler>,
) -> Result<Vec<&AhiHsdFileHandler>> {
    let Some(first) = handlers.first() else {
        return Err(RustySatError::invalid_input(
            "AHI HSD segment assembly requires at least one segment",
        ));
    };
    let expected_total = usize::from(first.segment.total_segments);
    let mut slots = vec![None; expected_total];
    for handler in handlers.iter().copied() {
        validate_segment_header_matches_filename(handler)?;
        if handler.segment.total_segments != first.segment.total_segments {
            return Err(RustySatError::invalid_input(format!(
                "AHI HSD segment total mismatch: expected {}, got {} for '{}'",
                first.segment.total_segments,
                handler.segment.total_segments,
                handler.filename().display()
            )));
        }
        validate_assembly_compatible(first, handler)?;
        let idx = usize::from(handler.segment.segment_number - 1);
        if slots[idx].is_some() {
            return Err(RustySatError::invalid_input(format!(
                "duplicate AHI HSD segment {} for dataset '{}'",
                handler.segment.segment_number,
                handler.band_name()
            )));
        }
        slots[idx] = Some(handler);
    }
    let missing = slots
        .iter()
        .enumerate()
        .filter_map(|(idx, handler)| handler.is_none().then_some(idx + 1))
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        return Err(RustySatError::invalid_input(format!(
            "missing AHI HSD segment(s) {:?} for complete {}-segment assembly",
            missing, expected_total
        )));
    }
    Ok(slots.into_iter().map(Option::unwrap).collect())
}

fn validate_segment_header_matches_filename(handler: &AhiHsdFileHandler) -> Result<()> {
    let Some(segment) = &handler.header.segment else {
        return Ok(());
    };
    if segment.total_segments != handler.segment.total_segments {
        return Err(RustySatError::invalid_input(format!(
            "AHI HSD block-7 total segments {} does not match filename total {} for '{}'",
            segment.total_segments,
            handler.segment.total_segments,
            handler.filename().display()
        )));
    }
    if segment.segment_sequence_number != handler.segment.segment_number {
        return Err(RustySatError::invalid_input(format!(
            "AHI HSD block-7 segment sequence {} does not match filename segment {} for '{}'",
            segment.segment_sequence_number,
            handler.segment.segment_number,
            handler.filename().display()
        )));
    }
    Ok(())
}

fn validate_assembly_compatible(
    first: &AhiHsdFileHandler,
    handler: &AhiHsdFileHandler,
) -> Result<()> {
    if first.file_type != handler.file_type {
        return Err(RustySatError::invalid_input(format!(
            "AHI HSD segment file type mismatch: '{}' vs '{}'",
            first.file_type, handler.file_type
        )));
    }
    if first.header.calibration.band_number != handler.header.calibration.band_number {
        return Err(RustySatError::invalid_input(format!(
            "AHI HSD segment band mismatch: {} vs {}",
            first.header.calibration.band_number, handler.header.calibration.band_number
        )));
    }
    if first.header.data.columns != handler.header.data.columns {
        return Err(RustySatError::invalid_input(format!(
            "AHI HSD segment column mismatch: {} vs {}",
            first.header.data.columns, handler.header.data.columns
        )));
    }
    if first.header.data.bits_per_pixel != handler.header.data.bits_per_pixel {
        return Err(RustySatError::invalid_input(format!(
            "AHI HSD segment bits-per-pixel mismatch: {} vs {}",
            first.header.data.bits_per_pixel, handler.header.data.bits_per_pixel
        )));
    }
    if first.header.calibration.central_wavelength != handler.header.calibration.central_wavelength
    {
        return Err(RustySatError::invalid_input(
            "AHI HSD segment wavelength mismatch",
        ));
    }
    Ok(())
}

fn ahi_area_metadata_value(
    id: String,
    description: String,
    proj_id: String,
    projection: BTreeMap<String, String>,
    height: usize,
    width: usize,
    area_extent: [f64; 4],
) -> Result<MetadataValue> {
    Ok(MetadataValue::map([
        ("type", MetadataValue::string("area")),
        ("id", MetadataValue::string(id)),
        ("description", MetadataValue::string(description)),
        ("proj_id", MetadataValue::string(proj_id)),
        (
            "projection",
            MetadataValue::Map(
                projection
                    .into_iter()
                    .map(|(key, value)| (key, MetadataValue::string(value)))
                    .collect(),
            ),
        ),
        ("height", MetadataValue::Integer(usize_to_i64(height)?)),
        ("width", MetadataValue::Integer(usize_to_i64(width)?)),
        (
            "area_extent",
            MetadataValue::List(
                area_extent
                    .into_iter()
                    .map(MetadataValue::float)
                    .collect::<Result<Vec<_>>>()?,
            ),
        ),
    ]))
}

fn usize_to_i64(value: usize) -> Result<i64> {
    i64::try_from(value).map_err(|_| RustySatError::invalid_input("value does not fit in i64"))
}

fn geos_area_extent(
    lines: usize,
    columns: usize,
    cfac: u32,
    lfac: u32,
    coff: f64,
    loff: f64,
    h: f64,
) -> [f64; 4] {
    // Reference: `satpy/satpy/readers/core/_geos_area.py` — `get_area_extent`.
    // For N2S scan (AHI standard), line 0.5 is at the top (north) and
    // line lines+0.5 is at the bottom (south).  The scanning angle y
    // decreases from top to bottom when loff > 0.
    // We compute LL and UR scanning angles, then ensure ll_y < ur_y
    // by swapping if needed (matching Satpy's area_extent convention).
    let ll = geos_xy_from_line_col(0.5, 0.5, loff, coff, lfac, cfac);
    let ur = geos_xy_from_line_col(
        lines as f64 + 0.5,
        columns as f64 + 0.5,
        loff,
        coff,
        lfac,
        cfac,
    );
    let ll_x = ll.0.to_radians() * h;
    let ll_y = ll.1.to_radians() * h;
    let ur_x = ur.0.to_radians() * h;
    let ur_y = ur.1.to_radians() * h;
    // Ensure ll_y < ur_y (Satpy convention: lower-left y < upper-right y).
    if ll_y < ur_y {
        [ll_x, ll_y, ur_x, ur_y]
    } else {
        [ll_x, ur_y, ur_x, ll_y]
    }
}

fn geos_xy_from_line_col(
    line: f64,
    col: f64,
    loff: f64,
    coff: f64,
    lfac: u32,
    cfac: u32,
) -> (f64, f64) {
    let x = (col - coff) / (f64::from(cfac) / 2_f64.powi(16));
    let y = (line - loff) / (f64::from(lfac) / 2_f64.powi(16));
    (x, y)
}

fn attach_projection_coordinates_to_any_array(
    array: AnyDataArray,
    area_extent: [f64; 4],
) -> Result<AnyDataArray> {
    Ok(match array {
        AnyDataArray::F32(array) => {
            attach_projection_coordinates_to_array(array, area_extent)?.into()
        }
        AnyDataArray::F64(array) => {
            attach_projection_coordinates_to_array(array, area_extent)?.into()
        }
        AnyDataArray::U8(array) => {
            attach_projection_coordinates_to_array(array, area_extent)?.into()
        }
        AnyDataArray::U16(array) => {
            attach_projection_coordinates_to_array(array, area_extent)?.into()
        }
        AnyDataArray::I16(array) => {
            attach_projection_coordinates_to_array(array, area_extent)?.into()
        }
    })
}

fn attach_projection_coordinates_to_array<T: NumericElement>(
    array: DataArray<T>,
    area_extent: [f64; 4],
) -> Result<DataArray<T>> {
    let (height, width) = array.shape_yx()?;
    let x_coords = projection_x_coords(width, area_extent);
    let y_coords = projection_y_coords(height, area_extent);
    array
        .with_coordinate("x", Coordinate::axis("x", x_coords)?)?
        .with_coordinate("y", Coordinate::axis("y", y_coords)?)
}

fn projection_x_coords(width: usize, area_extent: [f64; 4]) -> Vec<f64> {
    let pixel_size = (area_extent[2] - area_extent[0]) / width as f64;
    (0..width)
        .map(|x| area_extent[0] + (x as f64 + 0.5) * pixel_size)
        .collect()
}

fn projection_y_coords(height: usize, area_extent: [f64; 4]) -> Vec<f64> {
    let pixel_size = (area_extent[3] - area_extent[1]) / height as f64;
    (0..height)
        .map(|y| area_extent[3] - (y as f64 + 0.5) * pixel_size)
        .collect()
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
    let projection_offset = checked_block_offset(data_offset, data.block_length, "projection")?;
    let projection = AhiProjectionInfo::parse(take_block(
        bytes,
        projection_offset,
        PROJECTION_INFO_LEN,
        "projection information",
    )?)?;
    let navigation_offset =
        checked_block_offset(projection_offset, projection.block_length, "navigation")?;
    let navigation = AhiNavigationInfo::parse(take_block(
        bytes,
        navigation_offset,
        NAVIGATION_INFO_LEN,
        "navigation information",
    )?)?;
    let calibration_offset =
        checked_block_offset(navigation_offset, navigation.block_length, "calibration")?;
    let calibration_prefix = take_block(
        bytes,
        calibration_offset,
        CALIBRATION_INFO_LEN,
        "calibration information",
    )?;
    let calibration_block_len = usize::from(read_u16_le(
        calibration_prefix,
        1,
        "calibration blocklength",
    )?);
    let calibration = AhiCalibrationInfo::parse(take_block(
        bytes,
        calibration_offset,
        calibration_block_len.max(CALIBRATION_INFO_LEN),
        "calibration information",
    )?)?;
    let segment = parse_optional_segment_info(
        bytes,
        checked_block_offset(calibration_offset, calibration.block_length, "segment")?,
    )?;

    Ok(AhiHsdHeader {
        basic,
        data,
        projection,
        navigation,
        calibration,
        segment,
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
        let band_number = read_u16_le(bytes, 3, "calibration band number")?;
        Ok(Self {
            header_block_number: read_u8(bytes, 0, "calibration hblock_number")?,
            block_length: read_u16_le(bytes, 1, "calibration blocklength")?,
            band_number,
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
            band_calibration: parse_band_calibration(bytes, band_number)?,
        })
    }
}

impl AhiCalibration {
    pub fn name(self) -> &'static str {
        match self {
            Self::Counts => "counts",
            Self::Radiance => "radiance",
            Self::Reflectance => "reflectance",
            Self::BrightnessTemperature => "brightness_temperature",
        }
    }
}

impl AhiCalibrationMode {
    pub fn name(self) -> &'static str {
        match self {
            Self::Nominal => "nominal",
            Self::Update => "update",
        }
    }
}

impl AhiCalibrationOutput {
    pub fn name(self) -> &'static str {
        match self {
            Self::DisplayF32 => "f32",
            Self::ScientificF64 => "f64",
        }
    }
}

fn parse_band_calibration(bytes: &[u8], band_number: u16) -> Result<Option<AhiBandCalibration>> {
    let expected_len = if band_number <= 6 {
        VISIBLE_CALIBRATION_INFO_LEN
    } else {
        INFRARED_CALIBRATION_INFO_LEN
    };
    if bytes.len() < expected_len {
        return Ok(None);
    }
    if band_number <= 6 {
        Ok(Some(AhiBandCalibration::Visible {
            coeff_rad_to_albedo: read_f64_le(bytes, 35, "visible coeff rad to albedo")?,
            coeff_update_time_days: read_f64_le(bytes, 43, "visible coeff update time")?,
            calibrated_gain_count_to_radiance: read_f64_le(bytes, 51, "visible calibrated gain")?,
            calibrated_offset_count_to_radiance: read_f64_le(
                bytes,
                59,
                "visible calibrated offset",
            )?,
        }))
    } else {
        Ok(Some(AhiBandCalibration::Infrared {
            c0_rad_to_tb: read_f64_le(bytes, 35, "infrared c0 rad to tb")?,
            c1_rad_to_tb: read_f64_le(bytes, 43, "infrared c1 rad to tb")?,
            c2_rad_to_tb: read_f64_le(bytes, 51, "infrared c2 rad to tb")?,
            c0_tb_to_rad: read_f64_le(bytes, 59, "infrared c0 tb to rad")?,
            c1_tb_to_rad: read_f64_le(bytes, 67, "infrared c1 tb to rad")?,
            c2_tb_to_rad: read_f64_le(bytes, 75, "infrared c2 tb to rad")?,
            speed_of_light: read_f64_le(bytes, 83, "infrared speed of light")?,
            planck_constant: read_f64_le(bytes, 91, "infrared planck constant")?,
            boltzmann_constant: read_f64_le(bytes, 99, "infrared boltzmann constant")?,
        }))
    }
}

impl AhiSegmentBlockInfo {
    fn parse(bytes: &[u8]) -> Result<Self> {
        Ok(Self {
            header_block_number: read_u8(bytes, 0, "segment hblock_number")?,
            block_length: read_u16_le(bytes, 1, "segment blocklength")?,
            total_segments: read_u8(bytes, 3, "segment total segments")?,
            segment_sequence_number: read_u8(bytes, 4, "segment sequence number")?,
            first_line_number: read_u16_le(bytes, 5, "segment first line number")?,
        })
    }
}

fn parse_optional_segment_info(
    bytes: &[u8],
    block6_offset: usize,
) -> Result<Option<AhiSegmentBlockInfo>> {
    let Some(block6_prefix_end) = block6_offset.checked_add(3) else {
        return Ok(None);
    };
    let Some(block6_prefix) = bytes.get(block6_offset..block6_prefix_end) else {
        return Ok(None);
    };
    let block6_length = usize::from(read_u16_le(
        block6_prefix,
        1,
        "inter-calibration blocklength",
    )?);
    let Some(block7_offset) = block6_offset.checked_add(block6_length) else {
        return Ok(None);
    };
    let Some(block7_prefix_end) = block7_offset.checked_add(3) else {
        return Ok(None);
    };
    let Some(block7_prefix) = bytes.get(block7_offset..block7_prefix_end) else {
        return Ok(None);
    };
    let block7_length = usize::from(read_u16_le(block7_prefix, 1, "segment blocklength")?);
    Ok(Some(AhiSegmentBlockInfo::parse(take_block(
        bytes,
        block7_offset,
        block7_length.max(7),
        "segment information",
    )?)?))
}

fn take_block<'a>(bytes: &'a [u8], offset: usize, min_len: usize, name: &str) -> Result<&'a [u8]> {
    let end = offset.checked_add(min_len).ok_or_else(|| {
        RustySatError::invalid_input(format!("AHI HSD {name} block range overflow"))
    })?;
    bytes
        .get(offset..end)
        .ok_or_else(|| RustySatError::invalid_input(format!("AHI HSD {name} block is truncated")))
}

fn checked_block_offset(offset: usize, block_length: u16, name: &str) -> Result<usize> {
    offset
        .checked_add(usize::from(block_length))
        .ok_or_else(|| {
            RustySatError::invalid_input(format!("AHI HSD {name} offset overflows usize"))
        })
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
    let end = offset.checked_add(N).ok_or_else(|| {
        RustySatError::invalid_input(format!("AHI HSD field '{field}' offset overflows usize"))
    })?;
    bytes
        .get(offset..end)
        .and_then(|value| value.try_into().ok())
        .ok_or_else(|| truncated_field(field))
}

fn read_fixed_string(bytes: &[u8], offset: usize, len: usize, field: &str) -> Result<String> {
    let end = offset.checked_add(len).ok_or_else(|| {
        RustySatError::invalid_input(format!("AHI HSD field '{field}' offset overflows usize"))
    })?;
    let raw = bytes
        .get(offset..end)
        .ok_or_else(|| truncated_field(field))?;
    let text_end = raw.iter().position(|byte| *byte == 0).unwrap_or(raw.len());
    let text = std::str::from_utf8(&raw[..text_end]).map_err(|err| {
        RustySatError::invalid_input(format!("AHI HSD field '{field}' is not valid UTF-8: {err}"))
    })?;
    Ok(text.trim().to_string())
}

fn truncated_field(field: &str) -> RustySatError {
    RustySatError::invalid_input(format!("AHI HSD field '{field}' is truncated"))
}

fn required_filename_u8(file_match: &FileMatch, key: &str) -> Result<u8> {
    match file_match.filename_info().get(key) {
        Some(PatternValue::Integer(value)) => u8::try_from(*value).map_err(|_| {
            RustySatError::invalid_input(format!("AHI HSD filename field '{key}' must fit in u8"))
        }),
        Some(value) => Err(RustySatError::invalid_input(format!(
            "AHI HSD filename field '{key}' must be an integer, got {value:?}"
        ))),
        None => Err(RustySatError::invalid_input(format!(
            "AHI HSD filename field '{key}' is required"
        ))),
    }
}

fn read_initial_hsd_header_prefix(path: &Path) -> Result<Vec<u8>> {
    if file_has_bzip2_magic(path)? {
        return read_bzip2_prefix(path, INITIAL_HEADER_PREFIX_LEN as usize);
    }
    let file = File::open(path).map_err(|err| {
        RustySatError::invalid_input(format!(
            "failed to open AHI HSD file '{}': {err}",
            path.display()
        ))
    })?;
    read_plain_prefix(file, path, INITIAL_HEADER_PREFIX_LEN)
}

fn read_plain_prefix(file: File, path: &Path, max_bytes: u64) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    file.take(max_bytes)
        .read_to_end(&mut bytes)
        .map_err(|err| {
            RustySatError::invalid_input(format!(
                "failed to read AHI HSD header from '{}': {err}",
                path.display()
            ))
        })?;
    Ok(bytes)
}

fn file_has_bzip2_magic(path: &Path) -> Result<bool> {
    let mut file = File::open(path).map_err(|err| {
        RustySatError::invalid_input(format!(
            "failed to open AHI HSD file '{}': {err}",
            path.display()
        ))
    })?;
    let mut magic = [0_u8; 3];
    let read = file.read(&mut magic).map_err(|err| {
        RustySatError::invalid_input(format!(
            "failed to inspect AHI HSD file '{}': {err}",
            path.display()
        ))
    })?;
    Ok(read == BZIP2_MAGIC.len() && magic == BZIP2_MAGIC)
}

fn read_bzip2_prefix(path: &Path, max_decompressed_bytes: usize) -> Result<Vec<u8>> {
    let file = File::open(path).map_err(|err| {
        RustySatError::invalid_input(format!(
            "failed to open compressed AHI HSD file '{}': {err}",
            path.display()
        ))
    })?;
    let mut decoder = MultiBzDecoder::new(BufReader::new(file));
    let mut bytes = Vec::new();
    decoder
        .by_ref()
        .take(max_decompressed_bytes as u64)
        .read_to_end(&mut bytes)
        .map_err(|err| {
            RustySatError::invalid_input(format!(
                "failed to decompress AHI HSD header from '{}': {err}",
                path.display()
            ))
        })?;
    Ok(bytes)
}

fn data_block_bytes(
    bytes: &[u8],
    data_offset: usize,
    byte_count: usize,
    compression_flag: u8,
) -> Result<Cow<'_, [u8]>> {
    match compression_flag {
        0 => {
            let data_end = data_offset
                .checked_add(byte_count)
                .ok_or_else(|| RustySatError::invalid_input("AHI HSD data range overflow"))?;
            let data = bytes.get(data_offset..data_end).ok_or_else(|| {
                RustySatError::invalid_input(format!(
                    "AHI HSD data block is truncated: need {byte_count} bytes at offset {data_offset}"
                ))
            })?;
            Ok(Cow::Borrowed(data))
        }
        1 => Err(RustySatError::unsupported(
            "AHI HSD gzip-compressed data blocks",
        )),
        2 => {
            let data = bytes.get(data_offset..).ok_or_else(|| {
                RustySatError::invalid_input(format!(
                    "AHI HSD compressed data block is truncated at offset {data_offset}"
                ))
            })?;
            let decompressed =
                read_bzip2_reader_bounded(BufReader::new(data), byte_count, "AHI HSD data block")?;
            if decompressed.len() != byte_count {
                return Err(RustySatError::invalid_input(format!(
                    "AHI HSD bzip2 data block decompressed to {} bytes, expected {byte_count}",
                    decompressed.len()
                )));
            }
            Ok(Cow::Owned(decompressed))
        }
        other => Err(RustySatError::unsupported(format!(
            "AHI HSD data compression flag {other}"
        ))),
    }
}

fn read_bzip2_reader_bounded<R: std::io::BufRead>(
    reader: R,
    max_decompressed_bytes: usize,
    context: &str,
) -> Result<Vec<u8>> {
    let mut decoder = MultiBzDecoder::new(reader);
    let mut bytes = Vec::new();
    let limit = max_decompressed_bytes
        .checked_add(1)
        .ok_or_else(|| RustySatError::invalid_input("AHI HSD decompression limit overflow"))?;
    decoder
        .by_ref()
        .take(limit as u64)
        .read_to_end(&mut bytes)
        .map_err(|err| {
            RustySatError::invalid_input(format!("failed to decompress {context}: {err}"))
        })?;
    if bytes.len() > max_decompressed_bytes {
        return Err(RustySatError::invalid_input(format!(
            "{context} decompressed data exceeds the current safety limit of {max_decompressed_bytes} bytes"
        )));
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;
    use crate::yaml_reader::YamlMetadataReader;
    use bzip2::write::BzEncoder;
    use bzip2::Compression;
    use rusty_sat_core::{DataQuery, DataValue, MetadataValue, Scene};
    use rusty_sat_resample::{
        area_from_metadata_value, resample_dataset_from_attrs, source_geometry_from_dataset,
        ResampleOptions, SourceGeometry,
    };
    use rusty_sat_writers::{FloatTiffWriter, SimpleImageWriter};
    use std::fs;
    use std::io::Write;
    use std::time::{SystemTime, UNIX_EPOCH};

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

    #[test]
    fn file_handler_from_yaml_match_tracks_segment_metadata() {
        let reader = YamlMetadataReader::from_yaml_str(AHI_HSD_STYLE_YAML).unwrap();
        let matches = reader
            .match_filenames(["/data/HS_H08_FLDK_B03_S0110.DAT"])
            .unwrap();

        let handler =
            AhiHsdFileHandler::from_file_match_and_header_bytes(&matches[0], &synthetic_header())
                .unwrap();
        let id = handler.dataset_id().unwrap();

        assert_eq!(handler.file_type(), "hsd_b03");
        assert_eq!(handler.segment(), AhiSegmentInfo::new(1, 10).unwrap());
        assert_eq!(handler.band_name(), "B03");
        assert_eq!(id.name(), "B03");
        assert_eq!(
            id.qualifier("calibration"),
            Some(&DataValue::Text("counts".to_string()))
        );
        assert!(id.qualifier("wavelength").is_some());
    }

    #[test]
    fn ahi_hsd_reader_inventory_and_stub_load() {
        let handler = AhiHsdFileHandler::from_header_bytes(
            "/data/HS_H08_FLDK_B03_S0110.DAT",
            "hsd_b03",
            AhiSegmentInfo::new(1, 10).unwrap(),
            &synthetic_header(),
        )
        .unwrap();
        let reader = AhiHsdReader::new([handler]).unwrap();
        let id = reader.available_dataset_ids().pop().unwrap();

        let inventory = reader.inventory().unwrap();
        let dataset = reader.handlers()[0].dataset_stub().unwrap();

        assert_eq!(reader.name(), "ahi_hsd");
        assert!(inventory.available_dataset_ids().contains(&id));
        assert!(dataset.array().is_none());
        assert_eq!(
            dataset.attr("segment_number"),
            Some(&MetadataValue::Integer(1))
        );
        assert_eq!(
            dataset.attr("total_segments"),
            Some(&MetadataValue::Integer(10))
        );
        assert_eq!(
            dataset
                .attr("platform_name")
                .and_then(MetadataValue::as_str),
            Some("Himawari-8")
        );
        assert_eq!(dataset.attr("columns"), Some(&MetadataValue::Integer(10)));
        assert_eq!(dataset.attr("lines"), Some(&MetadataValue::Integer(20)));
    }

    #[test]
    fn ahi_hsd_reader_drives_scene_load_resample_and_output_writers() {
        let hsd_path = temp_path("ahi_hsd_scene", "DAT");
        let png_path = temp_path("ahi_hsd_scene", "png");
        let tiff_path = temp_path("ahi_hsd_scene", "tif");
        fs::write(
            &hsd_path,
            synthetic_full_hsd_file_with_visible_calibration(),
        )
        .unwrap();

        let handler =
            AhiHsdFileHandler::from_path(&hsd_path, "hsd_b03", AhiSegmentInfo::new(1, 10).unwrap())
                .unwrap();
        let reader = AhiHsdReader::with_calibration(AhiCalibration::Reflectance, [handler])
            .unwrap()
            .with_output(AhiCalibrationOutput::ScientificF64);
        let inventory = reader.inventory().unwrap();
        let mut scene = Scene::new();
        let plan = scene
            .plan_reader_loads([DataQuery::named("B03").unwrap()], [&inventory])
            .unwrap();
        let id = plan
            .reader_datasets()
            .get(reader.name())
            .unwrap()
            .iter()
            .next()
            .unwrap()
            .clone();

        scene.insert_dataset(reader.load(&id).unwrap());
        let area = area_from_metadata_value(scene.get(&id).unwrap().attr("area").unwrap()).unwrap();
        let resampled =
            resample_dataset_from_attrs(scene.get(&id).unwrap(), &area, ResampleOptions::default())
                .unwrap();
        scene.insert_dataset(resampled);
        scene
            .save_dataset(
                &id,
                &SimpleImageWriter::default().with_16_bit_dataset_output(),
                &png_path,
            )
            .unwrap();

        assert_png_luma_dimensions_and_bit_depth(&png_path, 3, 2, 16);
        scene
            .save_dataset(&id, &FloatTiffWriter::default(), &tiff_path)
            .unwrap();
        assert_float_tiff_dimensions_and_first_pixel(&tiff_path, 3, 2);
        fs::remove_file(hsd_path).ok();
        fs::remove_file(png_path).ok();
        fs::remove_file(tiff_path).ok();
    }

    #[test]
    fn ahi_hsd_scene_lifecycle_loads_discovery_and_save() {
        let hsd_path = temp_path("ahi_hsd_scene_lifecycle", "DAT");
        let png_path = temp_path("ahi_hsd_scene_lifecycle", "png");
        fs::write(
            &hsd_path,
            synthetic_full_hsd_file_with_visible_calibration(),
        )
        .unwrap();

        let handler =
            AhiHsdFileHandler::from_path(&hsd_path, "hsd_b03", AhiSegmentInfo::new(1, 10).unwrap())
                .unwrap();
        let reader =
            AhiHsdReader::with_calibration(AhiCalibration::Reflectance, [handler]).unwrap();
        let mut scene = Scene::with_loader(reader);

        assert_eq!(scene.available_dataset_names(), vec!["B03".to_string()]);

        scene.load([DataQuery::named("B03").unwrap()]).unwrap();

        assert_eq!(scene.len(), 1);
        assert!(scene.missing_datasets().is_empty());
        let id = &scene.available_dataset_ids()[0];
        let dataset = scene.get(id).expect("loaded B03");
        assert!(dataset.attr("area").is_some());
        assert!(dataset.array().unwrap().coord("x").is_some());
        assert_eq!(
            dataset.attr("calibration").and_then(MetadataValue::as_str),
            Some("reflectance")
        );
        assert_eq!(scene.sensor_names(), vec!["ahi".to_string()]);

        scene
            .save_dataset(id, &SimpleImageWriter::default(), &png_path)
            .unwrap();
        assert_png_luma_dimensions_and_bit_depth(&png_path, 3, 2, 8);
        fs::remove_file(hsd_path).ok();
        fs::remove_file(png_path).ok();
    }

    #[test]
    fn file_handler_requires_segment_filename_fields() {
        let reader = YamlMetadataReader::from_yaml_str(
            r#"
reader:
  name: ahi_hsd
file_types:
  hsd_b03:
    file_patterns: ['HS_H08_FLDK_B03.DAT']
"#,
        )
        .unwrap();
        let matches = reader.match_filenames(["HS_H08_FLDK_B03.DAT"]).unwrap();

        let err =
            AhiHsdFileHandler::from_file_match_and_header_bytes(&matches[0], &synthetic_header())
                .unwrap_err();

        assert!(err.to_string().contains("segment"));
    }

    #[test]
    fn loads_raw_hsd_count_array_from_uncompressed_bytes() {
        let bytes = synthetic_full_hsd_file();
        let handler = AhiHsdFileHandler::from_header_bytes(
            "HS_H08_FLDK_B03_S0110.DAT",
            "hsd_b03",
            AhiSegmentInfo::new(1, 10).unwrap(),
            &bytes,
        )
        .unwrap();

        let dataset = handler.counts_dataset_from_bytes(&bytes).unwrap();
        let array = dataset.array().unwrap();

        assert_eq!(
            handler.header().segment.as_ref().unwrap().total_segments,
            10
        );
        assert_eq!(array.shape(), &[2, 3]);
        assert_eq!(array.dims(), &["y".to_string(), "x".to_string()]);
        let rusty_sat_core::AnyDataArray::U16(array) = array else {
            panic!("expected u16 raw count array");
        };
        assert_eq!(array.values(), &[1, 65535, 2, 65534, 3, 4]);
        assert_eq!(array.is_masked(0), Some(false));
        assert_eq!(array.is_masked(1), Some(true));
        assert_eq!(array.is_masked(3), Some(true));
        assert_eq!(
            dataset.attr("first_line_number"),
            Some(&MetadataValue::Integer(1))
        );
    }

    #[test]
    fn loads_whole_file_bzip2_hsd_from_path() {
        let hsd_path = temp_path("ahi_hsd_whole_file", "DAT.bz2");
        let compressed = bzip2_compress(&synthetic_full_hsd_file());
        fs::write(&hsd_path, compressed).unwrap();

        let handler =
            AhiHsdFileHandler::from_path(&hsd_path, "hsd_b03", AhiSegmentInfo::new(1, 10).unwrap())
                .unwrap();
        let dataset = handler.load_counts_dataset().unwrap();

        let rusty_sat_core::AnyDataArray::U16(array) = dataset.array().unwrap() else {
            panic!("expected u16 raw count array");
        };
        assert_eq!(array.values(), &[1, 65535, 2, 65534, 3, 4]);
        assert_eq!(array.is_masked(1), Some(true));
        fs::remove_file(hsd_path).ok();
    }

    #[test]
    fn loads_bzip2_compressed_hsd_data_block() {
        let bytes = synthetic_full_hsd_file_with_bzip2_data_block(&[1_u16, 65535, 2, 65534, 3, 4]);
        let handler = AhiHsdFileHandler::from_header_bytes(
            "HS_H08_FLDK_B03_S0110.DAT",
            "hsd_b03",
            AhiSegmentInfo::new(1, 10).unwrap(),
            &bytes,
        )
        .unwrap();

        let dataset = handler.counts_dataset_from_bytes(&bytes).unwrap();

        let rusty_sat_core::AnyDataArray::U16(array) = dataset.array().unwrap() else {
            panic!("expected u16 raw count array");
        };
        assert_eq!(array.values(), &[1, 65535, 2, 65534, 3, 4]);
        assert_eq!(array.is_masked(3), Some(true));
    }

    #[test]
    fn reader_assembles_complete_hsd_segments_in_line_order() {
        let seg1_path = temp_path("ahi_hsd_segment_1", "DAT");
        let seg2_path = temp_path("ahi_hsd_segment_2", "DAT");
        fs::write(
            &seg2_path,
            synthetic_full_hsd_segment(2, 2, &[20, 21, 22, 23, 65534, 25]),
        )
        .unwrap();
        fs::write(
            &seg1_path,
            synthetic_full_hsd_segment(1, 2, &[10, 11, 65535, 13, 14, 15]),
        )
        .unwrap();
        let seg2 =
            AhiHsdFileHandler::from_path(&seg2_path, "hsd_b03", AhiSegmentInfo::new(2, 2).unwrap())
                .unwrap();
        let seg1 =
            AhiHsdFileHandler::from_path(&seg1_path, "hsd_b03", AhiSegmentInfo::new(1, 2).unwrap())
                .unwrap();
        let reader = AhiHsdReader::new([seg2, seg1]).unwrap();
        let ids = reader.available_dataset_ids();

        assert_eq!(ids.len(), 1);
        let dataset = reader.load(&ids[0]).unwrap();

        let rusty_sat_core::AnyDataArray::U16(array) = dataset.array().unwrap() else {
            panic!("expected assembled u16 counts array");
        };
        assert_eq!(array.shape_nd(), &[4, 3]);
        assert_eq!(
            array.values(),
            &[10, 11, 65535, 13, 14, 15, 20, 21, 22, 23, 65534, 25]
        );
        assert_eq!(array.mask().unwrap().masked_count(), 2);
        assert_eq!(array.is_masked(2), Some(true));
        assert_eq!(array.is_masked(10), Some(true));
        assert_eq!(array.coord("x").unwrap().values().len(), 3);
        assert_eq!(array.coord("y").unwrap().values().len(), 4);
        let SourceGeometry::Area(area) = source_geometry_from_dataset(&dataset).unwrap() else {
            panic!("expected assembled area source geometry");
        };
        assert_eq!(area.shape(), (4, 3));
        assert_eq!(dataset.attr("lines"), Some(&MetadataValue::Integer(4)));
        assert_eq!(
            dataset.attr("assembled_segments"),
            Some(&MetadataValue::List(vec![
                MetadataValue::Integer(1),
                MetadataValue::Integer(2)
            ]))
        );
        fs::remove_file(seg1_path).ok();
        fs::remove_file(seg2_path).ok();
    }

    #[test]
    fn reader_rejects_missing_hsd_segment_for_assembly() {
        let seg1_path = temp_path("ahi_hsd_missing_segment", "DAT");
        let seg3_path = temp_path("ahi_hsd_missing_segment", "DAT");
        fs::write(
            &seg1_path,
            synthetic_full_hsd_segment(1, 3, &[10, 11, 12, 13, 14, 15]),
        )
        .unwrap();
        fs::write(
            &seg3_path,
            synthetic_full_hsd_segment(3, 3, &[30, 31, 32, 33, 34, 35]),
        )
        .unwrap();
        let seg1 =
            AhiHsdFileHandler::from_path(&seg1_path, "hsd_b03", AhiSegmentInfo::new(1, 3).unwrap())
                .unwrap();
        let seg3 =
            AhiHsdFileHandler::from_path(&seg3_path, "hsd_b03", AhiSegmentInfo::new(3, 3).unwrap())
                .unwrap();
        let reader = AhiHsdReader::new([seg1, seg3]).unwrap();
        let id = reader.available_dataset_ids().pop().unwrap();

        let err = reader.load(&id).unwrap_err();

        assert!(err.to_string().contains("missing AHI HSD segment"));
        assert!(err.to_string().contains("2"));
        fs::remove_file(seg1_path).ok();
        fs::remove_file(seg3_path).ok();
    }

    #[test]
    fn reader_rejects_duplicate_hsd_segment_for_assembly() {
        let seg1a_path = temp_path("ahi_hsd_duplicate_segment_a", "DAT");
        let seg1b_path = temp_path("ahi_hsd_duplicate_segment_b", "DAT");
        fs::write(
            &seg1a_path,
            synthetic_full_hsd_segment(1, 2, &[10, 11, 12, 13, 14, 15]),
        )
        .unwrap();
        fs::write(
            &seg1b_path,
            synthetic_full_hsd_segment(1, 2, &[20, 21, 22, 23, 24, 25]),
        )
        .unwrap();
        let seg1a = AhiHsdFileHandler::from_path(
            &seg1a_path,
            "hsd_b03",
            AhiSegmentInfo::new(1, 2).unwrap(),
        )
        .unwrap();
        let seg1b = AhiHsdFileHandler::from_path(
            &seg1b_path,
            "hsd_b03",
            AhiSegmentInfo::new(1, 2).unwrap(),
        )
        .unwrap();
        let reader = AhiHsdReader::new([seg1a, seg1b]).unwrap();
        let id = reader.available_dataset_ids().pop().unwrap();

        let err = reader.load(&id).unwrap_err();

        assert!(err.to_string().contains("duplicate AHI HSD segment 1"));
        fs::remove_file(seg1a_path).ok();
        fs::remove_file(seg1b_path).ok();
    }

    #[test]
    fn reader_rejects_hsd_segment_header_filename_mismatch() {
        let seg2_path = temp_path("ahi_hsd_segment_mismatch", "DAT");
        fs::write(
            &seg2_path,
            synthetic_full_hsd_segment(1, 2, &[10, 11, 12, 13, 14, 15]),
        )
        .unwrap();
        let handler =
            AhiHsdFileHandler::from_path(&seg2_path, "hsd_b03", AhiSegmentInfo::new(2, 2).unwrap())
                .unwrap();
        let other = AhiHsdFileHandler::from_header_bytes(
            "HS_H08_FLDK_B03_S0102.DAT",
            "hsd_b03",
            AhiSegmentInfo::new(1, 2).unwrap(),
            &synthetic_full_hsd_segment(1, 2, &[1, 2, 3, 4, 5, 6]),
        )
        .unwrap();
        let reader = AhiHsdReader::new([handler, other]).unwrap();
        let id = reader.available_dataset_ids().pop().unwrap();

        let err = reader.load(&id).unwrap_err();

        assert!(err.to_string().contains("block-7 segment sequence"));
        fs::remove_file(seg2_path).ok();
    }

    #[test]
    fn area_metadata_matches_satpy_region_navigation_case() {
        let mut bytes = synthetic_header();
        let data = BASIC_INFO_LEN;
        write_u16(&mut bytes, data + 5, 1000);
        write_u16(&mut bytes, data + 7, 1000);
        let proj = BASIC_INFO_LEN + DATA_INFO_LEN;
        write_f32(&mut bytes, proj + 19, -591.5);
        write_f32(&mut bytes, proj + 23, 5132.5);
        let handler = AhiHsdFileHandler::from_header_bytes(
            "HS_H08_FLDK_B03_S0101.DAT",
            "hsd_b03",
            AhiSegmentInfo::new(1, 1).unwrap(),
            &bytes,
        )
        .unwrap();

        let area = area_from_metadata_value(&handler.area_metadata_value().unwrap()).unwrap();

        assert_eq!(area.id(), "FLDK");
        assert_eq!(area.proj_id(), "geosh8");
        assert_eq!(area.shape(), (1000, 1000));
        assert_eq!(
            area.projection().get("proj").map(String::as_str),
            Some("geos")
        );
        assert_eq!(
            area.projection().get("lon_0").map(String::as_str),
            Some("140.7")
        );
        assert_close(
            area.area_extent(),
            [
                592000.0038256242,
                4132000.0267018233,
                1592000.0102878273,
                5132000.033164027,
            ],
            1.0e-6,
        );
    }

    #[test]
    fn area_metadata_matches_satpy_segment_navigation_case() {
        let mut bytes = synthetic_header();
        let data = BASIC_INFO_LEN;
        write_u16(&mut bytes, data + 5, 11000);
        write_u16(&mut bytes, data + 7, 1100);
        let handler = AhiHsdFileHandler::from_header_bytes(
            "HS_H08_FLDK_B03_S0810.DAT",
            "hsd_b03",
            AhiSegmentInfo::new(8, 10).unwrap(),
            &bytes,
        )
        .unwrap();

        let area = area_from_metadata_value(&handler.area_metadata_value().unwrap()).unwrap();

        assert_eq!(area.shape(), (1100, 11000));
        assert_close(
            area.area_extent(),
            [
                -5500000.035542117,
                -3300000.021325271,
                5500000.035542117,
                -2200000.0142168473,
            ],
            1.0e-6,
        );
    }

    #[test]
    fn hsd_dataset_area_attr_and_xy_coords_drive_resampling_pipeline() {
        let bytes = synthetic_full_hsd_file();
        let handler = AhiHsdFileHandler::from_header_bytes(
            "HS_H08_FLDK_B03_S0110.DAT",
            "hsd_b03",
            AhiSegmentInfo::new(1, 10).unwrap(),
            &bytes,
        )
        .unwrap();
        let dataset = handler
            .calibrated_dataset_from_bytes_f64(&bytes, AhiCalibration::Radiance)
            .unwrap();

        let SourceGeometry::Area(area) = source_geometry_from_dataset(&dataset).unwrap() else {
            panic!("expected area source geometry");
        };
        let array = dataset.array().unwrap();
        assert_eq!(array.coord("x").unwrap().values().len(), 3);
        assert_eq!(array.coord("y").unwrap().values().len(), 2);
        assert_eq!(area.shape(), (2, 3));

        let resampled =
            resample_dataset_from_attrs(&dataset, &area, ResampleOptions::default()).unwrap();

        let rusty_sat_core::AnyDataArray::F64(output) = resampled.array().unwrap() else {
            panic!("expected f64 output");
        };
        assert_eq!(
            output.values(),
            &[-0.99, 654.35, -0.98, 654.34, -0.97, -0.96]
        );
        assert_eq!(output.mask().unwrap().masked_count(), 2);
    }

    #[test]
    fn rejects_truncated_bzip2_data_block() {
        let mut bytes =
            synthetic_full_hsd_file_with_bzip2_data_block(&[1_u16, 65535, 2, 65534, 3, 4]);
        bytes.pop();
        let handler = AhiHsdFileHandler::from_header_bytes(
            "HS_H08_FLDK_B03_S0110.DAT",
            "hsd_b03",
            AhiSegmentInfo::new(1, 10).unwrap(),
            &bytes,
        )
        .unwrap();

        let err = handler.counts_dataset_from_bytes(&bytes).unwrap_err();

        assert!(err.to_string().contains("decompress"));
    }

    #[test]
    fn rejects_bzip2_data_block_that_expands_past_expected_size() {
        let bytes =
            synthetic_full_hsd_file_with_bzip2_data_block(&[1_u16, 65535, 2, 65534, 3, 4, 5]);
        let handler = AhiHsdFileHandler::from_header_bytes(
            "HS_H08_FLDK_B03_S0110.DAT",
            "hsd_b03",
            AhiSegmentInfo::new(1, 10).unwrap(),
            &bytes,
        )
        .unwrap();

        let err = handler.counts_dataset_from_bytes(&bytes).unwrap_err();

        assert!(err.to_string().contains("exceeds"));
    }

    #[test]
    fn rejects_gzip_hsd_data_block_until_supported() {
        let mut bytes = synthetic_full_hsd_file();
        write_u8(&mut bytes, BASIC_INFO_LEN + 9, 1);
        let handler = AhiHsdFileHandler::from_header_bytes(
            "HS_H08_FLDK_B03_S0110.DAT",
            "hsd_b03",
            AhiSegmentInfo::new(1, 10).unwrap(),
            &bytes,
        )
        .unwrap();

        let err = handler.counts_dataset_from_bytes(&bytes).unwrap_err();

        assert!(matches!(err, RustySatError::Unsupported { .. }));
        assert!(err.to_string().contains("gzip"));
    }

    #[test]
    fn raw_hsd_count_loading_rejects_truncated_data_block() {
        let bytes = synthetic_full_hsd_file();
        let handler = AhiHsdFileHandler::from_header_bytes(
            "HS_H08_FLDK_B03_S0110.DAT",
            "hsd_b03",
            AhiSegmentInfo::new(1, 10).unwrap(),
            &bytes,
        )
        .unwrap();

        let err = handler
            .counts_dataset_from_bytes(&bytes[..bytes.len() - 2])
            .unwrap_err();

        assert!(err.to_string().contains("data block is truncated"));
    }

    #[test]
    fn calibrates_visible_hsd_counts_to_radiance_and_reflectance() {
        let bytes = synthetic_full_hsd_file_with_visible_calibration();
        let handler = AhiHsdFileHandler::from_header_bytes(
            "HS_H08_FLDK_B03_S0110.DAT",
            "hsd_b03",
            AhiSegmentInfo::new(1, 10).unwrap(),
            &bytes,
        )
        .unwrap();

        let radiance = handler
            .calibrated_dataset_from_bytes(&bytes, AhiCalibration::Radiance)
            .unwrap();
        let reflectance = handler
            .calibrated_dataset_from_bytes(&bytes, AhiCalibration::Reflectance)
            .unwrap();

        assert_eq!(
            radiance.id().qualifier("calibration"),
            Some(&DataValue::Text("radiance".to_string()))
        );
        assert_eq!(
            reflectance.id().qualifier("calibration"),
            Some(&DataValue::Text("reflectance".to_string()))
        );
        let rusty_sat_core::AnyDataArray::F32(radiance) = radiance.array().unwrap() else {
            panic!("expected f32 radiance array");
        };
        let rusty_sat_core::AnyDataArray::F32(reflectance) = reflectance.array().unwrap() else {
            panic!("expected f32 reflectance array");
        };
        assert_eq!(radiance.values()[0], 0.0);
        assert_eq!(radiance.values()[2], 2.0);
        assert_eq!(reflectance.values()[0], 0.0);
        assert_eq!(reflectance.values()[2], 100.0);
        assert_eq!(reflectance.is_masked(1), Some(true));
    }

    #[test]
    fn calibrates_hsd_counts_to_f64_when_precision_is_requested() {
        let bytes = synthetic_full_hsd_file_with_visible_calibration();
        let handler = AhiHsdFileHandler::from_header_bytes(
            "HS_H08_FLDK_B03_S0110.DAT",
            "hsd_b03",
            AhiSegmentInfo::new(1, 10).unwrap(),
            &bytes,
        )
        .unwrap();

        let reflectance = handler
            .calibrated_dataset_from_bytes_f64(&bytes, AhiCalibration::Reflectance)
            .unwrap();

        assert_eq!(
            reflectance
                .attr("precision")
                .and_then(MetadataValue::as_str),
            Some("f64")
        );
        let rusty_sat_core::AnyDataArray::F64(reflectance) = reflectance.array().unwrap() else {
            panic!("expected f64 reflectance array");
        };
        assert_eq!(reflectance.values()[2], 100.0);
        assert_eq!(reflectance.is_masked(1), Some(true));
    }

    #[test]
    fn visible_calibration_can_use_nominal_or_update_with_fallback() {
        let bytes = synthetic_full_hsd_file_with_visible_calibration();
        let nominal = AhiHsdFileHandler::from_header_bytes(
            "HS_H08_FLDK_B03_S0110.DAT",
            "hsd_b03",
            AhiSegmentInfo::new(1, 10).unwrap(),
            &bytes,
        )
        .unwrap()
        .with_calibration_mode(AhiCalibrationMode::Nominal);
        let update = AhiHsdFileHandler::from_header_bytes(
            "HS_H08_FLDK_B03_S0110.DAT",
            "hsd_b03",
            AhiSegmentInfo::new(1, 10).unwrap(),
            &bytes,
        )
        .unwrap();

        assert_eq!(
            nominal
                .calibrate_counts_to_f32(&[0, 100, 200], AhiCalibration::Radiance)
                .unwrap(),
            vec![-1.0, 0.0, 1.0]
        );
        assert_eq!(
            update
                .calibrate_counts_to_f32(&[0, 100, 200], AhiCalibration::Radiance)
                .unwrap(),
            vec![-2.0, 0.0, 2.0]
        );

        let mut fallback_bytes = bytes.clone();
        let cal = BASIC_INFO_LEN + DATA_INFO_LEN + PROJECTION_INFO_LEN + NAVIGATION_INFO_LEN;
        write_f64(&mut fallback_bytes, cal + 51, 0.0);
        write_f64(&mut fallback_bytes, cal + 59, 0.0);
        let fallback = AhiHsdFileHandler::from_header_bytes(
            "HS_H08_FLDK_B03_S0110.DAT",
            "hsd_b03",
            AhiSegmentInfo::new(1, 10).unwrap(),
            &fallback_bytes,
        )
        .unwrap();

        assert_eq!(
            fallback
                .calibrate_counts_to_f32(&[0, 100, 200], AhiCalibration::Radiance)
                .unwrap(),
            vec![-1.0, 0.0, 1.0]
        );
    }

    #[test]
    fn user_calibration_matches_satpy_rad_and_dn_modes() {
        let mut bytes = synthetic_full_hsd_file_with_infrared_calibration();
        let cal = BASIC_INFO_LEN + DATA_INFO_LEN + PROJECTION_INFO_LEN + NAVIGATION_INFO_LEN;
        write_f64(&mut bytes, cal + 19, -0.0037);
        write_f64(&mut bytes, cal + 27, 15.20);
        let base = AhiHsdFileHandler::from_header_bytes(
            "HS_H08_FLDK_B13_S0110.DAT",
            "hsd_b13",
            AhiSegmentInfo::new(1, 10).unwrap(),
            &bytes,
        )
        .unwrap();
        let rad = base.clone().with_user_calibration(
            AhiUserCalibration::radiance_correction([(
                "B13",
                AhiUserCalibrationCoefficients {
                    slope: 0.95,
                    offset: -0.1,
                },
            )])
            .unwrap(),
        );
        let dn = base.with_user_calibration(
            AhiUserCalibration::digital_number([(
                "B13",
                AhiUserCalibrationCoefficients {
                    slope: -0.0032,
                    offset: 15.20,
                },
            )])
            .unwrap(),
        );
        let counts = [0_u16, 1000, 2000, 5000];

        let rad_values = rad
            .calibrate_counts_to_f32(&counts, AhiCalibration::Radiance)
            .unwrap();
        let dn_values = dn
            .calibrate_counts_to_f32(&counts, AhiCalibration::Radiance)
            .unwrap();

        assert_close_vec(
            &rad_values,
            &[16.105263, 12.210526, 8.315789, -3.368421],
            1.0e-5,
        );
        assert_close_vec(&dn_values, &[15.2, 12.0, 8.8, -0.8], 1.0e-6);
    }

    #[test]
    fn ahi_hsd_reader_can_select_scientific_f64_output() {
        let bytes = synthetic_full_hsd_file_with_visible_calibration();
        let hsd_path = temp_path("ahi_hsd_reader_f64", "DAT");
        fs::write(&hsd_path, bytes).unwrap();
        let handler =
            AhiHsdFileHandler::from_path(&hsd_path, "hsd_b03", AhiSegmentInfo::new(1, 10).unwrap())
                .unwrap();
        let reader = AhiHsdReader::with_calibration(AhiCalibration::Reflectance, [handler])
            .unwrap()
            .with_output(AhiCalibrationOutput::ScientificF64);
        let id = reader.available_dataset_ids().pop().unwrap();

        let dataset = reader.load(&id).unwrap();

        assert_eq!(
            dataset.attr("precision").and_then(MetadataValue::as_str),
            Some("f64")
        );
        let rusty_sat_core::AnyDataArray::F64(array) = dataset.array().unwrap() else {
            panic!("expected f64 array");
        };
        assert_eq!(array.values()[2], 100.0);
        fs::remove_file(hsd_path).ok();
    }

    #[test]
    fn calibrates_infrared_hsd_counts_to_brightness_temperature() {
        let bytes = synthetic_full_hsd_file_with_infrared_calibration();
        let handler = AhiHsdFileHandler::from_header_bytes(
            "HS_H08_FLDK_B13_S0110.DAT",
            "hsd_b13",
            AhiSegmentInfo::new(1, 10).unwrap(),
            &bytes,
        )
        .unwrap();

        let dataset = handler
            .calibrated_dataset_from_bytes(&bytes, AhiCalibration::BrightnessTemperature)
            .unwrap();

        assert_eq!(
            dataset.id().qualifier("calibration"),
            Some(&DataValue::Text("brightness_temperature".to_string()))
        );
        let rusty_sat_core::AnyDataArray::F32(array) = dataset.array().unwrap() else {
            panic!("expected f32 brightness-temperature array");
        };
        assert!(array.values()[0].is_nan());
        assert!(array.values()[2].is_finite());
        assert!(array.values()[2] > 0.0);
        assert_eq!(array.is_masked(1), Some(true));
    }

    #[test]
    fn reflectance_requires_visible_calibration_extension() {
        let bytes = synthetic_full_hsd_file();
        let handler = AhiHsdFileHandler::from_header_bytes(
            "HS_H08_FLDK_B03_S0110.DAT",
            "hsd_b03",
            AhiSegmentInfo::new(1, 10).unwrap(),
            &bytes,
        )
        .unwrap();

        let err = handler
            .calibrated_dataset_from_bytes(&bytes, AhiCalibration::Reflectance)
            .unwrap_err();

        assert!(matches!(err, RustySatError::Unsupported { .. }));
    }

    const AHI_HSD_STYLE_YAML: &str = r#"
reader:
  name: ahi_hsd
  sensors: [ahi]
file_types:
  hsd_b03:
    file_patterns: ['HS_H08_{area:4s}_B03_S{segment:02d}{total_segments:02d}.DAT']
datasets:
  B03:
    name: B03
    file_type: hsd_b03
    wavelength: [0.63, 0.64, 0.65]
    calibration: counts
"#;

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

    fn synthetic_full_hsd_file() -> Vec<u8> {
        let mut bytes = synthetic_header();
        write_u16(&mut bytes, BASIC_INFO_LEN + 5, 3);
        write_u16(&mut bytes, BASIC_INFO_LEN + 7, 2);
        write_u32(&mut bytes, 74, 12);

        let mut block6 = vec![0; 259];
        write_u8(&mut block6, 0, 6);
        write_u16(&mut block6, 1, 259);
        bytes.extend(block6);

        let mut block7 = vec![0; 47];
        write_u8(&mut block7, 0, 7);
        write_u16(&mut block7, 1, 47);
        write_u8(&mut block7, 3, 10);
        write_u8(&mut block7, 4, 1);
        write_u16(&mut block7, 5, 1);
        bytes.extend(block7);

        let total_header_length = bytes.len() as u32;
        write_u32(&mut bytes, 70, total_header_length);
        for value in [1_u16, 65535, 2, 65534, 3, 4] {
            bytes.extend(value.to_le_bytes());
        }
        bytes
    }

    fn synthetic_full_hsd_file_with_visible_calibration() -> Vec<u8> {
        let mut bytes = synthetic_header();
        extend_block5_with_visible_calibration(&mut bytes);
        finish_full_hsd_file(bytes)
    }

    fn synthetic_full_hsd_file_with_infrared_calibration() -> Vec<u8> {
        let mut bytes = synthetic_header();
        let cal = BASIC_INFO_LEN + DATA_INFO_LEN + PROJECTION_INFO_LEN + NAVIGATION_INFO_LEN;
        write_u16(&mut bytes, cal + 3, 13);
        write_f64(&mut bytes, cal + 5, 10.4);
        extend_block5_with_infrared_calibration(&mut bytes);
        finish_full_hsd_file(bytes)
    }

    fn extend_block5_with_visible_calibration(bytes: &mut Vec<u8>) {
        let cal = BASIC_INFO_LEN + DATA_INFO_LEN + PROJECTION_INFO_LEN + NAVIGATION_INFO_LEN;
        write_u16(bytes, cal + 1, 147);
        let mut vis = vec![0; 112];
        write_f64(&mut vis, 0, 0.5);
        write_f64(&mut vis, 8, 60000.4);
        write_f64(&mut vis, 16, 0.02);
        write_f64(&mut vis, 24, -2.0);
        bytes.extend(vis);
    }

    fn extend_block5_with_infrared_calibration(bytes: &mut Vec<u8>) {
        let cal = BASIC_INFO_LEN + DATA_INFO_LEN + PROJECTION_INFO_LEN + NAVIGATION_INFO_LEN;
        write_u16(bytes, cal + 1, 147);
        let mut ir = vec![0; 112];
        write_f64(&mut ir, 0, 0.0);
        write_f64(&mut ir, 8, 1.0);
        write_f64(&mut ir, 16, 0.0);
        write_f64(&mut ir, 24, 0.0);
        write_f64(&mut ir, 32, 1.0);
        write_f64(&mut ir, 40, 0.0);
        write_f64(&mut ir, 48, 299_792_458.0);
        write_f64(&mut ir, 56, 6.626_070_15e-34);
        write_f64(&mut ir, 64, 1.380_649e-23);
        bytes.extend(ir);
    }

    fn finish_full_hsd_file(mut bytes: Vec<u8>) -> Vec<u8> {
        write_u16(&mut bytes, BASIC_INFO_LEN + 5, 3);
        write_u16(&mut bytes, BASIC_INFO_LEN + 7, 2);
        write_u32(&mut bytes, 74, 12);

        let mut block6 = vec![0; 259];
        write_u8(&mut block6, 0, 6);
        write_u16(&mut block6, 1, 259);
        bytes.extend(block6);

        let mut block7 = vec![0; 47];
        write_u8(&mut block7, 0, 7);
        write_u16(&mut block7, 1, 47);
        write_u8(&mut block7, 3, 10);
        write_u8(&mut block7, 4, 1);
        write_u16(&mut block7, 5, 1);
        bytes.extend(block7);

        let total_header_length = bytes.len() as u32;
        write_u32(&mut bytes, 70, total_header_length);
        for value in [100_u16, 65535, 200, 65534, 300, 400] {
            bytes.extend(value.to_le_bytes());
        }
        bytes
    }

    fn synthetic_full_hsd_file_with_bzip2_data_block(values: &[u16]) -> Vec<u8> {
        let mut bytes = synthetic_full_hsd_file();
        write_u8(&mut bytes, BASIC_INFO_LEN + 9, 2);
        let header = parse_initial_hsd_header(&bytes).unwrap();
        let data_offset = header.basic.total_header_length as usize;
        let raw = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let compressed = bzip2_compress(&raw);
        bytes.truncate(data_offset);
        bytes.extend(compressed);
        bytes
    }

    fn synthetic_full_hsd_segment(segment: u8, total_segments: u8, values: &[u16]) -> Vec<u8> {
        assert_eq!(values.len(), 6);
        let mut bytes = synthetic_full_hsd_file();
        let header = parse_initial_hsd_header(&bytes).unwrap();
        let data_offset = header.basic.total_header_length as usize;
        let block7_offset = data_offset - 47;
        write_u8(&mut bytes, block7_offset + 3, total_segments);
        write_u8(&mut bytes, block7_offset + 4, segment);
        let first_line = 1 + (u16::from(segment) - 1) * header.data.lines;
        write_u16(&mut bytes, block7_offset + 5, first_line);
        write_u32(&mut bytes, 74, (values.len() * 2) as u32);
        bytes.truncate(data_offset);
        for value in values {
            bytes.extend(value.to_le_bytes());
        }
        bytes
    }

    fn bzip2_compress(bytes: &[u8]) -> Vec<u8> {
        let mut encoder = BzEncoder::new(Vec::new(), Compression::best());
        encoder.write_all(bytes).unwrap();
        encoder.finish().unwrap()
    }

    fn assert_close(actual: [f64; 4], expected: [f64; 4], tolerance: f64) {
        for (actual, expected) in actual.into_iter().zip(expected) {
            assert!(
                (actual - expected).abs() <= tolerance,
                "expected {actual} to be within {tolerance} of {expected}"
            );
        }
    }

    fn assert_close_vec(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert!(
                (*actual - *expected).abs() <= tolerance,
                "expected {actual} to be within {tolerance} of {expected}"
            );
        }
    }

    fn temp_path(name: &str, extension: &str) -> std::path::PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time")
            .as_nanos();
        std::env::temp_dir().join(format!("rusty_sat_{name}_{nanos}.{extension}"))
    }

    fn assert_png_luma_dimensions_and_bit_depth(
        path: &std::path::Path,
        expected_width: u32,
        expected_height: u32,
        expected_bit_depth: u8,
    ) {
        let bytes = fs::read(path).unwrap();
        assert!(bytes.len() >= 26);
        assert_eq!(&bytes[..8], b"\x89PNG\r\n\x1a\n");
        assert_eq!(
            u32::from_be_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]),
            expected_width
        );
        assert_eq!(
            u32::from_be_bytes([bytes[20], bytes[21], bytes[22], bytes[23]]),
            expected_height
        );
        assert_eq!(bytes[24], expected_bit_depth);
        assert_eq!(bytes[25], 0);
    }

    fn assert_float_tiff_dimensions_and_first_pixel(
        path: &std::path::Path,
        expected_width: u32,
        expected_height: u32,
    ) {
        let bytes = fs::read(path).unwrap();
        assert_eq!(&bytes[..2], b"II");
        assert_eq!(read_le_u16(&bytes, 2), 42);
        assert_eq!(read_tiff_tag_u32(&bytes, 256), expected_width);
        assert_eq!(read_tiff_tag_u32(&bytes, 257), expected_height);
        assert_eq!(read_tiff_tag_u16(&bytes, 258), 32);
        assert_eq!(read_tiff_tag_u16(&bytes, 339), 3);
        assert!(read_tiff_tag_u32(&bytes, 33550) > 0);
        assert!(read_tiff_tag_u32(&bytes, 33922) > 0);
        assert!(read_tiff_tag_u32(&bytes, 34735) > 0);
        assert!(read_tiff_tag_u32(&bytes, 34737) > 0);
        let offset = read_tiff_tag_u32(&bytes, 273) as usize;
        assert!(read_le_f32(&bytes, offset).is_finite());
    }

    fn read_tiff_tag_u16(bytes: &[u8], tag: u16) -> u16 {
        read_le_u16(bytes, find_tiff_ifd_entry(bytes, tag) + 8)
    }

    fn read_tiff_tag_u32(bytes: &[u8], tag: u16) -> u32 {
        read_le_u32(bytes, find_tiff_ifd_entry(bytes, tag) + 8)
    }

    fn find_tiff_ifd_entry(bytes: &[u8], tag: u16) -> usize {
        let ifd_offset = read_le_u32(bytes, 4) as usize;
        let count = read_le_u16(bytes, ifd_offset) as usize;
        for index in 0..count {
            let offset = ifd_offset + 2 + index * 12;
            if read_le_u16(bytes, offset) == tag {
                return offset;
            }
        }
        panic!("missing TIFF tag {tag}");
    }

    fn read_le_u16(bytes: &[u8], offset: usize) -> u16 {
        u16::from_le_bytes([bytes[offset], bytes[offset + 1]])
    }

    fn read_le_u32(bytes: &[u8], offset: usize) -> u32 {
        u32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ])
    }

    fn read_le_f32(bytes: &[u8], offset: usize) -> f32 {
        f32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ])
    }

    fn write_u8(bytes: &mut [u8], offset: usize, value: u8) {
        bytes[offset] = value;
    }

    fn write_u16(bytes: &mut [u8], offset: usize, value: u16) {
        write_bytes(bytes, offset, &value.to_le_bytes());
    }

    fn write_u32(bytes: &mut [u8], offset: usize, value: u32) {
        write_bytes(bytes, offset, &value.to_le_bytes());
    }

    fn write_f32(bytes: &mut [u8], offset: usize, value: f32) {
        write_bytes(bytes, offset, &value.to_le_bytes());
    }

    fn write_f64(bytes: &mut [u8], offset: usize, value: f64) {
        write_bytes(bytes, offset, &value.to_le_bytes());
    }

    fn write_string(bytes: &mut [u8], offset: usize, len: usize, value: &str) {
        let raw = value.as_bytes();
        let count = raw.len().min(len);
        write_bytes(bytes, offset, &raw[..count]);
    }

    fn write_bytes(bytes: &mut [u8], offset: usize, raw: &[u8]) {
        let end = offset
            .checked_add(raw.len())
            .expect("test byte range overflow");
        debug_assert!(end <= bytes.len());
        bytes[offset..end].copy_from_slice(raw);
    }
}
