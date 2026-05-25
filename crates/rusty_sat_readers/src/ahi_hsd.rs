//! Himawari AHI HSD binary header foundations.
//!
//! Reference behavior inspected before implementation:
//! - Root `HS_D_users_guide_en_v12.pdf` is the local HSD user guide reference.
//! - `satpy/satpy/readers/ahi_hsd.py` defines the NumPy dtypes for HSD header
//!   blocks 1-5 and reads them in sequence before dataset loading.
//!
//! This module is intentionally limited to fixed-size initial header parsing,
//! uncompressed raw-count loading, and first-pass calibration. Satpy's display
//! calibration path uses float32-like arithmetic for memory-efficient imagery;
//! Rusty Sat also exposes f64 calibration helpers for future scientific/HDR
//! output paths where precision preservation matters more than display memory.

use crate::filename_pattern::PatternValue;
use crate::yaml_reader::FileMatch;
use crate::Reader;
use bzip2::bufread::MultiBzDecoder;
use rusty_sat_core::{
    AnyDataArray, DataArray, DataId, Dataset, MetadataValue, NumericElement, ReaderInventory,
    Result, RustySatError, ValidityMask, WavelengthRange,
};
use std::borrow::Cow;
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
        Ok(dataset)
    }

    pub fn counts_dataset_from_bytes(&self, bytes: &[u8]) -> Result<Dataset> {
        let values = self.raw_count_values_from_bytes(bytes)?;
        let mask = ValidityMask::from_masked_flags(values.iter().map(|value| {
            *value == self.header.calibration.error_pixel_count_value
                || *value == self.header.calibration.outside_scan_pixel_count_value
        }));
        let array = DataArray::<u16>::from_vec_named(
            vec![
                usize::from(self.header.data.lines),
                usize::from(self.header.data.columns),
            ],
            ["y", "x"],
            values,
        )?
        .with_mask(mask)?;

        let mut dataset = Dataset::new(self.dataset_id()?);
        self.attach_common_attrs(&mut dataset)?;
        dataset.insert_attr("calibration", "counts")?;
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
        let mask = ValidityMask::from_masked_flags(values.iter().map(|value| {
            *value == self.header.calibration.error_pixel_count_value
                || *value == self.header.calibration.outside_scan_pixel_count_value
        }));
        let calibrated_values = self.calibrate_counts_to_f32(&values, calibration)?;
        let array = DataArray::<f32>::from_vec_named(
            vec![
                usize::from(self.header.data.lines),
                usize::from(self.header.data.columns),
            ],
            ["y", "x"],
            calibrated_values,
        )?
        .with_mask(mask)?;

        let mut dataset = Dataset::new(self.dataset_id_for_calibration(calibration)?);
        self.attach_common_attrs(&mut dataset)?;
        dataset.insert_attr("calibration", calibration.name())?;
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
        let mask = ValidityMask::from_masked_flags(values.iter().map(|value| {
            *value == self.header.calibration.error_pixel_count_value
                || *value == self.header.calibration.outside_scan_pixel_count_value
        }));
        let calibrated_values = self.calibrate_counts_to_f64(&values, calibration)?;
        let array = DataArray::<f64>::from_vec_named(
            vec![
                usize::from(self.header.data.lines),
                usize::from(self.header.data.columns),
            ],
            ["y", "x"],
            calibrated_values,
        )?
        .with_mask(mask)?;

        let mut dataset = Dataset::new(self.dataset_id_for_calibration(calibration)?);
        self.attach_common_attrs(&mut dataset)?;
        dataset.insert_attr("calibration", calibration.name())?;
        dataset.insert_attr("precision", "f64")?;
        dataset.set_array(array);
        Ok(dataset)
    }

    pub fn calibrate_counts_to_f32(
        &self,
        counts: &[u16],
        calibration: AhiCalibration,
    ) -> Result<Vec<f32>> {
        let radiance = self.counts_to_radiance_f32(counts);
        match calibration {
            AhiCalibration::Counts => Ok(counts.iter().map(|value| f32::from(*value)).collect()),
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
        let radiance = self.counts_to_radiance_f64(counts);
        match calibration {
            AhiCalibration::Counts => Ok(counts.iter().map(|value| f64::from(*value)).collect()),
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
        let bytes = self.read_file_bytes()?;
        self.counts_dataset_from_bytes(&bytes)
    }

    pub fn load_calibrated_dataset(&self, calibration: AhiCalibration) -> Result<Dataset> {
        let bytes = self.read_file_bytes()?;
        self.calibrated_dataset_from_bytes(&bytes, calibration)
    }

    pub fn load_calibrated_dataset_f64(&self, calibration: AhiCalibration) -> Result<Dataset> {
        let bytes = self.read_file_bytes()?;
        self.calibrated_dataset_from_bytes_f64(&bytes, calibration)
    }

    fn read_file_bytes(&self) -> Result<Vec<u8>> {
        read_hsd_file_bytes_bounded(&self.filename)
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
        let gain = self.header.calibration.gain_count_to_radiance as f32;
        let offset = self.header.calibration.offset_count_to_radiance as f32;
        counts
            .iter()
            .map(|value| f32::from(*value) * gain + offset)
            .collect()
    }

    fn counts_to_radiance_f64(&self, counts: &[u16]) -> Vec<f64> {
        let gain = self.header.calibration.gain_count_to_radiance;
        let offset = self.header.calibration.offset_count_to_radiance;
        counts
            .iter()
            .map(|value| f64::from(*value) * gain + offset)
            .collect()
    }

    fn radiance_to_reflectance_f32(&self, radiance: Vec<f32>) -> Result<Vec<f32>> {
        let coeff = self.required_visible_albedo_coeff()? as f32;
        Ok(radiance
            .into_iter()
            .map(|value| (value * coeff * 100.0).max(0.0))
            .collect())
    }

    fn radiance_to_reflectance_f64(&self, radiance: Vec<f64>) -> Result<Vec<f64>> {
        let coeff = self.required_visible_albedo_coeff()?;
        Ok(radiance
            .into_iter()
            .map(|value| (value * coeff * 100.0).max(0.0))
            .collect())
    }

    fn radiance_to_brightness_temperature_f32(&self, radiance: Vec<f32>) -> Result<Vec<f32>> {
        Ok(self
            .radiance_to_brightness_temperature_f64(
                radiance.into_iter().map(f64::from).collect::<Vec<_>>(),
            )?
            .into_iter()
            .map(|value| value as f32)
            .collect())
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
}

#[derive(Debug, Clone, PartialEq)]
pub struct AhiHsdReader {
    name: String,
    handlers: Vec<AhiHsdFileHandler>,
    calibration: AhiCalibration,
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
        })
    }

    pub fn handlers(&self) -> &[AhiHsdFileHandler] {
        &self.handlers
    }

    pub fn calibration(&self) -> AhiCalibration {
        self.calibration
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
            [handler] => handler.load_calibrated_dataset(self.calibration),
            _ => self.load_assembled_dataset(matching_handlers),
        }
    }
}

impl AhiHsdReader {
    fn load_assembled_dataset(&self, handlers: Vec<&AhiHsdFileHandler>) -> Result<Dataset> {
        let sorted_handlers = sorted_complete_segment_handlers(handlers)?;
        let mut datasets = Vec::with_capacity(sorted_handlers.len());
        for handler in &sorted_handlers {
            datasets.push(handler.load_calibrated_dataset(self.calibration)?);
        }
        assemble_segment_datasets(datasets, &sorted_handlers, self.calibration)
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

fn assemble_segment_datasets(
    mut datasets: Vec<Dataset>,
    handlers: &[&AhiHsdFileHandler],
    calibration: AhiCalibration,
) -> Result<Dataset> {
    let Some(mut output) = datasets
        .first_mut()
        .map(|dataset| Dataset::new(dataset.id().clone()))
    else {
        return Err(RustySatError::invalid_input(
            "AHI HSD segment assembly requires at least one dataset",
        ));
    };
    let first_dataset = datasets.remove(0);
    let mut arrays = vec![first_dataset.into_array().ok_or_else(|| {
        RustySatError::invalid_input("AHI HSD segment dataset is missing array data")
    })?];
    for dataset in datasets {
        arrays.push(dataset.into_array().ok_or_else(|| {
            RustySatError::invalid_input("AHI HSD segment dataset is missing array data")
        })?);
    }
    let array = concatenate_yx_any_arrays(arrays)?;
    handlers[0].attach_common_attrs(&mut output)?;
    output.insert_attr("calibration", calibration.name())?;
    output.insert_attr(
        "lines",
        i64::try_from(array.shape()[0]).map_err(|_| {
            RustySatError::invalid_input("assembled AHI HSD line count does not fit in i64")
        })?,
    )?;
    output.insert_attr(
        "assembled_segments",
        MetadataValue::List(
            handlers
                .iter()
                .map(|handler| MetadataValue::Integer(i64::from(handler.segment.segment_number)))
                .collect(),
        ),
    )?;
    output.set_array(array);
    Ok(output)
}

fn concatenate_yx_any_arrays(arrays: Vec<AnyDataArray>) -> Result<AnyDataArray> {
    let Some(first) = arrays.first() else {
        return Err(RustySatError::invalid_input(
            "AHI HSD segment assembly requires at least one array",
        ));
    };
    match first {
        AnyDataArray::F32(_) => concatenate_yx_typed_arrays(
            arrays
                .into_iter()
                .map(|array| match array {
                    AnyDataArray::F32(array) => Ok(array),
                    other => Err(RustySatError::invalid_input(format!(
                        "AHI HSD segment dtype mismatch: expected f32, got {}",
                        other.dtype().name()
                    ))),
                })
                .collect::<Result<Vec<_>>>()?,
        )
        .map(Into::into),
        AnyDataArray::F64(_) => concatenate_yx_typed_arrays(
            arrays
                .into_iter()
                .map(|array| match array {
                    AnyDataArray::F64(array) => Ok(array),
                    other => Err(RustySatError::invalid_input(format!(
                        "AHI HSD segment dtype mismatch: expected f64, got {}",
                        other.dtype().name()
                    ))),
                })
                .collect::<Result<Vec<_>>>()?,
        )
        .map(Into::into),
        AnyDataArray::U8(_) => concatenate_yx_typed_arrays(
            arrays
                .into_iter()
                .map(|array| match array {
                    AnyDataArray::U8(array) => Ok(array),
                    other => Err(RustySatError::invalid_input(format!(
                        "AHI HSD segment dtype mismatch: expected u8, got {}",
                        other.dtype().name()
                    ))),
                })
                .collect::<Result<Vec<_>>>()?,
        )
        .map(Into::into),
        AnyDataArray::U16(_) => concatenate_yx_typed_arrays(
            arrays
                .into_iter()
                .map(|array| match array {
                    AnyDataArray::U16(array) => Ok(array),
                    other => Err(RustySatError::invalid_input(format!(
                        "AHI HSD segment dtype mismatch: expected u16, got {}",
                        other.dtype().name()
                    ))),
                })
                .collect::<Result<Vec<_>>>()?,
        )
        .map(Into::into),
        AnyDataArray::I16(_) => concatenate_yx_typed_arrays(
            arrays
                .into_iter()
                .map(|array| match array {
                    AnyDataArray::I16(array) => Ok(array),
                    other => Err(RustySatError::invalid_input(format!(
                        "AHI HSD segment dtype mismatch: expected i16, got {}",
                        other.dtype().name()
                    ))),
                })
                .collect::<Result<Vec<_>>>()?,
        )
        .map(Into::into),
    }
}

fn concatenate_yx_typed_arrays<T: NumericElement>(
    arrays: Vec<DataArray<T>>,
) -> Result<DataArray<T>> {
    let Some(first) = arrays.first() else {
        return Err(RustySatError::invalid_input(
            "AHI HSD segment assembly requires at least one typed array",
        ));
    };
    first.require_dims_exact(&["y", "x"])?;
    let width = first.shape_yx()?.1;
    let mut total_height = 0_usize;
    let mut total_len = 0_usize;
    for array in &arrays {
        array.require_dims_exact(&["y", "x"])?;
        let (height, array_width) = array.shape_yx()?;
        if array_width != width {
            return Err(RustySatError::invalid_input(format!(
                "AHI HSD segment width mismatch: expected {width}, got {array_width}"
            )));
        }
        total_height = total_height
            .checked_add(height)
            .ok_or_else(|| RustySatError::invalid_input("AHI HSD assembled height overflow"))?;
        total_len = total_len
            .checked_add(array.len())
            .ok_or_else(|| RustySatError::invalid_input("AHI HSD assembled length overflow"))?;
    }

    let mut values = Vec::with_capacity(total_len);
    let mut mask_flags: Option<Vec<bool>> = None;
    for array in arrays {
        let previous_len = values.len();
        let (segment_values, _, segment_mask) = array.into_parts();
        if let Some(flags) = &mut mask_flags {
            extend_mask_flags(flags, segment_mask.as_ref(), segment_values.len());
        } else if let Some(mask) = segment_mask.as_ref() {
            let mut flags = vec![false; previous_len];
            extend_mask_flags(&mut flags, Some(mask), segment_values.len());
            mask_flags = Some(flags);
        }
        values.extend(segment_values);
    }

    let mut array = DataArray::<T>::from_vec_named([total_height, width], ["y", "x"], values)?;
    if let Some(flags) = mask_flags {
        array.set_mask(ValidityMask::from_masked_flags(flags))?;
    }
    Ok(array)
}

fn extend_mask_flags(flags: &mut Vec<bool>, mask: Option<&ValidityMask>, len: usize) {
    flags.extend((0..len).map(|idx| mask.and_then(|mask| mask.is_masked(idx)).unwrap_or(false)));
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

fn read_hsd_file_bytes_bounded(path: &Path) -> Result<Vec<u8>> {
    if file_has_bzip2_magic(path)? {
        return read_bzip2_file_bounded(path, MAX_HSD_FILE_BYTES as usize);
    }

    let mut file = File::open(path).map_err(|err| {
        RustySatError::invalid_input(format!(
            "failed to open AHI HSD file '{}': {err}",
            path.display()
        ))
    })?;
    let file_len = file
        .metadata()
        .map(|metadata| metadata.len())
        .map_err(|err| {
            RustySatError::invalid_input(format!(
                "failed to inspect AHI HSD file '{}': {err}",
                path.display()
            ))
        })?;
    if file_len > MAX_HSD_FILE_BYTES {
        return Err(RustySatError::invalid_input(format!(
            "AHI HSD file '{}' is {file_len} bytes, exceeding the current safety limit of {MAX_HSD_FILE_BYTES} bytes",
            path.display()
        )));
    }
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(file_len as usize).map_err(|err| {
        RustySatError::invalid_input(format!(
            "failed to reserve memory for AHI HSD file '{}': {err}",
            path.display()
        ))
    })?;
    file.read_to_end(&mut bytes).map_err(|err| {
        RustySatError::invalid_input(format!(
            "failed to read AHI HSD file '{}': {err}",
            path.display()
        ))
    })?;
    Ok(bytes)
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

fn read_bzip2_file_bounded(path: &Path, max_decompressed_bytes: usize) -> Result<Vec<u8>> {
    let file_len = File::open(path)
        .and_then(|file| file.metadata())
        .map(|metadata| metadata.len())
        .map_err(|err| {
            RustySatError::invalid_input(format!(
                "failed to inspect compressed AHI HSD file '{}': {err}",
                path.display()
            ))
        })?;
    if file_len > MAX_HSD_FILE_BYTES {
        return Err(RustySatError::invalid_input(format!(
            "compressed AHI HSD file '{}' is {file_len} bytes, exceeding the current safety limit of {MAX_HSD_FILE_BYTES} bytes",
            path.display()
        )));
    }
    let file = File::open(path).map_err(|err| {
        RustySatError::invalid_input(format!(
            "failed to open compressed AHI HSD file '{}': {err}",
            path.display()
        ))
    })?;
    read_bzip2_reader_bounded(BufReader::new(file), max_decompressed_bytes, "AHI HSD file")
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
    use super::*;
    use crate::yaml_reader::YamlMetadataReader;
    use bzip2::write::BzEncoder;
    use bzip2::Compression;
    use rusty_sat_core::{DataQuery, DataValue, MetadataValue, Scene};
    use rusty_sat_writers::SimpleImageWriter;
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
    fn ahi_hsd_reader_drives_scene_load_and_png_output() {
        let hsd_path = temp_path("ahi_hsd_scene", "DAT");
        let png_path = temp_path("ahi_hsd_scene", "png");
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
        scene
            .save_dataset(&id, &SimpleImageWriter::default(), &png_path)
            .unwrap();

        assert!(png_path.metadata().unwrap().len() > 0);
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
        assert_eq!(radiance.values()[2], 1.0);
        assert_eq!(reflectance.values()[0], 0.0);
        assert_eq!(reflectance.values()[2], 50.0);
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
        assert_eq!(reflectance.values()[2], 50.0);
        assert_eq!(reflectance.is_masked(1), Some(true));
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
        let first_line = 1 + (u16::from(segment) - 1) * u16::from(header.data.lines);
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

    fn temp_path(name: &str, extension: &str) -> std::path::PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time")
            .as_nanos();
        std::env::temp_dir().join(format!("rusty_sat_{name}_{nanos}.{extension}"))
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
