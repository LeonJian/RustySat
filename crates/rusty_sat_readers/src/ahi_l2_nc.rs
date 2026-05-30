//! Himawari AHI L2 NetCDF reader metadata foundation.
//!
//! Reference behavior inspected before implementation:
//! - `satpy/satpy/readers/ahi_l2_nc.py`
//! - `satpy/satpy/etc/readers/ahi_l2_nc.yaml`
//! - `satpy/satpy/tests/reader_tests/test_ahi_l2_nc.py`
//!
//! This module intentionally starts with Satpy-compatible metadata and
//! inventory behavior over the portable `NetCdfFixtureSource`. Native NetCDF
//! IO, variable loading, and full output workflows remain separate roadmap
//! slices.

use crate::{
    NetCdfContent, NetCdfDataSource, NetCdfFileHandler, NetCdfFileTypeInfo, NetCdfFixtureSource,
    Reader,
};
use rusty_sat_core::{
    AnyDataArray, Coordinate, DataArray, DataId, Dataset, MetadataValue, NumericElement,
    ReaderInventory, Result, RustySatError, ValidityMask,
};
use std::collections::BTreeMap;
use std::path::Path;

const EXPECTED_DATA_AREA: &str = "Full Disk";
const FULL_DISK_SIZE: usize = 5500;
const AHI_L2_AREA_EXTENT: [f64; 4] = [
    -5_499_999.9012,
    -5_499_999.9012,
    5_499_999.9012,
    5_499_999.9012,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AhiL2NcFileType {
    Mask,
    Type,
    Height,
}

impl AhiL2NcFileType {
    pub fn name(self) -> &'static str {
        match self {
            Self::Mask => "ahi_l2_mask",
            Self::Type => "ahi_l2_type",
            Self::Height => "ahi_l2_height",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AhiL2DatasetDef {
    name: &'static str,
    file_key: &'static str,
    file_types: &'static [AhiL2NcFileType],
}

impl AhiL2DatasetDef {
    pub fn name(&self) -> &'static str {
        self.name
    }

    pub fn file_key(&self) -> &'static str {
        self.file_key
    }

    pub fn supports_file_type(&self, file_type: AhiL2NcFileType) -> bool {
        self.file_types.contains(&file_type)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AhiL2NcFileHandler {
    file_handler: NetCdfFileHandler,
    file_type: AhiL2NcFileType,
    sensor: String,
    platform_name: String,
    platform_shortname: String,
    start_time: String,
    end_time: String,
    rows: usize,
    columns: usize,
}

impl AhiL2NcFileHandler {
    pub fn from_source(
        filename: impl Into<String>,
        filename_info: BTreeMap<String, MetadataValue>,
        file_type: AhiL2NcFileType,
        source: &impl NetCdfDataSource,
    ) -> Result<Self> {
        let file_handler = NetCdfFileHandler::from_source(
            filename,
            filename_info,
            NetCdfFileTypeInfo::new(),
            source,
        )?;
        Self::new(file_handler, file_type)
    }

    pub fn new(file_handler: NetCdfFileHandler, file_type: AhiL2NcFileType) -> Result<Self> {
        let data_area = required_global_attr(&file_handler, "cdm_data_type")?;
        if data_area != EXPECTED_DATA_AREA {
            return Err(RustySatError::invalid_input(
                "File is not a full disk scene",
            ));
        }
        let sensor = required_global_attr(&file_handler, "instrument_name")?.to_ascii_lowercase();
        let platform_name = required_global_attr(&file_handler, "satellite_name")?.to_string();
        let start_time = required_global_attr(&file_handler, "time_coverage_start")?.to_string();
        let end_time = required_global_attr(&file_handler, "time_coverage_end")?.to_string();
        let platform_shortname = file_handler
            .filename_info()
            .get("platform")
            .and_then(MetadataValue::as_str)
            .ok_or_else(|| {
                RustySatError::invalid_input("AHI L2 NetCDF filename info requires 'platform'")
            })?
            .to_string();
        let rows = dimension_length(&file_handler, "Rows")?;
        let columns = dimension_length(&file_handler, "Columns")?;

        Ok(Self {
            file_handler,
            file_type,
            sensor,
            platform_name,
            platform_shortname,
            start_time,
            end_time,
            rows,
            columns,
        })
    }

    pub fn file_handler(&self) -> &NetCdfFileHandler {
        &self.file_handler
    }

    pub fn file_type(&self) -> AhiL2NcFileType {
        self.file_type
    }

    pub fn sensor(&self) -> &str {
        &self.sensor
    }

    pub fn platform_name(&self) -> &str {
        &self.platform_name
    }

    pub fn platform_shortname(&self) -> &str {
        &self.platform_shortname
    }

    pub fn start_time(&self) -> &str {
        &self.start_time
    }

    pub fn end_time(&self) -> &str {
        &self.end_time
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn columns(&self) -> usize {
        self.columns
    }

    pub fn available_dataset_defs(&self) -> Vec<&'static AhiL2DatasetDef> {
        AHI_L2_DATASETS
            .iter()
            .filter(|dataset| dataset.supports_file_type(self.file_type))
            .collect()
    }

    pub fn dataset_def(&self, name: &str) -> Option<&'static AhiL2DatasetDef> {
        self.available_dataset_defs()
            .into_iter()
            .find(|dataset| dataset.name() == name)
    }

    pub fn area_metadata_value(&self) -> Result<MetadataValue> {
        if self.rows != FULL_DISK_SIZE || self.columns != FULL_DISK_SIZE {
            return Err(RustySatError::invalid_input(
                "Input L2 file is not a full disk Himawari scene. Only full disk data is supported.",
            ));
        }
        Ok(MetadataValue::map([
            ("type", MetadataValue::string("area")),
            ("id", MetadataValue::string("Himawari_Area")),
            ("description", MetadataValue::string("AHI Full Disk area")),
            (
                "proj_id",
                MetadataValue::string(format!("geos{}", self.platform_shortname)),
            ),
            (
                "projection",
                MetadataValue::Map(ahi_l2_projection_metadata()),
            ),
            ("height", MetadataValue::Integer(self.rows as i64)),
            ("width", MetadataValue::Integer(self.columns as i64)),
            (
                "area_extent",
                MetadataValue::List(
                    AHI_L2_AREA_EXTENT
                        .into_iter()
                        .map(MetadataValue::float)
                        .collect::<Result<Vec<_>>>()?,
                ),
            ),
        ]))
    }

    pub fn load_dataset(
        &self,
        dataset_name: &str,
        source: &impl NetCdfDataSource,
    ) -> Result<Dataset> {
        let def = self.dataset_def(dataset_name).ok_or_else(|| {
            RustySatError::not_found(format!("AHI L2 NetCDF dataset '{dataset_name}'"))
        })?;
        let array = self
            .file_handler
            .load_variable_array(def.file_key(), source)?;
        let array = mask_ahi_l2_array(array, |key| {
            self.file_handler
                .attr(&format!("{}/attr/{key}", def.file_key()))
        })?
        .with_renamed_dims([("Rows", "y"), ("Columns", "x")])?;
        array.require_dims_exact(&["y", "x"])?;
        let (height, width) = array.shape_yx()?;
        let array = attach_projection_coordinates_to_any_array(array, AHI_L2_AREA_EXTENT)?;

        let mut dataset = Dataset::new(DataId::new(def.name())?).with_array(array);
        self.attach_common_dataset_attrs(&mut dataset, def)?;
        dataset.insert_attr("area", self.area_metadata_value_for_shape(height, width)?)?;
        Ok(dataset)
    }

    fn area_metadata_value_for_shape(&self, height: usize, width: usize) -> Result<MetadataValue> {
        Ok(MetadataValue::map([
            ("type", MetadataValue::string("area")),
            ("id", MetadataValue::string("Himawari_Area")),
            ("description", MetadataValue::string("AHI Full Disk area")),
            (
                "proj_id",
                MetadataValue::string(format!("geos{}", self.platform_shortname)),
            ),
            (
                "projection",
                MetadataValue::Map(ahi_l2_projection_metadata()),
            ),
            ("height", MetadataValue::Integer(height as i64)),
            ("width", MetadataValue::Integer(width as i64)),
            (
                "area_extent",
                MetadataValue::List(
                    AHI_L2_AREA_EXTENT
                        .into_iter()
                        .map(MetadataValue::float)
                        .collect::<Result<Vec<_>>>()?,
                ),
            ),
        ]))
    }

    fn attach_common_dataset_attrs(
        &self,
        dataset: &mut Dataset,
        def: &AhiL2DatasetDef,
    ) -> Result<()> {
        dataset.insert_attr("reader", "ahi_l2_nc")?;
        dataset.insert_attr("file_type", self.file_type.name())?;
        dataset.insert_attr("filename", self.file_handler.filename().to_string())?;
        dataset.insert_attr("variable", def.file_key())?;
        dataset.insert_attr("sensor", self.sensor.clone())?;
        dataset.insert_attr("platform_name", self.platform_name.clone())?;
        dataset.insert_attr("platform_shortname", self.platform_shortname.clone())?;
        dataset.insert_attr("start_time", self.start_time.clone())?;
        dataset.insert_attr("end_time", self.end_time.clone())?;
        let attr_prefix = format!("{}/attr/", def.file_key());
        for (path, content) in self.file_handler.metadata().iter() {
            if let Some(key) = path.strip_prefix(&attr_prefix) {
                if let NetCdfContent::Attribute(value) = content {
                    dataset.insert_attr(key, value.clone())?;
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct AhiL2NcFixtureReader {
    name: String,
    source: NetCdfFixtureSource,
    handler: AhiL2NcFileHandler,
}

impl AhiL2NcFixtureReader {
    pub fn from_fixture_path(
        filename: impl AsRef<Path>,
        filename_info: BTreeMap<String, MetadataValue>,
        file_type: AhiL2NcFileType,
    ) -> Result<Self> {
        let filename = filename.as_ref();
        let source = NetCdfFixtureSource::from_path(filename)?;
        Self::from_source(filename.to_string_lossy(), filename_info, file_type, source)
    }

    pub fn from_source(
        filename: impl Into<String>,
        filename_info: BTreeMap<String, MetadataValue>,
        file_type: AhiL2NcFileType,
        source: NetCdfFixtureSource,
    ) -> Result<Self> {
        let handler = AhiL2NcFileHandler::from_source(filename, filename_info, file_type, &source)?;
        Ok(Self {
            name: "ahi_l2_nc".to_string(),
            source,
            handler,
        })
    }

    pub fn source(&self) -> &impl NetCdfDataSource {
        &self.source
    }

    pub fn handler(&self) -> &AhiL2NcFileHandler {
        &self.handler
    }

    pub fn inventory(&self) -> Result<ReaderInventory> {
        ReaderInventory::new(self.name.clone(), self.available_dataset_ids())
    }
}

impl Reader for AhiL2NcFixtureReader {
    fn name(&self) -> &str {
        &self.name
    }

    fn available_dataset_ids(&self) -> Vec<DataId> {
        self.handler
            .available_dataset_defs()
            .into_iter()
            .filter_map(|dataset| DataId::new(dataset.name()).ok())
            .collect()
    }

    fn load(&self, id: &DataId) -> Result<Dataset> {
        if self.handler.dataset_def(id.name()).is_none() {
            return Err(RustySatError::not_found(format!(
                "AHI L2 NetCDF dataset '{}'",
                id.name()
            )));
        }
        self.handler.load_dataset(id.name(), &self.source)
    }
}

fn required_global_attr<'a>(handler: &'a NetCdfFileHandler, key: &str) -> Result<&'a str> {
    handler
        .attr(&format!("/attr/{key}"))
        .and_then(MetadataValue::as_str)
        .ok_or_else(|| RustySatError::not_found(format!("AHI L2 NetCDF global attribute '{key}'")))
}

fn dimension_length(handler: &NetCdfFileHandler, name: &str) -> Result<usize> {
    handler
        .get(&format!("dimension/{name}"))
        .and_then(NetCdfContent::as_dimension_length)
        .ok_or_else(|| RustySatError::not_found(format!("AHI L2 NetCDF dimension '{name}'")))
}

fn mask_ahi_l2_array<'a>(
    array: AnyDataArray,
    attr: impl Fn(&str) -> Option<&'a MetadataValue>,
) -> Result<AnyDataArray> {
    let valid_range = optional_valid_range(attr("valid_range"))?;
    let fill_value = attr("_FillValue").and_then(metadata_as_f64);
    if valid_range.is_none() && fill_value.is_none() && array.mask().is_none() {
        return Ok(array);
    }

    let mut mask = array
        .mask()
        .cloned()
        .unwrap_or_else(|| ValidityMask::all_valid(array.len()));
    let values = array.values_as_f64();
    for (idx, value) in values.iter().copied().enumerate() {
        if valid_range.is_some_and(|(min, max)| value < min || value > max)
            || fill_value.is_some_and(|fill| nearly_equal(value, fill))
        {
            mask.set_masked(idx, true);
        }
    }

    if mask.masked_count() == 0 && array.mask().is_none() {
        return Ok(array);
    }
    attach_mask(array, mask)
}

fn attach_mask(array: AnyDataArray, mask: ValidityMask) -> Result<AnyDataArray> {
    Ok(match array {
        AnyDataArray::F32(array) => array.with_mask(mask)?.into(),
        AnyDataArray::F64(array) => array.with_mask(mask)?.into(),
        AnyDataArray::U8(array) => array.with_mask(mask)?.into(),
        AnyDataArray::U16(array) => array.with_mask(mask)?.into(),
        AnyDataArray::I16(array) => array.with_mask(mask)?.into(),
    })
}

fn optional_valid_range(value: Option<&MetadataValue>) -> Result<Option<(f64, f64)>> {
    let Some(MetadataValue::List(values)) = value else {
        return Ok(None);
    };
    if values.len() != 2 {
        return Err(RustySatError::invalid_input(
            "AHI L2 NetCDF valid_range must contain exactly two values",
        ));
    }
    let min = metadata_as_f64(&values[0]).ok_or_else(|| {
        RustySatError::invalid_input("AHI L2 NetCDF valid_range minimum must be numeric")
    })?;
    let max = metadata_as_f64(&values[1]).ok_or_else(|| {
        RustySatError::invalid_input("AHI L2 NetCDF valid_range maximum must be numeric")
    })?;
    if min > max {
        return Err(RustySatError::invalid_input(
            "AHI L2 NetCDF valid_range minimum cannot exceed maximum",
        ));
    }
    Ok(Some((min, max)))
}

fn metadata_as_f64(value: &MetadataValue) -> Option<f64> {
    match value {
        MetadataValue::Integer(value) => Some(*value as f64),
        MetadataValue::Float(value) => Some(value.get()),
        _ => None,
    }
}

fn nearly_equal(left: f64, right: f64) -> bool {
    (left - right).abs() <= f64::EPSILON
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

fn ahi_l2_projection_metadata() -> BTreeMap<String, MetadataValue> {
    BTreeMap::from([
        ("a".to_string(), MetadataValue::string("6378137")),
        ("b".to_string(), MetadataValue::string("6356752.3")),
        ("h".to_string(), MetadataValue::string("35785863")),
        ("lon_0".to_string(), MetadataValue::string("140.7")),
        ("proj".to_string(), MetadataValue::string("geos")),
        ("rf".to_string(), MetadataValue::string("298.257024882273")),
        ("units".to_string(), MetadataValue::string("m")),
        ("x_0".to_string(), MetadataValue::string("0")),
        ("y_0".to_string(), MetadataValue::string("0")),
    ])
}

pub fn ahi_l2_dataset_defs() -> &'static [AhiL2DatasetDef] {
    AHI_L2_DATASETS
}

const MASK_ONLY: &[AhiL2NcFileType] = &[AhiL2NcFileType::Mask];
const TYPE_ONLY: &[AhiL2NcFileType] = &[AhiL2NcFileType::Type];
const HEIGHT_ONLY: &[AhiL2NcFileType] = &[AhiL2NcFileType::Height];
const ALL_TYPES: &[AhiL2NcFileType] = &[
    AhiL2NcFileType::Height,
    AhiL2NcFileType::Type,
    AhiL2NcFileType::Mask,
];

const AHI_L2_DATASETS: &[AhiL2DatasetDef] = &[
    AhiL2DatasetDef {
        name: "cloud_mask",
        file_key: "CloudMask",
        file_types: MASK_ONLY,
    },
    AhiL2DatasetDef {
        name: "cloud_mask_binary",
        file_key: "CloudMaskBinary",
        file_types: MASK_ONLY,
    },
    AhiL2DatasetDef {
        name: "cloud_probability",
        file_key: "CloudProbability",
        file_types: MASK_ONLY,
    },
    AhiL2DatasetDef {
        name: "ice_cloud_probability",
        file_key: "IceCloudProbability",
        file_types: MASK_ONLY,
    },
    AhiL2DatasetDef {
        name: "phase_uncertainty",
        file_key: "PhaseUncertainty",
        file_types: MASK_ONLY,
    },
    AhiL2DatasetDef {
        name: "dust_mask",
        file_key: "Dust_Mask",
        file_types: MASK_ONLY,
    },
    AhiL2DatasetDef {
        name: "fire_mask",
        file_key: "Fire_Mask",
        file_types: MASK_ONLY,
    },
    AhiL2DatasetDef {
        name: "smoke_mask",
        file_key: "Smoke_Mask",
        file_types: MASK_ONLY,
    },
    AhiL2DatasetDef {
        name: "cloud_phase",
        file_key: "CloudPhase",
        file_types: TYPE_ONLY,
    },
    AhiL2DatasetDef {
        name: "cloud_phase_flag",
        file_key: "CloudPhaseFlag",
        file_types: TYPE_ONLY,
    },
    AhiL2DatasetDef {
        name: "cloud_type",
        file_key: "CloudType",
        file_types: TYPE_ONLY,
    },
    AhiL2DatasetDef {
        name: "cloud_optical_depth",
        file_key: "CldOptDpth",
        file_types: HEIGHT_ONLY,
    },
    AhiL2DatasetDef {
        name: "cloud_top_emissivity",
        file_key: "CldTopEmss",
        file_types: HEIGHT_ONLY,
    },
    AhiL2DatasetDef {
        name: "cloud_top_height",
        file_key: "CldTopHght",
        file_types: HEIGHT_ONLY,
    },
    AhiL2DatasetDef {
        name: "cloud_top_pressure",
        file_key: "CldTopPres",
        file_types: HEIGHT_ONLY,
    },
    AhiL2DatasetDef {
        name: "cloud_top_pressure_low",
        file_key: "CldTopPresLow",
        file_types: HEIGHT_ONLY,
    },
    AhiL2DatasetDef {
        name: "cloud_top_temperature",
        file_key: "CldTopTemp",
        file_types: HEIGHT_ONLY,
    },
    AhiL2DatasetDef {
        name: "cloud_top_temperature_low",
        file_key: "CldTopTempLow",
        file_types: HEIGHT_ONLY,
    },
    AhiL2DatasetDef {
        name: "cloud_height_quality",
        file_key: "CloudHgtQF",
        file_types: HEIGHT_ONLY,
    },
    AhiL2DatasetDef {
        name: "retrieval_cost",
        file_key: "Cost",
        file_types: HEIGHT_ONLY,
    },
    AhiL2DatasetDef {
        name: "inversion_flag",
        file_key: "InverFlag",
        file_types: HEIGHT_ONLY,
    },
    AhiL2DatasetDef {
        name: "latitude_parallax_corrected",
        file_key: "Latitude_Pc",
        file_types: HEIGHT_ONLY,
    },
    AhiL2DatasetDef {
        name: "longitude_parallax_corrected",
        file_key: "Longitude_Pc",
        file_types: HEIGHT_ONLY,
    },
    AhiL2DatasetDef {
        name: "cloud_top_pressure_error",
        file_key: "PcError",
        file_types: HEIGHT_ONLY,
    },
    AhiL2DatasetDef {
        name: "processing_order",
        file_key: "ProcOrder",
        file_types: HEIGHT_ONLY,
    },
    AhiL2DatasetDef {
        name: "shadow_mask",
        file_key: "Shadow_Mask",
        file_types: HEIGHT_ONLY,
    },
    AhiL2DatasetDef {
        name: "cloud_top_temperature_error",
        file_key: "TcError",
        file_types: HEIGHT_ONLY,
    },
    AhiL2DatasetDef {
        name: "cloud_top_height_error",
        file_key: "ZcError",
        file_types: HEIGHT_ONLY,
    },
    AhiL2DatasetDef {
        name: "latitude",
        file_key: "Latitude",
        file_types: ALL_TYPES,
    },
    AhiL2DatasetDef {
        name: "longitude",
        file_key: "Longitude",
        file_types: ALL_TYPES,
    },
];

#[cfg(test)]
mod tests {
    use super::*;
    use rusty_sat_core::{DataQuery, Scene};
    use rusty_sat_resample::{
        area_from_metadata_value, resample_dataset_from_attrs, source_geometry_from_dataset,
        ResampleOptions, SourceGeometry,
    };
    use rusty_sat_writers::SimpleImageWriter;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    const AHI_L2_FULL_DISK_FIXTURE: &str = r#"
attrs:
  time_coverage_start: "2023-08-24T05:40:21Z"
  time_coverage_end: "2023-08-24T05:49:40Z"
  instrument_name: AHI
  satellite_name: Himawari-9
  cdm_data_type: Full Disk
dimensions:
  Rows: 5500
  Columns: 5500
variables:
  CloudMask:
    dtype: u16
    dimensions: [Rows, Columns]
    shape: [5500, 5500]
  Latitude:
    dtype: f32
    dimensions: [Rows, Columns]
    shape: [5500, 5500]
  Longitude:
    dtype: f32
    dimensions: [Rows, Columns]
    shape: [5500, 5500]
"#;

    const AHI_L2_SMALL_DATA_FIXTURE: &str = r#"
attrs:
  time_coverage_start: "2023-08-24T05:40:21Z"
  time_coverage_end: "2023-08-24T05:49:40Z"
  instrument_name: AHI
  satellite_name: Himawari-9
  cdm_data_type: Full Disk
dimensions:
  Rows: 2
  Columns: 3
variables:
  CloudMask:
    dtype: u16
    dimensions: [Rows, Columns]
    shape: [2, 3]
    attrs:
      units: "1"
      _FillValue: 65535
      valid_range: [0, 2]
    values: [0, 1, 2, 3, 65535, 1]
  CloudProbability:
    dtype: f32
    dimensions: [Rows, Columns]
    shape: [2, 3]
    attrs:
      units: "1"
    values: [0.0, 0.25, 0.5, 0.75, 1.0, 0.1]
  Latitude:
    dtype: f32
    dimensions: [Rows, Columns]
    shape: [2, 3]
    values: [10.0, 10.1, 10.2, 9.9, 9.8, 9.7]
  Longitude:
    dtype: f32
    dimensions: [Rows, Columns]
    shape: [2, 3]
    values: [140.0, 140.1, 140.2, 139.9, 139.8, 139.7]
"#;

    fn filename_info() -> BTreeMap<String, MetadataValue> {
        BTreeMap::from([("platform".to_string(), MetadataValue::string("h09"))])
    }

    #[test]
    fn ahi_l2_metadata_matches_satpy_handler_attrs() {
        let source = NetCdfFixtureSource::from_yaml_str(AHI_L2_FULL_DISK_FIXTURE).unwrap();
        let reader = AhiL2NcFixtureReader::from_source(
            "AHI-CMSK_v1r1_h09_s202308240540213_e202308240549407_c202308240557548.nc",
            filename_info(),
            AhiL2NcFileType::Mask,
            source,
        )
        .unwrap();
        let handler = reader.handler();

        assert_eq!(reader.name(), "ahi_l2_nc");
        assert_eq!(handler.sensor(), "ahi");
        assert_eq!(handler.platform_name(), "Himawari-9");
        assert_eq!(handler.platform_shortname(), "h09");
        assert_eq!(handler.start_time(), "2023-08-24T05:40:21Z");
        assert_eq!(handler.end_time(), "2023-08-24T05:49:40Z");
        assert_eq!(handler.rows(), 5500);
        assert_eq!(handler.columns(), 5500);
    }

    #[test]
    fn ahi_l2_inventory_follows_satpy_yaml_file_type_mapping() {
        let source = NetCdfFixtureSource::from_yaml_str(AHI_L2_FULL_DISK_FIXTURE).unwrap();
        let reader = AhiL2NcFixtureReader::from_source(
            "fixture.yaml",
            filename_info(),
            AhiL2NcFileType::Mask,
            source,
        )
        .unwrap();
        let names = reader
            .available_dataset_ids()
            .into_iter()
            .map(|id| id.name().to_string())
            .collect::<Vec<_>>();

        assert!(names.contains(&"cloud_mask".to_string()));
        assert!(names.contains(&"latitude".to_string()));
        assert!(names.contains(&"longitude".to_string()));
        assert!(!names.contains(&"cloud_type".to_string()));
        assert!(!names.contains(&"cloud_top_height".to_string()));
        assert_eq!(
            reader
                .handler()
                .dataset_def("cloud_mask")
                .unwrap()
                .file_key(),
            "CloudMask"
        );
    }

    #[test]
    fn ahi_l2_area_metadata_matches_satpy_full_disk_assumption() {
        let source = NetCdfFixtureSource::from_yaml_str(AHI_L2_FULL_DISK_FIXTURE).unwrap();
        let reader = AhiL2NcFixtureReader::from_source(
            "fixture.yaml",
            filename_info(),
            AhiL2NcFileType::Mask,
            source,
        )
        .unwrap();

        let area = reader.handler().area_metadata_value().unwrap();

        assert_eq!(
            area.get_path(&["id"]).and_then(MetadataValue::as_str),
            Some("Himawari_Area")
        );
        assert_eq!(
            area.get_path(&["proj_id"]).and_then(MetadataValue::as_str),
            Some("geosh09")
        );
        assert_eq!(
            area.get_path(&["projection", "proj"])
                .and_then(MetadataValue::as_str),
            Some("geos")
        );
        assert_eq!(
            area.get_path(&["projection", "lon_0"])
                .and_then(MetadataValue::as_str),
            Some("140.7")
        );
        let MetadataValue::List(extent) = area.get_path(&["area_extent"]).unwrap() else {
            panic!("expected area extent list");
        };
        assert_eq!(extent.len(), 4);
    }

    #[test]
    fn ahi_l2_rejects_non_full_disk_product() {
        let fixture =
            AHI_L2_FULL_DISK_FIXTURE.replace("cdm_data_type: Full Disk", "cdm_data_type: CONUS");
        let source = NetCdfFixtureSource::from_yaml_str(&fixture).unwrap();
        let err = AhiL2NcFixtureReader::from_source(
            "fixture.yaml",
            filename_info(),
            AhiL2NcFileType::Mask,
            source,
        )
        .unwrap_err();

        assert!(err.to_string().contains("File is not a full disk scene"));
    }

    #[test]
    fn ahi_l2_area_rejects_non_nominal_shape() {
        let fixture = AHI_L2_FULL_DISK_FIXTURE
            .replace("Rows: 5500", "Rows: 3000")
            .replace("shape: [5500, 5500]", "shape: [3000, 5500]");
        let source = NetCdfFixtureSource::from_yaml_str(&fixture).unwrap();
        let reader = AhiL2NcFixtureReader::from_source(
            "fixture.yaml",
            filename_info(),
            AhiL2NcFileType::Mask,
            source,
        )
        .unwrap();

        let err = reader.handler().area_metadata_value().unwrap_err();

        assert!(err.to_string().contains("Only full disk data is supported"));
    }

    #[test]
    fn ahi_l2_fixture_reader_drives_scene_planning() {
        let source = NetCdfFixtureSource::from_yaml_str(AHI_L2_FULL_DISK_FIXTURE).unwrap();
        let reader = AhiL2NcFixtureReader::from_source(
            "fixture.yaml",
            filename_info(),
            AhiL2NcFileType::Mask,
            source,
        )
        .unwrap();
        let inventory = reader.inventory().unwrap();
        let mut scene = Scene::new();

        let plan = scene
            .plan_reader_loads([DataQuery::named("cloud_mask").unwrap()], [&inventory])
            .unwrap();

        assert_eq!(plan.reader_datasets().get("ahi_l2_nc").unwrap().len(), 1);
    }

    #[test]
    fn ahi_l2_loads_dataset_with_rows_columns_renamed_to_yx() {
        let source = NetCdfFixtureSource::from_yaml_str(AHI_L2_SMALL_DATA_FIXTURE).unwrap();
        let reader = AhiL2NcFixtureReader::from_source(
            "fixture.yaml",
            filename_info(),
            AhiL2NcFileType::Mask,
            source,
        )
        .unwrap();
        let id = DataId::new("cloud_mask").unwrap();

        let dataset = reader.load(&id).unwrap();

        assert_eq!(
            dataset.attr("reader").and_then(MetadataValue::as_str),
            Some("ahi_l2_nc")
        );
        assert_eq!(
            dataset.attr("variable").and_then(MetadataValue::as_str),
            Some("CloudMask")
        );
        assert_eq!(
            dataset.attr("sensor").and_then(MetadataValue::as_str),
            Some("ahi")
        );
        assert_eq!(
            dataset
                .attr("platform_name")
                .and_then(MetadataValue::as_str),
            Some("Himawari-9")
        );
        assert_eq!(
            dataset.attr("units").and_then(MetadataValue::as_str),
            Some("1")
        );
        let rusty_sat_core::AnyDataArray::U16(array) = dataset.array().unwrap() else {
            panic!("expected u16 cloud mask");
        };
        assert_eq!(array.shape_nd(), &[2, 3]);
        assert_eq!(array.dims(), &["y".to_string(), "x".to_string()]);
        assert_eq!(array.coord("x").unwrap().values().len(), 3);
        assert_eq!(array.coord("y").unwrap().values().len(), 2);
        assert_eq!(array.values(), &[0, 1, 2, 3, 65535, 1]);
        assert_eq!(array.is_masked(2), Some(false));
        assert_eq!(array.is_masked(3), Some(true));
        assert_eq!(array.is_masked(4), Some(true));
        let SourceGeometry::Area(area) = source_geometry_from_dataset(&dataset).unwrap() else {
            panic!("expected area source geometry");
        };
        assert_eq!(area.shape(), (2, 3));
    }

    #[test]
    fn ahi_l2_load_preserves_float_dtype_without_mask_when_no_mask_attrs_exist() {
        let source = NetCdfFixtureSource::from_yaml_str(AHI_L2_SMALL_DATA_FIXTURE).unwrap();
        let reader = AhiL2NcFixtureReader::from_source(
            "fixture.yaml",
            filename_info(),
            AhiL2NcFileType::Mask,
            source,
        )
        .unwrap();
        let id = DataId::new("cloud_probability").unwrap();

        let dataset = reader.load(&id).unwrap();

        let rusty_sat_core::AnyDataArray::F32(array) = dataset.array().unwrap() else {
            panic!("expected f32 cloud probability");
        };
        assert_eq!(array.dims(), &["y".to_string(), "x".to_string()]);
        assert_eq!(array.values()[3], 0.75);
        assert!(array.mask().is_none());
    }

    #[test]
    fn ahi_l2_loaded_dataset_area_attr_and_xy_coords_drive_resampling_pipeline() {
        let fixture = AHI_L2_SMALL_DATA_FIXTURE.replacen("dtype: f32", "dtype: f64", 1);
        let source = NetCdfFixtureSource::from_yaml_str(&fixture).unwrap();
        let reader = AhiL2NcFixtureReader::from_source(
            "fixture.yaml",
            filename_info(),
            AhiL2NcFileType::Mask,
            source,
        )
        .unwrap();
        let dataset = reader
            .load(&DataId::new("cloud_probability").unwrap())
            .unwrap();

        let area = area_from_metadata_value(dataset.attr("area").unwrap()).unwrap();
        let array = dataset.array().unwrap();
        assert_eq!(
            array.coord("x").unwrap().values(),
            &[-3_666_666.6008, 0.0, 3_666_666.6008]
        );
        assert_eq!(
            array.coord("y").unwrap().values(),
            &[2_749_999.9506, -2_749_999.9506]
        );

        let resampled =
            resample_dataset_from_attrs(&dataset, &area, ResampleOptions::default()).unwrap();

        let rusty_sat_core::AnyDataArray::F64(output) = resampled.array().unwrap() else {
            panic!("expected f64 output");
        };
        assert_eq!(output.dims(), &["y".to_string(), "x".to_string()]);
        assert_eq!(output.values(), &[0.0, 0.25, 0.5, 0.75, 1.0, 0.1]);
    }

    #[test]
    fn ahi_l2_scene_load_resample_and_png_output_vertical_slice() {
        let fixture = AHI_L2_SMALL_DATA_FIXTURE.replace(
            "CloudProbability:\n    dtype: f32",
            "CloudProbability:\n    dtype: f64",
        );
        let source = NetCdfFixtureSource::from_yaml_str(&fixture).unwrap();
        let reader = AhiL2NcFixtureReader::from_source(
            "fixture.yaml",
            filename_info(),
            AhiL2NcFileType::Mask,
            source,
        )
        .unwrap();
        let inventory = reader.inventory().unwrap();
        let mut scene = Scene::new();
        let plan = scene
            .plan_reader_loads(
                [DataQuery::named("cloud_probability").unwrap()],
                [&inventory],
            )
            .unwrap();
        let planned = plan.reader_datasets().get("ahi_l2_nc").unwrap();
        let id = planned.iter().next().unwrap().clone();
        let dataset = reader.load(&id).unwrap();
        let area = area_from_metadata_value(dataset.attr("area").unwrap()).unwrap();
        scene.insert_dataset(dataset);

        let resampled =
            resample_dataset_from_attrs(scene.get(&id).unwrap(), &area, ResampleOptions::default())
                .unwrap();
        scene.insert_dataset(resampled);
        let output_path = temp_png_path("ahi_l2_scene_output");

        scene
            .save_dataset(&id, &SimpleImageWriter::default(), &output_path)
            .unwrap();

        let bytes = fs::read(&output_path).unwrap();
        assert_eq!(&bytes[..8], b"\x89PNG\r\n\x1a\n");
        fs::remove_file(output_path).ok();
    }

    #[test]
    fn ahi_l2_load_rejects_multidimensional_product_like_satpy_reader_note() {
        let fixture = r#"
attrs:
  time_coverage_start: "2023-08-24T05:40:21Z"
  time_coverage_end: "2023-08-24T05:49:40Z"
  instrument_name: AHI
  satellite_name: Himawari-9
  cdm_data_type: Full Disk
dimensions:
  Rows: 2
  Columns: 3
  Layer: 2
variables:
  CloudProbability:
    dtype: f32
    dimensions: [Rows, Columns, Layer]
    shape: [2, 3, 2]
    values: [0.0, 0.25, 0.5, 0.75, 1.0, 0.1, 0.0, 0.25, 0.5, 0.75, 1.0, 0.1]
  Latitude:
    dtype: f32
    dimensions: [Rows, Columns]
    shape: [2, 3]
    values: [10.0, 10.1, 10.2, 9.9, 9.8, 9.7]
  Longitude:
    dtype: f32
    dimensions: [Rows, Columns]
    shape: [2, 3]
    values: [140.0, 140.1, 140.2, 139.9, 139.8, 139.7]
"#;
        let source = NetCdfFixtureSource::from_yaml_str(&fixture).unwrap();
        let reader = AhiL2NcFixtureReader::from_source(
            "fixture.yaml",
            filename_info(),
            AhiL2NcFileType::Mask,
            source,
        )
        .unwrap();
        let id = DataId::new("cloud_probability").unwrap();

        let err = reader.load(&id).unwrap_err();

        assert!(err.to_string().contains("do not match expected"));
    }

    fn temp_png_path(name: &str) -> std::path::PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "rusty_sat_{name}_{}_{}.png",
            std::process::id(),
            nanos
        ))
    }
}
