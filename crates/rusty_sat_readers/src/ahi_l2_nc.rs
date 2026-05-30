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
use rusty_sat_core::{DataId, Dataset, MetadataValue, ReaderInventory, Result, RustySatError};
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
        let Some(def) = self.handler.dataset_def(id.name()) else {
            return Err(RustySatError::not_found(format!(
                "AHI L2 NetCDF dataset '{}'",
                id.name()
            )));
        };
        Err(RustySatError::unsupported(format!(
            "AHI L2 NetCDF data loading for '{}' ({})",
            def.name(),
            def.file_key()
        )))
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
}
