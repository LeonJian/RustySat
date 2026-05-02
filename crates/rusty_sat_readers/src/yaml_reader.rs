//! Satpy-style YAML reader metadata parsing.
//!
//! Reference behavior inspected before implementation:
//! - `satpy/doc/source/dev_guide/custom_reader.rst`
//! - `satpy/satpy/readers/core/yaml_reader.py`
//! - `satpy/satpy/etc/readers/seviri_l1b_nc.yaml`

use crate::Reader;
use rusty_sat_core::{
    DataId, Dataset, ModifierTuple, ReaderInventory, Result, RustySatError, WavelengthRange,
};
use serde_yaml::{Mapping, Value};
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReaderInfo {
    name: String,
    short_name: Option<String>,
    long_name: Option<String>,
    sensors: Vec<String>,
    supports_fsspec: Option<bool>,
}

impl ReaderInfo {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn short_name(&self) -> Option<&str> {
        self.short_name.as_deref()
    }

    pub fn long_name(&self) -> Option<&str> {
        self.long_name.as_deref()
    }

    pub fn sensors(&self) -> &[String] {
        &self.sensors
    }

    pub fn supports_fsspec(&self) -> Option<bool> {
        self.supports_fsspec
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileTypeConfig {
    name: String,
    file_patterns: Vec<String>,
    requires: Vec<String>,
}

impl FileTypeConfig {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn file_patterns(&self) -> &[String] {
        &self.file_patterns
    }

    pub fn requires(&self) -> &[String] {
        &self.requires
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DatasetConfig {
    name: String,
    data_ids: Vec<DataId>,
    file_type: Option<String>,
    coordinates: Vec<String>,
}

impl DatasetConfig {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn data_ids(&self) -> &[DataId] {
        &self.data_ids
    }

    pub fn file_type(&self) -> Option<&str> {
        self.file_type.as_deref()
    }

    pub fn coordinates(&self) -> &[String] {
        &self.coordinates
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct YamlReaderConfig {
    info: ReaderInfo,
    file_types: BTreeMap<String, FileTypeConfig>,
    datasets: BTreeMap<String, DatasetConfig>,
}

impl YamlReaderConfig {
    pub fn from_str(yaml: &str) -> Result<Self> {
        let value: Value = serde_yaml::from_str(yaml)
            .map_err(|err| RustySatError::invalid_input(format!("invalid reader YAML: {err}")))?;
        let mapping = value
            .as_mapping()
            .ok_or_else(|| RustySatError::invalid_input("reader YAML root must be a mapping"))?;
        let info = parse_reader_info(required_value(mapping, "reader", "reader YAML")?)?;
        let file_types = parse_file_types(optional_value(mapping, "file_types"))?;
        let datasets = parse_datasets(optional_value(mapping, "datasets"))?;
        Ok(Self {
            info,
            file_types,
            datasets,
        })
    }

    pub fn info(&self) -> &ReaderInfo {
        &self.info
    }

    pub fn file_types(&self) -> &BTreeMap<String, FileTypeConfig> {
        &self.file_types
    }

    pub fn datasets(&self) -> &BTreeMap<String, DatasetConfig> {
        &self.datasets
    }

    pub fn all_dataset_ids(&self) -> Vec<DataId> {
        self.datasets
            .values()
            .flat_map(|dataset| dataset.data_ids.iter().cloned())
            .collect()
    }
}

#[derive(Debug, Clone)]
pub struct YamlMetadataReader {
    config: YamlReaderConfig,
}

impl YamlMetadataReader {
    pub fn from_config(config: YamlReaderConfig) -> Self {
        Self { config }
    }

    pub fn from_str(yaml: &str) -> Result<Self> {
        Ok(Self::from_config(YamlReaderConfig::from_str(yaml)?))
    }

    pub fn config(&self) -> &YamlReaderConfig {
        &self.config
    }

    pub fn inventory(&self) -> Result<ReaderInventory> {
        ReaderInventory::new(self.name().to_string(), self.available_dataset_ids())
    }
}

impl Reader for YamlMetadataReader {
    fn name(&self) -> &str {
        self.config.info.name()
    }

    fn available_dataset_ids(&self) -> Vec<DataId> {
        self.config.all_dataset_ids()
    }

    fn load(&self, _id: &DataId) -> Result<Dataset> {
        Err(RustySatError::unsupported(
            "YAML metadata reader does not load dataset arrays yet",
        ))
    }
}

fn parse_reader_info(value: &Value) -> Result<ReaderInfo> {
    let mapping = value
        .as_mapping()
        .ok_or_else(|| RustySatError::invalid_input("reader section must be a mapping"))?;
    let name = parse_required_string(mapping, "name", "reader")?;
    let short_name = parse_optional_string(mapping, "short_name")?;
    let long_name = parse_optional_string(mapping, "long_name")?;
    let sensors = optional_value(mapping, "sensors")
        .map(parse_string_list)
        .transpose()?
        .unwrap_or_default();
    let supports_fsspec = optional_value(mapping, "supports_fsspec").and_then(Value::as_bool);
    Ok(ReaderInfo {
        name,
        short_name,
        long_name,
        sensors,
        supports_fsspec,
    })
}

fn parse_file_types(value: Option<&Value>) -> Result<BTreeMap<String, FileTypeConfig>> {
    let Some(value) = value else {
        return Ok(BTreeMap::new());
    };
    let mapping = value
        .as_mapping()
        .ok_or_else(|| RustySatError::invalid_input("file_types section must be a mapping"))?;
    let mut file_types = BTreeMap::new();
    for (key, value) in mapping {
        let name = key
            .as_str()
            .ok_or_else(|| RustySatError::invalid_input("file type keys must be strings"))?;
        let file_type_mapping = value.as_mapping().ok_or_else(|| {
            RustySatError::invalid_input(format!("file type '{name}' must be a mapping"))
        })?;
        let file_patterns =
            parse_string_list(required_value(file_type_mapping, "file_patterns", name)?)?;
        let requires = optional_value(file_type_mapping, "requires")
            .map(parse_string_list)
            .transpose()?
            .unwrap_or_default();
        file_types.insert(
            name.to_string(),
            FileTypeConfig {
                name: name.to_string(),
                file_patterns,
                requires,
            },
        );
    }
    Ok(file_types)
}

fn parse_datasets(value: Option<&Value>) -> Result<BTreeMap<String, DatasetConfig>> {
    let Some(value) = value else {
        return Ok(BTreeMap::new());
    };
    let mapping = value
        .as_mapping()
        .ok_or_else(|| RustySatError::invalid_input("datasets section must be a mapping"))?;
    let mut datasets = BTreeMap::new();
    for (key, value) in mapping {
        let config_key = key
            .as_str()
            .ok_or_else(|| RustySatError::invalid_input("dataset keys must be strings"))?;
        let dataset = parse_dataset(config_key, value)?;
        datasets.insert(config_key.to_string(), dataset);
    }
    Ok(datasets)
}

fn parse_dataset(config_key: &str, value: &Value) -> Result<DatasetConfig> {
    let mapping = value.as_mapping().ok_or_else(|| {
        RustySatError::invalid_input(format!("dataset '{config_key}' must be a mapping"))
    })?;
    let name = parse_optional_string(mapping, "name")?.unwrap_or_else(|| config_key.to_string());
    let file_type = parse_optional_string(mapping, "file_type")?;
    let coordinates = optional_value(mapping, "coordinates")
        .map(parse_string_list)
        .transpose()?
        .unwrap_or_default();
    let calibrations = parse_calibrations(optional_value(mapping, "calibration"))?;
    let mut data_ids = Vec::new();
    for calibration in calibrations {
        let mut data_id = DataId::new(name.clone())?;
        if let Some(resolution) = optional_value(mapping, "resolution") {
            if let Some(resolution) = resolution.as_f64() {
                data_id = data_id.with_qualifier("resolution", resolution)?;
            }
        }
        if let Some(wavelength) = optional_value(mapping, "wavelength") {
            data_id = data_id.with_qualifier("wavelength", parse_wavelength(wavelength)?)?;
        }
        if let Some(polarization) = parse_optional_string(mapping, "polarization")? {
            data_id = data_id.with_qualifier("polarization", polarization)?;
        }
        if let Some(modifiers) = optional_value(mapping, "modifiers") {
            data_id = data_id.with_qualifier("modifiers", parse_modifiers(modifiers)?)?;
        }
        if let Some(calibration) = calibration {
            data_id = data_id.with_qualifier("calibration", calibration)?;
        }
        data_ids.push(data_id);
    }
    Ok(DatasetConfig {
        name,
        data_ids,
        file_type,
        coordinates,
    })
}

fn parse_calibrations(value: Option<&Value>) -> Result<Vec<Option<String>>> {
    match value {
        Some(Value::Mapping(mapping)) => {
            let mut calibrations = Vec::new();
            for key in mapping.keys() {
                calibrations.push(Some(
                    key.as_str()
                        .ok_or_else(|| {
                            RustySatError::invalid_input("calibration keys must be strings")
                        })?
                        .to_string(),
                ));
            }
            Ok(calibrations)
        }
        Some(Value::Sequence(values)) => values
            .iter()
            .map(|value| yaml_scalar_to_string(value).map(Some))
            .collect(),
        Some(value) => Ok(vec![Some(yaml_scalar_to_string(value)?)]),
        None => Ok(vec![None]),
    }
}

fn parse_wavelength(value: &Value) -> Result<WavelengthRange> {
    let values = value
        .as_sequence()
        .ok_or_else(|| RustySatError::invalid_input("wavelength must be a 3-value list"))?;
    if values.len() != 3 {
        return Err(RustySatError::invalid_input(
            "wavelength must be a 3-value list",
        ));
    }
    WavelengthRange::micrometers(
        parse_f64(&values[0], "wavelength minimum")?,
        parse_f64(&values[1], "wavelength central")?,
        parse_f64(&values[2], "wavelength maximum")?,
    )
}

fn parse_modifiers(value: &Value) -> Result<ModifierTuple> {
    let values = parse_string_list(value)?;
    ModifierTuple::new(values)
}

fn parse_string_list(value: &Value) -> Result<Vec<String>> {
    match value {
        Value::Sequence(values) => values.iter().map(yaml_scalar_to_string).collect(),
        _ => Ok(vec![yaml_scalar_to_string(value)?]),
    }
}

fn parse_required_string(mapping: &Mapping, key: &str, section: &str) -> Result<String> {
    yaml_scalar_to_string(required_value(mapping, key, section)?)
}

fn parse_optional_string(mapping: &Mapping, key: &str) -> Result<Option<String>> {
    optional_value(mapping, key)
        .map(yaml_scalar_to_string)
        .transpose()
}

fn parse_f64(value: &Value, name: &str) -> Result<f64> {
    value
        .as_f64()
        .ok_or_else(|| RustySatError::invalid_input(format!("{name} must be numeric")))
}

fn required_value<'a>(mapping: &'a Mapping, key: &str, section: &str) -> Result<&'a Value> {
    mapping
        .get(Value::String(key.to_string()))
        .ok_or_else(|| RustySatError::invalid_input(format!("{section} missing '{key}'")))
}

fn optional_value<'a>(mapping: &'a Mapping, key: &str) -> Option<&'a Value> {
    mapping.get(Value::String(key.to_string()))
}

fn yaml_scalar_to_string(value: &Value) -> Result<String> {
    match value {
        Value::String(value) => Ok(value.clone()),
        Value::Number(value) => Ok(value.to_string()),
        Value::Bool(value) => Ok(value.to_string()),
        Value::Tagged(value) => yaml_scalar_to_string(&value.value),
        _ => Err(RustySatError::invalid_input(
            "YAML value must be a scalar string, number, or bool",
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rusty_sat_core::DataValue;

    const SEVIRI_STYLE_YAML: &str = r#"
reader:
  name: seviri_l1b_nc
  short_name: SEVIRI L1b NetCDF4
  long_name: MSG SEVIRI Level 1b NetCDF4
  supports_fsspec: true
  sensors: [seviri]
  reader: !!python/name:satpy.readers.core.yaml_reader.GEOFlippableFileYAMLReader

file_types:
  seviri_l1b_nc:
    file_reader: !!python/name:satpy.readers.seviri_l1b_nc.NCSEVIRIFileHandler
    file_patterns: ['W_XX,{satid:4s}_{start_time:%Y%m%d%H%M%S}.nc']

datasets:
  VIS006:
    name: VIS006
    resolution: 3000.403165817
    wavelength: [0.56, 0.635, 0.71]
    calibration:
      reflectance:
        standard_name: toa_bidirectional_reflectance
        units: "%"
      radiance:
        standard_name: toa_outgoing_radiance_per_unit_wavenumber
        units: mW m-2 sr-1 (cm-1)-1
      counts:
        standard_name: counts
        units: count
    file_type: seviri_l1b_nc
    coordinates: [longitude, latitude]

  longitude:
    name: longitude
    file_type: seviri_l1b_nc
"#;

    #[test]
    fn parses_satpy_style_reader_yaml_metadata() {
        let config = YamlReaderConfig::from_str(SEVIRI_STYLE_YAML).unwrap();

        assert_eq!(config.info().name(), "seviri_l1b_nc");
        assert_eq!(config.info().short_name(), Some("SEVIRI L1b NetCDF4"));
        assert_eq!(config.info().sensors(), &["seviri".to_string()]);
        assert_eq!(config.info().supports_fsspec(), Some(true));
        assert_eq!(
            config
                .file_types()
                .get("seviri_l1b_nc")
                .unwrap()
                .file_patterns(),
            &["W_XX,{satid:4s}_{start_time:%Y%m%d%H%M%S}.nc".to_string()]
        );
    }

    #[test]
    fn creates_dataset_ids_for_calibration_variants() {
        let config = YamlReaderConfig::from_str(SEVIRI_STYLE_YAML).unwrap();
        let vis006 = config.datasets().get("VIS006").unwrap();

        assert_eq!(vis006.file_type(), Some("seviri_l1b_nc"));
        assert_eq!(
            vis006.coordinates(),
            &["longitude".to_string(), "latitude".to_string()]
        );
        assert_eq!(vis006.data_ids().len(), 3);
        assert!(vis006.data_ids().iter().any(|id| {
            id.qualifier("calibration") == Some(&DataValue::Text("reflectance".to_string()))
        }));
        assert!(vis006
            .data_ids()
            .iter()
            .all(|id| { id.qualifier("resolution") == Some(&DataValue::from(3000.403165817)) }));
        assert!(vis006
            .data_ids()
            .iter()
            .all(|id| id.qualifier("wavelength").is_some()));
    }

    #[test]
    fn metadata_reader_exposes_inventory_but_not_array_loading() {
        let reader = YamlMetadataReader::from_str(SEVIRI_STYLE_YAML).unwrap();
        let ids = reader.available_dataset_ids();

        assert_eq!(reader.name(), "seviri_l1b_nc");
        assert_eq!(ids.len(), 4);
        assert_eq!(reader.inventory().unwrap().name(), "seviri_l1b_nc");
        assert!(matches!(
            reader.load(&ids[0]).unwrap_err(),
            RustySatError::Unsupported { .. }
        ));
    }
}
