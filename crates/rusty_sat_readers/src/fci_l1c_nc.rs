//! MTG FCI L1C NetCDF reader vertical slice.
//!
//! Reference behavior inspected before implementation:
//! - `satpy/satpy/readers/fci_l1c_nc.py`
//! - `satpy/doc/source/examples/fci_l1c_natural_color.rst`
//!
//! This is not the full production NetCDF/HDF backend yet. It connects the
//! FCI L1C measured-channel loader to the `Reader` trait using the portable
//! `NetCdfFixtureSource` adapter, so Scene planning can load FCI-like counts
//! datasets from real fixture files while native backend selection remains a
//! separate roadmap step.

use crate::{
    FciL1cNetCdfHandler, NetCdfDataSource, NetCdfFileHandler, NetCdfFileTypeInfo,
    NetCdfFixtureSource, Reader,
};
use rusty_sat_core::{DataId, DataValue, Dataset, ReaderInventory, Result, RustySatError};
use std::collections::BTreeMap;
use std::path::Path;

#[derive(Debug, Clone, PartialEq)]
pub struct FciL1cFixtureReader {
    name: String,
    source: NetCdfFixtureSource,
    handler: FciL1cNetCdfHandler,
    channels: Vec<String>,
}

impl FciL1cFixtureReader {
    pub fn from_fixture_path(
        filename: impl AsRef<Path>,
        channels: impl IntoIterator<Item = impl Into<String>>,
    ) -> Result<Self> {
        let filename = filename.as_ref();
        let source = NetCdfFixtureSource::from_path(filename)?;
        Self::from_source(
            filename.to_string_lossy(),
            source,
            NetCdfFileTypeInfo::new(),
            channels,
        )
    }

    pub fn from_source(
        filename: impl Into<String>,
        source: NetCdfFixtureSource,
        filetype_info: NetCdfFileTypeInfo,
        channels: impl IntoIterator<Item = impl Into<String>>,
    ) -> Result<Self> {
        let filename = filename.into();
        let channels = normalize_channels(channels)?;
        let file_handler =
            NetCdfFileHandler::from_source(filename, BTreeMap::new(), filetype_info, &source)?;
        for channel in &channels {
            let path = FciL1cNetCdfHandler::effective_radiance_path(channel)?;
            file_handler.variable_shape(&path)?;
        }
        Ok(Self {
            name: "fci_l1c_nc".to_string(),
            source,
            handler: FciL1cNetCdfHandler::new(file_handler),
            channels,
        })
    }

    pub fn source(&self) -> &impl NetCdfDataSource {
        &self.source
    }

    pub fn handler(&self) -> &FciL1cNetCdfHandler {
        &self.handler
    }

    pub fn channels(&self) -> &[String] {
        &self.channels
    }

    pub fn inventory(&self) -> Result<ReaderInventory> {
        ReaderInventory::new(self.name.clone(), self.available_dataset_ids())
    }

    fn dataset_id(channel: &str) -> Result<DataId> {
        DataId::new(channel)?.with_qualifier("calibration", "counts")
    }

    fn supports_dataset(&self, id: &DataId) -> bool {
        self.channels.iter().any(|channel| channel == id.name())
            && id
                .qualifier("calibration")
                .map(|value| value == &DataValue::Text("counts".to_string()))
                .unwrap_or(false)
    }
}

impl Reader for FciL1cFixtureReader {
    fn name(&self) -> &str {
        &self.name
    }

    fn available_dataset_ids(&self) -> Vec<DataId> {
        self.channels
            .iter()
            .filter_map(|channel| Self::dataset_id(channel).ok())
            .collect()
    }

    fn load(&self, id: &DataId) -> Result<Dataset> {
        if !self.supports_dataset(id) {
            return Err(RustySatError::not_found(format!(
                "FCI L1C fixture dataset '{}'",
                id.name()
            )));
        }
        self.handler.load_counts_dataset(id.name(), &self.source)
    }
}

fn normalize_channels(
    channels: impl IntoIterator<Item = impl Into<String>>,
) -> Result<Vec<String>> {
    let mut normalized = Vec::new();
    for channel in channels {
        let channel = channel.into();
        FciL1cNetCdfHandler::effective_radiance_path(&channel)?;
        if !normalized.contains(&channel) {
            normalized.push(channel);
        }
    }
    if normalized.is_empty() {
        return Err(RustySatError::invalid_input(
            "FCI fixture reader requires at least one channel",
        ));
    }
    Ok(normalized)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;
    use rusty_sat_core::{DataQuery, MetadataValue, Scene};
    use std::fs;

    const FCI_FIXTURE: &str = r#"
attrs:
  platform: MTG-I1
groups:
  data:
    groups:
      vis_04:
        groups:
          measured:
            dimensions:
              y: 2
              x: 3
            variables:
              effective_radiance:
                dtype: u16
                dimensions: [y, x]
                shape: [2, 3]
                attrs:
                  units: mW m-2 sr-1 (cm-1)-1
                  ancillary_variables: pixel_quality
                  _FillValue: 65535
                  valid_range: [0, 4095]
                values: [10, 4096, 12, 13, 65535, 15]
"#;

    fn fixture_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "rusty_sat_fci_fixture_{}_{}.yaml",
            std::process::id(),
            name
        ))
    }

    #[test]
    fn fci_fixture_reader_exposes_counts_inventory() {
        let source = NetCdfFixtureSource::from_yaml_str(FCI_FIXTURE).unwrap();
        let reader = FciL1cFixtureReader::from_source(
            "fixture.yaml",
            source,
            NetCdfFileTypeInfo::new(),
            ["vis_04"],
        )
        .unwrap();
        let id = FciL1cFixtureReader::dataset_id("vis_04").unwrap();

        assert_eq!(reader.name(), "fci_l1c_nc");
        assert_eq!(reader.available_dataset_ids(), vec![id.clone()]);
        assert!(reader
            .inventory()
            .unwrap()
            .available_dataset_ids()
            .contains(&id));
    }

    #[test]
    fn fci_fixture_reader_loads_counts_dataset() {
        let source = NetCdfFixtureSource::from_yaml_str(FCI_FIXTURE).unwrap();
        let reader = FciL1cFixtureReader::from_source(
            "fixture.yaml",
            source,
            NetCdfFileTypeInfo::new(),
            ["vis_04"],
        )
        .unwrap();
        let id = FciL1cFixtureReader::dataset_id("vis_04").unwrap();

        let dataset = reader.load(&id).unwrap();
        let array = dataset.array().unwrap();
        let mask = array.mask().unwrap();

        assert_eq!(dataset.id(), &id);
        assert_eq!(array.shape(), &[2, 3]);
        assert_eq!(array.dtype().name(), "u16");
        assert_eq!(mask.masked_count(), 2);
        assert_eq!(
            dataset.attr("ancillary_variables"),
            Some(&MetadataValue::String("pixel_quality".to_string()))
        );
    }

    #[test]
    fn fci_fixture_reader_drives_scene_planning() {
        let path = fixture_path("scene_planning");
        fs::write(&path, FCI_FIXTURE).unwrap();
        let reader = FciL1cFixtureReader::from_fixture_path(&path, ["vis_04"]).unwrap();
        let inventory = reader.inventory().unwrap();
        let mut scene = Scene::new();
        let plan = scene
            .plan_reader_loads([DataQuery::named("vis_04").unwrap()], [&inventory])
            .unwrap();
        let ids = plan.reader_datasets().get("fci_l1c_nc").unwrap();

        for id in ids {
            scene.insert_dataset(reader.load(id).unwrap());
        }

        assert_eq!(scene.len(), 1);
        let id = FciL1cFixtureReader::dataset_id("vis_04").unwrap();
        assert_eq!(scene.get(&id).unwrap().array().unwrap().shape(), &[2, 3]);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn fci_fixture_reader_drives_scene_load_lifecycle() {
        let path = fixture_path("scene_lifecycle");
        fs::write(&path, FCI_FIXTURE).unwrap();
        let reader = FciL1cFixtureReader::from_fixture_path(&path, ["vis_04"]).unwrap();
        let mut scene = Scene::with_loader(reader);

        assert_eq!(scene.available_dataset_names(), vec!["vis_04".to_string()]);
        scene.load([DataQuery::named("vis_04").unwrap()]).unwrap();

        assert_eq!(scene.len(), 1);
        assert!(scene.missing_datasets().is_empty());
        let id = FciL1cFixtureReader::dataset_id("vis_04").unwrap();
        assert_eq!(scene.get(&id).unwrap().array().unwrap().shape(), &[2, 3]);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn fci_fixture_reader_rejects_unknown_calibration() {
        let source = NetCdfFixtureSource::from_yaml_str(FCI_FIXTURE).unwrap();
        let reader = FciL1cFixtureReader::from_source(
            "fixture.yaml",
            source,
            NetCdfFileTypeInfo::new(),
            ["vis_04"],
        )
        .unwrap();
        let id = DataId::new("vis_04")
            .unwrap()
            .with_qualifier("calibration", "radiance")
            .unwrap();

        assert!(reader.load(&id).is_err());
    }

    #[test]
    fn fci_fixture_reader_rejects_unknown_channel() {
        let source = NetCdfFixtureSource::from_yaml_str(FCI_FIXTURE).unwrap();

        let err = FciL1cFixtureReader::from_source(
            "fixture.yaml",
            source,
            NetCdfFileTypeInfo::new(),
            ["vis_05"],
        )
        .unwrap_err()
        .to_string();

        assert!(err.contains("vis_05"));
    }

    #[test]
    fn fci_fixture_reader_rejects_empty_channels() {
        let source = NetCdfFixtureSource::from_yaml_str(FCI_FIXTURE).unwrap();

        let err = FciL1cFixtureReader::from_source(
            "fixture.yaml",
            source,
            NetCdfFileTypeInfo::new(),
            Vec::<String>::new(),
        )
        .unwrap_err()
        .to_string();

        assert!(err.contains("at least one channel"));
    }

    #[test]
    fn fci_fixture_reader_deduplicates_channels() {
        let source = NetCdfFixtureSource::from_yaml_str(FCI_FIXTURE).unwrap();
        let reader = FciL1cFixtureReader::from_source(
            "fixture.yaml",
            source,
            NetCdfFileTypeInfo::new(),
            ["vis_04", "vis_04"],
        )
        .unwrap();

        assert_eq!(reader.channels().len(), 1);
        assert_eq!(reader.available_dataset_ids().len(), 1);
    }

    #[test]
    fn fci_fixture_reader_loads_multiple_channels() {
        let fixture = r#"
groups:
  data:
    groups:
      vis_04:
        groups:
          measured:
            dimensions: {y: 1, x: 2}
            variables:
              effective_radiance:
                dtype: u16
                dimensions: [y, x]
                shape: [1, 2]
                values: [1, 2]
      ir_38:
        groups:
          measured:
            dimensions: {y: 1, x: 2}
            variables:
              effective_radiance:
                dtype: u16
                dimensions: [y, x]
                shape: [1, 2]
                values: [3, 4]
"#;
        let source = NetCdfFixtureSource::from_yaml_str(fixture).unwrap();
        let reader = FciL1cFixtureReader::from_source(
            "fixture.yaml",
            source,
            NetCdfFileTypeInfo::new(),
            ["vis_04", "ir_38"],
        )
        .unwrap();

        assert_eq!(reader.channels().len(), 2);
        assert_eq!(reader.available_dataset_ids().len(), 2);

        let vis_id = FciL1cFixtureReader::dataset_id("vis_04").unwrap();
        let ir_id = FciL1cFixtureReader::dataset_id("ir_38").unwrap();
        let vis_ds = reader.load(&vis_id).unwrap();
        let ir_ds = reader.load(&ir_id).unwrap();

        assert_eq!(vis_ds.array().unwrap().values_as_f64(), vec![1.0, 2.0]);
        assert_eq!(ir_ds.array().unwrap().values_as_f64(), vec![3.0, 4.0]);
    }
}
