//! Reader framework foundations.

pub mod ahi_hsd;
pub mod ahi_l2_nc;
pub mod fci_l1c_nc;
pub mod filename_pattern;
pub mod netcdf;
pub mod text_grid;
pub mod yaml_reader;

pub use ahi_hsd::{
    parse_initial_hsd_header, AhiBandCalibration, AhiBasicInfo, AhiCalibration, AhiCalibrationInfo,
    AhiCalibrationMode, AhiCalibrationOutput, AhiDataInfo, AhiHsdFileHandler, AhiHsdHeader,
    AhiHsdReader, AhiNavigationInfo, AhiProjectionInfo, AhiSegmentBlockInfo, AhiSegmentInfo,
    AhiUserCalibration, AhiUserCalibrationCoefficients,
};
pub use ahi_l2_nc::{
    ahi_l2_dataset_defs, AhiL2DatasetDef, AhiL2NcFileHandler, AhiL2NcFileType, AhiL2NcFixtureReader,
};
pub use fci_l1c_nc::FciL1cFixtureReader;
pub use netcdf::{
    FciL1cNetCdfHandler, InMemoryNetCdfSource, NetCdfContent, NetCdfDataSource, NetCdfFileHandler,
    NetCdfFileTypeInfo, NetCdfFixtureSource, NetCdfGroup, NetCdfMetadata, NetCdfMetadataSource,
    NetCdfVariable,
};
pub use text_grid::{
    lazy_text_grid, load_text_grid, parse_text_grid, TextGridChunkSource, TextGridReader,
};
pub use yaml_reader::{
    yaml_to_metadata_value, DatasetConfig, FileMatch, FileTypeConfig, ReaderInfo,
    YamlMetadataReader, YamlReaderConfig,
};

use rusty_sat_core::{DataId, Dataset, ReaderInventory, Result, RustySatError};
use std::collections::BTreeMap;

pub trait Reader {
    fn name(&self) -> &str;

    fn available_dataset_ids(&self) -> Vec<DataId> {
        Vec::new()
    }

    fn load(&self, _id: &DataId) -> Result<Dataset>;
}

#[derive(Debug, Clone)]
pub struct FakeReader {
    name: String,
    datasets: BTreeMap<DataId, Dataset>,
}

impl FakeReader {
    pub fn new(name: impl Into<String>) -> Result<Self> {
        let name = name.into();
        if name.trim().is_empty() {
            return Err(RustySatError::invalid_input(
                "fake reader name cannot be empty",
            ));
        }
        Ok(Self {
            name,
            datasets: BTreeMap::new(),
        })
    }

    pub fn with_dataset(mut self, dataset: Dataset) -> Self {
        self.datasets.insert(dataset.id().clone(), dataset);
        self
    }

    pub fn insert_dataset(&mut self, dataset: Dataset) {
        self.datasets.insert(dataset.id().clone(), dataset);
    }

    pub fn inventory(&self) -> Result<ReaderInventory> {
        ReaderInventory::new(self.name.clone(), self.available_dataset_ids())
    }
}

impl Reader for FakeReader {
    fn name(&self) -> &str {
        &self.name
    }

    fn available_dataset_ids(&self) -> Vec<DataId> {
        self.datasets.keys().cloned().collect()
    }

    fn load(&self, id: &DataId) -> Result<Dataset> {
        self.datasets
            .get(id)
            .cloned()
            .ok_or_else(|| RustySatError::not_found(format!("dataset '{}'", id.name())))
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;
    use rusty_sat_core::{DataQuery, Scene};

    struct EmptyReader;

    impl Reader for EmptyReader {
        fn name(&self) -> &str {
            "empty"
        }

        fn load(&self, _id: &DataId) -> Result<Dataset> {
            Err(RustySatError::unsupported("empty reader load"))
        }
    }

    #[test]
    fn reader_trait_compiles() {
        let reader = EmptyReader;
        assert_eq!(reader.name(), "empty");
        assert!(reader.available_dataset_ids().is_empty());
    }

    #[test]
    fn fake_reader_exposes_inventory_and_loads_dataset() {
        let data_id = DataId::new("VIS006").unwrap();
        let dataset = Dataset::new(data_id.clone());
        let reader = FakeReader::new("fake")
            .unwrap()
            .with_dataset(dataset.clone());

        assert_eq!(reader.name(), "fake");
        assert_eq!(reader.available_dataset_ids(), vec![data_id.clone()]);
        assert_eq!(reader.inventory().unwrap().name(), "fake");
        assert_eq!(reader.load(&data_id).unwrap(), dataset);
    }

    #[test]
    fn fake_reader_drives_scene_planning_vertical_slice() {
        let low_res = DataId::new("VIS006")
            .unwrap()
            .with_qualifier("resolution", 3000.0)
            .unwrap();
        let high_res = DataId::new("VIS006")
            .unwrap()
            .with_qualifier("resolution", 1000.0)
            .unwrap();
        let reader = FakeReader::new("fake")
            .unwrap()
            .with_dataset(Dataset::new(low_res))
            .with_dataset(Dataset::new(high_res.clone()));
        let inventory = reader.inventory().unwrap();
        let mut scene = Scene::new();
        let plan = scene
            .plan_reader_loads([DataQuery::named("VIS006").unwrap()], [&inventory])
            .unwrap();
        let planned_ids = plan.reader_datasets().get("fake").unwrap();

        for id in planned_ids {
            scene.insert_dataset(reader.load(id).unwrap());
        }

        assert!(scene.get(&high_res).is_some());
        assert_eq!(scene.len(), 1);
    }
}
