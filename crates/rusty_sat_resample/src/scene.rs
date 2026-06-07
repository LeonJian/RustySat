//! Scene-level resampling integration.
//!
//! Satpy exposes resampling as a `Scene` workflow. Rusty Sat keeps the core
//! `Scene` crate free of resampling dependencies, so this crate provides an
//! extension trait for the current Rust-native API surface.

use crate::{AreaDefinition, Resampler};
use rusty_sat_core::{Result, Scene};

pub trait SceneResampleExt {
    fn resample_with(
        &self,
        resampler: &impl Resampler,
        destination: &AreaDefinition,
    ) -> Result<Scene>;

    fn resample_with_owned(
        self,
        resampler: &impl Resampler,
        destination: &AreaDefinition,
    ) -> Result<Scene>;
}

impl SceneResampleExt for Scene {
    fn resample_with(
        &self,
        resampler: &impl Resampler,
        destination: &AreaDefinition,
    ) -> Result<Scene> {
        let mut resampled = Scene::new();
        for (_id, dataset) in self.iter() {
            resampled.insert_dataset(resampler.resample(dataset, destination)?);
        }
        Ok(resampled)
    }

    fn resample_with_owned(
        self,
        resampler: &impl Resampler,
        destination: &AreaDefinition,
    ) -> Result<Scene> {
        let mut resampled = Scene::new();
        for dataset in self.into_datasets() {
            resampled.insert_dataset(resampler.resample_owned(dataset, destination)?);
        }
        Ok(resampled)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;
    use crate::NearestAreaResampler;
    use rusty_sat_core::{DataGrid, DataId, Dataset};
    use std::collections::BTreeMap;

    fn area(id: &str, height: usize, width: usize, extent: [f64; 4]) -> AreaDefinition {
        AreaDefinition::from_parts(
            id,
            id,
            id,
            BTreeMap::from([("proj".to_string(), "longlat".to_string())]),
            height,
            width,
            extent,
        )
        .unwrap()
    }

    #[test]
    fn scene_resample_extension_resamples_all_datasets() {
        let source = area("source", 2, 2, [0.0, 0.0, 2.0, 2.0]);
        let destination = area("destination", 1, 1, [0.0, 1.0, 1.0, 2.0]);
        let mut scene = Scene::new();
        let id = DataId::new("image").unwrap();
        let mut dataset = Dataset::new(id.clone())
            .with_data(DataGrid::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap());
        dataset.insert_metadata("units", "K").unwrap();
        scene.insert_dataset(dataset);
        let resampler = NearestAreaResampler::new(source).with_fill_value(-999.0);

        let resampled = scene.resample_with(&resampler, &destination).unwrap();

        let output = resampled.get(&id).unwrap();
        assert_eq!(output.data().unwrap().values(), &[1.0]);
        assert_eq!(
            output.metadata().get("area"),
            Some(&"destination".to_string())
        );
        assert_eq!(
            output.metadata().get("resampler"),
            Some(&"nearest_area".to_string())
        );
        assert_eq!(output.metadata().get("units"), Some(&"K".to_string()));
    }

    #[test]
    fn scene_resample_extension_returns_empty_scene_for_empty_input() {
        let source = area("source", 1, 1, [0.0, 0.0, 1.0, 1.0]);
        let destination = area("destination", 1, 1, [0.0, 0.0, 1.0, 1.0]);
        let scene = Scene::new();
        let resampler = NearestAreaResampler::new(source);

        let resampled = scene.resample_with(&resampler, &destination).unwrap();

        assert!(resampled.is_empty());
    }

    #[test]
    fn scene_resample_owned_produces_same_output_as_borrowed() {
        let source = area("source", 2, 2, [0.0, 0.0, 2.0, 2.0]);
        let destination = area("destination", 1, 1, [0.0, 1.0, 1.0, 2.0]);

        let mut borrowed_scene = Scene::new();
        let mut owned_scene = Scene::new();
        let id = DataId::new("image").unwrap();

        let mut ds1 = Dataset::new(id.clone())
            .with_data(DataGrid::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap());
        ds1.insert_metadata("units", "K").unwrap();
        borrowed_scene.insert_dataset(ds1);

        let mut ds2 = Dataset::new(id.clone())
            .with_data(DataGrid::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap());
        ds2.insert_metadata("units", "K").unwrap();
        owned_scene.insert_dataset(ds2);

        let resampler = NearestAreaResampler::new(source).with_fill_value(-999.0);

        let borrowed = borrowed_scene
            .resample_with(&resampler, &destination)
            .unwrap();
        let owned = owned_scene
            .resample_with_owned(&resampler, &destination)
            .unwrap();

        let borrowed_out = borrowed.get(&id).unwrap();
        let owned_out = owned.get(&id).unwrap();
        assert_eq!(
            borrowed_out.data().unwrap().values(),
            owned_out.data().unwrap().values()
        );
        assert_eq!(borrowed_out.metadata(), owned_out.metadata());
    }

    #[test]
    fn scene_resample_owned_returns_empty_scene_for_empty_input() {
        let source = area("source", 1, 1, [0.0, 0.0, 1.0, 1.0]);
        let destination = area("destination", 1, 1, [0.0, 0.0, 1.0, 1.0]);
        let scene = Scene::new();
        let resampler = NearestAreaResampler::new(source);

        let resampled = scene.resample_with_owned(&resampler, &destination).unwrap();

        assert!(resampled.is_empty());
    }
}
