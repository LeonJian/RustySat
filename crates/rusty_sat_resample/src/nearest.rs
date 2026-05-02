//! First nearest-neighbor resampling slice.
//!
//! Reference behavior inspected before implementation:
//! - `deps/pyresample/pyresample/kd_tree.py`
//! - `deps/pyresample/pyresample/geometry.py`
//! - `deps/pyresample/docs/source/concepts/resampling.rst`
//!
//! This module starts with projection-coordinate area-to-area resampling. It
//! uses pixel centers and an optional radius of influence like Pyresample, but
//! does not yet implement kd-tree lookup, swaths, CRS transforms, or masks.

use crate::{AreaDefinition, Resampler};
use rusty_sat_core::{DataGrid, Dataset, Result, RustySatError};

#[derive(Debug, Clone)]
pub struct NearestAreaResampler {
    source: AreaDefinition,
    radius_of_influence: Option<f64>,
    fill_value: f64,
}

impl NearestAreaResampler {
    pub fn new(source: AreaDefinition) -> Self {
        Self {
            source,
            radius_of_influence: None,
            fill_value: f64::NAN,
        }
    }

    pub fn with_radius_of_influence(mut self, radius_of_influence: f64) -> Result<Self> {
        if radius_of_influence < 0.0 {
            return Err(RustySatError::invalid_input(
                "radius_of_influence must be non-negative",
            ));
        }
        self.radius_of_influence = Some(radius_of_influence);
        Ok(self)
    }

    pub fn with_fill_value(mut self, fill_value: f64) -> Self {
        self.fill_value = fill_value;
        self
    }

    pub fn source(&self) -> &AreaDefinition {
        &self.source
    }
}

impl Resampler for NearestAreaResampler {
    fn name(&self) -> &str {
        "nearest_area"
    }

    fn resample(&self, dataset: &Dataset, destination: &AreaDefinition) -> Result<Dataset> {
        let source_grid = dataset.data().ok_or_else(|| {
            RustySatError::invalid_input("nearest resampling requires dataset grid values")
        })?;
        if source_grid.shape() != self.source.shape() {
            return Err(RustySatError::invalid_input(format!(
                "dataset grid shape {:?} does not match source area shape {:?}",
                source_grid.shape(),
                self.source.shape()
            )));
        }
        if self.source.projection() != destination.projection() {
            return Err(RustySatError::unsupported(
                "nearest area resampling between different projections",
            ));
        }
        let resampled = resample_area_nearest(
            source_grid,
            &self.source,
            destination,
            self.radius_of_influence,
            self.fill_value,
        )?;
        let metadata = dataset_metadata_pairs(dataset.metadata());
        let mut resampled_dataset = Dataset::new(dataset.id().clone()).with_data(resampled);
        for (key, value) in metadata {
            resampled_dataset.insert_metadata(key, value)?;
        }
        resampled_dataset.insert_metadata("area", destination.id())?;
        resampled_dataset.insert_metadata("resampler", self.name())?;
        Ok(resampled_dataset)
    }
}

fn dataset_metadata_pairs(
    metadata: &std::collections::BTreeMap<String, String>,
) -> Vec<(String, String)> {
    metadata
        .iter()
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect()
}

pub fn resample_area_nearest(
    source_grid: &DataGrid,
    source: &AreaDefinition,
    destination: &AreaDefinition,
    radius_of_influence: Option<f64>,
    fill_value: f64,
) -> Result<DataGrid> {
    if source_grid.shape() != source.shape() {
        return Err(RustySatError::invalid_input(format!(
            "source grid shape {:?} does not match source area shape {:?}",
            source_grid.shape(),
            source.shape()
        )));
    }
    let (dst_height, dst_width) = destination.shape();
    let mut values = Vec::with_capacity(dst_height * dst_width);
    for y in 0..dst_height {
        for x in 0..dst_width {
            let (dst_x, dst_y) = pixel_center(destination, y, x);
            let Some((src_y, src_x, distance)) = nearest_source_pixel(source, dst_x, dst_y) else {
                values.push(fill_value);
                continue;
            };
            if radius_of_influence.is_some_and(|radius| distance > radius) {
                values.push(fill_value);
                continue;
            }
            values.push(source_grid.get(src_y, src_x).unwrap_or(fill_value));
        }
    }
    DataGrid::new(dst_height, dst_width, values)
}

fn nearest_source_pixel(source: &AreaDefinition, x: f64, y: f64) -> Option<(usize, usize, f64)> {
    let extent = source.area_extent();
    let (pixel_size_x, pixel_size_y) = source.pixel_size();
    let src_x = ((x - extent[0]) / pixel_size_x - 0.5).round();
    let src_y = ((extent[3] - y) / pixel_size_y - 0.5).round();
    if src_x < 0.0 || src_y < 0.0 {
        return None;
    }
    let src_x = src_x as usize;
    let src_y = src_y as usize;
    let (height, width) = source.shape();
    if src_x >= width || src_y >= height {
        return None;
    }
    let (nearest_x, nearest_y) = pixel_center(source, src_y, src_x);
    let distance = ((nearest_x - x).powi(2) + (nearest_y - y).powi(2)).sqrt();
    Some((src_y, src_x, distance))
}

fn pixel_center(area: &AreaDefinition, y: usize, x: usize) -> (f64, f64) {
    let extent = area.area_extent();
    let (pixel_size_x, pixel_size_y) = area.pixel_size();
    (
        extent[0] + (x as f64 + 0.5) * pixel_size_x,
        extent[3] - (y as f64 + 0.5) * pixel_size_y,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn area(id: &str, height: usize, width: usize, area_extent: [f64; 4]) -> AreaDefinition {
        AreaDefinition::from_parts(
            id,
            id,
            id,
            BTreeMap::from([("proj".to_string(), "latlong".to_string())]),
            height,
            width,
            area_extent,
        )
        .unwrap()
    }

    #[test]
    fn nearest_resamples_area_to_finer_area() {
        let source = area("source", 2, 2, [0.0, 0.0, 2.0, 2.0]);
        let destination = area("destination", 4, 4, [0.0, 0.0, 2.0, 2.0]);
        let source_grid = DataGrid::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();

        let result =
            resample_area_nearest(&source_grid, &source, &destination, None, f64::NAN).unwrap();

        assert_eq!(result.shape(), (4, 4));
        assert_eq!(
            result.values(),
            &[
                1.0, 1.0, 2.0, 2.0, //
                1.0, 1.0, 2.0, 2.0, //
                3.0, 3.0, 4.0, 4.0, //
                3.0, 3.0, 4.0, 4.0,
            ]
        );
    }

    #[test]
    fn nearest_uses_fill_value_outside_radius() {
        let source = area("source", 1, 1, [0.0, 0.0, 1.0, 1.0]);
        let destination = area("destination", 1, 1, [1.0, 1.0, 2.0, 2.0]);
        let source_grid = DataGrid::new(1, 1, vec![5.0]).unwrap();

        let result =
            resample_area_nearest(&source_grid, &source, &destination, Some(0.25), -999.0).unwrap();

        assert_eq!(result.values(), &[-999.0]);
    }

    #[test]
    fn resampler_trait_returns_dataset_with_destination_area_metadata() {
        let source = area("source", 2, 2, [0.0, 0.0, 2.0, 2.0]);
        let destination = area("destination", 1, 1, [0.0, 1.0, 1.0, 2.0]);
        let id = rusty_sat_core::DataId::new("image").unwrap();
        let dataset =
            Dataset::new(id).with_data(DataGrid::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap());
        let resampler = NearestAreaResampler::new(source).with_fill_value(-999.0);

        let result = resampler.resample(&dataset, &destination).unwrap();

        assert_eq!(result.data().unwrap().values(), &[1.0]);
        assert_eq!(
            result.metadata().get("area"),
            Some(&"destination".to_string())
        );
        assert_eq!(
            result.metadata().get("resampler"),
            Some(&"nearest_area".to_string())
        );
    }
}
