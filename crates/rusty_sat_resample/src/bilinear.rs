//! First bilinear resampling slice.
//!
//! Reference behavior inspected before implementation:
//! - `satpy/satpy/resample/kdtree.py`
//! - `deps/pyresample/pyresample/image.py`
//! - `deps/pyresample/pyresample/bilinear/_numpy_resampler.py`
//!
//! Pyresample's production bilinear path supports irregular swaths, cached
//! coefficients, neighbour searches, and multiple dimensions. This module
//! starts with the rectangular area-to-area case where both areas share a
//! projection and source coordinates can be mapped directly to fractional
//! source pixels.

use crate::{AreaDefinition, Resampler};
use rayon::prelude::*;
use rusty_sat_core::{Coordinate, DataGrid, Dataset, Result, RustySatError, ValidityMask};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BilinearMissingPolicy {
    FillValue,
    Mask,
}

impl BilinearMissingPolicy {
    fn masks_missing(self) -> bool {
        matches!(self, Self::Mask)
    }
}

#[derive(Debug, Clone)]
pub struct BilinearAreaResampler {
    source: AreaDefinition,
    fill_value: f64,
    missing_policy: BilinearMissingPolicy,
}

impl BilinearAreaResampler {
    pub fn new(source: AreaDefinition) -> Self {
        Self {
            source,
            fill_value: f64::NAN,
            missing_policy: BilinearMissingPolicy::FillValue,
        }
    }

    pub fn with_fill_value(mut self, fill_value: f64) -> Self {
        self.fill_value = fill_value;
        self.missing_policy = BilinearMissingPolicy::FillValue;
        self
    }

    pub fn with_masked_missing(mut self) -> Self {
        self.missing_policy = BilinearMissingPolicy::Mask;
        self
    }

    pub fn source(&self) -> &AreaDefinition {
        &self.source
    }
}

impl Resampler for BilinearAreaResampler {
    fn name(&self) -> &str {
        "bilinear"
    }

    fn resample(&self, dataset: &Dataset, destination: &AreaDefinition) -> Result<Dataset> {
        let source_grid = dataset.data().ok_or_else(|| {
            RustySatError::invalid_input("bilinear resampling requires f64 dataset grid values")
        })?;
        validate_source_shape(source_grid, &self.source)?;
        validate_projection_compatibility(&self.source, destination)?;

        let resampled = resample_area_bilinear_with_policy(
            source_grid,
            &self.source,
            destination,
            self.fill_value,
            self.missing_policy,
        )?;
        let mut resampled_dataset = Dataset::new(dataset.id().clone()).with_data(resampled);
        for (key, value) in dataset.metadata() {
            resampled_dataset.insert_metadata(key.clone(), value.clone())?;
        }
        for (key, value) in dataset.attrs() {
            resampled_dataset.insert_attr(key.clone(), value.clone())?;
        }
        resampled_dataset.insert_metadata("area", destination.id())?;
        resampled_dataset.insert_metadata("resampler", self.name())?;
        Ok(resampled_dataset)
    }

    fn resample_owned(&self, dataset: Dataset, destination: &AreaDefinition) -> Result<Dataset> {
        let id = dataset.id().clone();
        let metadata = dataset.metadata().clone();
        let attrs = dataset.attrs().clone();
        let source_grid = dataset
            .into_array()
            .and_then(|array| array.into_f64())
            .ok_or_else(|| {
                RustySatError::invalid_input("bilinear resampling requires an f64 dataset grid")
            })?;
        validate_source_shape(&source_grid, &self.source)?;
        validate_projection_compatibility(&self.source, destination)?;

        let resampled = resample_area_bilinear_owned_with_policy(
            source_grid,
            &self.source,
            destination,
            self.fill_value,
            self.missing_policy,
        )?;
        let mut resampled_dataset = Dataset::new(id).with_data(resampled);
        for (key, value) in metadata {
            resampled_dataset.insert_metadata(key, value)?;
        }
        for (key, value) in attrs {
            resampled_dataset.insert_attr(key, value)?;
        }
        resampled_dataset.insert_metadata("area", destination.id())?;
        resampled_dataset.insert_metadata("resampler", self.name())?;
        Ok(resampled_dataset)
    }
}

pub fn resample_area_bilinear(
    source_grid: &DataGrid,
    source: &AreaDefinition,
    destination: &AreaDefinition,
    fill_value: f64,
) -> Result<DataGrid> {
    resample_area_bilinear_with_policy(
        source_grid,
        source,
        destination,
        fill_value,
        BilinearMissingPolicy::FillValue,
    )
}

pub fn resample_area_bilinear_masked_missing(
    source_grid: &DataGrid,
    source: &AreaDefinition,
    destination: &AreaDefinition,
) -> Result<DataGrid> {
    resample_area_bilinear_with_policy(
        source_grid,
        source,
        destination,
        f64::NAN,
        BilinearMissingPolicy::Mask,
    )
}

pub fn resample_area_bilinear_owned(
    source_grid: DataGrid,
    source: &AreaDefinition,
    destination: &AreaDefinition,
    fill_value: f64,
) -> Result<DataGrid> {
    resample_area_bilinear_owned_with_policy(
        source_grid,
        source,
        destination,
        fill_value,
        BilinearMissingPolicy::FillValue,
    )
}

fn resample_area_bilinear_with_policy(
    source_grid: &DataGrid,
    source: &AreaDefinition,
    destination: &AreaDefinition,
    fill_value: f64,
    missing_policy: BilinearMissingPolicy,
) -> Result<DataGrid> {
    validate_source_shape(source_grid, source)?;
    validate_projection_compatibility(source, destination)?;
    let (dst_height, dst_width) = destination.shape();
    let row_ys: Vec<f64> = destination.iter_projection_y_coords().collect();
    // Destination rows are independent: compute per-row vectors in parallel
    // (deterministic, bounded row-buffer memory) and concatenate in order.
    let rows: Vec<(Vec<f64>, Vec<bool>)> = (0..dst_height)
        .into_par_iter()
        .map(|y| {
            let mut row_values = Vec::with_capacity(dst_width);
            let mut row_masks = Vec::with_capacity(dst_width);
            let row_y = row_ys[y];
            for x in destination.iter_projection_x_coords() {
                match bilinear_sample(source_grid.values(), source_grid.mask(), source, x, row_y) {
                    Some(value) => {
                        row_values.push(value);
                        row_masks.push(false);
                    }
                    None => {
                        row_values.push(fill_value);
                        row_masks.push(missing_policy.masks_missing());
                    }
                }
            }
            (row_values, row_masks)
        })
        .collect();
    let (values, mask_flags) = flatten_bilinear_rows(rows, dst_height * dst_width);

    add_bilinear_coords(
        finish_bilinear_grid(dst_height, dst_width, values, mask_flags)?,
        Some(source_grid.coords()),
        destination,
    )
}

fn resample_area_bilinear_owned_with_policy(
    source_grid: DataGrid,
    source: &AreaDefinition,
    destination: &AreaDefinition,
    fill_value: f64,
    missing_policy: BilinearMissingPolicy,
) -> Result<DataGrid> {
    validate_source_shape(&source_grid, source)?;
    validate_projection_compatibility(source, destination)?;
    let (src_values, src_coords, src_mask) = source_grid.into_parts();
    let (dst_height, dst_width) = destination.shape();
    let row_ys: Vec<f64> = destination.iter_projection_y_coords().collect();
    let rows: Vec<(Vec<f64>, Vec<bool>)> = (0..dst_height)
        .into_par_iter()
        .map(|y| {
            let mut row_values = Vec::with_capacity(dst_width);
            let mut row_masks = Vec::with_capacity(dst_width);
            let row_y = row_ys[y];
            for x in destination.iter_projection_x_coords() {
                match bilinear_sample(&src_values, src_mask.as_ref(), source, x, row_y) {
                    Some(value) => {
                        row_values.push(value);
                        row_masks.push(false);
                    }
                    None => {
                        row_values.push(fill_value);
                        row_masks.push(missing_policy.masks_missing());
                    }
                }
            }
            (row_values, row_masks)
        })
        .collect();
    let (values, mask_flags) = flatten_bilinear_rows(rows, dst_height * dst_width);

    add_bilinear_coords_owned(
        finish_bilinear_grid(dst_height, dst_width, values, mask_flags)?,
        Some(src_coords),
        destination,
    )
}

fn flatten_bilinear_rows(
    rows: Vec<(Vec<f64>, Vec<bool>)>,
    total_len: usize,
) -> (Vec<f64>, Vec<bool>) {
    let mut values = Vec::with_capacity(total_len);
    let mut mask_flags = Vec::with_capacity(total_len);
    for (row_values, row_masks) in rows {
        values.extend(row_values);
        mask_flags.extend(row_masks);
    }
    (values, mask_flags)
}

fn bilinear_sample(
    values: &[f64],
    mask: Option<&ValidityMask>,
    source: &AreaDefinition,
    x: f64,
    y: f64,
) -> Option<f64> {
    let (height, width) = source.shape();
    let extent = source.area_extent();
    let (pixel_size_x, pixel_size_y) = source.pixel_size();
    let src_x = (x - extent[0]) / pixel_size_x - 0.5;
    let src_y = (extent[3] - y) / pixel_size_y - 0.5;
    if !src_x.is_finite()
        || !src_y.is_finite()
        || src_x < 0.0
        || src_y < 0.0
        || src_x > (width - 1) as f64
        || src_y > (height - 1) as f64
    {
        return None;
    }

    let x0 = src_x.floor() as usize;
    let y0 = src_y.floor() as usize;
    let x1 = src_x.ceil() as usize;
    let y1 = src_y.ceil() as usize;
    let wx = src_x - x0 as f64;
    let wy = src_y - y0 as f64;

    let v00 = valid_value(values, mask, y0 * width + x0)?;
    let v01 = valid_value(values, mask, y0 * width + x1)?;
    let v10 = valid_value(values, mask, y1 * width + x0)?;
    let v11 = valid_value(values, mask, y1 * width + x1)?;

    let top = v00 * (1.0 - wx) + v01 * wx;
    let bottom = v10 * (1.0 - wx) + v11 * wx;
    Some(top * (1.0 - wy) + bottom * wy)
}

fn valid_value(values: &[f64], mask: Option<&ValidityMask>, index: usize) -> Option<f64> {
    if mask.and_then(|mask| mask.is_masked(index)).unwrap_or(false) {
        return None;
    }
    let value = *values.get(index)?;
    if value.is_finite() {
        Some(value)
    } else {
        None
    }
}

fn validate_source_shape(source_grid: &DataGrid, source: &AreaDefinition) -> Result<()> {
    if source_grid.shape() != source.shape() {
        return Err(RustySatError::invalid_input(format!(
            "dataset grid shape {:?} does not match source area shape {:?}",
            source_grid.shape(),
            source.shape()
        )));
    }
    Ok(())
}

fn validate_projection_compatibility(
    source: &AreaDefinition,
    destination: &AreaDefinition,
) -> Result<()> {
    if source.projection() != destination.projection() {
        return Err(RustySatError::unsupported(
            "bilinear area resampling between different projections",
        ));
    }
    Ok(())
}

fn finish_bilinear_grid(
    height: usize,
    width: usize,
    values: Vec<f64>,
    mask_flags: Vec<bool>,
) -> Result<DataGrid> {
    let grid = DataGrid::new(height, width, values)?;
    if mask_flags.iter().any(|masked| *masked) {
        grid.with_mask(ValidityMask::from_masked_flags(mask_flags))
    } else {
        Ok(grid)
    }
}

fn add_bilinear_coords(
    mut grid: DataGrid,
    source_coords: Option<&BTreeMap<String, Coordinate>>,
    area: &AreaDefinition,
) -> Result<DataGrid> {
    if let Some(coords) = source_coords {
        for (name, coordinate) in coords {
            if should_preserve_coord(name, coordinate) {
                grid.set_coordinate(name.clone(), coordinate.clone())?;
            }
        }
    }
    grid.set_coordinate(
        "x",
        Coordinate::axis("x", area.iter_projection_x_coords().collect::<Vec<_>>())?,
    )?;
    grid.set_coordinate(
        "y",
        Coordinate::axis("y", area.iter_projection_y_coords().collect::<Vec<_>>())?,
    )?;
    Ok(grid)
}

fn add_bilinear_coords_owned(
    mut grid: DataGrid,
    source_coords: Option<BTreeMap<String, Coordinate>>,
    area: &AreaDefinition,
) -> Result<DataGrid> {
    if let Some(coords) = source_coords {
        for (name, coordinate) in coords {
            if should_preserve_coord(&name, &coordinate) {
                grid.set_coordinate(name, coordinate)?;
            }
        }
    }
    grid.set_coordinate(
        "x",
        Coordinate::axis("x", area.iter_projection_x_coords().collect::<Vec<_>>())?,
    )?;
    grid.set_coordinate(
        "y",
        Coordinate::axis("y", area.iter_projection_y_coords().collect::<Vec<_>>())?,
    )?;
    Ok(grid)
}

fn should_preserve_coord(name: &str, coordinate: &Coordinate) -> bool {
    const IGNORE_DIMS: [&str; 3] = ["y", "x", "crs"];
    !IGNORE_DIMS.contains(&name)
        && !coordinate
            .dims()
            .iter()
            .any(|dim| IGNORE_DIMS.contains(&dim.as_str()))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;
    use rusty_sat_core::{DataId, MetadataValue};

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
    fn bilinear_samples_area_fractional_pixel_center() {
        let source = area("source", 2, 2, [0.0, 0.0, 2.0, 2.0]);
        let destination = area("destination", 1, 1, [0.5, 0.5, 1.5, 1.5]);
        let grid = DataGrid::new(2, 2, vec![0.0, 10.0, 20.0, 30.0]).unwrap();

        let resampled = resample_area_bilinear(&grid, &source, &destination, -999.0).unwrap();

        assert_eq!(resampled.shape(), (1, 1));
        assert_eq!(resampled.values(), &[15.0]);
        assert!(resampled.mask().is_none());
    }

    #[test]
    fn bilinear_exact_pixel_centers_reproduce_source_values() {
        let source = area("source", 2, 2, [0.0, 0.0, 2.0, 2.0]);
        let grid = DataGrid::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();

        let resampled = resample_area_bilinear(&grid, &source, &source, -999.0).unwrap();

        assert_eq!(resampled.values(), grid.values());
    }

    #[test]
    fn bilinear_uses_fill_or_mask_for_missing_samples() {
        let source = area("source", 2, 2, [0.0, 0.0, 2.0, 2.0]);
        let destination = area("destination", 1, 1, [0.5, 0.5, 1.5, 1.5]);
        let grid = DataGrid::new(2, 2, vec![0.0, 10.0, 20.0, 30.0])
            .unwrap()
            .with_mask(ValidityMask::from_masked_flags([false, true, false, false]))
            .unwrap();

        let filled = resample_area_bilinear(&grid, &source, &destination, -999.0).unwrap();
        assert_eq!(filled.values(), &[-999.0]);
        assert!(filled.mask().is_none());

        let masked = resample_area_bilinear_masked_missing(&grid, &source, &destination).unwrap();
        assert!(masked.values()[0].is_nan());
        assert!(masked.is_masked(0).unwrap());
    }

    #[test]
    fn bilinear_owned_matches_borrowed() {
        let source = area("source", 2, 2, [0.0, 0.0, 2.0, 2.0]);
        let destination = area("destination", 1, 1, [0.5, 0.5, 1.5, 1.5]);
        let borrowed_grid = DataGrid::new(2, 2, vec![0.0, 10.0, 20.0, 30.0]).unwrap();
        let owned_grid = borrowed_grid.clone();

        let borrowed =
            resample_area_bilinear(&borrowed_grid, &source, &destination, -999.0).unwrap();
        let owned =
            resample_area_bilinear_owned(owned_grid, &source, &destination, -999.0).unwrap();

        assert_eq!(borrowed, owned);
    }

    #[test]
    fn resampler_rejects_different_projection_metadata() {
        let source = area("source", 2, 2, [0.0, 0.0, 2.0, 2.0]);
        let destination = AreaDefinition::from_parts(
            "destination",
            "destination",
            "destination",
            BTreeMap::from([("proj".to_string(), "merc".to_string())]),
            1,
            1,
            [0.5, 0.5, 1.5, 1.5],
        )
        .unwrap();
        let grid = DataGrid::new(2, 2, vec![0.0, 10.0, 20.0, 30.0]).unwrap();
        let dataset = Dataset::new(DataId::new("image").unwrap()).with_data(grid);
        let resampler = BilinearAreaResampler::new(source);

        let err = resampler.resample(&dataset, &destination).unwrap_err();

        assert!(err.to_string().contains("different projections"));
    }

    #[test]
    fn resampler_preserves_metadata_and_coords() {
        let source = area("source", 2, 2, [0.0, 0.0, 2.0, 2.0]);
        let destination = area("destination", 1, 1, [0.5, 0.5, 1.5, 1.5]);
        let grid = DataGrid::new(2, 2, vec![0.0, 10.0, 20.0, 30.0])
            .unwrap()
            .with_coordinate("time", Coordinate::scalar(1.0))
            .unwrap();
        let id = DataId::new("image").unwrap();
        let mut dataset = Dataset::new(id.clone()).with_data(grid);
        dataset.insert_metadata("units", "K").unwrap();
        dataset
            .insert_attr("nested", MetadataValue::string("kept"))
            .unwrap();
        let resampler = BilinearAreaResampler::new(source);

        let output = resampler.resample(&dataset, &destination).unwrap();

        assert_eq!(output.id(), &id);
        assert_eq!(output.data().unwrap().values(), &[15.0]);
        assert_eq!(output.metadata().get("units"), Some(&"K".to_string()));
        assert_eq!(
            output.metadata().get("resampler"),
            Some(&"bilinear".to_string())
        );
        assert!(output.data().unwrap().coord("time").is_some());
        assert!(output.data().unwrap().coord("x").is_some());
        assert!(output.data().unwrap().coord("y").is_some());
        assert_eq!(output.attr("nested"), Some(&MetadataValue::string("kept")));
    }

    #[test]
    fn resample_owned_through_trait_produces_same_output() {
        let source = area("source", 2, 2, [0.0, 0.0, 2.0, 2.0]);
        let destination = area("destination", 1, 1, [0.5, 0.5, 1.5, 1.5]);
        let id = DataId::new("image").unwrap();
        let ds1 = Dataset::new(id.clone())
            .with_data(DataGrid::new(2, 2, vec![0.0, 10.0, 20.0, 30.0]).unwrap());
        let ds2 =
            Dataset::new(id).with_data(DataGrid::new(2, 2, vec![0.0, 10.0, 20.0, 30.0]).unwrap());
        let resampler = BilinearAreaResampler::new(source);

        let borrowed = resampler.resample(&ds1, &destination).unwrap();
        let owned = resampler.resample_owned(ds2, &destination).unwrap();

        assert_eq!(borrowed.data().unwrap().values(), &[15.0]);
        assert_eq!(
            owned.data().unwrap().values(),
            borrowed.data().unwrap().values()
        );
    }

    #[test]
    fn bilinear_returns_fill_when_fully_outside_source_extent() {
        let source = area("source", 2, 2, [0.0, 0.0, 2.0, 2.0]);
        let destination = area("destination", 1, 1, [5.0, 5.0, 6.0, 6.0]);
        let grid = DataGrid::new(2, 2, vec![0.0, 10.0, 20.0, 30.0]).unwrap();

        let resampled = resample_area_bilinear(&grid, &source, &destination, -999.0).unwrap();

        assert_eq!(resampled.values(), &[-999.0]);
        assert!(resampled.mask().is_none());
    }

    #[test]
    fn bilinear_rejects_source_nans_during_interpolation() {
        let source = area("source", 2, 2, [0.0, 0.0, 2.0, 2.0]);
        let destination = area("destination", 1, 1, [0.5, 0.5, 1.5, 1.5]);
        let grid = DataGrid::new(2, 2, vec![f64::NAN, 10.0, 20.0, 30.0]).unwrap();

        let resampled = resample_area_bilinear(&grid, &source, &destination, -999.0).unwrap();

        assert_eq!(resampled.values(), &[-999.0]);
    }
}
