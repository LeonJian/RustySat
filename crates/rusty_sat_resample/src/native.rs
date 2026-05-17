//! Satpy-style native resolution resampling foundations.
//!
//! Reference behavior inspected before implementation:
//! - `satpy/satpy/resample/native.py`
//!
//! This first slice implements the core 2D rules from Satpy's
//! `NativeResampler`: integer expansion repeats samples, integer reduction
//! aggregates by mean, equal shapes are passed through unchanged, and mixed
//! expand/reduce directions are rejected.

use crate::{AreaDefinition, Resampler};
use rusty_sat_core::{Coordinate, DataGrid, Dataset, Result, RustySatError, ValidityMask};
use std::collections::BTreeMap;

#[derive(Debug, Clone)]
pub struct NativeResampler {
    source: AreaDefinition,
}

impl NativeResampler {
    pub fn new(source: AreaDefinition) -> Self {
        Self { source }
    }

    pub fn source(&self) -> &AreaDefinition {
        &self.source
    }
}

impl Resampler for NativeResampler {
    fn name(&self) -> &str {
        "native"
    }

    fn resample(&self, dataset: &Dataset, destination: &AreaDefinition) -> Result<Dataset> {
        let source_grid = dataset.data().ok_or_else(|| {
            RustySatError::invalid_input("native resampling requires f64 dataset grid values")
        })?;
        if source_grid.shape() != self.source.shape() {
            return Err(RustySatError::invalid_input(format!(
                "dataset grid shape {:?} does not match source area shape {:?}",
                source_grid.shape(),
                self.source.shape()
            )));
        }
        validate_native_area_compatibility(&self.source, destination)?;

        let resampled = native_resample_2d(source_grid, destination)?;
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
                RustySatError::invalid_input("native resampling requires an f64 dataset grid")
            })?;
        if source_grid.shape() != self.source.shape() {
            return Err(RustySatError::invalid_input(format!(
                "dataset grid shape {:?} does not match source area shape {:?}",
                source_grid.shape(),
                self.source.shape()
            )));
        }
        validate_native_area_compatibility(&self.source, destination)?;

        let resampled = native_resample_2d_owned(source_grid, destination)?;
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

pub fn native_resample_2d(
    source_grid: &DataGrid,
    destination: &AreaDefinition,
) -> Result<DataGrid> {
    let source_shape = source_grid.shape();
    let destination_shape = destination.shape();
    match native_scale(source_shape, destination_shape)? {
        NativeScale::Identity => {
            add_native_coords(source_grid.clone(), Some(source_grid.coords()), destination)
        }
        NativeScale::Repeat { y_factor, x_factor } => {
            let repeated = native_repeat_2d(source_grid, y_factor, x_factor)?;
            add_native_coords(repeated, Some(source_grid.coords()), destination)
        }
        NativeScale::Aggregate { y_factor, x_factor } => {
            let aggregated = native_aggregate_mean_2d(source_grid, y_factor, x_factor)?;
            add_native_coords(aggregated, Some(source_grid.coords()), destination)
        }
    }
}

pub fn native_resample_2d_owned(
    source_grid: DataGrid,
    destination: &AreaDefinition,
) -> Result<DataGrid> {
    let source_shape = source_grid.shape();
    let destination_shape = destination.shape();
    match native_scale(source_shape, destination_shape)? {
        NativeScale::Identity => {
            let (values, source_coords, mask) = source_grid.into_parts();
            let mut grid = DataGrid::new(destination_shape.0, destination_shape.1, values)?;
            if let Some(mask) = mask {
                grid.set_mask(mask)?;
            }
            add_native_coords_owned(grid, Some(source_coords), destination)
        }
        NativeScale::Repeat { y_factor, x_factor } => {
            let (values, source_coords, mask) = source_grid.into_parts();
            let repeated = repeat_2d_from_parts(
                source_shape.0,
                source_shape.1,
                values,
                mask,
                y_factor,
                x_factor,
            )?;
            add_native_coords_owned(repeated, Some(source_coords), destination)
        }
        NativeScale::Aggregate { y_factor, x_factor } => {
            let (values, source_coords, mask) = source_grid.into_parts();
            let aggregated = aggregate_mean_2d_from_parts(
                source_shape.0,
                source_shape.1,
                values,
                mask,
                y_factor,
                x_factor,
            )?;
            add_native_coords_owned(aggregated, Some(source_coords), destination)
        }
    }
}

pub fn native_repeat_2d(
    source_grid: &DataGrid,
    y_factor: usize,
    x_factor: usize,
) -> Result<DataGrid> {
    validate_repeat_factors(y_factor, x_factor)?;
    let (height, width) = source_grid.shape();
    let mut values = Vec::with_capacity(height * y_factor * width * x_factor);
    let mut mask_flags = Vec::new();
    if source_grid.mask().is_some() {
        mask_flags.reserve(values.capacity());
    }

    for y in 0..height {
        for _ in 0..y_factor {
            for x in 0..width {
                let src_idx = y * width + x;
                let value = source_grid.values()[src_idx];
                let masked = source_grid.is_masked(src_idx).unwrap_or(false);
                for _ in 0..x_factor {
                    values.push(value);
                    if source_grid.mask().is_some() {
                        mask_flags.push(masked);
                    }
                }
            }
        }
    }
    finish_native_grid(height * y_factor, width * x_factor, values, mask_flags)
}

pub fn native_repeat_2d_owned(
    source_grid: DataGrid,
    y_factor: usize,
    x_factor: usize,
) -> Result<DataGrid> {
    let (height, width) = source_grid.shape();
    let (source_values, _source_coords, source_mask) = source_grid.into_parts();
    repeat_2d_from_parts(
        height,
        width,
        source_values,
        source_mask,
        y_factor,
        x_factor,
    )
}

fn repeat_2d_from_parts(
    height: usize,
    width: usize,
    source_values: Vec<f64>,
    source_mask: Option<ValidityMask>,
    y_factor: usize,
    x_factor: usize,
) -> Result<DataGrid> {
    validate_repeat_factors(y_factor, x_factor)?;
    let mut values = Vec::with_capacity(height * y_factor * width * x_factor);
    let mut mask_flags = Vec::new();
    if source_mask.is_some() {
        mask_flags.reserve(values.capacity());
    }

    for y in 0..height {
        for _ in 0..y_factor {
            for x in 0..width {
                let src_idx = y * width + x;
                let value = source_values[src_idx];
                let masked = source_mask
                    .as_ref()
                    .and_then(|mask| mask.is_masked(src_idx))
                    .unwrap_or(false);
                for _ in 0..x_factor {
                    values.push(value);
                    if source_mask.is_some() {
                        mask_flags.push(masked);
                    }
                }
            }
        }
    }
    finish_native_grid(height * y_factor, width * x_factor, values, mask_flags)
}

pub fn native_aggregate_mean_2d(
    source_grid: &DataGrid,
    y_factor: usize,
    x_factor: usize,
) -> Result<DataGrid> {
    let (height, width) = source_grid.shape();
    validate_aggregate_factors(height, width, y_factor, x_factor)?;
    let out_height = height / y_factor;
    let out_width = width / x_factor;
    let mut values = Vec::with_capacity(out_height * out_width);
    let mut mask_flags = Vec::with_capacity(out_height * out_width);

    for out_y in 0..out_height {
        for out_x in 0..out_width {
            let (mean, masked) = aggregate_block_mean(
                source_grid.values(),
                source_grid.mask(),
                width,
                out_y,
                out_x,
                y_factor,
                x_factor,
            );
            values.push(mean);
            mask_flags.push(masked);
        }
    }
    finish_native_grid(out_height, out_width, values, mask_flags)
}

pub fn native_aggregate_mean_2d_owned(
    source_grid: DataGrid,
    y_factor: usize,
    x_factor: usize,
) -> Result<DataGrid> {
    let (height, width) = source_grid.shape();
    let (source_values, _source_coords, source_mask) = source_grid.into_parts();
    aggregate_mean_2d_from_parts(
        height,
        width,
        source_values,
        source_mask,
        y_factor,
        x_factor,
    )
}

fn aggregate_mean_2d_from_parts(
    height: usize,
    width: usize,
    source_values: Vec<f64>,
    source_mask: Option<ValidityMask>,
    y_factor: usize,
    x_factor: usize,
) -> Result<DataGrid> {
    validate_aggregate_factors(height, width, y_factor, x_factor)?;
    let out_height = height / y_factor;
    let out_width = width / x_factor;
    let mut values = Vec::with_capacity(out_height * out_width);
    let mut mask_flags = Vec::with_capacity(out_height * out_width);

    for out_y in 0..out_height {
        for out_x in 0..out_width {
            let (mean, masked) = aggregate_block_mean(
                &source_values,
                source_mask.as_ref(),
                width,
                out_y,
                out_x,
                y_factor,
                x_factor,
            );
            values.push(mean);
            mask_flags.push(masked);
        }
    }
    finish_native_grid(out_height, out_width, values, mask_flags)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NativeScale {
    Identity,
    Repeat { y_factor: usize, x_factor: usize },
    Aggregate { y_factor: usize, x_factor: usize },
}

fn native_scale(
    source_shape: (usize, usize),
    destination_shape: (usize, usize),
) -> Result<NativeScale> {
    let (src_h, src_w) = source_shape;
    let (dst_h, dst_w) = destination_shape;
    if source_shape == destination_shape {
        return Ok(NativeScale::Identity);
    }
    if dst_h >= src_h && dst_w >= src_w {
        if dst_h % src_h != 0 || dst_w % src_w != 0 {
            return Err(RustySatError::unsupported(
                "native expansion requires integer y/x repeat factors",
            ));
        }
        return Ok(NativeScale::Repeat {
            y_factor: dst_h / src_h,
            x_factor: dst_w / src_w,
        });
    }
    if dst_h <= src_h && dst_w <= src_w {
        if src_h % dst_h != 0 || src_w % dst_w != 0 {
            return Err(RustySatError::unsupported(
                "native reduction requires integer y/x aggregation factors",
            ));
        }
        return Ok(NativeScale::Aggregate {
            y_factor: src_h / dst_h,
            x_factor: src_w / dst_w,
        });
    }
    Err(RustySatError::unsupported(
        "native resampling cannot mix expansion and reduction axes",
    ))
}

fn validate_native_area_compatibility(
    source: &AreaDefinition,
    destination: &AreaDefinition,
) -> Result<()> {
    if source.projection() != destination.projection() {
        return Err(RustySatError::unsupported(
            "native resampling between different projections",
        ));
    }
    Ok(())
}

fn validate_repeat_factors(y_factor: usize, x_factor: usize) -> Result<()> {
    if y_factor == 0 || x_factor == 0 {
        return Err(RustySatError::invalid_input(
            "native repeat factors must be non-zero",
        ));
    }
    Ok(())
}

fn validate_aggregate_factors(
    height: usize,
    width: usize,
    y_factor: usize,
    x_factor: usize,
) -> Result<()> {
    validate_repeat_factors(y_factor, x_factor)?;
    if height % y_factor != 0 || width % x_factor != 0 {
        return Err(RustySatError::invalid_input(
            "native aggregation factors must evenly divide the source shape",
        ));
    }
    Ok(())
}

fn aggregate_block_mean(
    values: &[f64],
    mask: Option<&ValidityMask>,
    width: usize,
    out_y: usize,
    out_x: usize,
    y_factor: usize,
    x_factor: usize,
) -> (f64, bool) {
    let mut sum = 0.0;
    let mut count = 0usize;
    for dy in 0..y_factor {
        let src_y = out_y * y_factor + dy;
        for dx in 0..x_factor {
            let src_x = out_x * x_factor + dx;
            let src_idx = src_y * width + src_x;
            let masked = mask
                .and_then(|mask| mask.is_masked(src_idx))
                .unwrap_or(false);
            if masked {
                continue;
            }
            let value = values[src_idx];
            if value.is_finite() {
                sum += value;
                count += 1;
            }
        }
    }
    if count == 0 {
        (f64::NAN, true)
    } else {
        (sum / count as f64, false)
    }
}

fn finish_native_grid(
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

fn add_native_coords(
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

fn add_native_coords_owned(
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
    use super::*;
    use rusty_sat_core::{DataId, MetadataValue};

    fn area(id: &str, height: usize, width: usize) -> AreaDefinition {
        AreaDefinition::new(id, height, width).unwrap()
    }

    #[test]
    fn repeats_grid_and_mask_by_integer_factors() {
        let grid = DataGrid::new(2, 2, vec![1.0, 2.0, 3.0, 4.0])
            .unwrap()
            .with_mask(ValidityMask::from_masked_flags([false, true, false, false]))
            .unwrap();

        let repeated = native_repeat_2d(&grid, 2, 3).unwrap();

        assert_eq!(repeated.shape(), (4, 6));
        assert_eq!(
            repeated.values(),
            &[
                1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0,
                4.0, 4.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0
            ]
        );
        assert!(repeated.is_masked(3).unwrap());
        assert!(repeated.is_masked(9).unwrap());
        assert!(!repeated.is_masked(12).unwrap());
    }

    #[test]
    fn aggregates_grid_by_nanmean_and_masks_empty_blocks() {
        let grid = DataGrid::new(
            4,
            4,
            vec![
                1.0,
                2.0,
                5.0,
                7.0,
                3.0,
                f64::NAN,
                11.0,
                13.0,
                17.0,
                19.0,
                23.0,
                29.0,
                31.0,
                37.0,
                f64::NAN,
                f64::NAN,
            ],
        )
        .unwrap()
        .with_mask(ValidityMask::from_masked_flags([
            false, false, false, false, false, true, false, false, false, false, false, false,
            false, false, true, true,
        ]))
        .unwrap();

        let aggregated = native_aggregate_mean_2d(&grid, 2, 2).unwrap();

        assert_eq!(aggregated.shape(), (2, 2));
        assert_eq!(aggregated.values()[0], 2.0);
        assert_eq!(aggregated.values()[1], 9.0);
        assert_eq!(aggregated.values()[2], 26.0);
        assert_eq!(aggregated.values()[3], 26.0);
        assert!(aggregated.mask().is_none());
    }

    #[test]
    fn native_resampler_rejects_mixed_axis_direction() {
        let grid = DataGrid::new(2, 4, (0..8).map(f64::from).collect()).unwrap();
        let err = native_resample_2d(&grid, &area("mixed", 4, 2)).unwrap_err();
        assert!(err
            .to_string()
            .contains("cannot mix expansion and reduction"));
    }

    #[test]
    fn resampler_preserves_metadata_and_adds_destination_coords() {
        let source = area("source", 2, 2);
        let destination = area("destination", 4, 4);
        let id = DataId::new("counts").unwrap();
        let grid = DataGrid::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let mut dataset = Dataset::new(id.clone()).with_data(grid);
        dataset.insert_metadata("sensor", "ahi").unwrap();
        dataset
            .insert_attr("nested", MetadataValue::string("kept"))
            .unwrap();

        let resampler = NativeResampler::new(source);
        let resampled = resampler.resample(&dataset, &destination).unwrap();

        assert_eq!(resampled.id(), &id);
        assert_eq!(resampled.metadata().get("sensor").unwrap(), "ahi");
        assert_eq!(resampled.metadata().get("area").unwrap(), "destination");
        assert_eq!(resampled.metadata().get("resampler").unwrap(), "native");
        let array = resampled.data().unwrap();
        assert_eq!(array.shape(), (4, 4));
        assert!(array.coord("x").is_some());
        assert!(array.coord("y").is_some());
        assert_eq!(
            resampled.attr("nested"),
            Some(&MetadataValue::string("kept"))
        );
    }
}
