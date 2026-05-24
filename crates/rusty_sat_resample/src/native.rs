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
use rusty_sat_core::{
    AnyDataArray, Coordinate, DataArray, DataGrid, Dataset, NumericElement, Result, RustySatError,
    ValidityMask,
};
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
        let source_array = dataset.array().ok_or_else(|| {
            RustySatError::invalid_input("native resampling requires dataset array values")
        })?;
        if source_array.shape_yx()? != self.source.shape() {
            return Err(RustySatError::invalid_input(format!(
                "dataset grid y/x shape {:?} does not match source area shape {:?}",
                source_array.shape_yx()?,
                self.source.shape()
            )));
        }
        validate_native_area_compatibility(&self.source, destination)?;

        let resampled = native_resample_any_yx(source_array, destination)?;
        let mut resampled_dataset = Dataset::new(dataset.id().clone()).with_array(resampled);
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
        let source_array = dataset.into_array().ok_or_else(|| {
            RustySatError::invalid_input("native resampling requires dataset array values")
        })?;
        if source_array.shape_yx()? != self.source.shape() {
            return Err(RustySatError::invalid_input(format!(
                "dataset grid y/x shape {:?} does not match source area shape {:?}",
                source_array.shape_yx()?,
                self.source.shape()
            )));
        }
        validate_native_area_compatibility(&self.source, destination)?;

        let resampled = native_resample_any_yx_owned(source_array, destination)?;
        let mut resampled_dataset = Dataset::new(id).with_array(resampled);
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

pub fn native_resample_any_yx(
    source_array: &AnyDataArray,
    destination: &AreaDefinition,
) -> Result<AnyDataArray> {
    let source_shape = source_array.shape_yx()?;
    let destination_shape = destination.shape();
    match native_scale(source_shape, destination_shape)? {
        NativeScale::Identity => add_native_coords_any(source_array.clone(), destination),
        NativeScale::Repeat { y_factor, x_factor } => {
            let repeated = match source_array {
                AnyDataArray::F32(array) => {
                    native_repeat_yx_typed(array, y_factor, x_factor)?.into()
                }
                AnyDataArray::F64(array) => native_repeat_yx(array, y_factor, x_factor)?.into(),
                AnyDataArray::U8(array) => {
                    native_repeat_yx_typed(array, y_factor, x_factor)?.into()
                }
                AnyDataArray::U16(array) => {
                    native_repeat_yx_typed(array, y_factor, x_factor)?.into()
                }
                AnyDataArray::I16(array) => {
                    native_repeat_yx_typed(array, y_factor, x_factor)?.into()
                }
            };
            add_native_coords_any(repeated, destination)
        }
        NativeScale::Aggregate { y_factor, x_factor } => aggregate_any_mean_yx_from_parts(
            source_array.shape().to_vec(),
            source_array.dims().to_vec(),
            source_shape,
            source_array.values_as_f64(),
            source_array.mask().cloned(),
            source_array.coords().clone(),
            y_factor,
            x_factor,
            destination,
        ),
    }
}

pub fn native_resample_any_yx_owned(
    source_array: AnyDataArray,
    destination: &AreaDefinition,
) -> Result<AnyDataArray> {
    let source_shape = source_array.shape_yx()?;
    let destination_shape = destination.shape();
    match native_scale(source_shape, destination_shape)? {
        NativeScale::Identity => add_native_coords_any(source_array, destination),
        NativeScale::Repeat { y_factor, x_factor } => {
            let repeated = match source_array {
                AnyDataArray::F32(array) => {
                    native_repeat_yx_typed_owned(array, y_factor, x_factor)?.into()
                }
                AnyDataArray::F64(array) => {
                    native_repeat_yx_owned(array, y_factor, x_factor)?.into()
                }
                AnyDataArray::U8(array) => {
                    native_repeat_yx_typed_owned(array, y_factor, x_factor)?.into()
                }
                AnyDataArray::U16(array) => {
                    native_repeat_yx_typed_owned(array, y_factor, x_factor)?.into()
                }
                AnyDataArray::I16(array) => {
                    native_repeat_yx_typed_owned(array, y_factor, x_factor)?.into()
                }
            };
            add_native_coords_any(repeated, destination)
        }
        NativeScale::Aggregate { y_factor, x_factor } => {
            let shape = source_array.shape().to_vec();
            let dims = source_array.dims().to_vec();
            let coords = source_array.coords().clone();
            let (values, mask) = source_array.into_f64_values_and_mask();
            aggregate_any_mean_yx_from_parts(
                shape,
                dims,
                source_shape,
                values,
                mask,
                coords,
                y_factor,
                x_factor,
                destination,
            )
        }
    }
}

pub fn native_resample_yx(
    source_grid: &DataGrid,
    destination: &AreaDefinition,
) -> Result<DataGrid> {
    let source_shape = source_grid.shape_yx()?;
    let destination_shape = destination.shape();
    match native_scale(source_shape, destination_shape)? {
        NativeScale::Identity => {
            add_native_coords(source_grid.clone(), Some(source_grid.coords()), destination)
        }
        NativeScale::Repeat { y_factor, x_factor } => {
            let repeated = native_repeat_yx(source_grid, y_factor, x_factor)?;
            add_native_coords(repeated, Some(source_grid.coords()), destination)
        }
        NativeScale::Aggregate { y_factor, x_factor } => {
            let aggregated = native_aggregate_mean_yx(source_grid, y_factor, x_factor)?;
            add_native_coords(aggregated, Some(source_grid.coords()), destination)
        }
    }
}

pub fn native_resample_yx_owned(
    source_grid: DataGrid,
    destination: &AreaDefinition,
) -> Result<DataGrid> {
    let source_shape = source_grid.shape_yx()?;
    let destination_shape = destination.shape();
    match native_scale(source_shape, destination_shape)? {
        NativeScale::Identity => {
            let shape = source_grid.shape_nd().to_vec();
            let dims = source_grid.dims().to_vec();
            let (values, source_coords, mask) = source_grid.into_parts();
            let mut grid = DataArray::from_vec_named(shape, dims, values)?;
            if let Some(mask) = mask {
                grid.set_mask(mask)?;
            }
            add_native_coords_owned(grid, Some(source_coords), destination)
        }
        NativeScale::Repeat { y_factor, x_factor } => {
            let shape = source_grid.shape_nd().to_vec();
            let dims = source_grid.dims().to_vec();
            let (values, source_coords, mask) = source_grid.into_parts();
            let repeated =
                repeat_yx_from_parts(shape, dims, values, mask, source_coords, y_factor, x_factor)?;
            add_native_coords(repeated, None, destination)
        }
        NativeScale::Aggregate { y_factor, x_factor } => {
            let shape = source_grid.shape_nd().to_vec();
            let dims = source_grid.dims().to_vec();
            let (values, source_coords, mask) = source_grid.into_parts();
            let aggregated = aggregate_mean_yx_from_parts(
                shape,
                dims,
                source_shape,
                values,
                mask,
                source_coords,
                y_factor,
                x_factor,
            )?;
            add_native_coords(aggregated, None, destination)
        }
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

pub fn native_repeat_yx(
    source_grid: &DataGrid,
    y_factor: usize,
    x_factor: usize,
) -> Result<DataGrid> {
    validate_repeat_factors(y_factor, x_factor)?;
    source_grid.shape_yx()?;
    let source_values = source_grid.values().to_vec();
    let source_mask = source_grid.mask().cloned();
    let source_coords = source_grid.coords().clone();
    repeat_yx_from_parts(
        source_grid.shape_nd().to_vec(),
        source_grid.dims().to_vec(),
        source_values,
        source_mask,
        source_coords,
        y_factor,
        x_factor,
    )
}

pub fn native_repeat_yx_owned(
    source_grid: DataGrid,
    y_factor: usize,
    x_factor: usize,
) -> Result<DataGrid> {
    source_grid.shape_yx()?;
    let shape = source_grid.shape_nd().to_vec();
    let dims = source_grid.dims().to_vec();
    let (source_values, source_coords, source_mask) = source_grid.into_parts();
    repeat_yx_from_parts(
        shape,
        dims,
        source_values,
        source_mask,
        source_coords,
        y_factor,
        x_factor,
    )
}

pub fn native_repeat_yx_typed<T: NumericElement>(
    source_array: &DataArray<T>,
    y_factor: usize,
    x_factor: usize,
) -> Result<DataArray<T>> {
    source_array.shape_yx()?;
    repeat_yx_typed_from_parts(
        source_array.shape_nd().to_vec(),
        source_array.dims().to_vec(),
        source_array.values().to_vec(),
        source_array.mask().cloned(),
        source_array.coords().clone(),
        y_factor,
        x_factor,
    )
}

pub fn native_repeat_yx_typed_owned<T: NumericElement>(
    source_array: DataArray<T>,
    y_factor: usize,
    x_factor: usize,
) -> Result<DataArray<T>> {
    source_array.shape_yx()?;
    let shape = source_array.shape_nd().to_vec();
    let dims = source_array.dims().to_vec();
    let (source_values, source_coords, source_mask) = source_array.into_parts();
    repeat_yx_typed_from_parts(
        shape,
        dims,
        source_values,
        source_mask,
        source_coords,
        y_factor,
        x_factor,
    )
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

pub fn native_aggregate_mean_yx(
    source_grid: &DataGrid,
    y_factor: usize,
    x_factor: usize,
) -> Result<DataGrid> {
    let source_shape = source_grid.shape_yx()?;
    let source_values = source_grid.values().to_vec();
    let source_mask = source_grid.mask().cloned();
    let source_coords = source_grid.coords().clone();
    aggregate_mean_yx_from_parts(
        source_grid.shape_nd().to_vec(),
        source_grid.dims().to_vec(),
        source_shape,
        source_values,
        source_mask,
        source_coords,
        y_factor,
        x_factor,
    )
}

pub fn native_aggregate_mean_yx_owned(
    source_grid: DataGrid,
    y_factor: usize,
    x_factor: usize,
) -> Result<DataGrid> {
    let source_shape = source_grid.shape_yx()?;
    let shape = source_grid.shape_nd().to_vec();
    let dims = source_grid.dims().to_vec();
    let (source_values, source_coords, source_mask) = source_grid.into_parts();
    aggregate_mean_yx_from_parts(
        shape,
        dims,
        source_shape,
        source_values,
        source_mask,
        source_coords,
        y_factor,
        x_factor,
    )
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

fn repeat_yx_from_parts(
    source_shape_nd: Vec<usize>,
    dims: Vec<String>,
    source_values: Vec<f64>,
    source_mask: Option<ValidityMask>,
    source_coords: BTreeMap<String, Coordinate>,
    y_factor: usize,
    x_factor: usize,
) -> Result<DataGrid> {
    validate_repeat_factors(y_factor, x_factor)?;
    let (y_dim, x_dim) = yx_dim_indices(&dims)?;
    let mut output_shape = source_shape_nd.clone();
    output_shape[y_dim] = output_shape[y_dim].checked_mul(y_factor).ok_or_else(|| {
        RustySatError::invalid_input("native y repeat output size overflows usize")
    })?;
    output_shape[x_dim] = output_shape[x_dim].checked_mul(x_factor).ok_or_else(|| {
        RustySatError::invalid_input("native x repeat output size overflows usize")
    })?;
    let source_strides = row_major_strides(&source_shape_nd)?;
    let output_strides = row_major_strides(&output_shape)?;
    let output_size = checked_shape_size(&output_shape)?;
    let mut values = Vec::with_capacity(output_size);
    let mut mask_flags = Vec::new();
    if source_mask.is_some() {
        mask_flags.reserve(output_size);
    }

    for output_idx in 0..output_size {
        let mut indexes = unravel_index(output_idx, &output_shape, &output_strides);
        indexes[y_dim] /= y_factor;
        indexes[x_dim] /= x_factor;
        let source_idx = linear_index(&indexes, &source_strides);
        values.push(source_values[source_idx]);
        if source_mask.is_some() {
            mask_flags.push(
                source_mask
                    .as_ref()
                    .and_then(|mask| mask.is_masked(source_idx))
                    .unwrap_or(false),
            );
        }
    }

    let grid = finish_native_array(output_shape, dims, values, mask_flags)?;
    add_preserved_native_coords_owned(grid, source_coords)
}

fn repeat_yx_typed_from_parts<T: NumericElement>(
    source_shape_nd: Vec<usize>,
    dims: Vec<String>,
    source_values: Vec<T>,
    source_mask: Option<ValidityMask>,
    source_coords: BTreeMap<String, Coordinate>,
    y_factor: usize,
    x_factor: usize,
) -> Result<DataArray<T>> {
    validate_repeat_factors(y_factor, x_factor)?;
    let (y_dim, x_dim) = yx_dim_indices(&dims)?;
    let mut output_shape = source_shape_nd.clone();
    output_shape[y_dim] = output_shape[y_dim].checked_mul(y_factor).ok_or_else(|| {
        RustySatError::invalid_input("native y repeat output size overflows usize")
    })?;
    output_shape[x_dim] = output_shape[x_dim].checked_mul(x_factor).ok_or_else(|| {
        RustySatError::invalid_input("native x repeat output size overflows usize")
    })?;
    let source_strides = row_major_strides(&source_shape_nd)?;
    let output_strides = row_major_strides(&output_shape)?;
    let output_size = checked_shape_size(&output_shape)?;
    let mut values = Vec::with_capacity(output_size);
    let mut mask_flags = Vec::new();
    if source_mask.is_some() {
        mask_flags.reserve(output_size);
    }

    for output_idx in 0..output_size {
        let mut indexes = unravel_index(output_idx, &output_shape, &output_strides);
        indexes[y_dim] /= y_factor;
        indexes[x_dim] /= x_factor;
        let source_idx = linear_index(&indexes, &source_strides);
        values.push(source_values[source_idx]);
        if source_mask.is_some() {
            mask_flags.push(
                source_mask
                    .as_ref()
                    .and_then(|mask| mask.is_masked(source_idx))
                    .unwrap_or(false),
            );
        }
    }

    let array = finish_native_array_typed(output_shape, dims, values, mask_flags)?;
    add_preserved_native_coords_owned_typed(array, source_coords)
}

fn aggregate_mean_yx_from_parts(
    source_shape_nd: Vec<usize>,
    dims: Vec<String>,
    source_yx_shape: (usize, usize),
    source_values: Vec<f64>,
    source_mask: Option<ValidityMask>,
    source_coords: BTreeMap<String, Coordinate>,
    y_factor: usize,
    x_factor: usize,
) -> Result<DataGrid> {
    let (height, width) = source_yx_shape;
    validate_aggregate_factors(height, width, y_factor, x_factor)?;
    let (y_dim, x_dim) = yx_dim_indices(&dims)?;
    let mut output_shape = source_shape_nd.clone();
    output_shape[y_dim] /= y_factor;
    output_shape[x_dim] /= x_factor;
    let source_strides = row_major_strides(&source_shape_nd)?;
    let output_strides = row_major_strides(&output_shape)?;
    let output_size = checked_shape_size(&output_shape)?;
    let mut values = Vec::with_capacity(output_size);
    let mut mask_flags = Vec::with_capacity(output_size);

    for output_idx in 0..output_size {
        let output_indexes = unravel_index(output_idx, &output_shape, &output_strides);
        let mut sum = 0.0;
        let mut count = 0usize;
        for dy in 0..y_factor {
            for dx in 0..x_factor {
                let mut source_indexes = output_indexes.clone();
                source_indexes[y_dim] = output_indexes[y_dim] * y_factor + dy;
                source_indexes[x_dim] = output_indexes[x_dim] * x_factor + dx;
                let source_idx = linear_index(&source_indexes, &source_strides);
                let masked = source_mask
                    .as_ref()
                    .and_then(|mask| mask.is_masked(source_idx))
                    .unwrap_or(false);
                if masked {
                    continue;
                }
                let value = source_values[source_idx];
                if value.is_finite() {
                    sum += value;
                    count += 1;
                }
            }
        }
        if count == 0 {
            values.push(f64::NAN);
            mask_flags.push(true);
        } else {
            values.push(sum / count as f64);
            mask_flags.push(false);
        }
    }

    let grid = finish_native_array(output_shape, dims, values, mask_flags)?;
    add_preserved_native_coords_owned(grid, source_coords)
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
    if !height.is_multiple_of(y_factor) || !width.is_multiple_of(x_factor) {
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

fn finish_native_array(
    shape: Vec<usize>,
    dims: Vec<String>,
    values: Vec<f64>,
    mask_flags: Vec<bool>,
) -> Result<DataGrid> {
    let grid = DataArray::from_vec_named(shape, dims, values)?;
    if mask_flags.iter().any(|masked| *masked) {
        grid.with_mask(ValidityMask::from_masked_flags(mask_flags))
    } else {
        Ok(grid)
    }
}

fn finish_native_array_typed<T: NumericElement>(
    shape: Vec<usize>,
    dims: Vec<String>,
    values: Vec<T>,
    mask_flags: Vec<bool>,
) -> Result<DataArray<T>> {
    let array = DataArray::from_vec_named(shape, dims, values)?;
    if mask_flags.iter().any(|masked| *masked) {
        array.with_mask(ValidityMask::from_masked_flags(mask_flags))
    } else {
        Ok(array)
    }
}

fn yx_dim_indices(dims: &[String]) -> Result<(usize, usize)> {
    let y_dim = dims.iter().position(|dim| dim == "y").ok_or_else(|| {
        RustySatError::invalid_input("native y/x resampling requires a 'y' dimension")
    })?;
    let x_dim = dims.iter().position(|dim| dim == "x").ok_or_else(|| {
        RustySatError::invalid_input("native y/x resampling requires an 'x' dimension")
    })?;
    Ok((y_dim, x_dim))
}

fn checked_shape_size(shape: &[usize]) -> Result<usize> {
    crate::nd_utils::checked_shape_size(shape)
}

fn row_major_strides(shape: &[usize]) -> Result<Vec<usize>> {
    crate::nd_utils::row_major_strides(shape)
}

fn unravel_index(mut index: usize, shape: &[usize], strides: &[usize]) -> Vec<usize> {
    let mut indexes = Vec::with_capacity(shape.len());
    for (dim, stride) in shape.iter().zip(strides) {
        let value = index / *stride;
        indexes.push(value % *dim);
        index %= *stride;
    }
    indexes
}

fn linear_index(indexes: &[usize], strides: &[usize]) -> usize {
    indexes
        .iter()
        .zip(strides)
        .map(|(index, stride)| index * stride)
        .sum()
}

fn add_preserved_native_coords_owned(
    mut grid: DataGrid,
    source_coords: BTreeMap<String, Coordinate>,
) -> Result<DataGrid> {
    for (name, coordinate) in source_coords {
        if should_preserve_coord(&name, &coordinate) {
            grid.set_coordinate(name, coordinate)?;
        }
    }
    Ok(grid)
}

fn add_preserved_native_coords_owned_typed<T: NumericElement>(
    mut array: DataArray<T>,
    source_coords: BTreeMap<String, Coordinate>,
) -> Result<DataArray<T>> {
    for (name, coordinate) in source_coords {
        if should_preserve_coord(&name, &coordinate) {
            array.set_coordinate(name, coordinate)?;
        }
    }
    Ok(array)
}

fn aggregate_any_mean_yx_from_parts(
    source_shape_nd: Vec<usize>,
    dims: Vec<String>,
    source_yx_shape: (usize, usize),
    source_values: Vec<f64>,
    source_mask: Option<ValidityMask>,
    source_coords: BTreeMap<String, Coordinate>,
    y_factor: usize,
    x_factor: usize,
    destination: &AreaDefinition,
) -> Result<AnyDataArray> {
    let aggregated = aggregate_mean_yx_from_parts(
        source_shape_nd,
        dims,
        source_yx_shape,
        source_values,
        source_mask,
        source_coords,
        y_factor,
        x_factor,
    )?;
    Ok(add_native_coords(aggregated, None, destination)?.into())
}

fn add_native_coords_any(array: AnyDataArray, area: &AreaDefinition) -> Result<AnyDataArray> {
    match array {
        AnyDataArray::F32(array) => Ok(add_native_coords_typed(array, area)?.into()),
        AnyDataArray::F64(array) => Ok(add_native_coords_owned(array, None, area)?.into()),
        AnyDataArray::U8(array) => Ok(add_native_coords_typed(array, area)?.into()),
        AnyDataArray::U16(array) => Ok(add_native_coords_typed(array, area)?.into()),
        AnyDataArray::I16(array) => Ok(add_native_coords_typed(array, area)?.into()),
    }
}

fn add_native_coords_typed<T: NumericElement>(
    mut array: DataArray<T>,
    area: &AreaDefinition,
) -> Result<DataArray<T>> {
    array.set_coordinate(
        "x",
        Coordinate::axis("x", area.iter_projection_x_coords().collect::<Vec<_>>())?,
    )?;
    array.set_coordinate(
        "y",
        Coordinate::axis("y", area.iter_projection_y_coords().collect::<Vec<_>>())?,
    )?;
    Ok(array)
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
    fn repeats_named_yx_axes_in_band_major_array() {
        let array = DataArray::from_vec_named(
            vec![2, 2, 2],
            ["bands", "y", "x"],
            vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
        )
        .unwrap()
        .with_mask(ValidityMask::from_masked_flags([
            false, true, false, false, false, false, true, false,
        ]))
        .unwrap()
        .with_coordinate("bands", Coordinate::axis("bands", vec![0.6, 0.8]).unwrap())
        .unwrap();

        let repeated = native_repeat_yx(&array, 2, 2).unwrap();

        assert_eq!(repeated.shape_nd(), &[2, 4, 4]);
        assert_eq!(repeated.dims(), &["bands", "y", "x"]);
        assert_eq!(
            &repeated.values()[..8],
            &[1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0]
        );
        assert_eq!(
            &repeated.values()[16..24],
            &[10.0, 10.0, 20.0, 20.0, 10.0, 10.0, 20.0, 20.0]
        );
        assert_eq!(repeated.mask().unwrap().masked_count(), 8);
        assert!(repeated.coord("bands").is_some());
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
    fn aggregates_named_yx_axes_in_band_major_array() {
        let array = DataArray::from_vec_named(
            vec![2, 2, 2],
            ["bands", "y", "x"],
            vec![1.0, 3.0, 5.0, 7.0, 10.0, 30.0, 50.0, f64::NAN],
        )
        .unwrap()
        .with_mask(ValidityMask::from_masked_flags([
            false, false, false, false, false, false, false, true,
        ]))
        .unwrap();

        let aggregated = native_aggregate_mean_yx(&array, 2, 2).unwrap();

        assert_eq!(aggregated.shape_nd(), &[2, 1, 1]);
        assert_eq!(aggregated.dims(), &["bands", "y", "x"]);
        assert_eq!(aggregated.values(), &[4.0, 30.0]);
        assert!(aggregated.mask().is_none());
    }

    #[test]
    fn native_resampler_accepts_higher_dimensional_yx_arrays() {
        let source = area("source", 2, 2);
        let destination = area("destination", 4, 4);
        let id = DataId::new("rgb").unwrap();
        let array = DataArray::from_vec_named(
            vec![2, 2, 2],
            ["bands", "y", "x"],
            vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
        )
        .unwrap();
        let dataset = Dataset::new(id).with_array(array);

        let output = NativeResampler::new(source)
            .resample(&dataset, &destination)
            .unwrap();

        let output_array = output.array().unwrap();
        assert_eq!(output_array.shape(), &[2, 4, 4]);
        assert_eq!(output_array.dims(), &["bands", "y", "x"]);
        assert_eq!(
            output.metadata().get("resampler"),
            Some(&"native".to_string())
        );
    }

    #[test]
    fn native_resampler_repeats_runtime_typed_arrays_without_promoting_dtype() {
        let source = area("source", 2, 2);
        let destination = area("destination", 4, 4);
        let id = DataId::new("counts").unwrap();
        let array = DataArray::from_vec_named(vec![2, 2], ["y", "x"], vec![1_u16, 2, 3, 4])
            .unwrap()
            .with_mask(ValidityMask::from_masked_flags([false, true, false, false]))
            .unwrap();
        let dataset = Dataset::new(id).with_array(array);

        let output = NativeResampler::new(source)
            .resample(&dataset, &destination)
            .unwrap();

        let output_array = output.array().unwrap();
        assert_eq!(output_array.dtype(), rusty_sat_core::DataType::U16);
        assert_eq!(output_array.shape(), &[4, 4]);
        assert_eq!(output_array.values_as_f64()[..4], [1.0, 1.0, 2.0, 2.0]);
        assert_eq!(output_array.mask().unwrap().masked_count(), 4);
        assert!(output_array.coord("x").is_some());
        assert!(output_array.coord("y").is_some());
    }

    #[test]
    fn native_resample_any_yx_owned_identity_preserves_dtype() {
        let destination = area("destination", 2, 2);
        let array = DataArray::from_vec_named(vec![2, 2], ["y", "x"], vec![1_u8, 2, 3, 4])
            .unwrap()
            .with_mask(ValidityMask::from_masked_flags([false, false, true, false]))
            .unwrap()
            .with_coordinate("quality", Coordinate::scalar(1.0))
            .unwrap();

        let output = native_resample_any_yx_owned(array.into(), &destination).unwrap();

        assert_eq!(output.dtype(), rusty_sat_core::DataType::U8);
        assert_eq!(output.shape(), &[2, 2]);
        assert_eq!(output.values_as_f64(), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(output.mask().unwrap().masked_count(), 1);
        assert!(output.coord("x").is_some());
        assert!(output.coord("y").is_some());
        assert_eq!(output.coord("quality").unwrap().values(), &[1.0]);
    }

    #[test]
    fn native_resampler_aggregates_runtime_typed_arrays_to_f64_means() {
        let source = area("source", 4, 4);
        let destination = area("destination", 2, 2);
        let id = DataId::new("counts").unwrap();
        let array = DataArray::from_vec_named(
            vec![4, 4],
            ["y", "x"],
            vec![1_u16, 3, 5, 7, 9, 11, 13, 15, 2, 4, 6, 8, 10, 12, 14, 16],
        )
        .unwrap();
        let dataset = Dataset::new(id).with_array(array);

        let output = NativeResampler::new(source)
            .resample_owned(dataset, &destination)
            .unwrap();

        let output_array = output.array().unwrap();
        assert_eq!(output_array.dtype(), rusty_sat_core::DataType::F64);
        assert_eq!(output_array.shape(), &[2, 2]);
        assert_eq!(output_array.values_as_f64(), &[6.0, 10.0, 7.0, 11.0]);
        assert!(output_array.coord("x").is_some());
        assert!(output_array.coord("y").is_some());
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

    #[test]
    fn identity_passes_through_unchanged() {
        let destination = area("destination", 2, 3);
        let grid = DataGrid::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap()
            .with_mask(ValidityMask::from_masked_flags([
                false, true, false, false, false, false,
            ]))
            .unwrap();

        let result = native_resample_2d(&grid, &destination).unwrap();

        assert_eq!(result.shape(), (2, 3));
        assert_eq!(result.values(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(result.mask().unwrap().masked_count(), 1);
        assert!(result.coord("x").is_some());
        assert!(result.coord("y").is_some());
    }

    #[test]
    fn owned_repeat_and_aggregate_produce_same_output_as_borrowed() {
        let grid = DataGrid::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let borrowed = native_repeat_2d(&grid, 2, 2).unwrap();
        let owned = native_repeat_2d_owned(grid, 2, 2).unwrap();
        assert_eq!(borrowed.values(), owned.values());
        assert_eq!(borrowed.mask(), owned.mask());

        let grid = DataGrid::new(4, 4, (0..16).map(f64::from).collect()).unwrap();
        let borrowed = native_aggregate_mean_2d(&grid, 2, 2).unwrap();
        let owned = native_aggregate_mean_2d_owned(grid, 2, 2).unwrap();
        assert_eq!(borrowed.values(), owned.values());
        assert_eq!(borrowed.mask(), owned.mask());
    }

    #[test]
    fn repeat_rejects_zero_factors() {
        let grid = DataGrid::new(1, 1, vec![1.0]).unwrap();

        assert!(native_repeat_2d(&grid, 0, 2).is_err());
        assert!(native_repeat_2d(&grid, 2, 0).is_err());
    }

    #[test]
    fn aggregate_rejects_non_integer_factors() {
        let grid = DataGrid::new(3, 3, (0..9).map(f64::from).collect()).unwrap();

        assert!(native_aggregate_mean_2d(&grid, 2, 2).is_err());
    }

    #[test]
    fn native_resampler_rejects_different_projections() {
        let source = AreaDefinition::from_parts(
            "source",
            "source",
            "source",
            BTreeMap::from([("proj".to_string(), "longlat".to_string())]),
            2,
            2,
            [0.0, 0.0, 2.0, 2.0],
        )
        .unwrap();
        let destination = AreaDefinition::from_parts(
            "destination",
            "destination",
            "destination",
            BTreeMap::from([("proj".to_string(), "merc".to_string())]),
            2,
            2,
            [0.0, 0.0, 2.0, 2.0],
        )
        .unwrap();
        let id = DataId::new("image").unwrap();
        let dataset =
            Dataset::new(id).with_data(DataGrid::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap());
        let resampler = NativeResampler::new(source);

        assert!(resampler.resample(&dataset, &destination).is_err());
    }

    #[test]
    fn yx_methods_reject_missing_y_or_x_dimensions() {
        let array =
            DataArray::from_vec_named(vec![2, 2], ["bands", "time"], vec![1.0, 2.0, 3.0, 4.0])
                .unwrap();

        assert!(native_repeat_yx(&array, 2, 2).is_err());
        assert!(native_aggregate_mean_yx(&array, 2, 2).is_err());
    }

    #[test]
    fn yx_owned_produces_same_output_as_borrowed() {
        let array = DataArray::from_vec_named(
            vec![2, 2, 2],
            ["bands", "y", "x"],
            vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
        )
        .unwrap()
        .with_mask(ValidityMask::from_masked_flags([
            false, true, false, false, false, false, true, false,
        ]))
        .unwrap();

        let borrowed_repeat = native_repeat_yx(&array, 2, 2).unwrap();
        let owned_repeat = native_repeat_yx_owned(array.clone(), 2, 2).unwrap();
        assert_eq!(borrowed_repeat.values(), owned_repeat.values());
        assert_eq!(borrowed_repeat.mask(), owned_repeat.mask());

        let borrowed_agg = native_aggregate_mean_yx(&array, 2, 2).unwrap();
        let owned_agg = native_aggregate_mean_yx_owned(array, 2, 2).unwrap();
        assert_eq!(borrowed_agg.values(), owned_agg.values());
        assert_eq!(borrowed_agg.mask(), owned_agg.mask());
    }

    #[test]
    fn native_resample_yx_owned_produces_same_output_as_borrowed() {
        let destination = area("destination", 4, 4);
        let borrowed = DataArray::from_vec_named(
            vec![2, 2, 2],
            ["bands", "y", "x"],
            vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
        )
        .unwrap();
        let owned = borrowed.clone();

        let borrowed_result = native_resample_yx(&borrowed, &destination).unwrap();
        let owned_result = native_resample_yx_owned(owned, &destination).unwrap();

        assert_eq!(borrowed_result.values(), owned_result.values());
        assert_eq!(borrowed_result.shape_nd(), owned_result.shape_nd());
        assert_eq!(borrowed_result.dims(), owned_result.dims());
    }
}
