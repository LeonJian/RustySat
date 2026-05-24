//! Line/sample indexing helpers.
//!
//! Reference behavior inspected before implementation:
//! - `deps/pyresample/pyresample/grid.py::get_image_from_linesample`
//! - `deps/pyresample/pyresample/image.py::ImageContainer.get_array_from_linesample`
//!
//! This is the first Rust-native foundation for Pyresample's line/sample path.
//! It supports both the early `DataGrid` API and runtime-typed arrays so callers
//! can keep integer/HDR buffers out of unnecessary `f64` promotion.

use rusty_sat_core::{
    AnyDataArray, Coordinate, DataArray, DataGrid, NumericElement, Result, RustySatError,
    ValidityMask,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LineSampleGrid {
    height: usize,
    width: usize,
    rows: Vec<isize>,
    cols: Vec<isize>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LineSampleFillValue {
    F32(f32),
    F64(f64),
    U8(u8),
    U16(u16),
    I16(i16),
}

impl LineSampleFillValue {
    pub fn f32(value: f32) -> Self {
        Self::F32(value)
    }

    pub fn f64(value: f64) -> Self {
        Self::F64(value)
    }

    pub fn u8(value: u8) -> Self {
        Self::U8(value)
    }

    pub fn u16(value: u16) -> Self {
        Self::U16(value)
    }

    pub fn i16(value: i16) -> Self {
        Self::I16(value)
    }

    fn as_f32(self) -> Result<f32> {
        match self {
            Self::F32(value) => Ok(value),
            _ => Err(fill_type_error("f32")),
        }
    }

    fn as_f64(self) -> Result<f64> {
        match self {
            Self::F64(value) => Ok(value),
            _ => Err(fill_type_error("f64")),
        }
    }

    fn as_u8(self) -> Result<u8> {
        match self {
            Self::U8(value) => Ok(value),
            _ => Err(fill_type_error("u8")),
        }
    }

    fn as_u16(self) -> Result<u16> {
        match self {
            Self::U16(value) => Ok(value),
            _ => Err(fill_type_error("u16")),
        }
    }

    fn as_i16(self) -> Result<i16> {
        match self {
            Self::I16(value) => Ok(value),
            _ => Err(fill_type_error("i16")),
        }
    }
}

impl LineSampleGrid {
    pub fn new(
        height: usize,
        width: usize,
        rows: impl Into<Vec<isize>>,
        cols: impl Into<Vec<isize>>,
    ) -> Result<Self> {
        if height == 0 || width == 0 {
            return Err(RustySatError::invalid_input(
                "line/sample target shape must be non-zero",
            ));
        }
        let rows = rows.into();
        let cols = cols.into();
        let expected = height.checked_mul(width).ok_or_else(|| {
            RustySatError::invalid_input("line/sample target shape overflows usize")
        })?;
        if rows.len() != expected || cols.len() != expected {
            return Err(RustySatError::invalid_input(format!(
                "line/sample rows and cols must both have {expected} values"
            )));
        }
        Ok(Self {
            height,
            width,
            rows,
            cols,
        })
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.height, self.width)
    }

    pub fn rows(&self) -> &[isize] {
        &self.rows
    }

    pub fn cols(&self) -> &[isize] {
        &self.cols
    }

    pub fn len(&self) -> usize {
        self.rows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }
}

pub fn get_image_from_linesample(
    rows: &[isize],
    cols: &[isize],
    target_height: usize,
    target_width: usize,
    source: &DataGrid,
    fill_value: f64,
) -> Result<DataGrid> {
    let linesample =
        LineSampleGrid::new(target_height, target_width, rows.to_vec(), cols.to_vec())?;
    sample_grid_from_linesample(source, &linesample, fill_value, false)
}

pub fn get_image_from_linesample_masked_missing(
    rows: &[isize],
    cols: &[isize],
    target_height: usize,
    target_width: usize,
    source: &DataGrid,
) -> Result<DataGrid> {
    let linesample =
        LineSampleGrid::new(target_height, target_width, rows.to_vec(), cols.to_vec())?;
    sample_grid_from_linesample(source, &linesample, f64::NAN, true)
}

pub fn sample_grid_from_linesample(
    source: &DataGrid,
    linesample: &LineSampleGrid,
    fill_value: f64,
    mask_missing: bool,
) -> Result<DataGrid> {
    sample_array_from_linesample(source, linesample, fill_value, mask_missing)
}

pub fn sample_any_from_linesample(
    source: &AnyDataArray,
    linesample: &LineSampleGrid,
    fill_value: LineSampleFillValue,
    mask_missing: bool,
) -> Result<AnyDataArray> {
    Ok(match source {
        AnyDataArray::F32(array) => {
            sample_array_from_linesample(array, linesample, fill_value.as_f32()?, mask_missing)?
                .into()
        }
        AnyDataArray::F64(array) => {
            sample_array_from_linesample(array, linesample, fill_value.as_f64()?, mask_missing)?
                .into()
        }
        AnyDataArray::U8(array) => {
            sample_array_from_linesample(array, linesample, fill_value.as_u8()?, mask_missing)?
                .into()
        }
        AnyDataArray::U16(array) => {
            sample_array_from_linesample(array, linesample, fill_value.as_u16()?, mask_missing)?
                .into()
        }
        AnyDataArray::I16(array) => {
            sample_array_from_linesample(array, linesample, fill_value.as_i16()?, mask_missing)?
                .into()
        }
    })
}

pub fn sample_array_from_linesample<T: NumericElement>(
    source: &DataArray<T>,
    linesample: &LineSampleGrid,
    fill_value: T,
    mask_missing: bool,
) -> Result<DataArray<T>> {
    let y_axis = source.dim_index("y").ok_or_else(|| {
        RustySatError::invalid_input("line/sample source array is missing 'y' dimension")
    })?;
    let x_axis = source.dim_index("x").ok_or_else(|| {
        RustySatError::invalid_input("line/sample source array is missing 'x' dimension")
    })?;
    let source_shape = source.shape_nd();
    let source_height = source_shape[y_axis];
    let source_width = source_shape[x_axis];
    let mut output_shape = source_shape.to_vec();
    output_shape[y_axis] = linesample.height();
    output_shape[x_axis] = linesample.width();
    let output_len = crate::nd_utils::checked_shape_size(&output_shape)?;
    let source_strides = crate::nd_utils::row_major_strides(source_shape)?;
    let output_strides = crate::nd_utils::row_major_strides(&output_shape)?;
    let source_values = source.values();
    let source_mask = source.mask();
    let mut output = Vec::with_capacity(output_len);
    let mut mask_flags = mask_missing.then(|| Vec::with_capacity(output_len));

    for output_offset in 0..output_len {
        let y = (output_offset / output_strides[y_axis]) % output_shape[y_axis];
        let x = (output_offset / output_strides[x_axis]) % output_shape[x_axis];
        let line_sample_offset = y * linesample.width() + x;
        let row = linesample.rows()[line_sample_offset];
        let col = linesample.cols()[line_sample_offset];
        let source_index = valid_source_index(row, col, source_height, source_width);
        let is_missing = source_index
            .map(|yx_index| {
                let index = remap_output_to_source_offset(
                    output_offset,
                    &output_shape,
                    &output_strides,
                    &source_strides,
                    y_axis,
                    x_axis,
                    yx_index,
                    source_width,
                );
                source_mask
                    .and_then(|mask| mask.is_masked(index))
                    .unwrap_or(false)
            })
            .unwrap_or(true);

        if is_missing {
            output.push(fill_value);
            if let Some(flags) = mask_flags.as_mut() {
                flags.push(true);
            }
        } else {
            let yx_index = source_index.expect("non-missing line/sample must have a source index");
            let index = remap_output_to_source_offset(
                output_offset,
                &output_shape,
                &output_strides,
                &source_strides,
                y_axis,
                x_axis,
                yx_index,
                source_width,
            );
            output.push(source_values[index]);
            if let Some(flags) = mask_flags.as_mut() {
                flags.push(false);
            }
        }
    }

    let mut array =
        DataArray::from_vec_named(output_shape.clone(), source.dims().to_vec(), output)?;
    for (name, coord) in source.coords() {
        let sampled_coord = sample_coordinate_from_linesample(
            coord,
            source.dims(),
            source_shape,
            &output_shape,
            linesample,
            source_height,
            source_width,
        )?;
        array.set_coordinate(name.clone(), sampled_coord)?;
    }
    if let Some(flags) = mask_flags {
        array.set_mask(ValidityMask::from_masked_flags(flags))?;
    }
    Ok(array)
}

fn valid_source_index(
    row: isize,
    col: isize,
    source_height: usize,
    source_width: usize,
) -> Option<usize> {
    let row = usize::try_from(row).ok()?;
    let col = usize::try_from(col).ok()?;
    if row >= source_height || col >= source_width {
        return None;
    }
    Some(row * source_width + col)
}

#[allow(clippy::too_many_arguments)]
fn remap_output_to_source_offset(
    output_offset: usize,
    output_shape: &[usize],
    output_strides: &[usize],
    source_strides: &[usize],
    y_axis: usize,
    x_axis: usize,
    yx_index: usize,
    source_width: usize,
) -> usize {
    let source_y = yx_index / source_width;
    let source_x = yx_index % source_width;
    let mut source_offset = 0;
    for axis in 0..output_shape.len() {
        let index = if axis == y_axis {
            source_y
        } else if axis == x_axis {
            source_x
        } else {
            (output_offset / output_strides[axis]) % output_shape[axis]
        };
        source_offset += index * source_strides[axis];
    }
    source_offset
}

fn sample_coordinate_from_linesample(
    coord: &Coordinate,
    source_dims: &[String],
    source_shape: &[usize],
    output_shape: &[usize],
    linesample: &LineSampleGrid,
    source_height: usize,
    source_width: usize,
) -> Result<Coordinate> {
    if coord.dims().iter().all(|dim| dim != "y" && dim != "x") {
        return Ok(coord.clone());
    }

    let output_dims = sampled_coordinate_dims(coord, source_dims);
    let coord_source_shape = coordinate_shape(coord.dims(), source_dims, source_shape)?;
    let coord_output_shape = coordinate_shape(&output_dims, source_dims, output_shape)?;
    let coord_source_strides = crate::nd_utils::row_major_strides(&coord_source_shape)?;
    let coord_output_strides = crate::nd_utils::row_major_strides(&coord_output_shape)?;
    let coord_output_len = crate::nd_utils::checked_shape_size(&coord_output_shape)?;
    let mut values = Vec::with_capacity(coord_output_len);

    for output_offset in 0..coord_output_len {
        let output_indices =
            unravel_offset(output_offset, &coord_output_shape, &coord_output_strides);
        let y_index = coordinate_dim_index(&output_dims, "y")
            .map(|axis| output_indices[axis])
            .unwrap_or(0);
        let x_index = coordinate_dim_index(&output_dims, "x")
            .map(|axis| output_indices[axis])
            .unwrap_or(0);
        let line_sample_offset = y_index * linesample.width() + x_index;
        let source_index = valid_source_index(
            linesample.rows()[line_sample_offset],
            linesample.cols()[line_sample_offset],
            source_height,
            source_width,
        );

        let Some(yx_index) = source_index else {
            values.push(f64::NAN);
            continue;
        };

        let source_y = yx_index / source_width;
        let source_x = yx_index % source_width;
        let mut coord_source_offset = 0usize;
        for (axis, dim) in coord.dims().iter().enumerate() {
            let index = if dim == "y" {
                source_y
            } else if dim == "x" {
                source_x
            } else {
                let output_axis = coordinate_dim_index(&output_dims, dim).expect(
                    "non-spatial coordinate dim must be present in sampled coordinate dims",
                );
                output_indices[output_axis]
            };
            coord_source_offset += index * coord_source_strides[axis];
        }
        values.push(coord.values()[coord_source_offset]);
    }

    Coordinate::new(output_dims, values)
}

fn sampled_coordinate_dims(coord: &Coordinate, source_dims: &[String]) -> Vec<String> {
    source_dims
        .iter()
        .filter(|dim| dim.as_str() == "y" || dim.as_str() == "x" || coord.dims().contains(dim))
        .cloned()
        .collect()
}

fn coordinate_shape(
    coord_dims: &[String],
    data_dims: &[String],
    data_shape: &[usize],
) -> Result<Vec<usize>> {
    coord_dims
        .iter()
        .map(|dim| {
            data_dims
                .iter()
                .position(|candidate| candidate == dim)
                .map(|axis| data_shape[axis])
                .ok_or_else(|| {
                    RustySatError::invalid_input(format!(
                        "coordinate dimension '{dim}' is not a data dimension"
                    ))
                })
        })
        .collect()
}

fn unravel_offset(offset: usize, shape: &[usize], strides: &[usize]) -> Vec<usize> {
    (0..shape.len())
        .map(|axis| (offset / strides[axis]) % shape[axis])
        .collect()
}

fn coordinate_dim_index(dims: &[String], dim: &str) -> Option<usize> {
    dims.iter().position(|candidate| candidate == dim)
}

fn fill_type_error(expected: &str) -> RustySatError {
    RustySatError::invalid_input(format!(
        "line/sample fill value must match source dtype {expected}"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rusty_sat_core::DataType;

    fn source_grid() -> DataGrid {
        DataGrid::new(3, 3, (0..9).map(f64::from).collect()).unwrap()
    }

    #[test]
    fn line_sample_grid_validates_shape_and_lengths() {
        let lines = LineSampleGrid::new(2, 2, [0, 0, 1, 1], [0, 1, 0, 1]).unwrap();
        assert_eq!(lines.shape(), (2, 2));
        assert_eq!(lines.len(), 4);

        let err = LineSampleGrid::new(2, 2, [0, 1], [0, 1]).unwrap_err();
        assert!(err.to_string().contains("must both have 4 values"));
        assert!(LineSampleGrid::new(0, 2, Vec::<isize>::new(), Vec::<isize>::new()).is_err());
    }

    #[test]
    fn samples_grid_with_fill_for_invalid_indices() {
        let output =
            get_image_from_linesample(&[0, 1, -1, 3], &[0, 2, 0, 0], 2, 2, &source_grid(), -999.0)
                .unwrap();

        assert_eq!(output.shape(), (2, 2));
        assert_eq!(output.values(), &[0.0, 5.0, -999.0, -999.0]);
        assert!(output.mask().is_none());
    }

    #[test]
    fn masked_missing_marks_invalid_indices_and_source_masks() {
        let source = source_grid()
            .with_mask(ValidityMask::from_masked_flags([
                false, false, false, false, false, true, false, false, false,
            ]))
            .unwrap();
        let lines = LineSampleGrid::new(2, 2, [0, 1, -1, 2], [0, 2, 0, 2]).unwrap();

        let output = sample_grid_from_linesample(&source, &lines, f64::NAN, true).unwrap();

        assert_eq!(output.values()[0], 0.0);
        assert!(output.values()[1].is_nan());
        assert!(output.values()[2].is_nan());
        assert_eq!(output.values()[3], 8.0);
        assert_eq!(output.mask().unwrap().masked_count(), 2);
        assert_eq!(output.mask().unwrap().is_masked(1), Some(true));
        assert_eq!(output.mask().unwrap().is_masked(2), Some(true));
    }

    #[test]
    fn source_masks_are_filled_when_not_masking_missing() {
        let source = source_grid()
            .with_mask(ValidityMask::from_masked_flags([
                false, false, false, false, false, true, false, false, false,
            ]))
            .unwrap();
        let output = get_image_from_linesample(&[1], &[2], 1, 1, &source, -1.0).unwrap();

        assert_eq!(output.values(), &[-1.0]);
        assert!(output.mask().is_none());
    }

    #[test]
    fn samples_runtime_typed_arrays_without_dtype_promotion() {
        let source = AnyDataArray::from(
            DataArray::<u16>::from_vec_named(vec![2, 2], ["y", "x"], vec![10, 20, 30, 40]).unwrap(),
        );
        let lines = LineSampleGrid::new(2, 2, [0, 1, -1, 0], [0, 1, 0, 4]).unwrap();

        let output =
            sample_any_from_linesample(&source, &lines, LineSampleFillValue::u16(999), false)
                .unwrap();

        assert_eq!(output.dtype(), DataType::U16);
        assert_eq!(output.values_as_f64(), vec![10.0, 40.0, 999.0, 999.0]);
        assert!(output.mask().is_none());
    }

    #[test]
    fn runtime_typed_masked_missing_preserves_dtype_and_masks() {
        let source = AnyDataArray::from(
            DataArray::<i16>::from_vec_named(vec![2, 2], ["y", "x"], vec![1, 2, 3, 4])
                .unwrap()
                .with_mask(ValidityMask::from_masked_flags([false, true, false, false]))
                .unwrap(),
        );
        let lines = LineSampleGrid::new(2, 2, [0, 0, 1, -1], [0, 1, 1, 0]).unwrap();

        let output =
            sample_any_from_linesample(&source, &lines, LineSampleFillValue::i16(-999), true)
                .unwrap();

        assert_eq!(output.dtype(), DataType::I16);
        assert_eq!(output.values_as_f64(), vec![1.0, -999.0, 4.0, -999.0]);
        assert_eq!(output.mask().unwrap().masked_count(), 2);
    }

    #[test]
    fn runtime_typed_sampling_rejects_mismatched_fill_dtype() {
        let source = AnyDataArray::from(
            DataArray::<u8>::from_vec_named(vec![1, 1], ["y", "x"], vec![5]).unwrap(),
        );
        let lines = LineSampleGrid::new(1, 1, [0], [0]).unwrap();

        let err =
            sample_any_from_linesample(&source, &lines, LineSampleFillValue::f64(-1.0), false)
                .unwrap_err();

        assert!(err.to_string().contains("must match source dtype u8"));
    }

    #[test]
    fn samples_band_major_yx_arrays_without_flattening_bands() {
        let source = DataArray::<u16>::from_vec_named(
            vec![2, 2, 3],
            ["bands", "y", "x"],
            vec![
                10, 11, 12, 13, 14, 15, //
                20, 21, 22, 23, 24, 25,
            ],
        )
        .unwrap()
        .with_coordinate(
            "bands",
            rusty_sat_core::Coordinate::axis("bands", vec![0.6, 0.8]).unwrap(),
        )
        .unwrap();
        let lines = LineSampleGrid::new(1, 2, [0, 1], [2, 0]).unwrap();

        let output = sample_array_from_linesample(&source, &lines, 999, false).unwrap();

        assert_eq!(output.shape_nd(), &[2, 1, 2]);
        assert_eq!(output.dims(), &["bands", "y", "x"]);
        assert_eq!(output.values(), &[12, 13, 22, 23]);
        assert_eq!(output.coord("bands").unwrap().values(), &[0.6_f64, 0.8_f64]);
        assert!(output.coord("y").is_none());
        assert!(output.coord("x").is_none());
    }

    #[test]
    fn samples_yx_axes_not_at_end() {
        let source = DataArray::<u8>::from_vec_named(
            vec![2, 2, 2],
            ["y", "x", "bands"],
            vec![1, 2, 3, 4, 5, 6, 7, 8],
        )
        .unwrap();
        let lines = LineSampleGrid::new(1, 2, [0, 1], [1, 0]).unwrap();

        let output = sample_array_from_linesample(&source, &lines, 255, false).unwrap();

        assert_eq!(output.shape_nd(), &[1, 2, 2]);
        assert_eq!(output.dims(), &["y", "x", "bands"]);
        assert_eq!(output.values(), &[3, 4, 5, 6]);
    }

    #[test]
    fn band_major_sampling_propagates_per_band_masks() {
        let source = DataArray::<i16>::from_vec_named(
            vec![2, 2, 2],
            ["bands", "y", "x"],
            vec![1, 2, 3, 4, 5, 6, 7, 8],
        )
        .unwrap()
        .with_mask(ValidityMask::from_masked_flags([
            false, true, false, false, false, false, true, false,
        ]))
        .unwrap();
        let lines = LineSampleGrid::new(1, 2, [0, 1], [1, 0]).unwrap();

        let output = sample_array_from_linesample(&source, &lines, -999, true).unwrap();

        assert_eq!(output.shape_nd(), &[2, 1, 2]);
        assert_eq!(output.values(), &[-999, 3, 6, -999]);
        assert_eq!(output.mask().unwrap().masked_count(), 2);
        assert_eq!(output.mask().unwrap().is_masked(0), Some(true));
        assert_eq!(output.mask().unwrap().is_masked(3), Some(true));
    }

    #[test]
    fn line_sample_promotes_axis_coordinates_to_sampled_2d_coords() {
        let source =
            DataArray::<u8>::from_vec_named(vec![2, 3], ["y", "x"], vec![1, 2, 3, 4, 5, 6])
                .unwrap()
                .with_coordinate("y", Coordinate::axis("y", vec![10.0, 20.0]).unwrap())
                .unwrap()
                .with_coordinate(
                    "x",
                    Coordinate::axis("x", vec![100.0, 200.0, 300.0]).unwrap(),
                )
                .unwrap();
        let lines = LineSampleGrid::new(2, 2, [0, 1, -1, 0], [2, 0, 0, 3]).unwrap();

        let output = sample_array_from_linesample(&source, &lines, 255, false).unwrap();

        assert_eq!(output.shape_nd(), &[2, 2]);
        assert_eq!(output.values(), &[3, 4, 255, 255]);
        assert_eq!(output.coord("y").unwrap().dims(), &["y", "x"]);
        assert_eq!(output.coord("x").unwrap().dims(), &["y", "x"]);
        assert_eq!(output.coord("y").unwrap().values()[0..2], [10.0, 20.0]);
        assert!(output.coord("y").unwrap().values()[2].is_nan());
        assert!(output.coord("y").unwrap().values()[3].is_nan());
        assert_eq!(output.coord("x").unwrap().values()[0..2], [300.0, 100.0]);
        assert!(output.coord("x").unwrap().values()[2].is_nan());
        assert!(output.coord("x").unwrap().values()[3].is_nan());
    }

    #[test]
    fn line_sample_remaps_2d_and_band_spatial_coordinates() {
        let source = DataArray::<u16>::from_vec_named(
            vec![2, 2, 3],
            ["bands", "y", "x"],
            vec![
                10, 11, 12, 13, 14, 15, //
                20, 21, 22, 23, 24, 25,
            ],
        )
        .unwrap()
        .with_coordinate(
            "longitude",
            Coordinate::new(["y", "x"], vec![100.0, 101.0, 102.0, 110.0, 111.0, 112.0]).unwrap(),
        )
        .unwrap()
        .with_coordinate(
            "band_quality",
            Coordinate::new(
                ["bands", "y"],
                vec![
                    0.0, 1.0, //
                    10.0, 11.0,
                ],
            )
            .unwrap(),
        )
        .unwrap();
        let lines = LineSampleGrid::new(1, 2, [0, 1], [2, 0]).unwrap();

        let output = sample_array_from_linesample(&source, &lines, 999, false).unwrap();

        assert_eq!(output.coord("longitude").unwrap().dims(), &["y", "x"]);
        assert_eq!(output.coord("longitude").unwrap().values(), &[102.0, 110.0]);
        assert_eq!(
            output.coord("band_quality").unwrap().dims(),
            &["bands", "y", "x"]
        );
        assert_eq!(
            output.coord("band_quality").unwrap().values(),
            &[0.0, 1.0, 10.0, 11.0]
        );
    }
}
