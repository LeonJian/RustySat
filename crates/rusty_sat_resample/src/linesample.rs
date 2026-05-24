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
    AnyDataArray, DataArray, DataGrid, NumericElement, Result, RustySatError, ValidityMask,
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
    let (source_height, source_width) = source.shape_yx()?;
    let source_values = source.values();
    let source_mask = source.mask();
    let mut output = Vec::with_capacity(linesample.len());
    let mut mask_flags = mask_missing.then(|| Vec::with_capacity(linesample.len()));

    for (&row, &col) in linesample.rows().iter().zip(linesample.cols()) {
        let source_index = valid_source_index(row, col, source_height, source_width);
        let is_missing = source_index
            .map(|index| {
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
            let index = source_index.expect("non-missing line/sample must have a source index");
            output.push(source_values[index]);
            if let Some(flags) = mask_flags.as_mut() {
                flags.push(false);
            }
        }
    }

    let mut array = DataArray::from_vec_named(
        vec![linesample.height(), linesample.width()],
        ["y", "x"],
        output,
    )?;
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
}
