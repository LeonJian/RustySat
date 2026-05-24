//! Line/sample indexing helpers.
//!
//! Reference behavior inspected before implementation:
//! - `deps/pyresample/pyresample/grid.py::get_image_from_linesample`
//! - `deps/pyresample/pyresample/image.py::ImageContainer.get_array_from_linesample`
//!
//! This is the first Rust-native foundation for Pyresample's line/sample path.
//! It deliberately starts with 2D `f64` grids because current Rusty Sat area
//! resamplers still use `DataGrid` internally for the common sampling paths.

use rusty_sat_core::{DataGrid, Result, RustySatError, ValidityMask};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LineSampleGrid {
    height: usize,
    width: usize,
    rows: Vec<isize>,
    cols: Vec<isize>,
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
    let (source_height, source_width) = source.shape();
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

    let mut grid = DataGrid::new(linesample.height(), linesample.width(), output)?;
    if let Some(flags) = mask_flags {
        grid.set_mask(ValidityMask::from_masked_flags(flags))?;
    }
    Ok(grid)
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
