//! Shared N-dimensional stride/index helpers.

use rusty_sat_core::{Result, RustySatError};

pub(crate) fn checked_shape_size(shape: &[usize]) -> Result<usize> {
    shape.iter().try_fold(1usize, |acc, dim| {
        if *dim == 0 {
            return Err(RustySatError::invalid_input(
                "shape dimensions must be non-zero",
            ));
        }
        acc.checked_mul(*dim)
            .ok_or_else(|| RustySatError::invalid_input("shape size overflows usize"))
    })
}

pub(crate) fn row_major_strides(shape: &[usize]) -> Result<Vec<usize>> {
    let mut strides = vec![1; shape.len()];
    let mut stride = 1usize;
    for (idx, dim) in shape.iter().enumerate().rev() {
        strides[idx] = stride;
        stride = stride
            .checked_mul(*dim)
            .ok_or_else(|| RustySatError::invalid_input("stride size overflows usize"))?;
    }
    Ok(strides)
}
