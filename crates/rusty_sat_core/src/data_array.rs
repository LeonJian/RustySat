//! Rust-native data array foundations.
//!
//! Reference behavior inspected before implementation:
//! - `satpy/doc/source/reading.rst` documents Satpy's xarray `DataArray`
//!   expectations: named dimensions, coordinates, attrs, and common `y`/`x`
//!   dimensions for 2D data.
//! - `deps/trollimage/trollimage/xrimage.py` uses xarray data backed by lazy
//!   dask arrays and standardizes image data around `bands`, `y`, and `x`.
//! - `deps/pyresample/README.md` notes that Pyresample works with numpy,
//!   masked arrays, xarray objects, and dask-backed data.
//!
//! This first foundation step only adds an owned, eager, typed array model.
//! Masks, lazy chunks, coordinates, and nested attrs are separate roadmap
//! items and should not be silently faked here.

use crate::{Result, RustySatError};

/// Numeric element types supported by the first Rusty Sat data-array layer.
pub trait NumericElement:
    Copy + Clone + PartialEq + Send + Sync + std::fmt::Debug + 'static
{
    const DTYPE: DataType;
}

impl NumericElement for f32 {
    const DTYPE: DataType = DataType::F32;
}

impl NumericElement for f64 {
    const DTYPE: DataType = DataType::F64;
}

impl NumericElement for u8 {
    const DTYPE: DataType = DataType::U8;
}

impl NumericElement for u16 {
    const DTYPE: DataType = DataType::U16;
}

impl NumericElement for i16 {
    const DTYPE: DataType = DataType::I16;
}

/// Runtime dtype marker for Satpy-style numeric datasets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DataType {
    F32,
    F64,
    U8,
    U16,
    I16,
}

impl DataType {
    pub fn name(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F64 => "f64",
            Self::U8 => "u8",
            Self::U16 => "u16",
            Self::I16 => "i16",
        }
    }
}

/// Owned n-dimensional numeric data.
#[derive(Debug, Clone, PartialEq)]
pub struct DataArray<T: NumericElement> {
    shape: Vec<usize>,
    values: Vec<T>,
}

/// Backwards-compatible name for the early 2D f64 grid vertical slices.
pub type DataGrid = DataArray<f64>;

impl<T: NumericElement> DataArray<T> {
    pub fn from_vec(shape: impl Into<Vec<usize>>, values: Vec<T>) -> Result<Self> {
        let shape = shape.into();
        validate_shape_and_len(&shape, values.len())?;
        Ok(Self { shape, values })
    }

    pub fn dtype(&self) -> DataType {
        T::DTYPE
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    pub fn shape_nd(&self) -> &[usize] {
        &self.shape
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn values(&self) -> &[T] {
        &self.values
    }

    pub fn get_nd(&self, indexes: &[usize]) -> Option<T> {
        let offset = row_major_offset(&self.shape, indexes)?;
        self.values.get(offset).copied()
    }

    pub fn into_values(self) -> Vec<T> {
        self.values
    }
}

impl DataArray<f64> {
    pub fn new(height: usize, width: usize, values: Vec<f64>) -> Result<Self> {
        Self::from_vec(vec![height, width], values)
    }

    pub fn shape(&self) -> (usize, usize) {
        debug_assert_eq!(self.shape.len(), 2);
        (self.shape[0], self.shape[1])
    }

    pub fn get(&self, y: usize, x: usize) -> Option<f64> {
        self.get_nd(&[y, x])
    }
}

/// Runtime-typed owned numeric data.
#[derive(Debug, Clone, PartialEq)]
pub enum AnyDataArray {
    F32(DataArray<f32>),
    F64(DataArray<f64>),
    U8(DataArray<u8>),
    U16(DataArray<u16>),
    I16(DataArray<i16>),
}

impl AnyDataArray {
    pub fn dtype(&self) -> DataType {
        match self {
            Self::F32(array) => array.dtype(),
            Self::F64(array) => array.dtype(),
            Self::U8(array) => array.dtype(),
            Self::U16(array) => array.dtype(),
            Self::I16(array) => array.dtype(),
        }
    }

    pub fn ndim(&self) -> usize {
        self.shape().len()
    }

    pub fn shape(&self) -> &[usize] {
        match self {
            Self::F32(array) => array.shape_nd(),
            Self::F64(array) => array.shape_nd(),
            Self::U8(array) => array.shape_nd(),
            Self::U16(array) => array.shape_nd(),
            Self::I16(array) => array.shape_nd(),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::F32(array) => array.len(),
            Self::F64(array) => array.len(),
            Self::U8(array) => array.len(),
            Self::U16(array) => array.len(),
            Self::I16(array) => array.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn as_f64(&self) -> Option<&DataArray<f64>> {
        match self {
            Self::F64(array) => Some(array),
            _ => None,
        }
    }

    pub fn shape_2d(&self) -> Result<(usize, usize)> {
        let shape = self.shape();
        if shape.len() != 2 {
            return Err(RustySatError::invalid_input(format!(
                "expected 2D array shape, got {:?}",
                shape
            )));
        }
        Ok((shape[0], shape[1]))
    }

    pub fn values_as_f64(&self) -> Vec<f64> {
        match self {
            Self::F32(array) => array
                .values()
                .iter()
                .map(|value| f64::from(*value))
                .collect(),
            Self::F64(array) => array.values().to_vec(),
            Self::U8(array) => array
                .values()
                .iter()
                .map(|value| f64::from(*value))
                .collect(),
            Self::U16(array) => array
                .values()
                .iter()
                .map(|value| f64::from(*value))
                .collect(),
            Self::I16(array) => array
                .values()
                .iter()
                .map(|value| f64::from(*value))
                .collect(),
        }
    }
}

impl From<DataArray<f32>> for AnyDataArray {
    fn from(value: DataArray<f32>) -> Self {
        Self::F32(value)
    }
}

impl From<DataArray<f64>> for AnyDataArray {
    fn from(value: DataArray<f64>) -> Self {
        Self::F64(value)
    }
}

impl From<DataArray<u8>> for AnyDataArray {
    fn from(value: DataArray<u8>) -> Self {
        Self::U8(value)
    }
}

impl From<DataArray<u16>> for AnyDataArray {
    fn from(value: DataArray<u16>) -> Self {
        Self::U16(value)
    }
}

impl From<DataArray<i16>> for AnyDataArray {
    fn from(value: DataArray<i16>) -> Self {
        Self::I16(value)
    }
}

fn validate_shape_and_len(shape: &[usize], actual_len: usize) -> Result<()> {
    if shape.is_empty() {
        return Err(RustySatError::invalid_input(
            "data array shape must have at least one dimension",
        ));
    }
    if shape.iter().any(|dim| *dim == 0) {
        return Err(RustySatError::invalid_input(
            "data array dimensions must be non-zero",
        ));
    }
    let expected_len = shape
        .iter()
        .try_fold(1usize, |acc, dim| acc.checked_mul(*dim))
        .ok_or_else(|| RustySatError::invalid_input("data array shape is too large"))?;
    if actual_len != expected_len {
        return Err(RustySatError::invalid_input(format!(
            "data array has {actual_len} values but shape {:?} requires {expected_len}",
            shape
        )));
    }
    Ok(())
}

fn row_major_offset(shape: &[usize], indexes: &[usize]) -> Option<usize> {
    if shape.len() != indexes.len() {
        return None;
    }
    let mut offset = 0usize;
    for (dim_len, index) in shape.iter().zip(indexes.iter()) {
        if index >= dim_len {
            return None;
        }
        offset = offset.checked_mul(*dim_len)?.checked_add(*index)?;
    }
    Some(offset)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constructs_generic_numeric_arrays() {
        let array = DataArray::<u16>::from_vec(vec![2, 2], vec![1, 2, 3, 4]).unwrap();

        assert_eq!(array.dtype(), DataType::U16);
        assert_eq!(array.ndim(), 2);
        assert_eq!(array.shape_nd(), &[2, 2]);
        assert_eq!(array.get_nd(&[1, 0]), Some(3));
        assert_eq!(array.get_nd(&[2, 0]), None);
    }

    #[test]
    fn keeps_data_grid_compatible_with_2d_f64_vertical_slices() {
        let grid = DataGrid::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        assert_eq!(grid.dtype(), DataType::F64);
        assert_eq!(grid.shape(), (2, 3));
        assert_eq!(grid.shape_nd(), &[2, 3]);
        assert_eq!(grid.get(1, 2), Some(6.0));
    }

    #[test]
    fn validates_shape_and_length() {
        let err = DataArray::<u8>::from_vec(vec![2, 3], vec![1, 2]).unwrap_err();

        assert!(err.to_string().contains("requires 6"));
        assert!(DataArray::<i16>::from_vec(vec![0, 2], vec![]).is_err());
        assert!(DataArray::<f32>::from_vec(Vec::<usize>::new(), vec![]).is_err());
    }

    #[test]
    fn stores_runtime_typed_arrays() {
        let array =
            AnyDataArray::from(DataArray::<i16>::from_vec(vec![3], vec![-1, 0, 1]).unwrap());

        assert_eq!(array.dtype(), DataType::I16);
        assert_eq!(array.ndim(), 1);
        assert_eq!(array.shape(), &[3]);
        assert_eq!(array.len(), 3);
        assert!(array.as_f64().is_none());
        assert_eq!(array.values_as_f64(), vec![-1.0, 0.0, 1.0]);
    }

    #[test]
    fn reports_2d_shape_for_runtime_typed_arrays() {
        let array = AnyDataArray::from(DataArray::<u8>::from_vec(vec![2, 3], vec![0; 6]).unwrap());
        let vector = AnyDataArray::from(DataArray::<u8>::from_vec(vec![6], vec![0; 6]).unwrap());

        assert_eq!(array.shape_2d().unwrap(), (2, 3));
        assert!(vector.shape_2d().is_err());
    }
}
