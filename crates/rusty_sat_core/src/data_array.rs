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
//! This foundation still stores values eagerly. Masks, chunk metadata, and
//! coordinate axes are explicit. Lazy chunk loading and nested attrs live in
//! their focused modules.

use crate::{Result, RustySatError};
use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Numeric element types supported by the first Rusty Sat data-array layer.
pub trait NumericElement:
    Copy + Clone + PartialEq + Send + Sync + std::fmt::Debug + 'static
{
    const DTYPE: DataType;

    fn to_f64(self) -> f64;
}

impl NumericElement for f32 {
    const DTYPE: DataType = DataType::F32;

    fn to_f64(self) -> f64 {
        f64::from(self)
    }
}

impl NumericElement for f64 {
    const DTYPE: DataType = DataType::F64;

    fn to_f64(self) -> f64 {
        self
    }
}

impl NumericElement for u8 {
    const DTYPE: DataType = DataType::U8;

    fn to_f64(self) -> f64 {
        f64::from(self)
    }
}

impl NumericElement for u16 {
    const DTYPE: DataType = DataType::U16;

    fn to_f64(self) -> f64 {
        f64::from(self)
    }
}

impl NumericElement for i16 {
    const DTYPE: DataType = DataType::I16;

    fn to_f64(self) -> f64 {
        f64::from(self)
    }
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
    dims: Vec<String>,
    coords: BTreeMap<String, Coordinate>,
    values: Vec<T>,
    mask: Option<ValidityMask>,
    chunks: Option<ChunkShape>,
}

/// Backwards-compatible name for the early 2D f64 grid vertical slices.
pub type DataGrid = DataArray<f64>;

/// Numeric coordinate attached to one or more named data dimensions.
#[derive(Debug, Clone, PartialEq)]
pub struct Coordinate {
    dims: Vec<String>,
    values: Vec<f64>,
}

impl Coordinate {
    pub fn new(
        dims: impl IntoIterator<Item = impl Into<String>>,
        values: Vec<f64>,
    ) -> Result<Self> {
        let dims = dims.into_iter().map(Into::into).collect::<Vec<_>>();
        if dims.iter().any(|dim| dim.trim().is_empty()) {
            return Err(RustySatError::invalid_input(
                "coordinate dimension name cannot be empty",
            ));
        }
        if values.is_empty() {
            return Err(RustySatError::invalid_input(
                "coordinate must have at least one value",
            ));
        }
        if dims.is_empty() && values.len() != 1 {
            return Err(RustySatError::invalid_input(
                "scalar coordinate must have exactly one value",
            ));
        }
        Ok(Self { dims, values })
    }

    pub fn axis(dim: impl Into<String>, values: Vec<f64>) -> Result<Self> {
        Self::new([dim], values)
    }

    pub fn scalar(value: f64) -> Self {
        Self {
            dims: Vec::new(),
            values: vec![value],
        }
    }

    pub fn dims(&self) -> &[String] {
        &self.dims
    }

    pub fn values(&self) -> &[f64] {
        &self.values
    }
}

/// Desired chunk size per dimension for future lazy/chunked execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkShape(Vec<usize>);

impl ChunkShape {
    pub fn new(chunks: impl Into<Vec<usize>>) -> Result<Self> {
        let chunks = chunks.into();
        if chunks.is_empty() {
            return Err(RustySatError::invalid_input(
                "chunk shape must have at least one dimension",
            ));
        }
        if chunks.contains(&0) {
            return Err(RustySatError::invalid_input(
                "chunk dimensions must be non-zero",
            ));
        }
        Ok(Self(chunks))
    }

    pub fn as_slice(&self) -> &[usize] {
        &self.0
    }

    pub fn validate_for_shape(&self, shape: &[usize]) -> Result<()> {
        validate_chunks(shape, self)
    }

    pub fn chunk_count_for_shape(&self, shape: &[usize]) -> Result<usize> {
        validate_chunks(shape, self)?;
        Ok(chunk_count(shape, self))
    }
}

/// Packed mask where a set bit means the corresponding data value is invalid.
pub struct ValidityMask {
    len: usize,
    bits: Vec<u8>,
    cached_count: AtomicUsize,
}

const UNCACHED_COUNT: usize = usize::MAX;

impl Clone for ValidityMask {
    fn clone(&self) -> Self {
        Self {
            len: self.len,
            bits: self.bits.clone(),
            cached_count: AtomicUsize::new(self.cached_count.load(Ordering::Relaxed)),
        }
    }
}

impl std::fmt::Debug for ValidityMask {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ValidityMask")
            .field("len", &self.len)
            .field("bits", &self.bits)
            .field("masked_count", &self.masked_count())
            .finish()
    }
}

impl PartialEq for ValidityMask {
    fn eq(&self, other: &Self) -> bool {
        self.len == other.len && self.bits == other.bits
    }
}

impl Eq for ValidityMask {}

impl ValidityMask {
    pub fn all_valid(len: usize) -> Self {
        Self {
            len,
            bits: vec![0; len.div_ceil(8)],
            cached_count: AtomicUsize::new(0),
        }
    }

    pub fn from_masked_flags(flags: impl IntoIterator<Item = bool>) -> Self {
        let flags = flags.into_iter().collect::<Vec<_>>();
        let mut mask = Self::all_valid(flags.len());
        let mut count = 0;
        for (idx, masked) in flags.into_iter().enumerate() {
            if masked {
                mask.set_masked(idx, true);
                count += 1;
            }
        }
        mask.cached_count.store(count, Ordering::Relaxed);
        mask
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn is_masked(&self, index: usize) -> Option<bool> {
        if index >= self.len {
            return None;
        }
        let byte = self.bits[index / 8];
        let bit = index % 8;
        Some(byte & (1 << bit) != 0)
    }

    pub fn set_masked(&mut self, index: usize, masked: bool) {
        assert!(index < self.len, "mask index out of bounds");
        let byte = &mut self.bits[index / 8];
        let bit = index % 8;
        if masked {
            *byte |= 1 << bit;
        } else {
            *byte &= !(1 << bit);
        }
        self.cached_count.store(UNCACHED_COUNT, Ordering::Relaxed);
    }

    pub fn masked_count(&self) -> usize {
        let cached = self.cached_count.load(Ordering::Relaxed);
        if cached != UNCACHED_COUNT {
            return cached;
        }
        let count = (0..self.len)
            .filter(|idx| self.is_masked(*idx).unwrap_or(false))
            .count();
        self.cached_count.store(count, Ordering::Relaxed);
        count
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bits
    }
}

impl<T: NumericElement> DataArray<T> {
    pub fn from_vec(shape: impl Into<Vec<usize>>, values: Vec<T>) -> Result<Self> {
        let shape = shape.into();
        validate_shape_and_len(&shape, values.len())?;
        let dims = default_dim_names(shape.len());
        Ok(Self {
            shape,
            dims,
            coords: BTreeMap::new(),
            values,
            mask: None,
            chunks: None,
        })
    }

    pub fn from_vec_named(
        shape: impl Into<Vec<usize>>,
        dims: impl IntoIterator<Item = impl Into<String>>,
        values: Vec<T>,
    ) -> Result<Self> {
        let shape = shape.into();
        let dims = dims.into_iter().map(Into::into).collect::<Vec<_>>();
        validate_shape_and_len(&shape, values.len())?;
        validate_dims(&shape, &dims)?;
        Ok(Self {
            shape,
            dims,
            coords: BTreeMap::new(),
            values,
            mask: None,
            chunks: None,
        })
    }

    pub fn with_mask(mut self, mask: ValidityMask) -> Result<Self> {
        self.set_mask(mask)?;
        Ok(self)
    }

    pub fn with_chunks(mut self, chunks: ChunkShape) -> Result<Self> {
        self.set_chunks(chunks)?;
        Ok(self)
    }

    pub fn with_coordinate(
        mut self,
        name: impl Into<String>,
        coordinate: Coordinate,
    ) -> Result<Self> {
        self.set_coordinate(name, coordinate)?;
        Ok(self)
    }

    pub fn with_renamed_dims(
        mut self,
        replacements: impl IntoIterator<Item = (impl Into<String>, impl Into<String>)>,
    ) -> Result<Self> {
        let replacements = replacements
            .into_iter()
            .map(|(from, to)| (from.into(), to.into()))
            .collect::<BTreeMap<_, _>>();
        if replacements.is_empty() {
            return Ok(self);
        }
        for from in replacements.keys() {
            if !self.dims.iter().any(|dim| dim == from) {
                return Err(RustySatError::invalid_input(format!(
                    "cannot rename missing dimension '{from}'"
                )));
            }
        }
        let renamed_dims = self
            .dims
            .iter()
            .map(|dim| replacements.get(dim).unwrap_or(dim).clone())
            .collect::<Vec<_>>();
        validate_dims(&self.shape, &renamed_dims)?;
        let mut renamed_coords = BTreeMap::new();
        for (name, coord) in self.coords {
            let renamed_coord = Coordinate::new(
                coord
                    .dims
                    .into_iter()
                    .map(|dim| replacements.get(&dim).unwrap_or(&dim).clone()),
                coord.values,
            )?;
            renamed_coords.insert(name, renamed_coord);
        }
        self.dims = renamed_dims;
        self.coords = renamed_coords;
        Ok(self)
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

    pub fn dims(&self) -> &[String] {
        &self.dims
    }

    pub fn coords(&self) -> &BTreeMap<String, Coordinate> {
        &self.coords
    }

    pub fn coord(&self, name: &str) -> Option<&Coordinate> {
        self.coords.get(name)
    }

    pub fn set_coordinate(
        &mut self,
        name: impl Into<String>,
        coordinate: Coordinate,
    ) -> Result<()> {
        let name = name.into();
        if name.trim().is_empty() {
            return Err(RustySatError::invalid_input(
                "coordinate name cannot be empty",
            ));
        }
        validate_coordinate(&self.shape, &self.dims, &coordinate)?;
        self.coords.insert(name, coordinate);
        Ok(())
    }

    pub fn clear_coordinate(&mut self, name: &str) -> Option<Coordinate> {
        self.coords.remove(name)
    }

    pub fn dim(&self, index: usize) -> Option<&str> {
        self.dims.get(index).map(String::as_str)
    }

    pub fn dim_index(&self, dim: &str) -> Option<usize> {
        self.dims.iter().position(|candidate| candidate == dim)
    }

    pub fn size_of_dim(&self, dim: &str) -> Option<usize> {
        self.dim_index(dim).map(|index| self.shape[index])
    }

    pub fn shape_yx(&self) -> Result<(usize, usize)> {
        let Some(y) = self.size_of_dim("y") else {
            return Err(RustySatError::invalid_input(
                "data array requires a 'y' dimension",
            ));
        };
        let Some(x) = self.size_of_dim("x") else {
            return Err(RustySatError::invalid_input(
                "data array requires an 'x' dimension",
            ));
        };
        Ok((y, x))
    }

    pub fn require_dims_exact(&self, expected: &[&str]) -> Result<()> {
        if self.dims.len() != expected.len()
            || !self
                .dims
                .iter()
                .zip(expected)
                .all(|(left, right)| left == right)
        {
            return Err(RustySatError::invalid_input(format!(
                "data array dimensions {:?} do not match expected {:?}",
                self.dims, expected
            )));
        }
        Ok(())
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

    pub fn mask(&self) -> Option<&ValidityMask> {
        self.mask.as_ref()
    }

    pub fn set_mask(&mut self, mask: ValidityMask) -> Result<()> {
        validate_mask_len(self.values.len(), &mask)?;
        self.mask = Some(mask);
        Ok(())
    }

    pub fn clear_mask(&mut self) {
        self.mask = None;
    }

    pub fn chunks(&self) -> Option<&ChunkShape> {
        self.chunks.as_ref()
    }

    pub fn set_chunks(&mut self, chunks: ChunkShape) -> Result<()> {
        validate_chunks(&self.shape, &chunks)?;
        self.chunks = Some(chunks);
        Ok(())
    }

    pub fn clear_chunks(&mut self) {
        self.chunks = None;
    }

    pub fn chunk_count(&self) -> Option<usize> {
        self.chunks
            .as_ref()
            .map(|chunks| chunk_count(&self.shape, chunks))
    }

    pub fn is_masked(&self, index: usize) -> Option<bool> {
        if index >= self.values.len() {
            return None;
        }
        Some(
            self.mask
                .as_ref()
                .and_then(|mask| mask.is_masked(index))
                .unwrap_or(false),
        )
    }

    pub fn get_nd(&self, indexes: &[usize]) -> Option<T> {
        let offset = row_major_offset(&self.shape, indexes)?;
        self.values.get(offset).copied()
    }

    pub fn slice_yx(&self, y: Range<usize>, x: Range<usize>) -> Result<Self> {
        slice_array_yx(self, y, x)
    }

    pub fn slice_yx_owned(self, y: Range<usize>, x: Range<usize>) -> Result<Self> {
        if is_full_yx_slice(&self, &y, &x)? {
            return Ok(self);
        }
        slice_array_yx(&self, y, x)
    }

    pub fn into_values(self) -> Vec<T> {
        self.values
    }

    pub fn into_parts(self) -> (Vec<T>, BTreeMap<String, Coordinate>, Option<ValidityMask>) {
        let Self {
            values,
            coords,
            mask,
            ..
        } = self;
        (values, coords, mask)
    }
}

impl DataArray<f64> {
    pub fn new(height: usize, width: usize, values: Vec<f64>) -> Result<Self> {
        Self::from_vec_named(vec![height, width], ["y", "x"], values)
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

    pub fn dims(&self) -> &[String] {
        match self {
            Self::F32(array) => array.dims(),
            Self::F64(array) => array.dims(),
            Self::U8(array) => array.dims(),
            Self::U16(array) => array.dims(),
            Self::I16(array) => array.dims(),
        }
    }

    pub fn coords(&self) -> &BTreeMap<String, Coordinate> {
        match self {
            Self::F32(array) => array.coords(),
            Self::F64(array) => array.coords(),
            Self::U8(array) => array.coords(),
            Self::U16(array) => array.coords(),
            Self::I16(array) => array.coords(),
        }
    }

    pub fn coord(&self, name: &str) -> Option<&Coordinate> {
        match self {
            Self::F32(array) => array.coord(name),
            Self::F64(array) => array.coord(name),
            Self::U8(array) => array.coord(name),
            Self::U16(array) => array.coord(name),
            Self::I16(array) => array.coord(name),
        }
    }

    pub fn mask(&self) -> Option<&ValidityMask> {
        match self {
            Self::F32(array) => array.mask(),
            Self::F64(array) => array.mask(),
            Self::U8(array) => array.mask(),
            Self::U16(array) => array.mask(),
            Self::I16(array) => array.mask(),
        }
    }

    pub fn chunks(&self) -> Option<&ChunkShape> {
        match self {
            Self::F32(array) => array.chunks(),
            Self::F64(array) => array.chunks(),
            Self::U8(array) => array.chunks(),
            Self::U16(array) => array.chunks(),
            Self::I16(array) => array.chunks(),
        }
    }

    pub fn chunk_count(&self) -> Option<usize> {
        match self {
            Self::F32(array) => array.chunk_count(),
            Self::F64(array) => array.chunk_count(),
            Self::U8(array) => array.chunk_count(),
            Self::U16(array) => array.chunk_count(),
            Self::I16(array) => array.chunk_count(),
        }
    }

    pub fn dim_index(&self, dim: &str) -> Option<usize> {
        match self {
            Self::F32(array) => array.dim_index(dim),
            Self::F64(array) => array.dim_index(dim),
            Self::U8(array) => array.dim_index(dim),
            Self::U16(array) => array.dim_index(dim),
            Self::I16(array) => array.dim_index(dim),
        }
    }

    pub fn size_of_dim(&self, dim: &str) -> Option<usize> {
        match self {
            Self::F32(array) => array.size_of_dim(dim),
            Self::F64(array) => array.size_of_dim(dim),
            Self::U8(array) => array.size_of_dim(dim),
            Self::U16(array) => array.size_of_dim(dim),
            Self::I16(array) => array.size_of_dim(dim),
        }
    }

    pub fn shape_yx(&self) -> Result<(usize, usize)> {
        match self {
            Self::F32(array) => array.shape_yx(),
            Self::F64(array) => array.shape_yx(),
            Self::U8(array) => array.shape_yx(),
            Self::U16(array) => array.shape_yx(),
            Self::I16(array) => array.shape_yx(),
        }
    }

    pub fn slice_yx(&self, y: Range<usize>, x: Range<usize>) -> Result<Self> {
        Ok(match self {
            Self::F32(array) => array.slice_yx(y, x)?.into(),
            Self::F64(array) => array.slice_yx(y, x)?.into(),
            Self::U8(array) => array.slice_yx(y, x)?.into(),
            Self::U16(array) => array.slice_yx(y, x)?.into(),
            Self::I16(array) => array.slice_yx(y, x)?.into(),
        })
    }

    pub fn slice_yx_owned(self, y: Range<usize>, x: Range<usize>) -> Result<Self> {
        Ok(match self {
            Self::F32(array) => array.slice_yx_owned(y, x)?.into(),
            Self::F64(array) => array.slice_yx_owned(y, x)?.into(),
            Self::U8(array) => array.slice_yx_owned(y, x)?.into(),
            Self::U16(array) => array.slice_yx_owned(y, x)?.into(),
            Self::I16(array) => array.slice_yx_owned(y, x)?.into(),
        })
    }

    pub fn require_dims_exact(&self, expected: &[&str]) -> Result<()> {
        match self {
            Self::F32(array) => array.require_dims_exact(expected),
            Self::F64(array) => array.require_dims_exact(expected),
            Self::U8(array) => array.require_dims_exact(expected),
            Self::U16(array) => array.require_dims_exact(expected),
            Self::I16(array) => array.require_dims_exact(expected),
        }
    }

    pub fn with_renamed_dims(
        self,
        replacements: impl IntoIterator<Item = (impl Into<String>, impl Into<String>)>,
    ) -> Result<Self> {
        let replacements = replacements
            .into_iter()
            .map(|(from, to)| (from.into(), to.into()))
            .collect::<BTreeMap<_, _>>();
        Ok(match self {
            Self::F32(array) => array.with_renamed_dims(replacements.clone())?.into(),
            Self::F64(array) => array.with_renamed_dims(replacements.clone())?.into(),
            Self::U8(array) => array.with_renamed_dims(replacements.clone())?.into(),
            Self::U16(array) => array.with_renamed_dims(replacements.clone())?.into(),
            Self::I16(array) => array.with_renamed_dims(replacements)?.into(),
        })
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

    pub fn into_f64(self) -> Option<DataArray<f64>> {
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
            Self::F32(array) => array.values().iter().map(|value| value.to_f64()).collect(),
            Self::F64(array) => array.values().to_vec(),
            Self::U8(array) => array.values().iter().map(|value| value.to_f64()).collect(),
            Self::U16(array) => array.values().iter().map(|value| value.to_f64()).collect(),
            Self::I16(array) => array.values().iter().map(|value| value.to_f64()).collect(),
        }
    }

    pub fn into_f64_values(self) -> Vec<f64> {
        match self {
            Self::F32(array) => numeric_values_into_f64(array.into_values()),
            Self::F64(array) => array.into_values(),
            Self::U8(array) => numeric_values_into_f64(array.into_values()),
            Self::U16(array) => numeric_values_into_f64(array.into_values()),
            Self::I16(array) => numeric_values_into_f64(array.into_values()),
        }
    }

    pub fn into_mask(self) -> Option<ValidityMask> {
        match self {
            Self::F32(array) => array.into_parts().2,
            Self::F64(array) => array.into_parts().2,
            Self::U8(array) => array.into_parts().2,
            Self::U16(array) => array.into_parts().2,
            Self::I16(array) => array.into_parts().2,
        }
    }

    pub fn into_f64_values_and_mask(self) -> (Vec<f64>, Option<ValidityMask>) {
        match self {
            Self::F32(array) => into_numeric_f64_values_and_mask(array),
            Self::F64(array) => {
                let (values, _, mask) = array.into_parts();
                (values, mask)
            }
            Self::U8(array) => into_numeric_f64_values_and_mask(array),
            Self::U16(array) => into_numeric_f64_values_and_mask(array),
            Self::I16(array) => into_numeric_f64_values_and_mask(array),
        }
    }
}

fn numeric_values_into_f64<T: NumericElement>(values: Vec<T>) -> Vec<f64> {
    values.into_iter().map(NumericElement::to_f64).collect()
}

fn into_numeric_f64_values_and_mask<T: NumericElement>(
    array: DataArray<T>,
) -> (Vec<f64>, Option<ValidityMask>) {
    let (values, _, mask) = array.into_parts();
    (numeric_values_into_f64(values), mask)
}

fn slice_array_yx<T: NumericElement>(
    array: &DataArray<T>,
    y: Range<usize>,
    x: Range<usize>,
) -> Result<DataArray<T>> {
    validate_range(&y, array.size_of_dim("y"), "y")?;
    validate_range(&x, array.size_of_dim("x"), "x")?;

    let ranges = ranges_for_yx(array, y, x)?;
    let output_shape = ranges
        .iter()
        .map(|range| range.end - range.start)
        .collect::<Vec<_>>();
    let output_len = output_shape.iter().product();
    let mut values = Vec::with_capacity(output_len);
    let mut mask_flags = array.mask.as_ref().map(|_| Vec::with_capacity(output_len));

    for output_offset in 0..output_len {
        let output_indexes = unravel_offset(&output_shape, output_offset)?;
        let source_indexes = output_indexes
            .iter()
            .zip(&ranges)
            .map(|(idx, range)| range.start + idx)
            .collect::<Vec<_>>();
        let source_offset = row_major_offset(&array.shape, &source_indexes).ok_or_else(|| {
            RustySatError::invalid_input("computed source slice index outside array shape")
        })?;
        values.push(array.values[source_offset]);
        if let Some(flags) = mask_flags.as_mut() {
            flags.push(
                array
                    .mask
                    .as_ref()
                    .and_then(|mask| mask.is_masked(source_offset))
                    .unwrap_or(false),
            );
        }
    }

    let mut sliced = DataArray::from_vec_named(output_shape, array.dims.clone(), values)?;
    for (name, coord) in &array.coords {
        sliced.set_coordinate(
            name.clone(),
            slice_coordinate(coord, &array.shape, &array.dims, &ranges)?,
        )?;
    }
    if let Some(flags) = mask_flags {
        sliced.set_mask(ValidityMask::from_masked_flags(flags))?;
    }
    Ok(sliced)
}

fn is_full_yx_slice<T: NumericElement>(
    array: &DataArray<T>,
    y: &Range<usize>,
    x: &Range<usize>,
) -> Result<bool> {
    validate_range(y, array.size_of_dim("y"), "y")?;
    validate_range(x, array.size_of_dim("x"), "x")?;
    let y_dim = array.size_of_dim("y").unwrap();
    let x_dim = array.size_of_dim("x").unwrap();
    Ok(y.start == 0 && y.end == y_dim && x.start == 0 && x.end == x_dim)
}

fn ranges_for_yx<T: NumericElement>(
    array: &DataArray<T>,
    y: Range<usize>,
    x: Range<usize>,
) -> Result<Vec<Range<usize>>> {
    let y_dim = array.dim_index("y").ok_or_else(|| {
        RustySatError::invalid_input("data array requires a 'y' dimension for slicing")
    })?;
    let x_dim = array.dim_index("x").ok_or_else(|| {
        RustySatError::invalid_input("data array requires an 'x' dimension for slicing")
    })?;
    let mut ranges = array.shape.iter().map(|dim| 0..*dim).collect::<Vec<_>>();
    ranges[y_dim] = y;
    ranges[x_dim] = x;
    Ok(ranges)
}

fn validate_range(range: &Range<usize>, dim_len: Option<usize>, dim_name: &str) -> Result<()> {
    let dim_len = dim_len.ok_or_else(|| {
        RustySatError::invalid_input(format!("data array requires a '{dim_name}' dimension"))
    })?;
    if range.start >= range.end || range.end > dim_len {
        return Err(RustySatError::invalid_input(format!(
            "{dim_name} slice {:?} is outside dimension length {dim_len}",
            range
        )));
    }
    Ok(())
}

fn slice_coordinate(
    coord: &Coordinate,
    array_shape: &[usize],
    array_dims: &[String],
    array_ranges: &[Range<usize>],
) -> Result<Coordinate> {
    if coord.dims().is_empty() {
        return Ok(coord.clone());
    }
    if !coord.dims().iter().any(|dim| dim == "y" || dim == "x") {
        return Ok(coord.clone());
    }

    let mut coord_shape = Vec::with_capacity(coord.dims().len());
    let mut coord_ranges = Vec::with_capacity(coord.dims().len());
    for dim in coord.dims() {
        let dim_index = array_dims
            .iter()
            .position(|candidate| candidate == dim)
            .ok_or_else(|| {
                RustySatError::invalid_input(format!(
                    "coordinate dimension '{dim}' is not a data dimension"
                ))
            })?;
        coord_shape.push(array_shape[dim_index]);
        coord_ranges.push(array_ranges[dim_index].clone());
    }

    let output_shape = coord_ranges
        .iter()
        .map(|range| range.end - range.start)
        .collect::<Vec<_>>();
    let output_len = output_shape.iter().product();
    let mut values = Vec::with_capacity(output_len);
    for output_offset in 0..output_len {
        let output_indexes = unravel_offset(&output_shape, output_offset)?;
        let source_indexes = output_indexes
            .iter()
            .zip(&coord_ranges)
            .map(|(idx, range)| range.start + idx)
            .collect::<Vec<_>>();
        let source_offset = row_major_offset(&coord_shape, &source_indexes).ok_or_else(|| {
            RustySatError::invalid_input("computed source coordinate slice outside shape")
        })?;
        values.push(coord.values()[source_offset]);
    }
    Coordinate::new(coord.dims().to_vec(), values)
}

fn unravel_offset(shape: &[usize], mut offset: usize) -> Result<Vec<usize>> {
    if shape.is_empty() {
        return Err(RustySatError::invalid_input(
            "cannot unravel offset for empty shape",
        ));
    }
    let mut indexes = vec![0; shape.len()];
    for dim in (0..shape.len()).rev() {
        let dim_len = shape[dim];
        if dim_len == 0 {
            return Err(RustySatError::invalid_input(
                "cannot unravel offset for zero-sized dimension",
            ));
        }
        indexes[dim] = offset % dim_len;
        offset /= dim_len;
    }
    Ok(indexes)
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
    if shape.contains(&0) {
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

fn validate_mask_len(values_len: usize, mask: &ValidityMask) -> Result<()> {
    if values_len != mask.len() {
        return Err(RustySatError::invalid_input(format!(
            "data array mask has {} values but data has {}",
            mask.len(),
            values_len
        )));
    }
    Ok(())
}

fn validate_chunks(shape: &[usize], chunks: &ChunkShape) -> Result<()> {
    if chunks.as_slice().len() != shape.len() {
        return Err(RustySatError::invalid_input(format!(
            "chunk shape has {} dimensions but data has {}",
            chunks.as_slice().len(),
            shape.len()
        )));
    }
    for (dim, chunk) in shape.iter().zip(chunks.as_slice()) {
        if chunk > dim {
            return Err(RustySatError::invalid_input(format!(
                "chunk dimension {chunk} cannot exceed data dimension {dim}"
            )));
        }
    }
    Ok(())
}

fn validate_coordinate(shape: &[usize], dims: &[String], coordinate: &Coordinate) -> Result<()> {
    let mut expected_len = 1usize;
    for dim in coordinate.dims() {
        let Some(dim_index) = dims.iter().position(|candidate| candidate == dim) else {
            return Err(RustySatError::invalid_input(format!(
                "coordinate dimension '{dim}' is not a data dimension"
            )));
        };
        expected_len = expected_len
            .checked_mul(shape[dim_index])
            .ok_or_else(|| RustySatError::invalid_input("coordinate shape is too large"))?;
    }
    if coordinate.values().len() != expected_len {
        return Err(RustySatError::invalid_input(format!(
            "coordinate has {} values but dimensions {:?} require {}",
            coordinate.values().len(),
            coordinate.dims(),
            expected_len
        )));
    }
    Ok(())
}

fn chunk_count(shape: &[usize], chunks: &ChunkShape) -> usize {
    shape
        .iter()
        .zip(chunks.as_slice())
        .map(|(dim, chunk)| dim.div_ceil(*chunk))
        .product()
}

fn validate_dims(shape: &[usize], dims: &[String]) -> Result<()> {
    if dims.len() != shape.len() {
        return Err(RustySatError::invalid_input(format!(
            "data array has {} dimensions but {} dimension names",
            shape.len(),
            dims.len()
        )));
    }
    let mut seen = BTreeSet::new();
    for dim in dims {
        if dim.trim().is_empty() {
            return Err(RustySatError::invalid_input(
                "data array dimension name cannot be empty",
            ));
        }
        if !seen.insert(dim) {
            return Err(RustySatError::invalid_input(format!(
                "duplicate data array dimension name '{dim}'"
            )));
        }
    }
    Ok(())
}

fn default_dim_names(ndim: usize) -> Vec<String> {
    match ndim {
        1 => vec!["y".to_string()],
        2 => vec!["y".to_string(), "x".to_string()],
        3 => vec!["bands".to_string(), "y".to_string(), "x".to_string()],
        4 => vec![
            "time".to_string(),
            "bands".to_string(),
            "y".to_string(),
            "x".to_string(),
        ],
        _ => (0..ndim).map(|idx| format!("dim_{idx}")).collect(),
    }
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
        assert_eq!(array.dims(), &["y".to_string(), "x".to_string()]);
        assert_eq!(array.get_nd(&[1, 0]), Some(3));
        assert_eq!(array.get_nd(&[2, 0]), None);
        assert_eq!(array.mask(), None);
    }

    #[test]
    fn constructs_arrays_with_named_dimensions() {
        let array =
            DataArray::<f32>::from_vec_named(vec![3, 2, 2], ["bands", "y", "x"], vec![0.0; 12])
                .unwrap();

        assert_eq!(array.ndim(), 3);
        assert_eq!(array.shape_nd(), &[3, 2, 2]);
        assert_eq!(
            array.dims(),
            &["bands".to_string(), "y".to_string(), "x".to_string()]
        );
        assert_eq!(array.dim(1), Some("y"));
        assert_eq!(array.dim_index("x"), Some(2));
        assert_eq!(array.size_of_dim("bands"), Some(3));
        assert_eq!(array.shape_yx().unwrap(), (2, 2));
        array.require_dims_exact(&["bands", "y", "x"]).unwrap();
        assert!(array.require_dims_exact(&["y", "x"]).is_err());
    }

    #[test]
    fn stores_named_coordinate_axes() {
        let array = DataArray::<u8>::from_vec(vec![2, 3], vec![0; 6])
            .unwrap()
            .with_coordinate("y", Coordinate::axis("y", vec![10.0, 20.0]).unwrap())
            .unwrap()
            .with_coordinate("x", Coordinate::axis("x", vec![1.0, 2.0, 3.0]).unwrap())
            .unwrap();

        assert_eq!(array.coords().len(), 2);
        assert_eq!(array.coord("y").unwrap().dims(), &["y".to_string()]);
        assert_eq!(array.coord("y").unwrap().values(), &[10.0, 20.0]);
        assert_eq!(array.coord("x").unwrap().values(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn renames_dimensions_without_copying_values_or_losing_mask() {
        let array = DataArray::<u16>::from_vec_named(
            vec![2, 3],
            ["Rows", "Columns"],
            vec![1, 2, 3, 4, 5, 6],
        )
        .unwrap()
        .with_mask(ValidityMask::from_masked_flags([
            false, true, false, false, false, true,
        ]))
        .unwrap()
        .with_coordinate(
            "row_index",
            Coordinate::axis("Rows", vec![10.0, 20.0]).unwrap(),
        )
        .unwrap()
        .with_coordinate(
            "scan",
            Coordinate::new(["Rows", "Columns"], vec![0.0; 6]).unwrap(),
        )
        .unwrap();

        let renamed = array
            .with_renamed_dims([("Rows", "y"), ("Columns", "x")])
            .unwrap();

        assert_eq!(renamed.dims(), &["y".to_string(), "x".to_string()]);
        assert_eq!(renamed.values(), &[1, 2, 3, 4, 5, 6]);
        assert_eq!(renamed.mask().unwrap().masked_count(), 2);
        assert_eq!(
            renamed.coord("row_index").unwrap().dims(),
            &["y".to_string()]
        );
        assert_eq!(
            renamed.coord("scan").unwrap().dims(),
            &["y".to_string(), "x".to_string()]
        );
    }

    #[test]
    fn runtime_typed_array_renames_dimensions_and_rejects_conflicts() {
        let array = AnyDataArray::from(
            DataArray::<u8>::from_vec_named(vec![2, 3], ["Rows", "Columns"], vec![0; 6]).unwrap(),
        );

        let renamed = array
            .with_renamed_dims([("Rows", "y"), ("Columns", "x")])
            .unwrap();

        assert_eq!(renamed.dtype(), DataType::U8);
        assert_eq!(renamed.dims(), &["y".to_string(), "x".to_string()]);
        assert!(renamed
            .with_renamed_dims([("missing", "y")])
            .unwrap_err()
            .to_string()
            .contains("missing dimension"));
    }

    #[test]
    fn stores_multidimensional_coordinates() {
        let array = DataArray::<u8>::from_vec(vec![2, 3], vec![0; 6])
            .unwrap()
            .with_coordinate(
                "longitude",
                Coordinate::new(["y", "x"], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
            )
            .unwrap();

        assert_eq!(
            array.coord("longitude").unwrap().dims(),
            &["y".to_string(), "x".to_string()]
        );
    }

    #[test]
    fn stores_scalar_coordinates() {
        let array = DataArray::<u8>::from_vec(vec![2, 3], vec![0; 6])
            .unwrap()
            .with_coordinate("acq_time", Coordinate::scalar(123.0))
            .unwrap();

        assert!(array.coord("acq_time").unwrap().dims().is_empty());
        assert_eq!(array.coord("acq_time").unwrap().values(), &[123.0]);
        assert!(Coordinate::new(Vec::<String>::new(), vec![1.0, 2.0]).is_err());
    }

    #[test]
    fn validates_coordinate_dimensions_and_length() {
        let mut array = DataArray::<u8>::from_vec(vec![2, 3], vec![0; 6]).unwrap();

        assert!(array
            .set_coordinate("bad", Coordinate::axis("row", vec![1.0, 2.0]).unwrap())
            .is_err());
        assert!(array
            .set_coordinate("x", Coordinate::axis("x", vec![1.0, 2.0]).unwrap())
            .is_err());
    }

    #[test]
    fn keeps_data_grid_compatible_with_2d_f64_vertical_slices() {
        let grid = DataGrid::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        assert_eq!(grid.dtype(), DataType::F64);
        assert_eq!(grid.shape(), (2, 3));
        assert_eq!(grid.shape_nd(), &[2, 3]);
        assert_eq!(grid.dims(), &["y".to_string(), "x".to_string()]);
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
    fn validates_dimension_names() {
        assert!(DataArray::<u8>::from_vec_named(vec![2, 2], ["y"], vec![0; 4]).is_err());
        assert!(DataArray::<u8>::from_vec_named(vec![2, 2], ["y", "y"], vec![0; 4]).is_err());
        assert!(DataArray::<u8>::from_vec_named(vec![2, 2], ["y", ""], vec![0; 4]).is_err());
    }

    #[test]
    fn stores_independent_validity_mask() {
        let mask = ValidityMask::from_masked_flags([false, true, false, true]);
        let array = DataArray::<u8>::from_vec(vec![2, 2], vec![1, 2, 3, 4])
            .unwrap()
            .with_mask(mask)
            .unwrap();

        assert_eq!(array.mask().unwrap().len(), 4);
        assert_eq!(array.mask().unwrap().masked_count(), 2);
        assert_eq!(array.is_masked(0), Some(false));
        assert_eq!(array.is_masked(1), Some(true));
        assert_eq!(array.is_masked(4), None);
        assert_eq!(array.mask().unwrap().bytes().len(), 1);
    }

    #[test]
    fn validates_mask_length() {
        let mask = ValidityMask::from_masked_flags([true, false]);
        let err = DataArray::<u8>::from_vec(vec![2, 2], vec![1, 2, 3, 4])
            .unwrap()
            .with_mask(mask)
            .unwrap_err();

        assert!(err.to_string().contains("mask has 2 values"));
    }

    #[test]
    fn stores_chunk_shape_metadata() {
        let array = DataArray::<u8>::from_vec(vec![5, 6], vec![0; 30])
            .unwrap()
            .with_chunks(ChunkShape::new(vec![2, 3]).unwrap())
            .unwrap();

        assert_eq!(array.chunks().unwrap().as_slice(), &[2, 3]);
        assert_eq!(array.chunk_count(), Some(6));
    }

    #[test]
    fn validates_chunk_shape_metadata() {
        assert!(ChunkShape::new(Vec::<usize>::new()).is_err());
        assert!(ChunkShape::new(vec![2, 0]).is_err());

        let err = DataArray::<u8>::from_vec(vec![5, 6], vec![0; 30])
            .unwrap()
            .with_chunks(ChunkShape::new(vec![2]).unwrap())
            .unwrap_err();
        assert!(err.to_string().contains("chunk shape has 1 dimensions"));

        let err = DataArray::<u8>::from_vec(vec![5, 6], vec![0; 30])
            .unwrap()
            .with_chunks(ChunkShape::new(vec![6, 2]).unwrap())
            .unwrap_err();
        assert!(err.to_string().contains("cannot exceed"));
    }

    #[test]
    fn stores_runtime_typed_arrays() {
        let array =
            AnyDataArray::from(DataArray::<i16>::from_vec(vec![3], vec![-1, 0, 1]).unwrap());

        assert_eq!(array.dtype(), DataType::I16);
        assert_eq!(array.ndim(), 1);
        assert_eq!(array.shape(), &[3]);
        assert_eq!(array.dims(), &["y".to_string()]);
        assert_eq!(array.len(), 3);
        assert!(array.as_f64().is_none());
        assert_eq!(array.values_as_f64(), vec![-1.0, 0.0, 1.0]);
    }

    #[test]
    fn slices_yx_dimensions_and_preserves_dtype_mask_and_coords() {
        let array = DataArray::<u16>::from_vec_named(
            vec![2, 3, 4],
            ["bands", "y", "x"],
            (0..24).collect::<Vec<_>>(),
        )
        .unwrap()
        .with_mask(ValidityMask::from_masked_flags(
            (0..24).map(|idx| idx == 6 || idx == 21),
        ))
        .unwrap()
        .with_coordinate("y", Coordinate::axis("y", vec![10.0, 20.0, 30.0]).unwrap())
        .unwrap()
        .with_coordinate(
            "longitude",
            Coordinate::new(
                ["y", "x"],
                vec![
                    0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0, 20.0, 21.0, 22.0, 23.0,
                ],
            )
            .unwrap(),
        )
        .unwrap()
        .with_coordinate("acq_time", Coordinate::scalar(42.0))
        .unwrap();

        let sliced = array.slice_yx(1..3, 1..4).unwrap();

        assert_eq!(sliced.dtype(), DataType::U16);
        assert_eq!(sliced.shape_nd(), &[2, 2, 3]);
        assert_eq!(sliced.dims(), &["bands", "y", "x"]);
        assert_eq!(
            sliced.values(),
            &[5, 6, 7, 9, 10, 11, 17, 18, 19, 21, 22, 23]
        );
        assert_eq!(sliced.mask().unwrap().masked_count(), 2);
        assert_eq!(sliced.is_masked(1), Some(true));
        assert_eq!(sliced.is_masked(9), Some(true));
        assert_eq!(sliced.coord("y").unwrap().values(), &[20.0, 30.0]);
        assert_eq!(
            sliced.coord("longitude").unwrap().values(),
            &[11.0, 12.0, 13.0, 21.0, 22.0, 23.0]
        );
        assert_eq!(sliced.coord("acq_time").unwrap().values(), &[42.0]);
        assert!(sliced.chunks().is_none());
    }

    #[test]
    fn runtime_typed_owned_slice_preserves_dtype_and_full_slice_can_keep_chunks() {
        let array = DataArray::<u8>::from_vec(vec![2, 3], vec![1, 2, 3, 4, 5, 6])
            .unwrap()
            .with_chunks(ChunkShape::new(vec![1, 3]).unwrap())
            .unwrap();

        let full = AnyDataArray::from(array.clone())
            .slice_yx_owned(0..2, 0..3)
            .unwrap();
        let cropped = AnyDataArray::from(array)
            .slice_yx_owned(1..2, 1..3)
            .unwrap();

        assert_eq!(full.dtype(), DataType::U8);
        assert_eq!(full.chunks().unwrap().as_slice(), &[1, 3]);
        assert_eq!(cropped.dtype(), DataType::U8);
        assert_eq!(cropped.shape(), &[1, 2]);
        assert_eq!(cropped.values_as_f64(), vec![5.0, 6.0]);
        assert!(cropped.chunks().is_none());
    }

    #[test]
    fn yx_slice_rejects_missing_or_out_of_bounds_dimensions() {
        let vector = DataArray::<u8>::from_vec_named(vec![3], ["x"], vec![1, 2, 3]).unwrap();
        let array = DataArray::<u8>::from_vec(vec![2, 2], vec![1, 2, 3, 4]).unwrap();

        assert!(vector.slice_yx(0..1, 0..1).is_err());
        assert!(array.slice_yx(0..3, 0..1).is_err());
        assert!(array.slice_yx(1..1, 0..1).is_err());
    }

    #[test]
    fn exposes_coordinates_from_runtime_typed_arrays() {
        let array = AnyDataArray::from(
            DataArray::<u8>::from_vec(vec![2], vec![1, 2])
                .unwrap()
                .with_coordinate("y", Coordinate::axis("y", vec![0.5, 1.5]).unwrap())
                .unwrap(),
        );

        assert_eq!(array.coord("y").unwrap().values(), &[0.5, 1.5]);
    }

    #[test]
    fn reports_2d_shape_for_runtime_typed_arrays() {
        let array = AnyDataArray::from(DataArray::<u8>::from_vec(vec![2, 3], vec![0; 6]).unwrap());
        let vector = AnyDataArray::from(DataArray::<u8>::from_vec(vec![6], vec![0; 6]).unwrap());

        assert_eq!(array.shape_2d().unwrap(), (2, 3));
        assert!(vector.shape_2d().is_err());
    }

    #[test]
    fn into_parts_destructures_array() {
        let array = DataArray::<u8>::from_vec(vec![2, 2], vec![1, 2, 3, 4])
            .unwrap()
            .with_coordinate("acq_time", Coordinate::scalar(123.0))
            .unwrap()
            .with_mask(ValidityMask::from_masked_flags([false, true, false, false]))
            .unwrap();

        let (values, coords, mask) = array.into_parts();

        assert_eq!(values, vec![1, 2, 3, 4]);
        assert_eq!(coords.len(), 1);
        assert_eq!(coords.get("acq_time").unwrap().values(), &[123.0]);
        assert_eq!(mask.unwrap().masked_count(), 1);
    }

    #[test]
    fn runtime_typed_array_consumes_into_f64_values_without_copying_f64() {
        let f64_array =
            AnyDataArray::from(DataArray::<f64>::from_vec(vec![2], vec![1.5, 2.5]).unwrap());
        let u16_array =
            AnyDataArray::from(DataArray::<u16>::from_vec(vec![2], vec![3, 4]).unwrap());

        assert_eq!(f64_array.into_f64_values(), vec![1.5, 2.5]);
        assert_eq!(u16_array.into_f64_values(), vec![3.0, 4.0]);
    }

    #[test]
    fn runtime_typed_array_into_f64_consumes_full_grid() {
        let grid = DataGrid::new(1, 2, vec![10.0, 20.0])
            .unwrap()
            .with_mask(ValidityMask::from_masked_flags([false, true]))
            .unwrap()
            .with_coordinate("acq_time", Coordinate::scalar(123.0))
            .unwrap();
        let array = AnyDataArray::from(grid);
        let consumed: DataGrid = array.into_f64().unwrap();

        assert_eq!(consumed.shape(), (1, 2));
        assert_eq!(consumed.values(), &[10.0, 20.0]);
        assert_eq!(consumed.mask().unwrap().masked_count(), 1);
        assert_eq!(consumed.coord("acq_time").unwrap().values(), &[123.0]);
    }

    #[test]
    fn runtime_typed_array_into_f64_rejects_non_f64() {
        let array = AnyDataArray::from(DataArray::<u16>::from_vec(vec![2], vec![1, 2]).unwrap());
        assert!(array.into_f64().is_none());
    }

    #[test]
    fn runtime_typed_array_consumes_mask() {
        let array = AnyDataArray::from(
            DataArray::<u8>::from_vec(vec![3], vec![1, 2, 3])
                .unwrap()
                .with_mask(ValidityMask::from_masked_flags([false, true, false]))
                .unwrap(),
        );

        assert_eq!(array.into_mask().unwrap().masked_count(), 1);
    }

    #[test]
    fn runtime_typed_array_consumes_values_and_mask_together() {
        let array = AnyDataArray::from(
            DataArray::<i16>::from_vec(vec![2], vec![-1, 2])
                .unwrap()
                .with_mask(ValidityMask::from_masked_flags([true, false]))
                .unwrap(),
        );

        let (values, mask) = array.into_f64_values_and_mask();

        assert_eq!(values, vec![-1.0, 2.0]);
        assert_eq!(mask.unwrap().is_masked(0), Some(true));
    }
}
