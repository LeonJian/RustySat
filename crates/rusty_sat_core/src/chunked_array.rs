//! Lazy chunked array foundations.
//!
//! Reference behavior inspected before implementation:
//! - `deps/trollimage/trollimage/xrimage.py` keeps image data as xarray objects
//!   backed by dask arrays, preserving chunking for lazy image operations.
//! - `deps/pyresample/pyresample/kd_tree.py` builds delayed KD-tree and
//!   resampling work around dask chunks and tries to preserve chunk sizes.
//! - `satpy/satpy/scene.py` exposes chunk, persist, compute, and save paths
//!   that can return delayed work instead of immediately materializing data.
//!
//! This module only defines Rusty Sat's deferred loading contract. It does not
//! implement file-backed readers, scheduling, caching, or parallel execution.

use crate::{ChunkShape, DataArray, DataType, NumericElement, Result, RustySatError};
use std::collections::BTreeSet;
use std::fmt;
use std::sync::Arc;

/// N-dimensional region requested from a lazy chunk source.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkRegion {
    origin: Vec<usize>,
    shape: Vec<usize>,
}

impl ChunkRegion {
    pub fn new(
        full_shape: &[usize],
        origin: impl Into<Vec<usize>>,
        shape: impl Into<Vec<usize>>,
    ) -> Result<Self> {
        let origin = origin.into();
        let shape = shape.into();
        validate_region(full_shape, &origin, &shape)?;
        Ok(Self { origin, shape })
    }

    pub fn origin(&self) -> &[usize] {
        &self.origin
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
}

/// Deferred source of chunk data.
pub trait ChunkSource<T: NumericElement>: Send + Sync {
    fn read_chunk(&self, region: &ChunkRegion) -> Result<DataArray<T>>;
}

/// Metadata and source handle for a deferred n-dimensional array.
#[derive(Clone)]
pub struct LazyDataArray<T: NumericElement> {
    shape: Vec<usize>,
    dims: Vec<String>,
    chunks: ChunkShape,
    source: Arc<dyn ChunkSource<T>>,
}

impl<T: NumericElement> fmt::Debug for LazyDataArray<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LazyDataArray")
            .field("dtype", &T::DTYPE)
            .field("shape", &self.shape)
            .field("dims", &self.dims)
            .field("chunks", &self.chunks)
            .finish_non_exhaustive()
    }
}

impl<T: NumericElement> LazyDataArray<T> {
    pub fn new(
        shape: impl Into<Vec<usize>>,
        dims: impl IntoIterator<Item = impl Into<String>>,
        chunks: ChunkShape,
        source: Arc<dyn ChunkSource<T>>,
    ) -> Result<Self> {
        let shape = shape.into();
        let dims = dims.into_iter().map(Into::into).collect::<Vec<_>>();
        validate_shape(&shape)?;
        validate_dims(&shape, &dims)?;
        chunks.validate_for_shape(&shape)?;
        Ok(Self {
            shape,
            dims,
            chunks,
            source,
        })
    }

    pub fn from_shape(
        shape: impl Into<Vec<usize>>,
        chunks: ChunkShape,
        source: Arc<dyn ChunkSource<T>>,
    ) -> Result<Self> {
        let shape = shape.into();
        let dims = default_dim_names(shape.len());
        Self::new(shape, dims, chunks, source)
    }

    pub fn dtype(&self) -> DataType {
        T::DTYPE
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn dims(&self) -> &[String] {
        &self.dims
    }

    pub fn chunks(&self) -> &ChunkShape {
        &self.chunks
    }

    pub fn shape_yx(&self) -> Result<(usize, usize)> {
        let Some(y_index) = self.dims.iter().position(|dim| dim == "y") else {
            return Err(RustySatError::invalid_input(
                "lazy data array requires a 'y' dimension",
            ));
        };
        let Some(x_index) = self.dims.iter().position(|dim| dim == "x") else {
            return Err(RustySatError::invalid_input(
                "lazy data array requires an 'x' dimension",
            ));
        };
        Ok((self.shape[y_index], self.shape[x_index]))
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
                "lazy data array dimensions {:?} do not match expected {:?}",
                self.dims, expected
            )));
        }
        Ok(())
    }

    pub fn chunk_count(&self) -> usize {
        self.chunks
            .chunk_count_for_shape(&self.shape)
            .expect("chunks are validated during construction")
    }

    pub fn chunk_region(&self, chunk_index: &[usize]) -> Result<ChunkRegion> {
        if chunk_index.len() != self.shape.len() {
            return Err(RustySatError::invalid_input(format!(
                "chunk index has {} dimensions but data has {}",
                chunk_index.len(),
                self.shape.len()
            )));
        }

        let mut origin = Vec::with_capacity(chunk_index.len());
        let mut shape = Vec::with_capacity(chunk_index.len());
        for ((dim, chunk), index) in self
            .shape
            .iter()
            .zip(self.chunks.as_slice())
            .zip(chunk_index)
        {
            let start = index.checked_mul(*chunk).ok_or_else(|| {
                RustySatError::invalid_input("chunk index multiplication overflowed")
            })?;
            if start >= *dim {
                return Err(RustySatError::invalid_input(format!(
                    "chunk index {index} starts outside dimension {dim}"
                )));
            }
            origin.push(start);
            shape.push((*chunk).min(dim - start));
        }

        ChunkRegion::new(&self.shape, origin, shape)
    }

    pub fn chunk_regions(&self) -> Vec<ChunkRegion> {
        let counts = self
            .shape
            .iter()
            .zip(self.chunks.as_slice())
            .map(|(dim, chunk)| dim.div_ceil(*chunk))
            .collect::<Vec<_>>();
        let mut indexes = Vec::new();
        enumerate_chunk_indexes(&counts, &mut Vec::new(), &mut indexes);
        indexes
            .iter()
            .map(|index| {
                self.chunk_region(index)
                    .expect("enumerated chunk indexes are valid")
            })
            .collect()
    }

    pub fn read_chunk(&self, chunk_index: &[usize]) -> Result<DataArray<T>> {
        let region = self.chunk_region(chunk_index)?;
        let chunk = self.source.read_chunk(&region)?;
        if chunk.shape_nd() != region.shape() {
            return Err(RustySatError::invalid_input(format!(
                "chunk source returned shape {:?} for requested region {:?}",
                chunk.shape_nd(),
                region.shape()
            )));
        }
        Ok(chunk)
    }
}

fn enumerate_chunk_indexes(
    counts: &[usize],
    current: &mut Vec<usize>,
    output: &mut Vec<Vec<usize>>,
) {
    if current.len() == counts.len() {
        output.push(current.clone());
        return;
    }
    let dim_index = current.len();
    for index in 0..counts[dim_index] {
        current.push(index);
        enumerate_chunk_indexes(counts, current, output);
        current.pop();
    }
}

fn validate_region(full_shape: &[usize], origin: &[usize], shape: &[usize]) -> Result<()> {
    if full_shape.is_empty() {
        return Err(RustySatError::invalid_input(
            "chunk region requires a non-empty full shape",
        ));
    }
    if origin.len() != full_shape.len() || shape.len() != full_shape.len() {
        return Err(RustySatError::invalid_input(format!(
            "chunk region dimensions do not match full shape {:?}",
            full_shape
        )));
    }
    for ((dim, start), len) in full_shape.iter().zip(origin).zip(shape) {
        if *len == 0 {
            return Err(RustySatError::invalid_input(
                "chunk region shape dimensions must be non-zero",
            ));
        }
        let end = start
            .checked_add(*len)
            .ok_or_else(|| RustySatError::invalid_input("chunk region end overflowed"))?;
        if end > *dim {
            return Err(RustySatError::invalid_input(format!(
                "chunk region {:?}+{:?} exceeds full shape {:?}",
                origin, shape, full_shape
            )));
        }
    }
    Ok(())
}

fn validate_shape(shape: &[usize]) -> Result<()> {
    if shape.is_empty() {
        return Err(RustySatError::invalid_input(
            "lazy data array shape must have at least one dimension",
        ));
    }
    if shape.contains(&0) {
        return Err(RustySatError::invalid_input(
            "lazy data array dimensions must be non-zero",
        ));
    }
    Ok(())
}

fn validate_dims(shape: &[usize], dims: &[String]) -> Result<()> {
    if dims.len() != shape.len() {
        return Err(RustySatError::invalid_input(format!(
            "lazy data array has {} dimensions but {} dimension names",
            shape.len(),
            dims.len()
        )));
    }
    let mut seen = BTreeSet::new();
    for dim in dims {
        if dim.trim().is_empty() {
            return Err(RustySatError::invalid_input(
                "lazy data array dimension name cannot be empty",
            ));
        }
        if !seen.insert(dim) {
            return Err(RustySatError::invalid_input(format!(
                "duplicate lazy data array dimension name '{dim}'"
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

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;
    use std::sync::Mutex;

    #[derive(Debug)]
    struct RecordingSource {
        requests: Mutex<Vec<ChunkRegion>>,
    }

    impl RecordingSource {
        fn new() -> Self {
            Self {
                requests: Mutex::new(Vec::new()),
            }
        }
    }

    impl ChunkSource<u8> for RecordingSource {
        fn read_chunk(&self, region: &ChunkRegion) -> Result<DataArray<u8>> {
            self.requests.lock().unwrap().push(region.clone());
            let len = region.shape().iter().product();
            DataArray::from_vec_named(region.shape().to_vec(), ["y", "x"], vec![7; len])
        }
    }

    #[test]
    fn creates_lazy_array_with_validated_chunks() {
        let source = Arc::new(RecordingSource::new());
        let array =
            LazyDataArray::from_shape(vec![5, 6], ChunkShape::new(vec![2, 4]).unwrap(), source)
                .unwrap();

        assert_eq!(array.dtype(), DataType::U8);
        assert_eq!(array.shape(), &[5, 6]);
        assert_eq!(array.dims(), &["y".to_string(), "x".to_string()]);
        assert_eq!(array.chunks().as_slice(), &[2, 4]);
        assert_eq!(array.chunk_count(), 6);
    }

    #[test]
    fn enumerates_partial_edge_chunk_regions() {
        let source = Arc::new(RecordingSource::new());
        let array =
            LazyDataArray::from_shape(vec![5, 6], ChunkShape::new(vec![2, 4]).unwrap(), source)
                .unwrap();
        let regions = array.chunk_regions();

        assert_eq!(regions.len(), 6);
        assert_eq!(regions[0].origin(), &[0, 0]);
        assert_eq!(regions[0].shape(), &[2, 4]);
        assert_eq!(regions[1].origin(), &[0, 4]);
        assert_eq!(regions[1].shape(), &[2, 2]);
        assert_eq!(regions[5].origin(), &[4, 4]);
        assert_eq!(regions[5].shape(), &[1, 2]);
    }

    #[test]
    fn reads_chunk_through_source_without_materializing_everything() {
        let source = Arc::new(RecordingSource::new());
        let array = LazyDataArray::new(
            vec![5, 6],
            ["y", "x"],
            ChunkShape::new(vec![2, 4]).unwrap(),
            source.clone(),
        )
        .unwrap();

        let chunk = array.read_chunk(&[2, 1]).unwrap();

        assert_eq!(chunk.shape_nd(), &[1, 2]);
        assert_eq!(chunk.values(), &[7, 7]);
        let requests = source.requests.lock().unwrap();
        assert_eq!(
            requests.as_slice(),
            &[ChunkRegion::new(&[5, 6], [4, 4], [1, 2]).unwrap()]
        );
    }

    #[test]
    fn rejects_invalid_lazy_array_metadata() {
        let source = Arc::new(RecordingSource::new());

        assert!(LazyDataArray::<u8>::from_shape(
            vec![5, 6],
            ChunkShape::new(vec![6, 1]).unwrap(),
            source
        )
        .is_err());
    }

    #[test]
    fn rejects_out_of_range_chunk_indices() {
        let source = Arc::new(RecordingSource::new());
        let array =
            LazyDataArray::from_shape(vec![5, 6], ChunkShape::new(vec![2, 4]).unwrap(), source)
                .unwrap();

        assert!(array.chunk_region(&[3, 0]).is_err());
        assert!(array.chunk_region(&[0]).is_err());
    }
}
