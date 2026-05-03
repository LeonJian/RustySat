//! First nearest-neighbor resampling slice.
//!
//! Reference behavior inspected before implementation:
//! - `deps/pyresample/pyresample/kd_tree.py`
//! - `deps/pyresample/pyresample/geometry.py`
//! - `deps/pyresample/docs/source/concepts/resampling.rst`
//!
//! This module starts with projection-coordinate area-to-area resampling. It
//! uses pixel centers and an optional radius of influence like Pyresample, but
//! does not yet implement kd-tree lookup, CRS transforms, or full fill-vs-mask
//! policy.

use crate::{AreaDefinition, Resampler, SwathDefinition};
use rusty_sat_core::{
    Coordinate, DataGrid, Dataset, LazyDataArray, MetadataValue, Result, RustySatError,
    ValidityMask,
};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MissingValuePolicy {
    FillValue,
    Mask,
}

impl MissingValuePolicy {
    fn masks_missing(self) -> bool {
        matches!(self, Self::Mask)
    }
}

#[derive(Debug, Clone)]
pub struct NearestAreaResampler {
    source: AreaDefinition,
    radius_of_influence: Option<f64>,
    fill_value: f64,
    missing_value_policy: MissingValuePolicy,
}

impl NearestAreaResampler {
    pub fn new(source: AreaDefinition) -> Self {
        Self {
            source,
            radius_of_influence: None,
            fill_value: f64::NAN,
            missing_value_policy: MissingValuePolicy::FillValue,
        }
    }

    pub fn with_radius_of_influence(mut self, radius_of_influence: f64) -> Result<Self> {
        if radius_of_influence < 0.0 {
            return Err(RustySatError::invalid_input(
                "radius_of_influence must be non-negative",
            ));
        }
        self.radius_of_influence = Some(radius_of_influence);
        Ok(self)
    }

    pub fn with_fill_value(mut self, fill_value: f64) -> Self {
        self.fill_value = fill_value;
        self.missing_value_policy = MissingValuePolicy::FillValue;
        self
    }

    pub fn with_masked_missing(mut self) -> Self {
        self.missing_value_policy = MissingValuePolicy::Mask;
        self
    }

    pub fn source(&self) -> &AreaDefinition {
        &self.source
    }
}

impl Resampler for NearestAreaResampler {
    fn name(&self) -> &str {
        "nearest_area"
    }

    fn resample(&self, dataset: &Dataset, destination: &AreaDefinition) -> Result<Dataset> {
        let source_grid = dataset.data().ok_or_else(|| {
            RustySatError::invalid_input("nearest resampling requires dataset grid values")
        })?;
        if source_grid.shape() != self.source.shape() {
            return Err(RustySatError::invalid_input(format!(
                "dataset grid shape {:?} does not match source area shape {:?}",
                source_grid.shape(),
                self.source.shape()
            )));
        }
        if self.source.projection() != destination.projection() {
            return Err(RustySatError::unsupported(
                "nearest area resampling between different projections",
            ));
        }
        let resampled = resample_area_nearest_with_policy(
            source_grid,
            &self.source,
            destination,
            self.radius_of_influence,
            self.fill_value,
            self.missing_value_policy,
        )?;
        let mut resampled_dataset = Dataset::new(dataset.id().clone()).with_data(resampled);
        for (key, value) in dataset_metadata_pairs(dataset.metadata()) {
            resampled_dataset.insert_metadata(key, value)?;
        }
        for (key, value) in dataset_attr_pairs(dataset.attrs()) {
            resampled_dataset.insert_attr(key, value)?;
        }
        resampled_dataset.insert_metadata("area", destination.id())?;
        resampled_dataset.insert_metadata("resampler", self.name())?;
        Ok(resampled_dataset)
    }
}

fn dataset_metadata_pairs(
    metadata: &std::collections::BTreeMap<String, String>,
) -> Vec<(String, String)> {
    metadata
        .iter()
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect()
}

fn dataset_attr_pairs(
    attrs: &std::collections::BTreeMap<String, MetadataValue>,
) -> Vec<(String, MetadataValue)> {
    attrs
        .iter()
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect()
}

pub fn resample_area_nearest(
    source_grid: &DataGrid,
    source: &AreaDefinition,
    destination: &AreaDefinition,
    radius_of_influence: Option<f64>,
    fill_value: f64,
) -> Result<DataGrid> {
    resample_area_nearest_with_policy(
        source_grid,
        source,
        destination,
        radius_of_influence,
        fill_value,
        MissingValuePolicy::FillValue,
    )
}

pub fn resample_area_nearest_masked_missing(
    source_grid: &DataGrid,
    source: &AreaDefinition,
    destination: &AreaDefinition,
    radius_of_influence: Option<f64>,
) -> Result<DataGrid> {
    resample_area_nearest_with_policy(
        source_grid,
        source,
        destination,
        radius_of_influence,
        f64::NAN,
        MissingValuePolicy::Mask,
    )
}

pub fn resample_area_nearest_lazy(
    source_grid: &LazyDataArray<f64>,
    source: &AreaDefinition,
    destination: &AreaDefinition,
    radius_of_influence: Option<f64>,
    fill_value: f64,
) -> Result<DataGrid> {
    resample_area_nearest_lazy_with_policy(
        source_grid,
        source,
        destination,
        radius_of_influence,
        fill_value,
        MissingValuePolicy::FillValue,
    )
}

pub fn resample_area_nearest_lazy_masked_missing(
    source_grid: &LazyDataArray<f64>,
    source: &AreaDefinition,
    destination: &AreaDefinition,
    radius_of_influence: Option<f64>,
) -> Result<DataGrid> {
    resample_area_nearest_lazy_with_policy(
        source_grid,
        source,
        destination,
        radius_of_influence,
        f64::NAN,
        MissingValuePolicy::Mask,
    )
}

fn resample_area_nearest_with_policy(
    source_grid: &DataGrid,
    source: &AreaDefinition,
    destination: &AreaDefinition,
    radius_of_influence: Option<f64>,
    fill_value: f64,
    missing_value_policy: MissingValuePolicy,
) -> Result<DataGrid> {
    if source_grid.shape() != source.shape() {
        return Err(RustySatError::invalid_input(format!(
            "source grid shape {:?} does not match source area shape {:?}",
            source_grid.shape(),
            source.shape()
        )));
    }
    let (dst_height, dst_width) = destination.shape();
    let mut values = Vec::with_capacity(dst_height * dst_width);
    let mut mask_flags = Vec::with_capacity(dst_height * dst_width);
    for y in 0..dst_height {
        for x in 0..dst_width {
            let (dst_x, dst_y) = pixel_center(destination, y, x);
            let Some((src_y, src_x, distance)) = nearest_source_pixel(source, dst_x, dst_y) else {
                values.push(fill_value);
                mask_flags.push(missing_value_policy.masks_missing());
                continue;
            };
            if radius_of_influence.is_some_and(|radius| distance > radius) {
                values.push(fill_value);
                mask_flags.push(missing_value_policy.masks_missing());
                continue;
            }
            let source_index = src_y * source.shape().1 + src_x;
            let source_masked = source_grid.is_masked(source_index).unwrap_or(false);
            values.push(source_grid.get(src_y, src_x).unwrap_or(fill_value));
            mask_flags.push(source_masked);
        }
    }
    add_resampled_coords(
        finish_resampled_grid(dst_height, dst_width, values, mask_flags)?,
        Some(source_grid),
        destination,
    )
}

fn resample_area_nearest_lazy_with_policy(
    source_grid: &LazyDataArray<f64>,
    source: &AreaDefinition,
    destination: &AreaDefinition,
    radius_of_influence: Option<f64>,
    fill_value: f64,
    missing_value_policy: MissingValuePolicy,
) -> Result<DataGrid> {
    if source_grid.shape_yx()? != source.shape() {
        return Err(RustySatError::invalid_input(format!(
            "lazy source grid shape {:?} does not match source area shape {:?}",
            source_grid.shape_yx()?,
            source.shape()
        )));
    }
    source_grid.require_dims_exact(&["y", "x"])?;
    let (dst_height, dst_width) = destination.shape();
    let mut values = Vec::with_capacity(dst_height * dst_width);
    let mut mask_flags = Vec::with_capacity(dst_height * dst_width);
    let mut source_chunks = SourceChunkCache::new(source_grid);
    for y in 0..dst_height {
        for x in 0..dst_width {
            let (dst_x, dst_y) = pixel_center(destination, y, x);
            let Some((src_y, src_x, distance)) = nearest_source_pixel(source, dst_x, dst_y) else {
                values.push(fill_value);
                mask_flags.push(missing_value_policy.masks_missing());
                continue;
            };
            if radius_of_influence.is_some_and(|radius| distance > radius) {
                values.push(fill_value);
                mask_flags.push(missing_value_policy.masks_missing());
                continue;
            }
            let (value, masked) = source_chunks.value_at(src_y, src_x, fill_value)?;
            values.push(value);
            mask_flags.push(masked);
        }
    }
    add_resampled_coords(
        finish_resampled_grid(dst_height, dst_width, values, mask_flags)?,
        None,
        destination,
    )
}

struct SourceChunkCache<'a> {
    source_grid: &'a LazyDataArray<f64>,
    chunks: BTreeMap<(usize, usize), DataGrid>,
}

impl<'a> SourceChunkCache<'a> {
    fn new(source_grid: &'a LazyDataArray<f64>) -> Self {
        Self {
            source_grid,
            chunks: BTreeMap::new(),
        }
    }

    fn value_at(&mut self, y: usize, x: usize, fill_value: f64) -> Result<(f64, bool)> {
        let chunk_y = self.source_grid.chunks().as_slice()[0];
        let chunk_x = self.source_grid.chunks().as_slice()[1];
        let chunk_index = (y / chunk_y, x / chunk_x);
        if !self.chunks.contains_key(&chunk_index) {
            let chunk = self
                .source_grid
                .read_chunk(&[chunk_index.0, chunk_index.1])?;
            self.chunks.insert(chunk_index, chunk);
        }
        let chunk = self
            .chunks
            .get(&chunk_index)
            .expect("chunk inserted or existed");
        let local_y = y % chunk_y;
        let local_x = x % chunk_x;
        let (_, width) = chunk.shape();
        let local_index = local_y * width + local_x;
        Ok((
            chunk.get(local_y, local_x).unwrap_or(fill_value),
            chunk.is_masked(local_index).unwrap_or(false),
        ))
    }
}

pub fn resample_swath_nearest(
    source_grid: &DataGrid,
    source: &SwathDefinition,
    destination: &AreaDefinition,
    radius_of_influence: Option<f64>,
    fill_value: f64,
) -> Result<DataGrid> {
    resample_swath_nearest_with_policy(
        source_grid,
        source,
        destination,
        radius_of_influence,
        fill_value,
        MissingValuePolicy::FillValue,
    )
}

pub fn resample_swath_nearest_masked_missing(
    source_grid: &DataGrid,
    source: &SwathDefinition,
    destination: &AreaDefinition,
    radius_of_influence: Option<f64>,
) -> Result<DataGrid> {
    resample_swath_nearest_with_policy(
        source_grid,
        source,
        destination,
        radius_of_influence,
        f64::NAN,
        MissingValuePolicy::Mask,
    )
}

fn resample_swath_nearest_with_policy(
    source_grid: &DataGrid,
    source: &SwathDefinition,
    destination: &AreaDefinition,
    radius_of_influence: Option<f64>,
    fill_value: f64,
    missing_value_policy: MissingValuePolicy,
) -> Result<DataGrid> {
    if source_grid.shape() != source.shape() {
        return Err(RustySatError::invalid_input(format!(
            "source grid shape {:?} does not match source swath shape {:?}",
            source_grid.shape(),
            source.shape()
        )));
    }
    require_lonlat_area(destination)?;
    let lons = source.lons().ok_or_else(|| {
        RustySatError::invalid_input("swath nearest resampling requires longitude coordinates")
    })?;
    let lats = source.lats().ok_or_else(|| {
        RustySatError::invalid_input("swath nearest resampling requires latitude coordinates")
    })?;
    let (dst_height, dst_width) = destination.shape();
    let mut values = Vec::with_capacity(dst_height * dst_width);
    let mut mask_flags = Vec::with_capacity(dst_height * dst_width);
    for y in 0..dst_height {
        for x in 0..dst_width {
            let (dst_x, dst_y) = pixel_center(destination, y, x);
            let Some((source_index, distance)) = nearest_swath_point(lons, lats, dst_x, dst_y)
            else {
                values.push(fill_value);
                mask_flags.push(missing_value_policy.masks_missing());
                continue;
            };
            if radius_of_influence.is_some_and(|radius| distance > radius) {
                values.push(fill_value);
                mask_flags.push(missing_value_policy.masks_missing());
                continue;
            }
            let source_masked = source_grid.is_masked(source_index).unwrap_or(false);
            values.push(source_grid.values()[source_index]);
            mask_flags.push(source_masked);
        }
    }
    add_resampled_coords(
        finish_resampled_grid(dst_height, dst_width, values, mask_flags)?,
        Some(source_grid),
        destination,
    )
}

fn finish_resampled_grid(
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

fn add_resampled_coords(
    mut grid: DataGrid,
    source_grid: Option<&DataGrid>,
    area: &AreaDefinition,
) -> Result<DataGrid> {
    if let Some(source_grid) = source_grid {
        for (name, coordinate) in source_grid.coords() {
            if should_preserve_coord(name, coordinate) {
                grid.set_coordinate(name.clone(), coordinate.clone())?;
            }
        }
    }
    grid.set_coordinate("x", Coordinate::axis("x", area.projection_x_coords())?)?;
    grid.set_coordinate("y", Coordinate::axis("y", area.projection_y_coords())?)?;
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

fn nearest_source_pixel(source: &AreaDefinition, x: f64, y: f64) -> Option<(usize, usize, f64)> {
    let extent = source.area_extent();
    let (pixel_size_x, pixel_size_y) = source.pixel_size();
    let (height, width) = source.shape();
    let src_x = clamp_pixel_index((x - extent[0]) / pixel_size_x - 0.5, width)?;
    let src_y = clamp_pixel_index((extent[3] - y) / pixel_size_y - 0.5, height)?;
    let (nearest_x, nearest_y) = pixel_center(source, src_y, src_x);
    let distance = ((nearest_x - x).powi(2) + (nearest_y - y).powi(2)).sqrt();
    Some((src_y, src_x, distance))
}

fn nearest_swath_point(lons: &[f64], lats: &[f64], x: f64, y: f64) -> Option<(usize, f64)> {
    lons.iter()
        .zip(lats)
        .enumerate()
        .filter_map(|(idx, (lon, lat))| {
            if !lon.is_finite() || !lat.is_finite() {
                return None;
            }
            let distance = ((*lon - x).powi(2) + (*lat - y).powi(2)).sqrt();
            Some((idx, distance))
        })
        .min_by(|left, right| left.1.total_cmp(&right.1))
}

fn require_lonlat_area(area: &AreaDefinition) -> Result<()> {
    let projection = area.projection();
    let Some(proj) = projection.get("proj").or_else(|| projection.get("proj4")) else {
        return Err(RustySatError::unsupported(
            "swath nearest resampling without lon/lat destination projection metadata",
        ));
    };
    if proj.contains("latlong") || proj.contains("longlat") {
        return Ok(());
    }
    Err(RustySatError::unsupported(
        "swath nearest resampling to non-lon/lat destination area",
    ))
}

fn clamp_pixel_index(value: f64, size: usize) -> Option<usize> {
    if !value.is_finite() || size == 0 {
        return None;
    }
    let max_index = (size - 1) as f64;
    Some(value.round().clamp(0.0, max_index) as usize)
}

fn pixel_center(area: &AreaDefinition, y: usize, x: usize) -> (f64, f64) {
    let extent = area.area_extent();
    let (pixel_size_x, pixel_size_y) = area.pixel_size();
    (
        extent[0] + (x as f64 + 0.5) * pixel_size_x,
        extent[3] - (y as f64 + 0.5) * pixel_size_y,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use rusty_sat_core::{ChunkRegion, ChunkShape, ChunkSource, DataArray};
    use std::collections::BTreeMap;
    use std::sync::{Arc, Mutex};

    fn area(id: &str, height: usize, width: usize, area_extent: [f64; 4]) -> AreaDefinition {
        AreaDefinition::from_parts(
            id,
            id,
            id,
            BTreeMap::from([("proj".to_string(), "latlong".to_string())]),
            height,
            width,
            area_extent,
        )
        .unwrap()
    }

    #[derive(Debug)]
    struct MatrixSource {
        width: usize,
        values: Vec<f64>,
        requests: Mutex<Vec<ChunkRegion>>,
    }

    impl MatrixSource {
        fn new(width: usize, values: Vec<f64>) -> Self {
            Self {
                width,
                values,
                requests: Mutex::new(Vec::new()),
            }
        }
    }

    impl ChunkSource<f64> for MatrixSource {
        fn read_chunk(&self, region: &ChunkRegion) -> Result<DataArray<f64>> {
            self.requests.lock().unwrap().push(region.clone());
            let [origin_y, origin_x] = region.origin() else {
                panic!("test source only supports 2D regions");
            };
            let [height, width] = region.shape() else {
                panic!("test source only supports 2D regions");
            };
            let mut values = Vec::with_capacity(height * width);
            for y in 0..*height {
                for x in 0..*width {
                    let source_idx = (*origin_y + y) * self.width + *origin_x + x;
                    values.push(self.values[source_idx]);
                }
            }
            DataArray::from_vec_named(vec![*height, *width], ["y", "x"], values)
        }
    }

    #[test]
    fn nearest_resamples_area_to_finer_area() {
        let source = area("source", 2, 2, [0.0, 0.0, 2.0, 2.0]);
        let destination = area("destination", 4, 4, [0.0, 0.0, 2.0, 2.0]);
        let source_grid = DataGrid::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();

        let result =
            resample_area_nearest(&source_grid, &source, &destination, None, f64::NAN).unwrap();

        assert_eq!(result.shape(), (4, 4));
        assert_eq!(
            result.values(),
            &[
                1.0, 1.0, 2.0, 2.0, //
                1.0, 1.0, 2.0, 2.0, //
                3.0, 3.0, 4.0, 4.0, //
                3.0, 3.0, 4.0, 4.0,
            ]
        );
        assert_eq!(
            result.coord("x").unwrap().values(),
            &[0.25, 0.75, 1.25, 1.75]
        );
        assert_eq!(
            result.coord("y").unwrap().values(),
            &[1.75, 1.25, 0.75, 0.25]
        );
    }

    #[test]
    fn nearest_propagates_source_mask_for_area_resampling() {
        let source = area("source", 2, 2, [0.0, 0.0, 2.0, 2.0]);
        let destination = area("destination", 2, 2, [0.0, 0.0, 2.0, 2.0]);
        let source_grid = DataGrid::new(2, 2, vec![1.0, 2.0, 3.0, 4.0])
            .unwrap()
            .with_mask(ValidityMask::from_masked_flags([false, true, false, false]))
            .unwrap();

        let result =
            resample_area_nearest(&source_grid, &source, &destination, None, -999.0).unwrap();

        assert_eq!(result.values(), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(result.mask().unwrap().masked_count(), 1);
        assert_eq!(result.is_masked(1), Some(true));
    }

    #[test]
    fn nearest_preserves_non_xy_coordinates_and_replaces_xy_axes() {
        let source = area("source", 2, 2, [0.0, 0.0, 2.0, 2.0]);
        let destination = area("destination", 1, 1, [0.0, 1.0, 1.0, 2.0]);
        let source_grid = DataGrid::new(2, 2, vec![1.0, 2.0, 3.0, 4.0])
            .unwrap()
            .with_coordinate("acq_time", Coordinate::scalar(123.0))
            .unwrap()
            .with_coordinate("x", Coordinate::axis("x", vec![0.5, 1.5]).unwrap())
            .unwrap()
            .with_coordinate(
                "longitude",
                Coordinate::new(["y", "x"], vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
            )
            .unwrap();

        let result =
            resample_area_nearest(&source_grid, &source, &destination, None, f64::NAN).unwrap();

        assert_eq!(result.coord("acq_time").unwrap().values(), &[123.0]);
        assert!(result.coord("longitude").is_none());
        assert_eq!(result.coord("x").unwrap().values(), &[0.5]);
        assert_eq!(result.coord("y").unwrap().values(), &[1.5]);
    }

    #[test]
    fn nearest_resamples_lazy_area_source_by_loading_source_chunks() {
        let source = area("source", 2, 2, [0.0, 0.0, 2.0, 2.0]);
        let destination = area("destination", 4, 4, [0.0, 0.0, 2.0, 2.0]);
        let source_values = Arc::new(MatrixSource::new(2, vec![1.0, 2.0, 3.0, 4.0]));
        let source_grid = LazyDataArray::from_shape(
            vec![2, 2],
            ChunkShape::new(vec![1, 1]).unwrap(),
            source_values.clone(),
        )
        .unwrap();

        let result =
            resample_area_nearest_lazy(&source_grid, &source, &destination, None, f64::NAN)
                .unwrap();

        assert_eq!(result.shape(), (4, 4));
        assert_eq!(result.coord("x").unwrap().dims(), &["x".to_string()]);
        assert_eq!(result.coord("y").unwrap().dims(), &["y".to_string()]);
        assert_eq!(
            result.values(),
            &[
                1.0, 1.0, 2.0, 2.0, //
                1.0, 1.0, 2.0, 2.0, //
                3.0, 3.0, 4.0, 4.0, //
                3.0, 3.0, 4.0, 4.0,
            ]
        );
        assert_eq!(source_values.requests.lock().unwrap().len(), 4);
    }

    #[test]
    fn nearest_lazy_area_source_uses_fill_value_outside_radius() {
        let source = area("source", 1, 1, [0.0, 0.0, 1.0, 1.0]);
        let destination = area("destination", 1, 1, [1.0, 1.0, 2.0, 2.0]);
        let source_values = Arc::new(MatrixSource::new(1, vec![5.0]));
        let source_grid = LazyDataArray::from_shape(
            vec![1, 1],
            ChunkShape::new(vec![1, 1]).unwrap(),
            source_values,
        )
        .unwrap();

        let result =
            resample_area_nearest_lazy(&source_grid, &source, &destination, Some(0.25), -999.0)
                .unwrap();

        assert_eq!(result.values(), &[-999.0]);
    }

    #[test]
    fn nearest_uses_fill_value_outside_radius() {
        let source = area("source", 1, 1, [0.0, 0.0, 1.0, 1.0]);
        let destination = area("destination", 1, 1, [1.0, 1.0, 2.0, 2.0]);
        let source_grid = DataGrid::new(1, 1, vec![5.0]).unwrap();

        let result =
            resample_area_nearest(&source_grid, &source, &destination, Some(0.25), -999.0).unwrap();

        assert_eq!(result.values(), &[-999.0]);
    }

    #[test]
    fn nearest_can_mask_missing_area_pixels() {
        let source = area("source", 1, 1, [0.0, 0.0, 1.0, 1.0]);
        let destination = area("destination", 1, 1, [1.0, 1.0, 2.0, 2.0]);
        let source_grid = DataGrid::new(1, 1, vec![5.0]).unwrap();

        let result =
            resample_area_nearest_masked_missing(&source_grid, &source, &destination, Some(0.25))
                .unwrap();

        assert!(result.values()[0].is_nan());
        assert_eq!(result.mask().unwrap().masked_count(), 1);
        assert_eq!(result.is_masked(0), Some(true));
    }

    #[test]
    fn nearest_uses_edge_pixel_outside_extent_when_inside_radius() {
        let source = area("source", 1, 1, [0.0, 0.0, 1.0, 1.0]);
        let destination = area("destination", 1, 1, [1.0, 0.0, 2.0, 1.0]);
        let source_grid = DataGrid::new(1, 1, vec![5.0]).unwrap();

        let result =
            resample_area_nearest(&source_grid, &source, &destination, Some(1.0), -999.0).unwrap();

        assert_eq!(result.values(), &[5.0]);
    }

    #[test]
    fn nearest_without_radius_uses_nearest_edge_pixel_for_outside_target() {
        let source = area("source", 2, 2, [0.0, 0.0, 2.0, 2.0]);
        let destination = area("destination", 1, 1, [2.0, 1.0, 3.0, 2.0]);
        let source_grid = DataGrid::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();

        let result =
            resample_area_nearest(&source_grid, &source, &destination, None, -999.0).unwrap();

        assert_eq!(result.values(), &[2.0]);
    }

    #[test]
    fn nearest_zero_radius_only_accepts_exact_pixel_center_matches() {
        let source = area("source", 1, 1, [0.0, 0.0, 1.0, 1.0]);
        let exact_destination = area("exact", 1, 1, [0.0, 0.0, 1.0, 1.0]);
        let shifted_destination = area("shifted", 1, 1, [0.1, 0.0, 1.1, 1.0]);
        let source_grid = DataGrid::new(1, 1, vec![5.0]).unwrap();

        let exact =
            resample_area_nearest(&source_grid, &source, &exact_destination, Some(0.0), -999.0)
                .unwrap();
        let shifted = resample_area_nearest(
            &source_grid,
            &source,
            &shifted_destination,
            Some(0.0),
            -999.0,
        )
        .unwrap();

        assert_eq!(exact.values(), &[5.0]);
        assert_eq!(shifted.values(), &[-999.0]);
    }

    #[test]
    fn resampler_rejects_different_projection_metadata() {
        let source = area("source", 1, 1, [0.0, 0.0, 1.0, 1.0]);
        let destination = AreaDefinition::from_parts(
            "destination",
            "destination",
            "destination",
            BTreeMap::from([("proj".to_string(), "merc".to_string())]),
            1,
            1,
            [0.0, 0.0, 1.0, 1.0],
        )
        .unwrap();
        let id = rusty_sat_core::DataId::new("image").unwrap();
        let dataset = Dataset::new(id).with_data(DataGrid::new(1, 1, vec![5.0]).unwrap());
        let resampler = NearestAreaResampler::new(source);

        assert!(matches!(
            resampler.resample(&dataset, &destination).unwrap_err(),
            RustySatError::Unsupported { .. }
        ));
    }

    #[test]
    fn nearest_resamples_swath_points_to_lonlat_area() {
        let swath =
            SwathDefinition::from_lonlats(2, 2, vec![0.5, 1.5, 0.5, 1.5], vec![1.5, 1.5, 0.5, 0.5])
                .unwrap();
        let source_grid = DataGrid::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let destination = area("destination", 2, 2, [0.0, 0.0, 2.0, 2.0]);

        let result =
            resample_swath_nearest(&source_grid, &swath, &destination, Some(0.0), -999.0).unwrap();

        assert_eq!(result.values(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn nearest_propagates_source_mask_for_swath_resampling() {
        let swath =
            SwathDefinition::from_lonlats(2, 2, vec![0.5, 1.5, 0.5, 1.5], vec![1.5, 1.5, 0.5, 0.5])
                .unwrap();
        let source_grid = DataGrid::new(2, 2, vec![1.0, 2.0, 3.0, 4.0])
            .unwrap()
            .with_mask(ValidityMask::from_masked_flags([false, false, true, false]))
            .unwrap();
        let destination = area("destination", 2, 2, [0.0, 0.0, 2.0, 2.0]);

        let result =
            resample_swath_nearest(&source_grid, &swath, &destination, Some(0.0), -999.0).unwrap();

        assert_eq!(result.values(), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(result.mask().unwrap().masked_count(), 1);
        assert_eq!(result.is_masked(2), Some(true));
    }

    #[test]
    fn nearest_swath_uses_fill_value_when_outside_radius() {
        let swath = SwathDefinition::from_lonlats(1, 1, vec![0.5], vec![0.5]).unwrap();
        let source_grid = DataGrid::new(1, 1, vec![7.0]).unwrap();
        let destination = area("destination", 1, 1, [1.0, 1.0, 2.0, 2.0]);

        let result =
            resample_swath_nearest(&source_grid, &swath, &destination, Some(0.25), -999.0).unwrap();

        assert_eq!(result.values(), &[-999.0]);
    }

    #[test]
    fn nearest_can_mask_missing_swath_pixels() {
        let swath = SwathDefinition::from_lonlats(1, 1, vec![0.5], vec![0.5]).unwrap();
        let source_grid = DataGrid::new(1, 1, vec![7.0]).unwrap();
        let destination = area("destination", 1, 1, [1.0, 1.0, 2.0, 2.0]);

        let result =
            resample_swath_nearest_masked_missing(&source_grid, &swath, &destination, Some(0.25))
                .unwrap();

        assert!(result.values()[0].is_nan());
        assert_eq!(result.mask().unwrap().masked_count(), 1);
        assert_eq!(result.is_masked(0), Some(true));
    }

    #[test]
    fn nearest_swath_requires_coordinates() {
        let swath = SwathDefinition::new(1, 1).unwrap();
        let source_grid = DataGrid::new(1, 1, vec![7.0]).unwrap();
        let destination = area("destination", 1, 1, [0.0, 0.0, 1.0, 1.0]);

        assert!(matches!(
            resample_swath_nearest(&source_grid, &swath, &destination, None, -999.0).unwrap_err(),
            RustySatError::InvalidInput { .. }
        ));
    }

    #[test]
    fn resampler_trait_returns_dataset_with_destination_area_metadata() {
        let source = area("source", 2, 2, [0.0, 0.0, 2.0, 2.0]);
        let destination = area("destination", 1, 1, [0.0, 1.0, 1.0, 2.0]);
        let id = rusty_sat_core::DataId::new("image").unwrap();
        let mut dataset =
            Dataset::new(id).with_data(DataGrid::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap());
        dataset.insert_metadata("units", "K").unwrap();
        dataset
            .insert_attr(
                "orbital_parameters",
                MetadataValue::map([(
                    "satellite_nominal_longitude",
                    MetadataValue::float(140.7).unwrap(),
                )]),
            )
            .unwrap();
        let resampler = NearestAreaResampler::new(source).with_fill_value(-999.0);

        let result = resampler.resample(&dataset, &destination).unwrap();

        assert_eq!(result.data().unwrap().values(), &[1.0]);
        assert_eq!(result.metadata().get("units"), Some(&"K".to_string()));
        assert_eq!(
            result.attr("units").and_then(MetadataValue::as_str),
            Some("K")
        );
        assert_eq!(
            result
                .attr("orbital_parameters")
                .and_then(|value| { value.get_path(&["satellite_nominal_longitude"]) }),
            Some(&MetadataValue::float(140.7).unwrap())
        );
        assert_eq!(
            result.metadata().get("area"),
            Some(&"destination".to_string())
        );
        assert_eq!(
            result.metadata().get("resampler"),
            Some(&"nearest_area".to_string())
        );
    }

    #[test]
    fn resampler_trait_can_mask_missing_area_pixels() {
        let source = area("source", 1, 1, [0.0, 0.0, 1.0, 1.0]);
        let destination = area("destination", 1, 1, [1.0, 1.0, 2.0, 2.0]);
        let id = rusty_sat_core::DataId::new("image").unwrap();
        let dataset = Dataset::new(id).with_data(DataGrid::new(1, 1, vec![5.0]).unwrap());
        let resampler = NearestAreaResampler::new(source)
            .with_radius_of_influence(0.25)
            .unwrap()
            .with_masked_missing();

        let result = resampler.resample(&dataset, &destination).unwrap();

        assert!(result.data().unwrap().values()[0].is_nan());
        assert_eq!(result.data().unwrap().mask().unwrap().masked_count(), 1);
    }
}
