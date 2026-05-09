//! Neighbour-info foundations modeled after Pyresample KD-tree output.
//!
//! Reference behavior inspected before implementation:
//! - `deps/pyresample/pyresample/kd_tree.py::get_neighbour_info`
//! - `deps/pyresample/pyresample/kd_tree.py::_query_resample_kdtree`
//! - `deps/pyresample/pyresample/kd_tree.py::_create_empty_info`
//!
//! This module does not build a KD-tree yet. It defines the reusable
//! neighbour-info contract and provides a projection-coordinate area-to-area
//! nearest query that future KD-tree code can replace internally.

use crate::AreaDefinition;
use rusty_sat_core::{DataGrid, Result, RustySatError, ValidityMask};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SampleMissingPolicy {
    FillValue,
    Mask,
}

impl SampleMissingPolicy {
    fn masks_missing(self) -> bool {
        matches!(self, Self::Mask)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Neighbour {
    reduced_source_index: usize,
    distance: f64,
}

impl Neighbour {
    pub fn new(reduced_source_index: usize, distance: f64) -> Result<Self> {
        if !distance.is_finite() || distance < 0.0 {
            return Err(RustySatError::invalid_input(
                "neighbour distance must be finite and non-negative",
            ));
        }
        Ok(Self {
            reduced_source_index,
            distance,
        })
    }

    pub fn reduced_source_index(&self) -> usize {
        self.reduced_source_index
    }

    pub fn distance(&self) -> f64 {
        self.distance
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct NeighbourInfo {
    source_size: usize,
    target_size: usize,
    neighbours: usize,
    valid_input_index: Vec<bool>,
    valid_output_index: Vec<bool>,
    index_array: Vec<usize>,
    distance_array: Vec<f64>,
}

impl NeighbourInfo {
    pub fn new(
        source_size: usize,
        target_size: usize,
        neighbours: usize,
        valid_input_index: Vec<bool>,
        valid_output_index: Vec<bool>,
        index_array: Vec<usize>,
        distance_array: Vec<f64>,
    ) -> Result<Self> {
        if neighbours == 0 {
            return Err(RustySatError::invalid_input(
                "neighbour count must be non-zero",
            ));
        }
        if valid_input_index.len() != source_size {
            return Err(RustySatError::invalid_input(format!(
                "valid_input_index length {} does not match source size {source_size}",
                valid_input_index.len()
            )));
        }
        if valid_output_index.len() != target_size {
            return Err(RustySatError::invalid_input(format!(
                "valid_output_index length {} does not match target size {target_size}",
                valid_output_index.len()
            )));
        }
        let expected_neighbour_entries = valid_output_index
            .iter()
            .filter(|valid| **valid)
            .count()
            .checked_mul(neighbours)
            .ok_or_else(|| RustySatError::invalid_input("neighbour info size overflows usize"))?;
        if index_array.len() != expected_neighbour_entries {
            return Err(RustySatError::invalid_input(format!(
                "index_array length {} does not match valid outputs * neighbours {expected_neighbour_entries}",
                index_array.len()
            )));
        }
        if distance_array.len() != expected_neighbour_entries {
            return Err(RustySatError::invalid_input(format!(
                "distance_array length {} does not match valid outputs * neighbours {expected_neighbour_entries}",
                distance_array.len()
            )));
        }
        Ok(Self {
            source_size,
            target_size,
            neighbours,
            valid_input_index,
            valid_output_index,
            index_array,
            distance_array,
        })
    }

    pub fn source_size(&self) -> usize {
        self.source_size
    }

    pub fn target_size(&self) -> usize {
        self.target_size
    }

    pub fn neighbours(&self) -> usize {
        self.neighbours
    }

    pub fn valid_input_index(&self) -> &[bool] {
        &self.valid_input_index
    }

    pub fn valid_output_index(&self) -> &[bool] {
        &self.valid_output_index
    }

    pub fn index_array(&self) -> &[usize] {
        &self.index_array
    }

    pub fn distance_array(&self) -> &[f64] {
        &self.distance_array
    }

    pub fn valid_input_count(&self) -> usize {
        self.valid_input_index
            .iter()
            .filter(|valid| **valid)
            .count()
    }

    pub fn valid_output_count(&self) -> usize {
        self.valid_output_index
            .iter()
            .filter(|valid| **valid)
            .count()
    }

    pub fn missing_neighbour_index(&self) -> usize {
        self.valid_input_count()
    }

    pub fn neighbours_for_output(
        &self,
        output_index: usize,
    ) -> Result<Option<Vec<Option<Neighbour>>>> {
        let Some(valid_output) = self.valid_output_index.get(output_index) else {
            return Err(RustySatError::invalid_input(format!(
                "output index {output_index} is outside target size {}",
                self.target_size
            )));
        };
        if !valid_output {
            return Ok(None);
        }
        let Some(valid_output_ordinal) = self.valid_output_ordinal(output_index) else {
            return Ok(None);
        };
        let start = valid_output_ordinal * self.neighbours;
        let missing_index = self.missing_neighbour_index();
        Ok(Some(
            (0..self.neighbours)
                .map(|offset| {
                    let reduced_source_index = self.index_array[start + offset];
                    let distance = self.distance_array[start + offset];
                    if reduced_source_index == missing_index || !distance.is_finite() {
                        None
                    } else {
                        Some(Neighbour {
                            reduced_source_index,
                            distance,
                        })
                    }
                })
                .collect(),
        ))
    }

    pub fn first_neighbour_for_output(&self, output_index: usize) -> Result<Option<Neighbour>> {
        Ok(self
            .neighbours_for_output(output_index)?
            .and_then(|neighbours| neighbours.into_iter().next().flatten()))
    }

    pub fn source_index_for_reduced_index(
        &self,
        reduced_source_index: usize,
    ) -> Result<Option<usize>> {
        let missing_index = self.missing_neighbour_index();
        if reduced_source_index == missing_index {
            return Ok(None);
        }
        let mut reduced = 0usize;
        for (source_index, valid) in self.valid_input_index.iter().enumerate() {
            if !valid {
                continue;
            }
            if reduced == reduced_source_index {
                return Ok(Some(source_index));
            }
            reduced += 1;
        }
        Err(RustySatError::invalid_input(format!(
            "reduced source index {reduced_source_index} is outside valid input count {missing_index}"
        )))
    }

    pub fn first_source_index_for_output(
        &self,
        output_index: usize,
    ) -> Result<Option<(usize, f64)>> {
        let Some(neighbour) = self.first_neighbour_for_output(output_index)? else {
            return Ok(None);
        };
        Ok(self
            .source_index_for_reduced_index(neighbour.reduced_source_index)?
            .map(|source_index| (source_index, neighbour.distance)))
    }

    fn valid_output_ordinal(&self, output_index: usize) -> Option<usize> {
        self.valid_output_index
            .iter()
            .take(output_index + 1)
            .filter(|valid| **valid)
            .count()
            .checked_sub(1)
    }
}

pub fn get_area_neighbour_info(
    source: &AreaDefinition,
    target: &AreaDefinition,
    radius_of_influence: Option<f64>,
) -> Result<NeighbourInfo> {
    if radius_of_influence.is_some_and(|radius| radius < 0.0) {
        return Err(RustySatError::invalid_input(
            "radius_of_influence must be non-negative",
        ));
    }
    if source.projection() != target.projection() {
        return Err(RustySatError::unsupported(
            "area neighbour info between different projections",
        ));
    }

    let source_size = source.height() * source.width();
    let target_size = target.height() * target.width();
    let valid_input_index = vec![true; source_size];
    let valid_output_index = vec![true; target_size];
    let mut index_array = Vec::with_capacity(target_size);
    let mut distance_array = Vec::with_capacity(target_size);
    // Sentinel must equal valid_input_count() so that
    // NeighbourInfo::missing_neighbour_index() stays consistent
    // when valid_input_index is later narrowed by masking.
    let missing_index = valid_input_index.iter().filter(|v| **v).count();

    for (target_x, target_y) in target.iter_projection_coords() {
        let Some((source_index, distance)) = nearest_area_pixel(source, target_x, target_y) else {
            index_array.push(missing_index);
            distance_array.push(f64::INFINITY);
            continue;
        };
        if radius_of_influence.is_some_and(|radius| distance > radius) {
            index_array.push(missing_index);
            distance_array.push(f64::INFINITY);
        } else {
            index_array.push(source_index);
            distance_array.push(distance);
        }
    }

    NeighbourInfo::new(
        source_size,
        target_size,
        1,
        valid_input_index,
        valid_output_index,
        index_array,
        distance_array,
    )
}

pub fn sample_nearest_from_neighbour_info(
    source_grid: &DataGrid,
    output_shape: (usize, usize),
    info: &NeighbourInfo,
    fill_value: f64,
    missing_policy: SampleMissingPolicy,
) -> Result<DataGrid> {
    if info.neighbours() != 1 {
        return Err(RustySatError::unsupported(
            "nearest neighbour sampling from multi-neighbour index arrays",
        ));
    }
    if source_grid.values().len() != info.source_size() {
        return Err(RustySatError::invalid_input(format!(
            "source grid has {} values but neighbour info source size is {}",
            source_grid.values().len(),
            info.source_size()
        )));
    }
    let output_size = output_shape
        .0
        .checked_mul(output_shape.1)
        .ok_or_else(|| RustySatError::invalid_input("output shape size overflows usize"))?;
    if output_size != info.target_size() {
        return Err(RustySatError::invalid_input(format!(
            "output shape {:?} has {output_size} pixels but neighbour info target size is {}",
            output_shape,
            info.target_size()
        )));
    }

    let mut values = Vec::with_capacity(output_size);
    let mut mask_flags = Vec::with_capacity(output_size);
    for output_index in 0..output_size {
        let Some((source_index, _distance)) = info.first_source_index_for_output(output_index)?
        else {
            values.push(fill_value);
            mask_flags.push(missing_policy.masks_missing());
            continue;
        };
        let source_masked = source_grid
            .is_masked(source_index)
            .expect("source_index is validated against source_size");
        values.push(source_grid.values()[source_index]);
        mask_flags.push(source_masked);
    }

    let has_mask = mask_flags.iter().any(|masked| *masked);
    let grid = DataGrid::new(output_shape.0, output_shape.1, values)?;
    if has_mask {
        grid.with_mask(ValidityMask::from_masked_flags(mask_flags))
    } else {
        Ok(grid)
    }
}

fn nearest_area_pixel(source: &AreaDefinition, x: f64, y: f64) -> Option<(usize, f64)> {
    let extent = source.area_extent();
    let src_x = clamp_pixel_index(
        (x - extent[0]) / source.pixel_size_x() - 0.5,
        source.width(),
    )?;
    let src_y = clamp_pixel_index(
        (extent[3] - y) / source.pixel_size_y() - 0.5,
        source.height(),
    )?;
    let nearest_x = source.projection_x_coord(src_x).ok()?;
    let nearest_y = source.projection_y_coord(src_y).ok()?;
    let dx = nearest_x - x;
    let dy = nearest_y - y;
    Some((src_y * source.width() + src_x, (dx * dx + dy * dy).sqrt()))
}

fn clamp_pixel_index(value: f64, size: usize) -> Option<usize> {
    if !value.is_finite() || size == 0 {
        return None;
    }
    let max_index = (size - 1) as f64;
    Some(value.round().clamp(0.0, max_index) as usize)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn area(id: &str, height: usize, width: usize, extent: [f64; 4]) -> AreaDefinition {
        AreaDefinition::from_parts(
            id,
            id,
            "test",
            BTreeMap::from([("proj".to_string(), "longlat".to_string())]),
            height,
            width,
            extent,
        )
        .unwrap()
    }

    #[test]
    fn constructs_neighbour_info_like_pyresample_tuple() {
        let info = NeighbourInfo::new(
            4,
            3,
            1,
            vec![true, true, false, true],
            vec![true, false, true],
            vec![0, 2],
            vec![0.0, 1.5],
        )
        .unwrap();

        assert_eq!(info.source_size(), 4);
        assert_eq!(info.target_size(), 3);
        assert_eq!(info.neighbours(), 1);
        assert_eq!(info.valid_input_count(), 3);
        assert_eq!(info.valid_output_count(), 2);
        assert_eq!(info.missing_neighbour_index(), 3);
        assert_eq!(
            info.first_neighbour_for_output(0).unwrap(),
            Some(Neighbour {
                reduced_source_index: 0,
                distance: 0.0
            })
        );
        assert_eq!(info.first_neighbour_for_output(1).unwrap(), None);
        assert_eq!(
            info.first_source_index_for_output(2).unwrap(),
            Some((3, 1.5))
        );
    }

    #[test]
    fn validates_neighbour_info_lengths() {
        let err = NeighbourInfo::new(1, 2, 1, vec![true], vec![true, true], vec![0], vec![0.0])
            .unwrap_err();

        assert!(err.to_string().contains("valid outputs"));
    }

    #[test]
    fn area_neighbour_info_matches_current_area_nearest_geometry() {
        let source = area("source", 2, 2, [0.0, 0.0, 2.0, 2.0]);
        let target = area("target", 2, 2, [0.0, 0.0, 2.0, 2.0]);

        let info = get_area_neighbour_info(&source, &target, None).unwrap();

        assert_eq!(info.valid_input_index(), &[true, true, true, true]);
        assert_eq!(info.valid_output_index(), &[true, true, true, true]);
        assert_eq!(info.index_array(), &[0, 1, 2, 3]);
        assert_eq!(info.distance_array(), &[0.0, 0.0, 0.0, 0.0]);
        assert_eq!(
            info.first_source_index_for_output(3).unwrap(),
            Some((3, 0.0))
        );
    }

    #[test]
    fn area_neighbour_info_uses_radius_sentinel_for_missing_neighbours() {
        let source = area("source", 1, 1, [0.0, 0.0, 1.0, 1.0]);
        let target = area("target", 1, 1, [2.0, 2.0, 3.0, 3.0]);

        let info = get_area_neighbour_info(&source, &target, Some(0.25)).unwrap();

        assert_eq!(info.index_array(), &[1]);
        assert!(info.distance_array()[0].is_infinite());
        assert_eq!(info.first_neighbour_for_output(0).unwrap(), None);
    }

    #[test]
    fn area_neighbour_info_rejects_different_projection_metadata() {
        let source = area("source", 1, 1, [0.0, 0.0, 1.0, 1.0]);
        let mut projection = BTreeMap::new();
        projection.insert("proj".to_string(), "stere".to_string());
        let target = AreaDefinition::from_parts(
            "target",
            "target",
            "target",
            projection,
            1,
            1,
            [0.0, 0.0, 1.0, 1.0],
        )
        .unwrap();

        assert!(matches!(
            get_area_neighbour_info(&source, &target, None).unwrap_err(),
            RustySatError::Unsupported { .. }
        ));
    }

    #[test]
    fn samples_nearest_values_from_neighbour_info() {
        let source = area("source", 2, 2, [0.0, 0.0, 2.0, 2.0]);
        let target = area("target", 2, 2, [0.0, 0.0, 2.0, 2.0]);
        let source_grid = DataGrid::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let info = get_area_neighbour_info(&source, &target, None).unwrap();

        let sampled = sample_nearest_from_neighbour_info(
            &source_grid,
            target.shape(),
            &info,
            -999.0,
            SampleMissingPolicy::FillValue,
        )
        .unwrap();

        assert_eq!(sampled.values(), &[1.0, 2.0, 3.0, 4.0]);
        assert!(sampled.mask().is_none());
    }

    #[test]
    fn samples_fill_or_mask_for_missing_neighbours() {
        let source = area("source", 1, 1, [0.0, 0.0, 1.0, 1.0]);
        let target = area("target", 1, 1, [2.0, 2.0, 3.0, 3.0]);
        let source_grid = DataGrid::new(1, 1, vec![7.0]).unwrap();
        let info = get_area_neighbour_info(&source, &target, Some(0.25)).unwrap();

        let filled = sample_nearest_from_neighbour_info(
            &source_grid,
            target.shape(),
            &info,
            -999.0,
            SampleMissingPolicy::FillValue,
        )
        .unwrap();
        let masked = sample_nearest_from_neighbour_info(
            &source_grid,
            target.shape(),
            &info,
            f64::NAN,
            SampleMissingPolicy::Mask,
        )
        .unwrap();

        assert_eq!(filled.values(), &[-999.0]);
        assert!(filled.mask().is_none());
        assert!(masked.values()[0].is_nan());
        assert_eq!(masked.mask().unwrap().masked_count(), 1);
    }

    #[test]
    fn sampling_propagates_source_mask() {
        let source = area("source", 1, 2, [0.0, 0.0, 2.0, 1.0]);
        let target = area("target", 1, 2, [0.0, 0.0, 2.0, 1.0]);
        let source_grid = DataGrid::new(1, 2, vec![1.0, 2.0])
            .unwrap()
            .with_mask(ValidityMask::from_masked_flags([false, true]))
            .unwrap();
        let info = get_area_neighbour_info(&source, &target, None).unwrap();

        let sampled = sample_nearest_from_neighbour_info(
            &source_grid,
            target.shape(),
            &info,
            -999.0,
            SampleMissingPolicy::FillValue,
        )
        .unwrap();

        assert_eq!(sampled.values(), &[1.0, 2.0]);
        assert_eq!(sampled.mask().unwrap().masked_count(), 1);
        assert_eq!(sampled.is_masked(1), Some(true));
    }

    #[test]
    fn sampling_validates_geometry_sizes() {
        let source_grid = DataGrid::new(1, 1, vec![7.0]).unwrap();
        let info =
            NeighbourInfo::new(2, 1, 1, vec![true, true], vec![true], vec![0], vec![0.0]).unwrap();

        let err = sample_nearest_from_neighbour_info(
            &source_grid,
            (1, 1),
            &info,
            -999.0,
            SampleMissingPolicy::FillValue,
        )
        .unwrap_err();

        assert!(err.to_string().contains("source grid has 1 values"));
    }
}
