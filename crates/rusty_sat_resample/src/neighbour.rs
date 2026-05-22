//! Neighbour-info foundations modeled after Pyresample KD-tree output.
//!
//! Reference behavior inspected before implementation:
//! - `deps/pyresample/pyresample/kd_tree.py::get_neighbour_info`
//! - `deps/pyresample/pyresample/kd_tree.py::_query_resample_kdtree`
//! - `deps/pyresample/pyresample/kd_tree.py::_create_empty_info`
//! - `deps/pyresample/pyresample/kd_tree.py::get_sample_from_neighbour_info`
//!
//! This module does not build a geocentric KD-tree yet. It defines the
//! reusable neighbour-info contract and provides projection-coordinate
//! area-to-area queries that future KD-tree code can replace internally.

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
    valid_input_count: usize,
    valid_output_count: usize,
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
        let valid_input_count = valid_input_index.iter().filter(|v| **v).count();
        let valid_output_count = valid_output_index.iter().filter(|v| **v).count();
        let expected_neighbour_entries = valid_output_count
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
            valid_input_count,
            valid_output_count,
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
        self.valid_input_count
    }

    pub fn valid_output_count(&self) -> usize {
        self.valid_output_count
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
    get_area_neighbour_info_with_neighbours(source, target, radius_of_influence, 1)
}

pub fn get_area_neighbour_info_with_neighbours(
    source: &AreaDefinition,
    target: &AreaDefinition,
    radius_of_influence: Option<f64>,
    neighbours: usize,
) -> Result<NeighbourInfo> {
    if neighbours == 0 {
        return Err(RustySatError::invalid_input(
            "neighbour count must be non-zero",
        ));
    }
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
    let mut index_array = Vec::with_capacity(target_size * neighbours);
    let mut distance_array = Vec::with_capacity(target_size * neighbours);
    // Sentinel must equal valid_input_count() so that
    // NeighbourInfo::missing_neighbour_index() stays consistent
    // when valid_input_index is later narrowed by masking.
    let missing_index = valid_input_index.iter().filter(|v| **v).count();

    for (target_x, target_y) in target.iter_projection_coords() {
        let neighbours_for_target =
            nearest_area_pixels(source, target_x, target_y, radius_of_influence, neighbours);
        for offset in 0..neighbours {
            if let Some((source_index, distance)) = neighbours_for_target.get(offset) {
                index_array.push(*source_index);
                distance_array.push(*distance);
            } else {
                index_array.push(missing_index);
                distance_array.push(f64::INFINITY);
            }
        }
    }

    NeighbourInfo::new(
        source_size,
        target_size,
        neighbours,
        valid_input_index,
        valid_output_index,
        index_array,
        distance_array,
    )
}

pub fn sample_weighted_from_neighbour_info(
    source_grid: &DataGrid,
    output_shape: (usize, usize),
    info: &NeighbourInfo,
    fill_value: f64,
    missing_policy: SampleMissingPolicy,
    weight_fn: impl Fn(f64) -> f64,
) -> Result<DataGrid> {
    validate_sampling_geometry(source_grid.values().len(), output_shape, info)?;
    let output_size = output_shape.0 * output_shape.1;
    let mut values = Vec::with_capacity(output_size);
    let mut mask_flags = Vec::with_capacity(output_size);
    for output_index in 0..output_size {
        let Some(weighted_value) = weighted_value_for_output(
            output_index,
            info,
            fill_value,
            &weight_fn,
            |source_index| {
                let masked = source_grid
                    .is_masked(source_index)
                    .expect("source_index is validated against source_size");
                (source_grid.values()[source_index], masked)
            },
        )?
        else {
            values.push(fill_value);
            mask_flags.push(missing_policy.masks_missing());
            continue;
        };
        values.push(weighted_value);
        mask_flags.push(false);
    }
    build_sampled_grid(output_shape, values, mask_flags)
}

pub fn sample_weighted_from_neighbour_info_owned(
    source_grid: DataGrid,
    output_shape: (usize, usize),
    info: &NeighbourInfo,
    fill_value: f64,
    missing_policy: SampleMissingPolicy,
    weight_fn: impl Fn(f64) -> f64,
) -> Result<DataGrid> {
    validate_sampling_geometry(source_grid.values().len(), output_shape, info)?;
    let (src_values, _src_coords, src_mask) = source_grid.into_parts();
    let output_size = output_shape.0 * output_shape.1;
    let mut values = Vec::with_capacity(output_size);
    let mut mask_flags = Vec::with_capacity(output_size);
    for output_index in 0..output_size {
        let Some(weighted_value) = weighted_value_for_output(
            output_index,
            info,
            fill_value,
            &weight_fn,
            |source_index| {
                let masked = src_mask
                    .as_ref()
                    .and_then(|mask| mask.is_masked(source_index))
                    .unwrap_or(false);
                (src_values[source_index], masked)
            },
        )?
        else {
            values.push(fill_value);
            mask_flags.push(missing_policy.masks_missing());
            continue;
        };
        values.push(weighted_value);
        mask_flags.push(false);
    }
    build_sampled_grid(output_shape, values, mask_flags)
}

pub fn gaussian_weight(distance: f64, sigma: f64) -> Result<f64> {
    if !distance.is_finite() || distance < 0.0 {
        return Err(RustySatError::invalid_input(
            "gaussian distance must be finite and non-negative",
        ));
    }
    if !sigma.is_finite() || sigma <= 0.0 {
        return Err(RustySatError::invalid_input(
            "gaussian sigma must be finite and positive",
        ));
    }
    Ok((-distance * distance / (2.0 * sigma * sigma)).exp())
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
    let output_size = validate_sampling_geometry(source_grid.values().len(), output_shape, info)?;

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

    build_sampled_grid(output_shape, values, mask_flags)
}

pub fn sample_nearest_from_neighbour_info_owned(
    source_grid: DataGrid,
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
    let output_size = validate_sampling_geometry(source_grid.values().len(), output_shape, info)?;

    let (src_values, _src_coords, src_mask) = source_grid.into_parts();
    let mut values = Vec::with_capacity(output_size);
    let mut mask_flags = Vec::with_capacity(output_size);
    for output_index in 0..output_size {
        let Some((source_index, _distance)) = info.first_source_index_for_output(output_index)?
        else {
            values.push(fill_value);
            mask_flags.push(missing_policy.masks_missing());
            continue;
        };
        let source_masked = src_mask
            .as_ref()
            .and_then(|m| m.is_masked(source_index))
            .unwrap_or(false);
        values.push(src_values[source_index]);
        mask_flags.push(source_masked);
    }

    build_sampled_grid(output_shape, values, mask_flags)
}

fn nearest_area_pixels(
    source: &AreaDefinition,
    x: f64,
    y: f64,
    radius_of_influence: Option<f64>,
    neighbours: usize,
) -> Vec<(usize, f64)> {
    if neighbours == 1 {
        return nearest_area_pixel_fast(source, x, y, radius_of_influence)
            .into_iter()
            .collect();
    }
    let mut candidates = Vec::with_capacity(source.height() * source.width());
    for (source_index, (source_x, source_y)) in source.iter_projection_coords().enumerate() {
        let dx = source_x - x;
        let dy = source_y - y;
        let distance = (dx * dx + dy * dy).sqrt();
        if radius_of_influence.is_none_or(|radius| distance <= radius) {
            candidates.push((source_index, distance));
        }
    }
    candidates.sort_by(|left, right| {
        left.1
            .total_cmp(&right.1)
            .then_with(|| left.0.cmp(&right.0))
    });
    candidates.truncate(neighbours);
    candidates
}

fn nearest_area_pixel_fast(
    source: &AreaDefinition,
    x: f64,
    y: f64,
    radius_of_influence: Option<f64>,
) -> Option<(usize, f64)> {
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
    let distance = (dx * dx + dy * dy).sqrt();
    if radius_of_influence.is_some_and(|radius| distance > radius) {
        None
    } else {
        Some((src_y * source.width() + src_x, distance))
    }
}

fn validate_sampling_geometry(
    source_len: usize,
    output_shape: (usize, usize),
    info: &NeighbourInfo,
) -> Result<usize> {
    if source_len != info.source_size() {
        return Err(RustySatError::invalid_input(format!(
            "source grid has {source_len} values but neighbour info source size is {}",
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
    Ok(output_size)
}

fn weighted_value_for_output(
    output_index: usize,
    info: &NeighbourInfo,
    fill_value: f64,
    weight_fn: &impl Fn(f64) -> f64,
    source_value: impl Fn(usize) -> (f64, bool),
) -> Result<Option<f64>> {
    let Some(neighbours) = info.neighbours_for_output(output_index)? else {
        return Ok(None);
    };
    let mut weighted_sum = 0.0;
    let mut norm = 0.0;
    for neighbour in neighbours.into_iter().flatten() {
        let Some(source_index) =
            info.source_index_for_reduced_index(neighbour.reduced_source_index())?
        else {
            continue;
        };
        let (value, masked) = source_value(source_index);
        if masked || !value.is_finite() || (fill_value.is_finite() && value == fill_value) {
            continue;
        }
        let weight = weight_fn(neighbour.distance());
        if !weight.is_finite() || weight <= 0.0 {
            continue;
        }
        weighted_sum += value * weight;
        norm += weight;
    }
    if norm > 0.0 {
        Ok(Some(weighted_sum / norm))
    } else {
        Ok(None)
    }
}

fn build_sampled_grid(
    output_shape: (usize, usize),
    values: Vec<f64>,
    mask_flags: Vec<bool>,
) -> Result<DataGrid> {
    let has_mask = mask_flags.iter().any(|masked| *masked);
    let grid = DataGrid::new(output_shape.0, output_shape.1, values)?;
    if has_mask {
        grid.with_mask(ValidityMask::from_masked_flags(mask_flags))
    } else {
        Ok(grid)
    }
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
    fn area_neighbour_info_can_return_multiple_neighbours() {
        let source = area("source", 1, 3, [0.0, 0.0, 3.0, 1.0]);
        let target = area("target", 1, 1, [1.0, 0.0, 2.0, 1.0]);

        let info = get_area_neighbour_info_with_neighbours(&source, &target, None, 3).unwrap();

        assert_eq!(info.neighbours(), 3);
        assert_eq!(info.index_array(), &[1, 0, 2]);
        assert_eq!(info.distance_array(), &[0.0, 1.0, 1.0]);
        assert_eq!(
            info.neighbours_for_output(0).unwrap().unwrap(),
            vec![
                Some(Neighbour {
                    reduced_source_index: 1,
                    distance: 0.0
                }),
                Some(Neighbour {
                    reduced_source_index: 0,
                    distance: 1.0
                }),
                Some(Neighbour {
                    reduced_source_index: 2,
                    distance: 1.0
                })
            ]
        );
    }

    #[test]
    fn area_neighbour_info_pads_missing_multi_neighbours() {
        let source = area("source", 1, 2, [0.0, 0.0, 2.0, 1.0]);
        let target = area("target", 1, 1, [0.0, 0.0, 1.0, 1.0]);

        let info =
            get_area_neighbour_info_with_neighbours(&source, &target, Some(0.25), 3).unwrap();

        assert_eq!(info.index_array(), &[0, 2, 2]);
        assert_eq!(info.distance_array()[0], 0.0);
        assert!(info.distance_array()[1].is_infinite());
        assert!(info.distance_array()[2].is_infinite());
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

    #[test]
    fn owned_sampling_produces_same_output_as_borrowed() {
        let source = area("source", 2, 2, [0.0, 0.0, 2.0, 2.0]);
        let target = area("target", 2, 2, [0.0, 0.0, 2.0, 2.0]);
        let borrowed_grid = DataGrid::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let owned_grid = DataGrid::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let info = get_area_neighbour_info(&source, &target, None).unwrap();

        let borrowed = sample_nearest_from_neighbour_info(
            &borrowed_grid,
            target.shape(),
            &info,
            -999.0,
            SampleMissingPolicy::FillValue,
        )
        .unwrap();
        let owned = sample_nearest_from_neighbour_info_owned(
            owned_grid,
            target.shape(),
            &info,
            -999.0,
            SampleMissingPolicy::FillValue,
        )
        .unwrap();

        assert_eq!(borrowed.values(), owned.values());
        assert_eq!(borrowed.mask(), owned.mask());
    }

    #[test]
    fn owned_sampling_propagates_source_mask() {
        let source = area("source", 1, 2, [0.0, 0.0, 2.0, 1.0]);
        let target = area("target", 1, 2, [0.0, 0.0, 2.0, 1.0]);
        let source_grid = DataGrid::new(1, 2, vec![1.0, 2.0])
            .unwrap()
            .with_mask(ValidityMask::from_masked_flags([false, true]))
            .unwrap();
        let info = get_area_neighbour_info(&source, &target, None).unwrap();

        let result = sample_nearest_from_neighbour_info_owned(
            source_grid,
            target.shape(),
            &info,
            -999.0,
            SampleMissingPolicy::FillValue,
        )
        .unwrap();

        assert_eq!(result.values(), &[1.0, 2.0]);
        assert_eq!(result.mask().unwrap().masked_count(), 1);
        assert_eq!(result.is_masked(1), Some(true));
    }

    #[test]
    fn weighted_sampling_averages_multiple_neighbours() {
        let source = area("source", 1, 3, [0.0, 0.0, 3.0, 1.0]);
        let target = area("target", 1, 1, [1.0, 0.0, 2.0, 1.0]);
        let source_grid = DataGrid::new(1, 3, vec![10.0, 20.0, 30.0]).unwrap();
        let info = get_area_neighbour_info_with_neighbours(&source, &target, None, 3).unwrap();

        let sampled = sample_weighted_from_neighbour_info(
            &source_grid,
            target.shape(),
            &info,
            -999.0,
            SampleMissingPolicy::FillValue,
            |_| 1.0,
        )
        .unwrap();

        assert_eq!(sampled.values(), &[20.0]);
        assert!(sampled.mask().is_none());
    }

    #[test]
    fn weighted_sampling_skips_source_masks_and_can_mask_missing_output() {
        let source = area("source", 1, 2, [0.0, 0.0, 2.0, 1.0]);
        let target = area("target", 1, 1, [0.0, 0.0, 1.0, 1.0]);
        let source_grid = DataGrid::new(1, 2, vec![10.0, 20.0])
            .unwrap()
            .with_mask(ValidityMask::from_masked_flags([true, false]))
            .unwrap();
        let info =
            get_area_neighbour_info_with_neighbours(&source, &target, Some(0.25), 2).unwrap();

        let sampled = sample_weighted_from_neighbour_info(
            &source_grid,
            target.shape(),
            &info,
            -999.0,
            SampleMissingPolicy::Mask,
            |_| 1.0,
        )
        .unwrap();

        assert_eq!(sampled.values(), &[-999.0]);
        assert_eq!(sampled.mask().unwrap().is_masked(0), Some(true));
    }

    #[test]
    fn weighted_sampling_owned_matches_borrowed() {
        let source = area("source", 1, 2, [0.0, 0.0, 2.0, 1.0]);
        let target = area("target", 1, 1, [0.0, 0.0, 2.0, 1.0]);
        let source_grid = DataGrid::new(1, 2, vec![10.0, 20.0]).unwrap();
        let info = get_area_neighbour_info_with_neighbours(&source, &target, None, 2).unwrap();

        let borrowed = sample_weighted_from_neighbour_info(
            &source_grid,
            target.shape(),
            &info,
            -999.0,
            SampleMissingPolicy::FillValue,
            |distance| gaussian_weight(distance, 2.0).unwrap(),
        )
        .unwrap();
        let owned = sample_weighted_from_neighbour_info_owned(
            source_grid,
            target.shape(),
            &info,
            -999.0,
            SampleMissingPolicy::FillValue,
            |distance| gaussian_weight(distance, 2.0).unwrap(),
        )
        .unwrap();

        assert_eq!(borrowed.values(), owned.values());
    }

    #[test]
    fn gaussian_weight_validates_inputs() {
        assert_eq!(gaussian_weight(0.0, 2.0).unwrap(), 1.0);
        assert!(gaussian_weight(-1.0, 2.0).is_err());
        assert!(gaussian_weight(1.0, 0.0).is_err());
    }

    #[test]
    fn area_neighbour_info_rejects_zero_neighbours() {
        let source = area("source", 1, 1, [0.0, 0.0, 1.0, 1.0]);
        let target = area("target", 1, 1, [0.0, 0.0, 1.0, 1.0]);

        assert!(get_area_neighbour_info_with_neighbours(&source, &target, None, 0).is_err());
    }

    #[test]
    fn weighted_sampling_works_with_single_neighbour() {
        let source = area("source", 1, 2, [0.0, 0.0, 2.0, 1.0]);
        let target = area("target", 1, 1, [0.0, 0.0, 1.0, 1.0]);
        let source_grid = DataGrid::new(1, 2, vec![10.0, 20.0]).unwrap();
        let info = get_area_neighbour_info(&source, &target, None).unwrap();

        let sampled = sample_weighted_from_neighbour_info(
            &source_grid,
            target.shape(),
            &info,
            -999.0,
            SampleMissingPolicy::FillValue,
            |_| 1.0,
        )
        .unwrap();

        assert_eq!(sampled.values(), &[10.0]);
    }

    #[test]
    fn weighted_sampling_owned_propagates_source_mask() {
        let source = area("source", 1, 2, [0.0, 0.0, 2.0, 1.0]);
        let target = area("target", 1, 1, [0.0, 0.0, 1.0, 1.0]);
        let source_grid = DataGrid::new(1, 2, vec![10.0, 20.0])
            .unwrap()
            .with_mask(ValidityMask::from_masked_flags([false, true]))
            .unwrap();
        let info = get_area_neighbour_info(&source, &target, None).unwrap();

        let result = sample_weighted_from_neighbour_info_owned(
            source_grid,
            target.shape(),
            &info,
            -999.0,
            SampleMissingPolicy::FillValue,
            |_| 1.0,
        )
        .unwrap();

        assert_eq!(result.values(), &[10.0]);
        assert!(result.mask().is_none());
    }
}
