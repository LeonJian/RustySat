//! Pyresample-style coarse data-reduction helpers.
//!
//! Reference behavior inspected before implementation:
//! - `deps/pyresample/pyresample/data_reduce.py`
//! - `deps/pyresample/pyresample/test/test_data_reduce.py`
//!
//! This first S7 data-reduction slice exposes lon/lat-grid and boundary based
//! validity filtering. It does not slice datasets yet; Scene-level slicing and
//! crop integration remain later S7 work.

use rusty_sat_core::{Result, RustySatError};

const EARTH_RADIUS_METERS: f64 = 6_370_997.0;

#[derive(Debug, Clone, PartialEq)]
pub struct LonLatBoundaries {
    side1: Vec<f64>,
    side2: Vec<f64>,
    side3: Vec<f64>,
    side4: Vec<f64>,
}

impl LonLatBoundaries {
    pub fn new(
        side1: impl Into<Vec<f64>>,
        side2: impl Into<Vec<f64>>,
        side3: impl Into<Vec<f64>>,
        side4: impl Into<Vec<f64>>,
    ) -> Result<Self> {
        let boundaries = Self {
            side1: side1.into(),
            side2: side2.into(),
            side3: side3.into(),
            side4: side4.into(),
        };
        boundaries.validate()?;
        Ok(boundaries)
    }

    pub fn side1(&self) -> &[f64] {
        &self.side1
    }

    pub fn side2(&self) -> &[f64] {
        &self.side2
    }

    pub fn side3(&self) -> &[f64] {
        &self.side3
    }

    pub fn side4(&self) -> &[f64] {
        &self.side4
    }

    fn validate(&self) -> Result<()> {
        for side in [&self.side1, &self.side2, &self.side3, &self.side4] {
            if side.is_empty() {
                return Err(RustySatError::invalid_input(
                    "lon/lat boundary sides must not be empty",
                ));
            }
            if side.iter().any(|value| !value.is_finite()) {
                return Err(RustySatError::invalid_input(
                    "lon/lat boundary values must be finite",
                ));
            }
        }
        Ok(())
    }
}

pub fn lonlat_grid_boundaries(
    height: usize,
    width: usize,
    values: &[f64],
) -> Result<LonLatBoundaries> {
    validate_grid_shape(height, width, values.len())?;

    let side1 = values[..width].to_vec();
    let side2 = (0..height)
        .map(|y| values[y * width + width - 1])
        .collect::<Vec<_>>();
    let side3 = values[(height - 1) * width..height * width]
        .iter()
        .rev()
        .copied()
        .collect::<Vec<_>>();
    let side4 = (0..height)
        .rev()
        .map(|y| values[y * width])
        .collect::<Vec<_>>();

    LonLatBoundaries::new(side1, side2, side3, side4)
}

pub fn get_valid_index_from_lonlat_grid(
    height: usize,
    width: usize,
    grid_lons: &[f64],
    grid_lats: &[f64],
    lons: &[f64],
    lats: &[f64],
    radius_of_influence: f64,
) -> Result<Vec<bool>> {
    validate_grid_shape(height, width, grid_lons.len())?;
    validate_grid_shape(height, width, grid_lats.len())?;
    let boundary_lons = lonlat_grid_boundaries(height, width, grid_lons)?;
    let boundary_lats = lat_grid_boundaries(height, width, grid_lats)?;
    get_valid_index_from_lonlat_boundaries(
        &boundary_lons,
        &boundary_lats,
        lons,
        lats,
        radius_of_influence,
    )
}

pub fn get_valid_index_from_lonlat_boundaries(
    boundary_lons: &LonLatBoundaries,
    boundary_lats: &LonLatBoundaries,
    lons: &[f64],
    lats: &[f64],
    radius_of_influence: f64,
) -> Result<Vec<bool>> {
    validate_points(lons, lats)?;
    validate_radius(radius_of_influence)?;
    boundary_lons.validate()?;
    boundary_lats.validate()?;
    Ok(valid_index_from_boundaries(
        boundary_lons,
        boundary_lats,
        lons,
        lats,
        radius_of_influence,
    ))
}

fn lat_grid_boundaries(height: usize, width: usize, values: &[f64]) -> Result<LonLatBoundaries> {
    validate_grid_shape(height, width, values.len())?;

    let side1 = values[..width].to_vec();
    let side2 = (0..height)
        .map(|y| values[y * width + width - 1])
        .collect::<Vec<_>>();
    let side3 = values[(height - 1) * width..height * width].to_vec();
    let side4 = (0..height).map(|y| values[y * width]).collect::<Vec<_>>();

    LonLatBoundaries::new(side1, side2, side3, side4)
}

fn valid_index_from_boundaries(
    boundary_lons: &LonLatBoundaries,
    boundary_lats: &LonLatBoundaries,
    lons: &[f64],
    lats: &[f64],
    radius_of_influence: f64,
) -> Vec<bool> {
    if has_illegal_lonlat(boundary_lons, boundary_lats) {
        return vec![true; lons.len()];
    }

    let angle_sum = boundary_lons
        .sides()
        .iter()
        .map(|side| side_angle_sum(side))
        .sum::<f64>();
    let lat_min = boundary_lats
        .sides()
        .iter()
        .flat_map(|side| side.iter().copied())
        .fold(f64::INFINITY, f64::min);
    let lat_max = boundary_lats
        .sides()
        .iter()
        .flat_map(|side| side.iter().copied())
        .fold(f64::NEG_INFINITY, f64::max);
    let lat_buffer = (radius_of_influence / EARTH_RADIUS_METERS).to_degrees();
    let lat_min_buffered = lat_min - lat_buffer;
    let lat_max_buffered = lat_max + lat_buffer;

    match angle_sum.round() as i64 {
        -360 => lats
            .iter()
            .map(|lat| lat.is_finite() && *lat >= lat_min_buffered)
            .collect(),
        360 => lats
            .iter()
            .map(|lat| lat.is_finite() && *lat <= lat_max_buffered)
            .collect(),
        0 => valid_no_pole_index(
            boundary_lons,
            boundary_lats,
            lons,
            lats,
            radius_of_influence,
            lat_min_buffered,
            lat_max_buffered,
        ),
        _ => vec![true; lons.len()],
    }
}

fn valid_no_pole_index(
    boundary_lons: &LonLatBoundaries,
    boundary_lats: &LonLatBoundaries,
    lons: &[f64],
    lats: &[f64],
    radius_of_influence: f64,
    lat_min_buffered: f64,
    lat_max_buffered: f64,
) -> Vec<bool> {
    let lon_min_buffered =
        boundary_lons.side4_min() - lon_buffer_degrees(boundary_lats.side4(), radius_of_influence);
    let lon_max_buffered =
        boundary_lons.side2_max() + lon_buffer_degrees(boundary_lats.side2(), radius_of_influence);
    let crosses_date_line = boundary_lons.side2_min() <= boundary_lons.side4_max();

    lons.iter()
        .zip(lats)
        .map(|(lon, lat)| {
            if !lon.is_finite() || !lat.is_finite() {
                return false;
            }
            let valid_lat = *lat >= lat_min_buffered && *lat <= lat_max_buffered;
            let valid_lon = if crosses_date_line {
                (*lon >= lon_min_buffered && *lon <= 180.0)
                    || (*lon <= lon_max_buffered && *lon >= -180.0)
            } else {
                *lon >= lon_min_buffered && *lon <= lon_max_buffered
            };
            valid_lat && valid_lon
        })
        .collect()
}

fn lon_buffer_degrees(boundary_lats: &[f64], radius_of_influence: f64) -> f64 {
    if radius_of_influence == 0.0 {
        return 0.0;
    }
    let max_angle = boundary_lats
        .iter()
        .map(|lat| lat.abs())
        .fold(0.0_f64, f64::max);
    let denominator = max_angle.to_radians().sin() * EARTH_RADIUS_METERS;
    if denominator.abs() <= f64::EPSILON {
        f64::INFINITY
    } else {
        (radius_of_influence / denominator).to_degrees()
    }
}

fn side_angle_sum(side: &[f64]) -> f64 {
    side.windows(2)
        .map(|window| {
            let mut delta = window[1] - window[0];
            if delta.abs() > 180.0 {
                delta = (delta.abs() - 360.0) * delta.signum();
            }
            delta
        })
        .sum()
}

fn has_illegal_lonlat(boundary_lons: &LonLatBoundaries, boundary_lats: &LonLatBoundaries) -> bool {
    boundary_lons
        .sides()
        .iter()
        .flat_map(|side| side.iter())
        .any(|lon| *lon < -180.0 || *lon > 180.0)
        || boundary_lats
            .sides()
            .iter()
            .flat_map(|side| side.iter())
            .any(|lat| *lat < -90.0 || *lat > 90.0)
}

impl LonLatBoundaries {
    fn sides(&self) -> [&[f64]; 4] {
        [&self.side1, &self.side2, &self.side3, &self.side4]
    }

    fn side2_min(&self) -> f64 {
        min_value(&self.side2)
    }

    fn side2_max(&self) -> f64 {
        max_value(&self.side2)
    }

    fn side4_min(&self) -> f64 {
        min_value(&self.side4)
    }

    fn side4_max(&self) -> f64 {
        max_value(&self.side4)
    }
}

fn min_value(values: &[f64]) -> f64 {
    values.iter().copied().fold(f64::INFINITY, f64::min)
}

fn max_value(values: &[f64]) -> f64 {
    values.iter().copied().fold(f64::NEG_INFINITY, f64::max)
}

fn validate_grid_shape(height: usize, width: usize, len: usize) -> Result<()> {
    if height == 0 || width == 0 {
        return Err(RustySatError::invalid_input(
            "lon/lat grid dimensions must be non-zero",
        ));
    }
    let expected = height
        .checked_mul(width)
        .ok_or_else(|| RustySatError::invalid_input("lon/lat grid size overflows usize"))?;
    if len != expected {
        return Err(RustySatError::invalid_input(format!(
            "lon/lat grid length {len} does not match shape {height}x{width}",
        )));
    }
    Ok(())
}

fn validate_points(lons: &[f64], lats: &[f64]) -> Result<()> {
    if lons.len() != lats.len() {
        return Err(RustySatError::invalid_input(format!(
            "longitude length {} does not match latitude length {}",
            lons.len(),
            lats.len()
        )));
    }
    Ok(())
}

fn validate_radius(radius_of_influence: f64) -> Result<()> {
    if !radius_of_influence.is_finite() || radius_of_influence < 0.0 {
        return Err(RustySatError::invalid_input(
            "radius_of_influence must be finite and non-negative",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    fn square_boundary() -> (LonLatBoundaries, LonLatBoundaries) {
        (
            LonLatBoundaries::new([0.0, 10.0], [10.0, 10.0], [10.0, 0.0], [0.0, 0.0]).unwrap(),
            LonLatBoundaries::new([10.0, 10.0], [10.0, 0.0], [0.0, 0.0], [0.0, 10.0]).unwrap(),
        )
    }

    #[test]
    fn lon_grid_boundaries_follow_pyresample_side_order() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let boundaries = lonlat_grid_boundaries(2, 3, &values).unwrap();

        assert_eq!(boundaries.side1(), &[1.0, 2.0, 3.0]);
        assert_eq!(boundaries.side2(), &[3.0, 6.0]);
        assert_eq!(boundaries.side3(), &[6.0, 5.0, 4.0]);
        assert_eq!(boundaries.side4(), &[4.0, 1.0]);
    }

    #[test]
    fn lat_grid_boundaries_follow_pyresample_side_order() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let boundaries = lat_grid_boundaries(2, 3, &values).unwrap();

        assert_eq!(boundaries.side1(), &[1.0, 2.0, 3.0]);
        assert_eq!(boundaries.side2(), &[3.0, 6.0]);
        assert_eq!(boundaries.side3(), &[4.0, 5.0, 6.0]);
        assert_eq!(boundaries.side4(), &[1.0, 4.0]);
    }

    #[test]
    fn valid_index_from_lonlat_boundaries_filters_square_region() {
        let (boundary_lons, boundary_lats) = square_boundary();
        let lons = vec![5.0, -1.0, 11.0, 5.0, f64::NAN];
        let lats = vec![5.0, 5.0, 5.0, 11.0, 5.0];

        let valid = get_valid_index_from_lonlat_boundaries(
            &boundary_lons,
            &boundary_lats,
            &lons,
            &lats,
            0.0,
        )
        .unwrap();

        assert_eq!(valid, vec![true, false, false, false, false]);
    }

    #[test]
    fn valid_index_from_lonlat_grid_matches_boundary_filter() {
        let grid_lons = vec![0.0, 10.0, 0.0, 10.0];
        let grid_lats = vec![10.0, 10.0, 0.0, 0.0];
        let lons = vec![5.0, 15.0];
        let lats = vec![5.0, 5.0];

        let valid =
            get_valid_index_from_lonlat_grid(2, 2, &grid_lons, &grid_lats, &lons, &lats, 0.0)
                .unwrap();

        assert_eq!(valid, vec![true, false]);
    }

    #[test]
    fn date_line_crossing_accepts_both_lon_segments() {
        let boundary_lons = LonLatBoundaries::new(
            [170.0, -170.0],
            [-170.0, -170.0],
            [-170.0, 170.0],
            [170.0, 170.0],
        )
        .unwrap();
        let boundary_lats =
            LonLatBoundaries::new([10.0, 10.0], [10.0, 0.0], [0.0, 0.0], [0.0, 10.0]).unwrap();
        let lons = vec![175.0, -175.0, 0.0];
        let lats = vec![5.0, 5.0, 5.0];

        let valid = get_valid_index_from_lonlat_boundaries(
            &boundary_lons,
            &boundary_lats,
            &lons,
            &lats,
            0.0,
        )
        .unwrap();

        assert_eq!(valid, vec![true, true, false]);
    }

    #[test]
    fn illegal_target_boundary_disables_reduction_like_pyresample() {
        let boundary_lons = LonLatBoundaries::new([200.0], [200.0], [200.0], [200.0]).unwrap();
        let boundary_lats = LonLatBoundaries::new([0.0], [0.0], [0.0], [0.0]).unwrap();
        let lons = vec![0.0, f64::NAN];
        let lats = vec![0.0, 0.0];

        let valid = get_valid_index_from_lonlat_boundaries(
            &boundary_lons,
            &boundary_lats,
            &lons,
            &lats,
            0.0,
        )
        .unwrap();

        assert_eq!(valid, vec![true, true]);
    }

    #[test]
    fn north_pole_area_filters_by_lat_min() {
        let boundary_lons =
            LonLatBoundaries::new([0.0, 90.0], [90.0, 90.0], [90.0, 0.0], [0.0, 0.0]).unwrap();
        let boundary_lats =
            LonLatBoundaries::new([80.0, 80.0], [80.0, 90.0], [90.0, 90.0], [90.0, 80.0]).unwrap();
        let lons = vec![45.0, 180.0];
        let lats = vec![85.0, 70.0];

        let valid = get_valid_index_from_lonlat_boundaries(
            &boundary_lons,
            &boundary_lats,
            &lons,
            &lats,
            0.0,
        )
        .unwrap();

        assert_eq!(valid, vec![true, false]);
    }

    #[test]
    fn radius_of_influence_expands_acceptance_region() {
        let (boundary_lons, boundary_lats) = square_boundary();
        let lons = vec![9.5, 10.2];
        let lats = vec![5.0, 5.0];

        let tight = get_valid_index_from_lonlat_boundaries(
            &boundary_lons,
            &boundary_lats,
            &lons,
            &lats,
            0.0,
        )
        .unwrap();
        let wide = get_valid_index_from_lonlat_boundaries(
            &boundary_lons,
            &boundary_lats,
            &lons,
            &lats,
            100_000.0,
        )
        .unwrap();

        assert_eq!(tight, vec![true, false]);
        assert_eq!(wide, vec![true, true]);
    }

    #[test]
    fn validate_grid_shape_rejects_zero_dimensions() {
        assert!(validate_grid_shape(0, 1, 0).is_err());
        assert!(validate_grid_shape(1, 0, 0).is_err());
    }
}
