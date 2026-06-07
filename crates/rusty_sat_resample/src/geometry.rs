//! Shared Pyresample-style geometry definition foundations.
//!
//! Reference behavior inspected before implementation:
//! - `deps/pyresample/pyresample/geometry.py`:
//!   `BaseDefinition`, `CoordinateDefinition`, and `GridDefinition`
//! - `deps/pyresample/docs/source/concepts/geometries.rst`

use rusty_sat_core::{Result, RustySatError};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeometryKind {
    Area,
    Swath,
    Coordinate,
    Grid,
}

pub trait GeometryDefinition {
    fn kind(&self) -> GeometryKind;

    fn shape(&self) -> Vec<usize>;

    fn ndim(&self) -> usize {
        self.shape().len()
    }

    fn size(&self) -> usize {
        self.shape().iter().product()
    }

    fn is_empty(&self) -> bool {
        self.size() == 0
    }
}

pub trait ProjectionDefinition: GeometryDefinition {
    fn width(&self) -> usize;

    fn height(&self) -> usize;

    fn area_extent(&self) -> [f64; 4];

    fn pixel_size(&self) -> (f64, f64);

    fn upper_left_extent(&self) -> (f64, f64) {
        let extent = self.area_extent();
        (extent[0], extent[3])
    }

    fn pixel_upper_left(&self) -> (f64, f64) {
        let (pixel_size_x, pixel_size_y) = self.pixel_size();
        let (upper_left_x, upper_left_y) = self.upper_left_extent();
        (
            upper_left_x + pixel_size_x / 2.0,
            upper_left_y - pixel_size_y / 2.0,
        )
    }

    fn pixel_offset(&self) -> (f64, f64) {
        let (pixel_size_x, pixel_size_y) = self.pixel_size();
        let extent = self.area_extent();
        (-extent[0] / pixel_size_x, extent[3] / pixel_size_y)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CoordinateDefinition {
    shape: Vec<usize>,
    lons: Vec<f64>,
    lats: Vec<f64>,
}

impl CoordinateDefinition {
    pub fn from_lonlats(
        shape: impl Into<Vec<usize>>,
        lons: Vec<f64>,
        lats: Vec<f64>,
    ) -> Result<Self> {
        let shape = shape.into();
        validate_shape(&shape, "coordinate definition")?;
        validate_lonlat_vectors(&shape, &lons, &lats, "coordinate definition")?;
        Ok(Self { shape, lons, lats })
    }

    pub fn lons(&self) -> &[f64] {
        &self.lons
    }

    pub fn lats(&self) -> &[f64] {
        &self.lats
    }

    pub fn into_lonlats(self) -> (Vec<f64>, Vec<f64>) {
        (self.lons, self.lats)
    }

    pub fn approximate_eq(&self, other: &Self, abs_tol: f64, rel_tol: f64) -> bool {
        self.shape == other.shape
            && coords_approx_eq(&self.lons, &other.lons, abs_tol, rel_tol)
            && coords_approx_eq(&self.lats, &other.lats, abs_tol, rel_tol)
    }
}

impl GeometryDefinition for CoordinateDefinition {
    fn kind(&self) -> GeometryKind {
        GeometryKind::Coordinate
    }

    fn shape(&self) -> Vec<usize> {
        self.shape.clone()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct GridDefinition {
    coordinates: CoordinateDefinition,
}

impl GridDefinition {
    pub fn from_lonlats(
        height: usize,
        width: usize,
        lons: Vec<f64>,
        lats: Vec<f64>,
    ) -> Result<Self> {
        let coordinates = CoordinateDefinition::from_lonlats(vec![height, width], lons, lats)?;
        Ok(Self { coordinates })
    }

    pub fn coordinates(&self) -> &CoordinateDefinition {
        &self.coordinates
    }

    pub fn lons(&self) -> &[f64] {
        self.coordinates.lons()
    }

    pub fn lats(&self) -> &[f64] {
        self.coordinates.lats()
    }
}

impl GeometryDefinition for GridDefinition {
    fn kind(&self) -> GeometryKind {
        GeometryKind::Grid
    }

    fn shape(&self) -> Vec<usize> {
        self.coordinates.shape()
    }
}

pub(crate) fn validate_shape(shape: &[usize], context: &str) -> Result<()> {
    if shape.is_empty() {
        return Err(RustySatError::invalid_input(format!(
            "{context} shape cannot be empty"
        )));
    }
    if shape.contains(&0) {
        return Err(RustySatError::invalid_input(format!(
            "{context} dimensions must be non-zero"
        )));
    }
    Ok(())
}

pub(crate) fn validate_lonlat_vectors(
    shape: &[usize],
    lons: &[f64],
    lats: &[f64],
    context: &str,
) -> Result<()> {
    if lons.len() != lats.len() {
        return Err(RustySatError::invalid_input(format!(
            "{context} lons and lats must have the same length"
        )));
    }
    let expected = shape.iter().try_fold(1usize, |acc, value| {
        acc.checked_mul(*value).ok_or_else(|| {
            RustySatError::invalid_input(format!("{context} shape size overflows usize"))
        })
    })?;
    if lons.len() != expected {
        return Err(RustySatError::invalid_input(format!(
            "{context} coordinate length {} does not match shape {:?}",
            lons.len(),
            shape
        )));
    }
    if lons
        .iter()
        .chain(lats.iter())
        .any(|value| !value.is_finite())
    {
        return Err(RustySatError::invalid_input(format!(
            "{context} coordinates must be finite"
        )));
    }
    Ok(())
}

fn coords_approx_eq(left: &[f64], right: &[f64], abs_tol: f64, rel_tol: f64) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right.iter())
            .all(|(left, right)| scalar_approx_eq(*left, *right, abs_tol, rel_tol))
}

fn scalar_approx_eq(left: f64, right: f64, abs_tol: f64, rel_tol: f64) -> bool {
    if left == right {
        return true;
    }
    let diff = (left - right).abs();
    diff <= abs_tol || diff <= rel_tol * left.abs().max(right.abs())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    #[test]
    fn coordinate_definition_validates_shape_and_coordinates() {
        let definition = CoordinateDefinition::from_lonlats(
            vec![2, 2],
            vec![0.0, 1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0, 7.0],
        )
        .unwrap();

        assert_eq!(definition.kind(), GeometryKind::Coordinate);
        assert_eq!(definition.shape(), vec![2, 2]);
        assert_eq!(definition.ndim(), 2);
        assert_eq!(definition.size(), 4);
    }

    #[test]
    fn coordinate_definition_rejects_mismatched_lengths() {
        let err = CoordinateDefinition::from_lonlats(vec![2, 2], vec![0.0, 1.0], vec![0.0, 1.0])
            .unwrap_err();

        assert!(err.to_string().contains("does not match shape"));
    }

    #[test]
    fn coordinate_definition_rejects_non_finite_coordinates() {
        let err =
            CoordinateDefinition::from_lonlats(vec![1], vec![f64::NAN], vec![35.0]).unwrap_err();

        assert!(err.to_string().contains("coordinates must be finite"));
    }

    #[test]
    fn grid_definition_is_two_dimensional_coordinate_definition() {
        let grid =
            GridDefinition::from_lonlats(1, 2, vec![140.0, 141.0], vec![35.0, 36.0]).unwrap();

        assert_eq!(grid.kind(), GeometryKind::Grid);
        assert_eq!(grid.shape(), vec![1, 2]);
        assert_eq!(grid.lons(), &[140.0, 141.0]);
    }

    #[test]
    fn coordinate_approximate_equality_matches_pyresample_tolerance_style() {
        let left = CoordinateDefinition::from_lonlats(vec![1], vec![140.0], vec![35.0]).unwrap();
        let right =
            CoordinateDefinition::from_lonlats(vec![1], vec![140.0 + 1.0e-7], vec![35.0]).unwrap();

        assert!(left.approximate_eq(&right, 1.0e-6, 5.0e-9));
        assert!(!left.approximate_eq(&right, 1.0e-9, 1.0e-12));
    }
}
