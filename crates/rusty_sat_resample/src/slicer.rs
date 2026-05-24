//! Area slicing helpers for Satpy/Pyresample-style data reduction.
//!
//! Reference behavior inspected before implementation:
//! - `deps/pyresample/pyresample/future/geometry/_subset.py`
//! - `deps/pyresample/pyresample/slicer.py`
//! - `deps/pyresample/pyresample/resampler.py`
//! - `satpy/satpy/scene.py::_reduce_data`
//!
//! This module intentionally starts with the same-projection `AreaDefinition`
//! path used by Satpy's common data-reduction flow. Cross-projection polygon
//! slicing remains a later S7 task because Rusty Sat does not have a real
//! projection transform backend yet.

use crate::AreaDefinition;
use rusty_sat_core::{Result, RustySatError};
use std::ops::Range;

const ROUND_HALF_EVEN_EPSILON: f64 = 1.0e-10;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AreaSlice {
    x: Range<usize>,
    y: Range<usize>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AreaCrop {
    source_area: AreaDefinition,
    slices: AreaSlice,
}

impl AreaSlice {
    pub fn new(x: Range<usize>, y: Range<usize>) -> Result<Self> {
        if x.start >= x.end || y.start >= y.end {
            return Err(RustySatError::invalid_input(
                "area slices must have non-empty x and y ranges",
            ));
        }
        Ok(Self { x, y })
    }

    pub fn x(&self) -> Range<usize> {
        self.x.clone()
    }

    pub fn y(&self) -> Range<usize> {
        self.y.clone()
    }

    pub fn width(&self) -> usize {
        self.x.end - self.x.start
    }

    pub fn height(&self) -> usize {
        self.y.end - self.y.start
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.height(), self.width())
    }
}

impl AreaCrop {
    pub fn source_area(&self) -> &AreaDefinition {
        &self.source_area
    }

    pub fn slices(&self) -> &AreaSlice {
        &self.slices
    }

    pub fn into_parts(self) -> (AreaDefinition, AreaSlice) {
        (self.source_area, self.slices)
    }
}

pub fn get_area_slices(source: &AreaDefinition, target: &AreaDefinition) -> Result<AreaSlice> {
    get_area_slices_with_divisibility(source, target, None)
}

pub fn get_area_slices_with_divisibility(
    source: &AreaDefinition,
    target: &AreaDefinition,
    shape_divisible_by: Option<usize>,
) -> Result<AreaSlice> {
    ensure_same_projection(source, target)?;

    let [target_llx, target_lly, target_urx, target_ury] = target.area_extent();
    if !target_llx.is_finite()
        || !target_lly.is_finite()
        || !target_urx.is_finite()
        || !target_ury.is_finite()
    {
        return Err(RustySatError::invalid_input(
            "target area extent must be finite",
        ));
    }

    let (x0, y0) = array_coordinates_from_projection(source, target_llx, target_lly)?;
    let (x1, y1) = array_coordinates_from_projection(source, target_urx, target_ury)?;
    let mut x = bounded_range(
        py_round_half_even(x0)?,
        py_round_half_even(x1)?.saturating_add(1),
        source.width(),
    )?;
    let mut y = bounded_range(
        py_round_half_even(y1)?,
        py_round_half_even(y0)?.saturating_add(1),
        source.height(),
    )?;

    if let Some(factor) = shape_divisible_by {
        if factor == 0 {
            return Err(RustySatError::invalid_input(
                "shape_divisible_by must be non-zero",
            ));
        }
        x = make_range_divisible(x, source.width(), factor)?;
        y = make_range_divisible(y, source.height(), factor)?;
    }

    AreaSlice::new(x, y)
}

pub fn slice_area(source: &AreaDefinition, slices: &AreaSlice) -> Result<AreaDefinition> {
    if slices.x.end > source.width() || slices.y.end > source.height() {
        return Err(RustySatError::invalid_input(
            "area slices exceed source shape",
        ));
    }
    let [llx, _lly, _urx, ury] = source.area_extent();
    let (pixel_size_x, pixel_size_y) = source.pixel_size();
    let area_extent = [
        llx + slices.x.start as f64 * pixel_size_x,
        ury - slices.y.end as f64 * pixel_size_y,
        llx + slices.x.end as f64 * pixel_size_x,
        ury - slices.y.start as f64 * pixel_size_y,
    ];
    AreaDefinition::from_parts(
        source.id().to_string(),
        source.description().to_string(),
        source.proj_id().to_string(),
        source.projection().clone(),
        slices.height(),
        slices.width(),
        area_extent,
    )
}

pub fn crop_source_area(source: &AreaDefinition, target: &AreaDefinition) -> Result<AreaCrop> {
    let slices = get_area_slices(source, target)?;
    let source_area = slice_area(source, &slices)?;
    Ok(AreaCrop {
        source_area,
        slices,
    })
}

fn ensure_same_projection(source: &AreaDefinition, target: &AreaDefinition) -> Result<()> {
    let source_crs = source.crs()?;
    let target_crs = target.crs()?;
    if source_crs.equivalent_to(&target_crs) {
        return Ok(());
    }
    Err(RustySatError::unsupported(
        "cross-projection area slicing before transform backend",
    ))
}

fn array_coordinates_from_projection(
    source: &AreaDefinition,
    x: f64,
    y: f64,
) -> Result<(f64, f64)> {
    let [llx, _lly, _urx, ury] = source.area_extent();
    let (pixel_size_x, pixel_size_y) = source.pixel_size();
    if pixel_size_x <= 0.0 || pixel_size_y <= 0.0 {
        return Err(RustySatError::invalid_input(
            "source pixel sizes must be positive",
        ));
    }
    let upper_left_pixel_x = llx + 0.5 * pixel_size_x;
    let upper_left_pixel_y = ury - 0.5 * pixel_size_y;
    let x_coord = (x - upper_left_pixel_x) / pixel_size_x;
    let y_coord = (y - upper_left_pixel_y) / -pixel_size_y;
    if !x_coord.is_finite() || !y_coord.is_finite() {
        return Err(RustySatError::invalid_input(
            "slice coordinates must be finite",
        ));
    }
    Ok((x_coord, y_coord))
}

fn py_round_half_even(value: f64) -> Result<isize> {
    if !value.is_finite() || value < i64::MIN as f64 || value > i64::MAX as f64 {
        return Err(RustySatError::invalid_input(
            "slice coordinate is outside finite i64 range",
        ));
    }
    let floor = value.floor();
    let rounded = if (value - floor - 0.5).abs() <= ROUND_HALF_EVEN_EPSILON {
        let floor_i64 = floor as i64;
        if floor_i64 % 2 == 0 {
            floor_i64
        } else {
            floor_i64 + 1
        }
    } else {
        value.round() as i64
    };
    isize::try_from(rounded)
        .map_err(|err| RustySatError::invalid_input(format!("slice index out of range: {err}")))
}

fn bounded_range(start: isize, stop: isize, max_size: usize) -> Result<Range<usize>> {
    let max_size = isize::try_from(max_size)
        .map_err(|err| RustySatError::invalid_input(format!("area size out of range: {err}")))?;
    let start = start.max(0) as usize;
    let stop = stop.clamp(0, max_size) as usize;
    if start >= stop {
        return Err(RustySatError::invalid_input(
            "areas do not overlap enough to produce a slice",
        ));
    }
    Ok(start..stop)
}

fn make_range_divisible(
    mut range: Range<usize>,
    max_size: usize,
    factor: usize,
) -> Result<Range<usize>> {
    let len = range.end - range.start;
    let rem = len % factor;
    if rem == 0 {
        return Ok(range);
    }

    let adjustment = factor - rem;
    if range.end.saturating_add(1).saturating_add(rem) < max_size {
        range.end = range.end.saturating_add(adjustment);
    } else if range.start > 0 {
        range.start = range.start.saturating_sub(adjustment);
    } else {
        range.end = range.end.saturating_sub(rem);
    }
    if range.start >= range.end {
        return Err(RustySatError::invalid_input(
            "shape_divisible_by produced an empty slice",
        ));
    }
    Ok(range)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn lonlat_area(id: &str, height: usize, width: usize, extent: [f64; 4]) -> AreaDefinition {
        AreaDefinition::from_parts(
            id,
            id,
            "longlat",
            BTreeMap::from([("proj".to_string(), "longlat".to_string())]),
            height,
            width,
            extent,
        )
        .unwrap()
    }

    #[test]
    fn computes_same_projection_area_slices_with_pyresample_style_margin() {
        let source = lonlat_area("source", 4, 4, [0.0, 0.0, 4.0, 4.0]);
        let target = lonlat_area("target", 2, 2, [1.0, 1.0, 3.0, 3.0]);

        let slices = get_area_slices(&source, &target).unwrap();

        assert_eq!(slices.x(), 0..3);
        assert_eq!(slices.y(), 0..3);
        assert_eq!(slices.shape(), (3, 3));
    }

    #[test]
    fn slices_area_extent_and_shape() {
        let source = lonlat_area("source", 4, 4, [0.0, 0.0, 4.0, 4.0]);
        let slices = AreaSlice::new(1..3, 1..4).unwrap();

        let cropped = slice_area(&source, &slices).unwrap();

        assert_eq!(cropped.shape(), (3, 2));
        assert_eq!(cropped.area_extent(), [1.0, 0.0, 3.0, 3.0]);
    }

    #[test]
    fn crop_source_area_returns_slices_and_cropped_area() {
        let source = lonlat_area("source", 4, 4, [0.0, 0.0, 4.0, 4.0]);
        let target = lonlat_area("target", 2, 2, [1.1, 1.1, 2.9, 2.9]);

        let crop = crop_source_area(&source, &target).unwrap();

        assert_eq!(crop.slices().x(), 1..3);
        assert_eq!(crop.slices().y(), 1..3);
        assert_eq!(crop.source_area().shape(), (2, 2));
        assert_eq!(crop.source_area().area_extent(), [1.0, 1.0, 3.0, 3.0]);
    }

    #[test]
    fn rejects_non_overlapping_area() {
        let source = lonlat_area("source", 4, 4, [0.0, 0.0, 4.0, 4.0]);
        let target = lonlat_area("target", 1, 1, [10.0, 10.0, 11.0, 11.0]);

        assert!(get_area_slices(&source, &target).is_err());
    }

    #[test]
    fn can_expand_slice_to_divisible_shape() {
        let source = lonlat_area("source", 8, 8, [0.0, 0.0, 8.0, 8.0]);
        let target = lonlat_area("target", 3, 3, [2.1, 2.1, 4.9, 4.9]);

        let slices = get_area_slices_with_divisibility(&source, &target, Some(4)).unwrap();

        assert_eq!(slices.shape(), (4, 4));
    }

    #[test]
    fn rejects_cross_projection_until_transform_backend_exists() {
        let source = lonlat_area("source", 4, 4, [0.0, 0.0, 4.0, 4.0]);
        let target = AreaDefinition::from_parts(
            "target",
            "target",
            "merc",
            BTreeMap::from([("proj".to_string(), "merc".to_string())]),
            4,
            4,
            [0.0, 0.0, 4.0, 4.0],
        )
        .unwrap();

        assert!(matches!(
            get_area_slices(&source, &target),
            Err(RustySatError::Unsupported { .. })
        ));
    }

    #[test]
    fn area_slice_rejects_empty_ranges() {
        assert!(AreaSlice::new(1..1, 0..2).is_err());
        assert!(AreaSlice::new(0..2, 1..1).is_err());
    }

    #[test]
    fn slices_clamp_when_target_straddles_source_edge() {
        let source = lonlat_area("source", 4, 4, [0.0, 0.0, 4.0, 4.0]);
        // Target extends from inside source to well outside it.
        let target = lonlat_area("straddle", 2, 2, [1.0, 1.0, 6.0, 6.0]);

        let slices = get_area_slices(&source, &target).unwrap();

        // x stop clamps to source width; y starts from the lower-left corner.
        assert_eq!(slices.x().end, 4);
        assert_eq!(slices.y().start, 0);
    }
}
