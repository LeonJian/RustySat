//! Day/night fill compositors — Satpy `fill.py` equivalents.
//!
//! Reference: `satpy/satpy/composites/fill.py` — `DayNightCompositor`.
//!
//! The [`DayNightCompositor`] blends a corrected (day) and an uncorrected
//! (night) band-major RGB composite with per-pixel solar-zenith weights:
//! `out = w * day + (1 - w) * night`. The weights are computed externally
//! (e.g. `rusty_sat_modifiers::sun_zenith::daynight_blend_weights`), keeping
//! this crate independent of the angle machinery.

use rusty_sat_core::{
    AnyDataArray, Coordinate, DataArray, DataId, Dataset, MetadataValue, Result, RustySatError,
    ValidityMask,
};

/// Blend two band-major RGB datasets by per-pixel day/night weights.
///
/// Equivalent of Satpy's `DayNightCompositor` for `day_night == "day_night"`:
///
/// ```text
/// out = w * day + (1 - w) * night      (per band, per pixel)
/// ```
///
/// `w` is 1 on the day side (SZA ≤ `lim_low`, e.g. the corrected composite)
/// and 0 on the night side (SZA ≥ `lim_high`, e.g. the uncorrected composite).
/// The weights are precomputed externally (for example with
/// `rusty_sat_modifiers::sun_zenith::daynight_blend_weights`).
#[derive(Debug, Clone, PartialEq)]
pub struct DayNightCompositor {
    name: String,
    lim_low: f64,
    lim_high: f64,
}

impl DayNightCompositor {
    /// Create a day/night blend compositor.
    ///
    /// `lim_low`/`lim_high` are the solar-zenith blend limits; they are stored
    /// as metadata and documented here, while the blend itself consumes the
    /// externally computed weights.
    pub fn new(name: impl Into<String>, lim_low: f64, lim_high: f64) -> Result<Self> {
        let name = name.into();
        if name.trim().is_empty() {
            return Err(RustySatError::invalid_input(
                "day/night compositor name cannot be empty",
            ));
        }
        if !lim_low.is_finite() || !lim_high.is_finite() || lim_low >= lim_high {
            return Err(RustySatError::invalid_input(format!(
                "day/night blend limits must satisfy lim_low < lim_high, got {lim_low} / {lim_high}"
            )));
        }
        Ok(Self {
            name,
            lim_low,
            lim_high,
        })
    }

    /// Lower solar-zenith blend limit (SZA below this uses `day`).
    pub fn lim_low(&self) -> f64 {
        self.lim_low
    }

    /// Upper solar-zenith blend limit (SZA above this uses `night`).
    pub fn lim_high(&self) -> f64 {
        self.lim_high
    }

    /// Blend two band-major `[3, y, x]` datasets with per-pixel weights.
    ///
    /// `day` is the corrected composite (weight 1 at SZA ≤ `lim_low`), `night`
    /// the uncorrected composite (weight 1 at SZA ≥ `lim_high`). `weights`
    /// holds one value per pixel (length `y * x`) in `[0, 1]`. A pixel whose
    /// corresponding band is masked in either input is masked in the output.
    pub fn compose(&self, day: &Dataset, night: &Dataset, weights: &[f32]) -> Result<Dataset> {
        let (height, width, pixel_count) = validate_inputs(day, night, weights.len())?;
        let day_array = day
            .array()
            .ok_or_else(|| missing_array_error(day.id().name()))?;
        let night_array = night
            .array()
            .ok_or_else(|| missing_array_error(night.id().name()))?;
        let day_values = values_as_f32(day_array);
        let night_values = values_as_f32(night_array);
        let values = blend_values(&day_values, &night_values, weights, pixel_count);
        let mask = or_masks(day_array.mask(), night_array.mask());
        finish_dataset(&self.name, height, width, values, mask)
    }

    /// Consuming variant of [`DayNightCompositor::compose`].
    pub fn compose_owned(self, day: Dataset, night: Dataset, weights: Vec<f32>) -> Result<Dataset> {
        let (height, width, pixel_count) = validate_inputs(&day, &night, weights.len())?;
        let day_name = day.id().name().to_string();
        let night_name = night.id().name().to_string();
        let day_array = day
            .into_array()
            .ok_or_else(|| missing_array_error(&day_name))?;
        let night_array = night
            .into_array()
            .ok_or_else(|| missing_array_error(&night_name))?;
        let day_mask = day_array.mask().cloned();
        let night_mask = night_array.mask().cloned();
        let day_values = owned_values_as_f32(day_array);
        let night_values = owned_values_as_f32(night_array);
        let values = blend_values(&day_values, &night_values, &weights, pixel_count);
        let mask = or_masks(day_mask.as_ref(), night_mask.as_ref());
        finish_dataset(&self.name, height, width, values, mask)
    }
}

fn missing_array_error(name: &str) -> RustySatError {
    RustySatError::invalid_input(format!("dataset '{name}' has no array data"))
}

fn validate_inputs(
    day: &Dataset,
    night: &Dataset,
    weight_count: usize,
) -> Result<(usize, usize, usize)> {
    let day_shape = day
        .array()
        .ok_or_else(|| missing_array_error(day.id().name()))?
        .shape()
        .to_vec();
    let night_shape = night
        .array()
        .ok_or_else(|| missing_array_error(night.id().name()))?
        .shape()
        .to_vec();
    if day_shape != night_shape {
        return Err(RustySatError::invalid_input(format!(
            "day and night composites must have identical shapes, got {day_shape:?} vs {night_shape:?}"
        )));
    }
    if day_shape.len() != 3 {
        return Err(RustySatError::invalid_input(format!(
            "day/night blend requires band-major [bands, y, x] datasets, got {:?}D",
            day_shape.len()
        )));
    }
    let height = day_shape[1];
    let width = day_shape[2];
    let pixel_count = height
        .checked_mul(width)
        .ok_or_else(|| RustySatError::invalid_input("day/night blend shape is too large"))?;
    if weight_count != pixel_count {
        return Err(RustySatError::invalid_input(format!(
            "day/night blend needs one weight per pixel ({pixel_count}), got {weight_count}"
        )));
    }
    Ok((height, width, pixel_count))
}

/// Per-pixel weighted blend `w * day + (1 - w) * night` over band-major values.
///
/// Values are promoted to f64 for the blend (matching Satpy's float blend) and
/// written back as f32. The band loop is rayon-parallel; each band is one
/// independent chunk so weights (y*x) are read once per band.
fn blend_values(day: &[f32], night: &[f32], weights: &[f32], pixel_count: usize) -> Vec<f32> {
    debug_assert_eq!(day.len(), night.len());
    let mut out = vec![0.0_f32; day.len()];
    use rayon::prelude::*;
    out.par_chunks_mut(pixel_count)
        .enumerate()
        .for_each(|(band, band_out)| {
            let day_band = &day[band * pixel_count..(band + 1) * pixel_count];
            let night_band = &night[band * pixel_count..(band + 1) * pixel_count];
            for (p, slot) in band_out.iter_mut().enumerate() {
                let w = f64::from(weights[p]);
                let value = w * f64::from(day_band[p]) + (1.0 - w) * f64::from(night_band[p]);
                *slot = value as f32;
            }
        });
    out
}

/// Per-band OR of two masks: a pixel is masked if either input masks it.
fn or_masks(
    day_mask: Option<&ValidityMask>,
    night_mask: Option<&ValidityMask>,
) -> Option<ValidityMask> {
    match (day_mask, night_mask) {
        (None, None) => None,
        (Some(mask), None) | (None, Some(mask)) => Some(mask.clone()),
        (Some(day), Some(night)) => {
            let mut out = day.clone();
            for i in 0..out.len() {
                if night.is_masked(i).unwrap_or(false) {
                    out.set_masked(i, true);
                }
            }
            Some(out)
        }
    }
}

fn values_as_f32(array: &AnyDataArray) -> Vec<f32> {
    match array {
        AnyDataArray::F32(a) => a.values().to_vec(),
        AnyDataArray::F64(a) => a.values().iter().map(|v| *v as f32).collect(),
        AnyDataArray::U8(a) => a.values().iter().map(|v| f32::from(*v)).collect(),
        AnyDataArray::U16(a) => a.values().iter().map(|v| f32::from(*v)).collect(),
        AnyDataArray::I16(a) => a.values().iter().map(|v| f32::from(*v)).collect(),
    }
}

fn owned_values_as_f32(array: AnyDataArray) -> Vec<f32> {
    match array {
        AnyDataArray::F32(a) => a.into_values(),
        AnyDataArray::F64(a) => a.into_values().into_iter().map(|v| v as f32).collect(),
        AnyDataArray::U8(a) => a.into_values().into_iter().map(f32::from).collect(),
        AnyDataArray::U16(a) => a.into_values().into_iter().map(f32::from).collect(),
        AnyDataArray::I16(a) => a.into_values().into_iter().map(f32::from).collect(),
    }
}

fn finish_dataset(
    name: &str,
    height: usize,
    width: usize,
    values: Vec<f32>,
    mask: Option<ValidityMask>,
) -> Result<Dataset> {
    let mut array =
        DataArray::<f32>::from_vec_named(vec![3, height, width], ["bands", "y", "x"], values)?
            .with_coordinate("bands", Coordinate::axis("bands", vec![0.0, 1.0, 2.0])?)?;
    if let Some(mask) = mask {
        array.set_mask(mask)?;
    }
    let mut dataset = Dataset::new(DataId::new(name)?).with_array(array);
    dataset.insert_attr("mode", MetadataValue::string("RGB"))?;
    Ok(dataset)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rgb_dataset(name: &str, shape: [usize; 3], value: f32) -> Dataset {
        let count = shape.iter().product();
        let array = DataArray::<f32>::from_vec_named(
            shape.to_vec(),
            ["bands", "y", "x"],
            vec![value; count],
        )
        .expect("array");
        Dataset::new(DataId::new(name).expect("id")).with_array(array)
    }

    #[test]
    fn new_validates_name_and_limits() {
        assert!(DayNightCompositor::new("", 73.0, 85.0).is_err());
        assert!(DayNightCompositor::new("tcr", 85.0, 73.0).is_err());
        assert!(DayNightCompositor::new("tcr", 73.0, 73.0).is_err());
        assert!(DayNightCompositor::new("tcr", f64::NAN, 85.0).is_err());
        let c = DayNightCompositor::new("tcr", 73.0, 85.0).expect("ok");
        assert_eq!(c.lim_low(), 73.0);
        assert_eq!(c.lim_high(), 85.0);
    }

    #[test]
    fn weight_one_keeps_day_weight_zero_keeps_night() {
        let day = rgb_dataset("day", [3, 2, 2], 100.0);
        let night = rgb_dataset("night", [3, 2, 2], 10.0);
        let c = DayNightCompositor::new("blend", 73.0, 85.0).expect("ok");
        let out = c
            .compose_owned(day, night, vec![1.0, 1.0, 1.0, 1.0])
            .expect("blend");
        let values = out.array().expect("arr").values_as_f64();
        for v in values {
            assert!((v - 100.0).abs() < 1e-6, "day-only pixel {v}");
        }

        let day = rgb_dataset("day", [3, 2, 2], 100.0);
        let night = rgb_dataset("night", [3, 2, 2], 10.0);
        let c = DayNightCompositor::new("blend", 73.0, 85.0).expect("ok");
        let out = c
            .compose(&day, &night, &[0.0, 0.0, 0.0, 0.0])
            .expect("blend");
        let values = out.array().expect("arr").values_as_f64();
        for v in values {
            assert!((v - 10.0).abs() < 1e-6, "night-only pixel {v}");
        }
    }

    #[test]
    fn half_weight_averages() {
        let day = rgb_dataset("day", [3, 2, 2], 100.0);
        let night = rgb_dataset("night", [3, 2, 2], 10.0);
        let c = DayNightCompositor::new("blend", 73.0, 85.0).expect("ok");
        let out = c.compose(&day, &night, &[0.5; 4]).expect("blend");
        let values = out.array().expect("arr").values_as_f64();
        for v in values {
            assert!((v - 55.0).abs() < 1e-6, "half blend pixel {v}");
        }
        assert_eq!(
            out.attr("mode").and_then(MetadataValue::as_str),
            Some("RGB")
        );
    }

    #[test]
    fn weight_is_per_pixel() {
        // 2×2 with different weights per pixel: pixel 0 is day-only, pixel 1
        // night-only, pixels 2/3 half.
        let day = rgb_dataset("day", [3, 2, 2], 100.0);
        let night = rgb_dataset("night", [3, 2, 2], 10.0);
        let c = DayNightCompositor::new("blend", 73.0, 85.0).expect("ok");
        let out = c
            .compose_owned(day, night, vec![1.0, 0.0, 0.5, 0.5])
            .expect("blend");
        let values = out.array().expect("arr").values_as_f64();
        let band_count = 3;
        let pixel_count = 4;
        for band in 0..band_count {
            let base = band * pixel_count;
            assert!((values[base] - 100.0).abs() < 1e-6);
            assert!((values[base + 1] - 10.0).abs() < 1e-6);
            assert!((values[base + 2] - 55.0).abs() < 1e-6);
            assert!((values[base + 3] - 55.0).abs() < 1e-6);
        }
    }

    #[test]
    fn mask_is_or_propagated() {
        let mut day_array =
            DataArray::<f32>::from_vec_named([3, 2, 2], ["bands", "y", "x"], vec![100.0_f32; 12])
                .expect("arr");
        let mut mask = ValidityMask::all_valid(12);
        mask.set_masked(1, true);
        day_array = day_array.with_mask(mask).expect("mask");
        let day = Dataset::new(DataId::new("day").expect("id")).with_array(day_array);

        let night = rgb_dataset("night", [3, 2, 2], 10.0);
        let c = DayNightCompositor::new("blend", 73.0, 85.0).expect("ok");
        let out = c.compose(&day, &night, &[1.0; 4]).expect("blend");
        let mask = out.array().expect("arr").mask().expect("mask present");
        assert!(mask.is_masked(1).expect("idx"), "day-masked pixel");
        assert!(mask.is_masked(3 * 4 - 1).is_some());
        // pixel 0 unmasked on both sides → valid.
        assert!(!mask.is_masked(0).expect("idx"));
    }

    #[test]
    fn validates_shapes_and_weights() {
        let day = rgb_dataset("day", [3, 2, 2], 100.0);
        let night = rgb_dataset("night", [3, 3, 3], 10.0);
        let c = DayNightCompositor::new("blend", 73.0, 85.0).expect("ok");
        assert!(c.compose(&day, &night, &[1.0; 4]).is_err());

        let night = rgb_dataset("night", [3, 2, 2], 10.0);
        assert!(c.compose(&day, &night, &[1.0; 5]).is_err());

        let two_d = Dataset::new(DataId::new("2d").expect("id")).with_array(
            DataArray::<f32>::from_vec_named(vec![2, 2], ["y", "x"], vec![0.0; 4]).expect("arr"),
        );
        assert!(c.compose(&two_d, &night, &[1.0; 4]).is_err());
    }
}
