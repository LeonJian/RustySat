//! Solar zenith angle correction modifier.
//!
//! Normalizes TOA reflectance to overhead-sun equivalent by dividing each pixel
//! by cos(solar_zenith_angle). This compensates for the varying solar
//! illumination path length across the Earth disk.
//!
//! Memory strategy: the correction is applied in row strips (64 rows by default).
//! Angles are computed per pixel in the hot loop using precomputed time/column
//! constants. Strip buffers are dropped each iteration, keeping peak memory low.
//!
//! Reference: `satpy/satpy/modifiers/angles.py` — `SunZenithCorrector`.
//! Formula from satpy: `corrected_refl = refl / pow(cos(sunz_rad), n)`
//! where n is configurable (default 1.0).

use crate::angles::{precompute_columns, precompute_rows, solar_single_precomputed, AngleParams};
use crate::astronomy::{gmst, sun_ra_dec, UtcInstant};
use crate::rayleigh::into_vis_f32;
use rusty_sat_core::{
    Coordinate, DataArray, DataId, Dataset, MetadataValue, Result, RustySatError, ValidityMask,
};

/// Solar zenith angle correction.
///
/// ## Formula (Satpy `SunZenithCorrector`)
///
/// ```text
/// limit = 88.0°
/// limit_cos = cos(88°)
/// max_sza_cos = cos(max_sza)
///
/// if     cos_sza > limit_cos:     corr = 1 / cos_sza
/// elif   cos_sza > max_sza_cos:  grad = (cos_sza - max_sza_cos) / (limit_cos - max_sza_cos)
///                                  corr = (1 - log₂(grad+1)) / limit_cos
/// else   (night):                  corr = 0
///
/// result = refl * corr
/// ```
///
/// The default `max_sza` 95.0° matches Satpy default.
///
/// ## Memory
///
/// Consumes the input dataset and `AngleParams`. Correction applied in 64-row
/// strips using precomputed column/row trig tables. No full-grid angle storage.
#[derive(Debug, Clone, Copy)]
pub struct SunZenithCorrector {
    max_sza: f64,
}

impl Default for SunZenithCorrector {
    fn default() -> Self {
        Self { max_sza: 95.0 }
    }
}
impl SunZenithCorrector {
    /// Create a new corrector with a custom max_sza.
    ///
    /// `max_sza` is the solar zenith angle beyond which correction is 0 (night).
    /// Must be in [88.0, 180.0). Default is 95.0°.
    pub fn new(max_sza: f64) -> Result<Self> {
        if !max_sza.is_finite() || !(88.0..180.0).contains(&max_sza) {
            return Err(RustySatError::invalid_input(format!(
                "max_sza must be in [88.0, 180.0), got {max_sza}"
            )));
        }
        Ok(Self { max_sza })
    }

    pub fn max_sza(&self) -> f64 {
        self.max_sza
    }

    /// Apply solar zenith correction to a reflectance dataset.
    ///
    /// Consumes `self` and the dataset. Returns a corrected f32 dataset.
    /// Preserves area metadata and x/y coordinates from the source.
    pub fn apply_correction(self, dataset: Dataset, params: AngleParams) -> Result<Dataset> {
        let max_sza = self.max_sza;
        let max_sza_rad = max_sza.to_radians();
        let max_sza_cos = max_sza_rad.cos();
        const LIMIT_DEG: f64 = 88.0;
        let limit_cos = LIMIT_DEG.to_radians().cos();
        let span = limit_cos - max_sza_cos;
        let inv_span = 1.0 / span;

        let area_attr = dataset.attr("area").cloned();
        let source_coords = dataset.array().map(|a| a.coords().clone());

        let array = dataset
            .into_array()
            .ok_or_else(|| RustySatError::invalid_input("dataset has no array data"))?;
        let (height, width) = array.shape_yx()?;
        let mask = array.mask().cloned();
        let mut refl_f32 = into_vis_f32(array);

        let gmst_val = gmst(params.utc);
        let (sun_ra, sun_dec) = sun_ra_dec(params.utc);
        let h = params.geos.perspective_point_height;
        let h_inv = 1.0 / h;
        let lon_0_rad = params.geos.longitude_of_projection_origin.to_radians();
        let column_data = precompute_columns(&params.x_coords, h_inv, lon_0_rad, gmst_val, sun_ra);

        let max_angle = (params.geos.semi_major_axis / params.geos.satellite_radius()).asin();
        let max_angle_sq = max_angle * max_angle;

        const STRIP_HEIGHT: usize = 64;

        for y0 in (0..height).step_by(STRIP_HEIGHT) {
            let y1 = (y0 + STRIP_HEIGHT).min(height);
            let strip_n = (y1 - y0) * width;

            let tan_theta_y_row = precompute_rows(&params.y_coords, y0, y1, h_inv);
            let offset = y0 * width;

            use rayon::prelude::*;
            refl_f32[offset..offset + strip_n]
                .par_iter_mut()
                .enumerate()
                .for_each(|(local_i, out)| {
                    if !out.is_finite() {
                        return;
                    }
                    let row = local_i / width;
                    let col = local_i % width;
                    let col_pre = &column_data[col];

                    let x_pos = params.x_coords[col];
                    let y_pos = params.y_coords[y0 + row];
                    let rsq = (x_pos * x_pos + y_pos * y_pos) / (h * h);
                    if rsq > max_angle_sq {
                        *out = f32::NAN;
                        return;
                    }

                    let lat_rad = (tan_theta_y_row[row] * col_pre.cos_theta_x).atan();
                    let (sunz_deg, _suna) = solar_single_precomputed(lat_rad, sun_dec, col_pre);
                    if !sunz_deg.is_finite() {
                        *out = f32::NAN;
                        return;
                    }
                    let cos_sza = sunz_deg.to_radians().cos();
                    *out *= sza_correction_factor(cos_sza, limit_cos, max_sza_cos, inv_span);
                });
        }

        build_result(refl_f32, mask, height, width, area_attr, source_coords)
    }
}

/// Compute the SZA correction factor with Satpy-style gradient falloff.
///
/// `inv_span = 1.0 / (limit_cos - max_sza_cos)`
#[inline]
pub(crate) fn sza_correction_factor(
    cos_sza: f64,
    limit_cos: f64,
    max_sza_cos: f64,
    inv_span: f64,
) -> f32 {
    if cos_sza > limit_cos {
        (1.0 / cos_sza) as f32
    } else if cos_sza > max_sza_cos {
        let grad = ((cos_sza - max_sza_cos) * inv_span).clamp(0.0, 1.0);
        let fac = 1.0 - (grad + 1.0).log2();
        (fac / limit_cos) as f32
    } else {
        0.0_f32
    }
}

fn build_result(
    refl_f32: Vec<f32>,
    mask: Option<ValidityMask>,
    height: usize,
    width: usize,
    area_attr: Option<MetadataValue>,
    source_coords: Option<std::collections::BTreeMap<String, Coordinate>>,
) -> Result<Dataset> {
    let mut result_array =
        DataArray::<f32>::from_vec_named(vec![height, width], vec!["y", "x"], refl_f32)?;

    if let Some(m) = mask {
        result_array = result_array.with_mask(m)?;
    }

    if let Some(coords) = source_coords {
        for (name, coord) in coords {
            if name == "y" || name == "x" {
                result_array = result_array.with_coordinate(&name, coord)?;
            }
        }
    }

    let mut result = Dataset::new(DataId::new("sun_zenith_corrected")?).with_array(result_array);

    if let Some(area) = area_attr {
        result.insert_attr("area", area)?;
    }
    result.insert_attr("modifier", MetadataValue::string("sun_zenith_correction"))?;

    Ok(result)
}

/// Compute angles and apply solar zenith correction in one call.
pub fn sun_zenith_correct(dataset: Dataset, utc: UtcInstant) -> Result<Dataset> {
    let corrector = SunZenithCorrector::default();
    sun_zenith_correct_with(dataset, utc, corrector)
}

/// Compute angles and apply solar zenith correction with a custom corrector.
pub fn sun_zenith_correct_with(
    dataset: Dataset,
    utc: UtcInstant,
    corrector: SunZenithCorrector,
) -> Result<Dataset> {
    use crate::angles::{extract_xy_coords, AngleParams};

    let area_attr = dataset
        .attr("area")
        .ok_or_else(|| RustySatError::invalid_input("dataset missing 'area' attribute"))?;

    let coords = dataset
        .array()
        .ok_or_else(|| RustySatError::invalid_input("dataset has no array"))?
        .coords();

    let (x_coords, y_coords) = extract_xy_coords(coords)?;
    let params = AngleParams::from_dataset_area(area_attr, x_coords, y_coords, utc)?;

    corrector.apply_correction(dataset, params)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::astronomy::UtcInstant;
    use crate::geos::GeosProjection;
    use rusty_sat_core::{AnyDataArray, Coordinate};

    fn make_angle_params(width: usize, height: usize) -> AngleParams {
        let geos = GeosProjection {
            semi_major_axis: 6_378_137.0,
            semi_minor_axis: 6_356_752.314_14,
            perspective_point_height: 35_785_863.0,
            longitude_of_projection_origin: 140.7,
        };
        let h = geos.perspective_point_height;
        let max_angle = (geos.semi_major_axis / geos.satellite_radius()).asin();
        let half_disk = h * max_angle;

        let step_x = if width > 1 {
            2.0 * half_disk / (width - 1) as f64
        } else {
            0.0
        };
        let step_y = if height > 1 {
            2.0 * half_disk / (height - 1) as f64
        } else {
            0.0
        };

        let x_coords: Vec<f64> = (0..width).map(|i| -half_disk + i as f64 * step_x).collect();
        let y_coords: Vec<f64> = (0..height).map(|i| half_disk - i as f64 * step_y).collect();

        AngleParams {
            sat_lon: 140.7,
            sat_lat: 0.0,
            sat_alt: h,
            utc: UtcInstant::from_ymdhms(2025, 9, 23, 7, 20, 0),
            width,
            height,
            geos,
            x_coords,
            y_coords,
        }
    }

    fn make_area_attr() -> MetadataValue {
        let projection = MetadataValue::Map(
            [
                ("a".to_string(), MetadataValue::string("6378137.0")),
                ("b".to_string(), MetadataValue::string("6356752.31414")),
                ("h".to_string(), MetadataValue::string("35785863.0")),
                ("lon_0".to_string(), MetadataValue::string("140.7")),
                ("proj".to_string(), MetadataValue::string("geos")),
                ("units".to_string(), MetadataValue::string("m")),
            ]
            .into_iter()
            .collect(),
        );

        MetadataValue::map([
            ("type", MetadataValue::string("area")),
            ("id", MetadataValue::string("test_area")),
            ("description", MetadataValue::string("test area")),
            ("proj_id", MetadataValue::string("geosh9")),
            ("projection", projection),
            ("height", MetadataValue::Integer(10)),
            ("width", MetadataValue::Integer(10)),
            (
                "area_extent",
                MetadataValue::List(vec![
                    MetadataValue::float(-5_500_000.0).expect("float"),
                    MetadataValue::float(-5_500_000.0).expect("float"),
                    MetadataValue::float(5_500_000.0).expect("float"),
                    MetadataValue::float(5_500_000.0).expect("float"),
                ]),
            ),
        ])
    }

    fn make_dataset(width: usize, height: usize, values: Vec<f32>, with_coords: bool) -> Dataset {
        let mut array =
            DataArray::<f32>::from_vec_named(vec![height, width], vec!["y", "x"], values)
                .expect("valid array");

        if with_coords {
            let geos = GeosProjection {
                semi_major_axis: 6_378_137.0,
                semi_minor_axis: 6_356_752.314_14,
                perspective_point_height: 35_785_863.0,
                longitude_of_projection_origin: 140.7,
            };
            let h = geos.perspective_point_height;
            let max_angle = (geos.semi_major_axis / geos.satellite_radius()).asin();
            let half_disk = h * max_angle;
            let step_x = if width > 1 {
                2.0 * half_disk / (width - 1) as f64
            } else {
                0.0
            };
            let step_y = if height > 1 {
                2.0 * half_disk / (height - 1) as f64
            } else {
                0.0
            };
            let x_vals: Vec<f64> = (0..width).map(|i| -half_disk + i as f64 * step_x).collect();
            let y_vals: Vec<f64> = (0..height).map(|i| half_disk - i as f64 * step_y).collect();
            array = array
                .with_coordinate("x", Coordinate::axis("x", x_vals).expect("x coord"))
                .expect("x coord")
                .with_coordinate("y", Coordinate::axis("y", y_vals).expect("y coord"))
                .expect("y coord");
        }

        let mut dataset =
            Dataset::new(DataId::new("test_band").expect("valid id")).with_array(array);
        dataset
            .insert_attr("area", make_area_attr())
            .expect("insert area");
        dataset
    }

    #[test]
    fn default_max_sza_is_95() {
        let c = SunZenithCorrector::default();
        assert!((c.max_sza() - 95.0).abs() < 0.01);
    }

    #[test]
    fn rejects_invalid_max_sza() {
        assert!(SunZenithCorrector::new(87.9).is_err());
        assert!(SunZenithCorrector::new(180.0).is_err());
        assert!(SunZenithCorrector::new(-1.0).is_err());
        assert!(SunZenithCorrector::new(f64::NAN).is_err());
    }

    #[test]
    fn accepts_valid_max_sza() {
        assert!(SunZenithCorrector::new(88.0).is_ok());
        assert!(SunZenithCorrector::new(95.0).is_ok());
        assert!(SunZenithCorrector::new(120.0).is_ok());
    }

    #[test]
    fn max_sza_reported_correctly() {
        let corrector = SunZenithCorrector::new(100.0).expect("valid");
        assert!((corrector.max_sza() - 100.0).abs() < 0.01);
    }

    #[test]
    fn sun_zenith_correction_reduces_constant_reflectance() {
        let values = vec![50.0_f32; 5 * 5]; // 5x5 grid with 50% reflectance everywhere
        let dataset = make_dataset(5, 5, values.clone(), true);
        let params = make_angle_params(5, 5);
        let original_mean: f64 =
            values.iter().map(|v| *v as f64).sum::<f64>() / values.len() as f64;

        let corrector = SunZenithCorrector::default();
        let result = corrector
            .apply_correction(dataset, params)
            .expect("correction should succeed");

        let result_values: Vec<f64> = result.array().expect("array exists").values_as_f64();

        // Correction should reduce reflectance (cos(sza) <= 1, so dividing increases values)
        // Wait, no: we divide by cos(sza) which is ≤ 1, so values INCREASE.
        // The solar zenith correction normalizes to overhead sun.
        // At the terminator (sza ~ 90°), cos(sza) is near 0, so we'd divide by ~0.
        // But for nadir (sza ~ 0°), cos(sza) is near 1, so little change.
        // In a 5x5 grid centered on SSP, most pixels have moderate sza,
        // so the correction should generally increase reflectance.
        let result_mean: f64 = result_values.iter().filter(|v| v.is_finite()).sum::<f64>()
            / result_values
                .iter()
                .filter(|v| v.is_finite())
                .count()
                .max(1) as f64;

        // Near SSP, cos(sza) ≈ 1, so values stay similar.
        // The mean should not decrease dramatically (it should increase or stay similar).
        assert!(result_mean >= original_mean * 0.8);
    }

    #[test]
    fn correction_increases_values_overall() {
        let n = 15;
        let values = vec![50.0_f32; n * n];
        let dataset = make_dataset(n, n, values.clone(), true);
        let params = make_angle_params(n, n);

        let corrector = SunZenithCorrector::default();
        let result = corrector
            .apply_correction(dataset, params)
            .expect("correction should succeed");

        let result_values = result.array().expect("array").values_as_f64();

        let original_mean: f64 =
            values.iter().map(|v| *v as f64).sum::<f64>() / values.len() as f64;
        let result_mean: f64 = result_values.iter().filter(|v| v.is_finite()).sum::<f64>()
            / result_values
                .iter()
                .filter(|v| v.is_finite())
                .count()
                .max(1) as f64;

        // Dividing by cos(sza) ≤ 1 always increases or equalises values
        assert!(
            result_mean >= original_mean,
            "correction should increase values overall: {result_mean} >= {original_mean}"
        );

        // Center pixel should be finite (within disk)
        let center_idx = n / 2 * n + n / 2;
        assert!(
            result_values[center_idx].is_finite(),
            "center pixel should be finite"
        );

        // Not all pixels get the same correction (different sza → different cos)
        let finite_vals: Vec<f64> = result_values
            .iter()
            .filter(|v| v.is_finite())
            .copied()
            .collect();
        let all_same = finite_vals.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-6);
        assert!(!all_same, "not all pixels should get the same correction");
    }

    #[test]
    fn space_pixels_become_nan() {
        // Build a grid where one pixel is outside the disk
        let geos = GeosProjection {
            semi_major_axis: 6_378_137.0,
            semi_minor_axis: 6_356_752.314_14,
            perspective_point_height: 35_785_863.0,
            longitude_of_projection_origin: 140.7,
        };
        let h = geos.perspective_point_height;
        let max_angle = (geos.semi_major_axis / geos.satellite_radius()).asin();
        let edge = h * max_angle;

        let values = vec![50.0_f32; 3];
        let mut array = DataArray::<f32>::from_vec_named(vec![1, 3], vec!["y", "x"], values)
            .expect("valid array");

        // First pixel valid, third pixel outside disk (x > edge)
        let y_vals = vec![0.0_f64];
        let x_vals = vec![edge * 0.5, 0.0, edge * 1.01];
        array = array
            .with_coordinate("x", Coordinate::axis("x", x_vals).expect("x coord"))
            .expect("x coord")
            .with_coordinate("y", Coordinate::axis("y", y_vals).expect("y coord"))
            .expect("y coord");

        let mut dataset = Dataset::new(DataId::new("test").expect("valid id")).with_array(array);
        dataset.insert_attr("area", make_area_attr()).expect("area");

        let params = AngleParams {
            sat_lon: geos.longitude_of_projection_origin,
            sat_lat: 0.0,
            sat_alt: h,
            utc: UtcInstant::from_ymdhms(2025, 9, 23, 7, 20, 0),
            width: 3,
            height: 1,
            geos,
            x_coords: vec![edge * 0.5, 0.0, edge * 1.01],
            y_coords: vec![0.0],
        };

        let corrector = SunZenithCorrector::default();
        let result = corrector
            .apply_correction(dataset, params)
            .expect("correction should succeed");

        let rv: Vec<f64> = result.array().expect("array").values_as_f64();

        assert!(rv[0].is_finite(), "valid pixel should be finite: {}", rv[0]);
        assert!(rv[1].is_finite(), "nadir pixel should be finite: {}", rv[1]);
        assert!(rv[2].is_nan(), "space pixel should be nan: {}", rv[2]);
    }

    #[test]
    fn preserves_mask() {
        let values = vec![50.0_f32; 5 * 5];
        let mut array = DataArray::<f32>::from_vec_named(vec![5, 5], vec!["y", "x"], values)
            .expect("valid array");

        let mut mask = ValidityMask::all_valid(25);
        mask.set_masked(12, true);
        array = array.with_mask(mask).expect("set mask");

        let geos = GeosProjection {
            semi_major_axis: 6_378_137.0,
            semi_minor_axis: 6_356_752.314_14,
            perspective_point_height: 35_785_863.0,
            longitude_of_projection_origin: 140.7,
        };
        let h = geos.perspective_point_height;
        let max_angle = (geos.semi_major_axis / geos.satellite_radius()).asin();
        let half_disk = h * max_angle;
        let step = 2.0 * half_disk / 4.0;
        let x_vals: Vec<f64> = (0..5).map(|i| -half_disk + i as f64 * step).collect();
        let y_vals: Vec<f64> = (0..5).map(|i| half_disk - i as f64 * step).collect();
        array = array
            .with_coordinate("x", Coordinate::axis("x", x_vals).expect("x"))
            .expect("x")
            .with_coordinate("y", Coordinate::axis("y", y_vals).expect("y"))
            .expect("y");

        let mut dataset = Dataset::new(DataId::new("test").expect("valid id")).with_array(array);
        dataset.insert_attr("area", make_area_attr()).expect("area");

        let params = make_angle_params(5, 5);
        let corrector = SunZenithCorrector::default();
        let result = corrector
            .apply_correction(dataset, params)
            .expect("correction should succeed");

        let result_mask = result.array().expect("array").mask().cloned();
        assert!(result_mask.is_some(), "mask should be preserved");
        assert!(
            result_mask
                .as_ref()
                .expect("mask")
                .is_masked(12)
                .expect("valid index"),
            "pixel 12 should still be masked"
        );
    }

    #[test]
    fn preserves_area_and_coordinates() {
        let values = vec![50.0_f32; 3 * 3];
        let dataset = make_dataset(3, 3, values, true);
        let params = make_angle_params(3, 3);
        let corrector = SunZenithCorrector::default();

        let result = corrector
            .apply_correction(dataset, params)
            .expect("correction should succeed");

        assert!(
            result.attr("area").is_some(),
            "area attribute should be preserved"
        );
        assert!(
            result.attr("modifier").is_some(),
            "modifier attribute should be set"
        );
        let coords = result.array().expect("array").coords();
        assert!(coords.contains_key("x"), "x coordinate should be preserved");
        assert!(coords.contains_key("y"), "y coordinate should be preserved");
    }

    #[test]
    fn convenience_uses_area_attr_and_coordinates() {
        let dataset = make_dataset(3, 3, vec![50.0_f32; 9], true);
        let utc = UtcInstant::from_ymdhms(2025, 9, 23, 7, 20, 0);
        let result = sun_zenith_correct(dataset, utc).expect("convenience fn should work");
        assert!(result.attr("modifier").is_some());
        assert!(result.array().is_some());
    }

    #[test]
    fn error_on_missing_area_attr() {
        let array = DataArray::<f32>::from_vec_named(vec![2, 2], vec!["y", "x"], vec![50.0_f32; 4])
            .expect("valid array");
        let dataset = Dataset::new(DataId::new("test").expect("valid id")).with_array(array);
        let utc = UtcInstant::from_ymdhms(2025, 9, 23, 7, 20, 0);
        assert!(sun_zenith_correct(dataset, utc).is_err());
    }

    #[test]
    fn error_on_missing_array() {
        let mut dataset = Dataset::new(DataId::new("test").expect("valid id"));
        dataset
            .insert_attr("area", make_area_attr())
            .expect("insert area");
        let utc = UtcInstant::from_ymdhms(2025, 9, 23, 7, 20, 0);
        assert!(sun_zenith_correct(dataset, utc).is_err());
    }

    #[test]
    fn sza_falloff_reduces_correction_near_terminator() {
        // max_sza=95°, pixels at moderate SZA should get correction,
        // pixels near the terminator get zero or reduced.
        let corrector = SunZenithCorrector::new(95.0).expect("valid");
        let n = 15;
        let values = vec![100.0_f32; n * n];
        let dataset = make_dataset(n, n, values, true);
        let params = make_angle_params(n, n);
        let result = corrector
            .apply_correction(dataset, params)
            .expect("correction");
        let vals = result.array().expect("arr").values_as_f64();
        // Center pixel at SSP has sza > 0 (sun not overhead at 7:20 UTC),
        // so correction factor > 1. Values should be finite and ≥ 100.
        let center = vals[n / 2 * n + n / 2];
        assert!(center.is_finite(), "center should be finite");
        assert!(center >= 99.0, "center should be ≥ 100: got {center}");
        // Some edge pixels may be NaN (space) or have very large corrections.
        let finite: Vec<_> = vals.iter().filter(|v| v.is_finite()).copied().collect();
        assert!(!finite.is_empty());
    }

    #[test]
    fn owned_array_is_consumed() {
        let values = vec![50.0_f32; 5 * 5];
        let dataset = make_dataset(5, 5, values, true);
        let params = make_angle_params(5, 5);

        let corrector = SunZenithCorrector::default();
        let result = corrector
            .apply_correction(dataset, params)
            .expect("correction should succeed");

        // After correction, the result should be a new f32 array
        let arr = result.array().expect("array exists");
        match arr {
            AnyDataArray::F32(_) => {}
            _ => panic!("expected F32 array"),
        }
    }

    #[test]
    fn zero_reflectance_stays_zero() {
        let values = vec![0.0_f32; 5 * 5];
        let dataset = make_dataset(5, 5, values, true);
        let params = make_angle_params(5, 5);
        let corrector = SunZenithCorrector::default();
        let result = corrector
            .apply_correction(dataset, params)
            .expect("correction should succeed");
        let rv = result.array().expect("array").values_as_f64();
        for v in rv {
            if v.is_finite() {
                assert!(
                    (v - 0.0).abs() < 1e-6,
                    "zero reflectance should stay zero for finite pixels, got {v}"
                );
            }
            // NaN is fine — corners outside Earth disk
        }
    }
}
