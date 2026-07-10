//! Rayleigh scattering correction modifier — Rust port of
//! `satpy.modifiers.atmosphere.PSPRayleighReflectance` +
//! `pyspectral.rayleigh.Rayleigh`.
//!
//! Reference:
//! - `satpy/satpy/modifiers/atmosphere.py` — `PSPRayleighReflectance.__call__`
//! - `deps/pyspectral/pyspectral/rayleigh.py` — `Rayleigh.get_reflectance`,
//!   `reduce_rayleigh_highzenith`, `_relax_rayleigh_refl_correction_where_cloudy`
//!
//! The correction subtracts the Rayleigh/aerosol reflectance contribution
//! from the visible-band reflectance, producing a corrected reflectance
//! that is closer to the true surface reflectance.
//!
//! ## Algorithm
//!
//! 1. Compute sun/satellite zenith and azimuth angles for each pixel.
//! 2. Compute relative azimuth from the azimuth angles.
//! 3. Load the Rayleigh LUT and select the wavelength-adjusted 3D slice.
//! 4. Interpolate the LUT at each pixel's angles → reflectance correction.
//! 5. Optionally relax the correction where clouds are present (red band > 20%).
//! 6. Optionally reduce the correction at high solar zenith angles.
//! 7. Subtract the correction from the visible reflectance.
//!
//! ## Memory strategy
//!
//! - Time-dependent astronomical constants are precomputed once per correction.
//! - Angle computation, LUT interpolation, cloud/high-zenith correction,
//!   and subtraction are performed in row strips (64 rows by default).
//!   Strip-sized buffers are allocated and freed per iteration, keeping
//!   peak memory proportional to ~5 GB for a full-disk AHI B03 image
//!   (vs ~23 GB for the previous full-array approach).
//! - The 4D LUT coordinate arrays are extracted (moved out) before the
//!   reflectance buffer is consumed by wavelength selection.
//! - Parallel processing via rayon for large grids (>10k pixels per strip).

use crate::angles::{
    precompute_columns, precompute_rows, satellite_single_precomputed, solar_single_precomputed,
};
use crate::astronomy::{gmst, observer_position, sun_ra_dec, UtcInstant};
use crate::rayleigh_lut::{lut_interpolate_single, RayleighLut};
use rusty_sat_core::{
    AnyDataArray, DataArray, DataId, Dataset, MetadataValue, Result, RustySatError,
};

/// Aerosol type for the Rayleigh LUT.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AerosolType {
    AntarcticAerosol,
    ContinentalAverageAerosol,
    ContinentalCleanAerosol,
    ContinentalPollutedAerosol,
    DesertAerosol,
    MarineCleanAerosol,
    MarinePollutedAerosol,
    MarineTropicalAerosol,
    RayleighOnly,
    RuralAerosol,
    UrbanAerosol,
}

impl AerosolType {
    /// Directory name as used by pyspectral.
    pub fn dir_name(&self) -> &'static str {
        match self {
            Self::AntarcticAerosol => "antarctic_aerosol",
            Self::ContinentalAverageAerosol => "continental_average_aerosol",
            Self::ContinentalCleanAerosol => "continental_clean_aerosol",
            Self::ContinentalPollutedAerosol => "continental_polluted_aerosol",
            Self::DesertAerosol => "desert_aerosol",
            Self::MarineCleanAerosol => "marine_clean_aerosol",
            Self::MarinePollutedAerosol => "marine_polluted_aerosol",
            Self::MarineTropicalAerosol => "marine_tropical_aerosol",
            Self::RayleighOnly => "rayleigh_only",
            Self::RuralAerosol => "rural_aerosol",
            Self::UrbanAerosol => "urban_aerosol",
        }
    }
}

/// Atmosphere type for the Rayleigh LUT.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Atmosphere {
    SubarcticSummer,
    SubarcticWinter,
    MidlatitudeSummer,
    MidlatitudeWinter,
    Tropical,
    UsStandard,
}

impl Atmosphere {
    /// File suffix as used by pyspectral.
    pub fn file_suffix(&self) -> &'static str {
        match self {
            Self::SubarcticSummer => "subarctic_summer",
            Self::SubarcticWinter => "subarctic_winter",
            Self::MidlatitudeSummer => "midlatitude_summer",
            Self::MidlatitudeWinter => "midlatitude_winter",
            Self::Tropical => "tropical",
            Self::UsStandard => "us-standard",
        }
    }
}

/// Configuration for the Rayleigh correction.
#[derive(Debug, Clone)]
pub struct RayleighConfig {
    /// Atmosphere type for the LUT.
    pub atmosphere: Atmosphere,
    /// Aerosol type for the LUT.
    pub aerosol_type: AerosolType,
    /// Solar zenith angle threshold for reduction (degrees).
    pub reduce_lim_low: f64,
    /// Solar zenith angle where reduction ends (degrees).
    pub reduce_lim_high: f64,
    /// Reduction strength (0.0–1.0).
    pub reduce_strength: f64,
}

impl Default for RayleighConfig {
    fn default() -> Self {
        Self {
            atmosphere: Atmosphere::UsStandard,
            aerosol_type: AerosolType::MarineCleanAerosol,
            reduce_lim_low: 70.0,
            reduce_lim_high: 105.0,
            reduce_strength: 0.0,
        }
    }
}

/// The Rayleigh reflectance corrector.
///
/// This is the Rust equivalent of `pyspectral.rayleigh.Rayleigh` +
/// `satpy.modifiers.PSPRayleighReflectance`.
pub struct RayleighCorrector {
    config: RayleighConfig,
    lut: RayleighLut,
}

impl RayleighCorrector {
    /// Create a corrector with the given LUT and default config.
    pub fn new(lut: RayleighLut) -> Self {
        Self {
            config: RayleighConfig::default(),
            lut,
        }
    }

    /// Create a corrector with the given LUT and custom config.
    pub fn with_config(lut: RayleighLut, config: RayleighConfig) -> Self {
        Self { config, lut }
    }

    /// Apply the Rayleigh correction to a visible-band reflectance dataset.
    ///
    /// This is the main entry point, equivalent to
    /// `PSPRayleighReflectance.__call__`.
    ///
    /// Consumes the visible dataset and returns the corrected dataset.
    /// The correction is computed in row strips to keep peak memory low.
    ///
    /// - `vis_dataset`: visible-band reflectance dataset (owned, consumed)
    /// - `red_dataset`: optional red-band reflectance for cloud relaxation
    /// - `params`: angle computation parameters (projection, coords, time);
    ///   consumed to compute angles per strip
    /// - `wavelength_nm`: central wavelength of the visible band (nm)
    pub fn apply_correction(
        self,
        vis_dataset: Dataset,
        red_dataset: Option<&Dataset>,
        params: crate::angles::AngleParams,
        wavelength_nm: f64,
    ) -> Result<Dataset> {
        let config = self.config;

        let sun_zenith_secant = self.lut.sun_zenith_secant.clone();
        let azimuth_difference = self.lut.azimuth_difference.clone();
        let satellite_zenith_secant = self.lut.satellite_zenith_secant.clone();

        let lut_3d = self.lut.into_wavelength_adjusted(wavelength_nm)?;
        if lut_3d.is_empty() {
            return rebuild_dataset(vis_dataset, "rayleigh_corrected");
        }

        let array = vis_dataset
            .into_array()
            .ok_or_else(|| RustySatError::invalid_input("visible dataset has no array data"))?;
        let (height, width) = array.shape_yx()?;
        let mask = array.mask().cloned();
        let mut vis_f32 = into_vis_f32(array);

        let red_values: Option<Vec<f64>> =
            red_dataset.map(|ds| ds.array().map(|a| a.values_as_f64()).unwrap_or_default());

        // Precompute time-dependent constants once for the entire grid.
        let gmst_val = gmst(params.utc);
        let (sun_ra, sun_dec) = sun_ra_dec(params.utc);
        let sat_alt_km = params.sat_alt / 1000.0;
        let (sat_x, sat_y, sat_z) =
            observer_position(params.utc, params.sat_lon, params.sat_lat, sat_alt_km);

        // Precompute column-dependent trig (saves ~5 trig calls per pixel).
        let h_inv = 1.0 / params.geos.perspective_point_height;
        let lon_0_rad = params.geos.longitude_of_projection_origin.to_radians();
        let column_data = precompute_columns(&params.x_coords, h_inv, lon_0_rad, gmst_val, sun_ra);

        // Precompute LUT interpolation constants once (saves ~8 fp ops per pixel).
        let lut_constants = RayleighLut::lut_interp_constants(
            &sun_zenith_secant,
            &azimuth_difference,
            &satellite_zenith_secant,
        );

        const STRIP_HEIGHT: usize = 64;

        for y0 in (0..height).step_by(STRIP_HEIGHT) {
            let y1 = (y0 + STRIP_HEIGHT).min(height);
            let strip_n = (y1 - y0) * width;

            // Precompute tan(theta_y) per row in this strip.
            let tan_theta_y_row = precompute_rows(&params.y_coords, y0, y1, h_inv);
            let red = red_values.as_ref();
            let offset = y0 * width;

            use rayon::prelude::*;
            vis_f32[offset..offset + strip_n]
                .par_iter_mut()
                .enumerate()
                .for_each(|(local_i, out)| {
                    if !out.is_finite() {
                        return;
                    }
                    let row = local_i / width;
                    let col = local_i % width;
                    let col_pre = &column_data[col];

                    // 1. Projection inverse → lat_rad (lon is implicit in col precomputation)
                    let lat_rad = (tan_theta_y_row[row] * col_pre.cos_theta_x).atan();

                    // 2. Solar zenith + azimuth
                    let (sunz, suna) = solar_single_precomputed(lat_rad, sun_dec, col_pre);

                    // 3. Satellite zenith + azimuth
                    let (satz, sata) =
                        satellite_single_precomputed(lat_rad, col_pre, sat_x, sat_y, sat_z);

                    // 4. LUT interpolation
                    let mut correction = lut_interpolate_single(
                        &lut_3d,
                        &lut_constants,
                        &sun_zenith_secant,
                        &azimuth_difference,
                        &satellite_zenith_secant,
                        sunz,
                        satz,
                        suna,
                        sata,
                    );

                    // 5. Cloud relaxation
                    if let Some(red_vals) = red {
                        let r = red_vals[offset + local_i];
                        if r.is_finite() && r >= 20.0 {
                            correction *= 1.0 - (r - 20.0) / 80.0;
                        }
                    }

                    // 6. High-zenith reduction
                    if config.reduce_strength > 0.0 && sunz.is_finite() {
                        let (lo, hi) = if config.reduce_lim_low > config.reduce_lim_high {
                            (config.reduce_lim_high, config.reduce_lim_low)
                        } else {
                            (config.reduce_lim_low, config.reduce_lim_high)
                        };
                        let span = hi - lo;
                        if span > 0.0 {
                            let t = if sunz < lo {
                                0.0
                            } else {
                                ((sunz - lo) / span).min(1.0)
                            };
                            let factor = (1.0 - config.reduce_strength * t).clamp(0.0, 1.0);
                            correction *= factor;
                        }
                    }

                    // 7. Subtract from reflectance (f32 ← f64 correction)
                    let corrected = (*out as f64 - correction).max(0.0);
                    *out = corrected as f32;
                });

            // Strip data dropped here — no intermediate angle/refl_cor buffers
        }

        drop(lut_3d);

        let mut result_array =
            DataArray::<f32>::from_vec_named(vec![height, width], vec!["y", "x"], vis_f32)?;

        if let Some(m) = mask {
            result_array = result_array.with_mask(m)?;
        }

        let mut result = Dataset::new(DataId::new("rayleigh_corrected")?).with_array(result_array);

        result.insert_attr("modifier", MetadataValue::string("rayleigh_correction"))?;
        result.insert_attr(
            "atmosphere",
            MetadataValue::string(config.atmosphere.file_suffix()),
        )?;
        result.insert_attr(
            "aerosol_type",
            MetadataValue::string(config.aerosol_type.dir_name()),
        )?;
        result.insert_attr("wavelength_nm", MetadataValue::float(wavelength_nm)?)?;

        Ok(result)
    }
}

/// Subtract the Rayleigh correction from the visible reflectance in-place.
///
/// Clamps to ≥ 0.0.  NaN pixels are left unchanged.  Uses rayon for large arrays.
#[cfg(test)]
#[inline]
fn subtract_correction(vis: &mut [f64], correction: &[f64]) {
    use rayon::prelude::*;
    let n = vis.len().min(correction.len());
    if n > 10_000 {
        vis.par_iter_mut().take(n).enumerate().for_each(|(i, v)| {
            if v.is_finite() {
                *v -= correction[i];
                if *v < 0.0 {
                    *v = 0.0;
                }
            }
        });
    } else {
        for i in 0..n {
            if vis[i].is_finite() {
                vis[i] -= correction[i];
                if vis[i] < 0.0 {
                    vis[i] = 0.0;
                }
            }
        }
    }
}

/// Extract vis reflectance values as f32 from an AnyDataArray.
///
/// For f32 input (AHI reflectance), this is zero-copy via `into_values()`.
/// Other dtypes are converted element-by-element.
fn into_vis_f32(array: AnyDataArray) -> Vec<f32> {
    match array {
        AnyDataArray::F32(da) => da.into_values(),
        AnyDataArray::F64(da) => da.into_values().into_iter().map(|v| v as f32).collect(),
        AnyDataArray::U8(da) => da.into_values().into_iter().map(|v| v as f32).collect(),
        AnyDataArray::U16(da) => da.into_values().into_iter().map(|v| v as f32).collect(),
        AnyDataArray::I16(da) => da.into_values().into_iter().map(|v| v as f32).collect(),
    }
}

/// Rebuild a dataset with a new ID but unchanged data (for the zero-correction path).
fn rebuild_dataset(mut dataset: Dataset, new_name: &str) -> Result<Dataset> {
    let mut new_ds = Dataset::new(DataId::new(new_name)?);
    if let Some(arr) = dataset.take_array() {
        new_ds = new_ds.with_array(arr);
    }
    Ok(new_ds)
}

/// Relax the Rayleigh correction where clouds are present.
///
/// Ported from `pyspectral.rayleigh._relax_rayleigh_refl_correction_where_cloudy`.
///
/// Where the red-band reflectance is > 20%, the correction is gradually
/// reduced to zero by 100% red reflectance.  Operates in-place.
#[cfg(test)]
#[inline]
fn relax_cloudy_correction(red_band: &[f64], rayleigh_refl: &mut [f64]) {
    use rayon::prelude::*;
    let n = red_band.len().min(rayleigh_refl.len());
    if n > 10_000 {
        rayleigh_refl
            .par_iter_mut()
            .take(n)
            .enumerate()
            .for_each(|(i, slot)| {
                let red = red_band[i];
                if !red.is_finite() {
                    return;
                }
                if red >= 20.0 {
                    *slot *= 1.0 - (red - 20.0) / 80.0;
                }
            });
    } else {
        for i in 0..n {
            let red = red_band[i];
            if !red.is_finite() {
                continue;
            }
            if red >= 20.0 {
                rayleigh_refl[i] *= 1.0 - (red - 20.0) / 80.0;
            }
        }
    }
}

/// Reduce the Rayleigh correction at high solar zenith angles.
///
/// Ported from `pyspectral.rayleigh.reduce_rayleigh_highzenith`.
///
/// Linearly scales the correction between `thresh_zen` and `maxzen`,
/// reducing it by `strength` fraction at `maxzen`.  Operates in-place.
#[cfg(test)]
#[inline]
fn reduce_high_zenith(
    zenith: &[f64],
    rayref: &mut [f64],
    thresh_zen: f64,
    maxzen: f64,
    strength: f64,
) {
    use rayon::prelude::*;
    let n = zenith.len().min(rayref.len());
    let (lo, hi) = if thresh_zen > maxzen {
        (maxzen, thresh_zen)
    } else {
        (thresh_zen, maxzen)
    };
    let span = hi - lo;
    if span <= 0.0 {
        return;
    }
    if n > 10_000 {
        rayref
            .par_iter_mut()
            .take(n)
            .enumerate()
            .for_each(|(i, slot)| {
                let z = zenith[i];
                if !z.is_finite() {
                    return;
                }
                let t = if z < lo {
                    0.0
                } else {
                    ((z - lo) / span).min(1.0)
                };
                let factor = (1.0 - strength * t).clamp(0.0, 1.0);
                *slot *= factor;
            });
    } else {
        for i in 0..n {
            let z = zenith[i];
            if !z.is_finite() {
                continue;
            }
            let t = if z < lo {
                0.0
            } else {
                ((z - lo) / span).min(1.0)
            };
            let factor = (1.0 - strength * t).clamp(0.0, 1.0);
            rayref[i] *= factor;
        }
    }
}

/// Compute angles and apply Rayleigh correction in one call.
///
/// This is a convenience function that:
/// 1. Extracts geometry from the visible dataset's area attribute.
/// 2. Computes the four sun/satellite angles.
/// 3. Applies the Rayleigh correction.
///
/// The `red_dataset` is optional (for cloud relaxation).
/// The `utc` time is the observation time.
/// The `wavelength_nm` is the central wavelength of the visible band.
pub fn rayleigh_correct(
    corrector: RayleighCorrector,
    vis_dataset: Dataset,
    red_dataset: Option<&Dataset>,
    utc: UtcInstant,
    wavelength_nm: f64,
) -> Result<Dataset> {
    use crate::angles::{extract_xy_coords, AngleParams};

    let area_attr = vis_dataset
        .attr("area")
        .ok_or_else(|| RustySatError::invalid_input("dataset missing 'area' attribute"))?;

    let coords = vis_dataset
        .array()
        .ok_or_else(|| RustySatError::invalid_input("dataset has no array"))?
        .coords();

    let (x_coords, y_coords) = extract_xy_coords(coords)?;
    let params = AngleParams::from_dataset_area(area_attr, x_coords, y_coords, utc)?;

    corrector.apply_correction(vis_dataset, red_dataset, params, wavelength_nm)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_lut() -> RayleighLut {
        let wavelengths = vec![631.0, 636.0];
        let sun_zenith_secant = vec![1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75];
        let azimuth_difference = vec![100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0];
        let satellite_zenith_secant = vec![1.0, 1.25, 1.5, 1.75, 2.0, 2.25];

        let n_wvl = 2;
        let n_sunz = 8;
        let n_azid = 7;
        let n_satz = 6;
        let total = n_wvl * n_sunz * n_azid * n_satz;

        let mut reflectance = vec![0.0; total];
        for w in 0..n_wvl {
            for s in 0..n_sunz {
                for a in 0..n_azid {
                    for t in 0..n_satz {
                        let idx = ((w * n_sunz + s) * n_azid + a) * n_satz + t;
                        let base = 0.08 + 0.01 * s as f64 + 0.005 * a as f64 + 0.01 * t as f64;
                        reflectance[idx] = base + 0.002 * w as f64;
                    }
                }
            }
        }

        RayleighLut {
            reflectance,
            wavelengths,
            sun_zenith_secant,
            azimuth_difference,
            satellite_zenith_secant,
        }
    }

    fn make_vis_ds(values: Vec<f64>) -> Dataset {
        let n = values.len();
        let side = (n as f64).sqrt() as usize;
        let array = DataArray::<f64>::from_vec_named(vec![side, side], vec!["y", "x"], values)
            .expect("valid test array");
        Dataset::new(DataId::new("B03").expect("valid DataId")).with_array(array)
    }

    fn make_test_params(side: usize) -> crate::angles::AngleParams {
        let geos = crate::geos::GeosProjection {
            semi_major_axis: 6_378_137.0,
            semi_minor_axis: 6_378_137.0,
            perspective_point_height: 35_785_863.0,
            longitude_of_projection_origin: 0.0,
        };
        let h = geos.perspective_point_height;
        let step = h * 1.0_f64.to_radians() / side.max(1) as f64;
        let x_coords: Vec<f64> = (0..side).map(|i| i as f64 * step).collect();
        let y_coords: Vec<f64> = (0..side).map(|i| i as f64 * step).collect();
        crate::angles::AngleParams {
            sat_lon: 0.0,
            sat_lat: 0.0,
            sat_alt: h,
            utc: crate::astronomy::UtcInstant::from_ymdhms(2025, 9, 23, 7, 20, 0),
            width: side,
            height: side,
            geos,
            x_coords,
            y_coords,
        }
    }

    #[test]
    fn relax_cloudy_preserves_low_red() {
        let red = vec![10.0, 20.0, 50.0, 100.0];
        let mut rr = vec![5.0, 5.0, 5.0, 5.0];
        relax_cloudy_correction(&red, &mut rr);
        assert!((rr[0] - 5.0).abs() < 1e-10);
        assert!((rr[1] - 5.0).abs() < 1e-10);
        assert!((rr[2] - 3.125).abs() < 1e-10);
        assert!((rr[3] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn reduce_high_zenith_reduces_at_high_angles() {
        let zenith = vec![10.0, 70.0, 90.0, 105.0];
        let mut rr = vec![10.0, 10.0, 10.0, 10.0];
        reduce_high_zenith(&zenith, &mut rr, 70.0, 105.0, 0.5);
        assert!((rr[0] - 10.0).abs() < 1e-10);
        assert!((rr[1] - 10.0).abs() < 1e-10);
        assert!(rr[2] < 10.0 && rr[2] > 7.0);
        assert!((rr[3] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn apply_correction_subtracts_rayleigh() {
        let lut = make_test_lut();
        let corrector = RayleighCorrector::new(lut);
        let vis_ds = make_vis_ds(vec![50.0; 4]);
        let params = make_test_params(2);
        let result = corrector
            .apply_correction(vis_ds, None, params, 634.0)
            .expect("correction should succeed");
        let vals = result
            .array()
            .expect("result should have array")
            .values_as_f64();
        assert!(vals.iter().all(|v| *v < 50.0));
        assert!(vals.iter().all(|v| *v >= 0.0));
    }

    #[test]
    fn apply_correction_with_red_band_reduces_effect() {
        let lut = make_test_lut();
        let corrector = RayleighCorrector::new(lut);
        let vis_ds = make_vis_ds(vec![50.0; 4]);
        let red_array = DataArray::<f64>::from_vec_named(vec![2, 2], vec!["y", "x"], vec![80.0; 4])
            .expect("valid red array");
        let red_ds = Dataset::new(DataId::new("B02").expect("valid DataId")).with_array(red_array);
        let params = make_test_params(2);
        let result = corrector
            .apply_correction(vis_ds, Some(&red_ds), params, 634.0)
            .expect("correction with red band should succeed");
        let vals = result
            .array()
            .expect("result should have array")
            .values_as_f64();
        assert!(vals.iter().all(|v| *v > 40.0));
    }

    #[test]
    fn out_of_range_wavelength_returns_original() {
        let lut = make_test_lut();
        let corrector = RayleighCorrector::new(lut);
        let vis_ds = make_vis_ds(vec![50.0; 4]);
        let params = make_test_params(2);
        let result = corrector
            .apply_correction(vis_ds, None, params, 1200.0)
            .expect("correction should succeed");
        let vals = result
            .array()
            .expect("result should have array")
            .values_as_f64();
        assert!(vals.iter().all(|v| (*v - 50.0).abs() < 1e-10));
    }

    #[test]
    fn subtract_correction_clamps_to_zero() {
        let mut vis = vec![5.0, 10.0, f64::NAN, 3.0];
        let corr = vec![10.0, 5.0, 100.0, 1.0];
        subtract_correction(&mut vis, &corr);
        assert_eq!(vis[0], 0.0);
        assert_eq!(vis[1], 5.0);
        assert!(vis[2].is_nan());
        assert_eq!(vis[3], 2.0);
    }

    #[test]
    fn parallel_subtract_matches_serial() {
        let n = 20_000;
        let mut vis_par = vec![50.0; n];
        let vis_ser = vec![50.0; n];
        let corr: Vec<f64> = (0..n).map(|i| (i as f64 % 30.0) + 1.0).collect();
        subtract_correction(&mut vis_par, &corr);
        for i in 0..n {
            let expected = (50.0 - corr[i]).max(0.0);
            assert!((vis_par[i] - expected).abs() < 1e-10, "mismatch at {i}");
        }
        let _ = vis_ser;
    }
}
