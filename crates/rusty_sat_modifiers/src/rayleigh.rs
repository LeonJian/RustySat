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
//! - Angle arrays are allocated once and reused.
//! - The 4D LUT coordinate arrays are extracted (moved out) before the
//!   reflectance buffer is consumed by wavelength selection.
//! - The 3D wavelength-adjusted slice is dropped before the final subtraction.
//! - The correction array is freed immediately after the subtraction.
//! - Parallel processing via rayon for large grids (>10k pixels).

use crate::angles::AngleSet;
use crate::astronomy::UtcInstant;
use crate::rayleigh_lut::RayleighLut;
use rusty_sat_core::{DataArray, DataId, Dataset, MetadataValue, Result, RustySatError};

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
    /// The correction is computed and subtracted in-place.
    ///
    /// - `vis_dataset`: visible-band reflectance dataset (owned, consumed)
    /// - `red_dataset`: optional red-band reflectance for cloud relaxation
    /// - `angles`: pre-computed angle set (avoids recomputation)
    /// - `wavelength_nm`: central wavelength of the visible band (nm)
    pub fn apply_correction(
        self,
        vis_dataset: Dataset,
        red_dataset: Option<&Dataset>,
        angles: &AngleSet,
        wavelength_nm: f64,
    ) -> Result<Dataset> {
        let config = self.config;

        // Extract coordinate arrays from the LUT *before* consuming the
        // reflectance buffer.  This avoids keeping the full 4D array alive
        // longer than necessary.
        let sun_zenith_secant = self.lut.sun_zenith_secant.clone();
        let azimuth_difference = self.lut.azimuth_difference.clone();
        let satellite_zenith_secant = self.lut.satellite_zenith_secant.clone();

        // Step 1: Wavelength selection — consumes the 4D reflectance buffer.
        let lut_3d = self.lut.into_wavelength_adjusted(wavelength_nm)?;

        // If LUT is empty (out-of-range wavelength), return the original
        // values unchanged (matching pyspectral's zero-correction behavior).
        if lut_3d.is_empty() {
            return rebuild_dataset(vis_dataset, "rayleigh_corrected");
        }

        // Extract visible reflectance values as f64 (consuming the array).
        let array = vis_dataset
            .into_array()
            .ok_or_else(|| RustySatError::invalid_input("visible dataset has no array data"))?;
        let (height, width) = array.shape_yx()?;
        let (mut vis_values, mask) = array.into_f64_values_and_mask();

        // Extract red band reflectance if available.
        let red_values: Option<Vec<f64>> =
            red_dataset.map(|ds| ds.array().map(|a| a.values_as_f64()).unwrap_or_default());

        // Compute relative azimuth.
        let azidiff = angles.relative_azimuth();

        // Step 2: Interpolate the LUT for each pixel (parallel).
        let mut refl_cor = RayleighLut::interpolate_pixels_parallel(
            &lut_3d,
            &sun_zenith_secant,
            &azimuth_difference,
            &satellite_zenith_secant,
            &angles.sun_zenith,
            &angles.sat_zenith,
            &azidiff,
        );

        // The 3D LUT is no longer needed — drop it explicitly.
        drop(lut_3d);

        // Step 3: Optionally relax correction where cloudy.
        if let Some(ref red_vals) = red_values {
            relax_cloudy_correction(red_vals, &mut refl_cor);
        }

        // Step 4: Optionally reduce at high zenith.
        if config.reduce_strength > 0.0 {
            reduce_high_zenith(
                &angles.sun_zenith,
                &mut refl_cor,
                config.reduce_lim_low,
                config.reduce_lim_high,
                config.reduce_strength,
            );
        }

        // Step 5: Subtract correction from visible reflectance (in-place, parallel).
        subtract_correction(&mut vis_values, &refl_cor);

        // The correction array is freed here.
        drop(refl_cor);

        // Build the output dataset.
        let mut result_array =
            DataArray::<f64>::from_vec_named(vec![height, width], vec!["y", "x"], vis_values)?;

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
    let params = AngleParams::from_dataset_area(area_attr, &x_coords, &y_coords, utc)?;
    let angles = params.compute_angles();

    corrector.apply_correction(vis_dataset, red_dataset, &angles, wavelength_nm)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::angles::AngleSet;

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

    fn make_angles(n: usize) -> AngleSet {
        AngleSet {
            sat_azimuth: vec![180.0; n],
            sat_zenith: vec![10.0; n],
            sun_azimuth: vec![0.0; n],
            sun_zenith: vec![40.0; n],
        }
    }

    fn make_vis_ds(values: Vec<f64>) -> Dataset {
        let n = values.len();
        let side = (n as f64).sqrt() as usize;
        let array = DataArray::<f64>::from_vec_named(vec![side, side], vec!["y", "x"], values)
            .expect("valid test array");
        Dataset::new(DataId::new("B03").expect("valid DataId")).with_array(array)
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
        let angles = make_angles(4);
        let result = corrector
            .apply_correction(vis_ds, None, &angles, 634.0)
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
        let angles = make_angles(4);
        let result = corrector
            .apply_correction(vis_ds, Some(&red_ds), &angles, 634.0)
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
        let angles = make_angles(4);
        let result = corrector
            .apply_correction(vis_ds, None, &angles, 1200.0)
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
