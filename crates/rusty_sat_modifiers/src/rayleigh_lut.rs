//! Rayleigh LUT data model and multilinear interpolation — port of
//! `pyspectral.rayleigh` LUT interpolation.
//!
//! Reference:
//! - `deps/pyspectral/pyspectral/rayleigh.py` — `_get_wavelength_adjusted_lut_rayleigh_reflectance`,
//!   `_interp_rayleigh_refl_by_angles`, `get_reflectance_lut_from_file`.
//! - `deps/pyspectral/pyspectral/tests/data.py` — test LUT data and expected results.
//! - `geotiepoints.multilinear.MultilinearInterpolator` — the interpolation
//!   method used by pyspectral.
//!
//! The LUT is a 4D array indexed as:
//!   `[wavelength, sun_zenith_secant, azimuth_difference, satellite_zenith_secant]`
//!
//! The interpolation takes 3D query points `(sunzsec, 180-azidiff, satzsec)`
//! and performs multilinear (trilinear after wavelength selection) interpolation.
//!
//! ## Memory strategy
//!
//! The LUT is stored as a single contiguous `Vec<f64>` in row-major order.
//! After wavelength selection (which produces a 3D slice), the 3D data is
//! extracted into a separate owned buffer and the full 4D LUT is dropped,
//! freeing memory before the per-pixel interpolation begins.

use rusty_sat_core::{Result, RustySatError};

/// The 4D Rayleigh LUT and its coordinate axes.
///
/// All coordinate arrays are monotonically increasing.
#[derive(Debug, Clone)]
pub struct RayleighLut {
    /// Reflectance values, shape `[n_wvl, n_sunz, n_azid, n_satz]`.
    /// Stored row-major: `index = ((w * n_sunz + s) * n_azid + a) * n_satz + t`.
    pub reflectance: Vec<f64>,
    /// Wavelength coordinate (nm).
    pub wavelengths: Vec<f64>,
    /// Sun zenith secant coordinate.
    pub sun_zenith_secant: Vec<f64>,
    /// Azimuth difference coordinate (degrees).
    pub azimuth_difference: Vec<f64>,
    /// Satellite zenith secant coordinate.
    pub satellite_zenith_secant: Vec<f64>,
}

impl RayleighLut {
    /// Dimensions: `(n_wvl, n_sunz, n_azid, n_satz)`.
    pub fn dims(&self) -> (usize, usize, usize, usize) {
        (
            self.wavelengths.len(),
            self.sun_zenith_secant.len(),
            self.azimuth_difference.len(),
            self.satellite_zenith_secant.len(),
        )
    }

    /// Index into the 4D reflectance array.
    #[inline]
    fn index(&self, w: usize, s: usize, a: usize, t: usize) -> usize {
        let (_, n_sunz, n_azid, n_satz) = self.dims();
        ((w * n_sunz + s) * n_azid + a) * n_satz + t
    }

    /// Get a single LUT value.
    #[inline]
    fn get(&self, w: usize, s: usize, a: usize, t: usize) -> f64 {
        self.reflectance[self.index(w, s, a, t)]
    }

    /// Select a wavelength-adjusted 3D slice of the LUT.
    ///
    /// Performs linear interpolation between the two nearest wavelength
    /// entries, then returns a 3D array of shape `[n_sunz, n_azid, n_satz]`.
    ///
    /// The 4D LUT is consumed to free memory.
    ///
    /// Ported from `pyspectral.rayleigh._get_wavelength_adjusted_lut_rayleigh_reflectance`.
    pub fn into_wavelength_adjusted(self, wavelength_nm: f64) -> Result<Vec<f64>> {
        let (n_wvl, n_sunz, n_azid, n_satz) = self.dims();

        if n_wvl == 0 {
            return Err(RustySatError::invalid_input(
                "LUT has no wavelength entries",
            ));
        }

        // If wavelength is outside the LUT range, return zero reflectance.
        if wavelength_nm < self.wavelengths[0] || wavelength_nm > self.wavelengths[n_wvl - 1] {
            // Return zeros — matches pyspectral behavior for out-of-range wavelengths.
            return Ok(vec![0.0; n_sunz * n_azid * n_satz]);
        }

        let (w_idx, w_factor) = wavelength_index_and_factor(&self.wavelengths, wavelength_nm);

        // Allocate the 3D output
        let total_3d = n_sunz * n_azid * n_satz;
        let mut result = vec![0.0; total_3d];

        // Interpolate between wavelength indices
        for s in 0..n_sunz {
            for a in 0..n_azid {
                for t in 0..n_satz {
                    let v0 = if w_idx > 0 {
                        self.get(w_idx - 1, s, a, t)
                    } else {
                        self.get(0, s, a, t)
                    };
                    let v1 = self.get(w_idx, s, a, t);
                    let idx3d = ((s * n_azid) + a) * n_satz + t;
                    result[idx3d] = w_factor * v0 + (1.0 - w_factor) * v1;
                }
            }
        }

        Ok(result)
    }

    /// Interpolate the Rayleigh reflectance for a set of pixel angles.
    ///
    /// Inputs are flattened pixel arrays:
    /// - `sun_zenith_deg`: solar zenith angle (degrees)
    /// - `sat_zenith_deg`: satellite zenith angle (degrees)
    /// - `azidiff_deg`: relative azimuth angle (degrees, 0–180)
    /// - `lut_3d`: wavelength-adjusted 3D LUT from `into_wavelength_adjusted`
    ///
    /// Returns the reflectance correction (0–100 range) for each pixel.
    ///
    /// Ported from `pyspectral.rayleigh._interp_rayleigh_refl_by_angles`.
    pub fn interpolate_pixels(
        lut_3d: &[f64],
        sun_zenith_secant_coords: &[f64],
        azimuth_difference_coords: &[f64],
        satellite_zenith_secant_coords: &[f64],
        sun_zenith_deg: &[f64],
        sat_zenith_deg: &[f64],
        azidiff_deg: &[f64],
    ) -> Vec<f64> {
        let n_sunz = sun_zenith_secant_coords.len();
        let n_azid = azimuth_difference_coords.len();
        let n_satz = satellite_zenith_secant_coords.len();

        let n_pixels = sun_zenith_deg
            .len()
            .min(sat_zenith_deg.len())
            .min(azidiff_deg.len());

        let mut result = vec![0.0; n_pixels];

        // Clip angles to the valid coordinate range.
        let sunz_sec_max = sun_zenith_secant_coords[n_sunz - 1];
        let satz_sec_max = satellite_zenith_secant_coords[n_satz - 1];

        // The clip angle is arccos(1/secant_max) in degrees.
        let sunz_clip_deg = (1.0 / sunz_sec_max).acos().to_degrees();
        let satz_clip_deg = (1.0 / satz_sec_max).acos().to_degrees();

        let sunz_min = sun_zenith_secant_coords[0];
        let sunz_max = sun_zenith_secant_coords[n_sunz - 1];
        let azid_min = azimuth_difference_coords[0];
        let azid_max = azimuth_difference_coords[n_azid - 1];
        let satz_min = satellite_zenith_secant_coords[0];
        let satz_max = satellite_zenith_secant_coords[n_satz - 1];

        for i in 0..n_pixels {
            let sunz_raw = sun_zenith_deg[i];
            let satz_raw = sat_zenith_deg[i];
            let azid_raw = azidiff_deg[i];

            // NaN angles → zero reflectance
            if !sunz_raw.is_finite() || !satz_raw.is_finite() || !azid_raw.is_finite() {
                result[i] = 0.0;
                continue;
            }

            // Clip zenith angles to valid range and convert to secant.
            let sunz = sunz_raw.clamp(0.0, sunz_clip_deg);
            let satz = satz_raw.clamp(0.0, satz_clip_deg);

            let sunzsec = 1.0 / sunz.to_radians().cos();
            let satzsec = 1.0 / satz.to_radians().cos();

            // pyspectral uses (180 - azidiff) as the azimuth coordinate.
            let azid_query = 180.0 - azid_raw;

            // Clamp to LUT coordinate range.
            let sunzsec_q = sunzsec.clamp(sunz_min, sunz_max);
            let azid_q = azid_query.clamp(azid_min, azid_max);
            let satzsec_q = satzsec.clamp(satz_min, satz_max);

            // Trilinear interpolation.
            let (s0, s1, sf) = find_interval(sun_zenith_secant_coords, sunzsec_q);
            let (a0, a1, af) = find_interval(azimuth_difference_coords, azid_q);
            let (t0, t1, tf) = find_interval(satellite_zenith_secant_coords, satzsec_q);

            // Get the 8 corner values from the 3D LUT.
            let c000 = lut_3d[((s0 * n_azid) + a0) * n_satz + t0];
            let c001 = lut_3d[((s0 * n_azid) + a0) * n_satz + t1];
            let c010 = lut_3d[((s0 * n_azid) + a1) * n_satz + t0];
            let c011 = lut_3d[((s0 * n_azid) + a1) * n_satz + t1];
            let c100 = lut_3d[((s1 * n_azid) + a0) * n_satz + t0];
            let c101 = lut_3d[((s1 * n_azid) + a0) * n_satz + t1];
            let c110 = lut_3d[((s1 * n_azid) + a1) * n_satz + t0];
            let c111 = lut_3d[((s1 * n_azid) + a1) * n_satz + t1];

            // Trilinear interpolation formula.
            let c00 = c000 * (1.0 - sf) + c100 * sf;
            let c01 = c001 * (1.0 - sf) + c101 * sf;
            let c10 = c010 * (1.0 - sf) + c110 * sf;
            let c11 = c011 * (1.0 - sf) + c111 * sf;

            let c0 = c00 * (1.0 - af) + c10 * af;
            let c1 = c01 * (1.0 - af) + c11 * af;

            let val = c0 * (1.0 - tf) + c1 * tf;

            // Scale by 100 (pyspectral multiplies by 100 at the end).
            result[i] = (val * 100.0).clamp(0.0, 100.0);
        }

        result
    }

    /// Parallel version of `interpolate_pixels` using rayon.
    ///
    /// Splits the pixel arrays into chunks and processes them in parallel.
    /// This is the preferred path for large full-disk images.
    pub fn interpolate_pixels_parallel(
        lut_3d: &[f64],
        sun_zenith_secant_coords: &[f64],
        azimuth_difference_coords: &[f64],
        satellite_zenith_secant_coords: &[f64],
        sun_zenith_deg: &[f64],
        sat_zenith_deg: &[f64],
        azidiff_deg: &[f64],
    ) -> Vec<f64> {
        let n_pixels = sun_zenith_deg
            .len()
            .min(sat_zenith_deg.len())
            .min(azidiff_deg.len());

        if n_pixels <= 10_000 {
            return Self::interpolate_pixels(
                lut_3d,
                sun_zenith_secant_coords,
                azimuth_difference_coords,
                satellite_zenith_secant_coords,
                sun_zenith_deg,
                sat_zenith_deg,
                azidiff_deg,
            );
        }

        use rayon::prelude::*;
        let mut result = vec![0.0; n_pixels];

        result.par_iter_mut().enumerate().for_each(|(i, slot)| {
            let n_sunz = sun_zenith_secant_coords.len();
            let n_azid = azimuth_difference_coords.len();
            let n_satz = satellite_zenith_secant_coords.len();

            let sunz_raw = sun_zenith_deg[i];
            let satz_raw = sat_zenith_deg[i];
            let azid_raw = azidiff_deg[i];

            if !sunz_raw.is_finite() || !satz_raw.is_finite() || !azid_raw.is_finite() {
                *slot = 0.0;
                return;
            }

            let sunz_sec_max = sun_zenith_secant_coords[n_sunz - 1];
            let satz_sec_max = satellite_zenith_secant_coords[n_satz - 1];
            let sunz_clip_deg = (1.0 / sunz_sec_max).acos().to_degrees();
            let satz_clip_deg = (1.0 / satz_sec_max).acos().to_degrees();

            let sunz_min = sun_zenith_secant_coords[0];
            let sunz_max = sun_zenith_secant_coords[n_sunz - 1];
            let azid_min = azimuth_difference_coords[0];
            let azid_max = azimuth_difference_coords[n_azid - 1];
            let satz_min = satellite_zenith_secant_coords[0];
            let satz_max = satellite_zenith_secant_coords[n_satz - 1];

            let sunz = sunz_raw.clamp(0.0, sunz_clip_deg);
            let satz = satz_raw.clamp(0.0, satz_clip_deg);

            let sunzsec = 1.0 / sunz.to_radians().cos();
            let satzsec = 1.0 / satz.to_radians().cos();
            let azid_query = 180.0 - azid_raw;

            let sunzsec_q = sunzsec.clamp(sunz_min, sunz_max);
            let azid_q = azid_query.clamp(azid_min, azid_max);
            let satzsec_q = satzsec.clamp(satz_min, satz_max);

            let (s0, s1, sf) = find_interval(sun_zenith_secant_coords, sunzsec_q);
            let (a0, a1, af) = find_interval(azimuth_difference_coords, azid_q);
            let (t0, t1, tf) = find_interval(satellite_zenith_secant_coords, satzsec_q);

            let c000 = lut_3d[((s0 * n_azid) + a0) * n_satz + t0];
            let c001 = lut_3d[((s0 * n_azid) + a0) * n_satz + t1];
            let c010 = lut_3d[((s0 * n_azid) + a1) * n_satz + t0];
            let c011 = lut_3d[((s0 * n_azid) + a1) * n_satz + t1];
            let c100 = lut_3d[((s1 * n_azid) + a0) * n_satz + t0];
            let c101 = lut_3d[((s1 * n_azid) + a0) * n_satz + t1];
            let c110 = lut_3d[((s1 * n_azid) + a1) * n_satz + t0];
            let c111 = lut_3d[((s1 * n_azid) + a1) * n_satz + t1];

            let c00 = c000 * (1.0 - sf) + c100 * sf;
            let c01 = c001 * (1.0 - sf) + c101 * sf;
            let c10 = c010 * (1.0 - sf) + c110 * sf;
            let c11 = c011 * (1.0 - sf) + c111 * sf;

            let c0 = c00 * (1.0 - af) + c10 * af;
            let c1 = c01 * (1.0 - af) + c11 * af;

            let val = c0 * (1.0 - tf) + c1 * tf;
            *slot = (val * 100.0).clamp(0.0, 100.0);
        });

        result
    }
}

/// Find the interpolation interval and fractional position for a value
/// within a sorted coordinate array.
///
/// Returns `(i0, i1, frac)` where `i0` and `i1` are the lower and upper
/// indices and `frac` is the fractional position between them (0.0 = at i0,
/// 1.0 = at i1).
#[inline]
fn find_interval(coords: &[f64], value: f64) -> (usize, usize, f64) {
    let n = coords.len();
    if n == 1 {
        return (0, 0, 0.0);
    }
    if value <= coords[0] {
        return (0, 1, 0.0);
    }
    if value >= coords[n - 1] {
        return (n - 2, n - 1, 1.0);
    }

    // Binary search for the interval.
    let mut lo = 0usize;
    let mut hi = n - 1;
    while hi - lo > 1 {
        let mid = lo + (hi - lo) / 2;
        if coords[mid] <= value {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    let span = coords[hi] - coords[lo];
    if span <= 0.0 {
        return (lo, hi, 0.0);
    }
    let frac = (value - coords[lo]) / span;
    (lo, hi, frac)
}

/// Find the wavelength index and interpolation factor.
///
/// Ported from `pyspectral.rayleigh._get_wavelength_index_and_factor`.
fn wavelength_index_and_factor(wvl_coord: &[f64], wvl: f64) -> (usize, f64) {
    let n = wvl_coord.len();
    if n == 0 {
        return (0, 0.0);
    }
    if n == 1 {
        return (0, 0.0);
    }

    // searchsorted (right side) to find insertion point.
    let mut idx = n;
    for (i, &v) in wvl_coord.iter().enumerate() {
        if v > wvl {
            idx = i;
            break;
        }
    }
    if idx == 0 {
        return (0, 0.0);
    }
    if idx >= n {
        return (n - 1, 0.0);
    }

    let wvl1 = wvl_coord[idx - 1];
    let wvl2 = wvl_coord[idx];
    let span = wvl2 - wvl1;
    if span <= 0.0 {
        return (idx, 0.0);
    }
    let factor = (wvl2 - wvl) / span;
    (idx, factor)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a small test LUT matching the pyspectral test data structure.
    /// Uses the same dimensions as `deps/pyspectral/pyspectral/tests/data.py`.
    fn make_test_lut() -> RayleighLut {
        // Use simplified data: 2 wavelengths, 8 sunz, 7 azid, 6 satz.
        let wavelengths = vec![631.0, 636.0];
        let sun_zenith_secant = vec![1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75];
        let azimuth_difference = vec![100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0];
        let satellite_zenith_secant = vec![1.0, 1.25, 1.5, 1.75, 2.0, 2.25];

        let n_wvl = 2;
        let n_sunz = 8;
        let n_azid = 7;
        let n_satz = 6;
        let total = n_wvl * n_sunz * n_azid * n_satz;

        // Create a simple gradient: higher sunz → higher reflectance.
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

    #[test]
    fn wavelength_adjusted_produces_3d_array() {
        let lut = make_test_lut();
        let dims = lut.dims();
        let lut_3d = lut
            .into_wavelength_adjusted(634.0)
            .expect("valid wavelength");
        assert_eq!(lut_3d.len(), dims.1 * dims.2 * dims.3);
    }

    #[test]
    fn wavelength_outside_range_returns_zeros() {
        let lut = make_test_lut();
        let dims = lut.dims();
        let lut_3d = lut
            .into_wavelength_adjusted(1200.0)
            .expect("valid wavelength");
        assert_eq!(lut_3d.len(), dims.1 * dims.2 * dims.3);
        assert!(lut_3d.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn interpolate_returns_finite_values() {
        let lut = make_test_lut();
        let lut_3d = lut
            .into_wavelength_adjusted(634.0)
            .expect("valid wavelength");
        let sun_zenith_deg = vec![50.0, 30.0];
        let sat_zenith_deg = vec![20.0, 10.0];
        let azidiff_deg = vec![140.0, 130.0];

        let result = RayleighLut::interpolate_pixels(
            &lut_3d,
            &make_test_lut().sun_zenith_secant,
            &make_test_lut().azimuth_difference,
            &make_test_lut().satellite_zenith_secant,
            &sun_zenith_deg,
            &sat_zenith_deg,
            &azidiff_deg,
        );

        assert_eq!(result.len(), 2);
        assert!(result.iter().all(|v| v.is_finite()));
        assert!(result.iter().all(|v| *v >= 0.0 && *v <= 100.0));
    }

    #[test]
    fn interpolate_handles_nan_angles() {
        let lut = make_test_lut();
        let lut_3d = lut
            .into_wavelength_adjusted(634.0)
            .expect("valid wavelength");
        let lut2 = make_test_lut();

        let sun_zenith_deg = vec![f64::NAN, 30.0];
        let sat_zenith_deg = vec![20.0, 10.0];
        let azidiff_deg = vec![140.0, 130.0];

        let result = RayleighLut::interpolate_pixels(
            &lut_3d,
            &lut2.sun_zenith_secant,
            &lut2.azimuth_difference,
            &lut2.satellite_zenith_secant,
            &sun_zenith_deg,
            &sat_zenith_deg,
            &azidiff_deg,
        );

        assert_eq!(result.len(), 2);
        assert_eq!(result[0], 0.0); // NaN → zero
        assert!(result[1] > 0.0);
    }

    #[test]
    fn parallel_matches_serial() {
        let lut = make_test_lut();
        let lut_3d = lut
            .clone()
            .into_wavelength_adjusted(634.0)
            .expect("valid wavelength");
        let lut2 = make_test_lut();

        // Create enough pixels to trigger parallel path (>10_000)
        let n = 20_000;
        let sun_zenith_deg: Vec<f64> = (0..n).map(|i| 10.0 + (i as f64 % 60.0)).collect();
        let sat_zenith_deg: Vec<f64> = (0..n).map(|i| 5.0 + (i as f64 % 30.0)).collect();
        let azidiff_deg: Vec<f64> = (0..n).map(|i| 100.0 + (i as f64 % 60.0)).collect();

        let serial = RayleighLut::interpolate_pixels(
            &lut_3d,
            &lut2.sun_zenith_secant,
            &lut2.azimuth_difference,
            &lut2.satellite_zenith_secant,
            &sun_zenith_deg,
            &sat_zenith_deg,
            &azidiff_deg,
        );

        let parallel = RayleighLut::interpolate_pixels_parallel(
            &lut_3d,
            &lut2.sun_zenith_secant,
            &lut2.azimuth_difference,
            &lut2.satellite_zenith_secant,
            &sun_zenith_deg,
            &sat_zenith_deg,
            &azidiff_deg,
        );

        for i in 0..n {
            assert!(
                (serial[i] - parallel[i]).abs() < 1e-12,
                "mismatch at {i}: serial={}, parallel={}",
                serial[i],
                parallel[i]
            );
        }
    }

    #[test]
    fn find_interval_works_at_boundaries() {
        let coords = vec![1.0, 2.0, 3.0, 4.0];
        let (i0, i1, f) = find_interval(&coords, 1.0);
        assert_eq!((i0, i1), (0, 1));
        assert_eq!(f, 0.0);

        let (i0, i1, f) = find_interval(&coords, 4.0);
        assert_eq!((i0, i1), (2, 3));
        assert_eq!(f, 1.0);

        let (i0, i1, f) = find_interval(&coords, 2.5);
        assert_eq!((i0, i1), (1, 2));
        assert!((f - 0.5).abs() < 1e-10);
    }

    #[test]
    fn wavelength_index_and_factor_interpolates() {
        let wvls = vec![631.0, 636.0];
        let (idx, factor) = wavelength_index_and_factor(&wvls, 634.0);
        assert_eq!(idx, 1);
        // factor = (636 - 634) / (636 - 631) = 2/5 = 0.4
        assert!((factor - 0.4).abs() < 1e-10);
    }
}
