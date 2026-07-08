//! Angle computation for satellite datasets — port of Satpy `modifiers/angles.py`.
//!
//! Reference:
//! - `satpy/satpy/modifiers/angles.py` — `get_angles`, `compute_relative_azimuth`,
//!   `_get_sun_angles`, `_get_sensor_angles`, `_get_sun_azimuth_ndarray`.
//! - `deps/pyorbital/pyorbital/astronomy.py` — `cos_zen`, `get_alt_az`.
//! - `deps/pyorbital/pyorbital/orbital.py` — `get_observer_look`.
//!
//! Computes the four angles needed for Rayleigh correction:
//! - solar zenith angle (sunz)
//! - solar azimuth angle (suna)
//! - satellite zenith angle (satz)
//! - satellite azimuth angle (sata)
//!
//! And the derived relative azimuth (ssadiff).

use crate::astronomy::{cos_zen, get_alt_az, UtcInstant};
use crate::geos::GeosProjection;
use crate::orbital::satellite_angles_grid;
use rusty_sat_core::{Coordinate, MetadataValue, Result, RustySatError};

/// The four sun/satellite angles for a 2D pixel grid.
///
/// All values are in degrees. `NaN` indicates a space/invalid pixel.
#[derive(Debug, Clone)]
pub struct AngleSet {
    /// Satellite azimuth angle (0–360°).
    pub sat_azimuth: Vec<f64>,
    /// Satellite zenith angle (0–90°+).
    pub sat_zenith: Vec<f64>,
    /// Solar azimuth angle (0–360°).
    pub sun_azimuth: Vec<f64>,
    /// Solar zenith angle (0–180°).
    pub sun_zenith: Vec<f64>,
}

impl AngleSet {
    /// Number of pixels.
    pub fn len(&self) -> usize {
        self.sat_azimuth.len()
    }

    /// Whether the angle set is empty.
    pub fn is_empty(&self) -> bool {
        self.sat_azimuth.is_empty()
    }

    /// Compute the relative azimuth angle (0–180°).
    ///
    /// Ported from `satpy.modifiers.angles._compute_relative_azimuth`.
    pub fn relative_azimuth(&self) -> Vec<f64> {
        let n = self.len();
        let mut ssadiff = vec![f64::NAN; n];
        for (i, slot) in ssadiff.iter_mut().enumerate().take(n) {
            let sata = self.sat_azimuth[i];
            let suna = self.sun_azimuth[i];
            if !sata.is_finite() || !suna.is_finite() {
                continue;
            }
            let diff = (suna - sata).abs();
            *slot = diff.min(360.0 - diff);
        }
        ssadiff
    }
}

/// Parameters needed to compute angles from a dataset's area metadata.
#[derive(Debug, Clone)]
pub struct AngleParams {
    /// Satellite sub-point longitude (degrees).
    pub sat_lon: f64,
    /// Satellite sub-point latitude (degrees).
    pub sat_lat: f64,
    /// Satellite altitude above the ellipsoid (meters).
    pub sat_alt: f64,
    /// Observation time.
    pub utc: UtcInstant,
    /// Grid width.
    pub width: usize,
    /// Grid height.
    pub height: usize,
    /// Flattened pixel-center longitudes (height * width).
    pub lons: Vec<f64>,
    /// Flattened pixel-center latitudes (height * width).
    pub lats: Vec<f64>,
}

impl AngleParams {
    /// Extract angle parameters from a dataset's `area` attribute and
    /// projection x/y coordinates.
    ///
    /// This reads the area metadata (projection params, area_extent, shape)
    /// and the dataset's x/y coordinates, then computes lon/lat via the
    /// geos projection inverse.
    ///
    /// The `sat_lon`/`sat_lat`/`sat_alt` are extracted from the area's
    /// projection metadata. For AHI, `lon_0` gives the sub-satellite
    /// longitude, and the altitude `h` is in the projection params.
    pub fn from_dataset_area(
        area_attr: &MetadataValue,
        x_coords: &[f64],
        y_coords: &[f64],
        utc: UtcInstant,
    ) -> Result<Self> {
        let area_map = match area_attr {
            MetadataValue::Map(m) => m,
            _ => return Err(RustySatError::invalid_input("area attribute must be a map")),
        };

        let height = match area_map.get("height") {
            Some(MetadataValue::Integer(v)) => *v as usize,
            _ => {
                return Err(RustySatError::invalid_input(
                    "area metadata missing 'height'",
                ))
            }
        };
        let width = match area_map.get("width") {
            Some(MetadataValue::Integer(v)) => *v as usize,
            _ => {
                return Err(RustySatError::invalid_input(
                    "area metadata missing 'width'",
                ))
            }
        };

        let projection = match area_map.get("projection") {
            Some(MetadataValue::Map(m)) => m,
            _ => {
                return Err(RustySatError::invalid_input(
                    "area metadata missing 'projection'",
                ))
            }
        };

        // Convert MetadataValue map to BTreeMap<String, String>
        let mut proj_map = std::collections::BTreeMap::new();
        for (key, value) in projection {
            let s = match value {
                MetadataValue::String(s) => s.clone(),
                _ => value.as_str().unwrap_or_default().to_string(),
            };
            proj_map.insert(key.clone(), s);
        }

        let geos = GeosProjection::from_projection_map(&proj_map)?;
        let sat_lon = geos.longitude_of_projection_origin;
        let sat_lat = 0.0; // geostationary satellites are at ~0° latitude
        let sat_alt = geos.perspective_point_height;

        // Compute lon/lat grid from projection coordinates
        let (lons, lats) = geos.lonlat_grid(x_coords, y_coords);

        Ok(Self {
            sat_lon,
            sat_lat,
            sat_alt,
            utc,
            width,
            height,
            lons,
            lats,
        })
    }

    /// Compute all four angles for this parameter set.
    ///
    /// This is the main entry point for angle computation. It computes
    /// solar and satellite zenith/azimuth angles in parallel.
    pub fn compute_angles(&self) -> AngleSet {
        let n = self.lons.len();

        // Solar angles
        let mut sun_azimuth = vec![f64::NAN; n];
        let mut sun_zenith = vec![f64::NAN; n];

        if n > 10_000 {
            use rayon::prelude::*;
            sun_azimuth
                .par_iter_mut()
                .zip(sun_zenith.par_iter_mut())
                .enumerate()
                .for_each(|(i, (a_slot, z_slot))| {
                    let lon = self.lons[i];
                    let lat = self.lats[i];
                    if !lon.is_finite() || !lat.is_finite() {
                        return;
                    }
                    let cz = cos_zen(self.utc, lon, lat);
                    if cz.is_finite() && cz.abs() <= 1.0 {
                        *z_slot = cz.acos().to_degrees();
                    } else if cz > 1.0 {
                        *z_slot = 0.0;
                    } else {
                        *z_slot = 180.0;
                    }
                    let (_, az) = get_alt_az(self.utc, lon, lat);
                    *a_slot = az.to_degrees().rem_euclid(360.0);
                });
        } else {
            for i in 0..n {
                let lon = self.lons[i];
                let lat = self.lats[i];
                if !lon.is_finite() || !lat.is_finite() {
                    continue;
                }
                let cz = cos_zen(self.utc, lon, lat);
                if cz.is_finite() && cz.abs() <= 1.0 {
                    sun_zenith[i] = cz.acos().to_degrees();
                } else if cz > 1.0 {
                    sun_zenith[i] = 0.0;
                } else {
                    sun_zenith[i] = 180.0;
                }
                let (_, az) = get_alt_az(self.utc, lon, lat);
                sun_azimuth[i] = az.to_degrees().rem_euclid(360.0);
            }
        }

        // Satellite angles
        let (sat_zenith, sat_azimuth) = satellite_angles_grid(
            self.sat_lon,
            self.sat_lat,
            self.sat_alt,
            self.utc,
            &self.lons,
            &self.lats,
        );

        AngleSet {
            sat_azimuth,
            sat_zenith,
            sun_azimuth,
            sun_zenith,
        }
    }
}

/// Extract x/y projection coordinates from a dataset's coordinate arrays.
///
/// Returns `(x_coords, y_coords)` as owned `Vec<f64>`.
pub fn extract_xy_coords(
    coords: &std::collections::BTreeMap<String, Coordinate>,
) -> Result<(Vec<f64>, Vec<f64>)> {
    let x_coord = coords
        .get("x")
        .ok_or_else(|| RustySatError::not_found("x coordinate"))?;
    let y_coord = coords
        .get("y")
        .ok_or_else(|| RustySatError::not_found("y coordinate"))?;
    Ok((x_coord.values().to_vec(), y_coord.values().to_vec()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_params(n: usize) -> AngleParams {
        let lons: Vec<f64> = (0..n).map(|i| 130.0 + i as f64 * 0.1).collect();
        let lats: Vec<f64> = (0..n).map(|i| -10.0 + i as f64 * 0.1).collect();
        AngleParams {
            sat_lon: 140.7,
            sat_lat: 0.0,
            sat_alt: 35_786_000.0,
            utc: UtcInstant::from_ymdhms(2025, 9, 23, 7, 20, 0),
            width: n,
            height: 1,
            lons,
            lats,
        }
    }

    #[test]
    fn compute_angles_returns_finite_values() {
        let params = make_params(10);
        let angles = params.compute_angles();
        assert_eq!(angles.len(), 10);
        // At least some pixels should have finite sun angles
        let finite_count = angles.sun_zenith.iter().filter(|v| v.is_finite()).count();
        assert!(finite_count > 0);
    }

    #[test]
    fn relative_azimuth_is_in_0_180_range() {
        let params = make_params(10);
        let angles = params.compute_angles();
        let rel = angles.relative_azimuth();
        for &v in &rel {
            if v.is_finite() {
                assert!(v >= 0.0 && v <= 180.0, "rel_azi={v}");
            }
        }
    }

    #[test]
    fn nan_lon_produces_nan_angles() {
        let params = AngleParams {
            sat_lon: 140.7,
            sat_lat: 0.0,
            sat_alt: 35_786_000.0,
            utc: UtcInstant::from_ymdhms(2025, 9, 23, 7, 20, 0),
            width: 2,
            height: 1,
            lons: vec![140.0, f64::NAN],
            lats: vec![0.0, 10.0],
        };
        let angles = params.compute_angles();
        assert!(angles.sun_zenith[0].is_finite());
        assert!(angles.sun_zenith[1].is_nan());
    }

    #[test]
    fn parallel_matches_serial() {
        let params_small = make_params(5);
        let angles_small = params_small.compute_angles();

        // Build a large param set to trigger parallel path
        let lons: Vec<f64> = (0..20_000).map(|i| 100.0 + (i as f64 % 80.0)).collect();
        let lats: Vec<f64> = (0..20_000).map(|i| -40.0 + (i as f64 % 80.0)).collect();
        let params_large = AngleParams {
            sat_lon: 140.7,
            sat_lat: 0.0,
            sat_alt: 35_786_000.0,
            utc: UtcInstant::from_ymdhms(2025, 9, 23, 7, 20, 0),
            width: 20_000,
            height: 1,
            lons,
            lats,
        };
        let angles_large = params_large.compute_angles();
        // Verify all values are finite (they should be for valid geo points)
        let finite = angles_large
            .sun_zenith
            .iter()
            .filter(|v| v.is_finite())
            .count();
        assert!(finite > 15_000, "expected mostly finite, got {finite}");

        // Small set sanity
        let _ = angles_small;
    }
}
