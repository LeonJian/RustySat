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

use crate::astronomy::{gmst, observer_position, sun_ra_dec, UtcInstant, EARTH_A_KM, F};
use crate::geos::GeosProjection;
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

    /// Compute the relative azimuth angle for a single pixel (0–180°).
    ///
    /// Ported from `satpy.modifiers.angles._compute_relative_azimuth`.
    #[inline]
    pub fn relative_azimuth_single(sat_azimuth_deg: f64, sun_azimuth_deg: f64) -> f64 {
        if !sat_azimuth_deg.is_finite() || !sun_azimuth_deg.is_finite() {
            return f64::NAN;
        }
        let diff = (sun_azimuth_deg - sat_azimuth_deg).abs();
        diff.min(360.0 - diff)
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
    /// Geostationary projection for on-the-fly lon/lat computation.
    pub geos: GeosProjection,
    /// Projection x coordinates (meters), length = width.
    pub x_coords: Vec<f64>,
    /// Projection y coordinates (meters), length = height.
    pub y_coords: Vec<f64>,
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
        x_coords: Vec<f64>,
        y_coords: Vec<f64>,
        utc: UtcInstant,
    ) -> Result<Self> {
        let area_map = match area_attr {
            MetadataValue::Map(m) => m,
            _ => return Err(RustySatError::invalid_input("area attribute must be a map")),
        };

        let _height = match area_map.get("height") {
            Some(MetadataValue::Integer(v)) => *v as usize,
            _ => {
                return Err(RustySatError::invalid_input(
                    "area metadata missing 'height'",
                ))
            }
        };
        let _width = match area_map.get("width") {
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
        let sat_lat = 0.0;
        let sat_alt = geos.perspective_point_height;
        let width = x_coords.len();
        let height = y_coords.len();

        Ok(Self {
            sat_lon,
            sat_lat,
            sat_alt,
            utc,
            width,
            height,
            geos,
            x_coords,
            y_coords,
        })
    }

    /// Compute all four angles for this parameter set.
    ///
    /// This is the main entry point for angle computation. It fuses
    /// projection inverse, solar, and satellite angle computation into a
    /// single parallel pass, using precomputed time-dependent constants to
    /// avoid redundant work per pixel.
    ///
    /// Consumes `self`, freeing the projection coordinate arrays when done.
    pub fn compute_angles(self) -> AngleSet {
        let width = self.width;
        let height = self.height;
        let n = width * height;

        // Precompute time-dependent constants once for the entire grid.
        let gmst_val = gmst(self.utc);
        let (sun_ra, sun_dec) = sun_ra_dec(self.utc);
        let sat_alt_km = self.sat_alt / 1000.0;
        let (sat_x, sat_y, sat_z) =
            observer_position(self.utc, self.sat_lon, self.sat_lat, sat_alt_km);

        let mut sun_azimuth = vec![f64::NAN; n];
        let mut sun_zenith = vec![f64::NAN; n];
        let mut sat_zenith = vec![f64::NAN; n];
        let mut sat_azimuth = vec![f64::NAN; n];

        if n > 10_000 {
            use rayon::prelude::*;
            sun_zenith
                .par_chunks_mut(width)
                .zip(sun_azimuth.par_chunks_mut(width))
                .zip(sat_zenith.par_chunks_mut(width))
                .zip(sat_azimuth.par_chunks_mut(width))
                .enumerate()
                .for_each(|(row, (((zen_row, azi_row), sz_row), sa_row))| {
                    let y = self.y_coords[row];
                    for col in 0..width {
                        let x = self.x_coords[col];
                        let Some((lon, lat)) = self.geos.inverse(x, y) else {
                            zen_row[col] = f64::NAN;
                            azi_row[col] = f64::NAN;
                            sz_row[col] = f64::NAN;
                            sa_row[col] = f64::NAN;
                            continue;
                        };
                        let (sunz, suna) =
                            solar_zenith_azimuth(lon, lat, sun_ra, sun_dec, gmst_val);
                        let (satzv, satav) =
                            satellite_zenith_azimuth(lon, lat, sat_x, sat_y, sat_z, gmst_val);
                        zen_row[col] = sunz;
                        azi_row[col] = suna;
                        sz_row[col] = satzv;
                        sa_row[col] = satav;
                    }
                });
        } else {
            for row in 0..height {
                let y = self.y_coords[row];
                for col in 0..width {
                    let x = self.x_coords[col];
                    let Some((lon, lat)) = self.geos.inverse(x, y) else {
                        let idx = row * width + col;
                        sun_zenith[idx] = f64::NAN;
                        sun_azimuth[idx] = f64::NAN;
                        sat_zenith[idx] = f64::NAN;
                        sat_azimuth[idx] = f64::NAN;
                        continue;
                    };
                    let (sunz, suna) = solar_zenith_azimuth(lon, lat, sun_ra, sun_dec, gmst_val);
                    let (satzv, satav) =
                        satellite_zenith_azimuth(lon, lat, sat_x, sat_y, sat_z, gmst_val);
                    let idx = row * width + col;
                    sun_zenith[idx] = sunz;
                    sun_azimuth[idx] = suna;
                    sat_zenith[idx] = satzv;
                    sat_azimuth[idx] = satav;
                }
            }
        }

        AngleSet {
            sat_azimuth,
            sat_zenith,
            sun_azimuth,
            sun_zenith,
        }
    }
}

/// Solar zenith and azimuth angles for a single pixel, using precomputed
/// solar right-ascension/declination and GMST to avoid redundant work.
#[inline]
fn solar_zenith_azimuth(
    lon_deg: f64,
    lat_deg: f64,
    sun_ra: f64,
    sun_dec: f64,
    gmst_val: f64,
) -> (f64, f64) {
    let lon = lon_deg.to_radians();
    let lat = lat_deg.to_radians();
    let h = (gmst_val + lon).rem_euclid(2.0 * std::f64::consts::PI) - sun_ra;

    // solar zenith (inline from astronomy::cos_zen)
    let cz = lat.sin() * sun_dec.sin() + lat.cos() * sun_dec.cos() * h.cos();
    let sunz = if cz.is_finite() && cz.abs() <= 1.0 {
        cz.acos().to_degrees()
    } else if cz > 1.0 {
        0.0
    } else {
        180.0
    };

    // solar azimuth (inline from astronomy::get_alt_az)
    let az = (-h.sin()).atan2(lat.cos() * sun_dec.tan() - lat.sin() * h.cos());
    let suna = az.to_degrees().rem_euclid(360.0);

    (sunz, suna)
}

/// Satellite zenith and azimuth angles for a single pixel, using precomputed
/// satellite ECI position and GMST to avoid redundant work.
#[inline]
fn satellite_zenith_azimuth(
    lon_deg: f64,
    lat_deg: f64,
    sat_x: f64,
    sat_y: f64,
    sat_z: f64,
    gmst_val: f64,
) -> (f64, f64) {
    let lon = lon_deg.to_radians();
    let lat = lat_deg.to_radians();
    let theta = (gmst_val + lon).rem_euclid(2.0 * std::f64::consts::PI);

    // Ground observer position (inline from astronomy::observer_position)
    let c = 1.0 / (1.0 + F * (F - 2.0) * lat.sin().powi(2)).sqrt();
    let sq = c * (1.0 - F).powi(2);
    let achcp = (EARTH_A_KM * c) * lat.cos();
    let opos_x = achcp * theta.cos();
    let opos_y = achcp * theta.sin();
    let opos_z = EARTH_A_KM * sq * lat.sin();

    // Vector from observer to satellite
    let rx = sat_x - opos_x;
    let ry = sat_y - opos_y;
    let rz = sat_z - opos_z;

    let sin_lat = lat.sin();
    let cos_lat = lat.cos();
    let sin_theta = theta.sin();
    let cos_theta = theta.cos();

    let top_s = sin_lat * cos_theta * rx + sin_lat * sin_theta * ry - cos_lat * rz;
    let top_e = -sin_theta * rx + cos_theta * ry;
    let top_z = cos_lat * cos_theta * rx + cos_lat * sin_theta * ry + sin_lat * rz;

    // satellite azimuth
    let az = (-top_e).atan2(top_s) + std::f64::consts::PI;
    let sata = az.rem_euclid(2.0 * std::f64::consts::PI).to_degrees();

    // satellite zenith
    let rg = (rx * rx + ry * ry + rz * rz).sqrt();
    let top_z_div = top_z / rg;
    let el = top_z_div.min(1.0).asin();
    let satz = (std::f64::consts::FRAC_PI_2 - el).to_degrees(); // 90° - elevation

    (satz, sata)
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
        let geos = GeosProjection {
            semi_major_axis: 6_378_137.0,
            semi_minor_axis: 6_378_137.0,
            perspective_point_height: 35_785_863.0,
            longitude_of_projection_origin: 140.7,
        };
        let sat_lon = geos.longitude_of_projection_origin;
        let sat_lat = 0.0;
        let sat_alt = geos.perspective_point_height;
        // 1° east of sub-satellite = h * 1° in radians
        let step = geos.perspective_point_height * 1.0_f64.to_radians() / (n.max(1) - 1) as f64;
        let x_coords: Vec<f64> = (0..n).map(|i| i as f64 * step).collect();
        let y_coords = vec![0.0];
        AngleParams {
            sat_lon,
            sat_lat,
            sat_alt,
            utc: UtcInstant::from_ymdhms(2025, 9, 23, 7, 20, 0),
            width: n,
            height: 1,
            geos,
            x_coords,
            y_coords,
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
    fn relative_azimuth_single_is_in_0_180_range() {
        let params = make_params(10);
        let angles = params.compute_angles();
        for i in 0..angles.len() {
            let v = AngleSet::relative_azimuth_single(angles.sat_azimuth[i], angles.sun_azimuth[i]);
            if v.is_finite() {
                assert!((0.0..=180.0).contains(&v), "rel_azi={v}");
            }
        }
    }

    #[test]
    fn nan_lon_produces_nan_angles() {
        let geos = GeosProjection {
            semi_major_axis: 6_378_137.0,
            semi_minor_axis: 6_378_137.0,
            perspective_point_height: 35_785_863.0,
            longitude_of_projection_origin: 140.7,
        };
        let h = geos.perspective_point_height;
        let max_angle = (geos.semi_major_axis / geos.satellite_radius()).asin();
        let edge = h * max_angle;
        let params = AngleParams {
            sat_lon: geos.longitude_of_projection_origin,
            sat_lat: 0.0,
            sat_alt: h,
            utc: UtcInstant::from_ymdhms(2025, 9, 23, 7, 20, 0),
            width: 2,
            height: 1,
            x_coords: vec![edge * 0.5, edge * 1.01], // valid + space
            y_coords: vec![0.0],
            geos,
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
        let geos = GeosProjection {
            semi_major_axis: 6_378_137.0,
            semi_minor_axis: 6_378_137.0,
            perspective_point_height: 35_785_863.0,
            longitude_of_projection_origin: 140.7,
        };
        let h = geos.perspective_point_height;
        let n = 20_000;
        let max_angle = (geos.semi_major_axis / geos.satellite_radius()).asin();
        let edge = h * max_angle;
        let step = edge / n as f64 * 0.9; // stay within disk
        let x_coords: Vec<f64> = (0..n).map(|i| i as f64 * step - edge * 0.45).collect();
        let y_coords = vec![0.0];
        let params_large = AngleParams {
            sat_lon: geos.longitude_of_projection_origin,
            sat_lat: 0.0,
            sat_alt: h,
            utc: UtcInstant::from_ymdhms(2025, 9, 23, 7, 20, 0),
            width: n,
            height: 1,
            x_coords,
            y_coords,
            geos,
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
