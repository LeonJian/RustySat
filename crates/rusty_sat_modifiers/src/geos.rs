//! Geostationary projection inverse — converts geos projection
//! coordinates (meters) to geographic lon/lat (degrees).
//!
//! Reference:
//! - PROJ `geos` projection source code (spherical forward/inverse).
//! - `deps/pyresample/pyresample/geometry.py` — `AreaDefinition.get_lonlats`.
//! - `satpy/satpy/readers/ahi_hsd.py` — area extent computation.
//! - `crates/rusty_sat_readers/src/ahi_hsd.rs` — `geos_area_extent` uses
//!   `h` (height above surface) as the projection coordinate scale factor:
//!   `proj_coord = scanning_angle_radians * h`.
//!
//! ## Convention
//!
//! The existing Rusty Sat code stores projection coordinates as
//! `scanning_angle_radians * h`, where `h` is the perspective point height
//! above the surface.  In the standard PROJ geos convention, coordinates are
//! `(a + h) * tan(scanning_angle)`.  We recover the scanning angle from the
//! stored coordinates and then convert to geocentric lon/lat using the
//! proper geostationary perspective geometry.
//!
//! From the PROJ geos forward (spherical), the direction vector from the
//! satellite is:
//! ```text
//!   Vx = cos(φ) * sin(λ_s)     // λ_s = geocentric lon offset
//!   Vy = sin(φ)                 // φ = geocentric latitude
//!   Vz = cos(φ) * cos(λ_s)
//! ```
//!
//! The column scanning angle `θ_x = atan(Vx / Vz) = λ_s` and the line
//! scanning angle `θ_y = atan(Vy / Vz) = atan(tan(φ) / cos(λ_s))`.
//!
//! Inverting: `λ_s = θ_x`, `φ = atan(tan(θ_y) * cos(θ_x))`.

use rusty_sat_core::{Result, RustySatError};

/// Parameters describing a geostationary projection.
#[derive(Debug, Clone, PartialEq)]
pub struct GeosProjection {
    /// Semi-major axis (equatorial radius) in meters.
    pub semi_major_axis: f64,
    /// Semi-minor axis (polar radius) in meters.
    pub semi_minor_axis: f64,
    /// Perspective point height above the ellipsoid in meters.
    pub perspective_point_height: f64,
    /// Longitude of projection origin (sub-satellite point) in degrees.
    pub longitude_of_projection_origin: f64,
}

impl GeosProjection {
    /// Create from the Satpy/AHI `geos_projection` BTreeMap of string values.
    pub fn from_projection_map(
        projection: &std::collections::BTreeMap<String, String>,
    ) -> Result<Self> {
        let parse = |key: &str| -> Result<f64> {
            projection
                .get(key)
                .and_then(|v| v.parse::<f64>().ok())
                .ok_or_else(|| {
                    RustySatError::invalid_input(format!(
                        "geos projection missing or invalid '{key}'"
                    ))
                })
        };
        let a = parse("a")?;
        let b = parse("b")?;
        let h = parse("h")?;
        let lon_0 = parse("lon_0")?;
        Ok(Self {
            semi_major_axis: a,
            semi_minor_axis: b,
            perspective_point_height: h,
            longitude_of_projection_origin: lon_0,
        })
    }

    /// Total distance from Earth's center to the satellite (meters).
    pub fn satellite_radius(&self) -> f64 {
        self.semi_major_axis + self.perspective_point_height
    }

    /// Whether the Earth model is spherical (a == b).
    pub fn is_spherical(&self) -> bool {
        (self.semi_major_axis - self.semi_minor_axis).abs() < 1.0
    }

    /// Maximum scanning angle (radians) before the horizon is reached.
    ///
    /// Pixels beyond this angle are space pixels.
    fn max_scanning_angle(&self) -> f64 {
        let d = self.satellite_radius();
        let a = self.semi_major_axis;
        (a / d).asin()
    }

    /// Inverse transform: geos projection x/y (meters) → lon/lat (degrees).
    ///
    /// Uses the scanning-angle convention from the existing Rusty Sat code:
    /// `θ = proj_coord / h` (radians), then converts to geocentric lon/lat.
    ///
    /// Returns `(lon_deg, lat_deg)` or `None` if the point is a space pixel.
    pub fn inverse(&self, x_meters: f64, y_meters: f64) -> Option<(f64, f64)> {
        if !x_meters.is_finite() || !y_meters.is_finite() {
            return None;
        }

        let h = self.perspective_point_height;
        if h <= 0.0 {
            return None;
        }

        // Recover scanning angles (radians) from projection coordinates.
        let theta_x = x_meters / h; // column scanning angle = geocentric lon offset
        let theta_y = y_meters / h; // line scanning angle

        // Visibility check: total scanning angle must be within the horizon.
        let total_angle_sq = theta_x * theta_x + theta_y * theta_y;
        let max_angle = self.max_scanning_angle();
        if total_angle_sq.sqrt() >= max_angle {
            return None; // space pixel
        }

        // Geocentric longitude offset = scanning angle θ_x.
        let lon_offset_rad = theta_x;

        // Geocentric latitude: φ = atan(tan(θ_y) * cos(θ_x))
        let lat_rad = (theta_y.tan() * theta_x.cos()).atan();

        let lon_deg = lon_offset_rad.to_degrees() + self.longitude_of_projection_origin;
        let lat_deg = lat_rad.to_degrees();

        Some((normalize_lon(lon_deg), lat_deg))
    }

    /// Batch inverse: convert arrays of x/y projection coordinates to
    /// lon/lat arrays. Space pixels get `(f64::NAN, f64::NAN)`.
    pub fn inverse_batch(&self, xs: &[f64], ys: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = xs.len().min(ys.len());
        let mut lons = vec![f64::NAN; n];
        let mut lats = vec![f64::NAN; n];
        for i in 0..n {
            if let Some((lon, lat)) = self.inverse(xs[i], ys[i]) {
                lons[i] = lon;
                lats[i] = lat;
            }
        }
        (lons, lats)
    }

    /// Generate lon/lat grids from pixel-center projection coordinates.
    ///
    /// `x_coords` and `y_coords` are 1D arrays of pixel-center projection
    /// coordinates (meters). The output is a flattened `(height * width)`
    /// array of longitudes and latitudes.
    ///
    /// Uses rayon for parallel computation when the grid is large enough.
    pub fn lonlat_grid(&self, x_coords: &[f64], y_coords: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let height = y_coords.len();
        let width = x_coords.len();
        let total = height * width;
        let mut lons = vec![f64::NAN; total];
        let mut lats = vec![f64::NAN; total];

        if total > 10_000 {
            use rayon::prelude::*;
            lons.par_chunks_mut(width)
                .zip(lats.par_chunks_mut(width))
                .enumerate()
                .for_each(|(row, (lon_row, lat_row))| {
                    let y = y_coords[row];
                    for (col, (lon_slot, lat_slot)) in
                        lon_row.iter_mut().zip(lat_row.iter_mut()).enumerate()
                    {
                        let x = x_coords[col];
                        if let Some((lon, lat)) = self.inverse(x, y) {
                            *lon_slot = lon;
                            *lat_slot = lat;
                        }
                    }
                });
        } else {
            for row in 0..height {
                let y = y_coords[row];
                for col in 0..width {
                    let x = x_coords[col];
                    if let Some((lon, lat)) = self.inverse(x, y) {
                        lons[row * width + col] = lon;
                        lats[row * width + col] = lat;
                    }
                }
            }
        }

        (lons, lats)
    }
}

/// Normalize longitude to [-180, 180].
fn normalize_lon(lon: f64) -> f64 {
    let mut lon = lon;
    while lon > 180.0 {
        lon -= 360.0;
    }
    while lon < -180.0 {
        lon += 360.0;
    }
    lon
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn ahi_geos() -> GeosProjection {
        GeosProjection {
            semi_major_axis: 6_378_137.0,
            semi_minor_axis: 6_356_752.3,
            perspective_point_height: 35_785_863.0,
            longitude_of_projection_origin: 140.7,
        }
    }

    fn ahi_spherical() -> GeosProjection {
        GeosProjection {
            semi_major_axis: 6_378_137.0,
            semi_minor_axis: 6_378_137.0,
            perspective_point_height: 35_785_863.0,
            longitude_of_projection_origin: 140.7,
        }
    }

    #[test]
    fn from_projection_map_parses_ahi_params() {
        let mut m = BTreeMap::new();
        m.insert("a".to_string(), "6378137.0".to_string());
        m.insert("b".to_string(), "6356752.3".to_string());
        m.insert("h".to_string(), "35785863.0".to_string());
        m.insert("lon_0".to_string(), "140.7".to_string());
        let g = GeosProjection::from_projection_map(&m).unwrap();
        assert!((g.semi_major_axis - 6_378_137.0).abs() < 1.0);
        assert!((g.longitude_of_projection_origin - 140.7).abs() < 1e-10);
        assert!(!g.is_spherical());
    }

    #[test]
    fn inverse_at_origin_returns_subsatellite_point() {
        let g = ahi_spherical();
        let (lon, lat) = g.inverse(0.0, 0.0).unwrap();
        assert!((lon - 140.7).abs() < 1e-10, "lon={lon}");
        assert!(lat.abs() < 1e-10, "lat={lat}");
    }

    #[test]
    fn inverse_returns_none_for_space_pixels() {
        let g = ahi_spherical();
        let h = g.perspective_point_height;
        // The max scanning angle is arcsin(a / (a+h)) ≈ 8.7°
        // In projection coordinates: h * 8.7° in radians = h * 0.152
        let max_angle = (g.semi_major_axis / g.satellite_radius()).asin();
        let edge = h * max_angle;
        // Just beyond the edge → space pixel
        assert!(g.inverse(edge * 1.01, 0.0).is_none());
        // Well within the disk → valid pixel
        assert!(g.inverse(edge * 0.5, 0.0).is_some());
    }

    #[test]
    fn inverse_at_equator_east_of_ssp() {
        let g = ahi_spherical();
        let h = g.perspective_point_height;
        // 1° east of sub-satellite point
        let theta_x = 1.0_f64.to_radians();
        let x = h * theta_x;
        let (lon, lat) = g.inverse(x, 0.0).unwrap();
        assert!((lon - 141.7).abs() < 0.01, "lon={lon}");
        assert!(lat.abs() < 0.01, "lat={lat}");
    }

    #[test]
    fn inverse_north_of_ssp() {
        let g = ahi_spherical();
        let h = g.perspective_point_height;
        // 1° north of sub-satellite point (line scanning angle)
        let theta_y = 1.0_f64.to_radians();
        let y = h * theta_y;
        let (lon, lat) = g.inverse(0.0, y).unwrap();
        assert!((lon - 140.7).abs() < 0.01, "lon={lon}");
        // At x=0, cos(theta_x) = 1, so lat = theta_y = 1°
        assert!((lat - 1.0).abs() < 0.01, "lat={lat}");
    }

    #[test]
    fn lonlat_grid_shape_matches_input() {
        let g = ahi_spherical();
        let xs = vec![0.0, 100_000.0, 200_000.0];
        let ys = vec![0.0, 100_000.0];
        let (lons, lats) = g.lonlat_grid(&xs, &ys);
        assert_eq!(lons.len(), 6);
        assert_eq!(lats.len(), 6);
    }

    #[test]
    fn lonlat_grid_parallel_matches_serial() {
        let g = ahi_spherical();
        let xs: Vec<f64> = (0..50).map(|i| i as f64 * 100_000.0).collect();
        let ys: Vec<f64> = (0..50).map(|i| i as f64 * 100_000.0).collect();
        let (lons_par, lats_par) = g.lonlat_grid(&xs, &ys);

        let mut lons_ser = vec![f64::NAN; 2500];
        let mut lats_ser = vec![f64::NAN; 2500];
        for (row, &y) in ys.iter().enumerate() {
            for (col, &x) in xs.iter().enumerate() {
                if let Some((lon, lat)) = g.inverse(x, y) {
                    lons_ser[row * 50 + col] = lon;
                    lats_ser[row * 50 + col] = lat;
                }
            }
        }
        for i in 0..2500 {
            if lons_par[i].is_nan() {
                assert!(lons_ser[i].is_nan(), "lon NaN mismatch at {i}");
            } else {
                assert!(
                    (lons_par[i] - lons_ser[i]).abs() < 1e-10,
                    "lon mismatch at {i}"
                );
            }
            if lats_par[i].is_nan() {
                assert!(lats_ser[i].is_nan(), "lat NaN mismatch at {i}");
            } else {
                assert!(
                    (lats_par[i] - lats_ser[i]).abs() < 1e-10,
                    "lat mismatch at {i}"
                );
            }
        }
    }

    #[test]
    fn ellipsoidal_inverse_produces_finite_lonlat() {
        let g = ahi_geos();
        let (lon, lat) = g.inverse(500_000.0, 300_000.0).unwrap();
        assert!(lon.is_finite());
        assert!(lat.is_finite());
    }
}
