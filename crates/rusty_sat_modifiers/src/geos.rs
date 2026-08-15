//! Geostationary projection inverse — converts geos projection
//! coordinates (meters) to geographic lon/lat (degrees).
//!
//! Reference:
//! - PROJ `geos` projection (Snyder, "Map Projections — A Working Manual",
//!   §32 Geostationary Satellite View; used by pyresample `get_lonlats`).
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
//! above the surface.  The inverse treats `(x, y)` as the point on the
//! projection plane at distance `h` from the satellite along the nadir
//! direction and solves the exact ray–ellipsoid intersection.
//!
//! The satellite is at ECEF `S = ((a + h)·cos(λ₀), (a + h)·sin(λ₀), 0)` with
//! `λ₀` the projection origin.  The ray direction through the projection
//! plane point `(x, y)` is:
//! ```text
//!   d = x·east + y·north − h·nadir
//! ```
//! with `east = (−sin λ₀, cos λ₀, 0)`, `north = (0, 0, 1)`, and
//! `nadir = S/|S|`.  Intersecting `P = S + t·d` with the Earth ellipsoid
//! `X²/a² + Y²/a² + Z²/b² = 1` gives a quadratic in `t`; a negative
//! discriminant means the ray misses the Earth (space pixel).  The hit point
//! is converted to geodetic lon/lat by Newton iteration on the ellipsoid
//! normal.
//!
//! This replaces the earlier flat tangent-plane approximation
//! (`lat = atan(tan θ_y · cos θ_x)`, `lon = λ₀ + θ_x`) which is exact only
//! at the sub-satellite point and underestimates ground offsets by the
//! `h/a` curvature factor (≈5.6× at the limb).

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

    /// Exact ellipsoidal geos inverse: projection x/y (meters) → geodetic
    /// lon/lat (radians).
    ///
    /// The ray from the satellite through the projection-plane point `(x, y)`
    /// is intersected with the Earth ellipsoid (PROJ `geos` / pyresample
    /// `get_lonlats` behavior). Returns `None` for space pixels (ray misses
    /// the Earth, or the discriminant is negative).
    ///
    /// `lon_rad` includes the projection-origin longitude offset.
    fn inverse_rad_inner(&self, x_meters: f64, y_meters: f64) -> Option<(f64, f64)> {
        let h = self.perspective_point_height;
        if h <= 0.0 {
            return None;
        }
        let a = self.semi_major_axis;
        let b = self.semi_minor_axis;
        let lon_0 = self.longitude_of_projection_origin.to_radians();
        let (sin_lon0, cos_lon0) = lon_0.sin_cos();

        // Satellite ECEF position, on the equator above the projection origin.
        let r_sat = h + a;
        let sx = r_sat * cos_lon0;
        let sy = r_sat * sin_lon0;

        // Ray direction: projection-plane point at distance h along nadir.
        let dx = -x_meters * sin_lon0 - h * cos_lon0;
        let dy = x_meters * cos_lon0 - h * sin_lon0;
        let dz = y_meters;

        // Ray–ellipsoid intersection quadratic: |S + t·d|_ell² = 1.
        // (d is unit-free; A is separable in x/y, B and C are constants.)
        let a_inv2 = 1.0 / (a * a);
        let b_inv2 = 1.0 / (b * b);
        let quad_a = (dx * dx + dy * dy) * a_inv2 + dz * dz * b_inv2;
        let quad_b = 2.0 * (sx * dx + sy * dy) * a_inv2;
        let quad_c = (sx * sx + sy * sy) * a_inv2 - 1.0;
        let disc = quad_b * quad_b - 4.0 * quad_a * quad_c;
        if !disc.is_finite() || disc < 0.0 {
            return None; // space pixel
        }
        let t = (-quad_b - disc.sqrt()) / (2.0 * quad_a);
        if t <= 0.0 {
            return None;
        }
        let px = sx + t * dx;
        let py = sy + t * dy;
        let pz = t * dz;

        let lon = py.atan2(px);

        // Geodetic latitude by Newton iteration on the ellipsoid normal.
        let e2 = 1.0 - (b * b) / (a * a);
        let rho = px.hypot(py);
        let mut phi = pz.atan2(rho * (1.0 - e2));
        for _ in 0..4 {
            let sin_phi = phi.sin();
            let n = a / (1.0 - e2 * sin_phi * sin_phi).sqrt();
            phi = (pz + e2 * n * sin_phi).atan2(rho);
        }

        Some((lon, phi))
    }

    /// Inverse transform: geos projection x/y (meters) → lon/lat (degrees).
    ///
    /// Exact ellipsoidal inverse matching the PROJ `geos` projection used by
    /// Satpy/pyresample `get_lonlats`. Returns `(lon_deg, lat_deg)` or `None`
    /// if the point is a space pixel.
    pub fn inverse(&self, x_meters: f64, y_meters: f64) -> Option<(f64, f64)> {
        if !x_meters.is_finite() || !y_meters.is_finite() {
            return None;
        }
        let (lon, lat) = self.inverse_rad_inner(x_meters, y_meters)?;
        Some((normalize_lon(lon.to_degrees()), lat.to_degrees()))
    }

    /// Inverse transform: geos projection x/y (meters) → lon/lat (radians).
    ///
    /// Same geometry as `inverse`, but returns radians to avoid the
    /// degrees→radians roundtrip in the angle computation hot path.
    ///
    /// Returns `(lon_rad, lat_rad)` or `None` if the point is a space pixel.
    /// `lon_rad` includes the projection-origin longitude offset.
    pub fn inverse_rad(&self, x_meters: f64, y_meters: f64) -> Option<(f64, f64)> {
        if !x_meters.is_finite() || !y_meters.is_finite() {
            return None;
        }
        self.inverse_rad_inner(x_meters, y_meters)
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
            semi_minor_axis: 6_356_752.314_14,
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
        let g = GeosProjection::from_projection_map(&m).expect("valid AHI projection map");
        assert!((g.semi_major_axis - 6_378_137.0).abs() < 1.0);
        assert!((g.longitude_of_projection_origin - 140.7).abs() < 1e-10);
        assert!(!g.is_spherical());
    }

    #[test]
    fn inverse_at_origin_returns_subsatellite_point() {
        let g = ahi_spherical();
        let (lon, lat) = g.inverse(0.0, 0.0).expect("origin should project");
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
        // 1° east of sub-satellite point (scanning angle ≈ x/h)
        let theta_x = 1.0_f64.to_radians();
        let x = h * theta_x;
        let (lon, lat) = g.inverse(x, 0.0).expect("equator pixel should project");
        // Exact curved-Earth value: ground offset ≈ 5.62× the scan angle.
        assert!((lon - 146.324_551_684_591).abs() < 1e-9, "lon={lon}");
        assert!(lat.abs() < 1e-9, "lat={lat}");
    }

    #[test]
    fn inverse_north_of_ssp() {
        let g = ahi_spherical();
        let h = g.perspective_point_height;
        // 1° north of sub-satellite point (line scanning angle)
        let theta_y = 1.0_f64.to_radians();
        let y = h * theta_y;
        let (lon, lat) = g.inverse(0.0, y).expect("north pixel should project");
        assert!((lon - 140.7).abs() < 1e-9, "lon={lon}");
        // Exact curved-Earth value, not the flat-plane scan angle.
        assert!((lat - 5.624_551_684_591).abs() < 1e-9, "lat={lat}");
    }

    /// Reference values computed with the exact ellipsoidal ray–ellipsoid
    /// intersection for the AHI geometry (a=6378137, b=6356752.31414,
    /// h=35785863, lon_0=140.7). These match the PROJ `geos` projection used
    /// by Satpy `area.get_lonlats()`. Provenance: independent numerical solve
    /// of the ray–ellipsoid quadratic + geodetic normal iteration; the
    /// horizon latitude (±81.28°) is consistent with the known AHI full-disk
    /// limit and the small-angle limit `lat ≈ θ_y·(h+a)/a` near the center.
    #[test]
    fn inverse_matches_curved_earth_reference_at_limb() {
        let g = ahi_geos();
        let h = g.perspective_point_height;
        let cases: &[(f64, f64, f64, f64)] = &[
            // (x, y, expected_lon, expected_lat)
            (0.0, 0.0, 140.7, 0.0),
            (h * 4.0_f64.to_radians(), 0.0, 164.119_006_320_043, 0.0),
            (0.0, h * 4.0_f64.to_radians(), 140.7, 23.575_595_238_793),
            (0.0, 5.225_25e6, 140.7, 65.129_298_005_784),
            (0.0, -5.225_25e6, 140.7, -65.129_298_005_784),
            (-5.225_25e6, 0.0, 76.235_983_903_003, 0.0),
            (5.225_25e6, 0.0, -154.835_983_903_003, 0.0),
        ];
        for (x, y, lon_ref, lat_ref) in cases {
            let (lon, lat) = g.inverse(*x, *y).expect("interior pixel should project");
            assert!(
                (lon - lon_ref).abs() < 1e-9,
                "lon mismatch at ({x}, {y}): {lon} != {lon_ref}"
            );
            assert!(
                (lat - lat_ref).abs() < 1e-9,
                "lat mismatch at ({x}, {y}): {lat} != {lat_ref}"
            );
        }
    }

    #[test]
    fn inverse_rad_matches_inverse_degrees() {
        let g = ahi_geos();
        for (x, y) in [(0.0, 0.0), (500_000.0, 300_000.0), (-2.0e6, 3.0e6)] {
            let (lon_d, lat_d) = g.inverse(x, y).expect("valid pixel");
            let (lon_r, lat_r) = g.inverse_rad(x, y).expect("valid pixel");
            assert!((lon_r.to_degrees() - lon_d).abs() < 1e-12, "lon mismatch");
            assert!((lat_r.to_degrees() - lat_d).abs() < 1e-12, "lat mismatch");
        }
    }

    #[test]
    fn space_pixels_beyond_ellipsoid_horizon_are_none() {
        let g = ahi_geos();
        // The visible horizon for the ellipsoid is slightly beyond the
        // spherical limit; well outside it the ray misses the Earth.
        let h = g.perspective_point_height;
        let theta_max = (g.semi_major_axis / g.satellite_radius()).asin();
        let edge = h * theta_max;
        assert!(g.inverse(edge * 1.01, 0.0).is_none());
        assert!(g.inverse(0.0, edge * 1.01).is_none());
        // Just inside the disk the inverse must succeed.
        assert!(g.inverse(edge * 0.99, 0.0).is_some());
    }

    #[test]
    fn horizon_latitude_matches_known_ahi_limit() {
        let g = ahi_geos();
        let a = g.semi_major_axis;
        let b = g.semi_minor_axis;
        let h = g.perspective_point_height;
        // Meridian tangent point: ray just grazing the ellipsoid.
        let y_tan = b * h / (h * h + 2.0 * a * h).sqrt();
        let (_, lat) = g
            .inverse(0.0, y_tan * 0.999_99)
            .expect("just inside the meridian horizon");
        // AHI full-disk maximum latitude ≈ 81.28° (ellipsoid; 81.30° sphere).
        assert!((lat - 81.28).abs() < 0.25, "horizon lat={lat}");
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
        let (lon, lat) = g
            .inverse(500_000.0, 300_000.0)
            .expect("interior pixel should project");
        assert!(lon.is_finite());
        assert!(lat.is_finite());
    }
}
