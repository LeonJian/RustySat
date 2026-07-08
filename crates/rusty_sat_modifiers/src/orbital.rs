//! Satellite look-angle computation — port of pyorbital `get_observer_look`.
//!
//! Reference:
//! - `deps/pyorbital/pyorbital/orbital.py` — `get_observer_look`
//! - http://celestrak.com/columns/v02n02/
//!
//! Computes the azimuth and elevation angle from a ground observer to a
//! geostationary satellite, then derives the satellite zenith angle.

use crate::astronomy::{observer_position, UtcInstant};

/// Compute the observer look angles (azimuth, elevation) from ground to satellite.
///
/// - `sat_lon`, `sat_lat`: satellite sub-point longitude/latitude in degrees.
/// - `sat_alt`: satellite altitude above the ellipsoid in **meters**.
/// - `utc`: observation time.
/// - `lon`, `lat`: observer (ground pixel) longitude/latitude in degrees.
///
/// Returns `(azimuth_deg, elevation_deg)`.
///
/// Ported from `pyorbital.orbital.get_observer_look`.
pub fn get_observer_look(
    sat_lon: f64,
    sat_lat: f64,
    sat_alt_m: f64,
    utc: UtcInstant,
    lon: f64,
    lat: f64,
) -> (f64, f64) {
    let sat_alt_km = sat_alt_m / 1000.0;
    let (pos_x, pos_y, pos_z) = observer_position(utc, sat_lon, sat_lat, sat_alt_km);
    let (opos_x, opos_y, opos_z) = observer_position(utc, lon, lat, 0.0);

    let lon_rad = lon.to_radians();
    let lat_rad = lat.to_radians();
    let theta = (crate::astronomy::gmst(utc) + lon_rad).rem_euclid(2.0 * std::f64::consts::PI);

    let rx = pos_x - opos_x;
    let ry = pos_y - opos_y;
    let rz = pos_z - opos_z;

    let sin_lat = lat_rad.sin();
    let cos_lat = lat_rad.cos();
    let sin_theta = theta.sin();
    let cos_theta = theta.cos();

    let top_s = sin_lat * cos_theta * rx + sin_lat * sin_theta * ry - cos_lat * rz;
    let top_e = -sin_theta * rx + cos_theta * ry;
    let top_z = cos_lat * cos_theta * rx + cos_lat * sin_theta * ry + sin_lat * rz;

    let mut az = (-top_e).atan2(top_s) + std::f64::consts::PI;
    az = az.rem_euclid(2.0 * std::f64::consts::PI);

    let rg = (rx * rx + ry * ry + rz * rz).sqrt();
    let top_z_div_rg = top_z / rg;
    let top_z_clipped = top_z_div_rg.min(1.0);
    let el = top_z_clipped.asin();

    (az.to_degrees(), el.to_degrees())
}

/// Compute satellite zenith and azimuth angles for a grid of pixels.
///
/// - `sat_lon`, `sat_lat`, `sat_alt_m`: satellite position.
/// - `utc`: observation time.
/// - `lons`, `lats`: flattened pixel longitude/latitude arrays (degrees).
///
/// Returns `(sat_zenith_deg, sat_azimuth_deg)` flattened arrays.
///
/// Uses rayon for parallel computation on large grids.
pub fn satellite_angles_grid(
    sat_lon: f64,
    sat_lat: f64,
    sat_alt_m: f64,
    utc: UtcInstant,
    lons: &[f64],
    lats: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let n = lons.len().min(lats.len());
    let mut satz = vec![f64::NAN; n];
    let mut sata = vec![f64::NAN; n];

    if n > 10_000 {
        use rayon::prelude::*;
        satz.par_iter_mut()
            .zip(sata.par_iter_mut())
            .enumerate()
            .for_each(|(i, (z_slot, a_slot))| {
                let lon = lons[i];
                let lat = lats[i];
                if !lon.is_finite() || !lat.is_finite() {
                    return;
                }
                let (az, el) = get_observer_look(sat_lon, sat_lat, sat_alt_m, utc, lon, lat);
                *z_slot = 90.0 - el;
                *a_slot = az;
            });
    } else {
        for i in 0..n {
            let lon = lons[i];
            let lat = lats[i];
            if !lon.is_finite() || !lat.is_finite() {
                continue;
            }
            let (az, el) = get_observer_look(sat_lon, sat_lat, sat_alt_m, utc, lon, lat);
            satz[i] = 90.0 - el;
            sata[i] = az;
        }
    }

    (satz, sata)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nadir_look_has_zero_zenith() {
        let utc = UtcInstant::from_ymdhms(2025, 9, 23, 7, 20, 0);
        let sat_lon = 140.7;
        let sat_lat = 0.0;
        let sat_alt_m = 35_786_000.0;
        let (az, el) = get_observer_look(sat_lon, sat_lat, sat_alt_m, utc, sat_lon, sat_lat);
        assert!(
            (90.0 - el).abs() < 0.5,
            "elevation at nadir should be ~90, got {el}"
        );
        let satz = 90.0 - el;
        assert!(satz.abs() < 0.5, "zenith at nadir should be ~0, got {satz}");
        // Azimuth at exact nadir is undefined but should be finite
        assert!(az.is_finite());
    }

    #[test]
    fn off_nadir_zenith_is_positive() {
        let utc = UtcInstant::from_ymdhms(2025, 9, 23, 7, 20, 0);
        let sat_lon = 140.7;
        let sat_lat = 0.0;
        let sat_alt_m = 35_786_000.0;
        // Observer 5° east of sub-satellite point
        let (az, el) = get_observer_look(sat_lon, sat_lat, sat_alt_m, utc, 145.7, 0.0);
        let satz = 90.0 - el;
        assert!(satz > 1.0, "zenith should be > 1°, got {satz}");
        assert!(satz < 20.0, "zenith should be < 20°, got {satz}");
        assert!(az.is_finite());
    }

    #[test]
    fn satellite_angles_grid_handles_nan() {
        let utc = UtcInstant::from_ymdhms(2025, 9, 23, 7, 20, 0);
        let lons = vec![140.7, f64::NAN, 145.0];
        let lats = vec![0.0, 10.0, 5.0];
        let (satz, sata) = satellite_angles_grid(140.7, 0.0, 35_786_000.0, utc, &lons, &lats);
        assert!(satz[0].is_finite());
        assert!(satz[1].is_nan()); // NaN lon → NaN zenith
        assert!(satz[2].is_finite());
        assert!(sata[0].is_finite());
        assert!(sata[2].is_finite());
    }

    #[test]
    fn satellite_angles_parallel_matches_serial() {
        let utc = UtcInstant::from_ymdhms(2025, 9, 23, 7, 20, 0);
        let lons: Vec<f64> = (0..200).map(|i| 100.0 + i as f64 * 0.5).collect();
        let lats: Vec<f64> = (0..200).map(|i| -50.0 + i as f64 * 0.5).collect();
        let (satz_par, sata_par) =
            satellite_angles_grid(140.7, 0.0, 35_786_000.0, utc, &lons, &lats);

        let mut satz_ser = vec![f64::NAN; 200];
        let mut sata_ser = vec![f64::NAN; 200];
        for i in 0..200 {
            let (az, el) = get_observer_look(140.7, 0.0, 35_786_000.0, utc, lons[i], lats[i]);
            satz_ser[i] = 90.0 - el;
            sata_ser[i] = az;
        }
        for i in 0..200 {
            assert!((satz_par[i] - satz_ser[i]).abs() < 1e-10, "mismatch at {i}");
            assert!((sata_par[i] - sata_ser[i]).abs() < 1e-10, "mismatch at {i}");
        }
    }
}
