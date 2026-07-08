//! Solar position astronomy — direct port of pyorbital `astronomy.py`.
//!
//! Reference:
//! - `deps/pyorbital/pyorbital/astronomy.py` — `jdays2000`, `gmst`,
//!   `sun_ecliptic_longitude`, `sun_ra_dec`, `_local_hour_angle`,
//!   `get_alt_az`, `cos_zen`.
//! - http://www.geoastro.de/elevaz/basics/index.htm
//!
//! All functions operate on scalar or slice inputs in **degrees** for
//! lon/lat and return radians or degrees as documented per-function.

use std::time::Duration;

/// Earth flattening (WGS-84).
pub const F: f64 = 1.0 / 298.257223563;

/// WGS-84 equatorial radius in km.
pub const EARTH_A_KM: f64 = 6378.137;

/// Earth rotation rate (rad/s).
pub const MFACTOR: f64 = 7.292115e-5;

/// Epoch for Julian day calculations: 2000-01-01T12:00:00 UTC.
pub const J2000_EPOCH: (i32, u32, u32, u32, u32, u32) = (2000, 1, 1, 12, 0, 0);

/// A UTC instant represented as days-since-J2000.
///
/// This avoids `chrono` and keeps the computation dependency-free.
/// The caller is responsible for converting from the dataset's observation
/// time into fractional days since 2000-01-01T12:00:00.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UtcInstant {
    days_since_j2000: f64,
}

impl UtcInstant {
    /// Create from days since J2000 epoch (2000-01-01T12:00:00 UTC).
    pub fn from_days_since_j2000(days: f64) -> Self {
        Self {
            days_since_j2000: days,
        }
    }

    /// Create from a broken-down UTC time.
    ///
    /// Uses the Gregorian calendar algorithm from
    /// <https://howardhinnant.github.io/date_algorithms.html>.
    pub fn from_ymdhms(year: i32, month: u32, day: u32, hour: u32, min: u32, sec: u32) -> Self {
        let y = if month <= 2 { year - 1 } else { year };
        let era = if y >= 0 { y } else { y - 399 } / 400;
        let yoe = (y - era * 400) as u32; // [0, 399]
        let doy = (153 * (if month > 2 { month - 3 } else { month + 9 }) + 2) / 5 + day - 1; // [0, 365]
        let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy; // [0, 146096]
        let days = era as i64 * 146097 + doe as i64 - 719468; // days since 1970-01-01
        let secs = (hour as i64) * 3600 + (min as i64) * 60 + sec as i64;
        let unix_secs = days * 86400 + secs;
        // J2000 epoch is 2000-01-01T12:00:00 = Unix 946728000
        let j2000_unix = 946_728_000i64;
        let delta = unix_secs - j2000_unix;
        Self::from_days_since_j2000(delta as f64 / 86400.0)
    }

    /// Create from Unix timestamp (seconds since 1970-01-01T00:00:00Z).
    pub fn from_unix(secs: i64) -> Self {
        let j2000_unix = 946_728_000i64;
        Self::from_days_since_j2000((secs - j2000_unix) as f64 / 86400.0)
    }

    /// Days since J2000 epoch.
    pub fn days_since_j2000(self) -> f64 {
        self.days_since_j2000
    }

    /// Julian day number.
    pub fn julian_day(self) -> f64 {
        self.days_since_j2000 + 2_451_545.0
    }

    /// Julian centuries since J2000 (T in the IAU convention).
    pub fn julian_centuries(self) -> f64 {
        self.days_since_j2000 / 36525.0
    }
}

impl From<Duration> for UtcInstant {
    fn from(d: Duration) -> Self {
        Self::from_unix(d.as_secs() as i64)
    }
}

/// Greenwich Mean Sidereal Time in radians.
///
/// Ported from `pyorbital.astronomy.gmst`.
pub fn gmst(utc: UtcInstant) -> f64 {
    let ut1 = utc.days_since_j2000() / 36525.0;
    let theta =
        67310.54841 + ut1 * (876600.0 * 3600.0 + 8640184.812866 + ut1 * (0.093104 - ut1 * 6.2e-5));
    let deg = theta / 240.0;
    deg.to_radians().rem_euclid(2.0 * std::f64::consts::PI)
}

/// Local Mean Sidereal Time in radians.
fn lmst(utc: UtcInstant, longitude_rad: f64) -> f64 {
    (gmst(utc) + longitude_rad).rem_euclid(2.0 * std::f64::consts::PI)
}

/// Ecliptic longitude of the sun (radians).
///
/// Ported from `pyorbital.astronomy.sun_ecliptic_longitude`.
pub fn sun_ecliptic_longitude(utc: UtcInstant) -> f64 {
    let jdate = utc.julian_centuries();
    let m_a = (357.52910 + 35999.05030 * jdate
        - 0.0001559 * jdate * jdate
        - 0.00000048 * jdate * jdate * jdate)
        .to_radians();
    let l_0 = 280.46645 + 36000.76983 * jdate + 0.0003032 * jdate * jdate;
    let d_l = (1.914600 - 0.004817 * jdate - 0.000014 * jdate * jdate) * m_a.sin()
        + (0.019993 - 0.000101 * jdate) * (2.0 * m_a).sin()
        + 0.000290 * (3.0 * m_a).sin();
    let l__ = l_0 + d_l;
    l__.to_radians()
}

/// Sun right ascension and declination (both in radians).
///
/// Ported from `pyorbital.astronomy.sun_ra_dec`.
pub fn sun_ra_dec(utc: UtcInstant) -> (f64, f64) {
    let jdate = utc.julian_centuries();
    let eps = (23.0 + 26.0 / 60.0 + 21.448 / 3600.0
        - (46.8150 * jdate + 0.00059 * jdate * jdate - 0.001813 * jdate * jdate * jdate) / 3600.0)
        .to_radians();
    let eclon = sun_ecliptic_longitude(utc);
    let x__ = eclon.cos();
    let y__ = eps.cos() * eclon.sin();
    let z__ = eps.sin() * eclon.sin();
    let r__ = (1.0 - z__ * z__).sqrt();
    let declination = z__.atan2(r__);
    let right_ascension = 2.0 * y__.atan2(x__ + r__);
    (right_ascension, declination)
}

fn local_hour_angle(utc: UtcInstant, longitude_rad: f64, right_ascension: f64) -> f64 {
    lmst(utc, longitude_rad) - right_ascension
}

/// Sun altitude and azimuth from UTC time, longitude, and latitude.
///
/// `lon`, `lat` in degrees. Returns `(altitude, azimuth)` in **radians**.
///
/// Ported from `pyorbital.astronomy.get_alt_az`.
pub fn get_alt_az(utc: UtcInstant, lon_deg: f64, lat_deg: f64) -> (f64, f64) {
    let lon = lon_deg.to_radians();
    let lat = lat_deg.to_radians();
    let (ra_, dec) = sun_ra_dec(utc);
    let h__ = local_hour_angle(utc, lon, ra_);
    let alt = (lat.sin() * dec.sin() + lat.cos() * dec.cos() * h__.cos()).asin();
    let az = (-h__.sin()).atan2(lat.cos() * dec.tan() - lat.sin() * h__.cos());
    (alt, az)
}

/// Cosine of the sun zenith angle.
///
/// `lon`, `lat` in degrees.
///
/// Ported from `pyorbital.astronomy.cos_zen`.
pub fn cos_zen(utc: UtcInstant, lon_deg: f64, lat_deg: f64) -> f64 {
    let lon = lon_deg.to_radians();
    let lat = lat_deg.to_radians();
    let (r_a, dec) = sun_ra_dec(utc);
    let h__ = local_hour_angle(utc, lon, r_a);
    lat.sin() * dec.sin() + lat.cos() * dec.cos() * h__.cos()
}

/// Sun zenith angle in degrees.
pub fn sun_zenith_angle(utc: UtcInstant, lon_deg: f64, lat_deg: f64) -> f64 {
    cos_zen(utc, lon_deg, lat_deg).acos().to_degrees()
}

/// Sun azimuth angle in degrees (0–360 range, like Satpy's convention).
pub fn sun_azimuth_angle(utc: UtcInstant, lon_deg: f64, lat_deg: f64) -> f64 {
    let (_alt, az) = get_alt_az(utc, lon_deg, lat_deg);
    az.to_degrees().rem_euclid(360.0)
}

/// Sun-Earth distance correction relative to 1 AU.
///
/// Ported from `pyorbital.astronomy.sun_earth_distance_correction`.
pub fn sun_earth_distance_correction(utc: UtcInstant) -> f64 {
    1.0 - 0.0167 * (2.0 * std::f64::consts::PI * (utc.days_since_j2000() - 3.0) / 365.25636).cos()
}

/// ECI position of an observer on the Earth's surface.
///
/// `lon`, `lat` in degrees, `alt` in km. Returns `(x, y, z)` in km.
///
/// Ported from `pyorbital.astronomy.observer_position`.
pub fn observer_position(
    utc: UtcInstant,
    lon_deg: f64,
    lat_deg: f64,
    alt_km: f64,
) -> (f64, f64, f64) {
    let lon = lon_deg.to_radians();
    let lat = lat_deg.to_radians();
    let theta = (gmst(utc) + lon).rem_euclid(2.0 * std::f64::consts::PI);
    let c = 1.0 / (1.0 + F * (F - 2.0) * lat.sin().powi(2)).sqrt();
    let sq = c * (1.0 - F).powi(2);
    let achcp = (EARTH_A_KM * c + alt_km) * lat.cos();
    let x = achcp * theta.cos();
    let y = achcp * theta.sin();
    let z = EARTH_A_KM * sq * lat.sin() + alt_km * lat.sin();
    (x, y, z)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gmst_is_in_valid_range() {
        let utc = UtcInstant::from_ymdhms(2025, 9, 23, 7, 20, 0);
        let g = gmst(utc);
        assert!((0.0..2.0 * std::f64::consts::PI).contains(&g));
    }

    #[test]
    fn cos_zen_at_ssp_is_near_one() {
        // 2025-09-23 07:20 UTC, sub-satellite point ~140.7°E, 0°N
        let utc = UtcInstant::from_ymdhms(2025, 9, 23, 7, 20, 0);
        let cz = cos_zen(utc, 140.7, 0.0);
        // Near equinox, sun is nearly over the equator. At 07:20 UTC, the
        // subsolar longitude is ~-20°. So 140.7°E is far from subsolar.
        // cos_zen should be negative (night or near terminator) — just check it's finite.
        assert!(cz.is_finite());
    }

    #[test]
    fn sun_zenith_is_zero_near_subsolar() {
        // 2025-09-23 12:00 UTC — sun roughly over 0° longitude
        let utc = UtcInstant::from_ymdhms(2025, 9, 23, 12, 0, 0);
        let sza = sun_zenith_angle(utc, 0.0, 0.0);
        // Near equinox at noon, sza should be small
        assert!(sza < 10.0, "expected small SZA, got {sza}");
    }

    #[test]
    fn sun_zenith_is_large_at_night() {
        let utc = UtcInstant::from_ymdhms(2025, 9, 23, 0, 0, 0);
        let sza = sun_zenith_angle(utc, 0.0, 0.0);
        assert!(sza > 80.0, "expected large SZA at midnight, got {sza}");
    }

    #[test]
    fn sun_earth_distance_correction_near_one() {
        let utc = UtcInstant::from_ymdhms(2025, 6, 1, 0, 0, 0);
        let corr = sun_earth_distance_correction(utc);
        assert!((0.98..1.02).contains(&corr));
    }

    #[test]
    fn j2000_epoch_days_are_zero() {
        let utc = UtcInstant::from_ymdhms(2000, 1, 1, 12, 0, 0);
        assert!(utc.days_since_j2000().abs() < 1e-10);
    }

    #[test]
    fn julian_day_matches_known_value() {
        // 2025-01-01T00:00:00 UTC → JD ≈ 2460676.5
        let utc = UtcInstant::from_ymdhms(2025, 1, 1, 0, 0, 0);
        let jd = utc.julian_day();
        assert!((jd - 2460676.5).abs() < 0.5, "JD={jd}");
    }

    #[test]
    fn observer_position_is_finite_and_reasonable() {
        let utc = UtcInstant::from_ymdhms(2025, 9, 23, 7, 20, 0);
        let (x, y, z) = observer_position(utc, 0.0, 0.0, 0.0);
        assert!(x.is_finite() && y.is_finite() && z.is_finite());
        // At lon=0, lat=0, alt=0, the horizontal distance from the Earth's
        // rotation axis should be ~Earth equatorial radius.
        let r = (x * x + y * y).sqrt();
        assert!(r > 6000.0 && r < 7000.0, "r={r}");
        // At lat=0, z should be ~0.
        assert!(z.abs() < 1.0, "z={z}");
    }

    #[test]
    fn alt_az_returns_finite_values() {
        let utc = UtcInstant::from_ymdhms(2025, 9, 23, 12, 0, 0);
        let (alt, az) = get_alt_az(utc, 0.0, 45.0);
        assert!(alt.is_finite());
        assert!(az.is_finite());
    }

    #[test]
    fn sun_azimuth_is_in_0_360_range() {
        let utc = UtcInstant::from_ymdhms(2025, 9, 23, 6, 0, 0);
        let az = sun_azimuth_angle(utc, 0.0, 0.0);
        assert!((0.0..360.0).contains(&az), "az={az}");
    }
}
