//! Coordinate reference system metadata wrapper.
//!
//! Reference behavior inspected before implementation:
//! - `deps/pyresample/pyresample/utils/proj4.py` stores CRS information via
//!   `pyproj.CRS`, converts PROJ strings/dicts, and uses `Transformer` for
//!   later coordinate transforms.
//! - `deps/pyresample/docs/source/howtos/geometry_utils.rst` documents that
//!   Pyresample stores CRS internally as pyproj CRS objects and may normalize
//!   or rename parameters.
//! - `satpy/doc/source/reading.rst` documents `crs` as a scalar pyproj CRS
//!   coordinate for projected data, with swaths defaulting to WGS84 longlat.
//!
//! This module intentionally does not bind to PROJ yet. `P0.2.2` adds the
//! public transform API and identity behavior only where it is safe. Real
//! cross-projection math still requires a backend selected in a later step.

use rusty_sat_core::{Result, RustySatError};
use std::collections::BTreeMap;

use crate::geo_keys::{
    GeoKeyDef, ANGULAR_UNIT_DEGREE, CT_GEOSTATIONARY_SATELLITE,
    CT_LAMBERT_AZIMUTHAL_EQUAL_AREA, CT_MERCATOR, CT_POLAR_STEREOGRAPHIC, CT_STEREOGRAPHIC,
    EPSG_DATUM_WGS_84, EPSG_WGS_84, GEOGRAPHIC_TYPE_GEO_KEY, GEOG_ANGULAR_UNITS_GEO_KEY,
    GEOG_CITATION_GEO_KEY, GEOG_GEODETIC_DATUM_GEO_KEY, GEO_USER_DEFINED, GT_MODEL_TYPE_GEO_KEY,
    LINEAR_UNIT_METER, MODEL_TYPE_GEOGRAPHIC, MODEL_TYPE_PROJECTED, PROJ_CENTER_LONG_GEO_KEY,
    PROJ_COORD_TRANS_GEO_KEY, PROJ_FALSE_EASTING_GEO_KEY, PROJ_FALSE_NORTHING_GEO_KEY,
    PROJ_INV_FLATTENING_GEO_KEY, PROJ_LINEAR_UNITS_GEO_KEY, PROJ_NAT_ORIGIN_LAT_GEO_KEY,
    PROJ_NAT_ORIGIN_LONG_GEO_KEY, PROJ_SEMI_MAJOR_AXIS_GEO_KEY, PROJ_SEMI_MINOR_AXIS_GEO_KEY,
    PROJECTED_CITATION_GEO_KEY, PROJECTED_CS_TYPE_GEO_KEY,
};

/// Current projection dependency decision for Rusty Sat.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProjectionBackendStrategy {
    /// Pure Rust CRS metadata only. Coordinate transformation is unsupported.
    MetadataOnly,
    /// Future work: add a transform backend after build assumptions are known.
    DeferredTransformBackend,
}

/// Typed PROJ/CRS metadata used by areas and swaths.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProjCrs {
    params: BTreeMap<String, Option<String>>,
    source: CrsSource,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrsSource {
    Wgs84LongLat,
    ProjMap,
    ProjString,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransformDirection {
    Forward,
    Inverse,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Coordinate2D {
    x: f64,
    y: f64,
}

impl Coordinate2D {
    pub fn new(x: f64, y: f64) -> Result<Self> {
        if !x.is_finite() || !y.is_finite() {
            return Err(RustySatError::invalid_input(
                "coordinate values must be finite",
            ));
        }
        Ok(Self { x, y })
    }

    pub fn x(self) -> f64 {
        self.x
    }

    pub fn y(self) -> f64 {
        self.y
    }
}

impl ProjCrs {
    pub fn wgs84_longlat() -> Self {
        Self {
            params: BTreeMap::from([
                ("datum".to_string(), Some("WGS84".to_string())),
                ("proj".to_string(), Some("longlat".to_string())),
            ]),
            source: CrsSource::Wgs84LongLat,
        }
    }

    pub fn from_projection_map(projection: &BTreeMap<String, String>) -> Result<Self> {
        if projection.is_empty() {
            return Err(RustySatError::invalid_input(
                "projection map cannot be empty",
            ));
        }
        if let Some(proj4) = projection.get("proj4") {
            return Self::from_proj4_str(proj4);
        }
        let mut params = BTreeMap::new();
        for (key, value) in projection {
            let key = normalize_key(key)?;
            let value = normalize_value_for_key(&key, value)?;
            insert_param(&mut params, key, value)?;
        }
        normalize_epsg_init(&mut params)?;
        validate_params(&params)?;
        Ok(Self {
            params,
            source: CrsSource::ProjMap,
        })
    }

    pub fn from_proj4_str(proj4: &str) -> Result<Self> {
        let proj4 = proj4.trim();
        if proj4.is_empty() {
            return Err(RustySatError::invalid_input("PROJ string cannot be empty"));
        }
        if proj4.to_ascii_uppercase().starts_with("EPSG:") {
            let epsg = normalize_epsg_code(&proj4[5..])?;
            return Ok(Self {
                params: BTreeMap::from([("epsg".to_string(), Some(epsg))]),
                source: CrsSource::ProjString,
            });
        }

        let mut params = BTreeMap::new();
        for token in proj4.split_whitespace() {
            let token = token.strip_prefix('+').unwrap_or(token);
            if token.is_empty() {
                continue;
            }
            let mut parts = token.splitn(2, '=');
            let key = normalize_key(parts.next().unwrap_or_default())?;
            let value = parts
                .next()
                .map(|value| normalize_value_for_key(&key, value))
                .transpose()?
                .unwrap_or(None);
            insert_param(&mut params, key, value)?;
        }
        normalize_epsg_init(&mut params)?;
        validate_params(&params)?;
        Ok(Self {
            params,
            source: CrsSource::ProjString,
        })
    }

    pub fn backend_strategy() -> ProjectionBackendStrategy {
        ProjectionBackendStrategy::MetadataOnly
    }

    pub fn transform_backend_strategy() -> ProjectionBackendStrategy {
        ProjectionBackendStrategy::DeferredTransformBackend
    }

    pub fn source(&self) -> CrsSource {
        self.source
    }

    pub fn params(&self) -> &BTreeMap<String, Option<String>> {
        &self.params
    }

    pub fn param(&self, key: &str) -> Option<Option<&str>> {
        self.params.get(key).map(Option::as_deref)
    }

    pub fn projection_name(&self) -> Option<&str> {
        self.param("proj").flatten()
    }

    pub fn is_geographic(&self) -> bool {
        match self.projection_name() {
            Some("longlat" | "latlong" | "lonlat") => true,
            Some(_) => false,
            None => self.param("epsg") == Some(Some("4326")),
        }
    }

    /// Parse a CRS parameter as f64.
    ///
    /// Used by `to_geotiff_geo_key_defs` to extract numeric projection
    /// parameters from the params map.
    pub fn parse_param_f64(&self, key: &str) -> Option<f64> {
        self.param(key).flatten().and_then(|s| s.parse::<f64>().ok())
    }

    /// Generate GeoTIFF GeoKey definitions for this CRS.
    ///
    /// Returns logical GeoKey entries that describe the CRS for TIFF
    /// embedding. The caller should add `GTRasterTypeGeoKey` separately
    /// (always `RasterPixelIsArea=1` for satellite imagery).
    ///
    /// Supported projection types: `longlat`, `geos`, `stere`, `laea`,
    /// `merc`. Unknown projection types return an empty definition set.
    pub fn to_geotiff_geo_key_defs(&self) -> Vec<GeoKeyDef> {
        match self.projection_name() {
            Some("longlat") | Some("latlong") | Some("lonlat") => self.geographic_geo_keys(),
            Some("geos") => self.geos_geo_keys(),
            Some("stere") => self.stere_geo_keys(),
            Some("laea") => self.laea_geo_keys(),
            Some("merc") => self.merc_geo_keys(),
            // EPSG-only CRS: treat as geographic with known EPSG code
            None if self.param("epsg").is_some() => self.epsg_geo_keys(),
            _ => vec![],
        }
    }

    // ------------------------------------------------------------------
    // Geographic CRS (WGS84 longlat)
    // ------------------------------------------------------------------

    fn geographic_geo_keys(&self) -> Vec<GeoKeyDef> {
        let datum = self
            .param("datum")
            .flatten()
            .unwrap_or("WGS84")
            .to_uppercase();
        let is_wgs84 = datum == "WGS84";
        vec![
            GeoKeyDef::short(GT_MODEL_TYPE_GEO_KEY, MODEL_TYPE_GEOGRAPHIC),
            if is_wgs84 {
                GeoKeyDef::short(GEOGRAPHIC_TYPE_GEO_KEY, EPSG_WGS_84)
            } else {
                GeoKeyDef::short(GEOGRAPHIC_TYPE_GEO_KEY, GEO_USER_DEFINED)
            },
            if is_wgs84 {
                GeoKeyDef::short(GEOG_GEODETIC_DATUM_GEO_KEY, EPSG_DATUM_WGS_84)
            } else {
                GeoKeyDef::ascii(
                    GEOG_CITATION_GEO_KEY,
                    format!("datum={}", datum.to_lowercase()),
                )
            },
            GeoKeyDef::short(GEOG_ANGULAR_UNITS_GEO_KEY, ANGULAR_UNIT_DEGREE),
        ]
    }

    fn epsg_geo_keys(&self) -> Vec<GeoKeyDef> {
        let epsg_code = self
            .param("epsg")
            .flatten()
            .and_then(|s| s.parse::<u16>().ok())
            .unwrap_or(GEO_USER_DEFINED);
        if epsg_code == EPSG_WGS_84 {
            return vec![
                GeoKeyDef::short(GT_MODEL_TYPE_GEO_KEY, MODEL_TYPE_GEOGRAPHIC),
                GeoKeyDef::short(GEOGRAPHIC_TYPE_GEO_KEY, EPSG_WGS_84),
                GeoKeyDef::short(GEOG_GEODETIC_DATUM_GEO_KEY, EPSG_DATUM_WGS_84),
                GeoKeyDef::short(GEOG_ANGULAR_UNITS_GEO_KEY, ANGULAR_UNIT_DEGREE),
            ];
        }
        // For non-4326 EPSG codes, emit UserDefined with citation
        let citation = format!("EPSG:{}", epsg_code);
        vec![
            GeoKeyDef::short(GT_MODEL_TYPE_GEO_KEY, MODEL_TYPE_GEOGRAPHIC),
            GeoKeyDef::short(GEOGRAPHIC_TYPE_GEO_KEY, epsg_code),
            GeoKeyDef::short(GEOG_ANGULAR_UNITS_GEO_KEY, ANGULAR_UNIT_DEGREE),
            GeoKeyDef::ascii(GEOG_CITATION_GEO_KEY, citation),
        ]
    }

    // ------------------------------------------------------------------
    // Geostationary (geos) — AHI primary target
    // ------------------------------------------------------------------

    fn geos_geo_keys(&self) -> Vec<GeoKeyDef> {
        let lon_0 = self.parse_param_f64("lon_0").unwrap_or(0.0);
        let a = self.parse_param_f64("a");
        let b = self.parse_param_f64("b")
            .or_else(|| {
                let semi_major = a?;
                let rf = self.parse_param_f64("rf")?;
                Some(semi_major * (1.0 - 1.0 / rf))
            });
        let x_0 = self.parse_param_f64("x_0");
        let y_0 = self.parse_param_f64("y_0");
        let citation = self.geos_citation(lon_0, a, b);

        let mut keys = vec![
            GeoKeyDef::short(GT_MODEL_TYPE_GEO_KEY, MODEL_TYPE_PROJECTED),
            GeoKeyDef::short(PROJECTED_CS_TYPE_GEO_KEY, GEO_USER_DEFINED),
            GeoKeyDef::ascii(PROJECTED_CITATION_GEO_KEY, citation),
            GeoKeyDef::short(PROJ_COORD_TRANS_GEO_KEY, CT_GEOSTATIONARY_SATELLITE),
            GeoKeyDef::short(PROJ_LINEAR_UNITS_GEO_KEY, LINEAR_UNIT_METER),
            GeoKeyDef::double(PROJ_CENTER_LONG_GEO_KEY, lon_0),
            GeoKeyDef::double(PROJ_NAT_ORIGIN_LONG_GEO_KEY, lon_0),
        ];

        if let Some(v) = x_0 {
            keys.push(GeoKeyDef::double(PROJ_FALSE_EASTING_GEO_KEY, v));
        }
        if let Some(v) = y_0 {
            keys.push(GeoKeyDef::double(PROJ_FALSE_NORTHING_GEO_KEY, v));
        }
        if let Some(v) = a {
            keys.push(GeoKeyDef::double(PROJ_SEMI_MAJOR_AXIS_GEO_KEY, v));
        }
        if let Some(v) = b {
            keys.push(GeoKeyDef::double(PROJ_SEMI_MINOR_AXIS_GEO_KEY, v));
        }
        if a.is_none() && b.is_none() {
            if let Some(rf) = self.parse_param_f64("rf") {
                keys.push(GeoKeyDef::double(PROJ_INV_FLATTENING_GEO_KEY, rf));
            }
        }
        keys
    }

    fn geos_citation(&self, lon_0: f64, a: Option<f64>, b: Option<f64>) -> String {
        let mut parts = vec![format!("GEOS lon_0={lon_0}")];
        if let Some(v) = a {
            parts.push(format!("a={v}"));
        }
        if let Some(v) = b {
            parts.push(format!("b={v}"));
        }
        if let Some(h) = self.parse_param_f64("h") {
            parts.push(format!("h={h}"));
        }
        parts.join(" ")
    }

    // ------------------------------------------------------------------
    // Stereographic (stere) — Polar vs. Oblique
    // ------------------------------------------------------------------

    fn stere_geo_keys(&self) -> Vec<GeoKeyDef> {
        let lat_0 = self.parse_param_f64("lat_0").unwrap_or(0.0);
        let lon_0 = self.parse_param_f64("lon_0").unwrap_or(0.0);
        let x_0 = self.parse_param_f64("x_0");
        let y_0 = self.parse_param_f64("y_0");
        let is_polar = (lat_0 - 90.0).abs() < f64::EPSILON
            || (lat_0 + 90.0).abs() < f64::EPSILON;
        let coord_trans = if is_polar {
            CT_POLAR_STEREOGRAPHIC
        } else {
            CT_STEREOGRAPHIC
        };
        let citation = format!("stere lat_0={lat_0} lon_0={lon_0}");

        let mut keys = vec![
            GeoKeyDef::short(GT_MODEL_TYPE_GEO_KEY, MODEL_TYPE_PROJECTED),
            GeoKeyDef::short(PROJECTED_CS_TYPE_GEO_KEY, GEO_USER_DEFINED),
            GeoKeyDef::ascii(PROJECTED_CITATION_GEO_KEY, citation),
            GeoKeyDef::short(PROJ_COORD_TRANS_GEO_KEY, coord_trans),
            GeoKeyDef::short(PROJ_LINEAR_UNITS_GEO_KEY, LINEAR_UNIT_METER),
            GeoKeyDef::double(PROJ_NAT_ORIGIN_LAT_GEO_KEY, lat_0),
            GeoKeyDef::double(PROJ_CENTER_LONG_GEO_KEY, lon_0),
            GeoKeyDef::double(PROJ_NAT_ORIGIN_LONG_GEO_KEY, lon_0),
        ];
        if let Some(v) = x_0 {
            keys.push(GeoKeyDef::double(PROJ_FALSE_EASTING_GEO_KEY, v));
        }
        if let Some(v) = y_0 {
            keys.push(GeoKeyDef::double(PROJ_FALSE_NORTHING_GEO_KEY, v));
        }
        keys
    }

    // ------------------------------------------------------------------
    // Lambert Azimuthal Equal Area (laea)
    // ------------------------------------------------------------------

    fn laea_geo_keys(&self) -> Vec<GeoKeyDef> {
        let lat_0 = self.parse_param_f64("lat_0").unwrap_or(0.0);
        let lon_0 = self.parse_param_f64("lon_0").unwrap_or(0.0);
        let x_0 = self.parse_param_f64("x_0");
        let y_0 = self.parse_param_f64("y_0");
        let citation = format!("laea lat_0={lat_0} lon_0={lon_0}");

        let mut keys = vec![
            GeoKeyDef::short(GT_MODEL_TYPE_GEO_KEY, MODEL_TYPE_PROJECTED),
            GeoKeyDef::short(PROJECTED_CS_TYPE_GEO_KEY, GEO_USER_DEFINED),
            GeoKeyDef::ascii(PROJECTED_CITATION_GEO_KEY, citation),
            GeoKeyDef::short(
                PROJ_COORD_TRANS_GEO_KEY,
                CT_LAMBERT_AZIMUTHAL_EQUAL_AREA,
            ),
            GeoKeyDef::short(PROJ_LINEAR_UNITS_GEO_KEY, LINEAR_UNIT_METER),
            GeoKeyDef::double(PROJ_NAT_ORIGIN_LAT_GEO_KEY, lat_0),
            GeoKeyDef::double(PROJ_NAT_ORIGIN_LONG_GEO_KEY, lon_0),
        ];
        if let Some(v) = x_0 {
            keys.push(GeoKeyDef::double(PROJ_FALSE_EASTING_GEO_KEY, v));
        }
        if let Some(v) = y_0 {
            keys.push(GeoKeyDef::double(PROJ_FALSE_NORTHING_GEO_KEY, v));
        }
        keys
    }

    // ------------------------------------------------------------------
    // Mercator (merc)
    // ------------------------------------------------------------------

    fn merc_geo_keys(&self) -> Vec<GeoKeyDef> {
        let lon_0 = self.parse_param_f64("lon_0").unwrap_or(0.0);
        let lat_ts = self.parse_param_f64("lat_ts");
        let x_0 = self.parse_param_f64("x_0");
        let y_0 = self.parse_param_f64("y_0");
        let citation = if let Some(lt) = lat_ts {
            format!("merc lon_0={lon_0} lat_ts={lt}")
        } else {
            format!("merc lon_0={lon_0}")
        };

        let mut keys = vec![
            GeoKeyDef::short(GT_MODEL_TYPE_GEO_KEY, MODEL_TYPE_PROJECTED),
            GeoKeyDef::short(PROJECTED_CS_TYPE_GEO_KEY, GEO_USER_DEFINED),
            GeoKeyDef::ascii(PROJECTED_CITATION_GEO_KEY, citation),
            GeoKeyDef::short(PROJ_COORD_TRANS_GEO_KEY, CT_MERCATOR),
            GeoKeyDef::short(PROJ_LINEAR_UNITS_GEO_KEY, LINEAR_UNIT_METER),
            GeoKeyDef::double(PROJ_NAT_ORIGIN_LONG_GEO_KEY, lon_0),
        ];
        if let Some(v) = lat_ts {
            keys.push(GeoKeyDef::double(PROJ_NAT_ORIGIN_LAT_GEO_KEY, v));
        }
        if let Some(v) = x_0 {
            keys.push(GeoKeyDef::double(PROJ_FALSE_EASTING_GEO_KEY, v));
        }
        if let Some(v) = y_0 {
            keys.push(GeoKeyDef::double(PROJ_FALSE_NORTHING_GEO_KEY, v));
        }
        keys
    }

    pub fn equivalent_to(&self, other: &Self) -> bool {
        self.params == other.params
    }

    pub fn transform_coordinate(
        &self,
        direction: TransformDirection,
        coordinate: Coordinate2D,
    ) -> Result<Coordinate2D> {
        if self.is_geographic() {
            return Ok(coordinate);
        }
        Err(RustySatError::unsupported(format!(
            "{direction:?} coordinate transform for CRS '{}'",
            self.to_proj4_string()
        )))
    }

    pub fn transform_coordinates(
        &self,
        direction: TransformDirection,
        coordinates: impl IntoIterator<Item = Coordinate2D>,
    ) -> Result<Vec<Coordinate2D>> {
        coordinates
            .into_iter()
            .map(|coordinate| self.transform_coordinate(direction, coordinate))
            .collect()
    }

    pub fn transform_to(&self, target: &Self, coordinate: Coordinate2D) -> Result<Coordinate2D> {
        if self.equivalent_to(target) || (self.is_geographic() && target.is_geographic()) {
            return Ok(coordinate);
        }
        Err(RustySatError::unsupported(format!(
            "coordinate transform from '{}' to '{}'",
            self.to_proj4_string(),
            target.to_proj4_string()
        )))
    }

    pub fn transform_many_to(
        &self,
        target: &Self,
        coordinates: impl IntoIterator<Item = Coordinate2D>,
    ) -> Result<Vec<Coordinate2D>> {
        coordinates
            .into_iter()
            .map(|coordinate| self.transform_to(target, coordinate))
            .collect()
    }

    pub fn to_proj4_string(&self) -> String {
        self.params
            .iter()
            .map(|(key, value)| match value {
                Some(value) => format!("+{key}={value}"),
                None => format!("+{key}"),
            })
            .collect::<Vec<_>>()
            .join(" ")
    }
}

fn validate_params(params: &BTreeMap<String, Option<String>>) -> Result<()> {
    if params.is_empty() {
        return Err(RustySatError::invalid_input(
            "CRS parameters cannot be empty",
        ));
    }
    if !params.contains_key("proj") && !params.contains_key("epsg") {
        return Err(RustySatError::invalid_input(
            "CRS parameters require 'proj' or an EPSG code",
        ));
    }
    Ok(())
}

fn normalize_key(key: &str) -> Result<String> {
    let key = key.trim().strip_prefix('+').unwrap_or(key.trim());
    if key.is_empty() {
        return Err(RustySatError::invalid_input(
            "CRS parameter key cannot be empty",
        ));
    }
    Ok(key.to_string())
}

fn normalize_value_for_key(key: &str, value: &str) -> Result<Option<String>> {
    let Some(value) = normalize_value(value) else {
        return Ok(None);
    };
    if key == "proj" {
        return Ok(Some(normalize_projection_name(&value)));
    }
    if key == "epsg" {
        return Ok(Some(normalize_epsg_code(&value)?));
    }
    Ok(Some(normalize_numeric_string(&value)))
}

fn normalize_value(value: &str) -> Option<String> {
    let value = value.trim();
    if value.is_empty()
        || value.eq_ignore_ascii_case("none")
        || value.eq_ignore_ascii_case("null")
        || value.eq_ignore_ascii_case("true")
    {
        None
    } else {
        Some(value.to_string())
    }
}

fn normalize_projection_name(value: &str) -> String {
    match value.to_ascii_lowercase().as_str() {
        "latlong" | "lonlat" => "longlat".to_string(),
        _ => value.to_string(),
    }
}

fn normalize_numeric_string(value: &str) -> String {
    let Ok(number) = value.parse::<f64>() else {
        return value.to_string();
    };
    if !number.is_finite() {
        return value.to_string();
    }
    number.to_string()
}

fn normalize_epsg_code(value: &str) -> Result<String> {
    let value = value.trim();
    if value.is_empty() || !value.chars().all(|ch| ch.is_ascii_digit()) {
        return Err(RustySatError::invalid_input(format!(
            "EPSG code '{value}' must contain only digits"
        )));
    }
    let normalized = value.trim_start_matches('0');
    Ok(if normalized.is_empty() {
        "0".to_string()
    } else {
        normalized.to_string()
    })
}

fn insert_param(
    params: &mut BTreeMap<String, Option<String>>,
    key: String,
    value: Option<String>,
) -> Result<()> {
    if params.contains_key(&key) {
        return Err(RustySatError::invalid_input(format!(
            "duplicate CRS parameter '{key}'"
        )));
    }
    params.insert(key, value);
    Ok(())
}

fn normalize_epsg_init(params: &mut BTreeMap<String, Option<String>>) -> Result<()> {
    let Some(init) = params.remove("init") else {
        return Ok(());
    };
    let Some(init) = init else {
        return Err(RustySatError::invalid_input(
            "CRS init parameter requires a value",
        ));
    };
    let Some((authority, code)) = init.split_once(':') else {
        return Err(RustySatError::invalid_input(format!(
            "unsupported CRS init authority '{init}'"
        )));
    };
    if !authority.eq_ignore_ascii_case("epsg") {
        return Err(RustySatError::invalid_input(format!(
            "unsupported CRS init authority '{authority}'"
        )));
    }
    insert_param(params, "epsg".to_string(), Some(normalize_epsg_code(code)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geo_keys::GeoKeyValue;

    #[test]
    fn constructs_wgs84_longlat_crs() {
        let crs = ProjCrs::wgs84_longlat();

        assert_eq!(crs.source(), CrsSource::Wgs84LongLat);
        assert_eq!(crs.projection_name(), Some("longlat"));
        assert_eq!(crs.param("datum"), Some(Some("WGS84")));
        assert!(crs.is_geographic());
        assert_eq!(
            ProjCrs::backend_strategy(),
            ProjectionBackendStrategy::MetadataOnly
        );
        assert_eq!(
            ProjCrs::transform_backend_strategy(),
            ProjectionBackendStrategy::DeferredTransformBackend
        );
    }

    #[test]
    fn constructs_crs_from_projection_map() {
        let crs = ProjCrs::from_projection_map(&BTreeMap::from([
            ("+proj".to_string(), "laea".to_string()),
            ("lat_0".to_string(), "-90.0".to_string()),
            ("lon_0".to_string(), "0.000".to_string()),
            ("no_defs".to_string(), "None".to_string()),
            ("units".to_string(), "m".to_string()),
        ]))
        .unwrap();

        assert_eq!(crs.source(), CrsSource::ProjMap);
        assert_eq!(crs.projection_name(), Some("laea"));
        assert_eq!(crs.param("lat_0"), Some(Some("-90")));
        assert_eq!(crs.param("lon_0"), Some(Some("0")));
        assert_eq!(crs.param("no_defs"), Some(None));
        assert!(!crs.is_geographic());
    }

    #[test]
    fn constructs_crs_from_proj4_string() {
        let crs = ProjCrs::from_proj4_str("+proj=longlat +datum=WGS84 +no_defs").unwrap();

        assert_eq!(crs.source(), CrsSource::ProjString);
        assert_eq!(crs.projection_name(), Some("longlat"));
        assert_eq!(crs.param("no_defs"), Some(None));
        assert!(crs.to_proj4_string().contains("+datum=WGS84"));
        assert!(crs.to_proj4_string().contains("+proj=longlat"));
    }

    #[test]
    fn accepts_epsg_string_as_symbolic_crs() {
        let crs = ProjCrs::from_proj4_str("EPSG:04326").unwrap();

        assert_eq!(crs.param("epsg"), Some(Some("4326")));
        assert!(crs.is_geographic());
        assert_eq!(crs.to_proj4_string(), "+epsg=4326");
    }

    #[test]
    fn normalizes_proj_aliases_and_deprecated_epsg_init() {
        let longlat = ProjCrs::from_proj4_str("+proj=latlong +datum=WGS84").unwrap();
        let init = ProjCrs::from_proj4_str("+init=EPSG:4326").unwrap();

        assert_eq!(longlat.projection_name(), Some("longlat"));
        assert_eq!(init.param("epsg"), Some(Some("4326")));
        assert!(init.params().get("init").is_none());
    }

    #[test]
    fn rejects_duplicate_and_malformed_crs_parameters() {
        assert!(ProjCrs::from_proj4_str("+proj=merc +proj=laea").is_err());
        assert!(ProjCrs::from_proj4_str("EPSG:abcd").is_err());
        assert!(ProjCrs::from_proj4_str("+init=IGNF:LAMB1").is_err());
    }

    #[test]
    fn coordinate_constructor_rejects_non_finite_values() {
        assert!(Coordinate2D::new(1.0, 2.0).is_ok());
        assert!(Coordinate2D::new(f64::NAN, 2.0).is_err());
        assert!(Coordinate2D::new(1.0, f64::INFINITY).is_err());
    }

    #[test]
    fn geographic_forward_and_inverse_transforms_are_identity() {
        let crs = ProjCrs::wgs84_longlat();
        let coordinate = Coordinate2D::new(12.5, -45.25).unwrap();

        assert_eq!(
            crs.transform_coordinate(TransformDirection::Forward, coordinate)
                .unwrap(),
            coordinate
        );
        assert_eq!(
            crs.transform_coordinate(TransformDirection::Inverse, coordinate)
                .unwrap(),
            coordinate
        );
    }

    #[test]
    fn same_crs_transform_is_identity() {
        let source = ProjCrs::from_proj4_str("+proj=laea +lat_0=-90 +lon_0=0 +units=m").unwrap();
        let target = ProjCrs::from_projection_map(&BTreeMap::from([
            ("proj".to_string(), "laea".to_string()),
            ("lat_0".to_string(), "-90".to_string()),
            ("lon_0".to_string(), "0".to_string()),
            ("units".to_string(), "m".to_string()),
        ]))
        .unwrap();
        let coordinate = Coordinate2D::new(100.0, -200.0).unwrap();

        assert!(source.equivalent_to(&target));
        assert_eq!(
            source.transform_to(&target, coordinate).unwrap(),
            coordinate
        );
        assert_eq!(
            source
                .transform_many_to(&target, [coordinate, Coordinate2D::new(0.0, 0.0).unwrap()])
                .unwrap(),
            vec![coordinate, Coordinate2D::new(0.0, 0.0).unwrap()]
        );
    }

    #[test]
    fn geographic_alias_transforms_are_identity() {
        let epsg = ProjCrs::from_proj4_str("EPSG:4326").unwrap();
        let longlat = ProjCrs::from_proj4_str("+proj=longlat +datum=WGS84").unwrap();
        let coordinate = Coordinate2D::new(12.5, -45.25).unwrap();

        assert!(!epsg.equivalent_to(&longlat));
        assert_eq!(epsg.transform_to(&longlat, coordinate).unwrap(), coordinate);
        assert_eq!(longlat.transform_to(&epsg, coordinate).unwrap(), coordinate);
    }

    #[test]
    fn projected_and_cross_crs_transforms_require_backend() {
        let projected = ProjCrs::from_proj4_str("+proj=laea +lat_0=-90 +lon_0=0 +units=m").unwrap();
        let geographic = ProjCrs::wgs84_longlat();
        let coordinate = Coordinate2D::new(100.0, -200.0).unwrap();

        assert!(matches!(
            projected.transform_coordinate(TransformDirection::Forward, coordinate),
            Err(RustySatError::Unsupported { .. })
        ));
        assert!(matches!(
            projected.transform_to(&geographic, coordinate),
            Err(RustySatError::Unsupported { .. })
        ));
    }

    #[test]
    fn transform_coordinates_stops_on_unsupported_backend() {
        let projected = ProjCrs::from_proj4_str("+proj=stere +lat_0=90 +lon_0=0").unwrap();
        let coordinates = [
            Coordinate2D::new(0.0, 0.0).unwrap(),
            Coordinate2D::new(1.0, 1.0).unwrap(),
        ];

        assert!(matches!(
            projected.transform_coordinates(TransformDirection::Inverse, coordinates),
            Err(RustySatError::Unsupported { .. })
        ));
    }

    #[test]
    fn rejects_missing_projection_identifier() {
        let err = ProjCrs::from_projection_map(&BTreeMap::from([(
            "datum".to_string(),
            "WGS84".to_string(),
        )]))
        .unwrap_err();

        assert!(matches!(err, RustySatError::InvalidInput { .. }));
    }

    // ------------------------------------------------------------------
    // parse_param_f64
    // ------------------------------------------------------------------

    #[test]
    fn parse_param_f64_parses_numeric_values() {
        let crs = ProjCrs::from_proj4_str("+proj=geos +lon_0=140.7 +a=6378137").unwrap();
        assert!((crs.parse_param_f64("lon_0").unwrap() - 140.7).abs() < 1e-10);
        assert!((crs.parse_param_f64("a").unwrap() - 6_378_137.0).abs() < 1e-10);
    }

    #[test]
    fn parse_param_f64_returns_none_for_missing_key() {
        let crs = ProjCrs::wgs84_longlat();
        assert!(crs.parse_param_f64("lat_0").is_none());
    }

    #[test]
    fn parse_param_f64_returns_none_for_non_numeric_value() {
        let crs = ProjCrs::wgs84_longlat();
        assert!(crs.parse_param_f64("datum").is_none());
    }

    // ------------------------------------------------------------------
    // to_geotiff_geo_key_defs — longlat (WGS84 geographic)
    // ------------------------------------------------------------------

    #[test]
    fn geotiff_keys_longlat_wgs84() {
        let crs = ProjCrs::wgs84_longlat();
        let keys = crs.to_geotiff_geo_key_defs();
        assert_eq!(keys.len(), 4);
        // GTModelTypeGeoKey = Geographic
        assert_eq!(keys[0].key_id, GT_MODEL_TYPE_GEO_KEY);
        assert_eq!(keys[0].value, GeoKeyValue::Short(MODEL_TYPE_GEOGRAPHIC));
        // GeographicTypeGeoKey = 4326
        assert_eq!(keys[1].key_id, GEOGRAPHIC_TYPE_GEO_KEY);
        assert_eq!(keys[1].value, GeoKeyValue::Short(EPSG_WGS_84));
        // GeogGeodeticDatumGeoKey = 6326
        assert_eq!(keys[2].key_id, GEOG_GEODETIC_DATUM_GEO_KEY);
        assert_eq!(keys[2].value, GeoKeyValue::Short(EPSG_DATUM_WGS_84));
        // GeogAngularUnitsGeoKey = degree
        assert_eq!(keys[3].key_id, GEOG_ANGULAR_UNITS_GEO_KEY);
        assert_eq!(keys[3].value, GeoKeyValue::Short(ANGULAR_UNIT_DEGREE));
    }

    #[test]
    fn geotiff_keys_longlat_non_wgs84_datum_uses_citation() {
        let crs = ProjCrs::from_proj4_str("+proj=longlat +datum=NAD83").unwrap();
        let keys = crs.to_geotiff_geo_key_defs();
        // GTModelTypeGeoKey = Geographic
        assert_eq!(keys[0].key_id, GT_MODEL_TYPE_GEO_KEY);
        assert_eq!(keys[0].value, GeoKeyValue::Short(MODEL_TYPE_GEOGRAPHIC));
        // GeographicTypeGeoKey = UserDefined (not WGS84)
        assert_eq!(keys[1].key_id, GEOGRAPHIC_TYPE_GEO_KEY);
        assert_eq!(keys[1].value, GeoKeyValue::Short(GEO_USER_DEFINED));
        // Has GEOG_CITATION_GEO_KEY as Ascii
        let citation = keys.iter().find(|k| k.key_id == GEOG_CITATION_GEO_KEY);
        assert!(citation.is_some());
        if let Some(c) = citation {
            assert!(matches!(c.value, GeoKeyValue::Ascii(_)));
        }
    }

    // ------------------------------------------------------------------
    // to_geotiff_geo_key_defs — geos (AHI primary target)
    // ------------------------------------------------------------------

    #[test]
    fn geotiff_keys_geos_ahi_realistic() {
        let crs = ProjCrs::from_projection_map(&BTreeMap::from([
            ("proj".to_string(), "geos".to_string()),
            ("a".to_string(), "6378137".to_string()),
            ("b".to_string(), "6356752.3".to_string()),
            ("h".to_string(), "35785863".to_string()),
            ("lon_0".to_string(), "140.7".to_string()),
            ("units".to_string(), "m".to_string()),
        ]))
        .unwrap();
        let keys = crs.to_geotiff_geo_key_defs();

        // Must have at least 9 keys (no x_0/y_0 in this fixture)
        assert_eq!(keys.len(), 9, "expected 9 keys, got {}", keys.len());

        // GTModelTypeGeoKey = Projected
        assert_eq!(keys[0].key_id, GT_MODEL_TYPE_GEO_KEY);
        assert_eq!(keys[0].value, GeoKeyValue::Short(MODEL_TYPE_PROJECTED));
        // ProjectedCSTypeGeoKey = UserDefined
        assert_eq!(keys[1].key_id, PROJECTED_CS_TYPE_GEO_KEY);
        assert_eq!(keys[1].value, GeoKeyValue::Short(GEO_USER_DEFINED));
        // ProjectedCitationGeoKey = Ascii
        assert_eq!(keys[2].key_id, PROJECTED_CITATION_GEO_KEY);
        assert!(matches!(keys[2].value, GeoKeyValue::Ascii(_)));
        // ProjCoordTransGeoKey = CT_GeostationarySatellite (28)
        assert_eq!(keys[3].key_id, PROJ_COORD_TRANS_GEO_KEY);
        assert_eq!(
            keys[3].value,
            GeoKeyValue::Short(CT_GEOSTATIONARY_SATELLITE)
        );
        // ProjLinearUnitsGeoKey = Meter
        assert_eq!(keys[4].key_id, PROJ_LINEAR_UNITS_GEO_KEY);
        assert_eq!(keys[4].value, GeoKeyValue::Short(LINEAR_UNIT_METER));

        // Find double params by key_id
        let find_double = |key_id: u16| -> Option<f64> {
            keys.iter().find_map(|k| {
                if k.key_id == key_id {
                    if let GeoKeyValue::Double(v) = k.value {
                        Some(v)
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
        };
        assert!(
            (find_double(PROJ_CENTER_LONG_GEO_KEY).unwrap() - 140.7).abs() < 1e-10
        );
        assert!(
            (find_double(PROJ_NAT_ORIGIN_LONG_GEO_KEY).unwrap() - 140.7).abs() < 1e-10
        );
        assert!(
            (find_double(PROJ_SEMI_MAJOR_AXIS_GEO_KEY).unwrap() - 6_378_137.0).abs() < 1e-10
        );
        assert!(
            (find_double(PROJ_SEMI_MINOR_AXIS_GEO_KEY).unwrap() - 6_356_752.3).abs()
                < 1e-6
        );
    }

    #[test]
    fn geotiff_keys_geos_with_rf_computes_b() {
        // AHI L2 NC style: has rf, no direct b
        let crs = ProjCrs::from_projection_map(&BTreeMap::from([
            ("proj".to_string(), "geos".to_string()),
            ("a".to_string(), "6378137".to_string()),
            ("h".to_string(), "35785863".to_string()),
            ("lon_0".to_string(), "140.7".to_string()),
            ("rf".to_string(), "298.257024882273".to_string()),
            ("units".to_string(), "m".to_string()),
            ("x_0".to_string(), "0".to_string()),
            ("y_0".to_string(), "0".to_string()),
        ]))
        .unwrap();
        let keys = crs.to_geotiff_geo_key_defs();

        // b should be computed from a and rf: b = a * (1 - 1/rf)
        let b = keys
            .iter()
            .find_map(|k| {
                if k.key_id == PROJ_SEMI_MINOR_AXIS_GEO_KEY {
                    if let GeoKeyValue::Double(v) = k.value {
                        Some(v)
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .unwrap();
        let expected_b = 6_378_137.0 * (1.0 - 1.0 / 298.257024882273_f64);
        assert!((b - expected_b).abs() < 1e-6);
        // x_0 and y_0 should be present
        assert!(keys.iter().any(|k| k.key_id == PROJ_FALSE_EASTING_GEO_KEY));
        assert!(keys.iter().any(|k| k.key_id == PROJ_FALSE_NORTHING_GEO_KEY));
    }

    #[test]
    fn geotiff_keys_geos_without_rf_no_b() {
        let crs = ProjCrs::from_projection_map(&BTreeMap::from([
            ("proj".to_string(), "geos".to_string()),
            ("lon_0".to_string(), "140.7".to_string()),
        ]))
        .unwrap();
        let keys = crs.to_geotiff_geo_key_defs();
        // b and a not available; should have rf if present
        assert!(!keys
            .iter()
            .any(|k| k.key_id == PROJ_SEMI_MAJOR_AXIS_GEO_KEY));
        assert!(!keys
            .iter()
            .any(|k| k.key_id == PROJ_SEMI_MINOR_AXIS_GEO_KEY));
    }

    // ------------------------------------------------------------------
    // to_geotiff_geo_key_defs — stere
    // ------------------------------------------------------------------

    #[test]
    fn geotiff_keys_stere_polar() {
        let crs = ProjCrs::from_proj4_str("+proj=stere +lat_0=90 +lon_0=0 +units=m").unwrap();
        let keys = crs.to_geotiff_geo_key_defs();
        assert!(keys.len() >= 8);
        // ProjCoordTransGeoKey = PolarStereographic
        assert_eq!(keys[3].key_id, PROJ_COORD_TRANS_GEO_KEY);
        assert_eq!(keys[3].value, GeoKeyValue::Short(CT_POLAR_STEREOGRAPHIC));
        // ProjNatOriginLatGeoKey = 90
        let lat = keys
            .iter()
            .find_map(|k| {
                if k.key_id == PROJ_NAT_ORIGIN_LAT_GEO_KEY {
                    if let GeoKeyValue::Double(v) = k.value {
                        Some(v)
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .unwrap();
        assert!((lat - 90.0).abs() < 1e-10);
    }

    #[test]
    fn geotiff_keys_stere_oblique() {
        let crs =
            ProjCrs::from_proj4_str("+proj=stere +lat_0=60 +lon_0=10 +units=m").unwrap();
        let keys = crs.to_geotiff_geo_key_defs();
        // ProjCoordTransGeoKey = Stereographic (NOT polar)
        assert_eq!(keys[3].key_id, PROJ_COORD_TRANS_GEO_KEY);
        assert_eq!(keys[3].value, GeoKeyValue::Short(CT_STEREOGRAPHIC));
    }

    // ------------------------------------------------------------------
    // to_geotiff_geo_key_defs — laea
    // ------------------------------------------------------------------

    #[test]
    fn geotiff_keys_laea() {
        let crs =
            ProjCrs::from_proj4_str("+proj=laea +lat_0=45 +lon_0=-100 +units=m").unwrap();
        let keys = crs.to_geotiff_geo_key_defs();
        assert!(keys.len() >= 7);
        // ProjCoordTransGeoKey = LambertAzimEqualArea
        assert_eq!(keys[3].key_id, PROJ_COORD_TRANS_GEO_KEY);
        assert_eq!(
            keys[3].value,
            GeoKeyValue::Short(CT_LAMBERT_AZIMUTHAL_EQUAL_AREA)
        );
        // Contains lat_0 and lon_0 as doubles
        let lat = keys.iter().find(|k| k.key_id == PROJ_NAT_ORIGIN_LAT_GEO_KEY);
        assert!(lat.is_some());
        let lon = keys
            .iter()
            .find(|k| k.key_id == PROJ_NAT_ORIGIN_LONG_GEO_KEY);
        assert!(lon.is_some());
    }

    // ------------------------------------------------------------------
    // to_geotiff_geo_key_defs — merc
    // ------------------------------------------------------------------

    #[test]
    fn geotiff_keys_merc() {
        let crs = ProjCrs::from_proj4_str("+proj=merc +lon_0=0 +units=m").unwrap();
        let keys = crs.to_geotiff_geo_key_defs();
        assert!(keys.len() >= 6);
        // ProjCoordTransGeoKey = Mercator
        assert_eq!(keys[3].key_id, PROJ_COORD_TRANS_GEO_KEY);
        assert_eq!(keys[3].value, GeoKeyValue::Short(CT_MERCATOR));
        // ProjNatOriginLongGeoKey = lon_0
        let lon = keys
            .iter()
            .find(|k| k.key_id == PROJ_NAT_ORIGIN_LONG_GEO_KEY);
        assert!(lon.is_some());
    }

    #[test]
    fn geotiff_keys_merc_with_lat_ts() {
        let crs = ProjCrs::from_proj4_str(
            "+proj=merc +lon_0=140.7 +lat_ts=30 +units=m",
        )
        .unwrap();
        let keys = crs.to_geotiff_geo_key_defs();
        let lat = keys
            .iter()
            .find(|k| k.key_id == PROJ_NAT_ORIGIN_LAT_GEO_KEY);
        assert!(lat.is_some());
        if let Some(k) = lat {
            if let GeoKeyValue::Double(v) = k.value {
                assert!((v - 30.0).abs() < 1e-10);
            } else {
                panic!("expected Double for lat_ts");
            }
        }
    }

    // ------------------------------------------------------------------
    // to_geotiff_geo_key_defs — EPSG-only CRS
    // ------------------------------------------------------------------

    #[test]
    fn geotiff_keys_epsg_4326() {
        let crs = ProjCrs::from_proj4_str("EPSG:4326").unwrap();
        let keys = crs.to_geotiff_geo_key_defs();
        assert!(keys.len() >= 3);
        assert_eq!(keys[0].value, GeoKeyValue::Short(MODEL_TYPE_GEOGRAPHIC));
        assert_eq!(keys[1].value, GeoKeyValue::Short(EPSG_WGS_84));
    }

    // ------------------------------------------------------------------
    // to_geotiff_geo_key_defs — unknown projection
    // ------------------------------------------------------------------

    #[test]
    fn geotiff_keys_unknown_projection_returns_empty() {
        let crs = ProjCrs::from_proj4_str("+proj=ob_tran +o_proj=longlat").unwrap();
        let keys = crs.to_geotiff_geo_key_defs();
        assert!(keys.is_empty());
    }
}
