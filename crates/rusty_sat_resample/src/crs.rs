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
}
