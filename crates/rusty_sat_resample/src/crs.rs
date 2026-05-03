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
//! This module intentionally does not bind to PROJ yet. `P0.2.1` needs a
//! typed CRS representation and a dependency strategy; real forward/inverse
//! transforms belong to `P0.2.2`.

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
            params.insert(key, normalize_value(value));
        }
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
            return Ok(Self {
                params: BTreeMap::from([("epsg".to_string(), Some(proj4[5..].to_string()))]),
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
            let value = parts.next().map(normalize_value).unwrap_or(None);
            params.insert(key, value);
        }
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
        matches!(
            self.projection_name(),
            Some("longlat" | "latlong" | "lonlat")
        )
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
            ("lat_0".to_string(), "-90".to_string()),
            ("lon_0".to_string(), "0".to_string()),
            ("no_defs".to_string(), "None".to_string()),
            ("units".to_string(), "m".to_string()),
        ]))
        .unwrap();

        assert_eq!(crs.source(), CrsSource::ProjMap);
        assert_eq!(crs.projection_name(), Some("laea"));
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
        let crs = ProjCrs::from_proj4_str("EPSG:4326").unwrap();

        assert_eq!(crs.param("epsg"), Some(Some("4326")));
        assert_eq!(crs.to_proj4_string(), "+epsg=4326");
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
