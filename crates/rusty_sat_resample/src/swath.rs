//! Swath definition model and small YAML fixtures.
//!
//! Reference behavior inspected before implementation:
//! - `deps/pyresample/pyresample/geometry.py`
//! - `deps/pyresample/docs/source/concepts/geometries.rst`

use crate::geometry::{GeometryDefinition, GeometryKind};
use crate::ProjCrs;
use rusty_sat_core::{Coordinate, DataArray, NumericElement, Result, RustySatError};
use serde_norway::{Mapping, Value};
use std::collections::BTreeMap;

const MAX_SWATH_YAML_BYTES: usize = 8 * 1024 * 1024;
const MAX_SWATH_YAML_DEPTH: usize = 96;

#[derive(Debug, Clone, PartialEq)]
pub struct SwathDefinition {
    height: usize,
    width: usize,
    lons: Option<Vec<f64>>,
    lats: Option<Vec<f64>>,
    crs: BTreeMap<String, String>,
}

impl SwathDefinition {
    pub fn new(height: usize, width: usize) -> Result<Self> {
        validate_shape(height, width)?;
        Ok(Self {
            height,
            width,
            lons: None,
            lats: None,
            crs: default_lonlat_crs(),
        })
    }

    pub fn from_lonlats(
        height: usize,
        width: usize,
        lons: Vec<f64>,
        lats: Vec<f64>,
    ) -> Result<Self> {
        validate_shape(height, width)?;
        validate_coordinates(height, width, &lons, &lats)?;
        Ok(Self {
            height,
            width,
            lons: Some(lons),
            lats: Some(lats),
            crs: default_lonlat_crs(),
        })
    }

    pub fn with_crs(mut self, crs: BTreeMap<String, String>) -> Result<Self> {
        if crs.is_empty() {
            return Err(RustySatError::invalid_input("swath CRS cannot be empty"));
        }
        self.crs = crs;
        Ok(self)
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.height, self.width)
    }

    pub fn size(&self) -> usize {
        self.height * self.width
    }

    pub fn has_coordinates(&self) -> bool {
        self.lons.is_some() && self.lats.is_some()
    }

    pub fn lons(&self) -> Option<&[f64]> {
        self.lons.as_deref()
    }

    pub fn lats(&self) -> Option<&[f64]> {
        self.lats.as_deref()
    }

    pub fn crs(&self) -> &BTreeMap<String, String> {
        &self.crs
    }

    pub fn crs_definition(&self) -> Result<ProjCrs> {
        ProjCrs::from_projection_map(&self.crs)
    }

    pub fn longitude_coordinate(&self) -> Result<Coordinate> {
        let Some(lons) = self.lons.as_ref() else {
            return Err(RustySatError::invalid_input(
                "swath has no longitude coordinates",
            ));
        };
        Coordinate::new(["y", "x"], lons.clone())
    }

    pub fn latitude_coordinate(&self) -> Result<Coordinate> {
        let Some(lats) = self.lats.as_ref() else {
            return Err(RustySatError::invalid_input(
                "swath has no latitude coordinates",
            ));
        };
        Coordinate::new(["y", "x"], lats.clone())
    }

    pub fn attach_coordinates_to_array<T: NumericElement>(
        &self,
        mut array: DataArray<T>,
    ) -> Result<DataArray<T>> {
        if array.shape_yx()? != self.shape() {
            return Err(RustySatError::invalid_input(format!(
                "data array y/x shape {:?} does not match swath shape {:?}",
                array.shape_yx()?,
                self.shape()
            )));
        }
        array.set_coordinate("longitude", self.longitude_coordinate()?)?;
        array.set_coordinate("latitude", self.latitude_coordinate()?)?;
        Ok(array)
    }
}

impl GeometryDefinition for SwathDefinition {
    fn kind(&self) -> GeometryKind {
        GeometryKind::Swath
    }

    fn shape(&self) -> Vec<usize> {
        vec![self.height, self.width]
    }
}

pub fn load_swath_from_str(yaml: &str, swath_id: &str) -> Result<SwathDefinition> {
    load_swaths_from_str(yaml)?
        .remove(swath_id)
        .ok_or_else(|| RustySatError::not_found(format!("swath definition '{swath_id}'")))
}

pub fn load_swaths_from_str(yaml: &str) -> Result<BTreeMap<String, SwathDefinition>> {
    let value = parse_swath_yaml_value(yaml)?;
    let mapping = value
        .as_mapping()
        .ok_or_else(|| RustySatError::invalid_input("swath YAML root must be a mapping"))?;
    let mut swaths = BTreeMap::new();
    for (key, value) in mapping {
        let swath_id = key
            .as_str()
            .ok_or_else(|| RustySatError::invalid_input("swath id keys must be strings"))?;
        swaths.insert(swath_id.to_string(), parse_swath(swath_id, value)?);
    }
    Ok(swaths)
}

fn parse_swath_yaml_value(yaml: &str) -> Result<Value> {
    if yaml.len() > MAX_SWATH_YAML_BYTES {
        return Err(RustySatError::invalid_input(format!(
            "swath YAML exceeds size limit of {MAX_SWATH_YAML_BYTES} bytes"
        )));
    }
    let value: Value = serde_norway::from_str(yaml)
        .map_err(|err| RustySatError::invalid_input(format!("invalid swath YAML: {err}")))?;
    validate_swath_yaml_depth(&value, 0)?;
    Ok(value)
}

fn validate_swath_yaml_depth(value: &Value, depth: usize) -> Result<()> {
    if depth > MAX_SWATH_YAML_DEPTH {
        return Err(RustySatError::invalid_input(format!(
            "swath YAML exceeds nesting depth limit of {MAX_SWATH_YAML_DEPTH}"
        )));
    }
    if let Some(sequence) = value.as_sequence() {
        for child in sequence {
            validate_swath_yaml_depth(child, depth + 1)?;
        }
    }
    if let Some(mapping) = value.as_mapping() {
        for (key, child) in mapping {
            validate_swath_yaml_depth(key, depth + 1)?;
            validate_swath_yaml_depth(child, depth + 1)?;
        }
    }
    if let Value::Tagged(tagged) = value {
        validate_swath_yaml_depth(&tagged.value, depth + 1)?;
    }
    Ok(())
}

fn parse_swath(swath_id: &str, value: &Value) -> Result<SwathDefinition> {
    let mapping = value.as_mapping().ok_or_else(|| {
        RustySatError::invalid_input(format!("swath '{swath_id}' must be a mapping"))
    })?;
    let (height, width, lons) =
        parse_coordinate_array(required_value(mapping, "lons", swath_id)?, "lons")?;
    let (lat_height, lat_width, lats) =
        parse_coordinate_array(required_value(mapping, "lats", swath_id)?, "lats")?;
    if (height, width) != (lat_height, lat_width) {
        return Err(RustySatError::invalid_input(format!(
            "swath '{swath_id}' lons and lats must have the same shape"
        )));
    }
    let mut swath = SwathDefinition::from_lonlats(height, width, lons, lats)?;
    if let Some(crs) = optional_value(mapping, "crs") {
        swath = swath.with_crs(parse_crs(crs)?)?;
    }
    Ok(swath)
}

fn parse_coordinate_array(value: &Value, name: &str) -> Result<(usize, usize, Vec<f64>)> {
    let rows = value
        .as_sequence()
        .ok_or_else(|| RustySatError::invalid_input(format!("{name} must be a list")))?;
    if rows.is_empty() {
        return Err(RustySatError::invalid_input(format!(
            "{name} cannot be empty"
        )));
    }
    if rows.iter().all(Value::is_sequence) {
        parse_2d_coordinate_array(rows, name)
    } else {
        let values = parse_f64_values(rows, name)?;
        Ok((1, values.len(), values))
    }
}

fn parse_2d_coordinate_array(rows: &[Value], name: &str) -> Result<(usize, usize, Vec<f64>)> {
    let mut values = Vec::new();
    let mut width = None;
    for row in rows {
        let row_values = row
            .as_sequence()
            .ok_or_else(|| RustySatError::invalid_input(format!("{name} rows must be lists")))?;
        if row_values.is_empty() {
            return Err(RustySatError::invalid_input(format!(
                "{name} rows cannot be empty"
            )));
        }
        match width {
            Some(width) if width != row_values.len() => {
                return Err(RustySatError::invalid_input(format!(
                    "{name} rows must have consistent width"
                )));
            }
            None => width = Some(row_values.len()),
            _ => {}
        }
        values.extend(parse_f64_values(row_values, name)?);
    }
    Ok((rows.len(), width.unwrap_or(0), values))
}

fn parse_f64_values(values: &[Value], name: &str) -> Result<Vec<f64>> {
    values
        .iter()
        .map(|value| {
            value.as_f64().ok_or_else(|| {
                RustySatError::invalid_input(format!("{name} values must be numeric"))
            })
        })
        .collect()
}

fn parse_crs(value: &Value) -> Result<BTreeMap<String, String>> {
    match value {
        Value::String(proj) => Ok(BTreeMap::from([("proj4".to_string(), proj.clone())])),
        Value::Mapping(mapping) => {
            let mut crs = BTreeMap::new();
            for (key, value) in mapping {
                let key = key
                    .as_str()
                    .ok_or_else(|| RustySatError::invalid_input("CRS keys must be strings"))?;
                crs.insert(key.to_string(), yaml_scalar_to_string(value)?);
            }
            Ok(crs)
        }
        _ => Err(RustySatError::invalid_input(
            "CRS must be a mapping or PROJ string",
        )),
    }
}

fn validate_shape(height: usize, width: usize) -> Result<()> {
    if height == 0 || width == 0 {
        return Err(RustySatError::invalid_input(
            "swath dimensions must be non-zero",
        ));
    }
    Ok(())
}

fn validate_coordinates(height: usize, width: usize, lons: &[f64], lats: &[f64]) -> Result<()> {
    if lons.len() != lats.len() {
        return Err(RustySatError::invalid_input(
            "lons and lats must have the same length",
        ));
    }
    let expected_len = height * width;
    if lons.len() != expected_len {
        return Err(RustySatError::invalid_input(format!(
            "coordinate length {} does not match swath shape {}x{}",
            lons.len(),
            height,
            width
        )));
    }
    Ok(())
}

fn default_lonlat_crs() -> BTreeMap<String, String> {
    BTreeMap::from([
        ("proj".to_string(), "longlat".to_string()),
        ("ellps".to_string(), "WGS84".to_string()),
    ])
}

fn required_value<'a>(mapping: &'a Mapping, key: &str, swath_id: &str) -> Result<&'a Value> {
    mapping
        .get(Value::String(key.to_string()))
        .ok_or_else(|| RustySatError::invalid_input(format!("swath '{swath_id}' missing '{key}'")))
}

fn optional_value<'a>(mapping: &'a Mapping, key: &str) -> Option<&'a Value> {
    mapping.get(Value::String(key.to_string()))
}

fn yaml_scalar_to_string(value: &Value) -> Result<String> {
    match value {
        Value::String(value) => Ok(value.clone()),
        Value::Number(value) => Ok(value.to_string()),
        Value::Bool(value) => Ok(value.to_string()),
        _ => Err(RustySatError::invalid_input(
            "YAML value must be a scalar string, number, or bool",
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constructs_dimension_only_swath() {
        let swath = SwathDefinition::new(5, 6).unwrap();

        assert_eq!(swath.shape(), (5, 6));
        assert_eq!(swath.size(), 30);
        assert!(!swath.has_coordinates());
        assert_eq!(swath.crs().get("proj").unwrap(), "longlat");
        assert_eq!(swath.crs().get("ellps").unwrap(), "WGS84");
        assert_eq!(
            swath.crs_definition().unwrap().projection_name(),
            Some("longlat")
        );
    }

    #[test]
    fn constructs_coordinate_swath() {
        let swath =
            SwathDefinition::from_lonlats(2, 2, vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0])
                .unwrap();

        assert_eq!(swath.shape(), (2, 2));
        assert_eq!(swath.lons().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(swath.lats().unwrap(), &[5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn attaches_lonlat_coordinates_to_matching_data_array() {
        let swath =
            SwathDefinition::from_lonlats(2, 2, vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0])
                .unwrap();
        let array = DataArray::<u16>::from_vec(vec![2, 2], vec![10, 20, 30, 40]).unwrap();

        let array = swath.attach_coordinates_to_array(array).unwrap();

        assert_eq!(
            array.coord("longitude").unwrap().dims(),
            &["y".to_string(), "x".to_string()]
        );
        assert_eq!(
            array.coord("longitude").unwrap().values(),
            &[1.0, 2.0, 3.0, 4.0]
        );
        assert_eq!(
            array.coord("latitude").unwrap().values(),
            &[5.0, 6.0, 7.0, 8.0]
        );
    }

    #[test]
    fn rejects_swath_coordinate_attachment_without_matching_shape_or_coordinates() {
        let swath =
            SwathDefinition::from_lonlats(2, 2, vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0])
                .unwrap();
        let bad_shape = DataArray::<u8>::from_vec(vec![1, 4], vec![0; 4]).unwrap();
        assert!(swath.attach_coordinates_to_array(bad_shape).is_err());

        let dimension_only = SwathDefinition::new(2, 2).unwrap();
        let array = DataArray::<u8>::from_vec(vec![2, 2], vec![0; 4]).unwrap();
        assert!(dimension_only.attach_coordinates_to_array(array).is_err());
    }

    #[test]
    fn rejects_mismatched_coordinate_lengths() {
        let err = SwathDefinition::from_lonlats(2, 2, vec![1.0, 2.0], vec![1.0]).unwrap_err();

        assert!(matches!(err, RustySatError::InvalidInput { .. }));
    }

    #[test]
    fn loads_2d_swath_yaml() {
        let yaml = r#"
granule:
  lons:
    - [10.0, 11.0]
    - [12.0, 13.0]
  lats:
    - [20.0, 21.0]
    - [22.0, 23.0]
  crs:
    proj: longlat
    ellps: WGS84
"#;

        let swath = load_swath_from_str(yaml, "granule").unwrap();

        assert_eq!(swath.shape(), (2, 2));
        assert_eq!(swath.lons().unwrap(), &[10.0, 11.0, 12.0, 13.0]);
        assert_eq!(swath.lats().unwrap(), &[20.0, 21.0, 22.0, 23.0]);
        assert_eq!(swath.crs().get("proj").unwrap(), "longlat");
    }

    #[test]
    fn loads_1d_swath_yaml() {
        let yaml = r#"
line:
  lons: [10.0, 11.0, 12.0]
  lats: [20.0, 21.0, 22.0]
"#;

        let swath = load_swath_from_str(yaml, "line").unwrap();

        assert_eq!(swath.shape(), (1, 3));
        assert_eq!(swath.size(), 3);
    }
}
