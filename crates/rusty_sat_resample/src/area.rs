//! Area definition model and Satpy/Pyresample-style YAML loading.
//!
//! Reference behavior inspected before implementation:
//! - `deps/pyresample/pyresample/area_config.py`
//! - `deps/pyresample/docs/source/concepts/geometries.rst`
//! - `satpy/utils/coord2area_def.py`

use rusty_sat_core::{Result, RustySatError};
use serde_yaml::{Mapping, Value};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, PartialEq)]
pub struct AreaDefinition {
    id: String,
    description: String,
    proj_id: String,
    projection: BTreeMap<String, String>,
    height: usize,
    width: usize,
    area_extent: [f64; 4],
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PixelResolution {
    x: f64,
    y: f64,
}

impl PixelResolution {
    pub fn new(x: f64, y: f64) -> Result<Self> {
        if x <= 0.0 || y <= 0.0 {
            return Err(RustySatError::invalid_input(
                "resolution values must be positive",
            ));
        }
        Ok(Self { x, y })
    }

    pub fn square(value: f64) -> Result<Self> {
        Self::new(value, value)
    }

    pub fn x(&self) -> f64 {
        self.x
    }

    pub fn y(&self) -> f64 {
        self.y
    }
}

impl AreaDefinition {
    pub fn new(id: impl Into<String>, height: usize, width: usize) -> Result<Self> {
        Self::from_parts(
            id.into(),
            String::new(),
            String::new(),
            BTreeMap::new(),
            height,
            width,
            [0.0, 0.0, width as f64, height as f64],
        )
    }

    pub fn from_parts(
        id: impl Into<String>,
        description: impl Into<String>,
        proj_id: impl Into<String>,
        projection: BTreeMap<String, String>,
        height: usize,
        width: usize,
        area_extent: [f64; 4],
    ) -> Result<Self> {
        let id = id.into();
        if id.trim().is_empty() {
            return Err(RustySatError::invalid_input("area id cannot be empty"));
        }
        if height == 0 || width == 0 {
            return Err(RustySatError::invalid_input(
                "area dimensions must be non-zero",
            ));
        }
        if area_extent[0] >= area_extent[2] || area_extent[1] >= area_extent[3] {
            return Err(RustySatError::invalid_input(
                "area_extent must be [lower_left_x, lower_left_y, upper_right_x, upper_right_y]",
            ));
        }
        let description = description.into();
        let proj_id = proj_id.into();
        Ok(Self {
            id: id.clone(),
            description: if description.is_empty() {
                id.clone()
            } else {
                description
            },
            proj_id: if proj_id.is_empty() {
                id.clone()
            } else {
                proj_id
            },
            projection,
            height,
            width,
            area_extent,
        })
    }

    pub fn from_extent_and_resolution(
        id: impl Into<String>,
        description: impl Into<String>,
        proj_id: impl Into<String>,
        projection: BTreeMap<String, String>,
        area_extent: [f64; 4],
        resolution: PixelResolution,
    ) -> Result<Self> {
        let (height, width) = shape_from_extent_and_resolution(area_extent, resolution)?;
        Self::from_parts(
            id,
            description,
            proj_id,
            projection,
            height,
            width,
            area_extent,
        )
    }

    pub fn from_center_radius_resolution(
        id: impl Into<String>,
        description: impl Into<String>,
        proj_id: impl Into<String>,
        projection: BTreeMap<String, String>,
        center: [f64; 2],
        radius: [f64; 2],
        resolution: PixelResolution,
    ) -> Result<Self> {
        if radius[0] <= 0.0 || radius[1] <= 0.0 {
            return Err(RustySatError::invalid_input(
                "radius values must be positive",
            ));
        }
        let area_extent = [
            center[0] - radius[0],
            center[1] - radius[1],
            center[0] + radius[0],
            center[1] + radius[1],
        ];
        Self::from_extent_and_resolution(
            id,
            description,
            proj_id,
            projection,
            area_extent,
            resolution,
        )
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn description(&self) -> &str {
        &self.description
    }

    pub fn proj_id(&self) -> &str {
        &self.proj_id
    }

    pub fn projection(&self) -> &BTreeMap<String, String> {
        &self.projection
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.height, self.width)
    }

    pub fn area_extent(&self) -> [f64; 4] {
        self.area_extent
    }

    pub fn pixel_size(&self) -> (f64, f64) {
        (
            (self.area_extent[2] - self.area_extent[0]) / self.width as f64,
            (self.area_extent[3] - self.area_extent[1]) / self.height as f64,
        )
    }

    pub fn projection_x_coords(&self) -> Vec<f64> {
        let (pixel_size_x, _) = self.pixel_size();
        (0..self.width)
            .map(|x| self.area_extent[0] + (x as f64 + 0.5) * pixel_size_x)
            .collect()
    }

    pub fn projection_y_coords(&self) -> Vec<f64> {
        let (_, pixel_size_y) = self.pixel_size();
        (0..self.height)
            .map(|y| self.area_extent[3] - (y as f64 + 0.5) * pixel_size_y)
            .collect()
    }
}

fn shape_from_extent_and_resolution(
    area_extent: [f64; 4],
    resolution: PixelResolution,
) -> Result<(usize, usize)> {
    if area_extent[0] >= area_extent[2] || area_extent[1] >= area_extent[3] {
        return Err(RustySatError::invalid_input(
            "area_extent must be [lower_left_x, lower_left_y, upper_right_x, upper_right_y]",
        ));
    }
    let width = round_dimension((area_extent[2] - area_extent[0]) / resolution.x, "width")?;
    let height = round_dimension((area_extent[3] - area_extent[1]) / resolution.y, "height")?;
    Ok((height, width))
}

fn round_dimension(value: f64, name: &str) -> Result<usize> {
    if !value.is_finite() || value <= 0.0 {
        return Err(RustySatError::invalid_input(format!(
            "{name} derived from extent and resolution must be positive"
        )));
    }
    let rounded = value.round();
    if (value - rounded).abs() > 1e-8 && value - value.floor() >= 0.01 {
        return usize::try_from(value.ceil() as u64)
            .map_err(|err| RustySatError::invalid_input(format!("{name} out of range: {err}")));
    }
    usize::try_from(rounded as u64)
        .map_err(|err| RustySatError::invalid_input(format!("{name} out of range: {err}")))
}

pub fn load_area_from_file(path: impl AsRef<Path>, area_id: &str) -> Result<AreaDefinition> {
    let contents = fs::read_to_string(path.as_ref()).map_err(|err| {
        RustySatError::not_found(format!("area file '{}': {err}", path.as_ref().display()))
    })?;
    load_area_from_str(&contents, area_id)
}

pub fn load_area_from_str(yaml: &str, area_id: &str) -> Result<AreaDefinition> {
    load_areas_from_str(yaml)?
        .remove(area_id)
        .ok_or_else(|| RustySatError::not_found(format!("area definition '{area_id}'")))
}

pub fn load_areas_from_str(yaml: &str) -> Result<BTreeMap<String, AreaDefinition>> {
    let value: Value = serde_yaml::from_str(yaml)
        .map_err(|err| RustySatError::invalid_input(format!("invalid area YAML: {err}")))?;
    let mapping = value
        .as_mapping()
        .ok_or_else(|| RustySatError::invalid_input("area YAML root must be a mapping"))?;
    let mut areas = BTreeMap::new();
    for (key, value) in mapping {
        let area_id = key
            .as_str()
            .ok_or_else(|| RustySatError::invalid_input("area id keys must be strings"))?;
        let area = parse_area(area_id, value)?;
        areas.insert(area_id.to_string(), area);
    }
    Ok(areas)
}

fn parse_area(area_id: &str, value: &Value) -> Result<AreaDefinition> {
    let mapping = value.as_mapping().ok_or_else(|| {
        RustySatError::invalid_input(format!("area '{area_id}' must be a mapping"))
    })?;
    let description =
        optional_string(mapping, "description")?.unwrap_or_else(|| area_id.to_string());
    let proj_id = optional_string(mapping, "proj_id")?.unwrap_or_else(|| area_id.to_string());
    let projection = parse_projection(required_value(mapping, "projection", area_id)?)?;
    let shape = optional_value(mapping, "shape")
        .map(parse_shape)
        .transpose()?;
    let area_extent = optional_value(mapping, "area_extent")
        .map(parse_area_extent)
        .transpose()?;
    let resolution = optional_value(mapping, "resolution")
        .map(parse_resolution)
        .transpose()?;
    let center = optional_value(mapping, "center")
        .map(|value| parse_xy(value, "center"))
        .transpose()?;
    let radius = optional_value(mapping, "radius")
        .map(|value| parse_xy(value, "radius"))
        .transpose()?;

    match (shape, area_extent, resolution, center, radius) {
        (Some((height, width)), Some(area_extent), _, _, _) => AreaDefinition::from_parts(
            area_id.to_string(),
            description,
            proj_id,
            projection,
            height,
            width,
            area_extent,
        ),
        (None, Some(area_extent), Some(resolution), _, _) => AreaDefinition::from_extent_and_resolution(
            area_id.to_string(),
            description,
            proj_id,
            projection,
            area_extent,
            resolution,
        ),
        (None, None, Some(resolution), Some(center), Some(radius)) => {
            AreaDefinition::from_center_radius_resolution(
                area_id.to_string(),
                description,
                proj_id,
                projection,
                center,
                radius,
                resolution,
            )
        }
        _ => Err(RustySatError::invalid_input(format!(
            "area '{area_id}' requires shape and area_extent, area_extent and resolution, or center/radius/resolution"
        ))),
    }
}

fn required_value<'a>(mapping: &'a Mapping, key: &str, area_id: &str) -> Result<&'a Value> {
    mapping
        .get(Value::String(key.to_string()))
        .ok_or_else(|| RustySatError::invalid_input(format!("area '{area_id}' missing '{key}'")))
}

fn optional_value<'a>(mapping: &'a Mapping, key: &str) -> Option<&'a Value> {
    mapping.get(Value::String(key.to_string()))
}

fn optional_string(mapping: &Mapping, key: &str) -> Result<Option<String>> {
    mapping
        .get(Value::String(key.to_string()))
        .map(yaml_scalar_to_string)
        .transpose()
}

fn parse_projection(value: &Value) -> Result<BTreeMap<String, String>> {
    match value {
        Value::String(proj) => Ok(BTreeMap::from([("proj4".to_string(), proj.clone())])),
        Value::Mapping(mapping) => {
            let mut projection = BTreeMap::new();
            for (key, value) in mapping {
                let key = key.as_str().ok_or_else(|| {
                    RustySatError::invalid_input("projection keys must be strings")
                })?;
                projection.insert(key.to_string(), yaml_scalar_to_string(value)?);
            }
            Ok(projection)
        }
        _ => Err(RustySatError::invalid_input(
            "projection must be a mapping or PROJ string",
        )),
    }
}

fn parse_shape(value: &Value) -> Result<(usize, usize)> {
    let mapping = value
        .as_mapping()
        .ok_or_else(|| RustySatError::invalid_input("shape must be a mapping"))?;
    let height = parse_usize(required_value(mapping, "height", "shape")?, "shape.height")?;
    let width = parse_usize(required_value(mapping, "width", "shape")?, "shape.width")?;
    Ok((height, width))
}

fn parse_area_extent(value: &Value) -> Result<[f64; 4]> {
    if let Value::Sequence(values) = value {
        return parse_f64_quad(values, "area_extent");
    }
    let mapping = value
        .as_mapping()
        .ok_or_else(|| RustySatError::invalid_input("area_extent must be a mapping or list"))?;
    let lower_left = parse_f64_pair(
        required_value(mapping, "lower_left_xy", "area_extent")?
            .as_sequence()
            .ok_or_else(|| RustySatError::invalid_input("lower_left_xy must be a list"))?,
        "lower_left_xy",
    )?;
    let upper_right = parse_f64_pair(
        required_value(mapping, "upper_right_xy", "area_extent")?
            .as_sequence()
            .ok_or_else(|| RustySatError::invalid_input("upper_right_xy must be a list"))?,
        "upper_right_xy",
    )?;
    Ok([lower_left[0], lower_left[1], upper_right[0], upper_right[1]])
}

fn parse_resolution(value: &Value) -> Result<PixelResolution> {
    match value {
        Value::Number(_) => PixelResolution::square(parse_f64(value, "resolution")?),
        Value::Sequence(values) => {
            let values = parse_f64_pair(values, "resolution")?;
            PixelResolution::new(values[0], values[1])
        }
        Value::Mapping(mapping) => {
            if let Some(value) = optional_value(mapping, "resolution") {
                return parse_resolution(value);
            }
            let dx = parse_f64(
                required_value(mapping, "dx", "resolution")?,
                "resolution.dx",
            )?;
            let dy = parse_f64(
                required_value(mapping, "dy", "resolution")?,
                "resolution.dy",
            )?;
            PixelResolution::new(dx, dy)
        }
        _ => Err(RustySatError::invalid_input(
            "resolution must be a number, list, or mapping",
        )),
    }
}

fn parse_xy(value: &Value, name: &str) -> Result<[f64; 2]> {
    match value {
        Value::Sequence(values) => parse_f64_pair(values, name),
        Value::Mapping(mapping) => {
            if let Some(value) = optional_value(mapping, name) {
                return parse_xy(value, name);
            }
            if let (Some(x_value), Some(y_value)) =
                (optional_value(mapping, "x"), optional_value(mapping, "y"))
            {
                let x = parse_f64(x_value, &format!("{name}.x"))?;
                let y = parse_f64(y_value, &format!("{name}.y"))?;
                return Ok([x, y]);
            }
            if let (Some(dx_value), Some(dy_value)) =
                (optional_value(mapping, "dx"), optional_value(mapping, "dy"))
            {
                let dx = parse_f64(dx_value, &format!("{name}.dx"))?;
                let dy = parse_f64(dy_value, &format!("{name}.dy"))?;
                return Ok([dx, dy]);
            }
            Err(RustySatError::invalid_input(format!(
                "{name} must provide x/y, dx/dy, or {name}"
            )))
        }
        _ => Err(RustySatError::invalid_input(format!(
            "{name} must be a list or mapping"
        ))),
    }
}

fn parse_f64_pair(values: &[Value], name: &str) -> Result<[f64; 2]> {
    if values.len() != 2 {
        return Err(RustySatError::invalid_input(format!(
            "{name} must contain 2 numeric values"
        )));
    }
    Ok([parse_f64(&values[0], name)?, parse_f64(&values[1], name)?])
}

fn parse_f64_quad(values: &[Value], name: &str) -> Result<[f64; 4]> {
    if values.len() != 4 {
        return Err(RustySatError::invalid_input(format!(
            "{name} must contain 4 numeric values"
        )));
    }
    Ok([
        parse_f64(&values[0], name)?,
        parse_f64(&values[1], name)?,
        parse_f64(&values[2], name)?,
        parse_f64(&values[3], name)?,
    ])
}

fn parse_usize(value: &Value, name: &str) -> Result<usize> {
    let number = value.as_u64().ok_or_else(|| {
        RustySatError::invalid_input(format!("{name} must be a positive integer"))
    })?;
    usize::try_from(number)
        .map_err(|err| RustySatError::invalid_input(format!("{name} out of range: {err}")))
}

fn parse_f64(value: &Value, name: &str) -> Result<f64> {
    value
        .as_f64()
        .ok_or_else(|| RustySatError::invalid_input(format!("{name} must be numeric")))
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
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn constructs_area_definition_with_pyresample_shape_order() {
        let area = AreaDefinition::from_parts(
            "test_area",
            "Test Area",
            "test_proj",
            BTreeMap::from([("proj".to_string(), "stere".to_string())]),
            10,
            20,
            [-100.0, -50.0, 100.0, 50.0],
        )
        .unwrap();

        assert_eq!(area.id(), "test_area");
        assert_eq!(area.description(), "Test Area");
        assert_eq!(area.proj_id(), "test_proj");
        assert_eq!(area.shape(), (10, 20));
        assert_eq!(area.area_extent(), [-100.0, -50.0, 100.0, 50.0]);
        assert_eq!(area.pixel_size(), (10.0, 10.0));
        assert_eq!(area.projection_x_coords()[..3], [-95.0, -85.0, -75.0]);
        assert_eq!(area.projection_y_coords()[..3], [45.0, 35.0, 25.0]);
    }

    #[test]
    fn loads_satpy_style_area_yaml() {
        let yaml = r#"
france:
  description: france
  projection:
    proj: stere
    ellps: WGS84
    lat_0: 46.75
    lon_0: 1.25
  shape:
    height: 703
    width: 746
  area_extent:
    lower_left_xy: [-559750.381098, -505020.675776]
    upper_right_xy: [559750.381098, 549517.351948]
"#;

        let area = load_area_from_str(yaml, "france").unwrap();

        assert_eq!(area.description(), "france");
        assert_eq!(area.shape(), (703, 746));
        assert_eq!(area.projection().get("proj").unwrap(), "stere");
        assert_eq!(area.projection().get("ellps").unwrap(), "WGS84");
        assert_eq!(
            area.area_extent(),
            [-559750.381098, -505020.675776, 559750.381098, 549517.351948]
        );
    }

    #[test]
    fn loads_flat_area_extent_yaml() {
        let yaml = r#"
simple:
  projection: "+proj=latlong"
  shape:
    height: 2
    width: 3
  area_extent: [-10.0, -5.0, 10.0, 5.0]
"#;

        let area = load_area_from_str(yaml, "simple").unwrap();

        assert_eq!(area.projection().get("proj4").unwrap(), "+proj=latlong");
        assert_eq!(area.shape(), (2, 3));
        assert_eq!(area.area_extent(), [-10.0, -5.0, 10.0, 5.0]);
    }

    #[test]
    fn derives_shape_from_extent_and_resolution() {
        let area = AreaDefinition::from_extent_and_resolution(
            "derived",
            "Derived",
            "derived_proj",
            BTreeMap::from([("proj".to_string(), "latlong".to_string())]),
            [-10.0, -5.0, 10.0, 5.0],
            PixelResolution::new(2.0, 5.0).unwrap(),
        )
        .unwrap();

        assert_eq!(area.shape(), (2, 10));
        assert_eq!(area.pixel_size(), (2.0, 5.0));
    }

    #[test]
    fn derives_extent_and_shape_from_center_radius_resolution_yaml() {
        let yaml = r#"
centered:
  projection:
    proj: laea
  center:
    x: 100.0
    y: 200.0
  radius:
    dx: 40.0
    dy: 20.0
  resolution:
    dx: 10.0
    dy: 5.0
"#;

        let area = load_area_from_str(yaml, "centered").unwrap();

        assert_eq!(area.area_extent(), [60.0, 180.0, 140.0, 220.0]);
        assert_eq!(area.shape(), (8, 8));
        assert_eq!(area.pixel_size(), (10.0, 5.0));
    }

    #[test]
    fn derives_shape_from_yaml_extent_and_scalar_resolution() {
        let yaml = r#"
dynamic_like:
  projection:
    proj: latlong
  area_extent: [-20.0, -10.0, 20.0, 10.0]
  resolution: 10.0
"#;

        let area = load_area_from_str(yaml, "dynamic_like").unwrap();

        assert_eq!(area.shape(), (2, 4));
        assert_eq!(area.pixel_size(), (10.0, 10.0));
    }

    #[test]
    fn reports_missing_area() {
        let err = load_area_from_str("known: {}", "missing").unwrap_err();

        assert!(matches!(
            err,
            RustySatError::InvalidInput { .. } | RustySatError::NotFound { .. }
        ));
    }

    #[test]
    fn loads_area_from_file() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("rusty_sat_area_{nonce}.yaml"));
        fs::write(
            &path,
            r#"
file_area:
  projection:
    proj: latlong
  shape:
    height: 1
    width: 1
  area_extent: [0.0, 0.0, 1.0, 1.0]
"#,
        )
        .unwrap();

        let area = load_area_from_file(&path, "file_area").unwrap();
        fs::remove_file(path).unwrap();

        assert_eq!(area.id(), "file_area");
    }
}
