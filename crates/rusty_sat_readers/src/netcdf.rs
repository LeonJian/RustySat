//! NetCDF reader metadata foundations.
//!
//! Reference behavior inspected before implementation:
//! - `satpy/satpy/readers/core/netcdf.py`
//! - `satpy/satpy/readers/fci_l1c_nc.py`
//! - `satpy/doc/source/examples/fci_l1c_natural_color.rst`
//!
//! Satpy's `NetCDF4FileHandler` builds a path-addressable `file_content`
//! index before datasets are loaded. This module implements the same metadata
//! index shape without choosing a native NetCDF backend yet. Future adapters
//! can fill `NetCdfGroup` from `netcdf`, `hdf5`, or another backend while
//! preserving the public lookup behavior.

use rusty_sat_core::{MetadataValue, Result, RustySatError};
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NetCdfVariable {
    name: String,
    dtype: String,
    dimensions: Vec<String>,
    shape: Vec<usize>,
    attrs: BTreeMap<String, MetadataValue>,
}

impl NetCdfVariable {
    pub fn new(
        name: impl Into<String>,
        dtype: impl Into<String>,
        dimensions: impl IntoIterator<Item = impl Into<String>>,
        shape: impl IntoIterator<Item = usize>,
    ) -> Result<Self> {
        let name = validate_component_name("variable", name)?;
        let dtype = dtype.into();
        if dtype.trim().is_empty() {
            return Err(RustySatError::invalid_input(
                "NetCDF variable dtype cannot be empty",
            ));
        }
        let dimensions = dimensions.into_iter().map(Into::into).collect::<Vec<_>>();
        for dimension in &dimensions {
            validate_component_ref("dimension", dimension)?;
        }
        let shape = shape.into_iter().collect::<Vec<_>>();
        if dimensions.len() != shape.len() {
            return Err(RustySatError::invalid_input(format!(
                "NetCDF variable '{name}' has {} dimensions but {} shape entries",
                dimensions.len(),
                shape.len()
            )));
        }
        Ok(Self {
            name,
            dtype,
            dimensions,
            shape,
            attrs: BTreeMap::new(),
        })
    }

    pub fn with_attr(
        mut self,
        key: impl Into<String>,
        value: impl Into<MetadataValue>,
    ) -> Result<Self> {
        self.insert_attr(key, value)?;
        Ok(self)
    }

    pub fn insert_attr(
        &mut self,
        key: impl Into<String>,
        value: impl Into<MetadataValue>,
    ) -> Result<()> {
        let key = validate_attr_name(key)?;
        self.attrs.insert(key, value.into());
        Ok(())
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn dtype(&self) -> &str {
        &self.dtype
    }

    pub fn dimensions(&self) -> &[String] {
        &self.dimensions
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn attrs(&self) -> &BTreeMap<String, MetadataValue> {
        &self.attrs
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NetCdfGroup {
    name: String,
    attrs: BTreeMap<String, MetadataValue>,
    dimensions: BTreeMap<String, usize>,
    variables: BTreeMap<String, NetCdfVariable>,
    groups: BTreeMap<String, NetCdfGroup>,
}

impl NetCdfGroup {
    pub fn root() -> Self {
        Self {
            name: String::new(),
            attrs: BTreeMap::new(),
            dimensions: BTreeMap::new(),
            variables: BTreeMap::new(),
            groups: BTreeMap::new(),
        }
    }

    pub fn new(name: impl Into<String>) -> Result<Self> {
        Ok(Self {
            name: validate_component_name("group", name)?,
            attrs: BTreeMap::new(),
            dimensions: BTreeMap::new(),
            variables: BTreeMap::new(),
            groups: BTreeMap::new(),
        })
    }

    pub fn with_attr(
        mut self,
        key: impl Into<String>,
        value: impl Into<MetadataValue>,
    ) -> Result<Self> {
        self.insert_attr(key, value)?;
        Ok(self)
    }

    pub fn with_dimension(mut self, name: impl Into<String>, length: usize) -> Result<Self> {
        self.insert_dimension(name, length)?;
        Ok(self)
    }

    pub fn with_variable(mut self, variable: NetCdfVariable) -> Result<Self> {
        self.insert_variable(variable)?;
        Ok(self)
    }

    pub fn with_group(mut self, group: NetCdfGroup) -> Result<Self> {
        self.insert_group(group)?;
        Ok(self)
    }

    pub fn insert_attr(
        &mut self,
        key: impl Into<String>,
        value: impl Into<MetadataValue>,
    ) -> Result<()> {
        let key = validate_attr_name(key)?;
        self.attrs.insert(key, value.into());
        Ok(())
    }

    pub fn insert_dimension(&mut self, name: impl Into<String>, length: usize) -> Result<()> {
        let name = validate_component_name("dimension", name)?;
        self.dimensions.insert(name, length);
        Ok(())
    }

    pub fn insert_variable(&mut self, variable: NetCdfVariable) -> Result<()> {
        self.variables.insert(variable.name.clone(), variable);
        Ok(())
    }

    pub fn insert_group(&mut self, group: NetCdfGroup) -> Result<()> {
        if group.name.is_empty() {
            return Err(RustySatError::invalid_input(
                "nested NetCDF group name cannot be empty",
            ));
        }
        self.groups.insert(group.name.clone(), group);
        Ok(())
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn attrs(&self) -> &BTreeMap<String, MetadataValue> {
        &self.attrs
    }

    pub fn dimensions(&self) -> &BTreeMap<String, usize> {
        &self.dimensions
    }

    pub fn variables(&self) -> &BTreeMap<String, NetCdfVariable> {
        &self.variables
    }

    pub fn groups(&self) -> &BTreeMap<String, NetCdfGroup> {
        &self.groups
    }

    fn group_at_path(&self, path: &str) -> Option<&Self> {
        if path.is_empty() {
            return Some(self);
        }
        let mut group = self;
        for component in path.split('/') {
            group = group.groups.get(component)?;
        }
        Some(group)
    }

    fn variable_at_path(&self, path: &str) -> Option<(&Self, &NetCdfVariable)> {
        let (group_path, variable_name) = split_parent_path(path);
        let group = self.group_at_path(group_path)?;
        group
            .variables
            .get(variable_name)
            .map(|variable| (group, variable))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NetCdfContent {
    Group,
    Variable,
    DType(String),
    Shape(Vec<usize>),
    Dimensions(Vec<String>),
    DimensionLength(usize),
    Attribute(MetadataValue),
    Attributes(BTreeMap<String, MetadataValue>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NetCdfMetadata {
    content: BTreeMap<String, NetCdfContent>,
}

impl NetCdfMetadata {
    pub fn collect(root: &NetCdfGroup) -> Result<Self> {
        let mut metadata = Self {
            content: BTreeMap::new(),
        };
        metadata.collect_group("", root);
        Ok(metadata)
    }

    pub fn collect_required(
        root: &NetCdfGroup,
        required_variables: impl IntoIterator<Item = impl AsRef<str>>,
        replacements: &BTreeMap<String, Vec<String>>,
    ) -> Result<Self> {
        let mut metadata = Self {
            content: BTreeMap::new(),
        };
        for path in expand_required_variable_names(required_variables, replacements)? {
            metadata.collect_required_path(root, &path)?;
        }
        Ok(metadata)
    }

    pub fn contains(&self, key: &str) -> bool {
        self.content.contains_key(key)
    }

    pub fn get(&self, key: &str) -> Option<&NetCdfContent> {
        self.content.get(key)
    }

    pub fn get_attr(&self, key: &str) -> Option<&MetadataValue> {
        match self.get(key) {
            Some(NetCdfContent::Attribute(value)) => Some(value),
            _ => None,
        }
    }

    pub fn global_attrs(&self) -> Option<&BTreeMap<String, MetadataValue>> {
        match self.get("/attrs") {
            Some(NetCdfContent::Attributes(attrs)) => Some(attrs),
            _ => None,
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = (&String, &NetCdfContent)> {
        self.content.iter()
    }

    fn collect_group(&mut self, path: &str, group: &NetCdfGroup) {
        if !path.is_empty() {
            self.content.insert(path.to_string(), NetCdfContent::Group);
            self.collect_attrs(path, group.attrs());
        } else {
            self.collect_global_attrs(group.attrs());
        }
        self.collect_dimensions(path, group.dimensions());
        for (name, child) in group.groups() {
            let child_path = join_path(path, name);
            self.collect_group(&child_path, child);
        }
        for (name, variable) in group.variables() {
            let variable_path = join_path(path, name);
            self.collect_variable(&variable_path, variable);
        }
    }

    fn collect_variable(&mut self, path: &str, variable: &NetCdfVariable) {
        self.content
            .insert(path.to_string(), NetCdfContent::Variable);
        self.content.insert(
            format!("{path}/dtype"),
            NetCdfContent::DType(variable.dtype().to_string()),
        );
        self.content.insert(
            format!("{path}/shape"),
            NetCdfContent::Shape(variable.shape().to_vec()),
        );
        self.content.insert(
            format!("{path}/dimensions"),
            NetCdfContent::Dimensions(variable.dimensions().to_vec()),
        );
        self.collect_attrs(path, variable.attrs());
    }

    fn collect_global_attrs(&mut self, attrs: &BTreeMap<String, MetadataValue>) {
        for (key, value) in attrs {
            self.content.insert(
                format!("/attr/{key}"),
                NetCdfContent::Attribute(value.clone()),
            );
        }
        self.content.insert(
            "/attrs".to_string(),
            NetCdfContent::Attributes(attrs.clone()),
        );
    }

    fn collect_attrs(&mut self, path: &str, attrs: &BTreeMap<String, MetadataValue>) {
        for (key, value) in attrs {
            self.content.insert(
                format!("{path}/attr/{key}"),
                NetCdfContent::Attribute(value.clone()),
            );
        }
    }

    fn collect_dimensions(&mut self, path: &str, dimensions: &BTreeMap<String, usize>) {
        for (name, length) in dimensions {
            self.content.insert(
                format!("{path}/dimension/{name}"),
                NetCdfContent::DimensionLength(*length),
            );
        }
    }

    fn collect_required_path(&mut self, root: &NetCdfGroup, path: &str) -> Result<()> {
        if let Some((object_path, attr_name)) = split_attr_path(path) {
            self.collect_required_attr(root, object_path, attr_name)
        } else {
            let (group, variable) = root
                .variable_at_path(path)
                .ok_or_else(|| RustySatError::not_found(format!("NetCDF variable '{path}'")))?;
            self.collect_variable(path, variable);
            let (group_path, _) = split_parent_path(path);
            self.collect_dimensions(group_path, group.dimensions());
            Ok(())
        }
    }

    fn collect_required_attr(
        &mut self,
        root: &NetCdfGroup,
        object_path: &str,
        attr_name: &str,
    ) -> Result<()> {
        if object_path.is_empty() {
            let value = root.attrs().get(attr_name).ok_or_else(|| {
                RustySatError::not_found(format!("NetCDF global attribute '{attr_name}'"))
            })?;
            self.content.insert(
                format!("/attr/{attr_name}"),
                NetCdfContent::Attribute(value.clone()),
            );
            return Ok(());
        }
        if let Some(group) = root.group_at_path(object_path) {
            let value = group.attrs().get(attr_name).ok_or_else(|| {
                RustySatError::not_found(format!(
                    "NetCDF group attribute '{object_path}/attr/{attr_name}'"
                ))
            })?;
            self.content.insert(
                format!("{object_path}/attr/{attr_name}"),
                NetCdfContent::Attribute(value.clone()),
            );
            return Ok(());
        }
        let (_, variable) = root.variable_at_path(object_path).ok_or_else(|| {
            RustySatError::not_found(format!(
                "NetCDF object for attribute '{object_path}/attr/{attr_name}'"
            ))
        })?;
        let value = variable.attrs().get(attr_name).ok_or_else(|| {
            RustySatError::not_found(format!(
                "NetCDF variable attribute '{object_path}/attr/{attr_name}'"
            ))
        })?;
        self.content.insert(
            format!("{object_path}/attr/{attr_name}"),
            NetCdfContent::Attribute(value.clone()),
        );
        Ok(())
    }
}

fn expand_required_variable_names(
    required_variables: impl IntoIterator<Item = impl AsRef<str>>,
    replacements: &BTreeMap<String, Vec<String>>,
) -> Result<Vec<String>> {
    let mut expanded = Vec::new();
    for variable in required_variables {
        let variable = variable.as_ref();
        let Some((key, values)) = replacements
            .iter()
            .find(|(key, _)| variable.contains(&format!("{{{key}}}")))
        else {
            expanded.push(variable.to_string());
            continue;
        };
        for value in values {
            if value.trim().is_empty() {
                return Err(RustySatError::invalid_input(format!(
                    "NetCDF variable replacement '{key}' cannot contain an empty value"
                )));
            }
            expanded.push(variable.replace(&format!("{{{key}}}"), value));
        }
    }
    Ok(expanded)
}

fn split_attr_path(path: &str) -> Option<(&str, &str)> {
    let (object_path, attr_name) = path.rsplit_once("/attr/")?;
    Some((object_path, attr_name))
}

fn split_parent_path(path: &str) -> (&str, &str) {
    path.rsplit_once('/').unwrap_or(("", path))
}

fn join_path(parent: &str, child: &str) -> String {
    if parent.is_empty() {
        child.to_string()
    } else {
        format!("{parent}/{child}")
    }
}

fn validate_component_name(kind: &str, name: impl Into<String>) -> Result<String> {
    let name = name.into();
    validate_component_ref(kind, &name)?;
    Ok(name)
}

fn validate_component_ref(kind: &str, name: &str) -> Result<()> {
    if name.trim().is_empty() {
        return Err(RustySatError::invalid_input(format!(
            "NetCDF {kind} name cannot be empty"
        )));
    }
    if name.contains('/') {
        return Err(RustySatError::invalid_input(format!(
            "NetCDF {kind} name cannot contain '/'"
        )));
    }
    Ok(())
}

fn validate_attr_name(key: impl Into<String>) -> Result<String> {
    let key = key.into();
    if key.trim().is_empty() {
        return Err(RustySatError::invalid_input(
            "NetCDF attribute name cannot be empty",
        ));
    }
    if key.contains('/') {
        return Err(RustySatError::invalid_input(
            "NetCDF attribute name cannot contain '/'",
        ));
    }
    Ok(key)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fci_like_root() -> NetCdfGroup {
        let effective_radiance =
            NetCdfVariable::new("effective_radiance", "u16", ["y", "x"], [2, 3])
                .unwrap()
                .with_attr("units", "mW m-2 sr-1 (cm-1)-1")
                .unwrap()
                .with_attr("scale_factor", MetadataValue::float(0.01).unwrap())
                .unwrap();
        let measured = NetCdfGroup::new("measured")
            .unwrap()
            .with_dimension("y", 2)
            .unwrap()
            .with_dimension("x", 3)
            .unwrap()
            .with_variable(effective_radiance)
            .unwrap();
        let channel = NetCdfGroup::new("vis_04")
            .unwrap()
            .with_group(measured)
            .unwrap();
        let data = NetCdfGroup::new("data")
            .unwrap()
            .with_group(channel)
            .unwrap();
        NetCdfGroup::root()
            .with_attr("platform", "MTG-I1")
            .unwrap()
            .with_group(data)
            .unwrap()
    }

    #[test]
    fn collects_satpy_style_file_content_keys() {
        let metadata = NetCdfMetadata::collect(&fci_like_root()).unwrap();

        assert!(metadata.contains("data"));
        assert!(metadata.contains("data/vis_04"));
        assert!(metadata.contains("data/vis_04/measured/effective_radiance"));
        assert_eq!(
            metadata.get("data/vis_04/measured/effective_radiance/dtype"),
            Some(&NetCdfContent::DType("u16".to_string()))
        );
        assert_eq!(
            metadata.get("data/vis_04/measured/effective_radiance/shape"),
            Some(&NetCdfContent::Shape(vec![2, 3]))
        );
        assert_eq!(
            metadata.get("data/vis_04/measured/effective_radiance/dimensions"),
            Some(&NetCdfContent::Dimensions(vec![
                "y".to_string(),
                "x".to_string()
            ]))
        );
        assert_eq!(
            metadata.get_attr("data/vis_04/measured/effective_radiance/attr/units"),
            Some(&MetadataValue::String("mW m-2 sr-1 (cm-1)-1".to_string()))
        );
        assert_eq!(
            metadata
                .global_attrs()
                .and_then(|attrs| attrs.get("platform")),
            Some(&MetadataValue::String("MTG-I1".to_string()))
        );
        assert_eq!(
            metadata.get("data/vis_04/measured/dimension/x"),
            Some(&NetCdfContent::DimensionLength(3))
        );
    }

    #[test]
    fn collect_required_expands_satpy_variable_replacements() {
        let mut replacements = BTreeMap::new();
        replacements.insert("channel".to_string(), vec!["vis_04".to_string()]);

        let metadata = NetCdfMetadata::collect_required(
            &fci_like_root(),
            [
                "data/{channel}/measured/effective_radiance",
                "data/{channel}/measured/effective_radiance/attr/units",
                "/attr/platform",
            ],
            &replacements,
        )
        .unwrap();

        assert!(metadata.contains("data/vis_04/measured/effective_radiance"));
        assert_eq!(
            metadata.get("data/vis_04/measured/effective_radiance/shape"),
            Some(&NetCdfContent::Shape(vec![2, 3]))
        );
        assert_eq!(
            metadata.get_attr("data/vis_04/measured/effective_radiance/attr/units"),
            Some(&MetadataValue::String("mW m-2 sr-1 (cm-1)-1".to_string()))
        );
        assert_eq!(
            metadata.get_attr("/attr/platform"),
            Some(&MetadataValue::String("MTG-I1".to_string()))
        );
        assert!(!metadata.contains("data"));
    }

    #[test]
    fn rejects_shape_dimension_mismatch() {
        let err = NetCdfVariable::new("radiance", "f32", ["y", "x"], [2])
            .unwrap_err()
            .to_string();

        assert!(err.contains("has 2 dimensions but 1 shape entries"));
    }

    #[test]
    fn required_collection_reports_missing_variable() {
        let err = NetCdfMetadata::collect_required(
            &fci_like_root(),
            ["data/vis_05/measured/radiance"],
            &BTreeMap::new(),
        )
        .unwrap_err()
        .to_string();

        assert!(err.contains("NetCDF variable 'data/vis_05/measured/radiance'"));
    }
}
