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

use rusty_sat_core::{
    AnyDataArray, DataId, Dataset, MetadataValue, Result, RustySatError, ValidityMask,
};
use std::collections::BTreeMap;

pub trait NetCdfMetadataSource {
    fn read_metadata_tree(&self, filename: &str, auto_mask_and_scale: bool) -> Result<NetCdfGroup>;
}

pub trait NetCdfDataSource: NetCdfMetadataSource {
    fn read_array(
        &self,
        filename: &str,
        variable_path: &str,
        auto_mask_and_scale: bool,
    ) -> Result<AnyDataArray>;
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct InMemoryNetCdfSource {
    root: NetCdfGroup,
    arrays: BTreeMap<String, AnyDataArray>,
}

impl InMemoryNetCdfSource {
    pub fn new(root: NetCdfGroup) -> Self {
        Self {
            root,
            arrays: BTreeMap::new(),
        }
    }

    pub fn with_array(
        mut self,
        variable_path: impl Into<String>,
        array: impl Into<AnyDataArray>,
    ) -> Result<Self> {
        self.insert_array(variable_path, array)?;
        Ok(self)
    }

    pub fn insert_array(
        &mut self,
        variable_path: impl Into<String>,
        array: impl Into<AnyDataArray>,
    ) -> Result<()> {
        let variable_path = variable_path.into();
        validate_netcdf_path("NetCDF array", &variable_path)?;
        self.arrays.insert(variable_path, array.into());
        Ok(())
    }
}

impl NetCdfMetadataSource for InMemoryNetCdfSource {
    fn read_metadata_tree(
        &self,
        _filename: &str,
        _auto_mask_and_scale: bool,
    ) -> Result<NetCdfGroup> {
        Ok(self.root.clone())
    }
}

impl NetCdfDataSource for InMemoryNetCdfSource {
    fn read_array(
        &self,
        _filename: &str,
        variable_path: &str,
        _auto_mask_and_scale: bool,
    ) -> Result<AnyDataArray> {
        self.arrays.get(variable_path).cloned().ok_or_else(|| {
            RustySatError::not_found(format!("NetCDF variable data '{variable_path}'"))
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct NetCdfFileTypeInfo {
    required_netcdf_variables: Vec<String>,
    variable_name_replacements: BTreeMap<String, Vec<String>>,
}

impl NetCdfFileTypeInfo {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_required_variables(
        mut self,
        variables: impl IntoIterator<Item = impl Into<String>>,
    ) -> Result<Self> {
        self.set_required_variables(variables)?;
        Ok(self)
    }

    pub fn with_variable_name_replacement(
        mut self,
        key: impl Into<String>,
        values: impl IntoIterator<Item = impl Into<String>>,
    ) -> Result<Self> {
        self.insert_variable_name_replacement(key, values)?;
        Ok(self)
    }

    pub fn set_required_variables(
        &mut self,
        variables: impl IntoIterator<Item = impl Into<String>>,
    ) -> Result<()> {
        self.required_netcdf_variables.clear();
        for variable in variables {
            let variable = variable.into();
            validate_netcdf_path("required NetCDF variable", &variable)?;
            self.required_netcdf_variables.push(variable);
        }
        Ok(())
    }

    pub fn insert_variable_name_replacement(
        &mut self,
        key: impl Into<String>,
        values: impl IntoIterator<Item = impl Into<String>>,
    ) -> Result<()> {
        let key = key.into();
        if key.trim().is_empty() {
            return Err(RustySatError::invalid_input(
                "NetCDF variable replacement key cannot be empty",
            ));
        }
        if key.contains(['{', '}', '/']) {
            return Err(RustySatError::invalid_input(
                "NetCDF variable replacement key cannot contain braces or '/'",
            ));
        }
        let values = values.into_iter().map(Into::into).collect::<Vec<_>>();
        if values.is_empty() {
            return Err(RustySatError::invalid_input(format!(
                "NetCDF variable replacement '{key}' must have at least one value"
            )));
        }
        for value in &values {
            if value.trim().is_empty() {
                return Err(RustySatError::invalid_input(format!(
                    "NetCDF variable replacement '{key}' cannot contain an empty value"
                )));
            }
        }
        self.variable_name_replacements.insert(key, values);
        Ok(())
    }

    pub fn required_netcdf_variables(&self) -> &[String] {
        &self.required_netcdf_variables
    }

    pub fn variable_name_replacements(&self) -> &BTreeMap<String, Vec<String>> {
        &self.variable_name_replacements
    }

    pub fn has_required_variables(&self) -> bool {
        !self.required_netcdf_variables.is_empty()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NetCdfFileHandler {
    filename: String,
    filename_info: BTreeMap<String, MetadataValue>,
    filetype_info: NetCdfFileTypeInfo,
    metadata: NetCdfMetadata,
    auto_mask_and_scale: bool,
}

impl NetCdfFileHandler {
    pub fn from_source(
        filename: impl Into<String>,
        filename_info: BTreeMap<String, MetadataValue>,
        filetype_info: NetCdfFileTypeInfo,
        source: &impl NetCdfMetadataSource,
    ) -> Result<Self> {
        Self::from_source_with_options(filename, filename_info, filetype_info, source, false)
    }

    pub fn from_source_with_options(
        filename: impl Into<String>,
        filename_info: BTreeMap<String, MetadataValue>,
        filetype_info: NetCdfFileTypeInfo,
        source: &impl NetCdfMetadataSource,
        auto_mask_and_scale: bool,
    ) -> Result<Self> {
        let filename = validate_filename(filename)?;
        let root = source.read_metadata_tree(&filename, auto_mask_and_scale)?;
        Self::from_root_with_options(
            filename,
            filename_info,
            filetype_info,
            &root,
            auto_mask_and_scale,
        )
    }

    pub fn from_root(
        filename: impl Into<String>,
        filename_info: BTreeMap<String, MetadataValue>,
        filetype_info: NetCdfFileTypeInfo,
        root: &NetCdfGroup,
    ) -> Result<Self> {
        Self::from_root_with_options(filename, filename_info, filetype_info, root, false)
    }

    pub fn from_root_with_options(
        filename: impl Into<String>,
        filename_info: BTreeMap<String, MetadataValue>,
        filetype_info: NetCdfFileTypeInfo,
        root: &NetCdfGroup,
        auto_mask_and_scale: bool,
    ) -> Result<Self> {
        let filename = validate_filename(filename)?;
        let metadata = if filetype_info.has_required_variables() {
            NetCdfMetadata::collect_required(
                root,
                filetype_info.required_netcdf_variables(),
                filetype_info.variable_name_replacements(),
            )?
        } else {
            NetCdfMetadata::collect(root)?
        };
        Ok(Self {
            filename,
            filename_info,
            filetype_info,
            metadata,
            auto_mask_and_scale,
        })
    }

    pub fn filename(&self) -> &str {
        &self.filename
    }

    pub fn filename_info(&self) -> &BTreeMap<String, MetadataValue> {
        &self.filename_info
    }

    pub fn filetype_info(&self) -> &NetCdfFileTypeInfo {
        &self.filetype_info
    }

    pub fn metadata(&self) -> &NetCdfMetadata {
        &self.metadata
    }

    pub fn auto_mask_and_scale(&self) -> bool {
        self.auto_mask_and_scale
    }

    pub fn contains(&self, key: &str) -> bool {
        self.metadata.contains(key)
    }

    pub fn get(&self, key: &str) -> Option<&NetCdfContent> {
        self.metadata.get(key)
    }

    pub fn get_or_err(&self, key: &str) -> Result<&NetCdfContent> {
        self.get(key)
            .ok_or_else(|| RustySatError::not_found(format!("NetCDF key '{key}'")))
    }

    pub fn attr(&self, key: &str) -> Option<&MetadataValue> {
        self.metadata.get_attr(key)
    }

    pub fn variable_shape(&self, variable_path: &str) -> Result<&[usize]> {
        self.get_or_err(&format!("{variable_path}/shape"))?
            .as_shape()
            .ok_or_else(|| {
                RustySatError::invalid_input(format!(
                    "'{variable_path}/shape' is not a shape entry"
                ))
            })
    }

    pub fn variable_dimensions(&self, variable_path: &str) -> Result<&[String]> {
        self.get_or_err(&format!("{variable_path}/dimensions"))?
            .as_dimensions()
            .ok_or_else(|| {
                RustySatError::invalid_input(format!(
                    "'{variable_path}/dimensions' is not a dimensions entry"
                ))
            })
    }

    pub fn variable_dtype(&self, variable_path: &str) -> Result<&str> {
        self.get_or_err(&format!("{variable_path}/dtype"))?
            .as_dtype()
            .ok_or_else(|| {
                RustySatError::invalid_input(format!(
                    "'{variable_path}/dtype' is not a dtype entry"
                ))
            })
    }

    pub fn load_variable_array(
        &self,
        variable_path: &str,
        source: &impl NetCdfDataSource,
    ) -> Result<AnyDataArray> {
        let array = source.read_array(&self.filename, variable_path, self.auto_mask_and_scale)?;
        self.validate_loaded_array(variable_path, &array)?;
        Ok(array)
    }

    fn validate_loaded_array(&self, variable_path: &str, array: &AnyDataArray) -> Result<()> {
        let expected_shape = self.variable_shape(variable_path)?;
        if array.shape() != expected_shape {
            return Err(RustySatError::invalid_input(format!(
                "NetCDF variable '{variable_path}' data shape {:?} does not match metadata shape {:?}",
                array.shape(),
                expected_shape
            )));
        }
        let expected_dims = self.variable_dimensions(variable_path)?;
        if array.dims() != expected_dims {
            return Err(RustySatError::invalid_input(format!(
                "NetCDF variable '{variable_path}' data dimensions {:?} do not match metadata dimensions {:?}",
                array.dims(),
                expected_dims
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FciL1cNetCdfHandler {
    file_handler: NetCdfFileHandler,
}

impl FciL1cNetCdfHandler {
    pub fn new(file_handler: NetCdfFileHandler) -> Self {
        Self { file_handler }
    }

    pub fn file_handler(&self) -> &NetCdfFileHandler {
        &self.file_handler
    }

    pub fn channel_measured_group_path(channel: &str) -> Result<String> {
        validate_fci_channel_name(channel)?;
        Ok(format!("data/{channel}/measured"))
    }

    pub fn effective_radiance_path(channel: &str) -> Result<String> {
        Ok(format!(
            "{}/effective_radiance",
            Self::channel_measured_group_path(channel)?
        ))
    }

    pub fn load_counts_dataset(
        &self,
        channel: &str,
        source: &impl NetCdfDataSource,
    ) -> Result<Dataset> {
        let variable_path = Self::effective_radiance_path(channel)?;
        let raw_array = self
            .file_handler
            .load_variable_array(&variable_path, source)?;
        let array = mask_fci_counts_array(&raw_array, |key| {
            self.file_handler
                .attr(&format!("{variable_path}/attr/{key}"))
        })?;
        let id = DataId::new(channel)?.with_qualifier("calibration", "counts")?;
        let mut dataset = Dataset::new(id).with_array(array);
        dataset.insert_metadata("reader", "fci_l1c_nc")?;
        dataset.insert_metadata("file", self.file_handler.filename())?;
        dataset.insert_metadata("variable", variable_path.clone())?;
        dataset.insert_metadata("calibration", "counts")?;
        for key in ["units", "standard_name", "ancillary_variables"] {
            if let Some(value) = self
                .file_handler
                .attr(&format!("{variable_path}/attr/{key}"))
            {
                dataset.insert_attr(key, value.clone())?;
            }
        }
        Ok(dataset)
    }
}

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

impl Default for NetCdfGroup {
    fn default() -> Self {
        Self::root()
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

impl NetCdfContent {
    pub fn as_dtype(&self) -> Option<&str> {
        match self {
            Self::DType(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_shape(&self) -> Option<&[usize]> {
        match self {
            Self::Shape(shape) => Some(shape),
            _ => None,
        }
    }

    pub fn as_dimensions(&self) -> Option<&[String]> {
        match self {
            Self::Dimensions(dims) => Some(dims),
            _ => None,
        }
    }

    pub fn as_dimension_length(&self) -> Option<usize> {
        match self {
            Self::DimensionLength(len) => Some(*len),
            _ => None,
        }
    }

    pub fn as_attribute(&self) -> Option<&MetadataValue> {
        match self {
            Self::Attribute(value) => Some(value),
            _ => None,
        }
    }

    pub fn as_attributes(&self) -> Option<&BTreeMap<String, MetadataValue>> {
        match self {
            Self::Attributes(attrs) => Some(attrs),
            _ => None,
        }
    }
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
                join_path(path, &format!("dimension/{name}")),
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

fn validate_filename(filename: impl Into<String>) -> Result<String> {
    let filename = filename.into();
    if filename.trim().is_empty() {
        return Err(RustySatError::invalid_input(
            "NetCDF filename cannot be empty",
        ));
    }
    Ok(filename)
}

fn validate_netcdf_path(kind: &str, path: &str) -> Result<()> {
    if path.trim().is_empty() {
        return Err(RustySatError::invalid_input(format!(
            "{kind} path cannot be empty"
        )));
    }
    if path.split('/').any(str::is_empty) && !path.starts_with("/attr/") {
        return Err(RustySatError::invalid_input(format!(
            "{kind} path contains an empty component"
        )));
    }
    Ok(())
}

fn validate_fci_channel_name(channel: &str) -> Result<()> {
    if channel.trim().is_empty() {
        return Err(RustySatError::invalid_input(
            "FCI channel name cannot be empty",
        ));
    }
    if channel.contains('/') {
        return Err(RustySatError::invalid_input(
            "FCI channel name cannot contain '/'",
        ));
    }
    if !["vis_", "nir_", "ir_", "wv_"]
        .iter()
        .any(|prefix| channel.starts_with(prefix))
    {
        return Err(RustySatError::invalid_input(format!(
            "FCI channel '{channel}' is not a measured channel"
        )));
    }
    Ok(())
}

fn mask_fci_counts_array<'a>(
    array: &AnyDataArray,
    attr: impl Fn(&str) -> Option<&'a MetadataValue>,
) -> Result<AnyDataArray> {
    let (valid_min, valid_max) = valid_range(attr("valid_range"))?;
    let fill_value = attr("_FillValue").and_then(metadata_as_f64);
    let values = array.values_as_f64();
    let mut mask = array
        .mask()
        .cloned()
        .unwrap_or_else(|| ValidityMask::all_valid(array.len()));
    for (idx, value) in values.iter().copied().enumerate() {
        if value < valid_min
            || value > valid_max
            || fill_value.is_some_and(|fill| nearly_equal(value, fill))
        {
            mask.set_masked(idx, true);
        }
    }
    clone_array_with_mask(array, mask)
}

fn clone_array_with_mask(array: &AnyDataArray, mask: ValidityMask) -> Result<AnyDataArray> {
    match array {
        AnyDataArray::F32(array) => Ok(array.clone().with_mask(mask)?.into()),
        AnyDataArray::F64(array) => Ok(array.clone().with_mask(mask)?.into()),
        AnyDataArray::U8(array) => Ok(array.clone().with_mask(mask)?.into()),
        AnyDataArray::U16(array) => Ok(array.clone().with_mask(mask)?.into()),
        AnyDataArray::I16(array) => Ok(array.clone().with_mask(mask)?.into()),
    }
}

fn valid_range(value: Option<&MetadataValue>) -> Result<(f64, f64)> {
    let Some(MetadataValue::List(values)) = value else {
        return Ok((f64::NEG_INFINITY, f64::INFINITY));
    };
    if values.len() != 2 {
        return Err(RustySatError::invalid_input(
            "NetCDF valid_range must contain exactly two values",
        ));
    }
    let min = metadata_as_f64(&values[0]).ok_or_else(|| {
        RustySatError::invalid_input("NetCDF valid_range minimum must be numeric")
    })?;
    let max = metadata_as_f64(&values[1]).ok_or_else(|| {
        RustySatError::invalid_input("NetCDF valid_range maximum must be numeric")
    })?;
    if min > max {
        return Err(RustySatError::invalid_input(
            "NetCDF valid_range minimum cannot exceed maximum",
        ));
    }
    Ok((min, max))
}

fn metadata_as_f64(value: &MetadataValue) -> Option<f64> {
    match value {
        MetadataValue::Integer(value) => Some(*value as f64),
        MetadataValue::Float(value) => Some(value.get()),
        _ => None,
    }
}

fn nearly_equal(left: f64, right: f64) -> bool {
    (left - right).abs() <= f64::EPSILON
}

#[cfg(test)]
mod tests {
    use super::*;
    use rusty_sat_core::DataArray;

    fn fci_like_root() -> NetCdfGroup {
        let effective_radiance =
            NetCdfVariable::new("effective_radiance", "u16", ["y", "x"], [2, 3])
                .unwrap()
                .with_attr("units", "mW m-2 sr-1 (cm-1)-1")
                .unwrap()
                .with_attr("scale_factor", MetadataValue::float(0.01).unwrap())
                .unwrap()
                .with_attr(
                    "valid_range",
                    MetadataValue::List(vec![
                        MetadataValue::Integer(0),
                        MetadataValue::Integer(4095),
                    ]),
                )
                .unwrap()
                .with_attr("_FillValue", MetadataValue::Integer(65535))
                .unwrap()
                .with_attr("ancillary_variables", "pixel_quality")
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

    // — accessor helpers —

    #[test]
    fn netcdf_content_accessors_return_none_for_wrong_variant() {
        let group = NetCdfContent::Group;
        assert!(group.as_dtype().is_none());
        assert!(group.as_shape().is_none());
        assert!(group.as_dimension_length().is_none());
        assert!(group.as_attribute().is_none());
        assert!(group.as_attributes().is_none());

        let dtype = NetCdfContent::DType("f32".to_string());
        assert_eq!(dtype.as_dtype(), Some("f32"));
        assert!(dtype.as_shape().is_none());

        let shape = NetCdfContent::Shape(vec![2, 3]);
        assert_eq!(shape.as_shape(), Some(&[2, 3][..]));

        let dims = NetCdfContent::Dimensions(vec!["y".to_string(), "x".to_string()]);
        assert_eq!(
            dims.as_dimensions(),
            Some(&["y".to_string(), "x".to_string()][..])
        );

        let dim_len = NetCdfContent::DimensionLength(5);
        assert_eq!(dim_len.as_dimension_length(), Some(5));

        let attr = NetCdfContent::Attribute(MetadataValue::String("K".to_string()));
        assert_eq!(
            attr.as_attribute(),
            Some(&MetadataValue::String("K".to_string()))
        );

        let attrs = NetCdfContent::Attributes(BTreeMap::from([(
            "units".to_string(),
            MetadataValue::String("K".to_string()),
        )]));
        let map = attrs.as_attributes().unwrap();
        assert_eq!(
            map.get("units"),
            Some(&MetadataValue::String("K".to_string()))
        );
    }

    // — get_attr returns None —

    #[test]
    fn get_attr_returns_none_for_non_attribute_key() {
        let metadata = NetCdfMetadata::collect(&fci_like_root()).unwrap();

        assert!(metadata
            .get_attr("data/vis_04/measured/effective_radiance/shape")
            .is_none());
        assert!(metadata
            .get_attr("data/vis_04/measured/effective_radiance/dtype")
            .is_none());
        assert!(metadata.get_attr("nonexistent").is_none());
    }

    // — root-level dimensions —

    #[test]
    fn collects_root_level_dimensions_without_leading_slash() {
        let root = NetCdfGroup::root()
            .with_dimension("y", 100)
            .unwrap()
            .with_dimension("x", 200)
            .unwrap();

        let metadata = NetCdfMetadata::collect(&root).unwrap();

        assert_eq!(
            metadata.get("dimension/y"),
            Some(&NetCdfContent::DimensionLength(100))
        );
        assert_eq!(
            metadata.get("dimension/x"),
            Some(&NetCdfContent::DimensionLength(200))
        );
        // Sanity: root-level dimensions must not have a leading /.
        assert!(!metadata.contains("/dimension/y"));
    }

    // — collect_required group attributes —

    #[test]
    fn collect_required_picks_up_group_attributes() {
        let inner = NetCdfGroup::new("inner")
            .unwrap()
            .with_attr("description", "test group")
            .unwrap();
        let root = NetCdfGroup::root().with_group(inner).unwrap();

        let metadata =
            NetCdfMetadata::collect_required(&root, ["inner/attr/description"], &BTreeMap::new())
                .unwrap();

        assert_eq!(
            metadata.get_attr("inner/attr/description"),
            Some(&MetadataValue::String("test group".to_string()))
        );
    }

    // — expand_required_variable_names —

    #[test]
    fn collect_required_expands_multiple_values_and_errors_on_missing() {
        let mut replacements = BTreeMap::new();
        replacements.insert(
            "channel".to_string(),
            vec!["vis_04".to_string(), "ir_38".to_string()],
        );

        let err = NetCdfMetadata::collect_required(
            &fci_like_root(),
            ["data/{channel}/measured/effective_radiance"],
            &replacements,
        )
        .unwrap_err()
        .to_string();

        assert!(
            err.contains("ir_38"),
            "expected error mentioning the missing channel, got: {err}"
        );
    }

    #[test]
    fn expand_required_rejects_empty_replacement_value() {
        let mut replacements = BTreeMap::new();
        replacements.insert("channel".to_string(), vec!["".to_string()]);

        let err =
            expand_required_variable_names(["data/{channel}/measured/radiance"], &replacements)
                .unwrap_err()
                .to_string();

        assert!(err.contains("cannot contain an empty value"));
        assert!(err.contains("channel"));
    }

    #[test]
    fn file_handler_collects_full_metadata_like_satpy_when_no_required_list_is_set() {
        let handler = NetCdfFileHandler::from_root(
            "fci.nc",
            BTreeMap::from([("repeat_cycle".to_string(), MetadataValue::Integer(1))]),
            NetCdfFileTypeInfo::new(),
            &fci_like_root(),
        )
        .unwrap();

        assert_eq!(handler.filename(), "fci.nc");
        assert_eq!(
            handler.filename_info().get("repeat_cycle"),
            Some(&MetadataValue::Integer(1))
        );
        assert!(handler.contains("data/vis_04/measured/effective_radiance"));
        assert_eq!(
            handler
                .variable_shape("data/vis_04/measured/effective_radiance")
                .unwrap(),
            &[2, 3]
        );
        assert_eq!(
            handler
                .variable_dimensions("data/vis_04/measured/effective_radiance")
                .unwrap(),
            &["y".to_string(), "x".to_string()]
        );
        assert_eq!(
            handler
                .variable_dtype("data/vis_04/measured/effective_radiance")
                .unwrap(),
            "u16"
        );
        assert_eq!(
            handler.attr("/attr/platform"),
            Some(&MetadataValue::String("MTG-I1".to_string()))
        );
    }

    #[test]
    fn file_handler_collects_required_variables_with_replacements() {
        let filetype_info = NetCdfFileTypeInfo::new()
            .with_required_variables([
                "data/{channel}/measured/effective_radiance",
                "/attr/platform",
            ])
            .unwrap()
            .with_variable_name_replacement("channel", ["vis_04"])
            .unwrap();

        let handler = NetCdfFileHandler::from_root(
            "fci.nc",
            BTreeMap::new(),
            filetype_info,
            &fci_like_root(),
        )
        .unwrap();

        assert!(handler.contains("data/vis_04/measured/effective_radiance"));
        assert!(handler.contains("/attr/platform"));
        assert!(!handler.contains("data"));
    }

    #[derive(Debug)]
    struct RecordingSource {
        root: NetCdfGroup,
        expected_auto_mask_and_scale: bool,
    }

    impl NetCdfMetadataSource for RecordingSource {
        fn read_metadata_tree(
            &self,
            filename: &str,
            auto_mask_and_scale: bool,
        ) -> Result<NetCdfGroup> {
            if filename != "fci.nc" {
                return Err(RustySatError::invalid_input("unexpected filename"));
            }
            if auto_mask_and_scale != self.expected_auto_mask_and_scale {
                return Err(RustySatError::invalid_input(
                    "unexpected auto mask/scale setting",
                ));
            }
            Ok(self.root.clone())
        }
    }

    #[test]
    fn file_handler_passes_auto_mask_and_scale_to_metadata_source() {
        let source = RecordingSource {
            root: fci_like_root(),
            expected_auto_mask_and_scale: true,
        };

        let handler = NetCdfFileHandler::from_source_with_options(
            "fci.nc",
            BTreeMap::new(),
            NetCdfFileTypeInfo::new(),
            &source,
            true,
        )
        .unwrap();

        assert!(handler.auto_mask_and_scale());
        assert!(handler.contains("data/vis_04/measured/effective_radiance"));
    }

    #[test]
    fn file_type_info_validates_required_variables_and_replacements() {
        let err = NetCdfFileTypeInfo::new()
            .with_required_variables(["data//radiance"])
            .unwrap_err()
            .to_string();
        assert!(err.contains("empty component"));

        let err = NetCdfFileTypeInfo::new()
            .with_variable_name_replacement("channel", Vec::<String>::new())
            .unwrap_err()
            .to_string();
        assert!(err.contains("at least one value"));
    }

    #[test]
    fn file_handler_loads_variable_array_from_data_source() {
        let variable_path = "data/vis_04/measured/effective_radiance";
        let source = InMemoryNetCdfSource::new(fci_like_root())
            .with_array(
                variable_path,
                DataArray::<u16>::from_vec_named(vec![2, 3], ["y", "x"], vec![1, 2, 3, 4, 5, 6])
                    .unwrap(),
            )
            .unwrap();
        let handler = NetCdfFileHandler::from_source(
            "fci.nc",
            BTreeMap::new(),
            NetCdfFileTypeInfo::new(),
            &source,
        )
        .unwrap();

        let array = handler.load_variable_array(variable_path, &source).unwrap();

        assert_eq!(array.shape(), &[2, 3]);
        assert_eq!(array.dims(), &["y".to_string(), "x".to_string()]);
        assert_eq!(array.values_as_f64(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn file_handler_rejects_loaded_array_shape_mismatch() {
        let variable_path = "data/vis_04/measured/effective_radiance";
        let source = InMemoryNetCdfSource::new(fci_like_root())
            .with_array(
                variable_path,
                DataArray::<u16>::from_vec_named(vec![3, 2], ["y", "x"], vec![1, 2, 3, 4, 5, 6])
                    .unwrap(),
            )
            .unwrap();
        let handler = NetCdfFileHandler::from_source(
            "fci.nc",
            BTreeMap::new(),
            NetCdfFileTypeInfo::new(),
            &source,
        )
        .unwrap();

        let err = handler
            .load_variable_array(variable_path, &source)
            .unwrap_err()
            .to_string();

        assert!(err.contains("does not match metadata shape"));
    }

    #[test]
    fn fci_handler_loads_counts_dataset_and_masks_invalid_values() {
        let variable_path = "data/vis_04/measured/effective_radiance";
        let source = InMemoryNetCdfSource::new(fci_like_root())
            .with_array(
                variable_path,
                DataArray::<u16>::from_vec_named(
                    vec![2, 3],
                    ["y", "x"],
                    vec![10, 4095, 4096, 65535, 12, 13],
                )
                .unwrap(),
            )
            .unwrap();
        let handler = NetCdfFileHandler::from_source(
            "fci.nc",
            BTreeMap::new(),
            NetCdfFileTypeInfo::new(),
            &source,
        )
        .unwrap();
        let fci = FciL1cNetCdfHandler::new(handler);

        let dataset = fci.load_counts_dataset("vis_04", &source).unwrap();
        let array = dataset.array().unwrap();
        let mask = array.mask().unwrap();

        assert_eq!(dataset.id().name(), "vis_04");
        assert_eq!(
            dataset.metadata().get("calibration"),
            Some(&"counts".to_string())
        );
        assert_eq!(
            dataset.attr("ancillary_variables"),
            Some(&MetadataValue::String("pixel_quality".to_string()))
        );
        assert_eq!(array.shape(), &[2, 3]);
        assert_eq!(array.dtype().name(), "u16");
        assert_eq!(mask.is_masked(0), Some(false));
        assert_eq!(mask.is_masked(2), Some(true));
        assert_eq!(mask.is_masked(3), Some(true));
        assert_eq!(mask.masked_count(), 2);
    }

    #[test]
    fn fci_handler_rejects_non_channel_dataset_names() {
        let err = FciL1cNetCdfHandler::effective_radiance_path("quality")
            .unwrap_err()
            .to_string();

        assert!(err.contains("not a measured channel"));
    }
}
