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
    AnyDataArray, DataArray, DataId, Dataset, MetadataValue, Result, RustySatError, ValidityMask,
};
use serde_norway::{Mapping, Value};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

const MAX_NETCDF_FIXTURE_YAML_BYTES: usize = 32 * 1024 * 1024;
const MAX_NETCDF_FIXTURE_YAML_DEPTH: usize = 96;

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

#[derive(Debug, Clone, PartialEq, Default)]
pub struct NetCdfFixtureSource {
    inner: InMemoryNetCdfSource,
}

impl NetCdfFixtureSource {
    pub fn from_yaml_str(yaml: &str) -> Result<Self> {
        if yaml.len() > MAX_NETCDF_FIXTURE_YAML_BYTES {
            return Err(RustySatError::invalid_input(format!(
                "NetCDF fixture YAML exceeds size limit of {MAX_NETCDF_FIXTURE_YAML_BYTES} bytes"
            )));
        }
        let value: Value = serde_norway::from_str(yaml).map_err(|err| {
            RustySatError::invalid_input(format!("invalid NetCDF fixture YAML: {err}"))
        })?;
        validate_fixture_yaml_depth(&value, 0)?;
        let mapping = value
            .as_mapping()
            .ok_or_else(|| RustySatError::invalid_input("NetCDF fixture root must be a mapping"))?;
        let mut arrays = BTreeMap::new();
        let root = parse_fixture_group(mapping, "", &mut arrays)?;
        Ok(Self {
            inner: InMemoryNetCdfSource { root, arrays },
        })
    }

    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let metadata = fs::metadata(path).map_err(|err| {
            RustySatError::invalid_input(format!(
                "failed to inspect NetCDF fixture '{}': {err}",
                path.display()
            ))
        })?;
        if metadata.len() as usize > MAX_NETCDF_FIXTURE_YAML_BYTES {
            return Err(RustySatError::invalid_input(format!(
                "NetCDF fixture '{}' exceeds size limit of {MAX_NETCDF_FIXTURE_YAML_BYTES} bytes",
                path.display()
            )));
        }
        let yaml = fs::read_to_string(path).map_err(|err| {
            RustySatError::invalid_input(format!(
                "failed to read NetCDF fixture '{}': {err}",
                path.display()
            ))
        })?;
        Self::from_yaml_str(&yaml)
    }

    pub fn inner(&self) -> &InMemoryNetCdfSource {
        &self.inner
    }
}

impl NetCdfMetadataSource for NetCdfFixtureSource {
    fn read_metadata_tree(&self, filename: &str, auto_mask_and_scale: bool) -> Result<NetCdfGroup> {
        self.inner.read_metadata_tree(filename, auto_mask_and_scale)
    }
}

impl NetCdfDataSource for NetCdfFixtureSource {
    fn read_array(
        &self,
        filename: &str,
        variable_path: &str,
        auto_mask_and_scale: bool,
    ) -> Result<AnyDataArray> {
        self.inner
            .read_array(filename, variable_path, auto_mask_and_scale)
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
        let array = mask_fci_counts_array(raw_array, |key| {
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
    array: AnyDataArray,
    attr: impl Fn(&str) -> Option<&'a MetadataValue>,
) -> Result<AnyDataArray> {
    let (valid_min, valid_max) = valid_range(attr("valid_range"))?;
    let fill_value = attr("_FillValue").and_then(metadata_as_f64);
    let existing_mask = array.mask().cloned();
    let mut mask = existing_mask.unwrap_or_else(|| ValidityMask::all_valid(array.len()));
    let min = valid_min;
    let max = valid_max;
    let fill = fill_value;

    match &array {
        AnyDataArray::U16(arr) => {
            for (idx, &raw) in arr.values().iter().enumerate() {
                let v = raw as f64;
                if v < min || v > max || fill.is_some_and(|f| nearly_equal(v, f)) {
                    mask.set_masked(idx, true);
                }
            }
        }
        _ => {
            let values = array.values_as_f64();
            for (idx, v) in values.iter().copied().enumerate() {
                if v < min || v > max || fill.is_some_and(|f| nearly_equal(v, f)) {
                    mask.set_masked(idx, true);
                }
            }
        }
    }

    into_array_with_mask(array, mask)
}

fn into_array_with_mask(array: AnyDataArray, mask: ValidityMask) -> Result<AnyDataArray> {
    match array {
        AnyDataArray::F32(array) => Ok(array.with_mask(mask)?.into()),
        AnyDataArray::F64(array) => Ok(array.with_mask(mask)?.into()),
        AnyDataArray::U8(array) => Ok(array.with_mask(mask)?.into()),
        AnyDataArray::U16(array) => Ok(array.with_mask(mask)?.into()),
        AnyDataArray::I16(array) => Ok(array.with_mask(mask)?.into()),
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

fn parse_fixture_group(
    mapping: &Mapping,
    path: &str,
    arrays: &mut BTreeMap<String, AnyDataArray>,
) -> Result<NetCdfGroup> {
    let mut group = if path.is_empty() {
        NetCdfGroup::root()
    } else {
        let (_, name) = split_parent_path(path);
        NetCdfGroup::new(name)?
    };
    if let Some(attrs) = optional_mapping(mapping, "attrs")? {
        for (key, value) in attrs {
            group.insert_attr(
                yaml_key_to_string(key)?,
                crate::yaml_reader::yaml_to_metadata_value(value)?,
            )?;
        }
    }
    if let Some(dimensions) = optional_mapping(mapping, "dimensions")? {
        for (key, value) in dimensions {
            group.insert_dimension(
                yaml_key_to_string(key)?,
                yaml_value_to_usize(value, "dimension length")?,
            )?;
        }
    }
    if let Some(variables) = optional_mapping(mapping, "variables")? {
        for (key, value) in variables {
            let variable_name = yaml_key_to_string(key)?;
            let variable_mapping = value.as_mapping().ok_or_else(|| {
                RustySatError::invalid_input(format!(
                    "NetCDF fixture variable '{variable_name}' must be a mapping"
                ))
            })?;
            let variable_path = join_path(path, &variable_name);
            let (variable, array) = parse_fixture_variable(&variable_name, variable_mapping)?;
            group.insert_variable(variable)?;
            if let Some(array) = array {
                arrays.insert(variable_path, array);
            }
        }
    }
    if let Some(groups) = optional_mapping(mapping, "groups")? {
        for (key, value) in groups {
            let group_name = yaml_key_to_string(key)?;
            let group_mapping = value.as_mapping().ok_or_else(|| {
                RustySatError::invalid_input(format!(
                    "NetCDF fixture group '{group_name}' must be a mapping"
                ))
            })?;
            let group_path = join_path(path, &group_name);
            group.insert_group(parse_fixture_group(group_mapping, &group_path, arrays)?)?;
        }
    }
    Ok(group)
}

fn parse_fixture_variable(
    name: &str,
    mapping: &Mapping,
) -> Result<(NetCdfVariable, Option<AnyDataArray>)> {
    let dtype = required_string(mapping, "dtype", "NetCDF fixture variable")?;
    let dimensions = required_string_list(mapping, "dimensions", "NetCDF fixture variable")?;
    let shape = required_usize_list(mapping, "shape", "NetCDF fixture variable")?;
    let mut variable = NetCdfVariable::new(name, dtype.clone(), dimensions.clone(), shape.clone())?;
    if let Some(attrs) = optional_mapping(mapping, "attrs")? {
        for (key, value) in attrs {
            variable.insert_attr(
                yaml_key_to_string(key)?,
                crate::yaml_reader::yaml_to_metadata_value(value)?,
            )?;
        }
    }
    let array = optional_value(mapping, "values")
        .map(|value| fixture_values_to_array(&dtype, &shape, &dimensions, value))
        .transpose()?;
    Ok((variable, array))
}

fn fixture_values_to_array(
    dtype: &str,
    shape: &[usize],
    dimensions: &[String],
    value: &Value,
) -> Result<AnyDataArray> {
    let values = value.as_sequence().ok_or_else(|| {
        RustySatError::invalid_input("NetCDF fixture variable values must be a sequence")
    })?;
    match dtype {
        "f32" => Ok(DataArray::<f32>::from_vec_named(
            shape.to_vec(),
            dimensions.iter().cloned(),
            values
                .iter()
                .map(|value| yaml_value_to_f64(value, "f32 value").map(|value| value as f32))
                .collect::<Result<Vec<_>>>()?,
        )?
        .into()),
        "f64" => Ok(DataArray::<f64>::from_vec_named(
            shape.to_vec(),
            dimensions.iter().cloned(),
            values
                .iter()
                .map(|value| yaml_value_to_f64(value, "f64 value"))
                .collect::<Result<Vec<_>>>()?,
        )?
        .into()),
        "u8" => Ok(DataArray::<u8>::from_vec_named(
            shape.to_vec(),
            dimensions.iter().cloned(),
            values
                .iter()
                .map(|value| yaml_value_to_u8(value, "u8 value"))
                .collect::<Result<Vec<_>>>()?,
        )?
        .into()),
        "u16" => Ok(DataArray::<u16>::from_vec_named(
            shape.to_vec(),
            dimensions.iter().cloned(),
            values
                .iter()
                .map(|value| yaml_value_to_u16(value, "u16 value"))
                .collect::<Result<Vec<_>>>()?,
        )?
        .into()),
        "i16" => Ok(DataArray::<i16>::from_vec_named(
            shape.to_vec(),
            dimensions.iter().cloned(),
            values
                .iter()
                .map(|value| yaml_value_to_i16(value, "i16 value"))
                .collect::<Result<Vec<_>>>()?,
        )?
        .into()),
        _ => Err(RustySatError::unsupported(format!(
            "NetCDF fixture dtype '{dtype}'"
        ))),
    }
}

fn validate_fixture_yaml_depth(value: &Value, depth: usize) -> Result<()> {
    if depth > MAX_NETCDF_FIXTURE_YAML_DEPTH {
        return Err(RustySatError::invalid_input(format!(
            "NetCDF fixture YAML exceeds nesting depth limit of {MAX_NETCDF_FIXTURE_YAML_DEPTH}"
        )));
    }
    if let Some(sequence) = value.as_sequence() {
        for child in sequence {
            validate_fixture_yaml_depth(child, depth + 1)?;
        }
    }
    if let Some(mapping) = value.as_mapping() {
        for (key, child) in mapping {
            validate_fixture_yaml_depth(key, depth + 1)?;
            validate_fixture_yaml_depth(child, depth + 1)?;
        }
    }
    if let Value::Tagged(tagged) = value {
        validate_fixture_yaml_depth(&tagged.value, depth + 1)?;
    }
    Ok(())
}

fn optional_value<'a>(mapping: &'a Mapping, key: &str) -> Option<&'a Value> {
    mapping.get(Value::String(key.to_string()))
}

fn optional_mapping<'a>(mapping: &'a Mapping, key: &str) -> Result<Option<&'a Mapping>> {
    optional_value(mapping, key)
        .map(|value| {
            value.as_mapping().ok_or_else(|| {
                RustySatError::invalid_input(format!(
                    "NetCDF fixture section '{key}' must be a mapping"
                ))
            })
        })
        .transpose()
}

fn required_string(mapping: &Mapping, key: &str, context: &str) -> Result<String> {
    optional_value(mapping, key)
        .ok_or_else(|| RustySatError::invalid_input(format!("{context} requires '{key}'")))?
        .as_str()
        .map(ToString::to_string)
        .ok_or_else(|| RustySatError::invalid_input(format!("{context} '{key}' must be a string")))
}

fn required_string_list(mapping: &Mapping, key: &str, context: &str) -> Result<Vec<String>> {
    let value = optional_value(mapping, key)
        .ok_or_else(|| RustySatError::invalid_input(format!("{context} requires '{key}'")))?;
    value
        .as_sequence()
        .ok_or_else(|| {
            RustySatError::invalid_input(format!("{context} '{key}' must be a sequence"))
        })?
        .iter()
        .map(|value| {
            value.as_str().map(ToString::to_string).ok_or_else(|| {
                RustySatError::invalid_input(format!("{context} '{key}' entries must be strings"))
            })
        })
        .collect()
}

fn required_usize_list(mapping: &Mapping, key: &str, context: &str) -> Result<Vec<usize>> {
    let value = optional_value(mapping, key)
        .ok_or_else(|| RustySatError::invalid_input(format!("{context} requires '{key}'")))?;
    value
        .as_sequence()
        .ok_or_else(|| {
            RustySatError::invalid_input(format!("{context} '{key}' must be a sequence"))
        })?
        .iter()
        .map(|value| yaml_value_to_usize(value, key))
        .collect()
}

fn yaml_key_to_string(value: &Value) -> Result<String> {
    value
        .as_str()
        .map(ToString::to_string)
        .ok_or_else(|| RustySatError::invalid_input("NetCDF fixture mapping keys must be strings"))
}

fn yaml_value_to_usize(value: &Value, context: &str) -> Result<usize> {
    let value = value.as_i64().ok_or_else(|| {
        RustySatError::invalid_input(format!("NetCDF fixture {context} must be an integer"))
    })?;
    usize::try_from(value).map_err(|_| {
        RustySatError::invalid_input(format!(
            "NetCDF fixture {context} must be a non-negative integer"
        ))
    })
}

fn yaml_value_to_u8(value: &Value, context: &str) -> Result<u8> {
    u8::try_from(yaml_value_to_usize(value, context)?).map_err(|_| {
        RustySatError::invalid_input(format!("NetCDF fixture {context} does not fit in u8"))
    })
}

fn yaml_value_to_u16(value: &Value, context: &str) -> Result<u16> {
    u16::try_from(yaml_value_to_usize(value, context)?).map_err(|_| {
        RustySatError::invalid_input(format!("NetCDF fixture {context} does not fit in u16"))
    })
}

fn yaml_value_to_i16(value: &Value, context: &str) -> Result<i16> {
    let value = value.as_i64().ok_or_else(|| {
        RustySatError::invalid_input(format!("NetCDF fixture {context} must be an integer"))
    })?;
    i16::try_from(value).map_err(|_| {
        RustySatError::invalid_input(format!("NetCDF fixture {context} does not fit in i16"))
    })
}

fn yaml_value_to_f64(value: &Value, context: &str) -> Result<f64> {
    value.as_f64().ok_or_else(|| {
        RustySatError::invalid_input(format!("NetCDF fixture {context} must be numeric"))
    })
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

    #[test]
    fn load_variable_array_rejects_dimension_mismatch() {
        let variable_path = "data/vis_04/measured/effective_radiance";
        let source = InMemoryNetCdfSource::new(fci_like_root())
            .with_array(
                variable_path,
                DataArray::<u16>::from_vec_named(vec![2, 3], ["z", "x"], vec![1, 2, 3, 4, 5, 6])
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

        assert!(err.contains("dimensions"));
    }

    #[test]
    fn valid_range_returns_default_when_not_a_list() {
        let result = valid_range(Some(&MetadataValue::Integer(5))).unwrap();

        assert_eq!(result, (f64::NEG_INFINITY, f64::INFINITY));
    }

    #[test]
    fn valid_range_rejects_min_greater_than_max() {
        let result = valid_range(Some(&MetadataValue::List(vec![
            MetadataValue::Integer(10),
            MetadataValue::Integer(0),
        ])));

        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("minimum cannot exceed maximum"));
    }

    #[test]
    fn fci_counts_masking_respects_absent_valid_range() {
        // root without valid_range attr on the variable
        let effective_radiance =
            NetCdfVariable::new("effective_radiance", "u16", ["y", "x"], [1, 2])
                .unwrap()
                .with_attr("units", "counts")
                .unwrap();
        let measured = NetCdfGroup::new("measured")
            .unwrap()
            .with_dimension("y", 1)
            .unwrap()
            .with_dimension("x", 2)
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
        let root = NetCdfGroup::root().with_group(data).unwrap();
        let variable_path = "data/vis_04/measured/effective_radiance";
        let source = InMemoryNetCdfSource::new(root)
            .with_array(
                variable_path,
                DataArray::<u16>::from_vec_named(vec![1, 2], ["y", "x"], vec![100, 200]).unwrap(),
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

        // without valid_range, all values survive
        assert_eq!(array.mask().map(|m| m.masked_count()), Some(0));
        assert_eq!(array.values_as_f64(), vec![100.0, 200.0]);
    }

    #[test]
    fn fci_counts_masking_ignores_non_numeric_fill_value() {
        let effective_radiance =
            NetCdfVariable::new("effective_radiance", "u16", ["y", "x"], [1, 2])
                .unwrap()
                .with_attr("_FillValue", MetadataValue::String("missing".to_string()))
                .unwrap();
        let measured = NetCdfGroup::new("measured")
            .unwrap()
            .with_dimension("y", 1)
            .unwrap()
            .with_dimension("x", 2)
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
        let root = NetCdfGroup::root().with_group(data).unwrap();
        let variable_path = "data/vis_04/measured/effective_radiance";
        let source = InMemoryNetCdfSource::new(root)
            .with_array(
                variable_path,
                DataArray::<u16>::from_vec_named(vec![1, 2], ["y", "x"], vec![10, 20]).unwrap(),
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

        assert_eq!(array.mask().map(|m| m.masked_count()), Some(0));
    }

    #[test]
    fn fixture_source_loads_fci_like_counts_dataset_from_yaml() {
        let fixture = r#"
attrs:
  platform: MTG-I1
groups:
  data:
    groups:
      vis_04:
        groups:
          measured:
            dimensions:
              y: 2
              x: 3
            variables:
              effective_radiance:
                dtype: u16
                dimensions: [y, x]
                shape: [2, 3]
                attrs:
                  units: mW m-2 sr-1 (cm-1)-1
                  ancillary_variables: pixel_quality
                  _FillValue: 65535
                  valid_range: [0, 4095]
                values: [10, 4096, 12, 13, 65535, 15]
"#;
        let source = NetCdfFixtureSource::from_yaml_str(fixture).unwrap();
        let handler = NetCdfFileHandler::from_source(
            "fixture.yaml",
            BTreeMap::new(),
            NetCdfFileTypeInfo::new(),
            &source,
        )
        .unwrap();
        let fci = FciL1cNetCdfHandler::new(handler);

        let dataset = fci.load_counts_dataset("vis_04", &source).unwrap();
        let array = dataset.array().unwrap();
        let mask = array.mask().unwrap();

        assert_eq!(array.shape(), &[2, 3]);
        assert_eq!(
            array.values_as_f64(),
            vec![10.0, 4096.0, 12.0, 13.0, 65535.0, 15.0]
        );
        assert_eq!(mask.is_masked(1), Some(true));
        assert_eq!(mask.is_masked(4), Some(true));
        assert_eq!(mask.masked_count(), 2);
    }

    #[test]
    fn fixture_source_loads_from_path() {
        let fixture = r#"
groups:
  data:
    groups:
      vis_04:
        groups:
          measured:
            dimensions: {y: 1, x: 2}
            variables:
              effective_radiance:
                dtype: u16
                dimensions: [y, x]
                shape: [1, 2]
                attrs:
                  valid_range: [0, 4095]
                  ancillary_variables: pixel_quality
                values: [1, 2]
"#;
        let path = std::env::temp_dir().join(format!(
            "rusty_sat_netcdf_fixture_{}_loads_from_path.yaml",
            std::process::id()
        ));
        fs::write(&path, fixture).unwrap();

        let source = NetCdfFixtureSource::from_path(&path).unwrap();
        let handler = NetCdfFileHandler::from_source(
            path.to_string_lossy(),
            BTreeMap::new(),
            NetCdfFileTypeInfo::new(),
            &source,
        )
        .unwrap();

        assert_eq!(
            handler
                .variable_shape("data/vis_04/measured/effective_radiance")
                .unwrap(),
            &[1, 2]
        );

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn fixture_source_rejects_value_count_mismatch() {
        let fixture = r#"
variables:
  broken:
    dtype: u16
    dimensions: [x]
    shape: [2]
    values: [1]
"#;

        let err = NetCdfFixtureSource::from_yaml_str(fixture)
            .unwrap_err()
            .to_string();

        assert!(err.contains("shape"));
    }

    #[test]
    fn fixture_source_rejects_excessive_size() {
        let fixture = "x".repeat(MAX_NETCDF_FIXTURE_YAML_BYTES + 1);

        let err = NetCdfFixtureSource::from_yaml_str(&fixture)
            .unwrap_err()
            .to_string();

        assert!(err.contains("exceeds size limit"));
    }

    #[test]
    fn fixture_source_rejects_excessive_depth() {
        let fixture = "- ".repeat(MAX_NETCDF_FIXTURE_YAML_DEPTH + 2);

        let err = NetCdfFixtureSource::from_yaml_str(&fixture)
            .unwrap_err()
            .to_string();

        assert!(err.contains("nesting depth"));
    }

    #[test]
    fn fixture_source_rejects_non_mapping_root() {
        let err = NetCdfFixtureSource::from_yaml_str("42")
            .unwrap_err()
            .to_string();

        assert!(err.contains("root must be a mapping"));
    }

    #[test]
    fn fixture_source_rejects_unsupported_dtype() {
        let fixture = r#"
variables:
  bad:
    dtype: int64
    dimensions: [x]
    shape: [1]
    values: [1]
"#;

        let err = NetCdfFixtureSource::from_yaml_str(fixture)
            .unwrap_err()
            .to_string();

        assert!(err.contains("dtype 'int64'"));
    }

    #[test]
    fn fixture_source_rejects_non_mapping_variable() {
        let fixture = r#"
variables:
  bad: "not a mapping"
"#;

        let err = NetCdfFixtureSource::from_yaml_str(fixture)
            .unwrap_err()
            .to_string();

        assert!(err.contains("must be a mapping"));
    }

    #[test]
    fn fixture_source_rejects_non_mapping_group_child() {
        let fixture = r#"
groups:
  bad: "not a mapping"
"#;

        let err = NetCdfFixtureSource::from_yaml_str(fixture)
            .unwrap_err()
            .to_string();

        assert!(err.contains("must be a mapping"));
    }

    #[test]
    fn fixture_source_from_path_reports_missing_file() {
        let path = std::env::temp_dir().join(format!(
            "rusty_sat_netcdf_nonexistent_{}.yaml",
            std::process::id()
        ));
        // Ensure the file does not exist.
        let _ = fs::remove_file(&path);

        let err = NetCdfFixtureSource::from_path(&path)
            .unwrap_err()
            .to_string();

        assert!(
            err.contains("failed to") || err.contains("No such file"),
            "unexpected error: {err}"
        );
    }
}
