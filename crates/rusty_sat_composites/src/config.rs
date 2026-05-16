//! YAML-driven compositor and enhancement registration foundations.
//!
//! Reference behavior inspected before implementation:
//! - `satpy/satpy/composites/config_loader.py`
//! - `satpy/satpy/enhancements/enhancer.py`
//! - `satpy/satpy/etc/composites/ahi.yaml`
//! - `satpy/satpy/etc/enhancements/generic.yaml`
//!
//! Satpy YAML uses Python tags like `!!python/name:...` to point at classes or
//! functions. Rusty Sat stores those tags as inert strings; this module never
//! executes or dynamically imports names from YAML.

use rusty_sat_core::{MetadataValue, Result, RustySatError};
use serde_norway::{Mapping, Value};
use std::collections::BTreeMap;

const MAX_COMPOSITE_YAML_BYTES: usize = 8 * 1024 * 1024;
const MAX_COMPOSITE_YAML_DEPTH: usize = 96;

#[derive(Debug, Clone, PartialEq)]
pub struct CompositeRegistryConfig {
    composites: BTreeMap<String, CompositeDefinition>,
    enhancements: BTreeMap<String, EnhancementDefinition>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CompositeDefinition {
    name: String,
    compositor: String,
    prerequisites: Vec<CompositeDependency>,
    optional_prerequisites: Vec<CompositeDependency>,
    attrs: BTreeMap<String, MetadataValue>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum CompositeDependency {
    Query(BTreeMap<String, MetadataValue>),
    InlineComposite(Box<CompositeDefinition>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct EnhancementDefinition {
    name: String,
    match_attrs: BTreeMap<String, MetadataValue>,
    operations: Vec<EnhancementOperation>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EnhancementOperation {
    name: String,
    method: String,
    args: Vec<MetadataValue>,
    kwargs: BTreeMap<String, MetadataValue>,
}

impl CompositeRegistryConfig {
    pub fn from_yaml_str(yaml: &str) -> Result<Self> {
        let value = parse_config_yaml_value(yaml)?;
        Self::from_yaml_value(&value)
    }

    pub fn from_yaml_value(value: &Value) -> Result<Self> {
        let mapping = value.as_mapping().ok_or_else(|| {
            RustySatError::invalid_input("composite config root must be a mapping")
        })?;
        let composites = optional_mapping(mapping, "composites")?
            .map(parse_composites)
            .transpose()?
            .unwrap_or_default();
        let enhancements = optional_mapping(mapping, "enhancements")?
            .map(parse_enhancements)
            .transpose()?
            .unwrap_or_default();
        Ok(Self {
            composites,
            enhancements,
        })
    }

    pub fn composites(&self) -> &BTreeMap<String, CompositeDefinition> {
        &self.composites
    }

    pub fn enhancements(&self) -> &BTreeMap<String, EnhancementDefinition> {
        &self.enhancements
    }
}

impl CompositeDefinition {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn compositor(&self) -> &str {
        &self.compositor
    }

    pub fn prerequisites(&self) -> &[CompositeDependency] {
        &self.prerequisites
    }

    pub fn optional_prerequisites(&self) -> &[CompositeDependency] {
        &self.optional_prerequisites
    }

    pub fn attrs(&self) -> &BTreeMap<String, MetadataValue> {
        &self.attrs
    }
}

impl EnhancementDefinition {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn match_attrs(&self) -> &BTreeMap<String, MetadataValue> {
        &self.match_attrs
    }

    pub fn operations(&self) -> &[EnhancementOperation] {
        &self.operations
    }
}

impl EnhancementOperation {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn method(&self) -> &str {
        &self.method
    }

    pub fn args(&self) -> &[MetadataValue] {
        &self.args
    }

    pub fn kwargs(&self) -> &BTreeMap<String, MetadataValue> {
        &self.kwargs
    }
}

fn parse_config_yaml_value(yaml: &str) -> Result<Value> {
    if yaml.len() > MAX_COMPOSITE_YAML_BYTES {
        return Err(RustySatError::invalid_input(format!(
            "composite config YAML exceeds size limit of {MAX_COMPOSITE_YAML_BYTES} bytes"
        )));
    }
    let yaml = sanitize_python_name_tags(yaml);
    let value = serde_norway::from_str(&yaml).map_err(|err| {
        RustySatError::invalid_input(format!("invalid composite config YAML: {err}"))
    })?;
    validate_yaml_depth(&value, 0)?;
    Ok(value)
}

fn validate_yaml_depth(value: &Value, depth: usize) -> Result<()> {
    if depth > MAX_COMPOSITE_YAML_DEPTH {
        return Err(RustySatError::invalid_input(format!(
            "composite config YAML exceeds nesting depth limit of {MAX_COMPOSITE_YAML_DEPTH}"
        )));
    }
    match value {
        Value::Sequence(values) => {
            for child in values {
                validate_yaml_depth(child, depth + 1)?;
            }
        }
        Value::Mapping(mapping) => {
            for (key, child) in mapping {
                validate_yaml_depth(key, depth + 1)?;
                validate_yaml_depth(child, depth + 1)?;
            }
        }
        Value::Tagged(tagged) => validate_yaml_depth(&tagged.value, depth + 1)?,
        _ => {}
    }
    Ok(())
}

fn sanitize_python_name_tags(yaml: &str) -> String {
    const TAG: &str = "!!python/name:";
    let mut out = String::with_capacity(yaml.len());
    let mut rest = yaml;
    while let Some(index) = rest.find(TAG) {
        out.push_str(&rest[..index]);
        let after_tag = &rest[index + TAG.len()..];
        let name_len = after_tag
            .char_indices()
            .find_map(|(idx, ch)| {
                if ch.is_whitespace() || matches!(ch, ',' | ']' | '}' | '#') {
                    Some(idx)
                } else {
                    None
                }
            })
            .unwrap_or(after_tag.len());
        out.push('"');
        out.push_str(&after_tag[..name_len]);
        out.push('"');
        rest = &after_tag[name_len..];
    }
    out.push_str(rest);
    out
}

fn parse_composites(mapping: &Mapping) -> Result<BTreeMap<String, CompositeDefinition>> {
    let mut composites = BTreeMap::new();
    for (name, value) in mapping {
        let name = yaml_scalar_to_string(name)?;
        let definition = parse_composite_definition(name.clone(), value)?;
        composites.insert(name, definition);
    }
    Ok(composites)
}

fn parse_composite_definition(name: String, value: &Value) -> Result<CompositeDefinition> {
    let mapping = value.as_mapping().ok_or_else(|| {
        RustySatError::invalid_input(format!("composite '{name}' definition must be a mapping"))
    })?;
    let compositor = yaml_python_name(required_value(mapping, "compositor", &name)?)?;
    let prerequisites = optional_sequence(mapping, "prerequisites")?
        .map(|values| parse_dependencies(values, &name))
        .transpose()?
        .unwrap_or_default();
    let optional_prerequisites = optional_sequence(mapping, "optional_prerequisites")?
        .map(|values| parse_dependencies(values, &name))
        .transpose()?
        .unwrap_or_default();
    let mut attrs = BTreeMap::new();
    for (key, value) in mapping {
        let key = yaml_scalar_to_string(key)?;
        if matches!(
            key.as_str(),
            "compositor" | "prerequisites" | "optional_prerequisites"
        ) {
            continue;
        }
        attrs.insert(key, yaml_to_metadata_value(value)?);
    }
    attrs.insert("name".to_string(), MetadataValue::string(name.clone()));
    Ok(CompositeDefinition {
        name,
        compositor,
        prerequisites,
        optional_prerequisites,
        attrs,
    })
}

fn parse_dependencies(values: &[Value], parent_name: &str) -> Result<Vec<CompositeDependency>> {
    values
        .iter()
        .enumerate()
        .map(|(index, value)| parse_dependency(value, parent_name, index))
        .collect()
}

fn parse_dependency(value: &Value, parent_name: &str, index: usize) -> Result<CompositeDependency> {
    match value {
        Value::Mapping(mapping) if optional_value(mapping, "compositor").is_some() => {
            let inline_name = optional_value(mapping, "name")
                .map(yaml_scalar_to_string)
                .transpose()?
                .unwrap_or_else(|| format!("_{parent_name}_dep_{index}"));
            parse_composite_definition(inline_name, value)
                .map(Box::new)
                .map(CompositeDependency::InlineComposite)
        }
        Value::Mapping(mapping) => {
            let mut query = BTreeMap::new();
            for (key, value) in mapping {
                query.insert(yaml_scalar_to_string(key)?, yaml_to_metadata_value(value)?);
            }
            Ok(CompositeDependency::Query(query))
        }
        _ => Ok(CompositeDependency::Query(BTreeMap::from([(
            "name".to_string(),
            yaml_to_metadata_value(value)?,
        )]))),
    }
}

fn parse_enhancements(mapping: &Mapping) -> Result<BTreeMap<String, EnhancementDefinition>> {
    let mut enhancements = BTreeMap::new();
    for (name, value) in mapping {
        let name = yaml_scalar_to_string(name)?;
        let definition = parse_enhancement_definition(name.clone(), value)?;
        enhancements.insert(name, definition);
    }
    Ok(enhancements)
}

fn parse_enhancement_definition(name: String, value: &Value) -> Result<EnhancementDefinition> {
    let mapping = value.as_mapping().ok_or_else(|| {
        RustySatError::invalid_input(format!("enhancement '{name}' definition must be a mapping"))
    })?;
    let operations = optional_sequence(mapping, "operations")?
        .map(|values| parse_enhancement_operations(values))
        .transpose()?
        .unwrap_or_default();
    let mut match_attrs = BTreeMap::new();
    for (key, value) in mapping {
        let key = yaml_scalar_to_string(key)?;
        if key == "operations" {
            continue;
        }
        match_attrs.insert(key, yaml_to_metadata_value(value)?);
    }
    Ok(EnhancementDefinition {
        name,
        match_attrs,
        operations,
    })
}

fn parse_enhancement_operations(values: &[Value]) -> Result<Vec<EnhancementOperation>> {
    values
        .iter()
        .map(|value| {
            let mapping = value.as_mapping().ok_or_else(|| {
                RustySatError::invalid_input("enhancement operation must be a mapping")
            })?;
            let name = optional_value(mapping, "name")
                .map(yaml_scalar_to_string)
                .transpose()?
                .unwrap_or_else(|| "unnamed".to_string());
            let method = yaml_python_name(required_value(mapping, "method", &name)?)?;
            let args = optional_sequence(mapping, "args")?
                .map(|values| values.iter().map(yaml_to_metadata_value).collect())
                .transpose()?
                .unwrap_or_default();
            let kwargs = optional_mapping(mapping, "kwargs")?
                .map(parse_metadata_mapping)
                .transpose()?
                .unwrap_or_default();
            Ok(EnhancementOperation {
                name,
                method,
                args,
                kwargs,
            })
        })
        .collect()
}

fn parse_metadata_mapping(mapping: &Mapping) -> Result<BTreeMap<String, MetadataValue>> {
    let mut attrs = BTreeMap::new();
    for (key, value) in mapping {
        attrs.insert(yaml_scalar_to_string(key)?, yaml_to_metadata_value(value)?);
    }
    Ok(attrs)
}

fn yaml_to_metadata_value(value: &Value) -> Result<MetadataValue> {
    match value {
        Value::Null => Ok(MetadataValue::Null),
        Value::Bool(value) => Ok(MetadataValue::Bool(*value)),
        Value::Number(value) => {
            if let Some(value) = value.as_i64() {
                Ok(MetadataValue::Integer(value))
            } else if let Some(value) = value.as_f64() {
                MetadataValue::float(value)
            } else {
                Err(RustySatError::invalid_input(
                    "unsupported YAML metadata number",
                ))
            }
        }
        Value::String(value) => Ok(MetadataValue::string(value)),
        Value::Sequence(values) => values
            .iter()
            .map(yaml_to_metadata_value)
            .collect::<Result<Vec<_>>>()
            .map(MetadataValue::List),
        Value::Mapping(mapping) => parse_metadata_mapping(mapping).map(MetadataValue::Map),
        Value::Tagged(tagged) => {
            if let Some(name) = python_name_from_tag(tagged.tag.to_string().as_str()) {
                if name.is_empty() {
                    yaml_scalar_to_string(&tagged.value).map(MetadataValue::string)
                } else {
                    Ok(MetadataValue::string(name))
                }
            } else {
                yaml_to_metadata_value(&tagged.value)
            }
        }
    }
}

fn yaml_python_name(value: &Value) -> Result<String> {
    match value {
        Value::Tagged(tagged) => {
            if let Some(name) = python_name_from_tag(tagged.tag.to_string().as_str()) {
                if name.is_empty() {
                    yaml_scalar_to_string(&tagged.value)
                } else {
                    Ok(name.to_string())
                }
            } else {
                yaml_scalar_to_string(&tagged.value)
            }
        }
        _ => yaml_scalar_to_string(value),
    }
}

fn python_name_from_tag(tag: &str) -> Option<&str> {
    tag.trim_start_matches('!').strip_prefix("python/name:")
}

fn yaml_scalar_to_string(value: &Value) -> Result<String> {
    match value {
        Value::String(value) => Ok(value.clone()),
        Value::Number(value) => Ok(value.to_string()),
        Value::Bool(value) => Ok(value.to_string()),
        Value::Tagged(tagged) => {
            if let Some(name) = python_name_from_tag(tagged.tag.to_string().as_str()) {
                if name.is_empty() {
                    yaml_scalar_to_string(&tagged.value)
                } else {
                    Ok(name.to_string())
                }
            } else {
                yaml_scalar_to_string(&tagged.value)
            }
        }
        _ => Err(RustySatError::invalid_input(
            "YAML value must be a scalar string, number, bool, or python/name tag",
        )),
    }
}

fn required_value<'a>(mapping: &'a Mapping, key: &str, section: &str) -> Result<&'a Value> {
    mapping
        .get(Value::String(key.to_string()))
        .ok_or_else(|| RustySatError::invalid_input(format!("{section} missing '{key}'")))
}

fn optional_value<'a>(mapping: &'a Mapping, key: &str) -> Option<&'a Value> {
    mapping.get(Value::String(key.to_string()))
}

fn optional_mapping<'a>(mapping: &'a Mapping, key: &str) -> Result<Option<&'a Mapping>> {
    optional_value(mapping, key)
        .map(|value| {
            value
                .as_mapping()
                .ok_or_else(|| RustySatError::invalid_input(format!("{key} must be a mapping")))
        })
        .transpose()
}

fn optional_sequence<'a>(mapping: &'a Mapping, key: &str) -> Result<Option<&'a Vec<Value>>> {
    optional_value(mapping, key)
        .map(|value| {
            value
                .as_sequence()
                .ok_or_else(|| RustySatError::invalid_input(format!("{key} must be a sequence")))
        })
        .transpose()
}

#[cfg(test)]
mod tests {
    use super::*;

    const CONFIG: &str = r#"
sensor_name: visir/ahi

composites:
  reproduced_green:
    compositor: !!python/name:satpy.composites.spectral.SpectralBlender
    fractions: [0.6321, 0.2928, 0.0751]
    prerequisites:
      - name: B02
        modifiers: [sunz_corrected, rayleigh_corrected]
      - name: B03
      - name: B04
    standard_name: none
  airmass:
    compositor: !!python/name:satpy.composites.core.GenericCompositor
    prerequisites:
      - compositor: !!python/name:satpy.composites.arithmetic.DifferenceCompositor
        prerequisites:
          - name: B08
          - name: B10
      - name: B08
    standard_name: airmass

enhancements:
  natural_color_default:
    standard_name: natural_color
    operations:
      - name: stretch
        method: !!python/name:satpy.enhancements.contrast.stretch
        kwargs: {stretch: crude, min_stretch: 0, max_stretch: 120}
      - name: gamma
        method: !!python/name:satpy.enhancements.contrast.gamma
        kwargs: {gamma: 1.8}
"#;

    #[test]
    fn parses_composite_yaml_registration_without_executing_python_tags() {
        let registry = CompositeRegistryConfig::from_yaml_str(CONFIG).unwrap();
        let green = registry.composites().get("reproduced_green").unwrap();

        assert_eq!(green.name(), "reproduced_green");
        assert_eq!(
            green.compositor(),
            "satpy.composites.spectral.SpectralBlender"
        );
        assert_eq!(green.prerequisites().len(), 3);
        assert_eq!(
            green.attrs().get("standard_name"),
            Some(&MetadataValue::string("none"))
        );
        assert!(matches!(
            &green.prerequisites()[0],
            CompositeDependency::Query(query)
                if query.get("name") == Some(&MetadataValue::string("B02"))
        ));
    }

    #[test]
    fn parses_inline_composite_dependencies() {
        let registry = CompositeRegistryConfig::from_yaml_str(CONFIG).unwrap();
        let airmass = registry.composites().get("airmass").unwrap();

        let CompositeDependency::InlineComposite(inline) = &airmass.prerequisites()[0] else {
            panic!("expected inline composite dependency");
        };
        assert_eq!(inline.name(), "_airmass_dep_0");
        assert_eq!(
            inline.compositor(),
            "satpy.composites.arithmetic.DifferenceCompositor"
        );
        assert_eq!(inline.prerequisites().len(), 2);
    }

    #[test]
    fn parses_enhancement_operations() {
        let registry = CompositeRegistryConfig::from_yaml_str(CONFIG).unwrap();
        let enhancement = registry
            .enhancements()
            .get("natural_color_default")
            .unwrap();

        assert_eq!(enhancement.name(), "natural_color_default");
        assert_eq!(
            enhancement.match_attrs().get("standard_name"),
            Some(&MetadataValue::string("natural_color"))
        );
        assert_eq!(enhancement.operations().len(), 2);
        assert_eq!(
            enhancement.operations()[0].method(),
            "satpy.enhancements.contrast.stretch"
        );
        assert_eq!(
            enhancement.operations()[0].kwargs().get("stretch"),
            Some(&MetadataValue::string("crude"))
        );
    }

    #[test]
    fn rejects_missing_compositor_and_invalid_sections() {
        assert!(CompositeRegistryConfig::from_yaml_str("composites:\n  bad: {}\n").is_err());
        assert!(CompositeRegistryConfig::from_yaml_str("enhancements: []\n").is_err());
    }
}
