//! Configuration loading foundations.
//!
//! Satpy is strongly configuration driven. This crate starts the Rusty Sat
//! equivalent with deterministic search paths and recursive YAML merging.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use rusty_sat_core::{Result, RustySatError};
use serde_norway::{Mapping, Value};

pub const DEFAULT_CONFIG_ENV: &str = "RUSTY_SAT_CONFIG_PATH";
pub const SATPY_COMPAT_CONFIG_ENV: &str = "SATPY_CONFIG_PATH";
const MAX_YAML_BYTES: usize = 8 * 1024 * 1024;
const MAX_YAML_DEPTH: usize = 96;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfigSearchPath {
    paths: Vec<PathBuf>,
}

impl Default for ConfigSearchPath {
    fn default() -> Self {
        Self {
            paths: vec![PathBuf::from("satpy/satpy/etc")],
        }
    }
}

impl ConfigSearchPath {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn empty() -> Self {
        Self { paths: Vec::new() }
    }

    pub fn from_env() -> Self {
        let mut search = Self::new();
        search.extend_env(DEFAULT_CONFIG_ENV);
        search.extend_env(SATPY_COMPAT_CONFIG_ENV);
        search
    }

    pub fn push(&mut self, path: impl Into<PathBuf>) {
        self.paths.push(path.into());
    }

    pub fn extend_env(&mut self, var_name: &str) {
        if let Some(value) = env::var_os(var_name) {
            self.paths.extend(env::split_paths(&value));
        }
    }

    pub fn paths(&self) -> &[PathBuf] {
        &self.paths
    }

    pub fn config_search_paths(&self, relative_file: impl AsRef<Path>) -> Vec<PathBuf> {
        let relative_file = relative_file.as_ref();
        let mut paths = Vec::new();

        if relative_file.is_absolute() || relative_file.exists() {
            paths.push(relative_file.to_path_buf());
        }

        for base in &self.paths {
            paths.push(base.join(relative_file));
        }

        dedupe_existing_files(paths)
    }

    pub fn component_config_paths(&self, component: ConfigComponent, name: &str) -> Vec<PathBuf> {
        let filename = if name.ends_with(".yaml") || name.ends_with(".yml") {
            name.to_string()
        } else {
            format!("{name}.yaml")
        };
        self.config_search_paths(component.directory().join(filename))
    }

    pub fn load_yaml_file(&self, relative_file: impl AsRef<Path>) -> Result<Value> {
        let paths = self.config_search_paths(relative_file);
        if paths.is_empty() {
            return Err(RustySatError::not_found("config file"));
        }
        load_and_merge_yaml_files(&paths)
    }

    pub fn load_component_yaml(&self, component: ConfigComponent, name: &str) -> Result<Value> {
        let paths = self.component_config_paths(component, name);
        if paths.is_empty() {
            return Err(RustySatError::not_found(format!(
                "{} config: {name}",
                component.as_str()
            )));
        }
        load_and_merge_yaml_files(&paths)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfigComponent {
    Readers,
    Writers,
    Composites,
    Enhancements,
}

impl ConfigComponent {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Readers => "readers",
            Self::Writers => "writers",
            Self::Composites => "composites",
            Self::Enhancements => "enhancements",
        }
    }

    pub fn directory(self) -> PathBuf {
        PathBuf::from(self.as_str())
    }
}

pub fn load_and_merge_yaml_files(paths: &[PathBuf]) -> Result<Value> {
    let mut merged = Value::Mapping(Mapping::new());
    for path in paths {
        let next = load_yaml_file(path)?;
        merge_yaml(&mut merged, next);
    }
    Ok(merged)
}

pub fn load_yaml_file(path: &Path) -> Result<Value> {
    let content = fs::read_to_string(path).map_err(|err| {
        RustySatError::invalid_input(format!("failed to read {}: {err}", path.display()))
    })?;
    parse_yaml_value(&content, &format!("{}", path.display()))
}

fn parse_yaml_value(yaml: &str, context: &str) -> Result<Value> {
    if yaml.len() > MAX_YAML_BYTES {
        return Err(RustySatError::invalid_input(format!(
            "{context} exceeds YAML size limit of {MAX_YAML_BYTES} bytes"
        )));
    }
    let value: Value = serde_norway::from_str(yaml)
        .map_err(|err| RustySatError::invalid_input(format!("failed to parse {context}: {err}")))?;
    validate_yaml_depth(&value, 0, context)?;
    Ok(value)
}

fn validate_yaml_depth(value: &Value, depth: usize, context: &str) -> Result<()> {
    if depth > MAX_YAML_DEPTH {
        return Err(RustySatError::invalid_input(format!(
            "{context} exceeds YAML nesting depth limit of {MAX_YAML_DEPTH}"
        )));
    }
    if let Some(sequence) = value.as_sequence() {
        for child in sequence {
            validate_yaml_depth(child, depth + 1, context)?;
        }
    }
    if let Some(mapping) = value.as_mapping() {
        for (key, child) in mapping {
            validate_yaml_depth(key, depth + 1, context)?;
            validate_yaml_depth(child, depth + 1, context)?;
        }
    }
    if let Value::Tagged(tagged) = value {
        validate_yaml_depth(&tagged.value, depth + 1, context)?;
    }
    Ok(())
}

pub fn merge_yaml(base: &mut Value, overlay: Value) {
    match (base, overlay) {
        (Value::Mapping(base_map), Value::Mapping(overlay_map)) => {
            merge_yaml_mapping(base_map, overlay_map);
        }
        (base_value, overlay_value) => {
            *base_value = overlay_value;
        }
    }
}

fn merge_yaml_mapping(base: &mut Mapping, overlay: Mapping) {
    for (key, overlay_value) in overlay {
        match base.get_mut(&key) {
            Some(base_value) => merge_yaml(base_value, overlay_value),
            None => {
                base.insert(key, overlay_value);
            }
        }
    }
}

fn dedupe_existing_files(paths: Vec<PathBuf>) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for path in paths {
        if !path.is_file() || out.iter().any(|existing| existing == &path) {
            continue;
        }
        out.push(path);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

    #[test]
    fn default_search_path_points_to_satpy_etc_reference() {
        let paths = ConfigSearchPath::new();
        assert_eq!(paths.paths(), &[PathBuf::from("satpy/satpy/etc")]);
    }

    #[test]
    fn finds_component_config_in_search_path() {
        let temp = TestTempDir::new();
        temp.write(
            "readers/example.yaml",
            "reader:\n  name: example\n  sensors: [fake]\n",
        );
        let mut paths = ConfigSearchPath::empty();
        paths.push(temp.path());

        let found = paths.component_config_paths(ConfigComponent::Readers, "example");
        assert_eq!(found, vec![temp.path().join("readers/example.yaml")]);
    }

    #[test]
    fn merges_yaml_recursively_with_later_files_winning() {
        let temp = TestTempDir::new();
        let base = temp.write(
            "base.yaml",
            "reader:\n  name: base\n  options:\n    a: 1\n    b: 2\n",
        );
        let overlay = temp.write("overlay.yaml", "reader:\n  options:\n    b: 3\n    c: 4\n");

        let merged = load_and_merge_yaml_files(&[base, overlay]).unwrap();
        assert_eq!(merged["reader"]["name"], Value::from("base"));
        assert_eq!(merged["reader"]["options"]["a"], Value::from(1));
        assert_eq!(merged["reader"]["options"]["b"], Value::from(3));
        assert_eq!(merged["reader"]["options"]["c"], Value::from(4));
    }

    #[test]
    fn rejects_excessively_nested_yaml() {
        let yaml = format!("root: {}", "[".repeat(MAX_YAML_DEPTH + 2))
            + "0"
            + &"]".repeat(MAX_YAML_DEPTH + 2);

        let err = parse_yaml_value(&yaml, "test YAML").unwrap_err();

        assert!(err.to_string().contains("nesting depth limit"));
    }

    #[test]
    fn reports_missing_component_config() {
        let paths = ConfigSearchPath::empty();
        let err = paths
            .load_component_yaml(ConfigComponent::Readers, "missing")
            .unwrap_err();
        assert!(err.to_string().contains("not found"));
    }

    struct TestTempDir {
        path: PathBuf,
    }

    impl TestTempDir {
        fn new() -> Self {
            let millis = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis();
            let path = env::temp_dir().join(format!(
                "rusty_sat_config_test_{}_{}_{}",
                std::process::id(),
                millis,
                TEMP_COUNTER.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir_all(&path).unwrap();
            Self { path }
        }

        fn path(&self) -> PathBuf {
            self.path.clone()
        }

        fn write(&self, relative: &str, content: &str) -> PathBuf {
            let path = self.path.join(relative);
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent).unwrap();
            }
            fs::write(&path, content).unwrap();
            path
        }
    }

    impl Drop for TestTempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }
}
