//! Core public API foundations for Rusty Sat.
//!
//! This crate intentionally starts small. It provides the shared result/error
//! type and lightweight versions of Satpy's central concepts so the rest of
//! the workspace can compile while features are ported incrementally.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt::{self, Display};
use std::hash::{Hash, Hasher};

pub type Result<T> = std::result::Result<T, RustySatError>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RustySatError {
    Unsupported { feature: String },
    InvalidInput { message: String },
    NotFound { item: String },
    Ambiguous { message: String },
}

impl RustySatError {
    pub fn unsupported(feature: impl Into<String>) -> Self {
        Self::Unsupported {
            feature: feature.into(),
        }
    }

    pub fn invalid_input(message: impl Into<String>) -> Self {
        Self::InvalidInput {
            message: message.into(),
        }
    }

    pub fn not_found(item: impl Into<String>) -> Self {
        Self::NotFound { item: item.into() }
    }

    pub fn ambiguous(message: impl Into<String>) -> Self {
        Self::Ambiguous {
            message: message.into(),
        }
    }
}

impl Display for RustySatError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unsupported { feature } => write!(f, "unsupported feature: {feature}"),
            Self::InvalidInput { message } => write!(f, "invalid input: {message}"),
            Self::NotFound { item } => write!(f, "not found: {item}"),
            Self::Ambiguous { message } => write!(f, "ambiguous result: {message}"),
        }
    }
}

impl Error for RustySatError {}

#[derive(Debug, Clone, Copy)]
pub struct FloatValue(f64);

impl FloatValue {
    pub fn new(value: f64) -> Result<Self> {
        if value.is_nan() {
            return Err(RustySatError::invalid_input(
                "floating point values cannot be NaN",
            ));
        }
        Ok(Self(value))
    }

    pub fn get(self) -> f64 {
        self.0
    }
}

impl PartialEq for FloatValue {
    fn eq(&self, other: &Self) -> bool {
        self.0.to_bits() == other.0.to_bits()
    }
}

impl Eq for FloatValue {}

impl PartialOrd for FloatValue {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for FloatValue {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0)
    }
}

impl Hash for FloatValue {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.to_bits().hash(state);
    }
}

impl Display for FloatValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct WavelengthRange {
    min: FloatValue,
    central: FloatValue,
    max: FloatValue,
    unit: String,
}

impl WavelengthRange {
    pub fn new(min: f64, central: f64, max: f64, unit: impl Into<String>) -> Result<Self> {
        if min > central || central > max {
            return Err(RustySatError::invalid_input(
                "wavelength range must satisfy min <= central <= max",
            ));
        }
        let unit = unit.into();
        if unit.trim().is_empty() {
            return Err(RustySatError::invalid_input(
                "wavelength unit cannot be empty",
            ));
        }
        Ok(Self {
            min: FloatValue::new(min)?,
            central: FloatValue::new(central)?,
            max: FloatValue::new(max)?,
            unit,
        })
    }

    pub fn micrometers(min: f64, central: f64, max: f64) -> Result<Self> {
        Self::new(min, central, max, "um")
    }

    pub fn contains_number(&self, value: f64) -> bool {
        self.min.get() <= value && value <= self.max.get()
    }

    pub fn contains_range(&self, other: &Self) -> bool {
        self.unit == other.unit && self.min <= other.min && other.max <= self.max
    }

    pub fn central(&self) -> f64 {
        self.central.get()
    }

    pub fn distance_to_number(&self, value: f64) -> f64 {
        if self.contains_number(value) {
            (self.central.get() - value).abs()
        } else {
            f64::INFINITY
        }
    }

    pub fn distance_to_range(&self, other: &Self) -> f64 {
        if self.contains_range(other) {
            (self.central.get() - other.central.get()).abs()
        } else {
            f64::INFINITY
        }
    }

    pub fn unit(&self) -> &str {
        &self.unit
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ModifierTuple(Vec<String>);

impl ModifierTuple {
    pub fn new(modifiers: impl IntoIterator<Item = impl Into<String>>) -> Result<Self> {
        let mut out = Vec::new();
        for modifier in modifiers {
            let modifier = modifier.into();
            if modifier.trim().is_empty() {
                return Err(RustySatError::invalid_input(
                    "modifier name cannot be empty",
                ));
            }
            out.push(modifier);
        }
        Ok(Self(out))
    }

    pub fn as_slice(&self) -> &[String] {
        &self.0
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DataValue {
    Text(String),
    Number(FloatValue),
    Wavelength(WavelengthRange),
    Modifiers(ModifierTuple),
}

impl DataValue {
    fn matches_query_value(&self, query_value: &QueryValue) -> bool {
        match query_value {
            QueryValue::Any => true,
            QueryValue::One(value) => self.matches_data_value(value),
            QueryValue::AnyOf(values) => values.iter().any(|value| self.matches_data_value(value)),
        }
    }

    fn matches_data_value(&self, requested: &DataValue) -> bool {
        match (self, requested) {
            (DataValue::Wavelength(range), DataValue::Number(number)) => {
                range.contains_number(number.get())
            }
            (DataValue::Wavelength(range), DataValue::Wavelength(requested_range)) => {
                range.contains_range(requested_range)
            }
            (DataValue::Number(number), DataValue::Wavelength(range)) => {
                range.contains_number(number.get())
            }
            _ => self == requested,
        }
    }

    fn absolute_distance(&self, key: &str) -> f64 {
        match self {
            Self::Text(value) if key == "calibration" => calibration_priority(value),
            Self::Text(_) => 0.0,
            Self::Number(value) => value.get(),
            Self::Wavelength(_) => 0.0,
            Self::Modifiers(modifiers) => modifiers.as_slice().len() as f64,
        }
    }

    fn distance_from_query_value(&self, query_value: &QueryValue) -> f64 {
        match query_value {
            QueryValue::Any => 0.0,
            QueryValue::One(value) => self.distance_from_data_value(value),
            QueryValue::AnyOf(values) => values
                .iter()
                .map(|value| self.distance_from_data_value(value))
                .fold(f64::INFINITY, f64::min),
        }
    }

    fn distance_from_data_value(&self, requested: &DataValue) -> f64 {
        match (self, requested) {
            (DataValue::Wavelength(range), DataValue::Number(number)) => {
                range.distance_to_number(number.get())
            }
            (DataValue::Wavelength(range), DataValue::Wavelength(requested_range)) => {
                range.distance_to_range(requested_range)
            }
            (DataValue::Number(number), DataValue::Wavelength(range)) => {
                range.distance_to_number(number.get())
            }
            (DataValue::Number(number), DataValue::Number(requested)) => {
                if number == requested {
                    number.get()
                } else {
                    f64::INFINITY
                }
            }
            _ if self == requested => 0.0,
            _ => f64::INFINITY,
        }
    }
}

fn calibration_priority(value: &str) -> f64 {
    match value {
        "reflectance" => 0.0,
        "brightness_temperature" => 1.0,
        "radiance" => 2.0,
        "radiance_wavenumber" => 3.0,
        "counts" => 4.0,
        _ => 1000.0,
    }
}

impl From<String> for DataValue {
    fn from(value: String) -> Self {
        Self::Text(value)
    }
}

impl From<&str> for DataValue {
    fn from(value: &str) -> Self {
        Self::Text(value.to_string())
    }
}

impl From<FloatValue> for DataValue {
    fn from(value: FloatValue) -> Self {
        Self::Number(value)
    }
}

impl From<f64> for DataValue {
    fn from(value: f64) -> Self {
        Self::Number(FloatValue::new(value).expect("DataValue cannot be created from NaN"))
    }
}

impl From<i64> for DataValue {
    fn from(value: i64) -> Self {
        Self::Number(FloatValue::new(value as f64).expect("integer conversion cannot produce NaN"))
    }
}

impl From<u64> for DataValue {
    fn from(value: u64) -> Self {
        Self::Number(FloatValue::new(value as f64).expect("integer conversion cannot produce NaN"))
    }
}

impl From<WavelengthRange> for DataValue {
    fn from(value: WavelengthRange) -> Self {
        Self::Wavelength(value)
    }
}

impl From<ModifierTuple> for DataValue {
    fn from(value: ModifierTuple) -> Self {
        Self::Modifiers(value)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QueryValue {
    Any,
    One(DataValue),
    AnyOf(Vec<DataValue>),
}

impl QueryValue {
    pub fn one(value: impl Into<DataValue>) -> Self {
        Self::One(value.into())
    }

    pub fn any_of(values: impl IntoIterator<Item = impl Into<DataValue>>) -> Self {
        Self::AnyOf(values.into_iter().map(Into::into).collect())
    }
}

impl From<DataValue> for QueryValue {
    fn from(value: DataValue) -> Self {
        Self::One(value)
    }
}

impl From<String> for QueryValue {
    fn from(value: String) -> Self {
        Self::One(DataValue::Text(value))
    }
}

impl From<&str> for QueryValue {
    fn from(value: &str) -> Self {
        if value == "*" {
            Self::Any
        } else {
            Self::One(DataValue::Text(value.to_string()))
        }
    }
}

impl From<f64> for QueryValue {
    fn from(value: f64) -> Self {
        Self::One(DataValue::from(value))
    }
}

impl From<WavelengthRange> for QueryValue {
    fn from(value: WavelengthRange) -> Self {
        Self::One(DataValue::Wavelength(value))
    }
}

impl From<ModifierTuple> for QueryValue {
    fn from(value: ModifierTuple) -> Self {
        Self::One(DataValue::Modifiers(value))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DataId {
    name: String,
    qualifiers: BTreeMap<String, DataValue>,
}

impl DataId {
    pub fn new(name: impl Into<String>) -> Result<Self> {
        let name = name.into();
        if name.trim().is_empty() {
            return Err(RustySatError::invalid_input("DataId name cannot be empty"));
        }
        Ok(Self {
            name,
            qualifiers: BTreeMap::new(),
        })
    }

    pub fn with_qualifier(
        mut self,
        key: impl Into<String>,
        value: impl Into<DataValue>,
    ) -> Result<Self> {
        let key = key.into();
        if key.trim().is_empty() {
            return Err(RustySatError::invalid_input(
                "DataId qualifier key cannot be empty",
            ));
        }
        self.qualifiers.insert(key, value.into());
        Ok(self)
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn qualifiers(&self) -> &BTreeMap<String, DataValue> {
        &self.qualifiers
    }

    pub fn qualifier(&self, key: &str) -> Option<&DataValue> {
        self.qualifiers.get(key)
    }

    fn sort_keys(&self) -> BTreeSet<String> {
        let mut keys = BTreeSet::new();
        keys.insert("name".to_string());
        keys.extend(self.qualifiers.keys().cloned());
        keys
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct DataQuery {
    name: Option<String>,
    filters: BTreeMap<String, QueryValue>,
}

impl DataQuery {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn named(name: impl Into<String>) -> Result<Self> {
        let name = name.into();
        if name.trim().is_empty() {
            return Err(RustySatError::invalid_input(
                "DataQuery name cannot be empty",
            ));
        }
        Ok(Self {
            name: Some(name),
            filters: BTreeMap::new(),
        })
    }

    pub fn with_filter(
        mut self,
        key: impl Into<String>,
        value: impl Into<QueryValue>,
    ) -> Result<Self> {
        let key = key.into();
        if key.trim().is_empty() {
            return Err(RustySatError::invalid_input(
                "DataQuery filter key cannot be empty",
            ));
        }
        self.filters.insert(key, value.into());
        Ok(self)
    }

    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    pub fn filters(&self) -> &BTreeMap<String, QueryValue> {
        &self.filters
    }

    pub fn matches(&self, data_id: &DataId) -> bool {
        let mut shared_key_matched = false;
        if let Some(name) = &self.name {
            if data_id.name() != name {
                return false;
            }
            shared_key_matched = true;
        }
        for (key, query_value) in &self.filters {
            let Some(value) = data_id.qualifier(key) else {
                continue;
            };
            shared_key_matched = true;
            if !value.matches_query_value(query_value) {
                return false;
            }
        }
        shared_key_matched
    }

    pub fn filter_data_ids<'a>(
        &self,
        data_ids: impl IntoIterator<Item = &'a DataId>,
    ) -> Vec<&'a DataId> {
        data_ids
            .into_iter()
            .filter(|data_id| self.matches(data_id))
            .collect()
    }

    pub fn sort_data_ids<'a>(
        &self,
        data_ids: impl IntoIterator<Item = &'a DataId>,
    ) -> Vec<ScoredDataId<'a>> {
        let mut scored: Vec<_> = data_ids
            .into_iter()
            .map(|data_id| ScoredDataId {
                data_id,
                distance: self.distance_to(data_id),
            })
            .collect();
        scored.sort_by(|left, right| {
            left.distance
                .total_cmp(&right.distance)
                .then_with(|| left.data_id.cmp(right.data_id))
        });
        scored
    }

    pub fn best_matches<'a>(
        &self,
        data_ids: impl IntoIterator<Item = &'a DataId>,
    ) -> Vec<&'a DataId> {
        let scored = self.sort_data_ids(self.filter_data_ids(data_ids));
        let Some(best) = scored.first() else {
            return Vec::new();
        };
        if !best.distance.is_finite() {
            return Vec::new();
        }
        scored
            .iter()
            .take_while(|score| score.distance == best.distance)
            .map(|score| score.data_id)
            .collect()
    }

    pub fn best_match<'a>(
        &self,
        data_ids: impl IntoIterator<Item = &'a DataId>,
    ) -> Result<&'a DataId> {
        let matches = self.best_matches(data_ids);
        match matches.as_slice() {
            [] => Err(RustySatError::not_found("dataset matching query")),
            [data_id] => Ok(data_id),
            _ => Err(RustySatError::ambiguous(format!(
                "query matched {} equally good datasets",
                matches.len()
            ))),
        }
    }

    fn distance_to(&self, data_id: &DataId) -> f64 {
        let mut keys = data_id.sort_keys();
        keys.extend(self.filters.keys().cloned());
        let mut distance = 0.0;

        if let Some(query_name) = &self.name {
            if query_name != data_id.name() {
                return f64::INFINITY;
            }
        }

        for key in keys {
            if key == "name" {
                continue;
            }
            let query_value = self.filters.get(&key).unwrap_or(&QueryValue::Any);
            match (query_value, data_id.qualifier(&key)) {
                (QueryValue::Any, Some(value)) => distance += value.absolute_distance(&key),
                (QueryValue::Any, None) => {}
                (_, Some(value)) => {
                    distance += value.distance_from_query_value(query_value);
                }
                (_, None) => distance += 100_000.0,
            }
            if !distance.is_finite() {
                break;
            }
        }

        distance
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScoredDataId<'a> {
    pub data_id: &'a DataId,
    pub distance: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Dataset {
    id: DataId,
    metadata: BTreeMap<String, String>,
}

impl Dataset {
    pub fn new(id: DataId) -> Self {
        Self {
            id,
            metadata: BTreeMap::new(),
        }
    }

    pub fn id(&self) -> &DataId {
        &self.id
    }

    pub fn metadata(&self) -> &BTreeMap<String, String> {
        &self.metadata
    }

    pub fn insert_metadata(
        &mut self,
        key: impl Into<String>,
        value: impl Into<String>,
    ) -> Result<()> {
        let key = key.into();
        if key.trim().is_empty() {
            return Err(RustySatError::invalid_input("metadata key cannot be empty"));
        }
        self.metadata.insert(key, value.into());
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DependencySource {
    UserProvided,
    Reader(String),
    Composite(String),
    Modifier(String),
    Unknown,
}

impl DependencySource {
    pub fn reader(name: impl Into<String>) -> Result<Self> {
        non_empty_named_source("reader", name).map(Self::Reader)
    }

    pub fn composite(name: impl Into<String>) -> Result<Self> {
        non_empty_named_source("composite", name).map(Self::Composite)
    }

    pub fn modifier(name: impl Into<String>) -> Result<Self> {
        non_empty_named_source("modifier", name).map(Self::Modifier)
    }
}

fn non_empty_named_source(kind: &str, name: impl Into<String>) -> Result<String> {
    let name = name.into();
    if name.trim().is_empty() {
        return Err(RustySatError::invalid_input(format!(
            "{kind} dependency source name cannot be empty"
        )));
    }
    Ok(name)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DependencyNode {
    id: DataId,
    source: DependencySource,
    dependencies: BTreeSet<DataId>,
}

impl DependencyNode {
    pub fn new(id: DataId, source: DependencySource) -> Self {
        Self {
            id,
            source,
            dependencies: BTreeSet::new(),
        }
    }

    pub fn id(&self) -> &DataId {
        &self.id
    }

    pub fn source(&self) -> &DependencySource {
        &self.source
    }

    pub fn dependencies(&self) -> &BTreeSet<DataId> {
        &self.dependencies
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct DependencyGraph {
    nodes: BTreeMap<DataId, DependencyNode>,
}

impl DependencyGraph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    pub fn add_node(&mut self, id: DataId, source: DependencySource) -> Result<()> {
        if let Some(existing) = self.nodes.get_mut(&id) {
            if existing.source != source && existing.source != DependencySource::Unknown {
                return Err(RustySatError::ambiguous(format!(
                    "dependency node '{}' already exists with a different source",
                    id.name()
                )));
            }
            existing.source = source;
            return Ok(());
        }
        self.nodes
            .insert(id.clone(), DependencyNode::new(id, source));
        Ok(())
    }

    pub fn add_leaf(&mut self, id: DataId) -> Result<()> {
        self.add_node(id, DependencySource::UserProvided)
    }

    pub fn add_dependency(&mut self, parent: DataId, dependency: DataId) -> Result<()> {
        if !self.contains(&parent) {
            self.add_node(parent.clone(), DependencySource::Unknown)?;
        }
        if !self.contains(&dependency) {
            self.add_node(dependency.clone(), DependencySource::Unknown)?;
        }
        let Some(node) = self.nodes.get_mut(&parent) else {
            return Err(RustySatError::not_found("dependency parent node"));
        };
        node.dependencies.insert(dependency);
        Ok(())
    }

    pub fn get(&self, id: &DataId) -> Option<&DependencyNode> {
        self.nodes.get(id)
    }

    pub fn contains(&self, id: &DataId) -> bool {
        self.nodes.contains_key(id)
    }

    pub fn remove(&mut self, id: &DataId) -> Option<DependencyNode> {
        for node in self.nodes.values_mut() {
            node.dependencies.remove(id);
        }
        self.nodes.remove(id)
    }

    pub fn dependencies_for(&self, id: &DataId) -> Result<&BTreeSet<DataId>> {
        self.nodes
            .get(id)
            .map(DependencyNode::dependencies)
            .ok_or_else(|| RustySatError::not_found(format!("dependency node '{}'", id.name())))
    }

    pub fn dependents_of(&self, id: &DataId) -> BTreeSet<DataId> {
        self.nodes
            .iter()
            .filter_map(|(node_id, node)| node.dependencies().contains(id).then(|| node_id.clone()))
            .collect()
    }

    pub fn leaves(&self) -> BTreeSet<DataId> {
        self.nodes
            .iter()
            .filter_map(|(id, node)| node.dependencies().is_empty().then(|| id.clone()))
            .collect()
    }
}

#[derive(Debug, Default)]
pub struct Scene {
    datasets: BTreeMap<DataId, Dataset>,
    wishlist: BTreeSet<DataId>,
    dependency_graph: DependencyGraph,
}

impl Scene {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_empty(&self) -> bool {
        self.datasets.is_empty()
    }

    pub fn len(&self) -> usize {
        self.datasets.len()
    }

    pub fn insert_dataset(&mut self, dataset: Dataset) {
        let id = dataset.id().clone();
        self.wishlist.insert(id.clone());
        self.dependency_graph
            .add_leaf(id.clone())
            .expect("DataId from Dataset must be valid dependency leaf");
        self.datasets.insert(id, dataset);
    }

    pub fn get(&self, id: &DataId) -> Option<&Dataset> {
        self.datasets.get(id)
    }

    pub fn remove_dataset(&mut self, id: &DataId) -> Option<Dataset> {
        self.wishlist.remove(id);
        self.dependency_graph.remove(id);
        self.datasets.remove(id)
    }

    pub fn wishlist(&self) -> &BTreeSet<DataId> {
        &self.wishlist
    }

    pub fn dependency_graph(&self) -> &DependencyGraph {
        &self.dependency_graph
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constructs_empty_scene() {
        let scene = Scene::new();
        assert!(scene.is_empty());
        assert_eq!(scene.len(), 0);
    }

    #[test]
    fn constructs_data_id_and_query() {
        let data_id = DataId::new("IR_108")
            .unwrap()
            .with_qualifier("calibration", "brightness_temperature")
            .unwrap();
        assert_eq!(data_id.name(), "IR_108");
        assert_eq!(
            data_id.qualifier("calibration"),
            Some(&DataValue::Text("brightness_temperature".to_string()))
        );

        let query = DataQuery::named("IR_108")
            .unwrap()
            .with_filter("resolution", 3000.0)
            .unwrap();
        assert_eq!(query.name(), Some("IR_108"));
        assert_eq!(
            query.filters().get("resolution"),
            Some(&QueryValue::one(3000.0))
        );
    }

    #[test]
    fn scene_stores_dataset_and_wishlist() {
        let data_id = DataId::new("VIS006").unwrap();
        let dataset = Dataset::new(data_id.clone());
        let mut scene = Scene::new();
        scene.insert_dataset(dataset);

        assert_eq!(scene.len(), 1);
        assert!(scene.get(&data_id).is_some());
        assert!(scene.wishlist().contains(&data_id));
        assert!(scene.dependency_graph().contains(&data_id));
        assert_eq!(
            scene.dependency_graph().get(&data_id).unwrap().source(),
            &DependencySource::UserProvided
        );
        assert!(scene.dependency_graph().leaves().contains(&data_id));
    }

    #[test]
    fn scene_removes_dataset_from_wishlist_and_dependency_graph() {
        let data_id = DataId::new("VIS006").unwrap();
        let dataset = Dataset::new(data_id.clone());
        let mut scene = Scene::new();
        scene.insert_dataset(dataset);

        assert!(scene.remove_dataset(&data_id).is_some());
        assert!(scene.is_empty());
        assert!(!scene.wishlist().contains(&data_id));
        assert!(!scene.dependency_graph().contains(&data_id));
    }

    #[test]
    fn dependency_graph_tracks_explicit_edges() {
        let composite = DataId::new("natural_color").unwrap();
        let red = DataId::new("VIS006").unwrap();
        let green = DataId::new("VIS008").unwrap();
        let mut graph = DependencyGraph::new();

        graph
            .add_node(
                composite.clone(),
                DependencySource::composite("natural_color").unwrap(),
            )
            .unwrap();
        graph
            .add_node(red.clone(), DependencySource::reader("seviri_l1b").unwrap())
            .unwrap();
        graph
            .add_dependency(composite.clone(), red.clone())
            .unwrap();
        graph
            .add_dependency(composite.clone(), green.clone())
            .unwrap();

        assert_eq!(graph.len(), 3);
        assert!(graph.dependencies_for(&composite).unwrap().contains(&red));
        assert!(graph.dependencies_for(&composite).unwrap().contains(&green));
        assert!(graph.dependents_of(&red).contains(&composite));
        assert!(graph.leaves().contains(&red));
        assert!(graph.leaves().contains(&green));
        assert!(!graph.leaves().contains(&composite));
        assert_eq!(
            graph.get(&green).unwrap().source(),
            &DependencySource::Unknown
        );
    }

    #[test]
    fn dependency_graph_rejects_conflicting_sources() {
        let data_id = DataId::new("VIS006").unwrap();
        let mut graph = DependencyGraph::new();
        graph
            .add_node(
                data_id.clone(),
                DependencySource::reader("reader_a").unwrap(),
            )
            .unwrap();
        let err = graph
            .add_node(data_id, DependencySource::reader("reader_b").unwrap())
            .unwrap_err();

        assert!(matches!(err, RustySatError::Ambiguous { .. }));
    }

    #[test]
    fn query_matches_strings_numbers_wildcards_and_lists() {
        let data_id = DataId::new("IR_108")
            .unwrap()
            .with_qualifier("calibration", "brightness_temperature")
            .unwrap()
            .with_qualifier("resolution", 3000.0)
            .unwrap();

        assert!(DataQuery::named("IR_108")
            .unwrap()
            .with_filter("calibration", "brightness_temperature")
            .unwrap()
            .matches(&data_id));
        assert!(DataQuery::named("IR_108")
            .unwrap()
            .with_filter("calibration", QueryValue::Any)
            .unwrap()
            .matches(&data_id));
        assert!(DataQuery::named("IR_108")
            .unwrap()
            .with_filter(
                "calibration",
                QueryValue::any_of(["reflectance", "brightness_temperature"]),
            )
            .unwrap()
            .matches(&data_id));
        assert!(!DataQuery::named("VIS006").unwrap().matches(&data_id));
    }

    #[test]
    fn query_matches_wavelength_number_inside_range() {
        let data_id = DataId::new("VIS006")
            .unwrap()
            .with_qualifier(
                "wavelength",
                WavelengthRange::micrometers(0.56, 0.64, 0.71).unwrap(),
            )
            .unwrap();

        assert!(DataQuery::named("VIS006")
            .unwrap()
            .with_filter("wavelength", 0.64)
            .unwrap()
            .matches(&data_id));
        assert!(!DataQuery::named("VIS006")
            .unwrap()
            .with_filter("wavelength", 10.8)
            .unwrap()
            .matches(&data_id));
    }

    #[test]
    fn stores_modifier_tuple_value() {
        let modifiers = ModifierTuple::new(["sunz_corrected", "rayleigh_corrected"]).unwrap();
        let data_id = DataId::new("VIS006")
            .unwrap()
            .with_qualifier("modifiers", modifiers.clone())
            .unwrap();
        assert_eq!(
            data_id.qualifier("modifiers"),
            Some(&DataValue::Modifiers(modifiers))
        );
    }

    #[test]
    fn best_match_prefers_smallest_resolution_when_unspecified() {
        let coarse = DataId::new("solar_zenith_angle")
            .unwrap()
            .with_qualifier("resolution", 1000.0)
            .unwrap();
        let fine = DataId::new("solar_zenith_angle")
            .unwrap()
            .with_qualifier("resolution", 250.0)
            .unwrap();

        let query = DataQuery::named("solar_zenith_angle").unwrap();
        assert_eq!(query.best_match([&coarse, &fine]).unwrap(), &fine);
    }

    #[test]
    fn best_match_prefers_calibration_priority_when_unspecified() {
        let counts = DataId::new("cheese_shops")
            .unwrap()
            .with_qualifier("calibration", "counts")
            .unwrap();
        let reflectance = DataId::new("cheese_shops")
            .unwrap()
            .with_qualifier("calibration", "reflectance")
            .unwrap();

        let query = DataQuery::named("cheese_shops").unwrap();
        assert_eq!(
            query.best_match([&counts, &reflectance]).unwrap(),
            &reflectance
        );
    }

    #[test]
    fn best_match_prefers_unmodified_dataset_when_unspecified() {
        let unmodified = DataId::new("VIS006")
            .unwrap()
            .with_qualifier(
                "modifiers",
                ModifierTuple::new(Vec::<String>::new()).unwrap(),
            )
            .unwrap();
        let modified = DataId::new("VIS006")
            .unwrap()
            .with_qualifier("modifiers", ModifierTuple::new(["sunz_corrected"]).unwrap())
            .unwrap();

        let query = DataQuery::named("VIS006").unwrap();
        assert_eq!(
            query.best_match([&modified, &unmodified]).unwrap(),
            &unmodified
        );
    }

    #[test]
    fn sort_data_ids_uses_wavelength_distance_and_other_metadata() {
        let hrv = DataId::new("HRV")
            .unwrap()
            .with_qualifier(
                "wavelength",
                WavelengthRange::micrometers(0.5, 0.7, 0.9).unwrap(),
            )
            .unwrap()
            .with_qualifier("resolution", 1000.0)
            .unwrap()
            .with_qualifier("calibration", "reflectance")
            .unwrap()
            .with_qualifier(
                "modifiers",
                ModifierTuple::new(Vec::<String>::new()).unwrap(),
            )
            .unwrap();
        let vis008 = DataId::new("VIS008")
            .unwrap()
            .with_qualifier(
                "wavelength",
                WavelengthRange::micrometers(0.74, 0.81, 0.88).unwrap(),
            )
            .unwrap()
            .with_qualifier("resolution", 3000.0)
            .unwrap()
            .with_qualifier("calibration", "reflectance")
            .unwrap()
            .with_qualifier(
                "modifiers",
                ModifierTuple::new(Vec::<String>::new()).unwrap(),
            )
            .unwrap();

        let query = DataQuery::new().with_filter("wavelength", 0.8).unwrap();
        assert_eq!(query.best_match([&vis008, &hrv]).unwrap(), &hrv);
    }

    #[test]
    fn best_match_reports_ambiguity_for_equal_scores() {
        let left = DataId::new("dup")
            .unwrap()
            .with_qualifier("resolution", 1000.0)
            .unwrap()
            .with_qualifier("polarization", "H")
            .unwrap();
        let right = DataId::new("dup")
            .unwrap()
            .with_qualifier("resolution", 1000.0)
            .unwrap()
            .with_qualifier("polarization", "V")
            .unwrap();

        let err = DataQuery::named("dup")
            .unwrap()
            .with_filter("polarization", QueryValue::Any)
            .unwrap()
            .best_match([&left, &right])
            .unwrap_err();
        assert!(matches!(err, RustySatError::Ambiguous { .. }));
    }

    #[test]
    fn satpy_compat_filtering_by_name_ignores_missing_extra_query_keys() {
        let composite_id = DataId::new("natural_color").unwrap();
        let query = DataQuery::named("natural_color")
            .unwrap()
            .with_filter("resolution", 250.0)
            .unwrap();

        assert_eq!(query.filter_data_ids([&composite_id]), vec![&composite_id]);
    }

    #[test]
    fn satpy_compat_query_without_shared_keys_does_not_match() {
        let static_image = DataId::new("static_image").unwrap();
        let query = DataQuery::new()
            .with_filter("wavelength", 0.22)
            .unwrap()
            .with_filter("modifiers", ModifierTuple::new(["mod1"]).unwrap())
            .unwrap();

        assert!(query.filter_data_ids([&static_image]).is_empty());
    }

    #[test]
    fn satpy_compat_seviri_hrv_priority_over_vis008_for_point_eight_micrometers() {
        let candidates = seviri_visible_candidates();
        let query = DataQuery::new().with_filter("wavelength", 0.8).unwrap();
        let best = query.best_match(candidates.iter()).unwrap();

        assert_eq!(best.name(), "HRV");
        assert_eq!(
            best.qualifier("calibration"),
            Some(&DataValue::Text("reflectance".to_string()))
        );
    }

    fn seviri_visible_candidates() -> Vec<DataId> {
        let mut candidates = Vec::new();
        for (name, wavelength, resolution) in [
            ("HRV", (0.5, 0.7, 0.9), 1000.134348869),
            ("VIS006", (0.56, 0.635, 0.71), 3000.403165817),
            ("VIS008", (0.74, 0.81, 0.88), 3000.403165817),
        ] {
            for calibration in ["reflectance", "radiance", "counts"] {
                candidates.push(
                    DataId::new(name)
                        .unwrap()
                        .with_qualifier(
                            "wavelength",
                            WavelengthRange::new(wavelength.0, wavelength.1, wavelength.2, "um")
                                .unwrap(),
                        )
                        .unwrap()
                        .with_qualifier("resolution", resolution)
                        .unwrap()
                        .with_qualifier("calibration", calibration)
                        .unwrap()
                        .with_qualifier(
                            "modifiers",
                            ModifierTuple::new(Vec::<String>::new()).unwrap(),
                        )
                        .unwrap(),
                );
            }
        }
        candidates
    }
}
