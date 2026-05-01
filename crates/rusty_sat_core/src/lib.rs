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
}

impl Display for RustySatError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unsupported { feature } => write!(f, "unsupported feature: {feature}"),
            Self::InvalidInput { message } => write!(f, "invalid input: {message}"),
            Self::NotFound { item } => write!(f, "not found: {item}"),
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
        if let Some(name) = &self.name {
            if data_id.name() != name {
                return false;
            }
        }
        self.filters
            .iter()
            .all(|(key, query_value)| match data_id.qualifier(key) {
                Some(value) => value.matches_query_value(query_value),
                None => matches!(query_value, QueryValue::Any),
            })
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

#[derive(Debug, Default)]
pub struct Scene {
    datasets: BTreeMap<DataId, Dataset>,
    wishlist: BTreeSet<DataId>,
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
        self.datasets.insert(id, dataset);
    }

    pub fn get(&self, id: &DataId) -> Option<&Dataset> {
        self.datasets.get(id)
    }

    pub fn wishlist(&self) -> &BTreeSet<DataId> {
        &self.wishlist
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
}
