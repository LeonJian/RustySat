//! Shared helpers for compositor implementations.

use rusty_sat_core::{AnyDataArray, Coordinate, Dataset, MetadataValue, Result, RustySatError};
use std::collections::BTreeMap;

#[derive(Debug, Clone)]
pub(crate) struct ArrayInfo {
    pub shape: Vec<usize>,
    pub dims: Vec<String>,
    pub coords: BTreeMap<String, Coordinate>,
}

impl ArrayInfo {
    pub fn value_count(&self) -> usize {
        self.shape.iter().product()
    }
}

const PROPAGATED_METADATA_KEYS: &[&str] = &["units", "standard_name", "ancillary_variables"];

pub(crate) fn extract_composite_metadata(
    attrs: &BTreeMap<String, MetadataValue>,
) -> Vec<(String, MetadataValue)> {
    PROPAGATED_METADATA_KEYS
        .iter()
        .filter_map(|key| {
            attrs
                .get(*key)
                .map(|value| (key.to_string(), value.clone()))
        })
        .collect()
}

pub(crate) fn require_two_arrays(inputs: &[Dataset]) -> Result<[&AnyDataArray; 2]> {
    if inputs.len() != 2 {
        return Err(RustySatError::invalid_input(format!(
            "compositor requires exactly 2 input datasets, got {}",
            inputs.len()
        )));
    }
    let [left, right] = inputs else {
        unreachable!("length checked above");
    };
    Ok([
        left.array()
            .ok_or_else(|| missing_array_error(left.id().name()))?,
        right
            .array()
            .ok_or_else(|| missing_array_error(right.id().name()))?,
    ])
}

pub(crate) fn require_two_owned_arrays(inputs: Vec<Dataset>) -> Result<[AnyDataArray; 2]> {
    if inputs.len() != 2 {
        return Err(RustySatError::invalid_input(format!(
            "compositor requires exactly 2 input datasets, got {}",
            inputs.len()
        )));
    }
    let mut arrays = Vec::with_capacity(2);
    for dataset in inputs {
        let name = dataset.id().name().to_string();
        let array = dataset
            .into_array()
            .ok_or_else(|| missing_array_error(&name))?;
        arrays.push(array);
    }
    arrays
        .try_into()
        .map_err(|_| RustySatError::invalid_input("compositor requires exactly 2 input datasets"))
}

pub(crate) fn missing_array_error(name: &str) -> RustySatError {
    RustySatError::invalid_input(format!("dataset '{name}' has no array data"))
}
