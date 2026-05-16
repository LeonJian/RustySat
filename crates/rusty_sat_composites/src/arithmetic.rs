//! Arithmetic composite foundations.
//!
//! Reference behavior inspected before implementation:
//! - `satpy/satpy/composites/arithmetic.py`
//! - `satpy/satpy/composites/core.py` for shape matching before composition.
//!
//! Satpy's arithmetic compositors operate on matched xarray projectables and
//! preserve combined metadata. This first Rust slice focuses on exact numeric
//! operations over matching runtime-typed arrays, mask propagation, and owned
//! variants that mutate the consumed left-hand buffer in place.

use crate::common::{require_two_arrays, require_two_owned_arrays, ArrayInfo};
use crate::Compositor;
use rusty_sat_core::{
    AnyDataArray, DataArray, DataId, Dataset, MetadataValue, Result, RustySatError, ValidityMask,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArithmeticOperation {
    Difference,
    Ratio,
    Sum,
    NormalizedDifference,
}

impl ArithmeticOperation {
    pub fn apply(self, left: f64, right: f64) -> f64 {
        match self {
            Self::Difference => left - right,
            Self::Ratio => left / right,
            Self::Sum => left + right,
            Self::NormalizedDifference => (left - right) / (left + right),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Difference => "difference",
            Self::Ratio => "ratio",
            Self::Sum => "sum",
            Self::NormalizedDifference => "normalized_difference",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArithmeticCompositor {
    name: String,
    operation: ArithmeticOperation,
}

impl ArithmeticCompositor {
    pub fn new(name: impl Into<String>, operation: ArithmeticOperation) -> Result<Self> {
        let name = name.into();
        if name.trim().is_empty() {
            return Err(RustySatError::invalid_input(
                "arithmetic compositor name cannot be empty",
            ));
        }
        Ok(Self { name, operation })
    }

    pub fn difference(name: impl Into<String>) -> Result<Self> {
        Self::new(name, ArithmeticOperation::Difference)
    }

    pub fn ratio(name: impl Into<String>) -> Result<Self> {
        Self::new(name, ArithmeticOperation::Ratio)
    }

    pub fn sum(name: impl Into<String>) -> Result<Self> {
        Self::new(name, ArithmeticOperation::Sum)
    }

    pub fn normalized_difference(name: impl Into<String>) -> Result<Self> {
        Self::new(name, ArithmeticOperation::NormalizedDifference)
    }

    pub fn operation(&self) -> ArithmeticOperation {
        self.operation
    }

    fn compose_arithmetic(&self, inputs: &[Dataset]) -> Result<Dataset> {
        let [left, right] = require_two_arrays(inputs)?;
        let array_info = require_matching_arrays(left, right)?;
        let left_values = left.values_as_f64();
        let right_values = right.values_as_f64();
        let values = left_values
            .into_iter()
            .zip(right_values)
            .map(|(left, right)| self.operation.apply(left, right))
            .collect::<Vec<_>>();
        let left_metadata = extract_composite_metadata(inputs[0].attrs());
        self.finish_dataset(
            array_info,
            values,
            build_binary_mask(left.mask(), right.mask()),
            &left_metadata,
        )
    }

    pub fn compose_owned(self, inputs: Vec<Dataset>) -> Result<Dataset> {
        let left_metadata = extract_composite_metadata(inputs[0].attrs());
        let [left, right] = require_two_owned_arrays(inputs)?;
        let array_info = require_matching_arrays(&left, &right)?;
        let (mut values, left_mask) = left.into_f64_values_and_mask();
        let (right_values, right_mask) = right.into_f64_values_and_mask();
        for (left, right) in values.iter_mut().zip(right_values) {
            *left = self.operation.apply(*left, right);
        }
        self.finish_dataset(
            array_info,
            values,
            build_binary_mask(left_mask.as_ref(), right_mask.as_ref()),
            &left_metadata,
        )
    }

    fn finish_dataset(
        &self,
        array_info: ArrayInfo,
        values: Vec<f64>,
        mask: Option<ValidityMask>,
        metadata: &Vec<(String, MetadataValue)>,
    ) -> Result<Dataset> {
        let mut array =
            DataArray::<f64>::from_vec_named(array_info.shape, array_info.dims, values)?;
        for (name, coordinate) in array_info.coords {
            array.set_coordinate(name, coordinate)?;
        }
        if let Some(mask) = mask {
            array.set_mask(mask)?;
        }
        let mut dataset = Dataset::new(DataId::new(self.name.clone())?).with_array(array);
        dataset.insert_attr("operation", MetadataValue::string(self.operation.name()))?;
        for (key, value) in metadata {
            dataset.insert_attr(key, value.clone())?;
        }
        Ok(dataset)
    }
}

impl Compositor for ArithmeticCompositor {
    fn name(&self) -> &str {
        &self.name
    }

    fn compose(&self, inputs: &[Dataset]) -> Result<Dataset> {
        self.compose_arithmetic(inputs)
    }
}

fn require_matching_arrays(left: &AnyDataArray, right: &AnyDataArray) -> Result<ArrayInfo> {
    if left.shape() != right.shape() {
        return Err(RustySatError::invalid_input(format!(
            "arithmetic compositor input shapes do not match: {:?} != {:?}",
            left.shape(),
            right.shape()
        )));
    }
    if left.dims() != right.dims() {
        return Err(RustySatError::invalid_input(format!(
            "arithmetic compositor input dimensions do not match: {:?} != {:?}",
            left.dims(),
            right.dims()
        )));
    }
    Ok(ArrayInfo {
        shape: left.shape().to_vec(),
        dims: left.dims().to_vec(),
        coords: left.coords().clone(),
    })
}

const PROPAGATED_METADATA_KEYS: &[&str] = &["units", "standard_name", "ancillary_variables"];

fn extract_composite_metadata(
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

fn build_binary_mask(
    left_mask: Option<&ValidityMask>,
    right_mask: Option<&ValidityMask>,
) -> Option<ValidityMask> {
    let (left_mask, right_mask) = match (left_mask, right_mask) {
        (None, None) => return None,
        masks => masks,
    };
    let len = left_mask
        .map(ValidityMask::len)
        .or_else(|| right_mask.map(ValidityMask::len))
        .expect("at least one mask exists");
    let mut output = ValidityMask::all_valid(len);
    for index in 0..len {
        let masked = left_mask
            .and_then(|mask| mask.is_masked(index))
            .unwrap_or(false)
            || right_mask
                .and_then(|mask| mask.is_masked(index))
                .unwrap_or(false);
        if masked {
            output.set_masked(index, true);
        }
    }
    Some(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rusty_sat_core::{Coordinate, DataType, NumericElement};

    #[test]
    fn arithmetic_difference_matches_satpy_binary_shape() -> Result<()> {
        let compositor = ArithmeticCompositor::difference("diff")?;
        let inputs = vec![
            dataset(
                "left",
                DataArray::<u16>::from_vec_named([1, 3], ["y", "x"], vec![10, 20, 30])?
                    .with_coordinate("x", Coordinate::axis("x", vec![1.0, 2.0, 3.0])?)?,
            ),
            dataset(
                "right",
                DataArray::<f32>::from_vec_named([1, 3], ["y", "x"], vec![1.0, 2.0, 3.0])?,
            ),
        ];

        let output = compositor.compose(&inputs)?;
        let array = output.array().expect("arithmetic output array");

        assert_eq!(output.id().name(), "diff");
        assert_eq!(array.dtype(), DataType::F64);
        assert_eq!(array.shape(), &[1, 3]);
        assert_eq!(array.dims(), &["y", "x"]);
        assert_eq!(array.values_as_f64(), vec![9.0, 18.0, 27.0]);
        assert!(array.coord("x").is_some());
        assert_eq!(
            output.attr("operation").and_then(MetadataValue::as_str),
            Some("difference")
        );
        Ok(())
    }

    #[test]
    fn arithmetic_supports_ratio_sum_and_normalized_difference() -> Result<()> {
        let left = dataset(
            "left",
            DataArray::<f64>::from_vec_named([1, 2], ["y", "x"], vec![6.0, 8.0])?,
        );
        let right = dataset(
            "right",
            DataArray::<f64>::from_vec_named([1, 2], ["y", "x"], vec![2.0, 4.0])?,
        );

        assert_eq!(
            ArithmeticCompositor::ratio("ratio")?
                .compose(&[left.clone(), right.clone()])?
                .array()
                .unwrap()
                .values_as_f64(),
            vec![3.0, 2.0]
        );
        assert_eq!(
            ArithmeticCompositor::sum("sum")?
                .compose(&[left.clone(), right.clone()])?
                .array()
                .unwrap()
                .values_as_f64(),
            vec![8.0, 12.0]
        );
        assert_eq!(
            ArithmeticCompositor::normalized_difference("nd")?
                .compose(&[left, right])?
                .array()
                .unwrap()
                .values_as_f64(),
            vec![0.5, 1.0 / 3.0]
        );
        Ok(())
    }

    #[test]
    fn arithmetic_propagates_binary_masks() -> Result<()> {
        let compositor = ArithmeticCompositor::sum("sum")?;
        let inputs = vec![
            dataset(
                "left",
                DataArray::<f64>::from_vec_named([1, 3], ["y", "x"], vec![1.0, 2.0, 3.0])?
                    .with_mask(ValidityMask::from_masked_flags([false, true, false]))?,
            ),
            dataset(
                "right",
                DataArray::<f64>::from_vec_named([1, 3], ["y", "x"], vec![4.0, 5.0, 6.0])?
                    .with_mask(ValidityMask::from_masked_flags([false, false, true]))?,
            ),
        ];

        let output = compositor.compose(&inputs)?;
        let mask = output.array().and_then(AnyDataArray::mask).unwrap();

        assert_eq!(mask.masked_count(), 2);
        assert_eq!(mask.is_masked(0), Some(false));
        assert_eq!(mask.is_masked(1), Some(true));
        assert_eq!(mask.is_masked(2), Some(true));
        Ok(())
    }

    #[test]
    fn arithmetic_owned_consumes_and_mutates_left_buffer() -> Result<()> {
        let inputs = vec![
            dataset(
                "left",
                DataArray::<f64>::from_vec_named([1, 2], ["y", "x"], vec![8.0, 9.0])?,
            ),
            dataset(
                "right",
                DataArray::<i16>::from_vec_named([1, 2], ["y", "x"], vec![2, 3])?,
            ),
        ];

        let output = ArithmeticCompositor::ratio("ratio")?.compose_owned(inputs)?;
        let array = output.array().expect("owned arithmetic output array");

        assert_eq!(array.values_as_f64(), vec![4.0, 3.0]);
        assert_eq!(
            output.attr("operation").and_then(MetadataValue::as_str),
            Some("ratio")
        );
        Ok(())
    }

    #[test]
    fn arithmetic_rejects_mismatched_inputs() -> Result<()> {
        let compositor = ArithmeticCompositor::difference("diff")?;
        let left = dataset(
            "left",
            DataArray::<f64>::from_vec_named([1, 2], ["y", "x"], vec![1.0, 2.0])?,
        );
        let right = dataset(
            "right",
            DataArray::<f64>::from_vec_named([2], ["x"], vec![1.0, 2.0])?,
        );

        assert!(compositor.compose(std::slice::from_ref(&left)).is_err());
        assert!(compositor.compose(&[left, right]).is_err());
        Ok(())
    }

    #[test]
    fn arithmetic_rejects_dimension_name_mismatch() -> Result<()> {
        let compositor = ArithmeticCompositor::difference("diff")?;
        let left = dataset(
            "left",
            DataArray::<f64>::from_vec_named([1, 2], ["y", "x"], vec![1.0, 2.0])?,
        );
        let right = dataset(
            "right",
            DataArray::<f64>::from_vec_named([1, 2], ["bands", "x"], vec![1.0, 2.0])?,
        );

        assert!(compositor.compose(&[left, right]).is_err());
        Ok(())
    }

    #[test]
    fn arithmetic_rejects_dataset_without_array() -> Result<()> {
        let compositor = ArithmeticCompositor::difference("diff")?;
        let empty = Dataset::new(DataId::new("empty")?);
        let right = dataset(
            "right",
            DataArray::<f64>::from_vec_named([1, 2], ["y", "x"], vec![1.0, 2.0])?,
        );

        assert!(compositor.compose(&[empty, right]).is_err());
        Ok(())
    }

    #[test]
    fn arithmetic_produces_no_mask_when_neither_input_has_one() -> Result<()> {
        let compositor = ArithmeticCompositor::sum("sum")?;
        let inputs = vec![
            dataset(
                "left",
                DataArray::<f64>::from_vec_named([1, 2], ["y", "x"], vec![1.0, 2.0])?,
            ),
            dataset(
                "right",
                DataArray::<f64>::from_vec_named([1, 2], ["y", "x"], vec![3.0, 4.0])?,
            ),
        ];

        let output = compositor.compose(&inputs)?;
        assert!(output.array().and_then(|a| a.mask()).is_none());
        Ok(())
    }

    #[test]
    fn arithmetic_owned_propagates_masks() -> Result<()> {
        let inputs = vec![
            dataset(
                "left",
                DataArray::<f64>::from_vec_named([1, 3], ["y", "x"], vec![1.0, 2.0, 3.0])?
                    .with_mask(ValidityMask::from_masked_flags([true, false, false]))?,
            ),
            dataset(
                "right",
                DataArray::<u16>::from_vec_named([1, 3], ["y", "x"], vec![4, 5, 6])?
                    .with_mask(ValidityMask::from_masked_flags([false, false, true]))?,
            ),
        ];

        let output = ArithmeticCompositor::sum("sum")?.compose_owned(inputs)?;
        let mask = output.array().and_then(AnyDataArray::mask).unwrap();

        assert_eq!(mask.masked_count(), 2);
        assert_eq!(mask.is_masked(0), Some(true));
        assert_eq!(mask.is_masked(1), Some(false));
        assert_eq!(mask.is_masked(2), Some(true));
        Ok(())
    }

    #[test]
    fn arithmetic_propagates_metadata_from_left_input() -> Result<()> {
        let compositor = ArithmeticCompositor::difference("diff")?;
        let mut left = dataset(
            "left",
            DataArray::<f64>::from_vec_named([1, 2], ["y", "x"], vec![1.0, 2.0])?,
        );
        left.insert_attr("units", MetadataValue::string("K"))?;
        left.insert_attr(
            "standard_name",
            MetadataValue::string("brightness_temperature"),
        )?;
        left.insert_attr("irrelevant", MetadataValue::string("should not propagate"))?;
        let right = dataset(
            "right",
            DataArray::<f64>::from_vec_named([1, 2], ["y", "x"], vec![1.0, 2.0])?,
        );

        let output = compositor.compose(&[left, right])?;

        assert_eq!(
            output.attr("units").and_then(MetadataValue::as_str),
            Some("K")
        );
        assert_eq!(
            output.attr("standard_name").and_then(MetadataValue::as_str),
            Some("brightness_temperature")
        );
        assert!(output.attr("irrelevant").is_none());
        Ok(())
    }

    fn dataset<T: NumericElement>(name: &str, array: DataArray<T>) -> Dataset
    where
        AnyDataArray: From<DataArray<T>>,
    {
        Dataset::new(DataId::new(name).expect("valid test data id")).with_array(array)
    }
}
