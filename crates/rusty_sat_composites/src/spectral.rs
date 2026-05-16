//! Spectral composite foundations.
//!
//! Reference behavior inspected before implementation:
//! - `satpy/satpy/composites/spectral.py`
//! - `satpy/satpy/composites/core.py` for matched projectable handling.
//!
//! Satpy's `SpectralBlender` computes weighted channel blends for corrected
//! green and related true-color products. This first Rust slice adds a
//! weighted single-band blender plus a band replacement compositor that can
//! patch one band in a band-major composite without rebuilding unrelated
//! bands when the caller can give up ownership.

use crate::common::{
    extract_composite_metadata, missing_array_error, require_two_arrays, require_two_owned_arrays,
    ArrayInfo,
};
use crate::Compositor;
use rusty_sat_core::{
    AnyDataArray, DataArray, DataId, Dataset, MetadataValue, Result, RustySatError, ValidityMask,
};

#[derive(Debug, Clone, PartialEq)]
pub struct SpectralBlender {
    name: String,
    weights: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BandReplacementCompositor {
    name: String,
    band_index: usize,
}

impl SpectralBlender {
    pub fn new(name: impl Into<String>, weights: impl Into<Vec<f64>>) -> Result<Self> {
        let name = name.into();
        if name.trim().is_empty() {
            return Err(RustySatError::invalid_input(
                "spectral blender name cannot be empty",
            ));
        }
        let weights = weights.into();
        validate_weights(&weights)?;
        Ok(Self { name, weights })
    }

    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    fn compose_blend(&self, inputs: &[Dataset]) -> Result<Dataset> {
        let arrays = require_weighted_arrays(inputs, self.weights.len())?;
        let array_info = require_matching_2d_arrays(&arrays)?;
        let value_count = array_info.value_count();
        let mut values = vec![0.0; value_count];
        for (array, weight) in arrays.iter().zip(&self.weights) {
            for (output, value) in values.iter_mut().zip(array.values_as_f64()) {
                *output += *weight * value;
            }
        }
        let metadata = extract_composite_metadata(inputs[0].attrs());
        self.finish_dataset(
            array_info,
            values,
            build_or_mask(arrays.iter().map(|array| array.mask())),
            &metadata,
        )
    }

    pub fn compose_owned(self, inputs: Vec<Dataset>) -> Result<Dataset> {
        let metadata = extract_composite_metadata(inputs[0].attrs());
        let arrays = require_weighted_owned_arrays(inputs, self.weights.len())?;
        let refs = arrays.iter().collect::<Vec<_>>();
        let array_info = require_matching_2d_arrays(&refs)?;
        let mut array_iter = arrays.into_iter();
        let first = array_iter
            .next()
            .expect("weights validation requires at least one input");
        let first_weight = self.weights[0];
        let (mut values, first_mask) = first.into_f64_values_and_mask();
        for value in &mut values {
            *value *= first_weight;
        }
        let mut masks = vec![first_mask];
        for (array, weight) in array_iter.zip(self.weights.iter().skip(1)) {
            let (array_values, mask) = array.into_f64_values_and_mask();
            for (output, value) in values.iter_mut().zip(array_values) {
                *output += *weight * value;
            }
            masks.push(mask);
        }
        self.finish_dataset(
            array_info,
            values,
            build_or_mask(masks.iter().map(Option::as_ref)),
            &metadata,
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
        dataset.insert_attr("operation", MetadataValue::string("spectral_blend"))?;
        dataset.insert_attr(
            "weights",
            MetadataValue::List(
                self.weights
                    .iter()
                    .copied()
                    .map(MetadataValue::float)
                    .collect::<Result<Vec<_>>>()?,
            ),
        )?;
        for (key, value) in metadata {
            dataset.insert_attr(key, value.clone())?;
        }
        Ok(dataset)
    }
}

impl Compositor for SpectralBlender {
    fn name(&self) -> &str {
        &self.name
    }

    fn compose(&self, inputs: &[Dataset]) -> Result<Dataset> {
        self.compose_blend(inputs)
    }
}

impl BandReplacementCompositor {
    pub fn new(name: impl Into<String>, band_index: usize) -> Result<Self> {
        let name = name.into();
        if name.trim().is_empty() {
            return Err(RustySatError::invalid_input(
                "band replacement compositor name cannot be empty",
            ));
        }
        Ok(Self { name, band_index })
    }

    pub fn band_index(&self) -> usize {
        self.band_index
    }

    fn compose_replacement(&self, inputs: &[Dataset]) -> Result<Dataset> {
        let [base, replacement] = require_two_arrays(inputs)?;
        let band_info = require_band_replacement_shapes(base, replacement, self.band_index)?;
        let mut values = base.values_as_f64();
        let replacement_values = replacement.values_as_f64();
        replace_band_values(&mut values, &replacement_values, &band_info);
        let mask = build_replacement_mask(base.mask(), replacement.mask(), &band_info);
        let metadata = extract_composite_metadata(inputs[0].attrs());
        self.finish_dataset(band_info.array_info, values, mask, &metadata)
    }

    pub fn compose_owned(self, inputs: Vec<Dataset>) -> Result<Dataset> {
        let metadata = extract_composite_metadata(inputs[0].attrs());
        let [base, replacement] = require_two_owned_arrays(inputs)?;
        let band_info = require_band_replacement_shapes(&base, &replacement, self.band_index)?;
        let (mut values, base_mask) = base.into_f64_values_and_mask();
        let (replacement_values, replacement_mask) = replacement.into_f64_values_and_mask();
        replace_band_values(&mut values, &replacement_values, &band_info);
        let mask =
            build_replacement_mask(base_mask.as_ref(), replacement_mask.as_ref(), &band_info);
        self.finish_dataset(band_info.array_info, values, mask, &metadata)
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
        dataset.insert_attr("operation", MetadataValue::string("band_replacement"))?;
        dataset.insert_attr("band_index", MetadataValue::Integer(self.band_index as i64))?;
        for (key, value) in metadata {
            dataset.insert_attr(key, value.clone())?;
        }
        Ok(dataset)
    }
}

impl Compositor for BandReplacementCompositor {
    fn name(&self) -> &str {
        &self.name
    }

    fn compose(&self, inputs: &[Dataset]) -> Result<Dataset> {
        self.compose_replacement(inputs)
    }
}

#[derive(Debug, Clone)]
struct BandInfo {
    array_info: ArrayInfo,
    band_index: usize,
    band_size: usize,
}

fn validate_weights(weights: &[f64]) -> Result<()> {
    if weights.is_empty() {
        return Err(RustySatError::invalid_input(
            "spectral blender requires at least one weight",
        ));
    }
    if weights.iter().any(|weight| !weight.is_finite()) {
        return Err(RustySatError::invalid_input(
            "spectral blender weights must be finite",
        ));
    }
    Ok(())
}

fn require_weighted_arrays(inputs: &[Dataset], weight_count: usize) -> Result<Vec<&AnyDataArray>> {
    if inputs.len() != weight_count {
        return Err(RustySatError::invalid_input(format!(
            "spectral blender requires {weight_count} input datasets, got {}",
            inputs.len()
        )));
    }
    inputs
        .iter()
        .map(|dataset| {
            dataset
                .array()
                .ok_or_else(|| missing_array_error(dataset.id().name()))
        })
        .collect()
}

fn require_weighted_owned_arrays(
    inputs: Vec<Dataset>,
    weight_count: usize,
) -> Result<Vec<AnyDataArray>> {
    if inputs.len() != weight_count {
        return Err(RustySatError::invalid_input(format!(
            "spectral blender requires {weight_count} input datasets, got {}",
            inputs.len()
        )));
    }
    inputs
        .into_iter()
        .map(|dataset| {
            let name = dataset.id().name().to_string();
            dataset
                .into_array()
                .ok_or_else(|| missing_array_error(&name))
        })
        .collect()
}

fn require_matching_2d_arrays(arrays: &[&AnyDataArray]) -> Result<ArrayInfo> {
    let first = arrays
        .first()
        .ok_or_else(|| RustySatError::invalid_input("at least one array is required"))?;
    let first_shape = require_2d_yx(first)?;
    for array in arrays.iter().skip(1) {
        let shape = require_2d_yx(array)?;
        if shape != first_shape || array.dims() != first.dims() {
            return Err(RustySatError::invalid_input(
                "spectral blender inputs must have matching y/x shapes and dimensions",
            ));
        }
    }
    Ok(ArrayInfo {
        shape: first.shape().to_vec(),
        dims: first.dims().to_vec(),
        coords: first.coords().clone(),
    })
}

fn require_2d_yx(array: &AnyDataArray) -> Result<(usize, usize)> {
    if array.ndim() != 2 {
        return Err(RustySatError::invalid_input(format!(
            "spectral input must be a 2D y/x array, got shape {:?}",
            array.shape()
        )));
    }
    array.shape_yx()
}

fn require_band_replacement_shapes(
    base: &AnyDataArray,
    replacement: &AnyDataArray,
    band_index: usize,
) -> Result<BandInfo> {
    if base.ndim() != 3 || base.dims().first().map(String::as_str) != Some("bands") {
        return Err(RustySatError::invalid_input(format!(
            "band replacement base must be a bands/y/x array, got shape {:?} with dims {:?}",
            base.shape(),
            base.dims()
        )));
    }
    let replacement_shape = require_2d_yx(replacement)?;
    if base.shape()[1] != replacement_shape.0 || base.shape()[2] != replacement_shape.1 {
        return Err(RustySatError::invalid_input(format!(
            "replacement shape {:?} does not match base y/x shape {:?}",
            replacement.shape(),
            &base.shape()[1..]
        )));
    }
    if band_index >= base.shape()[0] {
        return Err(RustySatError::invalid_input(format!(
            "band index {band_index} is outside base band count {}",
            base.shape()[0]
        )));
    }
    Ok(BandInfo {
        array_info: ArrayInfo {
            shape: base.shape().to_vec(),
            dims: base.dims().to_vec(),
            coords: base.coords().clone(),
        },
        band_index,
        band_size: replacement_shape.0 * replacement_shape.1,
    })
}

fn replace_band_values(values: &mut [f64], replacement_values: &[f64], band_info: &BandInfo) {
    let start = band_info.band_index * band_info.band_size;
    let end = start + band_info.band_size;
    values[start..end].copy_from_slice(replacement_values);
}

fn build_or_mask<'a>(
    masks: impl IntoIterator<Item = Option<&'a ValidityMask>>,
) -> Option<ValidityMask> {
    let masks = masks.into_iter().collect::<Vec<_>>();
    if masks.iter().all(Option::is_none) {
        return None;
    }
    let len = masks
        .iter()
        .find_map(|mask| mask.map(ValidityMask::len))
        .expect("at least one mask exists");
    let mut output = ValidityMask::all_valid(len);
    for index in 0..len {
        if masks
            .iter()
            .any(|mask| mask.and_then(|mask| mask.is_masked(index)).unwrap_or(false))
        {
            output.set_masked(index, true);
        }
    }
    Some(output)
}

fn build_replacement_mask(
    base_mask: Option<&ValidityMask>,
    replacement_mask: Option<&ValidityMask>,
    band_info: &BandInfo,
) -> Option<ValidityMask> {
    if base_mask.is_none() && replacement_mask.is_none() {
        return None;
    }
    let len = band_info.array_info.value_count();
    let mut output = ValidityMask::all_valid(len);
    if let Some(base_mask) = base_mask {
        for index in 0..len {
            if base_mask.is_masked(index).unwrap_or(false) {
                output.set_masked(index, true);
            }
        }
    }
    if let Some(replacement_mask) = replacement_mask {
        let start = band_info.band_index * band_info.band_size;
        for pixel_index in 0..band_info.band_size {
            output.set_masked(
                start + pixel_index,
                replacement_mask.is_masked(pixel_index).unwrap_or(false),
            );
        }
    }
    Some(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rusty_sat_core::{Coordinate, DataType, NumericElement};

    #[test]
    fn spectral_blender_computes_weighted_channel() -> Result<()> {
        let blender = SpectralBlender::new("corrected_green", vec![0.63, 0.29, 0.08])?;
        let inputs = vec![
            dataset(
                "green",
                DataArray::<f32>::from_vec_named([1, 2], ["y", "x"], vec![10.0, 20.0])?
                    .with_coordinate("x", Coordinate::axis("x", vec![0.5, 1.5])?)?,
            ),
            dataset(
                "red",
                DataArray::<u16>::from_vec_named([1, 2], ["y", "x"], vec![30, 40])?,
            ),
            dataset(
                "nir",
                DataArray::<f64>::from_vec_named([1, 2], ["y", "x"], vec![50.0, 60.0])?,
            ),
        ];

        let output = blender.compose(&inputs)?;
        let array = output.array().expect("spectral output");

        assert_eq!(output.id().name(), "corrected_green");
        assert_eq!(array.dtype(), DataType::F64);
        assert_eq!(array.shape(), &[1, 2]);
        assert_eq!(array.dims(), &["y", "x"]);
        assert_values_close(&array.values_as_f64(), &[19.0, 29.0], 1e-12);
        assert!(array.coord("x").is_some());
        assert_eq!(
            output.attr("operation").and_then(MetadataValue::as_str),
            Some("spectral_blend")
        );
        Ok(())
    }

    #[test]
    fn spectral_blender_owned_reuses_first_input_values() -> Result<()> {
        let inputs = vec![
            dataset(
                "green",
                DataArray::<f64>::from_vec_named([1, 2], ["y", "x"], vec![10.0, 20.0])?,
            ),
            dataset(
                "nir",
                DataArray::<i16>::from_vec_named([1, 2], ["y", "x"], vec![30, 40])?,
            ),
        ];

        let output =
            SpectralBlender::new("hybrid_green", vec![0.85, 0.15])?.compose_owned(inputs)?;

        assert_eq!(output.array().unwrap().values_as_f64(), vec![13.0, 23.0]);
        Ok(())
    }

    #[test]
    fn spectral_blender_propagates_any_input_mask() -> Result<()> {
        let inputs = vec![
            dataset(
                "a",
                DataArray::<f64>::from_vec_named([1, 3], ["y", "x"], vec![1.0, 2.0, 3.0])?
                    .with_mask(ValidityMask::from_masked_flags([false, true, false]))?,
            ),
            dataset(
                "b",
                DataArray::<f64>::from_vec_named([1, 3], ["y", "x"], vec![4.0, 5.0, 6.0])?
                    .with_mask(ValidityMask::from_masked_flags([false, false, true]))?,
            ),
        ];

        let output = SpectralBlender::new("blend", vec![0.5, 0.5])?.compose(&inputs)?;
        let mask = output.array().and_then(AnyDataArray::mask).unwrap();

        assert_eq!(mask.masked_count(), 2);
        assert_eq!(mask.is_masked(0), Some(false));
        assert_eq!(mask.is_masked(1), Some(true));
        assert_eq!(mask.is_masked(2), Some(true));
        Ok(())
    }

    #[test]
    fn band_replacement_replaces_requested_band() -> Result<()> {
        let base = dataset(
            "rgb",
            DataArray::<f64>::from_vec_named(
                [3, 1, 2],
                ["bands", "y", "x"],
                vec![1.0, 2.0, 10.0, 20.0, 100.0, 200.0],
            )?
            .with_coordinate("bands", Coordinate::axis("bands", vec![0.0, 1.0, 2.0])?)?,
        );
        let replacement = dataset(
            "corrected_green",
            DataArray::<u16>::from_vec_named([1, 2], ["y", "x"], vec![30, 40])?,
        );

        let output =
            BandReplacementCompositor::new("rgb_corrected", 1)?.compose(&[base, replacement])?;
        let array = output.array().expect("band replacement output");

        assert_eq!(array.shape(), &[3, 1, 2]);
        assert_eq!(
            array.values_as_f64(),
            vec![1.0, 2.0, 30.0, 40.0, 100.0, 200.0]
        );
        assert!(array.coord("bands").is_some());
        assert_eq!(
            output.attr("operation").and_then(MetadataValue::as_str),
            Some("band_replacement")
        );
        Ok(())
    }

    #[test]
    fn band_replacement_owned_propagates_replacement_mask_only_to_replaced_band() -> Result<()> {
        let base = dataset(
            "rgb",
            DataArray::<f64>::from_vec_named(
                [3, 1, 2],
                ["bands", "y", "x"],
                vec![1.0, 2.0, 10.0, 20.0, 100.0, 200.0],
            )?
            .with_mask(ValidityMask::from_masked_flags([
                false, false, false, false, false, true,
            ]))?,
        );
        let replacement = dataset(
            "corrected_green",
            DataArray::<f64>::from_vec_named([1, 2], ["y", "x"], vec![30.0, 40.0])?
                .with_mask(ValidityMask::from_masked_flags([true, false]))?,
        );

        let output = BandReplacementCompositor::new("rgb_corrected", 1)?
            .compose_owned(vec![base, replacement])?;
        let mask = output.array().and_then(AnyDataArray::mask).unwrap();

        assert_eq!(output.array().unwrap().values_as_f64()[2..4], [30.0, 40.0]);
        assert_eq!(mask.is_masked(0), Some(false));
        assert_eq!(mask.is_masked(2), Some(true));
        assert_eq!(mask.is_masked(3), Some(false));
        assert_eq!(mask.is_masked(5), Some(true));
        Ok(())
    }

    #[test]
    fn spectral_compositors_reject_invalid_inputs() -> Result<()> {
        assert!(SpectralBlender::new("bad", Vec::<f64>::new()).is_err());
        assert!(SpectralBlender::new("bad", vec![1.0, f64::NAN]).is_err());

        let left = dataset(
            "left",
            DataArray::<f64>::from_vec_named([1, 2], ["y", "x"], vec![1.0, 2.0])?,
        );
        let right = dataset(
            "right",
            DataArray::<f64>::from_vec_named([2], ["x"], vec![1.0, 2.0])?,
        );
        assert!(SpectralBlender::new("blend", vec![0.5, 0.5])?
            .compose(&[left.clone(), right])
            .is_err());
        assert!(BandReplacementCompositor::new("replace", 5)?
            .compose(&[left.clone(), left])
            .is_err());
        Ok(())
    }

    #[test]
    fn spectral_blender_works_with_single_weight() -> Result<()> {
        let inputs = vec![dataset(
            "green",
            DataArray::<f64>::from_vec_named([1, 2], ["y", "x"], vec![10.0, 20.0])?,
        )];

        let output = SpectralBlender::new("passthrough", vec![1.0])?.compose(&inputs)?;

        assert_eq!(output.array().unwrap().values_as_f64(), vec![10.0, 20.0]);
        Ok(())
    }

    #[test]
    fn band_replacement_no_masks_produces_no_output_mask() -> Result<()> {
        let base = dataset(
            "rgb",
            DataArray::<f64>::from_vec_named(
                [3, 1, 2],
                ["bands", "y", "x"],
                vec![1.0, 2.0, 10.0, 20.0, 100.0, 200.0],
            )?,
        );
        let replacement = dataset(
            "corrected",
            DataArray::<f64>::from_vec_named([1, 2], ["y", "x"], vec![30.0, 40.0])?,
        );

        let output =
            BandReplacementCompositor::new("rgb_corrected", 1)?.compose(&[base, replacement])?;

        assert!(output.array().and_then(|a| a.mask()).is_none());
        Ok(())
    }

    #[test]
    fn spectral_blender_owned_propagates_masks() -> Result<()> {
        let inputs = vec![
            dataset(
                "a",
                DataArray::<f64>::from_vec_named([1, 3], ["y", "x"], vec![1.0, 2.0, 3.0])?
                    .with_mask(ValidityMask::from_masked_flags([true, false, false]))?,
            ),
            dataset(
                "b",
                DataArray::<u16>::from_vec_named([1, 3], ["y", "x"], vec![4, 5, 6])?
                    .with_mask(ValidityMask::from_masked_flags([false, false, true]))?,
            ),
        ];

        let output = SpectralBlender::new("blend", vec![0.5, 0.5])?.compose_owned(inputs)?;
        let mask = output.array().and_then(AnyDataArray::mask).unwrap();

        assert_eq!(mask.masked_count(), 2);
        assert_eq!(mask.is_masked(0), Some(true));
        assert_eq!(mask.is_masked(1), Some(false));
        assert_eq!(mask.is_masked(2), Some(true));
        Ok(())
    }

    #[test]
    fn spectral_compositors_propagate_metadata_from_first_input() -> Result<()> {
        // Spectral blender
        let mut green = dataset(
            "green",
            DataArray::<f64>::from_vec_named([1, 2], ["y", "x"], vec![10.0, 20.0])?,
        );
        green.insert_attr("units", MetadataValue::string("W m-2 sr-1"))?;
        green.insert_attr("irrelevant", MetadataValue::string("no"))?;
        let nir = dataset(
            "nir",
            DataArray::<f64>::from_vec_named([1, 2], ["y", "x"], vec![5.0, 5.0])?,
        );

        let blend_output = SpectralBlender::new("blend", vec![0.5, 0.5])?.compose(&[green, nir])?;
        assert_eq!(
            blend_output.attr("units").and_then(MetadataValue::as_str),
            Some("W m-2 sr-1")
        );
        assert!(blend_output.attr("irrelevant").is_none());

        // Band replacement
        let mut base = dataset(
            "rgb",
            DataArray::<f64>::from_vec_named(
                [3, 1, 2],
                ["bands", "y", "x"],
                vec![1.0, 2.0, 10.0, 20.0, 100.0, 200.0],
            )?,
        );
        base.insert_attr("standard_name", MetadataValue::string("true_color"))?;
        let replacement = dataset(
            "corrected",
            DataArray::<f64>::from_vec_named([1, 2], ["y", "x"], vec![30.0, 40.0])?,
        );

        let repl_output =
            BandReplacementCompositor::new("rgb_corrected", 1)?.compose(&[base, replacement])?;
        assert_eq!(
            repl_output
                .attr("standard_name")
                .and_then(MetadataValue::as_str),
            Some("true_color")
        );
        Ok(())
    }

    fn dataset<T: NumericElement>(name: &str, array: DataArray<T>) -> Dataset
    where
        AnyDataArray: From<DataArray<T>>,
    {
        Dataset::new(DataId::new(name).expect("valid test data id")).with_array(array)
    }

    fn assert_values_close(left: &[f64], right: &[f64], tolerance: f64) {
        assert_eq!(left.len(), right.len());
        for (left, right) in left.iter().zip(right) {
            assert!((left - right).abs() <= tolerance, "{left} != {right}");
        }
    }
}
