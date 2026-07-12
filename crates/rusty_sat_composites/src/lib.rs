//! Satellite image compositing, spectral blending, arithmetic operations,
//! and enhancement execution.
//!
//! This crate implements the data combination and enhancement layers of the
//! Satpy processing pipeline. All compositors implement the [`Compositor`]
//! trait, which takes one or more input [`Dataset`]s and produces a new one.
//!
//! # Compositors
//!
//! - [`RgbCompositor`] — 3 single-band datasets → `[3, y, x]` RGB dataset
//!   with common-channel masking. Use for true color, false color, etc.
//! - [`SpectralBlender`](spectral::SpectralBlender) — weighted sum of N bands
//!   for corrected-green and similar products.
//! - [`ArithmeticCompositor`](arithmetic::ArithmeticCompositor) — binary ops:
//!   difference, ratio, sum, normalized-difference (NDVI-style).
//! - [`BandReplacementCompositor`](spectral::BandReplacementCompositor) —
//!   in-place band replacement in a band-major composite.
//!
//! # Enhancement
//!
//! - [`EnhancementExecutor`](enhancement::EnhancementExecutor) — safely
//!   executes enhancement operations (`stretch`, `gamma`, `invert`) from
//!   YAML definitions.
//! - [`CompositeRegistryConfig`](config::CompositeRegistryConfig) — parses
//!   Satpy-style YAML composite and enhancement configurations.
//!
//! # Usage
//!
//! ```ignore
//! use rusty_sat_composites::RgbCompositor;
//! let rgb = RgbCompositor::new("true_color")?
//!     .compose_rgb_owned(vec![red_ds, green_ds, blue_ds])?;
//! // rgb has shape [3, height, width], mode = "RGB"
//! ```

pub mod arithmetic;
mod common;
pub mod config;
pub mod enhancement;
pub mod self_sharpened;
pub mod spectral;

pub use arithmetic::{ArithmeticCompositor, ArithmeticOperation};
pub use config::{
    CompositeDefinition, CompositeDependency, CompositeRegistryConfig, EnhancementDefinition,
    EnhancementOperation,
};
pub use enhancement::EnhancementExecutor;
pub use self_sharpened::SelfSharpenedRgb;
pub use spectral::{BandReplacementCompositor, SpectralBlender};

use rusty_sat_core::{
    AnyDataArray, Coordinate, DataArray, DataId, Dataset, MetadataValue, NumericElement, Result,
    RustySatError, ValidityMask,
};

pub trait Compositor {
    fn name(&self) -> &str;

    fn compose(&self, _inputs: &[Dataset]) -> Result<Dataset> {
        Err(RustySatError::unsupported(format!(
            "{} compositor",
            self.name()
        )))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RgbCompositor {
    name: String,
    common_channel_mask: bool,
}

impl RgbCompositor {
    pub fn new(name: impl Into<String>) -> Result<Self> {
        let name = name.into();
        if name.trim().is_empty() {
            return Err(RustySatError::invalid_input(
                "RGB compositor name cannot be empty",
            ));
        }
        Ok(Self {
            name,
            common_channel_mask: true,
        })
    }

    pub fn with_common_channel_mask(mut self, common_channel_mask: bool) -> Self {
        self.common_channel_mask = common_channel_mask;
        self
    }

    fn compose_rgb(&self, inputs: &[Dataset]) -> Result<Dataset> {
        let input_arrays = require_three_arrays(inputs)?;
        let (height, width) = require_matching_yx_shapes(&input_arrays)?;
        let pixel_count = height
            .checked_mul(width)
            .ok_or_else(|| RustySatError::invalid_input("RGB composite shape is too large"))?;
        let mut values = Vec::with_capacity(pixel_count * 3);
        for array in &input_arrays {
            extend_values_as_f32(array, &mut values);
        }

        let mut array =
            DataArray::<f32>::from_vec_named(vec![3, height, width], ["bands", "y", "x"], values)?
                .with_coordinate("bands", Coordinate::axis("bands", vec![0.0, 1.0, 2.0])?)?;
        if let Some(mask) = build_rgb_mask(&input_arrays, pixel_count, self.common_channel_mask) {
            array.set_mask(mask)?;
        }

        let mut dataset = Dataset::new(DataId::new(self.name.clone())?).with_array(array);
        dataset.insert_attr("mode", MetadataValue::string("RGB"))?;
        Ok(dataset)
    }

    pub fn compose_rgb_owned(self, inputs: Vec<Dataset>) -> Result<Dataset> {
        let input_arrays = require_three_owned_arrays(inputs)?;
        let input_refs = [&input_arrays[0], &input_arrays[1], &input_arrays[2]];
        let (height, width) = require_matching_yx_shapes(&input_refs)?;
        let pixel_count = height
            .checked_mul(width)
            .ok_or_else(|| RustySatError::invalid_input("RGB composite shape is too large"))?;
        let mut values = Vec::with_capacity(pixel_count * 3);
        let mut masks = Vec::with_capacity(3);

        for array in input_arrays {
            let mask = array.mask().cloned();
            match array {
                AnyDataArray::F32(da) => values.extend_from_slice(&da.into_values()),
                AnyDataArray::F64(da) => {
                    for v in da.into_values() {
                        values.push(v as f32);
                    }
                }
                AnyDataArray::U8(da) => {
                    for v in da.into_values() {
                        values.push(v as f32);
                    }
                }
                AnyDataArray::U16(da) => {
                    for v in da.into_values() {
                        values.push(v as f32);
                    }
                }
                AnyDataArray::I16(da) => {
                    for v in da.into_values() {
                        values.push(v as f32);
                    }
                }
            }
            masks.push(mask);
        }

        let mut array =
            DataArray::<f32>::from_vec_named(vec![3, height, width], ["bands", "y", "x"], values)?
                .with_coordinate("bands", Coordinate::axis("bands", vec![0.0, 1.0, 2.0])?)?;
        if let Some(mask) =
            build_rgb_mask_from_owned_masks(&masks, pixel_count, self.common_channel_mask)
        {
            array.set_mask(mask)?;
        }

        let mut dataset = Dataset::new(DataId::new(self.name)?).with_array(array);
        dataset.insert_attr("mode", MetadataValue::string("RGB"))?;
        Ok(dataset)
    }
}

impl Compositor for RgbCompositor {
    fn name(&self) -> &str {
        &self.name
    }

    fn compose(&self, inputs: &[Dataset]) -> Result<Dataset> {
        self.compose_rgb(inputs)
    }
}

pub trait Modifier {
    fn name(&self) -> &str;

    fn apply(&self, _input: &Dataset) -> Result<Dataset> {
        Err(RustySatError::unsupported(format!(
            "{} modifier",
            self.name()
        )))
    }
}

fn require_three_arrays(inputs: &[Dataset]) -> Result<[&AnyDataArray; 3]> {
    if inputs.len() != 3 {
        return Err(RustySatError::invalid_input(format!(
            "RGB compositor requires exactly 3 input datasets, got {}",
            inputs.len()
        )));
    }
    let [red, green, blue] = inputs else {
        unreachable!("length checked above");
    };
    Ok([
        red.array()
            .ok_or_else(|| missing_array_error(red.id().name()))?,
        green
            .array()
            .ok_or_else(|| missing_array_error(green.id().name()))?,
        blue.array()
            .ok_or_else(|| missing_array_error(blue.id().name()))?,
    ])
}

fn missing_array_error(name: &str) -> RustySatError {
    RustySatError::invalid_input(format!("dataset '{name}' has no array data"))
}

fn require_three_owned_arrays(inputs: Vec<Dataset>) -> Result<[AnyDataArray; 3]> {
    if inputs.len() != 3 {
        return Err(RustySatError::invalid_input(format!(
            "RGB compositor requires exactly 3 input datasets, got {}",
            inputs.len()
        )));
    }
    let mut arrays = Vec::with_capacity(3);
    for dataset in inputs {
        let name = dataset.id().name().to_string();
        let array = dataset
            .into_array()
            .ok_or_else(|| missing_array_error(&name))?;
        arrays.push(array);
    }
    arrays.try_into().map_err(|_| {
        RustySatError::invalid_input("RGB compositor requires exactly 3 input datasets")
    })
}

fn require_matching_yx_shapes(arrays: &[&AnyDataArray; 3]) -> Result<(usize, usize)> {
    let first_shape = require_single_band_yx(arrays[0])?;
    for array in arrays.iter().skip(1) {
        let shape = require_single_band_yx(array)?;
        if shape != first_shape {
            return Err(RustySatError::invalid_input(format!(
                "RGB compositor input shapes do not match: {:?} != {:?}",
                shape, first_shape
            )));
        }
    }
    Ok(first_shape)
}

fn require_single_band_yx(array: &AnyDataArray) -> Result<(usize, usize)> {
    if array.ndim() != 2 {
        return Err(RustySatError::invalid_input(format!(
            "RGB compositor inputs must be 2D single-band arrays, got shape {:?}",
            array.shape()
        )));
    }
    array.shape_yx()
}

fn extend_values_as_f32(array: &AnyDataArray, values: &mut Vec<f32>) {
    match array {
        AnyDataArray::F32(array) => values.extend_from_slice(array.values()),
        AnyDataArray::F64(array) => extend_numeric_to_f32(array, values),
        AnyDataArray::U8(array) => extend_numeric_to_f32(array, values),
        AnyDataArray::U16(array) => extend_numeric_to_f32(array, values),
        AnyDataArray::I16(array) => extend_numeric_to_f32(array, values),
    }
}

fn extend_numeric_to_f32<T: NumericElement>(array: &DataArray<T>, values: &mut Vec<f32>) {
    values.extend(array.values().iter().map(|value| value.to_f64() as f32));
}

fn build_rgb_mask(
    arrays: &[&AnyDataArray; 3],
    pixel_count: usize,
    common_channel_mask: bool,
) -> Option<ValidityMask> {
    if arrays.iter().all(|array| array.mask().is_none()) {
        return None;
    }

    let mut output_mask = ValidityMask::all_valid(pixel_count * 3);
    if common_channel_mask {
        for pixel_index in 0..pixel_count {
            let masked = arrays.iter().any(|array| is_masked(array, pixel_index));
            if masked {
                for band in 0..3 {
                    output_mask.set_masked(band * pixel_count + pixel_index, true);
                }
            }
        }
    } else {
        for (band, array) in arrays.iter().enumerate() {
            for pixel_index in 0..pixel_count {
                if is_masked(array, pixel_index) {
                    output_mask.set_masked(band * pixel_count + pixel_index, true);
                }
            }
        }
    }
    Some(output_mask)
}

fn build_rgb_mask_from_owned_masks(
    masks: &[Option<ValidityMask>],
    pixel_count: usize,
    common_channel_mask: bool,
) -> Option<ValidityMask> {
    if masks.iter().all(Option::is_none) {
        return None;
    }

    let mut output_mask = ValidityMask::all_valid(pixel_count * 3);
    if common_channel_mask {
        for pixel_index in 0..pixel_count {
            let masked = masks.iter().any(|mask| {
                mask.as_ref()
                    .and_then(|mask| mask.is_masked(pixel_index))
                    .unwrap_or(false)
            });
            if masked {
                for band in 0..3 {
                    output_mask.set_masked(band * pixel_count + pixel_index, true);
                }
            }
        }
    } else {
        for (band, mask) in masks.iter().enumerate() {
            let Some(mask) = mask else {
                continue;
            };
            for pixel_index in 0..pixel_count {
                if mask.is_masked(pixel_index).unwrap_or(false) {
                    output_mask.set_masked(band * pixel_count + pixel_index, true);
                }
            }
        }
    }
    Some(output_mask)
}

fn is_masked(array: &AnyDataArray, pixel_index: usize) -> bool {
    array
        .mask()
        .and_then(|mask| mask.is_masked(pixel_index))
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;
    use rusty_sat_core::{DataArray, DataType};

    struct PlaceholderCompositor;

    impl Compositor for PlaceholderCompositor {
        fn name(&self) -> &str {
            "placeholder"
        }
    }

    #[test]
    fn compositor_trait_compiles() {
        let compositor = PlaceholderCompositor;
        assert_eq!(compositor.name(), "placeholder");
        assert!(compositor.compose(&[]).is_err());
    }

    #[test]
    fn rgb_compositor_concatenates_three_luma_datasets() -> Result<()> {
        let compositor = RgbCompositor::new("true_color")?;
        let inputs = vec![
            dataset(
                "red",
                DataArray::<u8>::from_vec_named([2, 2], ["y", "x"], vec![1, 2, 3, 4])?,
            ),
            dataset(
                "green",
                DataArray::<u16>::from_vec_named([2, 2], ["y", "x"], vec![5, 6, 7, 8])?,
            ),
            dataset(
                "blue",
                DataArray::<f32>::from_vec_named([2, 2], ["y", "x"], vec![9.0, 10.0, 11.0, 12.0])?,
            ),
        ];

        let output = compositor.compose(&inputs)?;
        let array = output.array().expect("RGB compositor output array");

        assert_eq!(output.id().name(), "true_color");
        assert_eq!(array.dtype(), DataType::F32);
        assert_eq!(array.shape(), &[3, 2, 2]);
        assert_eq!(array.dims(), &["bands", "y", "x"]);
        assert_eq!(
            array.values_as_f64(),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
        );
        assert_eq!(
            output.attr("mode").and_then(MetadataValue::as_str),
            Some("RGB")
        );
        assert!(array.coord("bands").is_some());
        Ok(())
    }

    #[test]
    fn rgb_compositor_rejects_missing_or_mismatched_inputs() -> Result<()> {
        let compositor = RgbCompositor::new("rgb")?;
        let red = dataset("red", DataArray::<f64>::new(1, 2, vec![1.0, 2.0])?);
        let green = dataset("green", DataArray::<f64>::new(1, 2, vec![3.0, 4.0])?);
        let blue = dataset("blue", DataArray::<f64>::new(2, 1, vec![5.0, 6.0])?);

        assert!(compositor.compose(&[red.clone(), green.clone()]).is_err());
        assert!(compositor.compose(&[red, green, blue]).is_err());
        Ok(())
    }

    #[test]
    fn rgb_compositor_applies_common_channel_mask() -> Result<()> {
        let compositor = RgbCompositor::new("rgb")?;
        let red_mask = ValidityMask::from_masked_flags([false, true, false]);
        let blue_mask = ValidityMask::from_masked_flags([false, false, true]);
        let inputs = vec![
            dataset(
                "red",
                DataArray::<f64>::from_vec_named([1, 3], ["y", "x"], vec![1.0, 2.0, 3.0])?
                    .with_mask(red_mask)?,
            ),
            dataset(
                "green",
                DataArray::<f64>::from_vec_named([1, 3], ["y", "x"], vec![4.0, 5.0, 6.0])?,
            ),
            dataset(
                "blue",
                DataArray::<f64>::from_vec_named([1, 3], ["y", "x"], vec![7.0, 8.0, 9.0])?
                    .with_mask(blue_mask)?,
            ),
        ];

        let output = compositor.compose(&inputs)?;
        let mask = output
            .array()
            .and_then(AnyDataArray::mask)
            .expect("common RGB mask");

        assert_eq!(mask.masked_count(), 6);
        for index in [1, 2, 4, 5, 7, 8] {
            assert_eq!(mask.is_masked(index), Some(true));
        }
        assert_eq!(mask.is_masked(0), Some(false));
        assert_eq!(mask.is_masked(3), Some(false));
        assert_eq!(mask.is_masked(6), Some(false));
        Ok(())
    }

    #[test]
    fn rgb_compositor_can_preserve_per_channel_masks() -> Result<()> {
        let compositor = RgbCompositor::new("rgb")?.with_common_channel_mask(false);
        let inputs = vec![
            dataset(
                "red",
                DataArray::<f64>::from_vec_named([1, 2], ["y", "x"], vec![1.0, 2.0])?
                    .with_mask(ValidityMask::from_masked_flags([false, true]))?,
            ),
            dataset(
                "green",
                DataArray::<f64>::from_vec_named([1, 2], ["y", "x"], vec![3.0, 4.0])?,
            ),
            dataset(
                "blue",
                DataArray::<f64>::from_vec_named([1, 2], ["y", "x"], vec![5.0, 6.0])?,
            ),
        ];

        let output = compositor.compose(&inputs)?;
        let mask = output
            .array()
            .and_then(AnyDataArray::mask)
            .expect("per-channel RGB mask");

        assert_eq!(mask.masked_count(), 1);
        assert_eq!(mask.is_masked(1), Some(true));
        assert_eq!(mask.is_masked(3), Some(false));
        assert_eq!(mask.is_masked(5), Some(false));
        Ok(())
    }

    #[test]
    fn rgb_compositor_owned_consumes_inputs() -> Result<()> {
        let compositor = RgbCompositor::new("rgb")?;
        let inputs = vec![
            dataset(
                "red",
                DataArray::<u8>::from_vec_named([1, 2], ["y", "x"], vec![1, 2])?,
            ),
            dataset(
                "green",
                DataArray::<f64>::from_vec_named([1, 2], ["y", "x"], vec![3.0, 4.0])?,
            ),
            dataset(
                "blue",
                DataArray::<i16>::from_vec_named([1, 2], ["y", "x"], vec![5, 6])?
                    .with_mask(ValidityMask::from_masked_flags([false, true]))?,
            ),
        ];

        let output = compositor.compose_rgb_owned(inputs)?;
        let array = output.array().expect("owned RGB output array");
        let mask = array.mask().expect("owned RGB common mask");

        assert_eq!(array.shape(), &[3, 1, 2]);
        assert_eq!(array.values_as_f64(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(mask.masked_count(), 3);
        assert_eq!(mask.is_masked(1), Some(true));
        assert_eq!(mask.is_masked(3), Some(true));
        assert_eq!(mask.is_masked(5), Some(true));
        Ok(())
    }

    fn dataset<T: NumericElement>(name: &str, array: DataArray<T>) -> Dataset
    where
        AnyDataArray: From<DataArray<T>>,
    {
        Dataset::new(DataId::new(name).expect("valid test data id")).with_array(array)
    }
}
