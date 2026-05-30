//! Safe execution for the supported subset of Satpy enhancement operations.
//!
//! Reference behavior inspected before implementation:
//! - `satpy/satpy/enhancements/enhancer.py` applies configured operations in
//!   order to a Trollimage `XRImage`.
//! - `satpy/satpy/enhancements/contrast.py` delegates `stretch`, `gamma`, and
//!   `invert` to Trollimage.
//! - `deps/trollimage/trollimage/xrimage.py` defines `stretch(..., "crude")`,
//!   `gamma`, and `invert` semantics.
//!
//! Rusty Sat intentionally does not execute Python names from YAML. This module
//! maps a small allow-list of method names to Rust-native `FloatImage` methods.

use crate::config::{EnhancementDefinition, EnhancementOperation};
use rusty_sat_core::{MetadataValue, Result, RustySatError};
use rusty_sat_image::{FloatImage, ImageFloat};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnhancementExecutor {
    strict: bool,
}

impl EnhancementExecutor {
    pub fn new() -> Self {
        Self { strict: true }
    }

    pub fn permissive() -> Self {
        Self { strict: false }
    }

    pub fn apply<T: ImageFloat>(
        &self,
        image: &mut FloatImage<T>,
        definition: &EnhancementDefinition,
    ) -> Result<()> {
        for operation in definition.operations() {
            self.apply_operation(image, operation)?;
        }
        Ok(())
    }

    pub fn apply_operation<T: ImageFloat>(
        &self,
        image: &mut FloatImage<T>,
        operation: &EnhancementOperation,
    ) -> Result<()> {
        match canonical_operation(operation.method()).as_deref() {
            Some("stretch") => apply_stretch(image, operation),
            Some("gamma") => apply_gamma(image, operation),
            Some("invert") => apply_invert(image, operation),
            Some(_) | None if self.strict => Err(RustySatError::unsupported(format!(
                "enhancement operation '{}'",
                operation.method()
            ))),
            Some(_) | None => Ok(()),
        }
    }
}

impl Default for EnhancementExecutor {
    fn default() -> Self {
        Self::new()
    }
}

fn apply_stretch<T: ImageFloat>(
    image: &mut FloatImage<T>,
    operation: &EnhancementOperation,
) -> Result<()> {
    let stretch = operation
        .kwargs()
        .get("stretch")
        .and_then(MetadataValue::as_str)
        .unwrap_or("crude");
    if !matches!(stretch, "crude" | "crude-stretch") {
        return Err(RustySatError::unsupported(format!(
            "enhancement stretch mode '{stretch}'"
        )));
    }
    let min_stretch = operation
        .kwargs()
        .get("min_stretch")
        .map(metadata_f64)
        .transpose()?
        .map(T::from_f64);
    let max_stretch = operation
        .kwargs()
        .get("max_stretch")
        .map(metadata_f64)
        .transpose()?
        .map(T::from_f64);
    image.crude_stretch_in_place(min_stretch, max_stretch);
    Ok(())
}

fn apply_gamma<T: ImageFloat>(
    image: &mut FloatImage<T>,
    operation: &EnhancementOperation,
) -> Result<()> {
    let gamma = if let Some(value) = operation.kwargs().get("gamma") {
        metadata_f64_or_list(value)?
    } else if let Some(value) = operation.args().first() {
        metadata_f64_or_list(value)?
    } else {
        vec![1.0]
    };
    if gamma.len() == 1 {
        image.gamma_in_place(T::from_f64(gamma[0]))
    } else {
        let gamma = gamma.into_iter().map(T::from_f64).collect::<Vec<_>>();
        image.gamma_channels_in_place(&gamma)
    }
}

fn apply_invert<T: ImageFloat>(
    image: &mut FloatImage<T>,
    operation: &EnhancementOperation,
) -> Result<()> {
    let invert = if let Some(value) = operation.args().first() {
        metadata_bool_or_list(value)?
    } else if let Some(value) = operation.kwargs().get("invert") {
        metadata_bool_or_list(value)?
    } else {
        vec![true]
    };
    if invert.len() == 1 {
        image.invert_in_place(invert[0]);
        Ok(())
    } else {
        image.invert_channels_in_place(&invert)
    }
}

fn canonical_operation(method: &str) -> Option<String> {
    let method = method.rsplit('.').next().unwrap_or(method).trim();
    if method.is_empty() {
        None
    } else {
        Some(method.to_string())
    }
}

fn metadata_f64_or_list(value: &MetadataValue) -> Result<Vec<f64>> {
    match value {
        MetadataValue::List(values) => values.iter().map(metadata_f64).collect(),
        _ => metadata_f64(value).map(|value| vec![value]),
    }
}

fn metadata_bool_or_list(value: &MetadataValue) -> Result<Vec<bool>> {
    match value {
        MetadataValue::Bool(value) => Ok(vec![*value]),
        MetadataValue::List(values) => values.iter().map(metadata_bool).collect(),
        _ => Err(RustySatError::invalid_input(
            "enhancement invert expects a boolean or list of booleans",
        )),
    }
}

fn metadata_f64(value: &MetadataValue) -> Result<f64> {
    match value {
        MetadataValue::Float(value) => Ok(value.get()),
        MetadataValue::Integer(value) => Ok(*value as f64),
        _ => Err(RustySatError::invalid_input(
            "enhancement numeric argument must be int or float",
        )),
    }
}

fn metadata_bool(value: &MetadataValue) -> Result<bool> {
    match value {
        MetadataValue::Bool(value) => Ok(*value),
        _ => Err(RustySatError::invalid_input(
            "enhancement boolean argument must be bool",
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::CompositeRegistryConfig;
    use rusty_sat_image::{FloatImage, ImageMode};

    #[test]
    fn executes_stretch_gamma_and_invert_from_yaml_order() -> Result<()> {
        let config = CompositeRegistryConfig::from_yaml_str(
            r#"
enhancements:
  ahi_true_color:
    name: true_color
    operations:
      - name: stretch
        method: !!python/name:satpy.enhancements.contrast.stretch
        kwargs: {stretch: crude, min_stretch: 0.0, max_stretch: 100.0}
      - name: gamma
        method: !!python/name:satpy.enhancements.contrast.gamma
        kwargs: {gamma: 2.0}
      - name: invert
        method: !!python/name:satpy.enhancements.contrast.invert
        args: [true]
"#,
        )?;
        let definition = config.enhancements().get("ahi_true_color").unwrap();
        let mut image =
            FloatImage::<f64>::from_pixels(ImageMode::Luma, 1, 3, vec![0.0, 25.0, 100.0])?;

        EnhancementExecutor::new().apply(&mut image, definition)?;

        let expected = [1.0, 0.5, 0.0];
        for (got, expected) in image.pixels().iter().zip(expected) {
            assert!((*got - expected).abs() < 1e-12);
        }
        assert_eq!(image.enhancement_history().len(), 1);
        assert_eq!(image.gamma_history().len(), 1);
        assert_eq!(image.invert_history().len(), 1);
        Ok(())
    }

    #[test]
    fn executes_channel_gamma_and_invert_lists() -> Result<()> {
        let config = CompositeRegistryConfig::from_yaml_str(
            r#"
enhancements:
  rgb:
    operations:
      - name: gamma
        method: gamma
        args: [[1.0, 2.0, 1.0]]
      - name: invert
        method: invert
        args: [[false, true, false]]
"#,
        )?;
        let definition = config.enhancements().get("rgb").unwrap();
        let mut image =
            FloatImage::<f64>::from_pixels(ImageMode::Rgb, 1, 1, vec![0.25, 0.25, 0.25])?;

        EnhancementExecutor::new().apply(&mut image, definition)?;

        assert!((image.pixels()[0] - 0.25).abs() < 1e-12);
        assert!((image.pixels()[1] - 0.5).abs() < 1e-12);
        assert!((image.pixels()[2] - 0.25).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn rejects_unsupported_enhancement_in_strict_mode() -> Result<()> {
        let config = CompositeRegistryConfig::from_yaml_str(
            r#"
enhancements:
  test:
    operations:
      - name: palettize
        method: satpy.enhancements.colormap.palettize
"#,
        )?;
        let operation = &config.enhancements()["test"].operations()[0];
        let mut image = FloatImage::<f32>::from_pixels(ImageMode::Luma, 1, 1, vec![0.5])?;

        let err = EnhancementExecutor::new()
            .apply_operation(&mut image, operation)
            .unwrap_err();

        assert!(err.to_string().contains("unsupported feature"));
        Ok(())
    }

    #[test]
    fn permissive_mode_skips_unsupported_enhancements() -> Result<()> {
        let config = CompositeRegistryConfig::from_yaml_str(
            r#"
enhancements:
  test:
    operations:
      - name: palettize
        method: satpy.enhancements.colormap.palettize
"#,
        )?;
        let operation = &config.enhancements()["test"].operations()[0];
        let mut image = FloatImage::<f32>::from_pixels(ImageMode::Luma, 1, 1, vec![0.5])?;

        EnhancementExecutor::permissive().apply_operation(&mut image, operation)?;

        assert_eq!(image.pixels(), &[0.5]);
        Ok(())
    }
}
