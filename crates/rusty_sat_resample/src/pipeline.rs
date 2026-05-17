//! Thin Satpy-style resampling pipeline helpers.
//!
//! Reference behavior inspected before implementation:
//! - `satpy/satpy/resample/base.py`
//!
//! Satpy separates resampler preparation from dataset resampling. Rusty Sat
//! keeps that split, but makes the selected resampler an explicit enum instead
//! of dynamically importing Python classes by name.

use crate::{
    AreaDefinition, BilinearAreaResampler, NativeResampler, NearestAreaResampler, Resampler,
};
use rusty_sat_core::{Dataset, Result, RustySatError};
use std::str::FromStr;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResamplerMethod {
    NearestArea,
    Bilinear,
    Native,
}

impl ResamplerMethod {
    pub fn from_name(name: &str) -> Result<Self> {
        name.parse()
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::NearestArea => "nearest_area",
            Self::Bilinear => "bilinear",
            Self::Native => "native",
        }
    }
}

impl FromStr for ResamplerMethod {
    type Err = RustySatError;

    fn from_str(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "nearest" | "nearest_area" | "kd_tree" => Ok(Self::NearestArea),
            "bilinear" => Ok(Self::Bilinear),
            "native" => Ok(Self::Native),
            other => Err(RustySatError::not_found(format!(
                "resampler method '{other}'"
            ))),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ResampleOptions {
    method: ResamplerMethod,
    radius_of_influence: Option<f64>,
    fill_value: f64,
    mask_missing: bool,
}

impl Default for ResampleOptions {
    fn default() -> Self {
        Self {
            method: ResamplerMethod::NearestArea,
            radius_of_influence: None,
            fill_value: f64::NAN,
            mask_missing: false,
        }
    }
}

impl ResampleOptions {
    pub fn new(method: ResamplerMethod) -> Self {
        Self {
            method,
            ..Self::default()
        }
    }

    pub fn nearest_area() -> Self {
        Self::new(ResamplerMethod::NearestArea)
    }

    pub fn native() -> Self {
        Self::new(ResamplerMethod::Native)
    }

    pub fn bilinear() -> Self {
        Self::new(ResamplerMethod::Bilinear)
    }

    pub fn method(&self) -> ResamplerMethod {
        self.method
    }

    pub fn radius_of_influence(&self) -> Option<f64> {
        self.radius_of_influence
    }

    pub fn fill_value(&self) -> f64 {
        self.fill_value
    }

    pub fn mask_missing(&self) -> bool {
        self.mask_missing
    }

    pub fn with_method(mut self, method: ResamplerMethod) -> Self {
        self.method = method;
        self
    }

    pub fn with_radius_of_influence(mut self, radius_of_influence: f64) -> Result<Self> {
        if radius_of_influence < 0.0 {
            return Err(RustySatError::invalid_input(
                "radius_of_influence must be non-negative",
            ));
        }
        self.radius_of_influence = Some(radius_of_influence);
        Ok(self)
    }

    pub fn with_fill_value(mut self, fill_value: f64) -> Self {
        self.fill_value = fill_value;
        self.mask_missing = false;
        self
    }

    pub fn with_masked_missing(mut self) -> Self {
        self.mask_missing = true;
        self
    }
}

#[derive(Debug, Clone)]
pub enum PreparedResampler {
    NearestArea(NearestAreaResampler),
    Bilinear(BilinearAreaResampler),
    Native(NativeResampler),
}

impl PreparedResampler {
    pub fn method(&self) -> ResamplerMethod {
        match self {
            Self::NearestArea(_) => ResamplerMethod::NearestArea,
            Self::Bilinear(_) => ResamplerMethod::Bilinear,
            Self::Native(_) => ResamplerMethod::Native,
        }
    }
}

impl Resampler for PreparedResampler {
    fn name(&self) -> &str {
        match self {
            Self::NearestArea(resampler) => resampler.name(),
            Self::Bilinear(resampler) => resampler.name(),
            Self::Native(resampler) => resampler.name(),
        }
    }

    fn resample(&self, dataset: &Dataset, destination: &AreaDefinition) -> Result<Dataset> {
        match self {
            Self::NearestArea(resampler) => resampler.resample(dataset, destination),
            Self::Bilinear(resampler) => resampler.resample(dataset, destination),
            Self::Native(resampler) => resampler.resample(dataset, destination),
        }
    }

    fn resample_owned(&self, dataset: Dataset, destination: &AreaDefinition) -> Result<Dataset> {
        match self {
            Self::NearestArea(resampler) => resampler.resample_owned(dataset, destination),
            Self::Bilinear(resampler) => resampler.resample_owned(dataset, destination),
            Self::Native(resampler) => resampler.resample_owned(dataset, destination),
        }
    }
}

/// Prepare a resampler from the given source area and options.
///
/// The `_destination` parameter is reserved for future CRS-aware resampler
/// setup (S7-next); it is currently unused because neither the nearest-area
/// nor the native resampler performs cross-projection coordinate transforms.
///
/// The `Bilinear` method currently uses `fill_value`/`mask_missing` but ignores
/// `radius_of_influence`; full Pyresample bilinear neighbour search belongs to
/// S3-next/S2-next.
///
/// The `Native` method ignores `radius_of_influence`, `fill_value`, and
/// `mask_missing` — native resampling is a pure geometric operation without
/// radiusing or fill-value semantics.
pub fn prepare_resampler(
    source: AreaDefinition,
    _destination: &AreaDefinition,
    options: ResampleOptions,
) -> Result<PreparedResampler> {
    match options.method {
        ResamplerMethod::NearestArea => {
            let mut resampler = NearestAreaResampler::new(source);
            if let Some(radius_of_influence) = options.radius_of_influence {
                resampler = resampler.with_radius_of_influence(radius_of_influence)?;
            }
            resampler = if options.mask_missing {
                resampler.with_masked_missing()
            } else {
                resampler.with_fill_value(options.fill_value)
            };
            Ok(PreparedResampler::NearestArea(resampler))
        }
        ResamplerMethod::Bilinear => {
            let mut resampler = BilinearAreaResampler::new(source);
            resampler = if options.mask_missing {
                resampler.with_masked_missing()
            } else {
                resampler.with_fill_value(options.fill_value)
            };
            Ok(PreparedResampler::Bilinear(resampler))
        }
        ResamplerMethod::Native => Ok(PreparedResampler::Native(NativeResampler::new(source))),
    }
}

pub fn resample_dataset(
    dataset: &Dataset,
    source: AreaDefinition,
    destination: &AreaDefinition,
    options: ResampleOptions,
) -> Result<Dataset> {
    let resampler = prepare_resampler(source, destination, options)?;
    resampler.resample(dataset, destination)
}

pub fn resample_dataset_owned(
    dataset: Dataset,
    source: AreaDefinition,
    destination: &AreaDefinition,
    options: ResampleOptions,
) -> Result<Dataset> {
    let resampler = prepare_resampler(source, destination, options)?;
    resampler.resample_owned(dataset, destination)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rusty_sat_core::{DataGrid, DataId};
    use std::collections::BTreeMap;

    fn area(id: &str, height: usize, width: usize, extent: [f64; 4]) -> AreaDefinition {
        AreaDefinition::from_parts(
            id,
            id,
            id,
            BTreeMap::from([("proj".to_string(), "longlat".to_string())]),
            height,
            width,
            extent,
        )
        .unwrap()
    }

    #[test]
    fn parses_resampler_method_names() {
        assert_eq!(
            ResamplerMethod::from_name("nearest_area").unwrap(),
            ResamplerMethod::NearestArea
        );
        assert_eq!(
            ResamplerMethod::from_name("kd_tree").unwrap(),
            ResamplerMethod::NearestArea
        );
        assert_eq!(
            ResamplerMethod::from_name("native").unwrap(),
            ResamplerMethod::Native
        );
        assert_eq!(
            ResamplerMethod::from_name("bilinear").unwrap(),
            ResamplerMethod::Bilinear
        );
        assert!(ResamplerMethod::from_name("ewa").is_err());
    }

    #[test]
    fn prepares_nearest_resampler_with_fill_options() {
        let source = area("source", 2, 2, [0.0, 0.0, 2.0, 2.0]);
        let destination = area("destination", 1, 1, [2.0, 2.0, 3.0, 3.0]);
        let options = ResampleOptions::nearest_area()
            .with_radius_of_influence(0.1)
            .unwrap()
            .with_fill_value(-999.0);
        let resampler = prepare_resampler(source, &destination, options).unwrap();
        assert_eq!(resampler.method(), ResamplerMethod::NearestArea);

        let dataset = Dataset::new(DataId::new("image").unwrap())
            .with_data(DataGrid::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap());
        let output = resampler.resample(&dataset, &destination).unwrap();

        assert_eq!(output.data().unwrap().values(), &[-999.0]);
        assert_eq!(
            output.metadata().get("resampler"),
            Some(&"nearest_area".to_string())
        );
    }

    #[test]
    fn resample_dataset_owned_uses_native_repeat() {
        let source = area("source", 2, 2, [0.0, 0.0, 2.0, 2.0]);
        let destination = area("destination", 4, 4, [0.0, 0.0, 4.0, 4.0]);
        let dataset = Dataset::new(DataId::new("image").unwrap())
            .with_data(DataGrid::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap());

        let output =
            resample_dataset_owned(dataset, source, &destination, ResampleOptions::native())
                .unwrap();

        assert_eq!(output.data().unwrap().shape(), (4, 4));
        assert_eq!(
            output.metadata().get("resampler"),
            Some(&"native".to_string())
        );
        assert_eq!(
            &output.data().unwrap().values()[..8],
            &[1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0]
        );
    }

    #[test]
    fn resample_dataset_uses_bilinear_method() {
        let source = area("source", 2, 2, [0.0, 0.0, 2.0, 2.0]);
        let destination = area("destination", 1, 1, [0.5, 0.5, 1.5, 1.5]);
        let dataset = Dataset::new(DataId::new("image").unwrap())
            .with_data(DataGrid::new(2, 2, vec![0.0, 10.0, 20.0, 30.0]).unwrap());

        let output =
            resample_dataset(&dataset, source, &destination, ResampleOptions::bilinear()).unwrap();

        assert_eq!(output.data().unwrap().values(), &[15.0]);
        assert_eq!(
            output.metadata().get("resampler"),
            Some(&"bilinear".to_string())
        );
    }

    #[test]
    fn resample_dataset_uses_default_nearest_method() {
        let source = area("source", 2, 2, [0.0, 0.0, 2.0, 2.0]);
        let destination = area("destination", 1, 1, [0.0, 1.0, 1.0, 2.0]);
        let dataset = Dataset::new(DataId::new("image").unwrap())
            .with_data(DataGrid::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap());

        let output =
            resample_dataset(&dataset, source, &destination, ResampleOptions::default()).unwrap();

        assert_eq!(output.data().unwrap().values(), &[1.0]);
        assert_eq!(
            output.metadata().get("resampler"),
            Some(&"nearest_area".to_string())
        );
    }

    #[test]
    fn prepare_resampler_with_native_method() {
        let source = area("source", 2, 2, [0.0, 0.0, 2.0, 2.0]);
        let destination = area("destination", 4, 4, [0.0, 0.0, 4.0, 4.0]);
        let options = ResampleOptions::native();
        let resampler = prepare_resampler(source, &destination, options).unwrap();

        assert_eq!(resampler.method(), ResamplerMethod::Native);

        let dataset = Dataset::new(DataId::new("image").unwrap())
            .with_data(DataGrid::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap());
        let output = resampler.resample(&dataset, &destination).unwrap();

        assert_eq!(output.data().unwrap().shape(), (4, 4));
    }

    #[test]
    fn mask_missing_option_produces_masked_output() {
        let source = area("source", 1, 1, [0.0, 0.0, 1.0, 1.0]);
        let destination = area("destination", 1, 1, [2.0, 2.0, 3.0, 3.0]);
        let options = ResampleOptions::nearest_area()
            .with_radius_of_influence(0.25)
            .unwrap()
            .with_masked_missing();
        let resampler = prepare_resampler(source, &destination, options).unwrap();

        let dataset = Dataset::new(DataId::new("image").unwrap())
            .with_data(DataGrid::new(1, 1, vec![5.0]).unwrap());
        let output = resampler.resample(&dataset, &destination).unwrap();

        assert!(output.data().unwrap().values()[0].is_nan());
        assert_eq!(output.data().unwrap().mask().unwrap().masked_count(), 1);
    }

    #[test]
    fn options_with_method_chains_to_switch_resampler() {
        let native = ResampleOptions::nearest_area().with_method(ResamplerMethod::Native);
        assert_eq!(native.method(), ResamplerMethod::Native);

        let nearest = ResampleOptions::native().with_method(ResamplerMethod::NearestArea);
        assert_eq!(nearest.method(), ResamplerMethod::NearestArea);
    }

    #[test]
    fn options_rejects_negative_radius() {
        assert!(ResampleOptions::nearest_area()
            .with_radius_of_influence(-0.5)
            .is_err());
    }
}
