//! Thin Satpy-style resampling pipeline helpers.
//!
//! Reference behavior inspected before implementation:
//! - `satpy/satpy/resample/base.py`
//!
//! Satpy separates resampler preparation from dataset resampling. Rusty Sat
//! keeps that split, but makes the selected resampler an explicit enum instead
//! of dynamically importing Python classes by name.

use crate::{
    reduce_area_dataset_owned_with_divisibility, reduce_area_dataset_with_divisibility,
    AreaDefinition, BilinearAreaResampler, BucketFractionResampler, BucketResampler,
    BucketStatistic, EwaOptions, EwaResampler, NativeResampler, NearestAreaResampler,
    NearestSwathResampler, Resampler, SwathDefinition,
};
use rusty_sat_core::{Dataset, Result, RustySatError};
use std::str::FromStr;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResamplerMethod {
    NearestArea,
    Bilinear,
    Native,
    BucketAverage,
    BucketSum,
    BucketCount,
    BucketFraction,
    Ewa,
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
            Self::BucketAverage => "bucket_avg",
            Self::BucketSum => "bucket_sum",
            Self::BucketCount => "bucket_count",
            Self::BucketFraction => "bucket_fraction",
            Self::Ewa => "ewa",
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
            "bucket" | "bucket_avg" | "bucket_average" => Ok(Self::BucketAverage),
            "bucket_sum" => Ok(Self::BucketSum),
            "bucket_count" => Ok(Self::BucketCount),
            "bucket_fraction" | "bucket_frac" => Ok(Self::BucketFraction),
            "ewa" | "ewa_legacy" => Ok(Self::Ewa),
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
    skipna: bool,
    reduce_data: bool,
    shape_divisible_by: Option<usize>,
    bucket_categories: Vec<f64>,
    bucket_categories_auto: bool,
}

impl Default for ResampleOptions {
    fn default() -> Self {
        Self {
            method: ResamplerMethod::NearestArea,
            radius_of_influence: None,
            fill_value: f64::NAN,
            mask_missing: false,
            skipna: true,
            reduce_data: false,
            shape_divisible_by: None,
            bucket_categories: Vec::new(),
            bucket_categories_auto: false,
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

    pub fn bucket_average() -> Self {
        Self::new(ResamplerMethod::BucketAverage)
    }

    pub fn bucket_sum() -> Self {
        Self::new(ResamplerMethod::BucketSum)
    }

    pub fn bucket_count() -> Self {
        Self::new(ResamplerMethod::BucketCount)
    }

    pub fn bucket_fraction(categories: impl Into<Vec<f64>>) -> Self {
        Self::new(ResamplerMethod::BucketFraction).with_bucket_categories(categories)
    }

    pub fn bucket_fraction_auto() -> Self {
        Self::new(ResamplerMethod::BucketFraction).with_auto_bucket_categories()
    }

    pub fn ewa() -> Self {
        Self::new(ResamplerMethod::Ewa)
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

    pub fn skipna(&self) -> bool {
        self.skipna
    }

    pub fn reduce_data(&self) -> bool {
        self.reduce_data
    }

    pub fn shape_divisible_by(&self) -> Option<usize> {
        self.shape_divisible_by
    }

    pub fn bucket_categories(&self) -> &[f64] {
        &self.bucket_categories
    }

    pub fn bucket_categories_auto(&self) -> bool {
        self.bucket_categories_auto
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

    pub fn with_skipna(mut self, skipna: bool) -> Self {
        self.skipna = skipna;
        self
    }

    pub fn with_reduce_data(mut self, reduce_data: bool) -> Self {
        self.reduce_data = reduce_data;
        if !reduce_data {
            self.shape_divisible_by = None;
        }
        self
    }

    pub fn with_data_reduction(self) -> Self {
        self.with_reduce_data(true)
    }

    pub fn without_data_reduction(self) -> Self {
        self.with_reduce_data(false)
    }

    pub fn with_shape_divisible_by(mut self, factor: usize) -> Result<Self> {
        if factor == 0 {
            return Err(RustySatError::invalid_input(
                "shape_divisible_by must be greater than zero",
            ));
        }
        self.shape_divisible_by = Some(factor);
        self.reduce_data = true;
        Ok(self)
    }

    pub fn with_bucket_categories(mut self, categories: impl Into<Vec<f64>>) -> Self {
        self.bucket_categories = categories.into();
        self.bucket_categories_auto = false;
        self
    }

    pub fn with_auto_bucket_categories(mut self) -> Self {
        self.bucket_categories.clear();
        self.bucket_categories_auto = true;
        self
    }
}

#[derive(Debug, Clone)]
pub enum PreparedResampler {
    NearestArea(NearestAreaResampler),
    NearestSwath(NearestSwathResampler),
    Bilinear(BilinearAreaResampler),
    Native(NativeResampler),
    Bucket(BucketResampler),
    BucketFraction(BucketFractionResampler),
    Ewa(EwaResampler),
}

impl PreparedResampler {
    pub fn method(&self) -> ResamplerMethod {
        match self {
            Self::NearestArea(_) | Self::NearestSwath(_) => ResamplerMethod::NearestArea,
            Self::Bilinear(_) => ResamplerMethod::Bilinear,
            Self::Native(_) => ResamplerMethod::Native,
            Self::Bucket(resampler) => match resampler.statistic() {
                BucketStatistic::Average => ResamplerMethod::BucketAverage,
                BucketStatistic::Sum => ResamplerMethod::BucketSum,
                BucketStatistic::Count => ResamplerMethod::BucketCount,
            },
            Self::BucketFraction(_) => ResamplerMethod::BucketFraction,
            Self::Ewa(_) => ResamplerMethod::Ewa,
        }
    }
}

impl Resampler for PreparedResampler {
    fn name(&self) -> &str {
        match self {
            Self::NearestArea(resampler) => resampler.name(),
            Self::NearestSwath(resampler) => resampler.name(),
            Self::Bilinear(resampler) => resampler.name(),
            Self::Native(resampler) => resampler.name(),
            Self::Bucket(resampler) => resampler.name(),
            Self::BucketFraction(resampler) => resampler.name(),
            Self::Ewa(resampler) => resampler.name(),
        }
    }

    fn resample(&self, dataset: &Dataset, destination: &AreaDefinition) -> Result<Dataset> {
        match self {
            Self::NearestArea(resampler) => resampler.resample(dataset, destination),
            Self::NearestSwath(resampler) => resampler.resample(dataset, destination),
            Self::Bilinear(resampler) => resampler.resample(dataset, destination),
            Self::Native(resampler) => resampler.resample(dataset, destination),
            Self::Bucket(resampler) => resampler.resample(dataset, destination),
            Self::BucketFraction(resampler) => resampler.resample(dataset, destination),
            Self::Ewa(resampler) => resampler.resample(dataset, destination),
        }
    }

    fn resample_owned(&self, dataset: Dataset, destination: &AreaDefinition) -> Result<Dataset> {
        match self {
            Self::NearestArea(resampler) => resampler.resample_owned(dataset, destination),
            Self::NearestSwath(resampler) => resampler.resample_owned(dataset, destination),
            Self::Bilinear(resampler) => resampler.resample_owned(dataset, destination),
            Self::Native(resampler) => resampler.resample_owned(dataset, destination),
            Self::Bucket(resampler) => resampler.resample_owned(dataset, destination),
            Self::BucketFraction(resampler) => resampler.resample_owned(dataset, destination),
            Self::Ewa(resampler) => resampler.resample_owned(dataset, destination),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum SourceGeometry {
    Area(AreaDefinition),
    Swath(SwathDefinition),
}

impl SourceGeometry {
    pub fn area(area: AreaDefinition) -> Self {
        Self::Area(area)
    }

    pub fn swath(swath: SwathDefinition) -> Self {
        Self::Swath(swath)
    }
}

#[derive(Debug, Clone, Default)]
pub struct ResamplerCache {
    entries: Vec<CachedResampler>,
}

#[derive(Debug, Clone)]
struct CachedResampler {
    source: SourceGeometry,
    destination: AreaDefinition,
    options: ResampleOptions,
    resampler: PreparedResampler,
}

impl ResamplerCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }

    pub fn prepare(
        &mut self,
        source: AreaDefinition,
        destination: &AreaDefinition,
        options: ResampleOptions,
    ) -> Result<&PreparedResampler> {
        self.prepare_for_geometry(SourceGeometry::Area(source), destination, options)
    }

    pub fn prepare_for_geometry(
        &mut self,
        source: SourceGeometry,
        destination: &AreaDefinition,
        options: ResampleOptions,
    ) -> Result<&PreparedResampler> {
        if let Some(index) = self.entries.iter().position(|entry| {
            entry.source == source
                && entry.destination == *destination
                && resample_options_equivalent(&entry.options, &options)
        }) {
            return Ok(&self.entries[index].resampler);
        }

        let resampler =
            prepare_resampler_for_geometry(source.clone(), destination, options.clone())?;
        self.entries.push(CachedResampler {
            source,
            destination: destination.clone(),
            options,
            resampler,
        });
        Ok(&self
            .entries
            .last()
            .expect("cache entry was just inserted")
            .resampler)
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
    prepare_resampler_for_geometry(SourceGeometry::Area(source), _destination, options)
}

pub fn prepare_resampler_for_geometry(
    source: SourceGeometry,
    _destination: &AreaDefinition,
    options: ResampleOptions,
) -> Result<PreparedResampler> {
    match options.method {
        ResamplerMethod::NearestArea => match source {
            SourceGeometry::Area(source) => {
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
            SourceGeometry::Swath(source) => {
                let mut resampler = NearestSwathResampler::new(source);
                if let Some(radius_of_influence) = options.radius_of_influence {
                    resampler = resampler.with_radius_of_influence(radius_of_influence)?;
                }
                resampler = if options.mask_missing {
                    resampler.with_masked_missing()
                } else {
                    resampler.with_fill_value(options.fill_value)
                };
                Ok(PreparedResampler::NearestSwath(resampler))
            }
        },
        ResamplerMethod::Bilinear => {
            let SourceGeometry::Area(source) = source else {
                return Err(RustySatError::unsupported(
                    "bilinear pipeline preparation from swath geometry",
                ));
            };
            let mut resampler = BilinearAreaResampler::new(source);
            resampler = if options.mask_missing {
                resampler.with_masked_missing()
            } else {
                resampler.with_fill_value(options.fill_value)
            };
            Ok(PreparedResampler::Bilinear(resampler))
        }
        ResamplerMethod::Native => {
            let SourceGeometry::Area(source) = source else {
                return Err(RustySatError::unsupported(
                    "native pipeline preparation from swath geometry",
                ));
            };
            Ok(PreparedResampler::Native(NativeResampler::new(source)))
        }
        ResamplerMethod::BucketAverage => {
            let SourceGeometry::Swath(source) = source else {
                return Err(RustySatError::unsupported(
                    "bucket average pipeline preparation from area geometry",
                ));
            };
            Ok(PreparedResampler::Bucket(
                BucketResampler::average(source)
                    .with_fill_value(options.fill_value)
                    .with_skipna(options.skipna),
            ))
        }
        ResamplerMethod::BucketSum => {
            let SourceGeometry::Swath(source) = source else {
                return Err(RustySatError::unsupported(
                    "bucket sum pipeline preparation from area geometry",
                ));
            };
            Ok(PreparedResampler::Bucket(
                BucketResampler::sum(source)
                    .with_fill_value(options.fill_value)
                    .with_skipna(options.skipna),
            ))
        }
        ResamplerMethod::BucketCount => {
            let SourceGeometry::Swath(source) = source else {
                return Err(RustySatError::unsupported(
                    "bucket count pipeline preparation from area geometry",
                ));
            };
            Ok(PreparedResampler::Bucket(BucketResampler::count(source)))
        }
        ResamplerMethod::BucketFraction => {
            let SourceGeometry::Swath(source) = source else {
                return Err(RustySatError::unsupported(
                    "bucket fraction pipeline preparation from area geometry",
                ));
            };
            let resampler = if options.bucket_categories_auto {
                BucketFractionResampler::auto_categories(source)
            } else {
                BucketFractionResampler::new(source, options.bucket_categories)?
            };
            Ok(PreparedResampler::BucketFraction(
                resampler.with_fill_value(options.fill_value),
            ))
        }
        ResamplerMethod::Ewa => {
            let SourceGeometry::Swath(source) = source else {
                return Err(RustySatError::unsupported(
                    "EWA pipeline preparation from area geometry",
                ));
            };
            let radius_of_influence = options.radius_of_influence.ok_or_else(|| {
                RustySatError::invalid_input(
                    "EWA pipeline preparation requires radius_of_influence",
                )
            })?;
            let mut ewa_options =
                EwaOptions::new(radius_of_influence)?.with_fill_value(options.fill_value);
            if options.mask_missing {
                ewa_options = ewa_options.with_masked_missing(true);
            }
            Ok(PreparedResampler::Ewa(EwaResampler::new(
                source,
                ewa_options,
            )))
        }
    }
}

pub fn resample_dataset(
    dataset: &Dataset,
    source: AreaDefinition,
    destination: &AreaDefinition,
    options: ResampleOptions,
) -> Result<Dataset> {
    if options.reduce_data {
        return resample_area_dataset_reduced(dataset, source, destination, options);
    }
    let resampler = prepare_resampler(source, destination, options)?;
    resampler.resample(dataset, destination)
}

pub fn resample_dataset_from_geometry(
    dataset: &Dataset,
    source: SourceGeometry,
    destination: &AreaDefinition,
    options: ResampleOptions,
) -> Result<Dataset> {
    let resampler = prepare_resampler_for_geometry(source, destination, options)?;
    resampler.resample(dataset, destination)
}

pub fn resample_dataset_cached(
    cache: &mut ResamplerCache,
    dataset: &Dataset,
    source: AreaDefinition,
    destination: &AreaDefinition,
    options: ResampleOptions,
) -> Result<Dataset> {
    if options.reduce_data {
        return resample_area_dataset_reduced_cached(cache, dataset, source, destination, options);
    }
    let resampler = cache.prepare(source, destination, options)?;
    resampler.resample(dataset, destination)
}

pub fn resample_area_dataset_reduced(
    dataset: &Dataset,
    source: AreaDefinition,
    destination: &AreaDefinition,
    options: ResampleOptions,
) -> Result<Dataset> {
    let shape_divisible_by = options.shape_divisible_by;
    let reduction =
        reduce_area_dataset_with_divisibility(dataset, &source, destination, shape_divisible_by)?;
    let (reduced_dataset, reduced_source, _) = reduction.into_parts();
    let resampler = prepare_resampler(reduced_source, destination, options)?;
    resampler.resample_owned(reduced_dataset, destination)
}

pub fn resample_area_dataset_reduced_cached(
    cache: &mut ResamplerCache,
    dataset: &Dataset,
    source: AreaDefinition,
    destination: &AreaDefinition,
    options: ResampleOptions,
) -> Result<Dataset> {
    let shape_divisible_by = options.shape_divisible_by;
    let reduction =
        reduce_area_dataset_with_divisibility(dataset, &source, destination, shape_divisible_by)?;
    let (reduced_dataset, reduced_source, _) = reduction.into_parts();
    let resampler = cache.prepare(reduced_source, destination, options)?;
    resampler.resample_owned(reduced_dataset, destination)
}

pub fn resample_dataset_from_geometry_cached(
    cache: &mut ResamplerCache,
    dataset: &Dataset,
    source: SourceGeometry,
    destination: &AreaDefinition,
    options: ResampleOptions,
) -> Result<Dataset> {
    let resampler = cache.prepare_for_geometry(source, destination, options)?;
    resampler.resample(dataset, destination)
}

pub fn resample_dataset_owned_from_geometry(
    dataset: Dataset,
    source: SourceGeometry,
    destination: &AreaDefinition,
    options: ResampleOptions,
) -> Result<Dataset> {
    let resampler = prepare_resampler_for_geometry(source, destination, options)?;
    resampler.resample_owned(dataset, destination)
}

pub fn resample_dataset_owned(
    dataset: Dataset,
    source: AreaDefinition,
    destination: &AreaDefinition,
    options: ResampleOptions,
) -> Result<Dataset> {
    if options.reduce_data {
        return resample_area_dataset_reduced_owned(dataset, source, destination, options);
    }
    let resampler = prepare_resampler(source, destination, options)?;
    resampler.resample_owned(dataset, destination)
}

pub fn resample_area_dataset_reduced_owned(
    dataset: Dataset,
    source: AreaDefinition,
    destination: &AreaDefinition,
    options: ResampleOptions,
) -> Result<Dataset> {
    let shape_divisible_by = options.shape_divisible_by;
    let reduction = reduce_area_dataset_owned_with_divisibility(
        dataset,
        &source,
        destination,
        shape_divisible_by,
    )?;
    let (reduced_dataset, reduced_source, _) = reduction.into_parts();
    let resampler = prepare_resampler(reduced_source, destination, options)?;
    resampler.resample_owned(reduced_dataset, destination)
}

pub fn resample_area_dataset_reduced_owned_cached(
    cache: &mut ResamplerCache,
    dataset: Dataset,
    source: AreaDefinition,
    destination: &AreaDefinition,
    options: ResampleOptions,
) -> Result<Dataset> {
    let shape_divisible_by = options.shape_divisible_by;
    let reduction = reduce_area_dataset_owned_with_divisibility(
        dataset,
        &source,
        destination,
        shape_divisible_by,
    )?;
    let (reduced_dataset, reduced_source, _) = reduction.into_parts();
    let resampler = cache.prepare(reduced_source, destination, options)?;
    resampler.resample_owned(reduced_dataset, destination)
}

pub fn resample_dataset_owned_cached(
    cache: &mut ResamplerCache,
    dataset: Dataset,
    source: AreaDefinition,
    destination: &AreaDefinition,
    options: ResampleOptions,
) -> Result<Dataset> {
    if options.reduce_data {
        return resample_area_dataset_reduced_owned_cached(
            cache,
            dataset,
            source,
            destination,
            options,
        );
    }
    let resampler = cache.prepare(source, destination, options)?;
    resampler.resample_owned(dataset, destination)
}

pub fn resample_dataset_owned_from_geometry_cached(
    cache: &mut ResamplerCache,
    dataset: Dataset,
    source: SourceGeometry,
    destination: &AreaDefinition,
    options: ResampleOptions,
) -> Result<Dataset> {
    let resampler = cache.prepare_for_geometry(source, destination, options)?;
    resampler.resample_owned(dataset, destination)
}

fn resample_options_equivalent(left: &ResampleOptions, right: &ResampleOptions) -> bool {
    left.method == right.method
        && optional_f64_bits_eq(left.radius_of_influence, right.radius_of_influence)
        && left.fill_value.to_bits() == right.fill_value.to_bits()
        && left.mask_missing == right.mask_missing
        && left.skipna == right.skipna
        && left.reduce_data == right.reduce_data
        && left.shape_divisible_by == right.shape_divisible_by
        && left.bucket_categories_auto == right.bucket_categories_auto
        && left.bucket_categories.len() == right.bucket_categories.len()
        && left
            .bucket_categories
            .iter()
            .zip(&right.bucket_categories)
            .all(|(left, right)| left.to_bits() == right.to_bits())
}

fn optional_f64_bits_eq(left: Option<f64>, right: Option<f64>) -> bool {
    match (left, right) {
        (Some(left), Some(right)) => left.to_bits() == right.to_bits(),
        (None, None) => true,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
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

    fn swath() -> SwathDefinition {
        SwathDefinition::from_lonlats(
            2,
            2,
            vec![0.25, 1.25, 0.25, 1.25],
            vec![1.25, 1.25, 0.25, 0.25],
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
        assert_eq!(
            ResamplerMethod::from_name("bucket_avg").unwrap(),
            ResamplerMethod::BucketAverage
        );
        assert_eq!(
            ResamplerMethod::from_name("bucket_sum").unwrap(),
            ResamplerMethod::BucketSum
        );
        assert_eq!(
            ResamplerMethod::from_name("bucket_count").unwrap(),
            ResamplerMethod::BucketCount
        );
        assert_eq!(
            ResamplerMethod::from_name("bucket_fraction").unwrap(),
            ResamplerMethod::BucketFraction
        );
        assert_eq!(
            ResamplerMethod::from_name("ewa").unwrap(),
            ResamplerMethod::Ewa
        );
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
    fn reduced_area_resample_crops_before_resampling() {
        let source = area("source", 4, 4, [0.0, 0.0, 4.0, 4.0]);
        let destination = area("destination", 2, 2, [1.0, 1.0, 3.0, 3.0]);
        let dataset = Dataset::new(DataId::new("image").unwrap())
            .with_data(DataGrid::new(4, 4, (0..16).map(f64::from).collect()).unwrap());

        let output = resample_area_dataset_reduced(
            &dataset,
            source,
            &destination,
            ResampleOptions::nearest_area(),
        )
        .unwrap();

        assert_eq!(output.data().unwrap().shape(), (2, 2));
        assert_eq!(output.data().unwrap().values(), &[5.0, 6.0, 9.0, 10.0]);
        assert_eq!(
            output.metadata().get("area"),
            Some(&"destination".to_string())
        );
    }

    #[test]
    fn reduced_area_resample_owned_matches_borrowed() {
        let source = area("source", 4, 4, [0.0, 0.0, 4.0, 4.0]);
        let destination = area("destination", 2, 2, [1.0, 1.0, 3.0, 3.0]);
        let dataset = Dataset::new(DataId::new("image").unwrap())
            .with_data(DataGrid::new(4, 4, (0..16).map(f64::from).collect()).unwrap());

        let borrowed = resample_area_dataset_reduced(
            &dataset,
            source.clone(),
            &destination,
            ResampleOptions::nearest_area(),
        )
        .unwrap();
        let owned = resample_area_dataset_reduced_owned(
            dataset,
            source,
            &destination,
            ResampleOptions::nearest_area(),
        )
        .unwrap();

        assert_eq!(borrowed, owned);
    }

    #[test]
    fn reduced_area_resample_cached_uses_reduced_source_area_key() {
        let source = area("source", 4, 4, [0.0, 0.0, 4.0, 4.0]);
        let destination = area("destination", 2, 2, [1.0, 1.0, 3.0, 3.0]);
        let dataset = Dataset::new(DataId::new("image").unwrap())
            .with_data(DataGrid::new(4, 4, (0..16).map(f64::from).collect()).unwrap());
        let mut cache = ResamplerCache::new();

        let first = resample_area_dataset_reduced_cached(
            &mut cache,
            &dataset,
            source.clone(),
            &destination,
            ResampleOptions::nearest_area(),
        )
        .unwrap();
        let second = resample_area_dataset_reduced_cached(
            &mut cache,
            &dataset,
            source,
            &destination,
            ResampleOptions::nearest_area(),
        )
        .unwrap();

        assert_eq!(first, second);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn resample_options_track_data_reduction_knobs() {
        let default = ResampleOptions::nearest_area();
        assert!(!default.reduce_data());
        assert_eq!(default.shape_divisible_by(), None);

        let reduced = default.clone().with_data_reduction();
        assert!(reduced.reduce_data());
        assert_eq!(reduced.shape_divisible_by(), None);

        let unreduced = reduced.without_data_reduction();
        assert!(!unreduced.reduce_data());
        assert_eq!(unreduced.shape_divisible_by(), None);

        let divisible = default.with_shape_divisible_by(4).unwrap();
        assert!(divisible.reduce_data());
        assert_eq!(divisible.shape_divisible_by(), Some(4));
        assert_eq!(
            divisible.without_data_reduction().shape_divisible_by(),
            None
        );

        assert!(ResampleOptions::nearest_area()
            .with_shape_divisible_by(0)
            .is_err());
    }

    #[test]
    fn resample_dataset_honors_reduction_option() {
        let source = area("source", 4, 4, [0.0, 0.0, 4.0, 4.0]);
        let destination = area("destination", 2, 2, [1.0, 1.0, 3.0, 3.0]);
        let dataset = Dataset::new(DataId::new("image").unwrap())
            .with_data(DataGrid::new(4, 4, (0..16).map(f64::from).collect()).unwrap());

        let output = resample_dataset(
            &dataset,
            source,
            &destination,
            ResampleOptions::nearest_area().with_data_reduction(),
        )
        .unwrap();

        assert_eq!(output.data().unwrap().shape(), (2, 2));
        assert_eq!(output.data().unwrap().values(), &[5.0, 6.0, 9.0, 10.0]);
    }

    #[test]
    fn resampler_cache_distinguishes_reduction_options() {
        let source = area("source", 4, 4, [0.0, 0.0, 4.0, 4.0]);
        let destination = area("destination", 2, 2, [1.0, 1.0, 3.0, 3.0]);
        let dataset = Dataset::new(DataId::new("image").unwrap())
            .with_data(DataGrid::new(4, 4, (0..16).map(f64::from).collect()).unwrap());
        let mut cache = ResamplerCache::new();

        resample_dataset_cached(
            &mut cache,
            &dataset,
            source.clone(),
            &destination,
            ResampleOptions::nearest_area().with_data_reduction(),
        )
        .unwrap();
        resample_dataset_cached(
            &mut cache,
            &dataset,
            source,
            &destination,
            ResampleOptions::nearest_area()
                .with_shape_divisible_by(2)
                .unwrap(),
        )
        .unwrap();

        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn resample_dataset_from_geometry_uses_bucket_average_method() {
        let destination = area("destination", 2, 2, [0.0, 0.0, 2.0, 2.0]);
        let dataset = Dataset::new(DataId::new("image").unwrap())
            .with_data(DataGrid::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap());
        let options = ResampleOptions::bucket_average().with_fill_value(-999.0);

        let output = resample_dataset_from_geometry(
            &dataset,
            SourceGeometry::swath(swath()),
            &destination,
            options,
        )
        .unwrap();

        assert_eq!(output.data().unwrap().values(), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(
            output.metadata().get("resampler"),
            Some(&"bucket_avg".to_string())
        );
    }

    #[test]
    fn resample_dataset_from_geometry_uses_bucket_fraction_method() {
        let destination = area("destination", 2, 2, [0.0, 0.0, 2.0, 2.0]);
        let dataset = Dataset::new(DataId::new("quality").unwrap())
            .with_data(DataGrid::new(2, 2, vec![0.0, 1.0, 1.0, 0.0]).unwrap());
        let options = ResampleOptions::bucket_fraction([0.0, 1.0]).with_fill_value(-1.0);

        let output = resample_dataset_from_geometry(
            &dataset,
            SourceGeometry::swath(swath()),
            &destination,
            options,
        )
        .unwrap();

        let array = output.array().unwrap();
        assert_eq!(array.shape(), &[2, 2, 2]);
        assert_eq!(array.dims(), &["categories", "y", "x"]);
        assert!(array.coord("categories").is_some());
        assert_eq!(
            output.metadata().get("resampler"),
            Some(&"bucket_fraction".to_string())
        );
    }

    #[test]
    fn resample_dataset_from_geometry_uses_auto_bucket_fraction_categories() {
        let destination = area("destination", 2, 2, [0.0, 0.0, 2.0, 2.0]);
        let dataset = Dataset::new(DataId::new("quality").unwrap())
            .with_data(DataGrid::new(2, 2, vec![2.0, 1.0, 2.0, 1.0]).unwrap());
        let options = ResampleOptions::bucket_fraction_auto().with_fill_value(-1.0);

        let output = resample_dataset_from_geometry(
            &dataset,
            SourceGeometry::swath(swath()),
            &destination,
            options,
        )
        .unwrap();

        let array = output.array().unwrap();
        assert_eq!(array.shape(), &[2, 2, 2]);
        assert_eq!(array.coord("categories").unwrap().values(), &[1.0, 2.0]);
    }

    #[test]
    fn bucket_fraction_pipeline_requires_categories() {
        let destination = area("destination", 1, 1, [0.0, 0.0, 1.0, 1.0]);

        let err = prepare_resampler_for_geometry(
            SourceGeometry::swath(swath()),
            &destination,
            ResampleOptions::new(ResamplerMethod::BucketFraction),
        )
        .unwrap_err();

        assert!(err.to_string().contains("at least one category"));
    }

    #[test]
    fn resample_dataset_owned_from_geometry_uses_ewa_method() {
        let destination = area("destination", 1, 2, [0.0, 0.0, 2.0, 1.0]);
        let source = SwathDefinition::from_lonlats(1, 2, vec![0.5, 1.5], vec![0.5, 0.5]).unwrap();
        let dataset = Dataset::new(DataId::new("image").unwrap())
            .with_data(DataGrid::new(1, 2, vec![10.0, 20.0]).unwrap());
        let options = ResampleOptions::ewa()
            .with_radius_of_influence(0.25)
            .unwrap()
            .with_fill_value(-999.0);

        let output = resample_dataset_owned_from_geometry(
            dataset,
            SourceGeometry::swath(source),
            &destination,
            options,
        )
        .unwrap();

        assert_eq!(output.data().unwrap().values(), &[10.0, 20.0]);
        assert_eq!(output.metadata().get("resampler"), Some(&"ewa".to_string()));
    }

    #[test]
    fn swath_pipeline_methods_require_swath_geometry() {
        let source = area("source", 1, 1, [0.0, 0.0, 1.0, 1.0]);
        let destination = area("destination", 1, 1, [0.0, 0.0, 1.0, 1.0]);

        assert!(matches!(
            prepare_resampler_for_geometry(
                SourceGeometry::area(source.clone()),
                &destination,
                ResampleOptions::bucket_count(),
            )
            .unwrap_err(),
            RustySatError::Unsupported { .. }
        ));
        assert!(prepare_resampler_for_geometry(
            SourceGeometry::area(source),
            &destination,
            ResampleOptions::ewa(),
        )
        .unwrap_err()
        .to_string()
        .contains("area geometry"));
    }

    #[test]
    fn swath_nearest_pipeline_preparation_builds_swath_resampler() {
        let destination = area("destination", 1, 1, [0.0, 0.0, 1.0, 1.0]);

        // Swath + nearest area now prepares the cached KD-indexed swath
        // resampler instead of erroring.
        let prepared = prepare_resampler_for_geometry(
            SourceGeometry::swath(swath()),
            &destination,
            ResampleOptions::nearest_area(),
        )
        .unwrap();
        assert_eq!(prepared.name(), "nearest_swath");

        let dataset = Dataset::new(DataId::new("swath_data").unwrap())
            .with_data(DataGrid::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap());
        let resampled = prepared.resample(&dataset, &destination).unwrap();
        assert_eq!(resampled.data().unwrap().shape(), (1, 1));
    }

    #[test]
    fn ewa_pipeline_requires_radius() {
        let destination = area("destination", 1, 1, [0.0, 0.0, 1.0, 1.0]);

        let err = prepare_resampler_for_geometry(
            SourceGeometry::swath(swath()),
            &destination,
            ResampleOptions::ewa(),
        )
        .unwrap_err();

        assert!(err.to_string().contains("radius_of_influence"));
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
    fn resampler_cache_reuses_matching_nan_default_options() {
        let source = area("source", 2, 2, [0.0, 0.0, 2.0, 2.0]);
        let destination = area("destination", 1, 1, [0.0, 1.0, 1.0, 2.0]);
        let mut cache = ResamplerCache::new();

        let first = cache
            .prepare(source.clone(), &destination, ResampleOptions::default())
            .unwrap()
            .method();
        let second = cache
            .prepare(source, &destination, ResampleOptions::default())
            .unwrap()
            .method();

        assert_eq!(first, ResamplerMethod::NearestArea);
        assert_eq!(second, ResamplerMethod::NearestArea);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn resampler_cache_distinguishes_options_and_can_clear() {
        let source = area("source", 2, 2, [0.0, 0.0, 2.0, 2.0]);
        let destination = area("destination", 1, 1, [0.0, 1.0, 1.0, 2.0]);
        let mut cache = ResamplerCache::new();

        cache
            .prepare(
                source.clone(),
                &destination,
                ResampleOptions::nearest_area().with_fill_value(-1.0),
            )
            .unwrap();
        cache
            .prepare(
                source,
                &destination,
                ResampleOptions::nearest_area().with_fill_value(-2.0),
            )
            .unwrap();

        assert_eq!(cache.len(), 2);
        cache.clear();
        assert!(cache.is_empty());
    }

    #[test]
    fn cached_resample_dataset_uses_prepared_resampler() {
        let source = area("source", 2, 2, [0.0, 0.0, 2.0, 2.0]);
        let destination = area("destination", 1, 1, [0.0, 1.0, 1.0, 2.0]);
        let dataset = Dataset::new(DataId::new("image").unwrap())
            .with_data(DataGrid::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap());
        let mut cache = ResamplerCache::new();

        let first = resample_dataset_cached(
            &mut cache,
            &dataset,
            source.clone(),
            &destination,
            ResampleOptions::default(),
        )
        .unwrap();
        let second = resample_dataset_cached(
            &mut cache,
            &dataset,
            source,
            &destination,
            ResampleOptions::default(),
        )
        .unwrap();

        assert_eq!(
            first.data().unwrap().values(),
            second.data().unwrap().values()
        );
        assert_eq!(cache.len(), 1);
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

        let bucket = ResampleOptions::bucket_sum().with_skipna(false);
        assert_eq!(bucket.method(), ResamplerMethod::BucketSum);
        assert!(!bucket.skipna());

        let fraction = ResampleOptions::bucket_fraction([0.0, 1.0]);
        assert_eq!(fraction.method(), ResamplerMethod::BucketFraction);
        assert_eq!(fraction.bucket_categories(), &[0.0, 1.0]);
        assert!(!fraction.bucket_categories_auto());

        let auto_fraction = ResampleOptions::bucket_fraction_auto();
        assert_eq!(auto_fraction.method(), ResamplerMethod::BucketFraction);
        assert!(auto_fraction.bucket_categories_auto());
        assert!(auto_fraction.bucket_categories().is_empty());
    }

    #[test]
    fn options_rejects_negative_radius() {
        assert!(ResampleOptions::nearest_area()
            .with_radius_of_influence(-0.5)
            .is_err());
    }

    #[test]
    fn resampler_cache_distinguishes_swath_and_area_geometry() {
        let destination = area("destination", 2, 2, [0.0, 0.0, 2.0, 2.0]);
        let mut cache = ResamplerCache::new();

        cache
            .prepare_for_geometry(
                SourceGeometry::area(area("src", 2, 2, [0.0, 0.0, 2.0, 2.0])),
                &destination,
                ResampleOptions::nearest_area(),
            )
            .unwrap();
        cache
            .prepare_for_geometry(
                SourceGeometry::swath(swath()),
                &destination,
                ResampleOptions::bucket_average(),
            )
            .unwrap();

        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn cached_owned_resample_dataset_works() {
        let source = area("source", 2, 2, [0.0, 0.0, 2.0, 2.0]);
        let destination = area("destination", 4, 4, [0.0, 0.0, 2.0, 2.0]);
        let dataset = Dataset::new(DataId::new("image").unwrap())
            .with_data(DataGrid::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap());
        let mut cache = ResamplerCache::new();

        let output = resample_dataset_owned_cached(
            &mut cache,
            dataset,
            source,
            &destination,
            ResampleOptions::native(),
        )
        .unwrap();

        assert_eq!(output.data().unwrap().shape(), (4, 4));
        assert_eq!(cache.len(), 1);
    }
}
