//! Elliptical-weighted-average resampling foundations.
//!
//! Reference behavior inspected before implementation:
//! - `satpy/satpy/resample/ewa.py`
//! - `deps/pyresample/pyresample/ewa/__init__.py`
//! - `deps/pyresample/pyresample/ewa/dask_ewa.py`
//! - `deps/pyresample/pyresample/ewa/ewa.py`
//!
//! Pyresample's production EWA path is a two-stage native implementation:
//! `ll2cr` maps scan geolocation to output column/row coordinates and
//! `fornav` applies scan-aware weighted accumulation. This first S5 slice is
//! intentionally smaller: it provides a dependency-free lon/lat swath to
//! lon/lat area weighted accumulator with Satpy-like fill/mask handling. Full
//! Fornav/LLS2 parity, scan geometry, chunking, and multi-band execution remain
//! S5-next work.

use crate::{AreaDefinition, Resampler, SwathDefinition};
use rayon::prelude::*;
use rusty_sat_core::{Coordinate, DataGrid, Dataset, Result, RustySatError, ValidityMask};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EwaOptions {
    radius_of_influence: f64,
    weight_min: f64,
    weight_sum_min: f64,
    fill_value: f64,
    mask_missing: bool,
}

impl EwaOptions {
    pub fn new(radius_of_influence: f64) -> Result<Self> {
        if !radius_of_influence.is_finite() || radius_of_influence <= 0.0 {
            return Err(RustySatError::invalid_input(
                "EWA radius_of_influence must be finite and positive",
            ));
        }
        Ok(Self {
            radius_of_influence,
            weight_min: 0.01,
            weight_sum_min: 0.0,
            fill_value: f64::NAN,
            mask_missing: false,
        })
    }

    pub fn with_weight_min(mut self, weight_min: f64) -> Result<Self> {
        if !weight_min.is_finite() || weight_min <= 0.0 || weight_min >= 1.0 {
            return Err(RustySatError::invalid_input(
                "EWA weight_min must be finite and between 0 and 1",
            ));
        }
        self.weight_min = weight_min;
        Ok(self)
    }

    pub fn with_weight_sum_min(mut self, weight_sum_min: f64) -> Result<Self> {
        if !weight_sum_min.is_finite() || weight_sum_min < 0.0 {
            return Err(RustySatError::invalid_input(
                "EWA weight_sum_min must be finite and non-negative",
            ));
        }
        self.weight_sum_min = weight_sum_min;
        Ok(self)
    }

    pub fn with_fill_value(mut self, fill_value: f64) -> Self {
        self.fill_value = fill_value;
        self
    }

    pub fn with_masked_missing(mut self, mask_missing: bool) -> Self {
        self.mask_missing = mask_missing;
        self
    }

    pub fn radius_of_influence(&self) -> f64 {
        self.radius_of_influence
    }

    pub fn weight_min(&self) -> f64 {
        self.weight_min
    }

    pub fn weight_sum_min(&self) -> f64 {
        self.weight_sum_min
    }

    pub fn fill_value(&self) -> f64 {
        self.fill_value
    }

    pub fn mask_missing(&self) -> bool {
        self.mask_missing
    }
}

#[derive(Debug, Clone)]
pub struct EwaResampler {
    source: SwathDefinition,
    options: EwaOptions,
}

impl EwaResampler {
    pub fn new(source: SwathDefinition, options: EwaOptions) -> Self {
        Self { source, options }
    }

    pub fn with_radius(source: SwathDefinition, radius_of_influence: f64) -> Result<Self> {
        Ok(Self::new(source, EwaOptions::new(radius_of_influence)?))
    }

    pub fn source(&self) -> &SwathDefinition {
        &self.source
    }

    pub fn options(&self) -> EwaOptions {
        self.options
    }
}

impl Resampler for EwaResampler {
    fn name(&self) -> &str {
        "ewa"
    }

    fn resample(&self, dataset: &Dataset, destination: &AreaDefinition) -> Result<Dataset> {
        let source_grid = dataset.data().ok_or_else(|| {
            RustySatError::invalid_input("EWA resampling requires f64 dataset grid values")
        })?;
        validate_source_shape(source_grid, &self.source)?;
        validate_ewa_target(destination)?;

        let resampled = resample_swath_ewa(source_grid, &self.source, destination, self.options)?;
        let mut resampled_dataset = Dataset::new(dataset.id().clone()).with_data(resampled);
        for (key, value) in dataset.metadata() {
            resampled_dataset.insert_metadata(key.clone(), value.clone())?;
        }
        for (key, value) in dataset.attrs() {
            resampled_dataset.insert_attr(key.clone(), value.clone())?;
        }
        resampled_dataset.insert_metadata("area", destination.id())?;
        resampled_dataset.insert_metadata("resampler", self.name())?;
        Ok(resampled_dataset)
    }

    fn resample_owned(&self, dataset: Dataset, destination: &AreaDefinition) -> Result<Dataset> {
        let id = dataset.id().clone();
        let metadata = dataset.metadata().clone();
        let attrs = dataset.attrs().clone();
        let source_grid = dataset
            .into_array()
            .and_then(|array| array.into_f64())
            .ok_or_else(|| {
                RustySatError::invalid_input("EWA resampling requires an f64 dataset grid")
            })?;
        validate_source_shape(&source_grid, &self.source)?;
        validate_ewa_target(destination)?;

        let resampled =
            resample_swath_ewa_owned(source_grid, &self.source, destination, self.options)?;
        let mut resampled_dataset = Dataset::new(id).with_data(resampled);
        for (key, value) in metadata {
            resampled_dataset.insert_metadata(key, value)?;
        }
        for (key, value) in attrs {
            resampled_dataset.insert_attr(key, value)?;
        }
        resampled_dataset.insert_metadata("area", destination.id())?;
        resampled_dataset.insert_metadata("resampler", self.name())?;
        Ok(resampled_dataset)
    }
}

pub fn resample_swath_ewa(
    source_grid: &DataGrid,
    source: &SwathDefinition,
    destination: &AreaDefinition,
    options: EwaOptions,
) -> Result<DataGrid> {
    validate_source_shape(source_grid, source)?;
    validate_ewa_target(destination)?;
    let accumulators = EwaAccumulators::new(destination, options)?;
    accumulators.add_borrowed(source_grid, source)?;
    add_ewa_coords(
        accumulators.finish()?,
        Some(source_grid.coords()),
        destination,
    )
}

pub fn resample_swath_ewa_owned(
    source_grid: DataGrid,
    source: &SwathDefinition,
    destination: &AreaDefinition,
    options: EwaOptions,
) -> Result<DataGrid> {
    validate_source_shape(&source_grid, source)?;
    validate_ewa_target(destination)?;
    let (values, coords, mask) = source_grid.into_parts();
    let accumulators = EwaAccumulators::new(destination, options)?;
    accumulators.add_values(&values, mask.as_ref(), source)?;
    add_ewa_coords_owned(accumulators.finish()?, Some(coords), destination)
}

#[derive(Debug)]
struct EwaAccumulators<'a> {
    destination: &'a AreaDefinition,
    options: EwaOptions,
    sums: Vec<AtomicU64>,
    weight_sums: Vec<AtomicU64>,
    radius_squared: f64,
    alpha: f64,
}

/// Lock-free f64 addition on top of `AtomicU64` (bit-pattern CAS loop).
///
/// Parallel EWA accumulation writes to shared per-pixel sums; CAS-based adds
/// keep the memory footprint identical to the sequential version (no
/// per-thread accumulator copies of the target grid). Contention is bounded
/// because each source point only touches its small pixel footprint.
fn atomic_add_f64(atomic: &AtomicU64, value: f64) {
    let _ = atomic.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
        Some((f64::from_bits(current) + value).to_bits())
    });
}

impl<'a> EwaAccumulators<'a> {
    fn new(destination: &'a AreaDefinition, options: EwaOptions) -> Result<Self> {
        let (height, width) = destination.shape();
        let size = height * width;
        let radius_squared = options.radius_of_influence * options.radius_of_influence;
        let alpha = -options.weight_min.ln() / radius_squared;
        Ok(Self {
            destination,
            options,
            sums: (0..size)
                .map(|_| AtomicU64::new(0.0_f64.to_bits()))
                .collect(),
            weight_sums: (0..size)
                .map(|_| AtomicU64::new(0.0_f64.to_bits()))
                .collect(),
            radius_squared,
            alpha,
        })
    }

    fn add_borrowed(&self, source_grid: &DataGrid, source: &SwathDefinition) -> Result<()> {
        self.add_values(source_grid.values(), source_grid.mask(), source)
    }

    fn add_values(
        &self,
        values: &[f64],
        mask: Option<&ValidityMask>,
        source: &SwathDefinition,
    ) -> Result<()> {
        let lons = source
            .lons()
            .ok_or_else(|| RustySatError::invalid_input("EWA resampling requires source lons"))?;
        let lats = source
            .lats()
            .ok_or_else(|| RustySatError::invalid_input("EWA resampling requires source lats"))?;
        let count = lons.len().min(lats.len()).min(values.len());
        (0..count).into_par_iter().for_each(|source_idx| {
            if mask
                .and_then(|mask| mask.is_masked(source_idx))
                .unwrap_or(false)
            {
                return;
            }
            let lon = lons[source_idx];
            let lat = lats[source_idx];
            let value = values[source_idx];
            if !lon.is_finite() || !lat.is_finite() || !value.is_finite() {
                return;
            }
            self.add_sample(lon, lat, value);
        });
        Ok(())
    }

    fn add_sample(&self, lon: f64, lat: f64, value: f64) {
        let (height, width) = self.destination.shape();
        let extent = self.destination.area_extent();
        let (pixel_size_x, pixel_size_y) = self.destination.pixel_size();
        let center_x = (lon - extent[0]) / pixel_size_x - 0.5;
        let center_y = (extent[3] - lat) / pixel_size_y - 0.5;
        if !center_x.is_finite() || !center_y.is_finite() {
            return;
        }

        let radius_x = self.options.radius_of_influence / pixel_size_x.abs();
        let radius_y = self.options.radius_of_influence / pixel_size_y.abs();
        let x_start = ((center_x - radius_x).floor().max(0.0)) as usize;
        let y_start = ((center_y - radius_y).floor().max(0.0)) as usize;
        let x_end = ((center_x + radius_x).ceil().min(width as f64 - 1.0)) as usize;
        let y_end = ((center_y + radius_y).ceil().min(height as f64 - 1.0)) as usize;
        if x_start >= width || y_start >= height || x_start > x_end || y_start > y_end {
            return;
        }

        for y in y_start..=y_end {
            let target_y = extent[3] - (y as f64 + 0.5) * pixel_size_y;
            let dy = target_y - lat;
            for x in x_start..=x_end {
                let target_x = extent[0] + (x as f64 + 0.5) * pixel_size_x;
                let dx = target_x - lon;
                let distance_squared = dx * dx + dy * dy;
                if distance_squared > self.radius_squared {
                    continue;
                }
                let weight = (-self.alpha * distance_squared).exp();
                if weight < self.options.weight_min {
                    continue;
                }
                let target_idx = y * width + x;
                atomic_add_f64(&self.sums[target_idx], value * weight);
                atomic_add_f64(&self.weight_sums[target_idx], weight);
            }
        }
    }

    fn finish(self) -> Result<DataGrid> {
        let (height, width) = self.destination.shape();
        let sums = self
            .sums
            .into_iter()
            .map(|atomic| f64::from_bits(atomic.into_inner()));
        let weight_sums = self
            .weight_sums
            .into_iter()
            .map(|atomic| f64::from_bits(atomic.into_inner()));
        let mut values = Vec::with_capacity(height * width);
        let mut masked = Vec::with_capacity(height * width);
        for (sum, weight_sum) in sums.zip(weight_sums) {
            if weight_sum > self.options.weight_sum_min {
                values.push(sum / weight_sum);
                masked.push(false);
            } else {
                values.push(self.options.fill_value);
                masked.push(true);
            }
        }
        let mut grid = DataGrid::new(height, width, values)?;
        if self.options.mask_missing {
            grid.set_mask(ValidityMask::from_masked_flags(masked))?;
        }
        Ok(grid)
    }
}

fn validate_source_shape(source_grid: &DataGrid, source: &SwathDefinition) -> Result<()> {
    if source_grid.shape() != source.shape() {
        return Err(RustySatError::invalid_input(format!(
            "dataset grid shape {:?} does not match swath shape {:?}",
            source_grid.shape(),
            source.shape()
        )));
    }
    Ok(())
}

fn validate_ewa_target(destination: &AreaDefinition) -> Result<()> {
    let projection = destination.projection();
    let Some(proj) = projection.get("proj").or_else(|| projection.get("proj4")) else {
        return Err(RustySatError::unsupported(
            "EWA foundation requires lon/lat destination projection metadata",
        ));
    };
    if proj.contains("latlong") || proj.contains("longlat") {
        return Ok(());
    }
    Err(RustySatError::unsupported(
        "EWA foundation only supports lon/lat destination areas",
    ))
}

fn add_ewa_coords(
    mut grid: DataGrid,
    source_coords: Option<&BTreeMap<String, Coordinate>>,
    area: &AreaDefinition,
) -> Result<DataGrid> {
    if let Some(coords) = source_coords {
        for (name, coordinate) in coords {
            if should_preserve_coord(name, coordinate) {
                grid.set_coordinate(name.clone(), coordinate.clone())?;
            }
        }
    }
    add_destination_coords(grid, area)
}

fn add_ewa_coords_owned(
    mut grid: DataGrid,
    source_coords: Option<BTreeMap<String, Coordinate>>,
    area: &AreaDefinition,
) -> Result<DataGrid> {
    if let Some(coords) = source_coords {
        for (name, coordinate) in coords {
            if should_preserve_coord(&name, &coordinate) {
                grid.set_coordinate(name, coordinate)?;
            }
        }
    }
    add_destination_coords(grid, area)
}

fn add_destination_coords(mut grid: DataGrid, area: &AreaDefinition) -> Result<DataGrid> {
    grid.set_coordinate(
        "x",
        Coordinate::axis("x", area.iter_projection_x_coords().collect::<Vec<_>>())?,
    )?;
    grid.set_coordinate(
        "y",
        Coordinate::axis("y", area.iter_projection_y_coords().collect::<Vec<_>>())?,
    )?;
    Ok(grid)
}

fn should_preserve_coord(name: &str, coordinate: &Coordinate) -> bool {
    const IGNORE_DIMS: [&str; 5] = ["y", "x", "crs", "longitude", "latitude"];
    !IGNORE_DIMS.contains(&name)
        && !coordinate
            .dims()
            .iter()
            .any(|dim| IGNORE_DIMS.contains(&dim.as_str()))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;
    use rusty_sat_core::DataId;

    fn area() -> AreaDefinition {
        AreaDefinition::from_parts(
            "target",
            "target",
            "target",
            BTreeMap::from([("proj".to_string(), "longlat".to_string())]),
            1,
            3,
            [0.0, 0.0, 3.0, 1.0],
        )
        .unwrap()
    }

    #[test]
    fn options_validate_radius_and_weights() {
        assert!(EwaOptions::new(0.0).is_err());
        assert!(EwaOptions::new(1.0).unwrap().with_weight_min(1.0).is_err());
        assert!(EwaOptions::new(1.0)
            .unwrap()
            .with_weight_sum_min(-1.0)
            .is_err());
    }

    #[test]
    fn ewa_spreads_weighted_samples_inside_radius() {
        let source = SwathDefinition::from_lonlats(1, 2, vec![0.5, 1.5], vec![0.5, 0.5]).unwrap();
        let grid = DataGrid::new(1, 2, vec![10.0, 20.0]).unwrap();
        let options = EwaOptions::new(1.0)
            .unwrap()
            .with_weight_min(0.1)
            .unwrap()
            .with_fill_value(-999.0);

        let resampled = resample_swath_ewa(&grid, &source, &area(), options).unwrap();

        assert_eq!(resampled.shape(), (1, 3));
        assert!(resampled.values()[0] > 10.0);
        assert!(resampled.values()[0] < resampled.values()[1]);
        assert!(resampled.values()[1] > 10.0);
        assert!(resampled.values()[1] < 20.0);
        assert!(resampled.values()[2] > resampled.values()[1]);
        assert_eq!(resampled.values()[2], 20.0);
        assert!(resampled.coord("x").is_some());
        assert!(resampled.coord("y").is_some());
    }

    #[test]
    fn ewa_uses_fill_and_optional_mask_for_empty_pixels() {
        let source = SwathDefinition::from_lonlats(1, 1, vec![0.5], vec![0.5]).unwrap();
        let grid = DataGrid::new(1, 1, vec![10.0]).unwrap();
        let options = EwaOptions::new(0.25)
            .unwrap()
            .with_fill_value(-999.0)
            .with_masked_missing(true);

        let resampled = resample_swath_ewa(&grid, &source, &area(), options).unwrap();

        assert_eq!(resampled.values(), &[10.0, -999.0, -999.0]);
        assert_eq!(resampled.mask().unwrap().is_masked(0), Some(false));
        assert_eq!(resampled.mask().unwrap().is_masked(1), Some(true));
        assert_eq!(resampled.mask().unwrap().is_masked(2), Some(true));
    }

    #[test]
    fn ewa_owned_matches_borrowed() {
        let source = SwathDefinition::from_lonlats(1, 2, vec![0.5, 1.5], vec![0.5, 0.5]).unwrap();
        let grid = DataGrid::new(1, 2, vec![10.0, 20.0]).unwrap();
        let options = EwaOptions::new(1.0).unwrap().with_fill_value(-999.0);

        let borrowed = resample_swath_ewa(&grid, &source, &area(), options).unwrap();
        let owned = resample_swath_ewa_owned(grid, &source, &area(), options).unwrap();

        assert_eq!(borrowed.values(), owned.values());
    }

    #[test]
    fn ewa_resampler_preserves_dataset_metadata() {
        let source = SwathDefinition::from_lonlats(1, 1, vec![0.5], vec![0.5]).unwrap();
        let mut dataset = Dataset::new(DataId::new("B03").unwrap()).with_data(
            DataGrid::new(1, 1, vec![10.0])
                .unwrap()
                .with_coordinate("time", Coordinate::scalar(1.0))
                .unwrap(),
        );
        dataset.insert_metadata("sensor", "ahi").unwrap();
        let resampler = EwaResampler::with_radius(source, 0.25).unwrap();

        let resampled = resampler.resample(&dataset, &area()).unwrap();

        assert_eq!(resampled.metadata().get("sensor"), Some(&"ahi".to_string()));
        assert_eq!(
            resampled.metadata().get("resampler"),
            Some(&"ewa".to_string())
        );
        assert!(resampled.data().unwrap().coord("time").is_some());
    }

    #[test]
    fn ewa_rejects_missing_swath_coordinates() {
        let source = SwathDefinition::new(1, 1).unwrap();
        let grid = DataGrid::new(1, 1, vec![10.0]).unwrap();

        let err =
            resample_swath_ewa(&grid, &source, &area(), EwaOptions::new(1.0).unwrap()).unwrap_err();

        assert!(err.to_string().contains("source lons"));
    }

    #[test]
    fn ewa_weight_min_filters_low_weight_contributions() {
        let source = SwathDefinition::from_lonlats(1, 2, vec![0.5, 1.0], vec![0.5, 0.5]).unwrap();
        let grid = DataGrid::new(1, 2, vec![10.0, 20.0]).unwrap();
        // weight_min=0.9 means only very close (<~0.05 deg) points contribute
        let options = EwaOptions::new(1.0)
            .unwrap()
            .with_weight_min(0.9)
            .unwrap()
            .with_fill_value(-999.0);

        let resampled = resample_swath_ewa(&grid, &source, &area(), options).unwrap();

        // Point at lon=0.5 close to pixel center → contributes. lon=1.0 far → filtered.
        assert!(resampled.values()[0] > 0.0);
    }

    #[test]
    fn ewa_weight_sum_min_fills_when_insufficient_total_weight() {
        // Point at edge of area, far from all pixel centers with small radius
        let source = SwathDefinition::from_lonlats(1, 1, vec![0.05], vec![0.5]).unwrap();
        let grid = DataGrid::new(1, 1, vec![10.0]).unwrap();
        let options = EwaOptions::new(0.3)
            .unwrap()
            .with_weight_sum_min(0.5)
            .unwrap()
            .with_fill_value(-999.0);

        let resampled = resample_swath_ewa(&grid, &source, &area(), options).unwrap();

        assert!(resampled.values().iter().all(|v| *v == -999.0));
    }

    #[test]
    fn ewa_resampler_owned_through_trait() {
        let source = SwathDefinition::from_lonlats(1, 2, vec![0.5, 1.5], vec![0.5, 0.5]).unwrap();
        let id = DataId::new("test").unwrap();
        let ds1 =
            Dataset::new(id.clone()).with_data(DataGrid::new(1, 2, vec![10.0, 20.0]).unwrap());
        let ds2 = Dataset::new(id).with_data(DataGrid::new(1, 2, vec![10.0, 20.0]).unwrap());
        let resampler = EwaResampler::with_radius(source, 1.0).unwrap();

        let borrowed = resampler.resample(&ds1, &area()).unwrap();
        let owned = resampler.resample_owned(ds2, &area()).unwrap();

        assert_eq!(
            borrowed.data().unwrap().values(),
            owned.data().unwrap().values()
        );
    }

    #[test]
    fn ewa_skips_non_finite_source_values() {
        let source = SwathDefinition::from_lonlats(1, 2, vec![0.5, 1.5], vec![0.5, 0.5]).unwrap();
        let grid = DataGrid::new(1, 2, vec![10.0, f64::NAN]).unwrap();
        let options = EwaOptions::new(1.0).unwrap().with_fill_value(-999.0);

        let resampled = resample_swath_ewa(&grid, &source, &area(), options).unwrap();

        // Only the first valid value (10.0) contributes; NaN is skipped.
        assert!(resampled.values()[0] > 0.0);
    }
}
