//! Drop-in-a-bucket resampling foundations.
//!
//! Reference behavior inspected before implementation:
//! - `satpy/satpy/resample/bucket.py`
//! - `deps/pyresample/pyresample/bucket/__init__.py`
//!
//! This first S4 slice supports lon/lat swath coordinates dropped into a
//! same-geographic target area. It implements average, sum, and count for 2D
//! f64 grids plus explicit and auto-discovered bucket fractions. Projected
//! target backends, multidimensional buckets, and chunked execution are future
//! S4 work.

use crate::{AreaDefinition, Resampler, SwathDefinition};
use rusty_sat_core::{
    Coordinate, DataArray, DataGrid, Dataset, MetadataValue, Result, RustySatError, ValidityMask,
};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BucketStatistic {
    Average,
    Sum,
    Count,
}

impl BucketStatistic {
    fn resampler_name(self) -> &'static str {
        match self {
            Self::Average => "bucket_avg",
            Self::Sum => "bucket_sum",
            Self::Count => "bucket_count",
        }
    }
}

#[derive(Debug, Clone)]
pub struct BucketResampler {
    source: SwathDefinition,
    statistic: BucketStatistic,
    fill_value: f64,
    skipna: bool,
}

#[derive(Debug, Clone)]
pub struct BucketFractionResampler {
    source: SwathDefinition,
    categories: Vec<f64>,
    discover_categories: bool,
    fill_value: f64,
}

impl BucketResampler {
    pub fn new(source: SwathDefinition, statistic: BucketStatistic) -> Self {
        Self {
            source,
            statistic,
            fill_value: f64::NAN,
            skipna: true,
        }
    }

    pub fn average(source: SwathDefinition) -> Self {
        Self::new(source, BucketStatistic::Average)
    }

    pub fn sum(source: SwathDefinition) -> Self {
        Self::new(source, BucketStatistic::Sum)
    }

    pub fn count(source: SwathDefinition) -> Self {
        Self::new(source, BucketStatistic::Count)
    }

    pub fn with_fill_value(mut self, fill_value: f64) -> Self {
        self.fill_value = fill_value;
        self
    }

    pub fn with_skipna(mut self, skipna: bool) -> Self {
        self.skipna = skipna;
        self
    }

    pub fn source(&self) -> &SwathDefinition {
        &self.source
    }

    pub fn statistic(&self) -> BucketStatistic {
        self.statistic
    }
}

impl BucketFractionResampler {
    pub fn new(source: SwathDefinition, categories: Vec<f64>) -> Result<Self> {
        validate_categories(&categories)?;
        Ok(Self {
            source,
            categories,
            discover_categories: false,
            fill_value: f64::NAN,
        })
    }

    pub fn auto_categories(source: SwathDefinition) -> Self {
        Self {
            source,
            categories: Vec::new(),
            discover_categories: true,
            fill_value: f64::NAN,
        }
    }

    pub fn with_fill_value(mut self, fill_value: f64) -> Self {
        self.fill_value = fill_value;
        self
    }

    pub fn source(&self) -> &SwathDefinition {
        &self.source
    }

    pub fn categories(&self) -> &[f64] {
        &self.categories
    }

    pub fn discovers_categories(&self) -> bool {
        self.discover_categories
    }

    fn categories_for_grid(&self, source_grid: &DataGrid) -> Result<Vec<f64>> {
        if self.discover_categories {
            discover_bucket_fraction_categories(source_grid)
        } else {
            Ok(self.categories.clone())
        }
    }
}

impl Resampler for BucketResampler {
    fn name(&self) -> &str {
        self.statistic.resampler_name()
    }

    fn resample(&self, dataset: &Dataset, destination: &AreaDefinition) -> Result<Dataset> {
        let source_grid = dataset.data().ok_or_else(|| {
            RustySatError::invalid_input("bucket resampling requires f64 dataset grid values")
        })?;
        validate_source_shape(source_grid, &self.source)?;
        validate_bucket_target(destination)?;

        let resampled = resample_bucket_with_statistic(
            source_grid,
            &self.source,
            destination,
            self.statistic,
            self.fill_value,
            self.skipna,
        )?;
        let mut resampled_dataset = Dataset::new(dataset.id().clone()).with_data(resampled);
        for (key, value) in dataset.metadata() {
            resampled_dataset.insert_metadata(key.clone(), value.clone())?;
        }
        for (key, value) in dataset.attrs() {
            resampled_dataset.insert_attr(key.clone(), value.clone())?;
        }
        adjust_bucket_attrs(&mut resampled_dataset, self.statistic)?;
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
                RustySatError::invalid_input("bucket resampling requires an f64 dataset grid")
            })?;
        validate_source_shape(&source_grid, &self.source)?;
        validate_bucket_target(destination)?;

        let resampled = resample_bucket_owned_with_statistic(
            source_grid,
            &self.source,
            destination,
            self.statistic,
            self.fill_value,
            self.skipna,
        )?;
        let mut resampled_dataset = Dataset::new(id).with_data(resampled);
        for (key, value) in metadata {
            resampled_dataset.insert_metadata(key, value)?;
        }
        for (key, value) in attrs {
            resampled_dataset.insert_attr(key, value)?;
        }
        adjust_bucket_attrs(&mut resampled_dataset, self.statistic)?;
        resampled_dataset.insert_metadata("area", destination.id())?;
        resampled_dataset.insert_metadata("resampler", self.name())?;
        Ok(resampled_dataset)
    }
}

impl Resampler for BucketFractionResampler {
    fn name(&self) -> &str {
        "bucket_fraction"
    }

    fn resample(&self, dataset: &Dataset, destination: &AreaDefinition) -> Result<Dataset> {
        let source_grid = dataset.data().ok_or_else(|| {
            RustySatError::invalid_input(
                "bucket fraction resampling requires f64 dataset grid values",
            )
        })?;
        let categories = self.categories_for_grid(source_grid)?;
        let resampled = resample_bucket_fraction(
            source_grid,
            &self.source,
            destination,
            &categories,
            self.fill_value,
        )?;
        let mut resampled_dataset = Dataset::new(dataset.id().clone()).with_array(resampled);
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
                RustySatError::invalid_input(
                    "bucket fraction resampling requires an f64 dataset grid",
                )
            })?;
        let categories = self.categories_for_grid(&source_grid)?;
        let resampled = resample_bucket_fraction(
            &source_grid,
            &self.source,
            destination,
            &categories,
            self.fill_value,
        )?;
        let mut resampled_dataset = Dataset::new(id).with_array(resampled);
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

pub fn resample_bucket_average(
    source_grid: &DataGrid,
    source: &SwathDefinition,
    destination: &AreaDefinition,
    fill_value: f64,
    skipna: bool,
) -> Result<DataGrid> {
    resample_bucket_with_statistic(
        source_grid,
        source,
        destination,
        BucketStatistic::Average,
        fill_value,
        skipna,
    )
}

pub fn resample_bucket_sum(
    source_grid: &DataGrid,
    source: &SwathDefinition,
    destination: &AreaDefinition,
    fill_value: f64,
    skipna: bool,
) -> Result<DataGrid> {
    resample_bucket_with_statistic(
        source_grid,
        source,
        destination,
        BucketStatistic::Sum,
        fill_value,
        skipna,
    )
}

pub fn resample_bucket_count(
    source_grid: &DataGrid,
    source: &SwathDefinition,
    destination: &AreaDefinition,
) -> Result<DataGrid> {
    resample_bucket_with_statistic(
        source_grid,
        source,
        destination,
        BucketStatistic::Count,
        f64::NAN,
        true,
    )
}

pub fn resample_bucket_fraction(
    source_grid: &DataGrid,
    source: &SwathDefinition,
    destination: &AreaDefinition,
    categories: &[f64],
    fill_value: f64,
) -> Result<DataArray<f64>> {
    validate_source_shape(source_grid, source)?;
    validate_bucket_target(destination)?;
    validate_categories(categories)?;
    let indices = bucket_indices(source, destination)?;
    let (height, width) = destination.shape();
    let bucket_count = height * width;
    let mut coordinate_counts = vec![0usize; bucket_count];
    let mut category_counts = vec![0usize; categories.len() * bucket_count];

    for (source_idx, target_idx) in indices.iter().enumerate() {
        let Some(target_idx) = target_idx else {
            continue;
        };
        coordinate_counts[*target_idx] += 1;
        let value = source_grid.values()[source_idx];
        if source_grid.is_masked(source_idx).unwrap_or(false) || !value.is_finite() {
            continue;
        }
        if let Some(category_index) = categories.iter().position(|category| *category == value) {
            category_counts[category_index * bucket_count + *target_idx] += 1;
        }
    }

    let mut values = Vec::with_capacity(categories.len() * bucket_count);
    for category_index in 0..categories.len() {
        let offset = category_index * bucket_count;
        for (bucket_idx, denominator) in coordinate_counts.iter().enumerate() {
            if *denominator == 0 {
                values.push(fill_value);
            } else {
                values.push(category_counts[offset + bucket_idx] as f64 / *denominator as f64);
            }
        }
    }
    let mut array = DataArray::from_vec_named(
        vec![categories.len(), height, width],
        ["categories", "y", "x"],
        values,
    )?;
    array.set_coordinate(
        "categories",
        Coordinate::axis("categories", categories.to_vec())?,
    )?;
    array.set_coordinate(
        "x",
        Coordinate::axis(
            "x",
            destination.iter_projection_x_coords().collect::<Vec<_>>(),
        )?,
    )?;
    array.set_coordinate(
        "y",
        Coordinate::axis(
            "y",
            destination.iter_projection_y_coords().collect::<Vec<_>>(),
        )?,
    )?;
    Ok(array)
}

pub fn resample_bucket_fraction_auto(
    source_grid: &DataGrid,
    source: &SwathDefinition,
    destination: &AreaDefinition,
    fill_value: f64,
) -> Result<DataArray<f64>> {
    let categories = discover_bucket_fraction_categories(source_grid)?;
    resample_bucket_fraction(source_grid, source, destination, &categories, fill_value)
}

fn resample_bucket_with_statistic(
    source_grid: &DataGrid,
    source: &SwathDefinition,
    destination: &AreaDefinition,
    statistic: BucketStatistic,
    fill_value: f64,
    skipna: bool,
) -> Result<DataGrid> {
    validate_source_shape(source_grid, source)?;
    validate_bucket_target(destination)?;
    let indices = bucket_indices(source, destination)?;
    let (height, width) = destination.shape();
    let mut accumulators = BucketAccumulators::new(height * width);
    accumulators.add_borrowed(source_grid, &indices, fill_value, statistic);
    add_bucket_coords(
        accumulators.finish(height, width, fill_value, statistic, skipna)?,
        Some(source_grid.coords()),
        destination,
    )
}

fn validate_categories(categories: &[f64]) -> Result<()> {
    if categories.is_empty() {
        return Err(RustySatError::invalid_input(
            "bucket fraction requires at least one category",
        ));
    }
    if categories.iter().any(|category| !category.is_finite()) {
        return Err(RustySatError::invalid_input(
            "bucket fraction categories must be finite",
        ));
    }
    Ok(())
}

fn discover_bucket_fraction_categories(source_grid: &DataGrid) -> Result<Vec<f64>> {
    let mut categories = Vec::new();
    for (idx, value) in source_grid.values().iter().copied().enumerate() {
        if source_grid.is_masked(idx).unwrap_or(false) || !value.is_finite() {
            continue;
        }
        categories.push(value);
    }
    categories.sort_by(f64::total_cmp);
    categories.dedup_by(|left, right| *left == *right);
    validate_categories(&categories)?;
    Ok(categories)
}

fn resample_bucket_owned_with_statistic(
    source_grid: DataGrid,
    source: &SwathDefinition,
    destination: &AreaDefinition,
    statistic: BucketStatistic,
    fill_value: f64,
    skipna: bool,
) -> Result<DataGrid> {
    validate_source_shape(&source_grid, source)?;
    validate_bucket_target(destination)?;
    let indices = bucket_indices(source, destination)?;
    let (values, coords, mask) = source_grid.into_parts();
    let (height, width) = destination.shape();
    let mut accumulators = BucketAccumulators::new(height * width);
    accumulators.add_values(&values, mask.as_ref(), &indices, fill_value, statistic);
    add_bucket_coords_owned(
        accumulators.finish(height, width, fill_value, statistic, skipna)?,
        Some(coords),
        destination,
    )
}

#[derive(Debug)]
struct BucketAccumulators {
    sums: Vec<f64>,
    valid_counts: Vec<usize>,
    coordinate_counts: Vec<usize>,
    invalid_seen: Vec<bool>,
}

impl BucketAccumulators {
    fn new(size: usize) -> Self {
        Self {
            sums: vec![0.0; size],
            valid_counts: vec![0; size],
            coordinate_counts: vec![0; size],
            invalid_seen: vec![false; size],
        }
    }

    fn add_borrowed(
        &mut self,
        source_grid: &DataGrid,
        indices: &[Option<usize>],
        fill_value: f64,
        statistic: BucketStatistic,
    ) {
        self.add_values(
            source_grid.values(),
            source_grid.mask(),
            indices,
            fill_value,
            statistic,
        );
    }

    fn add_values(
        &mut self,
        values: &[f64],
        mask: Option<&ValidityMask>,
        indices: &[Option<usize>],
        fill_value: f64,
        statistic: BucketStatistic,
    ) {
        for (source_idx, target_idx) in indices.iter().enumerate() {
            let Some(target_idx) = target_idx else {
                continue;
            };
            self.coordinate_counts[*target_idx] += 1;
            if statistic == BucketStatistic::Count {
                continue;
            }

            let invalid = is_invalid_sample(values[source_idx], mask, source_idx, fill_value);
            if invalid {
                self.invalid_seen[*target_idx] = true;
                continue;
            }
            self.sums[*target_idx] += values[source_idx];
            self.valid_counts[*target_idx] += 1;
        }
    }

    fn finish(
        self,
        height: usize,
        width: usize,
        fill_value: f64,
        statistic: BucketStatistic,
        skipna: bool,
    ) -> Result<DataGrid> {
        let mut output = Vec::with_capacity(height * width);
        match statistic {
            BucketStatistic::Average => {
                for idx in 0..self.sums.len() {
                    if self.valid_counts[idx] == 0 || (!skipna && self.invalid_seen[idx]) {
                        output.push(fill_value);
                    } else {
                        output.push(self.sums[idx] / self.valid_counts[idx] as f64);
                    }
                }
            }
            BucketStatistic::Sum => {
                for idx in 0..self.sums.len() {
                    if self.invalid_seen[idx] && !skipna {
                        output.push(fill_value);
                    } else if self.valid_counts[idx] == 0 {
                        output.push(0.0);
                    } else {
                        output.push(self.sums[idx]);
                    }
                }
            }
            BucketStatistic::Count => {
                output.extend(self.coordinate_counts.into_iter().map(|count| count as f64));
            }
        }
        DataGrid::new(height, width, output)
    }
}

fn is_invalid_sample(
    value: f64,
    mask: Option<&ValidityMask>,
    index: usize,
    fill_value: f64,
) -> bool {
    if mask.and_then(|mask| mask.is_masked(index)).unwrap_or(false) {
        return true;
    }
    if !value.is_finite() {
        return true;
    }
    fill_value.is_finite() && value == fill_value
}

fn bucket_indices(
    source: &SwathDefinition,
    destination: &AreaDefinition,
) -> Result<Vec<Option<usize>>> {
    let lons = source
        .lons()
        .ok_or_else(|| RustySatError::invalid_input("bucket resampling requires source lons"))?;
    let lats = source
        .lats()
        .ok_or_else(|| RustySatError::invalid_input("bucket resampling requires source lats"))?;
    let (height, width) = destination.shape();
    let extent = destination.area_extent();
    let (pixel_size_x, pixel_size_y) = destination.pixel_size();
    Ok(lons
        .iter()
        .zip(lats)
        .map(|(lon, lat)| {
            let x_idx = ((lon - extent[0]) / pixel_size_x).floor();
            let y_idx = ((extent[3] - lat) / pixel_size_y).floor();
            if x_idx < 0.0
                || y_idx < 0.0
                || x_idx >= width as f64
                || y_idx >= height as f64
                || !x_idx.is_finite()
                || !y_idx.is_finite()
            {
                None
            } else {
                Some(y_idx as usize * width + x_idx as usize)
            }
        })
        .collect())
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

fn validate_bucket_target(destination: &AreaDefinition) -> Result<()> {
    let projection = destination.projection();
    let Some(proj) = projection.get("proj").or_else(|| projection.get("proj4")) else {
        return Err(RustySatError::unsupported(
            "bucket resampling without lon/lat destination projection metadata",
        ));
    };
    if proj.contains("latlong") || proj.contains("longlat") {
        return Ok(());
    }
    Err(RustySatError::unsupported(
        "bucket resampling to non-lon/lat destination area",
    ))
}

fn adjust_bucket_attrs(dataset: &mut Dataset, statistic: BucketStatistic) -> Result<()> {
    if statistic == BucketStatistic::Count {
        dataset.insert_metadata("units", "")?;
        dataset.insert_metadata("calibration", "")?;
        dataset.insert_attr(
            "standard_name",
            MetadataValue::string("number_of_observations"),
        )?;
    }
    Ok(())
}

fn add_bucket_coords(
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

fn add_bucket_coords_owned(
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
    use super::*;
    use rusty_sat_core::DataId;

    fn area() -> AreaDefinition {
        AreaDefinition::from_parts(
            "target",
            "target",
            "target",
            BTreeMap::from([("proj".to_string(), "longlat".to_string())]),
            2,
            2,
            [0.0, 0.0, 2.0, 2.0],
        )
        .unwrap()
    }

    fn swath() -> SwathDefinition {
        SwathDefinition::from_lonlats(
            2,
            3,
            vec![0.25, 0.75, 1.25, 1.75, 3.0, 0.25],
            vec![1.75, 1.25, 1.25, 0.25, 0.5, 0.25],
        )
        .unwrap()
    }

    #[test]
    fn bucket_count_counts_coordinate_hits_inside_target() {
        let grid = DataGrid::new(2, 3, vec![1.0, 2.0, f64::NAN, 4.0, 5.0, 6.0]).unwrap();

        let resampled = resample_bucket_count(&grid, &swath(), &area()).unwrap();

        assert_eq!(resampled.values(), &[2.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn bucket_average_skips_invalid_values_by_default() {
        let grid = DataGrid::new(2, 3, vec![1.0, 2.0, f64::NAN, 4.0, 5.0, 6.0]).unwrap();

        let resampled = resample_bucket_average(&grid, &swath(), &area(), -999.0, true).unwrap();

        assert_eq!(resampled.values(), &[1.5, -999.0, 6.0, 4.0]);
    }

    #[test]
    fn bucket_average_can_fail_bucket_when_invalid_seen() {
        let grid = DataGrid::new(2, 3, vec![1.0, f64::NAN, f64::NAN, 4.0, 5.0, 6.0]).unwrap();

        let resampled = resample_bucket_average(&grid, &swath(), &area(), -999.0, false).unwrap();

        assert_eq!(resampled.values(), &[-999.0, -999.0, 6.0, 4.0]);
    }

    #[test]
    fn bucket_sum_skips_invalid_values_and_uses_zero_for_empty_buckets() {
        let grid = DataGrid::new(2, 3, vec![1.0, 2.0, f64::NAN, 4.0, 5.0, 6.0]).unwrap();

        let resampled = resample_bucket_sum(&grid, &swath(), &area(), -999.0, true).unwrap();

        assert_eq!(resampled.values(), &[3.0, 0.0, 6.0, 4.0]);
    }

    #[test]
    fn bucket_sum_skipna_false_fills_bucket_when_invalid_seen() {
        let swath = SwathDefinition::from_lonlats(1, 2, vec![0.5, 0.5], vec![1.5, 1.5]).unwrap();
        let target = AreaDefinition::from_parts(
            "target",
            "target",
            "target",
            BTreeMap::from([("proj".to_string(), "longlat".to_string())]),
            1,
            1,
            [0.0, 0.0, 2.0, 2.0],
        )
        .unwrap();
        let grid = DataGrid::new(1, 2, vec![10.0, f64::NAN]).unwrap();

        let resampled = resample_bucket_sum(&grid, &swath, &target, -999.0, false).unwrap();

        assert_eq!(resampled.values(), &[-999.0]);
    }

    #[test]
    fn bucket_fraction_returns_category_axis() {
        let swath =
            SwathDefinition::from_lonlats(1, 4, vec![0.5, 0.5, 0.5, 0.5], vec![1.5, 1.5, 1.5, 1.5])
                .unwrap();
        let target = AreaDefinition::from_parts(
            "target",
            "target",
            "target",
            BTreeMap::from([("proj".to_string(), "longlat".to_string())]),
            1,
            1,
            [0.0, 0.0, 2.0, 2.0],
        )
        .unwrap();
        let grid = DataGrid::new(1, 4, vec![0.0, 1.0, 1.0, f64::NAN]).unwrap();

        let fractions =
            resample_bucket_fraction(&grid, &swath, &target, &[0.0, 1.0], -1.0).unwrap();

        assert_eq!(fractions.shape_nd(), &[2, 1, 1]);
        assert_eq!(fractions.dims(), &["categories", "y", "x"]);
        assert_eq!(fractions.values(), &[0.25, 0.5]);
        assert_eq!(fractions.coord("categories").unwrap().values(), &[0.0, 1.0]);
        assert!(fractions.coord("x").is_some());
        assert!(fractions.coord("y").is_some());
    }

    #[test]
    fn bucket_fraction_uses_fill_for_empty_buckets() {
        let swath = SwathDefinition::from_lonlats(1, 2, vec![0.5, 0.5], vec![0.5, 0.5]).unwrap();
        let target = AreaDefinition::from_parts(
            "target",
            "target",
            "target",
            BTreeMap::from([("proj".to_string(), "longlat".to_string())]),
            1,
            2,
            [0.0, 0.0, 2.0, 1.0],
        )
        .unwrap();
        let grid = DataGrid::new(1, 2, vec![1.0, 1.0]).unwrap();

        let fractions = resample_bucket_fraction(&grid, &swath, &target, &[1.0], -1.0).unwrap();

        assert_eq!(fractions.shape_nd(), &[1, 1, 2]);
        assert_eq!(fractions.values(), &[1.0, -1.0]);
    }

    #[test]
    fn bucket_fraction_auto_discovers_sorted_finite_unmasked_categories() {
        let swath = SwathDefinition::from_lonlats(1, 5, vec![0.5; 5], vec![1.5; 5]).unwrap();
        let target = AreaDefinition::from_parts(
            "target",
            "target",
            "target",
            BTreeMap::from([("proj".to_string(), "longlat".to_string())]),
            1,
            1,
            [0.0, 0.0, 2.0, 2.0],
        )
        .unwrap();
        let grid = DataGrid::new(1, 5, vec![2.0, 1.0, 2.0, f64::NAN, 3.0])
            .unwrap()
            .with_mask(ValidityMask::from_masked_flags([
                false, false, false, false, true,
            ]))
            .unwrap();

        let fractions = resample_bucket_fraction_auto(&grid, &swath, &target, -1.0).unwrap();

        assert_eq!(fractions.shape_nd(), &[2, 1, 1]);
        assert_eq!(fractions.coord("categories").unwrap().values(), &[1.0, 2.0]);
        assert_eq!(fractions.values(), &[0.2, 0.4]);
    }

    #[test]
    fn bucket_fraction_rejects_empty_or_non_finite_categories() {
        let grid = DataGrid::new(2, 3, vec![1.0, 2.0, f64::NAN, 4.0, 5.0, 6.0]).unwrap();

        assert!(resample_bucket_fraction(&grid, &swath(), &area(), &[], -1.0).is_err());
        assert!(resample_bucket_fraction(&grid, &swath(), &area(), &[f64::NAN], -1.0).is_err());
    }

    #[test]
    fn bucket_fraction_auto_rejects_when_no_categories_exist() {
        let grid = DataGrid::new(2, 3, vec![f64::NAN; 6]).unwrap();

        let err = resample_bucket_fraction_auto(&grid, &swath(), &area(), -1.0).unwrap_err();

        assert!(err.to_string().contains("at least one category"));
    }

    #[test]
    fn bucket_resampler_preserves_metadata_and_updates_count_attrs() {
        let grid = DataGrid::new(2, 3, vec![1.0, 2.0, f64::NAN, 4.0, 5.0, 6.0])
            .unwrap()
            .with_coordinate("time", Coordinate::scalar(1.0))
            .unwrap();
        let id = DataId::new("obs").unwrap();
        let mut dataset = Dataset::new(id.clone()).with_data(grid);
        dataset.insert_metadata("sensor", "test").unwrap();
        let resampler = BucketResampler::count(swath());

        let output = resampler.resample(&dataset, &area()).unwrap();

        assert_eq!(output.id(), &id);
        assert_eq!(output.metadata().get("sensor"), Some(&"test".to_string()));
        assert_eq!(
            output.attr("standard_name"),
            Some(&MetadataValue::string("number_of_observations"))
        );
        assert_eq!(
            output.metadata().get("resampler"),
            Some(&"bucket_count".to_string())
        );
        assert!(output.data().unwrap().coord("time").is_some());
        assert!(output.data().unwrap().coord("x").is_some());
        assert!(output.data().unwrap().coord("y").is_some());
    }

    #[test]
    fn bucket_owned_matches_borrowed_average() {
        let grid = DataGrid::new(2, 3, vec![1.0, 2.0, f64::NAN, 4.0, 5.0, 6.0]).unwrap();
        let borrowed = resample_bucket_average(&grid, &swath(), &area(), -999.0, true).unwrap();
        let owned = resample_bucket_owned_with_statistic(
            grid,
            &swath(),
            &area(),
            BucketStatistic::Average,
            -999.0,
            true,
        )
        .unwrap();

        assert_eq!(borrowed, owned);
    }

    #[test]
    fn bucket_average_skipna_true_averages_valid_values_in_mixed_bucket() {
        // Two points land in same bucket: one valid (10.0), one NaN.
        let swath = SwathDefinition::from_lonlats(1, 2, vec![0.5, 0.5], vec![1.5, 1.5]).unwrap();
        let target = AreaDefinition::from_parts(
            "target",
            "target",
            "target",
            BTreeMap::from([("proj".to_string(), "longlat".to_string())]),
            1,
            1,
            [0.0, 0.0, 2.0, 2.0],
        )
        .unwrap();
        let grid = DataGrid::new(1, 2, vec![10.0, f64::NAN]).unwrap();

        let resampled = resample_bucket_average(&grid, &swath, &target, -999.0, true).unwrap();

        assert_eq!(resampled.values(), &[10.0]);
        assert!(resampled.mask().is_none());
    }

    #[test]
    fn bucket_resampler_through_trait_uses_correct_statistic_name() {
        let grid = DataGrid::new(2, 3, vec![1.0, 2.0, f64::NAN, 4.0, 5.0, 6.0]).unwrap();
        let id = DataId::new("obs").unwrap();
        let dataset = Dataset::new(id.clone()).with_data(grid);
        let resampler = BucketResampler::average(swath());

        let output = resampler.resample(&dataset, &area()).unwrap();

        assert_eq!(output.id(), &id);
        assert_eq!(
            output.metadata().get("resampler"),
            Some(&"bucket_avg".to_string())
        );
    }

    #[test]
    fn bucket_resampler_owned_through_trait_produces_same_output() {
        let id = DataId::new("obs").unwrap();
        let ds1 = Dataset::new(id.clone())
            .with_data(DataGrid::new(2, 3, vec![1.0, 2.0, f64::NAN, 4.0, 5.0, 6.0]).unwrap());
        let ds2 = Dataset::new(id)
            .with_data(DataGrid::new(2, 3, vec![1.0, 2.0, f64::NAN, 4.0, 5.0, 6.0]).unwrap());
        let resampler = BucketResampler::sum(swath()).with_fill_value(-999.0);

        let borrowed = resampler.resample(&ds1, &area()).unwrap();
        let owned = resampler.resample_owned(ds2, &area()).unwrap();

        assert_eq!(
            borrowed.data().unwrap().values(),
            owned.data().unwrap().values()
        );
    }

    #[test]
    fn bucket_factory_methods_create_correct_statistics() {
        assert_eq!(
            BucketResampler::average(swath()).statistic(),
            BucketStatistic::Average
        );
        assert_eq!(
            BucketResampler::sum(swath()).statistic(),
            BucketStatistic::Sum
        );
        assert_eq!(
            BucketResampler::count(swath()).statistic(),
            BucketStatistic::Count
        );
    }

    #[test]
    fn bucket_fraction_ignores_values_not_matching_any_category() {
        let swath =
            SwathDefinition::from_lonlats(1, 3, vec![0.5, 0.5, 0.5], vec![1.5, 1.5, 1.5]).unwrap();
        let target = AreaDefinition::from_parts(
            "target",
            "target",
            "target",
            BTreeMap::from([("proj".to_string(), "longlat".to_string())]),
            1,
            1,
            [0.0, 0.0, 2.0, 2.0],
        )
        .unwrap();
        let grid = DataGrid::new(1, 3, vec![0.0, 99.0, 1.0]).unwrap();

        let fractions =
            resample_bucket_fraction(&grid, &swath, &target, &[0.0, 1.0], -1.0).unwrap();

        assert_eq!(fractions.values(), &[1.0 / 3.0, 1.0 / 3.0]);
    }

    #[test]
    fn bucket_fraction_fills_all_when_all_points_masked() {
        let swath = SwathDefinition::from_lonlats(1, 2, vec![0.5, 0.5], vec![1.5, 1.5]).unwrap();
        let target = AreaDefinition::from_parts(
            "target",
            "target",
            "target",
            BTreeMap::from([("proj".to_string(), "longlat".to_string())]),
            1,
            1,
            [0.0, 0.0, 2.0, 2.0],
        )
        .unwrap();
        let grid = DataGrid::new(1, 2, vec![0.0, 1.0])
            .unwrap()
            .with_mask(ValidityMask::from_masked_flags([true, true]))
            .unwrap();

        let fractions =
            resample_bucket_fraction(&grid, &swath, &target, &[0.0, 1.0], -1.0).unwrap();

        // denominator = 2 (both points counted), numerator = 0 (both masked, skipped)
        assert_eq!(fractions.values(), &[0.0, 0.0]);
    }
}
