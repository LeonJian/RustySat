//! Pyresample-style image container helpers.
//!
//! Reference behavior inspected before implementation:
//! - `deps/pyresample/pyresample/image.py`
//! - `deps/pyresample/pyresample/grid.py`
//!
//! Pyresample's `ImageContainer*` classes bind an image array to a geometry and
//! return another container after resampling. Rusty Sat keeps the same concept,
//! but stores a typed `Dataset` and delegates the actual sampling to the
//! existing Rust-native resampling pipeline.

use crate::pipeline::{
    resample_dataset_from_geometry, resample_dataset_owned_from_geometry, ResampleOptions,
    SourceGeometry,
};
use crate::source_geometry::{source_geometry_from_dataset, with_area_attr};
use crate::{
    sample_any_from_linesample, sample_grid_from_linesample, LineSampleFillValue, LineSampleGrid,
};
use crate::{AreaDefinition, SwathDefinition};
use rusty_sat_core::{AnyDataArray, DataGrid, Dataset, Result, RustySatError};

#[derive(Debug, Clone, PartialEq)]
pub struct ImageContainer {
    dataset: Dataset,
    source: SourceGeometry,
}

impl ImageContainer {
    pub fn new(dataset: Dataset, source: SourceGeometry) -> Result<Self> {
        validate_dataset_geometry(&dataset, &source)?;
        Ok(Self { dataset, source })
    }

    pub fn from_area(dataset: Dataset, area: AreaDefinition) -> Result<Self> {
        Self::new(dataset, SourceGeometry::Area(area))
    }

    pub fn from_swath(dataset: Dataset, swath: SwathDefinition) -> Result<Self> {
        Self::new(dataset, SourceGeometry::Swath(swath))
    }

    pub fn from_dataset_attrs(dataset: Dataset) -> Result<Self> {
        let source = source_geometry_from_dataset(&dataset)?;
        Self::new(dataset, source)
    }

    pub fn dataset(&self) -> &Dataset {
        &self.dataset
    }

    pub fn source_geometry(&self) -> &SourceGeometry {
        &self.source
    }

    pub fn into_dataset(self) -> Dataset {
        self.dataset
    }

    pub fn into_parts(self) -> (Dataset, SourceGeometry) {
        (self.dataset, self.source)
    }

    pub fn get_array_from_linesample(
        &self,
        linesample: &LineSampleGrid,
        fill_value: f64,
    ) -> Result<DataGrid> {
        let SourceGeometry::Area(_) = self.source else {
            return Err(RustySatError::unsupported(
                "line/sample indexing from non-area image container geometry",
            ));
        };
        let source = self.dataset.data().ok_or_else(|| {
            RustySatError::invalid_input(format!(
                "image container dataset '{}' does not have a f64 grid",
                self.dataset.id().name()
            ))
        })?;
        sample_grid_from_linesample(source, linesample, fill_value, false)
    }

    pub fn get_array_from_linesample_masked_missing(
        &self,
        linesample: &LineSampleGrid,
    ) -> Result<DataGrid> {
        let SourceGeometry::Area(_) = self.source else {
            return Err(RustySatError::unsupported(
                "line/sample indexing from non-area image container geometry",
            ));
        };
        let source = self.dataset.data().ok_or_else(|| {
            RustySatError::invalid_input(format!(
                "image container dataset '{}' does not have a f64 grid",
                self.dataset.id().name()
            ))
        })?;
        sample_grid_from_linesample(source, linesample, f64::NAN, true)
    }

    pub fn get_any_array_from_linesample(
        &self,
        linesample: &LineSampleGrid,
        fill_value: LineSampleFillValue,
    ) -> Result<AnyDataArray> {
        let SourceGeometry::Area(_) = self.source else {
            return Err(RustySatError::unsupported(
                "line/sample indexing from non-area image container geometry",
            ));
        };
        let source = self.dataset.array().ok_or_else(|| {
            RustySatError::invalid_input(format!(
                "image container dataset '{}' has no array data",
                self.dataset.id().name()
            ))
        })?;
        sample_any_from_linesample(source, linesample, fill_value, false)
    }

    pub fn get_any_array_from_linesample_masked_missing(
        &self,
        linesample: &LineSampleGrid,
        fill_value: LineSampleFillValue,
    ) -> Result<AnyDataArray> {
        let SourceGeometry::Area(_) = self.source else {
            return Err(RustySatError::unsupported(
                "line/sample indexing from non-area image container geometry",
            ));
        };
        let source = self.dataset.array().ok_or_else(|| {
            RustySatError::invalid_input(format!(
                "image container dataset '{}' has no array data",
                self.dataset.id().name()
            ))
        })?;
        sample_any_from_linesample(source, linesample, fill_value, true)
    }

    pub fn resample(&self, destination: &AreaDefinition, options: ResampleOptions) -> Result<Self> {
        let dataset = resample_dataset_from_geometry(
            &self.dataset,
            self.source.clone(),
            destination,
            options,
        )?;
        Self::from_area(with_area_attr(dataset, destination)?, destination.clone())
    }

    pub fn resample_owned(
        self,
        destination: &AreaDefinition,
        options: ResampleOptions,
    ) -> Result<Self> {
        let dataset =
            resample_dataset_owned_from_geometry(self.dataset, self.source, destination, options)?;
        Self::from_area(with_area_attr(dataset, destination)?, destination.clone())
    }

    pub fn resample_nearest(
        &self,
        destination: &AreaDefinition,
        radius_of_influence: Option<f64>,
        fill_value: f64,
    ) -> Result<Self> {
        let mut options = ResampleOptions::nearest_area().with_fill_value(fill_value);
        if let Some(radius_of_influence) = radius_of_influence {
            options = options.with_radius_of_influence(radius_of_influence)?;
        }
        self.resample(destination, options)
    }

    pub fn resample_nearest_owned(
        self,
        destination: &AreaDefinition,
        radius_of_influence: Option<f64>,
        fill_value: f64,
    ) -> Result<Self> {
        let mut options = ResampleOptions::nearest_area().with_fill_value(fill_value);
        if let Some(radius_of_influence) = radius_of_influence {
            options = options.with_radius_of_influence(radius_of_influence)?;
        }
        self.resample_owned(destination, options)
    }
}

fn validate_dataset_geometry(dataset: &Dataset, source: &SourceGeometry) -> Result<()> {
    let array = dataset.array().ok_or_else(|| {
        RustySatError::invalid_input(format!(
            "image container dataset '{}' has no array data",
            dataset.id().name()
        ))
    })?;
    let shape = array.shape_yx()?;
    let expected = match source {
        SourceGeometry::Area(area) => area.shape(),
        SourceGeometry::Swath(swath) => swath.shape(),
    };
    if shape != expected {
        return Err(RustySatError::invalid_input(format!(
            "image container data y/x shape {:?} does not match geometry shape {:?}",
            shape, expected
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;
    use rusty_sat_core::{Coordinate, DataGrid, DataId};
    use std::collections::BTreeMap;

    fn area(id: &str, height: usize, width: usize, extent: [f64; 4]) -> AreaDefinition {
        AreaDefinition::from_parts(
            id,
            id,
            id,
            [("proj".to_string(), "longlat".to_string())]
                .into_iter()
                .collect::<BTreeMap<_, _>>(),
            height,
            width,
            extent,
        )
        .unwrap()
    }

    fn dataset(height: usize, width: usize) -> Dataset {
        Dataset::new(DataId::new("image").unwrap()).with_data(
            DataGrid::new(
                height,
                width,
                (0..height * width).map(|value| value as f64).collect(),
            )
            .unwrap(),
        )
    }

    #[test]
    fn image_container_validates_source_shape() {
        let source = area("source", 2, 2, [0.0, 0.0, 2.0, 2.0]);
        let container = ImageContainer::from_area(dataset(2, 2), source.clone()).unwrap();
        assert_eq!(container.source_geometry(), &SourceGeometry::Area(source));

        let err = ImageContainer::from_area(dataset(1, 2), area("bad", 2, 2, [0.0, 0.0, 2.0, 2.0]))
            .unwrap_err();
        assert!(err.to_string().contains("does not match geometry shape"));
    }

    #[test]
    fn image_container_resample_returns_destination_container() {
        let source = area("source", 2, 2, [0.0, 0.0, 2.0, 2.0]);
        let destination = area("destination", 1, 1, [0.0, 1.0, 1.0, 2.0]);
        let container = ImageContainer::from_area(dataset(2, 2), source).unwrap();

        let output = container
            .resample(&destination, ResampleOptions::nearest_area())
            .unwrap();

        assert_eq!(output.dataset().data().unwrap().values(), &[0.0]);
        assert_eq!(
            output.source_geometry(),
            &SourceGeometry::Area(destination.clone())
        );
        assert!(output.dataset().attr("area").is_some());
    }

    #[test]
    fn image_container_owned_resample_matches_borrowed() {
        let source = area("source", 2, 2, [0.0, 0.0, 2.0, 2.0]);
        let destination = area("destination", 1, 1, [0.0, 1.0, 1.0, 2.0]);
        let borrowed = ImageContainer::from_area(dataset(2, 2), source.clone())
            .unwrap()
            .resample(&destination, ResampleOptions::nearest_area())
            .unwrap();
        let owned = ImageContainer::from_area(dataset(2, 2), source)
            .unwrap()
            .resample_owned(&destination, ResampleOptions::nearest_area())
            .unwrap();

        assert_eq!(borrowed, owned);
    }

    #[test]
    fn image_container_can_infer_swath_from_dataset_coordinates() {
        let array = DataGrid::new(2, 2, vec![1.0, 2.0, 3.0, 4.0])
            .unwrap()
            .with_coordinate(
                "longitude",
                Coordinate::new(["y", "x"], vec![0.25, 1.25, 0.25, 1.25]).unwrap(),
            )
            .unwrap()
            .with_coordinate(
                "latitude",
                Coordinate::new(["y", "x"], vec![1.25, 1.25, 0.25, 0.25]).unwrap(),
            )
            .unwrap();
        let dataset = Dataset::new(DataId::new("image").unwrap()).with_data(array);

        let container = ImageContainer::from_dataset_attrs(dataset).unwrap();

        let SourceGeometry::Swath(swath) = container.source_geometry() else {
            panic!("expected swath geometry");
        };
        assert_eq!(swath.shape(), (2, 2));
    }

    #[test]
    fn resample_nearest_convenience_matches_explicit() {
        let source = area("source", 2, 2, [0.0, 0.0, 2.0, 2.0]);
        let destination = area("destination", 1, 1, [0.0, 1.0, 1.0, 2.0]);
        let container = ImageContainer::from_area(dataset(2, 2), source).unwrap();

        let explicit = container
            .resample(
                &destination,
                ResampleOptions::nearest_area().with_fill_value(-999.0),
            )
            .unwrap();
        let convenience = container
            .resample_nearest(&destination, None, -999.0)
            .unwrap();

        assert_eq!(explicit, convenience);
    }

    #[test]
    fn image_container_samples_area_linesample() {
        let source = area("source", 2, 2, [0.0, 0.0, 2.0, 2.0]);
        let linesample = LineSampleGrid::new(2, 2, [0, 1, -1, 0], [0, 1, 0, 4]).unwrap();
        let container = ImageContainer::from_area(dataset(2, 2), source).unwrap();

        let output = container
            .get_array_from_linesample(&linesample, -999.0)
            .unwrap();

        assert_eq!(output.values(), &[0.0, 3.0, -999.0, -999.0]);
    }

    #[test]
    fn image_container_samples_runtime_typed_linesample() {
        let source = area("source", 2, 2, [0.0, 0.0, 2.0, 2.0]);
        let dataset = Dataset::new(DataId::new("image").unwrap()).with_array(
            rusty_sat_core::DataArray::<u16>::from_vec_named(
                vec![2, 2],
                ["y", "x"],
                vec![10, 20, 30, 40],
            )
            .unwrap(),
        );
        let linesample = LineSampleGrid::new(1, 2, [0, -1], [1, 0]).unwrap();
        let container = ImageContainer::from_area(dataset, source).unwrap();

        let output = container
            .get_any_array_from_linesample(&linesample, LineSampleFillValue::u16(999))
            .unwrap();

        assert_eq!(output.values_as_f64(), vec![20.0, 999.0]);
    }
}
