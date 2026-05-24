//! Dataset attribute helpers for resolving resampling source geometry.
//!
//! Reference behavior inspected before implementation:
//! - `satpy/satpy/scene.py` (`Scene.resample` / `_reduce_data`)
//! - Satpy/xarray convention that dataset attrs carry an `area` object when
//!   gridded data already has a known source area.

use crate::pipeline::{
    resample_dataset, resample_dataset_cached, resample_dataset_from_geometry,
    resample_dataset_from_geometry_cached, resample_dataset_owned, resample_dataset_owned_cached,
    SourceGeometry,
};
use crate::{AreaDefinition, ResampleOptions, ResamplerCache, SwathDefinition};
use rusty_sat_core::{Dataset, MetadataValue, Result, RustySatError};
use std::collections::BTreeMap;

pub const AREA_ATTR_KEY: &str = "area";

pub fn area_to_metadata_value(area: &AreaDefinition) -> Result<MetadataValue> {
    Ok(MetadataValue::map([
        ("type", MetadataValue::string("area")),
        ("id", MetadataValue::string(area.id())),
        ("description", MetadataValue::string(area.description())),
        ("proj_id", MetadataValue::string(area.proj_id())),
        (
            "projection",
            MetadataValue::Map(
                area.projection()
                    .iter()
                    .map(|(key, value)| (key.clone(), MetadataValue::string(value.clone())))
                    .collect(),
            ),
        ),
        (
            "height",
            MetadataValue::Integer(usize_to_i64(area.height())?),
        ),
        ("width", MetadataValue::Integer(usize_to_i64(area.width())?)),
        (
            "area_extent",
            MetadataValue::List(
                area.area_extent()
                    .into_iter()
                    .map(MetadataValue::float)
                    .collect::<Result<Vec<_>>>()?,
            ),
        ),
    ]))
}

pub fn area_from_metadata_value(value: &MetadataValue) -> Result<AreaDefinition> {
    let MetadataValue::Map(map) = value else {
        return Err(RustySatError::invalid_input("area attr must be a map"));
    };
    if let Some(kind) = optional_string(map, "type")? {
        if kind != "area" {
            return Err(RustySatError::invalid_input(format!(
                "unsupported source geometry attr type '{kind}'"
            )));
        }
    }
    AreaDefinition::from_parts(
        required_string(map, "id")?,
        required_string(map, "description")?,
        required_string(map, "proj_id")?,
        required_string_map(map, "projection")?,
        required_usize(map, "height")?,
        required_usize(map, "width")?,
        required_f64_array4(map, "area_extent")?,
    )
}

pub fn set_dataset_area_attr(dataset: &mut Dataset, area: &AreaDefinition) -> Result<()> {
    dataset.insert_attr(AREA_ATTR_KEY, area_to_metadata_value(area)?)
}

pub fn with_area_attr(mut dataset: Dataset, area: &AreaDefinition) -> Result<Dataset> {
    set_dataset_area_attr(&mut dataset, area)?;
    Ok(dataset)
}

pub fn source_geometry_from_dataset(dataset: &Dataset) -> Result<SourceGeometry> {
    if let Some(area) = dataset.attr(AREA_ATTR_KEY) {
        return Ok(SourceGeometry::Area(area_from_metadata_value(area)?));
    }
    swath_geometry_from_dataset_coords(dataset).map(SourceGeometry::Swath)
}

pub fn resample_dataset_from_attrs(
    dataset: &Dataset,
    destination: &AreaDefinition,
    options: ResampleOptions,
) -> Result<Dataset> {
    match source_geometry_from_dataset(dataset)? {
        SourceGeometry::Area(area) => resample_dataset(dataset, area, destination, options),
        SourceGeometry::Swath(swath) => resample_dataset_from_geometry(
            dataset,
            SourceGeometry::Swath(swath),
            destination,
            options,
        ),
    }
}

pub fn resample_dataset_from_attrs_cached(
    cache: &mut ResamplerCache,
    dataset: &Dataset,
    destination: &AreaDefinition,
    options: ResampleOptions,
) -> Result<Dataset> {
    match source_geometry_from_dataset(dataset)? {
        SourceGeometry::Area(area) => {
            resample_dataset_cached(cache, dataset, area, destination, options)
        }
        SourceGeometry::Swath(swath) => resample_dataset_from_geometry_cached(
            cache,
            dataset,
            SourceGeometry::Swath(swath),
            destination,
            options,
        ),
    }
}

pub fn resample_dataset_owned_from_attrs(
    dataset: Dataset,
    destination: &AreaDefinition,
    options: ResampleOptions,
) -> Result<Dataset> {
    match source_geometry_from_dataset(&dataset)? {
        SourceGeometry::Area(area) => resample_dataset_owned(dataset, area, destination, options),
        SourceGeometry::Swath(swath) => crate::pipeline::resample_dataset_owned_from_geometry(
            dataset,
            SourceGeometry::Swath(swath),
            destination,
            options,
        ),
    }
}

pub fn resample_dataset_owned_from_attrs_cached(
    cache: &mut ResamplerCache,
    dataset: Dataset,
    destination: &AreaDefinition,
    options: ResampleOptions,
) -> Result<Dataset> {
    match source_geometry_from_dataset(&dataset)? {
        SourceGeometry::Area(area) => {
            resample_dataset_owned_cached(cache, dataset, area, destination, options)
        }
        SourceGeometry::Swath(swath) => {
            crate::pipeline::resample_dataset_owned_from_geometry_cached(
                cache,
                dataset,
                SourceGeometry::Swath(swath),
                destination,
                options,
            )
        }
    }
}

fn swath_geometry_from_dataset_coords(dataset: &Dataset) -> Result<SwathDefinition> {
    let array = dataset.array().ok_or_else(|| {
        RustySatError::not_found(format!(
            "source geometry attrs or lon/lat coordinates for dataset '{}'",
            dataset.id().name()
        ))
    })?;
    let (height, width) = array.shape_yx()?;
    let lons = coordinate_values(array, &["longitude", "lon", "lons"])?;
    let lats = coordinate_values(array, &["latitude", "lat", "lats"])?;
    SwathDefinition::from_lonlats(height, width, lons.to_vec(), lats.to_vec())
}

fn coordinate_values<'a>(
    array: &'a rusty_sat_core::AnyDataArray,
    names: &[&str],
) -> Result<&'a [f64]> {
    names
        .iter()
        .find_map(|name| array.coord(name))
        .map(|coord| coord.values())
        .ok_or_else(|| RustySatError::not_found(format!("coordinate '{}'", names[0])))
}

fn required_string(map: &BTreeMap<String, MetadataValue>, key: &str) -> Result<String> {
    map.get(key)
        .and_then(MetadataValue::as_str)
        .map(ToOwned::to_owned)
        .ok_or_else(|| RustySatError::invalid_input(format!("area attr missing string '{key}'")))
}

fn optional_string(map: &BTreeMap<String, MetadataValue>, key: &str) -> Result<Option<String>> {
    let Some(value) = map.get(key) else {
        return Ok(None);
    };
    value
        .as_str()
        .map(|value| Some(value.to_string()))
        .ok_or_else(|| RustySatError::invalid_input(format!("area attr '{key}' must be a string")))
}

fn required_string_map(
    map: &BTreeMap<String, MetadataValue>,
    key: &str,
) -> Result<BTreeMap<String, String>> {
    let Some(MetadataValue::Map(values)) = map.get(key) else {
        return Err(RustySatError::invalid_input(format!(
            "area attr missing map '{key}'"
        )));
    };
    values
        .iter()
        .map(|(key, value)| {
            value
                .as_str()
                .map(|value| (key.clone(), value.to_string()))
                .ok_or_else(|| {
                    RustySatError::invalid_input("area projection values must be strings")
                })
        })
        .collect()
}

fn required_usize(map: &BTreeMap<String, MetadataValue>, key: &str) -> Result<usize> {
    let Some(MetadataValue::Integer(value)) = map.get(key) else {
        return Err(RustySatError::invalid_input(format!(
            "area attr missing integer '{key}'"
        )));
    };
    usize::try_from(*value).map_err(|_| {
        RustySatError::invalid_input(format!("area attr '{key}' must be non-negative"))
    })
}

fn required_f64_array4(map: &BTreeMap<String, MetadataValue>, key: &str) -> Result<[f64; 4]> {
    let Some(MetadataValue::List(values)) = map.get(key) else {
        return Err(RustySatError::invalid_input(format!(
            "area attr missing list '{key}'"
        )));
    };
    if values.len() != 4 {
        return Err(RustySatError::invalid_input(format!(
            "area attr '{key}' must have four values"
        )));
    }
    let mut result = [0.0; 4];
    for (index, value) in values.iter().enumerate() {
        result[index] = metadata_f64(value, key)?;
    }
    Ok(result)
}

fn metadata_f64(value: &MetadataValue, key: &str) -> Result<f64> {
    match value {
        MetadataValue::Integer(value) => Ok(*value as f64),
        MetadataValue::Float(value) => Ok(value.get()),
        _ => Err(RustySatError::invalid_input(format!(
            "area attr '{key}' values must be numeric"
        ))),
    }
}

fn usize_to_i64(value: usize) -> Result<i64> {
    i64::try_from(value).map_err(|_| {
        RustySatError::invalid_input("area dimension is too large for metadata encoding")
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rusty_sat_core::{Coordinate, DataGrid, DataId};

    fn area(id: &str, height: usize, width: usize, extent: [f64; 4]) -> AreaDefinition {
        AreaDefinition::from_parts(
            id,
            id,
            "latlon",
            [("proj".to_string(), "longlat".to_string())]
                .into_iter()
                .collect(),
            height,
            width,
            extent,
        )
        .unwrap()
    }

    #[test]
    fn area_attr_round_trips_source_geometry() {
        let source = area("source", 2, 2, [0.0, 0.0, 2.0, 2.0]);
        let dataset = with_area_attr(
            Dataset::new(DataId::new("image").unwrap())
                .with_data(DataGrid::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap()),
            &source,
        )
        .unwrap();

        let SourceGeometry::Area(decoded) = source_geometry_from_dataset(&dataset).unwrap() else {
            panic!("expected area geometry");
        };
        assert_eq!(decoded, source);
    }

    #[test]
    fn resample_dataset_from_attrs_uses_area_metadata() {
        let source = area("source", 2, 2, [0.0, 0.0, 2.0, 2.0]);
        let destination = area("destination", 1, 1, [0.0, 1.0, 1.0, 2.0]);
        let dataset = with_area_attr(
            Dataset::new(DataId::new("image").unwrap())
                .with_data(DataGrid::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap()),
            &source,
        )
        .unwrap();

        let output = resample_dataset_from_attrs(
            &dataset,
            &destination,
            ResampleOptions::nearest_area().with_data_reduction(),
        )
        .unwrap();

        assert_eq!(output.data().unwrap().values(), &[1.0]);
        assert_eq!(
            output.metadata().get("resampler"),
            Some(&"nearest_area".to_string())
        );
    }

    #[test]
    fn source_geometry_falls_back_to_lonlat_coordinates_as_swath() {
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

        let SourceGeometry::Swath(swath) = source_geometry_from_dataset(&dataset).unwrap() else {
            panic!("expected swath geometry");
        };

        assert_eq!(swath.shape(), (2, 2));
        assert_eq!(swath.lons().unwrap(), &[0.25, 1.25, 0.25, 1.25]);
    }

    #[test]
    fn source_geometry_reports_missing_attrs_and_coordinates() {
        let dataset = Dataset::new(DataId::new("image").unwrap())
            .with_data(DataGrid::new(1, 1, vec![1.0]).unwrap());

        let err = source_geometry_from_dataset(&dataset).unwrap_err();
        assert!(err.to_string().contains("coordinate 'longitude'"));
    }
}
