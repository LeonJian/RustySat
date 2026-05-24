//! Geometry and resampling foundations.

pub mod area;
pub mod bilinear;
pub mod bucket;
pub mod crs;
pub mod data_reduce;
pub mod ewa;
pub mod geometry;
pub mod image_container;
pub mod linesample;
pub mod native;
mod nd_utils;
pub mod nearest;
pub mod neighbour;
pub mod pipeline;
pub mod scene;
pub mod slicer;
pub mod source_geometry;
pub mod spatial_index;
pub mod swath;

pub use area::{
    load_area_from_file, load_area_from_str, load_areas_from_str, AreaDefinition, PixelResolution,
};
pub use bilinear::{
    resample_area_bilinear, resample_area_bilinear_masked_missing, resample_area_bilinear_owned,
    BilinearAreaResampler, BilinearMissingPolicy,
};
pub use bucket::{
    resample_bucket_average, resample_bucket_count, resample_bucket_fraction,
    resample_bucket_fraction_auto, resample_bucket_sum, BucketFractionResampler, BucketResampler,
    BucketStatistic,
};
pub use crs::{Coordinate2D, CrsSource, ProjCrs, ProjectionBackendStrategy, TransformDirection};
pub use data_reduce::{
    get_valid_index_from_lonlat_boundaries, get_valid_index_from_lonlat_grid,
    lonlat_grid_boundaries, LonLatBoundaries,
};
pub use ewa::{resample_swath_ewa, resample_swath_ewa_owned, EwaOptions, EwaResampler};
pub use geometry::{
    CoordinateDefinition, GeometryDefinition, GeometryKind, GridDefinition, ProjectionDefinition,
};
pub use image_container::ImageContainer;
pub use linesample::{
    get_image_from_linesample, get_image_from_linesample_masked_missing,
    sample_any_from_linesample, sample_array_from_linesample, sample_grid_from_linesample,
    LineSampleFillValue, LineSampleGrid,
};
pub use native::{
    native_aggregate_mean_2d, native_aggregate_mean_2d_owned, native_aggregate_mean_yx,
    native_aggregate_mean_yx_owned, native_repeat_2d, native_repeat_2d_owned, native_repeat_yx,
    native_repeat_yx_owned, native_repeat_yx_typed, native_repeat_yx_typed_owned,
    native_resample_2d, native_resample_2d_owned, native_resample_any_yx,
    native_resample_any_yx_owned, native_resample_yx, native_resample_yx_owned, NativeResampler,
};
pub use nearest::{
    resample_area_nearest, resample_area_nearest_lazy, resample_area_nearest_owned,
    resample_swath_nearest, NearestAreaResampler,
};
pub use neighbour::{
    gaussian_weight, get_area_neighbour_info, get_area_neighbour_info_with_neighbours,
    get_swath_neighbour_info, sample_nearest_from_neighbour_info,
    sample_nearest_from_neighbour_info_owned, sample_weighted_from_neighbour_info,
    sample_weighted_from_neighbour_info_owned, Neighbour, NeighbourInfo, SampleMissingPolicy,
};
pub use pipeline::{
    prepare_resampler, prepare_resampler_for_geometry, resample_area_dataset_reduced,
    resample_area_dataset_reduced_cached, resample_area_dataset_reduced_owned,
    resample_area_dataset_reduced_owned_cached, resample_dataset, resample_dataset_cached,
    resample_dataset_from_geometry, resample_dataset_from_geometry_cached, resample_dataset_owned,
    resample_dataset_owned_cached, resample_dataset_owned_from_geometry,
    resample_dataset_owned_from_geometry_cached, PreparedResampler, ResampleOptions,
    ResamplerCache, ResamplerMethod, SourceGeometry,
};
pub use scene::SceneResampleExt;
pub use slicer::{
    crop_source_area, get_area_slices, get_area_slices_with_divisibility, reduce_area_dataset,
    reduce_area_dataset_owned, reduce_area_dataset_owned_with_divisibility,
    reduce_area_dataset_with_divisibility, slice_area, slice_dataset_yx, slice_dataset_yx_owned,
    AreaCrop, AreaDataReduction, AreaSlice,
};
pub use source_geometry::{
    area_from_metadata_value, area_to_metadata_value, resample_dataset_from_attrs,
    resample_dataset_from_attrs_cached, resample_dataset_owned_from_attrs,
    resample_dataset_owned_from_attrs_cached, set_dataset_area_attr, source_geometry_from_dataset,
    with_area_attr, AREA_ATTR_KEY,
};
pub use spatial_index::{KdPointIndex2D, NearestPoint, Point2D};
pub use swath::{load_swath_from_str, load_swaths_from_str, SwathDefinition};

use rusty_sat_core::{Dataset, Result, RustySatError};

pub trait Resampler {
    fn name(&self) -> &str;

    fn resample(&self, _dataset: &Dataset, _destination: &AreaDefinition) -> Result<Dataset> {
        Err(RustySatError::unsupported(format!(
            "{} resampling",
            self.name()
        )))
    }

    fn resample_owned(&self, dataset: Dataset, destination: &AreaDefinition) -> Result<Dataset> {
        self.resample(&dataset, destination)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constructs_area_and_swath() {
        let area = AreaDefinition::new("test_area", 10, 20).unwrap();
        assert_eq!(area.id(), "test_area");
        assert_eq!(area.shape(), (10, 20));
        assert_eq!(GeometryDefinition::kind(&area), GeometryKind::Area);
        assert_eq!(GeometryDefinition::shape(&area), vec![10, 20]);
        assert_eq!(GeometryDefinition::ndim(&area), 2);
        assert_eq!(GeometryDefinition::size(&area), 200);
        assert!(!GeometryDefinition::is_empty(&area));

        let swath = SwathDefinition::new(5, 6).unwrap();
        assert_eq!(swath.shape(), (5, 6));
        assert_eq!(GeometryDefinition::kind(&swath), GeometryKind::Swath);
        assert_eq!(GeometryDefinition::shape(&swath), vec![5, 6]);
        assert_eq!(GeometryDefinition::ndim(&swath), 2);
        assert_eq!(GeometryDefinition::size(&swath), 30);
        assert!(!GeometryDefinition::is_empty(&swath));
    }
}
