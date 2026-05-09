//! Geometry and resampling foundations.

pub mod area;
pub mod crs;
pub mod geometry;
pub mod nearest;
pub mod swath;

pub use area::{
    load_area_from_file, load_area_from_str, load_areas_from_str, AreaDefinition, PixelResolution,
};
pub use crs::{Coordinate2D, CrsSource, ProjCrs, ProjectionBackendStrategy, TransformDirection};
pub use geometry::{CoordinateDefinition, GeometryDefinition, GeometryKind, GridDefinition};
pub use nearest::{
    resample_area_nearest, resample_area_nearest_lazy, resample_area_nearest_owned,
    resample_swath_nearest, NearestAreaResampler,
};
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
