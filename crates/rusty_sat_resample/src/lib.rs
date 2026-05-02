//! Geometry and resampling foundations.

pub mod area;
pub mod swath;

pub use area::{
    load_area_from_file, load_area_from_str, load_areas_from_str, AreaDefinition, PixelResolution,
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

        let swath = SwathDefinition::new(5, 6).unwrap();
        assert_eq!(swath.shape(), (5, 6));
    }
}
