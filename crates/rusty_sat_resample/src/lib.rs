//! Geometry and resampling foundations.

use rusty_sat_core::{Dataset, Result, RustySatError};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AreaDefinition {
    id: String,
    height: usize,
    width: usize,
}

impl AreaDefinition {
    pub fn new(id: impl Into<String>, height: usize, width: usize) -> Result<Self> {
        let id = id.into();
        if id.trim().is_empty() {
            return Err(RustySatError::invalid_input("area id cannot be empty"));
        }
        if height == 0 || width == 0 {
            return Err(RustySatError::invalid_input(
                "area dimensions must be non-zero",
            ));
        }
        Ok(Self { id, height, width })
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.height, self.width)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SwathDefinition {
    height: usize,
    width: usize,
}

impl SwathDefinition {
    pub fn new(height: usize, width: usize) -> Result<Self> {
        if height == 0 || width == 0 {
            return Err(RustySatError::invalid_input(
                "swath dimensions must be non-zero",
            ));
        }
        Ok(Self { height, width })
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.height, self.width)
    }
}

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
