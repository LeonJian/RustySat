//! Portable Graymap image writer for the first image-output vertical slice.
//!
//! Reference behavior inspected before implementation:
//! - `satpy/satpy/writers/simple_image.py`
//! - `satpy/doc/source/writing.rst`
//! - `deps/trollimage/trollimage/xrimage.py`
//!
//! Satpy's simple image writer delegates enhanced image objects to Pillow.
//! Rusty Sat does not have the enhancement pipeline yet, so this first writer
//! writes a single-band `DataGrid` to the simple, documented PGM image format.

use crate::Writer;
use rusty_sat_core::{DataGrid, Dataset, Result, RustySatError};
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LinearScale {
    min: f64,
    max: f64,
}

impl LinearScale {
    pub fn new(min: f64, max: f64) -> Result<Self> {
        if !min.is_finite() || !max.is_finite() {
            return Err(RustySatError::invalid_input(
                "PGM scale bounds must be finite",
            ));
        }
        if min >= max {
            return Err(RustySatError::invalid_input(
                "PGM scale minimum must be less than maximum",
            ));
        }
        Ok(Self { min, max })
    }
}

#[derive(Debug, Clone)]
pub struct PgmWriter {
    scale: Option<LinearScale>,
    fill_value: u8,
}

impl PgmWriter {
    pub fn new() -> Self {
        Self {
            scale: None,
            fill_value: 0,
        }
    }

    pub fn with_scale(mut self, min: f64, max: f64) -> Result<Self> {
        self.scale = Some(LinearScale::new(min, max)?);
        Ok(self)
    }

    pub fn with_fill_value(mut self, fill_value: u8) -> Self {
        self.fill_value = fill_value;
        self
    }

    pub fn save_dataset(&self, dataset: &Dataset, path: impl AsRef<Path>) -> Result<()> {
        let grid = dataset
            .data()
            .ok_or_else(|| RustySatError::invalid_input("PGM writer requires dataset grid data"))?;
        write_pgm(grid, path, self.scale, self.fill_value)
    }
}

impl Default for PgmWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl Writer for PgmWriter {
    fn name(&self) -> &str {
        "pgm"
    }

    fn save_dataset(&self, dataset: &Dataset, path: &Path) -> Result<()> {
        PgmWriter::save_dataset(self, dataset, path)
    }
}

pub fn write_pgm(
    grid: &DataGrid,
    path: impl AsRef<Path>,
    scale: Option<LinearScale>,
    fill_value: u8,
) -> Result<()> {
    let bytes = encode_pgm(grid, scale, fill_value)?;
    fs::write(path.as_ref(), bytes).map_err(|err| {
        RustySatError::invalid_input(format!(
            "failed to write PGM '{}': {err}",
            path.as_ref().display()
        ))
    })
}

pub fn encode_pgm(grid: &DataGrid, scale: Option<LinearScale>, fill_value: u8) -> Result<Vec<u8>> {
    let scale = match scale {
        Some(scale) => scale,
        None => autoscale(grid, fill_value)?,
    };
    let (height, width) = grid.shape();
    let mut out = format!("P5\n{width} {height}\n255\n").into_bytes();
    out.extend(
        grid.values()
            .iter()
            .map(|value| scale_value(*value, scale, fill_value)),
    );
    Ok(out)
}

fn autoscale(grid: &DataGrid, fill_value: u8) -> Result<LinearScale> {
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for value in grid
        .values()
        .iter()
        .copied()
        .filter(|value| value.is_finite())
    {
        min = min.min(value);
        max = max.max(value);
    }
    if !min.is_finite() || !max.is_finite() {
        return LinearScale::new(0.0, 1.0).or_else(|_| {
            Err(RustySatError::invalid_input(format!(
                "cannot autoscale PGM data; using fill value {fill_value} failed unexpectedly"
            )))
        });
    }
    if min == max {
        return LinearScale::new(min, min + 1.0);
    }
    LinearScale::new(min, max)
}

fn scale_value(value: f64, scale: LinearScale, fill_value: u8) -> u8 {
    if !value.is_finite() {
        return fill_value;
    }
    let normalized = ((value - scale.min) / (scale.max - scale.min)).clamp(0.0, 1.0);
    (normalized * 255.0).round() as u8
}

#[cfg(test)]
mod tests {
    use super::*;
    use rusty_sat_core::DataId;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn encodes_dataset_grid_as_binary_pgm() {
        let grid = DataGrid::new(2, 2, vec![0.0, 0.5, 1.0, f64::NAN]).unwrap();
        let bytes = encode_pgm(&grid, Some(LinearScale::new(0.0, 1.0).unwrap()), 7).unwrap();

        assert_eq!(&bytes[..11], b"P5\n2 2\n255\n");
        assert_eq!(&bytes[11..], &[0, 128, 255, 7]);
    }

    #[test]
    fn autoscale_maps_min_and_max_to_byte_range() {
        let grid = DataGrid::new(1, 3, vec![10.0, 15.0, 20.0]).unwrap();
        let bytes = encode_pgm(&grid, None, 0).unwrap();

        assert_eq!(&bytes[..11], b"P5\n3 1\n255\n");
        assert_eq!(&bytes[11..], &[0, 128, 255]);
    }

    #[test]
    fn writer_saves_dataset_to_image_file() {
        let id = DataId::new("image").unwrap();
        let dataset = Dataset::new(id).with_data(DataGrid::new(1, 2, vec![0.0, 1.0]).unwrap());
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("rusty_sat_{nonce}.pgm"));

        PgmWriter::new()
            .with_scale(0.0, 1.0)
            .unwrap()
            .save_dataset(&dataset, &path)
            .unwrap();
        let bytes = fs::read(&path).unwrap();
        fs::remove_file(path).unwrap();

        assert_eq!(&bytes[..11], b"P5\n2 1\n255\n");
        assert_eq!(&bytes[11..], &[0, 255]);
    }

    #[test]
    fn pgm_writer_implements_writer_trait() {
        let writer: Box<dyn Writer> = Box::new(PgmWriter::new());
        assert_eq!(writer.name(), "pgm");
    }
}
