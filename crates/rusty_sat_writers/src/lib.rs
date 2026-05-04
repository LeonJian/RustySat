//! Writer framework foundations.

use std::path::Path;

pub mod pgm;
pub mod simple_image;

pub use pgm::{
    encode_pgm, encode_pgm_array, encode_pgm_from_f64, write_pgm, write_pgm_array, LinearScale,
    PgmWriter,
};
pub use simple_image::{write_png16_image, write_png_image, SimpleImageWriter};

use rusty_sat_core::{Dataset, DatasetWriter, Result, RustySatError};
use rusty_sat_image::Image;

pub trait Writer {
    fn name(&self) -> &str;

    fn save_image(&self, _image: &Image, _path: &Path) -> Result<()> {
        Err(RustySatError::unsupported(format!(
            "{} writer",
            self.name()
        )))
    }

    fn save_dataset(&self, _dataset: &Dataset, _path: &Path) -> Result<()> {
        Err(RustySatError::unsupported(format!(
            "{} dataset writer",
            self.name()
        )))
    }
}

impl DatasetWriter for PgmWriter {
    fn save_dataset(&self, dataset: &Dataset, path: &Path) -> Result<()> {
        PgmWriter::save_dataset(self, dataset, path)
    }
}

impl DatasetWriter for SimpleImageWriter {
    fn save_dataset(&self, dataset: &Dataset, path: &Path) -> Result<()> {
        Writer::save_dataset(self, dataset, path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct PlaceholderWriter;

    impl Writer for PlaceholderWriter {
        fn name(&self) -> &str {
            "placeholder"
        }
    }

    #[test]
    fn writer_trait_compiles() {
        let writer = PlaceholderWriter;
        assert_eq!(writer.name(), "placeholder");
    }
}
