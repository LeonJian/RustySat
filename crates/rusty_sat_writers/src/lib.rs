//! Writer framework foundations.

use std::path::Path;

use rusty_sat_core::{Result, RustySatError};
use rusty_sat_image::Image;

pub trait Writer {
    fn name(&self) -> &str;

    fn save_image(&self, _image: &Image, _path: &Path) -> Result<()> {
        Err(RustySatError::unsupported(format!(
            "{} writer",
            self.name()
        )))
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
