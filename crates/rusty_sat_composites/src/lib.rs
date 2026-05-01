//! Composite and modifier framework foundations.

use rusty_sat_core::{Dataset, Result, RustySatError};

pub trait Compositor {
    fn name(&self) -> &str;

    fn compose(&self, _inputs: &[Dataset]) -> Result<Dataset> {
        Err(RustySatError::unsupported(format!(
            "{} compositor",
            self.name()
        )))
    }
}

pub trait Modifier {
    fn name(&self) -> &str;

    fn apply(&self, _input: &Dataset) -> Result<Dataset> {
        Err(RustySatError::unsupported(format!(
            "{} modifier",
            self.name()
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct PlaceholderCompositor;

    impl Compositor for PlaceholderCompositor {
        fn name(&self) -> &str {
            "placeholder"
        }
    }

    #[test]
    fn compositor_trait_compiles() {
        let compositor = PlaceholderCompositor;
        assert_eq!(compositor.name(), "placeholder");
        assert!(compositor.compose(&[]).is_err());
    }
}
