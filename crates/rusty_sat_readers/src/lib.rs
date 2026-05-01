//! Reader framework foundations.

use rusty_sat_core::{DataId, Dataset, Result};

pub trait Reader {
    fn name(&self) -> &str;

    fn available_dataset_ids(&self) -> Vec<DataId> {
        Vec::new()
    }

    fn load(&self, _id: &DataId) -> Result<Dataset>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use rusty_sat_core::RustySatError;

    struct EmptyReader;

    impl Reader for EmptyReader {
        fn name(&self) -> &str {
            "empty"
        }

        fn load(&self, _id: &DataId) -> Result<Dataset> {
            Err(RustySatError::unsupported("empty reader load"))
        }
    }

    #[test]
    fn reader_trait_compiles() {
        let reader = EmptyReader;
        assert_eq!(reader.name(), "empty");
        assert!(reader.available_dataset_ids().is_empty());
    }
}
