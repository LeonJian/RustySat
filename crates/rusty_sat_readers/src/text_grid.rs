//! Tiny text-grid file reader for the first real array-loading vertical slice.
//!
//! This is intentionally not a Satpy production reader format. It mirrors the
//! Satpy file-handler shape from `satpy/doc/source/dev_guide/custom_reader.rst`:
//! YAML metadata identifies datasets and file types, then a format-specific
//! handler loads array values from matched files.

use crate::{FileMatch, Reader, YamlReaderConfig};
use rusty_sat_core::{DataGrid, DataId, Dataset, Result, RustySatError};
use std::fs;

#[derive(Debug, Clone)]
pub struct TextGridReader {
    config: YamlReaderConfig,
    files: Vec<FileMatch>,
}

impl TextGridReader {
    pub fn from_yaml_and_filenames(
        yaml: &str,
        filenames: impl IntoIterator<Item = impl AsRef<str>>,
    ) -> Result<Self> {
        let config = YamlReaderConfig::from_str(yaml)?;
        let files = config.match_filenames(filenames)?;
        Ok(Self { config, files })
    }

    pub fn files(&self) -> &[FileMatch] {
        &self.files
    }

    fn dataset_file_type(&self, id: &DataId) -> Result<&str> {
        self.config
            .datasets()
            .values()
            .find(|dataset| dataset.data_ids().iter().any(|candidate| candidate == id))
            .and_then(|dataset| dataset.file_type())
            .ok_or_else(|| {
                RustySatError::not_found(format!("file type for dataset '{}'", id.name()))
            })
    }

    fn first_file_for_type(&self, file_type: &str) -> Result<&FileMatch> {
        self.files
            .iter()
            .find(|file_match| file_match.file_type() == file_type)
            .ok_or_else(|| RustySatError::not_found(format!("matched file for type '{file_type}'")))
    }
}

impl Reader for TextGridReader {
    fn name(&self) -> &str {
        self.config.info().name()
    }

    fn available_dataset_ids(&self) -> Vec<DataId> {
        self.config.all_dataset_ids()
    }

    fn load(&self, id: &DataId) -> Result<Dataset> {
        let file_type = self.dataset_file_type(id)?;
        let file_match = self.first_file_for_type(file_type)?;
        let grid = load_text_grid(file_match.filename())?;
        let mut dataset = Dataset::new(id.clone()).with_data(grid);
        dataset.insert_metadata("filename", file_match.filename())?;
        dataset.insert_metadata("file_type", file_match.file_type())?;
        Ok(dataset)
    }
}

pub fn load_text_grid(filename: &str) -> Result<DataGrid> {
    let contents = fs::read_to_string(filename)
        .map_err(|err| RustySatError::not_found(format!("text grid file '{filename}': {err}")))?;
    parse_text_grid(&contents)
}

pub fn parse_text_grid(contents: &str) -> Result<DataGrid> {
    let mut values = Vec::new();
    let mut width = None;
    let mut height = 0;
    for (line_idx, line) in contents.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let row = parse_row(line, line_idx + 1)?;
        if row.is_empty() {
            continue;
        }
        match width {
            Some(width) if width != row.len() => {
                return Err(RustySatError::invalid_input(format!(
                    "row {} has {} columns but previous rows had {}",
                    line_idx + 1,
                    row.len(),
                    width
                )));
            }
            None => width = Some(row.len()),
            _ => {}
        }
        height += 1;
        values.extend(row);
    }
    let width =
        width.ok_or_else(|| RustySatError::invalid_input("text grid contains no numeric rows"))?;
    DataGrid::new(height, width, values)
}

fn parse_row(line: &str, line_number: usize) -> Result<Vec<f64>> {
    line.split(|ch: char| ch == ',' || ch.is_ascii_whitespace())
        .filter(|part| !part.is_empty())
        .map(|part| {
            part.parse::<f64>().map_err(|err| {
                RustySatError::invalid_input(format!(
                    "invalid number '{part}' on text grid line {line_number}: {err}"
                ))
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Reader;
    use std::time::{SystemTime, UNIX_EPOCH};

    const TEXT_GRID_YAML: &str = r#"
reader:
  name: text_grid
file_types:
  text_grid:
    file_patterns: ['GRID_{name:5s}_{start_time:%Y%m%d%H%M%S}_{nonce:d}.txt']
datasets:
  image:
    name: image
    resolution: 1000
    file_type: text_grid
"#;

    #[test]
    fn parses_whitespace_and_comma_text_grid() {
        let grid = parse_text_grid(
            r#"
# comment
1, 2, 3
4 5 6
"#,
        )
        .unwrap();

        assert_eq!(grid.shape(), (2, 3));
        assert_eq!(grid.values(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn rejects_ragged_text_grid() {
        let err = parse_text_grid("1 2 3\n4 5").unwrap_err();

        assert!(matches!(err, RustySatError::InvalidInput { .. }));
    }

    #[test]
    fn text_grid_reader_loads_matched_dataset_values() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("GRID_image_20200102030405_{nonce}.txt"));
        fs::write(&path, "1 2\n3 4\n").unwrap();

        let reader =
            TextGridReader::from_yaml_and_filenames(TEXT_GRID_YAML, [path.to_string_lossy()])
                .unwrap();
        let id = reader.available_dataset_ids().remove(0);
        let dataset = reader.load(&id).unwrap();
        fs::remove_file(path).unwrap();

        assert_eq!(reader.name(), "text_grid");
        assert_eq!(reader.files().len(), 1);
        assert_eq!(dataset.data().unwrap().shape(), (2, 2));
        assert_eq!(dataset.data().unwrap().values(), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(
            dataset.metadata().get("file_type"),
            Some(&"text_grid".to_string())
        );
    }
}
