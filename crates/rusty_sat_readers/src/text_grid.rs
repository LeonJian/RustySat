//! Tiny text-grid file reader for the first real array-loading vertical slice.
//!
//! This is intentionally not a Satpy production reader format. It mirrors the
//! Satpy file-handler shape from `satpy/doc/source/dev_guide/custom_reader.rst`:
//! YAML metadata identifies datasets and file types, then a format-specific
//! handler loads array values from matched files.

use crate::{FileMatch, Reader, YamlReaderConfig};
use rusty_sat_core::{
    ChunkRegion, ChunkShape, ChunkSource, DataArray, DataGrid, DataId, Dataset, LazyDataArray,
    Result, RustySatError,
};
use std::fs;
use std::sync::Arc;

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

    fn dataset_config(&self, id: &DataId) -> Result<&crate::DatasetConfig> {
        self.config
            .datasets()
            .values()
            .find(|dataset| dataset.data_ids().iter().any(|candidate| candidate == id))
            .ok_or_else(|| {
                RustySatError::not_found(format!("dataset config for dataset '{}'", id.name()))
            })
    }

    fn dataset_file_type(&self, id: &DataId) -> Result<&str> {
        self.dataset_config(id)?.file_type().ok_or_else(|| {
            RustySatError::not_found(format!("file type for dataset '{}'", id.name()))
        })
    }

    fn first_file_for_type(&self, file_type: &str) -> Result<&FileMatch> {
        self.files
            .iter()
            .find(|file_match| file_match.file_type() == file_type)
            .ok_or_else(|| RustySatError::not_found(format!("matched file for type '{file_type}'")))
    }

    pub fn lazy_array(&self, id: &DataId, chunks: ChunkShape) -> Result<LazyDataArray<f64>> {
        let file_type = self.dataset_file_type(id)?;
        let file_match = self.first_file_for_type(file_type)?;
        lazy_text_grid(file_match.filename(), chunks)
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
        let dataset_config = self.dataset_config(id)?;
        dataset.set_coordinate_names(dataset_config.coordinates().iter().cloned())?;
        for (key, value) in dataset_config.attrs() {
            dataset.insert_attr(key.clone(), value.clone())?;
        }
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

pub fn lazy_text_grid(filename: &str, chunks: ChunkShape) -> Result<LazyDataArray<f64>> {
    let source = Arc::new(TextGridChunkSource::open(filename)?);
    LazyDataArray::from_shape(
        vec![source.shape.0, source.shape.1],
        chunks,
        source as Arc<dyn ChunkSource<f64>>,
    )
}

#[derive(Debug, Clone)]
pub struct TextGridChunkSource {
    filename: String,
    shape: (usize, usize),
}

impl TextGridChunkSource {
    pub fn open(filename: impl Into<String>) -> Result<Self> {
        let filename = filename.into();
        let shape = scan_text_grid_shape(&filename)?;
        Ok(Self { filename, shape })
    }

    pub fn filename(&self) -> &str {
        &self.filename
    }

    pub fn shape(&self) -> (usize, usize) {
        self.shape
    }
}

impl ChunkSource<f64> for TextGridChunkSource {
    fn read_chunk(&self, region: &ChunkRegion) -> Result<DataArray<f64>> {
        let [origin_y, origin_x] = region.origin() else {
            return Err(RustySatError::invalid_input(
                "text grid chunk source requires 2D regions",
            ));
        };
        let [height, width] = region.shape() else {
            return Err(RustySatError::invalid_input(
                "text grid chunk source requires 2D regions",
            ));
        };
        let contents = fs::read_to_string(&self.filename).map_err(|err| {
            RustySatError::not_found(format!("text grid file '{}': {err}", self.filename))
        })?;
        let mut values = Vec::with_capacity(*height * *width);
        let y_end = *origin_y + *height;
        let x_end = *origin_x + *width;
        let mut data_row_idx = 0usize;
        for (line_idx, line) in contents.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let row = parse_row(line, line_idx + 1)?;
            if row.len() != self.shape.1 {
                return Err(RustySatError::invalid_input(format!(
                    "row {} has {} columns but text grid shape requires {}",
                    line_idx + 1,
                    row.len(),
                    self.shape.1
                )));
            }
            if (*origin_y..y_end).contains(&data_row_idx) {
                values.extend_from_slice(&row[*origin_x..x_end]);
            }
            data_row_idx += 1;
        }
        if values.len() != *height * *width {
            return Err(RustySatError::invalid_input(format!(
                "text grid chunk {:?}+{:?} was not fully read from '{}'",
                region.origin(),
                region.shape(),
                self.filename
            )));
        }
        DataArray::from_vec_named(vec![*height, *width], ["y", "x"], values)
    }
}

fn scan_text_grid_shape(filename: &str) -> Result<(usize, usize)> {
    let contents = fs::read_to_string(filename)
        .map_err(|err| RustySatError::not_found(format!("text grid file '{filename}': {err}")))?;
    let mut width = None;
    let mut height = 0usize;
    for (line_idx, line) in contents.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let row = parse_row(line, line_idx + 1)?;
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
    }
    let width =
        width.ok_or_else(|| RustySatError::invalid_input("text grid contains no numeric rows"))?;
    Ok((height, width))
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
    coordinates: [longitude, latitude]
    raw_metadata:
      platform: test-sat
      scan_lines: 2
      flags: [day, test]
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
            dataset.coordinate_names(),
            &["longitude".to_string(), "latitude".to_string()]
        );
        assert_eq!(
            dataset.metadata().get("file_type"),
            Some(&"text_grid".to_string())
        );
        assert_eq!(
            dataset
                .attr("raw_metadata")
                .and_then(|value| value.get_path(&["platform"]))
                .and_then(rusty_sat_core::MetadataValue::as_str),
            Some("test-sat")
        );
    }

    #[test]
    fn text_grid_chunk_source_reads_requested_region() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("rusty_sat_lazy_grid_{nonce}.txt"));
        fs::write(&path, "# comment\n1 2 3 4\n5 6 7 8\n9 10 11 12\n").unwrap();

        let source = TextGridChunkSource::open(path.to_string_lossy()).unwrap();
        let chunk = source
            .read_chunk(&ChunkRegion::new(&[3, 4], [1, 1], [2, 2]).unwrap())
            .unwrap();
        fs::remove_file(path).unwrap();

        assert_eq!(source.shape(), (3, 4));
        assert_eq!(chunk.shape(), (2, 2));
        assert_eq!(chunk.values(), &[6.0, 7.0, 10.0, 11.0]);
    }

    #[test]
    fn text_grid_reader_exposes_lazy_array_fixture() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("GRID_image_20200102030405_{nonce}.txt"));
        fs::write(&path, "1 2 3\n4 5 6\n").unwrap();

        let reader =
            TextGridReader::from_yaml_and_filenames(TEXT_GRID_YAML, [path.to_string_lossy()])
                .unwrap();
        let id = reader.available_dataset_ids().remove(0);
        let lazy = reader
            .lazy_array(&id, ChunkShape::new(vec![1, 2]).unwrap())
            .unwrap();
        let chunk = lazy.read_chunk(&[1, 1]).unwrap();
        fs::remove_file(path).unwrap();

        assert_eq!(lazy.shape(), &[2, 3]);
        assert_eq!(lazy.chunks().as_slice(), &[1, 2]);
        assert_eq!(chunk.shape(), (1, 1));
        assert_eq!(chunk.values(), &[6.0]);
    }
}
