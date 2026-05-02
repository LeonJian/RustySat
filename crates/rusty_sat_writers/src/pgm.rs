//! Portable Graymap image writer for the first image-output vertical slice.
//!
//! Reference behavior inspected before implementation:
//! - `satpy/satpy/writers/simple_image.py`
//! - `satpy/doc/source/writing.rst`
//! - `deps/trollimage/trollimage/xrimage.py`
//!
//! Satpy's simple image writer delegates enhanced image objects to Pillow.
//! Rusty Sat does not have the enhancement pipeline yet, so this first writer
//! writes a single-band numeric data array to the simple, documented PGM image
//! format.

use crate::Writer;
use rusty_sat_core::{
    AnyDataArray, DataArray, DataGrid, Dataset, LazyDataArray, NumericElement, Result,
    RustySatError, ValidityMask,
};
use std::fs;
use std::io::{self, Write};
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
        let array = dataset.array().ok_or_else(|| {
            RustySatError::invalid_input("PGM writer requires dataset array data")
        })?;
        write_pgm_array(array, path, self.scale, self.fill_value)
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

pub fn write_pgm_array(
    array: &AnyDataArray,
    path: impl AsRef<Path>,
    scale: Option<LinearScale>,
    fill_value: u8,
) -> Result<()> {
    let bytes = encode_pgm_array(array, scale, fill_value)?;
    fs::write(path.as_ref(), bytes).map_err(|err| {
        RustySatError::invalid_input(format!(
            "failed to write PGM '{}': {err}",
            path.as_ref().display()
        ))
    })
}

pub fn write_pgm_lazy<T: NumericElement>(
    array: &LazyDataArray<T>,
    path: impl AsRef<Path>,
    scale: Option<LinearScale>,
    fill_value: u8,
) -> Result<()> {
    let mut file = fs::File::create(path.as_ref()).map_err(|err| {
        RustySatError::invalid_input(format!(
            "failed to create PGM '{}': {err}",
            path.as_ref().display()
        ))
    })?;
    write_pgm_lazy_to_writer(array, &mut file, scale, fill_value)
}

pub fn encode_pgm(grid: &DataGrid, scale: Option<LinearScale>, fill_value: u8) -> Result<Vec<u8>> {
    encode_pgm_values(
        grid.shape(),
        grid.values().iter().copied(),
        grid.mask(),
        scale,
        fill_value,
    )
}

pub fn encode_pgm_array(
    array: &AnyDataArray,
    scale: Option<LinearScale>,
    fill_value: u8,
) -> Result<Vec<u8>> {
    array.require_dims_exact(&["y", "x"])?;
    let shape = array.shape_yx()?;
    encode_pgm_values(
        shape,
        array.values_as_f64(),
        array.mask(),
        scale,
        fill_value,
    )
}

pub fn encode_pgm_lazy<T: NumericElement>(
    array: &LazyDataArray<T>,
    scale: Option<LinearScale>,
    fill_value: u8,
) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    write_pgm_lazy_to_writer(array, &mut bytes, scale, fill_value)?;
    Ok(bytes)
}

fn encode_pgm_values(
    shape: (usize, usize),
    values: impl IntoIterator<Item = f64>,
    mask: Option<&ValidityMask>,
    scale: Option<LinearScale>,
    fill_value: u8,
) -> Result<Vec<u8>> {
    let values: Vec<_> = values.into_iter().collect();
    let scale = match scale {
        Some(scale) => scale,
        None => autoscale_values(&values, mask, fill_value)?,
    };
    let (height, width) = shape;
    let mut out = format!("P5\n{width} {height}\n255\n").into_bytes();
    out.extend(values.into_iter().enumerate().map(|(idx, value)| {
        if mask.is_some_and(|mask| mask.is_masked(idx).unwrap_or(false)) {
            fill_value
        } else {
            scale_value(value, scale, fill_value)
        }
    }));
    Ok(out)
}

fn write_pgm_lazy_to_writer<T: NumericElement>(
    array: &LazyDataArray<T>,
    writer: &mut impl Write,
    scale: Option<LinearScale>,
    fill_value: u8,
) -> Result<()> {
    array.require_dims_exact(&["y", "x"])?;
    let (height, width) = array.shape_yx()?;
    let scale = match scale {
        Some(scale) => scale,
        None => autoscale_lazy(array, fill_value)?,
    };
    writer
        .write_all(format!("P5\n{width} {height}\n255\n").as_bytes())
        .map_err(write_error)?;

    let chunk_y = array.chunks().as_slice()[0];
    let chunk_x = array.chunks().as_slice()[1];
    let y_chunks = height.div_ceil(chunk_y);
    let x_chunks = width.div_ceil(chunk_x);

    for cy in 0..y_chunks {
        let stripe_origin_y = cy * chunk_y;
        let stripe_height = chunk_y.min(height - stripe_origin_y);
        let mut stripe = vec![fill_value; stripe_height * width];
        for cx in 0..x_chunks {
            let chunk = array.read_chunk(&[cy, cx])?;
            let chunk_shape = chunk.shape_yx()?;
            let origin_x = cx * chunk_x;
            copy_scaled_chunk_into_stripe(
                &chunk,
                chunk_shape,
                origin_x,
                &mut stripe,
                width,
                scale,
                fill_value,
            );
        }
        writer.write_all(&stripe).map_err(write_error)?;
    }
    Ok(())
}

fn copy_scaled_chunk_into_stripe<T: NumericElement>(
    chunk: &DataArray<T>,
    chunk_shape: (usize, usize),
    origin_x: usize,
    stripe: &mut [u8],
    stripe_width: usize,
    scale: LinearScale,
    fill_value: u8,
) {
    let (chunk_height, chunk_width) = chunk_shape;
    for local_y in 0..chunk_height {
        for local_x in 0..chunk_width {
            let chunk_idx = local_y * chunk_width + local_x;
            let stripe_idx = local_y * stripe_width + origin_x + local_x;
            stripe[stripe_idx] = if chunk
                .mask()
                .is_some_and(|mask| mask.is_masked(chunk_idx).unwrap_or(false))
            {
                fill_value
            } else {
                scale_value(chunk.values()[chunk_idx].to_f64(), scale, fill_value)
            };
        }
    }
}

fn autoscale_lazy<T: NumericElement>(
    array: &LazyDataArray<T>,
    fill_value: u8,
) -> Result<LinearScale> {
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    let (height, width) = array.shape_yx()?;
    let chunk_y = array.chunks().as_slice()[0];
    let chunk_x = array.chunks().as_slice()[1];
    let y_chunks = height.div_ceil(chunk_y);
    let x_chunks = width.div_ceil(chunk_x);

    for cy in 0..y_chunks {
        for cx in 0..x_chunks {
            let chunk = array.read_chunk(&[cy, cx])?;
            for (idx, value) in chunk.values().iter().enumerate() {
                if chunk
                    .mask()
                    .is_some_and(|mask| mask.is_masked(idx).unwrap_or(false))
                {
                    continue;
                }
                let value = value.to_f64();
                if !value.is_finite() {
                    continue;
                }
                min = min.min(value);
                max = max.max(value);
            }
        }
    }

    if !min.is_finite() || !max.is_finite() {
        return LinearScale::new(0.0, 1.0).or_else(|_| {
            Err(RustySatError::invalid_input(format!(
                "cannot autoscale lazy PGM data; using fill value {fill_value} failed unexpectedly"
            )))
        });
    }
    if min == max {
        return LinearScale::new(min, min + 1.0);
    }
    LinearScale::new(min, max)
}

fn write_error(err: io::Error) -> RustySatError {
    RustySatError::invalid_input(format!("failed to write PGM bytes: {err}"))
}

fn autoscale_values(
    values: &[f64],
    mask: Option<&ValidityMask>,
    fill_value: u8,
) -> Result<LinearScale> {
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for (idx, value) in values.iter().copied().enumerate() {
        if mask.is_some_and(|mask| mask.is_masked(idx).unwrap_or(false)) || !value.is_finite() {
            continue;
        }
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
    use rusty_sat_core::{ChunkRegion, ChunkShape, ChunkSource, DataId};
    use std::sync::Mutex;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[derive(Debug)]
    struct MatrixSource {
        width: usize,
        values: Vec<u8>,
        requests: Mutex<Vec<ChunkRegion>>,
    }

    impl MatrixSource {
        fn new(width: usize, values: Vec<u8>) -> Self {
            Self {
                width,
                values,
                requests: Mutex::new(Vec::new()),
            }
        }
    }

    impl ChunkSource<u8> for MatrixSource {
        fn read_chunk(&self, region: &ChunkRegion) -> Result<DataArray<u8>> {
            self.requests.lock().unwrap().push(region.clone());
            let [origin_y, origin_x] = region.origin() else {
                panic!("test source only supports 2D regions");
            };
            let [height, width] = region.shape() else {
                panic!("test source only supports 2D regions");
            };
            let mut values = Vec::with_capacity(height * width);
            for y in 0..*height {
                for x in 0..*width {
                    let source_idx = (*origin_y + y) * self.width + *origin_x + x;
                    values.push(self.values[source_idx]);
                }
            }
            DataArray::from_vec_named(vec![*height, *width], ["y", "x"], values)
        }
    }

    #[test]
    fn encodes_dataset_grid_as_binary_pgm() {
        let grid = DataGrid::new(2, 2, vec![0.0, 0.5, 1.0, f64::NAN]).unwrap();
        let bytes = encode_pgm(&grid, Some(LinearScale::new(0.0, 1.0).unwrap()), 7).unwrap();

        assert_eq!(&bytes[..11], b"P5\n2 2\n255\n");
        assert_eq!(&bytes[11..], &[0, 128, 255, 7]);
    }

    #[test]
    fn masked_pixels_use_fill_value_and_do_not_affect_autoscale() {
        let grid = DataGrid::new(1, 3, vec![10.0, 9999.0, 20.0])
            .unwrap()
            .with_mask(ValidityMask::from_masked_flags([false, true, false]))
            .unwrap();
        let bytes = encode_pgm(&grid, None, 7).unwrap();

        assert_eq!(&bytes[..11], b"P5\n3 1\n255\n");
        assert_eq!(&bytes[11..], &[0, 7, 255]);
    }

    #[test]
    fn autoscale_maps_min_and_max_to_byte_range() {
        let grid = DataGrid::new(1, 3, vec![10.0, 15.0, 20.0]).unwrap();
        let bytes = encode_pgm(&grid, None, 0).unwrap();

        assert_eq!(&bytes[..11], b"P5\n3 1\n255\n");
        assert_eq!(&bytes[11..], &[0, 128, 255]);
    }

    #[test]
    fn encodes_runtime_typed_integer_array_as_binary_pgm() {
        let array =
            AnyDataArray::from(DataArray::<u16>::from_vec(vec![1, 3], vec![0, 128, 255]).unwrap());
        let bytes =
            encode_pgm_array(&array, Some(LinearScale::new(0.0, 255.0).unwrap()), 0).unwrap();

        assert_eq!(&bytes[..11], b"P5\n3 1\n255\n");
        assert_eq!(&bytes[11..], &[0, 128, 255]);
    }

    #[test]
    fn rejects_non_2d_runtime_typed_array() {
        let array = AnyDataArray::from(DataArray::<u8>::from_vec(vec![3], vec![0, 1, 2]).unwrap());
        let err = encode_pgm_array(&array, None, 0).unwrap_err();

        assert!(err.to_string().contains("do not match expected"));
    }

    #[test]
    fn rejects_2d_array_without_image_dimensions() {
        let array = AnyDataArray::from(
            DataArray::<u8>::from_vec_named(vec![1, 3], ["row", "col"], vec![0, 1, 2]).unwrap(),
        );
        let err = encode_pgm_array(&array, None, 0).unwrap_err();

        assert!(err.to_string().contains("do not match expected"));
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
    fn writer_saves_runtime_typed_dataset_to_image_file() {
        let id = DataId::new("image").unwrap();
        let dataset = Dataset::new(id)
            .with_array(DataArray::<u8>::from_vec(vec![1, 2], vec![0, 255]).unwrap());
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("rusty_sat_typed_{nonce}.pgm"));

        PgmWriter::new().save_dataset(&dataset, &path).unwrap();
        let bytes = fs::read(&path).unwrap();
        fs::remove_file(path).unwrap();

        assert_eq!(&bytes[..11], b"P5\n2 1\n255\n");
        assert_eq!(&bytes[11..], &[0, 255]);
    }

    #[test]
    fn encodes_lazy_array_by_reading_chunks_in_stripes() {
        let source = std::sync::Arc::new(MatrixSource::new(
            4,
            vec![0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110],
        ));
        let array = LazyDataArray::from_shape(
            vec![3, 4],
            ChunkShape::new(vec![2, 3]).unwrap(),
            source.clone(),
        )
        .unwrap();

        let bytes =
            encode_pgm_lazy(&array, Some(LinearScale::new(0.0, 110.0).unwrap()), 0).unwrap();

        assert_eq!(&bytes[..11], b"P5\n4 3\n255\n");
        assert_eq!(
            &bytes[11..],
            &[0, 23, 46, 70, 93, 116, 139, 162, 185, 209, 232, 255]
        );
        let requests = source.requests.lock().unwrap();
        assert_eq!(
            requests
                .iter()
                .map(|region| (region.origin().to_vec(), region.shape().to_vec()))
                .collect::<Vec<_>>(),
            vec![
                (vec![0, 0], vec![2, 3]),
                (vec![0, 3], vec![2, 1]),
                (vec![2, 0], vec![1, 3]),
                (vec![2, 3], vec![1, 1]),
            ]
        );
    }

    #[test]
    fn autoscale_lazy_array_uses_chunk_passes() {
        let source = std::sync::Arc::new(MatrixSource::new(3, vec![10, 20, 30, 40, 50, 60]));
        let array = LazyDataArray::from_shape(
            vec![2, 3],
            ChunkShape::new(vec![1, 2]).unwrap(),
            source.clone(),
        )
        .unwrap();

        let bytes = encode_pgm_lazy(&array, None, 0).unwrap();

        assert_eq!(&bytes[..11], b"P5\n3 2\n255\n");
        assert_eq!(&bytes[11..], &[0, 51, 102, 153, 204, 255]);
        assert_eq!(source.requests.lock().unwrap().len(), 8);
    }

    #[test]
    fn pgm_writer_implements_writer_trait() {
        let writer: Box<dyn Writer> = Box::new(PgmWriter::new());
        assert_eq!(writer.name(), "pgm");
    }
}
