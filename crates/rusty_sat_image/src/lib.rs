//! Image and enhancement foundations.

use rusty_sat_core::{AnyDataArray, DataArray, Dataset, NumericElement, Result, RustySatError};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageMode {
    Luma,
    Rgb,
    Rgba,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Image {
    mode: ImageMode,
    height: usize,
    width: usize,
    pixels: Vec<u8>,
}

impl Image {
    pub fn new(mode: ImageMode, height: usize, width: usize) -> Result<Self> {
        Self::from_pixels(
            mode,
            height,
            width,
            vec![0; checked_pixel_len(mode, height, width)?],
        )
    }

    pub fn from_pixels(
        mode: ImageMode,
        height: usize,
        width: usize,
        pixels: Vec<u8>,
    ) -> Result<Self> {
        if height == 0 || width == 0 {
            return Err(RustySatError::invalid_input(
                "image dimensions must be non-zero",
            ));
        }
        let expected = checked_pixel_len(mode, height, width)?;
        if pixels.len() != expected {
            return Err(RustySatError::invalid_input(format!(
                "image has {} pixels but {mode:?} shape ({height}, {width}) requires {expected} bytes",
                pixels.len()
            )));
        }
        Ok(Self {
            mode,
            height,
            width,
            pixels,
        })
    }

    pub fn from_luma_dataset(dataset: &Dataset) -> Result<Self> {
        let Some(array) = dataset.array() else {
            return Err(RustySatError::invalid_input(format!(
                "dataset '{}' has no array data",
                dataset.id().name()
            )));
        };
        Self::from_luma_array(array)
    }

    pub fn from_luma_array(array: &AnyDataArray) -> Result<Self> {
        let (height, width) = array.shape_yx()?;
        if array.shape().len() != 2 {
            return Err(RustySatError::invalid_input(format!(
                "luma image requires a 2D y/x array, got shape {:?}",
                array.shape()
            )));
        }
        let pixels = match array {
            AnyDataArray::F32(array) => autoscale_luma(array),
            AnyDataArray::F64(array) => autoscale_luma(array),
            AnyDataArray::U8(array) => autoscale_luma(array),
            AnyDataArray::U16(array) => autoscale_luma(array),
            AnyDataArray::I16(array) => autoscale_luma(array),
        };
        Self::from_pixels(ImageMode::Luma, height, width, pixels)
    }

    pub fn mode(&self) -> ImageMode {
        self.mode
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.height, self.width)
    }

    pub fn channels(&self) -> usize {
        self.mode.channels()
    }

    pub fn pixels(&self) -> &[u8] {
        &self.pixels
    }

    pub fn into_pixels(self) -> Vec<u8> {
        self.pixels
    }
}

impl ImageMode {
    pub fn channels(self) -> usize {
        match self {
            Self::Luma => 1,
            Self::Rgb => 3,
            Self::Rgba => 4,
        }
    }
}

fn checked_pixel_len(mode: ImageMode, height: usize, width: usize) -> Result<usize> {
    height
        .checked_mul(width)
        .and_then(|value| value.checked_mul(mode.channels()))
        .ok_or_else(|| RustySatError::invalid_input("image dimensions are too large"))
}

fn autoscale_luma<T: NumericElement>(array: &DataArray<T>) -> Vec<u8> {
    let mut min_value = f64::INFINITY;
    let mut max_value = f64::NEG_INFINITY;
    for (idx, value) in array.values().iter().enumerate() {
        if array.is_masked(idx).unwrap_or(false) {
            continue;
        }
        let value = value.to_f64();
        if !value.is_finite() {
            continue;
        }
        min_value = min_value.min(value);
        max_value = max_value.max(value);
    }

    if !min_value.is_finite() || !max_value.is_finite() {
        return vec![0; array.len()];
    }

    let scale = max_value - min_value;
    array
        .values()
        .iter()
        .enumerate()
        .map(|(idx, value)| {
            if array.is_masked(idx).unwrap_or(false) {
                return 0;
            }
            let value = value.to_f64();
            if !value.is_finite() {
                return 0;
            }
            if scale == 0.0 {
                return 0;
            }
            (((value - min_value) / scale) * 255.0)
                .clamp(0.0, 255.0)
                .round() as u8
        })
        .collect()
}

pub trait Enhancer {
    fn name(&self) -> &str;

    fn enhance(&self, _dataset: &Dataset) -> Result<Image> {
        Err(RustySatError::unsupported(format!(
            "{} enhancer",
            self.name()
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constructs_image() {
        let image = Image::new(ImageMode::Rgb, 10, 20).unwrap();
        assert_eq!(image.mode(), ImageMode::Rgb);
        assert_eq!(image.shape(), (10, 20));
        assert_eq!(image.channels(), 3);
        assert_eq!(image.pixels().len(), 600);
    }

    #[test]
    fn validates_pixel_buffer_length() {
        assert!(Image::from_pixels(ImageMode::Rgba, 2, 2, vec![0; 15]).is_err());
        assert!(Image::new(ImageMode::Luma, 0, 2).is_err());
    }

    #[test]
    fn creates_luma_image_from_dataset_array() {
        let array =
            DataArray::<u16>::from_vec_named(vec![2, 3], ["y", "x"], vec![10, 20, 30, 40, 50, 60])
                .unwrap();
        let dataset = Dataset::new(rusty_sat_core::DataId::new("test").unwrap()).with_array(array);
        let image = Image::from_luma_dataset(&dataset).unwrap();

        assert_eq!(image.mode(), ImageMode::Luma);
        assert_eq!(image.shape(), (2, 3));
        assert_eq!(image.pixels(), &[0, 51, 102, 153, 204, 255]);
    }

    #[test]
    fn creates_luma_image_with_masked_pixels_filled_black() {
        let mask = rusty_sat_core::ValidityMask::from_masked_flags([false, true, false, false]);
        let array =
            DataArray::<f32>::from_vec_named(vec![2, 2], ["y", "x"], vec![1.0, 99.0, 2.0, 3.0])
                .unwrap()
                .with_mask(mask)
                .unwrap();
        let image = Image::from_luma_array(&array.into()).unwrap();

        assert_eq!(image.pixels(), &[0, 0, 128, 255]);
    }
}
