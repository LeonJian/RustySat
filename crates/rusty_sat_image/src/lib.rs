//! Image and enhancement foundations.

use rusty_sat_core::{
    AnyDataArray, DataArray, Dataset, NumericElement, Result, RustySatError, ValidityMask,
};

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

#[derive(Debug, Clone, PartialEq)]
pub struct FloatImage {
    mode: ImageMode,
    height: usize,
    width: usize,
    pixels: Vec<f32>,
    mask: Option<ValidityMask>,
    enhancement_history: Vec<StretchRecord>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StretchRecord {
    scale: Vec<f32>,
    offset: Vec<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ChannelExtreme {
    Min,
    Max,
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
        FloatImage::from_luma_array(array)?
            .crude_stretched(None, None)
            .to_u8_image(0)
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

impl FloatImage {
    pub fn from_pixels(
        mode: ImageMode,
        height: usize,
        width: usize,
        pixels: Vec<f32>,
    ) -> Result<Self> {
        if height == 0 || width == 0 {
            return Err(RustySatError::invalid_input(
                "image dimensions must be non-zero",
            ));
        }
        let expected = checked_pixel_len(mode, height, width)?;
        if pixels.len() != expected {
            return Err(RustySatError::invalid_input(format!(
                "float image has {} pixels but {mode:?} shape ({height}, {width}) requires {expected} values",
                pixels.len()
            )));
        }
        Ok(Self {
            mode,
            height,
            width,
            pixels,
            mask: None,
            enhancement_history: Vec::new(),
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
        let mut image = match array {
            AnyDataArray::F32(array) => Self::from_numeric_luma_array(array, height, width)?,
            AnyDataArray::F64(array) => Self::from_numeric_luma_array(array, height, width)?,
            AnyDataArray::U8(array) => Self::from_numeric_luma_array(array, height, width)?,
            AnyDataArray::U16(array) => Self::from_numeric_luma_array(array, height, width)?,
            AnyDataArray::I16(array) => Self::from_numeric_luma_array(array, height, width)?,
        };
        image.mask = array.mask().cloned();
        Ok(image)
    }

    fn from_numeric_luma_array<T: NumericElement>(
        array: &DataArray<T>,
        height: usize,
        width: usize,
    ) -> Result<Self> {
        let pixels = array
            .values()
            .iter()
            .map(|value| value.to_f64() as f32)
            .collect();
        Self::from_pixels(ImageMode::Luma, height, width, pixels)
    }

    pub fn crude_stretched(mut self, min_stretch: Option<f32>, max_stretch: Option<f32>) -> Self {
        self.crude_stretch_in_place(min_stretch, max_stretch);
        self
    }

    pub fn crude_stretch_in_place(&mut self, min_stretch: Option<f32>, max_stretch: Option<f32>) {
        let channels = self.mode.channels();
        let mut scale = Vec::with_capacity(channels);
        let mut offset = Vec::with_capacity(channels);
        for channel in 0..channels {
            let (min_value, max_value) = self.channel_min_max(channel, min_stretch, max_stretch);
            let delta = max_value - min_value;
            let channel_scale = if delta.is_finite() && delta != 0.0 {
                1.0 / delta
            } else {
                0.0
            };
            let channel_offset = if channel_scale == 0.0 {
                0.0
            } else {
                -min_value * channel_scale
            };
            scale.push(channel_scale);
            offset.push(channel_offset);
            for idx in (channel..self.pixels.len()).step_by(channels) {
                if self.is_masked_pixel(idx / channels) {
                    continue;
                }
                let value = self.pixels[idx];
                self.pixels[idx] = if value.is_finite() {
                    value * channel_scale + channel_offset
                } else {
                    value
                };
            }
        }
        self.enhancement_history
            .push(StretchRecord { scale, offset });
    }

    fn channel_min_max(
        &self,
        channel: usize,
        min_stretch: Option<f32>,
        max_stretch: Option<f32>,
    ) -> (f32, f32) {
        let min_value =
            min_stretch.unwrap_or_else(|| self.channel_extreme(channel, ChannelExtreme::Min));
        let max_value =
            max_stretch.unwrap_or_else(|| self.channel_extreme(channel, ChannelExtreme::Max));
        (min_value, max_value)
    }

    fn channel_extreme(&self, channel: usize, extreme: ChannelExtreme) -> f32 {
        let channels = self.mode.channels();
        let mut result = match extreme {
            ChannelExtreme::Min => f32::INFINITY,
            ChannelExtreme::Max => f32::NEG_INFINITY,
        };
        for idx in (channel..self.pixels.len()).step_by(channels) {
            if self.is_masked_pixel(idx / channels) {
                continue;
            }
            let value = self.pixels[idx];
            if value.is_finite() {
                result = match extreme {
                    ChannelExtreme::Min => result.min(value),
                    ChannelExtreme::Max => result.max(value),
                };
            }
        }
        result
    }

    pub fn to_u8_image(&self, fill_value: u8) -> Result<Image> {
        let channels = self.mode.channels();
        let pixels = self
            .pixels
            .iter()
            .enumerate()
            .map(|(idx, value)| {
                if self.is_masked_pixel(idx / channels) || !value.is_finite() {
                    fill_value
                } else {
                    (value * 255.0).clamp(0.0, 255.0).round() as u8
                }
            })
            .collect();
        Image::from_pixels(self.mode, self.height, self.width, pixels)
    }

    pub fn mode(&self) -> ImageMode {
        self.mode
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.height, self.width)
    }

    pub fn pixels(&self) -> &[f32] {
        &self.pixels
    }

    pub fn enhancement_history(&self) -> &[StretchRecord] {
        &self.enhancement_history
    }

    fn is_masked_pixel(&self, pixel_index: usize) -> bool {
        self.mask
            .as_ref()
            .and_then(|mask| mask.is_masked(pixel_index))
            .unwrap_or(false)
    }
}

impl StretchRecord {
    pub fn scale(&self) -> &[f32] {
        &self.scale
    }

    pub fn offset(&self) -> &[f32] {
        &self.offset
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

    #[test]
    fn crude_stretch_normalizes_float_luma_in_place() {
        let mut image =
            FloatImage::from_pixels(ImageMode::Luma, 2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();

        image.crude_stretch_in_place(None, None);

        assert_pixels_close(image.pixels(), &[0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0]);
        assert_eq!(image.enhancement_history().len(), 1);
        assert_eq!(image.enhancement_history()[0].scale(), &[1.0 / 3.0]);
        assert_eq!(image.enhancement_history()[0].offset(), &[-1.0 / 3.0]);
    }

    #[test]
    fn crude_stretch_uses_explicit_limits_and_finalizes_to_u8() {
        let image = FloatImage::from_pixels(ImageMode::Luma, 1, 4, vec![0.0, 10.0, 20.0, 30.0])
            .unwrap()
            .crude_stretched(Some(0.0), Some(30.0));
        let final_image = image.to_u8_image(0).unwrap();

        assert_eq!(final_image.pixels(), &[0, 85, 170, 255]);
    }

    fn assert_pixels_close(left: &[f32], right: &[f32]) {
        assert_eq!(left.len(), right.len());
        for (left, right) in left.iter().zip(right) {
            assert!((left - right).abs() < 1e-6, "{left} != {right}");
        }
    }
}
