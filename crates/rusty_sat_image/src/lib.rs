//! Image types, color conversion, and enhancement operations.
//!
//! This crate bridges the gap between scientific float arrays and display-ready
//! pixel buffers. It converts [`Dataset`] arrays into [`Image`] (8-bit) or
//! [`Image16`] (16-bit) with automatic contrast stretching and optional gamma
//! correction.
//!
//! # Image Types
//!
//! - [`FloatImage<T>`] — intermediate float representation (f32 or f64).
//!   Enhancement operations (stretch, gamma, invert) are applied here and
//!   tracked in history for reproducibility.
//! - [`Image`] — 8-bit output (u8 pixels). Produced by `to_u8_image()` after
//!   all enhancements are applied.
//! - [`Image16`] — 16-bit output (u16 pixels). For HDR / scientific display.
//!
//! # ImageMode
//!
//! [`ImageMode::Luma`] (1 channel), [`ImageMode::Rgb`] (3 channels), or
//! [`ImageMode::Rgba`] (4 channels with alpha).
//!
//! # Quick Start
//!
//! ```ignore
//! use rusty_sat_image::{Image, FloatImage};
//! let img = Image::from_luma_dataset(&dataset)?;
//! // img.pixels() → &[u8] with auto-stretched 0–255 range
//! ```

use rusty_sat_core::{
    AnyDataArray, DataArray, Dataset, NumericElement, Result, RustySatError, ValidityMask,
};

pub trait ImageFloat: Copy + Clone + PartialEq + PartialOrd + std::fmt::Debug + 'static {
    fn from_f64(value: f64) -> Self;
    fn to_f64(self) -> f64;
    fn is_finite(self) -> bool;
}

impl ImageFloat for f32 {
    fn from_f64(value: f64) -> Self {
        value as f32
    }

    fn to_f64(self) -> f64 {
        f64::from(self)
    }

    fn is_finite(self) -> bool {
        self.is_finite()
    }
}

impl ImageFloat for f64 {
    fn from_f64(value: f64) -> Self {
        value
    }

    fn to_f64(self) -> f64 {
        self
    }

    fn is_finite(self) -> bool {
        self.is_finite()
    }
}

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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Image16 {
    mode: ImageMode,
    height: usize,
    width: usize,
    pixels: Vec<u16>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FloatImage<T: ImageFloat = f32> {
    mode: ImageMode,
    height: usize,
    width: usize,
    pixels: Vec<T>,
    mask: Option<ValidityMask>,
    enhancement_history: Vec<StretchRecord<T>>,
    gamma_history: Vec<Vec<T>>,
    invert_history: Vec<Vec<bool>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StretchRecord<T: ImageFloat = f32> {
    scale: Vec<T>,
    offset: Vec<T>,
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
        FloatImage::<f32>::from_luma_array(array)?
            .crude_stretched(None, None)
            .to_u8_image(0)
    }

    pub fn from_rgb_dataset(dataset: &Dataset) -> Result<Self> {
        let Some(array) = dataset.array() else {
            return Err(RustySatError::invalid_input(format!(
                "dataset '{}' has no array data",
                dataset.id().name()
            )));
        };
        Self::from_rgb_array(array)
    }

    pub fn from_rgb_array(array: &AnyDataArray) -> Result<Self> {
        FloatImage::<f32>::from_rgb_array(array)?
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

impl Image16 {
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
        pixels: Vec<u16>,
    ) -> Result<Self> {
        if height == 0 || width == 0 {
            return Err(RustySatError::invalid_input(
                "image dimensions must be non-zero",
            ));
        }
        let expected = checked_pixel_len(mode, height, width)?;
        if pixels.len() != expected {
            return Err(RustySatError::invalid_input(format!(
                "16-bit image has {} pixels but {mode:?} shape ({height}, {width}) requires {expected} values",
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
        FloatImage::<f64>::from_luma_array(array)?
            .crude_stretched(None, None)
            .to_u16_image(0)
    }

    pub fn from_rgb_dataset(dataset: &Dataset) -> Result<Self> {
        let Some(array) = dataset.array() else {
            return Err(RustySatError::invalid_input(format!(
                "dataset '{}' has no array data",
                dataset.id().name()
            )));
        };
        Self::from_rgb_array(array)
    }

    pub fn from_rgb_array(array: &AnyDataArray) -> Result<Self> {
        FloatImage::<f64>::from_rgb_array(array)?
            .crude_stretched(None, None)
            .to_u16_image(0)
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

    pub fn pixels(&self) -> &[u16] {
        &self.pixels
    }

    pub fn into_pixels(self) -> Vec<u16> {
        self.pixels
    }
}

impl<T: ImageFloat> FloatImage<T> {
    pub fn from_pixels(
        mode: ImageMode,
        height: usize,
        width: usize,
        pixels: Vec<T>,
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
            gamma_history: Vec::new(),
            invert_history: Vec::new(),
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

    pub fn from_rgb_dataset(dataset: &Dataset) -> Result<Self> {
        let Some(array) = dataset.array() else {
            return Err(RustySatError::invalid_input(format!(
                "dataset '{}' has no array data",
                dataset.id().name()
            )));
        };
        Self::from_rgb_array(array)
    }

    pub fn from_rgb_array(array: &AnyDataArray) -> Result<Self> {
        let (height, width, pixel_count) = require_band_major_rgb_shape(array)?;
        let mut image = match array {
            AnyDataArray::F32(array) => Self::from_numeric_rgb_array(array, height, width)?,
            AnyDataArray::F64(array) => Self::from_numeric_rgb_array(array, height, width)?,
            AnyDataArray::U8(array) => Self::from_numeric_rgb_array(array, height, width)?,
            AnyDataArray::U16(array) => Self::from_numeric_rgb_array(array, height, width)?,
            AnyDataArray::I16(array) => Self::from_numeric_rgb_array(array, height, width)?,
        };
        if let Some(mask) = array.mask() {
            image.mask = Some(pixel_mask_from_band_major_rgb_mask(mask, pixel_count)?);
        }
        Ok(image)
    }

    fn from_numeric_luma_array<U: NumericElement>(
        array: &DataArray<U>,
        height: usize,
        width: usize,
    ) -> Result<Self> {
        let pixels = array
            .values()
            .iter()
            .map(|value| T::from_f64(value.to_f64()))
            .collect();
        Self::from_pixels(ImageMode::Luma, height, width, pixels)
    }

    fn from_numeric_rgb_array<U: NumericElement>(
        array: &DataArray<U>,
        height: usize,
        width: usize,
    ) -> Result<Self> {
        let pixel_count = height
            .checked_mul(width)
            .ok_or_else(|| RustySatError::invalid_input("RGB image shape is too large"))?;
        let values = array.values();
        let mut pixels = Vec::with_capacity(pixel_count * ImageMode::Rgb.channels());
        for pixel_index in 0..pixel_count {
            for band in 0..ImageMode::Rgb.channels() {
                pixels.push(T::from_f64(
                    values[band * pixel_count + pixel_index].to_f64(),
                ));
            }
        }
        Self::from_pixels(ImageMode::Rgb, height, width, pixels)
    }

    pub fn crude_stretched(mut self, min_stretch: Option<T>, max_stretch: Option<T>) -> Self {
        self.crude_stretch_in_place(min_stretch, max_stretch);
        self
    }

    pub fn crude_stretch_in_place(&mut self, min_stretch: Option<T>, max_stretch: Option<T>) {
        let channels = self.mode.channels();
        let mut scale = Vec::with_capacity(channels);
        let mut offset = Vec::with_capacity(channels);
        for channel in 0..channels {
            let (min_value, max_value) = self.channel_min_max(channel, min_stretch, max_stretch);
            let min_value = min_value.to_f64();
            let max_value = max_value.to_f64();
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
            scale.push(T::from_f64(channel_scale));
            offset.push(T::from_f64(channel_offset));
            for idx in (channel..self.pixels.len()).step_by(channels) {
                if self.is_masked_pixel(idx / channels) {
                    continue;
                }
                let value = self.pixels[idx];
                self.pixels[idx] = if value.is_finite() {
                    T::from_f64(value.to_f64() * channel_scale + channel_offset)
                } else {
                    value
                };
            }
        }
        self.enhancement_history
            .push(StretchRecord { scale, offset });
    }

    pub fn gamma_corrected(mut self, gamma: T) -> Result<Self> {
        self.gamma_in_place(gamma)?;
        Ok(self)
    }

    pub fn gamma_in_place(&mut self, gamma: T) -> Result<()> {
        let gamma_values = vec![gamma; self.mode.channels()];
        self.gamma_channels_in_place(&gamma_values)
    }

    pub fn gamma_channels_in_place(&mut self, gamma: &[T]) -> Result<()> {
        let channels = self.mode.channels();
        validate_channel_count("gamma", gamma.len(), channels)?;
        let inverse_gamma: Vec<f64> = gamma
            .iter()
            .map(|value| {
                let value = value.to_f64();
                if !value.is_finite() || value <= 0.0 {
                    return Err(RustySatError::invalid_input(
                        "gamma values must be finite and positive",
                    ));
                }
                Ok(1.0 / value)
            })
            .collect::<Result<_>>()?;
        if inverse_gamma
            .iter()
            .all(|value| (*value - 1.0).abs() <= f64::EPSILON)
        {
            return Ok(());
        }
        for (channel, inverse_gamma) in inverse_gamma.into_iter().enumerate() {
            for idx in (channel..self.pixels.len()).step_by(channels) {
                if self.is_masked_pixel(idx / channels) {
                    continue;
                }
                let value = self.pixels[idx];
                if value.is_finite() {
                    self.pixels[idx] = T::from_f64(value.to_f64().max(0.0).powf(inverse_gamma));
                }
            }
        }
        self.gamma_history.push(gamma.to_vec());
        Ok(())
    }

    pub fn inverted(mut self, invert: bool) -> Self {
        self.invert_in_place(invert);
        self
    }

    pub fn invert_in_place(&mut self, invert: bool) {
        let invert_values = vec![invert; self.mode.channels()];
        self.invert_channels_in_place(&invert_values)
            .expect("scalar invert creates the correct channel count");
    }

    pub fn invert_channels_in_place(&mut self, invert: &[bool]) -> Result<()> {
        let channels = self.mode.channels();
        validate_channel_count("invert", invert.len(), channels)?;
        if invert.iter().all(|value| !*value) {
            return Ok(());
        }
        for (channel, invert) in invert.iter().copied().enumerate() {
            if !invert {
                continue;
            }
            for idx in (channel..self.pixels.len()).step_by(channels) {
                if self.is_masked_pixel(idx / channels) {
                    continue;
                }
                let value = self.pixels[idx];
                if value.is_finite() {
                    self.pixels[idx] = T::from_f64(1.0 - value.to_f64());
                }
            }
        }
        self.invert_history.push(invert.to_vec());
        Ok(())
    }

    fn channel_min_max(
        &self,
        channel: usize,
        min_stretch: Option<T>,
        max_stretch: Option<T>,
    ) -> (T, T) {
        let min_value =
            min_stretch.unwrap_or_else(|| self.channel_extreme(channel, ChannelExtreme::Min));
        let max_value =
            max_stretch.unwrap_or_else(|| self.channel_extreme(channel, ChannelExtreme::Max));
        (min_value, max_value)
    }

    fn channel_extreme(&self, channel: usize, extreme: ChannelExtreme) -> T {
        let channels = self.mode.channels();
        let mut result = match extreme {
            ChannelExtreme::Min => f64::INFINITY,
            ChannelExtreme::Max => f64::NEG_INFINITY,
        };
        for idx in (channel..self.pixels.len()).step_by(channels) {
            if self.is_masked_pixel(idx / channels) {
                continue;
            }
            let value = self.pixels[idx];
            if value.is_finite() {
                let value = value.to_f64();
                result = match extreme {
                    ChannelExtreme::Min => result.min(value),
                    ChannelExtreme::Max => result.max(value),
                };
            }
        }
        T::from_f64(result)
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
                    (value.to_f64() * 255.0).clamp(0.0, 255.0).round() as u8
                }
            })
            .collect();
        Image::from_pixels(self.mode, self.height, self.width, pixels)
    }

    pub fn to_u16_image(&self, fill_value: u16) -> Result<Image16> {
        let channels = self.mode.channels();
        let pixels = self
            .pixels
            .iter()
            .enumerate()
            .map(|(idx, value)| {
                if self.is_masked_pixel(idx / channels) || !value.is_finite() {
                    fill_value
                } else {
                    (value.to_f64() * 65_535.0).clamp(0.0, 65_535.0).round() as u16
                }
            })
            .collect();
        Image16::from_pixels(self.mode, self.height, self.width, pixels)
    }

    pub fn to_u8_rgba_image(
        &self,
        fill_value: u8,
        masked_alpha: u8,
        valid_alpha: u8,
    ) -> Result<Image> {
        rgba_image_from_float(
            &self.pixels,
            self.mask.as_ref(),
            self.mode,
            self.height,
            self.width,
            fill_value,
            masked_alpha,
            valid_alpha,
        )
    }

    pub fn into_u8_rgba_image(
        self,
        fill_value: u8,
        masked_alpha: u8,
        valid_alpha: u8,
    ) -> Result<Image> {
        // Consuming variant: reuses the same logic but takes ownership
        // so the float buffer is freed during conversion.
        rgba_image_from_float(
            &self.pixels,
            self.mask.as_ref(),
            self.mode,
            self.height,
            self.width,
            fill_value,
            masked_alpha,
            valid_alpha,
        )
    }

    pub fn mode(&self) -> ImageMode {
        self.mode
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.height, self.width)
    }

    pub fn pixels(&self) -> &[T] {
        &self.pixels
    }

    pub fn enhancement_history(&self) -> &[StretchRecord<T>] {
        &self.enhancement_history
    }

    pub fn gamma_history(&self) -> &[Vec<T>] {
        &self.gamma_history
    }

    pub fn invert_history(&self) -> &[Vec<bool>] {
        &self.invert_history
    }

    fn is_masked_pixel(&self, pixel_index: usize) -> bool {
        self.mask
            .as_ref()
            .and_then(|mask| mask.is_masked(pixel_index))
            .unwrap_or(false)
    }
}

impl<T: ImageFloat> StretchRecord<T> {
    pub fn scale(&self) -> &[T] {
        &self.scale
    }

    pub fn offset(&self) -> &[T] {
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

fn validate_channel_count(name: &str, provided: usize, expected: usize) -> Result<()> {
    if provided != expected {
        return Err(RustySatError::invalid_input(format!(
            "{name} provided {provided} channel values but image mode requires {expected}",
        )));
    }
    Ok(())
}

fn require_band_major_rgb_shape(array: &AnyDataArray) -> Result<(usize, usize, usize)> {
    let shape = array.shape();
    if shape.len() != 3 || shape[0] != ImageMode::Rgb.channels() {
        return Err(RustySatError::invalid_input(format!(
            "RGB image requires a band-major [3, y, x] array, got shape {:?}",
            shape
        )));
    }
    let dims = array.dims();
    if dims.len() != 3 || dims[0] != "bands" || dims[1] != "y" || dims[2] != "x" {
        return Err(RustySatError::invalid_input(format!(
            "RGB image requires dimensions ['bands', 'y', 'x'], got {:?}",
            dims
        )));
    }
    let height = shape[1];
    let width = shape[2];
    let pixel_count = height
        .checked_mul(width)
        .ok_or_else(|| RustySatError::invalid_input("RGB image shape is too large"))?;
    Ok((height, width, pixel_count))
}

fn pixel_mask_from_band_major_rgb_mask(
    mask: &ValidityMask,
    pixel_count: usize,
) -> Result<ValidityMask> {
    let expected = pixel_count
        .checked_mul(ImageMode::Rgb.channels())
        .ok_or_else(|| RustySatError::invalid_input("RGB mask shape is too large"))?;
    if mask.len() != expected {
        return Err(RustySatError::invalid_input(format!(
            "RGB mask length {} does not match expected band-major length {expected}",
            mask.len()
        )));
    }
    let mut output = ValidityMask::all_valid(pixel_count);
    for pixel_index in 0..pixel_count {
        let masked = (0..ImageMode::Rgb.channels()).any(|band| {
            mask.is_masked(band * pixel_count + pixel_index)
                .unwrap_or(false)
        });
        if masked {
            output.set_masked(pixel_index, true);
        }
    }
    Ok(output)
}

fn rgba_image_from_float<T: ImageFloat>(
    pixels: &[T],
    mask: Option<&ValidityMask>,
    mode: ImageMode,
    height: usize,
    width: usize,
    fill_value: u8,
    masked_alpha: u8,
    valid_alpha: u8,
) -> Result<Image> {
    let channels = mode.channels();
    let pixel_count = height * width;
    let mut rgba = Vec::with_capacity(pixel_count * ImageMode::Rgba.channels());
    for pixel_index in 0..pixel_count {
        let masked = mask
            .and_then(|mask| mask.is_masked(pixel_index))
            .unwrap_or(false);
        let base = pixel_index * channels;
        let to_u8 = |idx: usize, fill: u8, force_masked: bool| -> u8 {
            let value = pixels[idx];
            if force_masked || !value.is_finite() {
                fill
            } else {
                (value.to_f64() * 255.0).clamp(0.0, 255.0).round() as u8
            }
        };
        match mode {
            ImageMode::Luma => {
                let value = to_u8(base, fill_value, masked);
                rgba.extend_from_slice(&[value, value, value]);
                rgba.push(if masked { masked_alpha } else { valid_alpha });
            }
            ImageMode::Rgb => {
                for channel in 0..3 {
                    rgba.push(to_u8(base + channel, fill_value, masked));
                }
                rgba.push(if masked { masked_alpha } else { valid_alpha });
            }
            ImageMode::Rgba => {
                for channel in 0..3 {
                    rgba.push(to_u8(base + channel, fill_value, masked));
                }
                let alpha = if masked {
                    masked_alpha
                } else {
                    to_u8(base + 3, valid_alpha, false)
                };
                rgba.push(alpha);
            }
        }
    }
    Image::from_pixels(ImageMode::Rgba, height, width, rgba)
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
    fn constructs_16_bit_image() {
        let image = Image16::from_pixels(ImageMode::Rgb, 1, 2, vec![0, 1, 2, 3, 4, 65535]).unwrap();

        assert_eq!(image.mode(), ImageMode::Rgb);
        assert_eq!(image.shape(), (1, 2));
        assert_eq!(image.channels(), 3);
        assert_eq!(image.pixels(), &[0, 1, 2, 3, 4, 65535]);
        assert!(Image16::from_pixels(ImageMode::Rgb, 1, 2, vec![0; 5]).is_err());
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
    fn creates_16_bit_luma_image_from_dataset_array() {
        let array =
            DataArray::<f64>::from_vec_named(vec![1, 3], ["y", "x"], vec![1.0, 2.0, 3.0]).unwrap();
        let dataset = Dataset::new(rusty_sat_core::DataId::new("test").unwrap()).with_array(array);

        let image = Image16::from_luma_dataset(&dataset).unwrap();

        assert_eq!(image.mode(), ImageMode::Luma);
        assert_eq!(image.shape(), (1, 3));
        assert_eq!(image.pixels(), &[0, 32_768, 65_535]);
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
    fn creates_16_bit_luma_image_with_masked_pixels_filled_black() {
        let mask = rusty_sat_core::ValidityMask::from_masked_flags([false, true, false]);
        let array = DataArray::<f32>::from_vec_named(vec![1, 3], ["y", "x"], vec![1.0, 99.0, 3.0])
            .unwrap()
            .with_mask(mask)
            .unwrap();

        let image = Image16::from_luma_array(&array.into()).unwrap();

        assert_eq!(image.pixels(), &[0, 0, 65_535]);
    }

    #[test]
    fn creates_rgb_image_from_band_major_dataset_array() {
        let array = DataArray::<f64>::from_vec_named(
            vec![3, 1, 2],
            ["bands", "y", "x"],
            vec![0.0, 10.0, 0.0, 20.0, 0.0, 30.0],
        )
        .unwrap();
        let dataset =
            Dataset::new(rusty_sat_core::DataId::new("true_color").unwrap()).with_array(array);

        let image = Image::from_rgb_dataset(&dataset).unwrap();

        assert_eq!(image.mode(), ImageMode::Rgb);
        assert_eq!(image.shape(), (1, 2));
        assert_eq!(image.pixels(), &[0, 0, 0, 255, 255, 255]);
    }

    #[test]
    fn creates_float_rgb_image_by_interleaving_band_major_values() {
        let array = DataArray::<u16>::from_vec_named(
            vec![3, 1, 2],
            ["bands", "y", "x"],
            vec![1, 2, 10, 20, 100, 200],
        )
        .unwrap();

        let image = FloatImage::<f64>::from_rgb_array(&array.into()).unwrap();

        assert_eq!(image.mode(), ImageMode::Rgb);
        assert_eq!(image.shape(), (1, 2));
        assert_eq!(image.pixels(), &[1.0, 10.0, 100.0, 2.0, 20.0, 200.0]);
    }

    #[test]
    fn creates_rgb_image_with_any_masked_channel_filled_black() {
        let array = DataArray::<f64>::from_vec_named(
            vec![3, 1, 2],
            ["bands", "y", "x"],
            vec![0.0, 10.0, 0.0, 20.0, 0.0, 30.0],
        )
        .unwrap()
        .with_mask(rusty_sat_core::ValidityMask::from_masked_flags([
            false, false, false, true, false, false,
        ]))
        .unwrap();
        let dataset =
            Dataset::new(rusty_sat_core::DataId::new("true_color").unwrap()).with_array(array);

        let image = Image::from_rgb_dataset(&dataset).unwrap();

        assert_eq!(image.pixels(), &[0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn creates_16_bit_rgb_image_from_band_major_dataset_array() {
        let array = DataArray::<f64>::from_vec_named(
            vec![3, 1, 2],
            ["bands", "y", "x"],
            vec![0.0, 10.0, 0.0, 20.0, 0.0, 30.0],
        )
        .unwrap();
        let dataset =
            Dataset::new(rusty_sat_core::DataId::new("true_color").unwrap()).with_array(array);

        let image = Image16::from_rgb_dataset(&dataset).unwrap();

        assert_eq!(image.mode(), ImageMode::Rgb);
        assert_eq!(image.pixels(), &[0, 0, 0, 65_535, 65_535, 65_535]);
    }

    #[test]
    fn float_image_finalizes_to_16_bit_without_u8_quantization() {
        let image = FloatImage::from_pixels(ImageMode::Luma, 1, 3, vec![0.0_f64, 0.5, 1.0])
            .unwrap()
            .to_u16_image(0)
            .unwrap();

        assert_eq!(image.pixels(), &[0, 32_768, 65_535]);
    }

    #[test]
    fn crude_stretch_normalizes_float_luma_in_place() {
        let mut image =
            FloatImage::from_pixels(ImageMode::Luma, 2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();

        image.crude_stretch_in_place(None, None);

        assert_pixels_close(image.pixels(), &[0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0], 1e-6);
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

    #[test]
    fn crude_stretch_supports_f64_precision_path() {
        let mut image = FloatImage::<f64>::from_pixels(
            ImageMode::Luma,
            1,
            3,
            vec![1.0, 1.0 + f64::EPSILON, 1.0 + (2.0 * f64::EPSILON)],
        )
        .unwrap();

        image.crude_stretch_in_place(None, None);

        assert_pixels_close(image.pixels(), &[0.0, 0.5, 1.0], 1e-12);
        assert_eq!(image.enhancement_history().len(), 1);
        assert_eq!(image.enhancement_history()[0].scale().len(), 1);
    }

    #[test]
    fn f64_luma_conversion_preserves_precise_source_values() {
        let array =
            DataArray::<f64>::from_vec_named(vec![1, 2], ["y", "x"], vec![1.0, 1.0 + f64::EPSILON])
                .unwrap();
        let image = FloatImage::<f64>::from_luma_array(&array.into()).unwrap();

        assert_eq!(image.pixels(), &[1.0, 1.0 + f64::EPSILON]);
    }

    #[test]
    fn gamma_correction_clips_negative_values_and_preserves_f64_path() {
        let mut image =
            FloatImage::<f64>::from_pixels(ImageMode::Luma, 1, 3, vec![-1.0, 0.25, 1.0]).unwrap();

        image.gamma_in_place(2.0).unwrap();

        assert_pixels_close(image.pixels(), &[0.0, 0.5, 1.0], 1e-12);
        assert_eq!(image.gamma_history(), &[vec![2.0]]);
    }

    #[test]
    fn gamma_correction_supports_per_channel_values() {
        let mut image =
            FloatImage::from_pixels(ImageMode::Rgb, 1, 1, vec![0.25, 0.25, 0.25]).unwrap();

        image.gamma_channels_in_place(&[2.0, 1.0, 0.5]).unwrap();

        assert_pixels_close(image.pixels(), &[0.5, 0.25, 0.0625], 1e-6);
        assert!(image.gamma_channels_in_place(&[2.0]).is_err());
        assert!(image.gamma_in_place(0.0).is_err());
    }

    #[test]
    fn invert_uses_trollimage_black_white_semantics() {
        let mut image =
            FloatImage::from_pixels(ImageMode::Rgb, 1, 2, vec![0.0, 0.25, 1.0, 1.5, -0.5, 0.5])
                .unwrap();

        image
            .invert_channels_in_place(&[true, false, true])
            .unwrap();

        assert_pixels_close(image.pixels(), &[1.0, 0.25, 0.0, -0.5, -0.5, 0.5], 1e-6);
        assert_eq!(image.invert_history(), &[vec![true, false, true]]);
    }

    #[test]
    fn rgba_finalization_uses_mask_as_alpha_without_losing_float_source() {
        let mask = rusty_sat_core::ValidityMask::from_masked_flags([false, true]);
        let mut image =
            FloatImage::<f64>::from_pixels(ImageMode::Luma, 1, 2, vec![0.5, 0.75]).unwrap();
        image.mask = Some(mask);

        let rgba = image.to_u8_rgba_image(0, 0, 255).unwrap();

        assert_eq!(rgba.mode(), ImageMode::Rgba);
        assert_eq!(rgba.pixels(), &[128, 128, 128, 255, 0, 0, 0, 0]);
        assert_eq!(image.pixels(), &[0.5, 0.75]);
    }

    #[test]
    fn into_u8_rgba_image_consumes_float_buffer() {
        let mask = rusty_sat_core::ValidityMask::from_masked_flags([false, true]);
        let mut image =
            FloatImage::<f64>::from_pixels(ImageMode::Luma, 1, 2, vec![0.5, 0.75]).unwrap();
        image.mask = Some(mask);

        let rgba = image.into_u8_rgba_image(0, 0, 255).unwrap();

        assert_eq!(rgba.mode(), ImageMode::Rgba);
        assert_eq!(rgba.pixels(), &[128, 128, 128, 255, 0, 0, 0, 0]);
    }

    #[test]
    fn gamma_identity_is_noop_without_history() {
        let mut image = FloatImage::from_pixels(ImageMode::Luma, 1, 2, vec![0.25, 0.75]).unwrap();

        image.gamma_in_place(1.0).unwrap();

        assert_pixels_close(image.pixels(), &[0.25, 0.75], 1e-12);
        assert!(image.gamma_history().is_empty());
    }

    #[test]
    fn invert_all_false_is_noop_without_history() {
        let mut image = FloatImage::from_pixels(ImageMode::Luma, 1, 2, vec![0.25, 0.75]).unwrap();

        image.invert_channels_in_place(&[false]).unwrap();

        assert_pixels_close(image.pixels(), &[0.25, 0.75], 1e-12);
        assert!(image.invert_history().is_empty());
    }

    #[test]
    fn gamma_skips_masked_pixels() {
        let mask = rusty_sat_core::ValidityMask::from_masked_flags([false, true]);
        let mut image =
            FloatImage::<f64>::from_pixels(ImageMode::Luma, 1, 2, vec![0.25, 0.75]).unwrap();
        image.mask = Some(mask);

        image.gamma_in_place(2.0).unwrap();

        assert_pixels_close(image.pixels(), &[0.5, 0.75], 1e-12);
    }

    #[test]
    fn invert_skips_masked_pixels() {
        let mask = rusty_sat_core::ValidityMask::from_masked_flags([false, true]);
        let mut image =
            FloatImage::<f64>::from_pixels(ImageMode::Luma, 1, 2, vec![0.25, 0.75]).unwrap();
        image.mask = Some(mask);

        image.invert_channels_in_place(&[true]).unwrap();

        assert_pixels_close(image.pixels(), &[0.75, 0.75], 1e-12);
    }

    #[test]
    fn to_u8_rgba_image_handles_rgb_input() {
        let image = FloatImage::from_pixels(ImageMode::Rgb, 1, 1, vec![1.0, 0.5, 0.0]).unwrap();

        let rgba = image.to_u8_rgba_image(0, 0, 255).unwrap();

        assert_eq!(rgba.mode(), ImageMode::Rgba);
        assert_eq!(rgba.pixels(), &[255, 128, 0, 255]);
    }

    #[test]
    fn to_u8_rgba_image_preserves_source_alpha_for_rgba_input() {
        let image =
            FloatImage::from_pixels(ImageMode::Rgba, 1, 1, vec![1.0, 0.5, 0.0, 0.5]).unwrap();

        let rgba = image.to_u8_rgba_image(0, 0, 255).unwrap();

        assert_eq!(rgba.pixels(), &[255, 128, 0, 128]);
    }

    #[test]
    fn gamma_history_accumulates_multiple_calls() {
        let mut image = FloatImage::from_pixels(ImageMode::Luma, 1, 1, vec![0.25]).unwrap();

        image.gamma_in_place(2.0).unwrap();
        image.gamma_in_place(3.0).unwrap();

        assert_eq!(image.gamma_history(), &[vec![2.0], vec![3.0]]);
    }

    #[test]
    fn gamma_corrected_consuming_variant_produces_same_result() {
        let mut in_place_img =
            FloatImage::from_pixels(ImageMode::Luma, 1, 2, vec![0.25, 0.75]).unwrap();
        in_place_img.gamma_in_place(2.0).unwrap();

        let consuming_img =
            FloatImage::from_pixels(ImageMode::Luma, 1, 2, vec![0.25, 0.75]).unwrap();
        let result = consuming_img.gamma_corrected(2.0).unwrap();

        assert_pixels_close(result.pixels(), in_place_img.pixels(), 1e-12);
    }

    #[test]
    fn inverted_consuming_variant_produces_same_result() {
        let mut in_place_img =
            FloatImage::from_pixels(ImageMode::Luma, 1, 2, vec![0.25, 0.75]).unwrap();
        in_place_img.invert_in_place(true);

        let consuming_img =
            FloatImage::from_pixels(ImageMode::Luma, 1, 2, vec![0.25, 0.75]).unwrap();
        let result = consuming_img.inverted(true);

        assert_pixels_close(result.pixels(), in_place_img.pixels(), 1e-12);
    }

    fn assert_pixels_close<T: ImageFloat>(left: &[T], right: &[T], tolerance: f64) {
        assert_eq!(left.len(), right.len());
        for (left, right) in left.iter().zip(right) {
            let left = left.to_f64();
            let right = right.to_f64();
            assert!((left - right).abs() < tolerance, "{left} != {right}");
        }
    }
}
