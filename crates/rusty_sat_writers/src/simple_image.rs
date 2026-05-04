//! Simple image writer foundations.
//!
//! Reference behavior inspected before implementation:
//! - `satpy/satpy/writers/simple_image.py` `PillowWriter.save_image` delegates
//!   image saving to Trollimage.
//! - `deps/trollimage/trollimage/xrimage.py` `XRImage.pil_save` detects the
//!   output format from the filename extension when no explicit format is
//!   provided.
//!
//! This slice intentionally implements PNG output for already-finalized u8 and
//! u16 images. Dataset enhancement, compression controls, and metadata parity
//! are separate roadmap items.

use crate::Writer;
use image::{ColorType, ImageFormat};
use png::BitDepth;
use rusty_sat_core::{Dataset, Result, RustySatError};
use rusty_sat_image::{Image, Image16, ImageMode};
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SimpleImageWriter {
    name: String,
}

impl SimpleImageWriter {
    pub fn new(name: impl Into<String>) -> Result<Self> {
        let name = name.into();
        if name.trim().is_empty() {
            return Err(RustySatError::invalid_input(
                "simple image writer name cannot be empty",
            ));
        }
        Ok(Self { name })
    }

    pub fn save_image16(&self, image: &Image16, path: &Path) -> Result<()> {
        write_png16_image(image, path)
    }
}

impl Default for SimpleImageWriter {
    fn default() -> Self {
        Self {
            name: "simple_image".to_string(),
        }
    }
}

impl Writer for SimpleImageWriter {
    fn name(&self) -> &str {
        &self.name
    }

    fn save_image(&self, image: &Image, path: &Path) -> Result<()> {
        write_png_image(image, path)
    }

    fn save_dataset(&self, dataset: &Dataset, path: &Path) -> Result<()> {
        let image = Image::from_luma_dataset(dataset)?;
        self.save_image(&image, path)
    }
}

pub fn write_png_image(image: &Image, path: impl AsRef<Path>) -> Result<()> {
    let path = path.as_ref();
    let format = image_format_from_path(path)?;
    if format != ImageFormat::Png {
        return Err(RustySatError::unsupported(format!(
            "simple image format '{}'",
            path.extension()
                .and_then(|extension| extension.to_str())
                .unwrap_or("<missing>")
        )));
    }

    let (height, width) = image.shape();
    image::save_buffer_with_format(
        path,
        image.pixels(),
        u32::try_from(width)
            .map_err(|_| RustySatError::invalid_input("image width does not fit u32"))?,
        u32::try_from(height)
            .map_err(|_| RustySatError::invalid_input("image height does not fit u32"))?,
        color_type(image.mode()),
        format,
    )
    .map_err(|err| RustySatError::invalid_input(format!("failed to save PNG image: {err}")))
}

pub fn write_png16_image(image: &Image16, path: impl AsRef<Path>) -> Result<()> {
    let path = path.as_ref();
    let format = image_format_from_path(path)?;
    if format != ImageFormat::Png {
        return Err(RustySatError::unsupported(format!(
            "simple image format '{}'",
            path.extension()
                .and_then(|extension| extension.to_str())
                .unwrap_or("<missing>")
        )));
    }

    let (height, width) = image.shape();
    let file = File::create(path)
        .map_err(|err| RustySatError::invalid_input(format!("failed to create PNG file: {err}")))?;
    let writer = BufWriter::new(file);
    let mut encoder = png::Encoder::new(
        writer,
        u32::try_from(width)
            .map_err(|_| RustySatError::invalid_input("image width does not fit u32"))?,
        u32::try_from(height)
            .map_err(|_| RustySatError::invalid_input("image height does not fit u32"))?,
    );
    encoder.set_color(png_color_type(image.mode()));
    encoder.set_depth(BitDepth::Sixteen);
    let mut writer = encoder.write_header().map_err(|err| {
        RustySatError::invalid_input(format!("failed to write PNG header: {err}"))
    })?;
    let bytes = u16_pixels_to_png_bytes(image.pixels());
    writer
        .write_image_data(&bytes)
        .map_err(|err| RustySatError::invalid_input(format!("failed to save PNG image: {err}")))
}

fn image_format_from_path(path: &Path) -> Result<ImageFormat> {
    let Some(extension) = path.extension().and_then(|extension| extension.to_str()) else {
        return Err(RustySatError::invalid_input(
            "image filename must include an extension",
        ));
    };
    if extension.eq_ignore_ascii_case("png") {
        Ok(ImageFormat::Png)
    } else {
        Err(RustySatError::unsupported(format!(
            "simple image format '{extension}'"
        )))
    }
}

fn color_type(mode: ImageMode) -> ColorType {
    match mode {
        ImageMode::Luma => ColorType::L8,
        ImageMode::Rgb => ColorType::Rgb8,
        ImageMode::Rgba => ColorType::Rgba8,
    }
}

fn png_color_type(mode: ImageMode) -> png::ColorType {
    match mode {
        ImageMode::Luma => png::ColorType::Grayscale,
        ImageMode::Rgb => png::ColorType::Rgb,
        ImageMode::Rgba => png::ColorType::Rgba,
    }
}

fn u16_pixels_to_png_bytes(pixels: &[u16]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(pixels.len() * 2);
    for pixel in pixels {
        bytes.extend_from_slice(&pixel.to_be_bytes());
    }
    bytes
}

#[cfg(test)]
mod tests {
    use super::*;
    use rusty_sat_core::{DataArray, DataId, Scene};
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn writes_luma_png_image() -> Result<()> {
        let path = temp_png_path("writes_luma_png_image");
        let image = Image::from_pixels(ImageMode::Luma, 1, 2, vec![0, 255])?;

        write_png_image(&image, &path)?;

        let decoded = image::open(&path)
            .map_err(|err| RustySatError::invalid_input(err.to_string()))?
            .into_luma8();
        assert_eq!(decoded.dimensions(), (2, 1));
        assert_eq!(decoded.as_raw(), &[0, 255]);
        fs::remove_file(path).ok();
        Ok(())
    }

    #[test]
    fn writes_rgb_png_image() -> Result<()> {
        let path = temp_png_path("writes_rgb_png_image");
        let image = Image::from_pixels(ImageMode::Rgb, 1, 2, vec![255, 0, 0, 0, 255, 0])?;

        SimpleImageWriter::default().save_image(&image, &path)?;

        let decoded = image::open(&path)
            .map_err(|err| RustySatError::invalid_input(err.to_string()))?
            .into_rgb8();
        assert_eq!(decoded.dimensions(), (2, 1));
        assert_eq!(decoded.as_raw(), &[255, 0, 0, 0, 255, 0]);
        fs::remove_file(path).ok();
        Ok(())
    }

    #[test]
    fn writes_rgba_png_image() -> Result<()> {
        let path = temp_png_path("writes_rgba_png_image");
        let image = Image::from_pixels(ImageMode::Rgba, 1, 1, vec![1, 2, 3, 4])?;

        SimpleImageWriter::default().save_image(&image, &path)?;

        let decoded = image::open(&path)
            .map_err(|err| RustySatError::invalid_input(err.to_string()))?
            .into_rgba8();
        assert_eq!(decoded.as_raw(), &[1, 2, 3, 4]);
        fs::remove_file(path).ok();
        Ok(())
    }

    #[test]
    fn writes_16_bit_luma_png_image() -> Result<()> {
        let path = temp_png_path("writes_16_bit_luma_png_image");
        let image = Image16::from_pixels(ImageMode::Luma, 1, 2, vec![0, 65535])?;

        write_png16_image(&image, &path)?;

        let decoded = image::open(&path)
            .map_err(|err| RustySatError::invalid_input(err.to_string()))?
            .into_luma16();
        assert_eq!(decoded.dimensions(), (2, 1));
        assert_eq!(decoded.as_raw(), &[0, 65535]);
        fs::remove_file(path).ok();
        Ok(())
    }

    #[test]
    fn writes_16_bit_rgb_png_image() -> Result<()> {
        let path = temp_png_path("writes_16_bit_rgb_png_image");
        let image = Image16::from_pixels(ImageMode::Rgb, 1, 2, vec![65535, 0, 0, 0, 65535, 1])?;

        SimpleImageWriter::default().save_image16(&image, &path)?;

        let decoded = image::open(&path)
            .map_err(|err| RustySatError::invalid_input(err.to_string()))?
            .into_rgb16();
        assert_eq!(decoded.dimensions(), (2, 1));
        assert_eq!(decoded.as_raw(), &[65535, 0, 0, 0, 65535, 1]);
        fs::remove_file(path).ok();
        Ok(())
    }

    #[test]
    fn writes_16_bit_rgba_png_image() -> Result<()> {
        let path = temp_png_path("writes_16_bit_rgba_png_image");
        let image = Image16::from_pixels(
            ImageMode::Rgba,
            1,
            2,
            vec![65535, 0, 0, 65535, 0, 65535, 0, 32768],
        )?;

        SimpleImageWriter::default().save_image16(&image, &path)?;

        let decoded = image::open(&path)
            .map_err(|err| RustySatError::invalid_input(err.to_string()))?
            .into_rgba16();
        assert_eq!(decoded.dimensions(), (2, 1));
        assert_eq!(decoded.as_raw(), &[65535, 0, 0, 65535, 0, 65535, 0, 32768]);
        fs::remove_file(path).ok();
        Ok(())
    }

    #[test]
    fn saves_dataset_as_luma_png() -> Result<()> {
        let path = temp_png_path("saves_dataset_as_luma_png");
        let dataset = Dataset::new(DataId::new("VIS006")?).with_array(
            DataArray::<u8>::from_vec_named([1, 2], ["y", "x"], vec![0, 10])?,
        );

        SimpleImageWriter::default().save_dataset(&dataset, &path)?;

        let decoded = image::open(&path)
            .map_err(|err| RustySatError::invalid_input(err.to_string()))?
            .into_luma8();
        assert_eq!(decoded.dimensions(), (2, 1));
        assert_eq!(decoded.as_raw(), &[0, 255]);
        fs::remove_file(path).ok();
        Ok(())
    }

    #[test]
    fn scene_saves_dataset_as_luma_png() -> Result<()> {
        let path = temp_png_path("scene_saves_dataset_as_luma_png");
        let data_id = DataId::new("VIS006")?;
        let dataset = Dataset::new(data_id.clone()).with_array(DataArray::<u8>::from_vec_named(
            [1, 2],
            ["y", "x"],
            vec![0, 10],
        )?);
        let mut scene = Scene::new();
        scene.insert_dataset(dataset);

        scene.save_dataset(&data_id, &SimpleImageWriter::default(), &path)?;

        let decoded = image::open(&path)
            .map_err(|err| RustySatError::invalid_input(err.to_string()))?
            .into_luma8();
        assert_eq!(decoded.as_raw(), &[0, 255]);
        fs::remove_file(path).ok();
        Ok(())
    }

    #[test]
    fn rejects_non_png_extension() -> Result<()> {
        let image = Image::from_pixels(ImageMode::Luma, 1, 1, vec![0])?;
        let err =
            write_png_image(&image, temp_path("rejects_non_png_extension", "jpg")).unwrap_err();

        assert!(err.to_string().contains("unsupported feature"));
        Ok(())
    }

    #[test]
    fn converts_u16_pixels_to_png_big_endian_bytes() {
        assert_eq!(
            u16_pixels_to_png_bytes(&[0x1234, 0xabcd]),
            vec![0x12, 0x34, 0xab, 0xcd]
        );
    }

    fn temp_png_path(name: &str) -> std::path::PathBuf {
        temp_path(name, "png")
    }

    fn temp_path(name: &str, extension: &str) -> std::path::PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time")
            .as_nanos();
        std::env::temp_dir().join(format!("rusty_sat_{name}_{nanos}.{extension}"))
    }
}
