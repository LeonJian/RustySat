//! SelfSharpenedRGB compositor — Satpy `SelfSharpenedRGB` equivalent.
//!
//! Reference: `satpy/satpy/composites/resolution.py`
//!
//! Algorithm:
//! 1. R_low = mean4(R) — 2×2 nanmean, edge-padded, same shape as R(0.5km)
//! 2. ratio = clip(R / R_low, 0, 1.5), NaN/inf/neg → 1.0
//! 3. G_gm = repeat2x(G_1km), B_gm = repeat2x(B_1km)
//! 4. G = G_gm * ratio, B = B_gm * ratio
//! 5. stack [R, G, B] → band-major [3, y, x]
//!
//! Memory: consumes inputs, reuses R's buffer for ratio after computation,
//! uses rayon parallel per-row for down-sampling.

use rayon::prelude::*;
use rusty_sat_core::{
    AnyDataArray, DataArray, DataId, Dataset, MetadataValue, Result, RustySatError,
};

/// Self-sharpened RGB compositor.
///
/// Creates a true-color image where the high-resolution red channel (0.5 km)
/// sharpens the lower-resolution green and blue channels (1 km).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SelfSharpenedRgb {
    name: String,
}

impl SelfSharpenedRgb {
    pub fn new(name: impl Into<String>) -> Result<Self> {
        let name = name.into();
        if name.trim().is_empty() {
            return Err(RustySatError::invalid_input(
                "compositor name cannot be empty",
            ));
        }
        Ok(Self { name })
    }

    /// Consume three single-band datasets (R, G, B) and produce a
    /// self-sharpened band-major RGB dataset.
    ///
    /// R must be at target resolution (e.g. 0.5 km), G and B are
    /// up-sampled and sharpened by the red channel texture.
    pub fn compose_rgb_owned(self, inputs: Vec<Dataset>) -> Result<Dataset> {
        if inputs.len() != 3 {
            return Err(RustySatError::invalid_input(format!(
                "SelfSharpenedRgb requires exactly 3 bands, got {}",
                inputs.len()
            )));
        }
        let mut iter = inputs.into_iter();
        let r = iter.next().expect("red");
        let g = iter.next().expect("green");
        let b = iter.next().expect("blue");

        let (h_r, w_r) = validate_2d(&r, "red")?;
        let (h_g, w_g) = validate_2d(&g, "green")?;
        let (h_b, w_b) = validate_2d(&b, "blue")?;
        if (h_g, w_g) != (h_b, w_b) {
            return Err(RustySatError::invalid_input(format!(
                "green ({h_g}×{w_g}) and blue ({h_b}×{w_b}) must have same shape"
            )));
        }
        let scale = h_r / h_g;
        if scale == 0 || h_r != h_g * scale || w_r != w_g * scale {
            return Err(RustySatError::invalid_input(format!(
                "red ({h_r}×{w_r}) must be integer multiple of green ({h_g}×{w_g})"
            )));
        }

        // Step 1: extract f32 values
        let red = into_f32(r.into_array().ok_or_else(|| missing("red"))?);
        let green_1km = into_f32(g.into_array().ok_or_else(|| missing("green"))?);
        let blue_1km = into_f32(b.into_array().ok_or_else(|| missing("blue"))?);

        // Step 2: R_low = 2×2 nanmean of red, then overwrite R_low with ratio
        let mut ratio = mean4_2x2(&red, h_r, w_r, scale);
        compute_ratio(&mut ratio, &red); // ratio[i] = clip(red[i] / ratio[i], 0, 1.5)

        // Step 3-5: write the three band sections of the band-major output.
        // Red is moved in (freeing its buffer), then green and blue are
        // up-sampled and sharpened in a single fused pass reading `ratio`, so
        // no separate up-sampled 0.5 km intermediates are ever allocated.
        let band_count = h_r * w_r;
        let mut rgb = Vec::with_capacity(3 * band_count);
        rgb.extend(red);
        // Size the green/blue sections (zero-filled, then overwritten by the
        // fused pass below) so the output can be split into disjoint slices.
        rgb.resize(3 * band_count, 0.0);
        let (green_section, blue_section) = rgb[band_count..].split_at_mut(band_count);
        green_section
            .par_chunks_mut(w_r)
            .zip(blue_section.par_chunks_mut(w_r))
            .enumerate()
            .for_each(|(oi, (green_row, blue_row))| {
                let si = oi / scale;
                for oj in 0..w_r {
                    let r = ratio[oi * w_r + oj];
                    let src_idx = si * w_g + oj / scale;
                    let gv = green_1km[src_idx];
                    let bv = blue_1km[src_idx];
                    green_row[oj] = if gv.is_finite() && r.is_finite() {
                        gv * r
                    } else {
                        gv
                    };
                    blue_row[oj] = if bv.is_finite() && r.is_finite() {
                        bv * r
                    } else {
                        bv
                    };
                }
            });
        drop(ratio);
        drop(green_1km);
        drop(blue_1km);

        let array =
            DataArray::<f32>::from_vec_named(vec![3, h_r, w_r], vec!["bands", "y", "x"], rgb)?;
        let mut ds = Dataset::new(DataId::new(&self.name)?).with_array(array);
        ds.insert_attr("mode", MetadataValue::string("RGB"))?;
        Ok(ds)
    }
}

fn missing(name: &str) -> RustySatError {
    RustySatError::invalid_input(format!("{name} band has no array"))
}

fn validate_2d(ds: &Dataset, label: &str) -> Result<(usize, usize)> {
    let arr = ds
        .array()
        .ok_or_else(|| RustySatError::invalid_input(format!("{label} band has no array")))?;
    let shp = arr.shape();
    if shp.len() != 2 {
        return Err(RustySatError::invalid_input(format!(
            "{label} must be 2D, got {}D",
            shp.len()
        )));
    }
    Ok((shp[0], shp[1]))
}

/// Convert AnyDataArray to Vec<f32>.
fn into_f32(arr: AnyDataArray) -> Vec<f32> {
    match arr {
        AnyDataArray::F32(a) => a.into_values(),
        AnyDataArray::F64(a) => a.into_values().into_iter().map(|v| v as f32).collect(),
        AnyDataArray::U8(a) => a.into_values().into_iter().map(|v| v as f32).collect(),
        AnyDataArray::U16(a) => a.into_values().into_iter().map(|v| v as f32).collect(),
        AnyDataArray::I16(a) => a.into_values().into_iter().map(|v| v as f32).collect(),
    }
}

/// 2×2 nanmean with edge-padding. Output has same shape as input.
/// `scale` is the integer ratio (typically 2 for 0.5km → 1km down-sampling).
fn mean4_2x2(data: &[f32], h: usize, w: usize, scale: usize) -> Vec<f32> {
    let n = h * w;
    let mut low = vec![0.0_f32; n];
    let stride = scale;
    low.par_chunks_mut(w).enumerate().for_each(|(i, row_out)| {
        let i0 = (i / stride) * stride;
        let i1 = (i0 + stride - 1).min(h - 1);
        for (j, val) in row_out.iter_mut().enumerate() {
            let j0 = (j / stride) * stride;
            let j1 = (j0 + stride - 1).min(w - 1);
            let mut sum = 0.0_f64;
            let mut cnt = 0u32;
            for ri in i0..=i1 {
                for rj in j0..=j1 {
                    let v = data[ri * w + rj];
                    if v.is_finite() {
                        sum += v as f64;
                        cnt += 1;
                    }
                }
            }
            *val = if cnt > 0 {
                (sum / cnt as f64) as f32
            } else {
                f32::NAN
            };
        }
    });
    low
}

/// `out[i] = clip(data[i] / out[i], 0, 1.5)`, NaN/inf/neg → 1.0.
fn compute_ratio(out: &mut [f32], data: &[f32]) {
    out.par_iter_mut().zip(data.par_iter()).for_each(|(o, &d)| {
        if !(*o).is_finite() || !d.is_finite() || *o == 0.0 {
            *o = 1.0;
        } else {
            let r = d / *o;
            *o = if r.is_finite() && r >= 0.0 {
                r.clamp(0.0, 1.5)
            } else {
                1.0
            };
        }
    });
}

/// Integer up-sample: each pixel repeated `scale`×`scale` times.
///
/// Kept for unit tests; the compositor itself fuses up-sampling with the
/// ratio sharpening to avoid full-resolution intermediates.
#[cfg(test)]
fn repeat_2d(data: &[f32], h: usize, w: usize, scale: usize) -> Vec<f32> {
    let out_h = h * scale;
    let out_w = w * scale;
    let mut out = vec![0.0_f32; out_h * out_w];
    out.par_chunks_mut(out_w)
        .enumerate()
        .for_each(|(oi, row_out)| {
            let si = oi / scale;
            for (oj, val) in row_out.iter_mut().enumerate() {
                *val = data[si * w + oj / scale];
            }
        });
    out
}

/// Element-wise: `data[i] *= ratio[i]`.
///
/// Kept for unit tests; the compositor itself fuses the ratio into the
/// up-sampled write.
#[cfg(test)]
fn apply_ratio(data: &mut [f32], ratio: &[f32]) {
    data.par_iter_mut()
        .zip(ratio.par_iter())
        .for_each(|(d, &r)| {
            if d.is_finite() && r.is_finite() {
                *d *= r;
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mean4_2x2_uniform_returns_same() {
        let data = vec![5.0_f32; 4 * 4];
        let result = mean4_2x2(&data, 4, 4, 2);
        assert_eq!(result.len(), 16);
        for v in &result {
            assert!((v - 5.0).abs() < 0.001, "got {v}");
        }
    }

    #[test]
    fn mean4_2x2_averages_block() {
        // 2x2 blocks: [1,2;3,4] → mean=2.5
        let data = vec![1.0_f32, 2.0, 3.0, 4.0];
        let result = mean4_2x2(&data, 2, 2, 2);
        for v in &result {
            assert!((v - 2.5).abs() < 0.001, "got {v}");
        }
    }

    #[test]
    fn mean4_2x2_handles_nan() {
        let data = vec![1.0_f32, 2.0, 3.0, f32::NAN];
        let result = mean4_2x2(&data, 2, 2, 2);
        for v in &result {
            assert!((v - 2.0).abs() < 0.01, "got {v}"); // (1+2+3)/3 = 2.0
        }
    }

    #[test]
    fn ratio_clips_to_1_5() {
        let mut out = vec![1.0_f32];
        let data = vec![3.0_f32];
        compute_ratio(&mut out, &data);
        assert!((out[0] - 1.5).abs() < 0.001);
    }

    #[test]
    fn ratio_handles_nan_low() {
        let mut out = vec![f32::NAN];
        let data = vec![5.0_f32];
        compute_ratio(&mut out, &data);
        assert!((out[0] - 1.0).abs() < 0.001);
    }

    #[test]
    fn ratio_handles_zero_low() {
        let mut out = vec![0.0_f32];
        let data = vec![5.0_f32];
        compute_ratio(&mut out, &data);
        assert!((out[0] - 1.0).abs() < 0.001);
    }

    #[test]
    fn repeat_2x_doubles() {
        let data = vec![1.0_f32, 2.0, 3.0, 4.0];
        let result = repeat_2d(&data, 2, 2, 2);
        assert_eq!(
            result,
            vec![1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 3.0, 3.0, 4.0, 4.0,]
        );
    }

    #[test]
    fn apply_ratio_multiplies() {
        let mut data = vec![2.0_f32, 4.0, 6.0];
        let ratio = vec![0.5_f32, 1.0, 2.0];
        apply_ratio(&mut data, &ratio);
        assert_eq!(data, vec![1.0, 4.0, 12.0]);
    }
}
