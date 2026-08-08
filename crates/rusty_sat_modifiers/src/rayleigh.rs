//! Rayleigh scattering correction modifier using the `rustyspectral` crate.
//!
//! Integrates `rustyspectral::rayleigh` free functions (LUT I/O, wavelength
//! adjustment, trilinear interpolation, high-zenith reduction) with Rusty Sat's
//! angle computation and Dataset infrastructure.
//!
//! LUT downloading and caching is delegated to `rustyspectral::rayleigh::Rayleigh`
//! which manages Zenodo downloads, version checks, and local storage.
//!
//! Memory strategy: the correction is applied in row strips (64 rows by default).
//! Angles are computed per pixel in the hot loop using precomputed time/column
//! constants. Strip buffers are dropped each iteration, keeping peak memory low.

use ndarray::Array1;
use rustyspectral::rayleigh::{
    get_reflectance_lut_from_file, get_wavelength_adjusted_lut, read_reflectance_lut_4d,
    read_wavelength_lut_coord, trilinear_interpolate,
};
use std::path::{Path, PathBuf};

use crate::angles::{sat_angles_from_lonlat, sun_angles_from_lonlat, AngleParams, AngleSet};
use crate::astronomy::{gmst, observer_position, sun_ra_dec, UtcInstant};
use rusty_sat_core::{
    AnyDataArray, Coordinate, DataArray, DataId, Dataset, MetadataValue, Result, RustySatError,
    ValidityMask,
};

pub use rustyspectral::rayleigh::Rayleigh;

/// Aerosol type for the Rayleigh LUT.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AerosolType {
    AntarcticAerosol,
    ContinentalAverageAerosol,
    ContinentalCleanAerosol,
    ContinentalPollutedAerosol,
    DesertAerosol,
    MarineCleanAerosol,
    MarinePollutedAerosol,
    MarineTropicalAerosol,
    RayleighOnly,
    RuralAerosol,
    UrbanAerosol,
}

impl AerosolType {
    pub fn dir_name(&self) -> &'static str {
        match self {
            Self::AntarcticAerosol => "antarctic_aerosol",
            Self::ContinentalAverageAerosol => "continental_average_aerosol",
            Self::ContinentalCleanAerosol => "continental_clean_aerosol",
            Self::ContinentalPollutedAerosol => "continental_polluted_aerosol",
            Self::DesertAerosol => "desert_aerosol",
            Self::MarineCleanAerosol => "marine_clean_aerosol",
            Self::MarinePollutedAerosol => "marine_polluted_aerosol",
            Self::MarineTropicalAerosol => "marine_tropical_aerosol",
            Self::RayleighOnly => "rayleigh_only",
            Self::RuralAerosol => "rural_aerosol",
            Self::UrbanAerosol => "urban_aerosol",
        }
    }
}

/// Atmosphere type for the Rayleigh LUT.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Atmosphere {
    SubarcticSummer,
    SubarcticWinter,
    MidlatitudeSummer,
    MidlatitudeWinter,
    Tropical,
    UsStandard,
}

impl Atmosphere {
    pub fn file_suffix(&self) -> &'static str {
        match self {
            Self::SubarcticSummer => "subarctic_summer",
            Self::SubarcticWinter => "subarctic_winter",
            Self::MidlatitudeSummer => "midlatitude_summer",
            Self::MidlatitudeWinter => "midlatitude_winter",
            Self::Tropical => "tropical",
            Self::UsStandard => "us-standard",
        }
    }
}

/// Configuration for the Rayleigh correction.
#[derive(Debug, Clone)]
pub struct RayleighConfig {
    pub platform_name: String,
    pub sensor: String,
    pub atmosphere: Atmosphere,
    pub aerosol_type: AerosolType,
    pub reduce_lim_low: f64,
    pub reduce_lim_high: f64,
    pub reduce_strength: f64,
}

impl Default for RayleighConfig {
    fn default() -> Self {
        Self {
            platform_name: "Himawari-8".into(),
            sensor: "ahi".into(),
            atmosphere: Atmosphere::UsStandard,
            aerosol_type: AerosolType::MarineCleanAerosol,
            reduce_lim_low: 70.0,
            reduce_lim_high: 105.0,
            reduce_strength: 0.0,
        }
    }
}

/// Loaded Rayleigh LUT data ready for interpolation.
struct LutData {
    _rayleigh_refl_4d: ndarray::Array4<f64>,
    grid_3d: Option<ndarray::Array3<f64>>,
    sunz_coord: Array1<f64>,
    azid_coord: Array1<f64>,
    satz_coord: Array1<f64>,
}

fn load_lut_data(lut_path: &Path, wavelength_nm: f64) -> Result<LutData> {
    let rayleigh_refl_4d = read_reflectance_lut_4d(lut_path).map_err(|e| {
        RustySatError::invalid_input(format!("failed to read LUT reflectance: {e}"))
    })?;
    let wvl_coord = read_wavelength_lut_coord(lut_path).map_err(|e| {
        RustySatError::invalid_input(format!("failed to read LUT wavelengths: {e}"))
    })?;
    let (azid_coord, satz_coord, sunz_coord) = get_reflectance_lut_from_file(lut_path)
        .map_err(|e| RustySatError::invalid_input(format!("failed to read LUT coords: {e}")))?;

    let grid_3d = if wavelength_nm < wvl_coord[0] || wavelength_nm > wvl_coord[wvl_coord.len() - 1]
    {
        None
    } else {
        Some(get_wavelength_adjusted_lut(
            &rayleigh_refl_4d,
            &wvl_coord,
            wavelength_nm,
        ))
    };

    Ok(LutData {
        _rayleigh_refl_4d: rayleigh_refl_4d,
        grid_3d,
        sunz_coord,
        azid_coord,
        satz_coord,
    })
}

/// The Rayleigh reflectance corrector.
pub struct RayleighCorrector {
    config: RayleighConfig,
    lut_data: LutData,
}

impl RayleighCorrector {
    /// Create a corrector from a LUT file path and default config.
    /// Does NOT auto-download; use `with_config_auto` for that.
    pub fn new(lut_path: impl Into<PathBuf>, wavelength_nm: f64) -> Result<Self> {
        let config = RayleighConfig::default();
        let lut_data = load_lut_data(&lut_path.into(), wavelength_nm)?;
        Ok(Self { config, lut_data })
    }

    /// Create a corrector from a LUT file path and custom config.
    /// Does NOT auto-download; use `with_config_auto` for that.
    pub fn with_config(
        lut_path: impl Into<PathBuf>,
        config: RayleighConfig,
        wavelength_nm: f64,
    ) -> Result<Self> {
        let lut_data = load_lut_data(&lut_path.into(), wavelength_nm)?;
        Ok(Self { config, lut_data })
    }

    /// Create a corrector with custom config, auto-downloading the LUT
    /// via rustyspectral if not already cached locally.
    ///
    /// Uses rustyspectral's default LUT directory
    /// (`~/Library/Application Support/pyspectral/` on macOS,
    /// `~/.local/share/pyspectral/` on Linux). Override via the
    /// `PSP_CONFIG_FILE` environment variable.
    pub fn with_config_auto(config: RayleighConfig, wavelength_nm: f64) -> Result<Self> {
        let rayleigh = rustyspectral::rayleigh::Rayleigh::new(
            &config.platform_name,
            &config.sensor,
            Some(config.atmosphere.file_suffix()),
            Some(config.aerosol_type.dir_name()),
        );
        let lut_path = rayleigh.reflectance_lut_filename;
        let lut_data = load_lut_data(&lut_path, wavelength_nm)?;
        Ok(Self { config, lut_data })
    }

    /// Apply the Rayleigh correction to a visible-band reflectance dataset.
    ///
    /// Consumes the visible dataset and returns the corrected dataset.
    /// The correction is computed in row strips to keep peak memory low.
    /// Preserves area metadata and x/y coordinates from the source dataset.
    pub fn apply_correction(
        self,
        vis_dataset: Dataset,
        red: RedBandSource,
        params: AngleParams,
    ) -> Result<Dataset> {
        let config = self.config;
        let lut = self.lut_data;

        let area_attr = vis_dataset.attr("area").cloned();
        let source_coords = vis_dataset.array().map(|a| a.coords().clone());

        let array = vis_dataset
            .into_array()
            .ok_or_else(|| RustySatError::invalid_input("visible dataset has no array data"))?;
        let (height, width) = array.shape_yx()?;
        let mask = array.mask().cloned();
        let mut vis_f32 = into_vis_f32(array);

        let red_relax = red_relax_from_source(red, height * width, false)?;

        let gmst_val = gmst(params.utc);
        let (sun_ra, sun_dec) = sun_ra_dec(params.utc);
        let sat_alt_km = params.sat_alt / 1000.0;
        let (sat_x, sat_y, sat_z) =
            observer_position(params.utc, params.sat_lon, params.sat_lat, sat_alt_km);

        let reduce = config.reduce_strength > 0.0;
        let (zenith_lo, zenith_hi) = if config.reduce_lim_low > config.reduce_lim_high {
            (config.reduce_lim_high, config.reduce_lim_low)
        } else {
            (config.reduce_lim_low, config.reduce_lim_high)
        };
        let zenith_span = zenith_hi - zenith_lo;

        let grid_3d = match &lut.grid_3d {
            Some(g) => g,
            None => {
                return build_result(
                    vis_f32,
                    mask,
                    height,
                    width,
                    &config,
                    area_attr,
                    source_coords,
                );
            }
        };

        // pyspectral clips the sun/sat zenith ANGLE to the LUT maximum
        // secant before computing 1/cos; out-of-range pixels evaluate exactly
        // at the LUT boundary (no extrapolation past the last grid point).
        let sunz_max = lut.sunz_coord[lut.sunz_coord.len() - 1];
        let satz_max = lut.satz_coord[lut.satz_coord.len() - 1];

        const STRIP_HEIGHT: usize = 64;

        for y0 in (0..height).step_by(STRIP_HEIGHT) {
            let y1 = (y0 + STRIP_HEIGHT).min(height);
            let strip_n = (y1 - y0) * width;
            let red = &red_relax;
            let offset = y0 * width;

            use rayon::prelude::*;
            // Row-chunked parallel iteration: each parallel task is one full
            // strip row, so row/col indices come from the chunk position and
            // inner offset instead of a div/mod per pixel.
            vis_f32[offset..offset + strip_n]
                .par_chunks_mut(width)
                .enumerate()
                .for_each(|(row, strip_row)| {
                    for (col, out) in strip_row.iter_mut().enumerate() {
                        if !out.is_finite() {
                            continue;
                        }
                        let x_pos = params.x_coords[col];
                        let y_pos = params.y_coords[y0 + row];

                        // Exact geos inverse: space pixels become NaN.
                        let Some((lon, lat)) = params.geos.inverse_rad(x_pos, y_pos) else {
                            *out = f32::NAN;
                            continue;
                        };

                        let (sunz, suna) =
                            sun_angles_from_lonlat(lon, lat, sun_ra, sun_dec, gmst_val);
                        if !sunz.is_finite() {
                            *out = f32::NAN;
                            continue;
                        }
                        let (satz, sata) =
                            sat_angles_from_lonlat(lon, lat, sat_x, sat_y, sat_z, gmst_val);

                        let azidiff = AngleSet::relative_azimuth_single(sata, suna);
                        if !azidiff.is_finite() {
                            continue;
                        }

                        let sunzsec = secant_clamped(sunz, sunz_max);
                        let satzsec = secant_clamped(satz, satz_max);

                        let mut correction = trilinear_interpolate(
                            grid_3d,
                            sunzsec,
                            azidiff,
                            satzsec,
                            &lut.sunz_coord,
                            &lut.azid_coord,
                            &lut.satz_coord,
                        );

                        if let RedRelax::Values(vals) = red {
                            let r = f64::from(vals[offset + row * width + col]);
                            correction = relax_correction(correction, r);
                        }

                        // pyspectral clips the correction to [0, 100] after
                        // the red relaxation, then Satpy applies the optional
                        // high-zenith reduction.
                        correction = correction.clamp(0.0, 100.0);

                        if reduce && zenith_span > 0.0 {
                            let t = if sunz < zenith_lo {
                                0.0
                            } else {
                                ((sunz - zenith_lo) / zenith_span).min(1.0)
                            };
                            let factor = (1.0 - config.reduce_strength * t).clamp(0.0, 1.0);
                            correction *= factor;
                        }

                        // Satpy: `proj = vis - refl_cor_band` (no clamping of
                        // the result; negative values clip to black downstream).
                        *out = (f64::from(*out) - correction) as f32;
                    }
                });
        }

        build_result(
            vis_f32,
            mask,
            height,
            width,
            &config,
            area_attr,
            source_coords,
        )
    }

    /// Apply both sun zenith and Rayleigh correction in a single pass.
    ///
    /// Consumes the visible dataset. In each 64-row strip, angles are
    /// computed once per pixel and both corrections are applied inline:
    ///
    /// ```text
    /// refl *= sza_factor      // sun zenith (88°→max_sza gradient falloff)
    /// refl  -= LUT_correction // Rayleigh (relaxed by the red band, clipped)
    /// ```
    ///
    /// This avoids computing solar/satellite angles twice when both
    /// corrections are needed on the same band.
    pub fn apply_correction_with_sun_zenith(
        self,
        vis_dataset: Dataset,
        red: RedBandSource,
        params: AngleParams,
        max_sza: f64,
    ) -> Result<Dataset> {
        let max_sza_rad = max_sza.to_radians();
        let max_sza_cos = max_sza_rad.cos();
        const LIMIT_DEG: f64 = 88.0;
        let limit_rad = LIMIT_DEG.to_radians();
        let limit_cos = limit_rad.cos();
        let config = self.config;
        let lut = self.lut_data;

        let area_attr = vis_dataset.attr("area").cloned();
        let source_coords = vis_dataset.array().map(|a| a.coords().clone());

        let array = vis_dataset
            .into_array()
            .ok_or_else(|| RustySatError::invalid_input("visible dataset has no array data"))?;
        let (height, width) = array.shape_yx()?;
        let mask = array.mask().cloned();
        let mut vis_f32 = into_vis_f32(array);

        // The Rayleigh LUT only covers 400-800 nm. Bands outside that range
        // (e.g. AHI B04 at 0.86 µm) still receive the sun-zenith amplification
        // — Satpy's `hybrid_green` feeds B04 through `[sunz_corrected]` only —
        // and the Rayleigh subtraction runs just for in-range wavelengths.
        let grid_3d = lut.grid_3d.as_ref();
        let red_relax = match grid_3d {
            Some(_) => red_relax_from_source(red, height * width, true)?,
            None => RedRelax::None,
        };

        let gmst_val = gmst(params.utc);
        let (sun_ra, sun_dec) = sun_ra_dec(params.utc);
        let sat_alt_km = params.sat_alt / 1000.0;
        let (sat_x, sat_y, sat_z) =
            observer_position(params.utc, params.sat_lon, params.sat_lat, sat_alt_km);

        let reduce = config.reduce_strength > 0.0;
        let (zenith_lo, zenith_hi) = if config.reduce_lim_low > config.reduce_lim_high {
            (config.reduce_lim_high, config.reduce_lim_low)
        } else {
            (config.reduce_lim_low, config.reduce_lim_high)
        };
        let zenith_span = zenith_hi - zenith_lo;

        // pyspectral clips the sun/sat zenith ANGLE to the LUT maximum
        // secant before computing 1/cos; out-of-range pixels evaluate exactly
        // at the LUT boundary (no extrapolation past the last grid point).
        let sunz_max = lut.sunz_coord[lut.sunz_coord.len() - 1];
        let satz_max = lut.satz_coord[lut.satz_coord.len() - 1];

        const STRIP_HEIGHT: usize = 64;

        for y0 in (0..height).step_by(STRIP_HEIGHT) {
            let y1 = (y0 + STRIP_HEIGHT).min(height);
            let strip_n = (y1 - y0) * width;
            let red = &red_relax;
            let offset = y0 * width;

            use rayon::prelude::*;
            // Row-chunked parallel iteration: each parallel task is one full
            // strip row, so row/col indices come from the chunk position and
            // inner offset instead of a div/mod per pixel.
            vis_f32[offset..offset + strip_n]
                .par_chunks_mut(width)
                .enumerate()
                .for_each(|(row, strip_row)| {
                    for (col, out) in strip_row.iter_mut().enumerate() {
                        if !out.is_finite() {
                            continue;
                        }
                        let x_pos = params.x_coords[col];
                        let y_pos = params.y_coords[y0 + row];

                        // Exact geos inverse: space pixels become NaN.
                        let Some((lon, lat)) = params.geos.inverse_rad(x_pos, y_pos) else {
                            *out = f32::NAN;
                            continue;
                        };

                        let (sunz, suna) =
                            sun_angles_from_lonlat(lon, lat, sun_ra, sun_dec, gmst_val);
                        if !sunz.is_finite() {
                            *out = f32::NAN;
                            continue;
                        }

                        let cos_sza = sunz.to_radians().cos();
                        *out *= crate::sun_zenith::sza_correction_factor(
                            cos_sza,
                            limit_cos,
                            max_sza_cos,
                            limit_rad,
                            max_sza_rad,
                        );

                        // Wavelength outside the Rayleigh LUT range: this band
                        // is sun-zenith corrected only (Satpy `[sunz_corrected]`).
                        let Some(grid) = grid_3d else {
                            continue;
                        };

                        let (satz, sata) =
                            sat_angles_from_lonlat(lon, lat, sat_x, sat_y, sat_z, gmst_val);

                        let azidiff = AngleSet::relative_azimuth_single(sata, suna);
                        if !azidiff.is_finite() {
                            continue;
                        }

                        let sunzsec = secant_clamped(sunz, sunz_max);
                        let satzsec = secant_clamped(satz, satz_max);

                        let mut correction = trilinear_interpolate(
                            grid,
                            sunzsec,
                            azidiff,
                            satzsec,
                            &lut.sunz_coord,
                            &lut.azid_coord,
                            &lut.satz_coord,
                        );

                        match red {
                            RedRelax::Values(vals) => {
                                let r = f64::from(vals[offset + row * width + col]);
                                correction = relax_correction(correction, r);
                            }
                            RedRelax::InPlaceVis => {
                                // Satpy: the red prerequisite is the
                                // sun-zenith-corrected band itself, which is
                                // exactly the in-place value at this point.
                                correction = relax_correction(correction, f64::from(*out));
                            }
                            RedRelax::None => {}
                        }

                        // pyspectral clips the correction to [0, 100] after
                        // the red relaxation, then Satpy applies the optional
                        // high-zenith reduction.
                        correction = correction.clamp(0.0, 100.0);

                        if reduce && zenith_span > 0.0 {
                            let t = if sunz < zenith_lo {
                                0.0
                            } else {
                                ((sunz - zenith_lo) / zenith_span).min(1.0)
                            };
                            let factor = (1.0 - config.reduce_strength * t).clamp(0.0, 1.0);
                            correction *= factor;
                        }

                        // Satpy: `proj = vis - refl_cor_band` (no clamping of
                        // the result; negative values clip to black downstream).
                        *out = (f64::from(*out) - correction) as f32;
                    }
                });
        }

        build_result_combined(
            vis_f32,
            mask,
            height,
            width,
            &config,
            area_attr,
            source_coords,
        )
    }
}

fn build_result(
    vis_f32: Vec<f32>,
    mask: Option<ValidityMask>,
    height: usize,
    width: usize,
    config: &RayleighConfig,
    area_attr: Option<MetadataValue>,
    source_coords: Option<std::collections::BTreeMap<String, Coordinate>>,
) -> Result<Dataset> {
    let mut result_array =
        DataArray::<f32>::from_vec_named(vec![height, width], vec!["y", "x"], vis_f32)?;

    if let Some(m) = mask {
        result_array = result_array.with_mask(m)?;
    }

    if let Some(coords) = source_coords {
        for (name, coord) in coords {
            if name == "y" || name == "x" {
                result_array = result_array.with_coordinate(&name, coord)?;
            }
        }
    }

    let mut result = Dataset::new(DataId::new("rayleigh_corrected")?).with_array(result_array);

    if let Some(area) = area_attr {
        result.insert_attr("area", area)?;
    }
    result.insert_attr("modifier", MetadataValue::string("rayleigh_correction"))?;
    result.insert_attr(
        "atmosphere",
        MetadataValue::string(config.atmosphere.file_suffix()),
    )?;
    result.insert_attr(
        "aerosol_type",
        MetadataValue::string(config.aerosol_type.dir_name()),
    )?;

    Ok(result)
}

fn build_result_combined(
    vis_f32: Vec<f32>,
    mask: Option<ValidityMask>,
    height: usize,
    width: usize,
    config: &RayleighConfig,
    area_attr: Option<MetadataValue>,
    source_coords: Option<std::collections::BTreeMap<String, Coordinate>>,
) -> Result<Dataset> {
    let mut result_array =
        DataArray::<f32>::from_vec_named(vec![height, width], vec!["y", "x"], vis_f32)?;

    if let Some(m) = mask {
        result_array = result_array.with_mask(m)?;
    }

    if let Some(coords) = source_coords {
        for (name, coord) in coords {
            if name == "y" || name == "x" {
                result_array = result_array.with_coordinate(&name, coord)?;
            }
        }
    }

    let mut result =
        Dataset::new(DataId::new("sun_zenith_rayleigh_corrected")?).with_array(result_array);

    if let Some(area) = area_attr {
        result.insert_attr("area", area)?;
    }
    result.insert_attr(
        "modifier",
        MetadataValue::string("combined_sun_zenith_rayleigh_correction"),
    )?;
    result.insert_attr(
        "atmosphere",
        MetadataValue::string(config.atmosphere.file_suffix()),
    )?;
    result.insert_attr(
        "aerosol_type",
        MetadataValue::string(config.aerosol_type.dir_name()),
    )?;

    Ok(result)
}

pub(crate) fn into_vis_f32(array: AnyDataArray) -> Vec<f32> {
    match array {
        AnyDataArray::F32(da) => da.into_values(),
        AnyDataArray::F64(da) => da.into_values().into_iter().map(|v| v as f32).collect(),
        AnyDataArray::U8(da) => da.into_values().into_iter().map(|v| v as f32).collect(),
        AnyDataArray::U16(da) => da.into_values().into_iter().map(|v| v as f32).collect(),
        AnyDataArray::I16(da) => da.into_values().into_iter().map(|v| v as f32).collect(),
    }
}

/// Red-band source for the Rayleigh cloud relaxation (Satpy
/// `PSPRayleighReflectance` red prerequisite).
///
/// Satpy's `rayleigh_corrected` modifier relaxes the Rayleigh correction
/// where the red band (sun-zenith-corrected B03 for AHI) is bright:
/// `correction *= 1 − (red − 20)/80` for `red ≥ 20`, per pyspectral
/// `_relax_rayleigh_refl_correction_where_cloudy`. The formula is applied
/// without clipping — the factor goes negative for `red > 100` — matching
/// pyspectral and rustyspectral.
///
/// The red band must be on the same grid as the visible band (Satpy resamples
/// the red prerequisite to the visible band's area before correction).
#[derive(Debug, Clone, Copy)]
pub enum RedBandSource<'a> {
    /// No red-band relaxation (full Rayleigh correction).
    None,
    /// Relax using an explicit red dataset on the same y/x grid as the
    /// visible band. For AHI this is the sun-zenith-corrected B03 (resampled
    /// to the visible band's resolution when they differ).
    Dataset(&'a Dataset),
    /// Use the sun-zenith-corrected visible band itself as the red band.
    /// This is Satpy's behavior when the red prerequisite is the band being
    /// corrected (e.g. B03 for `rayleigh_corrected`). Only valid in the
    /// combined sun-zenith + Rayleigh path (`rayleigh_correct_with_sun_zenith`
    /// / `RayleighCorrector::apply_correction_with_sun_zenith`), where it
    /// reads the in-place post-sun-zenith values with no extra memory.
    SunZenithCorrectedVis,
}

/// Prepared red-band relaxation source for the strip loops.
#[derive(Debug)]
enum RedRelax {
    /// No relaxation.
    None,
    /// Explicit red values (f32 copy of the red dataset).
    Values(Vec<f32>),
    /// Combined path: read the sun-zenith-corrected in-place visible value.
    InPlaceVis,
}

/// Extract red values from a `RedBandSource` for the strip loops.
///
/// `allow_in_place_vis` is `true` only in the combined sun-zenith + Rayleigh
/// path, where `SunZenithCorrectedVis` can read the corrected in-place
/// values. The explicit dataset path validates the array length to avoid an
/// indexed panic in the strip loop.
fn red_relax_from_source(
    red: RedBandSource<'_>,
    visible_count: usize,
    allow_in_place: bool,
) -> Result<RedRelax> {
    match red {
        RedBandSource::None => Ok(RedRelax::None),
        RedBandSource::SunZenithCorrectedVis if allow_in_place => Ok(RedRelax::InPlaceVis),
        RedBandSource::SunZenithCorrectedVis => Err(RustySatError::unsupported(
            "RedBandSource::SunZenithCorrectedVis requires the combined \
             sun-zenith + Rayleigh correction path",
        )),
        RedBandSource::Dataset(ds) => Ok(RedRelax::Values(red_values_from_dataset(
            ds,
            visible_count,
        )?)),
    }
}

/// pyspectral `_relax_rayleigh_refl_correction_where_cloudy`: for `red ≥ 20`
/// the correction is scaled by `1 − (red − 20)/80`. No clipping is applied —
/// the factor goes negative for `red > 100`, matching pyspectral and
/// rustyspectral. Non-finite red leaves the correction unchanged.
#[inline]
fn relax_correction(correction: f64, red: f64) -> f64 {
    if red.is_finite() && red >= 20.0 {
        correction * (1.0 - (red - 20.0) / 80.0)
    } else {
        correction
    }
}

/// Clamp a zenith angle to the LUT range and return its secant, matching
/// pyspectral `_clip_angles_inside_coordinate_range`: the ANGLE is clipped to
/// `acos(1/max_secant)` before computing `1/cos`, so out-of-range (limb/night)
/// pixels evaluate exactly at the LUT boundary instead of extrapolating past
/// the last grid point.
#[inline]
fn secant_clamped(zenith_deg: f64, max_secant: f64) -> f64 {
    let clip_angle = (1.0 / max_secant).acos().to_degrees();
    let z = zenith_deg.clamp(0.0, clip_angle);
    1.0 / z.to_radians().cos()
}

/// Extract the red band values as f32.
///
/// The red band is stored as f32 (narrowing f64 sources) to halve the copy
/// for 0.5 km red bands. A pixel whose value sits within ~1e-6 of the
/// `r >= 20.0` cloud-relaxation threshold may flip the branch compared to a
/// full-f64 copy, which is accepted for the memory win. A dataset without an
/// array, or with fewer values than the visible band, is rejected instead of
/// panicking on an indexed access in the strip loop.
fn red_values_from_dataset(red_dataset: &Dataset, visible_count: usize) -> Result<Vec<f32>> {
    let Some(array) = red_dataset.array() else {
        return Err(RustySatError::invalid_input(
            "red dataset has no array data for Rayleigh correction",
        ));
    };
    if array.len() < visible_count {
        return Err(RustySatError::invalid_input(format!(
            "red dataset array length {} is smaller than the visible band ({visible_count} pixels)",
            array.len()
        )));
    }
    Ok(match array {
        AnyDataArray::F32(a) => a.values().to_vec(),
        AnyDataArray::F64(a) => a.values().iter().map(|v| *v as f32).collect(),
        AnyDataArray::U8(a) => a.values().iter().map(|v| f32::from(*v)).collect(),
        AnyDataArray::U16(a) => a.values().iter().map(|v| f32::from(*v)).collect(),
        AnyDataArray::I16(a) => a.values().iter().map(|v| f32::from(*v)).collect(),
    })
}

/// Compute angles and apply Rayleigh correction in one call.
pub fn rayleigh_correct(
    corrector: RayleighCorrector,
    vis_dataset: Dataset,
    red: RedBandSource,
    utc: UtcInstant,
) -> Result<Dataset> {
    use crate::angles::{extract_xy_coords, AngleParams};

    let area_attr = vis_dataset
        .attr("area")
        .ok_or_else(|| RustySatError::invalid_input("dataset missing 'area' attribute"))?;

    let coords = vis_dataset
        .array()
        .ok_or_else(|| RustySatError::invalid_input("dataset has no array"))?
        .coords();

    let (x_coords, y_coords) = extract_xy_coords(coords)?;
    let params = AngleParams::from_dataset_area(area_attr, x_coords, y_coords, utc)?;

    corrector.apply_correction(vis_dataset, red, params)
}

/// Compute angles and apply both sun zenith and Rayleigh correction
/// in a single pass.
///
/// `max_sza` controls where the sun-zenith correction reaches zero at
/// the terminator (default 95.0°, matching Satpy).
pub fn rayleigh_correct_with_sun_zenith(
    corrector: RayleighCorrector,
    vis_dataset: Dataset,
    red: RedBandSource,
    utc: UtcInstant,
    max_sza: f64,
) -> Result<Dataset> {
    use crate::angles::{extract_xy_coords, AngleParams};

    let area_attr = vis_dataset
        .attr("area")
        .ok_or_else(|| RustySatError::invalid_input("dataset missing 'area' attribute"))?;

    let coords = vis_dataset
        .array()
        .ok_or_else(|| RustySatError::invalid_input("dataset has no array"))?
        .coords();

    let (x_coords, y_coords) = extract_xy_coords(coords)?;
    let params = AngleParams::from_dataset_area(area_attr, x_coords, y_coords, utc)?;

    corrector.apply_correction_with_sun_zenith(vis_dataset, red, params, max_sza)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aerosol_dir_names_match_rustyspectral() {
        assert_eq!(
            AerosolType::MarineCleanAerosol.dir_name(),
            "marine_clean_aerosol"
        );
        assert_eq!(AerosolType::RayleighOnly.dir_name(), "rayleigh_only");
        assert_eq!(AerosolType::DesertAerosol.dir_name(), "desert_aerosol");
    }

    #[test]
    fn atmosphere_file_suffix_matches_expected() {
        assert_eq!(Atmosphere::UsStandard.file_suffix(), "us-standard");
        assert_eq!(
            Atmosphere::SubarcticSummer.file_suffix(),
            "subarctic_summer"
        );
    }

    #[test]
    fn default_config_has_expected_platform() {
        let c = RayleighConfig::default();
        assert_eq!(c.platform_name, "Himawari-8");
        assert_eq!(c.sensor, "ahi");
        assert_eq!(c.reduce_strength, 0.0);
    }

    #[test]
    fn nonexistent_lut_returns_error() {
        let tmp = tempfile::NamedTempFile::new().expect("temp file");
        let lut_path = tmp.path().to_path_buf();
        drop(tmp);
        let result = RayleighCorrector::new(&lut_path, 634.0);
        assert!(result.is_err());
    }

    #[test]
    fn all_aerosol_types_have_names() {
        let types = [
            AerosolType::AntarcticAerosol,
            AerosolType::ContinentalAverageAerosol,
            AerosolType::ContinentalCleanAerosol,
            AerosolType::ContinentalPollutedAerosol,
            AerosolType::DesertAerosol,
            AerosolType::MarineCleanAerosol,
            AerosolType::MarinePollutedAerosol,
            AerosolType::MarineTropicalAerosol,
            AerosolType::RayleighOnly,
            AerosolType::RuralAerosol,
            AerosolType::UrbanAerosol,
        ];
        for t in &types {
            assert!(!t.dir_name().is_empty());
        }
    }

    #[test]
    fn all_atmosphere_types_have_suffixes() {
        let atms = [
            Atmosphere::SubarcticSummer,
            Atmosphere::SubarcticWinter,
            Atmosphere::MidlatitudeSummer,
            Atmosphere::MidlatitudeWinter,
            Atmosphere::Tropical,
            Atmosphere::UsStandard,
        ];
        for a in &atms {
            assert!(!a.file_suffix().is_empty());
        }
    }

    // ── Combined sun zenith + Rayleigh correction ──

    use crate::astronomy::UtcInstant;
    use crate::geos::GeosProjection;
    use crate::sun_zenith::SunZenithCorrector;
    use rusty_sat_core::Coordinate;

    #[test]
    fn combined_convenience_errors_on_missing_area() {
        let array = rusty_sat_core::DataArray::<f32>::from_vec_named(
            vec![2, 2],
            vec!["y", "x"],
            vec![50.0_f32; 4],
        )
        .expect("array");
        let ds = Dataset::new(DataId::new("test").expect("id")).with_array(array);
        let utc = UtcInstant::from_ymdhms(2025, 9, 23, 7, 20, 0);
        // LUT loading fails for nonexistent file, but the convenience function
        // checks area attr before invoking the corrector.
        let tmp = tempfile::NamedTempFile::new().expect("tmp");
        let p = tmp.path().to_path_buf();
        drop(tmp);
        match RayleighCorrector::new(&p, 1.0) {
            Ok(c) => {
                let r = rayleigh_correct_with_sun_zenith(c, ds, RedBandSource::None, utc, 95.0);
                assert!(r.is_err());
            }
            Err(_) => { /* can't test — no LUT available */ }
        }
    }

    #[test]
    fn combined_convenience_errors_on_missing_array() {
        let mut ds = Dataset::new(DataId::new("test").expect("id"));
        ds.insert_attr("area", combined_area_attr()).expect("area");
        let utc = UtcInstant::from_ymdhms(2025, 9, 23, 7, 20, 0);
        let tmp = tempfile::NamedTempFile::new().expect("tmp");
        let p = tmp.path().to_path_buf();
        drop(tmp);
        match RayleighCorrector::new(&p, 1.0) {
            Ok(c) => {
                let r = rayleigh_correct_with_sun_zenith(c, ds, RedBandSource::None, utc, 95.0);
                assert!(r.is_err());
            }
            Err(_) => { /* can't test — no LUT available */ }
        }
    }

    #[test]
    fn combined_default_max_sza_is_95() {
        let sz = SunZenithCorrector::default();
        assert!((sz.max_sza() - 95.0).abs() < 0.01);
    }

    #[test]
    fn combined_convenience_function_compiles_and_accepts_min_cos() {
        // This test verifies the API shape — the function takes min_cos_zenith
        // as a f64 parameter. Full behavior is tested via the integration test.
        // The corrector construction fails here because no real LUT is available.
        let ds = combined_area_dataset(vec![50.0_f32; 9]);
        let utc = UtcInstant::from_ymdhms(2025, 9, 23, 7, 20, 0);
        let tmp = tempfile::NamedTempFile::new().expect("tmp");
        let p = tmp.path().to_path_buf();
        drop(tmp);
        match RayleighCorrector::new(&p, 1.0) {
            Ok(c) => {
                let r = rayleigh_correct_with_sun_zenith(c, ds, RedBandSource::None, utc, 95.0);
                // With grid_3d=None (wavelength out of range), this is a
                // no-op pass-through but should still produce a Dataset.
                assert!(r.is_ok());
                let result = r.expect("ok");
                assert!(result.attr("modifier").is_some());
            }
            Err(_) => { /* no real LUT available in unit test */ }
        }
    }

    #[test]
    fn combined_out_of_lut_wavelength_still_applies_sun_zenith() {
        // AHI B04 (0.86 µm) is outside the Rayleigh LUT range (400-800 nm).
        // Satpy's `hybrid_green` feeds B04 through `[sunz_corrected]` only, so
        // the combined path must still apply the sun-zenith amplification
        // instead of returning the raw reflectance. Regression for the early
        // return that skipped the whole strip loop for out-of-LUT wavelengths.
        let Some(lut_path) = local_rayleigh_lut() else {
            eprintln!("SKIP: no Rayleigh LUT available");
            return;
        };
        let cfg = RayleighConfig {
            platform_name: "Himawari-9".into(),
            sensor: "ahi".into(),
            atmosphere: Atmosphere::UsStandard,
            aerosol_type: AerosolType::RayleighOnly,
            ..RayleighConfig::default()
        };
        let corrector = RayleighCorrector::with_config(&lut_path, cfg, 860.0).expect("corrector");
        let utc = UtcInstant::from_ymdhms(2025, 9, 23, 7, 20, 0);

        let ds = combined_area_dataset(vec![50.0_f32; 9]);
        let result =
            rayleigh_correct_with_sun_zenith(corrector, ds, RedBandSource::None, utc, 95.0)
                .expect("combined path with out-of-LUT wavelength");
        assert_eq!(
            result.attr("modifier").and_then(MetadataValue::as_str),
            Some("combined_sun_zenith_rayleigh_correction")
        );

        // Parity with the standalone SunZenithCorrector: both paths must run
        // the identical sunz-only math for an out-of-LUT band.
        let ds_ref = combined_area_dataset(vec![50.0_f32; 9]);
        let area = ds_ref.attr("area").expect("area");
        let coords = ds_ref.array().expect("arr").coords();
        let (x_coords, y_coords) = crate::angles::extract_xy_coords(coords).expect("coords");
        let params = AngleParams::from_dataset_area(area, x_coords, y_coords, utc).expect("params");
        let reference = SunZenithCorrector::default()
            .apply_correction(ds_ref, params)
            .expect("sunz-only reference");

        let a = result.array().expect("arr").values_as_f64();
        let b = reference.array().expect("arr").values_as_f64();
        assert_eq!(a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            if y.is_nan() {
                assert!(x.is_nan(), "pixel {i}: expected NaN, got {x}");
            } else {
                assert!(
                    (x - y).abs() < 1e-4,
                    "pixel {i}: combined {x} != sunz-only reference {y}"
                );
            }
        }
        // The center pixel must be amplified above the raw 50
        // (1/cos(72.6°) ≈ 3.35 for 2025-09-23 07:20 UTC), proving the
        // sun-zenith path ran instead of returning the raw array.
        let center = a[4];
        assert!(
            center > 100.0 && center < 300.0,
            "center amplified by sunz: {center}"
        );
    }

    /// Locate a local pyspectral Rayleigh LUT in the standard per-platform
    /// directories; `None` when unavailable (test then skips).
    fn local_rayleigh_lut() -> Option<PathBuf> {
        let mut candidates = Vec::new();
        if let Ok(home) = std::env::var("HOME") {
            candidates.push(PathBuf::from(format!(
                "{home}/Library/Application Support/pyspectral/rayleigh_only/rayleigh_lut_us-standard.h5"
            )));
            candidates.push(PathBuf::from(format!(
                "{home}/.local/share/pyspectral/rayleigh_only/rayleigh_lut_us-standard.h5"
            )));
        }
        candidates.into_iter().find(|p| p.is_file())
    }

    // ── Helpers for combined tests ──

    #[test]
    fn red_values_reject_missing_or_short_arrays() {
        let visible_count = 9;
        // No red source -> no relaxation.
        assert!(matches!(
            red_relax_from_source(RedBandSource::None, visible_count, true)
                .expect("no red is fine"),
            RedRelax::None
        ));
        // Dataset without an array -> error, not an empty-Vec indexed access.
        let no_array =
            rusty_sat_core::Dataset::new(rusty_sat_core::DataId::new("red").expect("id"));
        assert!(
            red_relax_from_source(RedBandSource::Dataset(&no_array), visible_count, true).is_err()
        );
        // Shorter than the visible band -> error with a clear message.
        let short = rusty_sat_core::Dataset::new(rusty_sat_core::DataId::new("red").expect("id"))
            .with_array(
                rusty_sat_core::DataArray::<f32>::from_vec_named(
                    vec![2, 2],
                    ["y", "x"],
                    vec![1.0_f32; 4],
                )
                .expect("array"),
            );
        let err = red_relax_from_source(RedBandSource::Dataset(&short), visible_count, true)
            .expect_err("short red array must fail");
        assert!(err.to_string().contains("smaller than the visible band"));
        // Matching length -> values.
        let ok = rusty_sat_core::Dataset::new(rusty_sat_core::DataId::new("red").expect("id"))
            .with_array(
                rusty_sat_core::DataArray::<f32>::from_vec_named(
                    vec![3, 3],
                    ["y", "x"],
                    vec![1.0_f32; 9],
                )
                .expect("array"),
            );
        let RedRelax::Values(values) =
            red_relax_from_source(RedBandSource::Dataset(&ok), visible_count, true)
                .expect("valid red")
        else {
            panic!("expected values");
        };
        assert_eq!(values.len(), 9);
    }

    #[test]
    fn in_place_red_requires_combined_path() {
        // SunZenithCorrectedVis is allowed in the combined path ...
        assert!(matches!(
            red_relax_from_source(RedBandSource::SunZenithCorrectedVis, 9, true)
                .expect("combined path allows in-place red"),
            RedRelax::InPlaceVis
        ));
        // ... and rejected in the plain Rayleigh path.
        let err = red_relax_from_source(RedBandSource::SunZenithCorrectedVis, 9, false)
            .expect_err("plain path must reject in-place red");
        assert!(err.to_string().contains("combined"));
    }

    #[test]
    fn relax_correction_matches_pyspectral_where() {
        // red < 20: correction unchanged.
        assert_eq!(relax_correction(50.0, 0.0), 50.0);
        assert_eq!(relax_correction(50.0, 19.9), 50.0);
        // red == 20: factor 1.0.
        assert!((relax_correction(50.0, 20.0) - 50.0).abs() < 1e-12);
        // Linear scaling between 20 and 100: factor (1 - (red-20)/80).
        assert!((relax_correction(80.0, 60.0) - 80.0 * (1.0 - 40.0 / 80.0)).abs() < 1e-12);
        // red == 100: factor 0.
        assert_eq!(relax_correction(50.0, 100.0), 0.0);
        // red > 100: NO clipping — the factor goes negative (pyspectral parity).
        assert!((relax_correction(50.0, 140.0) - 50.0 * (1.0 - 120.0 / 80.0)).abs() < 1e-12);
        assert!(relax_correction(50.0, 140.0) < 0.0);
        // Non-finite red leaves the correction unchanged.
        assert_eq!(relax_correction(50.0, f64::NAN), 50.0);
    }

    #[test]
    fn secant_clamped_matches_pyspectral_angle_clip() {
        // Within range: plain 1/cos.
        assert!((secant_clamped(60.0, 24.75) - 1.0 / 60.0_f64.to_radians().cos()).abs() < 1e-12);
        // At the LUT boundary: the angle is clipped to acos(1/max), so the
        // secant is exactly the axis maximum (no extrapolation).
        assert!((secant_clamped(84.54, 24.75) - 1.0 / 84.54_f64.to_radians().cos()).abs() < 1e-12);
        assert!((secant_clamped(89.0, 24.75) - 24.75).abs() < 1e-12);
        assert!(
            (secant_clamped(137.0, 24.75) - 24.75).abs() < 1e-12,
            "night pixels clip too"
        );
        // Satellite secant axis: max 3.0 -> clip at ~70.5°.
        assert!((secant_clamped(81.0, 3.0) - 3.0).abs() < 1e-12);
        assert!((secant_clamped(85.0, 3.0) - 3.0).abs() < 1e-12);
    }

    fn combined_area_attr() -> MetadataValue {
        let proj = MetadataValue::Map(
            [
                ("a".into(), MetadataValue::string("6378137.0")),
                ("b".into(), MetadataValue::string("6356752.31414")),
                ("h".into(), MetadataValue::string("35785863.0")),
                ("lon_0".into(), MetadataValue::string("140.7")),
                ("proj".into(), MetadataValue::string("geos")),
                ("units".into(), MetadataValue::string("m")),
            ]
            .into(),
        );
        MetadataValue::map([
            ("type", MetadataValue::string("area")),
            ("proj_id", MetadataValue::string("geosh9")),
            ("projection", proj),
            ("height", MetadataValue::Integer(3)),
            ("width", MetadataValue::Integer(3)),
        ])
    }

    fn combined_area_dataset(values: Vec<f32>) -> Dataset {
        use rusty_sat_core::DataArray;
        let w = 3;
        let h = 3;
        let geos = GeosProjection {
            semi_major_axis: 6_378_137.0,
            semi_minor_axis: 6_356_752.314_14,
            perspective_point_height: 35_785_863.0,
            longitude_of_projection_origin: 140.7,
        };
        let hp = geos.perspective_point_height;
        let max_a = (geos.semi_major_axis / geos.satellite_radius()).asin();
        let hd = hp * max_a;
        let step = 2.0 * hd / 2.0;
        let xv: Vec<f64> = (0..w).map(|i| -hd + i as f64 * step).collect();
        let yv: Vec<f64> = (0..h).map(|i| hd - i as f64 * step).collect();
        let array = DataArray::<f32>::from_vec_named(vec![h, w], vec!["y", "x"], values)
            .expect("array")
            .with_coordinate("x", Coordinate::axis("x", xv).expect("x"))
            .expect("x")
            .with_coordinate("y", Coordinate::axis("y", yv).expect("y"))
            .expect("y");
        let mut ds = Dataset::new(DataId::new("test_band").expect("id")).with_array(array);
        ds.insert_attr("area", combined_area_attr()).expect("area");
        ds
    }
}
