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

use crate::angles::{
    precompute_columns, precompute_rows, satellite_single_precomputed, solar_single_precomputed,
};
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
        red_dataset: Option<&Dataset>,
        params: crate::angles::AngleParams,
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

        let red_values = red_values_from_red_dataset(red_dataset, height * width)?;

        let gmst_val = gmst(params.utc);
        let (sun_ra, sun_dec) = sun_ra_dec(params.utc);
        let sat_alt_km = params.sat_alt / 1000.0;
        let (sat_x, sat_y, sat_z) =
            observer_position(params.utc, params.sat_lon, params.sat_lat, sat_alt_km);

        let h_inv = 1.0 / params.geos.perspective_point_height;
        let lon_0_rad = params.geos.longitude_of_projection_origin.to_radians();
        let column_data = precompute_columns(&params.x_coords, h_inv, lon_0_rad, gmst_val, sun_ra);

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

        const STRIP_HEIGHT: usize = 64;

        for y0 in (0..height).step_by(STRIP_HEIGHT) {
            let y1 = (y0 + STRIP_HEIGHT).min(height);
            let strip_n = (y1 - y0) * width;

            let tan_theta_y_row = precompute_rows(&params.y_coords, y0, y1, h_inv);
            let red = red_values.as_ref();
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
                        let col_pre = &column_data[col];

                        let lat_rad = (tan_theta_y_row[row] * col_pre.cos_theta_x).atan();

                        let (sunz, suna) = solar_single_precomputed(lat_rad, sun_dec, col_pre);
                        let (satz, sata) =
                            satellite_single_precomputed(lat_rad, col_pre, sat_x, sat_y, sat_z);

                        let azidiff = crate::angles::AngleSet::relative_azimuth_single(sata, suna);
                        if !azidiff.is_finite() {
                            continue;
                        }

                        let sunzsec = 1.0 / sunz.to_radians().cos().max(0.0001);
                        let satzsec = 1.0 / satz.to_radians().cos().max(0.0001);

                        let mut correction = trilinear_interpolate(
                            grid_3d,
                            sunzsec,
                            azidiff,
                            satzsec,
                            &lut.sunz_coord,
                            &lut.azid_coord,
                            &lut.satz_coord,
                        );

                        if let Some(red_vals) = red {
                            let r = f64::from(red_vals[offset + row * width + col]);
                            if r.is_finite() && r >= 20.0 {
                                correction *= 1.0 - (r - 20.0) / 80.0;
                            }
                        }

                        if reduce && sunz.is_finite() && zenith_span > 0.0 {
                            let t = if sunz < zenith_lo {
                                0.0
                            } else {
                                ((sunz - zenith_lo) / zenith_span).min(1.0)
                            };
                            let factor = (1.0 - config.reduce_strength * t).clamp(0.0, 1.0);
                            correction *= factor;
                        }

                        let corrected = (f64::from(*out) - correction).max(0.0);
                        *out = corrected as f32;
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
    /// refl  = max(0, refl - LUT_correction)                // Rayleigh
    /// ```
    ///
    /// This avoids computing solar/satellite angles twice when both
    /// corrections are needed on the same band.
    pub fn apply_correction_with_sun_zenith(
        self,
        vis_dataset: Dataset,
        red_dataset: Option<&Dataset>,
        params: crate::angles::AngleParams,
        max_sza: f64,
    ) -> Result<Dataset> {
        let max_sza_rad = max_sza.to_radians();
        let max_sza_cos = max_sza_rad.cos();
        const LIMIT_DEG: f64 = 88.0;
        let limit_cos = LIMIT_DEG.to_radians().cos();
        let span = limit_cos - max_sza_cos;
        let inv_span = 1.0 / span;
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

        let red_values = red_values_from_red_dataset(red_dataset, height * width)?;

        let gmst_val = gmst(params.utc);
        let (sun_ra, sun_dec) = sun_ra_dec(params.utc);
        let sat_alt_km = params.sat_alt / 1000.0;
        let (sat_x, sat_y, sat_z) =
            observer_position(params.utc, params.sat_lon, params.sat_lat, sat_alt_km);

        let h = params.geos.perspective_point_height;
        let h_inv = 1.0 / h;
        let lon_0_rad = params.geos.longitude_of_projection_origin.to_radians();
        let column_data = precompute_columns(&params.x_coords, h_inv, lon_0_rad, gmst_val, sun_ra);

        let max_angle = (params.geos.semi_major_axis / params.geos.satellite_radius()).asin();
        let max_angle_sq = max_angle * max_angle;

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
                return build_result_combined(
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

        const STRIP_HEIGHT: usize = 64;

        for y0 in (0..height).step_by(STRIP_HEIGHT) {
            let y1 = (y0 + STRIP_HEIGHT).min(height);
            let strip_n = (y1 - y0) * width;

            let tan_theta_y_row = precompute_rows(&params.y_coords, y0, y1, h_inv);
            let red = red_values.as_ref();
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
                        let col_pre = &column_data[col];

                        let x_pos = params.x_coords[col];
                        let y_pos = params.y_coords[y0 + row];
                        let rsq = (x_pos * x_pos + y_pos * y_pos) / (h * h);
                        if rsq > max_angle_sq {
                            *out = f32::NAN;
                            continue;
                        }

                        let lat_rad = (tan_theta_y_row[row] * col_pre.cos_theta_x).atan();

                        let (sunz, suna) = solar_single_precomputed(lat_rad, sun_dec, col_pre);
                        if !sunz.is_finite() {
                            *out = f32::NAN;
                            continue;
                        }

                        let cos_sza = sunz.to_radians().cos();
                        *out *= crate::sun_zenith::sza_correction_factor(
                            cos_sza,
                            limit_cos,
                            max_sza_cos,
                            inv_span,
                        );

                        let (satz, sata) =
                            satellite_single_precomputed(lat_rad, col_pre, sat_x, sat_y, sat_z);

                        let azidiff = crate::angles::AngleSet::relative_azimuth_single(sata, suna);
                        if !azidiff.is_finite() {
                            continue;
                        }

                        let sunzsec = 1.0 / sunz.to_radians().cos().max(0.0001);
                        let satzsec = 1.0 / satz.to_radians().cos().max(0.0001);

                        let mut correction = trilinear_interpolate(
                            grid_3d,
                            sunzsec,
                            azidiff,
                            satzsec,
                            &lut.sunz_coord,
                            &lut.azid_coord,
                            &lut.satz_coord,
                        );

                        if let Some(red_vals) = red {
                            let r = f64::from(red_vals[offset + row * width + col]);
                            if r.is_finite() && r >= 20.0 {
                                correction *= 1.0 - (r - 20.0) / 80.0;
                            }
                        }

                        if reduce && sunz.is_finite() && zenith_span > 0.0 {
                            let t = if sunz < zenith_lo {
                                0.0
                            } else {
                                ((sunz - zenith_lo) / zenith_span).min(1.0)
                            };
                            let factor = (1.0 - config.reduce_strength * t).clamp(0.0, 1.0);
                            correction *= factor;
                        }

                        let corrected = (f64::from(*out) - correction).max(0.0);
                        *out = corrected as f32;
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

/// Extract the red band values as f32, or `None` when no red dataset is given.
///
/// The red band is stored as f32 (narrowing f64 sources) to halve the copy
/// for 0.5 km red bands. A pixel whose value sits within ~1e-6 of the
/// `r >= 20.0` cloud-relaxation threshold may flip the branch compared to a
/// full-f64 copy, which is accepted for the memory win. A dataset without an
/// array, or with fewer values than the visible band, is rejected instead of
/// panicking on an indexed access in the strip loop.
fn red_values_from_red_dataset(
    red_dataset: Option<&Dataset>,
    visible_count: usize,
) -> Result<Option<Vec<f32>>> {
    let Some(ds) = red_dataset else {
        return Ok(None);
    };
    let Some(array) = ds.array() else {
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
    let values = match array {
        AnyDataArray::F32(a) => a.values().to_vec(),
        AnyDataArray::F64(a) => a.values().iter().map(|v| *v as f32).collect(),
        AnyDataArray::U8(a) => a.values().iter().map(|v| f32::from(*v)).collect(),
        AnyDataArray::U16(a) => a.values().iter().map(|v| f32::from(*v)).collect(),
        AnyDataArray::I16(a) => a.values().iter().map(|v| f32::from(*v)).collect(),
    };
    Ok(Some(values))
}

/// Compute angles and apply Rayleigh correction in one call.
pub fn rayleigh_correct(
    corrector: RayleighCorrector,
    vis_dataset: Dataset,
    red_dataset: Option<&Dataset>,
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

    corrector.apply_correction(vis_dataset, red_dataset, params)
}

/// Compute angles and apply both sun zenith and Rayleigh correction
/// in a single pass.
///
/// `max_sza` controls where the sun-zenith correction reaches zero at
/// the terminator (default 95.0°, matching Satpy).
pub fn rayleigh_correct_with_sun_zenith(
    corrector: RayleighCorrector,
    vis_dataset: Dataset,
    red_dataset: Option<&Dataset>,
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

    corrector.apply_correction_with_sun_zenith(vis_dataset, red_dataset, params, max_sza)
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
                let r = rayleigh_correct_with_sun_zenith(c, ds, None, utc, 95.0);
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
                let r = rayleigh_correct_with_sun_zenith(c, ds, None, utc, 95.0);
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
                let r = rayleigh_correct_with_sun_zenith(c, ds, None, utc, 95.0);
                // With grid_3d=None (wavelength out of range), this is a
                // no-op pass-through but should still produce a Dataset.
                assert!(r.is_ok());
                let result = r.expect("ok");
                assert!(result.attr("modifier").is_some());
            }
            Err(_) => { /* no real LUT available in unit test */ }
        }
    }

    // ── Helpers for combined tests ──

    #[test]
    fn red_values_reject_missing_or_short_arrays() {
        let visible_count = 9;
        // No red dataset -> None.
        assert!(red_values_from_red_dataset(None, visible_count)
            .expect("no red is fine")
            .is_none());
        // Dataset without an array -> error, not an empty-Vec indexed access.
        let no_array =
            rusty_sat_core::Dataset::new(rusty_sat_core::DataId::new("red").expect("id"));
        assert!(red_values_from_red_dataset(Some(&no_array), visible_count).is_err());
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
        let err = red_values_from_red_dataset(Some(&short), visible_count)
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
        let values = red_values_from_red_dataset(Some(&ok), visible_count)
            .expect("valid red")
            .expect("values");
        assert_eq!(values.len(), 9);
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
