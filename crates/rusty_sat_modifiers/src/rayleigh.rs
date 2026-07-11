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

        let red_values: Option<Vec<f64>> =
            red_dataset.map(|ds| ds.array().map(|a| a.values_as_f64()).unwrap_or_default());

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
            vis_f32[offset..offset + strip_n]
                .par_iter_mut()
                .enumerate()
                .for_each(|(local_i, out)| {
                    if !out.is_finite() {
                        return;
                    }
                    let row = local_i / width;
                    let col = local_i % width;
                    let col_pre = &column_data[col];

                    let lat_rad = (tan_theta_y_row[row] * col_pre.cos_theta_x).atan();

                    let (sunz, suna) = solar_single_precomputed(lat_rad, sun_dec, col_pre);
                    let (satz, sata) =
                        satellite_single_precomputed(lat_rad, col_pre, sat_x, sat_y, sat_z);

                    let azidiff = crate::angles::AngleSet::relative_azimuth_single(sata, suna);
                    if !azidiff.is_finite() {
                        return;
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
                        let r = red_vals[offset + local_i];
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

                    let corrected = (*out as f64 - correction).max(0.0);
                    *out = corrected as f32;
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

fn into_vis_f32(array: AnyDataArray) -> Vec<f32> {
    match array {
        AnyDataArray::F32(da) => da.into_values(),
        AnyDataArray::F64(da) => da.into_values().into_iter().map(|v| v as f32).collect(),
        AnyDataArray::U8(da) => da.into_values().into_iter().map(|v| v as f32).collect(),
        AnyDataArray::U16(da) => da.into_values().into_iter().map(|v| v as f32).collect(),
        AnyDataArray::I16(da) => da.into_values().into_iter().map(|v| v as f32).collect(),
    }
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
}
