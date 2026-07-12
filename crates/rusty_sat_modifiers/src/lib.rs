//! Atmospheric and geometric modifiers for satellite imagery.
//!
//! This crate implements the modifier layer of the Satpy processing pipeline.
//! Rayleigh scattering correction is delegated to the `rustyspectral` crate.
//!
//! # Modules
//!
//! - [`astronomy`] — solar position math (cos_zen, sun azimuth/zenith, alt/az)
//!   ported from `pyorbital.astronomy`.
//! - [`geos`] — geostationary projection inverse (x/y meters → lon/lat degrees).
//! - [`orbital`] — satellite look angles (azimuth, elevation, zenith)
//!   ported from `pyorbital.orbital.get_observer_look`.
//! - [`angles`] — combined angle computation for a dataset grid.
//! - [`rayleigh`] — Rayleigh scattering correction modifier wrapping
//!   `rustyspectral::rayleigh`.
//! - [`sun_zenith`] — solar zenith angle correction normalizing TOA
//!   reflectance to overhead-sun equivalent.
//!
//! # Quick Start
//!
//! ```ignore
//! use rusty_sat_modifiers::{RayleighCorrector, RayleighConfig, rayleigh_correct};
//! use rusty_sat_modifiers::astronomy::UtcInstant;
//!
//! let config = RayleighConfig::default();
//! let corrector = RayleighCorrector::with_config(
//!     "rayleigh_lut_us-standard.h5", config, 640.0
//! )?;
//! let corrected = rayleigh_correct(
//!     corrector,
//!     vis_dataset,
//!     Some(&red_dataset),
//!     UtcInstant::from_ymdhms(2025, 9, 23, 7, 20, 0),
//! )?;
//! ```

pub mod angles;
pub mod astronomy;
pub mod geos;
pub mod orbital;
pub mod rayleigh;
pub mod sun_zenith;

pub use angles::{extract_xy_coords, AngleParams, AngleSet};
pub use astronomy::UtcInstant;
pub use geos::GeosProjection;
pub use orbital::{get_observer_look, satellite_angles_grid};
pub use rayleigh::{
    rayleigh_correct, rayleigh_correct_with_sun_zenith, AerosolType, Atmosphere, RayleighConfig,
    RayleighCorrector,
};
pub use sun_zenith::{sun_zenith_correct, sun_zenith_correct_with, SunZenithCorrector};
