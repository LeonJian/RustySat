//! Atmospheric and geometric modifiers for satellite imagery.
//!
//! This crate implements the modifier layer of the Satpy processing pipeline.
//! Currently focused on Rayleigh scattering correction.
//!
//! # Modules
//!
//! - [`astronomy`] — solar position math (cos_zen, sun azimuth/zenith, alt/az)
//!   ported from `pyorbital.astronomy`.
//! - [`geos`] — geostationary projection inverse (x/y meters → lon/lat degrees).
//! - [`orbital`] — satellite look angles (azimuth, elevation, zenith)
//!   ported from `pyorbital.orbital.get_observer_look`.
//! - [`angles`] — combined angle computation for a dataset grid.
//! - [`rayleigh_lut`] — Rayleigh LUT data model and multilinear interpolation
//!   ported from `pyspectral.rayleigh`.
//! - [`rayleigh`] — full Rayleigh scattering correction modifier
//!   ported from `satpy.modifiers.PSPRayleighReflectance`.
//!
//! # Quick Start
//!
//! ```ignore
//! use rusty_sat_modifiers::{RayleighCorrector, RayleighLut, rayleigh_correct};
//! use rusty_sat_modifiers::astronomy::UtcInstant;
//!
//! let lut = RayleighLut::load_from_hdf5("rayleigh_lut_us-standard.h5")?;
//! let corrector = RayleighCorrector::new(lut);
//! let corrected = rayleigh_correct(
//!     corrector,
//!     vis_dataset,
//!     Some(&red_dataset),
//!     UtcInstant::from_ymdhms(2025, 9, 23, 7, 20, 0),
//!     640.0, // wavelength in nm
//! )?;
//! ```

pub mod angles;
pub mod astronomy;
pub mod geos;
pub mod lut_loader;
pub mod orbital;
pub mod rayleigh;
pub mod rayleigh_lut;

pub use angles::{extract_xy_coords, AngleParams, AngleSet};
pub use astronomy::UtcInstant;
pub use geos::GeosProjection;
pub use lut_loader::{default_lut_dir, ensure_lut, load_lut_from_hdf5, load_or_download_lut};
pub use orbital::{get_observer_look, satellite_angles_grid};
pub use rayleigh::{rayleigh_correct, AerosolType, Atmosphere, RayleighConfig, RayleighCorrector};
pub use rayleigh_lut::RayleighLut;
