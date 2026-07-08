//! Rayleigh LUT loading from pyspectral HDF5 files and automatic download.
//!
//! Reference:
//! - `deps/pyspectral/pyspectral/rayleigh.py` — LUT file naming and structure.
//! - `deps/pyspectral/pyspectral/utils.py` — `get_rayleigh_lut_dir`,
//!   `HTTPS_RAYLEIGH_LUTS`, `download_luts`.
//! - `deps/pyspectral/pyspectral/tests/data.py` — test LUT data structure.
//!
//! The LUT HDF5 files are hosted on Zenodo:
//!   https://zenodo.org/record/1288441/files/pyspectral_atm_correction_luts_{name}.tgz
//!
//! Each archive extracts to a directory containing:
//!   rayleigh_lut_{atmosphere}.h5
//!
//! Each HDF5 file contains:
//! - `reflectance`: float64 [n_wvl, n_sunz, n_azid, n_satz]
//! - `wavelengths`: float64 [n_wvl]  (nm)
//! - `azimuth_difference`: float64 [n_azid]  (degrees)
//! - `sun_zenith_secant`: float64 [n_sunz]
//! - `satellite_zenith_secant`: float64 [n_satz]

use crate::rayleigh::{AerosolType, Atmosphere};
use crate::rayleigh_lut::RayleighLut;
use rusty_sat_core::{Result, RustySatError};
use std::fs;
use std::path::{Path, PathBuf};

const LUT_URL_PREFIX: &str =
    "https://zenodo.org/record/1288441/files/pyspectral_atm_correction_luts";

/// Default local cache directory for Rayleigh LUTs.
///
/// Uses `~/.cache/rusty_sat/rayleigh_luts` on Unix or
/// `%LOCALAPPDATA%\rusty_sat\rayleigh_luts` on Windows.
pub fn default_lut_dir() -> PathBuf {
    if let Some(home) = std::env::var_os("HOME") {
        PathBuf::from(home)
            .join(".cache")
            .join("rusty_sat")
            .join("rayleigh_luts")
    } else if let Some(local) = std::env::var_os("LOCALAPPDATA") {
        PathBuf::from(local).join("rusty_sat").join("rayleigh_luts")
    } else {
        PathBuf::from(".rusty_sat_rayleigh_luts")
    }
}

/// Get the LUT directory for a specific aerosol type.
pub fn lut_dir_for(base: &Path, aerosol: AerosolType) -> PathBuf {
    base.join(aerosol.dir_name())
}

/// Get the LUT file path for a specific aerosol type and atmosphere.
pub fn lut_file_path(base: &Path, aerosol: AerosolType, atm: Atmosphere) -> PathBuf {
    lut_dir_for(base, aerosol).join(format!("rayleigh_lut_{}.h5", atm.file_suffix()))
}

/// Get the download URL for a specific aerosol type's LUT archive.
pub fn lut_download_url(aerosol: AerosolType) -> String {
    let name = match aerosol {
        AerosolType::RayleighOnly => "no_aerosol",
        _ => aerosol.dir_name(),
    };
    format!("{LUT_URL_PREFIX}_{name}.tgz")
}

/// Load a Rayleigh LUT from a pyspectral HDF5 file.
///
/// The file must contain datasets: `reflectance`, `wavelengths`,
/// `azimuth_difference`, `sun_zenith_secant`, `satellite_zenith_secant`.
///
/// Uses `hdf5-pure` (pure Rust, no C dependency) to read the file.
pub fn load_lut_from_hdf5(path: &Path) -> Result<RayleighLut> {
    let bytes = fs::read(path).map_err(|e| {
        RustySatError::invalid_input(format!("failed to read LUT file '{}': {e}", path.display()))
    })?;

    let file = hdf5_pure::File::from_bytes(bytes)
        .map_err(|e| RustySatError::invalid_input(format!("failed to parse HDF5 LUT file: {e}")))?;

    let reflectance = file
        .dataset("reflectance")
        .map_err(|e| RustySatError::invalid_input(format!("missing 'reflectance' dataset: {e}")))?
        .read_f64()
        .map_err(|e| RustySatError::invalid_input(format!("failed to read reflectance: {e}")))?;

    let wavelengths = file
        .dataset("wavelengths")
        .map_err(|e| RustySatError::invalid_input(format!("missing 'wavelengths' dataset: {e}")))?
        .read_f64()
        .map_err(|e| RustySatError::invalid_input(format!("failed to read wavelengths: {e}")))?;

    let azimuth_difference = file
        .dataset("azimuth_difference")
        .map_err(|e| {
            RustySatError::invalid_input(format!("missing 'azimuth_difference' dataset: {e}"))
        })?
        .read_f64()
        .map_err(|e| {
            RustySatError::invalid_input(format!("failed to read azimuth_difference: {e}"))
        })?;

    let sun_zenith_secant = file
        .dataset("sun_zenith_secant")
        .map_err(|e| {
            RustySatError::invalid_input(format!("missing 'sun_zenith_secant' dataset: {e}"))
        })?
        .read_f64()
        .map_err(|e| {
            RustySatError::invalid_input(format!("failed to read sun_zenith_secant: {e}"))
        })?;

    let satellite_zenith_secant = file
        .dataset("satellite_zenith_secant")
        .map_err(|e| {
            RustySatError::invalid_input(format!("missing 'satellite_zenith_secant' dataset: {e}"))
        })?
        .read_f64()
        .map_err(|e| {
            RustySatError::invalid_input(format!("failed to read satellite_zenith_secant: {e}"))
        })?;

    Ok(RayleighLut {
        reflectance,
        wavelengths,
        sun_zenith_secant,
        azimuth_difference,
        satellite_zenith_secant,
    })
}

/// Ensure the LUT for the given aerosol type and atmosphere is available
/// locally, downloading it if necessary.
///
/// 1. Checks if the HDF5 file exists at `base/aerosol/rayleigh_lut_{atm}.h5`.
/// 2. If not, downloads the `.tgz` archive from Zenodo.
/// 3. Extracts the relevant `.h5` file.
/// 4. Returns the path to the extracted HDF5 file.
///
/// Requires network access for the download step.  The download uses
/// `curl` as a subprocess (to avoid pulling in a full HTTP client dependency).
/// Falls back to `wget` if `curl` is not available.
pub fn ensure_lut(base: &Path, aerosol: AerosolType, atm: Atmosphere) -> Result<PathBuf> {
    let target = lut_file_path(base, aerosol, atm);
    if target.is_file() {
        return Ok(target);
    }

    let dir = lut_dir_for(base, aerosol);
    fs::create_dir_all(&dir).map_err(|e| {
        RustySatError::invalid_input(format!("failed to create LUT directory: {e}"))
    })?;

    let url = lut_download_url(aerosol);
    let tgz_path = dir.join("luts.tgz");

    eprintln!("Downloading Rayleigh LUT from {url} ...");
    download_file(&url, &tgz_path)?;

    eprintln!("Extracting Rayleigh LUT ...");
    extract_tgz(&tgz_path, &dir)?;

    // Clean up the archive to save space.
    let _ = fs::remove_file(&tgz_path);

    if target.is_file() {
        Ok(target)
    } else {
        Err(RustySatError::not_found(format!(
            "LUT file not found after extraction: {}",
            target.display()
        )))
    }
}

/// Load a LUT, downloading it first if it is not cached locally.
pub fn load_or_download_lut(
    base: &Path,
    aerosol: AerosolType,
    atm: Atmosphere,
) -> Result<RayleighLut> {
    let path = ensure_lut(base, aerosol, atm)?;
    load_lut_from_hdf5(&path)
}

/// Download a file using `curl` or `wget`.
fn download_file(url: &str, dest: &Path) -> Result<()> {
    use std::process::Command;

    // Try curl first.
    let curl_result = Command::new("curl")
        .args(["-L", "--retry", "3", "--fail", "-o"])
        .arg(dest)
        .arg(url)
        .output();

    if let Ok(output) = curl_result {
        if output.status.success() {
            return Ok(());
        }
    }

    // Fall back to wget.
    let wget_result = Command::new("wget")
        .args(["-q", "--tries=3", "--timeout=120", "-O"])
        .arg(dest)
        .arg(url)
        .output();

    if let Ok(output) = wget_result {
        if output.status.success() {
            return Ok(());
        }
    }

    Err(RustySatError::invalid_input(format!(
        "failed to download LUT from {url} (neither curl nor wget succeeded)"
    )))
}

/// Extract a `.tgz` (gzip-compressed tar) file using the `tar` command.
fn extract_tgz(archive: &Path, dest_dir: &Path) -> Result<()> {
    use std::process::Command;

    let output = Command::new("tar")
        .args(["xzf"])
        .arg(archive)
        .args(["-C"])
        .arg(dest_dir)
        .output()
        .map_err(|e| RustySatError::invalid_input(format!("failed to run tar: {e}")))?;

    if !output.status.success() {
        // Fall back to manual gzip decompression + tar.
        return extract_tgz_manual(archive, dest_dir);
    }

    Ok(())
}

/// Fallback decompression using Python's tarfile module.
fn extract_tgz_manual(archive: &Path, dest_dir: &Path) -> Result<()> {
    let py_result = std::process::Command::new("python3")
        .args([
            "-c",
            "import tarfile,sys; tarfile.open(sys.argv[1]).extractall(sys.argv[2])",
            archive.to_str().unwrap_or(""),
            dest_dir.to_str().unwrap_or(""),
        ])
        .output();

    if let Ok(output) = py_result {
        if output.status.success() {
            return Ok(());
        }
    }

    Err(RustySatError::invalid_input(
        "failed to extract tgz archive (tar and python3 both failed)",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_lut_dir_is_nonempty() {
        let dir = default_lut_dir();
        assert!(!dir.as_os_str().is_empty());
    }

    #[test]
    fn lut_file_path_has_correct_name() {
        let base = PathBuf::from("/tmp/luts");
        let path = lut_file_path(
            &base,
            AerosolType::MarineCleanAerosol,
            Atmosphere::UsStandard,
        );
        assert!(path.to_str().unwrap().contains("marine_clean_aerosol"));
        assert!(path
            .to_str()
            .unwrap()
            .contains("rayleigh_lut_us-standard.h5"));
    }

    #[test]
    fn download_url_is_correct() {
        let url = lut_download_url(AerosolType::MarineCleanAerosol);
        assert!(url.contains("marine_clean_aerosol"));
        assert!(url.contains("zenodo.org"));

        let url_rayleigh = lut_download_url(AerosolType::RayleighOnly);
        assert!(url_rayleigh.contains("no_aerosol"));
    }

    #[test]
    fn ensure_lut_returns_existing_file() {
        // This test verifies the "already exists" path without downloading.
        let tmp = std::env::temp_dir().join("rusty_sat_lut_test");
        let dir = tmp.join("marine_clean_aerosol");
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("rayleigh_lut_us-standard.h5");
        std::fs::write(&file, b"dummy").unwrap();

        let result = ensure_lut(
            &tmp,
            AerosolType::MarineCleanAerosol,
            Atmosphere::UsStandard,
        );
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), file);

        let _ = std::fs::remove_dir_all(&tmp);
    }
}
