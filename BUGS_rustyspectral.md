## rustyspectral Bugs Found during Rusty Sat Integration

### Bug 1: LUT download URLs are malformed (fixed in v1.0.1)

**File**: `src/utils.rs`, function `get_https_rayleigh_luts()`

**Root cause**: `let base = "https://zenodo.org/records/"` on line 397 was defined but never prepended to any URL. Line 443 had `let _ = base;` which only silenced the unused-variable warning.

**Effect**: All 11 entries in `HTTPS_RAYLEIGH_LUTS` stored bare relative paths like `"19372152/files/pyspectral_atm_correction_lut_mca.tgz"`. `download_luts()` → `ureq::Agent::get(url)` hit `"19372152/..."` → `"http: invalid format"` error. Download always failed.

**Fix in 1.0.1**: Changed `HashMap<&'static str, &'static str>` → `HashMap<&'static str, String>`, used `format!("{base}19372152/files/...")`.

---

### Bug 2: Zenodo record 19372152 has zero files (still present in v1.0.1)

**File**: `src/utils.rs`, function `get_https_rayleigh_luts()`

**Root cause**: All 11 LUT download entries reference Zenodo record **19372152**, which exists on Zenodo (record page loads at `https://zenodo.org/records/19372152`) but contains **zero files**:

```
$ curl https://zenodo.org/api/records/19372152/versions?allversions=1
→ version id=19372152, files=0

$ curl -L https://zenodo.org/records/19372152/files/pyspectral_atm_correction_lut_mca.tgz
→ 404 (14,545 byte HTML error page)

$ curl https://zenodo.org/api/records/19372152/files
→ 403 Permission denied

$ curl https://zenodo.org/api/records/19372152/files-archive
→ 403 Permission denied
```

**Evidence the API works for other records** — RSR record `19373017` has 1 file and is publicly accessible:
```
$ curl https://zenodo.org/api/records/19373017
→ files=1, pyspectral_rsr_data.tgz (5.3 MB), public
```

**Effect**: `download_luts()` calls `download_file(url, &tarball_path)`. The URL returns 404 (HTML body). `ureq` downloads the 14KB HTML error page as the `.tgz` file. `extract_tarball()` → `GzDecoder::new()` fails on HTML → download returns I/O error. `with_config_auto()` and `check_and_download()` always fail.

**Proposed fix**: Upload the 11 `.tgz` LUT archive files to Zenodo record 19372152, or update the record IDs in `get_https_rayleigh_luts()` to point to records that actually have the LUT files.
