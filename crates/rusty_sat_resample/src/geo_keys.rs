//! GeoTIFF GeoKey definitions and serialization.
//!
//! Maps CRS projection parameters to GeoTIFF GeoKey directory entries
//! following the OGC GeoTIFF 1.1 specification and GDAL geotiff.h conventions.

// ---------------------------------------------------------------------------
// TIFF tag constants
// ---------------------------------------------------------------------------
pub const TIFFTAG_GEO_ASCII_PARAMS: u16 = 34737;
pub const TIFFTAG_GEO_DOUBLE_PARAMS: u16 = 34736;

// ---------------------------------------------------------------------------
// GeoKey directory header
// ---------------------------------------------------------------------------
pub const GEO_KEY_DIRECTORY_VERSION: u16 = 1;
pub const GEO_KEY_REVISION: u16 = 1;
pub const GEO_KEY_MINOR_REVISION: u16 = 0;

// ---------------------------------------------------------------------------
// GTModelTypeGeoKey (1024) values
// ---------------------------------------------------------------------------
pub const GT_MODEL_TYPE_GEO_KEY: u16 = 1024;
pub const MODEL_TYPE_PROJECTED: u16 = 1;
pub const MODEL_TYPE_GEOGRAPHIC: u16 = 2;

// ---------------------------------------------------------------------------
// GTRasterTypeGeoKey (1025) values
// ---------------------------------------------------------------------------
pub const GT_RASTER_TYPE_GEO_KEY: u16 = 1025;
pub const RASTER_PIXEL_IS_AREA: u16 = 1;

// ---------------------------------------------------------------------------
// Geographic CS keys
// ---------------------------------------------------------------------------
pub const GEOGRAPHIC_TYPE_GEO_KEY: u16 = 2048;
pub const GEOG_CITATION_GEO_KEY: u16 = 2049;
pub const GEOG_GEODETIC_DATUM_GEO_KEY: u16 = 2050;
pub const GEOG_ANGULAR_UNITS_GEO_KEY: u16 = 2054;
pub const GEOG_SEMI_MAJOR_AXIS_GEO_KEY: u16 = 2057;
pub const GEOG_SEMI_MINOR_AXIS_GEO_KEY: u16 = 2058;
pub const GEOG_INV_FLATTENING_GEO_KEY: u16 = 2059;

// ---------------------------------------------------------------------------
// Projected CS keys
// ---------------------------------------------------------------------------
pub const PROJECTED_CS_TYPE_GEO_KEY: u16 = 3072;
pub const PROJECTED_CITATION_GEO_KEY: u16 = 3073;
pub const PROJ_COORD_TRANS_GEO_KEY: u16 = 3075;
pub const PROJ_LINEAR_UNITS_GEO_KEY: u16 = 3076;
pub const PROJ_NAT_ORIGIN_LAT_GEO_KEY: u16 = 3081;
pub const PROJ_NAT_ORIGIN_LONG_GEO_KEY: u16 = 3080;
pub const PROJ_FALSE_EASTING_GEO_KEY: u16 = 3084;
pub const PROJ_FALSE_NORTHING_GEO_KEY: u16 = 3085;
pub const PROJ_CENTER_LONG_GEO_KEY: u16 = 3088;
pub const PROJ_CENTER_LAT_GEO_KEY: u16 = 3089;
pub const PROJ_SEMI_MAJOR_AXIS_GEO_KEY: u16 = 3089;
pub const PROJ_SEMI_MINOR_AXIS_GEO_KEY: u16 = 3090;
pub const PROJ_INV_FLATTENING_GEO_KEY: u16 = 3091;

// ---------------------------------------------------------------------------
// ProjCoordTransGeoKey (3075) values
// ---------------------------------------------------------------------------
pub const CT_MERCATOR: u16 = 7;
pub const CT_LAMBERT_AZIMUTHAL_EQUAL_AREA: u16 = 10;
pub const CT_STEREOGRAPHIC: u16 = 14;
pub const CT_POLAR_STEREOGRAPHIC: u16 = 15;
pub const CT_GEOSTATIONARY_SATELLITE: u16 = 28;

// ---------------------------------------------------------------------------
// Unit codes
// ---------------------------------------------------------------------------
pub const LINEAR_UNIT_METER: u16 = 9001;
pub const ANGULAR_UNIT_DEGREE: u16 = 9102;

// ---------------------------------------------------------------------------
// EPSG / authority codes
// ---------------------------------------------------------------------------
pub const EPSG_WGS_84: u16 = 4326;
pub const EPSG_DATUM_WGS_84: u16 = 6326;

// ---------------------------------------------------------------------------
// Sentinel
// ---------------------------------------------------------------------------
pub const GEO_USER_DEFINED: u16 = 32767;

// ---------------------------------------------------------------------------
// Logical GeoKey types
// ---------------------------------------------------------------------------

/// The logical value for a single GeoTIFF GeoKey.
#[derive(Debug, Clone, PartialEq)]
pub enum GeoKeyValue {
    /// u16 value stored inline in the GeoKey directory (TIFFTagLocation=0).
    Short(u16),
    /// String stored in the GeoAsciiParams tag (TIFFTagLocation=34737).
    Ascii(String),
    /// f64 value stored in the GeoDoubleParams tag (TIFFTagLocation=34736).
    Double(f64),
}

/// A single GeoKey entry in logical form (before offset computation).
#[derive(Debug, Clone, PartialEq)]
pub struct GeoKeyDef {
    pub key_id: u16,
    pub value: GeoKeyValue,
}

impl GeoKeyDef {
    pub fn short(key_id: u16, value: u16) -> Self {
        Self {
            key_id,
            value: GeoKeyValue::Short(value),
        }
    }

    pub fn ascii(key_id: u16, value: impl Into<String>) -> Self {
        Self {
            key_id,
            value: GeoKeyValue::Ascii(value.into()),
        }
    }

    pub fn double(key_id: u16, value: f64) -> Self {
        Self {
            key_id,
            value: GeoKeyValue::Double(value),
        }
    }
}

/// A finalized GeoKey set with computed byte offsets ready for TIFF encoding.
#[derive(Debug, Clone, PartialEq)]
pub struct GeoTiffGeoKeyFinal {
    /// Directory entries: [key_id, tiff_tag_location, count, value_offset].
    pub directory_entries: Vec<[u16; 4]>,
    /// GeoAsciiParams tag payload (concatenated strings with `|` terminators).
    pub ascii_params: Vec<u8>,
    /// GeoDoubleParams tag payload (little-endian f64 bytes).
    pub double_params: Vec<u8>,
}

impl GeoTiffGeoKeyFinal {
    /// Total number of GeoKey entries (used for the directory header).
    pub fn key_count(&self) -> u16 {
        u16::try_from(self.directory_entries.len()).unwrap_or(u16::MAX)
    }
}

/// Finalize logical GeoKey definitions into binary-ready form.
///
/// Assigns offsets into the ASCII and double parameter blobs.
/// Double param entries use *element index* (0-based), not byte offset —
/// this matches the GeoTIFF spec for `GeoDoubleParamsTag`.
pub fn finalize_geo_key_defs(defs: &[GeoKeyDef]) -> GeoTiffGeoKeyFinal {
    let mut ascii_blob = Vec::new();
    let mut ascii_offsets: Vec<u32> = Vec::with_capacity(defs.len());
    let mut double_elements: Vec<f64> = Vec::new();

    for def in defs {
        match &def.value {
            GeoKeyValue::Ascii(s) => {
                ascii_offsets.push(u32::try_from(ascii_blob.len()).unwrap_or(0));
                ascii_blob.extend_from_slice(s.as_bytes());
                ascii_blob.push(b'|');
            }
            GeoKeyValue::Double(_v) => {
                // offset is the element index, computed after collection
                let _ = _v; // pushed below
            }
            GeoKeyValue::Short(_) => {}
        }
    }

    let mut entries = Vec::with_capacity(defs.len());
    let mut ascii_idx: u32 = 0;
    let mut double_idx: u32 = 0;

    for def in defs {
        match &def.value {
            GeoKeyValue::Short(v) => {
                entries.push([def.key_id, 0, 1, *v]);
            }
            GeoKeyValue::Ascii(s) => {
                let offset = ascii_offsets[ascii_idx as usize];
                let count = u16::try_from(s.len()).unwrap_or(u16::MAX);
                entries.push([def.key_id, TIFFTAG_GEO_ASCII_PARAMS, count, offset as u16]);
                ascii_idx += 1;
            }
            GeoKeyValue::Double(v) => {
                double_elements.push(*v);
                entries.push([
                    def.key_id,
                    TIFFTAG_GEO_DOUBLE_PARAMS,
                    1,
                    double_idx as u16, // element index, NOT byte offset
                ]);
                double_idx += 1;
            }
        }
    }

    let double_params: Vec<u8> = double_elements
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();

    GeoTiffGeoKeyFinal {
        directory_entries: entries,
        ascii_params: ascii_blob,
        double_params,
    }
}

/// Serialize the full GeoKey directory to bytes (header + entries).
pub fn serialize_geo_key_directory(finalized: &GeoTiffGeoKeyFinal) -> Vec<u8> {
    let key_count = finalized.key_count();
    let entry_count = 4 + 4 * key_count as usize; // header(4) + entries(4 shorts each)
    let mut buf = Vec::with_capacity(entry_count * 2);
    // header
    for v in [
        GEO_KEY_DIRECTORY_VERSION,
        GEO_KEY_REVISION,
        GEO_KEY_MINOR_REVISION,
        key_count,
    ] {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    // entries
    for entry in &finalized.directory_entries {
        for v in entry {
            buf.extend_from_slice(&v.to_le_bytes());
        }
    }
    buf
}

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------
    // Constant value tests (cross-reference with GDAL geotiff.h)
    // ------------------------------------------------------------------

    #[test]
    fn geo_key_ids_match_gdal_geotiff_h() {
        assert_eq!(GT_MODEL_TYPE_GEO_KEY, 1024);
        assert_eq!(GT_RASTER_TYPE_GEO_KEY, 1025);
        assert_eq!(GEOGRAPHIC_TYPE_GEO_KEY, 2048);
        assert_eq!(GEOG_CITATION_GEO_KEY, 2049);
        assert_eq!(GEOG_GEODETIC_DATUM_GEO_KEY, 2050);
        assert_eq!(GEOG_ANGULAR_UNITS_GEO_KEY, 2054);
        assert_eq!(GEOG_SEMI_MAJOR_AXIS_GEO_KEY, 2057);
        assert_eq!(GEOG_SEMI_MINOR_AXIS_GEO_KEY, 2058);
        assert_eq!(GEOG_INV_FLATTENING_GEO_KEY, 2059);
        assert_eq!(PROJECTED_CS_TYPE_GEO_KEY, 3072);
        assert_eq!(PROJECTED_CITATION_GEO_KEY, 3073);
        assert_eq!(PROJ_COORD_TRANS_GEO_KEY, 3075);
        assert_eq!(PROJ_LINEAR_UNITS_GEO_KEY, 3076);
        assert_eq!(PROJ_NAT_ORIGIN_LONG_GEO_KEY, 3080);
        assert_eq!(PROJ_NAT_ORIGIN_LAT_GEO_KEY, 3081);
        assert_eq!(PROJ_FALSE_EASTING_GEO_KEY, 3084);
        assert_eq!(PROJ_FALSE_NORTHING_GEO_KEY, 3085);
        assert_eq!(PROJ_CENTER_LONG_GEO_KEY, 3088);
    }

    #[test]
    fn geo_key_values_match_gdal_geotiff_h() {
        assert_eq!(MODEL_TYPE_PROJECTED, 1);
        assert_eq!(MODEL_TYPE_GEOGRAPHIC, 2);
        assert_eq!(RASTER_PIXEL_IS_AREA, 1);
        assert_eq!(CT_MERCATOR, 7);
        assert_eq!(CT_LAMBERT_AZIMUTHAL_EQUAL_AREA, 10);
        assert_eq!(CT_STEREOGRAPHIC, 14);
        assert_eq!(CT_POLAR_STEREOGRAPHIC, 15);
        assert_eq!(CT_GEOSTATIONARY_SATELLITE, 28);
        assert_eq!(LINEAR_UNIT_METER, 9001);
        assert_eq!(ANGULAR_UNIT_DEGREE, 9102);
        assert_eq!(EPSG_WGS_84, 4326);
        assert_eq!(EPSG_DATUM_WGS_84, 6326);
        assert_eq!(GEO_USER_DEFINED, 32767);
    }

    #[test]
    fn tiff_tag_constants() {
        assert_eq!(TIFFTAG_GEO_ASCII_PARAMS, 34737);
        assert_eq!(TIFFTAG_GEO_DOUBLE_PARAMS, 34736);
    }

    // ------------------------------------------------------------------
    // GeoKeyDef constructors
    // ------------------------------------------------------------------

    #[test]
    fn geo_key_def_short() {
        let def = GeoKeyDef::short(GT_MODEL_TYPE_GEO_KEY, MODEL_TYPE_PROJECTED);
        assert_eq!(def.key_id, 1024);
        assert_eq!(def.value, GeoKeyValue::Short(1));
    }

    #[test]
    fn geo_key_def_ascii() {
        let def = GeoKeyDef::ascii(PROJECTED_CITATION_GEO_KEY, "test citation");
        assert_eq!(def.key_id, 3073);
        assert_eq!(def.value, GeoKeyValue::Ascii("test citation".to_string()));
    }

    #[test]
    fn geo_key_def_double() {
        let def = GeoKeyDef::double(PROJ_CENTER_LONG_GEO_KEY, 140.7);
        assert_eq!(def.key_id, 3088);
        assert_eq!(def.value, GeoKeyValue::Double(140.7));
    }

    // ------------------------------------------------------------------
    // finalize_geo_key_defs
    // ------------------------------------------------------------------

    #[test]
    fn finalize_short_only() {
        let defs = [
            GeoKeyDef::short(GT_MODEL_TYPE_GEO_KEY, MODEL_TYPE_PROJECTED),
            GeoKeyDef::short(GT_RASTER_TYPE_GEO_KEY, RASTER_PIXEL_IS_AREA),
        ];
        let result = finalize_geo_key_defs(&defs);
        assert_eq!(result.key_count(), 2);
        assert!(result.ascii_params.is_empty());
        assert!(result.double_params.is_empty());
        // Short entries: location=0, count=1, value=the u16
        assert_eq!(
            result.directory_entries[0],
            [GT_MODEL_TYPE_GEO_KEY, 0, 1, MODEL_TYPE_PROJECTED]
        );
        assert_eq!(
            result.directory_entries[1],
            [GT_RASTER_TYPE_GEO_KEY, 0, 1, RASTER_PIXEL_IS_AREA]
        );
    }

    #[test]
    fn finalize_single_ascii() {
        let defs = [GeoKeyDef::ascii(PROJECTED_CITATION_GEO_KEY, "GEOS 140.7")];
        let result = finalize_geo_key_defs(&defs);
        assert_eq!(result.key_count(), 1);
        assert_eq!(result.ascii_params, b"GEOS 140.7|");
        assert!(result.double_params.is_empty());
        // Ascii entry: location=34737, count=byte_len, value=byte_offset(0)
        assert_eq!(
            result.directory_entries[0],
            [PROJECTED_CITATION_GEO_KEY, TIFFTAG_GEO_ASCII_PARAMS, 10, 0]
        );
    }

    #[test]
    fn finalize_multiple_ascii_correct_offsets() {
        let defs = [
            GeoKeyDef::ascii(PROJECTED_CITATION_GEO_KEY, "first"),
            GeoKeyDef::ascii(GEOG_CITATION_GEO_KEY, "second"),
        ];
        let result = finalize_geo_key_defs(&defs);
        assert_eq!(result.ascii_params, b"first|second|");
        // "first" = 5 bytes, offset 0
        assert_eq!(
            result.directory_entries[0],
            [PROJECTED_CITATION_GEO_KEY, TIFFTAG_GEO_ASCII_PARAMS, 5, 0]
        );
        // "first|" = 6 bytes consumed, so "second" starts at offset 6
        assert_eq!(
            result.directory_entries[1],
            [GEOG_CITATION_GEO_KEY, TIFFTAG_GEO_ASCII_PARAMS, 6, 6]
        );
    }

    #[test]
    fn finalize_double_entries_use_element_index() {
        let defs = [
            GeoKeyDef::double(PROJ_CENTER_LONG_GEO_KEY, 140.7),
            GeoKeyDef::double(PROJ_NAT_ORIGIN_LAT_GEO_KEY, -90.0),
        ];
        let result = finalize_geo_key_defs(&defs);
        assert_eq!(result.key_count(), 2);
        assert!(result.ascii_params.is_empty());
        // double_params: 2 × 8 = 16 bytes
        assert_eq!(result.double_params.len(), 16);
        // First double: element index 0
        assert_eq!(
            result.directory_entries[0],
            [PROJ_CENTER_LONG_GEO_KEY, TIFFTAG_GEO_DOUBLE_PARAMS, 1, 0]
        );
        // Second double: element index 1 (NOT byte offset 8!)
        assert_eq!(
            result.directory_entries[1],
            [PROJ_NAT_ORIGIN_LAT_GEO_KEY, TIFFTAG_GEO_DOUBLE_PARAMS, 1, 1]
        );
        // Verify the actual f64 bytes
        let v0 = f64::from_le_bytes(result.double_params[0..8].try_into().unwrap());
        let v1 = f64::from_le_bytes(result.double_params[8..16].try_into().unwrap());
        assert!((v0 - 140.7).abs() < 1e-10);
        assert!((v1 + 90.0).abs() < 1e-10);
    }

    #[test]
    fn finalize_mixed_short_ascii_double() {
        let defs = [
            GeoKeyDef::short(GT_MODEL_TYPE_GEO_KEY, MODEL_TYPE_PROJECTED),
            GeoKeyDef::ascii(PROJECTED_CITATION_GEO_KEY, "test"),
            GeoKeyDef::double(PROJ_CENTER_LONG_GEO_KEY, 140.7),
            GeoKeyDef::short(GT_RASTER_TYPE_GEO_KEY, RASTER_PIXEL_IS_AREA),
        ];
        let result = finalize_geo_key_defs(&defs);
        assert_eq!(result.key_count(), 4);
        assert_eq!(result.ascii_params, b"test|");
        assert_eq!(result.double_params.len(), 8);
        // Short (inline)
        assert_eq!(
            result.directory_entries[0],
            [GT_MODEL_TYPE_GEO_KEY, 0, 1, MODEL_TYPE_PROJECTED]
        );
        // Ascii (tag 34737, offset 0, count 4)
        assert_eq!(
            result.directory_entries[1],
            [PROJECTED_CITATION_GEO_KEY, TIFFTAG_GEO_ASCII_PARAMS, 4, 0]
        );
        // Double (tag 34736, element index 0)
        assert_eq!(
            result.directory_entries[2],
            [PROJ_CENTER_LONG_GEO_KEY, TIFFTAG_GEO_DOUBLE_PARAMS, 1, 0]
        );
        // Short (inline)
        assert_eq!(
            result.directory_entries[3],
            [GT_RASTER_TYPE_GEO_KEY, 0, 1, RASTER_PIXEL_IS_AREA]
        );
    }

    #[test]
    fn finalize_empty_defs() {
        let result = finalize_geo_key_defs(&[]);
        assert_eq!(result.key_count(), 0);
        assert!(result.ascii_params.is_empty());
        assert!(result.double_params.is_empty());
        assert!(result.directory_entries.is_empty());
    }

    // ------------------------------------------------------------------
    // serialize_geo_key_directory
    // ------------------------------------------------------------------

    #[test]
    fn serialize_short_only_keys() {
        let defs = [
            GeoKeyDef::short(GT_MODEL_TYPE_GEO_KEY, MODEL_TYPE_GEOGRAPHIC),
            GeoKeyDef::short(GEOGRAPHIC_TYPE_GEO_KEY, EPSG_WGS_84),
        ];
        let finalized = finalize_geo_key_defs(&defs);
        let bytes = serialize_geo_key_directory(&finalized);
        // header: 4 × u16 = 8 bytes
        assert_eq!(u16::from_le_bytes(bytes[0..2].try_into().unwrap()), 1); // version
        assert_eq!(u16::from_le_bytes(bytes[2..4].try_into().unwrap()), 1); // revision
        assert_eq!(u16::from_le_bytes(bytes[4..6].try_into().unwrap()), 0); // minor
        assert_eq!(u16::from_le_bytes(bytes[6..8].try_into().unwrap()), 2); // key_count
                                                                            // entry 0: GTModelTypeGeoKey=1024, location=0, count=1, value=2
        assert_eq!(
            u16::from_le_bytes(bytes[8..10].try_into().unwrap()),
            GT_MODEL_TYPE_GEO_KEY
        );
        assert_eq!(u16::from_le_bytes(bytes[10..12].try_into().unwrap()), 0);
        assert_eq!(u16::from_le_bytes(bytes[12..14].try_into().unwrap()), 1);
        assert_eq!(
            u16::from_le_bytes(bytes[14..16].try_into().unwrap()),
            MODEL_TYPE_GEOGRAPHIC
        );
        // entry 1
        assert_eq!(
            u16::from_le_bytes(bytes[16..18].try_into().unwrap()),
            GEOGRAPHIC_TYPE_GEO_KEY
        );
    }

    #[test]
    fn serialize_with_ascii_and_double() {
        let defs = [
            GeoKeyDef::short(GT_MODEL_TYPE_GEO_KEY, MODEL_TYPE_PROJECTED),
            GeoKeyDef::ascii(PROJECTED_CITATION_GEO_KEY, "hi"),
            GeoKeyDef::double(PROJ_CENTER_LONG_GEO_KEY, 1.5),
        ];
        let finalized = finalize_geo_key_defs(&defs);
        let bytes = serialize_geo_key_directory(&finalized);
        // 4 header + 3×4 entries = 16 u16 = 32 bytes
        assert_eq!(bytes.len(), 32);
        assert_eq!(u16::from_le_bytes(bytes[6..8].try_into().unwrap()), 3); // key_count
                                                                            // ascii entry at index 1: key_id=3073, location=34737, count=2, offset=0
        let off = 8 + 8; // skip header + first entry
        assert_eq!(
            u16::from_le_bytes(bytes[off..off + 2].try_into().unwrap()),
            PROJECTED_CITATION_GEO_KEY
        );
        assert_eq!(
            u16::from_le_bytes(bytes[off + 2..off + 4].try_into().unwrap()),
            TIFFTAG_GEO_ASCII_PARAMS
        );
        assert_eq!(
            u16::from_le_bytes(bytes[off + 4..off + 6].try_into().unwrap()),
            2
        ); // count = "hi".len()
           // double entry at index 2: key_id=3088, location=34736, count=1, offset=0
        let off2 = off + 8;
        assert_eq!(
            u16::from_le_bytes(bytes[off2..off2 + 2].try_into().unwrap()),
            PROJ_CENTER_LONG_GEO_KEY
        );
        assert_eq!(
            u16::from_le_bytes(bytes[off2 + 2..off2 + 4].try_into().unwrap()),
            TIFFTAG_GEO_DOUBLE_PARAMS
        );
        assert_eq!(
            u16::from_le_bytes(bytes[off2 + 4..off2 + 6].try_into().unwrap()),
            1
        );
        assert_eq!(
            u16::from_le_bytes(bytes[off2 + 6..off2 + 8].try_into().unwrap()),
            0
        ); // element index
    }
}
