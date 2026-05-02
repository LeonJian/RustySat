//! Trollsift-compatible filename pattern helpers.
//!
//! This module intentionally starts with the common subset Satpy readers need:
//! Python-style named fields, non-greedy string matching, integer/float
//! extraction, strftime-shaped fields, validation, composing, and globifying.
//! Reference behavior inspected before implementation:
//! `deps/trollsift/doc/source/usage.rst` and `deps/trollsift/trollsift/parser.py`.

use regex::Regex;
use rusty_sat_core::{Result, RustySatError};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, PartialEq)]
pub enum PatternValue {
    Text(String),
    Integer(i64),
    Float(f64),
    DateTime(PatternDateTime),
}

impl PatternValue {
    fn as_compose_text(&self) -> String {
        match self {
            Self::Text(value) => value.clone(),
            Self::Integer(value) => value.to_string(),
            Self::Float(value) => value.to_string(),
            Self::DateTime(value) => value.to_compact_string(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PatternDateTime {
    pub year: i32,
    pub month: u8,
    pub day: u8,
    pub hour: u8,
    pub minute: u8,
    pub second: u8,
    pub ordinal_day: Option<u16>,
}

impl PatternDateTime {
    pub fn new(year: i32, month: u8, day: u8, hour: u8, minute: u8, second: u8) -> Result<Self> {
        if !(1..=12).contains(&month) {
            return Err(RustySatError::invalid_input(
                "datetime month must be in 1..=12",
            ));
        }
        if !(1..=31).contains(&day) {
            return Err(RustySatError::invalid_input(
                "datetime day must be in 1..=31",
            ));
        }
        if hour > 23 || minute > 59 || second > 59 {
            return Err(RustySatError::invalid_input(
                "datetime time must be within 24-hour clock bounds",
            ));
        }
        Ok(Self {
            year,
            month,
            day,
            hour,
            minute,
            second,
            ordinal_day: None,
        })
    }

    fn with_ordinal_day(mut self, ordinal_day: u16) -> Result<Self> {
        if !(1..=366).contains(&ordinal_day) {
            return Err(RustySatError::invalid_input(
                "datetime ordinal day must be in 1..=366",
            ));
        }
        self.ordinal_day = Some(ordinal_day);
        Ok(self)
    }

    fn to_compact_string(&self) -> String {
        format!(
            "{:04}{:02}{:02}{:02}{:02}{:02}",
            self.year, self.month, self.day, self.hour, self.minute, self.second
        )
    }
}

impl From<PatternDateTime> for PatternValue {
    fn from(value: PatternDateTime) -> Self {
        Self::DateTime(value)
    }
}

impl From<&str> for PatternValue {
    fn from(value: &str) -> Self {
        Self::Text(value.to_string())
    }
}

impl From<String> for PatternValue {
    fn from(value: String) -> Self {
        Self::Text(value)
    }
}

impl From<i64> for PatternValue {
    fn from(value: i64) -> Self {
        Self::Integer(value)
    }
}

impl From<f64> for PatternValue {
    fn from(value: f64) -> Self {
        Self::Float(value)
    }
}

#[derive(Debug, Clone)]
pub struct FilenamePattern {
    pattern: String,
    tokens: Vec<Token>,
    fields: Vec<Field>,
    regex: Regex,
}

impl FilenamePattern {
    pub fn new(pattern: impl Into<String>) -> Result<Self> {
        let pattern = pattern.into();
        let mut tokens = parse_tokens(&pattern)?;
        assign_capture_names(&mut tokens);
        let fields = tokens
            .iter()
            .filter_map(|token| match token {
                Token::Field(field) => Some(field.clone()),
                Token::Literal(_) => None,
            })
            .collect::<Vec<_>>();
        let regex = Regex::new(&format!("^{}$", regex_body(&tokens)?)).map_err(|err| {
            RustySatError::invalid_input(format!("invalid filename pattern regex: {err}"))
        })?;

        Ok(Self {
            pattern,
            tokens,
            fields,
            regex,
        })
    }

    pub fn pattern(&self) -> &str {
        &self.pattern
    }

    pub fn keys(&self) -> BTreeSet<String> {
        self.fields.iter().map(|field| field.name.clone()).collect()
    }

    pub fn parse(&self, filename: &str) -> Result<BTreeMap<String, PatternValue>> {
        let captures = self
            .regex
            .captures(filename)
            .ok_or_else(|| RustySatError::not_found("filename matching pattern"))?;
        let mut values = BTreeMap::new();
        let mut raw_values = BTreeMap::<String, String>::new();
        for field in &self.fields {
            let raw = captures
                .name(&field.capture_name)
                .ok_or_else(|| RustySatError::not_found(format!("field '{}'", field.name)))?
                .as_str();
            if let Some(previous) = raw_values.get(&field.name) {
                if previous != raw {
                    return Err(RustySatError::invalid_input(format!(
                        "repeated field '{}' did not match previous value",
                        field.name
                    )));
                }
                continue;
            }
            raw_values.insert(field.name.clone(), raw.to_string());
            values.insert(field.name.clone(), convert_value(raw, field)?);
        }
        Ok(values)
    }

    pub fn validate(&self, filename: &str) -> bool {
        self.parse(filename).is_ok()
    }

    pub fn compose(
        &self,
        values: &BTreeMap<String, PatternValue>,
        allow_partial: bool,
    ) -> Result<String> {
        let mut out = String::new();
        for token in &self.tokens {
            match token {
                Token::Literal(literal) => out.push_str(literal),
                Token::Field(field) => match values.get(&field.name) {
                    Some(value) => out.push_str(&format_value(value, field)?),
                    None if allow_partial => out.push_str(&field.original),
                    None => {
                        return Err(RustySatError::not_found(format!(
                            "compose value for field '{}'",
                            field.name
                        )));
                    }
                },
            }
        }
        Ok(out)
    }

    pub fn globify(&self, values: &BTreeMap<String, PatternValue>) -> Result<String> {
        let mut out = String::new();
        for token in &self.tokens {
            match token {
                Token::Literal(literal) => out.push_str(literal),
                Token::Field(field) => match values.get(&field.name) {
                    Some(value) => out.push_str(&format_value(value, field)?),
                    None => out.push_str(&glob_for_field(field)?),
                },
            }
        }
        Ok(out)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Token {
    Literal(String),
    Field(Field),
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Field {
    name: String,
    spec: String,
    conversion: Option<char>,
    capture_name: String,
    original: String,
}

fn parse_tokens(pattern: &str) -> Result<Vec<Token>> {
    let mut tokens = Vec::new();
    let mut literal = String::new();
    let mut chars = pattern.char_indices().peekable();

    while let Some((idx, ch)) = chars.next() {
        match ch {
            '{' => {
                if matches!(chars.peek(), Some((_, '{'))) {
                    chars.next();
                    literal.push('{');
                    continue;
                }
                flush_literal(&mut tokens, &mut literal);
                let end = pattern[idx + 1..]
                    .find('}')
                    .map(|offset| idx + 1 + offset)
                    .ok_or_else(|| {
                        RustySatError::invalid_input("unclosed filename pattern field")
                    })?;
                let raw = &pattern[idx + 1..end];
                tokens.push(Token::Field(parse_field(raw)?));
                while let Some((next_idx, _)) = chars.peek() {
                    if *next_idx <= end {
                        chars.next();
                    } else {
                        break;
                    }
                }
            }
            '}' => {
                if matches!(chars.peek(), Some((_, '}'))) {
                    chars.next();
                    literal.push('}');
                } else {
                    return Err(RustySatError::invalid_input(
                        "unmatched '}' in filename pattern",
                    ));
                }
            }
            _ => literal.push(ch),
        }
    }
    flush_literal(&mut tokens, &mut literal);
    Ok(tokens)
}

fn flush_literal(tokens: &mut Vec<Token>, literal: &mut String) {
    if !literal.is_empty() {
        tokens.push(Token::Literal(std::mem::take(literal)));
    }
}

fn parse_field(raw: &str) -> Result<Field> {
    let (name_and_conversion, spec) = raw.split_once(':').unwrap_or((raw, ""));
    let (name, conversion) = match name_and_conversion.split_once('!') {
        Some((name, conversion)) => {
            let mut chars = conversion.chars();
            let conversion = chars
                .next()
                .ok_or_else(|| RustySatError::invalid_input("empty filename pattern conversion"))?;
            if chars.next().is_some() {
                return Err(RustySatError::invalid_input(format!(
                    "unsupported filename pattern conversion '{conversion}'"
                )));
            }
            (name, Some(conversion))
        }
        None => (name_and_conversion, None),
    };
    let name = name.trim();
    if name.is_empty() {
        return Err(RustySatError::invalid_input(
            "filename pattern field name cannot be empty",
        ));
    }
    if !is_valid_field_name(name) {
        return Err(RustySatError::invalid_input(format!(
            "unsupported filename pattern field name '{name}'"
        )));
    }
    Ok(Field {
        name: name.to_string(),
        spec: spec.to_string(),
        conversion,
        capture_name: String::new(),
        original: format!("{{{raw}}}"),
    })
}

fn is_valid_field_name(name: &str) -> bool {
    let mut chars = name.chars();
    matches!(chars.next(), Some(ch) if ch == '_' || ch.is_ascii_alphabetic())
        && chars.all(|ch| ch == '_' || ch.is_ascii_alphanumeric())
}

fn assign_capture_names(tokens: &mut [Token]) {
    let mut counts = BTreeMap::<String, usize>::new();
    for token in tokens {
        let Token::Field(field) = token else {
            continue;
        };
        let count = counts.entry(field.name.clone()).or_default();
        field.capture_name = if *count == 0 {
            field.name.clone()
        } else {
            format!("{}__{}", field.name, count)
        };
        *count += 1;
    }
}

fn regex_body(tokens: &[Token]) -> Result<String> {
    let mut body = String::new();
    for token in tokens {
        match token {
            Token::Literal(literal) => body.push_str(&regex::escape(literal)),
            Token::Field(field) => {
                body.push_str(&format!(
                    r"(?P<{}>{})",
                    field.capture_name,
                    regex_for_spec(&field.spec)?
                ));
            }
        }
    }
    Ok(body)
}

fn regex_for_spec(spec: &str) -> Result<String> {
    if spec.is_empty() {
        return Ok(".*?".to_string());
    }
    if spec.contains('%') {
        return datetime_regex(spec);
    }

    let parsed = ParsedSpec::new(spec);
    let kind = parsed.kind;
    let width = parsed.width;
    let base = match kind {
        'd' | 'i' => r"[-+]?\d",
        'f' | 'e' | 'E' | 'g' => return Ok(float_regex(width, parsed.precision)),
        'x' | 'X' => r"[-+]?(0[xX])?[\dA-Fa-f]",
        'o' => r"[-+]?[0-7]",
        'b' => r"[-+]?[0-1]",
        'c' => r".",
        's' => r"\S",
        _ => {
            return Err(RustySatError::invalid_input(format!(
                "unsupported filename pattern format spec '{spec}'"
            )));
        }
    };
    Ok(match width {
        Some(width)
            if parsed.fill.is_some() || matches!(kind, 'd' | 'i' | 'x' | 'X' | 'o' | 'b') =>
        {
            format!(r".{{{width}}}")
        }
        Some(width) => format!("{base}{{{width}}}"),
        _ if matches!(kind, 'd' | 'i' | 'x' | 'X' | 'o' | 'b' | 's' | 'c') => format!("{base}*?"),
        _ => base.to_string(),
    })
}

fn float_regex(width: Option<usize>, precision: Option<usize>) -> String {
    if let Some(width) = width {
        return format!(r".{{{width}}}");
    }
    match precision {
        Some(precision) => {
            format!(r"[-+]?(\d+(\.\d{{{precision}}})|\.\d{{{precision}}})([eE][-+]?\d+)?")
        }
        None => r"[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?".to_string(),
    }
}

fn datetime_regex(spec: &str) -> Result<String> {
    let mut out = spec.to_string();
    for (key, value) in datetime_token_map() {
        out = out.replace(key, value);
    }
    Ok(out)
}

fn glob_for_field(field: &Field) -> Result<String> {
    let spec = field.spec.as_str();
    if spec.is_empty() {
        return Ok("*".to_string());
    }
    if spec.contains('%') {
        let mut out = spec.to_string();
        for (key, value) in datetime_glob_map() {
            out = out.replace(key, value);
        }
        return Ok(out);
    }
    Ok(match ParsedSpec::new(spec).width {
        Some(width) => "?".repeat(width),
        None => "*".to_string(),
    })
}

fn convert_value(raw: &str, field: &Field) -> Result<PatternValue> {
    let spec = field.spec.as_str();
    let parsed = ParsedSpec::new(spec);
    let kind = parsed.kind;
    if spec.contains('%') {
        return Ok(parse_datetime_value(raw, spec)
            .map(PatternValue::DateTime)
            .unwrap_or_else(|| PatternValue::Text(raw.to_string())));
    }
    if spec.is_empty() || matches!(kind, 's' | 'c') {
        return Ok(PatternValue::Text(raw.to_string()));
    }
    if matches!(kind, 'd' | 'i') {
        return parse_integer(raw, 10, field);
    }
    if matches!(kind, 'x' | 'X') {
        return parse_integer(raw, 16, field);
    }
    if kind == 'o' {
        return parse_integer(raw, 8, field);
    }
    if kind == 'b' {
        return parse_integer(raw, 2, field);
    }
    if matches!(kind, 'f' | 'e' | 'E' | 'g') {
        return cleaned_number_text(raw, &parsed)
            .parse::<f64>()
            .map(PatternValue::Float)
            .map_err(|err| {
                RustySatError::invalid_input(format!("invalid float field '{}': {err}", field.name))
            });
    }
    Ok(PatternValue::Text(raw.to_string()))
}

fn format_value(value: &PatternValue, field: &Field) -> Result<String> {
    if field.spec.contains('%') {
        if let PatternValue::DateTime(datetime) = value {
            return format_datetime_value(datetime, &field.spec);
        }
    }
    let converted = apply_conversion(value.as_compose_text(), field.conversion)?;
    let parsed = ParsedSpec::new(&field.spec);
    if let Some(width) = parsed.width {
        if matches!(parsed.kind, 'd' | 'i') {
            let number = converted.parse::<i64>().unwrap_or_default();
            let fill = parsed.fill.unwrap_or(if parsed.zero { '0' } else { ' ' });
            return Ok(pad_left(&number.to_string(), width, fill));
        }
    }
    Ok(converted)
}

fn parse_datetime_value(raw: &str, spec: &str) -> Option<PatternDateTime> {
    let mut raw_pos = 0;
    let mut year = 1900_i32;
    let mut month = 1_u8;
    let mut day = 1_u8;
    let mut hour = 0_u8;
    let mut minute = 0_u8;
    let mut second = 0_u8;
    let mut ordinal_day = None;
    let mut chars = spec.chars().peekable();

    while let Some(ch) = chars.next() {
        if ch != '%' {
            if raw[raw_pos..].chars().next()? != ch {
                return None;
            }
            raw_pos += ch.len_utf8();
            continue;
        }
        let directive = chars.next()?;
        match directive {
            'Y' => year = read_fixed_i32(raw, &mut raw_pos, 4)?,
            'y' => year = 2000 + read_fixed_i32(raw, &mut raw_pos, 2)?,
            'm' => month = read_fixed_u8(raw, &mut raw_pos, 2)?,
            'd' => day = read_fixed_u8(raw, &mut raw_pos, 2)?,
            'H' => hour = read_fixed_u8(raw, &mut raw_pos, 2)?,
            'M' => minute = read_fixed_u8(raw, &mut raw_pos, 2)?,
            'S' => second = read_fixed_u8(raw, &mut raw_pos, 2)?,
            'j' => ordinal_day = Some(read_fixed_u16(raw, &mut raw_pos, 3)?),
            '%' => {
                if raw[raw_pos..].chars().next()? != '%' {
                    return None;
                }
                raw_pos += 1;
            }
            _ => return None,
        }
    }
    if raw_pos != raw.len() {
        return None;
    }
    let datetime = PatternDateTime::new(year, month, day, hour, minute, second).ok()?;
    match ordinal_day {
        Some(day) => datetime.with_ordinal_day(day).ok(),
        None => Some(datetime),
    }
}

fn format_datetime_value(datetime: &PatternDateTime, spec: &str) -> Result<String> {
    let mut out = String::new();
    let mut chars = spec.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch != '%' {
            out.push(ch);
            continue;
        }
        let directive = chars
            .next()
            .ok_or_else(|| RustySatError::invalid_input("dangling datetime format '%'"))?;
        match directive {
            'Y' => out.push_str(&format!("{:04}", datetime.year)),
            'y' => out.push_str(&format!("{:02}", datetime.year.rem_euclid(100))),
            'm' => out.push_str(&format!("{:02}", datetime.month)),
            'd' => out.push_str(&format!("{:02}", datetime.day)),
            'H' => out.push_str(&format!("{:02}", datetime.hour)),
            'M' => out.push_str(&format!("{:02}", datetime.minute)),
            'S' => out.push_str(&format!("{:02}", datetime.second)),
            'j' => out.push_str(&format!("{:03}", datetime.ordinal_day.unwrap_or(1))),
            '%' => out.push('%'),
            other => {
                return Err(RustySatError::unsupported(format!(
                    "datetime format '%{other}'"
                )));
            }
        }
    }
    Ok(out)
}

fn read_fixed_i32(raw: &str, raw_pos: &mut usize, width: usize) -> Option<i32> {
    read_fixed(raw, raw_pos, width)?.parse().ok()
}

fn read_fixed_u8(raw: &str, raw_pos: &mut usize, width: usize) -> Option<u8> {
    read_fixed(raw, raw_pos, width)?.parse().ok()
}

fn read_fixed_u16(raw: &str, raw_pos: &mut usize, width: usize) -> Option<u16> {
    read_fixed(raw, raw_pos, width)?.parse().ok()
}

fn read_fixed<'a>(raw: &'a str, raw_pos: &mut usize, width: usize) -> Option<&'a str> {
    let end = *raw_pos + width;
    let value = raw.get(*raw_pos..end)?;
    value.chars().all(|ch| ch.is_ascii_digit()).then_some(())?;
    *raw_pos = end;
    Some(value)
}

fn parse_integer(raw: &str, radix: u32, field: &Field) -> Result<PatternValue> {
    let parsed = ParsedSpec::new(&field.spec);
    let cleaned = cleaned_number_text(raw, &parsed);
    let cleaned = cleaned.trim_start_matches("0x").trim_start_matches("0X");
    i64::from_str_radix(cleaned, radix)
        .map(PatternValue::Integer)
        .map_err(|err| {
            RustySatError::invalid_input(format!("invalid integer field '{}': {err}", field.name))
        })
}

fn cleaned_number_text<'a>(raw: &'a str, parsed: &ParsedSpec) -> String {
    let trimmed = raw.trim();
    match parsed.fill {
        Some(fill) => trimmed.trim_matches(fill).trim().to_string(),
        None => trimmed.to_string(),
    }
}

fn apply_conversion(mut value: String, conversion: Option<char>) -> Result<String> {
    match conversion {
        None | Some('s') => {}
        Some('c') => {
            let mut chars = value.chars();
            value = match chars.next() {
                Some(first) => {
                    first.to_uppercase().collect::<String>() + &chars.as_str().to_lowercase()
                }
                None => value,
            };
        }
        Some('l') => value = value.to_lowercase(),
        Some('t') => {
            value = value
                .split(' ')
                .map(|word| {
                    let mut chars = word.chars();
                    match chars.next() {
                        Some(first) => {
                            first.to_uppercase().collect::<String>()
                                + &chars.as_str().to_lowercase()
                        }
                        None => String::new(),
                    }
                })
                .collect::<Vec<_>>()
                .join(" ");
        }
        Some('u') => value = value.to_uppercase(),
        Some('h') => value = remove_separators(&value.to_lowercase()),
        Some('H') => value = remove_separators(&value.to_uppercase()),
        Some('R') => value = remove_separators(&value),
        Some('r') => value = format!("'{value}'"),
        Some(other) => {
            return Err(RustySatError::invalid_input(format!(
                "unsupported filename pattern conversion '{other}'"
            )));
        }
    }
    Ok(value)
}

fn remove_separators(value: &str) -> String {
    value
        .chars()
        .filter(|ch| !matches!(ch, '-' | '_' | ':' | ' '))
        .collect()
}

fn pad_left(value: &str, width: usize, fill: char) -> String {
    if value.len() >= width {
        return value.to_string();
    }
    format!("{}{}", fill.to_string().repeat(width - value.len()), value)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ParsedSpec {
    fill: Option<char>,
    zero: bool,
    width: Option<usize>,
    precision: Option<usize>,
    kind: char,
}

impl ParsedSpec {
    fn new(spec: &str) -> Self {
        let kind = spec
            .chars()
            .last()
            .filter(|ch| ch.is_ascii_alphabetic())
            .unwrap_or('s');
        let (fill, zero) = parse_fill(spec);
        let width = parse_width(spec);
        let precision = parse_precision(spec);
        Self {
            fill,
            zero,
            width,
            precision,
            kind,
        }
    }
}

fn parse_fill(spec: &str) -> (Option<char>, bool) {
    let mut chars = spec.chars();
    let first = chars.next();
    let second = chars.next();
    if matches!(second, Some('<' | '>' | '=' | '^')) {
        return (first, false);
    }
    (None, spec.starts_with('0'))
}

fn parse_width(spec: &str) -> Option<usize> {
    let mut digits = String::new();
    let mut started = false;
    for ch in spec.chars() {
        if ch == '.' {
            break;
        }
        if ch.is_ascii_digit() {
            started = true;
            digits.push(ch);
        } else if started {
            break;
        }
    }
    (!digits.is_empty()).then(|| digits.parse().ok()).flatten()
}

fn parse_precision(spec: &str) -> Option<usize> {
    let (_, precision) = spec.split_once('.')?;
    let digits = precision
        .chars()
        .take_while(|ch| ch.is_ascii_digit())
        .collect::<String>();
    (!digits.is_empty()).then(|| digits.parse().ok()).flatten()
}

fn datetime_token_map() -> &'static [(&'static str, &'static str)] {
    &[
        ("%Y", r"\d{4}"),
        ("%y", r"\d{2}"),
        ("%m", r"\d{2}"),
        ("%d", r"\d{2}"),
        ("%H", r"\d{2}"),
        ("%M", r"\d{2}"),
        ("%S", r"\d{2}"),
        ("%j", r"\d{3}"),
        ("%f", r"[^ \t\n\r\f\v\-_:]+"),
        ("%z", r"[^ \t\n\r\f\v\-_:]+"),
        ("%Z", r"[^ \t\n\r\f\v\-_:]+"),
        ("%%", "%"),
    ]
}

fn datetime_glob_map() -> &'static [(&'static str, &'static str)] {
    &[
        ("%Y", "????"),
        ("%y", "??"),
        ("%m", "??"),
        ("%d", "??"),
        ("%H", "??"),
        ("%M", "??"),
        ("%S", "??"),
        ("%j", "???"),
        ("%f", "*"),
        ("%z", "*"),
        ("%Z", "*"),
        ("%%", "?"),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_trollsift_usage_example() {
        let parser = FilenamePattern::new(
            "/somedir/{directory}/hrpt_{platform_name}_{start_time:%Y%m%d_%H%M}_{orbit:05d}.l1b",
        )
        .unwrap();
        let values = parser
            .parse("/somedir/otherdir/hrpt_noaa16_20140210_1004_69022.l1b")
            .unwrap();

        assert_eq!(
            values["directory"],
            PatternValue::Text("otherdir".to_string())
        );
        assert_eq!(
            values["platform_name"],
            PatternValue::Text("noaa16".to_string())
        );
        assert_eq!(
            values["start_time"],
            PatternValue::DateTime(PatternDateTime::new(2014, 2, 10, 10, 4, 0).unwrap())
        );
        assert_eq!(values["orbit"], PatternValue::Integer(69022));
    }

    #[test]
    fn string_fields_are_non_greedy_like_trollsift() {
        let parser = FilenamePattern::new("{field_one}_{field_two}").unwrap();
        let values = parser.parse("abc_def_ghi").unwrap();

        assert_eq!(values["field_one"], PatternValue::Text("abc".to_string()));
        assert_eq!(
            values["field_two"],
            PatternValue::Text("def_ghi".to_string())
        );
    }

    #[test]
    fn validates_and_rejects_full_match_failures() {
        let parser = FilenamePattern::new("A_{channel:3s}_{segment:3d}.dat").unwrap();

        assert!(parser.validate("A_VIS_007.dat"));
        assert!(!parser.validate("prefix_A_VIS_007.dat"));
        assert!(parser.parse("A_VIS_007.dat").is_ok());
        assert!(parser.parse("A_VIS_bad.dat").is_err());
    }

    #[test]
    fn compose_supports_strict_and_partial_modes() {
        let parser = FilenamePattern::new("{platform}_{orbit:05d}_{product}.dat").unwrap();
        let values = BTreeMap::from([
            ("platform".to_string(), PatternValue::from("noaa19")),
            ("orbit".to_string(), PatternValue::from(42_i64)),
        ]);

        assert!(parser.compose(&values, false).is_err());
        assert_eq!(
            parser.compose(&values, true).unwrap(),
            "noaa19_00042_{product}.dat"
        );
    }

    #[test]
    fn globify_matches_common_trollsift_patterns() {
        let parser =
            FilenamePattern::new("{a}_{start_time:%Y%m%d_%H%M}_{orbit:05d}_{kind}.dat").unwrap();
        let values = BTreeMap::from([("a".to_string(), PatternValue::from("hrpt"))]);

        assert_eq!(
            parser.globify(&values).unwrap(),
            "hrpt_????????_????_?????_*.dat"
        );
    }

    #[test]
    fn repeated_fields_must_match_same_text() {
        let parser = FilenamePattern::new("{platform}_{platform}_{orbit:03d}").unwrap();

        assert!(parser.validate("npp_npp_001"));
        assert!(!parser.validate("npp_noaa20_001"));
    }

    #[test]
    fn compose_supports_trollsift_string_conversions() {
        let value = PatternValue::from("this Is A-Test b_test c test");
        let values = BTreeMap::from([("a".to_string(), value)]);

        assert_eq!(
            FilenamePattern::new("{a!c}")
                .unwrap()
                .compose(&values, false)
                .unwrap(),
            "This is a-test b_test c test"
        );
        assert_eq!(
            FilenamePattern::new("{a!h}")
                .unwrap()
                .compose(&values, false)
                .unwrap(),
            "thisisatestbtestctest"
        );
        assert_eq!(
            FilenamePattern::new("{a!H}")
                .unwrap()
                .compose(&values, false)
                .unwrap(),
            "THISISATESTBTESTCTEST"
        );
        assert_eq!(
            FilenamePattern::new("{a!R}")
                .unwrap()
                .compose(&values, false)
                .unwrap(),
            "thisIsATestbtestctest"
        );
        assert_eq!(
            FilenamePattern::new("{a!u}")
                .unwrap()
                .compose(&values, false)
                .unwrap(),
            "THIS IS A-TEST B_TEST C TEST"
        );
        assert_eq!(
            FilenamePattern::new("{a!r}")
                .unwrap()
                .compose(&values, false)
                .unwrap(),
            "'this Is A-Test b_test c test'"
        );
    }

    #[test]
    fn parses_trollsift_integer_radices_and_padding() {
        for (fmt, string) in [
            ("{foo:x}", "7b"),
            ("{foo:X}", "7B"),
            ("{foo:03x}", "07b"),
            ("{foo:3x}", " 7b"),
            ("{foo:o}", "173"),
            ("{foo:_>4o}", "_173"),
            ("{foo:b}", "1111011"),
            ("{foo:8b}", " 1111011"),
        ] {
            let parser = FilenamePattern::new(fmt).unwrap();
            assert_eq!(
                parser.parse(string).unwrap()["foo"],
                PatternValue::Integer(123)
            );
        }
    }

    #[test]
    fn parses_trollsift_fixed_point_examples() {
        for (fmt, string, expected) in [
            ("{foo:f}", "12.34", 12.34),
            ("{foo:5.2f}", "-1.23", -1.23),
            ("{foo:5.2f}", " 1.23", 1.23),
            ("{foo:05.2f}", "01.23", 1.23),
            ("{foo:.2f}", "12.34", 12.34),
            ("{foo:4.2f}", "-.12", -0.12),
            ("{foo:7.2e}", "-1.23e4", -1.23e4),
        ] {
            let parser = FilenamePattern::new(fmt).unwrap();
            assert_eq!(
                parser.parse(string).unwrap()["foo"],
                PatternValue::Float(expected)
            );
        }
    }

    #[test]
    fn parses_and_composes_common_datetime_fields() {
        let parser = FilenamePattern::new("{start_time:%Y%m%d_%H%M%S}_{day:%Y%j}").unwrap();
        let values = parser.parse("20260502_123456_2026123").unwrap();

        assert_eq!(
            values["start_time"],
            PatternValue::DateTime(PatternDateTime::new(2026, 5, 2, 12, 34, 56).unwrap())
        );
        assert_eq!(
            values["day"],
            PatternValue::DateTime(
                PatternDateTime::new(2026, 1, 1, 0, 0, 0)
                    .unwrap()
                    .with_ordinal_day(123)
                    .unwrap()
            )
        );

        assert_eq!(
            parser.compose(&values, false).unwrap(),
            "20260502_123456_2026123"
        );
    }

    #[test]
    fn partial_compose_handles_similarly_named_fields() {
        let parser = FilenamePattern::new("{foo}{afooo}{fooo}.{bar}/{baz:%Y}/{bar:d}").unwrap();
        let values = BTreeMap::from([("afooo".to_string(), PatternValue::from("qux"))]);

        assert_eq!(
            parser.compose(&values, true).unwrap(),
            "{foo}qux{fooo}.{bar}/{baz:%Y}/{bar:d}"
        );
    }

    #[test]
    fn partial_compose_allows_repeated_fields_with_different_formats() {
        let parser =
            FilenamePattern::new("/foo/{start_time:%Y%m}/bar/{start_time:%Y%m%d_%H%M}.{format}")
                .unwrap();
        let values = BTreeMap::from([("format".to_string(), PatternValue::from("qux"))]);

        assert_eq!(
            parser.compose(&values, true).unwrap(),
            "/foo/{start_time:%Y%m}/bar/{start_time:%Y%m%d_%H%M}.qux"
        );
    }

    #[test]
    fn compose_rejects_unsupported_conversion() {
        let parser = FilenamePattern::new("{a!X}").unwrap();
        let values = BTreeMap::from([("a".to_string(), PatternValue::from("value"))]);

        assert!(parser.compose(&values, false).is_err());
    }
}
