/// Python-compatible JSON serialisation.
///
/// Laya serialises JSON states with `json.dumps(..., ensure_ascii=False)`
/// before tokenising, so the text must match Python byte for byte.
library;

/// Python's `json.dumps(value, ensure_ascii=False)` with the given separators.
///
/// Supports `null`, `bool`, `int`, `double`, `String`, `List` and `Map`.
/// Map keys that are not strings are converted with `toString()`. Throws
/// [ArgumentError] for any other value.
String pythonJsonDumps(
  Object? value, {
  String itemSeparator = ', ',
  String keySeparator = ': ',
}) {
  final out = StringBuffer();
  _write(out, value, itemSeparator, keySeparator);
  return out.toString();
}

void _write(StringBuffer out, Object? value, String item, String key) {
  if (value == null) {
    out.write('null');
  } else if (value is bool) {
    out.write(value ? 'true' : 'false');
  } else if (value is int) {
    out.write(value.toString());
  } else if (value is double) {
    out.write(pythonFloatRepr(value));
  } else if (value is String) {
    _writeString(out, value);
  } else if (value is List) {
    out.write('[');
    for (var i = 0; i < value.length; i++) {
      if (i > 0) out.write(item);
      _write(out, value[i], item, key);
    }
    out.write(']');
  } else if (value is Map) {
    out.write('{');
    var first = true;
    for (final entry in value.entries) {
      if (!first) out.write(item);
      first = false;
      final k = entry.key;
      _writeString(out, k is String ? k : k.toString());
      out.write(key);
      _write(out, entry.value, item, key);
    }
    out.write('}');
  } else {
    throw ArgumentError.value(
      value,
      'value',
      'Unsupported type ${value.runtimeType}',
    );
  }
}

void _writeString(StringBuffer out, String s) {
  out.write('"');
  for (final unit in s.codeUnits) {
    switch (unit) {
      case 0x22:
        out.write(r'\"');
      case 0x5c:
        out.write(r'\\');
      case 0x0a:
        out.write(r'\n');
      case 0x0d:
        out.write(r'\r');
      case 0x09:
        out.write(r'\t');
      case 0x08:
        out.write(r'\b');
      case 0x0c:
        out.write(r'\f');
      default:
        if (unit < 0x20) {
          out.write('\\u${unit.toRadixString(16).padLeft(4, '0')}');
        } else {
          out.writeCharCode(unit);
        }
    }
  }
  out.write('"');
}

/// Python's `repr(float)` / `json.dumps` text for a double.
///
/// Uses the shortest round-trip digits. Magnitudes in `[1e-4, 1e16)` print
/// positionally with at least one fractional digit; others print in
/// scientific notation with a signed exponent of at least two digits.
String pythonFloatRepr(double value) {
  if (value.isNaN) return 'NaN';
  if (value.isInfinite) return value < 0 ? '-Infinity' : 'Infinity';
  if (value == 0) return value.isNegative ? '-0.0' : '0.0';

  final negative = value < 0;
  final text = value.abs().toString();
  final eIndex = text.indexOf('e');
  final mantissa = eIndex < 0 ? text : text.substring(0, eIndex);
  final exp = eIndex < 0 ? 0 : int.parse(text.substring(eIndex + 1));

  final dot = mantissa.indexOf('.');
  var digits = mantissa.replaceFirst('.', '');
  // Decimal point position relative to the start of [digits].
  var point = (dot < 0 ? mantissa.length : dot) + exp;
  // Strip leading zeros, adjusting the point.
  final lead = digits.length - digits.replaceFirst(RegExp(r'^0+'), '').length;
  digits = digits.substring(lead);
  point -= lead;
  digits = digits.replaceFirst(RegExp(r'0+$'), '');
  if (digits.isEmpty) digits = '0';

  // Scientific exponent: value = d.ddd * 10^sciExp.
  final sciExp = point - 1;
  final String body;
  if (sciExp >= -4 && sciExp < 16) {
    if (point <= 0) {
      body = '0.${'0' * -point}$digits';
    } else if (point >= digits.length) {
      body = '${digits.padRight(point, '0')}.0';
    } else {
      body = '${digits.substring(0, point)}.${digits.substring(point)}';
    }
  } else {
    final m = digits.length == 1
        ? digits
        : '${digits[0]}.${digits.substring(1)}';
    final sign = sciExp < 0 ? '-' : '+';
    final e = sciExp.abs().toString().padLeft(2, '0');
    body = '${m}e$sign$e';
  }
  return negative ? '-$body' : body;
}
