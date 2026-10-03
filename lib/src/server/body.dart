/// Request-body decoding as laya-serve does it: Python's `json.loads(bytes)`.
library;

import 'dart:convert';
import 'dart:typed_data';

import 'errors.dart';

/// The deepest nesting of objects and arrays a body may have.
///
/// CPython's JSON scanner refuses deeper input with a `RecursionError`,
/// which laya-serve answers as invalid JSON. Its ceiling is the C recursion
/// budget left at the call, so it depends on the stack it runs on: in
/// laya-serve's app under Starlette's `TestClient` (CPython 3.12) a state
/// nested 2992 deep parses and 2993 does not, so a body (one level more)
/// parses up to 2993. `test/fixtures/serve.json` pins this.
const maxJsonDepth = 2993;

const _invalidJson = 'request body must be valid JSON';

/// A decoded request body.
typedef DecodedBody = ({
  /// The JSON value.
  Object? json,

  /// Whether the text held a surrogate code point decoded from the bytes
  /// themselves (Python's `surrogatepass`), which Python keeps as a lone
  /// surrogate even where two of them would pair up in a Dart string.
  bool rawSurrogate,
});

/// Python's `json.loads(bytes)`, as laya-serve's `_read_json_body` calls it.
///
/// The encoding is detected as Python's `json.detect_encoding` does
/// (UTF-8, UTF-8 with a BOM, UTF-16 or UTF-32, either byte order) and decoded
/// with `surrogatepass`. Throws [ApiException] 400 `request body must be
/// valid JSON` for bytes that do not decode or parse, or nest deeper than
/// [maxJsonDepth].
///
/// Two differences from Python remain: the non-standard `NaN` and
/// `Infinity` literals are refused, and integers beyond 64 bits decode as
/// doubles (Dart's `jsonDecode`).
DecodedBody decodeJsonBody(Uint8List bytes) {
  final (String text, bool raw) decoded;
  try {
    decoded = _decodeText(bytes);
  } on FormatException {
    throw const ApiException(400, _invalidJson);
  }
  final Object? json;
  try {
    json = jsonDecode(decoded.$1);
  } on FormatException {
    throw const ApiException(400, _invalidJson);
  }
  if (_depthOver(json, maxJsonDepth)) {
    throw const ApiException(400, _invalidJson);
  }
  return (json: json, rawSurrogate: decoded.$2);
}

/// Whether [value] nests objects and arrays deeper than [limit].
bool _depthOver(Object? value, int limit) {
  final stack = <(Object?, int)>[(value, 1)];
  while (stack.isNotEmpty) {
    final (item, depth) = stack.removeLast();
    final Iterable<Object?> children;
    if (item is Map) {
      children = item.values;
    } else if (item is List) {
      children = item;
    } else {
      continue;
    }
    if (depth > limit) return true;
    for (final child in children) {
      if (child is Map || child is List) stack.add((child, depth + 1));
    }
  }
  return false;
}

/// `json.detect_encoding` then `bytes.decode(encoding, 'surrogatepass')`.
/// Returns the text and whether it holds a surrogate code point.
(String, bool) _decodeText(Uint8List b) {
  bool startsWith(List<int> prefix) {
    if (b.length < prefix.length) return false;
    for (var i = 0; i < prefix.length; i++) {
      if (b[i] != prefix[i]) return false;
    }
    return true;
  }

  if (startsWith(const [0xff, 0xfe, 0, 0])) return _utf32(b, 4, Endian.little);
  if (startsWith(const [0, 0, 0xfe, 0xff])) return _utf32(b, 4, Endian.big);
  if (startsWith(const [0xff, 0xfe])) return _utf16(b, 2, Endian.little);
  if (startsWith(const [0xfe, 0xff])) return _utf16(b, 2, Endian.big);
  if (startsWith(const [0xef, 0xbb, 0xbf])) return _utf8(b, 3);
  if (b.length >= 4) {
    if (b[0] == 0) {
      return b[1] != 0 ? _utf16(b, 0, Endian.big) : _utf32(b, 0, Endian.big);
    }
    if (b[1] == 0) {
      return b[2] != 0 || b[3] != 0
          ? _utf16(b, 0, Endian.little)
          : _utf32(b, 0, Endian.little);
    }
  } else if (b.length == 2) {
    if (b[0] == 0) return _utf16(b, 0, Endian.big);
    if (b[1] == 0) return _utf16(b, 0, Endian.little);
  }
  return _utf8(b, 0);
}

bool _isSurrogate(int unit) => unit >= 0xd800 && unit <= 0xdfff;

/// UTF-16 from [start]: code units pass straight through, so a pair is one
/// character and an unpaired surrogate stays one (`surrogatepass`). An odd
/// byte count is an error.
(String, bool) _utf16(Uint8List b, int start, Endian endian) {
  final n = b.length - start;
  if (n.isOdd) throw const FormatException('truncated data');
  final data = ByteData.sublistView(b, start);
  final units = Uint16List(n ~/ 2);
  for (var i = 0; i < units.length; i++) {
    units[i] = data.getUint16(i * 2, endian);
  }
  final text = String.fromCharCodes(units);
  return (text, hasUnpairedSurrogate(text));
}

/// Whether [s] holds a UTF-16 surrogate that is not half of a pair.
bool hasUnpairedSurrogate(String s) {
  for (var i = 0; i < s.length; i++) {
    final u = s.codeUnitAt(i);
    if (u < 0xd800 || u > 0xdfff) continue;
    if (u <= 0xdbff && i + 1 < s.length) {
      final next = s.codeUnitAt(i + 1);
      if (next >= 0xdc00 && next <= 0xdfff) {
        i++;
        continue;
      }
    }
    return true;
  }
  return false;
}

/// UTF-32 from [start]: a surrogate code point passes through as one code
/// unit; anything past U+10FFFF, or a partial unit, is an error.
(String, bool) _utf32(Uint8List b, int start, Endian endian) {
  final n = b.length - start;
  if (n % 4 != 0) throw const FormatException('truncated data');
  final data = ByteData.sublistView(b, start);
  final out = StringBuffer();
  var surrogate = false;
  for (var i = 0; i < n; i += 4) {
    final cp = data.getUint32(i, endian);
    if (cp > 0x10ffff) throw const FormatException('code point not in range');
    if (_isSurrogate(cp)) surrogate = true;
    out.writeCharCode(cp);
  }
  return (out.toString(), surrogate);
}

/// UTF-8 from [start], strict except that an encoded surrogate (`ED A0..BF
/// xx`) decodes to that surrogate (`surrogatepass`).
(String, bool) _utf8(Uint8List b, int start) {
  final body = Uint8List.sublistView(b, start);
  // Dart's decoder drops a leading BOM, which Python keeps after the one
  // `utf-8-sig` strips; and it refuses encoded surrogates. Take the fast
  // path only when neither can matter.
  final bom =
      body.length >= 3 && body[0] == 0xef && body[1] == 0xbb && body[2] == 0xbf;
  if (!bom) {
    try {
      return (utf8.decode(body), false);
    } on FormatException {
      // Fall through: it may hold encoded surrogates.
    }
  }
  final units = <int>[];
  var surrogate = false;
  var i = 0;
  int cont(int at) {
    if (at >= body.length || body[at] & 0xc0 != 0x80) {
      throw const FormatException('invalid continuation byte');
    }
    return body[at] & 0x3f;
  }

  while (i < body.length) {
    final c = body[i];
    int cp;
    if (c < 0x80) {
      cp = c;
      i += 1;
    } else if (c >= 0xc2 && c <= 0xdf) {
      cp = (c & 0x1f) << 6 | cont(i + 1);
      i += 2;
    } else if (c >= 0xe0 && c <= 0xef) {
      final c1 = i + 1 < body.length ? body[i + 1] : -1;
      // Overlong (E0 80..9F) is an error; ED A0..BF is a surrogate, which
      // surrogatepass lets through.
      if (c == 0xe0 && c1 < 0xa0) {
        throw const FormatException('invalid continuation byte');
      }
      cp = (c & 0x0f) << 12 | cont(i + 1) << 6 | cont(i + 2);
      if (_isSurrogate(cp)) surrogate = true;
      i += 3;
    } else if (c >= 0xf0 && c <= 0xf4) {
      final c1 = i + 1 < body.length ? body[i + 1] : -1;
      if ((c == 0xf0 && c1 < 0x90) || (c == 0xf4 && c1 > 0x8f)) {
        throw const FormatException('invalid continuation byte');
      }
      cp =
          (c & 0x07) << 18 | cont(i + 1) << 12 | cont(i + 2) << 6 | cont(i + 3);
      i += 4;
    } else {
      throw const FormatException('invalid start byte');
    }
    if (cp >= 0x10000) {
      cp -= 0x10000;
      units
        ..add(0xd800 | cp >> 10)
        ..add(0xdc00 | cp & 0x3ff);
    } else {
      units.add(cp);
    }
  }
  return (String.fromCharCodes(units), surrogate);
}
