/// Python's `str()`, `repr()` and `type().__name__` of decoded JSON values,
/// as laya's error messages print them.
library;

import '../json/python_json.dart';
import 'py_unicode.dart';

/// Python's `str()` of a JSON scalar (`None`, `True`, `1`, `2.5`, `inf`,
/// text).
String pyStr(Object? value) => switch (value) {
  null => 'None',
  true => 'True',
  false => 'False',
  final String s => s,
  final double d when d.isNaN => 'nan',
  final double d when d.isInfinite => d.isNegative ? '-inf' : 'inf',
  final double d => pythonFloatRepr(d),
  _ => '$value',
};

/// Python's `repr()` of a decoded JSON value, as `%r` prints it.
String pyRepr(Object? value) {
  switch (value) {
    case final String s:
      return _strRepr(s);
    case final List<Object?> l:
      return '[${l.map(pyRepr).join(', ')}]';
    case final Map<Object?, Object?> m:
      return '{${m.entries.map((e) => '${pyRepr(e.key)}: ${pyRepr(e.value)}').join(', ')}}';
    default:
      return pyStr(value);
  }
}

String _strRepr(String s) {
  final quote = s.contains("'") && !s.contains('"') ? '"' : "'";
  final out = StringBuffer(quote);
  for (final r in s.runes) {
    final ch = String.fromCharCode(r);
    if (ch == quote || ch == r'\') {
      out.write('\\$ch');
    } else if (ch == '\n') {
      out.write(r'\n');
    } else if (ch == '\r') {
      out.write(r'\r');
    } else if (ch == '\t') {
      out.write(r'\t');
    } else if (r < 0x20 || r == 0x7f || (r > 0x7f && !pyIsPrintable(r))) {
      // Python escapes a non-printable code point by its width.
      final hex = r.toRadixString(16);
      out.write(
        r < 0x100
            ? '\\x${hex.padLeft(2, '0')}'
            : r < 0x10000
            ? '\\u${hex.padLeft(4, '0')}'
            : '\\U${hex.padLeft(8, '0')}',
      );
    } else {
      out.write(ch);
    }
  }
  out.write(quote);
  return out.toString();
}

/// Python type name of a decoded JSON value, as `type(v).__name__` prints it.
String pyTypeName(Object? value) => switch (value) {
  null => 'NoneType',
  bool() => 'bool',
  int() => 'int',
  double() => 'float',
  String() => 'str',
  List<Object?>() => 'list',
  Map<Object?, Object?>() => 'dict',
  _ => value.runtimeType.toString(),
};
