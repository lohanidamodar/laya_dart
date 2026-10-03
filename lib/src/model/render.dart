/// Option rendering, ported from laya's `common.py`.
library;

import '../json/python_json.dart';

/// laya `render_criterion`: strings pass through, else compact Python JSON.
String renderCriterion(Object? value) =>
    value is String ? value : pythonJsonDumps(value);

/// Python's `str()` of a JSON scalar (`None`, `True`, `1`, `2.5`, text).
String pyStr(Object? value) => switch (value) {
  null => 'None',
  true => 'True',
  false => 'False',
  final String s => s,
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
    } else if (r < 0x20 || r == 0x7f) {
      out.write('\\x${r.toRadixString(16).padLeft(2, '0')}');
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

/// laya `_resolve_noul_labels`: the (false, true) option words.
///
/// Returns null when [labels] is invalid; the caller supplies laya's message.
(String, String)? resolveNoulLabels(Object? labels) {
  if (labels == null) return ('false', 'true');
  if (labels is! Map ||
      labels.length != 2 ||
      !labels.containsKey('false') ||
      !labels.containsKey('true')) {
    return null;
  }
  final f = labels['false'];
  final t = labels['true'];
  if (f is! String || t is! String) return null;
  final fs = f.trim();
  final ts = t.trim();
  if (fs.isEmpty || ts.isEmpty || fs == ts) return null;
  return (fs, ts);
}

/// The message of laya's noul labels `ValueError`.
const noulLabelsMessage =
    "noul labels must map exactly 'false' and 'true' to distinct non-empty strings";
