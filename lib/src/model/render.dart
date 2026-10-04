/// Option rendering, ported from laya's `common.py`.
library;

import '../json/python_json.dart';
import '../python/py_unicode.dart';

/// laya `render_criterion`: strings pass through, else compact Python JSON.
String renderCriterion(Object? value) =>
    value is String ? value : pythonJsonDumps(value);

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
  // Python `str.strip()`, not Dart's trim: they differ on U+001C-U+001F
  // and U+FEFF.
  final fs = pyStripStr(f);
  final ts = pyStripStr(t);
  if (fs.isEmpty || ts.isEmpty || fs == ts) return null;
  return (fs, ts);
}

/// The message of laya's noul labels `ValueError`.
const noulLabelsMessage =
    "noul labels must map exactly 'false' and 'true' to distinct non-empty strings";
