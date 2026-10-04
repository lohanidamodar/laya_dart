/// Python 3.12's character behaviour (Unicode 15.0.0) for the `str`, `re` and
/// `unicodedata` calls laya's `lang.py` and `%r` make.
///
/// Dart's own `RegExp` `\p{...}` and `toLowerCase` follow a different Unicode
/// version and different rules, so every predicate here reads the tables
/// `tool/gen_nfc.dart` generates from the pinned UCD, using the flag
/// derivations of CPython's `makeunicodedata.py`. Text is a list of code
/// points, as Python `str` indexes it. `test/fixtures/unicode_props.json`
/// checks every code point against the reference.
library;

import 'dart:typed_data';

import 'unicode_data.g.dart';

/// The Unicode version of these tables (Python 3.12's
/// `unicodedata.unidata_version`).
const String pyUnicodeDataVersion = pyUnicodeVersion;

final _FlagTable _flags = _FlagTable(pyFlagRuns);

class _FlagTable {
  _FlagTable(String runs) {
    final entries = runs.split(' ');
    starts = Int32List(entries.length);
    flags = Uint16List(entries.length);
    for (final (i, e) in entries.indexed) {
      final colon = e.indexOf(':');
      starts[i] = int.parse(e.substring(0, colon), radix: 16);
      flags[i] = int.parse(e.substring(colon + 1), radix: 16);
    }
  }

  late final Int32List starts;
  late final Uint16List flags;

  int of(int cp) {
    var lo = 0;
    var hi = starts.length - 1;
    while (lo < hi) {
      final mid = (lo + hi + 1) >> 1;
      if (starts[mid] <= cp) {
        lo = mid;
      } else {
        hi = mid - 1;
      }
    }
    return flags[lo];
  }
}

final Map<int, int> _lowerSingle = () {
  final out = <int, int>{};
  for (final e in pyLowerRuns.split(' ')) {
    final [start, count, step, delta] = [
      for (final p in e.split('.')) int.parse(p, radix: 16),
    ];
    for (var i = 0; i < count; i++) {
      out[start + i * step] = start + i * step + delta;
    }
  }
  return out;
}();

final Map<int, List<int>> _lowerExpansions = {
  for (final e in pyLowerExpansions.split(' '))
    int.parse(e.substring(0, e.indexOf(':')), radix: 16): [
      for (final h in e.substring(e.indexOf(':') + 1).split('.'))
        int.parse(h, radix: 16),
    ],
};

bool _has(int cp, int flag) => _flags.of(cp) & flag != 0;

/// `ch.isalpha()`: general category Lm, Lt, Lu, Ll or Lo.
bool pyIsAlpha(int cp) => _has(cp, pyAlphaFlag);

/// `ch.isdecimal()`, which is also what `re`'s `\d` matches.
bool pyIsDecimal(int cp) => _has(cp, pyDecimalFlag);

/// `ch.isalnum()`: alphabetic, or with a numeric value (decimal and digit
/// values imply one).
bool pyIsAlnum(int cp) => _has(cp, pyAlphaFlag | pyNumericFlag);

/// `re`'s `\w` on a `str` pattern: alphanumeric or `_`.
bool pyIsWord(int cp) => cp == 0x5F || pyIsAlnum(cp);

/// `re`'s `[^\W\d_]`: a word character that is neither a decimal nor `_`.
bool pyIsLetter(int cp) => cp != 0x5F && pyIsAlnum(cp) && !pyIsDecimal(cp);

/// `ch.isspace()`: bidi class WS, B or S, or category Zs.
bool pyIsSpace(int cp) => _has(cp, pySpaceFlag);

/// `ch.islower()` for one character: the Lowercase property.
bool pyIsLower(int cp) => _has(cp, pyLowerFlag);

/// `ch.isupper()` for one character: the Uppercase property.
bool pyIsUpper(int cp) => _has(cp, pyUpperFlag);

/// `ch.isprintable()`: not in a C* or Z* category, or U+0020.
bool pyIsPrintable(int cp) => !_has(cp, pyNonPrintableFlag);

/// `unicodedata.combining(ch) != 0`.
bool pyIsCombining(int cp) => _has(cp, pyCombiningFlag);

/// `s.isupper()`: some character is uppercase and none is lowercase or
/// titlecase.
bool pyStrIsUpper(List<int> s) {
  var cased = false;
  for (final c in s) {
    if (_has(c, pyLowerFlag | pyTitleFlag)) return false;
    if (!cased && _has(c, pyUpperFlag)) cased = true;
  }
  return cased;
}

/// `s.lower()`: the full lowercase mapping, with CPython's Final_Sigma rule
/// for U+03A3 and no other context or language rule.
List<int> pyLower(List<int> s) {
  final out = <int>[];
  for (var i = 0; i < s.length; i++) {
    final c = s[i];
    if (c == 0x3A3) {
      out.add(_finalSigma(s, i) ? 0x3C2 : 0x3C3);
    } else if (_lowerExpansions[c] case final many?) {
      out.addAll(many);
    } else {
      out.add(_lowerSingle[c] ?? c);
    }
  }
  return out;
}

// CPython handle_capital_sigma: cased, case-ignorable*, SIGMA, and not
// followed by case-ignorable* then cased.
bool _finalSigma(List<int> s, int i) {
  var j = i - 1;
  while (j >= 0 && _has(s[j], pyCaseIgnorableFlag)) {
    j--;
  }
  if (j < 0 || !_has(s[j], pyCasedFlag)) return false;
  j = i + 1;
  while (j < s.length && _has(s[j], pyCaseIgnorableFlag)) {
    j++;
  }
  return j == s.length || !_has(s[j], pyCasedFlag);
}

/// `s.strip()`: without leading and trailing [pyIsSpace] characters.
List<int> pyStrip(List<int> s) {
  var start = 0;
  var end = s.length;
  while (start < end && pyIsSpace(s[start])) {
    start++;
  }
  while (end > start && pyIsSpace(s[end - 1])) {
    end--;
  }
  return s.sublist(start, end);
}

/// `s.split()`: the runs of non-[pyIsSpace] characters.
List<List<int>> pySplit(List<int> s) {
  final out = <List<int>>[];
  int? start;
  for (var i = 0; i <= s.length; i++) {
    final space = i == s.length || pyIsSpace(s[i]);
    if (space && start != null) {
      out.add(s.sublist(start, i));
      start = null;
    } else if (!space) {
      start ??= i;
    }
  }
  return out;
}

/// `s.split(sep)` for a single-code-point separator.
List<List<int>> pySplitOn(List<int> s, int sep) {
  final out = <List<int>>[];
  var start = 0;
  for (var i = 0; i < s.length; i++) {
    if (s[i] == sep) {
      out.add(s.sublist(start, i));
      start = i + 1;
    }
  }
  out.add(s.sublist(start));
  return out;
}

/// [pyStrip] of a Dart string: Python's `s.strip()`.
String pyStripStr(String s) => String.fromCharCodes(pyStrip(s.runes.toList()));

/// [pyLower] of a Dart string: Python's `s.lower()`.
String pyLowerStr(String s) => String.fromCharCodes(pyLower(s.runes.toList()));
