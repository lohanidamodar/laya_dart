import 'nfc_data.g.dart';

/// Unicode Normalization Form C, as Hugging Face `tokenizers` 0.23.2 computes it.
///
/// The tables are Unicode 9.0.0, the version that `tokenizers` actually implements (see
/// tool/gen_nfc.dart), so marks and composites from later versions are left as they are, exactly
/// as the reference leaves them. Text with every code unit below U+0300 is already NFC and is
/// returned as the same object. Unpaired surrogates pass through unchanged.
String nfc(String input) {
  var i = 0;
  while (i < input.length && input.codeUnitAt(i) < 0x300) {
    i++;
  }
  if (i == input.length) return input;

  final buffer = <int>[];
  for (final cp in input.runes) {
    _decompose(cp, buffer);
  }
  _orderMarks(buffer);
  _compose(buffer);
  return String.fromCharCodes(buffer);
}

final _tables = _Tables();

/// The generated tables, parsed once on first use.
class _Tables {
  _Tables() {
    for (final entry in cccRanges.split(' ')) {
      final [range, cls] = entry.split(':');
      final [start, ...rest] = range.split('-');
      final from = int.parse(start, radix: 16);
      final to = rest.isEmpty ? from : int.parse(rest.single, radix: 16);
      for (var cp = from; cp <= to; cp++) {
        ccc[cp] = int.parse(cls, radix: 16);
      }
    }
    for (final entry in decompositions.split(' ')) {
      final [cp, parts] = entry.split(':');
      decomposition[int.parse(cp, radix: 16)] = [
        for (final p in parts.split('.')) int.parse(p, radix: 16),
      ];
    }
    for (final entry in compositions.split(' ')) {
      final [pair, composite] = entry.split(':');
      final [a, b] = pair.split('.');
      composition[_pairKey(int.parse(a, radix: 16), int.parse(b, radix: 16))] =
          int.parse(composite, radix: 16);
    }
  }

  /// Nonzero canonical combining classes.
  final Map<int, int> ccc = {};

  /// Full canonical decompositions, Hangul excluded.
  final Map<int, List<int>> decomposition = {};

  /// Primary composites by [_pairKey], Hangul excluded.
  final Map<int, int> composition = {};
}

int _pairKey(int a, int b) => a << 21 | b;

int _ccc(int cp) => cp < 0x300 ? 0 : _tables.ccc[cp] ?? 0;

void _decompose(int cp, List<int> out) {
  final s = cp - hangulSBase;
  if (s >= 0 && s < hangulSCount) {
    out
      ..add(hangulLBase + s ~/ hangulNCount)
      ..add(hangulVBase + (s % hangulNCount) ~/ hangulTCount);
    final t = s % hangulTCount;
    if (t != 0) out.add(hangulTBase + t);
    return;
  }
  final parts = cp < 0xC0 ? null : _tables.decomposition[cp];
  if (parts == null) {
    out.add(cp);
  } else {
    out.addAll(parts);
  }
}

/// Stable-sorts each run of nonzero combining classes by class (canonical ordering).
void _orderMarks(List<int> cps) {
  var i = 0;
  while (i < cps.length) {
    if (_ccc(cps[i]) == 0) {
      i++;
      continue;
    }
    var end = i + 1;
    while (end < cps.length && _ccc(cps[end]) != 0) {
      end++;
    }
    if (end - i > 1) _sortRun(cps, i, end);
    i = end;
  }
}

/// Stable-sorts `cps[start, end)` (all nonzero classes) by class.
void _sortRun(List<int> cps, int start, int end) {
  if (end - start <= 16) {
    // Insertion sort; moving only past strictly greater classes keeps it stable.
    for (var i = start + 1; i < end; i++) {
      final cp = cps[i];
      final c = _ccc(cp);
      var j = i;
      while (j > start && _ccc(cps[j - 1]) > c) {
        cps[j] = cps[j - 1];
        j--;
      }
      cps[j] = cp;
    }
    return;
  }
  // A long run (hostile input) sorts in O(n log n); the index tie-break keeps it stable.
  final keyed = [
    for (var i = start; i < end; i++) (_ccc(cps[i]), i - start, cps[i]),
  ]..sort((a, b) => a.$1 != b.$1 ? a.$1 - b.$1 : a.$2 - b.$2);
  for (var i = 0; i < keyed.length; i++) {
    cps[start + i] = keyed[i].$3;
  }
}

int? _composePair(int a, int b) {
  final l = a - hangulLBase;
  if (l >= 0 && l < hangulLCount) {
    final v = b - hangulVBase;
    if (v >= 0 && v < hangulVCount) {
      return hangulSBase + (l * hangulVCount + v) * hangulTCount;
    }
    return null;
  }
  final s = a - hangulSBase;
  if (s >= 0 && s < hangulSCount && s % hangulTCount == 0) {
    final t = b - hangulTBase;
    // TBase itself is not a trailing consonant.
    if (t > 0 && t < hangulTCount) return a + t;
    return null;
  }
  return _tables.composition[_pairKey(a, b)];
}

/// Canonical composition in place (UAX #15): a character composes with the last starter
/// unless a character between them has a class equal to or above its own, or is a starter.
void _compose(List<int> cps) {
  if (cps.isEmpty) return;
  var starter = -1;
  var lastClass = 0;
  var write = 0;
  for (var read = 0; read < cps.length; read++) {
    final cp = cps[read];
    final cls = _ccc(cp);
    if (starter >= 0 && (lastClass < cls || lastClass == 0)) {
      final composite = _composePair(cps[starter], cp);
      if (composite != null) {
        cps[starter] = composite;
        continue;
      }
    }
    if (cls == 0) {
      starter = write;
      lastClass = 0;
    } else {
      lastClass = cls;
    }
    cps[write++] = cp;
  }
  cps.length = write;
}
