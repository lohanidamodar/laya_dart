// Generates lib/src/tokenizer/nfc_data.g.dart, the tables behind `nfc()`.
//
// Usage: dart run tool/gen_nfc.dart
//
// Unicode version: 9.0.0, which is what Hugging Face `tokenizers` 0.23.2 actually implements.
// Its NFC comes from the `unicode-normalization-alignments` crate, whose tables predate the
// `unicode-normalization` releases tracking current Unicode. The version was found by probing
// `tokenizers.normalizers.NFC().normalize_str` in .tooling/venv (tokenizers 0.23.2) with, for each
// Unicode version from 6.1 to 17.0, the first few code points whose combining class (0 < ccc <
// 230, probed as `x + U+0301 + mark`) or two-part canonical decomposition (probed as the
// decomposed pair) first appeared in that version:
//
//   6.1, 7.0, 8.0, 9.0   every probe reordered or composed (e.g. U+08E3, U+1172B, U+11442,
//                        U+1E94A, and the composites U+1134B, U+114BB, U+115BA);
//   10.0 to 17.0         every probe came back unchanged (e.g. U+0D3B and U+1DF7 from 10.0,
//                        U+0EBA from 12.0, the composite U+11938 from 13.0, U+105C9 from 16.0).
//
// A newer table would compose or reorder text that the reference leaves alone, so the tables are
// pinned to 9.0.0. test/fixtures/nfc.json (tool/reference/nfc.py) checks every code point with a
// decomposition or a combining class against the reference.
//
// The UCD files are cached under .tooling/ucd/<version>/ and checked against the sha256 values
// below, which were recorded from the first download.
//
// Structure: [Ucd] parses UnicodeData.txt into per-code-point records once; each table is its own
// emitter over those records. Another generated file (Python `unicodedata` categories and
// decompositions, for instance) is another emitter and another output with its own pinned version.
//
// Second output: lib/src/python/unicode_data.g.dart, the character data behind the Python `str`,
// `re` and `unicodedata` behaviour that laya's `lang.py` and `%r` rely on. It is pinned to
// Unicode 15.0.0, the version of Python 3.12's `unicodedata.unidata_version` (the venv reports
// 15.0.0), and is independent of the NFC pin above. The flags follow CPython's
// Tools/unicode/makeunicodedata.py; test/fixtures/unicode_props.json
// (tool/reference/unicode_props.py) checks every code point against the reference.
import 'dart:io';

import 'package:crypto/crypto.dart';

const _version = '9.0.0';

/// sha256 of each UCD file at [_version], recorded on first download.
const _sha256 = {
  'UnicodeData.txt':
      '68dfc414d28257b9b5d6ddbb8b466c768c00ebdf6cbf7784364a9b6cad55ee8f',
  'CompositionExclusions.txt':
      '5623df16856ad4007c60bdfff6f054e087521becd24cb4006be69c3a1d851aee',
};

const _out = 'lib/src/tokenizer/nfc_data.g.dart';

/// Returns the UCD file [name] at [version], downloading it on first use and checking its hash.
Future<String> _ucdFile(String version, String name, String sha256Hex) async {
  final file = File('.tooling/ucd/$version/$name');
  if (!file.existsSync()) {
    file.parent.createSync(recursive: true);
    final url = 'https://www.unicode.org/Public/$version/ucd/$name';
    final client = HttpClient();
    try {
      final response = await (await client.getUrl(Uri.parse(url))).close();
      if (response.statusCode != 200) {
        throw HttpException('GET $url -> ${response.statusCode}');
      }
      final part = File('${file.path}.part');
      await response.pipe(part.openWrite());
      part.renameSync(file.path);
    } finally {
      client.close();
    }
  }
  final bytes = file.readAsBytesSync();
  final actual = sha256.convert(bytes).toString();
  stdout.writeln('$name $version sha256 $actual');
  if (actual != sha256Hex) {
    throw StateError('${file.path}: sha256 $actual, expected $sha256Hex');
  }
  return String.fromCharCodes(bytes);
}

/// One UnicodeData.txt line.
class _Record {
  _Record(List<String> f)
    : cp = int.parse(f[0], radix: 16),
      category = f[2],
      ccc = int.parse(f[3]),
      decomposition = f[5];

  final int cp;
  final String category;
  final int ccc;

  /// The raw decomposition field: empty, `<tag> hex...` (compatibility) or `hex...` (canonical).
  final String decomposition;

  /// The one-step canonical decomposition, or null for none or a compatibility one.
  List<int>? get canonical => decomposition.isEmpty || decomposition[0] == '<'
      ? null
      : [for (final h in decomposition.split(' ')) int.parse(h, radix: 16)];
}

/// The parsed UCD at one version.
class _Ucd {
  _Ucd(String unicodeData, String exclusions)
    : records = {
        for (final line in unicodeData.split('\n'))
          if (line.trim().isNotEmpty)
            // Ranges (`<..., First>`/`<..., Last>`) carry ccc 0 and no decomposition, so their
            // two endpoint records are all that the NFC tables need.
            _Record(line.split(';')).cp: _Record(line.split(';')),
      },
      exclusions = {
        for (final line in exclusions.split('\n'))
          if (line.split('#').first.trim() case final cp when cp.isNotEmpty)
            int.parse(cp, radix: 16),
      };

  final Map<int, _Record> records;

  /// CompositionExclusions.txt: script-specific and post-composition-version exclusions.
  final Set<int> exclusions;

  int ccc(int cp) => records[cp]?.ccc ?? 0;

  /// The full canonical decomposition of [cp], recursively, not yet canonically ordered.
  List<int> fullDecomposition(int cp) {
    final step = records[cp]?.canonical;
    if (step == null) return [cp];
    return [for (final c in step) ...fullDecomposition(c)];
  }
}

String _hex(int v) => v.toRadixString(16);

/// `start[-end]:ccc` for each run of consecutive code points with the same nonzero class.
List<String> _cccEntries(_Ucd ucd) {
  final cps = ucd.records.keys.where((cp) => ucd.ccc(cp) != 0).toList()..sort();
  final out = <String>[];
  for (var i = 0; i < cps.length;) {
    var j = i;
    while (j + 1 < cps.length &&
        cps[j + 1] == cps[j] + 1 &&
        ucd.ccc(cps[j + 1]) == ucd.ccc(cps[i])) {
      j++;
    }
    final range = i == j ? _hex(cps[i]) : '${_hex(cps[i])}-${_hex(cps[j])}';
    out.add('$range:${_hex(ucd.ccc(cps[i]))}');
    i = j + 1;
  }
  return out;
}

/// `cp:d.d...` with each code point's full canonical decomposition.
List<String> _decompositionEntries(_Ucd ucd) => [
  for (final r in ucd.records.values.toList()..sort((a, b) => a.cp - b.cp))
    if (r.canonical != null)
      '${_hex(r.cp)}:${ucd.fullDecomposition(r.cp).map(_hex).join('.')}',
];

/// `a.b:c` for each primary composite: a two-part canonical decomposition that is neither in
/// CompositionExclusions.txt nor a non-starter decomposition. Singletons have one part, so
/// they never appear.
List<String> _compositionEntries(_Ucd ucd) => [
  for (final r in ucd.records.values.toList()..sort((a, b) => a.cp - b.cp))
    if (r.canonical case [
      final a,
      final b,
    ] when !ucd.exclusions.contains(r.cp) && r.ccc == 0 && ucd.ccc(a) == 0)
      '${_hex(a)}.${_hex(b)}:${_hex(r.cp)}',
];

/// A Dart `const String` built from adjacent literals, [entries] joined by spaces.
String _constString(String name, String doc, List<String> entries) {
  final lines = <String>[];
  var line = StringBuffer();
  for (final e in entries) {
    if (line.isNotEmpty && line.length + 1 + e.length > 72) {
      lines.add(line.toString());
      line = StringBuffer();
    }
    if (line.isNotEmpty) line.write(' ');
    line.write(e);
  }
  if (line.isNotEmpty) lines.add(line.toString());
  // Each literal but the last ends in a space, so the joined string is space-separated.
  final literals = [
    for (final (i, l) in lines.indexed)
      i == lines.length - 1 ? "    '$l'" : "    '$l '",
  ];
  return '/// $doc\nconst String $name =\n${literals.join('\n')};\n';
}

const _pyVersion = '15.0.0';

/// sha256 of each UCD file at [_pyVersion], recorded on first download.
const _pySha256 = {
  'UnicodeData.txt':
      '806e9aed65037197f1ec85e12be6e8cd870fc5608b4de0fffd990f689f376a73',
  'DerivedCoreProperties.txt':
      'd367290bc0867e6b484c68370530bdd1a08b6b32404601b8c7accaf83e05628d',
  'SpecialCasing.txt':
      '78b29c64b5840d25c11a9f31b665ee551b8a499eca6c70d770fcad7dd710f494',
};

const _pyOut = 'lib/src/python/unicode_data.g.dart';

/// Flag bits of the Python character table, as CPython's makeunicodedata.py derives them.
const _pyFlags = {
  'alpha': 1, // category Lm, Lt, Lu, Ll or Lo
  'decimal': 2, // UnicodeData field 6 present
  'numeric': 4, // UnicodeData field 8 present (digit and decimal imply it)
  'space': 8, // bidi WS, B or S, or category Zs
  'lower': 16, // DerivedCoreProperties Lowercase
  'upper': 32, // DerivedCoreProperties Uppercase
  'title': 64, // category Lt
  'cased': 128, // DerivedCoreProperties Cased
  'caseIgnorable': 256, // DerivedCoreProperties Case_Ignorable
  'nonPrintable': 512, // category C* or Z*, except U+0020
  'combining': 1024, // canonical combining class != 0
};

/// The Python character table at [_pyVersion]: flags per code point and the lowercase mapping.
class _PyUcd {
  _PyUcd(String unicodeData, String derived, String specialCasing) {
    String? rangeFirst;
    for (final line in unicodeData.split('\n')) {
      if (line.trim().isEmpty) continue;
      final f = line.split(';');
      final cp = int.parse(f[0], radix: 16);
      // `<..., First>`/`<..., Last>` pairs stand for every code point between them.
      if (f[1].endsWith(', First>')) {
        rangeFirst = f[0];
        continue;
      }
      final start = f[1].endsWith(', Last>')
          ? int.parse(rangeFirst!, radix: 16)
          : cp;
      final flags = _recordFlags(f);
      for (var c = start; c <= cp; c++) {
        this.flags[c] = flags;
        if (f[13].isNotEmpty) lower[c] = [int.parse(f[13], radix: 16)];
      }
    }
    const props = {
      'Lowercase': 'lower',
      'Uppercase': 'upper',
      'Cased': 'cased',
      'Case_Ignorable': 'caseIgnorable',
    };
    for (final line in derived.split('\n')) {
      final data = line.split('#').first.trim();
      if (data.isEmpty) continue;
      final parts = [for (final p in data.split(';')) p.trim()];
      final range = parts[0];
      final prop = parts[1];
      final bit = _pyFlags[props[prop]];
      if (bit == null) continue;
      final ends = range.split('..');
      final lo = int.parse(ends.first, radix: 16);
      final hi = int.parse(ends.last, radix: 16);
      for (var c = lo; c <= hi; c++) {
        flags[c] = (flags[c] ?? _pyFlags['nonPrintable']!) | bit;
      }
    }
    // CPython ignores every conditional mapping (it hardcodes Final_Sigma instead), so only
    // unconditional lines count; their lowercase replaces the simple one.
    for (final line in specialCasing.split('\n')) {
      final data = line.split('#').first.trim();
      if (data.isEmpty) continue;
      final f = [for (final p in data.split(';')) p.trim()];
      if (f.length > 4 && f[4].isNotEmpty) continue;
      lower[int.parse(f[0], radix: 16)] = [
        for (final h in f[1].split(' ')) int.parse(h, radix: 16),
      ];
    }
  }

  static int _recordFlags(List<String> f) {
    final category = f[2];
    final bidi = f[4];
    var flags = 0;
    if (const ['Lm', 'Lt', 'Lu', 'Ll', 'Lo'].contains(category)) {
      flags |= _pyFlags['alpha']!;
    }
    if (f[6].isNotEmpty) flags |= _pyFlags['decimal']!;
    if (f[8].isNotEmpty) flags |= _pyFlags['numeric']!;
    if (category == 'Zs' || const ['WS', 'B', 'S'].contains(bidi)) {
      flags |= _pyFlags['space']!;
    }
    if (category == 'Lt') flags |= _pyFlags['title']!;
    if (f[0] != '0020' && (category[0] == 'C' || category[0] == 'Z')) {
      flags |= _pyFlags['nonPrintable']!;
    }
    if (int.parse(f[3]) != 0) flags |= _pyFlags['combining']!;
    return flags;
  }

  /// Flags of each assigned code point; an unassigned one (category Cn) is only non-printable.
  final Map<int, int> flags = {};

  /// The full lowercase of each code point that has one.
  final Map<int, List<int>> lower = {};

  int flagsOf(int cp) => flags[cp] ?? _pyFlags['nonPrintable']!;

  /// `start:flags` at each code point where the flags change, from U+0000.
  List<String> flagRuns() {
    final out = <String>[];
    int? last;
    for (var cp = 0; cp < 0x110000; cp++) {
      final f = flagsOf(cp);
      if (f != last) out.add('${_hex(cp)}:${_hex(f)}');
      last = f;
    }
    return out;
  }

  /// One-to-one lowercase mappings as `start.count.step.delta`: [count] code points from
  /// [start], [step] apart, each mapping to itself plus [delta] (signed hex).
  List<String> lowerRuns() {
    final singles = [
      for (final e in lower.entries)
        if (e.value.length == 1 && e.value.single != e.key)
          (e.key, e.value.single - e.key),
    ]..sort((a, b) => a.$1 - b.$1);
    final out = <String>[];
    for (var i = 0; i < singles.length;) {
      final (start, delta) = singles[i];
      var step = 1;
      var count = 1;
      if (i + 1 < singles.length && singles[i + 1].$2 == delta) {
        step = singles[i + 1].$1 - start;
        while (i + count < singles.length &&
            singles[i + count].$2 == delta &&
            singles[i + count].$1 == start + count * step) {
          count++;
        }
      }
      final d = delta < 0 ? '-${_hex(-delta)}' : _hex(delta);
      out.add('${_hex(start)}.${_hex(count)}.${_hex(step)}.$d');
      i += count;
    }
    return out;
  }

  /// Lowercase mappings to more than one code point, as `cp:l.l...`.
  List<String> lowerExpansions() => [
    for (final e in lower.entries.toList()..sort((a, b) => a.key - b.key))
      if (e.value.length > 1) '${_hex(e.key)}:${e.value.map(_hex).join('.')}',
  ];
}

Future<void> _writeFormatted(String path, String content) async {
  File(path)
    ..parent.createSync(recursive: true)
    ..writeAsStringSync(content);
  final format = await Process.run(Platform.resolvedExecutable, [
    'format',
    path,
  ]);
  if (format.exitCode != 0) {
    throw ProcessException('dart', ['format', path], '${format.stderr}');
  }
  stdout.writeln('wrote $path (${File(path).lengthSync()} bytes)');
}

Future<void> _writePyUnicode() async {
  Future<String> file(String name) =>
      _ucdFile(_pyVersion, name, _pySha256[name]!);
  final ucd = _PyUcd(
    await file('UnicodeData.txt'),
    await file('DerivedCoreProperties.txt'),
    await file('SpecialCasing.txt'),
  );
  final out = StringBuffer()
    ..writeln(
      '// GENERATED by tool/gen_nfc.dart from the Unicode $_pyVersion UCD.',
    )
    ..writeln('// Do not edit by hand; rerun the tool. Numbers are hex.')
    ..writeln(
      '// Derived from the Unicode Character Database, (c) Unicode, Inc.,',
    )
    ..writeln('// under the Unicode License v3; see NOTICE.')
    ..writeln()
    ..writeln(
      '/// The Unicode version these tables implement: Python 3.12\'s '
      '`unicodedata.unidata_version`.',
    )
    ..writeln("const String pyUnicodeVersion = '$_pyVersion';")
    ..writeln();
  for (final e in _pyFlags.entries) {
    out
      ..writeln('/// Flag bit: ${e.key}.')
      ..writeln(
        'const int py${e.key[0].toUpperCase()}${e.key.substring(1)}Flag = '
        '${e.value};',
      )
      ..writeln();
  }
  out
    ..writeln(
      _constString(
        'pyFlagRuns',
        'Character flags as `start:flags`, one entry where they change, '
            'from U+0000 to U+10FFFF.',
        ucd.flagRuns(),
      ),
    )
    ..writeln(
      _constString(
        'pyLowerRuns',
        'One-to-one lowercase mappings, as `start.count.step.delta` '
            '(delta signed).',
        ucd.lowerRuns(),
      ),
    )
    ..writeln(
      _constString(
        'pyLowerExpansions',
        'Lowercase mappings to several code points, as `cp:l.l...`.',
        ucd.lowerExpansions(),
      ),
    );
  await _writeFormatted(_pyOut, out.toString());
}

Future<void> main() async {
  await _writeNfc();
  await _writePyUnicode();
}

Future<void> _writeNfc() async {
  final ucd = _Ucd(
    await _ucdFile(_version, 'UnicodeData.txt', _sha256['UnicodeData.txt']!),
    await _ucdFile(
      _version,
      'CompositionExclusions.txt',
      _sha256['CompositionExclusions.txt']!,
    ),
  );
  final out = StringBuffer()
    ..writeln(
      '// GENERATED by tool/gen_nfc.dart from the Unicode $_version UCD.',
    )
    ..writeln('// Do not edit by hand; rerun the tool. Code points are hex.')
    ..writeln(
      '// Derived from the Unicode Character Database, (c) Unicode, Inc.,',
    )
    ..writeln('// under the Unicode License v3; see NOTICE.')
    ..writeln()
    ..writeln('/// The Unicode version these tables implement.')
    ..writeln("const String unicodeVersion = '$_version';")
    ..writeln()
    ..writeln(
      _constString(
        'cccRanges',
        'Nonzero canonical combining classes, as `start[-end]:ccc`.',
        _cccEntries(ucd),
      ),
    )
    ..writeln(
      _constString(
        'decompositions',
        'Full canonical decompositions (Hangul excluded), as `cp:d.d...`.',
        _decompositionEntries(ucd),
      ),
    )
    ..writeln(
      _constString(
        'compositions',
        'Primary composites (Hangul excluded), as `first.second:composite`.',
        _compositionEntries(ucd),
      ),
    )
    ..writeln(
      '/// First precomposed Hangul syllable (The Unicode Standard, 3.12).',
    )
    ..writeln('const int hangulSBase = 0xAC00;')
    ..writeln()
    ..writeln('/// First leading consonant jamo.')
    ..writeln('const int hangulLBase = 0x1100;')
    ..writeln()
    ..writeln('/// First vowel jamo.')
    ..writeln('const int hangulVBase = 0x1161;')
    ..writeln()
    ..writeln('/// One before the first trailing consonant jamo.')
    ..writeln('const int hangulTBase = 0x11A7;')
    ..writeln()
    ..writeln('/// Number of leading consonants.')
    ..writeln('const int hangulLCount = 19;')
    ..writeln()
    ..writeln('/// Number of vowels.')
    ..writeln('const int hangulVCount = 21;')
    ..writeln()
    ..writeln('/// Number of trailing consonants, counting "none".')
    ..writeln('const int hangulTCount = 28;')
    ..writeln()
    ..writeln('/// Syllables per leading consonant.')
    ..writeln('const int hangulNCount = hangulVCount * hangulTCount;')
    ..writeln()
    ..writeln('/// Number of precomposed syllables.')
    ..writeln('const int hangulSCount = hangulLCount * hangulNCount;');
  await _writeFormatted(_out, out.toString());
}
