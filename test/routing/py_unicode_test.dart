import 'dart:convert';
import 'dart:io';

import 'package:laya_server/src/routing/py_unicode.dart';
import 'package:test/test.dart';

const _sigma = 0x3A3;
const _finalSigma = 0x3C2;

/// The bits tool/reference/unicode_props.py records, computed from the port.
int _bits(int cp) {
  var b = 0;
  if (pyIsAlpha(cp)) b |= 1;
  if (pyIsDecimal(cp)) b |= 2;
  if (pyIsWord(cp)) b |= 4;
  if (pyIsSpace(cp)) b |= 8;
  if (pyIsLower(cp)) b |= 16;
  if (pyIsUpper(cp)) b |= 32;
  if (!pyStrIsUpper([0x41, cp])) b |= 64;
  if (pyIsCombining(cp)) b |= 128;
  if (pyIsPrintable(cp)) b |= 256;
  if (pyLower([0x41, cp, _sigma]).last == _finalSigma) b |= 512;
  if (pyLower([cp, _sigma]).last == _finalSigma) b |= 1024;
  if (pyLower([0x41, _sigma, cp])[1] == _finalSigma) b |= 2048;
  if (pyIsDecimal(cp)) b |= 4096; // re's \d is str.isdecimal
  return b;
}

void main() {
  final fixture =
      jsonDecode(File('test/fixtures/unicode_props.json').readAsStringSync())
          as Map<String, Object?>;
  final data = (fixture['cases']! as List).single as Map<String, Object?>;

  test('the tables are the reference Unicode version', () {
    expect(pyUnicodeDataVersion, fixture['unicode']);
  });

  test('every code point behaves as Python 3.12 does', () {
    final runs = (data['runs']! as List).cast<List<Object?>>();
    expect(runs.length, greaterThan(1000));
    var mismatches = 0;
    final first = <String>[];
    for (var i = 0; i < runs.length; i++) {
      final start = runs[i][0]! as int;
      final end = i + 1 < runs.length ? runs[i + 1][0]! as int : 0x110000;
      final want = runs[i][1]! as int;
      for (var cp = start; cp < end; cp++) {
        final got = _bits(cp);
        if (got != want) {
          mismatches++;
          if (first.length < 20) {
            first.add('U+${cp.toRadixString(16)}: got $got, want $want');
          }
        }
      }
    }
    expect(mismatches, 0, reason: first.join('\n'));
  });

  test('every lowercase mapping matches str.lower', () {
    final want = {
      for (final e in (data['lower']! as List).cast<List<Object?>>())
        e[0]! as int: (e[1]! as List).cast<int>(),
    };
    final wrong = <String>[];
    for (var cp = 0; cp < 0x110000; cp++) {
      final got = pyLower([cp]);
      final expected = want[cp] ?? [cp];
      if (got.length != expected.length ||
          [
            for (var i = 0; i < got.length; i++) got[i] == expected[i],
          ].contains(false)) {
        wrong.add('U+${cp.toRadixString(16)}: $got, want $expected');
      }
    }
    expect(wrong, isEmpty);
  });

  test('strip and split use Python whitespace', () {
    expect(
      String.fromCharCodes(pyStrip('\u00a0 a b\u2028'.runes.toList())),
      'a b',
    );
    expect(
      [
        for (final w in pySplit('a\u3000b\x1cc  d'.runes.toList()))
          String.fromCharCodes(w),
      ],
      ['a', 'b', 'c', 'd'],
    );
    expect(pySplit(' \t '.runes.toList()), isEmpty);
  });
}
