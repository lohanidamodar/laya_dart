import 'dart:convert';
import 'dart:io';

import 'package:laya_server/src/tokenizer/nfc.dart';
import 'package:test/test.dart';

String _hex(String s) =>
    s.runes.map((r) => r.toRadixString(16).padLeft(4, '0')).join(' ');

void main() {
  test('composes', () {
    expect(nfc('Cafe\u{301}'), 'Caf\u{e9}');
    expect(nfc('\u{1100}\u{1161}'), '\u{ac00}');
  });

  test('leaves text below U+0300 untouched, as the same object', () {
    const ascii = 'plain text, \u{e9}\u{ff} and \u{2ff}';
    expect(identical(nfc(ascii), ascii), isTrue);
  });

  test('orders marks canonically before composing', () {
    // dot below (220) sorts before acute (230); a + dot below composes, then the acute stays.
    expect(nfc('a\u{301}\u{323}'), '\u{1ea1}\u{301}');
    // q has no composites, so only the reordering shows.
    expect(nfc('q\u{301}\u{323}'), 'q\u{323}\u{301}');
  });

  test('a long run of marks sorts stably (checked against HF tokenizers)', () {
    final input = 'a${'\u{301}\u{323}' * 1000}';
    expect(nfc(input), '\u{1ea1}${'\u{323}' * 999}${'\u{301}' * 1000}');
    // 100k marks in one run must not take quadratic time.
    final stopwatch = Stopwatch()..start();
    nfc('a${'\u{301}\u{323}' * 50000}');
    expect(stopwatch.elapsed, lessThan(const Duration(seconds: 2)));
  });

  test('a mark of the same class blocks composition', () {
    expect(nfc('a\u{301}\u{301}'), '\u{e1}\u{301}');
    expect(nfc('a\u{300}\u{301}'), '\u{e0}\u{301}');
  });

  test('singletons and composition exclusions stay decomposed', () {
    expect(nfc('\u{212b}'), '\u{c5}'); // ANGSTROM SIGN -> A with ring
    expect(nfc('\u{2126}'), '\u{3a9}'); // OHM SIGN -> GREEK OMEGA
    expect(nfc('\u{915}\u{93c}'), '\u{915}\u{93c}'); // U+0958 is excluded
    expect(nfc('\u{958}'), '\u{915}\u{93c}');
  });

  test('Hangul composes algorithmically', () {
    expect(nfc('\u{1100}\u{1161}\u{11a8}'), '\u{ac01}');
    expect(nfc('\u{ac00}\u{11a8}'), '\u{ac01}');
    expect(nfc('\u{ac01}\u{11a8}'), '\u{ac01}\u{11a8}');
    expect(nfc('\u{1100}\u{1161}\u{11a7}'), '\u{ac00}\u{11a7}');
  });

  test('handles astral code points and lone surrogates', () {
    // U+1109A = U+11099 U+110BA (Kaithi).
    expect(nfc('\u{11099}\u{110ba}'), '\u{1109a}');
    expect(nfc('x\u{301}\u{d800}'), 'x\u{301}\u{d800}');
  });

  test('matches HF tokenizers NFC on every fixture case', () {
    final fixture =
        jsonDecode(File('test/fixtures/nfc.json').readAsStringSync())
            as Map<String, Object?>;
    final cases = (fixture['cases']! as List).cast<Map<String, Object?>>();
    expect(cases, hasLength(greaterThan(8000)));
    final failures = <String>[];
    for (final c in cases) {
      final text = c['text']! as String;
      final got = nfc(text);
      if (got != c['nfc']) {
        failures.add(
          '${_hex(text)}: got ${_hex(got)}, want ${_hex(c['nfc']! as String)}',
        );
      }
    }
    expect(failures, isEmpty, reason: failures.take(20).join('\n'));
  });
}
