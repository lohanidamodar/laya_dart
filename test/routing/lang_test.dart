import 'dart:convert';
import 'dart:io';

import 'package:laya_server/src/routing/lang.dart';
import 'package:test/test.dart';

void main() {
  final cases =
      ((jsonDecode(File('test/fixtures/lang.json').readAsStringSync())
                  as Map)['cases']
              as List)
          .cast<Map<String, Object?>>();

  test('the fixture numbers print the same from Dart and Python', () {
    // The fixture is Python's json.dumps; the comparison below re-encodes it
    // with Dart's jsonEncode, so the two must agree on every number in it.
    final raw = File('test/fixtures/lang.json').readAsStringSync();
    final numbers = [
      for (final m in RegExp(
        r'"(?:diacritic_rate|non_latin_fraction)":([^,}]+)',
      ).allMatches(raw))
        m.group(1)!,
      for (final m in RegExp(r'"script_profile":\{([^}]*)\}').allMatches(raw))
        if (m.group(1)!.isNotEmpty)
          for (final entry in m.group(1)!.split(',')) entry.split(':').last,
    ];
    expect(numbers.length, greaterThan(200));
    for (final text in numbers) {
      expect(jsonEncode(jsonDecode(text)), text);
    }
  });

  for (final (i, c) in cases.indexed) {
    final input = c.containsKey('text') ? c['text'] : c['state'];
    test('case $i equals laya.lang.analyse', () {
      final got = analyse(input).toJson();
      expect(
        jsonEncode(got),
        jsonEncode(c['analysis']),
        reason: jsonEncode(input),
      );
    });
  }

  test('text and structured states both go through analyse', () {
    expect(analyse(null).script, 'unknown');
    expect(analyse(42).isEnglish, isTrue);
    // Recorded from laya.lang.analyse in the reference venv.
    final mixed = analyse({
      'subject': 'Order issue',
      'body':
          'Hello team, I am writing about the order that was delivered to the '
          'wrong address last week and the tracking page that is still not '
          'updated.\nQuero cancelar minha assinatura agora mesmo, por favor.\n'
          'Thanks for the help with this.',
    });
    expect(mixed.language, 'pt');
    expect(
      mixed.mixedSegment,
      'Quero cancelar minha assinatura agora mesmo, por favor.',
    );
  });
}
