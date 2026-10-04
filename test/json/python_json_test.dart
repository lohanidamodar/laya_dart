import 'dart:convert';
import 'dart:io';

import 'package:laya_server/src/json/python_json.dart';
import 'package:test/test.dart';

void main() {
  group('pythonFloatRepr', () {
    final cases = <double, String>{
      0.1: '0.1',
      1e-05: '1e-05',
      1e16: '1e+16',
      1e-7: '1e-07',
      123456789.123: '123456789.123',
      -0.0: '-0.0',
      2.5: '2.5',
      1.0: '1.0',
      1e15: '1000000000000000.0',
      0.0001: '0.0001',
      1.5e300: '1.5e+300',
      1.25e-10: '1.25e-10',
      -3.5e20: '-3.5e+20',
      double.infinity: 'Infinity',
      double.negativeInfinity: '-Infinity',
    };
    cases.forEach((value, text) {
      test('$value -> $text', () => expect(pythonFloatRepr(value), text));
    });
    test('NaN', () => expect(pythonFloatRepr(double.nan), 'NaN'));
  });

  final fixture =
      jsonDecode(File('test/fixtures/pyjson.json').readAsStringSync()) as Map;
  final cases = (fixture['cases'] as List).cast<Map<String, Object?>>();

  test('matches json.dumps on every fixture state', () {
    for (final c in cases) {
      expect(pythonJsonDumps(c['value']), c['dumps'], reason: '${c['value']}');
    }
  });

  test('matches render_criterion separators on every fixture', () {
    for (final c in cases) {
      final value = c['value'];
      final text = value is String
          ? value
          : pythonJsonDumps(value, itemSeparator: ', ', keySeparator: ': ');
      expect(text, c['criterion'], reason: '$value');
    }
  });

  test('escapes like Python with ensure_ascii=False', () {
    expect(
      pythonJsonDumps('a"b\\c\n\t\u0000\u001f é 😀'),
      r'"a\"b\\c\n\t\u0000\u001f é 😀"',
    );
    expect(pythonJsonDumps('\r\b\f'), r'"\r\b\f"');
  });

  test('custom separators and non-string keys', () {
    expect(
      pythonJsonDumps(
        {
          'a': 1,
          2: [true, null],
        },
        itemSeparator: ',',
        keySeparator: ':',
      ),
      '{"a":1,"2":[true,null]}',
    );
  });

  test('unsupported values throw', () {
    expect(() => pythonJsonDumps(Object()), throwsArgumentError);
  });
}
