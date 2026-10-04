import 'dart:convert';
import 'dart:io';

import 'package:laya_server/src/errors.dart';
import 'package:laya_server/src/model/question.dart';
import 'package:laya_server/src/model/render.dart';
import 'package:laya_server/src/python/py_str.dart';
import 'package:test/test.dart';

void main() {
  reprTests();
  test('renderCriterion passes strings and dumps the rest', () {
    expect(renderCriterion('a'), 'a');
    expect(
      renderCriterion({
        'd': [1, 2.0],
      }),
      '{"d": [1, 2.0]}',
    );
    expect(renderCriterion(5), '5');
    expect(renderCriterion('é'), 'é');
  });

  group('laya fixtures', () {
    final requests =
        (jsonDecode(File('tool/reference/cases.json').readAsStringSync())
                as Map)['requests']
            as List;
    final cases =
        (jsonDecode(File('test/fixtures/sequence.json').readAsStringSync())
                as Map)['cases']
            as List;
    final seen = <String>{};
    for (final c in cases.cast<Map<String, Object?>>()) {
      final index = c['request_index'] as int;
      final qid = c['qid'] as String;
      if (!seen.add('$index/$qid')) continue;
      final def = ((requests[index] as Map)['questions'] as Map)[qid];
      if (c.containsKey('error')) {
        final message = (c['error'] as String).substring('ValueError: '.length);
        test('request $index $qid is rejected', () {
          expect(
            () => Question.parse(qid, def),
            throwsA(
              isA<ApiException>()
                  .having((e) => e.status, 'status', 422)
                  .having((e) => e.detail, 'detail', message),
            ),
          );
        });
      } else {
        test('request $index $qid renders like laya', () {
          final q = Question.parse(qid, def);
          expect(q.renderOptions(), (c['options'] as List).cast<String>());
          expect(q.instructions, (c['internal'] as Map)['ins']);
        });
      }
    }
  });
}

// Python 3.12 repr of each input, recorded from the reference venv.
const _reprs = {
  '\u00a0': r"'\xa0'",
  '\u200b': r"'\u200b'",
  '\u{e0001}': r"'\U000e0001'",
  'a\u0085b': r"'a\x85b'",
  'é\u2028': r"'é\u2028'",
  'x\u0378': r"'x\u0378'",
};

void reprTests() {
  test('pyRepr escapes non-printable code points like Python', () {
    for (final e in _reprs.entries) {
      expect(pyRepr(e.key), e.value);
    }
    expect(pyRepr(String.fromCharCode(0xd800)), r"'\ud800'");
  });

  // Python str.strip(), checked against laya 0.3.24 `_resolve_noul_labels`:
  // U+001C-U+001F are whitespace to Python (not to Dart's trim), and U+FEFF
  // is not (Dart's trim strips it).
  test('noul labels strip as Python does', () {
    expect(resolveNoulLabels({'false': '\x1cno\x1c', 'true': 'yes\x1f'}), (
      'no',
      'yes',
    ));
    expect(resolveNoulLabels({'false': '\u{FEFF}', 'true': 'yes'}), (
      '\u{FEFF}',
      'yes',
    ));
    expect(resolveNoulLabels({'false': '\x1c', 'true': 'yes'}), isNull);
  });
}
