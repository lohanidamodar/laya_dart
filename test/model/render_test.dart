import 'dart:convert';
import 'dart:io';

import 'package:laya_server/src/model/question.dart';
import 'package:laya_server/src/model/render.dart';
import 'package:laya_server/src/server/errors.dart';
import 'package:test/test.dart';

void main() {
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
