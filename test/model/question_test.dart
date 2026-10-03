import 'package:laya_server/src/model/question.dart';
import 'package:laya_server/src/server/errors.dart';
import 'package:test/test.dart';

Matcher _invalid(String detail) => throwsA(
  isA<ApiException>()
      .having((e) => e.status, 'status', 422)
      .having((e) => e.detail, 'detail', detail),
);

void main() {
  test('choice renders key: criterion, bare key for null/empty', () {
    final q = Question.parse('q', {
      'type': 'choice',
      'instructions': 'Which team?',
      'criteria': {
        'billing': 'refunds',
        'tech': null,
        '1': '',
        'x': {'d': 1},
      },
    });
    expect(q.renderOptions(), ['billing: refunds', 'tech', '1', 'x: {"d": 1}']);
  });

  test('choice list criteria become labels', () {
    final q = Question.parse('q', {
      'type': 'choice',
      'instructions': 'i',
      'criteria': ['a', 'b'],
    });
    expect(q.renderOptions(), ['a', 'b']);
  });

  test('numeric and bool list labels render like Python str()', () {
    final q = Question.parse('q', {
      'type': 'choice',
      'instructions': 'i',
      'criteria': [1, 2.5, false],
    });
    expect(q.renderOptions(), ['1', '2.5', 'False']);
  });

  test('score levels', () {
    final q = Question.parse('q', {
      'type': 'score',
      'instructions': 'i',
      'criteria': ['low', 'high'],
    });
    expect(q.renderOptions(), ['level 0: low', 'level 1: high']);
  });

  test('noul defaults and custom criteria, false first', () {
    final q = Question.parse('q', {'type': 'noul', 'instructions': 'i'});
    expect(q.renderOptions(), [
      'false: no, the statement does not hold',
      'true: yes, the statement holds',
    ]);
    final c = Question.parse('q', {
      'type': 'noul',
      'instructions': 'i',
      'criteria': {'True': 'yes it is', 'false': ''},
    });
    expect(c.renderOptions(), [
      'false: no, the statement does not hold',
      'true: yes it is',
    ]);
  });

  test('noul labels replace the option words', () {
    final q = Question.parse('q', {
      'type': 'noul',
      'instructions': 'i',
      'labels': {'false': ' no ', 'true': 'yes'},
    });
    expect(q.renderOptions()[0], 'no: no, the statement does not hold');
    expect(q.renderOptions()[1], 'yes: yes, the statement holds');
  });

  test('object instructions are Python json', () {
    final q = Question.parse('q', {
      'type': 'noul',
      'instructions': {
        'a': [1, 2.0],
      },
    });
    expect(q.instructions, '{"a": [1, 2.0]}');
  });

  test('unknown type is 422 naming the question', () {
    expect(
      () => Question.parse('q1', {'type': 'rank', 'instructions': 'i'}),
      throwsA(
        isA<ApiException>()
            .having((e) => e.status, 'status', 422)
            .having((e) => e.detail, 'detail', contains('q1')),
      ),
    );
  });

  test('validation messages match laya', () {
    expect(
      () => Question.parse('q', {'type': 'rank', 'instructions': 'i'}),
      _invalid(
        "question 'q': unknown type 'rank'; use one of ['choice', 'noul', 'score']",
      ),
    );
    expect(
      () => Question.parse('q', 'x'),
      _invalid("question 'q': definition must be a dict, got str"),
    );
    expect(
      () => Question.parse('q', {'type': 'noul'}),
      _invalid(
        "question 'q': no 'instructions'; add the text the model should answer",
      ),
    );
    expect(
      () => Question.parse('q', {'type': 'noul', 'instructions': '  '}),
      _invalid(
        "question 'q': 'instructions' must not be empty; add the text the model should answer",
      ),
    );
    expect(
      () => Question.parse('q', {'type': 'choice', 'instructions': 'i'}),
      _invalid(
        "question 'q': a choice question takes 'criteria' as a dict of "
        'label -> description, or a list of labels',
      ),
    );
    expect(
      () => Question.parse('q', {
        'type': 'choice',
        'instructions': 'i',
        'criteria': ['a', 'b', 'a'],
      }),
      _invalid(
        "question 'q': choice label 2 ('a') repeats label 0; the labels are the "
        'answer keys, so every option needs its own (1, 1.0 and True are one key)',
      ),
    );
    expect(
      () => Question.parse('q', {
        'type': 'noul',
        'instructions': 'i',
        'criteria': {'maybe': 'x'},
      }),
      _invalid(
        "question 'q': a noul question takes 'criteria' keyed only "
        "'true'/'false' (either or both, and omitted is fine), got ['maybe']. "
        'Those keys are the option texts the model reads; any other key was '
        'silently dropped and replaced with the defaults. If you want the answer '
        "worded differently, keep 'criteria' keyed 'true'/'false' and set "
        "'labels' instead.",
      ),
    );
    expect(
      () => Question.parse('q', {
        'type': 'score',
        'instructions': 'i',
        'criteria': ['a', 'b'],
        'option_order': [0, 0],
      }),
      _invalid(
        "question 'q': 'option_order' must be a permutation of range(2) -- one "
        'slot per option, each option once -- got [0, 0]',
      ),
    );
    expect(
      () => Question.parse('q', {
        'type': 'choice',
        'instructions': 'i',
        'criteria': ['a'],
        'labels': {'false': 'a', 'true': 'b'},
      }),
      _invalid("question 'q': 'labels' is only supported for noul questions"),
    );
    expect(
      () => Question.parse('q', {
        'type': 'noul',
        'instructions': 'i',
        'labels': {'false': 'a', 'true': 'a'},
      }),
      _invalid(
        "question 'q': noul labels must map exactly 'false' and 'true' to "
        'distinct non-empty strings',
      ),
    );
  });
}
