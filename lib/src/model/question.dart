/// Typed questions: validation, normalisation and option rendering.
///
/// Ports laya's `Agent._check_question`, `_to_internal` and `render_options`.
/// The rendered option texts feed the tokenizer, so they must equal laya's
/// byte for byte.
library;

import '../json/python_json.dart';
import '../server/errors.dart';
import 'render.dart';

/// Question type, with the head's type id.
enum QuestionType {
  /// Pick one of several labelled options.
  choice(0),

  /// Pick a level, index 0 first.
  score(1),

  /// A yes/no statement.
  noul(2);

  const QuestionType(this.id);

  /// The decision head's type id.
  final int id;
}

/// One question in internal form (laya's `_to_internal`).
sealed class Question {
  const Question._(this.id, this.instructions, this.optionOrder);

  /// The question id from the request.
  final String id;

  /// Instructions as text: strings pass through, anything else is Python
  /// `json.dumps`.
  final String instructions;

  /// The validated `option_order` permutation, if the request gave one.
  final List<int>? optionOrder;

  /// The question type.
  QuestionType get type;

  /// Option texts in label-index order (laya `render_options`).
  List<String> renderOptions();

  /// Parses one question value from a request; throws [ApiException] 422
  /// with laya's message when it is invalid.
  static Question parse(String id, Object? json) {
    final q = _repr(id);
    Never fail(String message) => throw ApiException(422, message);

    if (id.trim().isEmpty) {
      fail('question id must be a non-empty string, got $q');
    }
    if (json is! Map) {
      fail('question $q: definition must be a dict, got ${pyTypeName(json)}');
    }
    final t = json['type'];
    final type = t is String
        ? QuestionType.values.where((v) => v.name == t).firstOrNull
        : null;
    if (type == null) {
      fail(
        'question $q: unknown type ${pyRepr(t)}; '
        "use one of ['choice', 'noul', 'score']",
      );
    }
    if (!json.containsKey('instructions')) {
      fail(
        "question $q: no 'instructions'; add the text the model should answer",
      );
    }
    final ins = json['instructions'];
    if (ins == null) {
      fail(
        "question $q: 'instructions' must not be None; "
        'add the text the model should answer',
      );
    }
    if ((ins is String && ins.trim().isEmpty) ||
        (ins is List && ins.isEmpty) ||
        (ins is Map && ins.isEmpty)) {
      fail(
        "question $q: 'instructions' must not be empty; "
        'add the text the model should answer',
      );
    }
    // JSON only yields str, dict, list, int, float and bool here, all allowed.
    final text = ins is String ? ins : pythonJsonDumps(ins);

    final crit = json['criteria'];
    final Question question;
    int optionCount;
    switch (type) {
      case QuestionType.choice:
        if (crit is! Map && crit is! List) {
          fail(
            "question $q: a choice question takes 'criteria' as a dict of "
            'label -> description, or a list of labels',
          );
        }
        if ((crit as dynamic).isEmpty as bool) {
          fail('question $q: a choice question needs at least one criterion');
        }
        final rawLabels = crit is List ? crit : (crit as Map).keys.toList();
        for (var i = 0; i < rawLabels.length; i++) {
          final label = rawLabels[i];
          if (label != null &&
              label is! String &&
              label is! num &&
              label is! bool) {
            fail(
              'question $q: choice label $i is a ${pyTypeName(label)}; a '
              'label is rendered as option text and used as the answer key, '
              'so it must be a scalar (a string, number or bool), '
              'got ${pyRepr(label)}',
            );
          }
          if (label == null) {
            fail(
              'question $q: choice label $i is null; a label is rendered as '
              'option text and used as the answer key, so it must be a '
              'string, number or bool -- a null label renders as the text '
              '"None" while its answer key is "null"',
            );
          }
        }
        if (crit is List) {
          // Python dict keys: 1, 1.0 and True are one key.
          final firstAt = <Object, int>{};
          for (var i = 0; i < crit.length; i++) {
            final label = crit[i];
            final key = label is bool ? (label ? 1 : 0) : label as Object;
            final first = firstAt[key];
            if (first != null) {
              fail(
                'question $q: choice label $i (${pyRepr(label)}) repeats '
                'label $first; the labels are the answer keys, so every '
                'option needs its own (1, 1.0 and True are one key)',
              );
            }
            firstAt[key] = i;
          }
        }
        optionCount = rawLabels.length;
        question = ChoiceQuestion._(
          id,
          text,
          _order(json),
          labels: [for (final l in rawLabels) pyStr(l)],
          criteria: crit is List
              ? List<Object?>.filled(crit.length, null)
              : (crit as Map).values.toList(),
        );
      case QuestionType.score:
        if (crit is! List) {
          fail(
            "question $q: a score question takes 'criteria' as a list of "
            'level descriptions, index 0 first',
          );
        }
        if (crit.isEmpty) {
          fail('question $q: a score question needs at least one level');
        }
        final nullAt = crit.indexOf(null);
        if (nullAt >= 0) {
          fail(
            'question $q: score level $nullAt is null; give every level a '
            'description, index 0 first',
          );
        }
        optionCount = crit.length;
        question = ScoreQuestion._(
          id,
          text,
          _order(json),
          levels: List<Object?>.of(crit),
        );
      case QuestionType.noul:
        if (crit != null && crit is! Map) {
          fail(
            "question $q: a noul question takes 'criteria' as a dict with "
            "optional 'true'/'false' descriptions, or omits it",
          );
        }
        final byKey = <String, Object?>{};
        if (crit is Map) {
          for (final e in crit.entries) {
            byKey[pyStr(e.key).toLowerCase()] = e.value;
          }
          if (!byKey.keys.every((k) => k == 'true' || k == 'false')) {
            final keys = byKey.keys.toList()..sort();
            fail(
              "question $q: a noul question takes 'criteria' keyed only "
              "'true'/'false' (either or both, and omitted is fine), got "
              '${pyRepr(keys)}. Those keys are the option texts the model '
              'reads; any other key was silently dropped and replaced with '
              'the defaults. If you want the answer worded differently, keep '
              "'criteria' keyed 'true'/'false' and set 'labels' instead.",
            );
          }
        }
        optionCount = 2;
        question = NoulQuestion._(
          id,
          text,
          _order(json),
          falseCriterion: byKey['false'],
          trueCriterion: byKey['true'],
          labels: const ('false', 'true'),
        );
    }

    final order = json['option_order'];
    if (json.containsKey('option_order')) {
      final ints = [
        for (final i in order is List ? order : const []) ...[if (i is int) i],
      ]..sort();
      if (order is! List ||
          order.length != optionCount ||
          ints.length != optionCount ||
          !_isRange(ints)) {
        fail(
          "question $q: 'option_order' must be a permutation of "
          'range($optionCount) -- one slot per option, each option once -- '
          'got ${pyRepr(order)}',
        );
      }
    }
    if (json.containsKey('labels')) {
      if (type != QuestionType.noul) {
        fail("question $q: 'labels' is only supported for noul questions");
      }
      final resolved = resolveNoulLabels(json['labels']);
      if (resolved == null) fail('question $q: $noulLabelsMessage');
      return NoulQuestion._(
        id,
        text,
        _order(json),
        falseCriterion: (question as NoulQuestion).falseCriterion,
        trueCriterion: question.trueCriterion,
        labels: resolved,
      );
    }
    return question;
  }

  static String _repr(String s) => pyRepr(s);

  static bool _isRange(List<int> sorted) {
    for (var i = 0; i < sorted.length; i++) {
      if (sorted[i] != i) return false;
    }
    return true;
  }

  static List<int>? _order(Map<Object?, Object?> json) {
    final o = json['option_order'];
    if (o is! List || o.any((i) => i is! int)) return null;
    return [for (final i in o) i as int];
  }
}

/// A choice question: option labels with optional descriptions.
final class ChoiceQuestion extends Question {
  const ChoiceQuestion._(
    super.id,
    super.instructions,
    super.optionOrder, {
    required this.labels,
    required this.criteria,
  }) : super._();

  /// The option labels as option text (Python `str(label)`), in order.
  final List<String> labels;

  /// The description of each label; null when the label stands alone.
  final List<Object?> criteria;

  @override
  QuestionType get type => QuestionType.choice;

  @override
  List<String> renderOptions() => [
    for (var i = 0; i < labels.length; i++)
      criteria[i] == null || criteria[i] == ''
          ? labels[i]
          : '${labels[i]}: ${renderCriterion(criteria[i])}',
  ];
}

/// A score question: one description per level, index 0 first.
final class ScoreQuestion extends Question {
  const ScoreQuestion._(
    super.id,
    super.instructions,
    super.optionOrder, {
    required this.levels,
  }) : super._();

  /// The level descriptions.
  final List<Object?> levels;

  @override
  QuestionType get type => QuestionType.score;

  @override
  List<String> renderOptions() => [
    for (var i = 0; i < levels.length; i++)
      'level $i: ${renderCriterion(levels[i])}',
  ];
}

/// A yes/no question; the options are always `[false, true]`.
final class NoulQuestion extends Question {
  const NoulQuestion._(
    super.id,
    super.instructions,
    super.optionOrder, {
    required this.falseCriterion,
    required this.trueCriterion,
    required this.labels,
  }) : super._();

  /// The description of the false option, or null for the default.
  final Object? falseCriterion;

  /// The description of the true option, or null for the default.
  final Object? trueCriterion;

  /// The (false, true) option words, trimmed.
  final (String, String) labels;

  @override
  QuestionType get type => QuestionType.noul;

  @override
  List<String> renderOptions() => [
    '${labels.$1}: ${_describe(falseCriterion, 'no, the statement does not hold')}',
    '${labels.$2}: ${_describe(trueCriterion, 'yes, the statement holds')}',
  ];

  static String _describe(Object? criterion, String fallback) =>
      criterion == null || criterion == ''
      ? fallback
      : renderCriterion(criterion);
}
