/// laya-serve's request guards (`laya/serve.py`): size limits, the body
/// controls' validators and the lone-surrogate walk, with its texts.
library;

import '../errors.dart';
import '../json/python_json.dart';
import '../model/confidence.dart';
import '../python/py_str.dart';
import 'body.dart' show hasUnpairedSurrogate;

/// At most this many questions per request (`MAX_QUESTIONS`).
const maxQuestions = 64;

/// At most this many characters of serialized state (`MAX_STATE_CHARS`).
const maxStateChars = 50000;

/// At most this many body bytes (`MAX_BODY_BYTES`, 2 MiB).
const maxBodyBytes = 2 * 1024 * 1024;

/// At most this many options per choice question (`MAX_CHOICE_OPTIONS`).
const maxChoiceOptions = 100;

/// At most this many levels per score question (`MAX_SCORE_LEVELS`).
const maxScoreLevels = 32;

/// At most this many options across all questions (`MAX_TOTAL_OPTIONS`).
const maxTotalOptions = 512;

/// How many values the state lower-bound walk inspects
/// (`_STATE_PROBE_VALUES`).
const _stateProbeValues = 64;

/// The 400 for a body holding an unpaired surrogate (`_LONE_SURROGATE_DETAIL`,
/// serve.py:617).
const loneSurrogateDetail =
    'request body contains an unpaired surrogate escape; '
    'those cannot be encoded as UTF-8';

/// The body keys a request may not send (`BODY_REFUSALS`, serve.py:104).
const bodyRefusals = [
  'hooks',
  'on_predict_start',
  'on_predict_end',
  'hooks_raise',
  'hooks_timeout',
];

/// Python's `len()` of a string: its code points, not its UTF-16 units.
int _pyLen(String s) => s.runes.length;

/// serve.py:429 `_state_length_lower_bound_over`: a lower bound on the
/// state's serialized length when that bound already exceeds [cap], else 0.
///
/// Sums string lengths, popping the most recently pushed value first; keys
/// are ignored. It gives up (0) at the first value that is neither a string
/// nor a container, and after [_stateProbeValues] values.
int stateLengthLowerBoundOver(Object? state, int cap) {
  var total = 0;
  var budget = _stateProbeValues;
  final stack = <Object?>[state];
  while (stack.isNotEmpty && budget > 0) {
    budget--;
    final item = stack.removeLast();
    if (item is String) {
      total += _pyLen(item);
      if (total > cap) return total;
    } else if (item is Map) {
      stack.addAll(item.values);
    } else if (item is List) {
      stack.addAll(item);
    } else {
      return 0;
    }
  }
  return 0;
}

/// serve.py:485 `_state_length`: the length of the text that will be
/// tokenized, the state itself for a string, else
/// `json.dumps(state, ensure_ascii=False)`.
///
/// (laya's 400 `'state' must be JSON-serializable` cannot happen here: a
/// decoded body is always serializable.)
int stateLength(Object state) {
  if (state is String) return _pyLen(state);
  final over = stateLengthLowerBoundOver(state, maxStateChars);
  if (over != 0) return over;
  return _pyLen(pythonJsonDumps(state));
}

/// serve.py:536 `_check_request_limits`: refuses an absent state, a
/// non-object `questions` (400), and oversized requests (413), and a score
/// question with a null level (422), in laya's order.
void checkRequestLimits(Object? state, Object? questions) {
  if (state == null) throw const ApiException(400, "'state' is required");
  if (questions is! Map) {
    throw const ApiException(400, "'questions' must be an object");
  }
  if (questions.length > maxQuestions) {
    throw ApiException(
      413,
      'too many questions (${questions.length} > $maxQuestions)',
    );
  }
  var totalOptions = 0;
  for (final MapEntry(key: qid, value: question) in questions.entries) {
    if (question is! Map) continue;
    final crit = question['criteria'];
    final qtype = question['type'];
    if (qtype == 'choice' && (crit is Map || crit is List)) {
      final count = crit is Map ? crit.length : (crit as List).length;
      totalOptions += count;
      if (count > maxChoiceOptions) {
        throw ApiException(
          413,
          'too many choice options for ${pyRepr(qid)} '
          '($count > $maxChoiceOptions)',
        );
      }
    } else if (qtype == 'score' && crit is List) {
      final count = crit.length;
      totalOptions += count;
      if (count > maxScoreLevels) {
        throw ApiException(
          413,
          'too many score levels for ${pyRepr(qid)} '
          '($count > $maxScoreLevels)',
        );
      }
      final hole = crit.indexOf(null);
      if (hole >= 0) {
        throw ApiException(
          422,
          'score question ${pyRepr(qid)} has a null level at index $hole; '
          'give every level a description',
        );
      }
    }
  }
  if (totalOptions > maxTotalOptions) {
    throw ApiException(
      413,
      'too many answer options across questions '
      '($totalOptions > $maxTotalOptions)',
    );
  }
  final length = stateLength(state);
  if (length > maxStateChars) {
    throw ApiException(413, 'state too large ($length > $maxStateChars chars)');
  }
}

/// serve.py:621 `_has_lone_surrogate`: whether any string in [value], key or
/// value, holds an unpaired surrogate. Iterative, so depth cannot overflow.
bool hasLoneSurrogate(Object? value) {
  final stack = <Object?>[value];
  while (stack.isNotEmpty) {
    final item = stack.removeLast();
    if (item is String) {
      if (hasUnpairedSurrogate(item)) return true;
    } else if (item is Map) {
      stack
        ..addAll(item.keys)
        ..addAll(item.values);
    } else if (item is List) {
      stack.addAll(item);
    }
  }
  return false;
}

/// serve.py:229 `_validate_budget_param`: `body[key]` as a token budget
/// override, null when absent or null; 422 unless a positive integer no
/// larger than [cap].
int? validateBudgetParam(Map<Object?, Object?> body, String key, int cap) {
  final value = body[key];
  if (value == null) return null;
  if (value is! int) throw ApiException(422, '$key must be an integer');
  if (value <= 0) throw ApiException(422, '$key must be a positive integer');
  if (value > cap) {
    throw ApiException(422, '$key exceeds server limit ($value > $cap)');
  }
  return value;
}

/// serve.py:250 `_refuse_body_refusals`: 422 naming every hook control the
/// body sends (non-null), sorted.
void refuseBodyRefusals(Map<Object?, Object?> body) {
  final given = [
    for (final key in bodyRefusals)
      if (body[key] != null) key,
  ]..sort();
  if (given.isNotEmpty) {
    throw ApiException(
      422,
      '${given.join(', ')} run inside the server process and cannot be sent '
      'to this endpoint; install them where laya-serve runs, or drop them',
    );
  }
}

/// serve.py:271 `_validate_language_param`: `body[key]` as a language code,
/// null when absent or null; 422 unless a string.
String? validateLanguageParam(Map<Object?, Object?> body, String key) {
  final value = body[key];
  if (value == null) return null;
  if (value is! String) {
    throw ApiException(
      422,
      '$key must be a language code string such as "de", or null',
    );
  }
  return value;
}

/// serve.py:290 `_validate_min_confidence`: the body's `min_confidence`
/// through laya's `check_min_confidence`, null when absent or null; 422 with
/// laya's message otherwise.
ConfidenceThreshold? validateMinConfidence(Map<Object?, Object?> body) {
  final value = body['min_confidence'];
  if (value == null) return null;
  try {
    return ConfidenceThreshold.check(value);
  } on ArgumentError catch (e) {
    throw ApiException(422, '${e.message}');
  }
}
