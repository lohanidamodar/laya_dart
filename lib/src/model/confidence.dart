/// The opt-in abstention threshold, `min_confidence`: laya's
/// `check_min_confidence` validator and `apply_confidence_gate`
/// (`laya/confidence.py`).
library;

import '../python/py_str.dart';

/// A validated `min_confidence`: one threshold for every answer
/// ([ScalarThreshold]), or one per option-count bucket ([BucketThresholds],
/// laya #394).
sealed class ConfidenceThreshold {
  const ConfidenceThreshold();

  /// One threshold, in 0 to 1, for every answer.
  const factory ConfidenceThreshold.scalar(double value) = ScalarThreshold;

  /// Per-bucket thresholds, in 0 to 1, keyed like `"choice:3-5"`, with an
  /// optional `"default"` for the buckets not named.
  const factory ConfidenceThreshold.buckets(Map<String, double> thresholds) =
      BucketThresholds;

  /// laya `check_min_confidence`: [value] decoded from JSON, as a threshold.
  ///
  /// A number in 0 to 1 gives a [ScalarThreshold]; a non-empty map of string
  /// keys to such numbers gives a [BucketThresholds]. Throws [ArgumentError]
  /// with laya's message for anything else (a bool is not a number here).
  static ConfidenceThreshold check(Object? value) {
    if (value is Map) {
      if (value.isEmpty) {
        throw ArgumentError(
          'a min_confidence map must be a non-empty dict of bucket -> float, '
          'got ${pyRepr(value)}',
        );
      }
      final out = <String, double>{};
      for (final MapEntry(:key, :value) in value.entries) {
        if (key is! String) {
          throw ArgumentError(
            "min_confidence map keys must be strings like 'choice:3-5', "
            'got ${pyRepr(key)}',
          );
        }
        out[key] = _checkOne(value);
      }
      return BucketThresholds(out);
    }
    return ScalarThreshold(_checkOne(value));
  }

  /// confidence.py `_check_one_threshold`.
  static double _checkOne(Object? v) {
    if (v is! num || !v.isFinite || v < 0.0 || v > 1.0) {
      throw ArgumentError(
        'min_confidence must be a float in [0.0, 1.0], got ${pyRepr(v)}',
      );
    }
    return v.toDouble();
  }

  /// laya `resolve_min_confidence`: the threshold [answer] is gated at.
  double resolve(Map<String, Object?> answer);

  /// laya `apply_confidence_gate` on one answer: adds `low_confidence: true`
  /// when the answer's confidence is below its threshold, then `abstention`
  /// (`passed`, `abstained` or `unevaluated`) and `abstention_threshold`.
  void gate(Map<String, Object?> answer) {
    final threshold = resolve(answer);
    final conf = _gateConfidence(answer);
    // flag_low_confidence: a scalar 0.0 flags nothing (nothing is below it).
    final skip = this is ScalarThreshold && threshold == 0.0;
    if (!skip && conf != null && conf < threshold) {
      answer['low_confidence'] = true;
    }
    answer['abstention'] = answer['low_confidence'] == true
        ? 'abstained'
        : conf == null
        ? 'unevaluated'
        : 'passed';
    answer['abstention_threshold'] = threshold;
  }

  /// The threshold as laya received it: a number or a map.
  Object toJson();
}

/// One threshold for every answer.
final class ScalarThreshold extends ConfidenceThreshold {
  /// Wraps [value], which the caller has checked is in 0 to 1.
  const ScalarThreshold(this.value);

  /// The threshold.
  final double value;

  @override
  double resolve(Map<String, Object?> answer) => value;

  @override
  Object toJson() => value;

  @override
  bool operator ==(Object other) =>
      other is ScalarThreshold && other.value == value;

  @override
  int get hashCode => value.hashCode;

  @override
  String toString() => 'ConfidenceThreshold.scalar($value)';
}

/// One threshold per option-count bucket (laya #394).
final class BucketThresholds extends ConfidenceThreshold {
  /// Wraps [thresholds], which the caller has checked are in 0 to 1.
  const BucketThresholds(this.thresholds);

  /// Thresholds keyed by bucket (`choice:2`, `choice:3-5`, `choice:6-10`,
  /// `choice:11+`, likewise `score:` and `noul:`), plus `default`.
  final Map<String, double> thresholds;

  /// The bucket's threshold, else `default`, else 0.0 (gate nothing).
  @override
  double resolve(Map<String, Object?> answer) {
    final key = _optionBucket(answer);
    if (key != null && thresholds.containsKey(key)) return thresholds[key]!;
    return thresholds['default'] ?? 0.0;
  }

  @override
  Object toJson() => Map.of(thresholds);

  @override
  bool operator ==(Object other) =>
      other is BucketThresholds &&
      other.thresholds.length == thresholds.length &&
      thresholds.entries.every((e) => other.thresholds[e.key] == e.value);

  @override
  int get hashCode => Object.hashAllUnordered(
    thresholds.entries.map((e) => Object.hash(e.key, e.value)),
  );

  @override
  String toString() => 'ConfidenceThreshold.buckets($thresholds)';
}

/// confidence.py `_option_bucket`: `"<type>:<size>"` from the answer's type
/// and its `probabilities` count (2 for a noul answer without them).
String? _optionBucket(Map<String, Object?> answer) {
  final type = answer['type'];
  if (type != 'choice' && type != 'score' && type != 'noul') return null;
  final probs = answer['probabilities'];
  final int k;
  if (probs is Map && probs.isNotEmpty) {
    k = probs.length;
  } else if (type == 'noul') {
    k = 2;
  } else {
    return null;
  }
  final size = k <= 2
      ? '2'
      : k <= 5
      ? '3-5'
      : k <= 10
      ? '6-10'
      : '11+';
  return '$type:$size';
}

/// confidence.py `_gate_confidence`: `answer_confidence`, else `confidence`,
/// else null when neither is a finite number.
double? _gateConfidence(Map<String, Object?> answer) {
  for (final key in ['answer_confidence', 'confidence']) {
    final v = answer[key];
    if (v is num && v.isFinite) return v.toDouble();
  }
  return null;
}
