/// Decoding of the decision head's output into Jev answers.
///
/// Ports laya's `ONNXAgent._decode_answers` (`onnx_agent.py:642-702`), the act
/// softmax in `ONNXAgent._infer_batch` (`onnx_agent.py:759-762`), the helpers
/// in `common.py:664-725` and the abstention gate in `confidence.py`.
///
/// laya does this arithmetic in numpy on the float32 arrays ONNX Runtime
/// returns, and numpy keeps float32 in float32: an operation between a float32
/// array or scalar and a Python float casts the Python float to float32
/// (NEP 50, numpy 2). Every step below that runs in float32 there runs in
/// float32 here, rounding through `f32`, with numpy's own float32 `exp` and
/// `log` kernels and its pairwise summation (`numpy_f32.dart`), so the
/// 4-decimal answers match laya exactly rather than nearly. The fixtures show
/// the difference: with correctly rounded float32 `exp`, a score case in
/// `test/fixtures/decode.json` rounds to 3.6536 where laya reports 3.6535.
/// The steps that leave float32 in laya (`float(...)`, the score's
/// `np.arange(k) * p`, the noul confidence) run in double here too.
library;

import 'dart:math' as math;
import 'dart:typed_data';

import '../json/python_json.dart';
import '../python/py_round.dart';
import 'checkpoint.dart';
import 'confidence.dart';
import 'numpy_f32.dart';
import 'question.dart';
import 'render.dart';

/// Decodes one question's head output into its Jev answer map, with the full
/// laya payload keys in laya's order.
///
/// [logits] is the question's row of the head's `logits` output, in slot
/// order; only its first k entries are read, k being the question's option
/// count (laya's marker count). [actLogits] is the row of `act_logits`.
///
/// With [minConfidence] set (scalar or per-bucket), the answer also carries the
/// gate's `abstention` and `abstention_threshold`, and `low_confidence: true`
/// when it abstained (laya `apply_confidence_gate`, applied per answer).
Map<String, Object?> decodeAnswer(
  Question q,
  List<double> logits,
  List<double> actLogits,
  CheckpointConfig cfg, {
  ConfidenceThreshold? minConfidence,
}) {
  final k = q.renderOptions().length;
  if (logits.length < k) {
    throw ArgumentError.value(
      logits.length,
      'logits',
      'has fewer entries than the question has options ($k)',
    );
  }
  if (actLogits.isEmpty) {
    throw ArgumentError.value(actLogits, 'actLogits', 'is empty');
  }

  // onnx_agent.py:651-656: `z = logits[r, :k] / t_scale`, then a
  // max-subtracted softmax. All float32: t_scale is cast to float32 first.
  final t = f32(cfg.temperatureFor(q.type, k));
  final z = Float32List(k);
  for (var i = 0; i < k; i++) {
    z[i] = f32(logits[i]) / t;
  }
  final p = _softmax(z);

  // onnx_agent.py:659, common.py:455 unpermute_probs.
  final order = q.optionOrder;
  final probs = order == null || order.length != k ? p : Float32List(k);
  if (!identical(probs, p)) {
    for (var s = 0; s < k; s++) {
      probs[order![s]] = p[s];
    }
  }

  final confidence = round4(_confidenceFromProbs(probs, k));
  final answerConfidence = round4(_answerConfidence(probs, k));
  // onnx_agent.py:668 reads the act softmax _infer_batch computed in float32.
  final action = {
    'act_probability': round4(_softmax(Float32List.fromList(actLogits))[0]),
  };

  final Map<String, Object?> answer;
  switch (q) {
    case ChoiceQuestion():
      final keys = q.keys;
      answer = {
        'type': 'choice',
        'choice': keys[_argmax(probs)],
        'probabilities': {
          for (var i = 0; i < k; i++) _jsonKey(keys[i]): round4(probs[i]),
        },
        'confidence': confidence,
        'answer_confidence': answerConfidence,
        'action': action,
      };
    case ScoreQuestion():
      // `(np.arange(k) * p).sum()`: int64 times float32 promotes to float64,
      // so the products and their (pairwise) sum are double.
      final weighted = Float64List(k);
      for (var i = 0; i < k; i++) {
        weighted[i] = i * probs[i];
      }
      answer = {
        'type': 'score',
        'score': round4(npPairwiseSum(weighted, inDouble: true)),
        'legend': {
          for (var i = 0; i < k; i++) '$i': renderCriterion(q.levels[i]),
        },
        'probabilities': {for (var i = 0; i < k; i++) '$i': round4(probs[i])},
        'confidence': confidence,
        'answer_confidence': answerConfidence,
        'action': action,
      };
    case NoulQuestion():
      // `float(p[1])` and `1.0 - float(p[1])` are double.
      final yes = probs[1];
      answer = {
        'type': 'noul',
        'noul': round4(yes),
        'confidence': round4(math.max(yes, 1.0 - yes)),
        'answer_confidence': answerConfidence,
        'action': action,
      };
  }
  minConfidence?.gate(answer);
  return answer;
}

/// Python `json.dumps`' text for a dict key: a string as is, a bool as
/// `true`/`false`, a float as its `repr`, an int in decimal.
String _jsonKey(Object key) => switch (key) {
  final String s => s,
  final bool b => b ? 'true' : 'false',
  final double d => pythonFloatRepr(d),
  _ => '$key',
};

/// `np.argmax`: the first maximum, where a NaN counts as the maximum.
int _argmax(List<double> p) {
  var best = 0;
  for (var i = 0; i < p.length; i++) {
    if (p[i].isNaN) return i;
    if (p[i] > p[best]) best = i;
  }
  return best;
}

/// `np.exp(z - z.max()); p / p.sum()` on a float32 row, in float32
/// (onnx_agent.py:656-658 for the option row, 760-761 for the act row).
Float32List _softmax(Float32List z) {
  var max = z[0];
  for (final v in z) {
    if (v > max || v.isNaN) max = v;
  }
  final e = Float32List(z.length);
  for (var i = 0; i < z.length; i++) {
    e[i] = npExpF32(f32(z[i] - max));
  }
  final sum = f32(npPairwiseSum(e));
  final p = Float32List(z.length);
  for (var i = 0; i < z.length; i++) {
    p[i] = e[i] / sum;
  }
  return p;
}

/// common.py:664 `answer_confidence`: `clip(max(p[:k]), 0, 1)`, float32.
double _answerConfidence(Float32List p, int k) {
  if (k < 1) return 1.0;
  var max = p[0];
  for (var i = 0; i < k; i++) {
    if (p[i] > max || p[i].isNaN) max = p[i];
  }
  return _clip01(max);
}

/// common.py:684 `confidence_from_probs`: `1 - H(p) / log(k)`, clipped.
///
/// Float32 throughout: the clip bound 1e-12 and `math.log(k)` are Python
/// floats meeting float32 values, so numpy casts them to float32.
double _confidenceFromProbs(Float32List p, int k) {
  if (k < 2) return 1.0;
  final floor = f32(1e-12);
  final terms = Float32List(k);
  for (var i = 0; i < k; i++) {
    final clipped = p[i] < floor ? floor : (p[i] > 1.0 ? 1.0 : p[i]);
    terms[i] = p[i] * npLogF32(clipped);
  }
  final ent = -f32(npPairwiseSum(terms));
  final ratio = f32(ent / f32(math.log(k)));
  return _clip01(f32(1.0 - ratio));
}

double _clip01(double v) => v < 0.0 ? 0.0 : (v > 1.0 ? 1.0 : v);
