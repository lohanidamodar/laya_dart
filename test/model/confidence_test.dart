import 'package:laya_server/laya_server.dart';
import 'package:test/test.dart';

/// The message `check_min_confidence` raises, as `ConfidenceThreshold.check`
/// throws it.
String _error(Object? value) {
  try {
    ConfidenceThreshold.check(value);
  } on ArgumentError catch (e) {
    return '${e.message}';
  }
  fail('accepted $value');
}

void main() {
  group('ConfidenceThreshold.check is laya check_min_confidence', () {
    test('accepts a number in [0, 1] as a scalar threshold', () {
      expect(ConfidenceThreshold.check(0), const ConfidenceThreshold.scalar(0));
      expect(ConfidenceThreshold.check(1), const ConfidenceThreshold.scalar(1));
      expect(
        ConfidenceThreshold.check(0.25),
        const ConfidenceThreshold.scalar(0.25),
      );
      expect(ConfidenceThreshold.check(0.5).toJson(), 0.5);
    });

    test('accepts a non-empty map of bucket -> number', () {
      final t = ConfidenceThreshold.check({'choice:3-5': 1, 'default': 0.2});
      expect(t, isA<BucketThresholds>());
      expect(t.toJson(), {'choice:3-5': 1.0, 'default': 0.2});
    });

    test('refuses with laya\'s texts (confidence.py)', () {
      const scalar = 'min_confidence must be a float in [0.0, 1.0], got';
      expect(_error(2), '$scalar 2');
      expect(_error(-0.1), '$scalar -0.1');
      expect(_error(1.5), '$scalar 1.5');
      expect(_error(true), '$scalar True');
      expect(_error('0.5'), "$scalar '0.5'");
      expect(_error([0.5]), '$scalar [0.5]');
      expect(_error(null), '$scalar None');
      expect(_error(double.infinity), '$scalar inf');
      expect(_error(double.negativeInfinity), '$scalar -inf');
      expect(_error(double.nan), '$scalar nan');
      expect(
        _error(<String, Object?>{}),
        'a min_confidence map must be a non-empty dict of bucket -> float, '
        'got {}',
      );
      expect(_error({'choice:2': 2}), '$scalar 2');
      expect(_error({'default': false}), '$scalar False');
      expect(
        _error({1: 0.5}),
        "min_confidence map keys must be strings like 'choice:3-5', got 1",
      );
    });
  });

  group('resolve is laya resolve_min_confidence', () {
    final map = ConfidenceThreshold.check({
      'choice:2': 0.9,
      'choice:3-5': 0.5,
      'choice:6-10': 0.6,
      'choice:11+': 0.7,
      'score:3-5': 0.4,
      'noul:2': 0.8,
    });
    Map<String, Object?> answer(String type, int k) => {
      'type': type,
      if (k > 0) 'probabilities': {for (var i = 0; i < k; i++) '$i': 0.1},
    };

    test('buckets by type and option count', () {
      expect(map.resolve(answer('choice', 2)), 0.9);
      expect(map.resolve(answer('choice', 1)), 0.9);
      expect(map.resolve(answer('choice', 5)), 0.5);
      expect(map.resolve(answer('choice', 6)), 0.6);
      expect(map.resolve(answer('choice', 10)), 0.6);
      expect(map.resolve(answer('choice', 11)), 0.7);
      expect(map.resolve(answer('score', 4)), 0.4);
      // noul answers carry no probabilities: two options.
      expect(map.resolve(answer('noul', 0)), 0.8);
    });

    test('falls back to "default", then 0.0', () {
      expect(map.resolve(answer('score', 2)), 0.0);
      final withDefault = ConfidenceThreshold.check({'default': 0.3});
      expect(withDefault.resolve(answer('score', 2)), 0.3);
      // No bucket at all: a choice without probabilities, an unknown type.
      expect(withDefault.resolve(answer('choice', 0)), 0.3);
      expect(withDefault.resolve(answer('other', 3)), 0.3);
    });

    test('a scalar threshold applies to every answer', () {
      expect(
        const ConfidenceThreshold.scalar(0.4).resolve(answer('noul', 0)),
        0.4,
      );
    });
  });

  group('gate is laya apply_confidence_gate on one answer', () {
    test('a map gates each answer at its bucket and echoes it', () {
      final t = ConfidenceThreshold.check({'choice:2': 0.9, 'default': 0.1});
      final low = {
        'type': 'choice',
        'probabilities': {'a': 0.6, 'b': 0.4},
        'answer_confidence': 0.6,
      };
      t.gate(low);
      expect(low['low_confidence'], isTrue);
      expect(low['abstention'], 'abstained');
      expect(low['abstention_threshold'], 0.9);

      final passed = {
        'type': 'noul',
        'noul': 0.6,
        'confidence': 0.6,
        'answer_confidence': 0.6,
      };
      t.gate(passed);
      expect(passed.containsKey('low_confidence'), isFalse);
      expect(passed['abstention'], 'passed');
      expect(passed['abstention_threshold'], 0.1);
    });

    test('no usable confidence is unevaluated', () {
      final a = <String, Object?>{'type': 'noul', 'noul': 0.5};
      ConfidenceThreshold.check({'default': 0.5}).gate(a);
      expect(a['abstention'], 'unevaluated');
      expect(a['abstention_threshold'], 0.5);
    });

    test('a scalar 0.0 passes everything and still reports', () {
      final a = <String, Object?>{'type': 'noul', 'answer_confidence': 0.0};
      const ConfidenceThreshold.scalar(0).gate(a);
      expect(a.containsKey('low_confidence'), isFalse);
      expect(a['abstention'], 'passed');
      expect(a['abstention_threshold'], 0.0);
    });
  });
}
