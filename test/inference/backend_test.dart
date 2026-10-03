import 'dart:typed_data';

import 'package:laya_server/src/inference/backend.dart';
import 'package:laya_server/src/inference/batch.dart';
import 'package:laya_server/src/model/question.dart';
import 'package:laya_server/src/model/sequence.dart';
import 'package:test/test.dart';

InferenceBatch _batch() => InferenceBatch.collate(
  [
    for (final markers in [
      [1, 2],
      [1, 2, 3],
    ])
      SequenceRow(
        ids: [0, 4, 4, 4, 2],
        markers: markers,
        options: OptionStats(
          options: markers.length,
          optionsDistinct: markers.length,
          tokensPerOption: null,
        ),
        truncation: TruncationStats(stateTokens: 0, stateTokensUsed: 0),
      ),
  ],
  [QuestionType.choice, QuestionType.choice],
  padId: 1,
);

void main() {
  group('InferenceOutput', () {
    final out = InferenceOutput(
      Float32List.fromList([0.1, 0.2, 0.3, 1.5, 2.5, 3.5]),
      Float32List.fromList([0.25, -0.25, 4, 5]),
      3,
    );

    test('logitsFor reads the first k of a row, as float32', () {
      expect(out.logitsFor(0, 2), [
        Float32List.fromList([0.1])[0],
        Float32List.fromList([0.2])[0],
      ]);
      expect(out.logitsFor(1, 3), [1.5, 2.5, 3.5]);
      expect(out.logitsFor(1, 3), isA<Float32List>());
    });

    test('actFor reads a row of act logits', () {
      expect(out.actFor(0), [0.25, -0.25]);
      expect(out.actFor(1), [4, 5]);
    });

    test('rejects out-of-range rows and k', () {
      expect(() => out.logitsFor(2, 1), throwsRangeError);
      expect(() => out.logitsFor(0, 4), throwsRangeError);
      expect(() => out.actFor(2), throwsRangeError);
    });
  });

  group('FakeBackend', () {
    test('logits come from the function, padded to the batch width', () async {
      final fake = FakeBackend(
        (row, markers) => [for (var i = 0; i < markers; i++) row * 10.0 + i],
      );
      final out = await fake.run(_batch());
      expect(out.markers, 3);
      expect(out.logitsFor(0, 3), [0, 1, 2]);
      expect(out.logitsFor(1, 3), [10, 11, 12]);
      expect(out.actFor(1), [0, 0]);
    });

    test('a short logits row is zero padded', () async {
      final fake = FakeBackend((row, markers) => [1]);
      final out = await fake.run(_batch());
      expect(out.logitsFor(0, 3), [1, 0, 0]);
    });

    test('a logits row wider than the batch is an error', () async {
      final fake = FakeBackend((row, markers) => [1, 2, 3, 4]);
      await expectLater(fake.run(_batch()), throwsArgumentError);
    });

    test('act logits come from the act function when given', () async {
      final fake = FakeBackend(
        (row, markers) => [0, 0, 0],
        act: (row) => [row.toDouble(), -1],
      );
      final out = await fake.run(_batch());
      expect(out.actFor(1), [1, -1]);
    });

    test('waits for the delay', () async {
      final fake = FakeBackend(
        (row, markers) => [0, 0, 0],
        delay: const Duration(milliseconds: 50),
      );
      final sw = Stopwatch()..start();
      await fake.run(_batch());
      expect(sw.elapsedMilliseconds, greaterThanOrEqualTo(45));
    });

    test('run after close throws StateError', () async {
      final fake = FakeBackend((row, markers) => [0, 0, 0]);
      await fake.close();
      await fake.close();
      await expectLater(
        fake.run(_batch()),
        throwsA(
          isA<StateError>().having(
            (e) => e.message,
            'message',
            'backend closed',
          ),
        ),
      );
    });
  });
}
