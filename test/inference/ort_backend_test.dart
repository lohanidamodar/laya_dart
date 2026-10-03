@Tags(['ort'])
library;

import 'dart:typed_data';

import 'package:laya_server/src/inference/batch.dart';
import 'package:laya_server/src/inference/ort/ort_library.dart';
import 'package:laya_server/src/inference/ort/ort_session.dart';
import 'package:laya_server/src/inference/ort_backend.dart';
import 'package:laya_server/src/model/question.dart';
import 'package:laya_server/src/model/sequence.dart';
import 'package:test/test.dart';

const _echo = 'test/fixtures/ort/echo.onnx';
const _stub = 'test/fixtures/ort/laya_stub.onnx';

/// A startup failure must surface, not hang.
const _startup = Duration(seconds: 30);

Map<String, OrtInput> _echoInputs(int seed) => {
  'ids': OrtInt64Input(
    Int64List.fromList([for (var i = 0; i < 6; i++) seed * 10 + i]),
    [2, 3],
  ),
  'mask': OrtBoolInput(
    Uint8List.fromList(seed.isEven ? [1, 0, 0, 1] : [0, 1, 1, 0]),
    [2, 2],
  ),
};

List<double> _idsF(int seed) => [
  for (var i = 0; i < 6; i++) (seed * 10 + i).toDouble(),
];

Matcher _closedError() => throwsA(
  isA<StateError>().having((e) => e.message, 'message', 'backend closed'),
);

SequenceRow _row(List<int> ids, List<int> markers) => SequenceRow(
  ids: ids,
  markers: markers,
  options: OptionStats(
    options: markers.length,
    optionsDistinct: markers.length,
    tokensPerOption: null,
  ),
  truncation: TruncationStats(stateTokens: 0, stateTokensUsed: 0),
);

void main() {
  group('OrtBackend.openRaw', () {
    test('a run round-trips through the worker isolate', () async {
      final b = await OrtBackend.openRaw(_echo, intraOpThreads: 1);
      addTearDown(b.close);
      expect(b.inputNames, ['ids', 'mask']);
      expect(b.outputNames, ['ids_f', 'mask_f']);
      final out = await b.run(_echoInputs(0));
      expect(out['ids_f']!.shape, [2, 3]);
      expect(out['ids_f']!.data, _idsF(0));
      expect(out['mask_f']!.shape, [2, 2]);
      expect(out['mask_f']!.data, [1, 0, 0, 1]);
    });

    test(
      'concurrent runs are serialised and each gets its own result',
      () async {
        final b = await OrtBackend.openRaw(_echo, intraOpThreads: 1);
        addTearDown(b.close);
        final results = await Future.wait([
          for (var seed = 0; seed < 32; seed++) b.run(_echoInputs(seed)),
        ]);
        for (var seed = 0; seed < 32; seed++) {
          expect(results[seed]['ids_f']!.data, _idsF(seed), reason: '$seed');
          expect(
            results[seed]['mask_f']!.data,
            seed.isEven ? [1, 0, 0, 1] : [0, 1, 1, 0],
            reason: '$seed',
          );
        }
      },
    );

    test(
      'a run error comes back as OrtException and the worker lives on',
      () async {
        final b = await OrtBackend.openRaw(_echo, intraOpThreads: 1);
        addTearDown(b.close);
        await expectLater(
          b.run({..._echoInputs(0), 'nope': _echoInputs(0)['ids']!}),
          throwsA(isA<OrtException>()),
        );
        await expectLater(
          b.run({
            'ids': OrtInt64Input(Int64List.fromList([1, 2, 3]), [2, 3]),
            'mask': OrtBoolInput(Uint8List.fromList([1, 0, 0, 1]), [2, 2]),
          }),
          throwsA(
            isA<OrtException>().having(
              (e) => e.message,
              'message',
              contains('6 elements'),
            ),
          ),
        );
        expect((await b.run(_echoInputs(3)))['ids_f']!.data, _idsF(3));
      },
    );

    test('close shuts the isolate down; later runs throw', () async {
      final b = await OrtBackend.openRaw(_echo, intraOpThreads: 1);
      await b.run(_echoInputs(1));
      await b.close().timeout(_startup);
      await b.workerExited.timeout(_startup);
      await b.close();
      await expectLater(b.run(_echoInputs(1)), _closedError());
    });

    test(
      'runs in flight when close is called complete; later ones throw',
      () async {
        final b = await OrtBackend.openRaw(_echo, intraOpThreads: 1);
        final inFlight = [
          for (var seed = 0; seed < 8; seed++) b.run(_echoInputs(seed)),
        ];
        final closing = b.close();
        await expectLater(b.run(_echoInputs(9)), _closedError());
        final results = await Future.wait(inFlight);
        for (var seed = 0; seed < 8; seed++) {
          expect(results[seed]['ids_f']!.data, _idsF(seed));
        }
        await closing.timeout(_startup);
        await b.workerExited.timeout(_startup);
      },
    );

    test('a bad model path fails open with OrtException', () async {
      await expectLater(
        OrtBackend.openRaw('missing.onnx').timeout(_startup),
        throwsA(
          isA<OrtException>().having(
            (e) => e.message,
            'message',
            contains('missing.onnx'),
          ),
        ),
      );
    });

    test('a missing library fails open with OrtException', () async {
      await expectLater(
        OrtBackend.openRaw(
          _echo,
          ortLibraryPath: 'no-such-onnxruntime.dll',
        ).timeout(_startup),
        throwsA(
          isA<OrtException>()
              .having((e) => e.code, 'code', -1)
              .having((e) => e.message, 'message', contains('LAYA_ORT_LIB')),
        ),
      );
    });
  });

  group('OrtBackend.open', () {
    test('rejects a graph without the Laya inputs, naming both sets', () async {
      await expectLater(
        OrtBackend.open(_echo).timeout(_startup),
        throwsA(
          isA<OrtException>().having(
            (e) => e.message,
            'message',
            allOf(
              contains('input_ids'),
              contains('attention_mask'),
              contains('marker_pos'),
              contains('marker_mask'),
              contains('qtype'),
              contains('ids, mask'),
            ),
          ),
        ),
      );
    });

    test('a bad model path fails open with OrtException', () async {
      await expectLater(
        OrtBackend.open('missing.onnx').timeout(_startup),
        throwsA(isA<OrtException>()),
      );
    });

    // laya_stub.onnx has the Laya graph's signature, with outputs that are
    // arithmetic over every input (see tool/reference/laya_stub_model.py):
    //   logits[i, j]     = marker_pos[i, j] + 100 * qtype[i]
    //   act_logits[i, 0] = sum(input_ids[i] * attention_mask[i])
    //   act_logits[i, 1] = sum(marker_mask[i])
    test('feeds a collated batch to every Laya input', () async {
      final b = await OrtBackend.open(_stub, intraOpThreads: 1);
      addTearDown(b.close);
      final batch = InferenceBatch.collate(
        [
          _row([5, 6, 7, 8, 9], [2, 3]),
          _row([10, 11, 12, 13, 14, 15, 16], [1, 3, 5]),
        ],
        [QuestionType.score, QuestionType.noul],
        padId: 1,
      );
      final out = await b.run(batch);
      expect(out.markers, 3);
      expect(out.logitsFor(0, 3), [102, 103, 100]);
      expect(out.logitsFor(1, 3), [201, 203, 205]);
      expect(out.logitsFor(0, 2), [102, 103]);
      expect(out.actFor(0), [35, 2]);
      expect(out.actFor(1), [91, 3]);
    });

    test('concurrent batches each get their own output', () async {
      final b = await OrtBackend.open(_stub, intraOpThreads: 1);
      addTearDown(b.close);
      final outs = await Future.wait([
        for (var i = 0; i < 16; i++)
          b.run(
            InferenceBatch.collate(
              [
                _row([i, i], [i % 2]),
              ],
              [QuestionType.choice],
              padId: 0,
            ),
          ),
      ]);
      for (var i = 0; i < 16; i++) {
        expect(outs[i].logitsFor(0, 1), [i % 2], reason: '$i');
        expect(outs[i].actFor(0), [2 * i, 1], reason: '$i');
      }
    });

    test('run after close throws StateError', () async {
      final b = await OrtBackend.open(_stub, intraOpThreads: 1);
      await b.close();
      await expectLater(
        b.run(
          InferenceBatch.collate(
            [
              _row([1], [0]),
            ],
            [QuestionType.choice],
            padId: 0,
          ),
        ),
        _closedError(),
      );
    });
  });
}
