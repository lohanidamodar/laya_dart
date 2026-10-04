import 'package:laya_server/src/inference/batch.dart';
import 'package:laya_server/src/model/question.dart';
import 'package:laya_server/src/model/sequence.dart';
import 'package:test/test.dart';

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
  // Hand-computed from laya common.collate_items (common.py:775-816):
  // n = 2, L = max len = 7, kmax = max markers = 3.
  //   ids  = full((n, L), pad_id); ids[i, :len] = row ids
  //   att  = zeros((n, L));        att[i, :len] = 1
  //   mpos = zeros((n, kmax));     mpos[i, :k] = markers
  //   mmask = zeros bool;          mmask[i, :k] = True
  //   qtype = [it.qtype for it in items]
  group('InferenceBatch.collate', () {
    final rows = [
      _row([5, 6, 7, 8, 9], [2, 3]),
      _row([10, 11, 12, 13, 14, 15, 16], [1, 3, 5]),
    ];
    final types = [QuestionType.choice, QuestionType.score];
    late InferenceBatch b;
    setUp(() => b = InferenceBatch.collate(rows, types, padId: 1));

    test('dimensions', () {
      expect(b.rows, 2);
      expect(b.seqLen, 7);
      expect(b.markers, 3);
    });

    test('input ids padded with padId', () {
      expect(b.inputIds, [5, 6, 7, 8, 9, 1, 1, 10, 11, 12, 13, 14, 15, 16]);
    });

    test('attention mask is ones then zeros', () {
      expect(b.attentionMask, [1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1]);
    });

    test('marker positions padded with 0', () {
      expect(b.markerPos, [2, 3, 0, 1, 3, 5]);
    });

    test('marker mask padded with 0', () {
      expect(b.markerMask, [1, 1, 0, 1, 1, 1]);
    });

    test('qtype per row', () {
      expect(b.qtype, [0, 1]);
      final noul = InferenceBatch.collate(
        [rows[0]],
        [QuestionType.noul],
        padId: 1,
      );
      expect(noul.qtype, [2]);
    });

    test('a single row is not padded', () {
      final one = InferenceBatch.collate(
        [rows[1]],
        [QuestionType.choice],
        padId: 0,
      );
      expect(one.inputIds, rows[1].ids);
      expect(one.attentionMask, everyElement(1));
      expect(one.markerPos, [1, 3, 5]);
      expect(one.markerMask, [1, 1, 1]);
    });

    test('rejects an empty batch and mismatched types', () {
      expect(
        () => InferenceBatch.collate([], [], padId: 0),
        throwsArgumentError,
      );
      expect(
        () => InferenceBatch.collate(rows, [QuestionType.choice], padId: 0),
        throwsArgumentError,
      );
    });
  });
}
