/// Collation of encoded rows into the decision graph's input tensors.
///
/// Ports laya's `collate_items` (`laya/common.py:775-816`) for inference: no
/// targets or labels.
library;

import 'dart:math' as math;
import 'dart:typed_data';

import 'package:meta/meta.dart';

import '../model/question.dart';
import '../model/sequence.dart';

/// Collated model inputs for N rows (laya `common.collate_items`).
///
/// Every tensor is row-major. [inputIds] and [attentionMask] are
/// `[rows, seqLen]`; [markerPos] and [markerMask] are `[rows, markers]`;
/// [qtype] is `[rows]`.
class InferenceBatch {
  InferenceBatch._(
    this.rows,
    this.seqLen,
    this.markers,
    this.inputIds,
    this.attentionMask,
    this.markerPos,
    this.markerMask,
    this.qtype,
  );

  /// Collates [rows], whose question types are [types] (one per row).
  ///
  /// Ids are right-padded with [padId] to the longest row; the attention mask
  /// is 1 over each row's ids and 0 over its padding. Marker positions are
  /// padded with 0 and the marker mask with 0 (false) to the most markers in
  /// any row. Throws [ArgumentError] for an empty batch or when [types] does
  /// not have one entry per row.
  ///
  /// Internal: `LayaAgent` builds batches; a backend only reads them.
  @internal
  factory InferenceBatch.collate(
    List<SequenceRow> rows,
    List<QuestionType> types, {
    required int padId,
  }) {
    if (rows.isEmpty) {
      throw ArgumentError.value(rows, 'rows', 'must not be empty');
    }
    if (types.length != rows.length) {
      throw ArgumentError.value(
        types.length,
        'types',
        'must have one entry per row (${rows.length})',
      );
    }
    final n = rows.length;
    final len = rows.map((r) => r.ids.length).reduce(math.max);
    final kmax = rows.map((r) => r.markers.length).reduce(math.max);
    final ids = Int64List(n * len)..fillRange(0, n * len, padId);
    final att = Int64List(n * len);
    final mpos = Int64List(n * kmax);
    final mmask = Uint8List(n * kmax);
    final qtype = Int64List(n);
    for (var i = 0; i < n; i++) {
      final r = rows[i];
      ids.setAll(i * len, r.ids);
      att.fillRange(i * len, i * len + r.ids.length, 1);
      mpos.setAll(i * kmax, r.markers);
      mmask.fillRange(i * kmax, i * kmax + r.markers.length, 1);
      qtype[i] = types[i].id;
    }
    return InferenceBatch._(n, len, kmax, ids, att, mpos, mmask, qtype);
  }

  /// The number of rows, N.
  final int rows;

  /// The longest row's length in tokens, L.
  final int seqLen;

  /// The most markers (options) in any row, K.
  final int markers;

  /// `input_ids`, int64 `[rows, seqLen]`.
  final Int64List inputIds;

  /// `attention_mask`, int64 `[rows, seqLen]`.
  final Int64List attentionMask;

  /// `marker_pos`, int64 `[rows, markers]`.
  final Int64List markerPos;

  /// `marker_mask`, bool `[rows, markers]`, one byte per element.
  final Uint8List markerMask;

  /// `qtype`, int64 `[rows]`: each row's [QuestionType.id].
  final Int64List qtype;
}
