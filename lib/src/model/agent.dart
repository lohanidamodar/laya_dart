/// One loaded checkpoint answering requests: laya's `system_one` path.
///
/// Ports `Agent.predict_batch` / `ONNXAgent._infer_batch` for a single state
/// (`laya/agent.py:1259-1448`, `laya/onnx_agent.py:704-790`), including the
/// marker-count check of `_encode_state` (`onnx_agent.py:603-641`).
library;

import 'dart:math' as math;

import '../errors.dart';
import '../inference/backend.dart';
import '../inference/batch.dart';
import '../python/py_str.dart';
import '../tokenizer/tokenizer.dart';
import 'checkpoint.dart';
import 'confidence.dart';
import 'decode.dart';
import 'question.dart';
import 'sequence.dart';

/// The `model` field of every result, as `laya-serve` (the torch `Agent`)
/// reports it.
const layaModelName = 'laya-rl-agent';

/// One loaded checkpoint: its tokenizer, its config and the backend that runs
/// its graph.
class LayaAgent {
  /// Wraps a loaded checkpoint. The agent owns [backend] only in the sense
  /// that a `LayaEngine` closes it on eviction; the agent never closes it.
  LayaAgent({
    required this.spec,
    required this.tokenizer,
    required this.config,
    required this.backend,
  });

  /// Which checkpoint this is.
  final CheckpointSpec spec;

  /// The checkpoint's tokenizer.
  final LayaTokenizer tokenizer;

  /// The checkpoint's windows and calibration.
  final CheckpointConfig config;

  /// Runs the checkpoint's graph.
  final InferenceBackend backend;

  /// Answers [questions] about [state] in one backend run.
  ///
  /// Returns laya's result, `{model, answers, usage}`, with the keys in laya's
  /// order: `answers` follows the order of [questions], and `usage` has
  /// `input_tokens`, `output_tokens`, `state_tokens`, `state_tokens_dropped`,
  /// `truncated` and `truncated_questions`, then `options` only when a
  /// question's options collapsed under the head budget. With no questions
  /// it returns empty answers and zero usage without running the backend.
  ///
  /// [state] is a string, or decoded JSON (a map or list); a list keeps its
  /// end when truncated. [maxLen] and [headMaxLen] override the config's
  /// windows. [minConfidence] (scalar or per-bucket) adds laya's abstention
  /// fields to every answer.
  ///
  /// Throws [ApiException] 422 with laya's message when a question's option
  /// markers do not all fit in [maxLen].
  Future<Map<String, Object?>> predict(
    Object state,
    Map<String, Question> questions, {
    int? maxLen,
    int? headMaxLen,
    ConfidenceThreshold? minConfidence,
  }) async {
    if (questions.isEmpty) {
      return {
        'model': layaModelName,
        'answers': <String, Object?>{},
        'usage': {'input_tokens': 0, 'output_tokens': 0},
      };
    }
    final len = maxLen ?? config.maxLen;
    final head = headMaxLen ?? config.headMaxLen;
    final ids = questions.keys.toList();
    final stateIds = encodeState(tokenizer, state);
    final rows = <SequenceRow>[];
    for (final qid in ids) {
      final q = questions[qid]!;
      final row = buildSequence(
        tokenizer,
        state,
        q,
        maxLen: len,
        headMaxLen: head,
        truncateLeft: state is List,
        stateIds: stateIds,
      );
      final options = q.renderOptions().length;
      if (row.markers.length != options) {
        throw ApiException(
          422,
          'question ${pyRepr(qid)}: only ${row.markers.length} of its '
          '$options option markers fit in max_len=$len with '
          'head_max_len=$head spent on the question; lower head_max_len, '
          'raise max_len, or use fewer options',
        );
      }
      rows.add(row);
    }

    final batch = InferenceBatch.collate(rows, [
      for (final qid in ids) questions[qid]!.type,
    ], padId: tokenizer.padId);
    final out = await backend.run(batch);

    final answers = <String, Object?>{
      for (var r = 0; r < ids.length; r++)
        ids[r]: decodeAnswer(
          questions[ids[r]]!,
          out.logitsFor(r, rows[r].markers.length),
          out.actFor(r),
          config,
          minConfidence: minConfidence,
        ),
    };

    final dropped = rows
        .map((r) => r.truncation.stateTokensDropped)
        .reduce(math.max);
    final usage = <String, Object?>{
      'input_tokens': rows.fold<int>(0, (s, r) => s + r.ids.length),
      'output_tokens': 0,
      'state_tokens': rows.first.truncation.stateTokens,
      'state_tokens_dropped': dropped,
      'truncated': dropped > 0,
      'truncated_questions': [
        for (var r = 0; r < ids.length; r++)
          if (rows[r].truncation.truncated) ids[r],
      ],
    };
    // laya common.collapsed_options: only when a question lost options.
    final collapsed = <String, Object?>{
      for (var r = 0; r < ids.length; r++)
        if (rows[r].options.optionsDistinct < rows[r].options.options)
          ids[r]: {
            'total': rows[r].options.options,
            'distinct': rows[r].options.optionsDistinct,
            'tokens_per_option': rows[r].options.tokensPerOption,
          },
    };
    if (collapsed.isNotEmpty) usage['options'] = collapsed;
    return {'model': layaModelName, 'answers': answers, 'usage': usage};
  }
}
