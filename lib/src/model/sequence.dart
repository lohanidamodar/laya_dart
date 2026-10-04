/// Sequence building: the exact token rows fed to the model.
///
/// Ports laya's `build_head`, `build_sequence` and `state_room`
/// (`laya/common.py`).
library;

import 'dart:math' as math;

import '../json/python_json.dart';
import '../tokenizer/tokenizer.dart';
import 'question.dart';

/// The most tokens one option keeps.
const int _maxOptionTokens = 48;

/// What the head budget did to the options.
class OptionStats {
  /// Creates option stats.
  const OptionStats({
    required this.options,
    required this.optionsDistinct,
    required this.tokensPerOption,
  });

  /// How many options the question defines.
  final int options;

  /// How many still have a token span of their own after capping.
  final int optionsDistinct;

  /// The cap applied to each option, or null when none was.
  final int? tokensPerOption;

  /// The stats as laya reports them, with its key names and order.
  Map<String, Object?> toJson() => {
    'options': options,
    'options_distinct': optionsDistinct,
    'tokens_per_option': tokensPerOption,
  };
}

/// What clamping the state to the remaining room did.
class TruncationStats {
  /// Creates truncation stats.
  const TruncationStats({
    required this.stateTokens,
    required this.stateTokensUsed,
  });

  /// The state's length in tokens.
  final int stateTokens;

  /// How many of them the row kept.
  final int stateTokensUsed;

  /// How many were cut.
  int get stateTokensDropped => stateTokens - stateTokensUsed;

  /// Whether any were cut.
  bool get truncated => stateTokensUsed < stateTokens;

  /// The stats as laya reports them, with its key names and order.
  Map<String, Object?> toJson() => {
    'state_tokens': stateTokens,
    'state_tokens_used': stateTokensUsed,
    'state_tokens_dropped': stateTokensDropped,
    'truncated': truncated,
  };
}

/// One encoded row: a question over a state.
class SequenceRow {
  /// Creates a row.
  const SequenceRow({
    required this.ids,
    required this.markers,
    required this.options,
    required this.truncation,
  });

  /// The token ids.
  final List<int> ids;

  /// The `[MASK]` position of each option, in option order.
  final List<int> markers;

  /// What the head budget did to the options.
  final OptionStats options;

  /// What the state clamp did.
  final TruncationStats truncation;
}

/// The question half of a sequence.
class _Head {
  _Head(this.ids, this.markers, this.stats);
  final List<int> ids;
  final List<int> markers;
  final OptionStats stats;
}

_Head _buildHead(LayaTokenizer tok, Question q, int headMaxLen) {
  final opts = q.renderOptions();
  final order = q.optionOrder ?? [for (var i = 0; i < opts.length; i++) i];
  final mask = tok.maskToken;
  final ins = q.instructions.replaceAll(mask, ' ');
  var headIds = tok.encode('${q.type.name} question: $ins');
  var optIds = <List<int>>[];
  for (final i in order) {
    final t = tok.encode(' ${opts[i].replaceAll(mask, ' ')}');
    optIds.add([
      tok.maskId,
      ...(t.length > _maxOptionTokens ? t.sublist(0, _maxOptionTokens) : t),
    ]);
  }
  int total() => optIds.fold(0, (s, o) => s + o.length);
  var budget = headMaxLen - total();
  int? perOption;
  if (budget < 16) {
    final per = math.max(4, (headMaxLen - 16) ~/ math.max(1, optIds.length));
    perOption = per;
    optIds = [for (final o in optIds) o.length > per ? o.sublist(0, per) : o];
    budget = headMaxLen - total();
  }
  final keep = math.max(8, budget);
  if (headIds.length > keep) headIds = headIds.sublist(0, keep);
  final ids = <int>[tok.clsId, ...headIds, tok.sepId];
  final markers = <int>[];
  for (final o in optIds) {
    markers.add(ids.length);
    ids.addAll(o);
  }
  ids.add(tok.sepId);
  final distinct = {for (final o in optIds) o.join(',')}.length;
  return _Head(
    ids,
    markers,
    OptionStats(
      options: optIds.length,
      optionsDistinct: distinct,
      tokensPerOption: perOption,
    ),
  );
}

/// The state as text: a string as-is, anything else Python `json.dumps`.
String _serializeState(Object state) =>
    state is String ? state : pythonJsonDumps(state);

/// Tokenizes the serialized state once (mask token replaced by a space).
List<int> encodeState(LayaTokenizer tok, Object state) =>
    tok.encode(_serializeState(state).replaceAll(tok.maskToken, ' '));

/// laya `build_sequence` (with option and truncation stats).
///
/// Format: `[CLS] TYPE instructions [SEP] [MASK] opt0 [MASK] opt1 ... [SEP]
/// state [SEP]`. The question's own `optionOrder` is applied when present.
/// [stateIds] lets a caller tokenize the state once for every question.
/// [truncateLeft] keeps the end of the state instead of the start (laya does
/// this for conversation turn lists).
SequenceRow buildSequence(
  LayaTokenizer tok,
  Object state,
  Question q, {
  required int maxLen,
  required int headMaxLen,
  bool truncateLeft = false,
  List<int>? stateIds,
}) {
  final head = _buildHead(tok, q, headMaxLen);
  final room = math.max(0, maxLen - head.ids.length - 1);
  final all = stateIds ?? encodeState(tok, state);
  final st = truncateLeft
      ? all.sublist(math.max(0, all.length - room))
      : all.sublist(0, math.min(room, all.length));
  var ids = [...head.ids, ...st, tok.sepId];
  if (ids.length > maxLen) ids = ids.sublist(0, maxLen);
  return SequenceRow(
    ids: ids,
    markers: [
      for (final m in head.markers)
        if (m < maxLen) m,
    ],
    options: head.stats,
    truncation: TruncationStats(
      stateTokens: all.length,
      stateTokensUsed: st.length,
    ),
  );
}

/// laya `state_room`: how many state tokens [q] leaves inside [maxLen].
int stateRoom(
  LayaTokenizer tok,
  Question q, {
  required int maxLen,
  required int headMaxLen,
}) => math.max(0, maxLen - _buildHead(tok, q, headMaxLen).ids.length - 1);
