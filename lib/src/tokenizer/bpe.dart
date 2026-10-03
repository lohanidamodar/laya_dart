import 'dart:typed_data';

import 'package:meta/meta.dart';

/// Rank-ordered BPE over pre-split words.
///
/// Mirrors Hugging Face `tokenizers` BPE (`Word::merge_all`): the adjacent pair with the lowest
/// merge rank is merged first, the leftmost one on ties, and merging continues one pair at a
/// time until no adjacent pair has a rank. A pair a merge creates competes immediately, so it can
/// beat the remaining occurrences of the pair that created it.
class BpeModel {
  /// Builds the model from a vocab and merges in rank order (rank = index in [merges]).
  ///
  /// A pair listed twice takes its later rank, as HF does. Throws a [FormatException] when a
  /// merge's parts or its result are missing from [vocab], which HF also refuses to load.
  BpeModel({
    required Map<String, int> vocab,
    required List<(String, String)> merges,
  }) : _vocab = vocab {
    for (var rank = 0; rank < merges.length; rank++) {
      final (a, b) = merges[rank];
      for (final token in [a, b, a + b]) {
        if (!vocab.containsKey(token)) {
          throw FormatException(
            'merge "$a $b" uses "$token", which is not in the vocab',
          );
        }
      }
      (_ranks[a] ??= <String, int>{})[b] = rank;
    }
  }

  /// Word length, in symbols, at or above which [mergeWord] uses the heap.
  ///
  /// Below it a rescan per merge is cheapest; above it the rescan's quadratic cost shows (an
  /// unspaced CJK paragraph arrives as one word).
  static const heapMinLength = 32;

  final Map<String, int> _vocab;

  /// Merge rank by left symbol, then right symbol.
  final Map<String, Map<String, int>> _ranks = {};

  /// Looks up a token id; null when absent.
  int? idOf(String token) => _vocab[token];

  /// Merges [symbols] (one word's initial symbols) and returns the final symbols.
  List<String> mergeWord(List<String> symbols) => symbols.length < heapMinLength
      ? mergeWordSimple(symbols)
      : mergeWordHeap(symbols);

  int? _rank(String a, String b) => _ranks[a]?[b];

  /// The rescan form of [mergeWord]: find the lowest-ranked leftmost pair, merge it, repeat.
  @visibleForTesting
  List<String> mergeWordSimple(List<String> symbols) {
    final word = List.of(symbols);
    while (word.length > 1) {
      var best = -1;
      var at = -1;
      for (var i = 0; i < word.length - 1; i++) {
        final rank = _rank(word[i], word[i + 1]);
        if (rank != null && (best < 0 || rank < best)) {
          best = rank;
          at = i;
        }
      }
      if (at < 0) break;
      word
        ..[at] = word[at] + word[at + 1]
        ..removeAt(at + 1);
    }
    return word;
  }

  /// The heap form of [mergeWord], O(n log n) in the word length.
  ///
  /// The word is a doubly linked list over the original slots, and candidate pairs sit in a
  /// binary min-heap keyed by (rank, slot). A slot only ever absorbs its right neighbour, so slot
  /// order is text order and the slot tie-break is the leftmost rule. Each slot carries a
  /// version that moves whenever its pair changes; a popped entry whose version has moved on is
  /// stale and skipped.
  @visibleForTesting
  List<String> mergeWordHeap(List<String> symbols) {
    final n = symbols.length;
    final tok = List.of(symbols);
    if (n <= 1) return tok;
    final next = Int32List(n);
    final prev = Int32List(n);
    final version = Int32List(n);
    for (var i = 0; i < n; i++) {
      prev[i] = i - 1;
      next[i] = i + 1 < n ? i + 1 : -1;
    }
    final heap = _PairHeap(3 * n);

    void offer(int slot) {
      final j = next[slot];
      if (j < 0) return;
      final rank = _rank(tok[slot], tok[j]);
      if (rank != null) heap.push(rank, slot, version[slot]);
    }

    for (var i = 0; i < n - 1; i++) {
      offer(i);
    }
    while (heap.isNotEmpty) {
      final (slot, ver) = heap.pop();
      // A slot that was absorbed has its version bumped below, so this also skips dead slots.
      if (version[slot] != ver) continue;
      final j = next[slot];
      tok[slot] = tok[slot] + tok[j];
      version[j]++;
      final k = next[j];
      next[slot] = k;
      if (k >= 0) prev[k] = slot;
      version[slot]++;
      offer(slot);
      final p = prev[slot];
      if (p >= 0) {
        version[p]++;
        offer(p);
      }
    }
    return [for (var i = 0; i >= 0; i = next[i]) tok[i]];
  }
}

/// A binary min-heap of (rank, slot, version) entries ordered by rank, then slot.
class _PairHeap {
  _PairHeap(int capacity)
    : _rank = Int32List(capacity),
      _slot = Int32List(capacity),
      _ver = Int32List(capacity);

  Int32List _rank;
  Int32List _slot;
  Int32List _ver;
  int _length = 0;

  bool get isNotEmpty => _length > 0;

  bool _less(int rankA, int slotA, int rankB, int slotB) =>
      rankA < rankB || (rankA == rankB && slotA < slotB);

  void push(int rank, int slot, int ver) {
    if (_length == _rank.length) {
      _rank = Int32List(_length * 2)..setAll(0, _rank);
      _slot = Int32List(_length * 2)..setAll(0, _slot);
      _ver = Int32List(_length * 2)..setAll(0, _ver);
    }
    var c = _length++;
    while (c > 0) {
      final p = (c - 1) >> 1;
      if (!_less(rank, slot, _rank[p], _slot[p])) break;
      _rank[c] = _rank[p];
      _slot[c] = _slot[p];
      _ver[c] = _ver[p];
      c = p;
    }
    _rank[c] = rank;
    _slot[c] = slot;
    _ver[c] = ver;
  }

  /// Removes the smallest entry and returns its (slot, version).
  (int, int) pop() {
    final top = (_slot[0], _ver[0]);
    _length--;
    final rank = _rank[_length];
    final slot = _slot[_length];
    final ver = _ver[_length];
    var c = 0;
    while (true) {
      final l = 2 * c + 1;
      if (l >= _length) break;
      final r = l + 1;
      final m = r < _length && _less(_rank[r], _slot[r], _rank[l], _slot[l])
          ? r
          : l;
      if (!_less(_rank[m], _slot[m], rank, slot)) break;
      _rank[c] = _rank[m];
      _slot[c] = _slot[m];
      _ver[c] = _ver[m];
      c = m;
    }
    _rank[c] = rank;
    _slot[c] = slot;
    _ver[c] = ver;
    return top;
  }
}
