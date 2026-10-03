import 'dart:math';

import 'package:laya_server/src/tokenizer/bpe.dart';
import 'package:test/test.dart';

void main() {
  test('lowest rank merges first, leftmost on ties', () {
    final bpe = BpeModel(
      vocab: {'a': 0, 'b': 1, 'ab': 2, 'abab': 3, 'ba': 4},
      merges: [('a', 'b'), ('ab', 'ab'), ('b', 'a')],
    );
    expect(bpe.mergeWord(['a', 'b', 'a', 'b']), ['abab']);
    expect(bpe.mergeWord(['b', 'a', 'b']), ['b', 'ab']);
  });

  test('long words take the heap path and agree with the simple path', () {
    final bpe = BpeModel(
      vocab: {'a': 0, 'aa': 1, 'aaaa': 2},
      merges: [('a', 'a'), ('aa', 'aa')],
    );
    expect(bpe.mergeWord(List.filled(40, 'a')), List.filled(10, 'aaaa'));
  });

  // Expected values observed with tokenizers 0.23.2 on a synthetic BPE whose merges are
  // [("ab","a"), ("a","b")]: HF merges one pair at a time, so a lower-ranked pair created by a
  // merge wins over the remaining occurrences of the pair that created it.
  test('merges one pair at a time, not every occurrence at once (HF)', () {
    final bpe = BpeModel(
      vocab: {'a': 0, 'b': 1, 'aba': 2, 'ab': 3},
      merges: [('ab', 'a'), ('a', 'b')],
    );
    expect(bpe.mergeWord(['a', 'b', 'a', 'b']), ['aba', 'b']);
    expect(bpe.mergeWord(['a', 'b', 'a', 'b', 'a', 'b']), ['aba', 'b', 'ab']);
    expect(
      bpe.mergeWord([
        for (var i = 0; i < 20; i++) ...['a', 'b'],
      ]),
      [
        for (var i = 0; i < 10; i++) ...['aba', 'b'],
      ],
    );
  });

  test('overlapping runs merge leftmost first (HF)', () {
    final bpe = BpeModel(
      vocab: {'a': 0, 'aa': 1, 'aaaa': 2},
      merges: [('a', 'a'), ('aa', 'aa')],
    );
    expect(bpe.mergeWord(['a', 'a', 'a']), ['aa', 'a']);
    expect(bpe.mergeWord(['a', 'a', 'a', 'a', 'a']), ['aaaa', 'a']);
  });

  test('a repeated merge takes its later rank (HF)', () {
    final bpe = BpeModel(
      vocab: {'a': 0, 'b': 1, 'c': 2, 'ab': 3, 'bc': 4},
      merges: [('a', 'b'), ('b', 'c'), ('a', 'b')],
    );
    expect(bpe.mergeWord(['a', 'b', 'c']), ['a', 'bc']);
  });

  test('empty, single and unmergeable words pass through', () {
    final bpe = BpeModel(vocab: {'a': 0, 'b': 1}, merges: const []);
    expect(bpe.mergeWord(const []), isEmpty);
    expect(bpe.mergeWord(['a']), ['a']);
    expect(bpe.mergeWord(['a', 'b', 'x']), ['a', 'b', 'x']);
  });

  test('idOf looks up the vocab', () {
    final bpe = BpeModel(vocab: {'a': 7}, merges: const []);
    expect(bpe.idOf('a'), 7);
    expect(bpe.idOf('z'), isNull);
  });

  test('a merge whose parts or result are not in the vocab is rejected', () {
    expect(
      () => BpeModel(vocab: {'a': 0, 'b': 1}, merges: [('a', 'b')]),
      throwsFormatException,
    );
    expect(
      () => BpeModel(vocab: {'a': 0, 'ab': 1}, merges: [('a', 'b')]),
      throwsFormatException,
    );
  });

  test('simple and heap paths agree on random tables at every length', () {
    final random = Random(42);
    const alphabet = ['a', 'b', 'c', 'd'];
    for (var table = 0; table < 40; table++) {
      // Random merges over the alphabet and the tokens they build.
      final vocab = <String, int>{
        for (final c in alphabet) c: alphabet.indexOf(c),
      };
      final merges = <(String, String)>[];
      final tokens = [...alphabet];
      for (var m = 0; m < 25; m++) {
        final a = tokens[random.nextInt(tokens.length)];
        final b = tokens[random.nextInt(tokens.length)];
        if (a.length + b.length > 6) continue;
        merges.add((a, b));
        if (!vocab.containsKey(a + b)) {
          vocab[a + b] = vocab.length;
          tokens.add(a + b);
        }
      }
      final bpe = BpeModel(vocab: vocab, merges: merges);
      for (var length = 0; length <= 80; length++) {
        final word = [
          for (var i = 0; i < length; i++)
            alphabet[random.nextInt(alphabet.length)],
        ];
        expect(
          bpe.mergeWordHeap(word),
          bpe.mergeWordSimple(word),
          reason: 'table $table, word ${word.join()}',
        );
      }
    }
  });
}
