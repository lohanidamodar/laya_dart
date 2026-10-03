import 'package:laya_server/src/tokenizer/added_tokens.dart';
import 'package:test/test.dart';

/// An added-token entry as tokenizer.json writes it; ids count up from 1 in list order.
List<Map<String, Object?>> _tokens(List<(String, Map<String, bool>)> specs) => [
  for (var i = 0; i < specs.length; i++)
    {
      'id': i + 1,
      'content': specs[i].$1,
      'single_word': false,
      'lstrip': false,
      'rstrip': false,
      'normalized': false,
      'special': true,
      ...specs[i].$2,
    },
];

/// Pieces as plain values: a [TextPiece] becomes its text, a [TokenPiece] its id.
List<Object> _flat(List<Piece> pieces) => [
  for (final p in pieces)
    switch (p) {
      TextPiece(:final text) => text,
      TokenPiece(:final id) => id,
    },
];

/// Asserts that [splitter] cuts each input into the expected pieces.
void _expectSplits(
  AddedTokenSplitter splitter,
  Map<String, List<Object>> cases,
) {
  for (final MapEntry(key: text, value: pieces) in cases.entries) {
    expect(
      _flat(splitter.split(text)),
      pieces,
      reason: 'input ${Uri.encodeFull(text)}',
    );
  }
}

String _identity(String s) => s;

// Every expected value below other than the first test was observed with Hugging Face
// tokenizers 0.23.2: a synthetic BPE tokenizer.json with one vocab entry per character, no
// pre-tokenizer and the listed added tokens, encoded with add_special_tokens=False. The tokens
// HF returned were regrouped into runs of plain characters (a TextPiece) and added-token ids.
void main() {
  test('lstrip on [MASK] eats the space before it', () {
    final splitter = AddedTokenSplitter([
      {
        'id': 50284,
        'content': '[MASK]',
        'lstrip': true,
        'rstrip': false,
        'normalized': false,
        'single_word': false,
        'special': true,
      },
    ], normalize: (s) => s);
    final pieces = splitter.split('a [MASK] b');
    expect(pieces.length, 3);
    expect((pieces[0] as TextPiece).text, 'a');
    expect((pieces[1] as TokenPiece).id, 50284);
    expect((pieces[2] as TextPiece).text, ' b');
  });

  test('overlapping added tokens match leftmost-longest', () {
    final splitter = AddedTokenSplitter(
      _tokens([('ab', {}), ('abc', {}), ('bcd', {}), ('cd', {})]),
      normalize: _identity,
    );
    _expectSplits(splitter, {
      'abcd': [2, 'd'],
      'xbcd': ['x', 3],
      'abcab': [2, 1],
      'bcdab': [3, 1],
    });
    final shorter = AddedTokenSplitter(
      _tokens([('ab', {}), ('bcd', {})]),
      normalize: _identity,
    );
    _expectSplits(shorter, {
      'abcd': [1, 'cd'],
      'xabcd': ['x', 1, 'cd'],
    });
  });

  test('lstrip and rstrip eat Unicode whitespace, never a previous match', () {
    final splitter = AddedTokenSplitter(
      _tokens([
        ('[M]', {'lstrip': true}),
        ('[R]', {'rstrip': true}),
        ('[X]', {}),
      ]),
      normalize: _identity,
    );
    _expectSplits(splitter, {
      'a [M] b': ['a', 1, ' b'],
      'a [R]  b': ['a ', 2, 'b'],
      'a\t\n [R] \t b': ['a\t\n ', 2, 'b'],
      '[X] [M]': [3, 1],
      '[R] [X]': [2, 3],
      '[R] [M]': [2, 1],
      '  [M]': [1],
      '[R]  ': [2],
      // U+00A0 and U+0085 are White_Space; U+FEFF is not.
      'a [M]': ['a', 1],
      'a\u0085[M]': ['a', 1],
      'a﻿[M]': ['a﻿', 1],
      'a [R]\u0085﻿b': ['a ', 2, '﻿b'],
    });
  });

  test('rstrip over a later match emits both, as HF does', () {
    final raw = AddedTokenSplitter(
      _tokens([
        ('[R]', {'rstrip': true}),
        ('  ', {}),
      ]),
      normalize: _identity,
    );
    _expectSplits(raw, {
      '[R]  b': [1, 2, 'b'],
      '[R]   b': [1, 2, ' b'],
      '[R] x  b': [1, 'x', 2, 'b'],
    });
    final normalized = AddedTokenSplitter(
      _tokens([
        ('[R]', {'rstrip': true}),
        ('  ', {'normalized': true}),
      ]),
      normalize: _identity,
    );
    _expectSplits(normalized, {
      '[R]  b': [1, 'b'],
      'c  [R]  b': ['c', 2, 1, 'b'],
    });
  });

  test('lstrip stays inside its own piece', () {
    final overlapping = AddedTokenSplitter(
      _tokens([
        ('[M]', {'lstrip': true}),
        ('x ', {}),
      ]),
      normalize: _identity,
    );
    _expectSplits(overlapping, {
      'x [M]': [2, 1],
      'x  [M]': [2, 1],
    });
    final normalizedStrip = AddedTokenSplitter(
      _tokens([
        ('[M]', {'lstrip': true, 'normalized': true}),
        ('[X]', {}),
      ]),
      normalize: _identity,
    );
    _expectSplits(normalizedStrip, {
      '[X] [M]': [2, 1],
      'a [M]': ['a', 1],
    });
  });

  test(
    'normalized tokens match only after normalization; text pieces come back normalized',
    () {
      final splitter = AddedTokenSplitter(
        _tokens([
          ('hello', {'normalized': true}),
          ('WORLD', {}),
          ('XY', {'normalized': true}),
          ('[M]', {'lstrip': true}),
        ]),
        normalize: (s) => s.toLowerCase(),
      );
      _expectSplits(splitter, {
        'HELLO world': [1, ' world'],
        'hello WORLD': [1, ' ', 2],
        'Hello': [1],
        // A normalized token's own content is normalized before matching.
        'xy XY': [3, ' ', 3],
        'HEL[M]LO': ['hel', 4, 'lo'],
        'AB [M]CD': ['ab', 4, 'cd'],
      });
      final replace = AddedTokenSplitter(
        _tokens([
          (' q', {'normalized': true}),
          ('z z', {}),
        ]),
        normalize: (s) => s.replaceAll(' ', '▁'),
      );
      _expectSplits(replace, {
        'a q': ['a', 1],
        'a▁q': ['a', 1],
        'z z': [2],
        'z▁z': ['z▁z'],
      });
    },
  );

  test(
    'single_word needs a non-word character or a piece boundary on both sides',
    () {
      final splitter = AddedTokenSplitter(
        _tokens([
          ('ab', {'single_word': true}),
        ]),
        normalize: _identity,
      );
      _expectSplits(splitter, {
        'ab': [1],
        'xab': ['xab'],
        'abx': ['abx'],
        'ab_': ['ab_'],
        '_ab': ['_ab'],
        'ab1': ['ab1'],
        'a ab.': ['a ', 1, '.'],
        'éab': ['éab'],
        'ab-c': [1, '-c'],
        '.ab.': ['.', 1, '.'],
        'ab ab': [1, ' ', 1],
      });
    },
  );

  test('single_word uses Unicode word characters', () {
    final splitter = AddedTokenSplitter(
      _tokens([
        ('ab', {'single_word': true}),
      ]),
      normalize: _identity,
    );
    // Word characters: a mark, connector punctuation, letter numbers, Join_Control, astral
    // letters, modifier letters and non-ASCII digits. Not word characters: No numbers.
    const word = ['́', '‿', 'Ⅻ', '‍', '\u{1d400}', 'ª', 'ʰ', '٣'];
    const notWord = ['½', '²'];
    for (final c in word) {
      expect(_flat(splitter.split('${c}ab')), [
        '${c}ab',
      ], reason: 'before ${c.runes}');
      expect(_flat(splitter.split('ab$c')), [
        'ab$c',
      ], reason: 'after ${c.runes}');
    }
    for (final c in notWord) {
      expect(_flat(splitter.split('${c}ab')), [
        c,
        1,
      ], reason: 'before ${c.runes}');
      expect(_flat(splitter.split('ab$c')), [1, c], reason: 'after ${c.runes}');
    }
  });

  test('a rejected single_word match is skipped, not retried shorter', () {
    final withShorter = AddedTokenSplitter(
      _tokens([
        ('ab', {'single_word': true}),
        ('b', {}),
      ]),
      normalize: _identity,
    );
    _expectSplits(withShorter, {
      'xab': ['xab'],
      'ab': [1],
    });
    final overlapping = AddedTokenSplitter(
      _tokens([
        ('ab', {'single_word': true}),
        ('a', {}),
        ('bc', {}),
      ]),
      normalize: _identity,
    );
    _expectSplits(overlapping, {
      'xabc': ['xabc'],
      'abc': ['abc'],
    });
  });

  test(
    'single_word on a normalized token sees phase-one pieces as boundaries',
    () {
      final splitter = AddedTokenSplitter(
        _tokens([
          ('[X]', {}),
          ('ab', {'single_word': true, 'normalized': true}),
        ]),
        normalize: _identity,
      );
      _expectSplits(splitter, {
        '[X]ab': [1, 2],
        'c[X]ab[X]c': ['c', 1, 2, 1, 'c'],
        'cab[X]': ['cab', 1],
      });
    },
  );

  test('single_word is checked before stripping', () {
    final splitter = AddedTokenSplitter(
      _tokens([
        ('ab', {'single_word': true, 'lstrip': true, 'rstrip': true}),
      ]),
      normalize: _identity,
    );
    _expectSplits(splitter, {
      'x ab y': ['x', 1, 'y'],
      'xab y': ['xab y'],
      'x ab  ': ['x', 1],
    });
  });

  test('empty contents are ignored and empty pieces dropped', () {
    final splitter = AddedTokenSplitter(
      _tokens([('', {}), ('a', {})]),
      normalize: _identity,
    );
    _expectSplits(splitter, {
      'bab': ['b', 2, 'b'],
      'a': [2],
      '': [],
    });
    // A piece the normalizer empties is dropped (HF, with a Replace "x" -> "" normalizer).
    final erasing = AddedTokenSplitter(
      _tokens([('[M]', {})]),
      normalize: (s) => s.replaceAll('x', ''),
    );
    _expectSplits(erasing, {
      'x[M]xx': [1],
      'xyx': ['y'],
    });
  });
}
