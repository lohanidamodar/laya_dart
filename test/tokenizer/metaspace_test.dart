import 'dart:convert';
import 'dart:io';

import 'package:laya_server/src/tokenizer/tokenizer.dart';
import 'package:test/test.dart';

import '../support/checkpoints.dart';

/// A minimal Metaspace tokenizer.json in the shape of the multilingual checkpoint's.
///
/// Every added token is also in the vocab under the same id: HF ignores an added token's `id`
/// when its content is not in the model vocab and assigns the next free id instead.
Map<String, Object?> _tiny({
  String prependScheme = 'always',
  bool split = true,
  bool byteFallback = true,
  bool fuseUnk = true,
  Map<String, int>? vocab,
  List<List<String>>? merges,
}) => {
  'added_tokens': [
    for (final (id, content) in [
      (0, '<pad>'),
      (1, '<eos>'),
      (2, '<bos>'),
      (3, '<unk>'),
      (4, '<mask>'),
      (12, '<t>'),
    ])
      {
        'id': id,
        'content': content,
        'single_word': false,
        'lstrip': content == '<mask>',
        'rstrip': false,
        'normalized': false,
        'special': content != '<t>',
      },
  ],
  'normalizer': {
    'type': 'Replace',
    'pattern': {'String': ' '},
    'content': '▁',
  },
  'pre_tokenizer': {
    'type': 'Metaspace',
    'replacement': '▁',
    'prepend_scheme': prependScheme,
    'split': split,
  },
  'model': {
    'type': 'BPE',
    'dropout': null,
    'unk_token': '<unk>',
    'continuing_subword_prefix': null,
    'end_of_word_suffix': null,
    'fuse_unk': fuseUnk,
    'byte_fallback': byteFallback,
    'ignore_merges': false,
    'vocab':
        vocab ??
        {
          '<pad>': 0,
          '<eos>': 1,
          '<bos>': 2,
          '<unk>': 3,
          '<mask>': 4,
          '▁': 5,
          'a': 6,
          'b': 7,
          '▁a': 8,
          '▁b': 9,
          'a▁': 10,
          '▁a▁': 11,
          '<t>': 12,
          '<0xC7>': 13,
          '<0x85>': 14,
          '<0xF0>': 15,
          '<0x9F>': 16,
          '<0x98>': 17,
          '<0x80>': 18,
        },
    'merges':
        merges ??
        [
          ['a', '▁'],
          ['▁', 'a▁'],
          ['▁', 'a'],
          ['▁', 'b'],
        ],
  },
};

/// What HF `tokenizers` 0.23.2 returns (`add_special_tokens=False`) for [_tiny] under each
/// (prepend_scheme, split, byte_fallback, fuse_unk), recorded with a throwaway probe against the
/// reference venv.
///
/// Note HF's unk ordering: a pending `<unk>` is flushed only by a character that is in the vocab,
/// not by byte-fallback tokens, so `"éǅa"` gives `▁ <0xC7> <0x85> <unk> a`, and with `fuse_unk`
/// the two `é`s of `"éǅé"` fuse into one `<unk>` after the bytes.
const _hf = <(String, bool, bool, bool), Map<String, List<int>>>{
  ('always', true, true, true): {
    'a b': [8, 9],
    'a<t>b': [8, 12, 9],
    'a <t> b': [8, 5, 12, 9],
    '<t>a': [12, 8],
    'a<t>': [8, 12],
    'a  b': [8, 5, 9],
    ' a': [8],
    'a ': [8, 5],
    '   ': [5, 5, 5],
    'b▁▁a': [9, 5, 8],
    'a <mask> b': [8, 4, 9],
    ' <mask>a': [4, 8],
    'x <mask>b': [5, 3, 4, 9],
    'ǅ': [5, 13, 14],
    'aǅb': [8, 13, 14, 7],
    'a😀b': [8, 15, 16, 17, 18, 7],
    'é': [5, 3],
    'éé': [5, 3],
    'éǅé': [5, 13, 14, 3],
    'éǅa': [5, 13, 14, 3, 6],
    'aéèb': [8, 3, 7],
    '中': [5, 3],
    'a a': [8, 8],
    'a  a': [8, 5, 8],
    'a<t>a ': [8, 12, 8, 5],
    '\t': [5, 3],
    '': [],
  },
  ('first', true, true, true): {
    'a b': [8, 9],
    'a<t>b': [8, 12, 7],
    'a <t> b': [8, 5, 12, 9],
    '<t>a': [12, 6],
    'a<t>': [8, 12],
    'a  b': [8, 5, 9],
    ' a': [8],
    'a ': [8, 5],
    '   ': [5, 5, 5],
    'b▁▁a': [9, 5, 8],
    'a <mask> b': [8, 4, 9],
    ' <mask>a': [4, 6],
    'x <mask>b': [5, 3, 4, 7],
    'ǅ': [5, 13, 14],
    'aǅb': [8, 13, 14, 7],
    'a😀b': [8, 15, 16, 17, 18, 7],
    'é': [5, 3],
    'éé': [5, 3],
    'éǅé': [5, 13, 14, 3],
    'éǅa': [5, 13, 14, 3, 6],
    'aéèb': [8, 3, 7],
    '中': [5, 3],
    'a a': [8, 8],
    'a  a': [8, 5, 8],
    'a<t>a ': [8, 12, 6, 5],
    '\t': [5, 3],
    '': [],
  },
  ('never', true, true, true): {
    'a b': [6, 9],
    'a<t>b': [6, 12, 7],
    'a <t> b': [6, 5, 12, 9],
    '<t>a': [12, 6],
    'a<t>': [6, 12],
    'a  b': [6, 5, 9],
    ' a': [8],
    'a ': [6, 5],
    '   ': [5, 5, 5],
    'b▁▁a': [7, 5, 8],
    'a <mask> b': [6, 4, 9],
    ' <mask>a': [4, 6],
    'x <mask>b': [3, 4, 7],
    'ǅ': [13, 14],
    'aǅb': [6, 13, 14, 7],
    'a😀b': [6, 15, 16, 17, 18, 7],
    'é': [3],
    'éé': [3],
    'éǅé': [13, 14, 3],
    'éǅa': [13, 14, 3, 6],
    'aéèb': [6, 3, 7],
    '中': [3],
    'a a': [6, 8],
    'a  a': [6, 5, 8],
    'a<t>a ': [6, 12, 6, 5],
    '\t': [3],
    '': [],
  },
  ('always', false, true, true): {
    'a b': [11, 7],
    'a<t>b': [8, 12, 9],
    'a <t> b': [11, 12, 9],
    '<t>a': [12, 8],
    'a<t>': [8, 12],
    'a  b': [11, 9],
    ' a': [8],
    'a ': [11],
    '   ': [5, 5, 5],
    'b▁▁a': [9, 5, 8],
    'a <mask> b': [8, 4, 9],
    ' <mask>a': [4, 8],
    'x <mask>b': [5, 3, 4, 9],
    'ǅ': [5, 13, 14],
    'aǅb': [8, 13, 14, 7],
    'a😀b': [8, 15, 16, 17, 18, 7],
    'é': [5, 3],
    'éé': [5, 3],
    'éǅé': [5, 13, 14, 3],
    'éǅa': [5, 13, 14, 3, 6],
    'aéèb': [8, 3, 7],
    '中': [5, 3],
    'a a': [11, 6],
    'a  a': [11, 8],
    'a<t>a ': [8, 12, 11],
    '\t': [5, 3],
    '': [],
  },
  ('first', false, true, true): {
    'a b': [11, 7],
    'a<t>b': [8, 12, 7],
    'a <t> b': [11, 12, 9],
    '<t>a': [12, 6],
    'a<t>': [8, 12],
    'a  b': [11, 9],
    ' a': [8],
    'a ': [11],
    '   ': [5, 5, 5],
    'b▁▁a': [9, 5, 8],
    'a <mask> b': [8, 4, 9],
    ' <mask>a': [4, 6],
    'x <mask>b': [5, 3, 4, 7],
    'ǅ': [5, 13, 14],
    'aǅb': [8, 13, 14, 7],
    'a😀b': [8, 15, 16, 17, 18, 7],
    'é': [5, 3],
    'éé': [5, 3],
    'éǅé': [5, 13, 14, 3],
    'éǅa': [5, 13, 14, 3, 6],
    'aéèb': [8, 3, 7],
    '中': [5, 3],
    'a a': [11, 6],
    'a  a': [11, 8],
    'a<t>a ': [8, 12, 10],
    '\t': [5, 3],
    '': [],
  },
  ('never', false, true, true): {
    'a b': [10, 7],
    'a<t>b': [6, 12, 7],
    'a <t> b': [10, 12, 9],
    '<t>a': [12, 6],
    'a<t>': [6, 12],
    'a  b': [10, 9],
    ' a': [8],
    'a ': [10],
    '   ': [5, 5, 5],
    'b▁▁a': [7, 5, 8],
    'a <mask> b': [6, 4, 9],
    ' <mask>a': [4, 6],
    'x <mask>b': [3, 4, 7],
    'ǅ': [13, 14],
    'aǅb': [6, 13, 14, 7],
    'a😀b': [6, 15, 16, 17, 18, 7],
    'é': [3],
    'éé': [3],
    'éǅé': [13, 14, 3],
    'éǅa': [13, 14, 3, 6],
    'aéèb': [6, 3, 7],
    '中': [3],
    'a a': [10, 6],
    'a  a': [10, 8],
    'a<t>a ': [6, 12, 10],
    '\t': [3],
    '': [],
  },
  ('always', true, false, true): {
    'a b': [8, 9],
    'a<t>b': [8, 12, 9],
    'a <t> b': [8, 5, 12, 9],
    '<t>a': [12, 8],
    'a<t>': [8, 12],
    'a  b': [8, 5, 9],
    ' a': [8],
    'a ': [8, 5],
    '   ': [5, 5, 5],
    'b▁▁a': [9, 5, 8],
    'a <mask> b': [8, 4, 9],
    ' <mask>a': [4, 8],
    'x <mask>b': [5, 3, 4, 9],
    'ǅ': [5, 3],
    'aǅb': [8, 3, 7],
    'a😀b': [8, 3, 7],
    'é': [5, 3],
    'éé': [5, 3],
    'éǅé': [5, 3],
    'éǅa': [5, 3, 6],
    'aéèb': [8, 3, 7],
    '中': [5, 3],
    'a a': [8, 8],
    'a  a': [8, 5, 8],
    'a<t>a ': [8, 12, 8, 5],
    '\t': [5, 3],
    '': [],
  },
  ('always', true, true, false): {
    'a b': [8, 9],
    'a<t>b': [8, 12, 9],
    'a <t> b': [8, 5, 12, 9],
    '<t>a': [12, 8],
    'a<t>': [8, 12],
    'a  b': [8, 5, 9],
    ' a': [8],
    'a ': [8, 5],
    '   ': [5, 5, 5],
    'b▁▁a': [9, 5, 8],
    'a <mask> b': [8, 4, 9],
    ' <mask>a': [4, 8],
    'x <mask>b': [5, 3, 4, 9],
    'ǅ': [5, 13, 14],
    'aǅb': [8, 13, 14, 7],
    'a😀b': [8, 15, 16, 17, 18, 7],
    'é': [5, 3],
    'éé': [5, 3, 3],
    'éǅé': [5, 13, 14, 3, 3],
    'éǅa': [5, 13, 14, 3, 6],
    'aéèb': [8, 3, 3, 7],
    '中': [5, 3],
    'a a': [8, 8],
    'a  a': [8, 5, 8],
    'a<t>a ': [8, 12, 8, 5],
    '\t': [5, 3],
    '': [],
  },
  ('always', true, false, false): {
    'a b': [8, 9],
    'a<t>b': [8, 12, 9],
    'a <t> b': [8, 5, 12, 9],
    '<t>a': [12, 8],
    'a<t>': [8, 12],
    'a  b': [8, 5, 9],
    ' a': [8],
    'a ': [8, 5],
    '   ': [5, 5, 5],
    'b▁▁a': [9, 5, 8],
    'a <mask> b': [8, 4, 9],
    ' <mask>a': [4, 8],
    'x <mask>b': [5, 3, 4, 9],
    'ǅ': [5, 3],
    'aǅb': [8, 3, 7],
    'a😀b': [8, 3, 7],
    'é': [5, 3],
    'éé': [5, 3, 3],
    'éǅé': [5, 3, 3, 3],
    'éǅa': [5, 3, 3, 6],
    'aéèb': [8, 3, 3, 7],
    '中': [5, 3],
    'a a': [8, 8],
    'a  a': [8, 5, 8],
    'a<t>a ': [8, 12, 8, 5],
    '\t': [5, 3],
    '': [],
  },
};

void main() {
  group('synthetic tokenizer matches HF tokenizers', () {
    for (final MapEntry(key: (scheme, split, fallback, fuse), value: cases)
        in _hf.entries) {
      test('prepend_scheme=$scheme split=$split byte_fallback=$fallback '
          'fuse_unk=$fuse', () {
        final tok = LayaTokenizer.fromJson(
          _tiny(
            prependScheme: scheme,
            split: split,
            byteFallback: fallback,
            fuseUnk: fuse,
          ),
        );
        for (final MapEntry(key: text, value: ids) in cases.entries) {
          expect(tok.encode(text), ids, reason: jsonEncode(text));
        }
      });
    }
  });

  test('a character outside the vocab falls back to its UTF-8 byte tokens', () {
    // The shape of laya-ts tokenizer-parity.test.ts:92; HF gives [5, 1, 2, 3, 4, 6].
    final tok = LayaTokenizer.fromJson(
      _tiny(
        vocab: {
          '<unk>': 0,
          '<0xF0>': 1,
          '<0x9F>': 2,
          '<0x98>': 3,
          '<0x80>': 4,
          '▁a': 5,
          'b': 6,
          '▁': 7,
          'a': 8,
          '<pad>': 9,
          '<eos>': 10,
          '<bos>': 11,
          '<mask>': 12,
          '<t>': 13,
        },
        merges: [
          ['▁', 'a'],
        ],
      ),
    );
    expect(tok.encode('a😀b'), [5, 1, 2, 3, 4, 6]);
  });

  test('byte tokens take part in merges', () {
    final json = _tiny(
      vocab: {
        '<pad>': 0,
        '<eos>': 1,
        '<bos>': 2,
        '<unk>': 3,
        '<mask>': 4,
        '▁': 5,
        '<0xC7>': 6,
        '<0x85>': 7,
        '<0xC7><0x85>': 8,
        '<t>': 9,
      },
      merges: [
        ['<0xC7>', '<0x85>'],
      ],
    );
    // HF gives [5, 8].
    expect(LayaTokenizer.fromJson(json).encode('ǅ'), [5, 8]);
  });

  test('a Sequence of Replace normalizers applies them in order', () {
    final json = _tiny();
    json['normalizer'] = {
      'type': 'Sequence',
      'normalizers': [
        {
          'type': 'Replace',
          'pattern': {'String': 'b'},
          'content': ' ',
        },
        {
          'type': 'Replace',
          'pattern': {'String': ' '},
          'content': '▁',
        },
      ],
    };
    // HF: "abba" -> "a  a" -> "a▁▁a" -> ▁a ▁ ▁a.
    expect(LayaTokenizer.fromJson(json).encode('abba'), [8, 5, 8]);
  });

  group('an unsupported option is rejected, naming it', () {
    void rejects(void Function(Map<String, Object?> json) edit, String name) {
      final json = _tiny();
      edit(json);
      expect(
        () => LayaTokenizer.fromJson(json),
        throwsA(
          isA<UnsupportedError>().having(
            (e) => e.message,
            'message',
            contains(name),
          ),
        ),
      );
    }

    Map<String, Object?> pre(Map<String, Object?> json) =>
        json['pre_tokenizer']! as Map<String, Object?>;

    test('normalizer type', () {
      rejects((json) => json['normalizer'] = {'type': 'NFC'}, 'NFC');
      rejects((json) => json['normalizer'] = null, 'normalizer');
    });

    test('Replace with a Regex pattern', () {
      rejects(
        (json) => json['normalizer'] = {
          'type': 'Replace',
          'pattern': {'Regex': ' +'},
          'content': '▁',
        },
        'Regex',
      );
    });

    test('prepend_scheme', () {
      rejects((json) => pre(json)['prepend_scheme'] = 'sometimes', 'sometimes');
      rejects((json) => pre(json).remove('prepend_scheme'), 'prepend_scheme');
    });

    test('split', () {
      rejects((json) => pre(json).remove('split'), 'split');
    });

    test('legacy add_prefix_space', () {
      rejects(
        (json) => pre(json)['add_prefix_space'] = true,
        'add_prefix_space',
      );
    });

    test('replacement', () {
      rejects((json) => pre(json)['replacement'] = '__', 'replacement');
    });

    test('an unk_token missing from the vocab', () {
      rejects(
        (json) => (json['model']! as Map)['unk_token'] = '<nope>',
        '<nope>',
      );
    });
  });

  group('multilingual checkpoint', () {
    late LayaTokenizer tok;
    late List<Map<String, Object?>> cases;
    setUpAll(() async {
      tok = await LayaTokenizer.load(tokenizerPath('multilingual'));
      final fixture =
          jsonDecode(
                File(
                  'test/fixtures/tokenizer_multilingual.json',
                ).readAsStringSync(),
              )
              as Map<String, Object?>;
      cases = (fixture['cases']! as List).cast<Map<String, Object?>>();
    });

    test('special ids', () {
      expect([tok.clsId, tok.sepId, tok.padId, tok.maskId], [2, 1, 0, 4]);
      expect(tok.maskToken, '<mask>');
    });

    test('the laya-ts reference vectors (research.md 2.4)', () {
      expect(
        tok.encode(
          'choice question: Which department should handle this request?',
        ),
        [6241, 2872, 235292, 12236, 9888, 1412, 6589, 736, 3853, 235336],
      );
      expect(tok.encode('मुझसे दो बार शुल्क लिया गया'), [
        39728,
        238385,
        20579,
        51484,
        99744,
        144905,
        228834,
        158850,
        44370,
      ]);
    });

    test('byte fallback and edge input on the real vocab', () {
      // Recorded from HF tokenizers 0.23.2 with a throwaway probe. The fixture corpus reaches
      // byte fallback only through U+0000; these add characters outside the 256k vocab.
      const hf = {
        '𓀀𝔘 \t ꙮ 㐀 뷁 ﷽ 🫠 👩🏽‍💻': [
          235248, 457, 364, 345, 345, 457, 374, 365, 369, 235248, 226, //
          235248, 255218, 235248, 444, 361, 345, 235248, 452, 400, 346,
          235248, 246908, 235248, 244497, 235248, 241355, 239518, 235879,
          242482,
        ],
        'a\tb\t\tc': [476, 226, 235268, 255969, 498],
        '\u{10FFFF}\u{E000}\u{FFFD}x': [
          235248, 253242, 240539, 236193, 235297, //
        ],
        '  leading and trailing  ': [
          235248, 8133, 578, 60732, 235248, 235248, //
        ],
        'x <mask>y <mask> z<mask>': [1141, 4, 597, 4, 868, 4],
        '中文没有空格的一段很长的文字测试一下合并是否正确': [
          102546, 6845, 226258, 13824, 236306, 235817, 60376, 25047, //
          28991, 18071, 131915, 17557, 50838,
        ],
      };
      for (final MapEntry(key: text, value: ids) in hf.entries) {
        expect(tok.encode(text), ids, reason: jsonEncode(text));
      }
    });

    test('every fixture text matches HF tokenizers', () {
      for (final c in cases) {
        expect(
          tok.encode(c['text']! as String),
          c['ids'],
          reason: jsonEncode(c['text']),
        );
      }
    });

    test('" " + text, cut to 48 ids, matches the option encoding', () {
      // option_ids is the raw `" " + text` through the fast tokenizer with max_length 48; the
      // server's own option path replaces <mask> first, which this fixture does not.
      for (final c in cases) {
        final ids = tok.encode(' ${c['text']}');
        expect(ids.take(48), c['option_ids'], reason: jsonEncode(c['text']));
      }
    });
  }, skip: tokenizerSkip('multilingual'));
}
