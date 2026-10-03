import 'dart:convert';
import 'dart:io';

import 'package:laya_server/src/tokenizer/tokenizer.dart';
import 'package:test/test.dart';

Map<String, Object?> _json({
  required String preTokenizer,
  List<Object> merges = const [],
  List<Map<String, Object?>>? added,
}) => {
  'added_tokens':
      added ??
      [
        for (final (id, content) in [
          (10, '[CLS]'),
          (11, '[SEP]'),
          (12, '[PAD]'),
          (13, '[MASK]'),
        ])
          {
            'id': id,
            'content': content,
            'single_word': false,
            'lstrip': content == '[MASK]',
            'rstrip': false,
            'normalized': false,
            'special': true,
          },
      ],
  'normalizer': {'type': 'NFC'},
  'pre_tokenizer': {'type': preTokenizer},
  'model': {
    'type': 'BPE',
    'dropout': null,
    'unk_token': null,
    'continuing_subword_prefix': null,
    'end_of_word_suffix': null,
    'fuse_unk': false,
    'byte_fallback': false,
    'ignore_merges': false,
    'vocab': {'a': 0, 'b': 1, 'c': 2, 'ab': 3, 'bb': 4, 'bbc': 5},
    'merges': merges,
  },
};

void main() {
  group('TokenizerSpec', () {
    test('reads merges as pairs or as space-separated strings', () {
      final pairs = TokenizerSpec.fromJson(
        _json(
          preTokenizer: 'ByteLevel',
          merges: [
            ['a', 'b'],
            ['b', 'b'],
            ['bb', 'c'],
          ],
        ),
      );
      final strings = TokenizerSpec.fromJson(
        _json(preTokenizer: 'ByteLevel', merges: ['a b', 'b b', 'bb c']),
      );
      for (final spec in [pairs, strings]) {
        expect(spec.merges, [('a', 'b'), ('b', 'b'), ('bb', 'c')]);
        expect(spec.bpe.mergeWord(['a', 'b', 'c']), ['ab', 'c']);
      }
    });

    test('resolves the special tokens by content', () {
      final spec = TokenizerSpec.fromJson(_json(preTokenizer: 'ByteLevel'));
      expect(spec.clsId, 10);
      expect(spec.sepId, 11);
      expect(spec.padId, 12);
      expect(spec.maskId, 13);
      expect(spec.maskToken, '[MASK]');
      expect(spec.preTokenizerType, 'ByteLevel');
      expect(spec.addedTokens, hasLength(4));
    });

    test('resolves Gemma-style special tokens', () {
      final spec = TokenizerSpec.fromJson(
        _json(
          preTokenizer: 'Metaspace',
          added: [
            for (final (id, content) in [
              (0, '<pad>'),
              (1, '<eos>'),
              (2, '<bos>'),
              (4, '<mask>'),
            ])
              {'id': id, 'content': content, 'special': true},
          ],
        ),
      );
      expect(
        [spec.clsId, spec.sepId, spec.padId, spec.maskId, spec.maskToken],
        [2, 1, 0, 4, '<mask>'],
      );
    });

    test('a missing special token is a FormatException', () {
      expect(
        () =>
            TokenizerSpec.fromJson(_json(preTokenizer: 'ByteLevel', added: [])),
        throwsFormatException,
      );
    });

    test('unsupported model options are rejected', () {
      final json = _json(preTokenizer: 'ByteLevel');
      (json['model']! as Map<String, Object?>)['continuing_subword_prefix'] =
          '##';
      expect(() => TokenizerSpec.fromJson(json), throwsUnsupportedError);
    });
  });

  group('LayaTokenizer.fromJson', () {
    test('ByteLevel and Metaspace are not implemented yet', () {
      expect(
        () => LayaTokenizer.fromJson(_json(preTokenizer: 'ByteLevel')),
        throwsUnsupportedError,
      );
      expect(
        () => LayaTokenizer.fromJson(_json(preTokenizer: 'Metaspace')),
        throwsUnsupportedError,
      );
    });

    test('other pre-tokenizers are unsupported', () {
      expect(
        () => LayaTokenizer.fromJson(_json(preTokenizer: 'Whitespace')),
        throwsUnsupportedError,
      );
    });
  });

  // The downloaded checkpoints live under the git-ignored .tooling/ (tool/gen_fixtures.dart).
  final checkpoints = {
    'english': (50281, 50282, 50283, 50284, '[MASK]'),
    'typed-decisions': (50281, 50282, 50283, 50284, '[MASK]'),
    'multilingual': (2, 1, 0, 4, '<mask>'),
  };
  for (final MapEntry(key: name, value: expected) in checkpoints.entries) {
    final file = File('.tooling/checkpoints/$name/tokenizer/tokenizer.json');
    test(
      'special ids of the $name checkpoint match its post_processor',
      () {
        final json =
            jsonDecode(file.readAsStringSync()) as Map<String, Object?>;
        final spec = TokenizerSpec.fromJson(json);
        expect((
          spec.clsId,
          spec.sepId,
          spec.padId,
          spec.maskId,
          spec.maskToken,
        ), expected);
        final special =
            (json['post_processor']! as Map<String, Object?>)['special_tokens']!
                as Map<String, Object?>;
        int idOf(String token) =>
            ((special[token]! as Map<String, Object?>)['ids']! as List).single
                as int;
        if (name == 'multilingual') {
          expect(spec.clsId, idOf('<bos>'));
          expect(spec.sepId, idOf('<eos>'));
        } else {
          for (final (token, id) in [
            ('[CLS]', spec.clsId),
            ('[SEP]', spec.sepId),
            ('[PAD]', spec.padId),
            ('[MASK]', spec.maskId),
          ]) {
            expect(id, idOf(token), reason: token);
          }
        }
      },
      skip: file.existsSync()
          ? false
          : 'run tool/gen_fixtures.dart to download $name',
    );
  }
}
