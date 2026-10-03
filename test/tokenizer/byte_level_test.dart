import 'dart:convert';
import 'dart:io';

import 'package:laya_server/src/tokenizer/tokenizer.dart';
import 'package:test/test.dart';

import '../support/checkpoints.dart';

/// A minimal ByteLevel tokenizer.json over the given vocab.
Map<String, Object?> _tiny(Map<String, int> vocab) => {
  'added_tokens': [
    for (final (id, content) in [
      (100, '[CLS]'),
      (101, '[SEP]'),
      (102, '[PAD]'),
      (103, '[MASK]'),
    ])
      {
        'id': id,
        'content': content,
        'lstrip': content == '[MASK]',
        'normalized': false,
        'special': true,
      },
  ],
  'normalizer': {'type': 'NFC'},
  'pre_tokenizer': {
    'type': 'ByteLevel',
    'add_prefix_space': false,
    'trim_offsets': true,
    'use_regex': true,
  },
  'model': {'type': 'BPE', 'vocab': vocab, 'merges': <Object>[]},
};

void main() {
  test('a symbol missing from the vocab is a StateError', () {
    final tok = LayaTokenizer.fromJson(_tiny({'a': 0}));
    expect(tok.encode('a'), [0]);
    expect(tok.encode('a[MASK]'), [0, 103]);
    expect(() => tok.encode('b'), throwsStateError);
  });

  test('an unsupported ByteLevel option is rejected', () {
    final json = _tiny({'a': 0});
    (json['pre_tokenizer']! as Map<String, Object?>)['add_prefix_space'] = true;
    expect(() => LayaTokenizer.fromJson(json), throwsUnsupportedError);
  });

  for (final checkpoint in ['english', 'typed-decisions']) {
    group('$checkpoint checkpoint', () {
      late LayaTokenizer tok;
      late List<Map<String, Object?>> cases;
      setUpAll(() async {
        tok = await LayaTokenizer.load(tokenizerPath(checkpoint));
        final fixture =
            jsonDecode(
                  File(
                    'test/fixtures/tokenizer_english.json',
                  ).readAsStringSync(),
                )
                as Map<String, Object?>;
        cases = (fixture['cases']! as List).cast<Map<String, Object?>>();
      });

      test('special ids', () {
        expect(
          [tok.clsId, tok.sepId, tok.padId, tok.maskId],
          [50281, 50282, 50283, 50284],
        );
        expect(tok.maskToken, '[MASK]');
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
        // server's own option path replaces [MASK] first, which this fixture does not.
        for (final c in cases) {
          final ids = tok.encode(' ${c['text']}');
          expect(ids.take(48), c['option_ids'], reason: jsonEncode(c['text']));
        }
      });
    }, skip: tokenizerSkip(checkpoint));
  }

  test(
    'the committed tokenizer.json is the downloaded english and typed-decisions file',
    () {
      final committed = File(
        'test/fixtures/tokenizers/english/tokenizer.json',
      ).readAsBytesSync();
      for (final checkpoint in ['english', 'typed-decisions']) {
        final downloaded = File(
          '.tooling/checkpoints/$checkpoint/tokenizer/tokenizer.json',
        );
        if (!downloaded.existsSync()) {
          markTestSkipped(missingTokenizer);
          return;
        }
        expect(
          _sameBytes(downloaded.readAsBytesSync(), committed),
          isTrue,
          reason: checkpoint,
        );
      }
    },
  );
}

bool _sameBytes(List<int> a, List<int> b) {
  if (a.length != b.length) return false;
  for (var i = 0; i < a.length; i++) {
    if (a[i] != b[i]) return false;
  }
  return true;
}
