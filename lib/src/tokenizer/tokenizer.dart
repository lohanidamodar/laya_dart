import 'dart:convert';
import 'dart:io';

import 'added_tokens.dart';
import 'bpe.dart';
import 'byte_level.dart';
import 'metaspace.dart';

/// A tokenizer loaded from a Hugging Face `tokenizer.json` (ByteLevel or Metaspace BPE).
///
/// Encodings reproduce Hugging Face `tokenizers` 0.23.2 with `add_special_tokens=False`.
abstract class LayaTokenizer {
  /// Parses [json] (the decoded tokenizer.json) and picks the implementation by pre_tokenizer.
  ///
  /// Throws a [FormatException] for a malformed file and an [UnsupportedError] for a model,
  /// option or pre-tokenizer this port does not implement.
  factory LayaTokenizer.fromJson(Map<String, Object?> json) {
    final spec = TokenizerSpec.fromJson(json);
    return switch (spec.preTokenizerType) {
      'ByteLevel' => _byteLevel(spec),
      'Metaspace' => _metaspace(spec),
      final type => throw UnsupportedError(
        'pre_tokenizer type "$type" is not supported',
      ),
    };
  }

  /// Reads and parses a tokenizer.json file.
  static Future<LayaTokenizer> load(String path) async {
    final json = jsonDecode(await File(path).readAsString());
    if (json is! Map<String, Object?>) {
      throw FormatException('$path: expected a JSON object');
    }
    return LayaTokenizer.fromJson(json);
  }

  /// Ids for [text] without special tokens (HF `add_special_tokens=False`).
  List<int> encode(String text);

  /// The CLS token id (`[CLS]` or `<bos>`).
  int get clsId;

  /// The SEP token id (`[SEP]` or `<eos>`).
  int get sepId;

  /// The PAD token id.
  int get padId;

  /// The MASK token id.
  int get maskId;

  /// The MASK token text (`[MASK]` or `<mask>`), which user text has replaced by a space.
  String get maskToken;
}

/// Construction point for the ModernBERT ByteLevel tokenizer.
LayaTokenizer _byteLevel(TokenizerSpec spec) => ByteLevelTokenizer(spec);

/// Construction point for the mmBERT Metaspace tokenizer.
LayaTokenizer _metaspace(TokenizerSpec spec) => MetaspaceTokenizer(spec);

/// The parts of a tokenizer.json that every implementation shares.
class TokenizerSpec {
  TokenizerSpec._({
    required this.model,
    required this.vocab,
    required this.merges,
    required this.addedTokens,
    required this.normalizer,
    required this.preTokenizer,
    required this.clsId,
    required this.sepId,
    required this.padId,
    required this.maskId,
    required this.maskToken,
  });

  /// Parses the shared parts of a decoded tokenizer.json.
  ///
  /// Throws a [FormatException] when a required part is missing or malformed, and an
  /// [UnsupportedError] for a model other than BPE or a BPE option this port does not implement
  /// (dropout, a subword prefix or suffix, `ignore_merges`).
  factory TokenizerSpec.fromJson(Map<String, Object?> json) {
    final model = _map(json['model'], 'model');
    if (model['type'] != 'BPE') {
      throw UnsupportedError('model type "${model['type']}" is not supported');
    }
    for (final option in [
      'dropout',
      'continuing_subword_prefix',
      'end_of_word_suffix',
    ]) {
      if (model[option] != null) {
        throw UnsupportedError('BPE $option is not supported');
      }
    }
    if (model['ignore_merges'] == true) {
      throw UnsupportedError('BPE ignore_merges is not supported');
    }
    final vocab = <String, int>{
      for (final MapEntry(:key, :value) in _map(
        model['vocab'],
        'model.vocab',
      ).entries)
        key: value is int
            ? value
            : throw FormatException('model.vocab["$key"] is not an int'),
    };
    final rawMerges = model['merges'] ?? const <Object?>[];
    if (rawMerges is! List) {
      throw const FormatException('model.merges is not a list');
    }
    final merges = [for (final merge in rawMerges) _merge(merge)];

    final addedRaw = json['added_tokens'] ?? const <Object?>[];
    if (addedRaw is! List) {
      throw const FormatException('added_tokens is not a list');
    }
    final addedTokens = [
      for (final (i, token) in addedRaw.indexed)
        _map(token, 'added_tokens[$i]'),
    ];
    for (final (i, token) in addedTokens.indexed) {
      if (token['id'] is! int || token['content'] is! String) {
        throw FormatException(
          'added_tokens[$i] needs an int id and a string content',
        );
      }
    }

    final preTokenizer = _map(json['pre_tokenizer'], 'pre_tokenizer');
    final normalizer = json['normalizer'] == null
        ? null
        : _map(json['normalizer'], 'normalizer');

    // A special token's id is looked up by content, as transformers' `<x>_token_id` does:
    // added tokens first, then the model vocab.
    final added = {
      for (final t in addedTokens) t['content']! as String: t['id']! as int,
    };
    (int, String) special(String role, List<String> aliases) {
      for (final alias in aliases) {
        final id = added[alias] ?? vocab[alias];
        if (id != null) return (id, alias);
      }
      throw FormatException(
        'no $role token (${aliases.join(' or ')}) in tokenizer.json',
      );
    }

    final (maskId, maskToken) = special('mask', ['[MASK]', '<mask>']);
    return TokenizerSpec._(
      model: model,
      vocab: vocab,
      merges: merges,
      addedTokens: addedTokens,
      normalizer: normalizer,
      preTokenizer: preTokenizer,
      clsId: special('CLS', ['[CLS]', '<bos>']).$1,
      sepId: special('SEP', ['[SEP]', '<eos>']).$1,
      padId: special('PAD', ['[PAD]', '<pad>']).$1,
      maskId: maskId,
      maskToken: maskToken,
    );
  }

  /// The `model` object, for model options such as `unk_token`, `byte_fallback`, `fuse_unk`.
  final Map<String, Object?> model;

  /// The model vocab, token to id.
  final Map<String, int> vocab;

  /// The merges in rank order.
  final List<(String, String)> merges;

  /// The `added_tokens` entries, as [AddedTokenSplitter] takes them.
  final List<Map<String, Object?>> addedTokens;

  /// The `normalizer` object, or null when the file has none.
  final Map<String, Object?>? normalizer;

  /// The `pre_tokenizer` object.
  final Map<String, Object?> preTokenizer;

  /// The `pre_tokenizer` type, which picks the implementation.
  String? get preTokenizerType => preTokenizer['type'] as String?;

  /// The CLS token id (`[CLS]` or `<bos>`).
  final int clsId;

  /// The SEP token id (`[SEP]` or `<eos>`).
  final int sepId;

  /// The PAD token id (`[PAD]` or `<pad>`).
  final int padId;

  /// The MASK token id (`[MASK]` or `<mask>`).
  final int maskId;

  /// The MASK token text (`[MASK]` or `<mask>`).
  final String maskToken;

  /// The BPE engine over [vocab] and [merges], built on first use.
  late final BpeModel bpe = BpeModel(vocab: vocab, merges: merges);

  /// The splitter for [addedTokens], given the tokenizer's [normalize].
  AddedTokenSplitter splitter(String Function(String) normalize) =>
      AddedTokenSplitter(addedTokens, normalize: normalize);

  static Map<String, Object?> _map(Object? value, String where) =>
      value is Map<String, Object?>
      ? value
      : throw FormatException('$where is missing or not an object');

  /// A merge as `["a", "b"]` or, in older files, `"a b"`.
  static (String, String) _merge(Object? merge) => switch (merge) {
    [final String a, final String b] => (a, b),
    final String s when s.split(' ').length == 2 => (
      s.split(' ')[0],
      s.split(' ')[1],
    ),
    _ => throw FormatException('malformed merge: $merge'),
  };
}
