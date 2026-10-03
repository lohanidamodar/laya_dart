import 'dart:convert';

import 'added_tokens.dart';
import 'tokenizer.dart';

/// The Metaspace prepend schemes of Hugging Face `tokenizers`.
enum _Prepend {
  /// Prefix every text piece that does not already start with the replacement.
  always,

  /// Prefix only a text piece that starts at offset 0 of the input.
  first,

  /// Never prefix.
  never,
}

/// The mmBERT (Gemma-style) tokenizer of the multilingual checkpoint: `Replace` normalizers,
/// Metaspace pre-tokenization and BPE with byte fallback.
///
/// Mirrors Hugging Face `tokenizers` 0.23.2 with `add_special_tokens=False`:
///
/// 1. added tokens are split out ([AddedTokenSplitter]), and the text between them is run
///    through the `Replace` normalizer(s) in order;
/// 2. in each text piece, spaces become the replacement (`▁`); the piece gets a leading
///    replacement unless it already has one (`prepend_scheme: always`), only when it starts the
///    input (`first`), or never (`never`); with `split` it is then cut before every replacement,
///    which stays at the front of the word that follows;
/// 3. each word is turned into symbols one character at a time, and merged by the BPE model.
///    A character missing from the vocab becomes one `<0xNN>` token per UTF-8 byte when
///    `byte_fallback` is set and all of them exist, else `<unk>`. A run of `<unk>`s is fused
///    into one with `fuse_unk`.
///
/// One HF quirk is kept: a pending `<unk>` is flushed only by a character found in the vocab,
/// not by byte-fallback tokens, so an unknown character followed by a byte-fallback one comes
/// out as the bytes first and the `<unk>` after them.
class MetaspaceTokenizer implements LayaTokenizer {
  /// Builds the tokenizer from a parsed tokenizer.json.
  ///
  /// Supports a `Replace` normalizer with a `String` pattern, or a `Sequence` of them, and a
  /// Metaspace pre-tokenizer with a one-character replacement, `prepend_scheme` `always`,
  /// `first` or `never`, and `split`. Throws an [UnsupportedError] naming any other normalizer,
  /// pattern or option.
  factory MetaspaceTokenizer(TokenizerSpec spec) {
    final replaces = _replaces(spec.normalizer);
    final pre = spec.preTokenizer;
    if (pre.containsKey('add_prefix_space')) {
      throw UnsupportedError(
        'Metaspace add_prefix_space (the legacy form of prepend_scheme) is not supported',
      );
    }
    final replacement = pre['replacement'];
    if (replacement is! String || replacement.runes.length != 1) {
      throw UnsupportedError(
        'Metaspace replacement "$replacement" is not supported; it must be one character',
      );
    }
    final scheme = pre['prepend_scheme'];
    final prepend = _Prepend.values.where((p) => p.name == scheme).firstOrNull;
    if (prepend == null) {
      throw UnsupportedError(
        'Metaspace prepend_scheme "$scheme" is not supported',
      );
    }
    final split = pre['split'];
    if (split is! bool) {
      throw UnsupportedError('Metaspace split "$split" is not supported');
    }

    final model = spec.model;
    bool flag(String name) => switch (model[name]) {
      null || false => false,
      true => true,
      final value => throw UnsupportedError(
        'BPE $name "$value" is not supported',
      ),
    };
    final unk = model['unk_token'];
    if (unk != null && unk is! String) {
      throw UnsupportedError('BPE unk_token "$unk" is not supported');
    }
    final byteFallback = flag('byte_fallback');
    final byteTokens = [
      for (var b = 0; b < 256; b++)
        if (byteFallback) _byteToken(b, spec.vocab) else null,
    ];

    String normalize(String text) {
      for (final (pattern, content) in replaces) {
        text = text.replaceAll(pattern, content);
      }
      return text;
    }

    return MetaspaceTokenizer._(
      spec,
      spec.splitter(normalize),
      replacement,
      prepend,
      split,
      byteFallback,
      flag('fuse_unk'),
      unk as String?,
      byteTokens,
    );
  }

  MetaspaceTokenizer._(
    this._spec,
    this._splitter,
    this._replacement,
    this._prepend,
    this._split,
    this._byteFallback,
    this._fuseUnk,
    this._unk,
    this._byteTokens,
  );

  final TokenizerSpec _spec;
  final AddedTokenSplitter _splitter;
  final String _replacement;
  final _Prepend _prepend;
  final bool _split;
  final bool _byteFallback;
  final bool _fuseUnk;

  /// The `unk_token`, or null when the file has none.
  final String? _unk;

  /// The `<0xNN>` token for each byte, or null when it is not in the vocab.
  final List<String?> _byteTokens;

  /// HF's byte-fallback token for [byte] (`<0x0A>`, uppercase hex), when [vocab] has it.
  static String? _byteToken(int byte, Map<String, int> vocab) {
    final token = '<0x${byte.toRadixString(16).toUpperCase().padLeft(2, '0')}>';
    return vocab.containsKey(token) ? token : null;
  }

  /// The (pattern, content) pairs of a `Replace` normalizer or a `Sequence` of them.
  static List<(String, String)> _replaces(Map<String, Object?>? normalizer) {
    (String, String) replace(Map<String, Object?> json) {
      final pattern = json['pattern'];
      final content = json['content'];
      if (pattern is! Map || pattern.keys.join() != 'String') {
        throw UnsupportedError(
          'Replace pattern ${jsonEncode(pattern)} is not supported; only a String pattern is '
          '(not a Regex)',
        );
      }
      final string = pattern['String'];
      if (string is! String || string.isEmpty || content is! String) {
        throw UnsupportedError(
          'Replace ${jsonEncode(pattern)} -> ${jsonEncode(content)} is not supported',
        );
      }
      return (string, content);
    }

    Never unsupported(Object? type) => throw UnsupportedError(
      'normalizer "$type" is not supported with Metaspace; only Replace or a Sequence of '
      'Replace is',
    );

    return switch (normalizer) {
      {'type': 'Replace'} => [replace(normalizer)],
      {'type': 'Sequence', 'normalizers': final List<Object?> list} => [
        for (final item in list)
          item is Map<String, Object?> && item['type'] == 'Replace'
              ? replace(item)
              : unsupported(item is Map ? item['type'] : item),
      ],
      null => unsupported(null),
      _ => unsupported(normalizer['type']),
    };
  }

  @override
  List<int> encode(String text) {
    final ids = <int>[];
    final pieces = _splitter.split(text);
    for (final (i, piece) in pieces.indexed) {
      switch (piece) {
        case TokenPiece(:final id):
          ids.add(id);
        case TextPiece(text: final normalized):
          // The splitter has already normalized text pieces. Splitting drops empty pieces, so
          // only the first piece of the output can start at offset 0 of the input.
          for (final word in _preTokenize(normalized, atStart: i == 0)) {
            _encodeWord(word, ids);
          }
      }
    }
    return ids;
  }

  /// HF `Metaspace::pre_tokenize` on one text piece.
  List<String> _preTokenize(String piece, {required bool atStart}) {
    var s = piece.replaceAll(' ', _replacement);
    final prepend = switch (_prepend) {
      _Prepend.always => true,
      _Prepend.first => atStart,
      _Prepend.never => false,
    };
    if (prepend && !s.startsWith(_replacement)) s = '$_replacement$s';
    if (!_split) return [s];
    // SplitDelimiterBehavior::MergedWithNext: cut before every replacement.
    final words = <String>[];
    var start = 0;
    var at = s.indexOf(_replacement, 1);
    while (at >= 0) {
      words.add(s.substring(start, at));
      start = at;
      at = s.indexOf(_replacement, at + _replacement.length);
    }
    words.add(s.substring(start));
    return words;
  }

  /// HF `BPE::merge_word` then the merges, on one word.
  void _encodeWord(String word, List<int> ids) {
    final vocab = _spec.vocab;
    final symbols = <String>[];
    var pendingUnk = false;
    for (final rune in word.runes) {
      final char = String.fromCharCode(rune);
      if (vocab.containsKey(char)) {
        if (pendingUnk) {
          symbols.add(_unk!);
          pendingUnk = false;
        }
        symbols.add(char);
        continue;
      }
      if (_byteFallback && _addBytes(char, symbols)) continue;
      final unk = _unk;
      if (unk == null || !vocab.containsKey(unk)) {
        throw StateError(
          '"$char" is not in the vocab and there is no usable unk_token',
        );
      }
      if (pendingUnk && !_fuseUnk) symbols.add(unk);
      pendingUnk = true;
    }
    if (pendingUnk) symbols.add(_unk!);
    for (final symbol in _spec.bpe.mergeWord(symbols)) {
      ids.add(_spec.bpe.idOf(symbol)!);
    }
  }

  /// Adds [char]'s byte-fallback tokens to [symbols]; false, adding none, when one is missing.
  bool _addBytes(String char, List<String> symbols) {
    final bytes = utf8.encode(char);
    for (final b in bytes) {
      if (_byteTokens[b] == null) return false;
    }
    for (final b in bytes) {
      symbols.add(_byteTokens[b]!);
    }
    return true;
  }

  @override
  int get clsId => _spec.clsId;

  @override
  int get sepId => _spec.sepId;

  @override
  int get padId => _spec.padId;

  @override
  int get maskId => _spec.maskId;

  @override
  String get maskToken => _spec.maskToken;
}
