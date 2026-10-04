import 'dart:convert';

import 'added_tokens.dart';
import 'bpe.dart';
import 'nfc.dart';
import 'tokenizer.dart';

/// The ModernBERT tokenizer of the english and typed-decisions checkpoints: an NFC normalizer,
/// GPT-2 ByteLevel pre-tokenization and BPE.
///
/// Mirrors Hugging Face `tokenizers` 0.23.2 with `add_special_tokens=False`:
///
/// 1. added tokens are split out ([AddedTokenSplitter]), and the text between them is NFC
///    normalized;
/// 2. each text piece is cut with the GPT-2 pre-tokenizer regex;
/// 3. each cut's UTF-8 bytes are mapped to the GPT-2 byte alphabet and merged by [BpeModel].
class ByteLevelTokenizer implements LayaTokenizer {
  /// Builds the tokenizer from a parsed tokenizer.json.
  ///
  /// Throws an [UnsupportedError] for a normalizer other than NFC, or a ByteLevel
  /// pre-tokenizer with `add_prefix_space` or without `use_regex`.
  factory ByteLevelTokenizer(TokenizerSpec spec) {
    final normalizer = spec.normalizer?['type'];
    if (normalizer != 'NFC') {
      throw UnsupportedError('normalizer "$normalizer" is not supported');
    }
    final pre = spec.preTokenizer;
    if (pre['add_prefix_space'] == true) {
      throw UnsupportedError('ByteLevel add_prefix_space is not supported');
    }
    if (pre['use_regex'] == false) {
      throw UnsupportedError('ByteLevel without use_regex is not supported');
    }
    return ByteLevelTokenizer._(spec, spec.splitter(nfc));
  }

  ByteLevelTokenizer._(this._spec, this._splitter);

  final TokenizerSpec _spec;
  final AddedTokenSplitter _splitter;

  /// HF's `\s`: Oniguruma's Unicode `White_Space`, which, unlike JavaScript's `\s`, includes
  /// U+0085 and excludes U+FEFF.
  static const _space =
      r'\t\n\v\f\r \x85\xA0\u{1680}\u{2000}-\u{200A}\u{2028}\u{2029}\u{202F}\u{205F}\u{3000}';

  /// The GPT-2 pre-tokenizer pattern, as HF's ByteLevel uses it.
  static final _split = RegExp(
    "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^$_space\\p{L}\\p{N}]+"
    '|[$_space]+(?![^$_space])|[$_space]+',
    unicode: true,
  );

  /// GPT-2's byte-to-character table (`bytes_to_unicode`): printable Latin-1 bytes map to
  /// themselves, the rest to U+0100 onward in byte order.
  static final List<String> _byteChars = () {
    final table = List<String>.filled(256, '');
    var next = 256;
    for (var b = 0; b < 256; b++) {
      final printable =
          (b >= 0x21 && b <= 0x7E) ||
          (b >= 0xA1 && b <= 0xAC) ||
          (b >= 0xAE && b <= 0xFF);
      table[b] = String.fromCharCode(printable ? b : next++);
    }
    return table;
  }();

  @override
  List<int> encode(String text) {
    final ids = <int>[];
    for (final piece in _splitter.split(text)) {
      switch (piece) {
        case TokenPiece(:final id):
          ids.add(id);
        case TextPiece(text: final normalized):
          // The splitter has already applied the NFC normalizer to text pieces.
          for (final match in _split.allMatches(normalized)) {
            _encodeWord(match[0]!, ids);
          }
      }
    }
    return ids;
  }

  void _encodeWord(String word, List<int> ids) {
    final symbols = [for (final b in utf8.encode(word)) _byteChars[b]];
    for (final symbol in _spec.bpe.mergeWord(symbols)) {
      final id = _spec.bpe.idOf(symbol);
      if (id == null) {
        // Every byte is in a ByteLevel vocab, so a miss is a bug or a broken file.
        throw StateError('"$symbol" is not in the vocab');
      }
      ids.add(id);
    }
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
