/// One piece of input after added-token splitting: either raw text or a matched added token.
sealed class Piece {}

/// A run of text that still has to be pre-tokenized and run through the model.
///
/// After [AddedTokenSplitter.split] this text is already normalized.
final class TextPiece extends Piece {
  /// Wraps [text].
  TextPiece(this.text);

  /// The text, normalized.
  final String text;

  @override
  String toString() => 'TextPiece(${Uri.encodeFull(text)})';
}

/// An added token that matched, emitted as its id.
final class TokenPiece extends Piece {
  /// Wraps [id].
  TokenPiece(this.id);

  /// The added token's id.
  final int id;

  @override
  String toString() => 'TokenPiece($id)';
}

/// One `added_tokens` entry of a tokenizer.json.
class _AddedToken {
  _AddedToken(Map<String, Object?> json)
    : id = json['id']! as int,
      content = json['content']! as String,
      singleWord = json['single_word'] as bool? ?? false,
      lstrip = json['lstrip'] as bool? ?? false,
      rstrip = json['rstrip'] as bool? ?? false,
      normalized = json['normalized'] as bool? ?? true;

  final int id;
  final String content;
  final bool singleWord;
  final bool lstrip;
  final bool rstrip;
  final bool normalized;
}

/// Leftmost-longest matching over one set of added tokens.
class _Matcher {
  _Matcher(this.byContent)
    : pattern = RegExp(
        (byContent.keys.toList()..sort((a, b) => b.length.compareTo(a.length)))
            .map(RegExp.escape)
            .join('|'),
      );

  /// Tokens by the content they match (for normalized tokens, the normalized content).
  final Map<String, _AddedToken> byContent;

  /// An alternation of every content, longest first, so the first alternative that matches at
  /// the leftmost position is the longest.
  final RegExp pattern;
}

/// Splits text around added tokens the way HF's AddedVocabulary does.
///
/// Mirrors `AddedVocabulary::extract_and_normalize` in Hugging Face `tokenizers` 0.23.2:
///
/// 1. tokens with `normalized: false` are matched on the raw text;
/// 2. every remaining text piece is normalized, and tokens with `normalized: true` are matched
///    on it, their own content normalized the same way.
///
/// In each phase matches are leftmost-longest and do not overlap. A `single_word` match needs a
/// non-word character or the piece's edge on both sides, and one that fails is skipped rather
/// than retried shorter. `lstrip` takes the whitespace before a match into the token, back to
/// the end of the previous match; `rstrip` takes the whitespace after it. Whitespace is Unicode
/// `White_Space`, as Rust's `char::is_whitespace`. Empty text pieces are dropped.
class AddedTokenSplitter {
  /// Builds a splitter from tokenizer.json `added_tokens` entries and the tokenizer's normalizer.
  ///
  /// Entries with empty content are ignored, as HF ignores them.
  factory AddedTokenSplitter(
    List<Map<String, Object?>> addedTokens, {
    required String Function(String) normalize,
  }) {
    final raw = <String, _AddedToken>{};
    final normalized = <String, _AddedToken>{};
    for (final json in addedTokens) {
      final token = _AddedToken(json);
      if (token.content.isEmpty) continue;
      if (token.normalized) {
        final content = normalize(token.content);
        if (content.isNotEmpty) normalized[content] = token;
      } else {
        raw[token.content] = token;
      }
    }
    return AddedTokenSplitter._(
      normalize,
      raw.isEmpty ? null : _Matcher(raw),
      normalized.isEmpty ? null : _Matcher(normalized),
    );
  }

  AddedTokenSplitter._(this._normalize, this._raw, this._normalized);

  final String Function(String) _normalize;
  final _Matcher? _raw;
  final _Matcher? _normalized;

  /// Splits [text] into normalized text pieces and matched added tokens, in order.
  List<Piece> split(String text) {
    final phase1 = _raw == null ? [TextPiece(text)] : _splitOn(text, _raw);
    final out = <Piece>[];
    for (final piece in phase1) {
      if (piece is! TextPiece) {
        out.add(piece);
        continue;
      }
      if (piece.text.isEmpty) continue;
      final normalized = _normalize(piece.text);
      if (normalized.isEmpty) continue;
      if (_normalized == null) {
        out.add(TextPiece(normalized));
      } else {
        out.addAll(_splitOn(normalized, _normalized));
      }
    }
    return out;
  }

  /// HF `AddedVocabulary::split_with_indices` on one piece.
  static List<Piece> _splitOn(String sentence, _Matcher matcher) {
    final out = <Piece>[];
    var startOffset = 0;
    for (final match in matcher.pattern.allMatches(sentence)) {
      final token = matcher.byContent[match[0]]!;
      var start = match.start;
      var stop = match.end;
      if (token.singleWord) {
        final startSpace =
            start == 0 || !_isWordChar(_codePointBefore(sentence, start));
        final stopSpace =
            stop == sentence.length ||
            !_isWordChar(_codePointAt(sentence, stop));
        if (!startSpace || !stopSpace) continue;
      }
      if (token.lstrip) {
        var newStart = start;
        while (newStart > 0 &&
            _isWhitespace(sentence.codeUnitAt(newStart - 1))) {
          newStart--;
        }
        // Strip only whitespace, never into the previous match.
        start = newStart > startOffset ? newStart : startOffset;
      }
      if (token.rstrip) {
        while (stop < sentence.length &&
            _isWhitespace(sentence.codeUnitAt(stop))) {
          stop++;
        }
      }
      if (startOffset < start) {
        out.add(TextPiece(sentence.substring(startOffset, start)));
      }
      out.add(TokenPiece(token.id));
      // As in HF, an rstrip can run past the start of the next match; that match is still
      // emitted and this offset then moves back to its end.
      startOffset = stop;
    }
    if (startOffset < sentence.length) {
      out.add(TextPiece(sentence.substring(startOffset)));
    }
    return out;
  }

  static int _codePointAt(String s, int i) {
    final unit = s.codeUnitAt(i);
    if (unit >= 0xD800 && unit < 0xDC00 && i + 1 < s.length) {
      final low = s.codeUnitAt(i + 1);
      if (low >= 0xDC00 && low < 0xE000) {
        return 0x10000 + ((unit - 0xD800) << 10) + (low - 0xDC00);
      }
    }
    return unit;
  }

  static int _codePointBefore(String s, int i) {
    final unit = s.codeUnitAt(i - 1);
    if (unit >= 0xDC00 && unit < 0xE000 && i >= 2) {
      final high = s.codeUnitAt(i - 2);
      if (high >= 0xD800 && high < 0xDC00) {
        return 0x10000 + ((high - 0xD800) << 10) + (unit - 0xDC00);
      }
    }
    return unit;
  }

  /// Unicode `\w`, as HF's single_word check uses: Alphabetic, marks, decimal digits,
  /// connector punctuation and Join_Control.
  static final _word = RegExp(
    r'^[\p{Alphabetic}\p{M}\p{Nd}\p{Pc}\p{Join_Control}]$',
    unicode: true,
  );

  static bool _isWordChar(int codePoint) =>
      _word.hasMatch(String.fromCharCode(codePoint));

  /// Unicode `White_Space` (Rust `char::is_whitespace`); every such character is in the BMP.
  static bool _isWhitespace(int unit) =>
      (unit >= 0x09 && unit <= 0x0D) ||
      unit == 0x20 ||
      unit == 0x85 ||
      unit == 0xA0 ||
      unit == 0x1680 ||
      (unit >= 0x2000 && unit <= 0x200A) ||
      unit == 0x2028 ||
      unit == 0x2029 ||
      unit == 0x202F ||
      unit == 0x205F ||
      unit == 0x3000;
}
