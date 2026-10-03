/// Language and script detection, ported from laya's `lang.py` (laya 0.3.24).
///
/// It decides one thing for routing: is a state English Latin text, or
/// something the English checkpoint cannot read? Script detection is exact;
/// the Latin-script language guess is laya's stopword and diacritic
/// heuristic, reproduced rule for rule so that [analyse] equals
/// `laya.lang.analyse` field for field and float for float.
///
/// Python `str` indexes code points, so text is handled as code-point lists,
/// and every character property comes from `py_unicode.dart` (Python 3.12,
/// Unicode 15.0.0) rather than Dart's own Unicode data.
library;

import '../model/decode.dart' show round4;
import 'py_unicode.dart';

/// `laya.lang.analyse(state)`; [toJson] is the Python dict, key for key.
class LanguageAnalysis {
  /// Creates an analysis.
  const LanguageAnalysis({
    required this.script,
    required this.scriptProfile,
    required this.language,
    required this.isEnglish,
    required this.languageUndecided,
    required this.diacriticRate,
    required this.nonLatinFraction,
    required this.mixedSegment,
  });

  /// The dominant script: `latin`, `han`, `devanagari`, ..., `other`, or
  /// `unknown` when the state has no letters.
  final String script;

  /// The fraction of letters in each script, Latin first.
  final Map<String, double> scriptProfile;

  /// The best-effort language code, or null when undecided or not Latin.
  final String? language;

  /// Whether the English checkpoint can be expected to read the state.
  final bool isEnglish;

  /// Whether nothing identified the language.
  final bool languageUndecided;

  /// The share of non-English letters, rounded to four places (Latin only).
  final double diacriticRate;

  /// The share of letters outside Latin, rounded to four places.
  final double nonLatinFraction;

  /// The line or field that made a mostly English state non-English.
  final String? mixedSegment;

  LanguageAnalysis _copyWith({
    String? language,
    bool? isEnglish,
    bool? languageUndecided,
    String? mixedSegment,
  }) => LanguageAnalysis(
    script: script,
    scriptProfile: scriptProfile,
    language: language,
    isEnglish: isEnglish ?? this.isEnglish,
    languageUndecided: languageUndecided ?? this.languageUndecided,
    diacriticRate: diacriticRate,
    nonLatinFraction: nonLatinFraction,
    mixedSegment: mixedSegment,
  );

  /// The analysis as laya's dict, in its key order.
  Map<String, Object?> toJson() => {
    'script': script,
    'script_profile': scriptProfile,
    'language': language,
    'is_english': isEnglish,
    'language_undecided': languageUndecided,
    'diacritic_rate': diacriticRate,
    'non_latin_fraction': nonLatinFraction,
    'mixed_segment': mixedSegment,
  };
}

// Unicode blocks the English checkpoint cannot read (lang.py:20).
const _scriptRanges = <(String, List<(int, int)>)>[
  ('greek', [(0x0370, 0x03FF), (0x1F00, 0x1FFF)]),
  ('cyrillic', [(0x0400, 0x052F), (0x2DE0, 0x2DFF), (0xA640, 0xA69F)]),
  ('armenian', [(0x0530, 0x058F)]),
  ('hebrew', [(0x0590, 0x05FF)]),
  (
    'arabic',
    [
      (0x0600, 0x06FF),
      (0x0750, 0x077F),
      (0x08A0, 0x08FF),
      (0xFB50, 0xFDFF),
      (0xFE70, 0xFEFF),
    ],
  ),
  ('devanagari', [(0x0900, 0x097F), (0xA8E0, 0xA8FF)]),
  ('bengali', [(0x0980, 0x09FF)]),
  ('gurmukhi', [(0x0A00, 0x0A7F)]),
  ('gujarati', [(0x0A80, 0x0AFF)]),
  ('oriya', [(0x0B00, 0x0B7F)]),
  ('tamil', [(0x0B80, 0x0BFF)]),
  ('telugu', [(0x0C00, 0x0C7F)]),
  ('kannada', [(0x0C80, 0x0CFF)]),
  ('malayalam', [(0x0D00, 0x0D7F)]),
  ('sinhala', [(0x0D80, 0x0DFF)]),
  ('thai', [(0x0E00, 0x0E7F)]),
  ('lao', [(0x0E80, 0x0EFF)]),
  ('tibetan', [(0x0F00, 0x0FFF)]),
  ('myanmar', [(0x1000, 0x109F)]),
  ('georgian', [(0x10A0, 0x10FF)]),
  ('ethiopic', [(0x1200, 0x137F)]),
  ('khmer', [(0x1780, 0x17FF)]),
  ('hangul', [(0x1100, 0x11FF), (0x3130, 0x318F), (0xAC00, 0xD7AF)]),
  ('kana', [(0x3040, 0x309F), (0x30A0, 0x30FF), (0x31F0, 0x31FF)]),
  ('han', [(0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF)]),
];

// Function words per language (lang.py:57). Python's `it` list repeats
// `sono`; a const set holds it once.
const _stop = <String, Set<String>>{
  'en': {
    'the', 'and', 'is', 'are', 'was', 'were', 'to', 'of', 'in', 'for', //
    'with', 'that', 'this', 'it', 'you', 'have', 'has', 'not', 'but', 'on',
    'at', 'be', 'as', 'from', 'will', 'can', 'would', 'there', 'their',
    'what', 'which', 'please', 'we', 'i',
  },
  'fr': {
    'le', 'la', 'les', 'des', 'une', 'est', 'pour', 'dans', 'que', 'qui', //
    'avec', 'sur', 'pas', 'plus', 'nous', 'vous', 'être', 'cette', 'mais',
    'sont', 'ont', 'aux', 'ce', 'et', 'du', 'au', 'ou', 'je', 'tu', 'il',
    'elle', 'ils', 'elles', 'mon', 'ton', 'ma', 'ta', 'sa', 'mes', 'tes',
    'ses', 'ces', 'deux', 'trois', 'très', 'bien', 'tout', 'tous', 'toute',
    'fait', 'veux', 'veut', 'peux', 'peut', 'dois', 'doit', 'merci',
    'bonjour', 'jour', 'jours', 'mois', 'fois', 'quand', 'comment',
    'pourquoi', 'alors', 'donc',
  },
  'de': {
    'der', 'die', 'das', 'und', 'ist', 'ein', 'eine', 'den', 'dem', //
    'nicht', 'mit', 'für', 'auf', 'von', 'zu', 'sich', 'auch', 'werden',
    'wurde', 'haben', 'sind', 'oder', 'aber', 'ich', 'wir', 'mir', 'mich',
    'dir', 'dich', 'uns', 'mein', 'meine', 'meinen', 'meinem', 'meiner',
    'diese', 'dieser', 'diesen', 'dieses', 'einen', 'einem', 'einer', 'wie',
    'wo', 'wann', 'welche', 'im', 'zum', 'zur', 'aus', 'bei', 'nach', 'noch',
    'bitte', 'heute', 'jetzt', 'kann', 'kannst', 'habe', 'gibt', 'wird',
    'in', 'was',
  },
  'es': {
    'el', 'los', 'las', 'que', 'por', 'con', 'para', 'una', 'es', 'se', //
    'del', 'como', 'pero', 'son', 'está', 'este', 'esta', 'todo', 'más',
    'muy', 'hay', 'sus', 'la', 'un', 'y', 'al', 'lo', 'le', 'les', 'su',
    'mi', 'tu', 'nos', 'ni', 'dos', 'tres', 'fue', 'fueron', 'ser', 'tiene',
    'tienen', 'tengo', 'puede', 'pueden', 'quiero', 'necesito', 'hemos',
    'han', 'sobre', 'entre', 'cuando', 'donde', 'porque', 'aunque',
    'también', 'ya', 'eso', 'esto', 'esa', 'ese', 'nada', 'algo', 'aquí',
    'hoy', 'gracias',
  },
  'pt': {
    'os', 'as', 'que', 'em', 'um', 'uma', 'para', 'com', 'não', 'é', 'se', //
    'do', 'da', 'dos', 'das', 'mas', 'são', 'está', 'este', 'esta', 'muito',
    'pelo', 'pela', 'o', 'e', 'na', 'nas', 'nos', 'ao', 'aos', 'por', 'foi',
    'era', 'ser', 'sou', 'tem', 'tenho', 'pode', 'podem', 'quero',
    'preciso', 'eu', 'meu', 'minha', 'seu', 'sua', 'isso', 'isto', 'aqui',
    'ali', 'como', 'quando', 'onde', 'porque', 'mais', 'já', 'ainda',
    'agora', 'hoje', 'ontem', 'dois', 'três', 'tudo', 'nada', 'obrigado',
    'olá', 'você', 'vocês', 'voce', 'voces', 'vc', 'vcs', 'nao', 'sao',
    'ja', 'até', 'tá', 'pra', 'gostaria', 'obrigada', 'também', 'tambem',
    'estou', 'estamos', 'meus', 'minhas', 'nosso', 'nossa', 'consigo',
    'cadê', 'boa', 'tarde', 'noite', 'depois', 'antes', 'então', 'entao',
    'ninguém', 'ninguem', 'alguém', 'alguem', 'nenhum', 'nenhuma', 'estava',
    'ficou', 'fiz', 'deu',
  },
  'it': {
    'il', 'lo', 'gli', 'che', 'di', 'per', 'con', 'non', 'è', 'si', 'del', //
    'della', 'sono', 'questo', 'questa', 'anche', 'come', 'più', 'nella',
    'alla', 'la', 'le', 'un', 'uno', 'una', 'e', 'ed', 'o', 'da', 'su',
    'tra', 'fra', 'mi', 'ci', 'ne', 'ho', 'hai', 'ha', 'abbiamo', 'avete',
    'hanno', 'era', 'stato', 'stata', 'devo', 'deve', 'devono', 'voglio',
    'vorrei', 'mio', 'mia', 'tuo', 'sua', 'quando', 'dove', 'perche',
    'molto', 'poco', 'sempre', 'mai', 'già', 'ancora', 'adesso', 'oggi',
    'ieri', 'grazie', 'ciao', 'scusa', 'nel', 'nell', 'negli', 'sul',
    'sulla', 'sulle', 'dal', 'dalla', 'dallo', 'dagli', 'dei', 'delle',
    'dello', 'degli', 'agli', 'alle', 'col',
  },
  'nl': {
    'het', 'een', 'van', 'is', 'op', 'te', 'dat', 'niet', 'met', 'voor', //
    'zijn', 'aan', 'door', 'maar', 'ook', 'worden', 'deze', 'naar', 'wordt',
  },
  'sv': {
    'jag', 'är', 'och', 'inte', 'att', 'från', 'till', 'behöver', 'får', //
    'skulle', 'ska', 'vill', 'måste', 'också', 'dessa', 'detta', 'säger',
    'upp', 'utan', 'mitt', 'min', 'om', 'kommer', 'här', 'två', 'vi',
    'nästa', 'gör', 'göra', 'hjälp', 'hjälpa', 'mig', 'återbetalning',
    'återbetala', 'faktura', 'gång', 'gånger', 'hittar', 'inställningen',
    'inställningarna', 'lösenord', 'när', 'öppnar', 'spårningen',
    'aterbetalning', 'aterbetala', 'behover', 'fel', 'ganger', 'hjalp',
    'hjalpa', 'installningen', 'installningarna', 'kraschar', 'kvittot',
    'losenord', 'nar', 'oppnar', 'paket', 'skicka', 'sparningen', 'tva',
    'uppdaterats', 'blivit', 'debiterade', 'appen',
  },
  'ro': {
    'și', 'să', 'este', 'sunt', 'care', 'pentru', 'din', 'dar', 'după', //
    'până', 'fără', 'ale', 'lui', 'în', 'fost', 'acum', 'vreau', 'trebuie',
    'foarte', 'acest', 'această', 'acesta', 'aceasta', 'mi', 'ți', 'vă',
    'nu',
  },
  'bn': {
    'ami', 'amar', 'amake', 'amra', 'amader', 'apni', 'apnar', 'apnake', //
    'apnara', 'tumi', 'tomar', 'tomake', 'tomra', 'tader', 'ota', 'eita',
    'oita', 'ekta', 'ei', 'oi', 'ki', 'keno', 'kivabe', 'kibhabe', 'kothay',
    'kokhon', 'kobe', 'koto', 'kintu', 'jodi', 'tahole', 'ar', 'theke',
    'jonno', 'sathe', 'shathe', 'diye', 'niye', 'moddhe', 'kore', 'korte',
    'korchi', 'korsi', 'korbo', 'korechi', 'koreche', 'korun', 'koren',
    'korlam', 'hobe', 'hoyeche', 'hoise', 'hocche', 'hoyni', 'chai',
    'chaina', 'lagbe', 'parchi', 'parbo', 'parchina', 'peyechi', 'paini',
    'dite', 'dilam', 'diyechi', 'nai', 'khub', 'onek', 'ekhon', 'akhon',
    'ekhono', 'abar', 'ekbar', 'duibar', 'ajke', 'kalke', 'taka', 'bhalo',
    'valo', 'kharap', 'shomossa', 'somossa', 'dhonnobad', 'bhai', 'shob',
    'keu', 'kichu', 'bolte', 'bolun', 'parben', 'asbe', 'jabe', 'pabo',
    'ferot', 'dorkar', 'hoye', 'geche', 'gese',
  },
  'az': {
    'və', 've', 'bir', 'bu', 'üçün', 'ucun', 'ilə', 'ile', 'olan', 'olub', //
    'olmasa', 'var', 'yox', 'yoxdur', 'mən', 'sən', 'biz', 'siz', 'onlar',
    'daha', 'çox', 'cox', 'hər', 'nə', 'kimi', 'görə', 'sonra', 'əgər',
    'eger', 'deyil', 'lakin', 'amma', 'ancaq', 'artıq', 'artiq', 'də', 'isə',
    'həm', 'yalnız', 'yalniz',
  },
};

// Swedish words specific enough for a two- or three-word state (lang.py:175).
const _shortSwedishWords = {
  'åtkomst', 'atkomst', 'lösenord', 'losenord', 'fakturan', 'betalningen', //
  'inloggningen', 'glömt', 'glomt', 'behöver', 'behover', 'återbetalning',
  'aterbetalning', 'kvitto', 'kvittot', 'spårningen', 'sparningen',
  'inställningen', 'installningen', 'felmeddelande', 'abonnemanget',
};

// Letters that ordinary English does not use (lang.py:185).
final Set<int> _nonEnDiacritics = {
  ...'àâäãáåçéèêëíìîïñóòôöõøúùûüýÿßæœ'.runes, // Western European
  ...'ăâîșțşţ'.runes, // Romanian
  ...'ąćęłńśźż'.runes, // Polish
  ...'čďěňřšťůž'.runes, // Czech / Slovak
  ...'őű'.runes, // Hungarian
  ...'ğı'.runes, // Turkish
  ...'āēģīķļņūž'.runes, // Baltic
  ...'đ'.runes, // Serbo-Croatian / Vietnamese
  ...'ə'.runes, // Azerbaijani
};

// Swedish words Danish shares (lang.py:205).
const _nordicOverlapWords = {
  'hej', 'ja', 'nej', 'jo', 'tack', 'mig', 'min', 'om', 'kommer', 'får', //
  'skulle', 'vi',
};

// Words more than one list claims, plus the Nordic overlap (lang.py:200).
final Set<String> _sharedWords = () {
  final counts = <String, int>{};
  for (final words in _stop.values) {
    for (final w in words) {
      counts[w] = (counts[w] ?? 0) + 1;
    }
  }
  return {
    for (final e in counts.entries)
      if (e.value > 1) e.key,
    ..._nordicOverlapWords,
  };
}();

// English function words no other list holds (lang.py:211).
final Set<String> _enOnlyWords = _stop['en']!.difference(_sharedWords);

// English words that count once however often they repeat (lang.py:222).
final Set<String> _enCollisionWords =
    {
      'come', 'son', 'do', 'care', 'todo', 'im', 'per', 'plus', //
    }.intersection({
      for (final e in _stop.entries)
        if (e.key != 'en') ...e.value,
    });

const _nonEnDiacriticRate = 0.02;
const _englishRescueDiacriticRate = 0.06;
const _nonLatinFraction = 0.2;
const _nonLatinMinFraction = 0.1;
const _nonLatinMinLetters = 10;

const _space = 0x20;
const _newline = 0x0A;

String _str(List<int> cps) => String.fromCharCodes(cps);

/// Python's `round(x)` for a float: nearest integer, ties to even.
int _roundHalfEven(double x) {
  final floor = x.floorToDouble();
  final diff = x - floor; // exact for a finite double
  final f = floor.toInt();
  if (diff > 0.5) return f + 1;
  if (diff < 0.5) return f;
  return f.isEven ? f : f + 1;
}

// _WORD.findall: maximal runs of `[^\W\d_]`.
List<String> _words(List<int> t) {
  final out = <String>[];
  int? start;
  for (var i = 0; i <= t.length; i++) {
    final letter = i < t.length && pyIsLetter(t[i]);
    if (letter) {
      start ??= i;
    } else if (start != null) {
      out.add(_str(t.sublist(start, i)));
      start = null;
    }
  }
  return out;
}

bool _isIdentChar(int c) => c == 0x2D || pyIsWord(c);

// _IDENTIFIER.sub(" ", t): `(?<![\w-])[\w-]*(?:[.@][\w-]+)+` becomes a space.
// `[\w-]` and `[.@]` are disjoint, so a match is the maximal run at its start
// followed by every `[.@]`-plus-run that comes after it.
List<int> _replaceIdentifiers(List<int> t) {
  final out = <int>[];
  var i = 0;
  while (i < t.length) {
    if (i == 0 || !_isIdentChar(t[i - 1])) {
      var k = i;
      while (k < t.length && _isIdentChar(t[k])) {
        k++;
      }
      var repeats = 0;
      while (k + 1 < t.length &&
          (t[k] == 0x2E || t[k] == 0x40) &&
          _isIdentChar(t[k + 1])) {
        k += 2;
        while (k < t.length && _isIdentChar(t[k])) {
          k++;
        }
        repeats++;
      }
      if (repeats > 0) {
        out.add(_space);
        i = k;
        continue;
      }
    }
    out.add(t[i]);
    i++;
  }
  return out;
}

// _CODE_LINE.search: `[=;{}\[\]]|\w\(`.
bool _isCodeLine(List<int> t) {
  for (var i = 0; i < t.length; i++) {
    final c = t[i];
    if (c == 0x3D ||
        c == 0x3B ||
        c == 0x7B ||
        c == 0x7D ||
        c == 0x5B ||
        c == 0x5D) {
      return true;
    }
    if (i + 1 < t.length && t[i + 1] == 0x28 && pyIsWord(c)) return true;
  }
  return false;
}

bool _isAlnumNotUnderscore(int c) => c != 0x5F && pyIsAlnum(c);

// _JOINED.search: `[^\W_][._/\\][^\W_]`.
bool _isJoined(List<int> t) {
  for (var i = 0; i + 2 < t.length; i++) {
    final j = t[i + 1];
    if ((j == 0x2E || j == 0x5F || j == 0x2F || j == 0x5C) &&
        _isAlnumNotUnderscore(t[i]) &&
        _isAlnumNotUnderscore(t[i + 2])) {
      return true;
    }
  }
  return false;
}

// _LETTER_RUN.sub: each maximal `[^\W\d_]` run of two or more that is all
// capitals becomes a space.
List<int> _dropAcronyms(List<int> t) {
  final out = <int>[];
  var i = 0;
  while (i < t.length) {
    if (!pyIsLetter(t[i])) {
      out.add(t[i]);
      i++;
      continue;
    }
    var j = i;
    while (j < t.length && pyIsLetter(t[j])) {
      j++;
    }
    final run = t.sublist(i, j);
    if (run.length >= 2 && pyStrIsUpper(run)) {
      out.add(_space);
    } else {
      out.addAll(run);
    }
    i = j;
  }
  return out;
}

// _iter_text: the string leaves of a state; keys are ignored.
List<List<int>> _iterText(Object? state, [int depth = 0]) {
  if (depth > 6 || state == null) return [];
  return switch (state) {
    final String s => [s.runes.toList()],
    final Map<Object?, Object?> m => [
      for (final v in m.values) ..._iterText(v, depth + 1),
    ],
    final List<Object?> l => [for (final v in l) ..._iterText(v, depth + 1)],
    _ => [],
  };
}

// state_text: the leaves joined by spaces, at most [maxChars] code points.
List<int> _stateText(Object? state, [int maxChars = 4000]) {
  final parts = <List<int>>[];
  var budget = maxChars;
  for (final leaf in _iterText(state)) {
    if (budget <= 0) break;
    if (leaf.length > budget) {
      parts.add(leaf.sublist(0, budget));
      break;
    }
    parts.add(leaf);
    budget -= leaf.length + 1;
  }
  final joined = <int>[];
  for (final (i, p) in parts.indexed) {
    if (i > 0) joined.add(_space);
    joined.addAll(p);
  }
  return joined.length > maxChars ? joined.sublist(0, maxChars) : joined;
}

int _countAlpha(List<int> t) => t.where(pyIsAlpha).length;

// _script_counts: letters per script, in first-seen order, Latin last.
Map<String, int> _scriptCounts(List<int> text) {
  final counts = <String, int>{};
  var latin = 0;
  for (final cp in text) {
    if (!pyIsAlpha(cp)) continue;
    if (cp < 0x02B0 ||
        (0x1E00 <= cp && cp <= 0x1EFF) ||
        (0xFF21 <= cp && cp <= 0xFF3A) ||
        (0xFF41 <= cp && cp <= 0xFF5A)) {
      latin++;
      continue;
    }
    final name = _rangeScript(cp) ?? 'other';
    counts[name] = (counts[name] ?? 0) + 1;
  }
  counts['latin'] = latin;
  return counts;
}

String? _rangeScript(int cp) {
  for (final (name, ranges) in _scriptRanges) {
    for (final (lo, hi) in ranges) {
      if (lo <= cp && cp <= hi) return name;
    }
  }
  return null;
}

// Python's max(..., key=...): the first of the largest.
String? _firstMax(Map<String, num> m) {
  String? best;
  num? value;
  for (final e in m.entries) {
    if (value == null || e.value > value) {
      best = e.key;
      value = e.value;
    }
  }
  return best;
}

String _scriptFromCounts(Map<String, int> counts) =>
    counts.values.any((v) => v != 0) ? _firstMax(counts)! : 'unknown';

Map<String, double> _profileFromCounts(Map<String, int> counts) {
  final total = counts.values.fold(0, (a, b) => a + b);
  if (total == 0) return {};
  final ordered = <String, int>{
    if (counts['latin'] case final latin? when latin != 0) 'latin': latin,
    for (final e in counts.entries)
      if (e.key != 'latin') e.key: e.value,
  };
  return {
    for (final e in ordered.entries)
      if (e.value != 0) e.key: e.value / total,
  };
}

// _script_of: the named non-Latin script of one character, else null.
String? _scriptOf(int cp) {
  if (cp < 0x0250 ||
      (0x1E00 <= cp && cp <= 0x1EFF) ||
      (0xFF21 <= cp && cp <= 0xFF3A) ||
      (0xFF41 <= cp && cp <= 0xFF5A)) {
    return null;
  }
  return _rangeScript(cp);
}

// _non_latin_words: non-Latin runs that read as words, not annotation.
List<List<int>> _nonLatinWords(List<int> text) {
  final runs = <List<int>>[];
  var cur = <int>[];
  String? script;
  for (final cp in text) {
    if (pyIsCombining(cp)) continue;
    final s = _scriptOf(cp);
    if (s != null && s == script) {
      cur.add(cp);
      continue;
    }
    if (cur.isNotEmpty) runs.add(cur);
    if (s != null) {
      cur = [cp];
      script = s;
    } else {
      cur = [];
      script = null;
    }
  }
  if (cur.isNotEmpty) runs.add(cur);
  return [
    for (final w in runs)
      if (w.length >= 2 && !pyIsUpper(w[0])) w,
  ];
}

bool _hasDiacritic(String w) => w.runes.any(_nonEnDiacritics.contains);

// _english_rescued_by_words (lang.py:410).
bool _englishRescuedByWords(List<String> words, double diacRate) {
  if (diacRate >= _englishRescueDiacriticRate) return false;
  final distinct = words.toSet();
  if (distinct.intersection(_enOnlyWords).length < 2) return false;
  return distinct.where(_hasDiacritic).length <= 1;
}

typedef _LatinProfile = ({
  String? language,
  int englishHits,
  double diacriticRate,
  bool looksNonEnglish,
});

// latin_profile (lang.py:427).
_LatinProfile _latinProfile(List<int> text) {
  final cleaned = [
    for (final c in _replaceIdentifiers(text)) c == 0x130 ? 0x69 : c,
  ];
  final words = _words(pyLower(cleaned));
  final lowered = pyLower(text);
  final diac = lowered.where(_nonEnDiacritics.contains).length;
  final diacRate = diac / (lowered.isEmpty ? 1 : lowered.length);
  final nonEnglish = diacRate >= _nonEnDiacriticRate;
  final distinct = words.toSet();
  final nordicOverlap =
      distinct.intersection(_nordicOverlapWords).isNotEmpty &&
      distinct.intersection(_enOnlyWords).isEmpty;
  if (1 < words.length &&
      words.length < 4 &&
      distinct.intersection(_shortSwedishWords).isNotEmpty) {
    return (
      language: 'sv',
      englishHits: 0,
      diacriticRate: diacRate,
      looksNonEnglish: nonEnglish,
    );
  }
  if (words.length < 4) {
    return (
      language: null,
      englishHits: 0,
      diacriticRate: diacRate,
      looksNonEnglish: nonEnglish || nordicOverlap,
    );
  }

  final counts = <String, int>{};
  for (final w in words) {
    counts[w] = (counts[w] ?? 0) + 1;
  }
  final scores = {
    for (final e in _stop.entries)
      e.key: counts.entries
          .where((c) => e.value.contains(c.key))
          .fold(
            0,
            (sum, c) => sum + (_enCollisionWords.contains(c.key) ? 1 : c.value),
          ),
  };
  final en = scores['en'] ?? 0;
  final evidenced = {
    for (final e in scores.entries)
      if (e.key != 'en' &&
          distinct
              .intersection(_stop[e.key]!)
              .any((w) => !_sharedWords.contains(w)))
        e.key: e.value,
  };
  final bestLg = _firstMax(evidenced);
  final best = bestLg == null ? 0 : evidenced[bestLg]!;

  String? lang;
  if (bestLg != null && best >= _max(2, en + 2)) {
    lang = bestLg;
  } else if (bestLg == 'sv' &&
      words.contains('inte') &&
      words.contains('kan') &&
      const {'kan', 'jag', 'vi'}.contains(words[0]) &&
      en <= 1) {
    lang = bestLg;
  } else if (bestLg != null && nonEnglish && best >= _max(2, en)) {
    lang = bestLg;
  } else if (en != 0 &&
      (!nonEnglish || _englishRescuedByWords(words, diacRate))) {
    lang = 'en';
  }
  return (
    language: lang,
    englishHits: en,
    diacriticRate: diacRate,
    looksNonEnglish: nonEnglish || (lang == null && nordicOverlap),
  );
}

int _max(int a, int b) => a > b ? a : b;

// _named_prose_language (lang.py:517).
String? _namedProseLanguage(List<int> segment) {
  if (pyStrip(segment).isEmpty || _isCodeLine(segment)) return null;
  var prose = <int>[];
  for (final tok in pySplit(segment)) {
    if (_isJoined(tok)) continue;
    if (prose.isNotEmpty) prose.add(_space);
    prose.addAll(tok);
  }
  if (prose.any(pyIsLower)) prose = _dropAcronyms(prose);
  final tokens = _words(prose);
  if (tokens.length < 4) return null;
  final lang = _latinProfile(prose).language;
  if (lang == null || lang == 'en') return null;
  final lowered = {for (final w in tokens) _str(pyLower(w.runes.toList()))};
  if (lowered.intersection(_stop[lang] ?? const {}).length < 2) return null;
  return lang;
}

// _non_english_segment (lang.py:539): the first line or field named a
// non-English language, as (language, stripped segment).
(String, String)? _nonEnglishSegment(Object? state, [int maxChars = 4000]) {
  var seen = 0;
  for (final leaf in _iterText(state)) {
    for (var seg in pySplitOn(leaf, _newline)) {
      if (seen >= maxChars) return null;
      if (seg.length > maxChars - seen) seg = seg.sublist(0, maxChars - seen);
      seen += seg.length;
      final lang = _namedProseLanguage(seg);
      if (lang != null) return (lang, _str(pyStrip(seg)));
    }
  }
  return null;
}

// _analyse_text (lang.py:562).
LanguageAnalysis _analyseText(List<int> text) {
  final counts = _scriptCounts(text);
  final prof = _profileFromCounts(counts);
  var script = _scriptFromCounts(counts);
  final nonLatin = prof.isNotEmpty ? round4(1.0 - (prof['latin'] ?? 0.0)) : 0.0;
  final nNonLatin = _roundHalfEven(nonLatin * _countAlpha(text));
  if (script == 'latin' &&
      _nonLatinWords(text).isNotEmpty &&
      (nonLatin >= _nonLatinFraction ||
          (nonLatin >= _nonLatinMinFraction &&
              nNonLatin >= _nonLatinMinLetters))) {
    script = _firstMax({
      for (final e in prof.entries)
        if (e.key != 'latin') e.key: e.value,
    })!;
  }
  if (script == 'unknown') {
    return LanguageAnalysis(
      script: 'unknown',
      scriptProfile: prof,
      language: null,
      isEnglish: true,
      languageUndecided: true,
      diacriticRate: 0.0,
      nonLatinFraction: 0.0,
      mixedSegment: null,
    );
  }
  if (script != 'latin') {
    return LanguageAnalysis(
      script: script,
      scriptProfile: prof,
      language: null,
      isEnglish: false,
      languageUndecided: true,
      diacriticRate: 0.0,
      nonLatinFraction: nonLatin,
      mixedSegment: null,
    );
  }
  final latin = _latinProfile(text);
  final lang = latin.language;
  final undecided = lang == null;
  return LanguageAnalysis(
    script: 'latin',
    scriptProfile: prof,
    language: lang,
    isEnglish: lang == 'en' || (undecided && !latin.looksNonEnglish),
    languageUndecided: undecided,
    diacriticRate: round4(latin.diacriticRate),
    nonLatinFraction: nonLatin,
    mixedSegment: null,
  );
}

// _leaf_non_english (lang.py:599).
LanguageAnalysis? _leafNonEnglish(List<int> leaf) {
  var bestN = -1;
  LanguageAnalysis? best;
  for (final line in pySplitOn(leaf, _newline)) {
    if (line.length < 7) continue;
    final sample = line.length > 4000 ? line.sublist(0, 4000) : line;
    if (pyStrip(sample).isEmpty || _isCodeLine(sample)) continue;
    final det = _analyseText(sample);
    if (det.isEnglish) continue;
    if (det.language != null && det.language != 'en') {
      if (_namedProseLanguage(sample) == null) continue;
    } else if (det.script != 'latin' && det.script != 'unknown') {
      if (!(_nonLatinWords(sample).isNotEmpty &&
          _countAlpha(sample) >= _nonLatinMinLetters)) {
        continue;
      }
    } else if (!(det.languageUndecided &&
        det.diacriticRate >= _nonEnDiacriticRate &&
        _words(sample).length >= 4)) {
      continue;
    }
    final nAlpha = _countAlpha(sample);
    if (nAlpha > bestN) {
      bestN = nAlpha;
      best = det;
    }
  }
  return best;
}

/// `laya.lang.analyse(state)`: the detection result for a state.
///
/// [state] is decoded JSON: a string, a map, a list, a scalar or null. Only
/// string values are read; map keys are ignored. One non-English line or
/// field makes an otherwise English state non-English.
LanguageAnalysis analyse(Object? state) {
  var result = _analyseText(_stateText(state));
  if (result.script == 'latin' && result.isEnglish) {
    final leaves = _iterText(state);
    if (leaves.length > 1 || leaves.any((l) => l.contains(_newline))) {
      final found = _nonEnglishSegment(state);
      if (found != null) {
        result = result._copyWith(
          language: found.$1,
          isEnglish: false,
          languageUndecided: false,
          mixedSegment: found.$2,
        );
      }
    }
  }
  if (state is String || state == null || !result.isEnglish) return result;
  var bestN = -1;
  LanguageAnalysis? best;
  for (final leaf in _iterText(state)) {
    final det = _leafNonEnglish(leaf);
    if (det == null) continue;
    final nAlpha = _countAlpha(
      leaf.length > 4000 ? leaf.sublist(0, 4000) : leaf,
    );
    if (nAlpha > bestN) {
      bestN = nAlpha;
      best = det;
    }
  }
  if (best == null) return result;
  return result._copyWith(
    language: best.language,
    isEnglish: false,
    languageUndecided: best.languageUndecided,
    mixedSegment: result.mixedSegment,
  );
}
