import 'dart:convert';
import 'dart:typed_data';

import 'package:laya_server/src/errors.dart';
import 'package:laya_server/src/json/python_json.dart';
import 'package:laya_server/src/server/body.dart';
import 'package:laya_server/src/server/limits.dart';
import 'package:test/test.dart';

/// The [ApiException] [body] throws, as `status detail`.
String _refusal(void Function() body) {
  try {
    body();
  } on ApiException catch (e) {
    return '${e.status} ${e.detail}';
  }
  return 'accepted';
}

void main() {
  group('hasLoneSurrogate walks like laya _has_lone_surrogate', () {
    test('finds lone surrogates in values, keys and nesting', () {
      expect(hasLoneSurrogate('\udc00'), isTrue);
      expect(hasLoneSurrogate('x\ud800'), isTrue);
      expect(hasLoneSurrogate('\ude00\ud83d'), isTrue); // reversed pair
      expect(hasLoneSurrogate('\ud83d😀'), isTrue);
      expect(
        hasLoneSurrogate({
          'a': [
            1,
            {'b': '\udfff'},
          ],
        }),
        isTrue,
      );
      expect(
        hasLoneSurrogate({
          '\ud800': 1,
          'b': [null, true, 2.5],
        }),
        isTrue,
      );
    });

    test('accepts paired surrogates and plain values', () {
      expect(hasLoneSurrogate('😀 emoji'), isFalse);
      expect(hasLoneSurrogate({'k': '\u{1F600}', 'n': 1}), isFalse);
      expect(hasLoneSurrogate([null, false, 1, 2.5, 'text']), isFalse);
    });

    test('is iterative: a deep body does not overflow the stack', () {
      Object deep = '\udc00';
      for (var i = 0; i < 200000; i++) {
        deep = [deep];
      }
      expect(hasLoneSurrogate(deep), isTrue);
    });
  });

  group('stateLengthLowerBoundOver is laya _state_length_lower_bound_over', () {
    test('sums string lengths in code points, ignoring keys', () {
      expect(stateLengthLowerBoundOver({'a': 'x' * 50001}, 50000), 50001);
      // Astral characters count once, as Python's len does.
      expect(stateLengthLowerBoundOver(['\u{1F600}' * 50001], 50000), 50001);
      expect(stateLengthLowerBoundOver({'k' * 60000: 'v'}, 50000), 0);
    });

    test('pops last-in first, and gives up at the first scalar', () {
      // `5` is popped first (last value), so the walk stops: not proven.
      expect(stateLengthLowerBoundOver({'a': 'x' * 60000, 'b': 5}, 50000), 0);
      expect(
        stateLengthLowerBoundOver({'b': 5, 'a': 'x' * 60000}, 50000),
        60000,
      );
      expect(stateLengthLowerBoundOver([null, 'x' * 60000], 50000), 60000);
    });

    test('inspects at most 64 values', () {
      final strings = List.filled(70, 'y' * 1000);
      // The list itself is one of the 64, then strings until past the cap.
      expect(stateLengthLowerBoundOver(strings, 50000), 51000);
      final small = List.filled(100, 'y' * 700);
      expect(stateLengthLowerBoundOver(small, 50000), 0);
    });
  });

  group('stateLength is laya _state_length', () {
    test('a string state is its length in code points', () {
      expect(stateLength('abc'), 3);
      expect(stateLength('\u{1F600}'), 1);
    });

    test('a JSON state is its json.dumps(ensure_ascii=False) length', () {
      final state = {'body': '"' * 30000};
      expect(stateLength(state), pythonJsonDumps(state).length);
      expect(stateLength(state), 60012);
      expect(stateLength(List.filled(20000, 1)), 60000);
      expect(stateLength({'z': '​'}), 10);
      expect(stateLength(5), 1);
    });
  });

  group('checkRequestLimits is laya _check_request_limits', () {
    Map<String, Object?> choice(int n) => {
      'type': 'choice',
      'criteria': {for (var i = 0; i < n; i++) 'o$i': 'o$i'},
    };
    Map<String, Object?> score(int n, {int? nullAt}) => {
      'type': 'score',
      'criteria': [for (var i = 0; i < n; i++) i == nullAt ? null : 'l$i'],
    };
    String check(Object? state, Object? questions) =>
        _refusal(() => checkRequestLimits(state, questions));

    test('state before questions, questions before counts', () {
      expect(check(null, []), "400 'state' is required");
      expect(check('x', null), "400 'questions' must be an object");
      expect(
        check('x', {for (var i = 0; i < 65; i++) 'q$i': <String, Object?>{}}),
        '413 too many questions (65 > 64)',
      );
    });

    test('option counts, with the id as Python repr', () {
      expect(
        check('x', {'q': choice(101)}),
        "413 too many choice options for 'q' (101 > 100)",
      );
      expect(
        check('x', {
          "it's": {'type': 'choice', 'criteria': List.filled(101, 'a')},
        }),
        '413 too many choice options for "it\'s" (101 > 100)',
      );
      expect(
        check('x', {'q': score(33, nullAt: 0)}),
        "413 too many score levels for 'q' (33 > 32)",
      );
      expect(
        check('x', {'q': score(4, nullAt: 2)}),
        "422 score question 'q' has a null level at index 2; "
        'give every level a description',
      );
      expect(
        check('x', {
          for (var i = 0; i < 5; i++) 'c$i': choice(100),
          's': score(13),
        }),
        '413 too many answer options across questions (513 > 512)',
      );
      expect(
        check('x', {
          for (var i = 0; i < 5; i++) 'c$i': choice(100),
          's': score(12),
        }),
        'accepted',
      );
    });

    test('ignores what it does not count', () {
      expect(
        check('x', {
          'a': 'not a question',
          'b': {'type': 'choice', 'criteria': 'text'},
          'c': {
            'type': 'score',
            'criteria': {'0': null},
          },
          'd': {'type': 'noul'},
        }),
        'accepted',
      );
    });

    test('the state size, measured or bounded', () {
      expect(check('x' * 50000, {}), 'accepted');
      expect(
        check('x' * 50001, {}),
        '413 state too large (50001 > 50000 chars)',
      );
      expect(
        check({'body': '"' * 30000}, {}),
        '413 state too large (60012 > 50000 chars)',
      );
      expect(
        check({'a': 'x' * 50001}, {}),
        '413 state too large (50001 > 50000 chars)',
      );
    });
  });

  group('body controls', () {
    String budget(Object? value, [int cap = 8192]) =>
        _refusal(() => validateBudgetParam({'max_len': value}, 'max_len', cap));

    test('validateBudgetParam is laya _validate_budget_param', () {
      expect(budget(0), '422 max_len must be a positive integer');
      expect(budget(-3), '422 max_len must be a positive integer');
      expect(budget('5'), '422 max_len must be an integer');
      expect(budget(true), '422 max_len must be an integer');
      expect(budget(5.0), '422 max_len must be an integer');
      expect(budget(9000), '422 max_len exceeds server limit (9000 > 8192)');
      expect(budget(101, 100), '422 max_len exceeds server limit (101 > 100)');
      expect(validateBudgetParam({'max_len': 8192}, 'max_len', 8192), 8192);
      expect(validateBudgetParam({'max_len': null}, 'max_len', 8192), isNull);
      expect(validateBudgetParam({}, 'max_len', 8192), isNull);
    });

    test('refuseBodyRefusals lists the sent hook keys, sorted', () {
      const tail =
          ' run inside the server process and cannot be sent to this '
          'endpoint; install them where laya-serve runs, or drop them';
      expect(
        _refusal(() => refuseBodyRefusals({'hooks': []})),
        '422 hooks$tail',
      );
      expect(
        _refusal(
          () => refuseBodyRefusals({
            'on_predict_start': 'x',
            'hooks_timeout': 1,
            'hooks_raise': false,
            'on_predict_end': null,
          }),
        ),
        '422 hooks_raise, hooks_timeout, on_predict_start$tail',
      );
      expect(_refusal(() => refuseBodyRefusals({'hooks': null})), 'accepted');
    });

    test('validateLanguageParam is laya _validate_language_param', () {
      expect(
        _refusal(() => validateLanguageParam({'lang': true}, 'lang')),
        '422 lang must be a language code string such as "de", or null',
      );
      expect(
        _refusal(() => validateLanguageParam({'lang_guess': 1}, 'lang_guess')),
        '422 lang_guess must be a language code string such as "de", or null',
      );
      expect(validateLanguageParam({'lang': 'de'}, 'lang'), 'de');
      expect(validateLanguageParam({'lang': null}, 'lang'), isNull);
    });

    test('validateMinConfidence round-trips check_min_confidence', () {
      expect(
        _refusal(() => validateMinConfidence({'min_confidence': 2})),
        '422 min_confidence must be a float in [0.0, 1.0], got 2',
      );
      expect(validateMinConfidence({'min_confidence': null}), isNull);
      expect(validateMinConfidence({'min_confidence': 0.5})?.toJson(), 0.5);
      expect(
        validateMinConfidence({
          'min_confidence': {'default': 0.5},
        })?.toJson(),
        {'default': 0.5},
      );
    });
  });

  group('decodeJsonBody is Python json.loads(bytes)', () {
    Object? decode(List<int> bytes) =>
        decodeJsonBody(Uint8List.fromList(bytes)).json;
    String refusal(List<int> bytes) =>
        _refusal(() => decodeJsonBody(Uint8List.fromList(bytes)));
    const invalid = '400 request body must be valid JSON';
    final doc = jsonEncode({'s': 'hé \u{1F600}'});

    test('detects UTF-8, UTF-16 and UTF-32, with or without a BOM', () {
      final want = jsonDecode(doc);
      List<int> utf16(String s, Endian e, {bool bom = false}) {
        final b = ByteData(s.codeUnits.length * 2);
        for (var i = 0; i < s.codeUnits.length; i++) {
          b.setUint16(i * 2, s.codeUnits[i], e);
        }
        return [
          if (bom) ...(e == Endian.little ? [0xff, 0xfe] : [0xfe, 0xff]),
          ...b.buffer.asUint8List(),
        ];
      }

      List<int> utf32(String s, Endian e, {bool bom = false}) {
        final runes = [if (bom) 0xfeff, ...s.runes];
        final b = ByteData(runes.length * 4);
        for (var i = 0; i < runes.length; i++) {
          b.setUint32(i * 4, runes[i], e);
        }
        return b.buffer.asUint8List();
      }

      expect(decode(utf8.encode(doc)), want);
      expect(decode([0xef, 0xbb, 0xbf, ...utf8.encode(doc)]), want);
      expect(decode(utf16(doc, Endian.little)), want);
      expect(decode(utf16(doc, Endian.big)), want);
      expect(decode(utf16(doc, Endian.little, bom: true)), want);
      expect(decode(utf16(doc, Endian.big, bom: true)), want);
      expect(decode(utf32(doc, Endian.little)), want);
      expect(decode(utf32(doc, Endian.big)), want);
      expect(decode(utf32(doc, Endian.little, bom: true)), want);
      expect(decode(utf32(doc, Endian.big, bom: true)), want);
    });

    test('refuses what Python refuses', () {
      expect(refusal([]), invalid);
      expect(refusal(utf8.encode('{')), invalid);
      expect(refusal([0x22, 0xff, 0x22]), invalid); // invalid UTF-8
      expect(refusal([0x22, 0xc0, 0x80, 0x22]), invalid); // overlong
      // Only one BOM is stripped; a second is not JSON.
      expect(
        refusal([0xef, 0xbb, 0xbf, 0xef, 0xbb, 0xbf, 0x7b, 0x7d]),
        invalid,
      );
      expect(refusal([0x7b, 0x00, 0x7d, 0x00, 0x20]), invalid); // odd UTF-16
      expect(refusal(utf8.encode('{"a":NaN}')), invalid); // see the report
    });

    test('UTF-8 surrogate bytes decode as lone surrogates (surrogatepass)', () {
      final r = decodeJsonBody(
        Uint8List.fromList([0x22, 0xed, 0xa0, 0x80, 0x22]),
      );
      expect(r.json, '\ud800');
      expect(r.rawSurrogate, isTrue);
      // An encoded pair is two surrogate code points to Python, not an emoji.
      final pair = decodeJsonBody(
        Uint8List.fromList([0x22, 0xed, 0xa0, 0xbd, 0xed, 0xb8, 0x80, 0x22]),
      );
      expect(pair.rawSurrogate, isTrue);
      expect(decodeJsonBody(utf8.encode('"\u{1F600}"')).rawSurrogate, isFalse);
    });

    test('nesting deeper than Python parses is invalid JSON', () {
      String nested(int depth) => '[' * depth + ']' * depth;
      expect(decode(utf8.encode(nested(maxJsonDepth))), isA<List<Object?>>());
      expect(refusal(utf8.encode(nested(maxJsonDepth + 1))), invalid);
      expect(refusal(utf8.encode(nested(500000))), invalid);
    });
  });
}
