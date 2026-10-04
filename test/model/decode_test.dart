import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:laya_server/src/model/checkpoint.dart';
import 'package:laya_server/src/model/confidence.dart';
import 'package:laya_server/src/model/decode.dart';
import 'package:laya_server/src/model/question.dart';
import 'package:test/test.dart';

const _names = ['english', 'multilingual', 'typed-decisions'];

void main() {
  final fixture =
      jsonDecode(File('test/fixtures/decode.json').readAsStringSync()) as Map;
  final configs = fixture['configs'] as Map;
  final temperatures = fixture['temperatures'] as Map;

  /// The config laya decoded with: windows from the onnx-community config,
  /// temperatures from upstream rl_agent_config.json as ONNXAgent clamps them.
  CheckpointConfig fixtureConfig(String name) {
    final laya = (configs[name] as Map)['laya'] as Map;
    final t = temperatures[name] as Map;
    return CheckpointConfig(
      maxLen: laya['max_len'] as int,
      headMaxLen: laya['head_max_len'] as int,
      temperature: [
        for (final v in t['temperature'] as List) (v as num).toDouble(),
      ],
      temperatureByOptions: {
        for (final e in (t['temperature_by_options'] as Map).entries)
          e.key as String: (e.value as num).toDouble(),
      },
    );
  }

  test('decodeAnswer equals laya predict_batch on every fixture row', () {
    final cfgs = {for (final n in _names) n: fixtureConfig(n)};
    final cases = (fixture['cases'] as List).cast<Map<String, Object?>>();
    expect(cases, hasLength(3960));
    final failures = <String>[];
    for (final c in cases) {
      final q = Question.parse('q', c['question']);
      final got = decodeAnswer(
        q,
        Float32List.fromList([
          for (final v in c['logits']! as List) (v as num).toDouble(),
        ]),
        Float32List.fromList([
          for (final v in c['act_logits']! as List) (v as num).toDouble(),
        ]),
        cfgs[c['checkpoint']]!,
        minConfidence: c['min_confidence'] == null
            ? null
            : ConfidenceThreshold.check(c['min_confidence']),
      );
      if (jsonEncode(got) != jsonEncode(c['answer'])) {
        failures.add('${jsonEncode(c)}\n  got ${jsonEncode(got)}');
      }
    }
    expect(failures, isEmpty, reason: failures.take(5).join('\n'));
  });

  for (final name in _names) {
    test('$name onnx config temperatures, clamped, equal laya\'s', () {
      final cfg = CheckpointConfig.fromOnnxConfig(
        (configs[name] as Map).cast<String, Object?>(),
      );
      final want = fixtureConfig(name);
      expect(cfg.maxLen, want.maxLen);
      expect(cfg.headMaxLen, want.headMaxLen);
      expect(cfg.temperature, want.temperature);
      expect(cfg.temperatureByOptions, want.temperatureByOptions);
    });

    final upstream = File('.tooling/checkpoints/$name/rl_agent_config.json');
    test(
      '$name onnx config laya section equals upstream rl_agent_config.json',
      () {
        final rl = jsonDecode(upstream.readAsStringSync()) as Map;
        final laya = (configs[name] as Map)['laya'] as Map;
        for (final key in [
          'max_len',
          'head_max_len',
          'temperature',
          'temperature_by_options',
        ]) {
          expect(laya[key], rl[key], reason: '$name $key');
        }
      },
      skip: upstream.existsSync()
          ? false
          : 'run dart run tool/gen_fixtures.dart first',
    );
  }

  group('CheckpointConfig temperatures', () {
    test('temperature clamp neutralises choice:11+ = 0.1006', () {
      final cfg = CheckpointConfig(
        maxLen: 512,
        headMaxLen: 192,
        temperature: [1, 1, 1],
        temperatureByOptions: {'choice:11+': 0.10058280825614929},
      );
      expect(cfg.temperatureFor(QuestionType.choice, 12), 0.5);
    });

    test('bucket, then type, then 1.0', () {
      final cfg = CheckpointConfig(
        maxLen: 512,
        headMaxLen: 192,
        temperature: [1.5, 2.5],
        temperatureByOptions: {'choice:3-5': 3.0, 'score:6-10': 9.0},
      );
      expect(cfg.temperatureFor(QuestionType.choice, 3), 3.0);
      expect(cfg.temperatureFor(QuestionType.choice, 2), 1.5);
      expect(cfg.temperatureFor(QuestionType.score, 10), 5.0);
      expect(cfg.temperatureFor(QuestionType.score, 11), 2.5);
      expect(cfg.temperatureFor(QuestionType.noul, 2), 1.0);
    });

    test('fromOnnxConfig turns non-numbers into 1.0 and honours the clamp', () {
      final cfg = CheckpointConfig.fromOnnxConfig({
        'laya': {
          'max_len': 256,
          'head_max_len': 64,
          'temperature': [true, 'x', 9.0],
          'temperature_by_options': {'noul:2': null, 'choice:2': 0.7},
          'temperature_clamp': [0.8, 4.0],
        },
      });
      expect(cfg.maxLen, 256);
      expect(cfg.headMaxLen, 64);
      expect(cfg.temperature, [1.0, 1.0, 4.0]);
      expect(cfg.temperatureByOptions, {'noul:2': 1.0, 'choice:2': 0.8});
    });

    test('fromOnnxConfig rejects a config without a laya section', () {
      expect(
        () => CheckpointConfig.fromOnnxConfig({'model_type': 'modernbert'}),
        throwsFormatException,
      );
    });
  });

  group('decodeAnswer details', () {
    final cfg = CheckpointConfig(
      maxLen: 512,
      headMaxLen: 192,
      temperature: [1, 1, 1],
      temperatureByOptions: {},
    );

    test(
      'choice keys are the raw labels, as Python json.dumps writes them',
      () {
        final q = Question.parse('q', {
          'type': 'choice',
          'instructions': 'x',
          'criteria': [true, 2, 2.5, 'a'],
        });
        final a = decodeAnswer(q, [0, 5, 0, 0], [0, 0], cfg);
        expect(a['choice'], 2);
        expect((a['probabilities']! as Map).keys, ['true', '2', '2.5', 'a']);
      },
    );

    test('option_order un-permutes the slot row', () {
      final q = Question.parse('q', {
        'type': 'choice',
        'instructions': 'x',
        'criteria': ['a', 'b', 'c'],
        'option_order': [2, 0, 1],
      });
      // Slot 0 holds option 2 ('c'), which gets the big logit.
      final a = decodeAnswer(q, [9, 0, 0], [0, 0], cfg);
      expect(a['choice'], 'c');
    });

    test('rejects a row shorter than the option count', () {
      final q = Question.parse('q', {'type': 'noul', 'instructions': 'x'});
      expect(() => decodeAnswer(q, [1], [0, 0], cfg), throwsArgumentError);
    });
  });
}
