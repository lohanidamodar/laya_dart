import 'dart:convert';
import 'dart:io';

import 'package:laya_server/src/errors.dart';
import 'package:laya_server/src/inference/backend.dart';
import 'package:laya_server/src/inference/batch.dart';
import 'package:laya_server/src/model/agent.dart';
import 'package:laya_server/src/model/checkpoint.dart';
import 'package:laya_server/src/model/confidence.dart';
import 'package:laya_server/src/model/question.dart';
import 'package:laya_server/src/model/sequence.dart';
import 'package:laya_server/src/tokenizer/tokenizer.dart';
import 'package:test/test.dart';

import '../support/checkpoints.dart';

/// Logits `[0, 1, 2, ...]` across the batch's marker width, for every row.
List<double> _ascending(int row, int markers) => [
  for (var i = 0; i < markers; i++) i.toDouble(),
];

Map<String, Question> _parse(Map<String, Object?> raw) => {
  for (final e in raw.entries) e.key: Question.parse(e.key, e.value),
};

/// Records each batch it runs, then defers to [inner].
class _Recording implements InferenceBackend {
  _Recording(this.inner);
  final InferenceBackend inner;
  final batches = <InferenceBatch>[];

  @override
  Future<InferenceOutput> run(InferenceBatch batch) {
    batches.add(batch);
    return inner.run(batch);
  }

  @override
  Future<void> close() => inner.close();
}

void main() {
  group('LayaAgent.predict', skip: tokenizerSkip('english'), () {
    late LayaTokenizer tok;
    final config = CheckpointConfig(
      maxLen: 512,
      headMaxLen: 192,
      temperature: [1.0, 1.0, 1.0],
      temperatureByOptions: const {},
    );
    setUpAll(
      () async => tok = await LayaTokenizer.load(tokenizerPath('english')),
    );

    LayaAgent agent(InferenceBackend backend) => LayaAgent(
      spec: CheckpointSpec.all['english']!,
      tokenizer: tok,
      config: config,
      backend: backend,
    );

    const state = 'I was charged twice for my March invoice, please refund me.';
    final raw = <String, Object?>{
      'urgency': {
        'type': 'score',
        'instructions': 'How urgent is this?',
        'criteria': ['low', 'medium', 'high'],
      },
      'department': {
        'type': 'choice',
        'instructions': 'Which department should handle this?',
        'criteria': ['support', 'sales', 'billing'],
      },
      'refund': {
        'type': 'noul',
        'instructions': 'The customer wants a refund.',
      },
    };

    test(
      'answers every question in request order, highest logit wins',
      () async {
        final result = await agent(
          FakeBackend(_ascending),
        ).predict(state, _parse(raw));
        expect(result.keys, ['model', 'answers', 'usage']);
        expect(result['model'], 'laya-rl-agent');
        final answers = result['answers']! as Map<String, Object?>;
        expect(answers.keys, ['urgency', 'department', 'refund']);
        expect((answers['department']! as Map)['choice'], 'billing');
        expect((answers['refund']! as Map)['noul'], greaterThan(0.5));
        final score = (answers['urgency']! as Map)['score']! as double;
        expect(score, greaterThan(1.0));
      },
    );

    test('usage counts this request\'s tokens, in laya\'s key order', () async {
      final backend = _Recording(FakeBackend(_ascending));
      final questions = _parse(raw);
      final result = await agent(backend).predict(state, questions);
      final ids = encodeState(tok, state);
      final rows = [
        for (final q in questions.values)
          buildSequence(
            tok,
            state,
            q,
            maxLen: 512,
            headMaxLen: 192,
            stateIds: ids,
          ),
      ];
      final total = rows.fold<int>(0, (s, r) => s + r.ids.length);
      expect(
        jsonEncode(result['usage']),
        jsonEncode({
          'input_tokens': total,
          'output_tokens': 0,
          'state_tokens': ids.length,
          'state_tokens_dropped': 0,
          'truncated': false,
          'truncated_questions': <String>[],
        }),
      );
      expect(backend.batches.single.rows, 3);
    });

    test('a 3,000-word state is truncated for every question', () async {
      final long = List.filled(3000, 'word').join(' ');
      final result = await agent(
        FakeBackend(_ascending),
      ).predict(long, _parse(raw));
      final usage = result['usage']! as Map<String, Object?>;
      expect(usage.keys, [
        'input_tokens',
        'output_tokens',
        'state_tokens',
        'state_tokens_dropped',
        'truncated',
        'truncated_questions',
      ]);
      expect(usage['truncated'], isTrue);
      expect(usage['truncated_questions'], ['urgency', 'department', 'refund']);
      expect(usage['state_tokens_dropped'], greaterThan(0));
      expect(usage['state_tokens'], encodeState(tok, long).length);
    });

    test('a list state keeps its end (truncate_left)', () async {
      final turns = [
        for (var i = 0; i < 400; i++) {'role': 'user', 'text': 'turn $i words'},
      ];
      final backend = _Recording(FakeBackend(_ascending));
      final q = _parse({'refund': raw['refund']});
      await agent(backend).predict(turns, q);
      final want = buildSequence(
        tok,
        turns,
        q['refund']!,
        maxLen: 512,
        headMaxLen: 192,
        truncateLeft: true,
      );
      final batch = backend.batches.single;
      expect(batch.inputIds.sublist(0, want.ids.length), want.ids);
    });

    test('collapsed options are reported under usage.options', () async {
      final many = {
        'route': {
          'type': 'choice',
          'instructions': 'Where should this go?',
          'criteria': [for (var i = 0; i < 40; i++) 'option label number $i'],
        },
        'refund': raw['refund'],
      };
      final result = await agent(
        FakeBackend(_ascending),
      ).predict(state, _parse(many));
      final usage = result['usage']! as Map<String, Object?>;
      expect(usage.keys.last, 'options');
      final options = usage['options']! as Map<String, Object?>;
      expect(options.keys, ['route']);
      expect(
        jsonEncode(options['route']),
        jsonEncode({'total': 40, 'distinct': 1, 'tokens_per_option': 4}),
      );
    });

    test('options past max_len are a 422 with laya\'s message', () async {
      final many = {
        'code': {
          'type': 'choice',
          'instructions': 'Pick a code.',
          'criteria': [for (var i = 0; i < 60; i++) 'code-$i'],
        },
      };
      await expectLater(
        agent(
          FakeBackend(_ascending),
        ).predict(state, _parse(many), maxLen: 48, headMaxLen: 32),
        throwsA(
          isA<ApiException>()
              .having((e) => e.status, 'status', 422)
              .having(
                (e) => e.detail,
                'detail',
                "question 'code': only 10 of its 60 option markers fit in "
                    'max_len=48 with head_max_len=32 spent on the question; '
                    'lower head_max_len, raise max_len, or use fewer options',
              ),
        ),
      );
    });

    test('no questions: empty answers and zero usage, no inference', () async {
      final backend = _Recording(FakeBackend(_ascending));
      final result = await agent(backend).predict(state, const {});
      expect(
        jsonEncode(result),
        '{"model":"laya-rl-agent","answers":{},'
        '"usage":{"input_tokens":0,"output_tokens":0}}',
      );
      expect(backend.batches, isEmpty);
    });

    test('min_confidence adds the gate fields to every answer', () async {
      final result = await agent(FakeBackend(_ascending)).predict(
        state,
        _parse(raw),
        minConfidence: const ConfidenceThreshold.scalar(0.99),
      );
      final answers = result['answers']! as Map<String, Object?>;
      for (final a in answers.values.cast<Map<String, Object?>>()) {
        expect(a['abstention_threshold'], 0.99);
        expect(a['abstention'], isNotNull);
      }
    });
  });

  group('LayaAgent.predict equals laya ONNXAgent.predict_batch', () {
    final fixture =
        jsonDecode(File('test/fixtures/predict.json').readAsStringSync())
            as Map<String, Object?>;
    final configs = fixture['configs']! as Map<String, Object?>;
    final requests = (fixture['requests']! as List)
        .cast<Map<String, Object?>>();
    final states = fixture['states']! as List;
    final cases = (fixture['cases']! as List).cast<Map<String, Object?>>();

    for (final name in ['english', 'multilingual', 'typed-decisions']) {
      test(name, skip: tokenizerSkip(name), () async {
        final tok = await LayaTokenizer.load(tokenizerPath(name));
        final config = CheckpointConfig.fromOnnxConfig(
          configs[name]! as Map<String, Object?>,
        );
        final mine = cases.where((c) => c['checkpoint'] == name).toList();
        expect(mine, hasLength(requests.length * 5));
        final failures = <String>[];
        for (final c in mine) {
          final req = requests[c['request_index']! as int];
          final variant = c['variant']! as Map<String, Object?>;
          final run = c['run'] as Map<String, Object?>?;
          List<double> row(String key, int r) => [
            for (final v in ((run![key]! as List)[r] as List))
              (v as num).toDouble(),
          ];
          final backend = _Recording(
            FakeBackend(
              (r, k) => row('logits', r),
              act: (r) => row('act_logits', r),
            ),
          );
          final agent = LayaAgent(
            spec: CheckpointSpec.all[name]!,
            tokenizer: tok,
            config: config,
            backend: backend,
          );
          final label = '#${c['request_index']} ${jsonEncode(variant)}';
          Object got;
          try {
            final questions = _parse(req['questions']! as Map<String, Object?>);
            got = await agent.predict(
              states[req['state']! as int] as Object,
              questions,
              maxLen: variant['max_len'] as int?,
              headMaxLen: variant['head_max_len'] as int?,
              minConfidence: variant['min_confidence'] == null
                  ? null
                  : ConfidenceThreshold.check(variant['min_confidence']),
            );
          } on ApiException catch (e) {
            got = 'ValueError: ${e.detail}';
            if (e.status != 422) failures.add('$label: status ${e.status}');
          }
          if (c.containsKey('error')) {
            if (got != c['error']) {
              failures.add('$label\n  want ${c['error']}\n  got  $got');
            }
            continue;
          }
          final want = Map.of(c['result']! as Map<String, Object?>);
          // ONNXAgent says "laya-rl-agent-onnx"; laya-serve's torch Agent says "laya-rl-agent".
          expect(want['model'], 'laya-rl-agent-onnx');
          want['model'] = 'laya-rl-agent';
          if (jsonEncode(got) != jsonEncode(want)) {
            failures.add(
              '$label\n  want ${jsonEncode(want)}\n  got  ${jsonEncode(got)}',
            );
          }
          if (run == null) {
            if (backend.batches.isNotEmpty) {
              failures.add('$label: ran inference');
            }
          } else {
            final b = backend.batches.single;
            if (b.rows != run['rows'] || b.markers != run['markers']) {
              failures.add('$label: batch ${b.rows}x${b.markers}, laya $run');
            }
          }
        }
        expect(failures, isEmpty, reason: failures.take(5).join('\n'));
      });
    }
  });
}
