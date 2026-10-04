import 'dart:convert';
import 'dart:io';

import 'package:laya_server/src/model/checkpoint.dart';
import 'package:laya_server/src/routing/router.dart';
import 'package:test/test.dart';

void main() {
  final fixture =
      jsonDecode(File('test/fixtures/route.json').readAsStringSync())
          as Map<String, Object?>;
  final corpus =
      jsonDecode(File('tool/reference/cases.json').readAsStringSync())
          as Map<String, Object?>;
  final states = corpus['states']! as List;
  final requests = (corpus['requests']! as List).cast<Map<String, Object?>>();
  final routers = {
    for (final e in (fixture['routers']! as Map<String, Object?>).entries)
      e.key: CheckpointRouter(
        autoTaskDetection:
            (e.value! as Map<String, Object?>)['auto_task_detection'] == true,
      ),
  };

  final cases = (fixture['cases']! as List).cast<Map<String, Object?>>();
  for (final (i, c) in cases.indexed) {
    final Object? state;
    final Map<String, Object?> questions;
    if (c.containsKey('request_index')) {
      final request = requests[c['request_index']! as int];
      state = states[request['state']! as int];
      questions = (request['questions']! as Map).cast<String, Object?>();
    } else {
      state = c['state'];
      questions = const {};
    }
    final kwargs = c['kwargs']! as Map<String, Object?>;
    final router = routers[c['router']]!;
    RouteDecision run() => router.route(
      state,
      questions,
      model: kwargs['model'] as String?,
      task: kwargs['task'] as String?,
      lang: kwargs['lang'] as String?,
      langGuess: kwargs['lang_guess'] as String?,
    );
    final label = '${c['router']} ${jsonEncode(kwargs)}';
    if (c.containsKey('error')) {
      final error = c['error']! as String;
      final message = error.substring('ValueError: '.length);
      test('case $i ($label) raises like laya', () {
        expect(error, startsWith('ValueError: '));
        expect(
          run,
          throwsA(
            isA<ArgumentError>().having((e) => e.message, 'message', message),
          ),
        );
      });
    } else {
      test('case $i ($label) equals Router.route', () {
        expect(jsonEncode(run().toJson()), jsonEncode(c['decision']));
      });
    }
  }

  test('aliases', () {
    expect(normaliseCheckpointName(' ML '), 'multilingual');
    expect(normaliseCheckpointName('typed_decisions'), 'typed-decisions');
    expect(() => normaliseCheckpointName('jev-1'), throwsArgumentError);
  });

  test('repos come from CheckpointSpec', () {
    for (final name in CheckpointSpec.all.keys) {
      final d = CheckpointRouter().route('x', const {}, model: name);
      expect(d.repo, CheckpointSpec.all[name]!.repo);
    }
  });

  test('the default model is normalised and used for undecided text', () {
    final router = CheckpointRouter(defaultModel: 'ML');
    final d = router.route('', const {});
    expect(d.model, 'multilingual');
    expect(
      d.reason,
      'no letters detected in state; using default (multilingual)',
    );
    expect(() => CheckpointRouter(defaultModel: 'gpt'), throwsArgumentError);
  });

  test('reasons repr non-printable characters like Python', () {
    final d = CheckpointRouter().route('x', const {}, model: '\u00a0en');
    expect(d.reason, r"explicit model='\xa0en'");
  });
}
