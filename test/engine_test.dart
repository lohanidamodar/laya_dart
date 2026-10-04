import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:laya_server/laya_server.dart';
import 'package:logging/logging.dart';
import 'package:test/test.dart';

import 'support/checkpoints.dart';

/// A [FakeBackend] whose closing and in-flight runs a test can see.
class _Tracked implements InferenceBackend {
  _Tracked(this.name, {Duration delay = Duration.zero})
    : _inner = FakeBackend(
        (r, k) => [for (var i = 0; i < k; i++) i.toDouble()],
        delay: delay,
      );
  final String name;
  final FakeBackend _inner;
  bool closed = false;
  int running = 0;
  int closedWhileRunning = 0;

  @override
  Future<InferenceOutput> run(InferenceBatch batch) async {
    running++;
    try {
      return await _inner.run(batch);
    } finally {
      running--;
    }
  }

  /// Makes [close] throw after marking the backend closed.
  bool failClose = false;

  @override
  Future<void> close() async {
    if (running > 0) closedWhileRunning++;
    closed = true;
    await _inner.close();
    if (failClose) throw StateError('release failed');
  }
}

const _english = 'I was charged twice for my March invoice, please refund me.';
const _hindi = 'मुझसे दो बार शुल्क लिया गया, कृपया मेरा पैसा वापस करें।';
const _questions = <String, Object?>{
  'refund': {'type': 'noul', 'instructions': 'The customer wants a refund.'},
};

void main() {
  group('LayaEngine', skip: tokenizerSkip('english'), () {
    late LayaTokenizer tok;
    setUpAll(
      () async => tok = await LayaTokenizer.load(tokenizerPath('english')),
    );

    late List<String> loads;
    late Map<String, List<_Tracked>> backends;
    Duration runDelay = Duration.zero;
    Duration loadDelay = Duration.zero;
    var failLoads = <String>{};
    var failCloses = <String>{};

    setUp(() {
      loads = [];
      backends = {};
      runDelay = Duration.zero;
      loadDelay = Duration.zero;
      failLoads = {};
      failCloses = {};
    });

    // Every fake checkpoint shares the English tokenizer: routing and residency are under test.
    Future<LayaAgent> loader(String name) async {
      loads.add(name);
      await Future<void>.delayed(loadDelay);
      if (failLoads.contains(name)) throw StateError('$name is not installed');
      final backend = _Tracked(name, delay: runDelay)
        ..failClose = failCloses.contains(name);
      (backends[name] ??= []).add(backend);
      final spec = CheckpointSpec.all[name]!;
      return LayaAgent(
        spec: spec,
        tokenizer: tok,
        config: CheckpointConfig(
          maxLen: spec.maxLen,
          headMaxLen: spec.headMaxLen,
          temperature: [1.0, 1.0, 1.0],
          temperatureByOptions: const {},
        ),
        backend: backend,
      );
    }

    Future<Map<String, Object?>> routing(
      LayaEngine e,
      Object state, [
      PredictOptions o = const PredictOptions(),
    ]) async =>
        (await e.predict(state, _questions, o))['routing']!
            as Map<String, Object?>;

    test(
      'routes Hindi to multilingual and English to english, with routing attached',
      () async {
        final engine = LayaEngine(loader: loader);
        addTearDown(engine.close);
        final hi = await engine.predict(
          _hindi,
          _questions,
          const PredictOptions(),
        );
        expect(hi.keys, ['model', 'answers', 'usage', 'routing']);
        expect((hi['routing']! as Map)['model'], 'multilingual');
        expect(
          (hi['routing']! as Map)['reason'],
          startsWith('non-Latin script (devanagari'),
        );
        final en = await routing(engine, _english);
        expect(en['model'], 'english');
        expect(en['reason'], 'English Latin text');
        expect(en['repo'], 'convaiinnovations/laya');
        expect(loads, ['multilingual', 'english']);
      },
    );

    test(
      'evicts the least recently used checkpoint and closes its backend',
      () async {
        final engine = LayaEngine(loader: loader, maxLoaded: 2);
        addTearDown(engine.close);
        await routing(engine, _english);
        await routing(engine, _hindi);
        await routing(engine, _english); // english is now the most recent
        expect(engine.loaded, ['multilingual', 'english']);
        await routing(engine, _english, const PredictOptions(model: 'typed'));
        expect(engine.loaded, ['english', 'typed-decisions']);
        await pumpEventQueue();
        expect(backends['multilingual']!.single.closed, isTrue);
        expect(backends['english']!.single.closed, isFalse);
        expect(loads, ['english', 'multilingual', 'typed-decisions']);
      },
    );

    test(
      'an evicted backend is closed only after its in-flight predict',
      () async {
        runDelay = const Duration(milliseconds: 200);
        final engine = LayaEngine(loader: loader, maxLoaded: 1);
        addTearDown(engine.close);
        final slow = engine.predict(
          _english,
          _questions,
          const PredictOptions(),
        );
        await Future<void>.delayed(const Duration(milliseconds: 50));
        runDelay = Duration.zero;
        await engine.predict(_hindi, _questions, const PredictOptions());
        expect(engine.loaded, ['multilingual']);
        final english = backends['english']!.single;
        expect(english.closed, isFalse);
        final result = await slow;
        expect((result['routing']! as Map)['model'], 'english');
        await pumpEventQueue();
        expect(english.closed, isTrue);
        expect(english.closedWhileRunning, 0);
      },
    );

    test(
      'explicit model: names and aliases pick, anything else auto-routes',
      () async {
        final engine = LayaEngine(loader: loader, maxLoaded: 3);
        addTearDown(engine.close);
        var r = await routing(
          engine,
          _english,
          const PredictOptions(model: 'jev-latest'),
        );
        expect(r['model'], 'english');
        expect(r['reason'], 'English Latin text');
        r = await routing(engine, _english, const PredictOptions(model: 'ml'));
        expect(r['model'], 'multilingual');
        expect(r['reason'], "explicit model='multilingual'");
      },
    );

    test('model resolution equals laya-serve _resolve_model', () async {
      final fixture =
          jsonDecode(File('test/fixtures/predict.json').readAsStringSync())
              as Map;
      final engine = LayaEngine(loader: loader, maxLoaded: 3);
      addTearDown(engine.close);
      for (final c
          in (fixture['resolve_model'] as List).cast<Map<String, Object?>>()) {
        final model = c['model'] as String?;
        final resolved = c['resolved'] as String?;
        final r = await routing(engine, _english, PredictOptions(model: model));
        if (resolved == null) {
          expect(
            r['reason'],
            'English Latin text',
            reason: '$model auto-routes',
          );
        } else {
          expect(r['model'], resolved, reason: '$model');
          expect(r['reason'], "explicit model='$resolved'", reason: '$model');
        }
      }
    });

    test('an unknown task is a 422 with laya\'s message', () async {
      final engine = LayaEngine(loader: loader);
      addTearDown(engine.close);
      await expectLater(
        engine.predict(_english, _questions, const PredictOptions(task: 'gpt')),
        throwsA(
          isA<ApiException>()
              .having((e) => e.status, 'status', 422)
              .having(
                (e) => e.detail,
                'detail',
                "unknown model 'gpt'; choose one of ['english', 'multilingual', "
                    "'typed-decisions'] (or an alias: ['decisions', 'default', 'en', "
                    "'laya', 'laya-multilingual', 'laya-typed-decisions', 'ml', "
                    "'multi', 'typed', 'typed_decisions'])",
              ),
        ),
      );
      expect(loads, isEmpty);
    });

    test('an invalid question is a 422 and loads nothing', () async {
      final engine = LayaEngine(loader: loader);
      addTearDown(engine.close);
      await expectLater(
        engine.predict(_english, {
          'q': {'type': 'nope', 'instructions': 'x'},
        }, const PredictOptions()),
        throwsA(isA<ApiException>().having((e) => e.status, 'status', 422)),
      );
      expect(loads, isEmpty);
    });

    test('concurrent predicts on one checkpoint load it once', () async {
      loadDelay = const Duration(milliseconds: 50);
      final engine = LayaEngine(loader: loader);
      addTearDown(engine.close);
      final results = await Future.wait([
        for (var i = 0; i < 8; i++)
          engine.predict(_english, _questions, const PredictOptions()),
      ]);
      expect(results, hasLength(8));
      expect(loads, ['english']);
    });

    test('a failed load evicts nothing', () async {
      failLoads = {'typed-decisions'};
      final engine = LayaEngine(loader: loader, maxLoaded: 2);
      addTearDown(engine.close);
      await routing(engine, _english);
      await routing(engine, _hindi);
      await expectLater(
        engine.predict(
          _english,
          _questions,
          const PredictOptions(task: 'typed'),
        ),
        throwsStateError,
      );
      await pumpEventQueue();
      expect(engine.loaded, ['english', 'multilingual']);
      expect(backends['english']!.single.closed, isFalse);
      expect(backends['multilingual']!.single.closed, isFalse);
      expect(loads, ['english', 'multilingual', 'typed-decisions']);
    });

    test('a failing backend close is logged as a warning', () async {
      failCloses = {'english'};
      final records = <LogRecord>[];
      final sub = Logger.root.onRecord
          .where((r) => r.loggerName == 'laya_server.engine')
          .listen(records.add);
      addTearDown(sub.cancel);
      final engine = LayaEngine(loader: loader, maxLoaded: 1);
      addTearDown(engine.close);
      await routing(engine, _english);
      await routing(engine, _hindi); // evicts english, whose close throws
      await pumpEventQueue();
      expect(backends['english']!.single.closed, isTrue);
      expect(records, hasLength(1));
      final r = records.single;
      expect(r.level, Level.WARNING);
      expect(r.message, contains('english'));
      expect(r.error, isA<StateError>());
      expect(r.stackTrace, isNotNull);
    });

    test('a failed load is retried on the next request', () async {
      var fail = true;
      final engine = LayaEngine(
        loader: (name) async {
          if (fail) {
            fail = false;
            throw StateError('boom');
          }
          return loader(name);
        },
      );
      addTearDown(engine.close);
      await expectLater(
        engine.predict(_english, _questions, const PredictOptions()),
        throwsStateError,
      );
      expect(engine.loaded, isEmpty);
      await engine.predict(_english, _questions, const PredictOptions());
      expect(engine.loaded, ['english']);
    });

    test(
      'preload loads by name or alias; close closes every backend',
      () async {
        final engine = LayaEngine(loader: loader, maxLoaded: 3);
        await engine.preload(['en', 'multilingual']);
        expect(engine.loaded, ['english', 'multilingual']);
        await engine.close();
        expect(backends.values.expand((b) => b).every((b) => b.closed), isTrue);
        await expectLater(
          engine.predict(_english, _questions, const PredictOptions()),
          throwsStateError,
        );
      },
    );

    test('token budget and min_confidence options reach the agent', () async {
      final engine = LayaEngine(loader: loader);
      addTearDown(engine.close);
      final long = List.filled(600, 'word').join(' ');
      final r = await engine.predict(
        long,
        _questions,
        const PredictOptions(
          maxLen: 64,
          minConfidence: ConfidenceThreshold.scalar(0.5),
        ),
      );
      expect((r['usage']! as Map)['truncated'], isTrue);
      final a = (r['answers']! as Map)['refund']! as Map;
      expect(a['abstention_threshold'], 0.5);
    });
  });

  group('LayaEngine.fileLoader', skip: tokenizerSkip('english'), () {
    late Directory dir;
    setUp(() => dir = Directory.systemTemp.createTempSync('laya_engine_'));
    tearDown(() => dir.deleteSync(recursive: true));

    void install(String name, {String model = 'model.onnx', bool data = true}) {
      final base = Directory('${dir.path}/$name/onnx')
        ..createSync(recursive: true);
      File(
        tokenizerPath('english'),
      ).copySync('${dir.path}/$name/tokenizer.json');
      final configs =
          (jsonDecode(File('test/fixtures/decode.json').readAsStringSync())
                  as Map)['configs']
              as Map;
      File(
        '${dir.path}/$name/config.json',
      ).writeAsStringSync(jsonEncode(configs[name]));
      File('test/fixtures/ort/laya_stub.onnx').copySync('${base.path}/$model');
      if (data) File('${base.path}/${model}_data').writeAsBytesSync(const []);
    }

    Matcher notInstalled(String name) => throwsA(
      isA<StateError>().having(
        (e) => e.message,
        'message',
        'checkpoint $name is not installed in ${dir.path}; '
            'run: dart run laya_server:laya_fetch $name',
      ),
    );

    test('a missing checkpoint names the fetch command', () async {
      await expectLater(
        LayaEngine.fileLoader(dir.path)('english'),
        notInstalled('english'),
      );
    });

    test('a missing _data sidecar counts as not installed', () async {
      install('english', data: false);
      await expectLater(
        LayaEngine.fileLoader(dir.path)('english'),
        notInstalled('english'),
      );
    });

    test('fp16 reads model_fp16.onnx', () async {
      install('english');
      await expectLater(
        LayaEngine.fileLoader(dir.path, variant: 'fp16')('english'),
        notInstalled('english'),
      );
    });

    test(
      'predicts end to end through ONNX Runtime on the stub graph',
      tags: ['ort'],
      () async {
        install('english');
        install('typed-decisions', model: 'model_fp16.onnx');
        final engine = LayaEngine(
          loader: LayaEngine.fileLoader(dir.path, threads: 1),
        );
        addTearDown(engine.close);
        // Stub logits are marker positions: the last option has the highest.
        final r = await engine.predict(_english, {
          'dept': {
            'type': 'choice',
            'instructions': 'Which department?',
            'criteria': ['support', 'sales', 'billing'],
          },
          ..._questions,
        }, const PredictOptions());
        expect(((r['answers']! as Map)['dept']! as Map)['choice'], 'billing');
        expect((r['routing']! as Map)['model'], 'english');
        expect(engine.loaded, ['english']);

        final fp16 = LayaEngine(
          loader: LayaEngine.fileLoader(dir.path, variant: 'fp16', threads: 1),
        );
        addTearDown(fp16.close);
        final t = await fp16.predict(
          _english,
          _questions,
          const PredictOptions(task: 'typed'),
        );
        expect((t['routing']! as Map)['model'], 'typed-decisions');
      },
    );
  });
}
